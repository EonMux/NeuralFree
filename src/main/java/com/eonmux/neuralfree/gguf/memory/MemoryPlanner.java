package com.eonmux.neuralfree.gguf.memory;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.MemoryKind;
import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.arch.SwaType;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;
import com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan.CompressedKvPlan;
import com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan.IndexerPlan;
import com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan.MsaPlan;
import com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan.SharedKv;
import com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan.StateTensor;
import com.eonmux.neuralfree.gguf.spec.AttentionSpec;
import com.eonmux.neuralfree.gguf.spec.DsaSpec;
import com.eonmux.neuralfree.gguf.spec.Dsv4Spec;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.MlaSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * Builds the byte-exact {@link MemoryPlan} of a model: the NeuralFree counterpart of llama.cpp
 * {@code llama_model::create_memory} (src/llama-model.cpp:2778-3290) and the cache constructors it selects.
 * The planner is driven by the resolved {@link ModelSpec}; it never reads GGUF keys.
 *
 * <p>Every plan has one slot per virtual layer (trunk, then NextN layers; {@code il} is the memory slot), but only the
 * layers of its context keep memory, as llama.cpp filters each cache by layer (src/llama-model.cpp:3240-3247 and
 * 2858-2862): the main context ({@link #planForContext}) plans the trunk and leaves the NextN layers empty, because
 * the trunk graph never attends them (a model made only of NextN layers keeps them all); the NextN / MTP draft
 * context ({@link #planNextn}) plans the NextN layers with the draft-context settings
 * ({@link MemoryConfig#forDraftContext()}) and leaves the trunk empty.
 *
 * <p>Per planned layer:
 * <ul>
 *   <li><b>No memory</b> when the model runs without a cache ({@link ModelSpec#memoryKind()} NONE: encoders,
 *       classifiers, diffusion) or is an LFM2 decision model ({@code causal_attn == false} and
 *       {@code n_layer_decision > 0}, llama-model.cpp:2781-2784), and on layers without a cached mixer.</li>
 *   <li><b>KV</b> for layers that read or write a cache ({@link LayerSpec#usesKv()}): standard attention stores
 *       {@code nHeadKv * headDimK} K and {@code nHeadKv * headDimV} V values per row; MLA stores latent rows of
 *       {@code kvLoraRank + headKRope} with V aliasing the first {@code kvLoraRank} values
 *       ({@link MlaSpec#latentCache()}), or per-head K and V when decompressed; DeepSeek-V4 stores a K-only raw row
 *       of {@code headDim} (MQA, K = V).</li>
 *   <li><b>Rows</b>: {@code nCtx}, or a ring of {@link KvLayout#ringRows} rows for causal STANDARD / CHUNKED
 *       sliding-window layers of the memory kinds llama.cpp builds with an ISWA cache (KV, HYBRID, DSA_ISWA and
 *       the DSV4 raw cache; llama-kv-cache-iswa.cpp:73). The other kinds (DSA, HYBRID_IDX, MSA, ENC_DEC) keep
 *       full-size caches and mask the window.</li>
 *   <li><b>Shared KV</b>: a layer whose {@link AttentionSpec#kvOwnerLayer()} is another layer reads that layer's
 *       cache, rows and rotation (gemma3n / gemma4 reuse); it owns nothing, including side caches.</li>
 *   <li><b>Recurrent state</b> ({@link LayerMemoryPlan#recurrentState}) on recurrent mixers, plus the qwen4exp
 *       PLE convolution history on the recurrent PLE layer (llama-memory-recurrent.cpp:110-114).</li>
 *   <li><b>Side caches</b>: the DSA lightning-indexer cache on layers with a full {@link DsaSpec},
 *       {@code indexer_head_size * kpool_row} wide for pooled indexers (llama-memory-hybrid-idx.cpp:58); the MSA
 *       index-key cache on sparse MSA layers; the DSV4 compressed caches with their compressor states (CSA plus
 *       the indexer for ratio 4, HCA for ratio 128; llama-kv-cache-dsv4.cpp:1244-1340), each compressor tensor
 *       with {@link MemoryConfig#rollbackSnapshots()} rollback snapshot planes ({@code n_rs_seq},
 *       llama-kv-cache-dsv4.cpp:968-970), at most the plan's ubatch since one rollback drops at most the positions
 *       of the last step; and the T5
 *       cross-attention K/V of {@code nCtx} encoder rows. Every indexer cache has the K type, as llama.cpp creates
 *       it with the context's {@code type_k} (llama-kv-cache-dsa.cpp:51, llama-kv-cache-dsv4.cpp:1308-1312,
 *       llama-memory-hybrid-idx.cpp:71), with each {@code indexer_head_size} slot whole blocks.</li>
 *   <li><b>KV dtypes</b>: {@link MemoryConfig#typeK()} / {@link MemoryConfig#typeV()}, any {@link KvDType}. As
 *       llama.cpp (llama-context.cpp:4074-4077), an MLA model or DeepSeek-V4 needs equal K and V types, which this
 *       planner rejects otherwise. Kernels address a head inside a row by its block offset (llama.cpp views each head
 *       with {@code ggml_row_size(type, head_dim)}, llama-kv-cache.cpp:1313-1318), so a block-type cache (Q8_0,
 *       Q4_0, Q4_1, IQ4_NL, Q5_0, Q5_1) needs every head (and, for an MLA latent row, the aliased {@code kv_lora} V
 *       prefix) to be whole 32-value blocks; otherwise the planner refuses the configuration, as llama.cpp's
 *       flash-attention check refuses the context (llama-context.cpp:4090-4110). Quantized V needs flash attention
 *       in llama.cpp (:4079-4088); every attention kernel here is the flash form.</li>
 *   <li><b>Rotation</b> ({@link RotationPlan}, llama-kv-cache.cpp:315-361): computed per cache domain (the full
 *       side and the sliding-window side of an ISWA cache) over the layers owning a cache, with the reference
 *       head dims {@code n_embd_head_k(0)} / {@code n_embd_head_v(0)}. A DSV4 layer's rotation applies to its
 *       raw, CSA and HCA caches (same type and head dim). The indexer cache of deepseek32, deepseek4, glm-dsa and
 *       dots3note is always rotated (its {@code n_embd_head_k_full == indexer_head_size} by construction), as is
 *       any indexer whose {@link DsaSpec#hadamard()} is set. The latent cache of an MLA or DSA layer is never
 *       rotated: llama.cpp reads it through the K-only attention inputs, which build no {@code self_k_rot}
 *       (llama-graph.cpp:3294-3376 and 3402-3478).</li>
 *   <li><b>KV groups</b>: layers with the same rotation domain, rows, ring flag and window share one
 *       {@code kv_pos} buffer; indexer and MSA caches use their layer's group.</li>
 * </ul>
 * The plan's {@code ubatch} is {@code min(MemoryConfig.ubatch, nCtx)} (llama.cpp never runs a ubatch larger than
 * the context). Stateless; every method is thread-safe.
 */
public final class MemoryPlanner {

    private static final Logger LOGGER = LogManager.getLogger(MemoryPlanner.class);

    /** Memory kinds llama.cpp builds with interleaved sliding-window (ring) caches. */
    private static final Set<MemoryKind> RING_KINDS =
            Collections.unmodifiableSet(EnumSet.of(MemoryKind.KV, MemoryKind.HYBRID, MemoryKind.DSA_ISWA, MemoryKind.DSV4));

    /** Architectures whose lightning-indexer cache is always Hadamard-rotated (llama-kv-cache.cpp:340-345). */
    private static final Set<Arch> INDEXER_ROTATION_ARCHS =
            Collections.unmodifiableSet(EnumSet.of(Arch.DEEPSEEK32, Arch.DEEPSEEK4, Arch.GLM_DSA, Arch.DOTS3NOTE));

    private MemoryPlanner() {
    }

    /**
     * Plans memory for the context {@link ContextFitter#fit} chooses: the user's {@code neuralfree.ctx}
     * (validated against the budget) or the largest context up to {@code n_ctx_train} that fits.
     *
     * @param budgetBytes bytes the plan may occupy ({@link ContextFitter#UNLIMITED} for no limit)
     * @throws ContextFitter.ContextBudgetException when no allowed context fits the budget
     * @throws IllegalArgumentException            when the spec cannot be planned (see {@link #planForContext})
     */
    public static MemoryPlan plan(ModelSpec spec, MemoryConfig cfg, long budgetBytes) {
        int nCtx = ContextFitter.fit(spec, cfg, budgetBytes);
        return planForContext(spec, cfg, nCtx);
    }

    /**
     * Plans memory for an explicit context length and logs the plan.
     *
     * @throws IllegalArgumentException for a non-positive context, a model that shares the KV of another model
     *                                  (use {@link #planDrafter}), or a layer whose parts contradict each other
     */
    public static MemoryPlan planForContext(ModelSpec spec, MemoryConfig cfg, int nCtx) {
        return logged(spec, new Planning(spec, cfg, nCtx, false, -1).run(), "memory plan");
    }

    /** {@link #planForContext} without logging; used by {@link ContextFitter} while searching. */
    static MemoryPlan planQuietly(ModelSpec spec, MemoryConfig cfg, int nCtx) {
        return new Planning(spec, cfg, nCtx, false, -1).run();
    }

    /**
     * Plans the NextN / MTP draft context of a model (llama.cpp {@code LLAMA_CONTEXT_TYPE_MTP},
     * src/llama-model.cpp:3240-3247): every NextN layer keeps its caches and the trunk keeps none. Rotation domains
     * and their shared head dims are taken over the NextN layers, the layers llama.cpp's MTP cache holds. The
     * context length and ubatch are the target plan's; the settings are the target plan's as a draft context runs
     * with them ({@link MemoryConfig#forDraftContext()}: F16 rows, so no rotation, and no state snapshots), as
     * llama.cpp creates the MTP context from the speculative parameters (common/speculative.cpp:2606-2607,
     * 2658-2669).
     *
     * @param spec   the model whose NextN layers draft: the target, or a draft file with NextN blocks of the target's
     *               model whose trunk slots are the target's
     * @param target the target's main-context plan ({@link #planForContext}), with or without slots for the NextN
     *               layers
     * @throws IllegalArgumentException when the model has no NextN layers or no trunk, or the target plan records
     *                                  no settings
     */
    public static MemoryPlan planNextn(ModelSpec spec, MemoryPlan target) {
        return logged(spec, planNextnContext(spec, target, -1), "NextN memory plan");
    }

    /**
     * {@link #planNextn(ModelSpec, MemoryPlan)} restricted to one NextN block: only layer {@code il} keeps caches
     * (with KV groups of its own), the memory of a drafter that runs each NextN block over its own caches. Rotation
     * is still decided over every NextN layer.
     *
     * @throws IllegalArgumentException as {@link #planNextn(ModelSpec, MemoryPlan)}, or when {@code il} is not a NextN
     *                                  layer
     */
    public static MemoryPlan planNextn(ModelSpec spec, MemoryPlan target, int il) {
        Objects.requireNonNull(spec, "spec");
        int first = spec.nLayer();
        int end = first + spec.nextnLayers().size();
        if (il < first || il >= end) {
            throw new IllegalArgumentException(spec.arch().ggufName() + ": layer " + il + " is not a NextN layer (NextN "
                    + "layers are " + first + ".." + (end - 1) + ")");
        }
        return logged(spec, planNextnContext(spec, target, il), "NextN memory plan for layer " + il);
    }

    /**
     * Bytes of the NextN draft memory that accompanies a main-context plan: the sum of the per-block plans of
     * {@link #planNextn(ModelSpec, MemoryPlan, int)} (each block with its own KV groups, which also bounds a single
     * plan of every NextN layer), or 0 when the model has no NextN layers or no trunk.
     */
    static long nextnBytes(ModelSpec spec, MemoryPlan target) {
        if (spec.nextnLayers().isEmpty() || spec.nLayer() == 0) {
            return 0L;
        }
        long bytes = 0;
        for (LayerSpec l : spec.nextnLayers()) {
            bytes += planNextnContext(spec, target, l.il()).totalBytes();
        }
        return bytes;
    }

    /**
     * Bytes of the NextN draft memory of {@code spec} beside a target context of {@code nCtx} tokens and a ubatch of
     * {@code ubatch} planned with {@code cfg}: the sum of the per-block plans of
     * {@link #planNextn(ModelSpec, MemoryPlan, int)} at those settings as a draft context runs with them
     * ({@link MemoryConfig#forDraftContext()}). This sizes the MTP context of a draft file with NextN blocks before
     * its target's plan exists.
     *
     * @throws IllegalArgumentException when the model has no NextN layers or no trunk, or the context or ubatch is
     *                                  not positive
     */
    public static long nextnBytes(ModelSpec spec, MemoryConfig cfg, int nCtx, int ubatch) {
        Objects.requireNonNull(spec, "spec");
        Objects.requireNonNull(cfg, "cfg");
        requireNextnModel(spec);
        if (nCtx < 1 || ubatch < 1) {
            throw new IllegalArgumentException(spec.arch().ggufName() + ": NextN memory of a context of " + nCtx
                    + " tokens and a ubatch of " + ubatch);
        }
        MemoryConfig nextnCfg = cfg.forDraftContext().toBuilder().nCtx(nCtx).ubatch(ubatch).build();
        long bytes = 0;
        for (LayerSpec l : spec.nextnLayers()) {
            bytes = Math.addExact(bytes, new Planning(spec, nextnCfg, nCtx, true, l.il()).run().totalBytes());
        }
        return bytes;
    }

    private static void requireNextnModel(ModelSpec spec) {
        String name = spec.arch().ggufName();
        if (spec.nextnLayers().isEmpty()) {
            throw new IllegalArgumentException(name + " has no NextN layers to draft with");
        }
        if (spec.nLayer() == 0) {
            throw new IllegalArgumentException(name + " has no trunk; its main memory plan already holds its NextN "
                    + "layers");
        }
    }

    private static MemoryPlan logged(ModelSpec spec, MemoryPlan plan, String what) {
        LOGGER.info("{} {}: {}", spec.arch().ggufName(), what, plan);
        return plan;
    }

    private static MemoryPlan planNextnContext(ModelSpec spec, MemoryPlan target, int only) {
        Objects.requireNonNull(spec, "spec");
        Objects.requireNonNull(target, "target");
        String name = spec.arch().ggufName();
        requireNextnModel(spec);
        MemoryConfig cfg = target.config();
        if (cfg == null) {
            throw new IllegalArgumentException(name + ": the target memory plan records no settings to plan the NextN "
                    + "context with; plan it with MemoryPlanner.planForContext");
        }
        int slots = target.layers().size();
        if (slots != spec.nLayer() + spec.nextnLayers().size() && slots != spec.nLayer()) {
            throw new IllegalArgumentException(name + ": the target memory plan has " + slots + " layer slots, the"
                    + " model " + spec.nLayer() + " trunk layers and " + spec.nextnLayers().size() + " NextN layers");
        }
        MemoryConfig nextnCfg = cfg.forDraftContext().toBuilder().nCtx(target.nCtx()).ubatch(target.ubatch()).build();
        return new Planning(spec, nextnCfg, target.nCtx(), true, only).run();
    }

    /**
     * Plans a drafter that reads the KV caches of its target model instead of owning any (gemma4-assistant,
     * llama-model.cpp:3250-3262): each attention layer of the drafter becomes a shared-KV layer reading target layer
     * {@code n_layer - 2} when it is a sliding-window layer and {@code n_layer - 1} otherwise, with the target
     * layer's rotation. The plan takes the target's context and ubatch.
     *
     * @param drafter    the drafter spec
     * @param target     the target spec ({@code n_layer} = its trunk length)
     * @param targetPlan the target's memory plan
     * @throws IllegalArgumentException when the drafter is not a cross-model KV sharer, has memory of its own, or
     *                                  its K/V widths or windows disagree with the target layers it reads
     */
    public static MemoryPlan planDrafter(ModelSpec drafter, ModelSpec target, MemoryPlan targetPlan) {
        Objects.requireNonNull(drafter, "drafter");
        Objects.requireNonNull(target, "target");
        Objects.requireNonNull(targetPlan, "targetPlan");
        String name = drafter.arch().ggufName();
        if (!sharesTargetKv(drafter.arch())) {
            throw new IllegalArgumentException(name + " does not read the KV cache of a target model");
        }
        int targetLayers = target.nLayer();
        if (targetLayers < 2) {
            throw new IllegalArgumentException(name + " needs a target with at least two layers, got " + targetLayers);
        }
        MemoryPlan.Builder b = MemoryPlan.builder(drafter.memoryKind(), targetPlan.nCtx(), targetPlan.ubatch());
        if (targetPlan.config() != null) {
            b.config(targetPlan.config());
        }
        for (LayerSpec l : allLayers(drafter)) {
            if (l.isRecurrent() || l.dsv4() != null || l.crossAttention() != null) {
                throw new IllegalArgumentException(name + " layer " + l.il() + " (" + l.mixerKind()
                        + ") needs memory of its own, which a KV-sharing drafter cannot have");
            }
            if (!l.usesKv()) {
                b.layer(LayerMemoryPlan.builder(l.il(), LayerMemoryPlan.Kind.NONE).build());
                continue;
            }
            AttentionSpec a = l.attention();
            if (a == null || l.mla() != null) {
                throw new IllegalArgumentException(name + " layer " + l.il() + ": only standard attention can read the target KV");
            }
            int targetIl = targetLayers - (a.isSwaLayer() ? 2 : 1);
            LayerMemoryPlan source = resolveOwner(targetPlan, targetIl);
            KvLayout kv = source.kv();
            if (kv == null) {
                throw new IllegalArgumentException(name + " layer " + l.il() + " reads target layer " + targetIl
                        + ", which has no KV cache");
            }
            if (kv.vAliasesK() || kv.widthK() != a.kDim() || kv.widthV() != a.vDim()) {
                throw new IllegalArgumentException(name + " layer " + l.il() + " K/V widths " + a.kDim() + "/" + a.vDim()
                        + " do not match target layer " + targetIl + " " + kv);
            }
            if (kv.window().windowed() != a.isSwaLayer()) {
                throw new IllegalArgumentException(name + " layer " + l.il() + (a.isSwaLayer() ? " is" : " is not")
                        + " a sliding-window layer but target layer " + targetIl + " has " + kv.window());
            }
            b.layer(LayerMemoryPlan.builder(l.il(), LayerMemoryPlan.Kind.SHARED_KV)
                    .sharedFrom(SharedKv.targetLayer(targetIl))
                    .rotation(source.rotation())
                    .build());
        }
        return b.build();
    }

    /** Architectures whose layers read the caches of a separate target model. */
    private static boolean sharesTargetKv(Arch arch) {
        return arch == Arch.GEMMA4_ASSISTANT;
    }

    /** Trunk then NextN layers, in memory-slot order. */
    private static List<LayerSpec> allLayers(ModelSpec spec) {
        List<LayerSpec> out = new ArrayList<>(spec.layers().size() + spec.nextnLayers().size());
        out.addAll(spec.layers());
        out.addAll(spec.nextnLayers());
        return out;
    }

    /** The plan of the layer whose cache {@code il} reads (itself, or its local shared-KV owner). */
    private static LayerMemoryPlan resolveOwner(MemoryPlan plan, int il) {
        LayerMemoryPlan p = plan.layer(il);
        if (p.kind() == LayerMemoryPlan.Kind.SHARED_KV && !p.sharedFrom().target()) {
            return plan.layer(p.sharedFrom().layer());
        }
        return p;
    }

    /** Per-layer facts gathered before rotation domains are known. */
    private static final class Draft {
        final LayerSpec layer;
        LayerMemoryPlan.Kind kind = LayerMemoryPlan.Kind.NONE;
        KvLayout kv;
        int group = -1;
        boolean swaDomain;
        int headDimK;
        int headDimV;
        int ownerLayer = -1;
        final List<StateTensor> states = new ArrayList<>();
        int indexerWidth;
        KvDType indexerType;
        boolean indexerForced;
        MsaPlan msa;
        final List<CompressedKvPlan> compressed = new ArrayList<>();
        KvLayout crossKv;

        Draft(LayerSpec layer) {
            this.layer = layer;
        }

        boolean ownsKv() {
            return kind == LayerMemoryPlan.Kind.KV || kind == LayerMemoryPlan.Kind.KV_RECURRENT
                    || kind == LayerMemoryPlan.Kind.DSV4;
        }
    }

    /** One planning run; mutable only while {@link #run()} executes. */
    private static final class Planning {
        private final ModelSpec spec;
        private final ModelHparams hp;
        private final MemoryConfig cfg;
        private final int nCtx;
        private final int ubatch;
        private final boolean nextnContext;
        private final int onlyLayer;
        private final Map<String, Integer> groupIndex = new LinkedHashMap<>();
        private final List<MemoryPlan.KvGroup> groups = new ArrayList<>();
        private int[] groupRemap = new int[0];

        /**
         * @param nextnContext plan the NextN / MTP draft context instead of the main context
         * @param onlyLayer    the one layer of the context that keeps memory, or -1 for all of them
         */
        Planning(ModelSpec spec, MemoryConfig cfg, int nCtx, boolean nextnContext, int onlyLayer) {
            this.spec = Objects.requireNonNull(spec, "spec");
            this.hp = spec.hparams();
            this.cfg = Objects.requireNonNull(cfg, "cfg");
            if (nCtx <= 0) {
                throw new IllegalArgumentException(spec.arch().ggufName() + ": context length must be positive, got " + nCtx);
            }
            checkKvTypes(spec, cfg);
            this.nCtx = nCtx;
            this.ubatch = Math.min(cfg.ubatch(), nCtx);
            this.nextnContext = nextnContext;
            this.onlyLayer = onlyLayer;
        }

        MemoryPlan run() {
            String name = spec.arch().ggufName();
            if (sharesTargetKv(spec.arch())) {
                throw new IllegalArgumentException(name + " reads the KV cache of its target model; plan it with "
                        + "MemoryPlanner.planDrafter");
            }
            List<LayerSpec> layers = allLayers(spec);
            boolean memoryless = memoryless();
            MemoryPlan.Builder b = MemoryPlan.builder(memoryless ? MemoryKind.NONE : spec.memoryKind(), nCtx, ubatch)
                    .config(cfg);
            List<Draft> drafts = new ArrayList<>(layers.size());
            for (LayerSpec l : layers) {
                Draft d = new Draft(l);
                if (!memoryless && inContext(l)) {
                    describe(d);
                }
                drafts.add(d);
            }
            resolveShared(drafts);
            int[] headKAll = new int[2];
            int[] headVAll = new int[2];
            int idxWidthAll = commonDims(drafts, headKAll, headVAll);
            for (MemoryPlan.KvGroup g : keptGroups(drafts)) {
                b.group(g);
            }
            for (Draft d : drafts) {
                b.layer(build(d, drafts, headKAll, headVAll, idxWidthAll));
            }
            return b.build();
        }

        /**
         * Whether a layer's caches belong to the context being planned: the trunk for the main context (every layer
         * when the model has no trunk), the NextN layers for the NextN context (llama-model.cpp:3240-3247).
         */
        private boolean inContext(LayerSpec l) {
            int nTrunk = spec.nLayer();
            boolean nextn = l.il() >= nTrunk;
            return nextnContext ? nextn : !nextn || nTrunk == 0;
        }

        /** Whether a layer of the context keeps memory in this plan. */
        private boolean kept(LayerSpec l) {
            return inContext(l) && (onlyLayer < 0 || l.il() == onlyLayer);
        }

        /** The KV groups the kept layers use, renumbered in creation order; fills {@link #groupRemap}. */
        private List<MemoryPlan.KvGroup> keptGroups(List<Draft> drafts) {
            boolean[] used = new boolean[groups.size()];
            for (Draft d : drafts) {
                if (d.group >= 0 && kept(d.layer)) {
                    used[d.group] = true;
                }
            }
            groupRemap = new int[groups.size()];
            List<MemoryPlan.KvGroup> out = new ArrayList<>();
            for (int g = 0; g < groups.size(); g++) {
                groupRemap[g] = -1;
                if (used[g]) {
                    MemoryPlan.KvGroup old = groups.get(g);
                    groupRemap[g] = out.size();
                    out.add(MemoryPlan.KvGroup.of(out.size(), old.rows(), old.ring(), old.window()));
                }
            }
            return out;
        }

        /** No memory for cache-less models and LFM2 decision models (llama-model.cpp:2781-2784). */
        private boolean memoryless() {
            if (spec.memoryKind() == MemoryKind.NONE) {
                return true;
            }
            return spec.arch() == Arch.LFM2 && !hp.causal() && hp.nLayerDecision() > 0;
        }

        /** Fills the layer's own caches (shared layers are completed by {@link #resolveShared}). */
        private void describe(Draft d) {
            LayerSpec l = d.layer;
            AttentionSpec a = l.attention();
            boolean shared = l.dsv4() == null && l.usesKv() && a != null && a.sharesKv(l.il());
            if (shared) {
                d.kind = LayerMemoryPlan.Kind.SHARED_KV;
                d.ownerLayer = a.kvOwnerLayer();
                if (l.isRecurrent() || l.crossAttention() != null) {
                    throw new IllegalArgumentException(layerName(l) + " shares the KV of layer " + d.ownerLayer
                            + " but also needs recurrent or cross-attention memory of its own");
                }
                return;
            }
            if (l.dsv4() != null) {
                describeDsv4(d, l.dsv4());
            } else if (l.usesKv()) {
                describeKv(d);
            }
            if (l.isRecurrent()) {
                d.states.addAll(LayerMemoryPlan.recurrentState(l.recurrent()));
            }
            addPleHistory(d);
            if (d.kv != null) {
                d.kind = d.layer.dsv4() != null ? LayerMemoryPlan.Kind.DSV4
                        : d.states.isEmpty() ? LayerMemoryPlan.Kind.KV : LayerMemoryPlan.Kind.KV_RECURRENT;
            } else if (!d.states.isEmpty()) {
                d.kind = LayerMemoryPlan.Kind.RECURRENT;
            }
            if (d.kind == LayerMemoryPlan.Kind.DSV4 && !d.states.isEmpty()) {
                throw new IllegalArgumentException(layerName(l) + ": a DeepSeek-V4 layer cannot keep recurrent state");
            }
            if (l.crossAttention() != null) {
                if (d.kind != LayerMemoryPlan.Kind.KV) {
                    throw new IllegalArgumentException(layerName(l) + ": cross-attention needs a decoder self-attention "
                            + "KV cache, got " + d.kind);
                }
                AttentionSpec x = l.crossAttention();
                d.crossKv = KvLayout.of(nCtx, false, x.kDim(), x.vDim(),
                        storeK(x.kDim(), x.headDimK(), "cross K"),
                        storeV(x.vDim(), x.headDimV(), "cross V"), KvWindow.FULL_BIDIRECTIONAL);
            }
        }

        /** Standard or MLA KV, plus the DSA indexer and MSA index caches. */
        private void describeKv(Draft d) {
            LayerSpec l = d.layer;
            AttentionSpec a = l.attention();
            MlaSpec mla = l.mla();
            KvWindow window = a != null ? KvWindow.of(a.effectiveSwaType(), a.nSwa(), a.causal()) : KvWindow.FULL_CAUSAL;
            boolean swaSide = ringEligible(window);
            int rows = rowsFor(window, swaSide);
            boolean ring = swaSide && rows < nCtx;
            if (mla != null) {
                checkMlaCacheMode(l, mla);
            }
            KvLayout kv;
            if (mla != null && mla.latentCache()) {
                int width = mla.latentRowWidth();
                kv = KvLayout.latent(rows, ring, width, mla.kvLoraRank(),
                        storeK(width, mla.kvLoraRank(), "MLA latent"), window);
                d.headDimK = width;
                d.headDimV = 0;
            } else if (mla != null) {
                int wk = mla.nHead() * mla.headK();
                int wv = mla.nHead() * mla.headV();
                kv = KvLayout.of(rows, ring, wk, wv, storeK(wk, mla.headK(), "attention K"),
                        storeV(wv, mla.headV(), "attention V"), window);
                d.headDimK = mla.headK();
                d.headDimV = mla.headV();
            } else {
                kv = KvLayout.of(rows, ring, a.kDim(), a.vDim(), storeK(a.kDim(), a.headDimK(), "attention K"),
                        storeV(a.vDim(), a.headDimV(), "attention V"), window);
                d.headDimK = a.headDimK();
                d.headDimV = a.headDimV();
            }
            assignKv(d, kv, swaSide);
            DsaSpec dsa = l.dsa();
            if (dsa != null && dsa.full()) {
                d.indexerWidth = dsa.indexerHeadDim() * (dsa.kpool() > 0 ? dsa.kpoolRow() : 1);
                d.indexerType = storeK(d.indexerWidth, dsa.indexerHeadDim(), "DSA indexer");
                d.indexerForced = dsa.hadamard()
                        || (INDEXER_ROTATION_ARCHS.contains(spec.arch()) && dsa.indexerHeadDim() == hp.indexerHeadSize()
                        && dsa.kpool() == 0);
            }
            if (l.msa() != null && l.msa().sparse()) {
                int width = l.msa().idxHeadDim();
                d.msa = MsaPlan.of(rows, width, storeK(width, width, "MSA index key"));
            }
        }

        /** The spec's resolved MLA cache form must be the one an explicit {@code neuralfree.mla.cache} asks for. */
        private void checkMlaCacheMode(LayerSpec l, MlaSpec mla) {
            MemoryConfig.MlaCacheMode mode = cfg.mlaCache();
            boolean contradicts = (mode == MemoryConfig.MlaCacheMode.LATENT && !mla.latentCache())
                    || (mode == MemoryConfig.MlaCacheMode.DECOMPRESSED && mla.latentCache());
            if (contradicts) {
                throw new IllegalArgumentException(layerName(l) + ": " + MemoryConfig.PROP_MLA_CACHE + "=" + mode
                        + " but the model spec was resolved with a " + (mla.latentCache() ? "latent" : "decompressed")
                        + " MLA cache" + (mla.absorbed() ? "" : " (the model has no absorbed attn_k_b/attn_v_b weights)"));
            }
        }

        /** DeepSeek-V4: K-only raw window cache plus the compressed caches of the layer's ratio. */
        private void describeDsv4(Draft d, Dsv4Spec v4) {
            LayerSpec l = d.layer;
            SwaType type = hp.swaType();
            if (type == SwaType.NONE) {
                throw new IllegalArgumentException(layerName(l) + ": the DeepSeek-V4 raw cache needs a sliding-window "
                        + "type, but the model has none");
            }
            KvWindow window = KvWindow.of(type, v4.window(), true);
            boolean swaSide = hp.isSwa(l.weightLayer()) && ringEligible(window);
            int rows = swaSide ? rowsFor(window, true) : nCtx;
            int width = v4.headDim();
            KvDType dtype = storeK(width, width, "DSV4 raw K");
            assignKv(d, KvLayout.latent(rows, swaSide && rows < nCtx, width, width, dtype, window), swaSide);
            d.headDimK = width;
            d.headDimV = 0;
            int snapshots = Math.min(cfg.rollbackSnapshots(), ubatch);
            if (v4.csa()) {
                int idx = v4.indexer().indexerHeadDim();
                d.compressed.add(CompressedKvPlan.of(CompressedKvPlan.Kind.CSA, nCtx, width, idx,
                        storeK(width, width, "DSV4 CSA"), snapshots));
                d.compressed.add(CompressedKvPlan.of(CompressedKvPlan.Kind.INDEXER, nCtx, width, idx,
                        storeK(idx, idx, "DSV4 indexer"), snapshots));
            } else if (v4.hca()) {
                d.compressed.add(CompressedKvPlan.of(CompressedKvPlan.Kind.HCA, nCtx, width, 0,
                        storeK(width, width, "DSV4 HCA"), snapshots));
            }
        }

        /** qwen4exp PLE convolution history: a row of the recurrent cache of the PLE layer. */
        private void addPleHistory(Draft d) {
            long floats = hp.pleConvState();
            LayerSpec l = d.layer;
            if (floats <= 0 || !hp.isPleLayer(l.weightLayer())) {
                return;
            }
            if (!l.isRecurrent()) {
                throw new IllegalArgumentException(layerName(l) + " is a PLE layer but not a linear-attention layer; the "
                        + "PLE convolution history lives in the recurrent cache");
            }
            d.states.add(StateTensor.of(LayerMemoryPlan.STATE_PLE_CONV, LayerMemoryPlan.StateLayout.PLE_CONV, floats));
        }

        /** Causal STANDARD / CHUNKED windows of ring memory kinds live on the sliding-window side. */
        private boolean ringEligible(KvWindow window) {
            return RING_KINDS.contains(spec.memoryKind()) && window.windowed() && window.causal()
                    && (window.type() == SwaType.STANDARD || window.type() == SwaType.CHUNKED);
        }

        private int rowsFor(KvWindow window, boolean swaSide) {
            return swaSide ? KvLayout.ringRows(nCtx, window.nSwa(), ubatch, cfg.swaFull()) : nCtx;
        }

        private void assignKv(Draft d, KvLayout kv, boolean swaSide) {
            KvWindow w = kv.window();
            String key = (swaSide ? "swa" : "full") + '|' + kv.rows() + '|' + kv.ring() + '|' + w.type() + '/' + w.nSwa()
                    + '/' + w.causal();
            Integer g = groupIndex.get(key);
            if (g == null) {
                g = groups.size();
                groups.add(MemoryPlan.KvGroup.of(g, kv.rows(), kv.ring(), w));
                groupIndex.put(key, g);
            }
            d.kv = kv;
            d.group = g;
            d.swaDomain = swaSide;
        }

        /** Points every shared-KV layer at its owner's layout and group. */
        private void resolveShared(List<Draft> drafts) {
            for (Draft d : drafts) {
                if (d.kind != LayerMemoryPlan.Kind.SHARED_KV) {
                    continue;
                }
                if (d.ownerLayer < 0 || d.ownerLayer >= drafts.size()) {
                    throw new IllegalArgumentException(layerName(d.layer) + " shares the KV of layer " + d.ownerLayer
                            + ", which does not exist");
                }
                Draft owner = drafts.get(d.ownerLayer);
                if (!owner.ownsKv()) {
                    throw new IllegalArgumentException(layerName(d.layer) + " shares the KV of layer " + d.ownerLayer
                            + ", which owns no cache (" + owner.kind + ")");
                }
                if (!sameCacheShape(d.layer, owner.kv)) {
                    throw new IllegalArgumentException(layerName(d.layer) + " cache rows do not match its KV owner layer "
                            + d.ownerLayer + " " + owner.kv);
                }
                AttentionSpec a = d.layer.attention();
                KvWindow own = KvWindow.of(a.effectiveSwaType(), a.nSwa(), a.causal());
                KvWindow ownerWindow = owner.kv.window();
                if (owner.kv.ring() && (own.type() != ownerWindow.type() || own.nSwa() > ownerWindow.nSwa())) {
                    throw new IllegalArgumentException(layerName(d.layer) + " attends with " + own
                            + " but its KV owner layer " + d.ownerLayer + " keeps only a ring of " + owner.kv.rows()
                            + " rows for " + ownerWindow);
                }
                d.kv = owner.kv;
                d.group = owner.group;
                d.swaDomain = owner.swaDomain;
            }
        }

        /**
         * Per rotation domain (index 0 full side, 1 sliding-window side) the head dims shared by every layer owning a
         * cache there: the common value, -1 when they differ, 0 when there is none. The V value is 0 as soon as one
         * cache of the domain has no V rows (llama.cpp leaves {@code n_embd_head_v_all} at 0 for MLA caches). Returns
         * the common indexer cache width the same way.
         */
        private int commonDims(List<Draft> drafts, int[] headKAll, int[] headVAll) {
            boolean[] noV = new boolean[2];
            int idx = 0;
            for (Draft d : drafts) {
                if (!d.ownsKv()) {
                    continue;
                }
                int dom = d.swaDomain ? 1 : 0;
                headKAll[dom] = merge(headKAll[dom], d.headDimK);
                if (d.headDimV == 0) {
                    noV[dom] = true;
                } else {
                    headVAll[dom] = merge(headVAll[dom], d.headDimV);
                }
                if (d.indexerWidth > 0) {
                    idx = merge(idx, d.indexerWidth);
                }
            }
            for (int dom = 0; dom < 2; dom++) {
                if (noV[dom]) {
                    headVAll[dom] = 0;
                }
            }
            return idx;
        }

        /** Rows of {@code kv} have the K/V widths and form {@code l} itself would cache. */
        private static boolean sameCacheShape(LayerSpec l, KvLayout kv) {
            MlaSpec mla = l.mla();
            if (mla != null && mla.latentCache()) {
                return kv.vAliasesK() && kv.widthK() == mla.latentRowWidth() && kv.widthV() == mla.kvLoraRank();
            }
            if (mla != null) {
                return !kv.vAliasesK() && kv.widthK() == mla.nHead() * mla.headK() && kv.widthV() == mla.nHead() * mla.headV();
            }
            AttentionSpec a = l.attention();
            return !kv.vAliasesK() && kv.widthK() == a.kDim() && kv.widthV() == a.vDim();
        }

        private static int merge(int all, int value) {
            if (all == 0) {
                return value;
            }
            return all == value ? all : -1;
        }

        private LayerMemoryPlan build(Draft d, List<Draft> drafts, int[] headKAll, int[] headVAll, int idxWidthAll) {
            LayerSpec l = d.layer;
            if (d.kind == LayerMemoryPlan.Kind.NONE || !kept(l)) {
                return LayerMemoryPlan.builder(l.il(), LayerMemoryPlan.Kind.NONE).build();
            }
            LayerMemoryPlan.Builder b = LayerMemoryPlan.builder(l.il(), d.kind);
            if (d.kind == LayerMemoryPlan.Kind.SHARED_KV) {
                Draft owner = drafts.get(d.ownerLayer);
                return b.sharedFrom(SharedKv.local(d.ownerLayer)).kv(d.kv, groupRemap[d.group])
                        .rotation(rotation(owner, headKAll, headVAll)).build();
            }
            if (d.kv != null) {
                RotationPlan rot = rotation(d, headKAll, headVAll);
                b.kv(d.kv, groupRemap[d.group]).rotation(rot);
            }
            b.states(d.states);
            if (d.indexerWidth > 0) {
                b.indexer(IndexerPlan.of(d.kv.rows(), d.indexerWidth, d.indexerType,
                        indexerRotation(d.indexerType, idxWidthAll, d.indexerWidth, d.indexerForced)));
            }
            if (d.msa != null) {
                b.msa(d.msa);
            }
            for (CompressedKvPlan c : d.compressed) {
                b.compressed(c);
            }
            if (d.crossKv != null) {
                b.crossKv(d.crossKv);
            }
            return b.build();
        }

        /**
         * Rotation of the main cache of an owning layer in its domain (llama-kv-cache.cpp:328-360). Latent MLA and DSA
         * caches are never rotated, although llama.cpp sizes {@code n_rot_k} for them: the MLA graphs read them through
         * the K-only input {@code build_attn_inp_k}, which builds no {@code self_k_rot} and applies no Hadamard
         * (llama-graph.cpp:3294-3376; that input only exists without sliding windows, :3304), and the DSA attention
         * leaves the cache's rotation to the indexer (llama-graph.cpp:3402-3478). Decompressed MLA caches go through
         * {@code build_attn_inp_kv} and are rotated like any other cache (llama-graph.cpp:3225-3231).
         */
        private RotationPlan rotation(Draft owner, int[] headKAll, int[] headVAll) {
            if (owner.layer.mixerKind() == MixerKind.DSA_MLA
                    || (owner.layer.mla() != null && owner.kv.vAliasesK())) {
                return RotationPlan.NONE;
            }
            int dom = owner.swaDomain ? 1 : 0;
            boolean forceK = INDEXER_ROTATION_ARCHS.contains(spec.arch()) && hp.headKFull() == hp.indexerHeadSize();
            KvDType typeV = owner.kv.vAliasesK() ? owner.kv.dtypeK() : owner.kv.dtypeV();
            return RotationPlan.of(owner.kv.dtypeK(), typeV, headKAll[dom], headVAll[dom], hp.headK(0), hp.headV(0),
                    cfg.kvRotation(), forceK);
        }

        /**
         * Rotation of an indexer cache's q/k. Only the forced rotation applies, whatever the cache type: llama.cpp
         * rotates indexer q/k solely in the graphs of the forced archs (deepseek32.cpp:246-248, glm-dsa.cpp:280-282,
         * dots3note.cpp:233-235, deepseek4.cpp:606-619 and 1072-1074), whose indexer caches always carry {@code k_rot}
         * (llama-kv-cache.cpp:331-334); the other indexers never apply one (hy-v4.cpp:377, the k-pool and MSA
         * graphs). {@code nRotK} follows the shared indexer width as in {@link RotationPlan#of}.
         */
        private static RotationPlan indexerRotation(KvDType type, int widthAll, int width, boolean forced) {
            return RotationPlan.of(type, type, widthAll, 0, width, 0, false, forced);
        }

        /**
         * llama.cpp's cache-type check (llama-context.cpp:4074-4077): an MLA model ({@code is_mla}) or DeepSeek-V4
         * stores K and V in one cache row, so the K and V types must be equal. A latent MLA cache here is the same
         * single row (its V aliases the K prefix), so it is held to the same rule.
         *
         * @throws IllegalArgumentException with llama.cpp's message when they differ
         */
        private static void checkKvTypes(ModelSpec spec, MemoryConfig cfg) {
            boolean shared = spec.hparams().isMla() || spec.arch() == Arch.DEEPSEEK4 || hasLatentMla(spec.layers())
                    || hasLatentMla(spec.nextnLayers());
            if (shared && cfg.typeK() != cfg.typeV()) {
                throw new IllegalArgumentException(spec.arch().ggufName() + ": model does not support different K ("
                        + cfg.typeK().name().toLowerCase(Locale.ROOT) + ") and V ("
                        + cfg.typeV().name().toLowerCase(Locale.ROOT) + ") cache types; set "
                        + MemoryConfig.PROP_KV_TYPE + " (or --kv-type) for both");
            }
        }

        private static boolean hasLatentMla(List<LayerSpec> layers) {
            for (LayerSpec l : layers) {
                if (l.mla() != null && l.mla().latentCache()) {
                    return true;
                }
            }
            return false;
        }

        /** {@link #storeType} of the configured K type. */
        private KvDType storeK(int width, int segment, String what) {
            return storeType(cfg.typeK(), width, segment, what, "K", MemoryConfig.PROP_KV_TYPE_K, "--kv-type-k");
        }

        /** {@link #storeType} of the configured V type. */
        private KvDType storeV(int width, int segment, String what) {
            return storeType(cfg.typeV(), width, segment, what, "V", MemoryConfig.PROP_KV_TYPE_V, "--kv-type-v");
        }

        /**
         * The configured type of a cache whose rows hold {@code width} values addressed in units of {@code segment}
         * values (a head, or the latent prefix). A block type must tile both exactly, as llama.cpp refuses a quantized
         * K or V type whose block size does not divide the head size (llama-context.cpp:4090-4110).
         *
         * @throws IllegalArgumentException when a block type does not tile the row or the unit
         */
        private KvDType storeType(KvDType wanted, int width, int segment, String what, String side, String property,
                                  String option) {
            if (!wanted.quantized() || (segment > 0 && wanted.supportsWidth(width) && wanted.supportsWidth(segment))) {
                return wanted;
            }
            throw new IllegalArgumentException(spec.arch().ggufName() + ": " + side + " cache type "
                    + wanted.name().toLowerCase(Locale.ROOT) + " with block size " + wanted.blockSize()
                    + " does not divide the " + what + " cache (rows of " + width + " values addressed in units of "
                    + segment + "); set " + property + " (or " + option + ") to f32, f16, bf16 or a block type that"
                    + " divides it");
        }

        private String layerName(LayerSpec l) {
            return spec.arch().ggufName() + " layer " + l.il();
        }
    }
}

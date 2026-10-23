package com.eonmux.neuralfree.gguf.spec.resolve;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.ArchEntry;
import com.eonmux.neuralfree.gguf.arch.ArchRegistry;
import com.eonmux.neuralfree.gguf.arch.MemoryKind;
import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.arch.ResidualKind;
import com.eonmux.neuralfree.gguf.arch.UnsupportedArchException;
import com.eonmux.neuralfree.gguf.hparams.ArchHparamsHook;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.hparams.HparamsLoader;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.spec.AttentionSpec;
import com.eonmux.neuralfree.gguf.spec.DsaSpec;
import com.eonmux.neuralfree.gguf.spec.Dsv4Spec;
import com.eonmux.neuralfree.gguf.spec.LayerSchedule;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.MlaSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.MsaSpec;
import com.eonmux.neuralfree.gguf.spec.RecurrentSpec;
import com.eonmux.neuralfree.gguf.spec.ResidualSpec;
import com.eonmux.neuralfree.gguf.spec.RopeSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorNames;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.function.Predicate;

/**
 * Assembles the {@link ModelSpec} of a GGUF file:
 * <ol>
 *   <li>architecture resolution from {@code general.architecture} and the {@link ArchRegistry} entry, rejecting
 *       {@link com.eonmux.neuralfree.gguf.arch.SupportClass#NA} ({@link ModelSpecValidator#checkSupported});</li>
 *   <li>{@link HparamsLoader} with the entry's hook; for deepseek4 the loader's NextN probe runs first: when
 *       {@code blk.<n_layer>.nextn.eh_proj.weight} is absent the NextN blocks are dropped and every block is a
 *       trunk block (models/deepseek4.cpp:20-25);</li>
 *   <li>the required tensors of the architecture's list, no tensor the architecture never loads
 *       ({@link TensorClaims}) and the graph's hparams asserts ({@link ModelSpecValidator});</li>
 *   <li>the resolvers of every trunk slot, NextN block and T5 stack layer, the embedding and the head;</li>
 *   <li>tensor shapes against the resolved spec ({@link ModelSpecValidator#checkShapes}).</li>
 * </ol>
 *
 * <p><b>Trunk.</b> The trunk has {@link ScheduleResolver#trunkLayers()} slots in {@link ScheduleResolver#schedule()}
 * order. Slot {@code s} reads the weights and per-layer hparams of {@code wl = schedule.weightLayer(s)} (looped
 * nanbeige and hrm_text schedules alias slots onto physical blocks); its {@link LayerSpec#il()} is {@code s} and an
 * owned KV cache is the slot's own. The NextN blocks follow the trunk ({@code il = trunk + i}). T5 keeps its encoder
 * stack ({@code enc.blk}, {@code n_layer} layers) apart and runs the decoder stack ({@code dec.blk},
 * {@code dec_n_layer} layers, layer-0 head counts and dims, models/t5.cpp:111-114) as the trunk; t5encoder runs its
 * encoder stack as the trunk.
 *
 * <p><b>Mixer per layer</b>, by the first rule that applies:
 * <ul>
 *   <li>wavtokenizer-dec: none. Its graph runs the posnet and ConvNeXt stacks and never reads a {@code blk} block
 *       (models/wavtokenizer-dec.cpp:118-262), so its block slots are pass-through layers without mixer or FFN;</li>
 *   <li>a recurrent layer ({@code hparams.is_recr}): its {@link RecurrentResolver#kind()}; falcon-h1 adds the
 *       parallel attention branch of every block (models/falcon-h1.cpp);</li>
 *   <li>deepseek4: {@link MixerKind#DSV4};</li>
 *   <li>MiniMax-M3: {@link MixerKind#MSA} over attention and its {@link MsaSpec} (dense lead layers included);</li>
 *   <li>a layer with {@code attn_kv_a_mqa}: MLA, as {@link MixerKind#DSA_MLA} with a lightning indexer, or
 *       {@link MixerKind#MSA} with a QSA k-pool indexer (glm5-next); the layer carries the MLA attention facet
 *       ({@link AttentionResolver#mlaFacet});</li>
 *   <li>Nemotron-H blocks with a feed-forward length (and no recurrent mixer) are FFN-only,
 *       {@link MixerKind#NONE}; the others are attention blocks (models/nemotron-h.cpp:227-235);</li>
 *   <li>otherwise softmax attention ({@link AttentionResolver#attentionKind(int)}: deci's attention-free and
 *       linear-replace layers), as {@link MixerKind#MSA} under a QSA k-pool indexer (qwen4exp).</li>
 * </ul>
 * The FFN follows {@link FfnResolver} (tensor presence decides dense, MoE and none; MoE is per layer by
 * {@code ffn_gate_inp} for jamba, mimo2, step35, hy_v3 and gemma4). Layer extras are listed in the order their
 * consumers require (see {@link #extras}). The LFM2 decision graph runs without memory
 * (llama-model.cpp:2781-2784). Stateless; each build reads only its arguments.
 */
public final class ModelSpecBuilder {

    private final ModelHparams hp;
    private final TensorCatalog tc;
    private final Arch arch;
    private final ScheduleResolver schedule;
    private final RopeResolver rope;
    private final NormResolver norm;
    private final AttentionResolver attention;
    private final ResidualResolver residual;
    private final FfnResolver ffn;
    private final MlaResolver mla;
    private final RecurrentResolver recurrent;
    private final SparseResolver sparse;

    private ModelSpecBuilder(ModelHparams hp, TensorCatalog tc, MemoryConfig cfg, int nCtxSeq) {
        this.hp = hp;
        this.tc = tc;
        this.arch = hp.arch();
        this.schedule = new ScheduleResolver(hp, tc);
        this.rope = new RopeResolver(hp, tc, nCtxSeq);
        this.norm = new NormResolver(hp, tc);
        this.attention = new AttentionResolver(hp, tc, rope, norm);
        this.residual = new ResidualResolver(hp, tc);
        this.ffn = new FfnResolver(hp, tc);
        this.mla = new MlaResolver(hp, tc, cfg);
        this.recurrent = new RecurrentResolver(hp, tc);
        this.sparse = new SparseResolver(hp, tc);
    }

    /**
     * The spec with the memory configuration of the system properties ({@link MemoryConfig#fromSystemProperties()}).
     *
     * @see #build(GgufKv, TensorCatalog, MemoryConfig, int)
     */
    public static ModelSpec build(GgufKv kv, TensorCatalog tc) {
        return build(kv, tc, MemoryConfig.fromSystemProperties());
    }

    /**
     * The spec for a session of {@code cfg}; the per-sequence context that picks the long or short RoPE factors is
     * {@code cfg.nCtx()}, or {@code n_ctx_train} with an automatic context.
     *
     * @see #build(GgufKv, TensorCatalog, MemoryConfig, int)
     */
    public static ModelSpec build(GgufKv kv, TensorCatalog tc, MemoryConfig cfg) {
        Objects.requireNonNull(cfg, "cfg");
        return build(kv, tc, cfg, 0);
    }

    /**
     * The spec for a session of {@code cfg} and per-sequence context {@code nCtxSeq}.
     *
     * @param cfg     memory configuration; its MLA cache mode decides latent or decompressed MLA caches
     * @param nCtxSeq per-sequence context ({@code cparams.n_ctx_seq}), or 0 to derive it from {@code cfg}
     * @throws UnsupportedArchException when the architecture is unknown, unregistered or not supported
     * @throws GgufFormatException      when metadata is missing or malformed, the tensor catalog belongs to another
     *                                  architecture, a graph assert fails or a tensor shape disagrees with the hparams
     * @throws com.eonmux.neuralfree.gguf.spec.MissingTensorException listing every missing required tensor
     * @throws com.eonmux.neuralfree.gguf.spec.UnusedTensorException listing every tensor the architecture never
     *                                  loads ({@link TensorClaims})
     * @throws UnsupportedOperationException for a file holding only the MTP blocks of its model
     *                                  ({@link ModelSpecValidator#isMtpOnly})
     * @throws IllegalStateException    for a layer layout no spec describes (named by the resolvers)
     * @throws IllegalArgumentException for a negative {@code nCtxSeq} or inconsistent resolved values
     */
    public static ModelSpec build(GgufKv kv, TensorCatalog tc, MemoryConfig cfg, int nCtxSeq) {
        Objects.requireNonNull(kv, "kv");
        Objects.requireNonNull(tc, "tc");
        Objects.requireNonNull(cfg, "cfg");
        if (nCtxSeq < 0) {
            throw new IllegalArgumentException("per-sequence context must be >= 0 (0 derives it), got " + nCtxSeq);
        }
        ArchEntry entry = resolveEntry(kv, tc);
        ModelSpecValidator.checkSupported(entry);
        ModelHparams hp = HparamsLoader.load(kv, withNextnProbe(entry, tc));
        ModelSpecValidator.requireTensors(hp, tc, entry.tensors());
        TensorClaims.requireAllClaimed(hp, tc, entry.tensors());
        refuseMtpOnly(hp, tc);
        ModelSpecValidator.checkGraphAsserts(hp);
        int ctx = nCtxSeq > 0 ? nCtxSeq : cfg.autoContext() ? hp.nCtxTrain() : cfg.nCtx();
        ModelSpec spec = new ModelSpecBuilder(hp, tc, cfg, ctx).assemble();
        ModelSpecValidator.checkShapes(spec, ModelSpecValidator.vocabTokens(kv, hp));
        return spec;
    }

    /**
     * Refuses an MTP-only file as a model of its own once its tensors passed the loader checks. llama.cpp loads it as
     * the draft-mtp context of its target (common/speculative.cpp:2414-2420 detects it, :2655-2690 creates the MTP
     * context over the target context); without trunk blocks it has no model of its own to run.
     * {@link #buildMtpDraft} builds it as the MTP drafter of a target.
     *
     * @throws UnsupportedOperationException when {@link ModelSpecValidator#isMtpOnly} holds
     */
    static void refuseMtpOnly(ModelHparams hp, TensorCatalog tc) {
        if (ModelSpecValidator.isMtpOnly(hp, tc)) {
            throw new UnsupportedOperationException(hp.arch().ggufName() + ": the file holds only the "
                    + (hp.nLayerAll() - hp.nLayer()) + " MTP block(s) of its model (no "
                    + ModelSpecValidator.trunkProbe(hp.arch()) + "); like llama.cpp, NeuralFree runs such a file only"
                    + " as the MTP drafter of its target model: open the target with this file as its draft model");
        }
    }

    /**
     * True when the file holds only the MTP blocks of its model ({@link ModelSpecValidator#isMtpOnly} over the
     * hparams its architecture's loader reads). Such a file cannot run as a model of its own ({@link #build}); it
     * drafts as an MTP draft file ({@link #isNextnDraft}).
     *
     * @throws UnsupportedArchException when the architecture is unknown, unregistered or not supported
     * @throws GgufFormatException      when metadata is missing or malformed
     */
    public static boolean isMtpOnly(GgufKv kv, TensorCatalog tc) {
        Objects.requireNonNull(kv, "kv");
        Objects.requireNonNull(tc, "tc");
        ArchEntry entry = resolveEntry(kv, tc);
        ModelSpecValidator.checkSupported(entry);
        return ModelSpecValidator.isMtpOnly(HparamsLoader.load(kv, withNextnProbe(entry, tc)), tc);
    }

    /**
     * True when a draft file is an MTP draft, as llama.cpp's {@code common_speculative_types_from_gguf} detects it
     * (common/speculative.cpp:2414-2420): its architecture is not {@code dflash} and its last block
     * {@code blk.<block_count - 1>} holds {@code nextn.eh_proj.weight}. The trunk is not consulted: a file with or
     * without its trunk blocks drafts with its NextN blocks only ({@link #buildMtpDraft}). A file without
     * {@code general.architecture} or {@code block_count} is not one.
     */
    public static boolean isNextnDraft(GgufKv kv, TensorCatalog tc) {
        Objects.requireNonNull(kv, "kv");
        Objects.requireNonNull(tc, "tc");
        String name = kv.archName();
        if (name == null || Arch.fromGgufName(name) == Arch.DFLASH) {
            return false;
        }
        int blocks = kv.u32(GgufKeys.BLOCK_COUNT, 0);
        return blocks > 0
                && tc.has(TensorNames.layerName(blocks - 1, TensorNames.weight(TensorNames.NEXTN_EH_PROJ)));
    }

    /**
     * The spec of an MTP draft file ({@link #isNextnDraft}) as the MTP drafter of {@code target}. llama.cpp loads the
     * file as a model and runs only its MTP graph against the target's hidden states
     * (common/speculative.cpp:2655-2690, {@code LLAMA_CONTEXT_TYPE_MTP}); a file holding only the MTP blocks loads
     * because {@code nextn_flags} makes its trunk tensors optional (llama-model.cpp:3901-3920). Here the drafter reads
     * none of the file's trunk, whether the file holds it or not: the trunk tensors are not required, the trunk slots
     * of the spec are the target's layers (the drafter never runs them), and the NextN blocks, the embedding stage and
     * the head are resolved from the file, the NextN blocks keeping their layer index {@code n_layer + i}. The file's
     * trunk tensors present are still matched against the architecture's tensor list ({@link TensorClaims}), and
     * {@link #mtpDraftReads} names the tensors the drafter places in memory.
     *
     * @param cfg memory configuration of the drafter's spec (its MLA cache form and the context that picks the RoPE
     *            factors): the target's draft-context settings with the target plan's context
     * @throws IllegalArgumentException when the file holds no NextN block or is not an MTP draft of the target's
     *                                  model ({@link #isMtpDraftOf}: architecture, trunk depth and schedule,
     *                                  {@code n_embd}, vocabulary), naming the difference
     * @throws UnsupportedArchException when the architecture is unknown, unregistered or not supported
     * @throws GgufFormatException      as {@link #build}
     * @throws com.eonmux.neuralfree.gguf.spec.MissingTensorException as {@link #build}, outside the trunk
     * @throws com.eonmux.neuralfree.gguf.spec.UnusedTensorException as {@link #build}
     */
    public static ModelSpec buildMtpDraft(GgufKv kv, TensorCatalog tc, MemoryConfig cfg, ModelSpec target) {
        Objects.requireNonNull(kv, "kv");
        Objects.requireNonNull(tc, "tc");
        Objects.requireNonNull(cfg, "cfg");
        Objects.requireNonNull(target, "target");
        ArchEntry entry = resolveEntry(kv, tc);
        ModelSpecValidator.checkSupported(entry);
        if (!isNextnDraft(kv, tc)) {
            throw new IllegalArgumentException(entry.arch().ggufName() + ": " + noNextnBlock(kv));
        }
        ModelHparams hp = HparamsLoader.load(kv, withNextnProbe(entry, tc));
        ModelSpecValidator.requireTensors(hp, tc, entry.tensors(), true);
        TensorClaims.requireAllClaimed(hp, tc, entry.tensors());
        ModelSpecValidator.checkGraphAsserts(hp);
        checkMtpTarget(hp, tc, target);
        int ctx = cfg.autoContext() ? hp.nCtxTrain() : cfg.nCtx();
        ModelSpec spec = new ModelSpecBuilder(hp, tc, cfg, ctx).assembleMtpDraft(target);
        ModelSpecValidator.checkMtpDraftShapes(spec, ModelSpecValidator.vocabTokens(kv, hp));
        return spec;
    }

    /**
     * The stored tensors of an MTP draft file that its drafter ({@link #buildMtpDraft}) reads, and so places in
     * memory: every tensor except those of the file's trunk blocks ({@link ModelSpecValidator#isTrunkTensor}).
     *
     * @param mtp the drafter's spec from {@link #buildMtpDraft}
     */
    public static Predicate<String> mtpDraftReads(ModelSpec mtp) {
        Objects.requireNonNull(mtp, "mtp");
        return trunkExcluded(mtp.hparams().nLayer());
    }

    /**
     * {@link #mtpDraftReads} of an MTP draft file ({@link #isNextnDraft}) from its metadata alone, before a target
     * spec exists.
     *
     * @throws IllegalArgumentException when the file is not an MTP draft file
     * @throws UnsupportedArchException when the architecture is unknown, unregistered or not supported
     * @throws GgufFormatException      when metadata is missing or malformed
     */
    public static Predicate<String> mtpDraftReads(GgufKv kv, TensorCatalog tc) {
        Objects.requireNonNull(kv, "kv");
        Objects.requireNonNull(tc, "tc");
        ArchEntry entry = resolveEntry(kv, tc);
        ModelSpecValidator.checkSupported(entry);
        if (!isNextnDraft(kv, tc)) {
            throw new IllegalArgumentException(entry.arch().ggufName() + ": the draft file holds no NextN block");
        }
        return trunkExcluded(HparamsLoader.load(kv, withNextnProbe(entry, tc)).nLayer());
    }

    private static Predicate<String> trunkExcluded(int nLayer) {
        return name -> !ModelSpecValidator.isTrunkTensor(name, nLayer);
    }

    /**
     * True when a draft file is an MTP draft of {@code target}'s model: an MTP draft file ({@link #isNextnDraft}) of
     * the target's architecture, trunk depth and schedule, {@code n_embd} and vocabulary, which
     * {@link #buildMtpDraft} accepts for that target ({@link #mtpDraftMismatch} is null). A file with NextN blocks
     * that fails it is another model of the family, which drafts as a separate draft model.
     *
     * @throws UnsupportedArchException when an MTP draft file's architecture is unknown, unregistered or not supported
     * @throws GgufFormatException      when an MTP draft file's metadata is missing or malformed
     * @throws IllegalStateException    when an MTP draft file's layer schedule is inconsistent with its hparams
     */
    public static boolean isMtpDraftOf(GgufKv kv, TensorCatalog tc, ModelSpec target) {
        return mtpDraftMismatch(kv, tc, target) == null;
    }

    /**
     * Why a draft file is not an MTP draft of {@code target}'s model ({@link #isMtpDraftOf}), or null when it is: it
     * holds no NextN block, or its architecture, trunk depth, trunk schedule, {@code n_embd} or vocabulary differs from
     * the target's. llama.cpp's MTP drafter asserts only the hidden width (common/speculative.cpp:1477-1479); the
     * drafter here also runs its NextN blocks at layer indices {@code n_layer + i} over the target's trunk slots, so
     * the trunk must match too.
     *
     * @throws UnsupportedArchException when an MTP draft file's architecture is unknown, unregistered or not supported
     * @throws GgufFormatException      when an MTP draft file's metadata is missing or malformed
     * @throws IllegalStateException    when an MTP draft file's layer schedule is inconsistent with its hparams
     */
    public static String mtpDraftMismatch(GgufKv kv, TensorCatalog tc, ModelSpec target) {
        Objects.requireNonNull(kv, "kv");
        Objects.requireNonNull(tc, "tc");
        Objects.requireNonNull(target, "target");
        if (!isNextnDraft(kv, tc)) {
            return noNextnBlock(kv);
        }
        ArchEntry entry = resolveEntry(kv, tc);
        ModelSpecValidator.checkSupported(entry);
        return mtpTargetMismatch(HparamsLoader.load(kv, withNextnProbe(entry, tc)), tc, target);
    }

    /** The reason a draft file without the {@link #isNextnDraft} probe tensor is not an MTP draft file. */
    private static String noNextnBlock(GgufKv kv) {
        return "the draft file holds no NextN block (no " + TensorNames.layerName(
                Math.max(0, kv.u32(GgufKeys.BLOCK_COUNT, 0) - 1), TensorNames.weight(TensorNames.NEXTN_EH_PROJ)) + ")";
    }

    /**
     * Why an MTP draft file of hparams {@code hp} does not draft for {@code target}, or null when it does: same
     * architecture, trunk depth, trunk schedule (slot count and the weight layer of every slot), width and vocabulary.
     */
    private static String mtpTargetMismatch(ModelHparams hp, TensorCatalog tc, ModelSpec target) {
        ModelHparams t = target.hparams();
        String name = hp.arch().ggufName();
        if (hp.arch() != target.arch()) {
            return "the MTP draft file is " + name + ", the target " + target.arch().ggufName();
        }
        if (hp.nLayer() != target.nLayer()) {
            return name + ": the MTP draft file follows " + hp.nLayer() + " trunk layers, the target has "
                    + target.nLayer();
        }
        if (hp.nEmbd() != t.nEmbd()) {
            return name + ": the MTP draft file has n_embd " + hp.nEmbd() + ", the target " + t.nEmbd();
        }
        if (hp.nVocab() != t.nVocab()) {
            return name + ": the MTP draft file has a vocabulary of " + hp.nVocab() + " tokens, the target "
                    + t.nVocab();
        }
        LayerSchedule s = new ScheduleResolver(hp, tc).schedule();
        List<LayerSpec> trunk = target.layers();
        if (s.slots() != trunk.size()) {
            return name + ": the MTP draft file schedules " + s.slots() + " trunk slots, the target " + trunk.size();
        }
        for (int slot = 0; slot < trunk.size(); slot++) {
            if (trunk.get(slot).weightLayer() != s.weightLayer(slot)) {
                return name + ": trunk slot " + slot + " of the target uses block " + trunk.get(slot).weightLayer()
                        + ", the MTP draft file schedules block " + s.weightLayer(slot);
            }
        }
        return null;
    }

    /**
     * Refuses an MTP draft file that is not of {@code target}'s model ({@link #mtpTargetMismatch}).
     *
     * @throws IllegalArgumentException naming the first difference
     */
    private static void checkMtpTarget(ModelHparams hp, TensorCatalog tc, ModelSpec target) {
        String mismatch = mtpTargetMismatch(hp, tc, target);
        if (mismatch != null) {
            throw new IllegalArgumentException(mismatch);
        }
    }

    /**
     * The registry entry named by {@code general.architecture}.
     *
     * @throws GgufFormatException      when the key is missing or the catalog was built for another architecture
     * @throws UnsupportedArchException when the name is not a llama.cpp architecture or has no table entry
     */
    static ArchEntry resolveEntry(GgufKv kv, TensorCatalog tc) {
        String name = kv.archName();
        if (name == null) {
            throw new GgufFormatException("required metadata key 'general.architecture' is missing");
        }
        Arch arch = Arch.fromGgufName(name);
        if (arch == Arch.UNKNOWN) {
            throw new UnsupportedArchException(arch, "unknown architecture '" + name
                    + "': general.architecture does not name a llama.cpp architecture");
        }
        if (!name.equals(tc.archName())) {
            throw new GgufFormatException("tensor catalog of architecture '" + tc.archName()
                    + "' does not belong to metadata of architecture '" + name + "'");
        }
        return ArchRegistry.get(arch);
    }

    /**
     * deepseek4's loader probe (models/deepseek4.cpp:20-25): with NextN blocks declared but no
     * {@code blk.<n_layer_all - n_nextn>.nextn.eh_proj.weight}, {@code n_layer_nextn} becomes 0 before the
     * architecture's keys are read, so every block is a trunk block. Other architectures keep their entry; their
     * absent NextN blocks are excluded by {@link ScheduleResolver}.
     */
    static ArchEntry withNextnProbe(ArchEntry entry, TensorCatalog tc) {
        if (entry.arch() != Arch.DEEPSEEK4) {
            return entry;
        }
        ArchHparamsHook hook = entry.hook();
        ArchHparamsHook probed = (b, kv) -> {
            if (b.nextnLayers() > 0) {
                int main = b.nLayerAll() - b.nextnLayers();
                String probe = TensorNames.layerName(main, TensorNames.weight(TensorNames.NEXTN_EH_PROJ));
                if (!tc.has(probe)) {
                    b.layerCounts(b.nLayerAll(), 0);
                }
            }
            hook.apply(b, kv);
        };
        return ArchEntry.of(entry.traits(), probed, entry.tensors());
    }

    // ================================================================== assembly

    /**
     * {@link #buildMtpDraft}: the target's trunk slots, then this file's NextN blocks, embedding and head. The trunk
     * schedule was matched against the target's ({@link #checkMtpTarget}).
     */
    private ModelSpec assembleMtpDraft(ModelSpec target) {
        if (schedule.nextnLayers() == 0) {
            throw new IllegalArgumentException(arch.ggufName() + ": the MTP draft file declares no NextN block"
                    + " (nextn_predict_layers is 0)");
        }
        ModelSpec.Builder b = ModelSpec.builder(hp, tc);
        b.schedule(schedule.schedule());
        for (LayerSpec slot : target.layers()) {
            b.layer(slot);
        }
        int first = schedule.trunkLayers();
        for (int i = 0; i < schedule.nextnLayers(); i++) {
            b.nextnLayer(mainLayer(first + i, first + i, false, false));
        }
        b.embedding(new EmbeddingResolver(hp, tc).resolve());
        b.head(new HeadResolver(hp, tc).resolve());
        return b.build();
    }

    private ModelSpec assemble() {
        ModelSpec.Builder b = ModelSpec.builder(hp, tc);
        if (arch == Arch.T5) {
            for (int i = 0; i < hp.nLayer(); i++) {
                b.encoderLayer(encoderLayer(i));
            }
            int nDec = hp.decNLayer();
            if (nDec <= 0) {
                throw new GgufFormatException(arch.ggufName() + ": the decoder has " + nDec + " blocks");
            }
            b.schedule(LayerSchedule.linear(nDec));
            for (int i = 0; i < nDec; i++) {
                b.layer(decoderLayer(i));
            }
        } else if (arch == Arch.T5ENCODER) {
            b.schedule(LayerSchedule.linear(hp.nLayer()));
            for (int i = 0; i < hp.nLayer(); i++) {
                b.layer(encoderLayer(i));
            }
        } else {
            LayerSchedule s = schedule.schedule();
            b.schedule(s);
            for (int slot = 0; slot < s.slots(); slot++) {
                b.layer(mainLayer(slot, s.weightLayer(slot), true, s.normAfterSlot(slot)));
            }
            int trunk = schedule.trunkLayers();
            for (int i = 0; i < schedule.nextnLayers(); i++) {
                b.nextnLayer(mainLayer(trunk + i, trunk + i, false, false));
            }
        }
        b.embedding(new EmbeddingResolver(hp, tc).resolve());
        b.head(new HeadResolver(hp, tc).resolve());
        if (arch == Arch.LFM2 && !hp.causal() && hp.nLayerDecision() > 0) {
            b.memoryKind(MemoryKind.NONE);
        }
        return b.build();
    }

    /** T5 / t5encoder encoder layer {@code i}: bidirectional attention without RoPE and a dense FFN. */
    private LayerSpec encoderLayer(int i) {
        String stack = TensorNames.STACK_ENC;
        FfnResolver.Result f = ffn.resolve(stack, i);
        return LayerSpec.builder(i, attention.attentionKind(stack, i), f.kind())
                .stack(stack)
                .norm(norm.resolve(stack, i))
                .residual(residual.resolve(stack, i))
                .attention(attention.resolve(stack, i))
                .ffn(f.ffn())
                .moe(f.moe())
                .build();
    }

    /** T5 decoder layer {@code i}: causal self-attention, cross-attention over the encoder output, dense FFN. */
    private LayerSpec decoderLayer(int i) {
        String stack = TensorNames.STACK_DEC;
        FfnResolver.Result f = ffn.resolve(stack, i);
        return LayerSpec.builder(i, attention.attentionKind(stack, i), f.kind())
                .stack(stack)
                .norm(norm.resolve(stack, i))
                .residual(residual.resolve(stack, i))
                .attention(attention.resolve(stack, i))
                .crossAttention(attention.crossAttention(i))
                .ffn(f.ffn())
                .moe(f.moe())
                .build();
    }

    /**
     * Main-stack layer at slot {@code slot} over the weights and hparams of block {@code wl}.
     *
     * @param trunk    a trunk slot (false for a NextN block)
     * @param loopNorm the schedule applies {@code output_norm} after this slot
     */
    private LayerSpec mainLayer(int slot, int wl, boolean trunk, boolean loopNorm) {
        Mixer m = mixer(wl);
        FfnResolver.Result f = ffn.resolve(wl);
        ResidualSpec r = residual.resolve(wl);
        LayerSpec.Builder b = LayerSpec.builder(slot, m.kind, f.kind())
                .weightLayer(wl)
                .norm(norm.resolve(wl))
                .residual(r)
                .attention(ownedBySlot(m.attention, slot, wl))
                .parallelAttention(m.parallel)
                .mla(m.mla)
                .dsa(m.dsa)
                .msa(m.msa)
                .dsv4(m.dsv4)
                .recurrent(m.recurrent)
                .rope(m.rope)
                .ffn(f.ffn())
                .moe(f.moe());
        if (slot != wl && (m.dsa != null || m.dsv4 != null)) {
            throw new IllegalStateException(arch.ggufName() + ": slot " + slot + " aliases block " + wl
                    + " but sparse indexers are not built for looped schedules");
        }
        for (String id : extras(wl, trunk, loopNorm, r, f)) {
            b.extra(id);
        }
        return b.build();
    }

    /**
     * The attention of a slot that aliases block {@code wl}: a cache owned by block {@code o} becomes the cache of
     * slot {@code slot + (o - wl)}, so every loop pass keeps its own KV rows (nanbeige expands its per-layer arrays
     * over the loop slots and keeps a KV index per slot, models/nanbeige.cpp:20-31, 66-68).
     */
    private static AttentionSpec ownedBySlot(AttentionSpec a, int slot, int wl) {
        if (a == null || slot == wl) {
            return a;
        }
        return a.toBuilder().kvOwnerLayer(slot + (a.kvOwnerLayer() - wl)).build();
    }

    // ================================================================== mixer

    /** The mixer facets of one main-stack layer. Immutable. */
    private static final class Mixer {
        private final MixerKind kind;
        private final AttentionSpec attention;
        private final boolean parallel;
        private final MlaSpec mla;
        private final DsaSpec dsa;
        private final MsaSpec msa;
        private final Dsv4Spec dsv4;
        private final RecurrentSpec recurrent;
        private final RopeSpec rope;

        private Mixer(MixerKind kind, AttentionSpec attention, boolean parallel, MlaSpec mla, DsaSpec dsa, MsaSpec msa,
                      Dsv4Spec dsv4, RecurrentSpec recurrent, RopeSpec rope) {
            this.kind = kind;
            this.attention = attention;
            this.parallel = parallel;
            this.mla = mla;
            this.dsa = dsa;
            this.msa = msa;
            this.dsv4 = dsv4;
            this.recurrent = recurrent;
            this.rope = rope;
        }

        /** A softmax-attention mixer (or none, for a null {@code attention}), optionally under a sparse facet. */
        static Mixer attention(MixerKind kind, AttentionSpec attention, DsaSpec dsa, MsaSpec msa, RopeSpec rope) {
            return new Mixer(kind, attention, false, null, dsa, msa, null, null, rope);
        }
    }

    private Mixer mixer(int il) {
        if (arch == Arch.WAVTOKENIZER_DEC) {
            return Mixer.attention(MixerKind.NONE, null, null, null, null);
        }
        if (recurrent.isRecurrentLayer(il)) {
            RecurrentSpec rec = recurrent.resolve(il);
            if (arch == Arch.FALCON_H1) {
                AttentionSpec a = attention.resolve(il);
                return new Mixer(recurrent.kind(), a, true, null, null, null, null, rec, attentionRope(il, a));
            }
            return new Mixer(recurrent.kind(), null, false, null, null, null, null, rec, null);
        }
        if (arch == Arch.DEEPSEEK4) {
            Dsv4Spec d = sparse.dsv4(il);
            RopeSpec r = rope.appliesRope(il) ? rope.forLayer(il, d.ropeDim(), d.headDim() - d.ropeDim()) : null;
            return new Mixer(MixerKind.DSV4, null, false, null, null, null, d, null, r);
        }
        MsaSpec msa = sparse.msa(il);
        if (msa != null) {
            AttentionSpec a = attention.resolve(il);
            return Mixer.attention(MixerKind.MSA, a, null, msa, attentionRope(il, a));
        }
        if (mla.isMlaLayer(il)) {
            return mlaMixer(il);
        }
        if (nemotronFfnOnlyLayer(il)) {
            return Mixer.attention(MixerKind.NONE, null, null, null, null);
        }
        DsaSpec dsa = sparse.dsa(il);
        MixerKind kind = attention.attentionKind(il);
        if (dsa != null && dsa.kpool() <= 0) {
            throw new IllegalStateException(arch.ggufName() + ": layer " + il + " has a lightning indexer over"
                    + " softmax attention, which llama.cpp builds only over MLA");
        }
        if (dsa != null && kind != MixerKind.ATTENTION) {
            throw new IllegalStateException(arch.ggufName() + ": layer " + il + " pairs a k-pool indexer with "
                    + kind);
        }
        AttentionSpec a = attention.resolve(il);
        return Mixer.attention(dsa != null ? MixerKind.MSA : kind, a, dsa, null, attentionRope(il, a));
    }

    /**
     * A Nemotron-H trunk layer with an FFN has no mixer (nemotron-h.cpp:226-234 selects attention by
     * {@code n_ff(il) == 0} inside the trunk loop only). The nemotron_h_moe NextN block always runs NoPE attention
     * before its MoE sub-layer (nemotron-h-moe.cpp {@code graph_mtp}), although its {@code feed_forward_length} entry
     * holds the expert width.
     */
    private boolean nemotronFfnOnlyLayer(int il) {
        return (arch == Arch.NEMOTRON_H || arch == Arch.NEMOTRON_H_MOE) && il < hp.nLayer() && hp.nFf(il) != 0;
    }

    /** MLA, DSA over MLA, or QSA over nope MLA (glm5-next), with the MLA attention facet. */
    private Mixer mlaMixer(int il) {
        MlaSpec spec = mla.resolve(il);
        DsaSpec dsa = sparse.dsa(il);
        MixerKind kind = dsa == null ? MixerKind.MLA : dsa.kpool() > 0 ? MixerKind.MSA : MixerKind.DSA_MLA;
        RopeSpec r = spec.ropeApplied() ? rope.forLayer(il, spec.headKRope(), spec.headKNope()) : null;
        return new Mixer(kind, attention.mlaFacet(il, spec), false, spec, dsa, null, null, null, r);
    }

    /** RoPE of a softmax-attention branch: its {@code nRot} leading dims, none for NoPE and linear-replace layers. */
    private RopeSpec attentionRope(int il, AttentionSpec a) {
        if (a == null || a.noRope() || a.linearReplace()) {
            return null;
        }
        return rope.forLayer(il, a.nRot(), 0);
    }

    // ================================================================== extras

    /**
     * Layer extras of block {@code wl}, in the order their consumers need them:
     * <ol>
     *   <li>{@code moe_layer_input} on the MoE layers of arctic (expert input from the layer input,
     *       models/arctic.cpp:136) and smallthinker (router on the layer input, models/smallthinker.cpp:109),
     *       first so it keeps the input before any other extra changes the stream;</li>
     *   <li>gemma3n trunk layers: {@code altup} then {@code laurel} (models/gemma3n.cpp);</li>
     *   <li>gemma4 with {@code per_layer_token_embd} and gemma-embedding2 trunk layers: {@code ple}
     *       (gemma4.cpp:376-403, gemma-embedding2.cpp:172-197);</li>
     *   <li>the qwen4exp PLE layer: {@code ngram_ple} before {@code hc} (qwen4exp.cpp:452-461);</li>
     *   <li>{@code hc} for hyper-connection residuals;</li>
     *   <li>kimi-k3 trunk layers with {@code attn_res_block_size > 0}: {@code attn_res} (kimi-k3.cpp:273-335);</li>
     *   <li>{@code resid_alpha_beta} (muse-spark) and {@code embd_skip} (talkie) from the residual spec;</li>
     *   <li>granite-switch trunk layers: {@code switched_lora} (granite-switch.cpp:115-130, 242-427);</li>
     *   <li>{@code loop_norm} on the slots the schedule closes with {@code output_norm} (nanbeige.cpp:164-170).</li>
     * </ol>
     */
    private List<String> extras(int wl, boolean trunk, boolean loopNorm, ResidualSpec r, FfnResolver.Result f) {
        List<String> ids = new ArrayList<>();
        if (f.moe() != null && (arch == Arch.ARCTIC || arch == Arch.SMALLTHINKER)) {
            ids.add(LayerSpec.Extras.MOE_LAYER_INPUT);
        }
        if (trunk && arch == Arch.GEMMA3N) {
            ids.add(LayerSpec.Extras.ALTUP);
            ids.add(LayerSpec.Extras.LAUREL);
        }
        if (trunk && usesPle()) {
            ids.add(LayerSpec.Extras.PLE);
        }
        if (arch == Arch.QWEN4EXP && hp.isPleLayer(wl)) {
            ids.add(LayerSpec.Extras.NGRAM_PLE);
        }
        if (r.kind() == ResidualKind.HYPER_CONNECTIONS) {
            ids.add(LayerSpec.Extras.HC);
        }
        if (trunk && arch == Arch.KIMI_K3 && hp.attnResBlockSize() > 0) {
            ids.add(LayerSpec.Extras.ATTN_RES);
        }
        if (r.alphaBeta()) {
            ids.add(LayerSpec.Extras.RESID_ALPHA_BETA);
        }
        if (r.embeddingSkip()) {
            ids.add(LayerSpec.Extras.EMBD_SKIP);
        }
        if (trunk && arch == Arch.GRANITE_SWITCH) {
            ids.add(LayerSpec.Extras.SWITCHED_LORA);
        }
        if (loopNorm) {
            ids.add(LayerSpec.Extras.LOOP_NORM);
        }
        return ids;
    }

    /**
     * Per-layer embeddings: gemma4 when the file has {@code per_layer_token_embd} (the graph tests the tensor,
     * gemma4.cpp:164-167), gemma-embedding2 always (gemma-embedding2.cpp:96-97).
     */
    private boolean usesPle() {
        if (arch == Arch.GEMMA_EMBEDDING2) {
            return true;
        }
        return arch == Arch.GEMMA4 && tc.get(TensorNames.weight(TensorNames.PER_LAYER_TOKEN_EMBD)) != null;
    }

    @Override
    public String toString() {
        return "ModelSpecBuilder{" + arch.ggufName() + '}';
    }
}

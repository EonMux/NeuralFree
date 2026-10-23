package com.eonmux.neuralfree.gguf.ops.flow;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.arch.ResidualKind;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuOps;
import com.eonmux.neuralfree.gguf.engine.DraftModel;
import com.eonmux.neuralfree.gguf.engine.EngineModel;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.engine.HiddenTap;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;
import com.eonmux.neuralfree.gguf.memory.MemoryPlanner;
import com.eonmux.neuralfree.gguf.memory.ModelMemory;
import com.eonmux.neuralfree.gguf.memory.SequenceState;
import com.eonmux.neuralfree.gguf.ops.flow.DraftModelLoader.Batch;
import com.eonmux.neuralfree.gguf.ops.flow.DraftModelLoader.Pick;
import com.eonmux.neuralfree.gguf.ops.flow.MtpHead.Rows;
import com.eonmux.neuralfree.gguf.ops.flow.MtpHead.Source;
import com.eonmux.neuralfree.gguf.ops.hc.CpuHyperConnections;
import com.eonmux.neuralfree.gguf.ops.hc.HcKernels;
import com.eonmux.neuralfree.gguf.ops.hc.HcSubBlock;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.ResidualSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.EnumSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.function.Predicate;

/**
 * Self-speculation with the target's NextN/MTP blocks (the {@code graph_mtp} heads of llama.cpp src/models, e.g.
 * qwen35.cpp:493-660, deepseek2.cpp:164-390, mimo2.cpp:260-398, step35.cpp:355-560, deepseek4.cpp:1366-1510,
 * qwen4exp.cpp:524-616, glm5-next.cpp:550-674), driven like llama.cpp common/speculative.cpp
 * {@code common_speculative_impl_draft_mtp} (1432-1876) without shared memory.
 *
 * <p>A head (NextN block {@code n_layer + j}): {@code e = embed(tok)} (from the block's {@code nextn.embed_tokens},
 * else the target's {@code token_embd}); {@code x = eh_proj . [enorm(e) | hnorm(h)]}; the block itself (run by the
 * layer executor over caches of its own, planned by {@link MemoryPlanner#planNextn(ModelSpec,
 * com.eonmux.neuralfree.gguf.memory.MemoryPlan, int)}); then the head norm (bailingmoe3's {@code layer_output_norm};
 * otherwise {@code nextn.shared_head_norm}, else mimo2's {@code layer_output_norm}, else {@code output_norm}) and the
 * head ({@code nextn.shared_head_head}, else the target's LM head; cohere2moe scales by its logit scale). The block's
 * output {@code h_nextn} feeds the next draft step: after the head norm, except mimo2 and step35 whose heads pass the
 * pre-norm stream.
 *
 * <p>Target hidden rows come from tap {@code n_layer} (the stream before the output head). Targets whose
 * {@code h_nextn} is the post-norm state get the target's output norm on the host first, so the heads always read
 * llama.cpp's {@code embeddings_nextn}.
 *
 * <p>The blocks, their embedding table, head norm and LM head are the target's own ({@link Source#TARGET}), or those
 * of an MTP draft file of the target's model ({@link Source#DRAFT}, with or without the file's trunk, which never
 * runs), which llama.cpp loads as the draft model of an MTP context over the target
 * (common/speculative.cpp:2655-2690): its NextN blocks, {@code token_embd}, {@code output_norm} and {@code output}
 * take the place of the target's, while the hidden rows, the target's output norm and the tap stay the target's. The
 * caches of every block are planned as a draft context ({@link MemoryPlanner#planNextn(ModelSpec,
 * com.eonmux.neuralfree.gguf.memory.MemoryPlan, int)}: F16 rows, no state snapshots), as llama.cpp creates the MTP
 * context from the speculative parameters (common/speculative.cpp:2606-2607, 2658-2669).
 *
 * <p>Hyper-connection targets ({@code hc} residual streams of {@code n_embd}):
 * <ul>
 *   <li>deepseek4 and qwen4exp read the flattened streams of the trunk output ({@code n_embd_out = hc * n_embd},
 *       deepseek4.cpp:1340-1347, qwen4exp.cpp:499-501) and run the head over streams: {@code enorm(e)} repeated
 *       for every stream, {@code hnorm} per stream (one shared vector on deepseek4, one per stream on qwen4exp),
 *       {@code eh_proj} per stream, the block with its hyper-connection sub-blocks, and {@code h_nextn} = the
 *       block's output streams. The logits collapse those streams first: deepseek4 through the model's HC head
 *       ({@code output_hc_*}) and then the head norm (deepseek4.cpp:1493-1501), qwen4exp through the block's own
 *       gated head ({@code nextn.hc_head_*}), which is also its final norm, into the target's LM head
 *       (qwen4exp.cpp:606-613). qwen4exp embeds with the target's {@code token_embd} only (qwen4exp.cpp:553-556);</li>
 *   <li>glm5-next reads the post-norm state of the collapsed trunk (stream mean, then {@code output_norm},
 *       glm5-next.cpp:785-798), computed here on the host, and runs a plain block without hyper-connections
 *       (glm5-next.cpp:612-663; its NextN blocks carry no {@code hc_*} tensors, glm5-next.cpp:85-92).</li>
 * </ul>
 *
 * <p>Protocol (positions are the target's): each target forward call over positions {@code [p0, p0 + m)} runs every
 * head over the pairs {@code (token[p], h[p - 1])}, writing that head's caches at every position, the first pair
 * taking the pending row of the previous call; the call's rows become the verify rows and the last one the pending
 * row. {@code accept(n)} makes verify row {@code min(n, rows - 1)} pending. Steps at a position first drop the head's
 * cache rows from that position on, so stale draft and verify rows are rewritten. A draft at {@code P = n_past}:
 * <ul>
 *   <li>one head (every architecture): starts from {@code (lastToken, pending)} at {@code P} and continues with each
 *       drafted token and the head's own {@code h_nextn} at {@code P + i};</li>
 *   <li>chained heads (hy-v3, mimo2 and step35 with several NextN blocks, speculative.cpp {@code chain_heads}):
 *       draft step {@code i} runs head {@code i} over the whole prefix {@code [lastToken, d_0 .. d_{i-1}]} at
 *       {@code P ..} with the hidden rows {@code [pending, h_0 .. h_{i-1}]}, where {@code h_k} is the last-row
 *       {@code h_nextn} of step {@code k}; a draft proposes at most one token per head.</li>
 * </ul>
 */
public final class NextnDraftModel implements DraftModel {

    /** Targets whose {@code h_nextn} is the output-norm state (the target graphs' {@code t_h_nextn}). */
    private static final Set<Arch> POST_NORM_TARGET = Collections.unmodifiableSet(EnumSet.of(Arch.DEEPSEEK2,
            Arch.DEEPSEEK32, Arch.GLM4_MOE, Arch.GLM_DSA, Arch.MELLUM, Arch.BAILINGMOE3, Arch.COHERE2MOE, Arch.HY_V3,
            Arch.QWEN3NEXT, Arch.QWEN35, Arch.QWEN35MOE, Arch.NEMOTRON_H_MOE, Arch.GLM5_NEXT));
    /** Targets and heads whose {@code h_nextn} is the pre-norm stream (mimo2.cpp:236-238, step35.cpp:339-341). */
    private static final Set<Arch> PRE_NORM = Collections.unmodifiableSet(EnumSet.of(Arch.MIMO2, Arch.STEP35));
    /**
     * MTP graphs that select their block by {@code nextn_layer_offset} without asserting a single block
     * (hy-v3.cpp:236, mimo2.cpp:265, step35.cpp:365); every other MTP graph asserts {@code n_layer_nextn == 1}.
     */
    private static final Set<Arch> CHAINED_HEADS = Collections.unmodifiableSet(EnumSet.of(Arch.HY_V3, Arch.MIMO2,
            Arch.STEP35));
    /**
     * MTP heads over the flattened hyper-connection streams of the trunk output, whose block keeps the streams
     * (deepseek4.cpp:1376, 1407-1427; qwen4exp.cpp:530, 581-588).
     */
    private static final Set<Arch> HC_STREAM = Collections.unmodifiableSet(EnumSet.of(Arch.DEEPSEEK4,
            Arch.QWEN4EXP));
    /**
     * Hyper-connection targets whose MTP head reads the stream mean after the output norm and whose block has no
     * hyper-connections (glm5-next.cpp:566, 785-798).
     */
    private static final Set<Arch> HC_MEAN_TARGET = Collections.unmodifiableSet(EnumSet.of(Arch.GLM5_NEXT));

    /** The tensors and layer one NextN block's head runs. Immutable. */
    private static final class Block {
        final LayerSpec layer;
        final String embedTable;
        final String ehProj;
        final String enorm;
        final String hnorm;
        /** Head norm; null when the stream collapse is the final norm (qwen4exp). */
        final String headNormWeight;
        final String headWeight;
        /** Stream collapse before the head norm; null for heads over one stream. */
        final HcSubBlock hcHead;

        Block(LayerSpec layer, String embedTable, String ehProj, String enorm, String hnorm, String headNormWeight,
              String headWeight, HcSubBlock hcHead) {
            this.layer = layer;
            this.embedTable = embedTable;
            this.ehProj = ehProj;
            this.enorm = enorm;
            this.hnorm = hnorm;
            this.headNormWeight = headNormWeight;
            this.headWeight = headWeight;
            this.hcHead = hcHead;
        }
    }

    private final MtpHead head;
    private final DraftModelLoader.Options options;
    /** The model whose NextN blocks, embedding and head run: the target, or an MTP draft file of its model. */
    private final ModelSpec spec;
    /** The target, whose trunk provides the hidden rows. */
    private final ModelSpec targetSpec;
    /** Where the tensors of {@link #spec} are read. */
    private final Source source;
    private final List<Block> blocks;
    private final boolean chained;
    private final int nEmbd;
    /** Streams of the head's hidden rows and block: hc for {@link #HC_STREAM}, else 1. */
    private final int streams;
    /** Floats per hidden row the head reads and its block maps: {@code streams * nEmbd}. */
    private final int hWidth;
    /** Floats per row at the target's tap {@code n_layer}. */
    private final int tapWidth;
    /** Streams the host averages before the target norm (glm5-next), else 1. */
    private final int meanStreams;
    /** True when {@code h_nextn} is the block output rather than the head norm's output. */
    private final boolean hFromBlock;
    private final int vocab;
    private final NormKind inNorm;
    private final float inEps;
    private final NormKind headNorm;
    private final float headEps;
    private final NormKind targetNormKind;
    private final float targetNormEps;
    private final boolean targetNormed;
    private final float logitScale;

    private ForwardBackend backend;
    private SequenceState targetSequence;
    private DraftModelLoader.TargetFeed feed;
    private MtpHead.Memory[] memories;
    private MtpHead.Stack[] stacks;
    private MtpHead.Collapse[] collapses;
    private int stepRows;
    private float[] targetNorm;
    private Rows hIn;
    private Rows embd;
    private Rows cat;
    private Rows x;
    private Rows xn;
    private float[] logits;

    private DraftModelLoader.HiddenRows hidden;

    /**
     * The target's own NextN blocks.
     *
     * @throws IllegalArgumentException      when the target has no NextN block, a block lacks a tensor, or the
     *                                       stream widths of the trunk and the block disagree
     * @throws UnsupportedOperationException for several blocks on an architecture whose MTP graph runs one, or an
     *                                       architecture without an MTP graph
     */
    NextnDraftModel(MtpHead head, DraftModelLoader.Options options) {
        this(head, Objects.requireNonNull(head, "head").target(), Source.TARGET, options);
    }

    /**
     * The NextN blocks of {@code model}: the target itself ({@link Source#TARGET}) or an MTP draft file of the
     * target's model ({@link Source#DRAFT}), whose trunk slots are the target's layers.
     *
     * @throws IllegalArgumentException      when the model has no NextN block, a block lacks a tensor, the stream
     *                                       widths of the trunk and the block disagree, or {@code model} is another
     *                                       architecture or trunk than the target
     * @throws UnsupportedOperationException for several blocks on an architecture whose MTP graph runs one, or an
     *                                       architecture without an MTP graph
     */
    NextnDraftModel(MtpHead head, ModelSpec model, Source source, DraftModelLoader.Options options) {
        this.head = Objects.requireNonNull(head, "head");
        this.options = Objects.requireNonNull(options, "options");
        this.spec = Objects.requireNonNull(model, "model");
        this.source = Objects.requireNonNull(source, "source");
        this.targetSpec = head.target();
        Arch arch = spec.arch();
        if (source == Source.TARGET && spec != targetSpec) {
            throw new IllegalArgumentException("the target's NextN blocks are described by the target's spec");
        }
        if (arch != targetSpec.arch() || spec.nLayer() != targetSpec.nLayer()) {
            throw new IllegalArgumentException("NextN blocks of " + arch.ggufName() + " after " + spec.nLayer()
                    + " trunk layers do not belong to the target " + targetSpec.arch().ggufName() + " ("
                    + targetSpec.nLayer() + " trunk layers)");
        }
        String name = arch.ggufName();
        boolean hcStream = HC_STREAM.contains(arch);
        if (!POST_NORM_TARGET.contains(arch) && !PRE_NORM.contains(arch) && !hcStream) {
            throw new UnsupportedOperationException(name + " has no NextN/MTP draft graph in llama.cpp");
        }
        List<LayerSpec> nextn = spec.nextnLayers();
        if (nextn.isEmpty()) {
            throw new IllegalArgumentException(name + " has no NextN blocks (nextn_predict_layers is 0 or the"
                    + " blocks are missing from the file)");
        }
        if (nextn.size() > 1 && !CHAINED_HEADS.contains(arch)) {
            throw new UnsupportedOperationException(name + " has " + nextn.size() + " NextN blocks; its MTP graph"
                    + " runs a single block (llama.cpp asserts n_layer_nextn == 1)");
        }
        if (targetSpec.layers().isEmpty()) {
            throw new IllegalArgumentException(name + " has no trunk to draft for");
        }
        this.chained = nextn.size() > 1;
        ModelHparams hp = spec.hparams();
        this.nEmbd = hp.nEmbd();
        if (targetSpec.hparams().nEmbd() != nEmbd) {
            throw new IllegalArgumentException(name + ": the NextN blocks read rows of n_embd " + nEmbd
                    + ", the target's n_embd is " + targetSpec.hparams().nEmbd());
        }
        this.tapWidth = targetSpec.layers().get(targetSpec.layers().size() - 1).residual().streamWidth();
        this.streams = hcStream ? hcStreams(nextn.get(0), name) : 1;
        this.hWidth = Math.multiplyExact(streams, nEmbd);
        this.meanStreams = HC_MEAN_TARGET.contains(arch) ? tapStreams(name) : 1;
        if (meanStreams == 1 && tapWidth != hWidth) {
            throw new IllegalArgumentException(name + ": the trunk ends with rows of " + tapWidth + " floats, the MTP"
                    + " head reads " + hWidth);
        }
        this.hFromBlock = PRE_NORM.contains(arch) || hcStream;
        // cohere2moe.cpp:295 picks the norm type from f_norm_rms_eps; the nemotron-h-moe head norm is LLM_NORM
        // (nemotron-h-moe.cpp:142)
        NormKind cohere = hp.normRmsEps() == 0f ? NormKind.LAYER : NormKind.RMS;
        this.inNorm = arch == Arch.COHERE2MOE ? cohere : NormKind.RMS;
        this.inEps = eps(hp, inNorm);
        this.headNorm = arch == Arch.COHERE2MOE ? cohere : arch == Arch.NEMOTRON_H_MOE ? NormKind.LAYER : NormKind.RMS;
        this.headEps = eps(hp, headNorm);
        // the target's own output norm (cohere2moe.cpp:258, nemotron-h.cpp:259, glm5-next.cpp:793, RMS elsewhere)
        this.targetNormKind = arch == Arch.COHERE2MOE ? cohere : NormKind.RMS;
        this.targetNormEps = eps(hp, targetNormKind);
        this.targetNormed = POST_NORM_TARGET.contains(arch);
        this.logitScale = arch == Arch.COHERE2MOE && hp.logitScale() != 0f ? hp.logitScale() : 1f;
        List<Block> bs = new ArrayList<>(nextn.size());
        for (LayerSpec l : nextn) {
            bs.add(block(l, arch));
        }
        this.blocks = Collections.unmodifiableList(bs);
        this.vocab = Math.toIntExact(head.shape(source, blocks.get(0).headWeight)[1]);
        for (Block b : blocks) {
            long v = head.shape(source, b.headWeight)[1];
            if (v != vocab) {
                throw new IllegalArgumentException(name + ": the head of NextN block " + b.layer.il() + " predicts "
                        + v + " tokens, block " + blocks.get(0).layer.il() + " " + vocab);
            }
        }
    }

    /** Streams of a hyper-connection NextN block: its residual must keep the hyper-connection streams. */
    private int hcStreams(LayerSpec l, String name) {
        ResidualSpec r = l.residual();
        if (r.kind() != ResidualKind.HYPER_CONNECTIONS || r.hcMult() < 2) {
            throw new IllegalArgumentException(name + ": NextN block " + l.il() + " has no hyper-connection streams ("
                    + r + "); its MTP graph runs the block over them");
        }
        return r.hcMult();
    }

    /** Streams of the trunk output that the host averages (glm5-next). */
    private int tapStreams(String name) {
        if (tapWidth % nEmbd != 0 || tapWidth / nEmbd < 2) {
            throw new IllegalArgumentException(name + ": the trunk ends with rows of " + tapWidth + " floats, not "
                    + "several streams of " + nEmbd);
        }
        return tapWidth / nEmbd;
    }

    private Block block(LayerSpec l, Arch arch) {
        int wl = l.weightLayer();
        String ehProj = nextnTensor(wl, TensorNames.NEXTN_EH_PROJ);
        String enorm = nextnTensor(wl, TensorNames.NEXTN_ENORM);
        String hnorm = nextnTensor(wl, TensorNames.NEXTN_HNORM);
        for (String t : new String[]{ehProj, enorm, hnorm}) {
            if (!head.has(source, t)) {
                throw new IllegalArgumentException(arch.ggufName() + ": the MTP block lacks " + t);
            }
        }
        String modelEmbd = required(spec.embedding().tokenEmbd() == null ? null
                : spec.embedding().tokenEmbd().name(), "token_embd");
        String modelHead = spec.head().output() == null ? null : spec.head().output().name();
        if (arch == Arch.QWEN4EXP) {
            return new Block(l, modelEmbd, ehProj, enorm, hnorm, null, required(modelHead, "the LM head"),
                    HcSubBlock.forNextnGatedHead(spec, l));
        }
        String ownEmbd = nextnTensor(wl, TensorNames.NEXTN_EMBED_TOKENS);
        String embedTable = head.has(source, ownEmbd) ? ownEmbd : modelEmbd;
        String ownHead = nextnTensor(wl, TensorNames.NEXTN_SHARED_HEAD_HEAD);
        String headWeight = head.has(source, ownHead) ? ownHead : required(modelHead, "the LM head");
        HcSubBlock hcHead = null;
        if (arch == Arch.DEEPSEEK4) {
            requireDsv4MtpBlock(l);
            hcHead = modelHcHead(arch);
        }
        String outputNorm = spec.head().outputNorm() == null ? null : spec.head().outputNorm().name();
        String headNormWeight = headNormWeight(arch, wl, t -> head.has(source, t), outputNorm);
        return new Block(l, embedTable, ehProj, enorm, hnorm, headNormWeight, headWeight, hcHead);
    }

    /**
     * The deepseek4 MTP graph attends over a plain sliding-window cache, so its block has no compressed attention
     * (deepseek4.cpp:944-945), and routes its experts by score (deepseek4.cpp:1460).
     */
    private void requireDsv4MtpBlock(LayerSpec l) {
        String name = spec.arch().ggufName();
        if (l.dsv4() != null && l.dsv4().compressRatio() != 0) {
            throw new IllegalArgumentException(name + ": NextN block " + l.il() + " has compress ratio "
                    + l.dsv4().compressRatio() + "; the MTP block attends without compression");
        }
        if (l.il() < spec.hparams().dsv4HashLayerCount()) {
            throw new IllegalArgumentException(name + ": NextN block " + l.il() + " routes its experts by token hash,"
                    + " which the MTP block does not support");
        }
    }

    /** The model's HC head over {@code output_hc_fn/scale/base}, which the deepseek4 MTP head reuses. */
    private HcSubBlock modelHcHead(Arch arch) {
        HcSubBlock h = HcSubBlock.forHead(spec);
        if (h.kind() != HcSubBlock.Kind.HEAD_WEIGHTS || h.hc() != streams) {
            throw new IllegalArgumentException(arch.ggufName() + ": the MTP head collapses " + streams + " streams"
                    + " with the model's output_hc_* weights, the model head is " + h);
        }
        return h;
    }

    private static float eps(ModelHparams hp, NormKind kind) {
        return kind == NormKind.LAYER ? hp.normEps() : hp.normRmsEps();
    }

    private static String nextnTensor(int wl, String base) {
        return TensorNames.layerName(wl, TensorNames.weight(base));
    }

    private String required(String tensor, String what) {
        return required(source, tensor, what);
    }

    private String required(Source s, String tensor, String what) {
        if (tensor == null || !head.has(s, tensor)) {
            throw new IllegalArgumentException(spec.arch().ggufName() + ": the MTP head needs " + what);
        }
        return tensor;
    }

    /**
     * The head-norm tensor of NextN block {@code wl}. bailingmoe3 loads its head norm from the block's
     * {@code layer_output_norm} and requires it (bailingmoe3.cpp:152, 430, 548). Every other head takes
     * {@code nextn.shared_head_norm}, which nemotron_h_moe requires; else mimo2's {@code layer_output_norm}
     * (mimo2.cpp:384-386); else the target's {@code output_norm}.
     *
     * @param present    whether the target file holds a tensor
     * @param outputNorm name of the target's {@code output_norm}, or null when it has none
     * @throws IllegalArgumentException when the tensor the architecture needs is missing
     */
    static String headNormWeight(Arch arch, int wl, Predicate<String> present, String outputNorm) {
        String layerOut = TensorNames.layerName(wl, TensorNames.weight(TensorNames.LAYER_OUT_NORM));
        if (arch == Arch.BAILINGMOE3) {
            return requireHeadNorm(arch, present, layerOut, layerOut);
        }
        String shared = nextnTensor(wl, TensorNames.NEXTN_SHARED_HEAD_NORM);
        if (arch == Arch.NEMOTRON_H_MOE) {
            return requireHeadNorm(arch, present, shared, shared);
        }
        if (present.test(shared)) {
            return shared;
        }
        if (arch == Arch.MIMO2 && present.test(layerOut)) {
            return layerOut;
        }
        return requireHeadNorm(arch, present, outputNorm, "a head norm");
    }

    private static String requireHeadNorm(Arch arch, Predicate<String> present, String tensor, String what) {
        if (tensor == null || !present.test(tensor)) {
            throw new IllegalArgumentException(arch.ggufName() + ": the MTP head needs " + what);
        }
        return tensor;
    }

    @Override
    public void bindTarget(EngineModel target) {
        Objects.requireNonNull(target, "target");
        if (backend != null) {
            throw new IllegalStateException("the NextN drafter is already bound");
        }
        ForwardBackend fb = target.forwardBackend();
        ModelMemory tm = fb.memory();
        if (tm == null) {
            throw new IllegalArgumentException(spec.arch().ggufName() + ": the target runs without memory");
        }
        MtpHead.Memory[] mems = new MtpHead.Memory[blocks.size()];
        MtpHead.Stack[] sts = new MtpHead.Stack[blocks.size()];
        MtpHead.Collapse[] cols = new MtpHead.Collapse[blocks.size()];
        for (int j = 0; j < blocks.size(); j++) {
            Block b = blocks.get(j);
            mems[j] = head.ownMemory(spec, MemoryPlanner.planNextn(spec, tm.plan(), b.layer.il()));
            sts[j] = head.stack(spec, source, mems[j], Collections.singletonList(b.layer));
            cols[j] = b.hcHead != null ? head.collapse(source, b.hcHead) : null;
        }
        this.memories = mems;
        this.stacks = sts;
        this.collapses = cols;
        this.stepRows = Math.min(head.ubatch(), mems[0].memory().plan().ubatch());
        if (targetNormed) {
            String normName = targetSpec.head().outputNorm() == null ? null
                    : targetSpec.head().outputNorm().name();
            this.targetNorm = head.vector(Source.TARGET, required(Source.TARGET, normName, "the target output norm"));
            if (targetNorm.length != nEmbd) {
                throw new IllegalArgumentException(normName + " has " + targetNorm.length + " values, n_embd is "
                        + nEmbd);
            }
        }
        this.hIn = head.rows("nextn.h", hWidth);
        this.embd = head.rows("nextn.e", nEmbd);
        this.cat = head.rows("nextn.eh", Math.multiplyExact(2 * streams, nEmbd));
        this.x = head.rows("nextn.x", hWidth);
        this.xn = head.rows("nextn.xn", nEmbd);
        this.logits = new float[vocab];
        this.hidden = new DraftModelLoader.HiddenRows(hWidth);
        this.targetSequence = tm.sequence();
        this.feed = new DraftModelLoader.TargetFeed(new int[]{targetSpec.nLayer()}, tapWidth, tm);
        fb.setHiddenTap(feed);
        this.backend = fb;
    }

    @Override
    public int[] draft(int lastToken, int n) {
        requireBound();
        if (n < 0) {
            throw new IllegalArgumentException("draft length " + n + " is negative");
        }
        sync();
        int pos0 = targetSequence.nPast();
        int nCtx = memories[0].memory().plan().nCtx();
        int count = Math.min(n, nCtx - pos0);
        if (chained) {
            count = Math.min(count, Math.min(blocks.size(), stepRows));
        }
        if (count <= 0) {
            return new int[0];
        }
        return chained ? draftChained(lastToken, pos0, count) : draftSingle(lastToken, pos0, count);
    }

    /** One head, one row per step at {@code pos0 + i} (speculative.cpp:1829-1833). */
    private int[] draftSingle(int lastToken, int pos0, int requested) {
        memories[0].align(pos0);
        int count = droppableRun(memories[0], requested);
        if (count == 0) {
            return new int[0];
        }
        float[] h = hidden.pending().clone();
        int[] out = new int[count];
        int made = 0;
        int[] one = new int[]{lastToken};
        for (int i = 0; i < count; i++) {
            step(0, pos0 + i, one, h, 1);
            int id = pickLast(0, 1);
            if (id < 0) {
                break;
            }
            out[made++] = id;
            one[0] = id;
            if (i + 1 < count) {
                head.readRow(hFromBlock ? x : xn, 0, h);
            }
        }
        return Arrays.copyOf(out, made);
    }

    /** Head {@code i} over the whole prefix at step {@code i} (speculative.cpp:1741-1753, 1813-1823). */
    private int[] draftChained(int lastToken, int pos0, int count) {
        int[] tokens = new int[count];
        tokens[0] = lastToken;
        float[] chainH = new float[Math.multiplyExact(count, hWidth)];
        System.arraycopy(hidden.pending(), 0, chainH, 0, hWidth);
        float[] row = new float[hWidth];
        int[] out = new int[count];
        int made = 0;
        for (int i = 0; i < count; i++) {
            int rows = i + 1;
            memories[i].align(pos0);
            if (droppableRun(memories[i], rows) < rows) {
                break;
            }
            step(i, pos0, Arrays.copyOf(tokens, rows), chainH, rows);
            int id = pickLast(i, rows);
            if (id < 0) {
                break;
            }
            out[made++] = id;
            if (i + 1 < count) {
                tokens[i + 1] = id;
                head.readRow(hFromBlock ? x : xn, rows - 1, row);
                System.arraycopy(row, 0, chainH, rows * hWidth, hWidth);
            }
        }
        return Arrays.copyOf(out, made);
    }

    /**
     * Longest run of at most {@code count} rows a drafting step can write into {@code mem} from its current end and
     * still drop at the next target call (which realigns the caches to the same position) without losing a key
     * that a sliding-window ring must keep. The answer is monotone in the run length: the position after the drop
     * is fixed while more rows overwrite the ring. A view of the target's caches writes nothing.
     */
    private static int droppableRun(MtpHead.Memory mem, int count) {
        if (!mem.ownsCaches()) {
            return count;
        }
        ModelMemory m = mem.memory();
        if (m.ringsKeepWindow(count, count)) {
            return count;
        }
        int lo = 0;
        int hi = count;
        while (hi - lo > 1) {
            int mid = (lo + hi) >>> 1;
            if (m.ringsKeepWindow(mid, mid)) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        return lo;
    }

    /**
     * Head of block {@code j} over the {@code rows} block outputs in {@code x} into {@code xn} (stream collapse, head
     * norm) and the draft pick of the last row; -1 when its probability is below {@code p_min}.
     */
    private int pickLast(int j, int rows) {
        Block b = blocks.get(j);
        if (collapses[j] != null) {
            collapses[j].run(x, xn, rows);
            if (b.headNormWeight != null) {
                head.norm(headNorm, source, b.headNormWeight, headEps, xn, xn, rows);
            }
        } else {
            head.norm(headNorm, source, b.headNormWeight, headEps, x, xn, rows);
        }
        head.logits(source, b.headWeight, xn, rows - 1, logits);
        if (logitScale != 1f) {
            CpuOps.scale(logits, 0, vocab, logitScale);
        }
        Pick p = DraftModelLoader.pick(logits, vocab);
        return p.p < options.pMin() ? -1 : p.id;
    }

    @Override
    public void accept(int nAccepted) {
        requireBound();
        if (nAccepted < 0) {
            throw new IllegalArgumentException("accepted count " + nAccepted + " is negative");
        }
        sync();
        hidden.accept(nAccepted);
    }

    @Override
    public void reset() {
        if (backend == null) {
            return;
        }
        feed.clear();
        for (MtpHead.Memory m : memories) {
            m.reset();
        }
        hidden.clear();
    }

    @Override
    public void close() {
        try {
            if (backend != null) {
                backend.setHiddenTap(HiddenTap.NONE);
            }
        } finally {
            backend = null;
            head.close();
        }
    }

    private void requireBound() {
        if (backend == null) {
            throw new IllegalStateException("the NextN drafter is not bound to a target");
        }
    }

    /** Runs the heads over every target call collected since the last sync. */
    private void sync() {
        for (Batch b : feed.drain()) {
            process(b);
        }
    }

    /** Every head over the pairs of one target call (speculative.cpp:1622-1667). */
    private void process(Batch b) {
        int m = b.size();
        float[] h = targetRows(b.rows[0], m);
        for (MtpHead.Memory mem : memories) {
            mem.align(b.pos0);
        }
        float[] before = hidden.at(b.pos0 - 1);
        float[] pairs = new float[Math.multiplyExact(Math.min(stepRows, m), hWidth)];
        for (int c = 0; c < m; c += stepRows) {
            int len = Math.min(stepRows, m - c);
            if (c == 0) {
                System.arraycopy(before != null ? before : hidden.pending(), 0, pairs, 0, hWidth);
                System.arraycopy(h, 0, pairs, hWidth, (len - 1) * hWidth);
            } else {
                System.arraycopy(h, (c - 1) * hWidth, pairs, 0, len * hWidth);
            }
            int[] tokens = Arrays.copyOfRange(b.tokens, c, c + len);
            for (int j = 0; j < blocks.size(); j++) {
                step(j, b.pos0 + c, tokens, pairs, len);
            }
        }
        hidden.record(b.pos0, h, m);
    }

    /**
     * The hidden rows the heads read from the {@code m} tap rows of one call: the stream mean (glm5-next), then the
     * target's output norm for post-norm targets; the tap rows themselves otherwise. May reuse {@code tap}.
     */
    private float[] targetRows(float[] tap, int m) {
        float[] h = tap;
        if (meanStreams > 1) {
            h = new float[Math.multiplyExact(m, nEmbd)];
            CpuHyperConnections.pre(HcKernels.PreParams.mean(meanStreams), meanStreams, nEmbd, m, tap, 0, null, 0, h,
                    0);
        }
        if (targetNormed) {
            CpuOps.norm(targetNormKind, targetNormEps, h, 0, h, 0, m, nEmbd, targetNorm, null);
        }
        return h;
    }

    /**
     * Head {@code j} up to its block output in {@code x}: {@code len} pairs of tokens and hidden rows at
     * {@code pos0 ..}. Over {@code streams} streams the embedding row is repeated for every stream and {@code hnorm}
     * and {@code eh_proj} apply per stream (deepseek4.cpp:1416-1427, qwen4exp.cpp:581-588).
     */
    private void step(int j, int pos0, int[] tokens, float[] hRows, int len) {
        Block b = blocks.get(j);
        head.write(hIn, hRows, len);
        head.embed(source, b.embedTable, tokens, len, embd);
        head.norm(inNorm, source, b.enorm, inEps, embd, embd, len);
        head.normStreams(inNorm, source, b.hnorm, inEps, hIn, hIn, len, streams);
        head.concatStreams(embd, hIn, cat, len, streams);
        head.projectStreams(source, b.ehProj, cat, x, len, streams);
        stacks[j].run(pos0, len, tokens, x);
    }

    @Override
    public String toString() {
        return "NextnDraftModel{" + spec.arch().ggufName() + ", blocks " + blocks.get(0).layer.il() + ".."
                + blocks.get(blocks.size() - 1).layer.il() + (chained ? " chained" : "")
                + (streams > 1 ? ", " + streams + " streams" : "") + (meanStreams > 1 ? ", stream mean" : "")
                + (hFromBlock ? ", pre-norm h" : "") + '}';
    }
}

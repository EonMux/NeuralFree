package com.eonmux.neuralfree.gguf.ops.flow;

import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.arch.ResidualKind;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuOps;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.engine.DraftModel;
import com.eonmux.neuralfree.gguf.engine.EngineModel;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.engine.HiddenTap;
import com.eonmux.neuralfree.gguf.engine.draft.DraftProbs;
import com.eonmux.neuralfree.gguf.engine.watermark.DraftWatermarker;
import com.eonmux.neuralfree.gguf.engine.watermark.WatermarkConfig;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardBackend;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuSplitForwardBackend;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;
import com.eonmux.neuralfree.gguf.memory.MemoryPlanner;
import com.eonmux.neuralfree.gguf.memory.ModelMemory;
import com.eonmux.neuralfree.gguf.memory.SequenceState;
import com.eonmux.neuralfree.gguf.ops.flow.DraftModelLoader.Batch;
import com.eonmux.neuralfree.gguf.ops.flow.DraftModelLoader.Pick;
import com.eonmux.neuralfree.gguf.ops.flow.MtpHead.Rows;
import com.eonmux.neuralfree.gguf.ops.flow.MtpHead.Source;
import com.eonmux.neuralfree.gguf.ops.flow.TargetDeviceFeed.Chunk;
import com.eonmux.neuralfree.gguf.ops.flow.TargetDeviceFeed.DeviceBatch;
import com.eonmux.neuralfree.gguf.ops.flow.TargetDeviceFeed.DeviceRows;
import com.eonmux.neuralfree.gguf.ops.hc.CpuHyperConnections;
import com.eonmux.neuralfree.gguf.ops.hc.HcKernels;
import com.eonmux.neuralfree.gguf.ops.hc.HcSubBlock;
import com.eonmux.neuralfree.gguf.quant.TapProduct;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.ResidualSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.gguf.spec.TensorRef;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.EnumSet;
import java.util.Iterator;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;
import java.util.Set;
import java.util.function.Predicate;

/**
 * Self-speculation with the target's NextN/MTP blocks (the {@code graph_mtp} heads of llama.cpp src/models, e.g.
 * qwen35.cpp:493-660, deepseek2.cpp:164-390, mimo2.cpp:260-398, step35.cpp:355-560, deepseek4.cpp:1366-1510,
 * qwen4exp.cpp:524-616, glm5-next.cpp:550-674) and the {@code qwen2} head of a MiMo checkpoint (vLLM
 * model_executor/models/mimo_mtp.py {@code MiMoMultiTokenPredictorLayer}), driven like llama.cpp common/speculative.cpp
 * {@code common_speculative_impl_draft_mtp} (1432-1876) without shared memory.
 *
 * <p>A head (NextN block {@code n_layer + j}): {@code e = embed(tok)} (from the block's {@code nextn.embed_tokens},
 * else the target's {@code token_embd}); {@code x = eh_proj . [enorm(e) | hnorm(h)]}; the block itself (run by the
 * layer executor over caches of its own, planned by {@link MemoryPlanner#planNextn(ModelSpec,
 * com.eonmux.neuralfree.gguf.memory.MemoryPlan, int)}); then the head norm (bailingmoe3's {@code layer_output_norm};
 * otherwise {@code nextn.shared_head_norm}, else mimo2's {@code layer_output_norm}, else {@code output_norm}) and the
 * head ({@code nextn.shared_head_head}, else the target's LM head; cohere2moe scales by its logit scale). An MTP
 * draft file may carry its own LM head cut to its most frequent tokens, whose {@code d2t} entry names the target
 * token of every row (qwen4exp.cpp:287-296): the pick runs over the cut rows and maps through {@code d2t}, which
 * is the scatter of the cut logits to the target vocabulary with {@code -inf} elsewhere that the reference graph
 * builds (qwen4exp.cpp:841-844), with the same argmax and the same pick distribution. The block's
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
 *       (qwen4exp.cpp:606-613). qwen4exp embeds with the target's {@code token_embd} only (qwen4exp.cpp:553-556);
 *       a draft file without its own {@code token_embd} or LM head takes the target's
 *       ({@code llama_model_share_mtp_io});</li>
 *   <li>glm5-next reads the post-norm state of the collapsed trunk (stream mean, then {@code output_norm},
 *       glm5-next.cpp:785-798), computed here on the host, and runs a plain block without hyper-connections
 *       (glm5-next.cpp:612-663; its NextN blocks carry no {@code hc_*} tensors, glm5-next.cpp:85-92).</li>
 * </ul>
 *
 * <p>Protocol (positions are the target's): each target forward call over positions {@code [p0, p0 + m)} runs every
 * head over the pairs {@code (token[p], h[p - 1])}, writing that head's caches at every position, the first pair
 * taking the pending row of the previous call; the call's rows become the verify rows and the last one the pending
 * row. {@code accept(n)} makes verify row {@code min(n, rows - 1)} pending. Steps at a position first drop the head's
 * cache rows from that position on, so stale draft and verify rows are rewritten. A generation that reuses a stored
 * prompt prefix ({@link #beginGeneration(int)}) keeps the rows of those positions and re-evaluates the last one,
 * whose pair the caches already hold; the call supplies the hidden row of the first new pair. A draft at
 * {@code P = n_past}:
 * <ul>
 *   <li>one head (every architecture): starts from {@code (lastToken, pending)} at {@code P} and continues with each
 *       drafted token and the head's own {@code h_nextn} at {@code P + i};</li>
 *   <li>chained heads (hy-v3, mimo2 and step35 with several NextN blocks, speculative.cpp {@code chain_heads}):
 *       draft step {@code i} runs head {@code i} over the whole prefix {@code [lastToken, d_0 .. d_{i-1}]} at
 *       {@code P ..} with the hidden rows {@code [pending, h_0 .. h_{i-1}]}, where {@code h_k} is the last-row
 *       {@code h_nextn} of step {@code k}; a draft proposes at most one token per head;</li>
 *   <li>cycling heads (exaone 4.5, exaone-moe, bailingmoe2, dots3-note and GLM-4 MoE Lite with several NextN
 *       blocks, vLLM {@code spec_step_idx % num_mtp_layers}): draft step {@code k} runs block {@code k mod n} over
 *       its own caches with the previous step's {@code h_nextn}, so the blocks repeat in order.</li>
 * </ul>
 *
 * <p>What a call costs the heads: the target's hidden rows reach them on the device when the target and the head
 * share it, or cross from the last trunk layer's device to the head's on a layer split ({@link TargetDeviceFeed};
 * rows a host path delivers anyway are uploaded), the pairs of a call's prompt batches write only their caches
 * ({@link MtpHead.Stack#runKV}: no reader consumes the rows of a call), and a small call (a verification, a decode)
 * defers its pairs, so its catch-up decodes together with the next draft's first pass — one run of the block per step
 * instead of two, its accepted rows rewritten with the verified pairs and the rejected ones never decoded
 * (llama.cpp's deferred catch-up rows, common/speculative.cpp draft_mtp).</p>
 */
public final class NextnDraftModel implements DraftModel, DraftProbs {

    /**
     * Targets whose {@code h_nextn} is the output-norm state (the target graphs' {@code t_h_nextn}; MiMo's
     * {@code mimo_mtp.py} head reads the Qwen2 model output after its final norm).
     */
    private static final Set<Arch> POST_NORM_TARGET = Collections.unmodifiableSet(EnumSet.of(Arch.DEEPSEEK2,
            Arch.DEEPSEEK32, Arch.GLM4_MOE, Arch.GLM_DSA, Arch.MELLUM, Arch.BAILINGMOE2, Arch.BAILINGMOE3,
            Arch.COHERE2MOE, Arch.HY_V3, Arch.QWEN3NEXT, Arch.QWEN35, Arch.QWEN35MOE, Arch.NEMOTRON_H,
            Arch.NEMOTRON_H_MOE, Arch.GLM5_NEXT, Arch.QWEN2, Arch.EXAONE4, Arch.EXAONE_MOE, Arch.DOTS3NOTE));
    /**
     * Targets and heads whose {@code h_nextn} is the pre-norm stream (mimo2.cpp:236-238, step35.cpp:339-341; the
     * Step5 predictor returns the decoder block's residual output and the shared head norms it afterwards).
     */
    private static final Set<Arch> PRE_NORM = Collections.unmodifiableSet(EnumSet.of(Arch.MIMO2, Arch.STEP35,
            Arch.STEP5));
    /**
     * MTP graphs that select their block by {@code nextn_layer_offset} without asserting a single block
     * (hy-v3.cpp:236, mimo2.cpp:265, step35.cpp:365); every other MTP graph asserts {@code n_layer_nextn == 1}.
     */
    private static final Set<Arch> CHAINED_HEADS = Collections.unmodifiableSet(EnumSet.of(Arch.HY_V3, Arch.MIMO2,
            Arch.STEP35));
    /**
     * MTP heads that cycle their blocks (vLLM {@code spec_step_idx % num_mtp_layers}): draft step {@code k} runs
     * block {@code k mod n}, each block with its own caches: exaone4_5_mtp.py:119, exaone_moe_mtp.py:114,
     * bailing_moe_mtp.py:177, deepseek_mtp.py:217 (GLM-4 MoE Lite) and the dots3_note predictor over
     * deepseek_v32's.
     */
    private static final Set<Arch> CYCLING_HEADS = Collections.unmodifiableSet(EnumSet.of(Arch.EXAONE4,
            Arch.EXAONE_MOE, Arch.BAILINGMOE2, Arch.DOTS3NOTE, Arch.DEEPSEEK2, Arch.STEP5));
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
    /**
     * False feeds the heads their target rows through the host again on a shared device, restoring the batch path
     * ({@code neuralfree.draft.mtp.device.rows}; llama.cpp's {@code LLAMA_MTP_H_DEVICE=0}).
     */
    private static boolean deviceRows() {
        return !"false".equals(System.getProperty("neuralfree.draft.mtp.device.rows", "true"));
    }

    /** True when {@code arch}'s MTP head cycles its blocks, one block per draft step, instead of chaining them. */
    public static boolean cyclesBlocks(Arch arch) {
        return CYCLING_HEADS.contains(arch);
    }

    /** The tensors and layer one NextN block's head runs. Immutable. */
    private static final class Block {
        final LayerSpec layer;
        /** Where the embedding table and the LM head are read: the block's model, or the target when shared. */
        final Source embedSource;
        final String embedTable;
        final String ehProj;
        final String enorm;
        final String hnorm;
        /** Head norm; null when the stream collapse is the final norm (qwen4exp). */
        final String headNormWeight;
        final Source headSource;
        final String headWeight;
        /** Stream collapse before the head norm; null for heads over one stream. */
        final HcSubBlock hcHead;

        Block(LayerSpec layer, Source embedSource, String embedTable, String ehProj, String enorm, String hnorm,
              String headNormWeight, Source headSource, String headWeight, HcSubBlock hcHead) {
            this.layer = layer;
            this.embedSource = embedSource;
            this.embedTable = embedTable;
            this.ehProj = ehProj;
            this.enorm = enorm;
            this.hnorm = hnorm;
            this.headNormWeight = headNormWeight;
            this.headSource = headSource;
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
    /** True when the head cycles its blocks, one block per draft step (vLLM {@code spec_step_idx % n}). */
    private final boolean cycling;
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
    /** {@code d2t} of a reduced (FR-Spec) draft head, or null over the full vocabulary. */
    private final String d2tName;
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
    /** The device feed when the target delivers its hidden rows on the device the head runs on, else null. */
    private TargetDeviceFeed deviceFeed;
    private DeviceRows devRows;
    /** Rows of one drafting step's hidden pairs on the device; null on the host feed. */
    private Rows devicePairs;
    private MtpHead.Memory[] memories;
    private MtpHead.Stack[] stacks;
    private MtpHead.Collapse[] collapses;
    private int stepRows;
    private float[] targetNorm;
    /** The target output norm's tensor, which a device feed's rows are normed through. */
    private String targetNormName;
    /** Target token of every draft-head row when {@link #d2tName} is set, read at bind. */
    private int[] d2t;
    private Rows hIn;
    private Rows embd;
    private Rows cat;
    private Rows x;
    private Rows xn;
    private float[] logits;

    /** The draft-side watermark of the current generation, or null for the ordinary greedy pick. */
    private DraftWatermarker draftWatermark;
    private SplittableRandom draftRng;
    private float draftTemperature = 1f;
    /** The proposal distributions of the last draft's emitted positions, by position (null entries for greedy). */
    private final List<float[]> draftProposals = new ArrayList<>();
    /** The tokens the last draft emitted. */
    private int draftCount;

    private DraftModelLoader.HiddenRows hidden;
    /** The catch-up rows of a small target call, decoded with the next draft's first pass. */
    private final List<Deferred> defer = new ArrayList<>();
    /** The rows of the call the current deferrals came from; 0 when they are gone. */
    private int deferBasis;

    /**
     * A catch-up row of a target call the next draft's first pass decodes: its token, its position and the hidden
     * row it pairs with, on the host ({@code row}) or on the device ({@code device} at {@code deviceRow}, held by
     * {@code owned} until then unless the rows are permanent).
     */
    private static final class Deferred {
        final int token;
        final int pos;
        final float[] row;
        final Rows device;
        final int deviceRow;
        final Chunk owned;

        Deferred(int token, int pos, float[] row, Rows device, int deviceRow, Chunk owned) {
            this.token = token;
            this.pos = pos;
            this.row = row;
            this.device = device;
            this.deviceRow = deviceRow;
            this.owned = owned;
        }
    }

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
        if (arch == Arch.DEEPSEEK41) {
            // models/deepseek41.cpp:221-224: no MTP graph; the DSpark draft is a separate DFLASH file
            throw new UnsupportedOperationException(name + " has no MTP graph: its DSpark draft is a separate DFLASH"
                    + " GGUF used with the dflash draft type (models/deepseek41.cpp:221-224)");
        }
        if (!POST_NORM_TARGET.contains(arch) && !PRE_NORM.contains(arch) && !hcStream) {
            throw new UnsupportedOperationException(name + " has no NextN/MTP draft graph in llama.cpp");
        }
        List<LayerSpec> nextn = spec.nextnLayers();
        if (nextn.isEmpty()) {
            throw new IllegalArgumentException(name + " has no NextN blocks (nextn_predict_layers is 0 or the"
                    + " blocks are missing from the file)");
        }
        boolean multi = nextn.size() > 1;
        if (multi && !CHAINED_HEADS.contains(arch) && !CYCLING_HEADS.contains(arch)) {
            String why = arch == Arch.QWEN2
                    ? "MiMo's MTP head predicts one token per step (vLLM mimo_mtp.py)"
                    : "its MTP graph runs a single block (llama.cpp asserts n_layer_nextn == 1)";
            throw new UnsupportedOperationException(name + " has " + nextn.size() + " NextN blocks; " + why);
        }
        if (targetSpec.layers().isEmpty()) {
            throw new IllegalArgumentException(name + " has no trunk to draft for");
        }
        this.cycling = multi && CYCLING_HEADS.contains(arch);
        this.chained = multi && !cycling;
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
        this.vocab = Math.toIntExact(head.shape(blocks.get(0).headSource, blocks.get(0).headWeight)[1]);
        for (Block b : blocks) {
            long v = head.shape(b.headSource, b.headWeight)[1];
            if (v != vocab) {
                throw new IllegalArgumentException(name + ": the head of NextN block " + b.layer.il() + " predicts "
                        + v + " tokens, block " + blocks.get(0).layer.il() + " " + vocab);
            }
        }
        // an MTP draft file may carry its own LM head cut to its most frequent tokens, its d2t entry naming the
        // target token of each row (qwen4exp.cpp:287-296); the target's own head is never cut
        TensorRef d2t = source == Source.DRAFT ? spec.head().d2t() : null;
        this.d2tName = d2t == null ? null : d2t.name();
        long targetVocab = targetSpec.head().vocabRows();
        if (d2tName != null) {
            if (d2t.ne(0) != vocab) {
                throw new IllegalArgumentException(name + ": " + d2tName + " has " + d2t.ne(0) + " entries, the head "
                        + blocks.get(0).headWeight + " predicts " + vocab + " tokens");
            }
        } else if (vocab != targetVocab) {
            throw new IllegalArgumentException(name + ": the head " + blocks.get(0).headWeight + " predicts " + vocab
                    + " tokens, the target's vocabulary has " + targetVocab + ", and there is no d2t table");
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
        String modelEmbd = spec.embedding().tokenEmbd() == null ? null : spec.embedding().tokenEmbd().name();
        String modelHead = spec.head().output() == null ? null : spec.head().output().name();
        if (arch == Arch.QWEN4EXP) {
            // A shared MTP-only file may lack the token embeddings and the LM head: it takes the target's
            // (qwen4exp.cpp:274-300, llama_model_share_mtp_io)
            Source embdSource = source;
            String embedTable = modelEmbd;
            if (embedTable == null) {
                embdSource = Source.TARGET;
                embedTable = requiredTarget(targetSpec.embedding().tokenEmbd(),
                        "token embeddings (own or the target's)");
            }
            Source lmSource = source;
            String headWeight = modelHead;
            if (headWeight == null) {
                lmSource = Source.TARGET;
                headWeight = requiredTarget(targetSpec.head().output(), "an LM head (own or the target's)");
            }
            return new Block(l, embdSource, embedTable, ehProj, enorm, hnorm, null, lmSource, headWeight,
                    HcSubBlock.forNextnGatedHead(spec, l));
        }
        required(modelEmbd, "token_embd");
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
        return new Block(l, source, embedTable, ehProj, enorm, hnorm, headNormWeight, source, headWeight, hcHead);
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

    /**
     * The target token of every row of a reduced (FR-Spec) head, validated against the head's rows and the target's
     * vocabulary. eagle3 maps its draft ids the same way (eagle3.cpp:286-299); the reference MTP graph instead
     * scatters the cut logits to the target vocabulary with {@code -inf} elsewhere (qwen4exp.cpp:841-844), whose
     * pick this mapping reproduces.
     */
    private int[] mapping() {
        int targetVocab = Math.toIntExact(targetSpec.head().vocabRows());
        int[] map = head.integers(source, d2tName);
        if (map.length != vocab) {
            throw new IllegalArgumentException(spec.arch().ggufName() + ": " + d2tName + " has " + map.length
                    + " entries for " + vocab + " draft tokens");
        }
        for (int i = 0; i < map.length; i++) {
            if (map[i] < 0 || map[i] >= targetVocab) {
                throw new IllegalArgumentException(spec.arch().ggufName() + ": " + d2tName + "[" + i + "] = " + map[i]
                        + " is outside the target vocabulary of " + targetVocab);
            }
        }
        return map;
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
     * The name of the target's tensor {@code ref}, which an MTP file without its own takes
     * ({@code llama_model_share_mtp_io}, llama-model.cpp:3054-3070). The shape needs no check here: the target's
     * IO was checked against its hparams at its load, and {@code ModelSpecBuilder} accepts only a draft file of the
     * target's width and vocabulary.
     *
     * @throws IllegalArgumentException when the target holds no such tensor
     */
    private String requiredTarget(TensorRef ref, String what) {
        if (ref == null || !head.has(Source.TARGET, ref.name())) {
            throw new IllegalArgumentException(spec.arch().ggufName() + ": the MTP head needs " + what);
        }
        return ref.name();
    }

    /**
     * The head-norm tensor of NextN block {@code wl}. bailingmoe3 loads its head norm from the block's
     * {@code layer_output_norm} and requires it (bailingmoe3.cpp:152, 430, 548). Every other head takes
     * {@code nextn.shared_head_norm}, which nemotron_h_moe requires; else mimo2's or bailingmoe2's
     * {@code layer_output_norm} (mimo2.cpp:384-386, bailing_moe_mtp.py {@code final_layernorm}); else the target's
     * {@code output_norm}.
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
        if ((arch == Arch.MIMO2 || arch == Arch.BAILINGMOE2) && present.test(layerOut)) {
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
        this.d2t = d2tName == null ? null : mapping();
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
            this.targetNormName = normName;
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
        if (deviceRows() && meanStreams == 1 && deviceFeed(fb)) {
            // the target's rows stay on the device the head runs on (one device), or reach it from the last
            // trunk layer's device when the target is a layer split
            ComputeQueue capture = captureQueue(fb);
            this.deviceFeed = new TargetDeviceFeed(head, tm, targetSpec.nLayer(), tapWidth, capture);
            this.devRows = deviceFeed.deviceRows();
            this.devicePairs = head.rows("flow.tap.pairs", hWidth);
            fb.setHiddenTap(deviceFeed);
        } else {
            this.feed = new DraftModelLoader.TargetFeed(new int[]{targetSpec.nLayer()}, tapWidth, tm);
            fb.setHiddenTap(feed);
        }
        this.backend = fb;
    }

    /** Whether {@code fb}'s forward calls deliver the head's rows on a device the feed can keep them on. */
    private boolean deviceFeed(ForwardBackend fb) {
        return head.gpuContext() != null && captureQueue(fb) != null;
    }

    /**
     * The queue that captures the target's tap rows: the head's own queue when the target runs there, else the last
     * layer's shard of a layer split (the feed copies the rows to the head's device). Null when neither applies.
     */
    private ComputeQueue captureQueue(ForwardBackend fb) {
        if (fb instanceof GpuForwardBackend && ((GpuForwardBackend) fb).context() == head.gpuContext()) {
            return head.gpuContext().queue();
        }
        if (fb instanceof GpuSplitForwardBackend) {
            GpuSplitForwardBackend split = (GpuSplitForwardBackend) fb;
            if (!split.isShardQueue(head.gpuContext().queue())) {
                return null;
            }
            return split.tapQueue(targetSpec.nLayer());
        }
        return null;
    }

    @Override
    public int[] draft(int lastToken, int n) {
        return draft(lastToken, n, 1f);
    }

    @Override
    public int[] draft(int lastToken, int n, float temperature) {
        requireBound();
        if (n < 0) {
            throw new IllegalArgumentException("draft length " + n + " is negative");
        }
        draftProposals.clear();
        draftCount = 0;
        draftTemperature = temperature;
        if (draftWatermark != null && temperature > 0f) {
            draftWatermark.sync(targetSequence.textTokens(), lastToken);
            draftWatermark.beginRound();
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
        int[] result = chained ? draftChained(lastToken, pos0, count)
                : deviceFeed != null ? draftSingleDevice(lastToken, pos0, count)
                : draftSingle(lastToken, pos0, count);
        draftCount = result.length;
        return result;
    }

    /** One head, one row per step at {@code pos0 + i}; a cycling head runs block {@code i mod n} at step i. */
    private int[] draftSingle(int lastToken, int pos0, int requested) {
        int nDef = pendingDeferred(pos0);
        alignMemories(pos0 - nDef);
        int count = droppableRun(requested + nDef) - nDef;
        if (count <= 0) {
            return new int[0];
        }
        int[] out = new int[count];
        int made = 0;
        float[] h = hidden.pending().clone();
        int[] one = new int[]{lastToken};
        int i = 0;
        if (nDef > 0) {
            // the catch-up rows of the last target call decode with this draft's first pass
            int len = nDef + 1;
            int[] tokens = new int[len];
            float[] pairs = new float[Math.multiplyExact(len, hWidth)];
            for (int d = 0; d < nDef; d++) {
                Deferred deferred = defer.get(d);
                tokens[d] = deferred.token;
                System.arraycopy(deferred.row, 0, pairs, d * hWidth, hWidth);
            }
            tokens[nDef] = lastToken;
            System.arraycopy(h, 0, pairs, nDef * hWidth, hWidth);
            step(0, pos0 - nDef, tokens, pairs, len, false);
            clearDeferred();
            int id = pickLast(0, len);
            if (id < 0) {
                return new int[0];
            }
            out[made++] = id;
            one[0] = id;
            head.readRow(hFromBlock ? x : xn, len - 1, h);
            i = 1;
        }
        for (; i < count; i++) {
            int j = blockAt(i);
            step(j, pos0 + i, one, h, 1, false);
            int id = pickLast(j, 1);
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

    /** {@link #draftSingle} over a device feed: the pending row and the chain stay on the device. */
    private int[] draftSingleDevice(int lastToken, int pos0, int requested) {
        int nDef = pendingDeferred(pos0);
        alignMemories(pos0 - nDef);
        int count = droppableRun(requested + nDef) - nDef;
        if (count <= 0) {
            return new int[0];
        }
        int[] out = new int[count];
        int made = 0;
        Rows pending = devRows.pending();
        int[] one = new int[]{lastToken};
        int i = 0;
        if (nDef > 0) {
            // the catch-up rows of the last target call decode with this draft's first pass
            int len = nDef + 1;
            int[] tokens = new int[len];
            for (int d = 0; d < nDef; d++) {
                Deferred deferred = defer.get(d);
                tokens[d] = deferred.token;
                head.copyRows(deferred.device, deferred.deviceRow, devicePairs, d, 1);
            }
            tokens[nDef] = lastToken;
            head.copyRows(pending, 0, devicePairs, nDef, 1);
            stepDevice(0, pos0 - nDef, tokens, devicePairs, len, false);
            clearDeferred();
            int id = pickLast(0, len);
            if (id < 0) {
                return new int[0];
            }
            out[made++] = id;
            one[0] = id;
            head.copyRows(hFromBlock ? x : xn, len - 1, pending, 0, 1);
            i = 1;
        }
        for (; i < count; i++) {
            int j = blockAt(i);
            stepDevice(j, pos0 + i, one, pending, 1, false);
            int id = pickLast(j, 1);
            if (id < 0) {
                break;
            }
            out[made++] = id;
            one[0] = id;
            if (i + 1 < count) {
                head.copyRows(hFromBlock ? x : xn, 0, pending, 0, 1);
            }
        }
        return Arrays.copyOf(out, made);
    }

    /** The block a draft step {@code i} runs: {@code i mod n} for a cycling head, block 0 otherwise. */
    private int blockAt(int i) {
        return cycling ? i % blocks.size() : 0;
    }

    /** Aligns every block's caches to {@code pos0}, the position the next draft step writes. */
    private void alignMemories(int pos0) {
        for (MtpHead.Memory m : memories) {
            m.align(pos0);
        }
    }

    /** {@link #droppableRun(MtpHead.Memory, int)} over every block's caches. */
    private int droppableRun(int count) {
        int run = count;
        for (MtpHead.Memory mem : memories) {
            run = Math.min(run, droppableRun(mem, count));
        }
        return run;
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
            step(i, pos0, Arrays.copyOf(tokens, rows), chainH, rows, false);
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
     * norm) and the draft pick of the last row, its id mapped through {@code d2t} over a reduced head; -1 when its
     * probability is below {@code p_min}.
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
        head.logits(b.headSource, b.headWeight, xn, rows - 1, logits);
        if (logitScale != 1f) {
            CpuOps.scale(logits, 0, vocab, logitScale);
        }
        boolean watermarked = draftWatermark != null && draftTemperature > 0f;
        Pick p;
        if (watermarked) {
            float[] row = d2t == null ? logits : DraftModelLoader.scatter(logits, vocab, d2t);
            p = DraftModelLoader.pick(row, row.length, draftWatermark, draftTemperature, draftRng::nextDouble);
        } else {
            p = DraftModelLoader.pick(logits, vocab);
        }
        if (p.p < options.pMin()) {
            return -1;
        }
        if (watermarked) {
            draftProposals.add(p.proposal);
            return p.id;
        }
        return d2t != null ? d2t[p.id] : p.id;
    }

    @Override
    public boolean supportsWatermarking() {
        return true;
    }

    @Override
    public void watermark(WatermarkConfig config, int promptLength, long seed) {
        this.draftWatermark = DraftWatermarker.of(config);
        if (draftWatermark != null) {
            draftWatermark.reset(promptLength);
        }
        this.draftRng = draftWatermark == null ? null : new SplittableRandom(seed);
    }

    @Override
    public float[] draftProbs(int position) {
        if (position < 0 || position >= draftCount) {
            throw new IllegalStateException("no drafted token at position " + position);
        }
        float[] row = position < draftProposals.size() ? draftProposals.get(position) : null;
        return row == null ? null : row.clone();
    }

    @Override
    public void accept(int nAccepted) {
        requireBound();
        if (nAccepted < 0) {
            throw new IllegalArgumentException("accepted count " + nAccepted + " is negative");
        }
        sync();
        if (deviceFeed != null) {
            devRows.accept(nAccepted);
        }
        hidden.accept(nAccepted);
        trimDeferred(nAccepted);
    }

    @Override
    public int beginGeneration(int reused) {
        if (reused < 0) {
            throw new IllegalArgumentException("reused prefix length " + reused + " is negative");
        }
        if (backend == null) {
            return 0;
        }
        if (deviceFeed != null) {
            deviceFeed.clear();
            devRows.clear();
        } else {
            feed.clear();
        }
        clearDeferred();
        hidden.clear();
        if (reused == 0) {
            for (MtpHead.Memory m : memories) {
                m.reset();
            }
            return 0;
        }
        if (!keepPrefix(reused)) {
            return reused;
        }
        // the decoder re-evaluates the last stored position: every head's rows below it are kept, and that call
        // supplies the hidden row the first new pair fuses
        return 1;
    }

    /** Keeps every head's cache rows of positions {@code [0, reused)}; false when they are gone or cannot drop. */
    private boolean keepPrefix(int reused) {
        for (MtpHead.Memory m : memories) {
            ModelMemory mm = m.memory();
            int past = mm.sequence().nPast();
            int drop = past - reused;
            if (past < reused || (drop > 0 && !mm.canRollback(drop))) {
                for (MtpHead.Memory reset : memories) {
                    reset.reset();
                }
                return false;
            }
        }
        for (MtpHead.Memory m : memories) {
            int drop = m.memory().sequence().nPast() - reused;
            if (drop > 0) {
                m.memory().rollback(drop);
            }
        }
        return true;
    }

    /** True when the heads' caches already hold the pair at {@code pos} (a row of a reused prompt prefix). */
    private boolean cached(int pos) {
        return memories[0].memory().sequence().nPast() > pos;
    }

    @Override
    public boolean readsTargetHidden() {
        return true;
    }

    @Override
    public void reset() {
        draftProposals.clear();
        draftCount = 0;
        if (backend == null) {
            return;
        }
        if (deviceFeed != null) {
            deviceFeed.clear();
            devRows.clear();
        } else {
            feed.clear();
        }
        clearDeferred();
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
        if (deviceFeed != null) {
            for (DeviceBatch b : deviceFeed.drain()) {
                processDevice(b);
                deviceFeed.releaseBatch(b);
            }
            return;
        }
        for (Batch b : feed.drain()) {
            process(b);
        }
    }

    /** Every head over the pairs of one host-fed target call (speculative.cpp:1622-1667). */
    private void process(Batch b) {
        // a call that rewrote deferred positions (a recurrent replay) drops their rows
        dropDeferredAtOrAfter(b.pos0);
        int m = b.size();
        if (!canDefer() || m > deferMax() || retainedBoundary(b)) {
            flushDeferred(b.pos0);
            runRows(b);
            return;
        }
        if (defer.size() + m > deferMax()) {
            // the catch-up rows the first pass could not take decode before this call defers
            flushDeferred(b.pos0);
        }
        deferBatch(b);
    }

    /**
     * True when {@code b} starts at the last position of a reused prompt prefix: its pair ({@code token[pos0]},
     * {@code h[pos0 - 1]}) is the one the caches already hold, so the call writes the next position's pair only.
     */
    private boolean retainedBoundary(Batch b) {
        return hidden.at(b.pos0 - 1) == null && cached(b.pos0);
    }

    /** {@link #retainedBoundary(Batch)} over a device feed. */
    private boolean retainedBoundary(DeviceBatch b) {
        return devRows.at(b.pos0 - 1) == null && cached(b.pos0);
    }

    /** Every head over the pairs of one target call, its cache writes only: no reader consumes the rows. */
    private void runRows(Batch b) {
        int m = b.size();
        float[] h = targetRows(b.rows[0], m);
        float[] before = hidden.at(b.pos0 - 1);
        int skip = before == null && cached(b.pos0) ? 1 : 0;
        for (MtpHead.Memory mem : memories) {
            mem.align(b.pos0 + skip);
        }
        float[] pairs = new float[Math.multiplyExact(Math.min(stepRows, m), hWidth)];
        for (int c = skip; c < m; c += stepRows) {
            int len = Math.min(stepRows, m - c);
            if (c == 0) {
                System.arraycopy(before != null ? before : hidden.pending(), 0, pairs, 0, hWidth);
                System.arraycopy(h, 0, pairs, hWidth, (len - 1) * hWidth);
            } else {
                System.arraycopy(h, (c - 1) * hWidth, pairs, 0, len * hWidth);
            }
            int[] tokens = Arrays.copyOfRange(b.tokens, c, c + len);
            for (int j = 0; j < blocks.size(); j++) {
                step(j, b.pos0 + c, tokens, pairs, len, true);
            }
        }
        hidden.record(b.pos0, h, m);
    }

    /** Every head over the pairs of one device-fed target call: the rows stay on the device. */
    private void processDevice(DeviceBatch b) {
        // a call that rewrote deferred positions (a recurrent replay) drops their rows
        dropDeferredAtOrAfter(b.pos0);
        int m = b.size();
        if (!canDefer() || m > deferMax() || retainedBoundary(b)) {
            flushDeferred(b.pos0);
            runDeviceRows(b);
            return;
        }
        if (defer.size() + m > deferMax()) {
            // the catch-up rows the first pass could not take decode before this call defers
            flushDeferred(b.pos0);
        }
        deferDeviceBatch(b);
    }

    /** The target norm the heads read the rows through, applied on the device, in place. */
    private void transform(Chunk c) {
        if (targetNormed) {
            head.norm(targetNormKind, MtpHead.Source.TARGET, targetNormName, targetNormEps, c.rows, c.rows, c.nRows);
        }
    }

    /** The call's rows through the heads, its cache writes only, in steps of at most {@code stepRows} rows. */
    private void runDeviceRows(DeviceBatch b) {
        int m = b.size();
        for (Chunk c : b.chunks) {
            transform(c);
        }
        int skip = devRows.at(b.pos0 - 1) == null && cached(b.pos0) ? 1 : 0;
        for (MtpHead.Memory mem : memories) {
            mem.align(b.pos0 + skip);
        }
        for (int c = skip; c < m; c += stepRows) {
            int len = Math.min(stepRows, m - c);
            if (c == 0) {
                TargetDeviceFeed.Located before = devRows.at(b.pos0 - 1);
                Rows src = before != null ? before.rows : devRows.pending();
                head.copyRows(src, before != null ? before.row : 0, devicePairs, 0, 1);
                copyRows(b.chunks, 0, len - 1, devicePairs, 1);
            } else {
                copyRows(b.chunks, c - 1, len, devicePairs, 0);
            }
            int[] tokens = Arrays.copyOfRange(b.tokens, c, c + len);
            for (int j = 0; j < blocks.size(); j++) {
                stepDevice(j, b.pos0 + c, tokens, devicePairs, len, true);
            }
        }
        devRows.record(b.chunks, b.pos0, m);
    }

    /** Copies {@code n} rows from global row {@code from} of a call's chunks into {@code dst} at {@code dstRow}. */
    private void copyRows(List<Chunk> chunks, int from, int n, Rows dst, int dstRow) {
        int base = 0;
        int done = 0;
        for (Chunk c : chunks) {
            int start = Math.max(from - base, 0);
            int end = Math.min(from + n - base, c.nRows);
            if (start < end) {
                head.copyRows(c.rows, start, dst, dstRow + (base + start - from), end - start);
                done += end - start;
            }
            base += c.nRows;
        }
        if (done != n) {
            throw new IllegalStateException("the call holds " + base + " rows, rows " + from + ".." + (from + n)
                    + " were asked");
        }
    }

    // ================================================================== deferred catch-up rows

    /** True when a small target call may defer its rows to the next draft's first pass. */
    private boolean canDefer() {
        return !chained && !cycling;
    }

    /** The most catch-up rows the next draft's first pass can decode beside its own row. */
    private int deferMax() {
        return Math.max(0, stepRows - 1);
    }

    /** Drops the deferred rows at or after {@code limit} (a later call rewrote their positions). */
    private void dropDeferredAtOrAfter(int limit) {
        for (Iterator<Deferred> it = defer.iterator(); it.hasNext(); ) {
            Deferred d = it.next();
            if (d.pos >= limit) {
                releaseDeferred(d);
                it.remove();
            }
        }
    }

    /** Drops the deferred rows at or after {@code limit} and returns how many are kept below it. */
    private int pendingDeferred(int limit) {
        dropDeferredAtOrAfter(limit);
        return defer.size();
    }

    private void releaseDeferred(Deferred d) {
        if (d.owned != null) {
            deviceFeed.release(d.owned);
        }
    }

    private void clearDeferred() {
        for (Deferred d : defer) {
            releaseDeferred(d);
        }
        defer.clear();
        deferBasis = 0;
    }

    /** Defers the call's catch-up rows: each token with the hidden row it pairs with, on the host. */
    private void deferBatch(Batch b) {
        int m = b.size();
        float[] h = targetRows(b.rows[0], m);
        float[] before = hidden.at(b.pos0 - 1);
        for (int i = 0; i < m; i++) {
            float[] row = new float[hWidth];
            if (i == 0) {
                System.arraycopy(before != null ? before : hidden.pending(), 0, row, 0, hWidth);
            } else {
                System.arraycopy(h, (i - 1) * hWidth, row, 0, hWidth);
            }
            defer.add(new Deferred(b.tokens[i], b.pos0 + i, row, null, 0, null));
        }
        hidden.record(b.pos0, h, m);
        deferBasis = m;
    }

    /** Defers the call's catch-up rows, their paired rows left on the device until the first pass decodes them. */
    private void deferDeviceBatch(DeviceBatch b) {
        int m = b.size();
        for (Chunk c : b.chunks) {
            transform(c);
        }
        for (int i = 0; i < m; i++) {
            Rows device;
            int deviceRow;
            Chunk owned = null;
            if (i == 0) {
                TargetDeviceFeed.Located at = devRows.at(b.pos0 - 1);
                if (at != null) {
                    device = at.rows;
                    deviceRow = at.row;
                } else {
                    device = devRows.pending();
                    deviceRow = 0;
                }
            } else {
                int[] local = new int[1];
                owned = chunkOf(b.chunks, i - 1, local);
                device = owned.rows;
                deviceRow = local[0];
            }
            if (owned != null) {
                deviceFeed.retain(owned);
            }
            defer.add(new Deferred(b.tokens[i], b.pos0 + i, null, device, deviceRow, owned));
        }
        devRows.record(b.chunks, b.pos0, m);
        deferBasis = m;
    }

    /** The chunk of a call's rows holding global row {@code row}, with its index there in {@code local[0]}. */
    private static Chunk chunkOf(List<Chunk> chunks, int row, int[] local) {
        int base = 0;
        for (Chunk c : chunks) {
            if (row < base + c.nRows) {
                local[0] = row - base;
                return c;
            }
            base += c.nRows;
        }
        throw new IllegalStateException("row " + row + " outside the call's " + base + " rows");
    }

    /** Keeps the deferred rows of the accepted prefix; the rejected draft tokens' rows are never decoded. */
    private void trimDeferred(int nAccepted) {
        int keep = Math.min(nAccepted, deferBasis - 1) + 1;
        while (defer.size() > Math.max(keep, 0)) {
            releaseDeferred(defer.remove(defer.size() - 1));
        }
    }

    /** Decodes the deferred catch-up rows below {@code limit} into the draft caches, their cache writes only. */
    private void flushDeferred(int limit) {
        dropDeferredAtOrAfter(limit);
        int n = defer.size();
        if (n == 0) {
            return;
        }
        int p0 = defer.get(0).pos;
        int[] tokens = new int[n];
        for (int i = 0; i < n; i++) {
            Deferred d = defer.get(i);
            if (d.pos != p0 + i) {
                throw new IllegalStateException("the deferred catch-up rows are not contiguous (" + p0 + " + " + i
                        + " is " + d.pos + ")");
            }
            tokens[i] = d.token;
        }
        for (MtpHead.Memory mem : memories) {
            mem.align(p0);
        }
        if (deviceFeed != null) {
            for (int i = 0; i < n; i++) {
                Deferred d = defer.get(i);
                head.copyRows(d.device, d.deviceRow, devicePairs, i, 1);
            }
            for (int j = 0; j < blocks.size(); j++) {
                stepDevice(j, p0, tokens, devicePairs, n, true);
            }
        } else {
            float[] pairs = new float[Math.multiplyExact(n, hWidth)];
            for (int i = 0; i < n; i++) {
                System.arraycopy(defer.get(i).row, 0, pairs, i * hWidth, hWidth);
            }
            for (int j = 0; j < blocks.size(); j++) {
                step(j, p0, tokens, pairs, n, true);
            }
        }
        clearDeferred();
    }

    /** The NextN blocks this drafter runs. */
    public int nextnLayers() {
        return blocks.size();
    }

    /** The stored GGUF name of a host weight the head multiplies, or null ({@link MtpHead#hostWeightName}). */
    public String hostWeightName(CpuTensorRef weight) {
        return head.hostWeightName(weight);
    }

    /**
     * Replays the head over the target's forward calls since the last replay, without drafting, so that the
     * matmuls of the NextN block are seen by an installed activation tap (llama.cpp tools/imatrix
     * {@code nextn_collector::decode}): each call runs the block over the pairs {@code (token[p], h[p - 1])} of its
     * positions, its first pair taking the previous call's last hidden row; a call without one (the first call
     * after {@link #reset()}, the start of a chunk) starts at its second position, as the reference skips the
     * pair that has no previous row. A block with its own LM head ({@code nextn.shared_head_head}) also runs its
     * head norm and head over every row, as the reference requests the MTP logits of every row exactly then.
     *
     * @throws UnsupportedOperationException for several NextN blocks (the reference collects one)
     */
    public void replay() {
        requireBound();
        if (blocks.size() > 1) {
            throw new UnsupportedOperationException("multi-layer NextN heads are not supported (found "
                    + blocks.size() + ")");
        }
        if (deviceFeed != null) {
            for (DeviceBatch b : deviceFeed.drain()) {
                replayCall(hostRows(b));
                deviceFeed.releaseBatch(b);
            }
            return;
        }
        for (Batch b : feed.drain()) {
            replayCall(b);
        }
    }

    /** A device batch's rows read to the host, for the replay path (an installed activation tap reads them). */
    private Batch hostRows(DeviceBatch b) {
        float[] rows = new float[Math.multiplyExact(b.nRows, tapWidth)];
        int off = 0;
        for (Chunk c : b.chunks) {
            float[] section = new float[Math.multiplyExact(c.nRows, tapWidth)];
            head.read(c.rows, c.nRows, section);
            System.arraycopy(section, 0, rows, off, section.length);
            off += section.length;
        }
        return new Batch(b.pos0, b.tokens, new float[][]{rows});
    }

    /** The block over the pairs of one target call, in steps of at most {@code stepRows} rows. */
    private void replayCall(Batch b) {
        int m = b.size();
        float[] h = targetRows(b.rows[0], m);
        float[] before = hidden.at(b.pos0 - 1);
        int skip = before == null ? 1 : 0;
        int rows = m - skip;
        Block block = blocks.get(0);
        boolean ownHead = block.headWeight.equals(
                nextnTensor(block.layer.weightLayer(), TensorNames.NEXTN_SHARED_HEAD_HEAD));
        for (int c = 0; c < rows; c += stepRows) {
            int len = Math.min(stepRows, rows - c);
            float[] pairs = new float[Math.multiplyExact(len, hWidth)];
            for (int i = 0; i < len; i++) {
                int src = skip + c + i - 1;
                System.arraycopy(src < 0 ? before : h, src < 0 ? 0 : src * hWidth, pairs, i * hWidth, hWidth);
            }
            int[] tokens = Arrays.copyOfRange(b.tokens, skip + c, skip + c + len);
            step(0, b.pos0 + skip + c, tokens, pairs, len, false);
            if (ownHead) {
                headRows(len);
            }
        }
        hidden.record(b.pos0, h, m);
    }

    /**
     * The head norm and the block's own LM head over the {@code len} block outputs in {@code x}, the head one row at
     * a time as one product of {@code len} rows for the matmul taps.
     */
    private void headRows(int len) {
        Block b = blocks.get(0);
        if (collapses[0] != null) {
            collapses[0].run(x, xn, len);
            if (b.headNormWeight != null) {
                head.norm(headNorm, source, b.headNormWeight, headEps, xn, xn, len);
            }
        } else {
            head.norm(headNorm, source, b.headNormWeight, headEps, x, xn, len);
        }
        try (TapProduct rows = TapProduct.dense(len)) {
            for (int r = 0; r < len; r++) {
                head.logits(b.headSource, b.headWeight, xn, r, logits);
            }
        }
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
     * and {@code eh_proj} apply per stream (deepseek4.cpp:1416-1427, qwen4exp.cpp:581-588). {@code kvOnly} runs the
     * stack's cache writes only: no reader consumes the rows of a call's pairs.
     */
    private void step(int j, int pos0, int[] tokens, float[] hRows, int len, boolean kvOnly) {
        head.write(hIn, hRows, len);
        stepBlocks(j, pos0, tokens, len, kvOnly);
    }

    /** {@link #step} over hidden rows already on the device ({@code hRows} rows {@code 0 ..}). */
    private void stepDevice(int j, int pos0, int[] tokens, Rows hRows, int len, boolean kvOnly) {
        head.copyRows(hRows, 0, hIn, 0, len);
        stepBlocks(j, pos0, tokens, len, kvOnly);
    }

    /** The blocks of head {@code j}, their input rows already in {@code hIn}. */
    private void stepBlocks(int j, int pos0, int[] tokens, int len, boolean kvOnly) {
        Block b = blocks.get(j);
        head.embed(b.embedSource, b.embedTable, tokens, len, embd);
        head.norm(inNorm, source, b.enorm, inEps, embd, embd, len);
        head.normStreams(inNorm, source, b.hnorm, inEps, hIn, hIn, len, streams);
        head.concatStreams(embd, hIn, cat, len, streams);
        head.projectStreams(source, b.ehProj, cat, x, len, streams);
        if (kvOnly) {
            stacks[j].runKV(pos0, len, tokens, x);
        } else {
            stacks[j].run(pos0, len, tokens, x);
        }
    }

    @Override
    public String toString() {
        return "NextnDraftModel{" + spec.arch().ggufName() + ", blocks " + blocks.get(0).layer.il() + ".."
                + blocks.get(blocks.size() - 1).layer.il() + (chained ? " chained" : cycling ? " cycled" : "")
                + (deviceFeed != null ? ", device rows" : "")
                + (streams > 1 ? ", " + streams + " streams" : "") + (meanStreams > 1 ? ", stream mean" : "")
                + (hFromBlock ? ", pre-norm h" : "") + (d2t != null ? ", d2t " + d2t.length + " of "
                + targetSpec.head().vocabRows() : "") + '}';
    }
}

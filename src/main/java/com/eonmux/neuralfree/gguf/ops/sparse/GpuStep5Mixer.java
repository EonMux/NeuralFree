package com.eonmux.neuralfree.gguf.ops.sparse;

import com.eonmux.neuralfree.acceleration.GpuModelUnsupportedException;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.gpu.block.GpuMixer;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.PreparedDispatch;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeights;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.KvLayout;
import com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan;
import com.eonmux.neuralfree.gguf.memory.StepLayout;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuKvLayer;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuMemory;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuRecurrentState;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuStepInputs;
import com.eonmux.neuralfree.gguf.ops.norm.NormKernels;
import com.eonmux.neuralfree.gguf.ops.rope.RopeKernels;
import com.eonmux.neuralfree.gguf.ops.rope.RopeMath;
import com.eonmux.neuralfree.gguf.spec.AttentionSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.MissingTensorException;
import com.eonmux.neuralfree.gguf.spec.RopeSpec;
import com.eonmux.neuralfree.gguf.spec.Step5Spec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.utils.Closeables;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Step5 sparse region attention mixer on the GPU ({@link com.eonmux.neuralfree.gguf.arch.MixerKind#STEP5}), the twin
 * of {@link CpuStep5Mixer} (vLLM {@code models/step5/nvidia/model.py}). Per step it projects Q/K/V, the head gate and
 * the indexer tensors with the context's matvec kernels; applies the per-head norms, the BF16/FP8 E4M3 roundings and
 * the RoPE with the shared norm and RoPE kernels; stores K/V at the step's cells; updates the region ring and the
 * closed-region summaries ({@link Step5Kernels#SUMMARY}); scores the closed regions
 * ({@link Step5Kernels#REGION_SCORES}), takes the per-token top-k ({@code LLM_TopK_Rows}) and lays the selection out
 * as {@link CpuStep5Sparse#selectRegions} does ({@link Step5Kernels#SELECT_REGIONS}); then gathers K/V over the
 * selected regions with the per-head gate ({@link Step5Kernels#attn(KvDType)}) and projects with {@code attn_output}.
 * A sliding block runs the dense windowed form of the same attention kernel.
 *
 * <p>The closed-region summaries and the open-region ring are the layer's {@code step5.summary} / {@code step5.ring}
 * recurrent-state buffers (slot 0, the contiguous single-sequence memory). The scorer runs in chunks of tokens bounded
 * by {@link #SCORE_BUDGET_FLOATS}. Per step only token counts, global sizes and the chunk staging buffers are
 * re-set; every working buffer is a {@code GpuScratch} temporary reserved at bind.</p>
 */
public final class GpuStep5Mixer implements GpuMixer {

    public static final String TEMP_Q = "step5.q";
    public static final String TEMP_K = "step5.k";
    public static final String TEMP_V = "step5.v";
    public static final String TEMP_GATE = "step5.gate";
    public static final String TEMP_ATT = "step5.att";
    public static final String TEMP_IQ = "step5.iq";
    public static final String TEMP_IK = "step5.ik";
    public static final String TEMP_IZ = "step5.iz";
    public static final String TEMP_IW = "step5.iw";
    public static final String TEMP_SCORES = "step5.scores";
    public static final String TEMP_TOPK = "step5.topk";
    public static final String TEMP_SEL = "step5.sel";
    public static final String TEMP_STAGE_IQ = "step5.stage.iq";
    public static final String TEMP_STAGE_IW = "step5.stage.iw";
    public static final String TEMP_STAGE_POS = "step5.stage.pos";
    public static final String TEMP_STAGE_SEL = "step5.stage.sel";

    /** Largest score buffer of one chunk: 8 Mi floats (32 MiB). */
    public static final long SCORE_BUDGET_FLOATS = 8L << 20;

    private final List<PreparedDispatch> prepared = new ArrayList<>();

    private GpuForwardContext ctx;
    private GpuOps ops;
    private GpuMemory memory;
    private GpuStepInputs steps;
    private String where;
    private int ubatch;
    private int local;
    private int nEmbd;
    private int nHead;
    private int nKv;
    private int headDim;
    private int headsPerGroup;
    private int qStride;
    private int kStride;
    private float kqScale;
    private int window;
    private boolean full;
    private Step5Spec step5;
    private AttentionSpec attn;

    private GpuKvLayer cache;
    private KvDType kvType;
    private int kvRows;
    private int group;
    private GpuBuffer summaryBuffer;
    private GpuBuffer ringBuffer;

    private RopeSpec rope;
    private RopeSpec idxRope;
    private GpuBuffer ropeFactors;
    private GpuBuffer idxFactors;

    private GpuWeight wQ;
    private GpuWeight wK;
    private GpuWeight wV;
    private GpuWeight wO;
    private GpuWeight wGate;
    private GpuWeight wIq;
    private GpuWeight wIk;
    private GpuWeight wIz;
    private GpuWeight wIProj;

    private int nIdx;
    private int proxy;
    private int topK;
    private int regionBlock;
    private int regions;
    private int selWidth;
    private int capacity;

    private ScratchSlot q;
    private ScratchSlot k;
    private ScratchSlot v;
    private ScratchSlot gate;
    private ScratchSlot att;
    private ScratchSlot iq;
    private ScratchSlot ik;
    private ScratchSlot iz;
    private ScratchSlot iw;
    private ScratchSlot scores;
    private ScratchSlot topk;
    private ScratchSlot sel;
    private ScratchSlot stageIq;
    private ScratchSlot stageIw;
    private ScratchSlot stagePos;
    private ScratchSlot stageSel;

    private PreparedDispatch qNorm;
    private PreparedDispatch kNorm;
    private PreparedDispatch iqNorm;
    private PreparedDispatch ikNorm;
    private PreparedDispatch castQBf16;
    private PreparedDispatch castKBf16;
    private PreparedDispatch castIqBf16;
    private PreparedDispatch castIqE4m3;
    private PreparedDispatch castIkBf16;
    private PreparedDispatch castIzBf16;
    private PreparedDispatch qRope;
    private PreparedDispatch kRope;
    private PreparedDispatch iqRope;
    private PreparedDispatch ikRope;
    private PreparedDispatch kStore;
    private PreparedDispatch vStore;
    private PreparedDispatch summary;
    private PreparedDispatch score;
    private PreparedDispatch selectTopK;
    private PreparedDispatch selectRegions;
    private PreparedDispatch attnDispatch;
    private int chunk;
    private int castLocal;
    private int summaryLocal;
    private int topKLocal;
    private int attnLocal;

    private boolean bound;
    private boolean closed;

    /** Use {@link Step5Blocks#gpuMixers()}. */
    GpuStep5Mixer() {
    }

    /**
     * Binds the mixer: validates the layer against its context, reserves the working buffers, binds the indexer of a
     * full layer and prepares every dispatch.
     *
     * @throws IllegalStateException         when already bound or closed, the context has no memory, or a tensor or
     *                                       cache contradicts the spec
     * @throws UnsupportedOperationException when a tensor type has no kernel or a per-token-head cache is bound
     * @throws MissingTensorException        when a required tensor is absent
     */
    @Override
    public void bind(GpuForwardContext context, LayerSpec layer) {
        Objects.requireNonNull(context, "context");
        Objects.requireNonNull(layer, "layer");
        if (bound || closed) {
            throw new IllegalStateException("Step5 mixer is already bound or closed");
        }
        ctx = context;
        try {
            bindLayer(layer);
            bound = true;
        } catch (RuntimeException e) {
            releaseAll(e);
            ctx = null;
            throw e;
        }
    }

    private void bindLayer(LayerSpec layer) {
        ops = ctx.ops();
        ubatch = ctx.scratch().ubatch();
        local = ops.local();
        step5 = layer.step5();
        if (step5 == null) {
            throw new IllegalArgumentException("layer " + Step5Blocks.describe(layer) + " has no Step5Spec");
        }
        where = "Step5 layer " + Step5Blocks.describe(layer);
        if (!ctx.hasMemory()) {
            throw new IllegalStateException(where + ": the Step5 mixer needs the model's KV memory");
        }
        memory = ctx.memory();
        steps = memory.stepInputs();
        attn = layer.attention();
        if (attn == null) {
            throw new IllegalArgumentException(where + ": the layer has no attention");
        }
        full = step5.full();
        bindAttention(layer);
        GpuWeights weights = ctx.weightSource().weightsFor(layer.weightLayer());
        wQ = matrix(weights.layer(layer, TensorNames.weight(TensorNames.ATTN_Q)), nEmbd, nHead * headDim, "attn_q");
        wK = matrix(weights.layer(layer, TensorNames.weight(TensorNames.ATTN_K)), nEmbd, nKv * headDim, "attn_k");
        wV = matrix(weights.layer(layer, TensorNames.weight(TensorNames.ATTN_V)), nEmbd, nKv * headDim, "attn_v");
        wO = matrix(weights.layer(layer, TensorNames.weight(TensorNames.ATTN_OUT)), nHead * headDim, nEmbd,
                "attn_output");
        wGate = matrix(weights.layer(layer, TensorNames.weight(TensorNames.ATTN_GATE)), nEmbd, nHead, "attn_gate");
        GpuWeight qNormW = vector(weights.layer(layer, TensorNames.weight(TensorNames.ATTN_Q_NORM)), headDim);
        GpuWeight kNormW = vector(weights.layer(layer, TensorNames.weight(TensorNames.ATTN_K_NORM)), headDim);

        q = temp(TEMP_Q, (long) ubatch * qStride);
        k = temp(TEMP_K, (long) ubatch * kStride);
        v = temp(TEMP_V, (long) ubatch * kStride);
        gate = temp(TEMP_GATE, (long) ubatch * nHead);
        att = temp(TEMP_ATT, (long) ubatch * qStride);

        NormKernels.checkReduceLocal(local);
        qNorm = prepareNorm(qNormW, q.buffer(), headDim, headDim, 0, NormKind.RMS);
        kNorm = prepareNorm(kNormW, k.buffer(), headDim, headDim, 0, NormKind.RMS);
        rope = layer.rope();
        ropeFactors = bindRopeFactors(weights, rope, layer.il());
        boolean hasMainFf = ropeFactors != null;
        if (rope != null) {
            qRope = prepare(RopeKernels.ROPE_EXT, ropeGlobal(rope, ubatch, nHead), new long[]{local},
                    RopeKernels.ropeExtArgs(rope, ubatch, nHead, headDim, qStride, rope.nOffs(), ubatch, 0,
                            hasMainFf, steps.ropePos(), hasMainFf ? ropeFactors : ph(), q.buffer()));
            kRope = prepare(RopeKernels.ROPE_EXT, ropeGlobal(rope, ubatch, nKv), new long[]{local},
                    RopeKernels.ropeExtArgs(rope, ubatch, nKv, headDim, kStride, rope.nOffs(), ubatch, 0,
                            hasMainFf, steps.ropePos(), hasMainFf ? ropeFactors : ph(), k.buffer()));
        }
        castLocal = Math.min(Step5Kernels.ELEMENT_LOCAL, local);
        castQBf16 = prepareCast(q, qStride);
        castKBf16 = prepareCast(k, kStride);
        bindStores(layer);
        if (full) {
            bindIndexer(layer, weights);
        }
    }

    private void bindAttention(LayerSpec layer) {
        if (attn.fusedQkv() || attn.headDimV() != attn.headDimK() || attn.vNorm()
                || attn.valueExperts() != null || attn.sinks() || attn.softcap() != 0f) {
            throw new IllegalArgumentException(where + " uses an attention feature the mixer does not implement"
                    + " (fused QKV, unequal K/V widths, V norm, value experts, sinks or a logit softcap)");
        }
        nEmbd = Step5Blocks.mixerWidth(layer);
        nHead = attn.nHead();
        nKv = attn.nHeadKv();
        headDim = attn.headDimK();
        if (nKv < 1 || nHead % nKv != 0) {
            throw new IllegalArgumentException(where + ": " + nHead + " query heads do not divide into " + nKv
                    + " KV heads");
        }
        headsPerGroup = nHead / nKv;
        qStride = nHead * headDim;
        kStride = nKv * headDim;
        kqScale = attn.kqScale();
        window = attn.isSwaLayer() ? attn.nSwa() : 0;
        cache = memory.kv(layer.il());
        if (cache == null) {
            throw new IllegalStateException(where + " has no KV cache");
        }
        KvLayout layout = cache.layout();
        if (layout.vAliasesK() || layout.nHeadKv() != nKv || layout.widthK() != nKv * headDim
                || layout.widthV() != nKv * headDim) {
            throw new IllegalStateException(where + ": the KV cache holds " + layout.nHeadKv() + " heads of "
                    + layout.widthK() + "/" + layout.widthV() + " values, the mixer needs " + nKv + " of "
                    + nKv * headDim);
        }
        kvType = layout.dtypeK();
        if (kvType.perTokenHead()) {
            throw new UnsupportedOperationException(where + ": the Step5 sparse attention has no kernel for the"
                    + " per-token-head KV type " + kvType);
        }
        if (!kvType.supportsWidth(layout.widthK()) || !kvType.supportsWidth(layout.widthV())) {
            throw new IllegalStateException(where + ": " + kvType + " rows cannot hold the " + layout.widthK() + "/"
                    + layout.widthV() + "-value attention rows");
        }
        kvRows = layout.rows();
        group = cache.group();
        steps.reserveSequenceAddressing(group);
    }

    private void bindStores(LayerSpec layer) {
        KvDType type = cache.layout().dtypeK();
        GpuBuffer scales = ctx.placeholders().kvScalesOnes();
        kStore = prepare(RopeKernels.kvStore(type),
                elementGlobal(RopeKernels.kvStoreWorkItems(type, ubatch, kStride)), new long[]{local},
                RopeKernels.kvStoreArgs(type, ubatch, kStride, kStride, 0, cache.layout().widthK(), headDim,
                        k.buffer(), steps.cellIdx(group), scales, cache.k()));
        if (!cache.layout().vAliasesK()) {
            vStore = prepare(RopeKernels.kvStore(type),
                    elementGlobal(RopeKernels.kvStoreWorkItems(type, ubatch, kStride)), new long[]{local},
                    RopeKernels.kvStoreArgs(type, ubatch, kStride, kStride, 0, cache.layout().widthV(), headDim,
                            v.buffer(), steps.cellIdx(group), scales, cache.v()));
        }
    }

    private void bindIndexer(LayerSpec layer, GpuWeights weights) {
        nIdx = step5.indexerHeads();
        proxy = step5.proxyDim();
        topK = step5.topK();
        regionBlock = step5.regionBlock();
        regions = step5.regions(memory.plan().nCtx());
        if (nIdx % nKv != 0) {
            throw new IllegalStateException(where + ": " + nIdx + " indexer heads do not divide into " + nKv
                    + " KV groups");
        }
        if (headsPerGroup != nIdx / nKv) {
            throw new IllegalStateException(where + ": " + nIdx + " indexer heads over " + nKv + " KV groups do not"
                    + " match " + headsPerGroup + " query heads per group");
        }
        selWidth = Math.min(topK, regions) + 1;
        wIq = matrix(weights.layer(layer, TensorNames.weight(TensorNames.INDEXER_ATTN_Q_B)), nEmbd, nIdx * proxy,
                "indexer.attn_q_b");
        wIk = matrix(weights.layer(layer, TensorNames.weight(TensorNames.INDEXER_ATTN_K)), nEmbd, proxy,
                "indexer.attn_k");
        wIz = matrix(weights.layer(layer, TensorNames.weight(TensorNames.INDEXER_Z)), nEmbd, proxy, "indexer.z");
        wIProj = matrix(weights.layer(layer, TensorNames.weight(TensorNames.INDEXER_PROJ)), nEmbd, nIdx,
                "indexer.proj");
        GpuWeight iqNormW = vector(weights.layer(layer, TensorNames.weight(TensorNames.INDEXER_Q_NORM)), proxy);
        GpuWeight ikNormW = vector(weights.layer(layer, TensorNames.weight(TensorNames.INDEXER_K_NORM)), proxy);
        GpuWeight ikNormB = vector(weights.layer(layer, TensorNames.bias(TensorNames.INDEXER_K_NORM)), proxy);

        iq = temp(TEMP_IQ, (long) ubatch * nIdx * proxy);
        ik = temp(TEMP_IK, (long) ubatch * proxy);
        iz = temp(TEMP_IZ, (long) ubatch * proxy);
        iw = temp(TEMP_IW, (long) ubatch * nIdx);
        iqNorm = prepareNorm(iqNormW, iq.buffer(), proxy, proxy, 0, NormKind.RMS);
        ikNorm = prepareNorm2(ikNormW, ikNormB, ik.buffer(), proxy, proxy, 0, NormKind.LAYER);
        idxRope = step5.indexerRope();
        idxFactors = bindRopeFactors(weights, idxRope, layer.il());
        boolean hasIdxFf = idxFactors != null;
        if (idxRope != null) {
            iqRope = prepare(RopeKernels.ROPE_EXT, ropeGlobal(idxRope, ubatch, nIdx), new long[]{local},
                    RopeKernels.ropeExtArgs(idxRope, ubatch, nIdx, proxy, nIdx * proxy, idxRope.nOffs(), ubatch, 0,
                            hasIdxFf, steps.ropePos(), hasIdxFf ? idxFactors : ph(), iq.buffer()));
            ikRope = prepare(RopeKernels.ROPE_EXT, ropeGlobal(idxRope, ubatch, 1), new long[]{local},
                    RopeKernels.ropeExtArgs(idxRope, ubatch, 1, proxy, proxy, idxRope.nOffs(), ubatch, 0,
                            hasIdxFf, steps.ropePos(), hasIdxFf ? idxFactors : ph(), ik.buffer()));
        }
        castIqBf16 = prepareCast(iq, nIdx * proxy);
        castIqE4m3 = prepareCast(iq, nIdx * proxy, Step5Kernels.CAST_MODE_E4M3);
        castIkBf16 = prepareCast(ik, proxy);
        castIzBf16 = prepareCast(iz, proxy);
        bindState(layer);
        prepareSelection();
    }

    private void bindState(LayerSpec layer) {
        GpuRecurrentState state = memory.recurrent(layer.il());
        if (state == null) {
            throw new IllegalStateException(where + ": the memory plan holds no Step5 region-summary state");
        }
        summaryBuffer = state.tensor(LayerMemoryPlan.STATE_STEP5_SUMMARY);
        ringBuffer = state.tensor(LayerMemoryPlan.STATE_STEP5_RING);
        int summaryFloats = rowFloats(state, LayerMemoryPlan.STATE_STEP5_SUMMARY);
        int ringFloats = rowFloats(state, LayerMemoryPlan.STATE_STEP5_RING);
        if (summaryFloats != regions * proxy) {
            throw new IllegalStateException(where + ": the summary state holds " + summaryFloats + " values, the"
                    + " mixer needs " + regions + " regions of " + proxy);
        }
        if (ringFloats % (2 * proxy) != 0) {
            throw new IllegalStateException(where + ": the ring state holds " + ringFloats + " values, not whole"
                    + " [key | z] rows of " + 2 * proxy);
        }
        capacity = ringFloats / (2 * proxy);
        long perToken = (long) Math.max(1, nKv) * regions;
        chunk = (int) Math.max(1L, Math.min(ubatch, SCORE_BUDGET_FLOATS / perToken));
        summaryLocal = Math.min(local, 256);
        summary = prepare(Step5Kernels.SUMMARY,
                Step5Kernels.summaryGlobal(ubatch, summaryLocal), Step5Kernels.local(summaryLocal),
                Step5Kernels.summaryArgs(ubatch, regionBlock, capacity, proxy, ik.buffer(), iz.buffer(),
                        steps.qPos(), ringBuffer, summaryBuffer));
    }

    private void prepareSelection() {
        int k = Math.min(topK, regions);
        scores = temp(TEMP_SCORES, (long) chunk * Math.max(1, nKv) * regions);
        topk = intTemp(TEMP_TOPK, (long) chunk * Math.max(1, nKv) * k);
        sel = intTemp(TEMP_SEL, (long) ubatch * Math.max(1, nKv) * selWidth);
        stageIq = temp(TEMP_STAGE_IQ, (long) chunk * nIdx * proxy);
        stageIw = temp(TEMP_STAGE_IW, (long) chunk * nIdx);
        stagePos = intTemp(TEMP_STAGE_POS, chunk);
        stageSel = intTemp(TEMP_STAGE_SEL, (long) chunk * Math.max(1, nKv) * selWidth);
        topKLocal = Math.min(SparseKernels.TOPK_LOCAL, local);
        SparseKernels.checkTopKLocal(topKLocal);
        score = prepare(Step5Kernels.REGION_SCORES,
                Step5Kernels.elementGlobal((long) chunk * Math.max(1, nKv) * regions, local), new long[]{local},
                Step5Kernels.regionScoreArgs(chunk, Math.max(1, nKv), nIdx, headsPerGroup, proxy, regions,
                        regionBlock, stageIq.buffer(), stageIw.buffer(), summaryBuffer, stagePos.buffer(),
                        scores.buffer()));
        PreparedDispatch topKDispatch = prepare(SparseKernels.TOPK_ROWS,
                SparseKernels.topKGlobal(chunk * Math.max(1, nKv), topKLocal),
                SparseKernels.local(topKLocal),
                new Object[]{chunk * Math.max(1, nKv), regions, k, scores.buffer(), topk.buffer()});
        selectRegions = prepare(Step5Kernels.SELECT_REGIONS,
                Step5Kernels.elementGlobal((long) chunk * Math.max(1, nKv),
                        Math.min(Step5Kernels.ELEMENT_LOCAL, local)),
                new long[]{Math.min(Step5Kernels.ELEMENT_LOCAL, local)},
                Step5Kernels.selectArgs(chunk * Math.max(1, nKv), k, Math.max(1, nKv), regionBlock,
                        regions, topk.buffer(), stagePos.buffer(), stageSel.buffer()));
        this.selectTopK = topKDispatch;
        attnLocal = Math.min(64, local);
        attnDispatch = prepare(Step5Kernels.attn(kvType),
                Step5Kernels.elementGlobal((long) ubatch * nHead, attnLocal), Step5Kernels.local(attnLocal),
                Step5Kernels.attnArgs(ubatch, nHead, nKv, headDim, regionBlock, selWidth,
                        window > 0 ? Step5Kernels.ATTN_MODE_WINDOW : Step5Kernels.ATTN_MODE_SPARSE, window, kqScale,
                        kvRows, cache.layout().widthK(), cache.layout().widthV(), q.buffer(), cache.k(), cache.v(),
                        steps.kvPos(group), sel.buffer(), steps.qPos(), gate.buffer(), att.buffer()));
    }

    // ------------------------------------------------------------------ steps

    @Override
    public void decode(int pos, ScratchSlot in, ScratchSlot out) {
        forward(pos, 1, in, out);
    }

    @Override
    public void prefill(int pos0, int nTokens, ScratchSlot in, ScratchSlot out) {
        forward(pos0, nTokens, in, out);
    }

    @Override
    public String batchRefusal() {
        return where + " runs the contiguous single-sequence step only; a paged multi-sequence Step5 step has no GPU"
                + " mixer";
    }

    private void forward(int pos0, int n, ScratchSlot in, ScratchSlot out) {
        ensureBound();
        checkStep(pos0, n, in, out);
        GpuBuffer x = in.buffer();
        int xOff = in.offset();
        // The attention and indexer projections of a single-token step all read this row, so its Q8_1
        // blocks are quantized once.
        GpuOps.SharedQ8 share = n == 1 && GpuOps.shareableQ8Activations(xOff, nEmbd)
                ? ops.shareQ8Activations(x, xOff, nEmbd) : null;
        try {
            projectMain(n, x, xOff);
            if (full) {
                projectIndexer(n, x, xOff);
                summary.setInt(Step5Kernels.ARG_SUMMARY_N_TOKENS, n);
                summary.setGlobal(Step5Kernels.summaryGlobal(n, summaryLocal));
                summary.dispatch();
                selectRegions(n);
            }
            int mode = window > 0 ? Step5Kernels.ATTN_MODE_WINDOW : Step5Kernels.ATTN_MODE_SPARSE;
            attnDispatch.setInt(Step5Kernels.ARG_ATTN_N_TOKENS, n);
            attnDispatch.setInt(Step5Kernels.ARG_ATTN_MODE, mode);
            attnDispatch.setGlobal(Step5Kernels.elementGlobal((long) n * nHead, attnLocal));
            attnDispatch.dispatch();
            ops.matmul(wO, att.buffer(), 0, out.buffer(), out.offset(), n);
        } finally {
            if (share != null) {
                share.close();
            }
        }
    }

    /** Q/K/V, the gate, the per-head norms, the BF16 roundings, RoPE and the cache stores. */
    private void projectMain(int n, GpuBuffer x, int xOff) {
        ops.matmul(wQ, x, xOff, q.buffer(), 0, n);
        ops.matmul(wK, x, xOff, k.buffer(), 0, n);
        ops.matmul(wV, x, xOff, v.buffer(), 0, n);
        ops.matmul(wGate, x, xOff, gate.buffer(), 0, n);
        rowNorm(qNorm, n * nHead);
        rowNorm(kNorm, n * nKv);
        runCast(castQBf16, (long) n * qStride);
        runCast(castKBf16, (long) n * kStride);
        rope(qRope, rope, n, nHead);
        rope(kRope, rope, n, nKv);
        runCast(castQBf16, (long) n * qStride);
        runCast(castKBf16, (long) n * kStride);
        kStore.setInt(RopeKernels.ARG_N_TOKENS, n);
        kStore.setGlobal(elementGlobal(RopeKernels.kvStoreWorkItems(kvType, n, kStride)));
        kStore.dispatch();
        if (vStore != null) {
            vStore.setInt(RopeKernels.ARG_N_TOKENS, n);
            vStore.setGlobal(elementGlobal(RopeKernels.kvStoreWorkItems(kvType, n, kStride)));
            vStore.dispatch();
        }
    }

    /** The indexer projections, norms, roundings and RoPE of the full layer. */
    private void projectIndexer(int n, GpuBuffer x, int xOff) {
        ops.matmul(wIq, x, xOff, iq.buffer(), 0, n);
        ops.matmul(wIk, x, xOff, ik.buffer(), 0, n);
        ops.matmul(wIz, x, xOff, iz.buffer(), 0, n);
        ops.matmul(wIProj, x, xOff, iw.buffer(), 0, n);
        rowNorm(iqNorm, n * nIdx);
        rowNorm(ikNorm, n);
        runCast(castIqBf16, (long) n * nIdx * proxy);
        runCast(castIkBf16, (long) n * proxy);
        rope(iqRope, idxRope, n, nIdx);
        rope(ikRope, idxRope, n, 1);
        runCast(castIqE4m3, (long) n * nIdx * proxy);
        runCast(castIkBf16, (long) n * proxy);
        runCast(castIzBf16, (long) n * proxy);
        float scale = step5.weightScale();
        if (scale != 1f) {
            ops.scale(iw.buffer(), 0, n * nIdx, scale);
        }
    }

    /** Scores and selects every token's closed regions in chunks bounded by {@link #SCORE_BUDGET_FLOATS}. */
    private void selectRegions(int n) {
        int k = Math.min(topK, regions);
        for (int t0 = 0; t0 < n; t0 += chunk) {
            int c = Math.min(chunk, n - t0);
            int rows = c * Math.max(1, nKv);
            int width = nIdx * proxy;
            ops.copy(iq.buffer(), t0 * width, stageIq.buffer(), 0, c * width);
            ops.copy(iw.buffer(), t0 * nIdx, stageIw.buffer(), 0, c * nIdx);
            ctx.queue().enqueueCopyBuffer(steps.qPos(), (long) t0 * Integer.BYTES, stagePos.buffer(), 0,
                    (long) c * Integer.BYTES);
            score.setInt(Step5Kernels.ARG_REGION_SCORES_N_TOKENS, c);
            score.setGlobal(Step5Kernels.elementGlobal((long) rows * regions, local));
            score.dispatch();
            selectTopK.setInt(SparseKernels.ARG_TOPK_N_ROWS, rows);
            selectTopK.setInt(SparseKernels.ARG_TOPK_N_COLS, regions);
            selectTopK.setInt(SparseKernels.ARG_TOPK_K, k);
            selectTopK.setGlobal(SparseKernels.topKGlobal(rows, topKLocal));
            selectTopK.dispatch();
            selectRegions.setInt(Step5Kernels.ARG_SELECT_N_ROWS, rows);
            selectRegions.setGlobal(Step5Kernels.elementGlobal(rows,
                    Math.min(Step5Kernels.ELEMENT_LOCAL, local)));
            selectRegions.dispatch();
            ctx.queue().enqueueCopyBuffer(stageSel.buffer(), 0, sel.buffer(),
                    (long) t0 * Math.max(1, nKv) * selWidth * Integer.BYTES,
                    (long) rows * selWidth * Integer.BYTES);
        }
    }

    // ------------------------------------------------------------------ bind helpers

    private PreparedDispatch prepareNorm(GpuWeight w, GpuBuffer x, int rowLen, int rowStride, int off, NormKind kind) {
        return prepareNorm2(w, null, x, rowLen, rowStride, off, kind);
    }

    private PreparedDispatch prepareNorm2(GpuWeight w, GpuWeight b, GpuBuffer x, int rowLen, int rowStride, int off,
                                          NormKind kind) {
        return prepare(NormKernels.ROW_NORM,
                NormKernels.reduceGlobal(NormKernels.rowNormGroups(ubatch), local), new long[]{local},
                NormKernels.rowNormArgs(kind, kind == NormKind.RMS ? ctx.spec().hparams().normRmsEps()
                                : ctx.spec().hparams().normEps(), ubatch, rowLen, rowStride, x, off, x, off,
                        w.shard(0), b == null ? null : b.shard(0), 1, ph()));
    }

    private PreparedDispatch prepareCast(ScratchSlot slot, int width) {
        return prepareCast(slot, width, Step5Kernels.CAST_MODE_BF16);
    }

    private PreparedDispatch prepareCast(ScratchSlot slot, int width, int mode) {
        return prepare(Step5Kernels.CAST, Step5Kernels.elementGlobal((long) ubatch * width, castLocal),
                Step5Kernels.local(castLocal),
                Step5Kernels.castArgs(ubatch * width, mode, slot.buffer(), slot.buffer()));
    }

    private void runCast(PreparedDispatch d, long n) {
        d.setInt(Step5Kernels.ARG_CAST_N, Math.toIntExact(n));
        d.setGlobal(Step5Kernels.elementGlobal(n, castLocal));
        d.dispatch();
    }

    private void rowNorm(PreparedDispatch d, int nRows) {
        d.setInt(NormKernels.ARG_ROW_NORM_N_ROWS, nRows);
        d.setGlobal(NormKernels.reduceGlobal(NormKernels.rowNormGroups(nRows), local));
        d.dispatch();
    }

    private void rope(PreparedDispatch d, RopeSpec r, int nTokens, int heads) {
        if (d == null) {
            return;
        }
        d.setInt(RopeKernels.ARG_N_TOKENS, nTokens);
        d.setInt(RopeKernels.ARG_POS_STRIDE, nTokens);
        d.setGlobal(elementGlobal(RopeKernels.ropeExtWorkItems(r.type(), nTokens, heads, r.nDims())));
        d.dispatch();
    }

    private long[] ropeGlobal(RopeSpec r, int n, int heads) {
        return elementGlobal(RopeKernels.ropeExtWorkItems(r.type(), n, heads, r.nDims()));
    }

    private long[] elementGlobal(long items) {
        return new long[]{RopeKernels.roundUp(Math.max(1L, items), local)};
    }

    /**
     * The layer's frequency factors as one device buffer, or null when its RoPE has none.
     *
     * @throws MissingTensorException when the spec names factors that are not resident
     * @throws IllegalStateException  when they are not one F32 buffer of at least {@code nDims / 2} values
     */
    private GpuBuffer bindRopeFactors(GpuWeights weights, RopeSpec r, int il) {
        if (r == null || r.freqFactors() == null) {
            return null;
        }
        GpuWeight w = weights.getOrNull(r.freqFactors().name());
        if (w == null) {
            throw new MissingTensorException(weights.archName(), il, r.freqFactors().name());
        }
        int pairs = RopeMath.pairCount(r.type(), r.nDims());
        if (w.type() != GGMLType.F32 || !w.isSingleShard() || w.rows() * w.cols() < pairs) {
            throw new IllegalStateException(where + ": " + w.name() + " must be one F32 buffer of at least " + pairs
                    + " frequency factors, got " + w);
        }
        return w.shard(0);
    }

    private GpuWeight matrix(GpuWeight w, long cols, long rows, String what) {
        Objects.requireNonNull(w, what);
        if (w.cols() != cols || w.rows() != rows) {
            throw new IllegalStateException(where + ": " + w.name() + " is " + w.cols() + "x" + w.rows()
                    + ", the Step5 mixer needs " + cols + "x" + rows);
        }
        if (!GpuOps.supportsWeightType(w.type())) {
            throw new GpuModelUnsupportedException(where + ": no GPU matvec kernels for " + w.name() + " of type "
                    + w.type());
        }
        return w;
    }

    private GpuWeight vector(GpuWeight w, int n) {
        if (w.type() != GGMLType.F32 || !w.isSingleShard() || w.cols() * w.rows() != n) {
            throw new IllegalStateException(where + ": " + w.name() + " must be a single-buffer F32 vector of " + n
                    + " values, got " + w);
        }
        return w;
    }

    private static int rowFloats(GpuRecurrentState state, String name) {
        for (LayerMemoryPlan.StateTensor t : state.tensors()) {
            if (t.name().equals(name)) {
                return Math.toIntExact(t.floats());
            }
        }
        throw new IllegalStateException("layer " + state.layer() + " has no state tensor '" + name + "' (has "
                + state.tensors() + ")");
    }

    private ScratchSlot temp(String name, long floats) {
        ScratchSlot s = ctx.scratch().temp(name, floats);
        if (s.offset() != 0) {
            throw new IllegalStateException(where + ": Step5 buffer " + s + " must start at element 0");
        }
        return s;
    }

    private ScratchSlot intTemp(String name, long ints) {
        return temp(name, ints);
    }

    private GpuBuffer ph() {
        return ctx.placeholders().readOnlyFloats();
    }

    private PreparedDispatch prepare(String kernel, long[] global, long[] localSize, Object[] args) {
        PreparedDispatch d = ops.prepare(kernel, global, localSize, args);
        prepared.add(d);
        return d;
    }

    private void checkStep(int pos0, int n, ScratchSlot in, ScratchSlot out) {
        if (n < 1 || n > ubatch || pos0 < 0) {
            throw new IllegalArgumentException(where + ": Step5 step of " + n + " tokens at " + pos0
                    + " is outside [1, " + ubatch + "] tokens at a non-negative position");
        }
        if (steps.pos0() != pos0 || steps.nTokens() != n) {
            throw new IllegalStateException(where + ": the step inputs hold " + steps.nTokens() + " tokens at "
                    + steps.pos0() + ", the Step5 step is " + n + " tokens at " + pos0);
        }
        long width = (long) n * nEmbd;
        Objects.requireNonNull(in, "in").require(width);
        Objects.requireNonNull(out, "out").require(width);
    }

    private void ensureBound() {
        if (closed) {
            throw new IllegalStateException("Step5 mixer is closed");
        }
        if (!bound) {
            throw new IllegalStateException("Step5 mixer is not bound");
        }
    }

    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        releaseAll(null);
    }

    private void releaseAll(RuntimeException cause) {
        RuntimeException failure = Closeables.closeAll(prepared, null);
        prepared.clear();
        if (failure != null) {
            if (cause != null) {
                cause.addSuppressed(failure);
            } else {
                throw failure;
            }
        }
    }

    @Override
    public String toString() {
        return "GpuStep5Mixer{" + (bound ? where + ", " + nHead + "/" + nKv + "x" + headDim
                + (window > 0 ? ", window " + window : ", topK " + topK + " of " + regions + " regions")
                : "unbound") + '}';
    }
}

package com.eonmux.neuralfree.gguf.ops.sparse;

import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.cpu.block.CpuMixer;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuForwardContext;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuScratch;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuOps;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.hparams.MropeTextPositions;
import com.eonmux.neuralfree.gguf.memory.KvLayout;
import com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan;
import com.eonmux.neuralfree.gguf.memory.StepLayout;
import com.eonmux.neuralfree.gguf.memory.cpu.CpuMemory;
import com.eonmux.neuralfree.gguf.memory.cpu.HeapModelMemory;
import com.eonmux.neuralfree.gguf.memory.cpu.KvAccessor;
import com.eonmux.neuralfree.gguf.memory.cpu.RecurrentStateAccessor;
import com.eonmux.neuralfree.gguf.memory.paged.PagedStep;
import com.eonmux.neuralfree.gguf.ops.rope.CpuRope;
import com.eonmux.neuralfree.gguf.ops.rope.RopeFreqTable;
import com.eonmux.neuralfree.gguf.ops.rope.RopePositions;
import com.eonmux.neuralfree.gguf.spec.AttentionSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.RopeSpec;
import com.eonmux.neuralfree.gguf.spec.Step5Spec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;

import java.util.Arrays;
import java.util.Objects;
import java.util.function.IntUnaryOperator;

/**
 * CPU twin of the Step5 GQA attention (vLLM {@code models/step5/nvidia/model.py} {@code Step5Attention} and
 * {@code common/ops}): per-head Q/K RMS norms with partial NeoX RoPE, the per-head sigmoid output gate, and on
 * full-attention layers the region indexer of {@link CpuStep5RegionCache}.
 *
 * <p>Call sequence of a step of {@code T} tokens at {@code pos0}:
 * <pre>
 *   matmul attn_q/k/v; norm RMS q/k per head and round BF16; NeoX rope; round BF16; writeK/writeV
 *   matmul attn_gate (the per-head logits)
 *   full indexer: matmul indexer.attn_q_b / attn_k / z / proj; RMS query norm; LayerNorm key; round BF16;
 *                 NeoX rope; round the query BF16 then FP8 and the key/z BF16; CpuStep5RegionCache.update per token;
 *                 score the closed regions of every (token, KV group) and select the top-k plus the open region
 *   attention: CpuStep5Attn.sparse over the selected regions, or CpuStep5Attn.windowed on a sliding block
 *   matmul attn_output
 * </pre>
 * The closed-region summaries and the open-region ring are the layer's {@code step5.summary} / {@code step5.ring}
 * recurrent-state tensors, so they follow a paged sequence's slot, snapshot/restore and session files. A contiguous
 * step runs directly; a paged multi-sequence step runs one segment at a time over each sequence's own state slot
 * ({@link #batch(StepLayout, float[], float[])}); a context shift of a contiguous sequence moves the ring and rebuilds
 * the closed-region summaries at their new positions ({@link #shiftContext(int, int)}). A paged sequence's shift is
 * refused by name, since its state lives in per-sequence slots the contiguous shift does not address. Not
 * thread-safe.
 */
public final class CpuStep5Mixer implements CpuMixer {

    private static final String T_Q = "step5.q";
    private static final String T_K = "step5.k";
    private static final String T_V = "step5.v";
    private static final String T_GATE = "step5.gate";
    private static final String T_ATT = "step5.att";
    private static final String T_IQ = "step5.iq";
    private static final String T_IK = "step5.ik";
    private static final String T_IZ = "step5.iz";
    private static final String T_IW = "step5.iw";
    private static final String T_SCORES = "step5.scores";
    private static final String T_SEL = "step5.sel";
    private static final String T_QPOS = "step5.qpos";
    private static final String T_CELLS = "step5.cells";
    private static final String T_ROPE_POS = "step5.rope_pos";
    private static final String T_ROW_OF_POS = "step5.row_of_pos";
    private static final String T_SEG_POS = "step5.seg_pos";
    private static final String T_SLOT_CELLS = "step5.slot_cells";
    private static final String T_SHIFT_ROW = "step5.shift_row";
    private static final String T_SHIFT_POS = "step5.shift_pos";

    private CpuForwardContext ctx;
    private CpuMatVec mv;
    private CpuMemory memory;
    private String where;
    private int ubatch;
    private int posLimit;
    private float rmsEps;
    private float normEps;
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
    private KvAccessor kv;
    private KvLayout kvLayout;
    private RopeSpec rope;
    private RopeFreqTable ropeTable;
    private RopeFreqTable shiftTable;
    private RopeSpec idxRope;
    private RopeFreqTable idxTable;
    private RopeFreqTable idxShiftTable;
    private int nIdx;
    private int proxy;
    private int topK;
    private int regionBlock;
    private int regions;
    private int selWidth;
    private int capacity;
    private CpuTensorRef wQ;
    private CpuTensorRef wK;
    private CpuTensorRef wV;
    private CpuTensorRef wO;
    private CpuTensorRef wGate;
    private CpuTensorRef wIq;
    private CpuTensorRef wIk;
    private CpuTensorRef wIz;
    private CpuTensorRef wIProj;
    private float[] qNorm;
    private float[] kNorm;
    private float[] iqNorm;
    private float[] ikNormW;
    private float[] ikNormB;
    private float[] q;
    private float[] k;
    private float[] v;
    private float[] gate;
    private float[] att;
    private float[] iq;
    private float[] ik;
    private float[] iz;
    private float[] iw;
    private float[] scores;
    private int[] sel;
    private int[] qPos;
    private int[] cells;
    private int[] ropePos;
    private float[] shiftRow;
    private int[] shiftPos;
    private int[] rowOfPos;
    private int[] segPos;
    private int[] slotCells;
    private CpuStep5RegionCache cache;
    private CpuStep5RegionCache[] slotCaches;
    private RecurrentStateAccessor state;
    private int summaryFloats;
    private int ringFloats;
    private int layerIl;
    private final float[] sums = new float[4];
    private int rowLimit;
    private IntUnaryOperator rowAt;
    private boolean bound;
    private boolean closed;

    // ================================================================== bind

    /**
     * @throws IllegalArgumentException when the layer is not a Step5 layer or uses a layout the mixer does not run
     * @throws IllegalStateException    when memory, a cache, a tensor or a planned state tensor has the wrong shape
     * @throws UnsupportedOperationException when the memory serves a paged multi-sequence plan
     * @throws com.eonmux.neuralfree.gguf.spec.MissingTensorException when a required tensor is absent
     */
    @Override
    public void bind(CpuForwardContext ctx, LayerSpec layer) {
        Objects.requireNonNull(ctx, "ctx");
        Objects.requireNonNull(layer, "layer");
        if (bound || closed) {
            throw new IllegalStateException("Step5 mixer is already bound or closed");
        }
        Step5Spec spec = layer.step5();
        if (spec == null) {
            throw new IllegalArgumentException("layer " + Step5Blocks.describe(layer) + " has no Step5Spec");
        }
        if (!ctx.hasMemory()) {
            throw new IllegalStateException("Step5 layer " + Step5Blocks.describe(layer) + " needs a KV cache");
        }
        this.ctx = ctx;
        this.step5 = spec;
        this.where = "Step5 layer " + Step5Blocks.describe(layer);
        this.mv = ctx.matVec();
        this.memory = ctx.memory();
        CpuScratch scratch = ctx.scratch();
        this.ubatch = scratch.ubatch();
        this.normEps = ctx.spec().hparams().normEps();
        this.rmsEps = ctx.spec().hparams().normRmsEps();
        this.attn = layer.attention();
        this.full = spec.full();
        bindAttention(layer, scratch);
        CpuWeights ws = ctx.weights();
        wQ = Step5Blocks.cpuMatrix(ws, layer, TensorNames.ATTN_Q, nEmbd, (long) nHead * headDim);
        wK = Step5Blocks.cpuMatrix(ws, layer, TensorNames.ATTN_K, nEmbd, (long) nKv * headDim);
        wV = Step5Blocks.cpuMatrix(ws, layer, TensorNames.ATTN_V, nEmbd, (long) nKv * headDim);
        wO = Step5Blocks.cpuMatrix(ws, layer, TensorNames.ATTN_OUT, (long) nHead * headDim, nEmbd);
        wGate = Step5Blocks.cpuMatrix(ws, layer, TensorNames.ATTN_GATE, nEmbd, nHead);
        qNorm = Step5Blocks.cpuVector(ws, layer, TensorNames.weight(TensorNames.ATTN_Q_NORM), headDim, true);
        kNorm = Step5Blocks.cpuVector(ws, layer, TensorNames.weight(TensorNames.ATTN_K_NORM), headDim, true);
        int ub = ubatch;
        q = scratch.temp(T_Q, Math.multiplyExact(ub, nHead * headDim));
        k = scratch.temp(T_K, Math.multiplyExact(ub, nKv * headDim));
        v = scratch.temp(T_V, Math.multiplyExact(ub, nKv * headDim));
        gate = scratch.temp(T_GATE, Math.multiplyExact(ub, nHead));
        att = scratch.temp(T_ATT, Math.multiplyExact(ub, nHead * headDim));
        qPos = scratch.intTemp(T_QPOS, ub);
        cells = scratch.intTemp(T_CELLS, ub);
        segPos = scratch.intTemp(T_SEG_POS, ub);
        ropePos = scratch.intTemp(T_ROPE_POS, Math.multiplyExact(MropeTextPositions.SECTIONS, ub));
        rowOfPos = scratch.intTemp(T_ROW_OF_POS, posLimit);
        slotCells = scratch.intTemp(T_SLOT_CELLS, posLimit);
        if (full) {
            bindIndexer(layer, scratch, ws);
        }
        rowAt = p -> p >= 0 && p < rowLimit ? rowOfPos[p] : -1;
        bound = true;
    }

    private void bindAttention(LayerSpec layer, CpuScratch scratch) {
        if (attn == null) {
            throw new IllegalArgumentException("Step5 layer " + Step5Blocks.describe(layer) + " has no attention");
        }
        if (attn.fusedQkv() || attn.headDimV() != attn.headDimK() || attn.vNorm()
                || attn.valueExperts() != null || attn.sinks() || attn.softcap() != 0f) {
            throw new IllegalArgumentException("Step5 layer " + Step5Blocks.describe(layer) + " uses an attention"
                    + " feature the mixer does not implement (fused QKV, unequal K/V widths, V norm, value experts,"
                    + " sinks or a logit softcap)");
        }
        nEmbd = Step5Blocks.mixerWidth(layer);
        nHead = attn.nHead();
        nKv = attn.nHeadKv();
        headDim = attn.headDimK();
        headsPerGroup = nHead / nKv;
        if (nHead % nKv != 0 || headsPerGroup * nKv != nHead) {
            throw new IllegalArgumentException(where + ": " + nHead + " query heads do not divide into " + nKv
                    + " KV heads");
        }
        qStride = nHead * headDim;
        kStride = nKv * headDim;
        kqScale = attn.kqScale();
        window = attn.isSwaLayer() ? attn.nSwa() : 0;
        kv = memory.kv(layer.il());
        if (kv == null) {
            throw new IllegalStateException(where + " has no KV cache");
        }
        kvLayout = kv.layout();
        if (kvLayout.vAliasesK() || kv.nHeadKv() != nKv || kv.headDimK() != headDim || kv.headDimV() != headDim) {
            throw new IllegalStateException(where + ": the KV cache holds " + kv.nHeadKv() + " heads of "
                    + kv.headDimK() + "/" + kv.headDimV() + ", the mixer needs " + nKv + " of " + headDim);
        }
        posLimit = ctx.memory().plan().nCtx();
        rope = layer.rope();
        float[] mainFactors = Step5Blocks.cpuFreqFactors(ctx.weights(), rope);
        ropeTable = rope == null ? null : RopeFreqTable.of(rope, mainFactors);
        shiftTable = rope == null ? null : RopeFreqTable.forShift(rope, mainFactors);
        shiftRow = scratch.temp(T_SHIFT_ROW, kStride);
        shiftPos = scratch.intTemp(T_SHIFT_POS, 1);
    }

    private void bindIndexer(LayerSpec layer, CpuScratch scratch, CpuWeights ws) {
        nIdx = step5.indexerHeads();
        proxy = step5.proxyDim();
        topK = step5.topK();
        regionBlock = step5.regionBlock();
        regions = step5.regions(posLimit);
        selWidth = Math.min(topK, regions) + 1;
        idxRope = step5.indexerRope();
        float[] idxFactors = Step5Blocks.cpuFreqFactors(ws, idxRope);
        idxTable = idxRope == null ? null : RopeFreqTable.of(idxRope, idxFactors);
        idxShiftTable = idxRope == null ? null : RopeFreqTable.forShift(idxRope, idxFactors);
        wIq = Step5Blocks.cpuMatrix(ws, layer, TensorNames.INDEXER_ATTN_Q_B, nEmbd, (long) nIdx * proxy);
        wIk = Step5Blocks.cpuMatrix(ws, layer, TensorNames.INDEXER_ATTN_K, nEmbd, proxy);
        wIz = Step5Blocks.cpuMatrix(ws, layer, TensorNames.INDEXER_Z, nEmbd, proxy);
        wIProj = Step5Blocks.cpuMatrix(ws, layer, TensorNames.INDEXER_PROJ, nEmbd, nIdx);
        iqNorm = Step5Blocks.cpuVector(ws, layer, TensorNames.weight(TensorNames.INDEXER_Q_NORM), proxy, true);
        ikNormW = Step5Blocks.cpuVector(ws, layer, TensorNames.weight(TensorNames.INDEXER_K_NORM), proxy, true);
        ikNormB = Step5Blocks.cpuVector(ws, layer, TensorNames.bias(TensorNames.INDEXER_K_NORM), proxy, true);
        int ub = ubatch;
        iq = scratch.temp(T_IQ, Math.multiplyExact(ub, nIdx * proxy));
        ik = scratch.temp(T_IK, Math.multiplyExact(ub, proxy));
        iz = scratch.temp(T_IZ, Math.multiplyExact(ub, proxy));
        iw = scratch.temp(T_IW, Math.multiplyExact(ub, nIdx));
        scores = scratch.temp(T_SCORES, Math.multiplyExact(Math.max(1, nKv), regions));
        sel = scratch.intTemp(T_SEL, Math.multiplyExact(Math.multiplyExact(ub, Math.max(1, nKv)), selWidth));
        bindCache(layer);
    }

    private void bindCache(LayerSpec layer) {
        layerIl = layer.il();
        state = memory.recurrent(layerIl);
        if (state == null) {
            throw new IllegalStateException(where + ": the memory plan holds no Step5 region-summary state");
        }
        summaryFloats = state.slotFloats(LayerMemoryPlan.STATE_STEP5_SUMMARY);
        ringFloats = state.slotFloats(LayerMemoryPlan.STATE_STEP5_RING);
        if (summaryFloats != regions * proxy) {
            throw new IllegalStateException(where + ": the summary state holds " + summaryFloats + " values, the"
                    + " mixer needs " + regions + " regions of " + proxy);
        }
        if (ringFloats % (2 * proxy) != 0) {
            throw new IllegalStateException(where + ": the ring state holds " + ringFloats + " values, not whole"
                    + " [key | z] rows of " + 2 * proxy);
        }
        capacity = ringFloats / (2 * proxy);
        slotCaches = new CpuStep5RegionCache[state.slots()];
        cache = slotCache(0);
    }

    /**
     * The region cache of state slot {@code slot}: a paged memory holds one per sequence, a contiguous one only
     * slot 0. Created on first use and reused afterwards.
     */
    private CpuStep5RegionCache slotCache(int slot) {
        if (slot < 0 || slot >= slotCaches.length) {
            throw new IllegalStateException(where + ": state slot " + slot + " is outside the " + slotCaches.length
                    + " slots of the Step5 state");
        }
        CpuStep5RegionCache c = slotCaches[slot];
        if (c == null) {
            c = new CpuStep5RegionCache(regionBlock, capacity, proxy, state.tensor(LayerMemoryPlan.STATE_STEP5_SUMMARY),
                    slot * summaryFloats, state.tensor(LayerMemoryPlan.STATE_STEP5_RING), slot * ringFloats);
            slotCaches[slot] = c;
        }
        return c;
    }

    // ================================================================== steps

    @Override
    public void decode(int pos, float[] in, float[] out) {
        run(pos, 1, in, out);
    }

    @Override
    public void prefill(int pos0, int n, float[] in, float[] out) {
        run(pos0, n, in, out);
    }

    private void run(int pos0, int n, float[] in, float[] out) {
        checkBound();
        checkStep(n, in, out);
        if (pos0 < 0 || pos0 + n > posLimit) {
            throw new IllegalArgumentException(where + ": step at positions [" + pos0 + ", " + (pos0 + n)
                    + ") exceeds the " + posLimit + " cached positions");
        }
        for (int t = 0; t < n; t++) {
            qPos[t] = pos0 + t;
            cells[t] = kvLayout.rowOf(pos0 + t);
        }
        projectMain(n, in);
        projectIndexer(n, in);
        int nPos = pos0 + n;
        Step5Blocks.fillContiguousRows(memory, kv.group(), kvLayout, nPos, rowOfPos);
        rowLimit = nPos;
        if (full) {
            updateRegions(0, n, cache);
        }
        if (window > 0) {
            CpuStep5Attn.windowed(n, nHead, nKv, headDim, window, q, 0, qStride, kqScale, kv, rowAt, qPos,
                    gate, 0, nHead, att, 0, qStride);
        } else {
            CpuStep5Attn.sparse(n, nHead, nKv, headDim, regionBlock, selWidth, q, 0, qStride, kqScale, kv, rowAt,
                    qPos, sel, 0, Math.max(1, nKv) * selWidth, selWidth, gate, 0, nHead, att, 0, qStride);
        }
        mv.matmul(wO, att, 0, out, 0, n);
    }

    /**
     * One step over the rows of {@code s}: the contiguous step when no paged step is current, else a paged
     * multi-sequence step. Every segment's rows project, normalize and rotate together (no stage mixes rows), are
     * stored at the step's cells, update the region state of their own sequence's slot, and attend through that
     * sequence's position &rarr; row map; the output projection runs over the whole step once.
     *
     * @throws UnsupportedOperationException when the step is not contiguous and the memory holds no paged step
     */
    @Override
    public void batch(StepLayout s, float[] in, float[] out) {
        checkBound();
        Objects.requireNonNull(s, "s");
        PagedStep step = memory.currentPagedStep();
        if (step == null) {
            CpuMixer.super.batch(s, in, out);
            return;
        }
        runPaged(s, step, in, out);
    }

    private void runPaged(StepLayout s, PagedStep step, float[] in, float[] out) {
        int n = s.nTokens();
        checkStep(n, in, out);
        if (ctx.stepNonCausal()) {
            throw new UnsupportedOperationException(where + ": the Step5 region attention and its summaries are"
                    + " causal; a non-causal step is not supported");
        }
        int group = kv.group();
        int[] stepCells = step.cellIdx(group);
        if (stepCells.length < n) {
            throw new IllegalStateException(where + ": the paged step addresses " + stepCells.length + " cells, the"
                    + " step holds " + n + " rows");
        }
        for (int seg = 0; seg < s.nSeg(); seg++) {
            int start = s.tokStart(seg);
            int len = s.len(seg);
            int pos0 = step.segPos0(seg);
            for (int i = 0; i < len; i++) {
                qPos[start + i] = pos0 + i;
                cells[start + i] = stepCells[start + i];
            }
        }
        projectMain(n, in);
        projectIndexer(n, in);
        for (int seg = 0; seg < s.nSeg(); seg++) {
            int start = s.tokStart(seg);
            int len = s.len(seg);
            CpuStep5RegionCache segCache = full ? slotCache(s.seq(seg)) : null;
            if (full) {
                updateRegions(start, len, segCache);
            }
            int nPos = MsaBlocks.fillSequenceRows(memory, group, step, s.seq(seg), slotCells, rowOfPos, where);
            System.arraycopy(qPos, start, segPos, 0, len);
            IntUnaryOperator rows = p -> p >= 0 && p < nPos ? rowOfPos[p] : -1;
            if (window > 0) {
                CpuStep5Attn.windowed(len, nHead, nKv, headDim, window, q, start * qStride, qStride, kqScale, kv,
                        rows, segPos, gate, start * nHead, nHead, att, start * qStride, qStride);
            } else {
                CpuStep5Attn.sparse(len, nHead, nKv, headDim, regionBlock, selWidth, q, start * qStride, qStride,
                        kqScale, kv, rows, segPos, sel, start * Math.max(1, nKv) * selWidth,
                        Math.max(1, nKv) * selWidth, selWidth, gate, start * nHead, nHead, att, start * qStride,
                        qStride);
            }
        }
        mv.matmul(wO, att, 0, out, 0, n);
    }

    /** Q/K/V and the gate projections, the per-head norms, RoPE and the cache writes. */
    private void projectMain(int n, float[] in) {
        mv.matmul(wQ, in, 0, q, 0, n);
        mv.matmul(wK, in, 0, k, 0, n);
        mv.matmul(wV, in, 0, v, 0, n);
        mv.matmul(wGate, in, 0, gate, 0, n);
        CpuOps.norm(NormKind.RMS, rmsEps, q, 0, q, 0, n * nHead, headDim, qNorm, null);
        CpuOps.norm(NormKind.RMS, rmsEps, k, 0, k, 0, n * nKv, headDim, kNorm, null);
        roundBf16(q, n * qStride);
        roundBf16(k, n * kStride);
        if (rope != null) {
            RopePositions.fill(rope.type(), rope.textPositions(), qPos, 0, n, ropePos);
            CpuRope.apply(ropeTable, q, 0, n, nHead, headDim, qStride, rope.nOffs(), ropePos, 0, n);
            CpuRope.apply(ropeTable, k, 0, n, nKv, headDim, kStride, rope.nOffs(), ropePos, 0, n);
        }
        roundBf16(q, n * qStride);
        roundBf16(k, n * kStride);
        for (int t = 0; t < n; t++) {
            kv.writeK(cells[t], k, t * kStride);
            kv.writeV(cells[t], v, t * kStride);
        }
    }

    /** The full layer's indexer projections, norms, RoPE and dtype roundings (the open row stores on update). */
    private void projectIndexer(int n, float[] in) {
        if (!full) {
            return;
        }
        mv.matmul(wIq, in, 0, iq, 0, n);
        mv.matmul(wIk, in, 0, ik, 0, n);
        mv.matmul(wIz, in, 0, iz, 0, n);
        mv.matmul(wIProj, in, 0, iw, 0, n);
        CpuOps.norm(NormKind.RMS, rmsEps, iq, 0, iq, 0, n * nIdx, proxy, iqNorm, null);
        CpuOps.norm(NormKind.LAYER, normEps, ik, 0, ik, 0, n, proxy, ikNormW, ikNormB);
        roundBf16(iq, n * nIdx * proxy);
        roundBf16(ik, n * proxy);
        if (idxTable != null) {
            RopePositions.fill(idxRope.type(), idxRope.textPositions(), qPos, 0, n, ropePos);
            CpuRope.apply(idxTable, iq, 0, n, nIdx, proxy, nIdx * proxy, idxRope.nOffs(), ropePos, 0, n);
            CpuRope.apply(idxTable, ik, 0, n, 1, proxy, proxy, idxRope.nOffs(), ropePos, 0, n);
        }
        float scale = step5.weightScale();
        for (int i = 0; i < n * nIdx; i++) {
            iw[i] *= scale;
        }
        for (int i = 0; i < n * nIdx * proxy; i++) {
            iq[i] = Step5Dtypes.e4m3(Step5Dtypes.bf16(iq[i]));
        }
        for (int i = 0; i < n * proxy; i++) {
            ik[i] = Step5Dtypes.bf16(ik[i]);
            iz[i] = Step5Dtypes.bf16(iz[i]);
        }
    }

    /** Rounds the first {@code n} values of {@code x} to BF16 in place (vLLM's normative cast points). */
    private static void roundBf16(float[] x, int n) {
        for (int i = 0; i < n; i++) {
            x[i] = Step5Dtypes.bf16(x[i]);
        }
    }

    /** Stores each token's open row and selects its regions once the regions it scores are closed. */
    private void updateRegions(int t0, int n, CpuStep5RegionCache regionCache) {
        for (int i = 0; i < n; i++) {
            int t = t0 + i;
            regionCache.update(qPos[t], ik, t * proxy, iz, t * proxy);
            for (int g = 0; g < nKv; g++) {
                select(t, g, regionCache);
            }
        }
    }

    /**
     * Scores the closed regions of token {@code t} and KV group {@code g} and writes its selection: the
     * {@code min(topK, regions)} best closed regions, then the open region. The vLLM score is
     * {@code sum_h relu(iq[t, g, h] . summary[r]) * w[t, g, h]}, accumulated in four independent FMA chains.
     */
    private void select(int t, int g, CpuStep5RegionCache regionCache) {
        int closed = Math.min(qPos[t] / regionBlock, regions);
        int outBase = (t * nKv + g) * selWidth;
        Arrays.fill(sel, outBase, outBase + selWidth, -1);
        if (closed == 0) {
            if (selWidth > 0) {
                sel[outBase + selWidth - 1] = 0;
            }
            return;
        }
        int scoreBase = g * regions;
        for (int r = 0; r < closed; r++) {
            Arrays.fill(sums, 0f);
            int sBase = regionCache.summaryOffset(r);
            for (int h = 0; h < headsPerGroup; h++) {
                int iqBase = t * nIdx * proxy + (g * headsPerGroup + h) * proxy;
                float dot = 0f;
                for (int d = 0; d < proxy; d++) {
                    dot += iq[iqBase + d] * regionCache.summaryArray()[sBase + d];
                }
                if (dot > 0f) {
                    sums[h & 3] += dot * iw[t * nIdx + g * headsPerGroup + h];
                }
            }
            scores[scoreBase + r] = ((sums[0] + sums[1]) + sums[2]) + sums[3];
        }
        int k = Math.min(topK, regions);
        if (selWidth > 0) {
            CpuStep5Sparse.selectRegions(scores, scoreBase, closed, k, sel, outBase);
        }
    }

    private void checkBound() {
        if (!bound || closed) {
            throw new IllegalStateException("Step5 mixer is not bound or already closed");
        }
    }

    private void checkStep(int n, float[] in, float[] out) {
        Objects.requireNonNull(in, "in");
        Objects.requireNonNull(out, "out");
        if (n < 1 || n > ubatch) {
            throw new IllegalArgumentException(where + ": step of " + n + " tokens is outside [1, " + ubatch + "]");
        }
    }

    @Override
    public String batchRefusal() {
        return where + " mixes several sequences per step only over paged memory (a currentPagedStep)";
    }

    /**
     * Applies a context shift: after the memory moved the KV rows, the region ring moves to the new positions,
     * its keys re-rotate by the shift delta, and every closed-region summary is rebuilt from the rows the region
     * now holds. The plan must have kept the whole sequence context in the ring
     * ({@code neuralfree.context.shift=true}).
     */
    @Override
    public void shiftContext(int nKeep, int nDiscard) {
        checkBound();
        if (nKeep < 0 || nDiscard <= 0) {
            throw new IllegalArgumentException("context shift needs nKeep >= 0 and nDiscard > 0, got " + nKeep + "/"
                    + nDiscard);
        }
        String refusal = shiftRefusal();
        if (refusal != null) {
            throw new UnsupportedOperationException("context shift unsupported: " + refusal);
        }
        int nNew = memory.sequence().nPast();
        int[] delta = ((HeapModelMemory) memory).shiftDelta()[kv.group()];
        rotateMainKeys(delta);
        if (!full) {
            return;
        }
        cache.shiftRows(nKeep, nDiscard, nNew);
        rotateRingKeys(nKeep, nDiscard, nNew);
        int closed = Math.min(nNew / regionBlock, regions);
        cache.rebuildSummaries(closed);
        cache.clearSummariesFrom(closed);
    }

    /** Re-rotates every stored K row by its shift delta ({@code LLM_KV_RopeShift}). */
    private void rotateMainKeys(int[] delta) {
        if (rope == null || shiftTable == null) {
            return;
        }
        int rows = Math.min(delta.length, kvLayout.rows());
        for (int r = 0; r < rows; r++) {
            for (int h = 0; h < nKv; h++) {
                kv.readK(r, h, shiftRow, h * headDim);
            }
            shiftPos[0] = delta[r];
            CpuRope.apply(shiftTable, shiftRow, 0, 1, nKv, headDim, kStride, rope.nOffs(), shiftPos, 0, 1);
            kv.writeK(r, shiftRow, 0);
        }
    }

    /** Re-rotates the ring keys of the positions a shift moved down, then rounds them to BF16 again. */
    private void rotateRingKeys(int nKeep, int nDiscard, int nNew) {
        if (idxRope == null || idxShiftTable == null) {
            return;
        }
        float[] ring = cache.ringArray();
        shiftPos[0] = -nDiscard;
        for (int p = nKeep; p < nNew; p++) {
            int off = cache.ringOffset(p);
            CpuRope.apply(idxShiftTable, ring, off, 1, 1, proxy, proxy, idxRope.nOffs(), shiftPos, 0, 1);
            for (int d = 0; d < proxy; d++) {
                ring[off + d] = Step5Dtypes.bf16(ring[off + d]);
            }
        }
    }

    /**
     * Null when {@link #shiftContext} can run: a contiguous memory whose plan kept the whole context in the Step5
     * region ring ({@code neuralfree.context.shift=true}). A paged sequence's state lives in per-sequence slots the
     * contiguous shift does not address.
     */
    @Override
    public String shiftRefusal() {
        checkBound();
        if (memory.pagedSequences() != null) {
            return where + " shifts a contiguous sequence only; a paged sequence's region state lives in per-sequence"
                    + " slots";
        }
        if (!(memory instanceof HeapModelMemory)) {
            return "context shift needs HeapModelMemory, the memory is " + memory.getClass().getSimpleName();
        }
        if (memory.plan().config() == null || !memory.plan().config().contextShift()) {
            return where + " was planned without context shift; set "
                    + com.eonmux.neuralfree.gguf.memory.MemoryConfig.PROP_CONTEXT_SHIFT + "=true";
        }
        if (full && capacity < posLimit) {
            return where + ": the region ring holds " + capacity + " rows, the " + posLimit + " positions a shift"
                    + " keeps; plan the memory with context shift so every kept row survives";
        }
        return null;
    }

    @Override
    public void close() {
        closed = true;
    }

    @Override
    public String toString() {
        if (!bound) {
            return "CpuStep5Mixer{unbound}";
        }
        return "CpuStep5Mixer{" + nHead + "/" + nKv + "x" + headDim + (window > 0 ? ", window " + window
                : ", topK " + topK + " of " + regions + " regions") + '}';
    }
}

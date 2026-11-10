package com.eonmux.neuralfree.gguf.ops.mla;

import com.eonmux.neuralfree.gguf.arch.AttnGateKind;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.cpu.block.CpuMixer;
import com.eonmux.neuralfree.gguf.cpu.block.CpuProjectionDeltas;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuForwardContext;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuScratch;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuParallel;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.memory.KvLayout;
import com.eonmux.neuralfree.gguf.memory.cpu.CpuMemory;
import com.eonmux.neuralfree.gguf.memory.cpu.HeapModelMemory;
import com.eonmux.neuralfree.gguf.memory.cpu.KvAccessor;
import com.eonmux.neuralfree.gguf.memory.paged.PagedStep;
import com.eonmux.neuralfree.gguf.ops.attn.AttnExtMath;
import com.eonmux.neuralfree.gguf.ops.attn.CpuAttentionCore;
import com.eonmux.neuralfree.gguf.ops.norm.NormMath;
import com.eonmux.neuralfree.gguf.ops.rope.CpuRope;
import com.eonmux.neuralfree.gguf.ops.rope.RopeFreqTable;
import com.eonmux.neuralfree.gguf.ops.rope.RopeKernels;
import com.eonmux.neuralfree.gguf.ops.rope.RopeMath;
import com.eonmux.neuralfree.gguf.ops.rope.RopePositions;
import com.eonmux.neuralfree.gguf.quant.CpuHadamard;
import com.eonmux.neuralfree.gguf.quant.HadamardSpec;
import com.eonmux.neuralfree.gguf.quant.TapProduct;
import com.eonmux.neuralfree.gguf.spec.AttentionSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.MissingTensorException;
import com.eonmux.neuralfree.gguf.spec.RopeSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;

import java.util.Objects;
import java.util.function.IntUnaryOperator;

/**
 * CPU twin of {@link GpuMlaMixer}: the same projections, RoPE, RMS norms, absorption, Hadamard rotations, cache
 * writes, attention and expansion, in the same order and on the same layouts, through {@link CpuMatVec},
 * {@link NormMath}, {@link CpuRope}, {@link AttnExtMath}, {@link CpuHadamard}, {@link KvAccessor} and
 * {@link CpuAttentionCore}. The per-head products with {@code attn_k_b} and {@code attn_v_b} run over every
 * (token, head) pair of a step in one parallel pass ({@link CpuMlaHeadProducts}).
 *
 * <p>Unlike the GPU, the CPU also runs a decompressed cache over absorbed weights: {@code k_nope = attn_k_b^T * kv}
 * per head from decoded {@code attn_k_b} rows, and {@code v = attn_v_b * kv} as one product over the
 * {@code [n_head * head_v][kv_lora]} rows of {@code attn_v_b}.</p>
 */
public final class CpuMlaMixer implements CpuMixer {

    static final String TEMP_Q_A = "mla.q_a";
    static final String TEMP_Q = "mla.q";
    static final String TEMP_KV_A = "mla.kv_a";
    static final String TEMP_LATENT = "mla.latent";
    static final String TEMP_Q_LATENT = "mla.q_latent";
    static final String TEMP_KV_B = "mla.kv_b";
    static final String TEMP_K = "mla.k";
    static final String TEMP_V = "mla.v";
    static final String TEMP_ATTN = "mla.attn";
    static final String TEMP_GATE = "mla.gate";
    static final String TEMP_WK_B = "mla.wk_b";
    static final String TEMP_SHIFT = "mla.shift";
    static final String TEMP_POS = "mla.pos";
    static final String TEMP_QPOS = "mla.qpos";
    static final String TEMP_DELTA = "mla.delta";
    /** Cache rows a context shift gathers, rotates and writes back per pass. */
    static final int SHIFT_ROWS = 64;

    private boolean bound;
    private boolean closed;

    private CpuMatVec mv;
    private CpuScratch scratch;
    private CpuMemory memory;
    private KvAccessor kv;
    private KvLayout layout;
    private IntUnaryOperator rowPos;
    private MlaBlocks.Geometry g;
    private boolean owns;
    private int il;
    private int nEmbd;
    private int nOut;
    private float eps;
    private float softcap;
    private AttnGateKind gateKind;
    private AttentionSpec.Temperature temperature;
    private CpuAttentionCore.Params params;
    /** {@link #params} without the causal mask, for the steps the context flags non-causal (a media chunk). */
    private CpuAttentionCore.Params nonCausalParams;
    private CpuForwardContext ctx;
    private RopeSpec rope;
    private float[] freqFactors;
    private RopeFreqTable ropeTable;
    private String shiftBlocker;
    private HadamardSpec rotKSpec;
    private HadamardSpec rotVSpec;
    /** The memory's current paged step, or null on the contiguous path ({@code CpuMemory.currentPagedStep}). */
    private PagedStep pagedStep;

    private CpuTensorRef wq;
    private CpuTensorRef wqA;
    private CpuTensorRef wqB;
    private CpuTensorRef wkvA;
    private CpuTensorRef wkvB;
    private CpuTensorRef wkB;
    private CpuTensorRef wvB;
    private CpuTensorRef wo;
    private CpuTensorRef wGate;
    private CpuProjectionDeltas.Site qDeltas;
    private CpuProjectionDeltas.Site qaDeltas;
    private CpuProjectionDeltas.Site qbDeltas;
    private CpuProjectionDeltas.Site kvaDeltas;
    private CpuProjectionDeltas.Site kvbDeltas;
    private CpuProjectionDeltas.Site outDeltas;
    private CpuProjectionDeltas.Site gateDeltas;
    private float[] qaNorm;
    private float[] kvaNorm;
    /** {@code attn_k_norm} of the shared rotary key, or null when the layer applies none. */
    private float[] kPeNorm;
    /** Per-head {@code attn_sinks} logits, or null without sinks. */
    private float[] sinks;
    private float[] gateBias;

    CpuMlaMixer() {
    }

    // ================================================================== bind

    /**
     * @throws IllegalArgumentException      when the layer is not an MLA layer
     * @throws UnsupportedOperationException for a variant {@link MlaBlocks#unsupportedCommon} names
     * @throws IllegalStateException         when the model has no memory, a tensor or the cache has the wrong shape,
     *                                       or the mixer is already bound or closed
     * @throws MissingTensorException        when a required tensor is absent
     */
    @Override
    public void bind(CpuForwardContext ctx, LayerSpec layer) {
        Objects.requireNonNull(ctx, "ctx");
        MlaBlocks.checkKind(layer);
        if (bound || closed) {
            throw new IllegalStateException("MLA mixer is already bound or closed");
        }
        String why = MlaBlocks.unsupportedCommon(layer);
        if (why != null) {
            throw new UnsupportedOperationException(why);
        }
        if (!ctx.hasMemory()) {
            throw new IllegalStateException("layer " + layer.il() + ": the MLA mixer needs the model's KV memory");
        }
        il = layer.il();
        this.ctx = ctx;
        mv = ctx.matVec();
        scratch = ctx.scratch();
        memory = ctx.memory();
        kv = memory.kv(il);
        if (kv == null) {
            throw new IllegalStateException("layer " + il + ": the memory holds no KV cache for this MLA layer");
        }
        layout = kv.layout();
        owns = memory.plan().layer(il).ownsKv();
        g = MlaBlocks.Geometry.of(layer, layout, memory.plan().layer(il).rotation());
        checkAccessor();
        final CpuMemory mem = memory;
        final int group = kv.group();
        rowPos = r -> mem.positionOfRow(group, r);
        rotKSpec = g.rotK > 0 ? HadamardSpec.rotation(g.rotK) : null;
        rotVSpec = g.rotV > 0 ? HadamardSpec.rotation(g.rotV) : null;
        eps = ctx.spec().hparams().normRmsEps();
        softcap = MlaBlocks.softcap(layer);
        gateKind = MlaBlocks.gateKind(layer);
        temperature = MlaBlocks.temperature(layer);
        bindWeights(ctx.weights(), layer);
        CpuAttentionCore.Builder pb = CpuAttentionCore.Params.builder(g.nHead).scale(g.kqScale).softcap(softcap)
                .window(layout.window());
        if (sinks != null) {
            pb.sinks(sinks);
        }
        params = pb.build();
        nonCausalParams = params.nonCausal();
        bindDeltas(ctx.projectionDeltas(), layer);
        bindRope(ctx.weights(), layer);
        reserveTemps(layer);
        bound = true;
    }

    /**
     * Requests every temporary a step or a context shift uses at its largest size (a full ubatch), so the steps are
     * served by these arrays and allocate nothing.
     */
    private void reserveTemps(LayerSpec layer) {
        long u = scratch.ubatch();
        long rows = u * g.nHead;
        scratch.intTemp(TEMP_QPOS, MlaBlocks.toInt(u, "MLA query positions"));
        scratch.temp(TEMP_Q, MlaBlocks.toInt(u * g.qWidth, "MLA query floats"));
        scratch.temp(TEMP_KV_A, MlaBlocks.toInt(u * g.kvaWidth, "MLA latent floats"));
        scratch.temp(TEMP_ATTN, MlaBlocks.toInt(rows * g.headV, "MLA attention floats"));
        if (wqA != null) {
            scratch.temp(TEMP_Q_A, MlaBlocks.toInt(u * g.qLora, "MLA low-rank query floats"));
        }
        if (rope != null) {
            scratch.intTemp(TEMP_POS, MlaBlocks.toInt(RopeMath.positionsPerToken(rope.type()) * u, "MLA positions"));
        }
        if (g.latent) {
            scratch.temp(TEMP_Q_LATENT, MlaBlocks.toInt(rows * g.dQ, "MLA absorbed query floats"));
            scratch.temp(TEMP_LATENT, MlaBlocks.toInt(rows * g.kvLora, "MLA latent output floats"));
        } else {
            scratch.temp(TEMP_K, MlaBlocks.toInt(rows * g.headK, "MLA key floats"));
            scratch.temp(TEMP_V, MlaBlocks.toInt(rows * g.headV, "MLA value floats"));
            if (wkvB != null) {
                scratch.temp(TEMP_KV_B, MlaBlocks.toInt(rows * (g.nope + g.headV), "MLA decompressed floats"));
            } else {
                scratch.temp(TEMP_WK_B, MlaBlocks.toInt((long) g.kvLora * g.nope, "MLA attn_k_b head floats"));
            }
        }
        if (wGate != null) {
            scratch.temp(TEMP_GATE, MlaBlocks.toInt(u * MlaBlocks.gateWidth(layer), "MLA gate floats"));
        }
        if (owns && rope != null && shiftBlocker == null) {
            scratch.temp(TEMP_SHIFT, MlaBlocks.toInt((long) SHIFT_ROWS * g.shiftRowStride, "MLA shift floats"));
            scratch.intTemp(TEMP_DELTA, 2 * SHIFT_ROWS);
        }
    }

    /** The accessor must expose the same heads the geometry attends: one latent head or {@code n_head} heads. */
    private void checkAccessor() {
        int wantHeads = g.nHeadKv;
        int wantK = g.latent ? g.kvaWidth : g.headK;
        int wantV = g.latent ? g.kvLora : g.headV;
        if (kv.nHeadKv() != wantHeads || kv.headDimK() != wantK || kv.headDimV() != wantV) {
            throw new IllegalStateException("layer " + il + ": the KV accessor exposes " + kv.nHeadKv() + " heads of "
                    + kv.headDimK() + "/" + kv.headDimV() + ", the MLA layer attends " + wantHeads + " heads of "
                    + wantK + "/" + wantV);
        }
    }

    private void bindWeights(CpuWeights ws, LayerSpec layer) {
        wkvA = ws.layer(layer, TensorNames.weight(TensorNames.ATTN_KV_A_MQA));
        nEmbd = MlaBlocks.toInt(wkvA.cols(), "MLA input width");
        MlaBlocks.checkShape(name(ws, wkvA), wkvA.cols(), wkvA.rows(), nEmbd, g.kvaWidth);
        if (g.qLora > 0) {
            wqA = ws.layer(layer, TensorNames.weight(TensorNames.ATTN_Q_A));
            wqB = ws.layer(layer, TensorNames.weight(TensorNames.ATTN_Q_B));
            MlaBlocks.checkShape(name(ws, wqA), wqA.cols(), wqA.rows(), nEmbd, g.qLora);
            MlaBlocks.checkShape(name(ws, wqB), wqB.cols(), wqB.rows(), g.qLora, g.qWidth);
            qaNorm = vector(ws, layer, TensorNames.ATTN_Q_A_NORM, g.qLora);
        } else {
            wq = ws.layer(layer, TensorNames.weight(TensorNames.ATTN_Q));
            MlaBlocks.checkShape(name(ws, wq), wq.cols(), wq.rows(), nEmbd, g.qWidth);
        }
        kvaNorm = vector(ws, layer, TensorNames.ATTN_KV_A_NORM, g.kvLora);
        if (MlaBlocks.kPeNorm(layer)) {
            kPeNorm = vector(ws, layer, TensorNames.ATTN_K_NORM, g.rope);
        }
        if (MlaBlocks.sinks(layer)) {
            sinks = vector(ws, layer, TensorNames.ATTN_SINKS, g.nHead);
        }
        if (g.latent || g.absorbed) {
            wkB = ws.layer(layer, TensorNames.weight(TensorNames.ATTN_K_B));
            wvB = ws.layer(layer, TensorNames.weight(TensorNames.ATTN_V_B));
            MlaBlocks.checkShape(name(ws, wkB), wkB.cols(), wkB.rows(), g.nope, (long) g.kvLora * g.nHead);
            MlaBlocks.checkShape(name(ws, wvB), wvB.cols(), wvB.rows(), g.kvLora, (long) g.headV * g.nHead);
            requirePlainHeads(ws, wkB);
            requirePlainHeads(ws, wvB);
            CpuMatVec.vecDot(wkB.type());
            CpuMatVec.vecDot(wvB.type());
        } else {
            wkvB = ws.layer(layer, TensorNames.weight(TensorNames.ATTN_KV_B));
            MlaBlocks.checkShape(name(ws, wkvB), wkvB.cols(), wkvB.rows(), g.kvLora,
                    (long) g.nHead * (g.nope + g.headV));
        }
        wo = ws.layer(layer, TensorNames.weight(TensorNames.ATTN_OUT));
        MlaBlocks.checkShape(name(ws, wo), wo.cols(), wo.rows(), (long) g.nHead * g.headV, wo.rows());
        nOut = MlaBlocks.toInt(wo.rows(), "MLA output width");
        if (gateKind != AttnGateKind.NONE) {
            wGate = ws.layer(layer, TensorNames.weight(TensorNames.ATTN_GATE));
            MlaBlocks.checkShape(name(ws, wGate), wGate.cols(), wGate.rows(), nEmbd, MlaBlocks.gateWidth(layer));
            gateBias = ws.layerFloatsOrNull(layer, TensorNames.bias(TensorNames.ATTN_GATE));
            if (gateBias != null && gateBias.length != g.nHead) {
                throw new IllegalStateException("layer " + il + ": attn_gate.bias holds " + gateBias.length
                        + " values, the gate needs one per head (" + g.nHead + ")");
            }
            if (gateBias == null && AttnExtMath.gateRequiresBias(gateKind)) {
                throw new MissingTensorException(ws.archName(), il, TensorNames.bias(TensorNames.ATTN_GATE));
            }
        }
    }

    /**
     * The delta sites of the projections this layer runs ({@code build_lora_mm}, llama-graph.cpp:1629-1667); the
     * latent and absorbed paths never multiply {@code attn_kv_b} and refuse deltas on it.
     */
    private void bindDeltas(CpuProjectionDeltas deltas, LayerSpec layer) {
        kvaDeltas = deltas.serve(layer, TensorNames.weight(TensorNames.ATTN_KV_A_MQA), wkvA, mv);
        if (wqA != null) {
            qaDeltas = deltas.serve(layer, TensorNames.weight(TensorNames.ATTN_Q_A), wqA, mv);
            qbDeltas = deltas.serve(layer, TensorNames.weight(TensorNames.ATTN_Q_B), wqB, mv);
        } else {
            qDeltas = deltas.serve(layer, TensorNames.weight(TensorNames.ATTN_Q), wq, mv);
        }
        if (wkvB != null) {
            kvbDeltas = deltas.serve(layer, TensorNames.weight(TensorNames.ATTN_KV_B), wkvB, mv);
        } else {
            deltas.serve(layer, TensorNames.weight(TensorNames.ATTN_KV_B), GpuMlaMixer.ABSORBED_KV_B_REFUSAL);
        }
        outDeltas = deltas.serve(layer, TensorNames.weight(TensorNames.ATTN_OUT), wo, mv);
        if (wGate != null) {
            gateDeltas = deltas.serve(layer, TensorNames.weight(TensorNames.ATTN_GATE), wGate, mv);
        }
    }

    private void bindRope(CpuWeights ws, LayerSpec layer) {
        rope = MlaBlocks.rope(layer);
        shiftBlocker = g.shiftBlocker(rope, layout);
        if (rope == null) {
            return;
        }
        if (rope.freqFactors() != null) {
            freqFactors = ws.floats(rope.freqFactors());
            if (freqFactors == null) {
                throw new MissingTensorException(ws.archName(), il, rope.freqFactors().name());
            }
        }
        ropeTable = RopeFreqTable.of(rope, freqFactors);
    }

    /**
     * {@code attn_k_b} / {@code attn_v_b} enter plain {@code ggml_mul_mat} products (deepseek2.cpp:120,
     * llama-graph.cpp build_attn {@code v_mla}); the GPU twin applies no transform to them either, the per-head
     * products ({@link CpuMlaHeadProducts}) apply none, and the decompressed path decodes their raw rows.
     */
    private void requirePlainHeads(CpuWeights ws, CpuTensorRef w) {
        CpuMatVec.Transforms t = mv.transformsOf(w);
        if (!t.transform().isIdentity() || t.hasScale()) {
            throw new UnsupportedOperationException("layer " + il + ": " + name(ws, w) + " carries " + t
                    + "; the per-head MLA products take plain weights");
        }
    }

    private static float[] vector(CpuWeights ws, LayerSpec layer, String base, int len) {
        String suffix = TensorNames.weight(base);
        float[] v = ws.layerFloatsOrNull(layer, suffix);
        if (v == null) {
            throw new MissingTensorException(ws.archName(), layer.il(), suffix);
        }
        if (v.length != len) {
            throw new IllegalStateException("layer " + layer.il() + ": " + suffix + " holds " + v.length
                    + " values, the MLA layer needs " + len);
        }
        return v;
    }

    private static String name(CpuWeights ws, CpuTensorRef w) {
        return ws.describe(w).name();
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

    /**
     * The cache writes of a step: the latent KV projection with its norms and RoPE, then the latent store (a latent
     * cache) or the {@code attn_kv_b} decompression, the K/V rotations and both stores. The query projection, the
     * absorption, the attention, the output expansion and the output projection do not run.
     */
    @Override
    public void prefillKV(int pos0, int n, float[] in) {
        checkBound();
        Objects.requireNonNull(in, "in");
        if (n < 1 || n > scratch.ubatch()) {
            throw new IllegalArgumentException("MLA step of " + n + " tokens is outside [1, " + scratch.ubatch() + "]");
        }
        if (pos0 < 0) {
            throw new IllegalArgumentException("MLA step position " + pos0 + " is negative");
        }
        if (in.length < (long) n * nEmbd) {
            throw new IllegalArgumentException("MLA step of " + n + " tokens needs " + n * (long) nEmbd + " inputs, "
                    + "got " + in.length);
        }
        if (!owns) {
            return;
        }
        pagedStep = memory.currentPagedStep();
        int[] qPos = scratch.intTemp(TEMP_QPOS, n);
        if (pagedStep != null) {
            System.arraycopy(scratch.ints(CpuScratch.POS), 0, qPos, 0, n);
        } else {
            for (int t = 0; t < n; t++) {
                qPos[t] = pos0 + t;
            }
        }
        float[] kva = scratch.temp(TEMP_KV_A, n * g.kvaWidth);
        projectKV(qPos, 0, n, in, kva);
        if (g.latent) {
            latentStepKV(qPos, 0, n, kva);
        } else {
            decompressedStepKV(qPos, 0, n, kva);
        }
    }

    @Override
    public String kvRefusal() {
        checkBound();
        return null;
    }

    /**
     * The latent row alone: the projection, the optional RMS norm of the shared rotary key, RoPE of the rotary
     * dims and the RMS norm of the latent prefix. The query's products, its norm and its RoPE do not run.
     */
    private void projectKV(int[] qPos, int rowOff, int n, float[] in, float[] kva) {
        int inOff = rowOff * nEmbd;
        mv.matmul(wkvA, in, inOff, kva, rowOff * g.kvaWidth, n);
        kvaDeltas.applyRows(n, in, inOff, kva, rowOff * g.kvaWidth);
        if (kPeNorm != null) {
            NormMath.rowNorm(NormKind.RMS, eps, n, g.rope, g.kvaWidth, 1, kva, rowOff * g.kvaWidth + g.kvLora,
                    kPeNorm, null, kva, rowOff * g.kvaWidth + g.kvLora);
        }
        if (rope != null) {
            int[] pos = scratch.intTemp(TEMP_POS, RopeMath.positionsPerToken(rope.type()) * n);
            RopePositions.fill(rope.type(), rope.textPositions(), qPos, rowOff, n, pos);
            CpuRope.apply(ropeTable, kva, rowOff * g.kvaWidth, n, 1, g.kvaWidth, g.kvaWidth, g.kvLora, pos, 0, n);
        }
        NormMath.rowNorm(NormKind.RMS, eps, n, g.kvLora, g.kvaWidth, 1, kva, rowOff * g.kvaWidth, kvaNorm, null,
                kva, rowOff * g.kvaWidth);
    }

    /** The latent step up to the store: the K rotation of the latent row and the store. */
    private void latentStepKV(int[] qPos, int rowOff, int len, float[] kva) {
        if (rotKSpec != null) {
            CpuHadamard.applyInPlace(rotKSpec, kva, rowOff * g.kvaWidth, len, g.kvaWidth);
        }
        int[] cells = pagedStep != null ? pagedStep.cellIdx(kv.group()) : null;
        for (int t = 0; t < len; t++) {
            int row = cells != null ? cells[rowOff + t] : layout.rowOf(qPos[rowOff + t]);
            kv.writeK(row, kva, (rowOff + t) * g.kvaWidth);
        }
    }

    /**
     * The decompressed step up to the stores: decompression, per-head K/V, the K and V rotations and both stores.
     * The temperature and the Q rotation (query-only) do not run.
     */
    private void decompressedStepKV(int[] qPos, int rowOff, int len, float[] kva) {
        int rows = g.headRows(len);
        float[] k = scratch.temp(TEMP_K, rows * g.headK);
        float[] v = scratch.temp(TEMP_V, rows * g.headV);
        if (wkvB != null) {
            int kvRow = g.nope + g.headV;
            float[] kvb = scratch.temp(TEMP_KV_B, rows * kvRow);
            mv.matmul(wkvB, kva, rowOff * g.kvaWidth, g.kvaWidth, kvb, 0, g.nHead * kvRow, len);
            kvbDeltas.apply(len, kva, rowOff * g.kvaWidth, g.kvaWidth, kvb, 0, g.nHead * kvRow);
            for (int s = 0; s < rows; s++) {
                System.arraycopy(kvb, s * kvRow, k, s * g.headK, g.nope);
                System.arraycopy(kvb, s * kvRow + g.nope, v, s * g.headV, g.headV);
            }
        } else {
            absorbedKNope(len, kva, rowOff * g.kvaWidth, k);
            mv.matmul(wvB, kva, rowOff * g.kvaWidth, g.kvaWidth, v, 0, g.nHead * g.headV, len);
        }
        for (int s = 0; s < rows; s++) {
            System.arraycopy(kva, (rowOff + s / g.nHead) * g.kvaWidth + g.kvLora, k, s * g.headK + g.nope, g.rope);
        }
        if (rotKSpec != null) {
            CpuHadamard.applyInPlace(rotKSpec, k, 0, rows, g.headK);
        }
        if (rotVSpec != null) {
            CpuHadamard.applyInPlace(rotVSpec, v, 0, rows, g.headV);
        }
        int[] cells = pagedStep != null ? pagedStep.cellIdx(kv.group()) : null;
        for (int t = 0; t < len; t++) {
            int row = cells != null ? cells[rowOff + t] : layout.rowOf(qPos[rowOff + t]);
            kv.writeK(row, k, t * g.qWidth);
            kv.writeV(row, v, t * g.nHead * g.headV);
        }
    }

    /**
     * One step over the rows of {@code s}: the contiguous step at {@code s.pos0(0)} when the memory runs no paged
     * step, the paged step through the memory's {@code CpuMemory.currentPagedStep} otherwise — one segment whole,
     * several segments one at a time, each storing at its rows' cells and attending through its sequence's page
     * table. Rows of different sequences never attend each other, so the per-segment attends compute the same
     * outputs as the same rows run one sequence at a time. The paged rows take their positions from the
     * {@link CpuScratch#POS} slot (a media chunk's rows can share one), and a step the context flags non-causal
     * ({@code CpuForwardContext.stepNonCausal}) attends without the causal mask.
     */
    @Override
    public void batch(com.eonmux.neuralfree.gguf.memory.StepLayout s, float[] in, float[] out) {
        if (memory.currentPagedStep() == null) {
            if (!s.contiguousSingle()) {
                throw new UnsupportedOperationException(batchRefusal());
            }
            if (s.nTokens() == 1) {
                decode(s.pos0(0), in, out);
            } else {
                prefill(s.pos0(0), s.nTokens(), in, out);
            }
            return;
        }
        if (s.nSeg() == 1) {
            run(s.pos0(0), s.nTokens(), in, out);
            return;
        }
        runSegmented(s, in, out);
    }

    private void run(int pos0, int n, float[] in, float[] out) {
        checkBound();
        Objects.requireNonNull(in, "in");
        Objects.requireNonNull(out, "out");
        if (n < 1 || n > scratch.ubatch()) {
            throw new IllegalArgumentException("MLA step of " + n + " tokens is outside [1, " + scratch.ubatch() + "]");
        }
        if (pos0 < 0) {
            throw new IllegalArgumentException("MLA step position " + pos0 + " is negative");
        }
        if (in.length < (long) n * nEmbd || out.length < (long) n * nOut) {
            throw new IllegalArgumentException("MLA step of " + n + " tokens needs " + n * (long) nEmbd + " inputs and "
                    + n * (long) nOut + " outputs, got " + in.length + " and " + out.length);
        }
        pagedStep = memory.currentPagedStep();
        int[] qPos = scratch.intTemp(TEMP_QPOS, n);
        if (pagedStep != null) {
            System.arraycopy(scratch.ints(CpuScratch.POS), 0, qPos, 0, n);
        } else {
            for (int t = 0; t < n; t++) {
                qPos[t] = pos0 + t;
            }
        }
        float[] q = scratch.temp(TEMP_Q, n * g.qWidth);
        float[] kva = scratch.temp(TEMP_KV_A, n * g.kvaWidth);
        float[] attn = scratch.temp(TEMP_ATTN, g.headRows(n) * g.headV);
        project(qPos, 0, n, in, q, kva);
        if (g.latent) {
            latentStep(qPos, 0, n, seq(), q, kva, attn);
        } else {
            decompressedStep(qPos, 0, n, seq(), q, kva, attn);
        }
        if (wGate != null) {
            int gw = (int) wGate.rows();
            float[] gate = scratch.temp(TEMP_GATE, n * gw);
            mv.matmul(wGate, in, 0, gate, 0, n);
            gateDeltas.applyRows(n, in, 0, gate, 0);
            AttnExtMath.applyOutGate(gateKind, n, g.nHead, g.headV, attn, 0, g.nHead * g.headV, gate, 0, gw,
                    gateBias, 0);
        }
        mv.matmul(wo, attn, 0, out, 0, n);
        outDeltas.applyRows(n, attn, 0, out, 0);
    }

    /**
     * The paged step over the segments of {@code s}: the token-wise phases (projections, RoPE, norms) run over the
     * whole step once, then every segment stores its latent (or decompressed) rows at the step's cells and attends
     * through its sequence's page table.
     */
    private void runSegmented(com.eonmux.neuralfree.gguf.memory.StepLayout s, float[] in, float[] out) {
        checkBound();
        int n = s.nTokens();
        if (n < 1 || n > scratch.ubatch()) {
            throw new IllegalArgumentException("MLA step of " + n + " tokens is outside [1, " + scratch.ubatch() + "]");
        }
        if (in.length < (long) n * nEmbd || out.length < (long) n * nOut) {
            throw new IllegalArgumentException("MLA step of " + n + " tokens needs " + n * (long) nEmbd + " inputs and "
                    + n * (long) nOut + " outputs, got " + in.length + " and " + out.length);
        }
        pagedStep = memory.currentPagedStep();
        int[] qPos = scratch.intTemp(TEMP_QPOS, n);
        System.arraycopy(scratch.ints(CpuScratch.POS), 0, qPos, 0, n);
        float[] q = scratch.temp(TEMP_Q, n * g.qWidth);
        float[] kva = scratch.temp(TEMP_KV_A, n * g.kvaWidth);
        float[] attn = scratch.temp(TEMP_ATTN, g.headRows(n) * g.headV);
        project(qPos, 0, n, in, q, kva);
        try (TapProduct batch = TapProduct.dense(n)) {
            for (int seg = 0; seg < s.nSeg(); seg++) {
                if (g.latent) {
                    latentStep(qPos, s.tokStart(seg), s.len(seg), s.seq(seg), q, kva, attn);
                } else {
                    decompressedStep(qPos, s.tokStart(seg), s.len(seg), s.seq(seg), q, kva, attn);
                }
            }
        }
        if (wGate != null) {
            int gw = (int) wGate.rows();
            float[] gate = scratch.temp(TEMP_GATE, n * gw);
            mv.matmul(wGate, in, 0, gate, 0, n);
            gateDeltas.applyRows(n, in, 0, gate, 0);
            AttnExtMath.applyOutGate(gateKind, n, g.nHead, g.headV, attn, 0, g.nHead * g.headV, gate, 0, gw,
                    gateBias, 0);
        }
        mv.matmul(wo, attn, 0, out, 0, n);
        outDeltas.applyRows(n, attn, 0, out, 0);
    }

    /** The paged step's one segment when {@link #run} drives a single-segment step, else its first segment. */
    private int seq() {
        return pagedStep != null ? pagedStep.segSeq(0) : 0;
    }

    /**
     * Q and the latent row: projections, the optional RMS norm of the shared rotary key, RoPE of the rotary dims,
     * RMS norm of the latent prefix (dots3note.cpp:330-344). Rows {@code [rowOff, rowOff + len)} carry positions
     * {@code qPos[rowOff ..]}.
     */
    private void project(int[] qPos, int rowOff, int n, float[] in, float[] q, float[] kva) {
        int inOff = rowOff * nEmbd;
        if (wqA != null) {
            float[] qa = scratch.temp(TEMP_Q_A, n * g.qLora);
            mv.matmul(wqA, in, inOff, qa, 0, n);
            qaDeltas.applyRows(n, in, inOff, qa, 0);
            NormMath.rowNorm(NormKind.RMS, eps, n, g.qLora, g.qLora, 1, qa, 0, qaNorm, null, qa, 0);
            mv.matmul(wqB, qa, 0, q, rowOff * g.qWidth, n);
            qbDeltas.applyRows(n, qa, 0, q, rowOff * g.qWidth);
        } else {
            mv.matmul(wq, in, inOff, q, rowOff * g.qWidth, n);
            qDeltas.applyRows(n, in, inOff, q, rowOff * g.qWidth);
        }
        mv.matmul(wkvA, in, inOff, kva, rowOff * g.kvaWidth, n);
        kvaDeltas.applyRows(n, in, inOff, kva, rowOff * g.kvaWidth);
        if (kPeNorm != null) {
            NormMath.rowNorm(NormKind.RMS, eps, n, g.rope, g.kvaWidth, 1, kva, rowOff * g.kvaWidth + g.kvLora,
                    kPeNorm, null, kva, rowOff * g.kvaWidth + g.kvLora);
        }
        if (rope != null) {
            int[] pos = scratch.intTemp(TEMP_POS, RopeMath.positionsPerToken(rope.type()) * n);
            RopePositions.fill(rope.type(), rope.textPositions(), qPos, rowOff, n, pos);
            CpuRope.apply(ropeTable, q, rowOff * g.qWidth, n, g.nHead, g.headK, g.qWidth, g.nope, pos, 0, n);
            CpuRope.apply(ropeTable, kva, rowOff * g.kvaWidth, n, 1, g.kvaWidth, g.kvaWidth, g.kvLora, pos, 0, n);
        }
        NormMath.rowNorm(NormKind.RMS, eps, n, g.kvLora, g.kvaWidth, 1, kva, rowOff * g.kvaWidth, kvaNorm, null,
                kva, rowOff * g.kvaWidth);
    }

    /** Absorbed Q, latent K row, attention over the latent cache, per-head expansion by {@code attn_v_b}. */
    private void latentStep(int[] qPos, int rowOff, int len, int seq, float[] q, float[] kva, float[] attn) {
        int rows = g.headRows(len);
        float[] qlat = scratch.temp(TEMP_Q_LATENT, rows * g.dQ);
        float[] lat = scratch.temp(TEMP_LATENT, rows * g.kvLora);
        CpuMlaHeadProducts.run(wkB, g.nHead, g.kvLora, len, q, rowOff * g.qWidth, g.qWidth, g.headK, qlat, 0,
                g.nHead * g.dQ, g.dQ);
        for (int s = 0; s < rows; s++) {
            int qBase = (rowOff + s / g.nHead) * g.qWidth + (s % g.nHead) * g.headK;
            System.arraycopy(q, qBase + g.nope, qlat, s * g.dQ + g.kvLora, g.rope);
        }
        if (temperature != null) {
            AttnExtMath.applyTemperature(temperature, len, g.nHead, g.dQ, qPos, rowOff, qlat, 0, g.nHead * g.dQ);
        }
        if (rotKSpec != null) {
            CpuHadamard.applyInPlace(rotKSpec, qlat, 0, rows, g.dQ);
            CpuHadamard.applyInPlace(rotKSpec, kva, rowOff * g.kvaWidth, len, g.kvaWidth);
        }
        if (owns) {
            int[] cells = pagedStep != null ? pagedStep.cellIdx(kv.group()) : null;
            for (int t = 0; t < len; t++) {
                int row = cells != null ? cells[rowOff + t] : layout.rowOf(qPos[rowOff + t]);
                kv.writeK(row, kva, (rowOff + t) * g.kvaWidth);
            }
        }
        attend(qPos, rowOff, len, seq, qlat, 0, lat, 0);
        if (rotKSpec != null) {
            CpuHadamard.applyInPlace(rotKSpec, lat, 0, rows, g.kvLora);
        }
        CpuMlaHeadProducts.run(wvB, g.nHead, g.headV, len, lat, 0, g.nHead * g.kvLora, g.kvLora, attn,
                rowOff * (g.nHead * g.headV), g.nHead * g.headV, g.headV);
    }
    /** Per-head K/V from {@code attn_kv_b} (or the absorbed pair), attention over the per-head cache. */
    private void decompressedStep(int[] qPos, int rowOff, int len, int seq, float[] q, float[] kva, float[] attn) {
        int rows = g.headRows(len);
        float[] k = scratch.temp(TEMP_K, rows * g.headK);
        float[] v = scratch.temp(TEMP_V, rows * g.headV);
        if (wkvB != null) {
            int kvRow = g.nope + g.headV;
            float[] kvb = scratch.temp(TEMP_KV_B, rows * kvRow);
            mv.matmul(wkvB, kva, rowOff * g.kvaWidth, g.kvaWidth, kvb, 0, g.nHead * kvRow, len);
            kvbDeltas.apply(len, kva, rowOff * g.kvaWidth, g.kvaWidth, kvb, 0, g.nHead * kvRow);
            for (int s = 0; s < rows; s++) {
                System.arraycopy(kvb, s * kvRow, k, s * g.headK, g.nope);
                System.arraycopy(kvb, s * kvRow + g.nope, v, s * g.headV, g.headV);
            }
        } else {
            absorbedKNope(len, kva, rowOff * g.kvaWidth, k);
            mv.matmul(wvB, kva, rowOff * g.kvaWidth, g.kvaWidth, v, 0, g.nHead * g.headV, len);
        }
        for (int s = 0; s < rows; s++) {
            System.arraycopy(kva, (rowOff + s / g.nHead) * g.kvaWidth + g.kvLora, k, s * g.headK + g.nope, g.rope);
        }
        if (temperature != null) {
            AttnExtMath.applyTemperature(temperature, len, g.nHead, g.headK, qPos, rowOff, q, rowOff * g.qWidth,
                    g.qWidth);
        }
        if (rotKSpec != null) {
            CpuHadamard.applyInPlace(rotKSpec, q, rowOff * g.qWidth, len * g.nHead, g.headK);
            CpuHadamard.applyInPlace(rotKSpec, k, 0, rows, g.headK);
        }
        if (rotVSpec != null) {
            CpuHadamard.applyInPlace(rotVSpec, v, 0, rows, g.headV);
        }
        if (owns) {
            int[] cells = pagedStep != null ? pagedStep.cellIdx(kv.group()) : null;
            for (int t = 0; t < len; t++) {
                int row = cells != null ? cells[rowOff + t] : layout.rowOf(qPos[rowOff + t]);
                kv.writeK(row, k, t * g.qWidth);
                kv.writeV(row, v, t * g.nHead * g.headV);
            }
        }
        attend(qPos, rowOff, len, seq, q, rowOff * (g.nHead * g.dQ), attn, rowOff * (g.nHead * g.headV));
        if (rotVSpec != null) {
            CpuHadamard.applyInPlace(rotVSpec, attn, rowOff * (g.nHead * g.headV), len * g.nHead, g.headV);
        }
    }

    /**
     * {@code k_nope[t][h][d] = sum_l attn_k_b[h][l][d] * kv_cmpr[t][l]}: the transpose of the absorbed product, from
     * head {@code h}'s decoded {@code [kv_lora][n_nope]} rows, parallel over tokens. Rows {@code [rowOff, ..)} read
     * {@code kva} from {@code kvaOff}.
     */
    private void absorbedKNope(int n, float[] kva, int kvaOff, float[] k) {
        float[] wh = scratch.temp(TEMP_WK_B, g.kvLora * g.nope);
        for (int h = 0; h < g.nHead; h++) {
            wkB.decodeRows((long) h * g.kvLora, g.kvLora, wh, 0);
            final int head = h;
            CpuParallel.parallelFor(n, 1, (from, to) -> {
                for (int t = from; t < to; t++) {
                    int kBase = (t * g.nHead + head) * g.headK;
                    int xBase = kvaOff + t * g.kvaWidth;
                    for (int d = 0; d < g.nope; d++) {
                        float acc = 0.0f;
                        for (int l = 0; l < g.kvLora; l++) {
                            acc += wh[l * g.nope + d] * kva[xBase + l];
                        }
                        k[kBase + d] = acc;
                    }
                }
            });
        }
    }

    /**
     * Attention of one segment's {@code len} rows at {@code [rowOff, rowOff + len)} of the step: query rows at
     * {@code qs[qsOff]} (the segment's own array start), positions {@code qPos[rowOff ..]}, outputs into
     * {@code os[osOff]}. Paged through the sequence's page table, else over the position window of the contiguous
     * layout. A tree verify's step carries the context's tree mask, which replaces the position rules over the
     * step's fresh rows (see {@link CpuAttentionCore.TreeMask}); the segment's rows are the call's fresh rows, so
     * the mask relativizes to them like {@code CpuAttentionMixer#segmentTreeMask} does.
     */
    private void attend(int[] qPos, int rowOff, int len, int seq, float[] qs, int qsOff, float[] os, int osOff) {
        int qStride = g.nHead * g.dQ;
        int outStride = g.nHead * g.dV;
        CpuAttentionCore.TreeMask tree = treeMask(rowOff, len);
        if (pagedStep != null) {
            int group = kv.group();
            int keys = pagedStep.seqSlot(group, seq);
            int[] table = new int[pagedStep.pageTableLength(group, seq)];
            pagedStep.pageTableRow(group, seq, table);
            CpuAttentionCore.Params stepParams = ctx.stepNonCausal() ? nonCausalParams : params;
            CpuAttentionCore.attendRows(stepParams, len, qs, qsOff, qStride, qPos, rowOff, kv, rowPos,
                    pageShift(pagedStep), table, 0, keys, os, osOff, outStride, tree);
            return;
        }
        int pos0 = qPos[rowOff];
        int start = MlaBlocks.kvStart(layout, pos0);
        int end = MlaBlocks.kvEnd(layout, qPos[rowOff + len - 1]);
        CpuAttentionCore.attendRows(params, len, qs, qsOff, qStride, qPos, rowOff, kv, rowPos, start, end,
                os, osOff, outStride, tree);
    }

    /**
     * The running step's tree mask relativized to the segment's rows at {@code [rowOff, rowOff + len)} of the
     * ubatch, or null: the backend set one per ubatch row and the segment's rows are the call's fresh rows (a
     * single-sequence tree verify runs one segment at {@code rowOff} 0, a batched masked step one segment per
     * sequence), as {@code CpuAttentionMixer#segmentTreeMask} builds for the standard attention.
     */
    private CpuAttentionCore.TreeMask treeMask(int rowOff, int len) {
        long[] bits = ctx.stepTreeMask();
        if (bits == null) {
            return null;
        }
        int fresh = ctx.stepTreeFresh();
        if (rowOff + len > fresh) {
            throw new IllegalStateException("layer " + il + " runs rows [" + rowOff + ", " + (rowOff + len)
                    + ") but the step's tree mask covers " + fresh + " fresh rows");
        }
        long[] rel = new long[rowOff + len];
        for (int t = 0; t < len; t++) {
            rel[rowOff + t] = bits[rowOff + t] >>> rowOff;
        }
        return new CpuAttentionCore.TreeMask(rel, len);
    }

    /** {@code log2(pageSize)}: slot {@code s} sits in page {@code s >> shift} (the kernel's page-table shift). */
    private static int pageShift(PagedStep step) {
        int size = step.pageSize();
        if (Integer.bitCount(size) != 1) {
            throw new IllegalArgumentException("page size " + size + " is not a power of two");
        }
        return Integer.numberOfTrailingZeros(size);
    }

    // ================================================================== context shift

    /**
     * Applies the per-row RoPE deltas of the memory's last {@code shiftPositions(nKeep, nDiscard)} to this layer's
     * K rows, as {@code LLM_KV_RopeShift_*} does: every row is read (delta 0 included, as llama.cpp shifts every cell
     * with its per-cell shift, llama-kv-cache.cpp:1545-1556), the Hadamard blocks covering the rotary
     * dims are undone when the rows are rotated, the dims {@code [nOffs, nOffs + n_rot)} of every head are rotated by
     * the delta, the blocks re-applied and the shifted dims written back: the rotary dims of every head, widened for a
     * block type to the kernel's work window, so the latent prefix and the other blocks keep their stored bits (the
     * {@code n_rot} view llama.cpp requantizes, llama-kv-cache.cpp:2058-2066).
     *
     * @throws UnsupportedOperationException for multi-position RoPE, a cache geometry the shift kernels reject, or a
     *                                       memory that does not record shift deltas
     */
    @Override
    public void shiftContext(int nKeep, int nDiscard) {
        String refusal = shiftRefusal();
        if (refusal != null) {
            throw new UnsupportedOperationException("layer " + il + ": context shift unsupported: " + refusal);
        }
        if (!owns || rope == null) {
            return;
        }
        rotateByDeltas();
    }

    /** {@link CpuMixer#applyShiftDeltas}: the paged shift's row moves already happened, rotate by the deltas. */
    @Override
    public void applyShiftDeltas() {
        String refusal = shiftRefusal();
        if (refusal != null) {
            throw new UnsupportedOperationException("layer " + il + ": context shift unsupported: " + refusal);
        }
        if (!owns || rope == null) {
            return;
        }
        rotateByDeltas();
    }

    /** Rotates every cached K row by its row's delta of the memory's last shift. */
    private void rotateByDeltas() {
        int[] delta = ((HeapModelMemory) memory).shiftDelta()[kv.group()];
        RopeFreqTable table = RopeFreqTable.forShift(rope, freqFactors);
        int rotK = g.shiftRotK();
        float[] buf = scratch.temp(TEMP_SHIFT, SHIFT_ROWS * g.shiftRowStride);
        int[] rowsOf = scratch.intTemp(TEMP_DELTA, 2 * SHIFT_ROWS);
        int count = 0;
        for (int r = 0; r < delta.length; r++) {
            rowsOf[count] = r;
            rowsOf[SHIFT_ROWS + count] = delta[r];
            count++;
            if (count == SHIFT_ROWS) {
                shiftRows(table, rotK, buf, rowsOf, count);
                count = 0;
            }
        }
        if (count > 0) {
            shiftRows(table, rotK, buf, rowsOf, count);
        }
    }

    /** Rows {@code rowsOf[0 .. count)} with deltas {@code rowsOf[SHIFT_ROWS ..]}: read, rotate, write back. */
    private void shiftRows(RopeFreqTable table, int rotK, float[] buf, int[] rowsOf, int count) {
        for (int i = 0; i < count; i++) {
            for (int h = 0; h < g.shiftHeads; h++) {
                kv.readK(rowsOf[i], h, buf, i * g.shiftRowStride + h * g.shiftHeadStride);
            }
        }
        if (rotK > 0) {
            rotateWindows(buf, count, rotK);
        }
        CpuRope.apply(table, buf, 0, count, g.shiftHeads, g.shiftHeadStride, g.shiftRowStride, g.shiftOffs, rowsOf,
                SHIFT_ROWS, count);
        if (rotK > 0) {
            rotateWindows(buf, count, rotK);
        }
        boolean blocks = layout.dtypeK().quantized();
        int w0 = blocks ? RopeKernels.shiftWindowStart(g.shiftOffs, rotK) : g.shiftOffs;
        int w1 = blocks ? RopeKernels.shiftWindowEnd(rope.nDims(), g.shiftOffs, rotK) : g.shiftOffs + rope.nDims();
        for (int i = 0; i < count; i++) {
            for (int h = 0; h < g.shiftHeads; h++) {
                int at = h * g.shiftHeadStride + w0;
                kv.writeKSlice(rowsOf[i], at, w1 - w0, buf, i * g.shiftRowStride + at);
            }
        }
    }

    /** The self-inverse Hadamard blocks covering the rotary dims of every head ({@code nf_rope_shift_w0..w1}). */
    private void rotateWindows(float[] buf, int count, int rotK) {
        HadamardSpec spec = HadamardSpec.rotation(rotK);
        int w0 = RopeKernels.shiftWindowStart(g.shiftOffs, rotK);
        int w1 = RopeKernels.shiftWindowEnd(rope.nDims(), g.shiftOffs, rotK);
        for (int i = 0; i < count; i++) {
            for (int h = 0; h < g.shiftHeads; h++) {
                int base = i * g.shiftRowStride + h * g.shiftHeadStride;
                for (int c = w0; c < w1; c += rotK) {
                    CpuHadamard.applyInPlace(spec, buf, base + c, 1, rotK);
                }
            }
        }
    }

    /** Null when {@link #shiftContext} can run: no own RoPE cache, or a shiftable geometry on heap memory. */
    @Override
    public String shiftRefusal() {
        checkBound();
        if (!owns || rope == null) {
            return null;
        }
        if (shiftBlocker != null) {
            return shiftBlocker;
        }
        if (!(memory instanceof HeapModelMemory)) {
            return "context shift needs HeapModelMemory, got " + memory.getClass().getSimpleName();
        }
        return null;
    }

    // ================================================================== lifecycle

    private void checkBound() {
        if (!bound || closed) {
            throw new IllegalStateException("MLA mixer is not bound or already closed");
        }
    }

    @Override
    public void close() {
        closed = true;
    }

    @Override
    public String toString() {
        return "CpuMlaMixer{layer=" + il + ", " + g + '}';
    }
}

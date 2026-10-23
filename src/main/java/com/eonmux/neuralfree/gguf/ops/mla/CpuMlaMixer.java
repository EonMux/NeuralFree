package com.eonmux.neuralfree.gguf.ops.mla;

import com.eonmux.neuralfree.gguf.arch.AttnGateKind;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.cpu.block.CpuMixer;
import com.eonmux.neuralfree.gguf.cpu.block.CpuProjectionDeltas;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuForwardContext;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuScratch;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.memory.KvLayout;
import com.eonmux.neuralfree.gguf.memory.cpu.CpuMemory;
import com.eonmux.neuralfree.gguf.memory.cpu.HeapModelMemory;
import com.eonmux.neuralfree.gguf.memory.cpu.KvAccessor;
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
import com.eonmux.neuralfree.gguf.spec.AttentionSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.MissingTensorException;
import com.eonmux.neuralfree.gguf.spec.RopeSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuParallel;

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
    private RopeSpec rope;
    private float[] freqFactors;
    private RopeFreqTable ropeTable;
    private String shiftBlocker;
    private HadamardSpec rotKSpec;
    private HadamardSpec rotVSpec;

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
            deltas.serve(layer, TensorNames.weight(TensorNames.ATTN_KV_B), GpuMlaMixer.LATENT_KV_B_REFUSAL);
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
        if (t.input() != null || t.hasScale()) {
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
        int[] qPos = scratch.intTemp(TEMP_QPOS, n);
        for (int t = 0; t < n; t++) {
            qPos[t] = pos0 + t;
        }
        float[] q = scratch.temp(TEMP_Q, n * g.qWidth);
        float[] kva = scratch.temp(TEMP_KV_A, n * g.kvaWidth);
        float[] attn = scratch.temp(TEMP_ATTN, g.headRows(n) * g.headV);
        project(pos0, n, in, q, kva);
        if (g.latent) {
            latentStep(pos0, n, q, kva, qPos, attn);
        } else {
            decompressedStep(pos0, n, q, kva, qPos, attn);
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
     * Q and the latent row: projections, the optional RMS norm of the shared rotary key, RoPE of the rotary dims,
     * RMS norm of the latent prefix (dots3note.cpp:330-344).
     */
    private void project(int pos0, int n, float[] in, float[] q, float[] kva) {
        if (wqA != null) {
            float[] qa = scratch.temp(TEMP_Q_A, n * g.qLora);
            mv.matmul(wqA, in, 0, qa, 0, n);
            qaDeltas.applyRows(n, in, 0, qa, 0);
            NormMath.rowNorm(NormKind.RMS, eps, n, g.qLora, g.qLora, 1, qa, 0, qaNorm, null, qa, 0);
            mv.matmul(wqB, qa, 0, q, 0, n);
            qbDeltas.applyRows(n, qa, 0, q, 0);
        } else {
            mv.matmul(wq, in, 0, q, 0, n);
            qDeltas.applyRows(n, in, 0, q, 0);
        }
        mv.matmul(wkvA, in, 0, kva, 0, n);
        kvaDeltas.applyRows(n, in, 0, kva, 0);
        if (kPeNorm != null) {
            NormMath.rowNorm(NormKind.RMS, eps, n, g.rope, g.kvaWidth, 1, kva, g.kvLora, kPeNorm, null, kva,
                    g.kvLora);
        }
        if (rope != null) {
            int[] pos = scratch.intTemp(TEMP_POS, RopeMath.positionsPerToken(rope.type()) * n);
            RopePositions.fill(rope.type(), rope.textPositions(), pos0, n, pos);
            CpuRope.apply(ropeTable, q, 0, n, g.nHead, g.headK, g.qWidth, g.nope, pos, 0, n);
            CpuRope.apply(ropeTable, kva, 0, n, 1, g.kvaWidth, g.kvaWidth, g.kvLora, pos, 0, n);
        }
        NormMath.rowNorm(NormKind.RMS, eps, n, g.kvLora, g.kvaWidth, 1, kva, 0, kvaNorm, null, kva, 0);
    }

    /** Absorbed Q, latent K row, attention over the latent cache, per-head expansion by {@code attn_v_b}. */
    private void latentStep(int pos0, int n, float[] q, float[] kva, int[] qPos, float[] attn) {
        int rows = g.headRows(n);
        float[] qlat = scratch.temp(TEMP_Q_LATENT, rows * g.dQ);
        float[] lat = scratch.temp(TEMP_LATENT, rows * g.kvLora);
        CpuMlaHeadProducts.run(wkB, g.nHead, g.kvLora, n, q, 0, g.qWidth, g.headK, qlat, 0, g.nHead * g.dQ, g.dQ);
        for (int s = 0; s < rows; s++) {
            int qBase = (s / g.nHead) * g.qWidth + (s % g.nHead) * g.headK;
            System.arraycopy(q, qBase + g.nope, qlat, s * g.dQ + g.kvLora, g.rope);
        }
        if (temperature != null) {
            AttnExtMath.applyTemperature(temperature, n, g.nHead, g.dQ, qPos, 0, qlat, 0, g.nHead * g.dQ);
        }
        if (rotKSpec != null) {
            CpuHadamard.applyInPlace(rotKSpec, qlat, 0, rows, g.dQ);
            CpuHadamard.applyInPlace(rotKSpec, kva, 0, n, g.kvaWidth);
        }
        if (owns) {
            for (int t = 0; t < n; t++) {
                kv.writeK(layout.rowOf(pos0 + t), kva, t * g.kvaWidth);
            }
        }
        attend(pos0, n, qlat, qPos, lat);
        if (rotKSpec != null) {
            CpuHadamard.applyInPlace(rotKSpec, lat, 0, rows, g.kvLora);
        }
        CpuMlaHeadProducts.run(wvB, g.nHead, g.headV, n, lat, 0, g.nHead * g.kvLora, g.kvLora, attn, 0,
                g.nHead * g.headV, g.headV);
    }

    /** Per-head K/V from {@code attn_kv_b} (or the absorbed pair), attention over the per-head cache. */
    private void decompressedStep(int pos0, int n, float[] q, float[] kva, int[] qPos, float[] attn) {
        int rows = g.headRows(n);
        float[] k = scratch.temp(TEMP_K, rows * g.headK);
        float[] v = scratch.temp(TEMP_V, rows * g.headV);
        if (wkvB != null) {
            int kvRow = g.nope + g.headV;
            float[] kvb = scratch.temp(TEMP_KV_B, rows * kvRow);
            mv.matmul(wkvB, kva, 0, g.kvaWidth, kvb, 0, g.nHead * kvRow, n);
            kvbDeltas.apply(n, kva, 0, g.kvaWidth, kvb, 0, g.nHead * kvRow);
            for (int s = 0; s < rows; s++) {
                System.arraycopy(kvb, s * kvRow, k, s * g.headK, g.nope);
                System.arraycopy(kvb, s * kvRow + g.nope, v, s * g.headV, g.headV);
            }
        } else {
            absorbedKNope(n, kva, k);
            mv.matmul(wvB, kva, 0, g.kvaWidth, v, 0, g.nHead * g.headV, n);
        }
        for (int s = 0; s < rows; s++) {
            System.arraycopy(kva, (s / g.nHead) * g.kvaWidth + g.kvLora, k, s * g.headK + g.nope, g.rope);
        }
        if (temperature != null) {
            AttnExtMath.applyTemperature(temperature, n, g.nHead, g.headK, qPos, 0, q, 0, g.qWidth);
        }
        if (rotKSpec != null) {
            CpuHadamard.applyInPlace(rotKSpec, q, 0, rows, g.headK);
            CpuHadamard.applyInPlace(rotKSpec, k, 0, rows, g.headK);
        }
        if (rotVSpec != null) {
            CpuHadamard.applyInPlace(rotVSpec, v, 0, rows, g.headV);
        }
        if (owns) {
            for (int t = 0; t < n; t++) {
                int row = layout.rowOf(pos0 + t);
                kv.writeK(row, k, t * g.qWidth);
                kv.writeV(row, v, t * g.nHead * g.headV);
            }
        }
        attend(pos0, n, q, qPos, attn);
        if (rotVSpec != null) {
            CpuHadamard.applyInPlace(rotVSpec, attn, 0, rows, g.headV);
        }
    }

    /**
     * {@code k_nope[t][h][d] = sum_l attn_k_b[h][l][d] * kv_cmpr[t][l]}: the transpose of the absorbed product, from
     * head {@code h}'s decoded {@code [kv_lora][n_nope]} rows, parallel over tokens.
     */
    private void absorbedKNope(int n, float[] kva, float[] k) {
        float[] wh = scratch.temp(TEMP_WK_B, g.kvLora * g.nope);
        for (int h = 0; h < g.nHead; h++) {
            wkB.decodeRows((long) h * g.kvLora, g.kvLora, wh, 0);
            final int head = h;
            CpuParallel.parallelFor(n, 1, (from, to) -> {
                for (int t = from; t < to; t++) {
                    int kBase = (t * g.nHead + head) * g.headK;
                    int xBase = t * g.kvaWidth;
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

    private void attend(int pos0, int n, float[] qs, int[] qPos, float[] os) {
        int start = MlaBlocks.kvStart(layout, pos0);
        int end = MlaBlocks.kvEnd(layout, pos0 + n - 1);
        CpuAttentionCore.attendRows(params, n, qs, 0, g.nHead * g.dQ, qPos, 0, kv, rowPos, start, end, os, 0,
                g.nHead * g.dV);
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

package com.eonmux.neuralfree.gguf.ops.rwkv;

import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.arch.GluOp;
import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.cpu.block.CpuMixer;
import com.eonmux.neuralfree.gguf.cpu.block.CpuProjectionDeltas;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuForwardContext;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuOps;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan;
import com.eonmux.neuralfree.gguf.ops.norm.NormMath;
import com.eonmux.neuralfree.gguf.spec.ActParams;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.MissingTensorException;
import com.eonmux.neuralfree.gguf.spec.RecurrentSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;

import java.util.Objects;

/**
 * CPU twin of {@link GpuRwkv7Mixer}: the RWKV7 time mix (models/rwkv7-base.cpp:32-137) of {@code rwkv7} and
 * {@code arwkv7} with the same step sequence, using {@link CpuRwkvMix}, {@link CpuWkv}, {@link NormMath},
 * {@link CpuOps} and {@link CpuMatVec}. Layer 0 keeps its value rows in the scratch temporary
 * {@link RwkvBlocks#V_FIRST} for the value residual of the later layers. Not thread-safe.
 */
public final class CpuRwkv7Mixer implements CpuMixer {

    private static final int PLANE_R = 0;
    private static final int PLANE_W = 1;
    private static final int PLANE_K = 2;
    private static final int PLANE_V = 3;
    private static final int PLANE_A = 4;
    private static final int PLANE_G = 5;

    private RwkvBlocks.CpuKit kit;
    private CpuMatVec mv;
    private boolean closed;

    private int nEmbd;
    private int headSize;
    private int nHead;
    private int nMix;
    private boolean firstLayer;
    private boolean gating;
    private int decayDim;
    private int gateDim;

    private CpuTensorRef wr;
    private CpuTensorRef wk;
    private CpuTensorRef wv;
    private CpuTensorRef wo;
    /**
     * LoRA deltas of receptance, key, value and output ({@code build_lora_mm}, models/rwkv7-base.cpp); the low-rank
     * w/a/v/g products are plain {@code ggml_mul_mat} in llama.cpp and take none.
     */
    private CpuProjectionDeltas.Site rDeltas;
    private CpuProjectionDeltas.Site kDeltas;
    private CpuProjectionDeltas.Site vDeltas;
    private CpuProjectionDeltas.Site oDeltas;
    private CpuTensorRef w1;
    private CpuTensorRef w2;
    private CpuTensorRef a1;
    private CpuTensorRef a2;
    private CpuTensorRef v1;
    private CpuTensorRef v2;
    private CpuTensorRef g1;
    private CpuTensorRef g2;

    private float[] lerpFused;
    private float[] w0;
    private float[] a0;
    private float[] v0;
    private float[] kK;
    private float[] kA;
    private float[] rK;
    private float[] ln;
    private float[] lnB;

    private float[] shift;
    private float[] state;

    private float[] planes;
    private float[] r;
    private float[] w;
    private float[] k;
    private float[] v;
    private float[] a;
    private float[] g;
    private float[] lora;
    private float[] tmp;
    private float[] tmp2;
    private float[] aa;
    private float[] bb;
    private float[] y;
    private float[] vFirst;

    /**
     * Resolves the layer's tensors and recurrent arrays and reserves the working temporaries (and
     * {@link RwkvBlocks#V_FIRST}).
     *
     * @throws IllegalStateException    when already bound or closed, the context has no recurrent memory, or a
     *                                  tensor's shape contradicts the spec
     * @throws IllegalArgumentException when the layer is not an RWKV7 layer
     * @throws MissingTensorException   when a required tensor is absent
     */
    @Override
    public void bind(CpuForwardContext ctx, LayerSpec layer) {
        if (closed || kit != null) {
            throw new IllegalStateException("RWKV7 time mix is " + (closed ? "closed" : "already bound"));
        }
        Objects.requireNonNull(ctx, "ctx");
        Objects.requireNonNull(layer, "layer");
        RecurrentSpec spec = RwkvBlocks.recurrent(layer, MixerKind.RWKV7);
        RwkvBlocks.CpuKit kt = new RwkvBlocks.CpuKit(ctx, layer, "RWKV7 time mix");
        nEmbd = spec.nEmbd();
        headSize = spec.wkvHeadSize();
        nHead = nEmbd / headSize;
        firstLayer = layer.il() == 0;
        int loraDim = bindTensors(kt, nEmbd);
        bindWorking(kt, nEmbd, loraDim);
        mv = kt.mv;
        kit = kt;
    }

    /** Resolves the tensors; returns the widest LoRA hidden width. */
    private int bindTensors(RwkvBlocks.CpuKit kt, int c) {
        wr = kt.matrix(TensorNames.TIME_MIX_RECEPTANCE, c, c);
        wk = kt.matrix(TensorNames.TIME_MIX_KEY, c, c);
        wv = kt.matrix(TensorNames.TIME_MIX_VALUE, c, c);
        wo = kt.matrix(TensorNames.TIME_MIX_OUTPUT, c, c);
        CpuProjectionDeltas deltas = kt.ctx.projectionDeltas();
        rDeltas = deltas.serve(kt.layer, TensorNames.weight(TensorNames.TIME_MIX_RECEPTANCE), wr, kt.mv);
        kDeltas = deltas.serve(kt.layer, TensorNames.weight(TensorNames.TIME_MIX_KEY), wk, kt.mv);
        vDeltas = deltas.serve(kt.layer, TensorNames.weight(TensorNames.TIME_MIX_VALUE), wv, kt.mv);
        oDeltas = deltas.serve(kt.layer, TensorNames.weight(TensorNames.TIME_MIX_OUTPUT), wo, kt.mv);
        w1 = kt.matrix(TensorNames.TIME_MIX_W1, c, 0);
        decayDim = Math.toIntExact(w1.rows());
        w2 = kt.matrix(TensorNames.TIME_MIX_W2, decayDim, c);
        a1 = kt.matrix(TensorNames.TIME_MIX_A1, c, 0);
        int iclrDim = Math.toIntExact(a1.rows());
        a2 = kt.matrix(TensorNames.TIME_MIX_A2, iclrDim, c);
        int valueDim = 0;
        if (!firstLayer) {
            v1 = kt.matrix(TensorNames.TIME_MIX_V1, c, 0);
            valueDim = Math.toIntExact(v1.rows());
            v2 = kt.matrix(TensorNames.TIME_MIX_V2, valueDim, c);
            v0 = kt.vector(TensorNames.TIME_MIX_V0, c);
        }
        g1 = kt.matrixOrNull(TensorNames.TIME_MIX_G1, c, 0);
        gateDim = g1 != null ? Math.toIntExact(g1.rows()) : 0;
        g2 = g1 != null ? kt.matrixOrNull(TensorNames.TIME_MIX_G2, gateDim, c) : null;
        gating = g1 != null && g2 != null;
        nMix = gating ? 6 : 5;
        lerpFused = kt.vector(TensorNames.TIME_MIX_LERP_FUSED, nMix * c);
        w0 = kt.vector(TensorNames.TIME_MIX_W0, c);
        a0 = kt.vector(TensorNames.TIME_MIX_A0, c);
        kK = kt.vector(TensorNames.TIME_MIX_K_K, c);
        kA = kt.vector(TensorNames.TIME_MIX_K_A, c);
        rK = kt.vector(TensorNames.TIME_MIX_R_K, c);
        float[] lnW = kt.vectorOrNull(TensorNames.TIME_MIX_LN, c);
        float[] lnBias = kt.biasOrNull(TensorNames.TIME_MIX_LN, c);
        ln = lnW != null && lnBias != null ? lnW : null;
        lnB = ln != null ? lnBias : null;
        return Math.max(Math.max(decayDim, iclrDim), Math.max(valueDim, gateDim));
    }

    private void bindWorking(RwkvBlocks.CpuKit kt, int c, int loraDim) {
        shift = kt.state(LayerMemoryPlan.STATE_SHIFT);
        state = kt.state(LayerMemoryPlan.STATE_SCAN);
        CpuWkv.checkRows("RWKV7 shift state", shift, 0, 1, c);
        CpuWkv.checkRows("RWKV7 WKV state", state, 0, nHead * headSize, headSize);
        planes = kt.temp("rwkv7.planes", nMix * c);
        r = kt.temp("rwkv7.r", c);
        w = kt.temp("rwkv7.w", c);
        k = kt.temp("rwkv7.k", c);
        v = kt.temp("rwkv7.v", c);
        a = kt.temp("rwkv7.a", c);
        g = gating ? kt.temp("rwkv7.g", c) : null;
        lora = kt.temp("rwkv7.lora", loraDim);
        tmp = kt.temp("rwkv7.tmp", c);
        tmp2 = kt.temp("rwkv7.tmp2", c);
        aa = kt.temp("rwkv7.aa", c);
        bb = kt.temp("rwkv7.bb", c);
        y = kt.temp("rwkv7.y", c);
        vFirst = kt.temp(RwkvBlocks.V_FIRST, c);
    }

    @Override
    public void decode(int pos, float[] in, float[] out) {
        prefill(pos, 1, in, out);
    }

    /**
     * Runs the time mix over {@code n} rows of {@code in} into {@code out} and advances the layer's shift and WKV
     * state; layer 0 also refreshes {@link RwkvBlocks#V_FIRST}.
     *
     * @throws IllegalStateException    when not bound or closed
     * @throws IllegalArgumentException for a token count outside {@code [1, ubatch]} or arrays too small for the rows
     */
    @Override
    public void prefill(int pos0, int n, float[] in, float[] out) {
        if (closed || kit == null) {
            throw new IllegalStateException("RWKV7 time mix is " + (closed ? "closed" : "not bound"));
        }
        int c = nEmbd;
        kit.checkStep(n, in, out, c);
        int plane = n * c;
        CpuRwkvMix.tokenShiftLerp(n, c, nMix, RwkvKernels.SHIFT_MODE_LERP, in, 0, shift, 0, null, lerpFused, planes,
                0);
        mv.matmul(wr, planes, PLANE_R * plane, r, 0, n);
        rDeltas.applyRows(n, planes, PLANE_R * plane, r, 0);
        mv.matmul(w1, planes, PLANE_W * plane, lora, 0, n);
        CpuOps.unary(ActOp.TANH, ActParams.NONE, lora, 0, lora, 0, n * decayDim);
        mv.matmul(w2, lora, 0, tmp, 0, n);
        CpuRwkvMix.decay7(plane, c, tmp, 0, w0, w, 0);
        mv.matmul(wk, planes, PLANE_K * plane, k, 0, n);
        kDeltas.applyRows(n, planes, PLANE_K * plane, k, 0);
        mv.matmul(wv, planes, PLANE_V * plane, v, 0, n);
        vDeltas.applyRows(n, planes, PLANE_V * plane, v, 0);
        valueResidual(n, plane);
        if (gating) {
            mv.matmul(g1, planes, PLANE_G * plane, lora, 0, n);
            CpuOps.unary(ActOp.SIGMOID, ActParams.NONE, lora, 0, lora, 0, n * gateDim);
            mv.matmul(g2, lora, 0, g, 0, n);
        }
        kit.project(a1, null, planes, PLANE_A * plane, lora, n);
        kit.project(a2, a0, lora, 0, a, n);
        CpuOps.unary(ActOp.SIGMOID, ActParams.NONE, a, 0, a, 0, plane);
        CpuRwkvMix.kkPrep(n, nHead, headSize, k, 0, kK, kA, a, 0, tmp, 0, tmp2, 0, aa, 0, bb, 0);
        CpuWkv.wkv7(n, c, nHead, r, 0, w, 0, tmp2, 0, v, 0, aa, 0, bb, 0, state, 0, y, 0);
        if (ln != null) {
            NormMath.rowNorm(NormKind.LAYER, RwkvBlocks.GROUP_NORM_EPS, n * nHead, headSize, headSize, nHead, y, 0,
                    ln, lnB, y, 0);
        }
        CpuRwkvMix.bonus(n, nHead, headSize, tmp2, 0, r, 0, rK, v, 0, y, 0);
        if (gating) {
            CpuOps.glu(GluOp.MUL, ActParams.NONE, y, 0, g, 0, y, 0, plane);
        }
        mv.matmul(wo, y, 0, out, 0, n);
        oDeltas.applyRows(n, y, 0, out, 0);
    }

    /** Layer 0 stores {@code v}; later layers compute {@code v += (v_first - v) * sigmoid(V2 V1 x_v + v0)}. */
    private void valueResidual(int n, int plane) {
        if (firstLayer) {
            System.arraycopy(v, 0, vFirst, 0, plane);
            return;
        }
        mv.matmul(v1, planes, PLANE_V * plane, lora, 0, n);
        kit.project(v2, v0, lora, 0, tmp, n);
        CpuOps.unary(ActOp.SIGMOID, ActParams.NONE, tmp, 0, tmp, 0, plane);
        System.arraycopy(vFirst, 0, tmp2, 0, plane);
        CpuOps.axpy(-1.0f, v, 0, tmp2, 0, plane);
        CpuOps.glu(GluOp.MUL, ActParams.NONE, tmp2, 0, tmp, 0, tmp2, 0, plane);
        CpuOps.add(v, 0, tmp2, 0, v, 0, plane);
    }

    @Override
    public void close() {
        closed = true;
    }

    @Override
    public String toString() {
        return "CpuRwkv7Mixer{" + (kit != null ? kit.where + (gating ? ", gated" : "") : "unbound")
                + (closed ? ", closed" : "") + '}';
    }
}

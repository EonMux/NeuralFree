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
 * CPU twin of {@link GpuRwkv6Mixer}: the RWKV6 / QRWKV time mix (models/rwkv6-base.cpp:30-165) with the same step
 * sequence, using {@link CpuRwkvMix}, {@link CpuWkv}, {@link NormMath}, {@link CpuOps} and {@link CpuMatVec}. W2's
 * plane {@code m} is the row block {@code [m*C, (m+1)*C)} applied per token with {@link CpuMatVec#matvecRows}.
 * Working rows are scratch temporaries; the shift and WKV state are the layer's live recurrent arrays. Not
 * thread-safe.
 */
public final class CpuRwkv6Mixer implements CpuMixer {

    private static final int N_MIX = 5;
    private static final int PLANE_W = 0;
    private static final int PLANE_K = 1;
    private static final int PLANE_V = 2;
    private static final int PLANE_R = 3;
    private static final int PLANE_G = 4;

    private RwkvBlocks.CpuKit kit;
    private CpuMatVec mv;
    private boolean closed;

    private int nEmbd;
    private int headSize;
    private int nHead;
    private int kvWidth;
    private int headGroups;
    private int extra;
    private int decayDim;
    private boolean qrwkv;

    private CpuTensorRef w1;
    private CpuTensorRef w2;
    private CpuTensorRef wr;
    private CpuTensorRef wk;
    private CpuTensorRef wv;
    private CpuTensorRef wg;
    private CpuTensorRef wo;
    /**
     * LoRA deltas of receptance, key, value, gate and output ({@code build_lora_mm}, models/rwkv6-base.cpp); the
     * low-rank lerp and decay products are plain {@code ggml_mul_mat} in llama.cpp and take none.
     */
    private CpuProjectionDeltas.Site rDeltas;
    private CpuProjectionDeltas.Site kDeltas;
    private CpuProjectionDeltas.Site vDeltas;
    private CpuProjectionDeltas.Site gDeltas;
    private CpuProjectionDeltas.Site oDeltas;
    private CpuTensorRef decayW1;
    private CpuTensorRef decayW2;

    private float[] lerpX;
    private float[] lerps;
    private float[] decayBase;
    private float[] first;
    private float[] ln;
    private float[] lnB;
    private float[] rBias;
    private float[] kBias;
    private float[] vBias;

    private float[] shift;
    private float[] state;

    private float[] xy;
    private float[] h;
    private float[] planes;
    private float[] r;
    private float[] k;
    private float[] v;
    private float[] kRaw;
    private float[] vRaw;
    private float[] g;
    private float[] dh;
    private float[] wRaw;
    private float[] w;

    /**
     * Resolves the layer's tensors and recurrent arrays and reserves the working temporaries.
     *
     * @throws IllegalStateException    when already bound or closed, the context has no recurrent memory, or a
     *                                  tensor's shape contradicts the spec
     * @throws IllegalArgumentException when the layer is not an RWKV6 / RWKV6_QWEN2 layer
     * @throws MissingTensorException   when a required tensor is absent
     */
    @Override
    public void bind(CpuForwardContext ctx, LayerSpec layer) {
        if (closed || kit != null) {
            throw new IllegalStateException("RWKV6 time mix is " + (closed ? "closed" : "already bound"));
        }
        Objects.requireNonNull(ctx, "ctx");
        Objects.requireNonNull(layer, "layer");
        if (layer.mixerKind() != MixerKind.RWKV6 && layer.mixerKind() != MixerKind.RWKV6_QWEN2) {
            throw new IllegalArgumentException("CpuRwkv6Mixer cannot serve " + layer.mixerKind() + " layer "
                    + layer.il());
        }
        RecurrentSpec spec = RwkvBlocks.recurrent(layer, layer.mixerKind());
        RwkvBlocks.CpuKit kt = new RwkvBlocks.CpuKit(ctx, layer, "RWKV6 time mix");
        nEmbd = spec.nEmbd();
        headSize = spec.wkvHeadSize();
        nHead = nEmbd / headSize;
        qrwkv = spec.qrwkv();
        bindTensors(kt, nEmbd);
        bindWorking(kt, nEmbd);
        mv = kt.mv;
        kit = kt;
    }

    private void bindTensors(RwkvBlocks.CpuKit kt, int c) {
        w1 = kt.matrix(TensorNames.TIME_MIX_W1, c, 0);
        if (w1.rows() % N_MIX != 0) {
            throw new IllegalStateException(kt.where + ": time_mix_w1 has " + w1.rows() + " rows, not a multiple of "
                    + N_MIX);
        }
        extra = Math.toIntExact(w1.rows() / N_MIX);
        w2 = kt.matrix(TensorNames.TIME_MIX_W2, extra, (long) N_MIX * c);
        wr = kt.matrix(TensorNames.TIME_MIX_RECEPTANCE, c, c);
        wk = kt.matrix(TensorNames.TIME_MIX_KEY, c, 0);
        kvWidth = Math.toIntExact(wk.rows());
        if (kvWidth % headSize != 0 || kvWidth == 0 || nHead % (kvWidth / headSize) != 0) {
            throw new IllegalStateException(kt.where + ": time_mix_key has " + kvWidth + " rows, not a head count"
                    + " dividing " + nHead + " heads of " + headSize);
        }
        headGroups = nHead / (kvWidth / headSize);
        wv = kt.matrix(TensorNames.TIME_MIX_VALUE, c, kvWidth);
        wg = kt.matrix(TensorNames.TIME_MIX_GATE, c, c);
        wo = kt.matrix(TensorNames.TIME_MIX_OUTPUT, c, c);
        CpuProjectionDeltas deltas = kt.ctx.projectionDeltas();
        rDeltas = deltas.serve(kt.layer, TensorNames.weight(TensorNames.TIME_MIX_RECEPTANCE), wr, kt.mv);
        kDeltas = deltas.serve(kt.layer, TensorNames.weight(TensorNames.TIME_MIX_KEY), wk, kt.mv);
        vDeltas = deltas.serve(kt.layer, TensorNames.weight(TensorNames.TIME_MIX_VALUE), wv, kt.mv);
        gDeltas = deltas.serve(kt.layer, TensorNames.weight(TensorNames.TIME_MIX_GATE), wg, kt.mv);
        oDeltas = deltas.serve(kt.layer, TensorNames.weight(TensorNames.TIME_MIX_OUTPUT), wo, kt.mv);
        decayW1 = kt.matrix(TensorNames.TIME_MIX_DECAY_W1, c, 0);
        decayDim = Math.toIntExact(decayW1.rows());
        decayW2 = kt.matrix(TensorNames.TIME_MIX_DECAY_W2, decayDim, c);
        lerpX = kt.vector(TensorNames.TIME_MIX_LERP_X, c);
        lerps = lerps(kt, c);
        decayBase = kt.vector(TensorNames.TIME_MIX_DECAY, c);
        rBias = kt.biasOrNull(TensorNames.TIME_MIX_RECEPTANCE, c);
        kBias = kt.biasOrNull(TensorNames.TIME_MIX_KEY, kvWidth);
        vBias = kt.biasOrNull(TensorNames.TIME_MIX_VALUE, kvWidth);
        if (!qrwkv) {
            first = kt.vector(TensorNames.TIME_MIX_FIRST, c);
            ln = kt.vector(TensorNames.TIME_MIX_LN, c);
            lnB = kt.biasOrNull(TensorNames.TIME_MIX_LN, c);
            if (lnB == null) {
                throw new MissingTensorException(kt.weights.archName(), kt.layer.il(),
                        kt.layer.tensorName(TensorNames.bias(TensorNames.TIME_MIX_LN)));
            }
        }
    }

    /** {@code [5][C]} lerp vectors in plane order w, k, v, r, g: the fused tensor or the five packed at bind time. */
    private static float[] lerps(RwkvBlocks.CpuKit kt, int c) {
        float[] fused = kt.vectorOrNull(TensorNames.TIME_MIX_LERP_FUSED, N_MIX * c);
        if (fused != null) {
            return fused;
        }
        String[] parts = {TensorNames.TIME_MIX_LERP_W, TensorNames.TIME_MIX_LERP_K, TensorNames.TIME_MIX_LERP_V,
                TensorNames.TIME_MIX_LERP_R, TensorNames.TIME_MIX_LERP_G};
        float[] packed = new float[N_MIX * c];
        for (int m = 0; m < N_MIX; m++) {
            System.arraycopy(kt.vector(parts[m], c), 0, packed, m * c, c);
        }
        return packed;
    }

    private void bindWorking(RwkvBlocks.CpuKit kt, int c) {
        shift = kt.state(LayerMemoryPlan.STATE_SHIFT);
        state = kt.state(LayerMemoryPlan.STATE_SCAN);
        CpuWkv.checkRows("RWKV6 shift state", shift, 0, 1, c);
        CpuWkv.checkRows("RWKV6 WKV state", state, 0, nHead * headSize, headSize);
        xy = kt.temp("rwkv6.xy", c);
        h = kt.temp("rwkv6.h", N_MIX * extra);
        planes = kt.temp("rwkv6.planes", N_MIX * c);
        r = kt.temp("rwkv6.r", c);
        k = kt.temp("rwkv6.k", c);
        v = kt.temp("rwkv6.v", c);
        kRaw = headGroups > 1 ? kt.temp("rwkv6.k_raw", kvWidth) : k;
        vRaw = headGroups > 1 ? kt.temp("rwkv6.v_raw", kvWidth) : v;
        g = kt.temp("rwkv6.g", c);
        dh = kt.temp("rwkv6.dh", decayDim);
        wRaw = kt.temp("rwkv6.w_raw", c);
        w = kt.temp("rwkv6.w", c);
    }

    @Override
    public void decode(int pos, float[] in, float[] out) {
        prefill(pos, 1, in, out);
    }

    /**
     * Runs the time mix over {@code n} rows of {@code in} into {@code out} and advances the layer's shift and WKV
     * state.
     *
     * @throws IllegalStateException    when not bound or closed
     * @throws IllegalArgumentException for a token count outside {@code [1, ubatch]} or arrays too small for the rows
     */
    @Override
    public void prefill(int pos0, int n, float[] in, float[] out) {
        if (closed || kit == null) {
            throw new IllegalStateException("RWKV6 time mix is " + (closed ? "closed" : "not bound"));
        }
        int c = nEmbd;
        kit.checkStep(n, in, out, c);
        int plane = n * c;
        CpuRwkvMix.tokenShiftLerp(n, c, 0, RwkvKernels.SHIFT_MODE_LERP_X, in, 0, shift, 0, lerpX, null, xy, 0);
        mv.matmul(w1, xy, 0, h, 0, n);
        CpuOps.unary(ActOp.TANH, ActParams.NONE, h, 0, h, 0, n * N_MIX * extra);
        for (int m = 0; m < N_MIX; m++) {
            for (int t = 0; t < n; t++) {
                mv.matvecRows(w2, (long) m * c, c, h, t * N_MIX * extra + m * extra, planes, m * plane + t * c);
            }
        }
        CpuRwkvMix.tokenShiftLerp(n, c, N_MIX, RwkvKernels.SHIFT_MODE_LORA, in, 0, shift, 0, null, lerps, planes, 0);
        project(wr, rDeltas, rBias, planes, PLANE_R * plane, r, n);
        project(wk, kDeltas, kBias, planes, PLANE_K * plane, kRaw, n);
        project(wv, vDeltas, vBias, planes, PLANE_V * plane, vRaw, n);
        mv.matmul(wg, planes, PLANE_G * plane, g, 0, n);
        gDeltas.applyRows(n, planes, PLANE_G * plane, g, 0);
        if (headGroups > 1) {
            repeatHeads(kRaw, k, n);
            repeatHeads(vRaw, v, n);
        }
        mv.matmul(decayW1, planes, PLANE_W * plane, dh, 0, n);
        CpuOps.unary(ActOp.TANH, ActParams.NONE, dh, 0, dh, 0, n * decayDim);
        mv.matmul(decayW2, dh, 0, wRaw, 0, n);
        CpuRwkvMix.decay6(plane, c, wRaw, 0, decayBase, w, 0);
        if (qrwkv) {
            CpuOps.glu(GluOp.MUL, ActParams.NONE, k, 0, w, 0, wRaw, 0, plane);
            CpuOps.axpy(-1.0f, wRaw, 0, k, 0, plane);
            CpuWkv.gla(n, c, nHead, RwkvKernels.GLA_G_PER_CHANNEL, RwkvKernels.glaScale(headSize), k, 0, v, 0, r, 0,
                    w, 0, state, 0, xy, 0);
        } else {
            CpuWkv.wkv6(n, c, nHead, k, 0, v, 0, r, 0, first, w, 0, state, 0, xy, 0);
            NormMath.rowNorm(NormKind.LAYER, RwkvBlocks.GROUP_NORM_EPS, n * nHead, headSize, headSize, nHead, xy, 0,
                    ln, lnB, xy, 0);
        }
        CpuOps.glu(qrwkv ? GluOp.SIGMUL : GluOp.SWIGLU, ActParams.NONE, g, 0, xy, 0, xy, 0, plane);
        mv.matmul(wo, xy, 0, out, 0, n);
        oDeltas.applyRows(n, xy, 0, out, 0);
    }

    /** {@code out[t] = W in[t] + deltas (+ b)} into rows from index 0 of {@code out}. */
    private void project(CpuTensorRef w, CpuProjectionDeltas.Site site, float[] bias, float[] in, int inOff,
                         float[] out, int n) {
        mv.matmul(w, in, inOff, out, 0, n);
        site.applyRows(n, in, inOff, out, 0);
        if (bias != null) {
            CpuOps.addRowBias(out, 0, n, (int) w.rows(), bias, 0);
        }
    }

    /** {@code dst[t][h] = src[t][h / G]} per head of {@code S} for {@code G = n_head / n_head_kv} (ggml_repeat). */
    private void repeatHeads(float[] src, float[] dst, int n) {
        for (int t = 0; t < n; t++) {
            for (int hd = 0; hd < nHead; hd++) {
                System.arraycopy(src, t * kvWidth + (hd / headGroups) * headSize, dst, t * nEmbd + hd * headSize,
                        headSize);
            }
        }
    }

    @Override
    public void close() {
        closed = true;
    }

    @Override
    public String toString() {
        return "CpuRwkv6Mixer{" + (kit != null ? kit.where + (qrwkv ? ", qrwkv" : "") : "unbound")
                + (closed ? ", closed" : "") + '}';
    }
}

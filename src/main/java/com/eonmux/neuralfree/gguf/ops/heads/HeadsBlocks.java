package com.eonmux.neuralfree.gguf.ops.heads;

import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.cpu.block.CpuFfnFactory;
import com.eonmux.neuralfree.gguf.cpu.block.CpuLayerExtraFactory;
import com.eonmux.neuralfree.gguf.cpu.block.CpuMixerFactory;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuOps;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.gpu.block.GpuFfnFactory;
import com.eonmux.neuralfree.gguf.gpu.block.GpuLayerExtraFactory;
import com.eonmux.neuralfree.gguf.gpu.block.GpuMixerFactory;
import com.eonmux.neuralfree.gguf.memory.KvWindow;
import com.eonmux.neuralfree.gguf.ops.attn.CpuAttentionCore;
import com.eonmux.neuralfree.gguf.spec.ActParams;

import java.util.Collections;
import java.util.List;

/**
 * Block factories of the heads family, exposed to the block catalogs, and the host math the heads share.
 *
 * <p>The family has no mixers and no FFN blocks. Its one layer extra is the graniteswitch switched LoRA
 * ({@code LayerSpec.Extras.SWITCHED_LORA}: {@link SwitchedLoraExtra}, {@link CpuSwitchedLoraExtra}). Its extras
 * share one model-level {@link SwitchedLoraExtra.Router}, which the factory catalog creates per model and passes
 * to {@link #gpuExtras(SwitchedLoraExtra.Router)} / {@link #cpuExtras(SwitchedLoraExtra.Router)}. The no-argument
 * lists register the feature id with factories whose creation fails with a message naming the missing router.</p>
 *
 * <p>The pooling, classifier and decision heads ({@link PoolingHead}, {@link ClassifierHead}, {@link ClefDecoder},
 * {@link VllmDecision2Head}, {@link Lfm2DecisionHead}, {@link ModernBertDecisionHead}) run once per request over the
 * final hidden rows ({@code ForwardBackend.hiddenRows}), on the host, with the package-private helpers below. Each
 * helper allocates its result; the heads are request-level, not per decode step.</p>
 */
public final class HeadsBlocks {

    private HeadsBlocks() {
    }

    // ================================================================== factory exposure

    /** The heads family has no mixers. */
    public static List<GpuMixerFactory> gpuMixers() {
        return Collections.emptyList();
    }

    /** The heads family has no mixers. */
    public static List<CpuMixerFactory> cpuMixers() {
        return Collections.emptyList();
    }

    /** The heads family has no FFN blocks. */
    public static List<GpuFfnFactory> gpuFfns() {
        return Collections.emptyList();
    }

    /** The heads family has no FFN blocks. */
    public static List<CpuFfnFactory> cpuFfns() {
        return Collections.emptyList();
    }

    /** GPU switched-LoRA extras without a router: creation fails until the catalog passes one. */
    public static List<GpuLayerExtraFactory> gpuExtras() {
        return gpuExtras(null);
    }

    /**
     * GPU switched-LoRA extras sharing {@code router}.
     *
     * @param router the model's adapter router, or null when the catalog has none (creation then fails)
     */
    public static List<GpuLayerExtraFactory> gpuExtras(SwitchedLoraExtra.Router router) {
        return Collections.singletonList(SwitchedLoraExtra.factory(router));
    }

    /** CPU switched-LoRA extras without a router: creation fails until the catalog passes one. */
    public static List<CpuLayerExtraFactory> cpuExtras() {
        return cpuExtras(null);
    }

    /**
     * CPU switched-LoRA extras sharing {@code router}.
     *
     * @param router the model's adapter router, or null when the catalog has none (creation then fails)
     */
    public static List<CpuLayerExtraFactory> cpuExtras(SwitchedLoraExtra.Router router) {
        return Collections.singletonList(CpuSwitchedLoraExtra.factory(router));
    }

    // ================================================================== shared host math

    /** Rows of {@code hidden} packed row-major, each checked to hold {@code width} floats. */
    static float[] pack(float[][] hidden, int width) {
        if (hidden == null || hidden.length == 0) {
            throw new IllegalArgumentException("the head needs at least one hidden row");
        }
        float[] out = new float[Math.multiplyExact(hidden.length, width)];
        for (int t = 0; t < hidden.length; t++) {
            if (hidden[t] == null || hidden[t].length != width) {
                throw new IllegalArgumentException("hidden row " + t + " has "
                        + (hidden[t] == null ? "no" : String.valueOf(hidden[t].length)) + " floats, expected " + width);
            }
            System.arraycopy(hidden[t], 0, out, t * width, width);
        }
        return out;
    }

    /**
     * {@code y[t] = W x[t] (+ b)} for {@code nRows} packed rows of {@code w.cols()} floats; returns the packed
     * {@code [nRows][w.rows()]} result. A one-element bias is broadcast (ggml_add of a {@code {1}} tensor).
     */
    static float[] linear(CpuMatVec mv, CpuTensorRef w, float[] bias, float[] x, int nRows) {
        int in = toInt(w.cols(), w);
        int out = toInt(w.rows(), w);
        if (x.length < (long) nRows * in) {
            throw new IllegalArgumentException("input of " + w + " holds " + x.length + " floats, " + nRows
                    + " rows of " + in + " are needed");
        }
        float[] y = new float[Math.multiplyExact(nRows, out)];
        mv.matmul(w, x, 0, in, y, 0, out, nRows);
        if (bias != null) {
            addBias(y, nRows, out, bias, w.toString());
        }
        return y;
    }

    /** Adds a {@code width}-float bias (or a broadcast one-element bias) to each of {@code nRows} packed rows. */
    static void addBias(float[] y, int nRows, int width, float[] bias, String what) {
        if (bias.length == width) {
            CpuOps.addRowBias(y, 0, nRows, width, bias, 0);
        } else if (bias.length == 1) {
            float b = bias[0];
            int n = nRows * width;
            for (int i = 0; i < n; i++) {
                y[i] += b;
            }
        } else {
            throw new IllegalArgumentException("bias of " + what + " has " + bias.length + " floats, expected "
                    + width + " or 1");
        }
    }

    /** In-place LayerNorm ({@code LLM_NORM}) of {@code nRows} packed rows; {@code b} may be null. */
    static void layerNorm(float[] x, int nRows, int width, float[] w, float[] b, float eps) {
        CpuOps.norm(NormKind.LAYER, eps, x, 0, x, 0, nRows, width, w, b);
    }

    /** LayerNorm of {@code nRows} packed rows into a new array; {@code b} may be null. */
    static float[] layerNormed(float[] x, int nRows, int width, float[] w, float[] b, float eps) {
        float[] y = new float[nRows * width];
        CpuOps.norm(NormKind.LAYER, eps, x, 0, y, 0, nRows, width, w, b);
        return y;
    }

    /** In-place {@code ggml_l2_norm} of {@code nRows} packed rows ({@link NormKind#L2_CLAMP}). */
    static void l2Norm(float[] x, int nRows, int width, float eps) {
        CpuOps.norm(NormKind.L2_CLAMP, eps, x, 0, x, 0, nRows, width, null, null);
    }

    /** In-place activation of the first {@code n} floats. */
    static void activate(ActOp op, float[] x, int n) {
        CpuOps.unary(op, ActParams.NONE, x, 0, x, 0, n);
    }

    /** {@code a[i] += b[i]} over the first {@code n} floats. */
    static void addInPlace(float[] a, float[] b, int n) {
        CpuOps.add(a, 0, b, 0, a, 0, n);
    }

    /** {@code a[aOff + i] += b[bOff + i]} for {@code i < n}. */
    static void addInPlace(float[] a, int aOff, float[] b, int bOff, int n) {
        CpuOps.add(a, aOff, b, bOff, a, aOff, n);
    }

    /**
     * Multi-head attention without mask or positions (every key visible): {@code nQ} packed query rows of
     * {@code nHead * headDim} against {@code nKv} rows whose K and V start at {@code kOff} / {@code vOff} with row
     * stride {@code kvStride}; returns the packed {@code [nQ][nHead * headDim]} context.
     */
    static float[] attendAll(float[] q, int qStride, int nQ, float[] kv, int kOff, int vOff, int kvStride, int nKv,
                             int nHead, int nHeadKv, int headDim, float scale) {
        CpuAttentionCore.Params p = CpuAttentionCore.Params.builder(nHead).scale(scale)
                .window(KvWindow.FULL_BIDIRECTIONAL).build();
        int[] kvPos = new int[nKv];
        int[] qPos = new int[nQ];
        CpuAttentionCore.DenseKv dense = CpuAttentionCore.DenseKv.of(kv, kOff, kvStride, headDim, kv, vOff, kvStride,
                headDim, nHeadKv, headDim, headDim, kvPos, 0);
        int width = nHead * headDim;
        float[] out = new float[nQ * width];
        CpuAttentionCore.attendRows(p, nQ, q, 0, qStride, qPos, 0, dense, 0, nKv, out, 0, width);
        return out;
    }

    /** A required tensor decoded to F32, checked to hold {@code n} floats. */
    static float[] vector(CpuWeights w, String name, int n) {
        float[] v = w.floats(name);
        if (v.length != n) {
            throw new IllegalArgumentException("tensor " + name + " has " + v.length + " floats, expected " + n);
        }
        return v;
    }

    /** A required matrix checked to map {@code cols} inputs to {@code rows} outputs ({@code rows <= 0}: any). */
    static CpuTensorRef matrix(CpuWeights w, String name, long cols, long rows) {
        CpuTensorRef m = w.get(name);
        if (m.cols() != cols || (rows > 0 && m.rows() != rows)) {
            throw new IllegalArgumentException("tensor " + name + " is " + m.cols() + " x " + m.rows()
                    + " (in x out), expected " + cols + " x " + (rows > 0 ? String.valueOf(rows) : "any"));
        }
        return m;
    }

    static int toInt(long v, Object what) {
        if (v <= 0 || v > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("dimension " + v + " of " + what + " is outside the int range");
        }
        return (int) v;
    }
}

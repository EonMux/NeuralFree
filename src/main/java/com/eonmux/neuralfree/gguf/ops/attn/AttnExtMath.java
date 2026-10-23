package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.gguf.arch.AttnGateKind;
import com.eonmux.neuralfree.gguf.ops.act.ActMath;
import com.eonmux.neuralfree.gguf.spec.AttentionSpec;

import java.util.Objects;

/**
 * CPU reference of the attention extras and the kernel codes they map to: the Java twin of
 * {@code LLM_AttnTempScale} and {@code LLM_AttnOutGate} ({@code kernels/llm/attn_ext.cl}).
 *
 * <p>Temperature (llama-graph.cpp:213-238), per query position {@code p}:
 * <ul>
 *   <li>log form {@code log(floor((p + offset) / floorScale) + 1) * scale + 1} (llama4, mistral3, deepseek2);</li>
 *   <li>pow form {@code exp(scale * log1p((p + offset) / floorScale))} (beam).</li>
 * </ul>
 * Output gates multiply the attention output before {@code wo} by {@code act(g + bias[h])}: element-wise
 * kinds read one gate per element ({@code n_head * head_dim} wide), per-head kinds one gate per head
 * (broadcast over the head's values). The activation is softplus for ELEMENTWISE_SOFTPLUS and
 * PER_HEAD_SOFTPLUS (models/laguna.cpp:247-258), the base-2 softplus {@code softplus(g * ln 2) * (1 / ln 2)}
 * for SOFTPLUS_BASE2 in ggml's order of a scale, a softplus and a scale (models/k2-horizon.cpp:283-288), and
 * sigmoid otherwise. Q_INTERLEAVED gates are split from the Q projection first and then applied as
 * ELEMENTWISE_SIGMOID.
 */
public final class AttnExtMath {

    /** {@code LLM_AttnOutGate} kind codes: {@link AttnGateKind} ordinal minus 2. */
    public static final int GATE_ELEMENTWISE_SIGMOID = 0;
    public static final int GATE_ELEMENTWISE_SOFTPLUS = 1;
    public static final int GATE_PER_HEAD_SIGMOID = 2;
    public static final int GATE_PER_HEAD_SIGMOID_BIAS = 3;
    public static final int GATE_PER_HEAD_SOFTPLUS = 4;
    public static final int GATE_SOFTPLUS_BASE2 = 5;

    /** {@code ln 2} as k2-horizon scales the gate before the softplus (models/k2-horizon.cpp:283). */
    static final float LN2 = 0.6931471805599453f;
    /** {@code 1 / ln 2} as k2-horizon scales the softplus after it (models/k2-horizon.cpp:288). */
    static final float INV_LN2 = 1.4426950408889634f;

    private AttnExtMath() {
    }

    /**
     * Temperature factor of a query position with llama.cpp's arithmetic: the log form floors a float
     * quotient and evaluates the logarithm in double; the pow form is computed in double.
     *
     * @throws IllegalArgumentException when {@code floorScale} is not positive or {@code scale} is 0 or NaN
     */
    public static float temperatureFactor(boolean powForm, int floorScale, float scale, float offset, int pos) {
        if (floorScale <= 0) {
            throw new IllegalArgumentException("attention temperature floor scale must be positive, got " + floorScale);
        }
        if (scale == 0f || Float.isNaN(scale)) {
            throw new IllegalArgumentException("attention temperature scale must be non-zero, got " + scale);
        }
        if (powForm) {
            return (float) Math.exp((double) scale * Math.log1p((pos + (double) offset) / (double) floorScale));
        }
        float x = (float) pos + offset;
        float q = (float) Math.floor(x / (float) floorScale);
        return (float) (Math.log(q + 1.0) * scale + 1.0);
    }

    /** Temperature factor of {@code t} at a position; 1 when {@code t} is disabled. */
    public static float temperatureFactor(AttentionSpec.Temperature t, int pos) {
        Objects.requireNonNull(t, "temperature");
        if (!t.enabled()) {
            return 1.0f;
        }
        return temperatureFactor(t.pow(), t.floorScale(), t.scale(), t.offset(), pos);
    }

    /**
     * CPU twin of {@code LLM_AttnTempScale}: {@code q[qOff + i*rowStride + e] *= factor(pos[posOff + i])}
     * for tokens {@code i < nTokens} and {@code e < nHeads * headDim}. A disabled temperature leaves q unchanged.
     *
     * @throws IllegalArgumentException when a view lies outside its array
     */
    public static void applyTemperature(AttentionSpec.Temperature t, int nTokens, int nHeads, int headDim,
                                        int[] pos, int posOff, float[] q, int qOff, int rowStride) {
        Objects.requireNonNull(t, "temperature");
        int width = rowWidth(nHeads, headDim);
        checkPositions(pos, posOff, nTokens);
        checkView("q", q, qOff, rowStride, nTokens, width);
        if (!t.enabled()) {
            return;
        }
        for (int i = 0; i < nTokens; i++) {
            float f = temperatureFactor(t, pos[posOff + i]);
            int base = qOff + i * rowStride;
            for (int e = 0; e < width; e++) {
                q[base + e] *= f;
            }
        }
    }

    /**
     * Kernel kind code of a gate.
     *
     * @throws IllegalArgumentException for NONE
     */
    public static int gateKernelKind(AttnGateKind kind) {
        switch (Objects.requireNonNull(kind, "kind")) {
            case Q_INTERLEAVED:
            case ELEMENTWISE_SIGMOID:
                return GATE_ELEMENTWISE_SIGMOID;
            case ELEMENTWISE_SOFTPLUS:
                return GATE_ELEMENTWISE_SOFTPLUS;
            case PER_HEAD_SIGMOID:
                return GATE_PER_HEAD_SIGMOID;
            case PER_HEAD_SIGMOID_BIAS:
                return GATE_PER_HEAD_SIGMOID_BIAS;
            case PER_HEAD_SOFTPLUS:
                return GATE_PER_HEAD_SOFTPLUS;
            case SOFTPLUS_BASE2:
                return GATE_SOFTPLUS_BASE2;
            case NONE:
            default:
                throw new IllegalArgumentException("attention gate kind " + kind + " has no output-gate kernel");
        }
    }

    /** True when the gate holds one value per head rather than one per element. */
    public static boolean gatePerHead(AttnGateKind kind) {
        return kernelKindPerHead(gateKernelKind(kind));
    }

    /** True when a kernel kind code reads one gate per head. */
    static boolean kernelKindPerHead(int kernelKind) {
        return kernelKind == GATE_PER_HEAD_SIGMOID || kernelKind == GATE_PER_HEAD_SIGMOID_BIAS
                || kernelKind == GATE_PER_HEAD_SOFTPLUS;
    }

    /** True when the kind requires a per-head bias (other kinds take one optionally, e.g. beam's element-wise gate). */
    public static boolean gateRequiresBias(AttnGateKind kind) {
        return gateKernelKind(kind) == GATE_PER_HEAD_SIGMOID_BIAS;
    }

    /** Gate values per token: {@code nHead} for per-head kinds, {@code nHead * headDim} otherwise. */
    public static int gateWidth(AttnGateKind kind, int nHead, int headDim) {
        return gatePerHead(kind) ? nHead : rowWidth(nHead, headDim);
    }

    /**
     * Gate activation of a kernel kind code: softplus for ELEMENTWISE_SOFTPLUS and PER_HEAD_SOFTPLUS,
     * {@code softplus(g * ln 2) * (1 / ln 2)} for SOFTPLUS_BASE2, sigmoid otherwise.
     */
    public static float gateActivation(int kernelKind, float g) {
        switch (kernelKind) {
            case GATE_ELEMENTWISE_SOFTPLUS:
            case GATE_PER_HEAD_SOFTPLUS:
                return ActMath.softplus(g);
            case GATE_SOFTPLUS_BASE2:
                return ActMath.softplus(g * LN2) * INV_LN2;
            default:
                return ActMath.sigmoid(g);
        }
    }

    /**
     * CPU twin of {@code LLM_AttnOutGate}:
     * {@code attn[attnOff + t*attnStride + h*headDim + d] *= act(gate[gateOff + t*gateStride + j] + bias[biasOff + h])}
     * with {@code j = h} for per-head kinds and {@code j = h*headDim + d} otherwise; {@code bias == null} adds no bias.
     *
     * @throws IllegalArgumentException for NONE, a missing required bias, or a view outside its array
     */
    public static void applyOutGate(AttnGateKind kind, int nTokens, int nHead, int headDim,
                                    float[] attn, int attnOff, int attnStride,
                                    float[] gate, int gateOff, int gateStride,
                                    float[] bias, int biasOff) {
        int k = gateKernelKind(kind);
        if (bias == null && k == GATE_PER_HEAD_SIGMOID_BIAS) {
            throw new IllegalArgumentException("attention gate " + kind + " requires a per-head bias");
        }
        int width = rowWidth(nHead, headDim);
        boolean perHead = kernelKindPerHead(k);
        checkView("attn", attn, attnOff, attnStride, nTokens, width);
        checkView("gate", gate, gateOff, gateStride, nTokens, perHead ? nHead : width);
        if (bias != null) {
            checkView("bias", bias, biasOff, nHead, 1, nHead);
        }
        for (int t = 0; t < nTokens; t++) {
            int a = attnOff + t * attnStride;
            int g = gateOff + t * gateStride;
            for (int h = 0; h < nHead; h++) {
                float b = bias == null ? 0f : bias[biasOff + h];
                int ah = a + h * headDim;
                if (perHead) {
                    float s = gateActivation(k, gate[g + h] + b);
                    for (int d = 0; d < headDim; d++) {
                        attn[ah + d] *= s;
                    }
                } else {
                    int gh = g + h * headDim;
                    for (int d = 0; d < headDim; d++) {
                        attn[ah + d] *= gateActivation(k, gate[gh + d] + b);
                    }
                }
            }
        }
    }

    private static int rowWidth(int nHeads, int headDim) {
        if (nHeads < 0 || headDim < 0) {
            throw new IllegalArgumentException("head shape must be non-negative, got " + nHeads + " x " + headDim);
        }
        long w = (long) nHeads * headDim;
        if (w > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("row width " + nHeads + " x " + headDim + " exceeds 2^31 - 1");
        }
        return (int) w;
    }

    private static void checkPositions(int[] pos, int posOff, int nTokens) {
        Objects.requireNonNull(pos, "pos");
        if (nTokens < 0 || posOff < 0 || (long) posOff + nTokens > pos.length) {
            throw new IllegalArgumentException("pos range [" + posOff + ", " + ((long) posOff + nTokens)
                    + ") is outside an array of length " + pos.length);
        }
    }

    private static void checkView(String name, float[] a, int off, int stride, int rows, int cols) {
        Objects.requireNonNull(a, name);
        if (rows < 0 || cols < 0 || off < 0) {
            throw new IllegalArgumentException(name + ": offset and shape must be non-negative");
        }
        if (rows > 1 && stride < cols) {
            throw new IllegalArgumentException(name + ": stride " + stride + " is below the row width " + cols);
        }
        long end = rows == 0 || cols == 0 ? off : (long) off + (long) (rows - 1) * stride + cols;
        if (end > a.length) {
            throw new IllegalArgumentException(name + " view [" + off + ", " + end + ") with stride " + stride
                    + " is outside an array of length " + a.length);
        }
    }
}

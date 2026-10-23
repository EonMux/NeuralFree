package com.eonmux.neuralfree.gguf.ops.act;

import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.spec.ActParams;

import java.util.Objects;

/**
 * CPU reference of the element-wise activations: the Java twin of {@code nf_act} in
 * {@code kernels/llm/common/math.h} and of the {@code LLM_Unary} kernel. Every formula follows the
 * llama.cpp CPU op it names, in single precision, with the parameter convention of
 * {@link ActParams#unaryArgs(ActOp)}:
 * <ul>
 *   <li>XIELU {@code p0 = alpha_n'}, {@code p1 = alpha_p'}, {@code p2 = beta}, {@code p3 = eps} (prepared);</li>
 *   <li>SCALE_BIAS {@code x * p0 + p1}; CLAMP {@code [p0, p1]}; SOFTCAP {@code p0 * tanh(x * p1)};</li>
 *   <li>the other ops take no parameters.</li>
 * </ul>
 * GELU uses the exact tanh / erf formulas (the GPU kernels' and ggml-cuda's), not the fp16 lookup table
 * of ggml-cpu. Stateless and thread-safe; callers shard ranges for parallelism.
 */
public final class ActMath {

    /** ggml {@code GELU_COEF_A} (ggml-cpu/vec.h:962). */
    public static final float GELU_COEF_A = 0.044715f;
    /** ggml {@code GELU_QUICK_COEF} (ggml-cpu/vec.h:963). */
    public static final float GELU_QUICK_COEF = -1.702f;
    /** ggml {@code SQRT_2_OVER_PI} (ggml-cpu/vec.h:964). */
    public static final float SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;
    /** ggml {@code SQRT_2_INV} (ggml-cpu/vec.h:965). */
    public static final float SQRT_2_INV = 0.70710678118654752440084436210484f;
    /** Softplus switches to the identity above this input (ggml-impl.h:107). */
    public static final float SOFTPLUS_THRESHOLD = 20.0f;

    private ActMath() {
    }

    /** {@code max(x, 0)} (ggml_vec_relu_f32: {@code x > 0 ? x : 0}). */
    public static float relu(float x) {
        return x > 0.0f ? x : 0.0f;
    }

    /** {@code relu(x)^2} (llama-graph.cpp LLM_FFN_RELU_SQR: ggml_relu then ggml_sqr). */
    public static float reluSqr(float x) {
        float r = relu(x);
        return r * r;
    }

    /** {@code 0.5 x (1 + tanh(sqrt(2/pi) x (1 + 0.044715 x^2)))} (ggml-cpu/vec.h:967-968). */
    public static float geluTanh(float x) {
        return 0.5f * x * (1.0f + (float) Math.tanh(SQRT_2_OVER_PI * x * (1.0f + GELU_COEF_A * x * x)));
    }

    /** {@code 0.5 x (1 + erf(x / sqrt(2)))} (ggml-cpu/vec.h ggml_vec_gelu_erf_f32). */
    public static float geluErf(float x) {
        return 0.5f * x * (1.0f + erf(x * SQRT_2_INV));
    }

    /** {@code x / (1 + e^(-1.702 x))} (ggml-cpu/vec.h:1029-1030). */
    public static float geluQuick(float x) {
        return x * (1.0f / (1.0f + (float) Math.exp(GELU_QUICK_COEF * x)));
    }

    /** {@code x / (1 + e^-x)} (ggml-cpu/vec.h:1060). */
    public static float silu(float x) {
        return x / (1.0f + (float) Math.exp(-x));
    }

    /** {@code 1 / (1 + e^-x)} (ggml-cpu/unary-ops.cpp op_sigmoid). */
    public static float sigmoid(float x) {
        return 1.0f / (1.0f + (float) Math.exp(-x));
    }

    /** {@code x > 20 ? x : log(1 + e^x)} (ggml-impl.h:107, ggml-cpu/unary-ops.cpp op_softplus). */
    public static float softplus(float x) {
        return x > SOFTPLUS_THRESHOLD ? x : (float) Math.log(1.0f + (float) Math.exp(x));
    }

    /** {@code tanh(x)}. */
    public static float tanh(float x) {
        return (float) Math.tanh(x);
    }

    /**
     * xIELU with load-time prepared parameters (ggml-cpu/unary-ops.cpp:55-62):
     * {@code x > 0 ? alphaP x^2 + beta x : (expm1(min(x, eps)) - x) alphaN + beta x}.
     */
    public static float xielu(float x, float alphaN, float alphaP, float beta, float eps) {
        if (x > 0.0f) {
            return alphaP * x * x + beta * x;
        }
        float minXEps = Math.min(x, eps);
        return ((float) Math.expm1(minXEps) - x) * alphaN + beta * x;
    }

    /** {@code x * scale + bias} (ggml_scale_bias). */
    public static float scaleBias(float x, float scale, float bias) {
        return x * scale + bias;
    }

    /** ggml_clamp: {@code MAX(MIN(x, hi), lo)} with the C macro comparisons (ggml-cpu/ops.cpp clamp_f32). */
    public static float clamp(float x, float lo, float hi) {
        float a = x < hi ? x : hi;
        return a > lo ? a : lo;
    }

    /** {@code e^x}. */
    public static float exp(float x) {
        return (float) Math.exp(x);
    }

    /** {@code sqrt(softplus(x))} (MoE SQRT_SOFTPLUS gating, llama-graph.cpp build_moe_ffn). */
    public static float sqrtSoftplus(float x) {
        return (float) Math.sqrt(softplus(x));
    }

    /** {@code p0 * tanh(x * p1)}: a softcap with {@code p0 = cap} and {@code p1 = scale / cap}. */
    public static float softcap(float x, float p0, float p1) {
        return p0 * (float) Math.tanh(x * p1);
    }

    /** Single-precision error function: the float rounding of {@link #erf(double)}. */
    public static float erf(float x) {
        return (float) erf((double) x);
    }

    /**
     * Error function in double precision (W. J. Cody, "Rational Chebyshev approximations for the error
     * function", Math. Comp. 23 (1969), algorithm CALERF; absolute error below 3e-16).
     */
    public static double erf(double x) {
        double y = Math.abs(x);
        if (Double.isNaN(x)) {
            return x;
        }
        if (y <= ERF_THRESHOLD) {
            double ysq = y > ERF_XSMALL ? x * x : 0.0;
            double xnum = ERF_A[4] * ysq;
            double xden = ysq;
            for (int i = 0; i < 3; i++) {
                xnum = (xnum + ERF_A[i]) * ysq;
                xden = (xden + ERF_B[i]) * ysq;
            }
            return x * (xnum + ERF_A[3]) / (xden + ERF_B[3]);
        }
        double erfc;
        if (y <= 4.0) {
            double xnum = ERF_C[8] * y;
            double xden = y;
            for (int i = 0; i < 7; i++) {
                xnum = (xnum + ERF_C[i]) * y;
                xden = (xden + ERF_D[i]) * y;
            }
            erfc = (xnum + ERF_C[7]) / (xden + ERF_D[7]);
        } else if (y >= ERF_XBIG) {
            erfc = 0.0;
        } else {
            double ysq = 1.0 / (y * y);
            double xnum = ERF_P[5] * ysq;
            double xden = ysq;
            for (int i = 0; i < 4; i++) {
                xnum = (xnum + ERF_P[i]) * ysq;
                xden = (xden + ERF_Q[i]) * ysq;
            }
            erfc = ysq * (xnum + ERF_P[4]) / (xden + ERF_Q[4]);
            erfc = (ERF_SQRPI - erfc) / y;
        }
        if (erfc != 0.0) {
            /* exp(-y^2) split as exp(-ysq^2) * exp(-(y - ysq)(y + ysq)) to keep the argument exact. */
            double ysq = Math.floor(y * 16.0) / 16.0;
            double del = (y - ysq) * (y + ysq);
            erfc = Math.exp(-ysq * ysq) * Math.exp(-del) * erfc;
        }
        double r = 1.0 - erfc;
        return x < 0.0 ? -r : r;
    }

    /* Cody CALERF coefficients and break points. */
    private static final double ERF_THRESHOLD = 0.46875;
    private static final double ERF_XSMALL = 1.11e-16;
    private static final double ERF_XBIG = 26.543;
    private static final double ERF_SQRPI = 5.6418958354775628695e-1;
    private static final double[] ERF_A = {
        3.16112374387056560e00, 1.13864154151050156e02, 3.77485237685302021e02,
        3.20937758913846947e03, 1.85777706184603153e-1};
    private static final double[] ERF_B = {
        2.36012909523441209e01, 2.44024637934444173e02, 1.28261652607737228e03,
        2.84423683343917062e03};
    private static final double[] ERF_C = {
        5.64188496988670089e-1, 8.88314979438837594e00, 6.61191906371416295e01,
        2.98635138197400131e02, 8.81952221241769090e02, 1.71204761263407058e03,
        2.05107837782607147e03, 1.23033935479799725e03, 2.15311535474403846e-8};
    private static final double[] ERF_D = {
        1.57449261107098347e01, 1.17693950891312499e02, 5.37181101862009858e02,
        1.62138957456669019e03, 3.29079923573345963e03, 4.36261909014324716e03,
        3.43936767414372164e03, 1.23033935480374942e03};
    private static final double[] ERF_P = {
        3.05326634961232344e-1, 3.60344899949804439e-1, 1.25781726111229246e-1,
        1.60837851487422766e-2, 6.58749161529837803e-4, 1.63153871373020978e-2};
    private static final double[] ERF_Q = {
        2.56852019228982242e00, 1.87295284992346725e00, 5.27905102951428412e-1,
        6.05183413124413191e-2, 2.33520497626869185e-3};

    /**
     * {@code nf_act(op, x, p0, p1, p2, p3)}.
     *
     * @throws NullPointerException when op is null
     */
    public static float apply(ActOp op, float x, float p0, float p1, float p2, float p3) {
        switch (Objects.requireNonNull(op, "op")) {
            case RELU:          return relu(x);
            case RELU_SQR:      return reluSqr(x);
            case GELU_TANH:     return geluTanh(x);
            case GELU_ERF:      return geluErf(x);
            case GELU_QUICK:    return geluQuick(x);
            case SILU:          return silu(x);
            case SIGMOID:       return sigmoid(x);
            case SOFTPLUS:      return softplus(x);
            case TANH:          return tanh(x);
            case XIELU:         return xielu(x, p0, p1, p2, p3);
            case SCALE_BIAS:    return scaleBias(x, p0, p1);
            case CLAMP:         return clamp(x, p0, p1);
            case EXP:           return exp(x);
            case SQRT_SOFTPLUS: return sqrtSoftplus(x);
            case IDENTITY:      return x;
            case SOFTCAP:       return softcap(x, p0, p1);
            default:
                throw new IllegalArgumentException("unhandled activation op " + op);
        }
    }

    /**
     * {@code nf_act} for a kernel op code ({@link ActOp#code()}).
     *
     * @throws IllegalArgumentException when no op has that code
     */
    public static float apply(int code, float x, float p0, float p1, float p2, float p3) {
        return apply(ActOp.fromCode(code), x, p0, p1, p2, p3);
    }

    /** One activation with the parameters of {@code p} (see {@link ActParams#unaryArgs(ActOp)}). */
    public static float apply(ActOp op, ActParams p, float x) {
        float[] a = Objects.requireNonNull(p, "params").unaryArgs(op);
        return apply(op, x, a[0], a[1], a[2], a[3]);
    }

    /** In-place form of {@link #apply(ActOp, ActParams, float[], int, float[], int, int)}. */
    public static void applyInPlace(ActOp op, ActParams p, float[] x, int off, int n) {
        apply(op, p, x, off, x, off, n);
    }

    /**
     * CPU twin of {@code LLM_Unary}: {@code out[outOff + i] = act(in[inOff + i])} for {@code i < n}.
     * {@code out} may be {@code in} when {@code outOff == inOff} or the ranges do not overlap.
     *
     * @throws IllegalArgumentException when a range lies outside its array or {@code n < 0}
     * @throws IllegalStateException    for SOFTCAP with a non-positive cap
     */
    public static void apply(ActOp op, ActParams p, float[] in, int inOff, float[] out, int outOff, int n) {
        Objects.requireNonNull(op, "op");
        checkRange("in", in, inOff, n);
        checkRange("out", out, outOff, n);
        float[] a = Objects.requireNonNull(p, "params").unaryArgs(op);
        float p0 = a[0];
        float p1 = a[1];
        switch (op) {
            case RELU:
                for (int i = 0; i < n; i++) out[outOff + i] = relu(in[inOff + i]);
                break;
            case RELU_SQR:
                for (int i = 0; i < n; i++) out[outOff + i] = reluSqr(in[inOff + i]);
                break;
            case GELU_TANH:
                for (int i = 0; i < n; i++) out[outOff + i] = geluTanh(in[inOff + i]);
                break;
            case GELU_ERF:
                for (int i = 0; i < n; i++) out[outOff + i] = geluErf(in[inOff + i]);
                break;
            case GELU_QUICK:
                for (int i = 0; i < n; i++) out[outOff + i] = geluQuick(in[inOff + i]);
                break;
            case SILU:
                for (int i = 0; i < n; i++) out[outOff + i] = silu(in[inOff + i]);
                break;
            case SIGMOID:
                for (int i = 0; i < n; i++) out[outOff + i] = sigmoid(in[inOff + i]);
                break;
            case SOFTPLUS:
                for (int i = 0; i < n; i++) out[outOff + i] = softplus(in[inOff + i]);
                break;
            case TANH:
                for (int i = 0; i < n; i++) out[outOff + i] = tanh(in[inOff + i]);
                break;
            case XIELU:
                for (int i = 0; i < n; i++) out[outOff + i] = xielu(in[inOff + i], p0, p1, a[2], a[3]);
                break;
            case SCALE_BIAS:
                for (int i = 0; i < n; i++) out[outOff + i] = scaleBias(in[inOff + i], p0, p1);
                break;
            case CLAMP:
                for (int i = 0; i < n; i++) out[outOff + i] = clamp(in[inOff + i], p0, p1);
                break;
            case EXP:
                for (int i = 0; i < n; i++) out[outOff + i] = exp(in[inOff + i]);
                break;
            case SQRT_SOFTPLUS:
                for (int i = 0; i < n; i++) out[outOff + i] = sqrtSoftplus(in[inOff + i]);
                break;
            case IDENTITY:
                if (in != out || inOff != outOff) {
                    System.arraycopy(in, inOff, out, outOff, n);
                }
                break;
            case SOFTCAP:
                for (int i = 0; i < n; i++) out[outOff + i] = softcap(in[inOff + i], p0, p1);
                break;
            default:
                throw new IllegalArgumentException("unhandled activation op " + op);
        }
    }

    /**
     * Validates that {@code [off, off + n)} lies inside {@code a}.
     *
     * @throws IllegalArgumentException naming the operand otherwise
     */
    static void checkRange(String name, float[] a, int off, int n) {
        Objects.requireNonNull(a, name);
        if (n < 0 || off < 0 || (long) off + n > a.length) {
            throw new IllegalArgumentException(name + " range [" + off + ", " + ((long) off + n)
                    + ") is outside an array of length " + a.length);
        }
    }
}

package com.eonmux.neuralfree.gguf.ops.act;

import com.eonmux.neuralfree.gguf.arch.GluOp;
import com.eonmux.neuralfree.gguf.spec.ActParams;

import java.util.Objects;

/**
 * CPU reference of the gated linear units {@code out = f(gate, up)}: the Java twin of {@code nf_glu} in
 * {@code kernels/llm/common/math.h} and of the {@code LLM_GLU} kernel. Parameters follow {@link ActParams}:
 * {@code limit} (SWIGLU_OAI, SWIGLU_CLAMP, SWIGLU_LIMITED), {@code alpha} (SWIGLU_OAI sigmoid slope),
 * {@code beta} = {@code situ_beta} and {@code lambda} = {@code situ_linear_beta} (SITU). The gate
 * activations are the {@link ActMath} functions. Stateless and thread-safe; callers shard rows for
 * parallelism.
 */
public final class GluMath {

    private GluMath() {
    }

    /** {@code std::min(a, b)}: {@code b < a ? b : a}. */
    private static float stdMin(float a, float b) {
        return b < a ? b : a;
    }

    /** {@code std::clamp(v, lo, hi)}: {@code v < lo ? lo : (hi < v ? hi : v)}. */
    private static float stdClamp(float v, float lo, float hi) {
        return v < lo ? lo : (hi < v ? hi : v);
    }

    /** gpt-oss {@code swiglu_oai}: {@code x = min(g, limit)}, {@code y = clamp(u, -limit, limit)}, {@code x sigmoid(alpha x) (y + 1)} (ggml-cpu/ops.cpp:3546-3549). */
    public static float swigluOai(float g, float u, float limit, float alpha) {
        float x = stdMin(g, limit);
        float y = stdClamp(u, -limit, limit);
        return x / (1.0f + (float) Math.exp(alpha * (-x))) * (y + 1.0f);
    }

    /** ggml {@code swiglu_clamp}: {@code silu(min(g, limit)) * clamp(u, -limit, limit)} (ggml-cpu/ops.cpp:3697-3700). */
    public static float swigluClamp(float g, float u, float limit) {
        float x = stdMin(g, limit);
        float y = stdClamp(u, -limit, limit);
        return x / (1.0f + (float) Math.exp(-x)) * y;
    }

    /**
     * Graph-level limited SwiGLU: {@code clamp(silu(g), -inf, limit) * clamp(u, -limit, limit)} with ggml_clamp
     * semantics (llama-graph.cpp:2010-2016 and :2430-2438).
     */
    public static float swigluLimited(float g, float u, float limit) {
        float y = ActMath.clamp(u, -limit, limit);
        float a = ActMath.clamp(ActMath.silu(g), Float.NEGATIVE_INFINITY, limit);
        return a * y;
    }

    /**
     * kimi-k3 SiTU: {@code beta tanh(g / beta) sigmoid(g) * (lambda > 0 ? lambda tanh(u / lambda) : u)}
     * (models/kimi-k3.cpp:173-183, llama-graph.cpp:2453-2466; division as multiplication by the reciprocal, as
     * {@code ggml_scale(x, 1/beta)}).
     */
    public static float situ(float g, float u, float beta, float lambda) {
        float a = beta * (float) Math.tanh(g * (1.0f / beta)) * ActMath.sigmoid(g);
        float b = lambda > 0.0f ? lambda * (float) Math.tanh(u * (1.0f / lambda)) : u;
        return a * b;
    }

    /**
     * {@code nf_glu(op, g, u, limit, alpha, beta, lambda)}.
     *
     * @throws NullPointerException when op is null
     */
    public static float apply(GluOp op, float g, float u, float limit, float alpha, float beta, float lambda) {
        switch (Objects.requireNonNull(op, "op")) {
            case SWIGLU:         return ActMath.silu(g) * u;
            case GEGLU_TANH:     return ActMath.geluTanh(g) * u;
            case GEGLU_ERF:      return ActMath.geluErf(g) * u;
            case GEGLU_QUICK:    return ActMath.geluQuick(g) * u;
            case REGLU:          return ActMath.relu(g) * u;
            case SWIGLU_OAI:     return swigluOai(g, u, limit, alpha);
            case SWIGLU_CLAMP:   return swigluClamp(g, u, limit);
            case SWIGLU_LIMITED: return swigluLimited(g, u, limit);
            case SITU:           return situ(g, u, beta, lambda);
            case MUL:            return g * u;
            case SIGMUL:         return ActMath.sigmoid(g) * u;
            default:
                throw new IllegalArgumentException("unhandled GLU op " + op);
        }
    }

    /**
     * {@code nf_glu} for a kernel op code ({@link GluOp#code()}).
     *
     * @throws IllegalArgumentException when no op has that code
     */
    public static float apply(int code, float g, float u, float limit, float alpha, float beta, float lambda) {
        return apply(GluOp.fromCode(code), g, u, limit, alpha, beta, lambda);
    }

    /** One GLU value with the parameters of {@code p}. */
    public static float apply(GluOp op, ActParams p, float g, float u) {
        Objects.requireNonNull(p, "params");
        return apply(op, g, u, p.limit(), p.alpha(), p.beta(), p.lambda());
    }

    /**
     * Contiguous form: {@code out[outOff + i] = f(gate[gateOff + i], up[upOff + i])} for {@code i < n}.
     * {@code out} may be {@code gate} or {@code up} at the same offset.
     *
     * @throws IllegalArgumentException when a range lies outside its array or {@code n < 0}
     */
    public static void apply(GluOp op, ActParams p, float[] gate, int gateOff, float[] up, int upOff,
                             float[] out, int outOff, int n) {
        Objects.requireNonNull(op, "op");
        Objects.requireNonNull(p, "params");
        ActMath.checkRange("gate", gate, gateOff, n);
        ActMath.checkRange("up", up, upOff, n);
        ActMath.checkRange("out", out, outOff, n);
        row(op, p.limit(), p.alpha(), p.beta(), p.lambda(), gate, gateOff, up, upOff, out, outOff, n);
    }

    /**
     * CPU twin of {@code LLM_GLU}: for {@code r < nRows} and {@code c < nCols},
     * {@code out[outOff + r*outStride + c] = f(gate[gateOff + r*gateStride + c], up[upOff + r*upStride + c])}.
     * The fused gate+up layout passes the same array for gate and up with {@code upOff = gateOff + n_ff}.
     * In-place use is exact when every output element coincides with its own gate (or up) element or
     * overlaps no input element.
     *
     * @throws IllegalArgumentException for negative sizes, a stride below {@code nCols} with several rows,
     *                                  or a view outside its array
     */
    public static void applyRows(GluOp op, ActParams p, int nRows, int nCols,
                                 float[] gate, int gateOff, int gateStride,
                                 float[] up, int upOff, int upStride,
                                 float[] out, int outOff, int outStride) {
        Objects.requireNonNull(op, "op");
        Objects.requireNonNull(p, "params");
        if (nRows < 0 || nCols < 0) {
            throw new IllegalArgumentException("GLU shape must be non-negative, got " + nRows + " x " + nCols);
        }
        checkView("gate", gate, gateOff, gateStride, nRows, nCols);
        checkView("up", up, upOff, upStride, nRows, nCols);
        checkView("out", out, outOff, outStride, nRows, nCols);
        float limit = p.limit();
        float alpha = p.alpha();
        float beta = p.beta();
        float lambda = p.lambda();
        for (int r = 0; r < nRows; r++) {
            row(op, limit, alpha, beta, lambda, gate, gateOff + r * gateStride, up, upOff + r * upStride,
                    out, outOff + r * outStride, nCols);
        }
    }

    /** One contiguous run of {@code n} elements, with the op switch hoisted out of the loop. */
    private static void row(GluOp op, float limit, float alpha, float beta, float lambda,
                            float[] g, int go, float[] u, int uo, float[] o, int oo, int n) {
        switch (op) {
            case SWIGLU:
                for (int i = 0; i < n; i++) o[oo + i] = ActMath.silu(g[go + i]) * u[uo + i];
                break;
            case GEGLU_TANH:
                for (int i = 0; i < n; i++) o[oo + i] = ActMath.geluTanh(g[go + i]) * u[uo + i];
                break;
            case GEGLU_ERF:
                for (int i = 0; i < n; i++) o[oo + i] = ActMath.geluErf(g[go + i]) * u[uo + i];
                break;
            case GEGLU_QUICK:
                for (int i = 0; i < n; i++) o[oo + i] = ActMath.geluQuick(g[go + i]) * u[uo + i];
                break;
            case REGLU:
                for (int i = 0; i < n; i++) o[oo + i] = ActMath.relu(g[go + i]) * u[uo + i];
                break;
            case SWIGLU_OAI:
                for (int i = 0; i < n; i++) o[oo + i] = swigluOai(g[go + i], u[uo + i], limit, alpha);
                break;
            case SWIGLU_CLAMP:
                for (int i = 0; i < n; i++) o[oo + i] = swigluClamp(g[go + i], u[uo + i], limit);
                break;
            case SWIGLU_LIMITED:
                for (int i = 0; i < n; i++) o[oo + i] = swigluLimited(g[go + i], u[uo + i], limit);
                break;
            case SITU:
                for (int i = 0; i < n; i++) o[oo + i] = situ(g[go + i], u[uo + i], beta, lambda);
                break;
            case MUL:
                for (int i = 0; i < n; i++) o[oo + i] = g[go + i] * u[uo + i];
                break;
            case SIGMUL:
                for (int i = 0; i < n; i++) o[oo + i] = ActMath.sigmoid(g[go + i]) * u[uo + i];
                break;
            default:
                throw new IllegalArgumentException("unhandled GLU op " + op);
        }
    }

    /** Validates a strided {@code [nRows][nCols]} view. */
    private static void checkView(String name, float[] a, int off, int stride, int nRows, int nCols) {
        Objects.requireNonNull(a, name);
        if (nRows > 1 && stride < nCols) {
            throw new IllegalArgumentException(name + " stride " + stride + " is below the row width " + nCols);
        }
        long end = nRows == 0 || nCols == 0 ? off : (long) off + (long) (nRows - 1) * stride + nCols;
        if (off < 0 || end > a.length) {
            throw new IllegalArgumentException(name + " view [" + off + ", " + end + ") with stride " + stride
                    + " is outside an array of length " + a.length);
        }
    }
}

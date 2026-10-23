package com.eonmux.neuralfree.gguf.spec;

import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.arch.GluOp;

import java.util.Objects;

/**
 * Scalar parameters of an activation or gated-linear-unit op, in the form the kernels consume
 * (kernels/llm/common/math.h {@code nf_act} and {@code nf_glu}).
 *
 * <ul>
 *   <li>GLU: {@code limit} (SWIGLU_OAI / SWIGLU_CLAMP / SWIGLU_LIMITED), {@code alpha} (SWIGLU_OAI),
 *       {@code beta} and {@code lambda} (SITU: {@code situ_beta}, {@code situ_linear_beta}).</li>
 *   <li>XIELU: the load-time prepared parameters of ggml {@code ggml_xielu} (ggml/src/ggml.c:2861-2872):
 *       {@code alpha_n' = beta + softplus(alpha_n)}, {@code alpha_p' = softplus(alpha_p)}, beta, eps, with
 *       {@code softplus(x) = x > 20 ? x : log(1 + exp(x))}. {@link Builder#xieluRaw} applies this preparation.</li>
 *   <li>SCALE_BIAS: {@code x * scale + bias}. CLAMP: {@code [clampLo, clampHi]}. SOFTCAP:
 *       {@code cap * tanh(x * softcapScale / cap)}.</li>
 * </ul>
 * Immutable.
 */
public final class ActParams {

    /** No parameters: unclamped GLUs, {@code alpha = beta = 1}, {@code lambda = 0}. */
    public static final ActParams NONE = builder().build();

    private final float limit;
    private final float alpha;
    private final float beta;
    private final float lambda;
    private final float xieluAlphaN;
    private final float xieluAlphaP;
    private final float xieluBeta;
    private final float xieluEps;
    private final float scale;
    private final float bias;
    private final float clampLo;
    private final float clampHi;
    private final float softcap;
    private final float softcapScale;

    private ActParams(Builder b) {
        this.limit = b.limit;
        this.alpha = b.alpha;
        this.beta = b.beta;
        this.lambda = b.lambda;
        this.xieluAlphaN = b.xieluAlphaN;
        this.xieluAlphaP = b.xieluAlphaP;
        this.xieluBeta = b.xieluBeta;
        this.xieluEps = b.xieluEps;
        this.scale = b.scale;
        this.bias = b.bias;
        this.clampLo = b.clampLo;
        this.clampHi = b.clampHi;
        this.softcap = b.softcap;
        this.softcapScale = b.softcapScale;
    }

    public static Builder builder() {
        return new Builder();
    }

    /** Builder pre-filled with this instance's values. */
    public Builder toBuilder() {
        Builder b = new Builder();
        b.limit = limit;
        b.alpha = alpha;
        b.beta = beta;
        b.lambda = lambda;
        b.xieluAlphaN = xieluAlphaN;
        b.xieluAlphaP = xieluAlphaP;
        b.xieluBeta = xieluBeta;
        b.xieluEps = xieluEps;
        b.scale = scale;
        b.bias = bias;
        b.clampLo = clampLo;
        b.clampHi = clampHi;
        b.softcap = softcap;
        b.softcapScale = softcapScale;
        return b;
    }

    /** GLU clamp limit; {@code +inf} when unclamped. */
    public float limit() { return limit; }
    /** SWIGLU_OAI sigmoid slope. */
    public float alpha() { return alpha; }
    /** SITU {@code situ_beta}. */
    public float beta() { return beta; }
    /** SITU {@code situ_linear_beta}; 0 leaves the up branch linear. */
    public float lambda() { return lambda; }
    /** Prepared XIELU {@code alpha_n' = beta + softplus(alpha_n)}. */
    public float xieluAlphaN() { return xieluAlphaN; }
    /** Prepared XIELU {@code alpha_p' = softplus(alpha_p)}. */
    public float xieluAlphaP() { return xieluAlphaP; }
    public float xieluBeta() { return xieluBeta; }
    public float xieluEps() { return xieluEps; }
    public float scale() { return scale; }
    public float bias() { return bias; }
    public float clampLo() { return clampLo; }
    public float clampHi() { return clampHi; }
    /** Softcap value; 0 disables. */
    public float softcap() { return softcap; }
    /** Pre-cap multiplier applied to {@code x} before {@code tanh(x * softcapScale / cap)}. */
    public float softcapScale() { return softcapScale; }

    /**
     * The four unary parameters {@code p0..p3} of {@code nf_act} for an op (math.h): XIELU
     * {@code (alpha_n', alpha_p', beta, eps)}, SCALE_BIAS {@code (scale, bias)}, CLAMP {@code (lo, hi)},
     * SOFTCAP {@code (cap, softcapScale / cap)}; zeros for the other ops.
     *
     * @throws IllegalStateException for SOFTCAP with a non-positive cap
     */
    public float[] unaryArgs(ActOp op) {
        Objects.requireNonNull(op, "op");
        switch (op) {
            case XIELU:
                return new float[]{xieluAlphaN, xieluAlphaP, xieluBeta, xieluEps};
            case SCALE_BIAS:
                return new float[]{scale, bias, 0f, 0f};
            case CLAMP:
                return new float[]{clampLo, clampHi, 0f, 0f};
            case SOFTCAP:
                if (!(softcap > 0f)) {
                    throw new IllegalStateException("SOFTCAP needs a positive cap, got " + softcap);
                }
                return new float[]{softcap, softcapScale / softcap, 0f, 0f};
            default:
                return new float[4];
        }
    }

    /** The GLU parameters {@code (limit, alpha, beta, lambda)} of {@code nf_glu}; ignored by ops that take none. */
    public float[] gluArgs(GluOp op) {
        Objects.requireNonNull(op, "op");
        return new float[]{limit, alpha, beta, lambda};
    }

    /** ggml {@code softplus}: {@code x > 20 ? x : log(1 + exp(x))} (ggml-impl.h:107). */
    public static float softplus(float x) {
        return x > 20.0f ? x : (float) Math.log(1.0 + Math.exp(x));
    }

    @Override
    public String toString() {
        return "ActParams{limit=" + limit + ", alpha=" + alpha + ", beta=" + beta + ", lambda=" + lambda
                + ", xielu=(" + xieluAlphaN + ", " + xieluAlphaP + ", " + xieluBeta + ", " + xieluEps + ")"
                + ", scale=" + scale + ", bias=" + bias + ", clamp=[" + clampLo + ", " + clampHi + "]"
                + ", softcap=" + softcap + ", softcapScale=" + softcapScale + '}';
    }

    public static final class Builder {
        private float limit = Float.POSITIVE_INFINITY;
        private float alpha = 1.0f;
        private float beta = 1.0f;
        private float lambda = 0.0f;
        private float xieluAlphaN;
        private float xieluAlphaP;
        private float xieluBeta;
        private float xieluEps;
        private float scale = 1.0f;
        private float bias;
        private float clampLo = Float.NEGATIVE_INFINITY;
        private float clampHi = Float.POSITIVE_INFINITY;
        private float softcap;
        private float softcapScale = 1.0f;

        private Builder() {
        }

        public Builder limit(float v) { this.limit = v; return this; }
        public Builder alpha(float v) { this.alpha = v; return this; }
        public Builder beta(float v) { this.beta = v; return this; }
        public Builder lambda(float v) { this.lambda = v; return this; }

        /** Sets already-prepared XIELU parameters. */
        public Builder xielu(float alphaNPrepared, float alphaPPrepared, float beta, float eps) {
            this.xieluAlphaN = alphaNPrepared;
            this.xieluAlphaP = alphaPPrepared;
            this.xieluBeta = beta;
            this.xieluEps = eps;
            return this;
        }

        /** Sets XIELU parameters from the raw GGUF values, preparing them as {@code ggml_xielu} does. */
        public Builder xieluRaw(float alphaN, float alphaP, float beta, float eps) {
            return xielu(beta + softplus(alphaN), softplus(alphaP), beta, eps);
        }

        public Builder scaleBias(float scale, float bias) { this.scale = scale; this.bias = bias; return this; }

        public Builder clamp(float lo, float hi) { this.clampLo = lo; this.clampHi = hi; return this; }

        /** {@code cap * tanh(x * scale / cap)}; a cap of 0 disables. */
        public Builder softcap(float cap, float scale) { this.softcap = cap; this.softcapScale = scale; return this; }

        /**
         * @throws IllegalArgumentException for a NaN parameter, a negative or zero limit, a non-positive SITU
         *                                  beta, a negative lambda, an inverted clamp or a negative softcap
         */
        public ActParams build() {
            float[] all = {limit, alpha, beta, lambda, xieluAlphaN, xieluAlphaP, xieluBeta, xieluEps, scale, bias,
                    clampLo, clampHi, softcap, softcapScale};
            for (float f : all) {
                if (Float.isNaN(f)) {
                    throw new IllegalArgumentException("activation parameter is NaN: " + new ActParams(this));
                }
            }
            if (!(limit > 0f)) {
                throw new IllegalArgumentException("GLU limit must be positive, got " + limit);
            }
            if (!(beta > 0f)) {
                throw new IllegalArgumentException("SITU beta must be positive, got " + beta);
            }
            if (lambda < 0f) {
                throw new IllegalArgumentException("SITU lambda must be >= 0, got " + lambda);
            }
            if (clampLo > clampHi) {
                throw new IllegalArgumentException("clamp range is inverted: [" + clampLo + ", " + clampHi + "]");
            }
            if (softcap < 0f) {
                throw new IllegalArgumentException("softcap must be >= 0, got " + softcap);
            }
            return new ActParams(this);
        }
    }
}

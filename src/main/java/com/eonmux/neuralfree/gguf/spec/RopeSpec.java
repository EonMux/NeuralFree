package com.eonmux.neuralfree.gguf.spec;

import com.eonmux.neuralfree.gguf.arch.RopeType;
import com.eonmux.neuralfree.gguf.hparams.MropeTextPositions;
import com.eonmux.neuralfree.gguf.ops.rope.RopeMath;

import java.util.Arrays;
import java.util.Objects;

/**
 * Rotary embedding of one layer, with every parameter of ggml {@code ggml_rope_multi} / {@code ggml_rope_ext}
 * already resolved (RopeResolver): {@code theta_scale = freqBase^(-2/nDims)}, YaRN {@code extFactor},
 * {@code attnFactor} and the correction dims of {@code ggml_rope_yarn_corr_dims} (ggml/src/ggml.c:4496-4504).
 *
 * <p>The rotation covers {@code nDims} dims starting at {@code nOffs} within each head ({@code nOffs > 0} for
 * {@code [nope | rope]} heads such as deepseek4). Multi-section types ({@link RopeMath#multiSection}: MROPE, IMROPE
 * and VISION) take four positions per token, laid out {@code pos[t + k * nTokens]}; text tokens get them from
 * {@link #textPositions()}. Immutable.
 */
public final class RopeSpec {

    private final RopeType type;
    private final int nDims;
    private final int nOffs;
    private final float freqBase;
    private final float freqScale;
    private final float thetaScale;
    private final float extFactor;
    private final float attnFactor;
    private final float betaFast;
    private final float betaSlow;
    private final float corrLo;
    private final float corrHi;
    private final int nCtxOrig;
    private final int[] sections;
    private final TensorRef freqFactors;
    private final MropeTextPositions textPositions;

    private RopeSpec(Builder b) {
        this.type = b.type;
        this.nDims = b.nDims;
        this.nOffs = b.nOffs;
        this.freqBase = b.freqBase;
        this.freqScale = b.freqScale;
        this.thetaScale = b.resolvedThetaScale();
        this.extFactor = b.extFactor;
        this.attnFactor = b.attnFactor;
        this.betaFast = b.betaFast;
        this.betaSlow = b.betaSlow;
        this.corrLo = b.corrLo;
        this.corrHi = b.corrHi;
        this.nCtxOrig = b.nCtxOrig;
        this.sections = b.sections.clone();
        this.freqFactors = b.freqFactors;
        this.textPositions = b.textPositions;
    }

    public static Builder builder(RopeType type) {
        return new Builder(type);
    }

    public RopeType type() { return type; }
    /** Rotated dims per head ({@code n_rot}). */
    public int nDims() { return nDims; }
    /** First rotated dim within the head. */
    public int nOffs() { return nOffs; }
    public float freqBase() { return freqBase; }
    public float freqScale() { return freqScale; }
    /** {@code freqBase^(-2/nDims)}. */
    public float thetaScale() { return thetaScale; }
    public float extFactor() { return extFactor; }
    public float attnFactor() { return attnFactor; }
    public float betaFast() { return betaFast; }
    public float betaSlow() { return betaSlow; }
    /** YaRN correction start dim ({@code dims[0]}). */
    public float corrLo() { return corrLo; }
    /** YaRN correction end dim ({@code dims[1]}). */
    public float corrHi() { return corrHi; }
    public int nCtxOrig() { return nCtxOrig; }
    /** Multi-section sizes (copy of 4 values; zeros for single-section types). */
    public int[] sections() { return sections.clone(); }
    /** Per-dim frequency divisors ({@code rope_freqs}, or the long/short factors chosen by context); null when absent. */
    public TensorRef freqFactors() { return freqFactors; }
    /** Section positions of text tokens for multi-section types. */
    public MropeTextPositions textPositions() { return textPositions; }

    /** Positions per token ({@link RopeMath#positionsPerToken}): 4 for multi-section types, else 1. */
    public int positionsPerToken() {
        return RopeMath.positionsPerToken(type);
    }

    @Override
    public String toString() {
        return "RopeSpec{" + type + ", nDims=" + nDims + ", nOffs=" + nOffs + ", base=" + freqBase + ", scale=" + freqScale
                + ", ext=" + extFactor + ", attn=" + attnFactor + ", corr=[" + corrLo + ", " + corrHi + "], sections="
                + Arrays.toString(sections) + ", factors=" + (freqFactors != null ? freqFactors.name() : "none") + '}';
    }

    public static final class Builder {
        private final RopeType type;
        private int nDims;
        private int nOffs;
        private float freqBase = 10000f;
        private float freqScale = 1f;
        private float thetaScale = Float.NaN;
        private float extFactor;
        private float attnFactor = 1f;
        private float betaFast = 32f;
        private float betaSlow = 1f;
        private float corrLo;
        private float corrHi;
        private int nCtxOrig;
        private int[] sections = new int[MropeTextPositions.SECTIONS];
        private TensorRef freqFactors;
        private MropeTextPositions textPositions = MropeTextPositions.PPP0;

        private Builder(RopeType type) {
            this.type = Objects.requireNonNull(type, "type");
        }

        public Builder nDims(int v) { this.nDims = v; return this; }
        public Builder nOffs(int v) { this.nOffs = v; return this; }
        public Builder freqBase(float v) { this.freqBase = v; return this; }
        public Builder freqScale(float v) { this.freqScale = v; return this; }
        public Builder thetaScale(float v) { this.thetaScale = v; return this; }
        public Builder yarn(float extFactor, float attnFactor, float betaFast, float betaSlow) {
            this.extFactor = extFactor;
            this.attnFactor = attnFactor;
            this.betaFast = betaFast;
            this.betaSlow = betaSlow;
            return this;
        }
        public Builder corrDims(float lo, float hi) { this.corrLo = lo; this.corrHi = hi; return this; }
        public Builder nCtxOrig(int v) { this.nCtxOrig = v; return this; }
        public Builder sections(int[] v) {
            Objects.requireNonNull(v, "sections");
            if (v.length != MropeTextPositions.SECTIONS) {
                throw new IllegalArgumentException("rope sections need " + MropeTextPositions.SECTIONS
                        + " values, got " + v.length);
            }
            this.sections = v.clone();
            return this;
        }
        public Builder freqFactors(TensorRef v) { this.freqFactors = v; return this; }
        public Builder textPositions(MropeTextPositions v) {
            this.textPositions = Objects.requireNonNull(v, "textPositions");
            return this;
        }

        /**
         * Builds the spec; an unset {@code thetaScale} becomes {@code powf(freqBase, -2.0f / nDims)}.
         *
         * @throws IllegalArgumentException for NONE, odd or non-positive dims, invalid frequencies, or sections that
         *                                  do not match the type: a multi-section type needs sections covering
         *                                  between 1 and {@link RopeMath#pairCount} pairs ({@code nDims} for VISION,
         *                                  {@code nDims / 2} otherwise)
         */
        public RopeSpec build() {
            if (type == RopeType.NONE) {
                throw new IllegalArgumentException("a RopeSpec needs a rotary type; NoPE layers carry no RopeSpec");
            }
            if (nDims <= 0 || nDims % 2 != 0 || nOffs < 0) {
                throw new IllegalArgumentException("rope nDims must be even and positive and nOffs >= 0, got "
                        + nDims + "/" + nOffs);
            }
            if (!(freqBase > 0f) || !(freqScale > 0f) || !(attnFactor > 0f)) {
                throw new IllegalArgumentException("rope freqBase, freqScale and attnFactor must be positive: "
                        + freqBase + ", " + freqScale + ", " + attnFactor);
            }
            float theta = resolvedThetaScale();
            if (!(theta > 0f) || Float.isInfinite(theta)) {
                throw new IllegalArgumentException("rope thetaScale must be a positive finite value, got " + theta);
            }
            if (extFactor != 0f && nCtxOrig <= 0) {
                throw new IllegalArgumentException("YaRN rope (extFactor " + extFactor + ") needs nCtxOrig > 0");
            }
            int sum = 0;
            for (int s : sections) {
                if (s < 0) {
                    throw new IllegalArgumentException("rope sections must be >= 0: " + Arrays.toString(sections));
                }
                sum += s;
            }
            int pairs = RopeMath.pairCount(type, nDims);
            if (RopeMath.multiSection(type) && (sum <= 0 || sum > pairs)) {
                throw new IllegalArgumentException(type + " rope sections " + Arrays.toString(sections)
                        + " must cover between 1 and " + pairs + " pairs");
            }
            if (freqFactors != null && freqFactors.numElements() < nDims / 2) {
                throw new IllegalArgumentException("rope freq factors " + freqFactors + " hold fewer than nDims/2 = "
                        + nDims / 2);
            }
            return new RopeSpec(this);
        }

        /** The set thetaScale, or {@code powf(freqBase, -2.0f / nDims)} in float arithmetic (ggml-cpu/ops.cpp:6455). */
        private float resolvedThetaScale() {
            if (!Float.isNaN(thetaScale)) {
                return thetaScale;
            }
            float exponent = -2.0f / nDims;
            return (float) Math.pow(freqBase, exponent);
        }
    }
}

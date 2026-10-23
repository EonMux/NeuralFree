package com.eonmux.neuralfree.gguf.spec;

import com.eonmux.neuralfree.gguf.arch.ResidualKind;

import java.util.Objects;

/**
 * How the mixer and FFN branches join the residual stream of one layer:
 * <ul>
 *   <li>STANDARD {@code x += branch}; SCALED {@code x += branch * residualScale};
 *       NORMED_INPUT_SCALED {@code x = branch + residualScale * attn_norm(x)}; DEPTH_SCALED
 *       {@code x = x * streamScale(ffn) + branch * branchScale(ffn)} per sub-block; PARALLEL: mixer and FFN read the
 *       same input and both branches are added after the FFN; HYPER_CONNECTIONS: the residual is consumed by the
 *       hyper-connection extra over a stream of {@link #streamWidth()} = {@code hcMult * n_embd} values.</li>
 *   <li>Norm-preserving residuals (muse-spark): {@code x = x * alpha + branch * beta} per sub-block, using
 *       {@code attn_resid_alpha/beta} and {@code ffn_resid_alpha/beta}; consumed by the {@code resid_alpha_beta}
 *       extra.</li>
 *   <li>deci: {@code n_ff == 0} sets {@link #discardLayerOutput()}: the layer passes its input through unchanged
 *       (models/deci.cpp:147-148); {@code n_head == 0} clears {@link #mixerResidual()} (models/deci.cpp:152).</li>
 *   <li>{@link #embeddingSkip()}: {@code embeddingSkipScale * embedding} is added to the layer output.</li>
 *   <li>{@link #blockOutputScale()}: the layer output is multiplied by this (RWKV6 halves the stream every
 *       {@code rescale_every_n_layers} layers); 1 leaves it unchanged.</li>
 * </ul>
 * Immutable.
 */
public final class ResidualSpec {

    private final ResidualKind kind;
    private final float residualScale;
    private final boolean mixerResidual;
    private final boolean discardLayerOutput;
    private final TensorRef attnAlpha;
    private final TensorRef attnBeta;
    private final TensorRef ffnAlpha;
    private final TensorRef ffnBeta;
    private final boolean embeddingSkip;
    private final float embeddingSkipScale;
    private final boolean ffnUsesAttnNorm2;
    private final int streamWidth;
    private final int hcMult;
    private final boolean hcPerSubBlock;
    private final float blockOutputScale;
    private final float mixerBranchScale;
    private final float mixerStreamScale;
    private final float ffnBranchScale;
    private final float ffnStreamScale;

    private ResidualSpec(Builder b) {
        this.kind = b.kind;
        this.residualScale = b.residualScale;
        this.mixerResidual = b.mixerResidual;
        this.discardLayerOutput = b.discardLayerOutput;
        this.attnAlpha = b.attnAlpha;
        this.attnBeta = b.attnBeta;
        this.ffnAlpha = b.ffnAlpha;
        this.ffnBeta = b.ffnBeta;
        this.embeddingSkip = b.embeddingSkip;
        this.embeddingSkipScale = b.embeddingSkipScale;
        this.ffnUsesAttnNorm2 = b.ffnUsesAttnNorm2;
        this.streamWidth = b.streamWidth();
        this.hcMult = b.hcMult;
        this.hcPerSubBlock = b.hcPerSubBlock;
        this.blockOutputScale = b.blockOutputScale;
        this.mixerBranchScale = b.mixerBranchScale;
        this.mixerStreamScale = b.mixerStreamScale;
        this.ffnBranchScale = b.ffnBranchScale;
        this.ffnStreamScale = b.ffnStreamScale;
    }

    /**
     * @param kind  residual kind
     * @param nEmbd model width; the stream width is {@code nEmbd}, or {@code hcMult * nEmbd} with hyper-connections
     */
    public static Builder builder(ResidualKind kind, int nEmbd) {
        return new Builder(kind, nEmbd);
    }

    public ResidualKind kind() { return kind; }
    /** Scale of SCALED / NORMED_INPUT_SCALED residuals; 1 otherwise. */
    public float residualScale() { return residualScale; }
    /** False when the mixer branch is not added to the stream (deci {@code n_head == 0}). */
    public boolean mixerResidual() { return mixerResidual; }
    public boolean discardLayerOutput() { return discardLayerOutput; }
    /** Stream multiplier of the mixer sub-block ({@code attn_resid_alpha}); null when absent. */
    public TensorRef attnAlpha() { return attnAlpha; }
    /** Branch multiplier of the mixer sub-block ({@code attn_resid_beta}); null when absent. */
    public TensorRef attnBeta() { return attnBeta; }
    public TensorRef ffnAlpha() { return ffnAlpha; }
    public TensorRef ffnBeta() { return ffnBeta; }
    public boolean embeddingSkip() { return embeddingSkip; }
    public float embeddingSkipScale() { return embeddingSkipScale; }
    /**
     * The block has the second input norm {@code attn_norm_2} of the layer input. RWKV6/7 feed it to the FFN;
     * falcon-40b's parallel block feeds it to the attention and {@code attn_norm} to the FFN (models/falcon.cpp:70-86, 125).
     */
    public boolean ffnUsesAttnNorm2() { return ffnUsesAttnNorm2; }
    /** Values per token in the residual stream: {@code n_embd}, or {@code hcMult * n_embd}. */
    public int streamWidth() { return streamWidth; }
    /** Hyper-connection stream count; 1 without hyper-connections. */
    public int hcMult() { return hcMult; }
    /** Hyper-connection pre/post maps per sub-block (mixer and FFN) instead of per layer. */
    public boolean hcPerSubBlock() { return hcPerSubBlock; }
    public float blockOutputScale() { return blockOutputScale; }
    /** DEPTH_SCALED multiplier of the FFN ({@code ffn}) or mixer branch; 1 for every other kind. */
    public float branchScale(boolean ffn) { return ffn ? ffnBranchScale : mixerBranchScale; }
    /** DEPTH_SCALED multiplier of the stream before the FFN ({@code ffn}) or mixer branch is added; 1 otherwise. */
    public float streamScale(boolean ffn) { return ffn ? ffnStreamScale : mixerStreamScale; }

    /** True when muse-spark style alpha/beta residuals are present. */
    public boolean alphaBeta() { return attnAlpha != null || ffnAlpha != null; }

    @Override
    public String toString() {
        return "ResidualSpec{" + kind + ", scale=" + residualScale + ", mixerResidual=" + mixerResidual + ", discard="
                + discardLayerOutput + ", alphaBeta=" + alphaBeta() + ", embdSkip=" + embeddingSkip + ", stream="
                + streamWidth + ", hc=" + hcMult + (hcPerSubBlock ? "/sub" : "") + ", outScale=" + blockOutputScale
                + (kind == ResidualKind.DEPTH_SCALED ? ", depth=" + mixerStreamScale + "/" + mixerBranchScale + ", "
                + ffnStreamScale + "/" + ffnBranchScale : "") + '}';
    }

    public static final class Builder {
        private final ResidualKind kind;
        private final int nEmbd;
        private float residualScale = 1.0f;
        private boolean mixerResidual = true;
        private boolean discardLayerOutput;
        private TensorRef attnAlpha;
        private TensorRef attnBeta;
        private TensorRef ffnAlpha;
        private TensorRef ffnBeta;
        private boolean embeddingSkip;
        private float embeddingSkipScale = 1.0f;
        private boolean ffnUsesAttnNorm2;
        private int hcMult = 1;
        private boolean hcPerSubBlock;
        private float blockOutputScale = 1.0f;
        private float mixerBranchScale = 1.0f;
        private float mixerStreamScale = 1.0f;
        private float ffnBranchScale = 1.0f;
        private float ffnStreamScale = 1.0f;

        private Builder(ResidualKind kind, int nEmbd) {
            this.kind = Objects.requireNonNull(kind, "kind");
            this.nEmbd = nEmbd;
        }

        public Builder residualScale(float v) { this.residualScale = v; return this; }
        public Builder mixerResidual(boolean v) { this.mixerResidual = v; return this; }
        public Builder discardLayerOutput(boolean v) { this.discardLayerOutput = v; return this; }
        public Builder attnAlphaBeta(TensorRef alpha, TensorRef beta) {
            this.attnAlpha = alpha;
            this.attnBeta = beta;
            return this;
        }
        public Builder ffnAlphaBeta(TensorRef alpha, TensorRef beta) {
            this.ffnAlpha = alpha;
            this.ffnBeta = beta;
            return this;
        }
        public Builder embeddingSkip(boolean on, float scale) {
            this.embeddingSkip = on;
            this.embeddingSkipScale = scale;
            return this;
        }
        public Builder ffnUsesAttnNorm2(boolean v) { this.ffnUsesAttnNorm2 = v; return this; }
        public Builder hyperConnections(int mult, boolean perSubBlock) {
            this.hcMult = mult;
            this.hcPerSubBlock = perSubBlock;
            return this;
        }
        public Builder blockOutputScale(float v) { this.blockOutputScale = v; return this; }
        /** DEPTH_SCALED branch and stream multipliers of the mixer and the FFN sub-block. */
        public Builder depthScales(float mixerBranch, float mixerStream, float ffnBranch, float ffnStream) {
            this.mixerBranchScale = mixerBranch;
            this.mixerStreamScale = mixerStream;
            this.ffnBranchScale = ffnBranch;
            this.ffnStreamScale = ffnStream;
            return this;
        }

        /** @throws IllegalArgumentException on inconsistent hyper-connection, scale or alpha/beta settings */
        public ResidualSpec build() {
            if (nEmbd <= 0) {
                throw new IllegalArgumentException("residual stream needs n_embd > 0, got " + nEmbd);
            }
            boolean hc = kind == ResidualKind.HYPER_CONNECTIONS;
            if (hc != (hcMult > 1) || hcMult < 1) {
                throw new IllegalArgumentException("hyper-connection multiplier " + hcMult
                        + " does not match residual kind " + kind);
            }
            if (hcPerSubBlock && !hc) {
                throw new IllegalArgumentException("hcPerSubBlock without hyper-connections");
            }
            boolean scaled = kind == ResidualKind.SCALED || kind == ResidualKind.NORMED_INPUT_SCALED;
            if (!scaled && residualScale != 1.0f) {
                throw new IllegalArgumentException(kind + " residual takes no scale, got " + residualScale);
            }
            if (!isFinite(residualScale) || !isFinite(embeddingSkipScale) || !(blockOutputScale > 0f)
                    || Float.isInfinite(blockOutputScale)) {
                throw new IllegalArgumentException("residual scales must be finite and the block output scale positive");
            }
            float[] depth = {mixerBranchScale, mixerStreamScale, ffnBranchScale, ffnStreamScale};
            for (float d : depth) {
                if (kind == ResidualKind.DEPTH_SCALED ? !(d > 0f) || Float.isInfinite(d) : d != 1.0f) {
                    throw new IllegalArgumentException(kind == ResidualKind.DEPTH_SCALED
                            ? "depth-scaled residual multipliers must be positive and finite, got " + d
                            : kind + " residual takes no depth scales, got " + d);
                }
            }
            if ((attnAlpha == null) != (attnBeta == null) || (ffnAlpha == null) != (ffnBeta == null)) {
                throw new IllegalArgumentException("residual alpha and beta tensors come in pairs");
            }
            return new ResidualSpec(this);
        }

        private int streamWidth() {
            return hcMult * nEmbd;
        }

        private static boolean isFinite(float v) {
            return !Float.isNaN(v) && !Float.isInfinite(v);
        }
    }
}

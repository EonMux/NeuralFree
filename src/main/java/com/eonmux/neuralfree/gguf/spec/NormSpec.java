package com.eonmux.neuralfree.gguf.spec;

import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.arch.PostNormRole;
import com.eonmux.neuralfree.gguf.arch.QkNormKind;

import java.util.Objects;

/**
 * Normalizations of one layer and where the layer executor applies them.
 *
 * <ul>
 *   <li>{@link #attnNorm()}: {@code attn_norm} before the mixer; {@link #attnNorm2()}: the second input norm
 *       {@code attn_norm_2} ({@code ResidualSpec.ffnUsesAttnNorm2}). RWKV6/7 feed it to the channel mix; falcon-40b
 *       feeds it to the attention and {@code attn_norm} to the FFN (models/falcon.cpp:70-86, 125).</li>
 *   <li>{@link PostNormRole#SANDWICH}: {@code post_attention_norm} on the mixer output and {@code post_ffw_norm} on
 *       the FFN output. {@link PostNormRole#PRE_FFN}: {@code post_attention_norm} is the FFN input norm (chosen only
 *       when {@code attn_norm} and {@code post_attention_norm} exist and {@code ffn_norm} is absent).
 *       {@link PostNormRole#POST_ONLY}: no pre-norms; post norms on both branches (olmo2, exaone4). Post norms use
 *       {@link #postEps()}; muse-spark's post-attention norm has no weight ({@link #postAttnWeightless()}) and its
 *       post-FFN norm normalizes the weighted branch ({@link #postFfnWeightFirst()}, muse-spark.cpp:157-158,
 *       203-205).</li>
 *   <li>{@link #sharedAttnNormForFfn()}: the FFN reuses the attn_norm output (plamo).</li>
 *   <li>{@link #groups()} {@code > 1}: {@code attn_norm} and {@code ffn_norm} are RMS norms over that many equal
 *       slices of the row, then multiplied by the full-width weight (k2-horizon, models/k2-horizon.cpp:124-135).</li>
 *   <li>{@link #ffnAda()}: the adaptive RMS norm of Voxtral realtime (mistral3). After {@code ffn_norm} the dense FFN
 *       input becomes {@code cur + cur * ada} with
 *       {@code ada = ffn_norm_ada_up · gelu_erf(ffn_norm_ada_down · t_cond)} and {@code t_cond} the sinusoidal
 *       embedding of {@link #ffnAdaDelay()} (models/mistral3.cpp:99-121, 152-159, 227-236).</li>
 *   <li>Sub-norms ({@code attn_sub_norm}, {@code ffn_sub_norm}, bitnet) apply inside the branch before its output
 *       projection; {@link #attnOutputNorm()} (grok) after it; {@link #layerOutputNorm()} on the layer output.</li>
 * </ul>
 * Norm biases are taken from the tensors when present. Immutable.
 */
public final class NormSpec {

    private final NormKind kind;
    private final float eps;
    private final boolean weightless;
    private final boolean attnNorm;
    private final boolean attnNorm2;
    private final boolean ffnNorm;
    private final PostNormRole postRole;
    private final boolean postAttnNorm;
    private final boolean postFfnNorm;
    private final boolean attnSubNorm;
    private final boolean ffnSubNorm;
    private final boolean attnOutputNorm;
    private final boolean layerOutputNorm;
    private final TensorRef layerOutputScale;
    private final boolean sharedAttnNormForFfn;
    private final QkNormKind qkNorm;
    private final float postEps;
    private final boolean postAttnWeightless;
    private final boolean postFfnWeightFirst;
    private final int groups;
    private final TensorRef ffnAdaDown;
    private final TensorRef ffnAdaUp;
    private final int ffnAdaDelay;

    private NormSpec(Builder b) {
        this.kind = b.kind;
        this.eps = b.eps;
        this.weightless = b.weightless;
        this.attnNorm = b.attnNorm;
        this.attnNorm2 = b.attnNorm2;
        this.ffnNorm = b.ffnNorm;
        this.postRole = b.postRole;
        this.postAttnNorm = b.postAttnNorm;
        this.postFfnNorm = b.postFfnNorm;
        this.attnSubNorm = b.attnSubNorm;
        this.ffnSubNorm = b.ffnSubNorm;
        this.attnOutputNorm = b.attnOutputNorm;
        this.layerOutputNorm = b.layerOutputNorm;
        this.layerOutputScale = b.layerOutputScale;
        this.sharedAttnNormForFfn = b.sharedAttnNormForFfn;
        this.qkNorm = b.qkNorm;
        this.postEps = b.postEps;
        this.postAttnWeightless = b.postAttnWeightless;
        this.postFfnWeightFirst = b.postFfnWeightFirst;
        this.groups = b.groups;
        this.ffnAdaDown = b.ffnAdaDown;
        this.ffnAdaUp = b.ffnAdaUp;
        this.ffnAdaDelay = b.ffnAdaDelay;
    }

    public static Builder builder(NormKind kind, float eps) {
        return new Builder(kind, eps);
    }

    public NormKind kind() { return kind; }
    public float eps() { return eps; }
    /** Norms carry no weight tensor (hrm_text stack norms); applied as plain normalization. */
    public boolean weightless() { return weightless; }
    public boolean attnNorm() { return attnNorm; }
    public boolean attnNorm2() { return attnNorm2; }
    public boolean ffnNorm() { return ffnNorm; }
    public PostNormRole postRole() { return postRole; }
    /** {@code post_attention_norm} present (its role is {@link #postRole()}). */
    public boolean postAttnNorm() { return postAttnNorm; }
    /** {@code post_ffw_norm} present. */
    public boolean postFfnNorm() { return postFfnNorm; }
    public boolean attnSubNorm() { return attnSubNorm; }
    public boolean ffnSubNorm() { return ffnSubNorm; }
    public boolean attnOutputNorm() { return attnOutputNorm; }
    public boolean layerOutputNorm() { return layerOutputNorm; }
    /** gemma4 per-layer output scalar ({@code layer_output_scale}); null when absent. */
    public TensorRef layerOutputScale() { return layerOutputScale; }
    public boolean sharedAttnNormForFfn() { return sharedAttnNormForFfn; }
    /** Q/K normalization; the same value as the layer's {@code AttentionSpec.qkNorm()}. */
    public QkNormKind qkNorm() { return qkNorm; }

    /** Epsilon of the post-attention and post-FFN norms: {@code f_post_norm_eps} when set, else {@link #eps()}. */
    public float postEps() { return postEps > 0f ? postEps : eps; }
    /** The post-attention norm is a plain normalization without a weight tensor (muse-spark.cpp:157-158). */
    public boolean postAttnWeightless() { return postAttnWeightless; }
    /** The post-FFN norm is {@code norm(branch * weight)} rather than {@code norm(branch) * weight} (muse-spark.cpp:203-205). */
    public boolean postFfnWeightFirst() { return postFfnWeightFirst; }

    /** Equal row slices each normalized on their own by {@code attn_norm} and {@code ffn_norm}; 1 for a plain row norm. */
    public int groups() { return groups; }
    /** True when the FFN input norm is conditioned by {@link #ffnAdaDown()} and {@link #ffnAdaUp()}. */
    public boolean ffnAda() { return ffnAdaDown != null; }
    /** {@code ffn_norm_ada_down} {@code [n_embd, n_ada]}: {@code t_cond} to the hidden width; null without ada. */
    public TensorRef ffnAdaDown() { return ffnAdaDown; }
    /** {@code ffn_norm_ada_up} {@code [n_ada, n_embd]}: the hidden width back to {@code n_embd}; null without ada. */
    public TensorRef ffnAdaUp() { return ffnAdaUp; }
    /** Output delay in tokens whose sinusoidal embedding is {@code t_cond} ({@code n_tokens_delay}). */
    public int ffnAdaDelay() { return ffnAdaDelay; }
    /** True when the FFN input norm is {@code post_attention_norm}. */
    public boolean ffnInputIsPostAttnNorm() { return postRole == PostNormRole.PRE_FFN; }

    @Override
    public String toString() {
        return "NormSpec{" + kind + ", eps=" + eps + ", attn=" + attnNorm + ", attn2=" + attnNorm2 + ", ffn=" + ffnNorm
                + ", post=" + postRole + ", qk=" + qkNorm + ", shared=" + sharedAttnNormForFfn
                + (groups > 1 ? ", groups=" + groups : "")
                + (ffnAdaDown != null ? ", ada=" + ffnAdaDown.name() + "/delay " + ffnAdaDelay : "") + '}';
    }

    public static final class Builder {
        private final NormKind kind;
        private final float eps;
        private boolean weightless;
        private boolean attnNorm = true;
        private boolean attnNorm2;
        private boolean ffnNorm = true;
        private PostNormRole postRole = PostNormRole.NONE;
        private boolean postAttnNorm;
        private boolean postFfnNorm;
        private boolean attnSubNorm;
        private boolean ffnSubNorm;
        private boolean attnOutputNorm;
        private boolean layerOutputNorm;
        private TensorRef layerOutputScale;
        private boolean sharedAttnNormForFfn;
        private QkNormKind qkNorm = QkNormKind.NONE;
        private float postEps;
        private boolean postAttnWeightless;
        private boolean postFfnWeightFirst;
        private int groups = 1;
        private TensorRef ffnAdaDown;
        private TensorRef ffnAdaUp;
        private int ffnAdaDelay;

        private Builder(NormKind kind, float eps) {
            this.kind = Objects.requireNonNull(kind, "kind");
            this.eps = eps;
        }

        public Builder weightless(boolean v) { this.weightless = v; return this; }
        public Builder attnNorm(boolean v) { this.attnNorm = v; return this; }
        public Builder attnNorm2(boolean v) { this.attnNorm2 = v; return this; }
        public Builder ffnNorm(boolean v) { this.ffnNorm = v; return this; }
        public Builder postRole(PostNormRole v) { this.postRole = Objects.requireNonNull(v, "postRole"); return this; }
        public Builder postAttnNorm(boolean v) { this.postAttnNorm = v; return this; }
        public Builder postFfnNorm(boolean v) { this.postFfnNorm = v; return this; }
        public Builder attnSubNorm(boolean v) { this.attnSubNorm = v; return this; }
        public Builder ffnSubNorm(boolean v) { this.ffnSubNorm = v; return this; }
        public Builder attnOutputNorm(boolean v) { this.attnOutputNorm = v; return this; }
        public Builder layerOutputNorm(boolean v) { this.layerOutputNorm = v; return this; }
        public Builder layerOutputScale(TensorRef v) { this.layerOutputScale = v; return this; }
        public Builder sharedAttnNormForFfn(boolean v) { this.sharedAttnNormForFfn = v; return this; }
        public Builder qkNorm(QkNormKind v) { this.qkNorm = Objects.requireNonNull(v, "qkNorm"); return this; }
        /** Post-norm epsilon; 0 uses the block epsilon. */
        public Builder postEps(float v) { this.postEps = v; return this; }
        public Builder postAttnWeightless(boolean v) { this.postAttnWeightless = v; return this; }
        public Builder postFfnWeightFirst(boolean v) { this.postFfnWeightFirst = v; return this; }
        /** Grouped block norms: {@code attn_norm} / {@code ffn_norm} over {@code v} equal slices; 1 disables. */
        public Builder groups(int v) { this.groups = v; return this; }
        /** Adaptive FFN input norm: both projections and the {@code t_cond} delay, or null projections for none. */
        public Builder ffnAda(TensorRef down, TensorRef up, int delay) {
            this.ffnAdaDown = down;
            this.ffnAdaUp = up;
            this.ffnAdaDelay = delay;
            return this;
        }

        /** @throws IllegalArgumentException when the role and the present norms disagree, or eps is invalid */
        public NormSpec build() {
            if (!(eps > 0f) || Float.isInfinite(eps)) {
                throw new IllegalArgumentException("norm eps must be a positive finite value, got " + eps);
            }
            if (!kind.isRowNorm()) {
                throw new IllegalArgumentException("layer norm kind must be a row norm, got " + kind);
            }
            switch (postRole) {
                case PRE_FFN:
                    if (!attnNorm || !postAttnNorm || ffnNorm) {
                        throw new IllegalArgumentException("PRE_FFN needs attn_norm, post_attention_norm and no ffn_norm");
                    }
                    break;
                case POST_ONLY:
                    if (attnNorm || ffnNorm) {
                        throw new IllegalArgumentException("POST_ONLY layers have no attn_norm or ffn_norm");
                    }
                    break;
                case SANDWICH:
                    if (!postAttnNorm && !postFfnNorm) {
                        throw new IllegalArgumentException("SANDWICH needs post_attention_norm or post_ffw_norm");
                    }
                    break;
                default:
                    if (postAttnNorm) {
                        throw new IllegalArgumentException("post_attention_norm is present but the post-norm role is NONE");
                    }
                    break;
            }
            if (sharedAttnNormForFfn && (!attnNorm || ffnNorm)) {
                throw new IllegalArgumentException("a shared attention norm for the FFN needs attn_norm and no ffn_norm");
            }
            if (attnNorm2 && !attnNorm) {
                throw new IllegalArgumentException("attn_norm_2 without attn_norm");
            }
            if (!(postEps >= 0f) || Float.isInfinite(postEps)) {
                throw new IllegalArgumentException("post-norm eps must be 0 or a positive finite value, got " + postEps);
            }
            boolean branchPost = postRole == PostNormRole.SANDWICH || postRole == PostNormRole.POST_ONLY;
            if (postAttnWeightless && !(branchPost && postAttnNorm)) {
                throw new IllegalArgumentException("a weightless post-attention norm needs a branch post-attention norm");
            }
            if (postFfnWeightFirst && (!(branchPost && postFfnNorm) || weightless)) {
                throw new IllegalArgumentException("a weight-first post-FFN norm needs a weighted branch post-FFN norm");
            }
            if (groups < 1 || (groups > 1 && (kind != NormKind.RMS || weightless))) {
                throw new IllegalArgumentException("grouped block norms need a weighted RMS norm and at least one group,"
                        + " got " + groups + " groups of " + kind + (weightless ? " (weightless)" : ""));
            }
            if ((ffnAdaDown == null) != (ffnAdaUp == null)) {
                throw new IllegalArgumentException("the adaptive FFN norm needs both ffn_norm_ada_down and"
                        + " ffn_norm_ada_up");
            }
            if (ffnAdaDown != null) {
                if (!ffnNorm || weightless || kind != NormKind.RMS || ffnAdaDelay < 0) {
                    throw new IllegalArgumentException("the adaptive FFN norm conditions a weighted RMS ffn_norm with a"
                            + " non-negative delay, got ffn_norm=" + ffnNorm + ", " + kind + ", delay " + ffnAdaDelay);
                }
                if (ffnAdaDown.rank() != 2 || ffnAdaUp.rank() != 2 || ffnAdaDown.ne(1) != ffnAdaUp.ne(0)
                        || ffnAdaDown.ne(0) != ffnAdaUp.ne(1)) {
                    throw new IllegalArgumentException("ffn_norm_ada_down " + ffnAdaDown + " and ffn_norm_ada_up "
                            + ffnAdaUp + " are not an [n_embd, n_ada] / [n_ada, n_embd] pair");
                }
            } else if (ffnAdaDelay != 0) {
                throw new IllegalArgumentException("an ada delay needs the adaptive FFN norm tensors");
            }
            return new NormSpec(this);
        }
    }
}

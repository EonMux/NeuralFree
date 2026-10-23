package com.eonmux.neuralfree.gguf.spec;

/**
 * Model input: token embedding lookup and everything applied before the first layer.
 * <ol>
 *   <li>{@code x = token_embd[token]} (width {@link #nEmbd()}), zero-padded to {@link #nEmbdInp()} when the
 *       first layer reads a wider input (deepstack);</li>
 *   <li>{@code + token_types[type]} and {@code + position_embd[pos]} when present (BERT family, GPT-2);</li>
 *   <li>{@code * embdScale} (gemma {@code sqrt(n_embd)}, granite {@code embedding_scale}); 1 leaves it;</li>
 *   <li>{@code token_embd_norm} (with bias when present) when present;</li>
 *   <li>a weightless RMS norm with {@code f_norm_rms_eps} when {@link #inputRmsNorm()} (talkie.cpp:50-51,
 *       muse-spark.cpp:86-87, muse-glimmer.cpp:69).</li>
 * </ol>
 * Per-layer embeddings (gemma3n PLE: {@code per_layer_token_embd}, {@code per_layer_model_proj},
 * {@code per_layer_proj_norm}; qwen4exp n-gram PLE) are computed once per token here and consumed by the
 * {@code ple} / {@code ngram_ple} layer extras. Immutable.
 */
public final class EmbeddingSpec {

    private final TensorRef tokenEmbd;
    private final TensorRef posEmbd;
    private final TensorRef tokenTypes;
    private final TensorRef tokenEmbdNorm;
    private final float embdScale;
    private final int nEmbd;
    private final int nEmbdInp;
    private final TensorRef perLayerTokenEmbd;
    private final TensorRef perLayerModelProj;
    private final TensorRef perLayerProjNorm;
    private final int nEmbdPerLayer;
    private final boolean ngramPle;
    private final boolean inputRmsNorm;

    private EmbeddingSpec(Builder b) {
        this.tokenEmbd = b.tokenEmbd;
        this.posEmbd = b.posEmbd;
        this.tokenTypes = b.tokenTypes;
        this.tokenEmbdNorm = b.tokenEmbdNorm;
        this.embdScale = b.embdScale;
        this.nEmbd = b.nEmbd;
        this.nEmbdInp = b.nEmbdInp;
        this.perLayerTokenEmbd = b.perLayerTokenEmbd;
        this.perLayerModelProj = b.perLayerModelProj;
        this.perLayerProjNorm = b.perLayerProjNorm;
        this.nEmbdPerLayer = b.nEmbdPerLayer;
        this.ngramPle = b.ngramPle;
        this.inputRmsNorm = b.inputRmsNorm;
    }

    /** The input rows get a weightless RMS norm with {@code f_norm_rms_eps} before the first layer. */
    public boolean inputRmsNorm() { return inputRmsNorm; }

    public static Builder builder(TensorRef tokenEmbd, int nEmbd) {
        return new Builder(tokenEmbd, nEmbd);
    }

    /** Token table; null for drafters that embed through their target model (eagle3, dflash). */
    public TensorRef tokenEmbd() { return tokenEmbd; }
    /** Absolute position table; null when absent. */
    public TensorRef posEmbd() { return posEmbd; }
    /** Token-type table; null when absent. */
    public TensorRef tokenTypes() { return tokenTypes; }
    /** Embedding norm weight; null when absent. */
    public TensorRef tokenEmbdNorm() { return tokenEmbdNorm; }
    public float embdScale() { return embdScale; }
    public int nEmbd() { return nEmbd; }
    /** Width of the first layer's input ({@code >= nEmbd}). */
    public int nEmbdInp() { return nEmbdInp; }
    public TensorRef perLayerTokenEmbd() { return perLayerTokenEmbd; }
    public TensorRef perLayerModelProj() { return perLayerModelProj; }
    public TensorRef perLayerProjNorm() { return perLayerProjNorm; }
    /** Per-layer embedding width ({@code n_embd_per_layer}); 0 without PLE. */
    public int nEmbdPerLayer() { return nEmbdPerLayer; }
    /** qwen4exp n-gram hash PLE (parameters in {@code ModelHparams.ple*}). */
    public boolean ngramPle() { return ngramPle; }

    /** True when gemma3n-style per-layer embeddings are present. */
    public boolean perLayerEmbeddings() { return perLayerTokenEmbd != null; }

    @Override
    public String toString() {
        return "EmbeddingSpec{" + (tokenEmbd != null ? tokenEmbd.name() : "target") + ", nEmbd=" + nEmbd
                + ", inp=" + nEmbdInp + ", scale=" + embdScale
                + ", pos=" + (posEmbd != null) + ", types=" + (tokenTypes != null) + ", norm=" + (tokenEmbdNorm != null)
                + ", ple=" + nEmbdPerLayer + (ngramPle ? "/ngram" : "") + '}';
    }

    public static final class Builder {
        private final TensorRef tokenEmbd;
        private final int nEmbd;
        private TensorRef posEmbd;
        private TensorRef tokenTypes;
        private TensorRef tokenEmbdNorm;
        private float embdScale = 1.0f;
        private int nEmbdInp;
        private TensorRef perLayerTokenEmbd;
        private TensorRef perLayerModelProj;
        private TensorRef perLayerProjNorm;
        private int nEmbdPerLayer;
        private boolean ngramPle;
        private boolean inputRmsNorm;

        private Builder(TensorRef tokenEmbd, int nEmbd) {
            this.tokenEmbd = tokenEmbd;
            this.nEmbd = nEmbd;
            this.nEmbdInp = nEmbd;
        }

        public Builder posEmbd(TensorRef v) { this.posEmbd = v; return this; }
        public Builder tokenTypes(TensorRef v) { this.tokenTypes = v; return this; }
        public Builder tokenEmbdNorm(TensorRef v) { this.tokenEmbdNorm = v; return this; }
        public Builder embdScale(float v) { this.embdScale = v; return this; }
        public Builder nEmbdInp(int v) { this.nEmbdInp = v; return this; }
        public Builder perLayer(TensorRef tokenEmbd, TensorRef modelProj, TensorRef projNorm, int nEmbdPerLayer) {
            this.perLayerTokenEmbd = tokenEmbd;
            this.perLayerModelProj = modelProj;
            this.perLayerProjNorm = projNorm;
            this.nEmbdPerLayer = nEmbdPerLayer;
            return this;
        }
        public Builder ngramPle(boolean v) { this.ngramPle = v; return this; }
        public Builder inputRmsNorm(boolean v) { this.inputRmsNorm = v; return this; }

        /** @throws IllegalArgumentException when widths disagree with the token table or PLE is incomplete */
        public EmbeddingSpec build() {
            if (nEmbd <= 0 || (tokenEmbd != null && tokenEmbd.cols() != nEmbd)) {
                throw new IllegalArgumentException("token embedding " + tokenEmbd + " does not have width n_embd " + nEmbd);
            }
            if (nEmbdInp < nEmbd) {
                throw new IllegalArgumentException("n_embd_inp " + nEmbdInp + " is smaller than n_embd " + nEmbd);
            }
            if (Float.isNaN(embdScale) || embdScale == 0f) {
                throw new IllegalArgumentException("embedding scale must be non-zero, got " + embdScale);
            }
            if (inputRmsNorm && tokenEmbdNorm != null) {
                throw new IllegalArgumentException("the weightless input norm and token_embd_norm are exclusive");
            }
            if ((perLayerTokenEmbd != null || ngramPle) && nEmbdPerLayer <= 0) {
                throw new IllegalArgumentException("per-layer embeddings need n_embd_per_layer > 0");
            }
            return new EmbeddingSpec(this);
        }
    }
}

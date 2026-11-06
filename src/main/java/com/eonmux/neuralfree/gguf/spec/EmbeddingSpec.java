package com.eonmux.neuralfree.gguf.spec;

/**
 * Model input: token embedding lookup and everything applied before the first layer.
 * <ol>
 *   <li>{@code x = token_embd[token]} (width {@link #nEmbd()}, or the input projection's input width), zero-padded
 *       to {@link #nEmbdInp()} when the first layer reads a wider input (deepstack);</li>
 *   <li>{@code x = project_in · x} when {@link #inputProjection()} is present (OPT {@code word_embed_proj_dim}
 *       different from {@code hidden_size});</li>
 *   <li>{@code + token_types[type]} and {@code + position_embd[pos + }{@link #posOffset()}{@code ]} when present
 *       (BERT family, GPT-2, OPT);</li>
 *   <li>{@code * embdScale} (gemma {@code sqrt(n_embd)}, granite {@code embedding_scale}); 1 leaves it;</li>
 *   <li>{@code token_embd_norm} (with bias when present) when present;</li>
 *   <li>a weightless RMS norm with {@code f_norm_rms_eps} when {@link #inputRmsNorm()} (talkie.cpp:50-51,
 *       muse-spark.cpp:86-87, muse-glimmer.cpp:69), or the same norm weighted when
 *       {@link #inputRmsNormWeight()} is set (Inkling {@code embed_norm}).</li>
 * </ol>
 * Per-layer embeddings (gemma3n PLE: {@code per_layer_token_embd}, {@code per_layer_model_proj},
 * {@code per_layer_proj_norm}; qwen4exp n-gram PLE) are computed once per token here and consumed by the
 * {@code ple} / {@code ngram_ple} layer extras. Immutable.
 */
public final class EmbeddingSpec {

    private final TensorRef tokenEmbd;
    private final TensorRef inputProjection;
    private final int posOffset;
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
    private final TensorRef ngramEmbd;
    private final TensorRef ngramProj;
    private final int ngramVocabSize;
    private final int ngramSplits;
    private final int ngramNeighbors;
    private final int ngramEosTokenId;
    private final TensorRef inputRmsNormWeight;

    private EmbeddingSpec(Builder b) {
        this.tokenEmbd = b.tokenEmbd;
        this.inputProjection = b.inputProjection;
        this.posOffset = b.posOffset;
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
        this.ngramEmbd = b.ngramEmbd;
        this.ngramProj = b.ngramProj;
        this.ngramVocabSize = b.ngramVocabSize;
        this.ngramSplits = b.ngramSplits;
        this.ngramNeighbors = b.ngramNeighbors;
        this.ngramEosTokenId = b.ngramEosTokenId;
        this.inputRmsNormWeight = b.inputRmsNormWeight;
    }

    /**
     * The input rows get a weightless RMS norm with {@code f_norm_rms_eps} before the first layer (or a weighted one
     * when {@link #inputRmsNormWeight()} is set).
     */
    public boolean inputRmsNorm() { return inputRmsNorm || inputRmsNormWeight != null; }

    /**
     * Weight of the input RMS norm ({@code token_embd_norm} on Inkling); null for the weightless norm of talkie,
     * muse-spark and muse-glimmer.
     */
    public TensorRef inputRmsNormWeight() { return inputRmsNormWeight; }

    public static Builder builder(TensorRef tokenEmbd, int nEmbd) {
        return new Builder(tokenEmbd, nEmbd);
    }

    /** Token table; null for drafters that embed through their target model (eagle3, dflash). */
    public TensorRef tokenEmbd() { return tokenEmbd; }
    /**
     * Projection {@code {tokenWidth, n_embd}} applied to the looked-up token rows before the position addition;
     * null when the token table already has {@code n_embd} columns.
     */
    public TensorRef inputProjection() { return inputProjection; }
    /** Added to every sequence position before the {@link #posEmbd()} lookup (OPT's two padding rows); 0 usually. */
    public int posOffset() { return posOffset; }
    /** Width of a token-table row: {@link #inputProjection()}'s input, else {@link #nEmbd()}. */
    public int tokenWidth() {
        return inputProjection != null ? (int) inputProjection.ne(0) : nEmbd;
    }
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

    /** True when the input is the fused n-gram embedding of LongCat-Flash-Lite. */
    public boolean ngram() { return ngramEmbd != null; }
    /**
     * The concatenated n-gram hash tables ({@code ngram_embd.weight}): one table per embedder, each sized
     * {@code vocabSize + 2 * embedder + 1} rows of {@link #ngramWidth()} values.
     */
    public TensorRef ngramEmbd() { return ngramEmbd; }
    /**
     * The stacked per-embedder projections ({@code ngram_proj.weight}, the transposed {@code post_projs} of the
     * checkpoint): an {@code [embedders * ngramWidth, n_embd]} matrix whose embedder {@code e} occupies rows
     * {@code [e * ngramWidth, (e + 1) * ngramWidth)}, so the gathered rows of one token are one contiguous vector.
     */
    public TensorRef ngramProj() { return ngramProj; }
    /** Base table size {@code m = ngram_vocab_size_ratio * vocab_size}. */
    public int ngramVocabSize() { return ngramVocabSize; }
    /** Hash tables per n-gram order ({@code emb_split_num}). */
    public int ngramSplits() { return ngramSplits; }
    /** Tokens per hash, the current one included ({@code emb_neighbor_num}). */
    public int ngramNeighbors() { return ngramNeighbors; }
    /** Token id that ends an n-gram walk ({@code eos_token_id}). */
    public int ngramEosTokenId() { return ngramEosTokenId; }
    /** Hash tables: {@code ngramSplits * (ngramNeighbors - 1)}. */
    public int ngramEmbedders() { return ngramSplits * (ngramNeighbors - 1); }
    /** Width of one table row and of one projection input ({@code n_embd / embedders}). */
    public int ngramWidth() { return nEmbd / ngramEmbedders(); }

    /** True when gemma3n-style per-layer embeddings are present. */
    public boolean perLayerEmbeddings() { return perLayerTokenEmbd != null; }

    @Override
    public String toString() {
        return "EmbeddingSpec{" + (tokenEmbd != null ? tokenEmbd.name() : "target") + ", nEmbd=" + nEmbd
                + ", inp=" + nEmbdInp + ", scale=" + embdScale
                + (inputProjection != null ? ", proj=" + inputProjection.name() : "")
                + ", pos=" + (posEmbd != null) + (posOffset != 0 ? "+" + posOffset : "")
                + ", types=" + (tokenTypes != null) + ", norm=" + (tokenEmbdNorm != null)
                + ", ple=" + nEmbdPerLayer + (ngramPle ? "/ngram" : "")
                + (ngramEmbd != null ? ", ngramEmbedders=" + ngramEmbedders() : "") + '}';
    }

    public static final class Builder {
        private final TensorRef tokenEmbd;
        private final int nEmbd;
        private TensorRef inputProjection;
        private int posOffset;
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
        private TensorRef ngramEmbd;
        private TensorRef ngramProj;
        private int ngramVocabSize;
        private int ngramSplits;
        private int ngramNeighbors;
        private int ngramEosTokenId = -1;
        private TensorRef inputRmsNormWeight;

        private Builder(TensorRef tokenEmbd, int nEmbd) {
            this.tokenEmbd = tokenEmbd;
            this.nEmbd = nEmbd;
            this.nEmbdInp = nEmbd;
        }

        public Builder posEmbd(TensorRef v) { this.posEmbd = v; return this; }
        /** Projection of the token-table rows to {@code n_embd}; null when the table is already {@code n_embd} wide. */
        public Builder inputProjection(TensorRef v) { this.inputProjection = v; return this; }
        /** Added to every sequence position before the position lookup; 0 uses the positions unchanged. */
        public Builder posOffset(int v) { this.posOffset = v; return this; }
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
        /** Weight of a weighted input RMS norm ({@code token_embd_norm}); null keeps the norm weightless. */
        public Builder inputRmsNormWeight(TensorRef v) { this.inputRmsNormWeight = v; return this; }

        /**
         * The fused n-gram input embedding: the concatenated hash tables {@code embd}, the stacked projections
         * {@code proj}, the base table size {@code vocabSize}, the tables per order {@code splits}, the tokens per
         * hash {@code neighbors} and the boundary token {@code eosTokenId}.
         */
        public Builder ngram(TensorRef embd, TensorRef proj, int vocabSize, int splits, int neighbors,
                             int eosTokenId) {
            this.ngramEmbd = embd;
            this.ngramProj = proj;
            this.ngramVocabSize = vocabSize;
            this.ngramSplits = splits;
            this.ngramNeighbors = neighbors;
            this.ngramEosTokenId = eosTokenId;
            return this;
        }

        /** @throws IllegalArgumentException when widths disagree with the token table or PLE is incomplete */
        public EmbeddingSpec build() {
            int tokenWidth = inputProjection != null ? (int) inputProjection.ne(0) : nEmbd;
            if (nEmbd <= 0 || (tokenEmbd != null && tokenEmbd.cols() != tokenWidth)) {
                throw new IllegalArgumentException("token embedding " + tokenEmbd
                        + (inputProjection != null ? " does not have the input projection's width " + tokenWidth
                                : " does not have width n_embd " + nEmbd));
            }
            if (inputProjection != null && (inputProjection.rank() != 2 || inputProjection.ne(1) != nEmbd
                    || tokenWidth <= 0)) {
                throw new IllegalArgumentException("input projection " + inputProjection + " is not a [token width, "
                        + nEmbd + "] matrix");
            }
            if (posOffset < 0) {
                throw new IllegalArgumentException("position offset must be >= 0, got " + posOffset);
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
            if (inputRmsNormWeight != null && tokenEmbdNorm != null) {
                throw new IllegalArgumentException("the weighted input norm and token_embd_norm are exclusive");
            }
            if ((perLayerTokenEmbd != null || ngramPle) && nEmbdPerLayer <= 0) {
                throw new IllegalArgumentException("per-layer embeddings need n_embd_per_layer > 0");
            }
            if (ngramEmbd == null) {
                if (ngramProj != null) {
                    throw new IllegalArgumentException("an n-gram projection without its hash tables");
                }
                return new EmbeddingSpec(this);
            }
            if (ngramSplits < 1 || ngramNeighbors < 2 || ngramVocabSize <= 0 || ngramEosTokenId < 0) {
                throw new IllegalArgumentException("the n-gram embedding needs splits >= 1, neighbors >= 2, a"
                        + " positive base table size and an EOS id, got " + ngramSplits + ", " + ngramNeighbors
                        + ", " + ngramVocabSize + ", " + ngramEosTokenId);
            }
            int embedders = ngramSplits * (ngramNeighbors - 1);
            if (nEmbd % embedders != 0) {
                throw new IllegalArgumentException("n_embd " + nEmbd + " does not split over the " + embedders
                        + " n-gram embedders");
            }
            long rows = 0;
            for (int i = 0; i < embedders; i++) {
                rows += ngramVocabSize + 2L * i + 1L;
            }
            if (ngramEmbd.rank() != 2 || ngramEmbd.ne(0) != nEmbd / embedders || ngramEmbd.ne(1) != rows) {
                throw new IllegalArgumentException("n-gram hash tables " + ngramEmbd + " are not ["
                        + (nEmbd / embedders) + ", " + rows + "]");
            }
            if (ngramProj == null || ngramProj.rank() != 2 || ngramProj.ne(0) != (long) (nEmbd / embedders) * embedders
                    || ngramProj.ne(1) != nEmbd) {
                throw new IllegalArgumentException("n-gram projections " + ngramProj + " are not ["
                        + ((nEmbd / embedders) * embedders) + ", " + nEmbd + "]");
            }
            return new EmbeddingSpec(this);
        }
    }
}

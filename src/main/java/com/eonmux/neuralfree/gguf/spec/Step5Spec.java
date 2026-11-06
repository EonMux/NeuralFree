package com.eonmux.neuralfree.gguf.spec;

/**
 * Step5 sparse region attention (vLLM {@code vllm/models/step5/nvidia/sparse_attention.py} and
 * {@code common/ops/summary_update.py}). The indexer scores whole {@link #regionBlock()}-token regions by their
 * summary key: one 256-wide proxy vector per closed region, an online softmax over the region's indexer keys weighted
 * per proxy dimension by {@code exp(z)}. Each KV group keeps {@link #headsPerGroup()} indexer query heads, and a token
 * attends the {@link #topK()} best closed regions plus the region its own position falls in.
 *
 * <p>Full-attention layers carry the indexer ({@link #full()}); sliding-window layers run dense windowed attention with
 * the same per-head output gate and no indexer. {@link #weightScale()} is {@code headsPerGroup^-0.5}, the forecast
 * vLLM applies to the per-head weights. The attention dims themselves come from the layer's {@link AttentionSpec}.
 * Immutable.
 */
public final class Step5Spec {

    private final int indexerHeads;
    private final int proxyDim;
    private final int indexerRopeDim;
    private final int topK;
    private final int regionBlock;
    private final int groups;
    private final boolean full;
    private final float weightScale;
    private final RopeSpec indexerRope;

    private Step5Spec(Builder b) {
        this.indexerHeads = b.indexerHeads;
        this.proxyDim = b.proxyDim;
        this.indexerRopeDim = b.indexerRopeDim;
        this.topK = b.topK;
        this.regionBlock = b.regionBlock;
        this.groups = b.groups;
        this.full = b.full;
        this.weightScale = b.weightScale;
        this.indexerRope = b.indexerRope;
    }

    public static Builder builder() {
        return new Builder();
    }

    /** Indexer query heads over all provider groups ({@code sparse_indexer_num_heads}). */
    public int indexerHeads() { return indexerHeads; }
    /** Width of the indexer proxy head ({@code proxy_dim}, 256). */
    public int proxyDim() { return proxyDim; }
    /** RoPE dims of each proxy head ({@code sparse_indexer_rope_dim}). */
    public int indexerRopeDim() { return indexerRopeDim; }
    /** Closed regions scored and selected per token and KV group ({@code topk}). */
    public int topK() { return topK; }
    /** Tokens per region ({@code region_block_size}, 8). */
    public int regionBlock() { return regionBlock; }
    /** Provider groups, one per KV head ({@code index_tp_size} / {@code num_provider_groups}). */
    public int groups() { return groups; }
    /** Indexer query heads per provider group. */
    public int headsPerGroup() { return indexerHeads / groups; }
    /** True when the layer scores and selects regions; false for a sliding-window layer. */
    public boolean full() { return full; }
    /** Per-group head-weight prescale {@code headsPerGroup^-0.5}. */
    public float weightScale() { return weightScale; }
    /** NeoX rotary of the indexer proxy heads; null for a sliding-window layer. */
    public RopeSpec indexerRope() { return indexerRope; }

    /** Regions a context of {@code nCtx} positions holds. */
    public int regions(int nCtx) {
        return (nCtx + regionBlock - 1) / regionBlock;
    }

    /**
     * Rows the open-region ring holds: a whole region of rows a closing update replays, plus the
     * {@code rollbackSnapshots} positions a speculative step writes over it and may then drop. vLLM sizes the ring
     * to {@code 7 + k} rows ({@code Step5CSATailRing.required_rows}); keeping one more covers a step that writes
     * {@code k + 1} rows over an open region that already holds seven.
     */
    public static int ringCapacity(int regionBlock, int rollbackSnapshots) {
        return Math.max(regionBlock, 8 + Math.max(0, rollbackSnapshots));
    }

    @Override
    public String toString() {
        return "Step5Spec{heads=" + indexerHeads + "/" + groups + ", proxy=" + proxyDim + ", rope=" + indexerRopeDim
                + ", topK=" + topK + ", region=" + regionBlock + (full ? ", indexer" : ", sliding") + '}';
    }

    public static final class Builder {
        private int indexerHeads;
        private int proxyDim = 256;
        private int indexerRopeDim;
        private int topK;
        private int regionBlock = 8;
        private int groups = 1;
        private boolean full = true;
        private float weightScale = Float.NaN;
        private RopeSpec indexerRope;

        private Builder() {
        }

        public Builder indexerHeads(int v) { this.indexerHeads = v; return this; }
        public Builder proxyDim(int v) { this.proxyDim = v; return this; }
        public Builder indexerRopeDim(int v) { this.indexerRopeDim = v; return this; }
        public Builder topK(int v) { this.topK = v; return this; }
        public Builder regionBlock(int v) { this.regionBlock = v; return this; }
        public Builder groups(int v) { this.groups = v; return this; }
        public Builder full(boolean v) { this.full = v; return this; }
        public Builder indexerRope(RopeSpec v) { this.indexerRope = v; return this; }

        /** Sets {@link #weightScale()} to {@code headsPerGroup^-0.5} in float arithmetic. */
        public Builder defaultWeightScale() {
            if (indexerHeads <= 0 || groups <= 0 || indexerHeads % groups != 0) {
                throw new IllegalStateException("set a divisible indexerHeads and groups before"
                        + " defaultWeightScale()");
            }
            float perGroup = (float) (indexerHeads / groups);
            this.weightScale = 1.0f / (float) Math.sqrt(perGroup);
            return this;
        }

        /**
         * @throws IllegalArgumentException for non-positive geometry, indexer heads that do not divide into the
         *                                  groups, or an invalid rotary width
         */
        public Step5Spec build() {
            if (proxyDim <= 0 || regionBlock <= 0 || topK <= 0 || indexerHeads <= 0 || groups <= 0) {
                throw new IllegalArgumentException("Step5 indexer geometry must be positive: proxy " + proxyDim
                        + ", region " + regionBlock + ", topK " + topK + ", heads " + indexerHeads + ", groups "
                        + groups);
            }
            if (indexerHeads % groups != 0) {
                throw new IllegalArgumentException("Step5 indexer heads " + indexerHeads + " do not divide into "
                        + groups + " provider groups");
            }
            if (indexerRopeDim < 0 || indexerRopeDim > proxyDim || indexerRopeDim % 2 != 0) {
                throw new IllegalArgumentException("Step5 indexer rope dim " + indexerRopeDim
                        + " must be even and within the proxy head " + proxyDim);
            }
            if (indexerRope != null && indexerRope.nDims() != indexerRopeDim) {
                throw new IllegalArgumentException("Step5 indexer RopeSpec rotates " + indexerRope.nDims()
                        + " dims, the spec declares " + indexerRopeDim);
            }
            if (!(weightScale > 0f) || Float.isInfinite(weightScale)) {
                throw new IllegalArgumentException("Step5 indexer weightScale must be a positive finite value, got "
                        + weightScale);
            }
            return new Step5Spec(this);
        }
    }
}

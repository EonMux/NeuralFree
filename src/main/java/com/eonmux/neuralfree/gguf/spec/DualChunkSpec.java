package com.eonmux.neuralfree.gguf.spec;

/**
 * Dual-chunk attention of one layer, vLLM's {@code dual_chunk_attention_config} of Qwen2.5-1M
 * (vllm/model_executor/layers/rotary_embedding/dual_chunk_rope.py and the dual-chunk flash-attention backend).
 *
 * <p>The query is rotated five ways ({@code query}, {@code query_succ}, {@code query_inter},
 * {@code query_succ_critical}, {@code query_inter_critical}) and the attention is split into the token's own chunk
 * ({@code intra}, causal), the preceding chunk ({@code succ}, non-causal) and every earlier chunk ({@code inter},
 * non-causal). The three stages share one softmax over the concatenated logits. Keys rotate at their position modulo
 * the chunk length ({@code chunkSize - localSize}), so a key carries the same rotation in every chunk it is read
 * from. {@link #originalMaxPositionEmbeddings()} scales the logits once a sequence outgrows the trained length
 * ({@code max(1, 0.1 * ln(seqLen / original) + 1)}).
 *
 * <p>The sparse-attention knobs vLLM reads are carried here too; NeuralFree does not run the vertical-slash sparse
 * kernel, so a configuration with {@link #sparseAttentionEnabled()} is refused by name at load.
 * Immutable.
 */
public final class DualChunkSpec {

    private final int chunkSize;
    private final int localSize;
    private final int originalMaxPositionEmbeddings;
    private final boolean sparseAttentionEnabled;
    private final int sparseAttentionThreshold;
    private final int sparseAttentionLastQ;

    private DualChunkSpec(Builder b) {
        this.chunkSize = b.chunkSize;
        this.localSize = b.localSize;
        this.originalMaxPositionEmbeddings = b.originalMaxPositionEmbeddings;
        this.sparseAttentionEnabled = b.sparseAttentionEnabled;
        this.sparseAttentionThreshold = b.sparseAttentionThreshold;
        this.sparseAttentionLastQ = b.sparseAttentionLastQ;
    }

    public static Builder builder() {
        return new Builder();
    }

    /** vLLM's default {@code chunk_size}. */
    public static final int DEFAULT_CHUNK_SIZE = 8192;
    /** vLLM's default {@code local_size}. */
    public static final int DEFAULT_LOCAL_SIZE = 1024;
    /** vLLM's default {@code sparse_attention_threshold}. */
    public static final int DEFAULT_SPARSE_THRESHOLD = 32768;
    /** vLLM's default {@code sparse_attention_last_q}. */
    public static final int DEFAULT_SPARSE_LAST_Q = 64;

    /** Tokens of one attention chunk ({@code chunk_size}). */
    public int chunkSize() { return chunkSize; }
    /** Tokens of the sliding part of a chunk ({@code local_size}); the chunks themselves are {@link #chunkLength()}. */
    public int localSize() { return localSize; }
    /** Chunk length {@code chunk_size - local_size}, the modulus of the key rotation. */
    public int chunkLength() { return chunkSize - localSize; }
    /** Trained context length of the scaling factor; 0 disables it. */
    public int originalMaxPositionEmbeddings() { return originalMaxPositionEmbeddings; }
    /** Whether the vertical-slash sparse attention was enabled (unsupported in NeuralFree). */
    public boolean sparseAttentionEnabled() { return sparseAttentionEnabled; }
    public int sparseAttentionThreshold() { return sparseAttentionThreshold; }
    public int sparseAttentionLastQ() { return sparseAttentionLastQ; }

    /**
     * Logit multiplier of a sequence of {@code sequenceLength} positions: 1 while the sequence fits
     * {@link #originalMaxPositionEmbeddings()}, else {@code max(1, 0.1 * ln(sequenceLength / original) + 1)}
     * (vLLM {@code DualChunkFlashAttentionMetadata.scaling_factor}).
     *
     * @throws IllegalArgumentException when {@code sequenceLength} is negative
     */
    public float scalingFactor(int sequenceLength) {
        if (sequenceLength < 0) {
            throw new IllegalArgumentException("dual-chunk sequence length must be >= 0, got " + sequenceLength);
        }
        if (originalMaxPositionEmbeddings <= 0 || sequenceLength <= originalMaxPositionEmbeddings) {
            return 1.0f;
        }
        return Math.max(1.0f, 0.1f * (float) Math.log((float) sequenceLength
                / (float) originalMaxPositionEmbeddings) + 1.0f);
    }

    /** True when the sequence scaling factor of {@code sequenceLength} is not the identity. */
    public boolean scales(int sequenceLength) {
        return scalingFactor(sequenceLength) != 1.0f;
    }

    @Override
    public String toString() {
        return "DualChunkSpec{chunk=" + chunkSize + ", local=" + localSize + ", orig="
                + originalMaxPositionEmbeddings + (sparseAttentionEnabled ? ", sparse threshold="
                + sparseAttentionThreshold + " lastQ=" + sparseAttentionLastQ : "") + '}';
    }

    public static final class Builder {
        private int chunkSize = DEFAULT_CHUNK_SIZE;
        private int localSize = DEFAULT_LOCAL_SIZE;
        private int originalMaxPositionEmbeddings;
        private boolean sparseAttentionEnabled;
        private int sparseAttentionThreshold = DEFAULT_SPARSE_THRESHOLD;
        private int sparseAttentionLastQ = DEFAULT_SPARSE_LAST_Q;

        private Builder() {
        }

        public Builder chunkSize(int v) { this.chunkSize = v; return this; }
        public Builder localSize(int v) { this.localSize = v; return this; }
        public Builder originalMaxPositionEmbeddings(int v) { this.originalMaxPositionEmbeddings = v; return this; }
        public Builder sparseAttentionEnabled(boolean v) { this.sparseAttentionEnabled = v; return this; }
        public Builder sparseAttentionThreshold(int v) { this.sparseAttentionThreshold = v; return this; }
        public Builder sparseAttentionLastQ(int v) { this.sparseAttentionLastQ = v; return this; }

        /**
         * @throws IllegalArgumentException when {@code chunk_size <= local_size}, either is not positive, the
         *                                  original context length is negative, or the sparse knobs are not positive
         */
        public DualChunkSpec build() {
            if (chunkSize <= 0 || localSize <= 0 || chunkSize <= localSize) {
                throw new IllegalArgumentException("dual-chunk attention needs 0 < local_size < chunk_size, got "
                        + localSize + " / " + chunkSize);
            }
            if (originalMaxPositionEmbeddings < 0) {
                throw new IllegalArgumentException("dual-chunk original_max_position_embeddings must be >= 0, got "
                        + originalMaxPositionEmbeddings);
            }
            if (sparseAttentionThreshold <= 0 || sparseAttentionLastQ <= 0) {
                throw new IllegalArgumentException("dual-chunk sparse_attention_threshold and sparse_attention_last_q"
                        + " must be positive, got " + sparseAttentionThreshold + " / " + sparseAttentionLastQ);
            }
            return new DualChunkSpec(this);
        }
    }
}

package com.eonmux.neuralfree.gguf.spec;

/**
 * MiniMax sparse attention (src/models/minimax-m3.cpp). One index query per GQA group
 * ({@code iq = rms(W_iq x) * q_norm}, {@link #idxHeads()} heads) and a single index key per token
 * ({@code ik = rms(W_ik x) * k_norm}, minimax-m3.cpp:84-87, 341-347), both with partial RoPE, score position blocks of
 * {@link #blockSize()} tokens; attention runs over the {@link #topKBlocks()} best blocks, among which the
 * {@link #localBlocks()} most recent ones are always selected (they are biased into the top-K, not added to it).
 * Layers below the dense lead use dense attention
 * ({@link #sparse()} false). Index keys ({@link #idxHeadDim()} values) are cached per cell ({@code GpuMsaCache}).
 * The attention dims themselves come from the layer's {@link AttentionSpec}. Immutable.
 */
public final class MsaSpec {

    private final int idxHeads;
    private final int idxHeadDim;
    private final int idxRopeDim;
    private final int blockSize;
    private final int localBlocks;
    private final int topKBlocks;
    private final boolean sparse;
    private final float kqScale;

    private MsaSpec(Builder b) {
        this.idxHeads = b.idxHeads;
        this.idxHeadDim = b.idxHeadDim;
        this.idxRopeDim = b.idxRopeDim;
        this.blockSize = b.blockSize;
        this.localBlocks = b.localBlocks;
        this.topKBlocks = b.topKBlocks;
        this.sparse = b.sparse;
        this.kqScale = b.kqScale;
    }

    public static Builder builder() {
        return new Builder();
    }

    /** Index heads ({@code indexer_n_head}): one per GQA group. */
    public int idxHeads() { return idxHeads; }
    /** Index key width ({@code indexer_head_size}); the index cache row width. */
    public int idxHeadDim() { return idxHeadDim; }
    public int idxRopeDim() { return idxRopeDim; }
    /** Tokens per position block ({@code indexer_block_size}). */
    public int blockSize() { return blockSize; }
    /** Most recent blocks always attended ({@code indexer_local_blocks}). */
    public int localBlocks() { return localBlocks; }
    /** Blocks selected by score ({@code indexer_top_k}). */
    public int topKBlocks() { return topKBlocks; }
    /** False for the dense lead layers. */
    public boolean sparse() { return sparse; }
    /** Attention {@code kq_scale} ({@code 1/sqrt(n_embd_head)}, minimax-m3.cpp:386). */
    public float kqScale() { return kqScale; }

    @Override
    public String toString() {
        return "MsaSpec{idxHeads=" + idxHeads + ", idxDim=" + idxHeadDim + ", rope=" + idxRopeDim + ", block=" + blockSize
                + ", local=" + localBlocks + ", topK=" + topKBlocks + ", sparse=" + sparse + '}';
    }

    public static final class Builder {
        private int idxHeads;
        private int idxHeadDim;
        private int idxRopeDim;
        private int blockSize;
        private int localBlocks;
        private int topKBlocks;
        private boolean sparse = true;
        private float kqScale = Float.NaN;

        private Builder() {
        }

        public Builder idxHeads(int v) { this.idxHeads = v; return this; }
        public Builder idxHeadDim(int v) { this.idxHeadDim = v; return this; }
        public Builder idxRopeDim(int v) { this.idxRopeDim = v; return this; }
        public Builder blockSize(int v) { this.blockSize = v; return this; }
        public Builder localBlocks(int v) { this.localBlocks = v; return this; }
        public Builder topKBlocks(int v) { this.topKBlocks = v; return this; }
        public Builder sparse(boolean v) { this.sparse = v; return this; }
        public Builder kqScale(float v) { this.kqScale = v; return this; }

        /** @throws IllegalArgumentException for non-positive dims or block parameters */
        public MsaSpec build() {
            if (idxHeads <= 0 || idxHeadDim <= 0 || blockSize <= 0 || topKBlocks <= 0 || localBlocks < 0) {
                throw new IllegalArgumentException("MSA heads, head dim, block size and top-k must be positive: "
                        + idxHeads + "/" + idxHeadDim + "/" + blockSize + "/" + topKBlocks + ", local " + localBlocks);
            }
            if (idxRopeDim < 0 || idxRopeDim > idxHeadDim || idxRopeDim % 2 != 0) {
                throw new IllegalArgumentException("MSA index rope dim " + idxRopeDim + " must be even and within "
                        + idxHeadDim);
            }
            if (!(kqScale > 0f) || Float.isInfinite(kqScale)) {
                throw new IllegalArgumentException("MSA kqScale must be a positive finite value, got " + kqScale);
            }
            return new MsaSpec(this);
        }
    }
}

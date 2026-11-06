package com.eonmux.neuralfree.gguf.spec;

/**
 * DeepSeek-V4.1 cross-layer sharing of one compressed-attention layer (models/deepseek41.cpp:35-86). Every
 * compressed layer (compress ratio 1 or 2) reads the compressed rows of the latest KV source layer at or before it
 * ({@code attention.kv_source_layers}) and the top-k selection of the latest indexer source layer
 * ({@code attention.indexer.source_layers}); only the source layer runs the compressor and owns the rows. The
 * optional two-level candidate selection ({@code attention.indexer.candidate_source_layer},
 * {@code candidate_top_k_blocks}, {@code candidate_block_size}) lets one indexer source pick whole blocks of
 * compressed rows that later indexer sources score inside.
 *
 * <p>The indexer geometry is part of the facet on every layer, because a consumer carries no indexer tensors of its
 * own but still sizes and reads the index keys stored beside the compressed rows. Immutable.</p>
 */
public final class Dsv41Spec {

    /** Compress ratio of the selected (CSA) layers of a V4.1 file. */
    public static final int RATIO_SELECTED = 2;
    /** Compress ratio of the dense (HCA) layers of a V4.1 file. */
    public static final int RATIO_DENSE = 1;

    private final int il;
    private final int kvSourceLayer;
    private final int topkSourceLayer;
    private final int candidateSourceLayer;
    private final int candidateTopKBlocks;
    private final int candidateBlockSize;
    private final int indexerHeads;
    private final int indexerHeadDim;
    private final int indexerTopK;

    private Dsv41Spec(Builder b) {
        this.il = b.il;
        this.kvSourceLayer = b.kvSourceLayer;
        this.topkSourceLayer = b.topkSourceLayer;
        this.candidateSourceLayer = b.candidateSourceLayer;
        this.candidateTopKBlocks = b.candidateTopKBlocks;
        this.candidateBlockSize = b.candidateBlockSize;
        this.indexerHeads = b.indexerHeads;
        this.indexerHeadDim = b.indexerHeadDim;
        this.indexerTopK = b.indexerTopK;
    }

    public static Builder builder(int il) {
        return new Builder(il);
    }

    /** Virtual layer index of the facet. */
    public int il() { return il; }
    /** Layer whose compressed rows this layer reads (itself when it owns them). */
    public int kvSourceLayer() { return kvSourceLayer; }
    /** Layer whose indexer top-k this layer reuses (itself when it computes it). */
    public int topkSourceLayer() { return topkSourceLayer; }
    /** Layer that picks the candidate blocks, or -1 without the two-level selection. */
    public int candidateSourceLayer() { return candidateSourceLayer; }
    /** Blocks the candidate source keeps per query; 0 without the two-level selection. */
    public int candidateTopKBlocks() { return candidateTopKBlocks; }
    /** Consecutive compressed rows of one candidate block; 0 without the two-level selection. */
    public int candidateBlockSize() { return candidateBlockSize; }
    public int indexerHeads() { return indexerHeads; }
    /** Indexer head size, the width of the index key stored beside each compressed row. */
    public int indexerHeadDim() { return indexerHeadDim; }
    public int indexerTopK() { return indexerTopK; }

    /** True when this layer owns (writes) the compressed rows and their compressor state. */
    public boolean ownsCompressed() { return kvSourceLayer == il; }
    /** True when this layer computes the indexer top-k instead of reading a source's. */
    public boolean computesTopk() { return topkSourceLayer == il; }
    /** True when this layer picks the candidate blocks (level one of the two-level selection). */
    public boolean picksCandidateBlocks() { return candidateSourceLayer == il; }
    /** True when this layer scores inside an earlier layer's candidate blocks (level two). */
    public boolean usesCandidateBlocks() { return candidateSourceLayer >= 0 && candidateSourceLayer < il; }

    /** Width of one compressed row: the compressed KV plus the index key stored beside it. */
    public int rowWidth(int headDim) { return headDim + indexerHeadDim; }

    @Override
    public String toString() {
        return "Dsv41Spec{il=" + il + ", kvSource=" + kvSourceLayer + ", topkSource=" + topkSourceLayer
                + (candidateSourceLayer >= 0 ? ", candidate=" + candidateSourceLayer + "x" + candidateTopKBlocks
                + "/" + candidateBlockSize : "") + ", indexer=" + indexerHeads + "x" + indexerHeadDim + " top"
                + indexerTopK + '}';
    }

    /** Builder of a V4.1 facet. */
    public static final class Builder {
        private final int il;
        private int kvSourceLayer = -1;
        private int topkSourceLayer = -1;
        private int candidateSourceLayer = -1;
        private int candidateTopKBlocks;
        private int candidateBlockSize;
        private int indexerHeads;
        private int indexerHeadDim;
        private int indexerTopK;

        private Builder(int il) {
            this.il = il;
        }

        public Builder kvSourceLayer(int v) { this.kvSourceLayer = v; return this; }
        public Builder topkSourceLayer(int v) { this.topkSourceLayer = v; return this; }
        public Builder candidate(int sourceLayer, int topKBlocks, int blockSize) {
            this.candidateSourceLayer = sourceLayer;
            this.candidateTopKBlocks = topKBlocks;
            this.candidateBlockSize = blockSize;
            return this;
        }
        public Builder indexer(int heads, int headDim, int topK) {
            this.indexerHeads = heads;
            this.indexerHeadDim = headDim;
            this.indexerTopK = topK;
            return this;
        }

        /**
         * @throws IllegalArgumentException for a negative layer, a source later than the layer or one of its top-k
         *                                  source, or a malformed candidate selection
         */
        public Dsv41Spec build() {
            if (il < 0) {
                throw new IllegalArgumentException("DSV4.1 facet layer " + il + " is negative");
            }
            if (kvSourceLayer < 0 || topkSourceLayer < 0) {
                if (kvSourceLayer != -1 || topkSourceLayer != -1) {
                    throw new IllegalArgumentException("DSV4.1 layer " + il + " has only one of its source layers:"
                            + " kv " + kvSourceLayer + ", indexer " + topkSourceLayer);
                }
                if (candidateSourceLayer >= 0 || candidateTopKBlocks != 0 || candidateBlockSize != 0) {
                    throw new IllegalArgumentException("DSV4.1 window-only layer " + il
                            + " cannot take part in the candidate selection");
                }
                if (indexerHeads <= 0 || indexerHeadDim <= 0 || indexerTopK <= 0) {
                    throw new IllegalArgumentException("DSV4.1 layer " + il + " needs positive indexer geometry, got "
                            + indexerHeads + " x " + indexerHeadDim + " top " + indexerTopK);
                }
                return new Dsv41Spec(this);
            }
            if (kvSourceLayer > il) {
                throw new IllegalArgumentException("DSV4.1 layer " + il + " needs a KV source in [" + 0 + ", " + il
                        + "], got " + kvSourceLayer);
            }
            if (topkSourceLayer < kvSourceLayer || topkSourceLayer > il) {
                throw new IllegalArgumentException("DSV4.1 layer " + il + " needs an indexer source in ["
                        + kvSourceLayer + ", " + il + "], got " + topkSourceLayer);
            }
            if (candidateSourceLayer < -1 || candidateSourceLayer > il) {
                throw new IllegalArgumentException("DSV4.1 candidate source of layer " + il + " must be at most the"
                        + " layer, got " + candidateSourceLayer);
            }
            if (candidateSourceLayer >= 0 && (candidateTopKBlocks <= 0 || candidateBlockSize <= 0)) {
                throw new IllegalArgumentException("DSV4.1 candidate selection needs positive block counts and a"
                        + " positive block size, got " + candidateTopKBlocks + " x " + candidateBlockSize);
            }
            if (candidateSourceLayer < 0 && (candidateTopKBlocks != 0 || candidateBlockSize != 0)) {
                throw new IllegalArgumentException("DSV4.1 layer " + il + " gives candidate blocks without a"
                        + " candidate source");
            }
            if (indexerHeads <= 0 || indexerHeadDim <= 0 || indexerTopK <= 0) {
                throw new IllegalArgumentException("DSV4.1 layer " + il + " needs positive indexer geometry, got "
                        + indexerHeads + " x " + indexerHeadDim + " top " + indexerTopK);
            }
            return new Dsv41Spec(this);
        }
    }
}

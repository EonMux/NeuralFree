package com.eonmux.neuralfree.gguf.spec;

import com.eonmux.neuralfree.gguf.arch.RopeType;

import java.util.Objects;

/**
 * DeepSeek sparse-attention lightning indexer of one layer (deepseek32, glm-dsa, dots3note, hy_v4, glm5-next,
 * qwen4exp, and the CSA layers of deepseek4). Score of query token {@code t} against cached key {@code s}:
 * {@code sum_h relu(iq[t,h] . ik[s]) * w[t,h]} with {@code w} pre-scaled by {@link #weightScale()}; attention is
 * restricted to the {@code min(n_kv, topK)} best cells.
 *
 * <p>Non-full layers ({@link #full()} false) reuse the top-k of {@link #sourceLayer()} instead of scoring
 * (glm-dsa {@code attention.indexer.types}). The kpool fields describe the QSA pooled indexer
 * (glm5-next, qwen4exp); {@code kpool == 0} disables pooling. Immutable.
 */
public final class DsaSpec {

    private final int indexerHeads;
    private final int indexerHeadDim;
    private final int indexerRopeDim;
    private final RopeType ropeType;
    private final boolean ropeTail;
    private final int topK;
    private final boolean full;
    private final int sourceLayer;
    private final boolean hadamard;
    private final int kpool;
    private final int kpoolRow;
    private final boolean kpoolByOrder;
    private final boolean kpoolSelectTail;
    private final float weightScale;

    private DsaSpec(Builder b) {
        this.indexerHeads = b.indexerHeads;
        this.indexerHeadDim = b.indexerHeadDim;
        this.indexerRopeDim = b.indexerRopeDim;
        this.ropeType = b.ropeType;
        this.ropeTail = b.ropeTail;
        this.topK = b.topK;
        this.full = b.full;
        this.sourceLayer = b.sourceLayer;
        this.hadamard = b.hadamard;
        this.kpool = b.kpool;
        this.kpoolRow = b.kpoolRow;
        this.kpoolByOrder = b.kpoolByOrder;
        this.kpoolSelectTail = b.kpoolSelectTail;
        this.weightScale = b.weightScale;
    }

    public static Builder builder() {
        return new Builder();
    }

    public int indexerHeads() { return indexerHeads; }
    /** Indexer head size ({@code indexer_head_size}); also the indexer cache row width. */
    public int indexerHeadDim() { return indexerHeadDim; }
    /** RoPE dims of each indexer head; placed per {@link #ropeTail()}. */
    public int indexerRopeDim() { return indexerRopeDim; }
    /**
     * False for the {@code [rope | nope]} head layout (deepseek32.cpp:173), true for {@code [nope | rope]}
     * (deepseek4 lightning indexer, deepseek4.cpp:600-616).
     */
    public boolean ropeTail() { return ropeTail; }
    /** NEOX for deepseek32, the model rope type for glm-dsa. */
    public RopeType ropeType() { return ropeType; }
    public int topK() { return topK; }
    /** This layer computes indexer scores; otherwise it reuses {@link #sourceLayer()}'s selection. */
    public boolean full() { return full; }
    /** Layer whose top-k this layer reuses; equals the layer itself when {@link #full()}. */
    public int sourceLayer() { return sourceLayer; }
    /** Indexer q/k are Hadamard-rotated (llama.cpp forces this when {@code n_embd_head_k_full == indexer_head_size}). */
    public boolean hadamard() { return hadamard; }
    /** QSA pooled rows per indexer score; 0 disables pooling. */
    public int kpool() { return kpool; }
    public int kpoolRow() { return kpoolRow; }
    public boolean kpoolByOrder() { return kpoolByOrder; }
    public boolean kpoolSelectTail() { return kpoolSelectTail; }
    /** Head-weight prescale {@code (indexerHeadDim * indexerHeads)^-0.5}. */
    public float weightScale() { return weightScale; }

    @Override
    public String toString() {
        return "DsaSpec{heads=" + indexerHeads + ", dim=" + indexerHeadDim + ", rope=" + indexerRopeDim + "/" + ropeType
                + ", topK=" + topK + ", full=" + full + ", source=" + sourceLayer + ", hadamard=" + hadamard
                + ", kpool=" + kpool + '}';
    }

    public static final class Builder {
        private int indexerHeads;
        private int indexerHeadDim;
        private int indexerRopeDim;
        private RopeType ropeType = RopeType.NEOX;
        private boolean ropeTail;
        private int topK;
        private boolean full = true;
        private int sourceLayer = -1;
        private boolean hadamard;
        private int kpool;
        private int kpoolRow;
        private boolean kpoolByOrder;
        private boolean kpoolSelectTail;
        private float weightScale = Float.NaN;

        private Builder() {
        }

        public Builder indexerHeads(int v) { this.indexerHeads = v; return this; }
        public Builder indexerHeadDim(int v) { this.indexerHeadDim = v; return this; }
        public Builder indexerRopeDim(int v) { this.indexerRopeDim = v; return this; }
        public Builder ropeType(RopeType v) { this.ropeType = Objects.requireNonNull(v, "ropeType"); return this; }
        public Builder ropeTail(boolean v) { this.ropeTail = v; return this; }
        public Builder topK(int v) { this.topK = v; return this; }
        public Builder full(boolean v) { this.full = v; return this; }
        public Builder sourceLayer(int v) { this.sourceLayer = v; return this; }
        public Builder hadamard(boolean v) { this.hadamard = v; return this; }
        public Builder kpool(int pool, int row, boolean byOrder, boolean selectTail) {
            this.kpool = pool;
            this.kpoolRow = row;
            this.kpoolByOrder = byOrder;
            this.kpoolSelectTail = selectTail;
            return this;
        }
        public Builder weightScale(float v) { this.weightScale = v; return this; }

        /**
         * Sets {@link #weightScale(float)} to {@code 1.0f / sqrtf(float(indexerHeadDim * indexerHeads))} in float
         * arithmetic (src/models/deepseek32.cpp:269).
         */
        public Builder defaultWeightScale() {
            if (indexerHeadDim <= 0 || indexerHeads <= 0) {
                throw new IllegalStateException("set indexerHeads and indexerHeadDim before defaultWeightScale()");
            }
            float product = (float) ((long) indexerHeadDim * indexerHeads);
            this.weightScale = 1.0f / (float) Math.sqrt(product);
            return this;
        }

        /** @throws IllegalArgumentException for non-positive dims, an invalid rope split, or a missing source layer */
        public DsaSpec build() {
            if (indexerHeads <= 0 || indexerHeadDim <= 0 || topK <= 0) {
                throw new IllegalArgumentException("DSA indexer heads, head dim and topK must be positive: "
                        + indexerHeads + "/" + indexerHeadDim + "/" + topK);
            }
            if (indexerRopeDim < 0 || indexerRopeDim > indexerHeadDim || indexerRopeDim % 2 != 0) {
                throw new IllegalArgumentException("DSA indexer rope dim " + indexerRopeDim
                        + " must be even and within the head dim " + indexerHeadDim);
            }
            if (sourceLayer < 0) {
                throw new IllegalArgumentException("DSA sourceLayer must be set (the layer itself when full)");
            }
            if (kpool < 0 || (kpool > 0 && kpoolRow <= 0)) {
                throw new IllegalArgumentException("DSA kpool " + kpool + " needs a positive kpool row, got " + kpoolRow);
            }
            if (!(weightScale > 0f) || Float.isInfinite(weightScale)) {
                throw new IllegalArgumentException("DSA weightScale must be a positive finite value, got " + weightScale);
            }
            return new DsaSpec(this);
        }
    }
}

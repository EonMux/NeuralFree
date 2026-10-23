package com.eonmux.neuralfree.gguf.spec;

/**
 * DeepSeek-V4 attention of one layer (src/models/deepseek4.cpp, src/llama-kv-cache-dsv4.cpp). Every layer keeps a raw
 * K-only sliding window of {@link #window()} tokens (shared K/V, MQA). Layers with {@code compressRatio == 4}
 * (CSA) add a compressed K cache scored through the lightning {@link #indexer()}; {@code compressRatio == 128}
 * (HCA) adds a heavily compressed cache attended densely; 0 is window-only. The output projection is grouped
 * ({@link #oGroupCount()} groups through a rank {@link #oLoraRank()} bottleneck). Immutable.
 */
public final class Dsv4Spec {

    /** Compression ratio of CSA layers. */
    public static final int RATIO_CSA = 4;
    /** Compression ratio of HCA layers. */
    public static final int RATIO_HCA = 128;

    private final int nHead;
    private final int compressRatio;
    private final int window;
    private final int headDim;
    private final int ropeDim;
    private final int qLoraRank;
    private final int oGroupCount;
    private final int oLoraRank;
    private final float compressRopeBase;
    private final DsaSpec indexer;
    private final boolean sinks;
    private final float kqScale;

    private Dsv4Spec(Builder b) {
        this.nHead = b.nHead;
        this.compressRatio = b.compressRatio;
        this.window = b.window;
        this.headDim = b.headDim;
        this.ropeDim = b.ropeDim;
        this.qLoraRank = b.qLoraRank;
        this.oGroupCount = b.oGroupCount;
        this.oLoraRank = b.oLoraRank;
        this.compressRopeBase = b.compressRopeBase;
        this.indexer = b.indexer;
        this.sinks = b.sinks;
        this.kqScale = b.kqScale;
    }

    public static Builder builder() {
        return new Builder();
    }

    public int nHead() { return nHead; }
    /** 0 (window only), {@link #RATIO_CSA} or {@link #RATIO_HCA}. */
    public int compressRatio() { return compressRatio; }
    /** Raw sliding window ({@code n_swa}). */
    public int window() { return window; }
    /** Shared K/V head dim ({@code n_embd_head_k}). */
    public int headDim() { return headDim; }
    /** RoPE dims at the end of each head. */
    public int ropeDim() { return ropeDim; }
    public int qLoraRank() { return qLoraRank; }
    public int oGroupCount() { return oGroupCount; }
    public int oLoraRank() { return oLoraRank; }
    /** RoPE base used on compressed rows ({@code compress_rope_freq_base}). */
    public float compressRopeBase() { return compressRopeBase; }
    /** Lightning indexer of a CSA layer; null otherwise. */
    public DsaSpec indexer() { return indexer; }
    public boolean sinks() { return sinks; }
    public float kqScale() { return kqScale; }

    public boolean csa() { return compressRatio == RATIO_CSA; }
    public boolean hca() { return compressRatio == RATIO_HCA; }

    @Override
    public String toString() {
        return "Dsv4Spec{nHead=" + nHead + ", ratio=" + compressRatio + ", window=" + window + ", head=" + headDim
                + ", rope=" + ropeDim + ", oGroups=" + oGroupCount + ", oLora=" + oLoraRank + ", indexer=" + indexer + '}';
    }

    public static final class Builder {
        private int nHead;
        private int compressRatio;
        private int window;
        private int headDim;
        private int ropeDim;
        private int qLoraRank;
        private int oGroupCount;
        private int oLoraRank;
        private float compressRopeBase;
        private DsaSpec indexer;
        private boolean sinks;
        private float kqScale = Float.NaN;

        private Builder() {
        }

        public Builder nHead(int v) { this.nHead = v; return this; }
        public Builder compressRatio(int v) { this.compressRatio = v; return this; }
        public Builder window(int v) { this.window = v; return this; }
        public Builder headDim(int v) { this.headDim = v; return this; }
        public Builder ropeDim(int v) { this.ropeDim = v; return this; }
        public Builder qLoraRank(int v) { this.qLoraRank = v; return this; }
        public Builder oGroupCount(int v) { this.oGroupCount = v; return this; }
        public Builder oLoraRank(int v) { this.oLoraRank = v; return this; }
        public Builder compressRopeBase(float v) { this.compressRopeBase = v; return this; }
        public Builder indexer(DsaSpec v) { this.indexer = v; return this; }
        public Builder sinks(boolean v) { this.sinks = v; return this; }
        public Builder kqScale(float v) { this.kqScale = v; return this; }

        /** @throws IllegalArgumentException for an unknown ratio, inconsistent indexer presence or bad dims */
        public Dsv4Spec build() {
            if (compressRatio != 0 && compressRatio != RATIO_CSA && compressRatio != RATIO_HCA) {
                throw new IllegalArgumentException("DSV4 compress ratio must be 0, 4 or 128, got " + compressRatio);
            }
            if ((compressRatio == RATIO_CSA) != (indexer != null)) {
                throw new IllegalArgumentException("DSV4 layers have an indexer exactly when the compress ratio is 4 (ratio "
                        + compressRatio + ", indexer " + (indexer != null) + ")");
            }
            if (nHead <= 0 || window <= 0 || headDim <= 0 || oGroupCount <= 0 || oLoraRank <= 0 || qLoraRank < 0) {
                throw new IllegalArgumentException("DSV4 dims must be positive: nHead=" + nHead + " window=" + window
                        + " head=" + headDim + " oGroups=" + oGroupCount + " oLora=" + oLoraRank);
            }
            if (nHead % oGroupCount != 0) {
                throw new IllegalArgumentException("DSV4 nHead " + nHead + " is not divisible by oGroupCount " + oGroupCount);
            }
            if (ropeDim < 0 || ropeDim > headDim || ropeDim % 2 != 0) {
                throw new IllegalArgumentException("DSV4 rope dim " + ropeDim + " must be even and within " + headDim);
            }
            if (compressRatio != 0 && !(compressRopeBase > 0f)) {
                throw new IllegalArgumentException("DSV4 compressed layers need compressRopeBase > 0, got " + compressRopeBase);
            }
            if (!(kqScale > 0f) || Float.isInfinite(kqScale)) {
                throw new IllegalArgumentException("DSV4 kqScale must be a positive finite value, got " + kqScale);
            }
            return new Dsv4Spec(this);
        }
    }
}

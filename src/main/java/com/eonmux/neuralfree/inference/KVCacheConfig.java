package com.eonmux.neuralfree.inference;

import java.util.Objects;

public final class KVCacheConfig {
    private final int numLayers;
    private final int numHeads;
    private final int headDim;
    private final int maxSeqLen;
    private final int batchSize;
    
    public KVCacheConfig(int numLayers, int numHeads, int headDim, int maxSeqLen, int batchSize) {
        if (numLayers <= 0) {
            throw new IllegalArgumentException("numLayers must be positive: " + numLayers);
        }
        if (numHeads <= 0) {
            throw new IllegalArgumentException("numHeads must be positive: " + numHeads);
        }
        if (headDim <= 0) {
            throw new IllegalArgumentException("headDim must be positive: " + headDim);
        }
        if (maxSeqLen <= 0) {
            throw new IllegalArgumentException("maxSeqLen must be positive: " + maxSeqLen);
        }
        if (batchSize <= 0) {
            throw new IllegalArgumentException("batchSize must be positive: " + batchSize);
        }
        
        this.numLayers = numLayers;
        this.numHeads = numHeads;
        this.headDim = headDim;
        this.maxSeqLen = maxSeqLen;
        this.batchSize = batchSize;
    }
    
    public int numLayers() {
        return numLayers;
    }
    
    public int numHeads() {
        return numHeads;
    }
    
    public int headDim() {
        return headDim;
    }
    
    public int maxSeqLen() {
        return maxSeqLen;
    }
    
    public int batchSize() {
        return batchSize;
    }
    
    public long memoryBytes() {
        return 2L * numLayers * batchSize * numHeads * maxSeqLen * headDim * 4L;
    }
    
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        KVCacheConfig that = (KVCacheConfig) o;
        return numLayers == that.numLayers &&
               numHeads == that.numHeads &&
               headDim == that.headDim &&
               maxSeqLen == that.maxSeqLen &&
               batchSize == that.batchSize;
    }
    
    @Override
    public int hashCode() {
        return Objects.hash(numLayers, numHeads, headDim, maxSeqLen, batchSize);
    }
    
    @Override
    public String toString() {
        return "KVCacheConfig{" +
            "numLayers=" + numLayers +
            ", numHeads=" + numHeads +
            ", headDim=" + headDim +
            ", maxSeqLen=" + maxSeqLen +
            ", batchSize=" + batchSize +
            ", memoryMB=" + (memoryBytes() / (1024 * 1024)) +
            '}';
    }
}

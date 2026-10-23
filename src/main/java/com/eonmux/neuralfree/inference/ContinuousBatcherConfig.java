package com.eonmux.neuralfree.inference;

import java.util.Objects;

public final class ContinuousBatcherConfig {
    private final int maxBatchSize;
    private final int maxQueueSize;
    private final int maxSeqLen;
    private final int maxTokensPerBatch;
    private final boolean enablePreemption;
    private final int preemptionThreshold;
    
    public ContinuousBatcherConfig(int maxBatchSize, int maxQueueSize, int maxSeqLen,
                                   int maxTokensPerBatch, boolean enablePreemption, 
                                   int preemptionThreshold) {
        if (maxBatchSize <= 0) {
            throw new IllegalArgumentException("maxBatchSize must be positive: " + maxBatchSize);
        }
        if (maxQueueSize <= 0) {
            throw new IllegalArgumentException("maxQueueSize must be positive: " + maxQueueSize);
        }
        if (maxSeqLen <= 0) {
            throw new IllegalArgumentException("maxSeqLen must be positive: " + maxSeqLen);
        }
        if (maxTokensPerBatch <= 0) {
            throw new IllegalArgumentException("maxTokensPerBatch must be positive: " + maxTokensPerBatch);
        }
        if (preemptionThreshold < 0) {
            throw new IllegalArgumentException("preemptionThreshold must be non-negative: " + preemptionThreshold);
        }
        
        this.maxBatchSize = maxBatchSize;
        this.maxQueueSize = maxQueueSize;
        this.maxSeqLen = maxSeqLen;
        this.maxTokensPerBatch = maxTokensPerBatch;
        this.enablePreemption = enablePreemption;
        this.preemptionThreshold = preemptionThreshold;
    }
    
    public static ContinuousBatcherConfig defaults() {
        return new ContinuousBatcherConfig(32, 1000, 4096, 8192, true, 80);
    }
    
    public static ContinuousBatcherConfig highThroughput() {
        return new ContinuousBatcherConfig(64, 2000, 8192, 16384, false, 0);
    }
    
    public static ContinuousBatcherConfig lowLatency() {
        return new ContinuousBatcherConfig(8, 100, 2048, 4096, true, 50);
    }
    
    public int maxBatchSize() {
        return maxBatchSize;
    }
    
    public int maxQueueSize() {
        return maxQueueSize;
    }
    
    public int maxSeqLen() {
        return maxSeqLen;
    }
    
    public int maxTokensPerBatch() {
        return maxTokensPerBatch;
    }
    
    public boolean enablePreemption() {
        return enablePreemption;
    }
    
    public int preemptionThreshold() {
        return preemptionThreshold;
    }
    
    public static Builder builder() {
        return new Builder();
    }
    
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        ContinuousBatcherConfig that = (ContinuousBatcherConfig) o;
        return maxBatchSize == that.maxBatchSize &&
               maxQueueSize == that.maxQueueSize &&
               maxSeqLen == that.maxSeqLen &&
               maxTokensPerBatch == that.maxTokensPerBatch &&
               enablePreemption == that.enablePreemption &&
               preemptionThreshold == that.preemptionThreshold;
    }
    
    @Override
    public int hashCode() {
        return Objects.hash(maxBatchSize, maxQueueSize, maxSeqLen, 
            maxTokensPerBatch, enablePreemption, preemptionThreshold);
    }
    
    @Override
    public String toString() {
        return "ContinuousBatcherConfig{" +
            "maxBatchSize=" + maxBatchSize +
            ", maxQueueSize=" + maxQueueSize +
            ", maxSeqLen=" + maxSeqLen +
            ", maxTokensPerBatch=" + maxTokensPerBatch +
            ", enablePreemption=" + enablePreemption +
            ", preemptionThreshold=" + preemptionThreshold +
            '}';
    }
    
    public static final class Builder {
        private int maxBatchSize = 32;
        private int maxQueueSize = 1000;
        private int maxSeqLen = 4096;
        private int maxTokensPerBatch = 8192;
        private boolean enablePreemption = true;
        private int preemptionThreshold = 80;
        
        public Builder maxBatchSize(int maxBatchSize) {
            this.maxBatchSize = maxBatchSize;
            return this;
        }
        
        public Builder maxQueueSize(int maxQueueSize) {
            this.maxQueueSize = maxQueueSize;
            return this;
        }
        
        public Builder maxSeqLen(int maxSeqLen) {
            this.maxSeqLen = maxSeqLen;
            return this;
        }
        
        public Builder maxTokensPerBatch(int maxTokensPerBatch) {
            this.maxTokensPerBatch = maxTokensPerBatch;
            return this;
        }
        
        public Builder enablePreemption(boolean enable) {
            this.enablePreemption = enable;
            return this;
        }
        
        public Builder preemptionThreshold(int threshold) {
            this.preemptionThreshold = threshold;
            return this;
        }
        
        public ContinuousBatcherConfig build() {
            return new ContinuousBatcherConfig(maxBatchSize, maxQueueSize, maxSeqLen,
                maxTokensPerBatch, enablePreemption, preemptionThreshold);
        }
    }
}

package com.eonmux.neuralfree.acceleration.pipeline.schedule;

import java.util.Objects;

public final class PipelineConfig {
    private final boolean enablePrefetch;
    private final int prefetchDepth;
    private final boolean enableKernelCache;
    private final boolean enableMemoryPool;

    private PipelineConfig(Builder builder) {
        this.enablePrefetch = builder.enablePrefetch;
        this.prefetchDepth = builder.prefetchDepth;
        this.enableKernelCache = builder.enableKernelCache;
        this.enableMemoryPool = builder.enableMemoryPool;
    }

    public boolean enablePrefetch() {
        return enablePrefetch;
    }

    public int prefetchDepth() {
        return prefetchDepth;
    }

    public boolean enableKernelCache() {
        return enableKernelCache;
    }

    public boolean enableMemoryPool() {
        return enableMemoryPool;
    }

    public int ringBufferSlotCount() {
        if (!enablePrefetch) {
            return 1;
        }
        return prefetchDepth + 1;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static PipelineConfig defaultConfig() {
        return builder().build();
    }

    public static final class Builder {
        private boolean enablePrefetch = true;
        private int prefetchDepth = 1;
        private boolean enableKernelCache = true;
        private boolean enableMemoryPool = true;

        public Builder enablePrefetch(boolean enablePrefetch) {
            this.enablePrefetch = enablePrefetch;
            return this;
        }

        public Builder prefetchDepth(int prefetchDepth) {
            if (prefetchDepth < 1 || prefetchDepth > 3) {
                throw new IllegalArgumentException("prefetchDepth must be between 1 and 3");
            }
            this.prefetchDepth = prefetchDepth;
            return this;
        }

        public Builder enableKernelCache(boolean enableKernelCache) {
            this.enableKernelCache = enableKernelCache;
            return this;
        }

        public Builder enableMemoryPool(boolean enableMemoryPool) {
            this.enableMemoryPool = enableMemoryPool;
            return this;
        }

        public PipelineConfig build() {
            return new PipelineConfig(this);
        }
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        PipelineConfig that = (PipelineConfig) o;
        return enablePrefetch == that.enablePrefetch &&
               prefetchDepth == that.prefetchDepth &&
               enableKernelCache == that.enableKernelCache &&
               enableMemoryPool == that.enableMemoryPool;
    }

    @Override
    public int hashCode() {
        return Objects.hash(enablePrefetch, prefetchDepth, enableKernelCache, enableMemoryPool);
    }

    @Override
    public String toString() {
        return "PipelineConfig{" +
               "enablePrefetch=" + enablePrefetch +
               ", prefetchDepth=" + prefetchDepth +
               ", enableKernelCache=" + enableKernelCache +
               ", enableMemoryPool=" + enableMemoryPool +
               '}';
    }
}

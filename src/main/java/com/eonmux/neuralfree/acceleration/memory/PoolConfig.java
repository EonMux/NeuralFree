package com.eonmux.neuralfree.acceleration.memory;

public final class PoolConfig {
    public static final long DEFAULT_MIN_BLOCK_SIZE = 4 * 1024;
    public static final long DEFAULT_BLOCK_SIZE = 1024 * 1024;
    public static final double DEFAULT_VRAM_USAGE_RATIO = 0.8;

    private final long maxPoolSizeBytes;
    private final long minBlockSizeBytes;
    private final long defaultBlockSizeBytes;

    private PoolConfig(Builder builder) {
        this.maxPoolSizeBytes = builder.maxPoolSizeBytes;
        this.minBlockSizeBytes = builder.minBlockSizeBytes;
        this.defaultBlockSizeBytes = builder.defaultBlockSizeBytes;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static PoolConfig defaultConfig(long totalVramBytes) {
        return builder()
                .maxPoolSizeBytes((long) (totalVramBytes * DEFAULT_VRAM_USAGE_RATIO))
                .minBlockSizeBytes(DEFAULT_MIN_BLOCK_SIZE)
                .defaultBlockSizeBytes(DEFAULT_BLOCK_SIZE)
                .build();
    }

    public long getMaxPoolSizeBytes() {
        return maxPoolSizeBytes;
    }

    public long getMinBlockSizeBytes() {
        return minBlockSizeBytes;
    }

    public long getDefaultBlockSizeBytes() {
        return defaultBlockSizeBytes;
    }

    public static final class Builder {
        private long maxPoolSizeBytes = 1024 * 1024 * 1024;
        private long minBlockSizeBytes = DEFAULT_MIN_BLOCK_SIZE;
        private long defaultBlockSizeBytes = DEFAULT_BLOCK_SIZE;

        public Builder maxPoolSizeBytes(long maxPoolSizeBytes) {
            this.maxPoolSizeBytes = maxPoolSizeBytes;
            return this;
        }

        public Builder minBlockSizeBytes(long minBlockSizeBytes) {
            this.minBlockSizeBytes = minBlockSizeBytes;
            return this;
        }

        public Builder defaultBlockSizeBytes(long defaultBlockSizeBytes) {
            this.defaultBlockSizeBytes = defaultBlockSizeBytes;
            return this;
        }

        public PoolConfig build() {
            return new PoolConfig(this);
        }
    }
}

package com.eonmux.neuralfree.nn.generative;

public final class SamplingConfig {
    private final float temperature;
    private final int topK;
    private final float topP;
    private final float repetitionPenalty;
    private final long seed;
    private final int beamWidth;

    private SamplingConfig(Builder builder) {
        this.temperature = builder.temperature;
        this.topK = builder.topK;
        this.topP = builder.topP;
        this.repetitionPenalty = builder.repetitionPenalty;
        this.seed = builder.seed;
        this.beamWidth = builder.beamWidth;
    }

    public static SamplingConfig defaults() {
        return builder().build();
    }

    public static Builder builder() {
        return new Builder();
    }

    public float temperature() {
        return temperature;
    }

    public int topK() {
        return topK;
    }

    public float topP() {
        return topP;
    }

    public float repetitionPenalty() {
        return repetitionPenalty;
    }

    public long seed() {
        return seed;
    }

    public int beamWidth() {
        return beamWidth;
    }

    @Override
    public String toString() {
        return "SamplingConfig{" +
            "temperature=" + temperature +
            ", topK=" + topK +
            ", topP=" + topP +
            ", repetitionPenalty=" + repetitionPenalty +
            ", seed=" + seed +
            ", beamWidth=" + beamWidth +
            '}';
    }

    public static final class Builder {
        private float temperature = 1.0f;
        private int topK = 50;
        private float topP = 0.9f;
        private float repetitionPenalty = 1.0f;
        private long seed = System.currentTimeMillis();
        private int beamWidth = 1;

        public Builder temperature(float temperature) {
            if (temperature <= 0) {
                throw new IllegalArgumentException("Temperature must be positive: " + temperature);
            }
            this.temperature = temperature;
            return this;
        }

        public Builder topK(int topK) {
            if (topK < 0) {
                throw new IllegalArgumentException("TopK must be non-negative: " + topK);
            }
            this.topK = topK;
            return this;
        }

        public Builder topP(float topP) {
            if (topP < 0.0f || topP > 1.0f) {
                throw new IllegalArgumentException("TopP must be in [0.0, 1.0]: " + topP);
            }
            this.topP = topP;
            return this;
        }

        public Builder repetitionPenalty(float repetitionPenalty) {
            if (repetitionPenalty < 1.0f) {
                throw new IllegalArgumentException("RepetitionPenalty must be >= 1.0: " + repetitionPenalty);
            }
            this.repetitionPenalty = repetitionPenalty;
            return this;
        }

        public Builder seed(long seed) {
            this.seed = seed;
            return this;
        }

        public Builder beamWidth(int beamWidth) {
            if (beamWidth < 1) {
                throw new IllegalArgumentException("BeamWidth must be at least 1: " + beamWidth);
            }
            this.beamWidth = beamWidth;
            return this;
        }

        public SamplingConfig build() {
            return new SamplingConfig(this);
        }
    }
}

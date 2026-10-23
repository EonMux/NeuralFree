package com.eonmux.neuralfree.inference;

import java.util.Objects;

public final class SpeculativeDecoderConfig {
    private final int draftSteps;
    private final float acceptThreshold;
    private final float temperature;
    
    public SpeculativeDecoderConfig(int draftSteps, float acceptThreshold, float temperature) {
        if (draftSteps <= 0) {
            throw new IllegalArgumentException("draftSteps must be positive: " + draftSteps);
        }
        if (acceptThreshold < 0.0f || acceptThreshold > 1.0f) {
            throw new IllegalArgumentException("acceptThreshold must be in [0, 1]: " + acceptThreshold);
        }
        if (temperature < 0.0f) {
            throw new IllegalArgumentException("temperature must be non-negative: " + temperature);
        }
        
        this.draftSteps = draftSteps;
        this.acceptThreshold = acceptThreshold;
        this.temperature = temperature;
    }
    
    public static SpeculativeDecoderConfig defaults() {
        return new SpeculativeDecoderConfig(4, 0.8f, 1.0f);
    }
    
    public static SpeculativeDecoderConfig aggressive() {
        return new SpeculativeDecoderConfig(8, 0.7f, 1.0f);
    }
    
    public static SpeculativeDecoderConfig conservative() {
        return new SpeculativeDecoderConfig(2, 0.9f, 0.8f);
    }
    
    public int draftSteps() {
        return draftSteps;
    }
    
    public float acceptThreshold() {
        return acceptThreshold;
    }
    
    public float temperature() {
        return temperature;
    }
    
    public SpeculativeDecoderConfig withDraftSteps(int newDraftSteps) {
        return new SpeculativeDecoderConfig(newDraftSteps, acceptThreshold, temperature);
    }
    
    public SpeculativeDecoderConfig withAcceptThreshold(float newThreshold) {
        return new SpeculativeDecoderConfig(draftSteps, newThreshold, temperature);
    }
    
    public SpeculativeDecoderConfig withTemperature(float newTemperature) {
        return new SpeculativeDecoderConfig(draftSteps, acceptThreshold, newTemperature);
    }
    
    public static Builder builder() {
        return new Builder();
    }
    
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        SpeculativeDecoderConfig that = (SpeculativeDecoderConfig) o;
        return draftSteps == that.draftSteps &&
               Float.compare(that.acceptThreshold, acceptThreshold) == 0 &&
               Float.compare(that.temperature, temperature) == 0;
    }
    
    @Override
    public int hashCode() {
        return Objects.hash(draftSteps, acceptThreshold, temperature);
    }
    
    @Override
    public String toString() {
        return "SpeculativeDecoderConfig{" +
            "draftSteps=" + draftSteps +
            ", acceptThreshold=" + acceptThreshold +
            ", temperature=" + temperature +
            '}';
    }
    
    public static final class Builder {
        private int draftSteps = 4;
        private float acceptThreshold = 0.8f;
        private float temperature = 1.0f;
        
        public Builder draftSteps(int draftSteps) {
            this.draftSteps = draftSteps;
            return this;
        }
        
        public Builder acceptThreshold(float threshold) {
            this.acceptThreshold = threshold;
            return this;
        }
        
        public Builder temperature(float temperature) {
            this.temperature = temperature;
            return this;
        }
        
        public SpeculativeDecoderConfig build() {
            return new SpeculativeDecoderConfig(draftSteps, acceptThreshold, temperature);
        }
    }
}

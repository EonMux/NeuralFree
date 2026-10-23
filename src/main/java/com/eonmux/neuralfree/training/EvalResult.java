package com.eonmux.neuralfree.training;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

public final class EvalResult {
    private final float loss;
    private final int numSamples;
    private final Map<String, Float> metrics;

    private EvalResult(Builder builder) {
        this.loss = builder.loss;
        this.numSamples = builder.numSamples;
        this.metrics = Collections.unmodifiableMap(new HashMap<>(builder.metrics));
    }

    public float loss() {
        return loss;
    }

    public int numSamples() {
        return numSamples;
    }

    public Map<String, Float> metrics() {
        return metrics;
    }

    public Float metric(String name) {
        return metrics.get(name);
    }

    public float accuracy() {
        return metrics.getOrDefault("accuracy", 0.0f);
    }

    public static Builder builder() {
        return new Builder();
    }

    public static final class Builder {
        private float loss = 0.0f;
        private int numSamples = 0;
        private Map<String, Float> metrics = new HashMap<>();

        public Builder loss(float loss) {
            this.loss = loss;
            return this;
        }

        public Builder numSamples(int numSamples) {
            this.numSamples = numSamples;
            return this;
        }

        public Builder metric(String name, float value) {
            this.metrics.put(name, value);
            return this;
        }

        public Builder metrics(Map<String, Float> metrics) {
            this.metrics.putAll(metrics);
            return this;
        }

        public Builder accuracy(float accuracy) {
            this.metrics.put("accuracy", accuracy);
            return this;
        }

        public EvalResult build() {
            return new EvalResult(this);
        }
    }

    @Override
    public String toString() {
        return "EvalResult{" +
            "loss=" + loss +
            ", numSamples=" + numSamples +
            ", metrics=" + metrics +
            '}';
    }
}

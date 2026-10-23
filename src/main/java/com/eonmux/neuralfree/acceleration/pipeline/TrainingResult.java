package com.eonmux.neuralfree.acceleration.pipeline;

import java.util.Arrays;
import java.util.Objects;

/**
 * The results of one {@link PipelineScheduler#executeTrainingIteration} step: the output rows of the forward pass
 * (before the update), the updated weights and biases, and the gradients the update applied, in the
 * {@code NeuralNetworkData.flattenWeights} and {@code flattenBiases} layouts. Immutable; arrays are copied in and
 * out.
 */
public final class TrainingResult {
    private final float[] forwardOutput;
    private final float[] updatedWeights;
    private final float[] updatedBiases;
    private final float[] gradientWeights;
    private final float[] gradientBiases;
    private final long executionTimeNanos;

    public TrainingResult(float[] forwardOutput, float[] updatedWeights, float[] updatedBiases,
                          float[] gradientWeights, float[] gradientBiases, long executionTimeNanos) {
        this.forwardOutput = forwardOutput != null ? forwardOutput.clone() : null;
        this.updatedWeights = updatedWeights != null ? updatedWeights.clone() : null;
        this.updatedBiases = updatedBiases != null ? updatedBiases.clone() : null;
        this.gradientWeights = gradientWeights != null ? gradientWeights.clone() : null;
        this.gradientBiases = gradientBiases != null ? gradientBiases.clone() : null;
        this.executionTimeNanos = executionTimeNanos;
    }

    public float[] forwardOutput() {
        return forwardOutput != null ? forwardOutput.clone() : null;
    }

    public float[] updatedWeights() {
        return updatedWeights != null ? updatedWeights.clone() : null;
    }

    public float[] updatedBiases() {
        return updatedBiases != null ? updatedBiases.clone() : null;
    }

    public float[] gradientWeights() {
        return gradientWeights != null ? gradientWeights.clone() : null;
    }

    public float[] gradientBiases() {
        return gradientBiases != null ? gradientBiases.clone() : null;
    }

    public long executionTimeNanos() {
        return executionTimeNanos;
    }

    public double executionTimeMillis() {
        return executionTimeNanos / 1_000_000.0;
    }

    public static Builder builder() {
        return new Builder();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        TrainingResult that = (TrainingResult) o;
        return executionTimeNanos == that.executionTimeNanos &&
               Arrays.equals(forwardOutput, that.forwardOutput) &&
               Arrays.equals(updatedWeights, that.updatedWeights) &&
               Arrays.equals(updatedBiases, that.updatedBiases) &&
               Arrays.equals(gradientWeights, that.gradientWeights) &&
               Arrays.equals(gradientBiases, that.gradientBiases);
    }

    @Override
    public int hashCode() {
        int result = Objects.hash(executionTimeNanos);
        result = 31 * result + Arrays.hashCode(forwardOutput);
        result = 31 * result + Arrays.hashCode(updatedWeights);
        result = 31 * result + Arrays.hashCode(updatedBiases);
        result = 31 * result + Arrays.hashCode(gradientWeights);
        result = 31 * result + Arrays.hashCode(gradientBiases);
        return result;
    }

    @Override
    public String toString() {
        return "TrainingResult{" +
               "forwardOutputSize=" + (forwardOutput != null ? forwardOutput.length : 0) +
               ", updatedWeightsSize=" + (updatedWeights != null ? updatedWeights.length : 0) +
               ", updatedBiasesSize=" + (updatedBiases != null ? updatedBiases.length : 0) +
               ", gradientWeightsSize=" + (gradientWeights != null ? gradientWeights.length : 0) +
               ", gradientBiasesSize=" + (gradientBiases != null ? gradientBiases.length : 0) +
               ", executionTimeMillis=" + executionTimeMillis() +
               '}';
    }

    public static final class Builder {
        private float[] forwardOutput;
        private float[] updatedWeights;
        private float[] updatedBiases;
        private float[] gradientWeights;
        private float[] gradientBiases;
        private long executionTimeNanos;

        public Builder forwardOutput(float[] forwardOutput) {
            this.forwardOutput = forwardOutput;
            return this;
        }

        public Builder updatedWeights(float[] updatedWeights) {
            this.updatedWeights = updatedWeights;
            return this;
        }

        public Builder updatedBiases(float[] updatedBiases) {
            this.updatedBiases = updatedBiases;
            return this;
        }

        public Builder gradientWeights(float[] gradientWeights) {
            this.gradientWeights = gradientWeights;
            return this;
        }

        public Builder gradientBiases(float[] gradientBiases) {
            this.gradientBiases = gradientBiases;
            return this;
        }

        public Builder executionTimeNanos(long executionTimeNanos) {
            this.executionTimeNanos = executionTimeNanos;
            return this;
        }

        public TrainingResult build() {
            return new TrainingResult(forwardOutput, updatedWeights, updatedBiases,
                                      gradientWeights, gradientBiases, executionTimeNanos);
        }
    }
}

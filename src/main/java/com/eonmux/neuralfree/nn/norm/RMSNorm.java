package com.eonmux.neuralfree.nn.norm;

import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.tensor.NormOps;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;

import java.util.Objects;

public final class RMSNorm extends Module {
    private final int normalizedShape;
    private final float epsilon;
    private final boolean hasWeight;
    
    private Parameter weight;

    public RMSNorm(int normalizedShape) {
        this(normalizedShape, 1e-8f, true);
    }

    public RMSNorm(int normalizedShape, float epsilon, boolean hasWeight) {
        if (normalizedShape <= 0) {
            throw new IllegalArgumentException("normalizedShape must be positive: " + normalizedShape);
        }
        if (epsilon <= 0) {
            throw new IllegalArgumentException("epsilon must be positive: " + epsilon);
        }
        
        this.normalizedShape = normalizedShape;
        this.epsilon = epsilon;
        this.hasWeight = hasWeight;
    }

    /** Creates the weight (ones) on {@code backend} when the module has one; runs under no-grad. */
    public void initialize(ComputeBackend backend) {
        Objects.requireNonNull(backend, "Backend cannot be null");

        if (hasWeight) {
            try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
                Tensor weightTensor = Tensor.ones(backend, Shape.vector(normalizedShape));
                this.weight = registerParameter("weight", new Parameter(weightTensor, "weight"));
            }
        }
    }

    /**
     * Normalizes each row of the last dimension by its root mean square, {@code rms = sqrt(sum(x^2) / n + eps)},
     * then scales elementwise by the weight: {@code x / rms * weight}.
     *
     * <p>Computed by {@link NormOps#rmsNorm}: on the train kernels when every operand is FLOAT32 on the input's
     * backend, else on its CPU reference, within the reduction tolerance class of the
     * {@linkplain com.eonmux.neuralfree.tensor.gpu numerical contract}. The call records one autograd node when grad
     * mode is on and the input or the weight requires grad, so the backward reaches the input and the weight.</p>
     *
     * @throws IllegalArgumentException when the input is a scalar or its last dim differs from
     *                                  {@code normalizedShape}
     */
    @Override
    public Tensor forward(Tensor input) {
        ensureInitialized();
        Objects.requireNonNull(input, "Input cannot be null");

        Shape inputShape = input.shape();
        int inputRank = inputShape.rank();
        if (inputRank < 1) {
            throw new IllegalArgumentException("Input must have at least 1 dimension, got rank " + inputRank);
        }
        int lastDim = inputShape.dim(inputRank - 1);
        if (lastDim != normalizedShape) {
            throw new IllegalArgumentException("Expected last dimension to be " + normalizedShape + ", got " + lastDim);
        }

        return NormOps.rmsNorm(input, hasWeight ? weight.tensor() : null, epsilon);
    }

    private void ensureInitialized() {
        if (hasWeight && weight == null) {
            throw new IllegalStateException("RMSNorm not initialized. Call initialize(backend) first.");
        }
    }

    public int normalizedShape() {
        return normalizedShape;
    }

    public float epsilon() {
        return epsilon;
    }

    public boolean hasWeight() {
        return hasWeight;
    }

    public Parameter weight() {
        if (!hasWeight) {
            throw new IllegalStateException("This RMSNorm has no weight parameter");
        }
        ensureInitialized();
        return weight;
    }

    @Override
    public String toString() {
        return "RMSNorm{" +
            "normalizedShape=" + normalizedShape +
            ", epsilon=" + epsilon +
            ", hasWeight=" + hasWeight +
            ", initialized=" + (weight != null || !hasWeight) +
            '}';
    }
}

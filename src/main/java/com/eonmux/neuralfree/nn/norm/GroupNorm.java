package com.eonmux.neuralfree.nn.norm;

import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.tensor.NormOps;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;

import java.util.Objects;

public final class GroupNorm extends Module {
    private final int numGroups;
    private final int numChannels;
    private final float epsilon;
    private final boolean affine;
    
    private Parameter weight;
    private Parameter bias;

    public GroupNorm(int numGroups, int numChannels) {
        this(numGroups, numChannels, 1e-5f, true);
    }

    public GroupNorm(int numGroups, int numChannels, float epsilon, boolean affine) {
        if (numGroups <= 0) {
            throw new IllegalArgumentException("numGroups must be positive: " + numGroups);
        }
        if (numChannels <= 0) {
            throw new IllegalArgumentException("numChannels must be positive: " + numChannels);
        }
        if (numChannels % numGroups != 0) {
            throw new IllegalArgumentException("numChannels (" + numChannels + ") must be divisible by numGroups (" + numGroups + ")");
        }
        if (epsilon <= 0) {
            throw new IllegalArgumentException("epsilon must be positive: " + epsilon);
        }
        
        this.numGroups = numGroups;
        this.numChannels = numChannels;
        this.epsilon = epsilon;
        this.affine = affine;
    }

    /** Creates the affine parameters (weight ones, bias zeros) on {@code backend}; runs under no-grad. */
    public void initialize(ComputeBackend backend) {
        Objects.requireNonNull(backend, "Backend cannot be null");

        if (affine) {
            try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
                Tensor weightTensor = Tensor.ones(backend, Shape.vector(numChannels));
                this.weight = registerParameter("weight", new Parameter(weightTensor, "weight"));

                Tensor biasTensor = Tensor.zeros(backend, Shape.vector(numChannels));
                this.bias = registerParameter("bias", new Parameter(biasTensor, "bias"));
            }
        }
    }

    /**
     * Normalizes an input {@code [B, C, *]} per (batch, group): the mean and biased variance over the
     * {@code (C / G) * prod(*)} elements of each group normalize them, followed by the per-channel affine step
     * {@code weight * y + bias}. The element of channel {@code c} at spatial index {@code s} is
     * {@code x[(b * C + c) * S + s]}.
     *
     * <p>Computed by {@link NormOps#groupNorm}: on the train kernels when every operand is FLOAT32 on the input's
     * backend, else on its CPU reference, within the reduction tolerance class of the
     * {@linkplain com.eonmux.neuralfree.tensor.gpu numerical contract}. The call records one autograd node when grad
     * mode is on and the input or an affine parameter requires grad, so the backward reaches the input, the weight and
     * the bias.</p>
     *
     * @throws IllegalArgumentException when the input has rank below 2 or another channel count
     */
    @Override
    public Tensor forward(Tensor input) {
        ensureInitialized();
        Objects.requireNonNull(input, "Input cannot be null");

        Shape inputShape = input.shape();
        if (inputShape.rank() < 2) {
            throw new IllegalArgumentException("GroupNorm expects at least 2D input, got rank " + inputShape.rank());
        }
        int channels = inputShape.dim(1);
        if (channels != this.numChannels) {
            throw new IllegalArgumentException("Expected " + this.numChannels + " channels, got " + channels);
        }

        Tensor gamma = affine ? weight.tensor() : null;
        Tensor beta = affine ? bias.tensor() : null;
        return NormOps.groupNorm(input, numGroups, gamma, beta, epsilon);
    }

    private void ensureInitialized() {
        if (affine && weight == null) {
            throw new IllegalStateException("GroupNorm not initialized. Call initialize(backend) first.");
        }
    }

    public int numGroups() {
        return numGroups;
    }

    public int numChannels() {
        return numChannels;
    }

    public float epsilon() {
        return epsilon;
    }

    public boolean hasAffine() {
        return affine;
    }

    public Parameter weight() {
        if (!affine) {
            throw new IllegalStateException("This GroupNorm has no affine parameters");
        }
        ensureInitialized();
        return weight;
    }

    public Parameter bias() {
        if (!affine) {
            throw new IllegalStateException("This GroupNorm has no affine parameters");
        }
        ensureInitialized();
        return bias;
    }

    @Override
    public String toString() {
        return "GroupNorm{" +
            "numGroups=" + numGroups +
            ", numChannels=" + numChannels +
            ", epsilon=" + epsilon +
            ", affine=" + affine +
            ", initialized=" + (weight != null || !affine) +
            '}';
    }
}

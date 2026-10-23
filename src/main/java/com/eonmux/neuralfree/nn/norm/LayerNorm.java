package com.eonmux.neuralfree.nn.norm;

import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.tensor.NormOps;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;

import java.util.Arrays;
import java.util.Objects;

public final class LayerNorm extends Module {
    private final int[] normalizedShape;
    private final float epsilon;
    private final boolean elementwiseAffine;
    
    private Parameter weight;
    private Parameter bias;

    public LayerNorm(int normalizedShape) {
        this(new int[]{normalizedShape});
    }

    public LayerNorm(int[] normalizedShape) {
        this(normalizedShape, 1e-5f, true);
    }

    public LayerNorm(int normalizedShape, float epsilon, boolean elementwiseAffine) {
        this(new int[]{normalizedShape}, epsilon, elementwiseAffine);
    }

    public LayerNorm(int[] normalizedShape, float epsilon, boolean elementwiseAffine) {
        Objects.requireNonNull(normalizedShape, "normalizedShape cannot be null");
        if (normalizedShape.length == 0) {
            throw new IllegalArgumentException("normalizedShape cannot be empty");
        }
        for (int dim : normalizedShape) {
            if (dim <= 0) {
                throw new IllegalArgumentException("All dimensions in normalizedShape must be positive: " + Arrays.toString(normalizedShape));
            }
        }
        if (epsilon <= 0) {
            throw new IllegalArgumentException("epsilon must be positive: " + epsilon);
        }
        
        this.normalizedShape = normalizedShape.clone();
        this.epsilon = epsilon;
        this.elementwiseAffine = elementwiseAffine;
    }

    /** Creates the affine parameters (weight ones, bias zeros) on {@code backend}; runs under no-grad. */
    public void initialize(ComputeBackend backend) {
        Objects.requireNonNull(backend, "Backend cannot be null");

        if (elementwiseAffine) {
            try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
                Tensor weightTensor = Tensor.ones(backend, Shape.of(normalizedShape));
                this.weight = registerParameter("weight", new Parameter(weightTensor, "weight"));

                Tensor biasTensor = Tensor.zeros(backend, Shape.of(normalizedShape));
                this.bias = registerParameter("bias", new Parameter(biasTensor, "bias"));
            }
        }
    }

    /**
     * Normalizes each row of the trailing {@code normalizedShape} elements: the row mean and biased variance
     * normalize it, followed by the elementwise affine step {@code weight * y + bias} over the normalized shape.
     *
     * <p>Computed by {@link NormOps#layerNorm}: on the train kernels when every operand is FLOAT32 on the input's
     * backend, else on its CPU reference, within the reduction tolerance class of the
     * {@linkplain com.eonmux.neuralfree.tensor.gpu numerical contract}. The call records one autograd node when grad
     * mode is on and the input or an affine parameter requires grad, so the backward reaches the input, the weight and
     * the bias.</p>
     *
     * @throws IllegalArgumentException when the input's trailing dims differ from {@code normalizedShape}, or a row
     *                                  of the normalized shape exceeds {@code 2^31-1} elements
     */
    @Override
    public Tensor forward(Tensor input) {
        ensureInitialized();
        Objects.requireNonNull(input, "Input cannot be null");

        Shape inputShape = input.shape();
        int inputRank = inputShape.rank();
        int normRank = normalizedShape.length;
        if (inputRank < normRank) {
            throw new IllegalArgumentException("Input rank (" + inputRank + ") must be at least " + normRank);
        }
        for (int i = 0; i < normRank; i++) {
            int inputDim = inputShape.dim(inputRank - normRank + i);
            if (inputDim != normalizedShape[i]) {
                throw new IllegalArgumentException("Input shape " + inputShape + " does not match normalizedShape " +
                    Arrays.toString(normalizedShape) + " at trailing dimension " + i);
            }
        }

        Tensor gamma = elementwiseAffine ? weight.tensor() : null;
        Tensor beta = elementwiseAffine ? bias.tensor() : null;
        return NormOps.layerNorm(input, rowSize(), gamma, beta, epsilon);
    }

    /**
     * The element count of one normalized row, the product of {@code normalizedShape}.
     *
     * @throws IllegalArgumentException when it exceeds {@code 2^31-1}, the limit of the normalization kernels and of
     *                                  the host arrays of their CPU reference
     */
    private int rowSize() {
        long size = 1;
        for (int dim : normalizedShape) {
            size *= dim;
            if (size > Integer.MAX_VALUE) {
                throw new IllegalArgumentException("normalizedShape " + Arrays.toString(normalizedShape)
                        + " has more than 2^31-1 elements per row, the limit of the normalization kernels");
            }
        }
        return (int) size;
    }

    private void ensureInitialized() {
        if (elementwiseAffine && weight == null) {
            throw new IllegalStateException("LayerNorm not initialized. Call initialize(backend) first.");
        }
        if (!elementwiseAffine && weight != null) {
            throw new IllegalStateException("LayerNorm not initialized properly for elementwiseAffine=false.");
        }
    }

    public int[] normalizedShape() {
        return normalizedShape.clone();
    }

    public float epsilon() {
        return epsilon;
    }

    public boolean hasElementwiseAffine() {
        return elementwiseAffine;
    }

    public Parameter weight() {
        if (!elementwiseAffine) {
            throw new IllegalStateException("This LayerNorm has no affine parameters");
        }
        ensureInitialized();
        return weight;
    }

    public Parameter bias() {
        if (!elementwiseAffine) {
            throw new IllegalStateException("This LayerNorm has no affine parameters");
        }
        ensureInitialized();
        return bias;
    }

    @Override
    public String toString() {
        return "LayerNorm{" +
            "normalizedShape=" + Arrays.toString(normalizedShape) +
            ", epsilon=" + epsilon +
            ", elementwiseAffine=" + elementwiseAffine +
            ", initialized=" + (weight != null || !elementwiseAffine) +
            '}';
    }
}

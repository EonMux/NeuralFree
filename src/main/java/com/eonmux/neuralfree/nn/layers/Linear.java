package com.eonmux.neuralfree.nn.layers;

import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;

import java.util.Objects;
import java.util.Random;

public final class Linear extends Module {
    private final int inFeatures;
    private final int outFeatures;
    private final boolean useBias;
    
    private Parameter weight;
    private Parameter bias;

    public Linear(int inFeatures, int outFeatures) {
        this(inFeatures, outFeatures, true);
    }

    public Linear(int inFeatures, int outFeatures, boolean useBias) {
        if (inFeatures <= 0) {
            throw new IllegalArgumentException("inFeatures must be positive: " + inFeatures);
        }
        if (outFeatures <= 0) {
            throw new IllegalArgumentException("outFeatures must be positive: " + outFeatures);
        }
        
        this.inFeatures = inFeatures;
        this.outFeatures = outFeatures;
        this.useBias = useBias;
    }

    public void initialize(ComputeBackend backend) {
        initialize(backend, new Random());
    }

    /** Initializes with weights drawn from N(0, 2 / (inFeatures + outFeatures)) and zero bias. */
    public void initialize(ComputeBackend backend, Random random) {
        initializeNormal(backend, random, (float) Math.sqrt(2.0 / (inFeatures + outFeatures)));
    }

    /** Initializes with weights drawn from N(0, 2 / inFeatures) and zero bias. */
    public void initializeKaiming(ComputeBackend backend, Random random) {
        initializeNormal(backend, random, (float) Math.sqrt(2.0 / inFeatures));
    }

    /** Initializes with weights drawn from N(0, 2 / (inFeatures + outFeatures)) and zero bias. */
    public void initializeXavier(ComputeBackend backend, Random random) {
        initializeNormal(backend, random, (float) Math.sqrt(2.0 / (inFeatures + outFeatures)));
    }

    /**
     * Registers the weight {@code randn(seed) * std}, with the seed drawn from {@code random}, and a zero bias when the
     * layer has one. Runs under {@link ComputationGraph#noGrad()}, so initialization records nothing; the draw and the
     * scaling temporaries are closed, and the parameters own their tensors (autograd leaves).
     */
    private void initializeNormal(ComputeBackend backend, Random random, float std) {
        Objects.requireNonNull(backend, "Backend cannot be null");
        Objects.requireNonNull(random, "Random cannot be null");

        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            Tensor weightTensor;
            try (Tensor normal = Tensor.randn(backend, Shape.matrix(outFeatures, inFeatures), random.nextLong());
                 Tensor scale = Tensor.scalar(backend, std)) {
                weightTensor = TensorOps.mul(normal, scale);
            }
            this.weight = registerParameter("weight", new Parameter(weightTensor, "weight"));

            if (useBias) {
                Tensor biasTensor = Tensor.zeros(backend, Shape.vector(outFeatures));
                this.bias = registerParameter("bias", new Parameter(biasTensor, "bias"));
            }
        }
    }

    /**
     * {@code input @ weight^T + bias} over the last dimension of {@code input}. The transposed weight is a recorded
     * view ({@code TransposeGrad}) and the product and bias sum record {@code MatmulGrad} and {@code AddGrad}, so in
     * grad mode the gradient reaches the weight and bias leaves.
     */
    @Override
    public Tensor forward(Tensor input) {
        ensureInitialized();
        Objects.requireNonNull(input, "Input cannot be null");
        
        Shape inputShape = input.shape();
        if (inputShape.rank() < 1) {
            throw new IllegalArgumentException("Input must have at least 1 dimension, got rank " + inputShape.rank());
        }
        
        int lastDim = inputShape.dim(inputShape.rank() - 1);
        if (lastDim != inFeatures) {
            throw new IllegalArgumentException("Input last dimension must be " + inFeatures + 
                ", got " + lastDim);
        }
        
        Tensor product;
        try (Tensor weightT = weight.tensor().transpose()) {
            product = TensorOps.matmul(input, weightT);
        }
        if (!useBias) {
            return product;
        }
        try {
            return TensorOps.add(product, bias.tensor());
        } finally {
            product.close();
        }
    }

    private void ensureInitialized() {
        if (weight == null) {
            throw new IllegalStateException("Linear layer not initialized. Call initialize(backend) first.");
        }
    }

    public int inFeatures() {
        return inFeatures;
    }

    public int outFeatures() {
        return outFeatures;
    }

    public boolean hasBias() {
        return useBias;
    }

    public Parameter weight() {
        ensureInitialized();
        return weight;
    }

    public Parameter bias() {
        if (!useBias) {
            throw new IllegalStateException("This Linear layer has no bias");
        }
        ensureInitialized();
        return bias;
    }

    @Override
    public String toString() {
        return "Linear{" +
            "inFeatures=" + inFeatures +
            ", outFeatures=" + outFeatures +
            ", useBias=" + useBias +
            ", initialized=" + (weight != null) +
            '}';
    }
}

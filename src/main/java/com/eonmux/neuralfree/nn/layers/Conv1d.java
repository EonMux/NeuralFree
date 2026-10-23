package com.eonmux.neuralfree.nn.layers;

import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.tensor.ConvOps;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;

import java.util.Objects;
import java.util.Random;

public final class Conv1d extends Module {
    private final int inChannels;
    private final int outChannels;
    private final int kernelSize;
    private final int stride;
    private final int padding;
    private final int dilation;
    private final int groups;
    private final boolean useBias;
    
    private Parameter weight;
    private Parameter bias;

    public Conv1d(int inChannels, int outChannels, int kernelSize) {
        this(inChannels, outChannels, kernelSize, 1, 0);
    }

    public Conv1d(int inChannels, int outChannels, int kernelSize, int stride, int padding) {
        this(inChannels, outChannels, kernelSize, stride, padding, 1, 1, true);
    }

    public Conv1d(int inChannels, int outChannels, int kernelSize, int stride, 
                  int padding, int dilation, int groups, boolean useBias) {
        if (inChannels <= 0) {
            throw new IllegalArgumentException("inChannels must be positive: " + inChannels);
        }
        if (outChannels <= 0) {
            throw new IllegalArgumentException("outChannels must be positive: " + outChannels);
        }
        if (kernelSize <= 0) {
            throw new IllegalArgumentException("kernelSize must be positive: " + kernelSize);
        }
        if (stride <= 0) {
            throw new IllegalArgumentException("stride must be positive: " + stride);
        }
        if (padding < 0) {
            throw new IllegalArgumentException("padding must be non-negative: " + padding);
        }
        if (dilation <= 0) {
            throw new IllegalArgumentException("dilation must be positive: " + dilation);
        }
        if (groups <= 0) {
            throw new IllegalArgumentException("groups must be positive: " + groups);
        }
        if (inChannels % groups != 0) {
            throw new IllegalArgumentException("inChannels (" + inChannels + ") must be divisible by groups (" + groups + ")");
        }
        if (outChannels % groups != 0) {
            throw new IllegalArgumentException("outChannels (" + outChannels + ") must be divisible by groups (" + groups + ")");
        }
        
        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.kernelSize = kernelSize;
        this.stride = stride;
        this.padding = padding;
        this.dilation = dilation;
        this.groups = groups;
        this.useBias = useBias;
    }

    public void initialize(ComputeBackend backend) {
        initialize(backend, new Random());
    }

    /**
     * Registers the weight {@code [outChannels, inChannels/groups, K]} drawn from N(0, 2 / fanIn) with
     * {@code fanIn = inChannels/groups * K}, and a zero bias when the layer has one. Runs under
     * {@link ComputationGraph#noGrad()}; the parameters own their tensors (autograd leaves).
     */
    public void initialize(ComputeBackend backend, Random random) {
        Objects.requireNonNull(backend, "Backend cannot be null");
        Objects.requireNonNull(random, "Random cannot be null");
        
        int weightInChannels = inChannels / groups;
        float std = (float) Math.sqrt(2.0 / (weightInChannels * kernelSize));
        
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            Tensor weightTensor = Conv2d.scaledNormal(backend,
                Shape.of(outChannels, weightInChannels, kernelSize),
                random.nextLong(), std);
            this.weight = registerParameter("weight", new Parameter(weightTensor, "weight"));
            
            if (useBias) {
                this.bias = registerParameter("bias",
                    new Parameter(Tensor.zeros(backend, Shape.vector(outChannels)), "bias"));
            }
        }
    }

    /**
     * 1-D convolution of {@code input [B, inChannels, L]} through {@link ConvOps#conv1d}: output
     * {@code [B, outChannels, OL]} with {@code OL = (L + 2*padding - dilation*(K-1) - 1) / stride + 1}. In grad mode
     * the op records {@code Conv1dGrad}, which routes gradients to the input, the weight and the bias.
     *
     * @throws IllegalArgumentException when the input is not 3-D, its channel count is not {@code inChannels}, or the
     *                                  output length is negative
     */
    @Override
    public Tensor forward(Tensor input) {
        ensureInitialized();
        Objects.requireNonNull(input, "Input cannot be null");
        
        Shape inputShape = input.shape();
        if (inputShape.rank() != 3) {
            throw new IllegalArgumentException("Conv1d expects 3D input [B, C, L], got rank " + inputShape.rank());
        }
        
        int inC = inputShape.dim(1);
        if (inC != inChannels) {
            throw new IllegalArgumentException("Input channels (" + inC + ") does not match layer inChannels (" + inChannels + ")");
        }
        
        return ConvOps.conv1d(input, weight.tensor(), useBias ? bias.tensor() : null,
            kernelSize, stride, padding, dilation, groups);
    }

    private void ensureInitialized() {
        if (weight == null) {
            throw new IllegalStateException("Conv1d layer not initialized. Call initialize(backend) first.");
        }
    }

    public int inChannels() {
        return inChannels;
    }

    public int outChannels() {
        return outChannels;
    }

    public int kernelSize() {
        return kernelSize;
    }

    public int stride() {
        return stride;
    }

    public int padding() {
        return padding;
    }

    public int dilation() {
        return dilation;
    }

    public int groups() {
        return groups;
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
            throw new IllegalStateException("This Conv1d layer has no bias");
        }
        ensureInitialized();
        return bias;
    }

    @Override
    public String toString() {
        return "Conv1d{" +
            "inChannels=" + inChannels +
            ", outChannels=" + outChannels +
            ", kernelSize=" + kernelSize +
            ", stride=" + stride +
            ", padding=" + padding +
            ", dilation=" + dilation +
            ", groups=" + groups +
            ", useBias=" + useBias +
            ", initialized=" + (weight != null) +
            '}';
    }
}

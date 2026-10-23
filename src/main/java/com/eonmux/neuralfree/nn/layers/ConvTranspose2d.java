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

public final class ConvTranspose2d extends Module {
    private final int inChannels;
    private final int outChannels;
    private final int[] kernelSize;
    private final int[] stride;
    private final int[] padding;
    private final int[] outputPadding;
    private final int[] dilation;
    private final int groups;
    private final boolean useBias;
    
    private Parameter weight;
    private Parameter bias;

    public ConvTranspose2d(int inChannels, int outChannels, int[] kernelSize) {
        this(inChannels, outChannels, kernelSize, new int[]{1, 1}, new int[]{0, 0});
    }

    public ConvTranspose2d(int inChannels, int outChannels, int[] kernelSize, int[] stride, int[] padding) {
        this(inChannels, outChannels, kernelSize, stride, padding, new int[]{0, 0}, new int[]{1, 1}, 1, true);
    }

    public ConvTranspose2d(int inChannels, int outChannels, int[] kernelSize, int[] stride, 
                           int[] padding, int[] outputPadding, int[] dilation, int groups, boolean useBias) {
        if (inChannels <= 0) {
            throw new IllegalArgumentException("inChannels must be positive: " + inChannels);
        }
        if (outChannels <= 0) {
            throw new IllegalArgumentException("outChannels must be positive: " + outChannels);
        }
        if (kernelSize == null || kernelSize.length != 2 || kernelSize[0] <= 0 || kernelSize[1] <= 0) {
            throw new IllegalArgumentException("kernelSize must be int[2] with positive values");
        }
        if (stride == null || stride.length != 2 || stride[0] <= 0 || stride[1] <= 0) {
            throw new IllegalArgumentException("stride must be int[2] with positive values");
        }
        if (padding == null || padding.length != 2 || padding[0] < 0 || padding[1] < 0) {
            throw new IllegalArgumentException("padding must be int[2] with non-negative values");
        }
        if (outputPadding == null || outputPadding.length != 2 || outputPadding[0] < 0 || outputPadding[1] < 0) {
            throw new IllegalArgumentException("outputPadding must be int[2] with non-negative values");
        }
        if (dilation == null || dilation.length != 2 || dilation[0] <= 0 || dilation[1] <= 0) {
            throw new IllegalArgumentException("dilation must be int[2] with positive values");
        }
        ConvOps.requireTransposeOutputPadding(outputPadding, stride, dilation);
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
        this.kernelSize = kernelSize.clone();
        this.stride = stride.clone();
        this.padding = padding.clone();
        this.outputPadding = outputPadding.clone();
        this.dilation = dilation.clone();
        this.groups = groups;
        this.useBias = useBias;
    }

    public void initialize(ComputeBackend backend) {
        initialize(backend, new Random());
    }

    /**
     * Registers the weight {@code [inChannels, outChannels/groups, KH, KW]} drawn from
     * N(0, 2 / (inChannels * KH * KW)), and a zero bias when the layer has one. Runs under
     * {@link ComputationGraph#noGrad()}; the parameters own their tensors (autograd leaves).
     */
    public void initialize(ComputeBackend backend, Random random) {
        Objects.requireNonNull(backend, "Backend cannot be null");
        Objects.requireNonNull(random, "Random cannot be null");
        
        int weightOutChannels = outChannels / groups;
        float std = (float) Math.sqrt(2.0 / (inChannels * kernelSize[0] * kernelSize[1]));
        
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            Tensor weightTensor = Conv2d.scaledNormal(backend,
                Shape.of(inChannels, weightOutChannels, kernelSize[0], kernelSize[1]),
                random.nextLong(), std);
            this.weight = registerParameter("weight", new Parameter(weightTensor, "weight"));
            
            if (useBias) {
                this.bias = registerParameter("bias",
                    new Parameter(Tensor.zeros(backend, Shape.vector(outChannels)), "bias"));
            }
        }
    }

    /**
     * Transposed 2-D convolution of {@code input [B, inChannels, H, W]} through {@link ConvOps#convTranspose2d}
     * (PyTorch semantics): output {@code [B, outChannels, OH, OW]} with
     * {@code OH = (H-1)*SH - 2*PH + DH*(KH-1) + outputPadding[0] + 1} (and its W twin). In grad mode the op records
     * {@code ConvTranspose2dGrad}, which routes gradients to the input, the weight and the bias.
     *
     * @throws IllegalArgumentException when the input is not 4-D, its channel count is not {@code inChannels}, or the
     *                                  output size is negative
     */
    @Override
    public Tensor forward(Tensor input) {
        ensureInitialized();
        Objects.requireNonNull(input, "Input cannot be null");
        
        Shape inputShape = input.shape();
        if (inputShape.rank() != 4) {
            throw new IllegalArgumentException("ConvTranspose2d expects 4D input [B, C, H, W], got rank " + inputShape.rank());
        }
        
        int inC = inputShape.dim(1);
        if (inC != inChannels) {
            throw new IllegalArgumentException("Input channels (" + inC + ") does not match layer inChannels (" + inChannels + ")");
        }
        
        return ConvOps.convTranspose2d(input, weight.tensor(), useBias ? bias.tensor() : null,
            kernelSize, stride, padding, outputPadding, dilation, groups);
    }

    private void ensureInitialized() {
        if (weight == null) {
            throw new IllegalStateException("ConvTranspose2d layer not initialized. Call initialize(backend) first.");
        }
    }

    public int inChannels() {
        return inChannels;
    }

    public int outChannels() {
        return outChannels;
    }

    public int[] kernelSize() {
        return kernelSize.clone();
    }

    public int[] stride() {
        return stride.clone();
    }

    public int[] padding() {
        return padding.clone();
    }

    public int[] outputPadding() {
        return outputPadding.clone();
    }

    public int[] dilation() {
        return dilation.clone();
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
            throw new IllegalStateException("This ConvTranspose2d layer has no bias");
        }
        ensureInitialized();
        return bias;
    }

    @Override
    public String toString() {
        return "ConvTranspose2d{" +
            "inChannels=" + inChannels +
            ", outChannels=" + outChannels +
            ", kernelSize=[" + kernelSize[0] + ", " + kernelSize[1] + "]" +
            ", stride=[" + stride[0] + ", " + stride[1] + "]" +
            ", padding=[" + padding[0] + ", " + padding[1] + "]" +
            ", outputPadding=[" + outputPadding[0] + ", " + outputPadding[1] + "]" +
            ", dilation=[" + dilation[0] + ", " + dilation[1] + "]" +
            ", groups=" + groups +
            ", useBias=" + useBias +
            ", initialized=" + (weight != null) +
            '}';
    }
}

package com.eonmux.neuralfree.nn.vision;

import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.layers.Conv2d;
import com.eonmux.neuralfree.nn.norm.BatchNorm2d;
import com.eonmux.neuralfree.nn.activations.ReLU;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;

import java.util.Objects;

/**
 * Depthwise separable convolution: a depthwise {@code kernelSize} convolution ({@code groups = inChannels}, no bias),
 * optional batch normalization and ReLU, then a pointwise 1x1 convolution, optional batch normalization and ReLU.
 * Built from the {@link Conv2d}, {@link BatchNorm2d} and {@link ReLU} modules, so in grad mode the output carries
 * their recorded graph back to every parameter and to the input.
 */
public final class DepthwiseSeparableConv extends Module {
    private final int inChannels;
    private final int outChannels;
    private final int[] kernelSize;
    private final int[] stride;
    private final int[] padding;
    private final boolean useBias;
    private final boolean useBatchNorm;
    
    private Conv2d depthwiseConv;
    private BatchNorm2d depthwiseBn;
    private ReLU activation1;
    private Conv2d pointwiseConv;
    private BatchNorm2d pointwiseBn;
    private ReLU activation2;

    public DepthwiseSeparableConv(int inChannels, int outChannels, int[] kernelSize) {
        this(inChannels, outChannels, kernelSize, new int[]{1, 1}, new int[]{0, 0}, true, true);
    }

    public DepthwiseSeparableConv(int inChannels, int outChannels, int[] kernelSize, 
                                   int[] stride, int[] padding) {
        this(inChannels, outChannels, kernelSize, stride, padding, true, true);
    }

    public DepthwiseSeparableConv(int inChannels, int outChannels, int[] kernelSize, 
                                   int[] stride, int[] padding, boolean useBias, boolean useBatchNorm) {
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
        
        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.kernelSize = kernelSize.clone();
        this.stride = stride.clone();
        this.padding = padding.clone();
        this.useBias = useBias;
        this.useBatchNorm = useBatchNorm;
        
        this.depthwiseConv = new Conv2d(inChannels, inChannels, kernelSize, stride, padding,
                                         new int[]{1, 1}, inChannels, false);
        this.depthwiseBn = useBatchNorm ? new BatchNorm2d(inChannels) : null;
        this.activation1 = new ReLU();
        this.pointwiseConv = new Conv2d(inChannels, outChannels, new int[]{1, 1},
                                         new int[]{1, 1}, new int[]{0, 0},
                                         new int[]{1, 1}, 1, useBias);
        this.pointwiseBn = useBatchNorm ? new BatchNorm2d(outChannels) : null;
        this.activation2 = new ReLU();
        
        registerModule("depthwise", depthwiseConv);
        if (depthwiseBn != null) {
            registerModule("depthwise_bn", depthwiseBn);
        }
        registerModule("pointwise", pointwiseConv);
        if (pointwiseBn != null) {
            registerModule("pointwise_bn", pointwiseBn);
        }
    }

    public void initialize(ComputeBackend backend) {
        Objects.requireNonNull(backend, "Backend cannot be null");
        
        depthwiseConv.initialize(backend);
        if (depthwiseBn != null) {
            depthwiseBn.initialize(backend);
        }
        pointwiseConv.initialize(backend);
        if (pointwiseBn != null) {
            pointwiseBn.initialize(backend);
        }
    }

    @Override
    public Tensor forward(Tensor input) {
        ensureInitialized();
        Objects.requireNonNull(input, "Input cannot be null");
        
        Tensor output = depthwiseConv.forward(input);
        if (depthwiseBn != null) {
            output = forwardStep(output, input, depthwiseBn::forward);
        }
        output = forwardStep(output, input, activation1::forward);
        output = forwardStep(output, input, pointwiseConv::forward);
        if (pointwiseBn != null) {
            output = forwardStep(output, input, pointwiseBn::forward);
        }
        return forwardStep(output, input, activation2::forward);
    }

    private void ensureInitialized() {
        if (depthwiseConv.weight() == null) {
            throw new IllegalStateException("DepthwiseSeparableConv not initialized. Call initialize(backend) first.");
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

    public boolean hasBias() {
        return useBias;
    }

    public boolean hasBatchNorm() {
        return useBatchNorm;
    }

    public Conv2d depthwiseConv() {
        return depthwiseConv;
    }

    public Conv2d pointwiseConv() {
        return pointwiseConv;
    }

    public BatchNorm2d depthwiseBatchNorm() {
        return depthwiseBn;
    }

    public BatchNorm2d pointwiseBatchNorm() {
        return pointwiseBn;
    }

    @Override
    public String toString() {
        return "DepthwiseSeparableConv{" +
            "inChannels=" + inChannels +
            ", outChannels=" + outChannels +
            ", kernelSize=[" + kernelSize[0] + ", " + kernelSize[1] + "]" +
            ", stride=[" + stride[0] + ", " + stride[1] + "]" +
            ", useBias=" + useBias +
            ", useBatchNorm=" + useBatchNorm +
            '}';
    }
}

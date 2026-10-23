package com.eonmux.neuralfree.nn.vision;

import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.layers.Conv2d;
import com.eonmux.neuralfree.nn.norm.BatchNorm2d;
import com.eonmux.neuralfree.nn.activations.ReLU;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;

import java.util.Objects;
import java.util.Random;

/**
 * ResNet basic block: two 3x3 convolutions (the first strided) with batch normalization and ReLU, added to the
 * shortcut (a strided 1x1 projection with batch normalization when the shape changes) before the final ReLU. Built
 * from recorded modules and {@link TensorOps#add}, so in grad mode the output carries the graph back to every
 * parameter and to the input.
 */
public final class ResidualBlock extends Module {
    private final int inChannels;
    private final int outChannels;
    private final int[] stride;
    private final boolean useProjection;
    
    private Conv2d conv1;
    private BatchNorm2d bn1;
    private ReLU activation1;
    private Conv2d conv2;
    private BatchNorm2d bn2;
    private Conv2d projection;
    private BatchNorm2d projectionBn;
    private ReLU finalActivation;

    public ResidualBlock(int inChannels, int outChannels) {
        this(inChannels, outChannels, new int[]{1, 1});
    }

    public ResidualBlock(int inChannels, int outChannels, int[] stride) {
        if (inChannels <= 0) {
            throw new IllegalArgumentException("inChannels must be positive: " + inChannels);
        }
        if (outChannels <= 0) {
            throw new IllegalArgumentException("outChannels must be positive: " + outChannels);
        }
        if (stride == null || stride.length != 2 || stride[0] <= 0 || stride[1] <= 0) {
            throw new IllegalArgumentException("stride must be int[2] with positive values");
        }
        
        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.stride = stride.clone();
        this.useProjection = (inChannels != outChannels) || (stride[0] != 1 || stride[1] != 1);
        
        this.conv1 = new Conv2d(inChannels, outChannels, new int[]{3, 3}, stride, new int[]{1, 1});
        this.bn1 = new BatchNorm2d(outChannels);
        this.activation1 = new ReLU();
        this.conv2 = new Conv2d(outChannels, outChannels, new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1});
        this.bn2 = new BatchNorm2d(outChannels);
        
        if (useProjection) {
            this.projection = new Conv2d(inChannels, outChannels, new int[]{1, 1}, stride, new int[]{0, 0});
            this.projectionBn = new BatchNorm2d(outChannels);
        } else {
            this.projection = null;
            this.projectionBn = null;
        }
        this.finalActivation = new ReLU();
        
        registerModule("conv1", conv1);
        registerModule("bn1", bn1);
        registerModule("conv2", conv2);
        registerModule("bn2", bn2);
        if (projection != null) {
            registerModule("projection", projection);
            registerModule("projection_bn", projectionBn);
        }
    }

    public void initialize(ComputeBackend backend) {
        initialize(backend, new Random());
    }

    public void initialize(ComputeBackend backend, Random random) {
        Objects.requireNonNull(backend, "Backend cannot be null");
        Objects.requireNonNull(random, "Random cannot be null");
        
        conv1.initialize(backend, random);
        bn1.initialize(backend);
        conv2.initialize(backend, random);
        bn2.initialize(backend);
        
        if (projection != null) {
            projection.initialize(backend, random);
            projectionBn.initialize(backend);
        }
    }

    @Override
    public Tensor forward(Tensor input) {
        ensureInitialized();
        Objects.requireNonNull(input, "Input cannot be null");
        
        Tensor out = conv1.forward(input);
        out = forwardStep(out, input, bn1::forward);
        out = forwardStep(out, input, activation1::forward);
        out = forwardStep(out, input, conv2::forward);
        out = forwardStep(out, input, bn2::forward);

        Tensor identity = input;
        Tensor sum;
        try {
            if (useProjection && projection != null) {
                identity = projection.forward(input);
                identity = forwardStep(identity, input, projectionBn::forward);
            }
            sum = TensorOps.add(out, identity);
        } finally {
            releaseIntermediate(out, input);
            releaseIntermediate(identity, input);
        }
        return forwardStep(sum, input, finalActivation::forward);
    }

    private void ensureInitialized() {
        if (conv1.weight() == null) {
            throw new IllegalStateException("ResidualBlock not initialized. Call initialize(backend) first.");
        }
    }

    public int inChannels() {
        return inChannels;
    }

    public int outChannels() {
        return outChannels;
    }

    public int[] stride() {
        return stride.clone();
    }

    public boolean hasProjection() {
        return useProjection;
    }

    public Conv2d conv1() {
        return conv1;
    }

    public Conv2d conv2() {
        return conv2;
    }

    public BatchNorm2d bn1() {
        return bn1;
    }

    public BatchNorm2d bn2() {
        return bn2;
    }

    public Conv2d projection() {
        return projection;
    }

    @Override
    public String toString() {
        return "ResidualBlock{" +
            "inChannels=" + inChannels +
            ", outChannels=" + outChannels +
            ", stride=[" + stride[0] + ", " + stride[1] + "]" +
            ", useProjection=" + useProjection +
            '}';
    }
}

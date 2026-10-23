package com.eonmux.neuralfree.nn.vision;

import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.layers.Conv2d;
import com.eonmux.neuralfree.nn.layers.Linear;
import com.eonmux.neuralfree.nn.norm.BatchNorm2d;
import com.eonmux.neuralfree.nn.activations.ReLU;
import com.eonmux.neuralfree.nn.activations.Sigmoid;
import com.eonmux.neuralfree.nn.layers.AdaptiveAvgPool2d;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;

import java.util.Objects;
import java.util.Random;

/**
 * ResNet bottleneck block: 1x1 reduce, 3x3 (strided) and 1x1 expand convolutions with batch normalization, an
 * optional squeeze-and-excitation gate on the 3x3 output, and a residual shortcut (a strided 1x1 projection when the
 * shape changes). The SE gate pools to {@code [batch, channels]} ({@link AdaptiveAvgPool2d} 1x1, which records the
 * adaptive average pool), runs its two linear layers and scales the feature map by a broadcast multiply, so the whole
 * block stays on the device. Every step is a recorded module, view or {@link TensorOps} operation, so in grad mode the
 * output carries the graph back to every parameter (the SE layers included) and to the input. Every intermediate is
 * closed once consumed: closing releases only this handle, since the graph routes gradients by node and each recorded
 * operation keeps its own handles of what its backward reads.
 */
public final class BottleneckBlock extends Module {
    private final int inChannels;
    private final int midChannels;
    private final int outChannels;
    private final int[] stride;
    private final boolean useProjection;
    private final boolean useSe;
    private final int seReduction;
    
    private Conv2d conv1;
    private BatchNorm2d bn1;
    private ReLU activation1;
    private Conv2d conv2;
    private BatchNorm2d bn2;
    private ReLU activation2;
    private Conv2d conv3;
    private BatchNorm2d bn3;
    
    private Conv2d projection;
    private BatchNorm2d projectionBn;
    
    private AdaptiveAvgPool2d sePool;
    private Linear seFc1;
    private ReLU seActivation;
    private Linear seFc2;
    private Sigmoid seSigmoid;
    
    private ReLU finalActivation;

    public BottleneckBlock(int inChannels, int midChannels, int outChannels) {
        this(inChannels, midChannels, outChannels, new int[]{1, 1}, false, 16);
    }

    public BottleneckBlock(int inChannels, int midChannels, int outChannels, int[] stride) {
        this(inChannels, midChannels, outChannels, stride, false, 16);
    }

    public BottleneckBlock(int inChannels, int midChannels, int outChannels, int[] stride, 
                           boolean useSe, int seReduction) {
        if (inChannels <= 0) {
            throw new IllegalArgumentException("inChannels must be positive: " + inChannels);
        }
        if (midChannels <= 0) {
            throw new IllegalArgumentException("midChannels must be positive: " + midChannels);
        }
        if (outChannels <= 0) {
            throw new IllegalArgumentException("outChannels must be positive: " + outChannels);
        }
        if (stride == null || stride.length != 2 || stride[0] <= 0 || stride[1] <= 0) {
            throw new IllegalArgumentException("stride must be int[2] with positive values");
        }
        if (seReduction <= 0) {
            throw new IllegalArgumentException("seReduction must be positive: " + seReduction);
        }
        
        this.inChannels = inChannels;
        this.midChannels = midChannels;
        this.outChannels = outChannels;
        this.stride = stride.clone();
        this.useProjection = (inChannels != outChannels) || (stride[0] != 1 || stride[1] != 1);
        this.useSe = useSe;
        this.seReduction = seReduction;
        
        this.conv1 = new Conv2d(inChannels, midChannels, new int[]{1, 1});
        this.bn1 = new BatchNorm2d(midChannels);
        this.activation1 = new ReLU();
        
        this.conv2 = new Conv2d(midChannels, midChannels, new int[]{3, 3}, stride, new int[]{1, 1});
        this.bn2 = new BatchNorm2d(midChannels);
        this.activation2 = new ReLU();
        
        this.conv3 = new Conv2d(midChannels, outChannels, new int[]{1, 1});
        this.bn3 = new BatchNorm2d(outChannels);
        
        if (useProjection) {
            this.projection = new Conv2d(inChannels, outChannels, new int[]{1, 1}, stride, new int[]{0, 0});
            this.projectionBn = new BatchNorm2d(outChannels);
        } else {
            this.projection = null;
            this.projectionBn = null;
        }
        
        if (useSe) {
            int seChannels = Math.max(1, midChannels / seReduction);
            this.sePool = new AdaptiveAvgPool2d(1, 1);
            this.seFc1 = new Linear(midChannels, seChannels);
            this.seActivation = new ReLU();
            this.seFc2 = new Linear(seChannels, midChannels);
            this.seSigmoid = new Sigmoid();
        } else {
            this.sePool = null;
            this.seFc1 = null;
            this.seActivation = null;
            this.seFc2 = null;
            this.seSigmoid = null;
        }
        
        this.finalActivation = new ReLU();
        
        registerModule("conv1", conv1);
        registerModule("bn1", bn1);
        registerModule("conv2", conv2);
        registerModule("bn2", bn2);
        registerModule("conv3", conv3);
        registerModule("bn3", bn3);
        
        if (projection != null) {
            registerModule("projection", projection);
            registerModule("projection_bn", projectionBn);
        }
        
        if (useSe) {
            registerModule("se_pool", sePool);
            registerModule("se_fc1", seFc1);
            registerModule("se_fc2", seFc2);
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
        conv3.initialize(backend, random);
        bn3.initialize(backend);
        
        if (projection != null) {
            projection.initialize(backend, random);
            projectionBn.initialize(backend);
        }
        
        if (useSe) {
            seFc1.initialize(backend, random);
            seFc2.initialize(backend, random);
        }
    }

    /** Returns a new tensor of the block output; intermediates are closed and {@code input} stays open. */
    @Override
    public Tensor forward(Tensor input) {
        ensureInitialized();
        Objects.requireNonNull(input, "Input cannot be null");
        
        Tensor out = conv1.forward(input);
        try {
            out = advance(out, bn1.forward(out), input);
            out = advance(out, activation1.forward(out), input);
            
            out = advance(out, conv2.forward(out), input);
            out = advance(out, bn2.forward(out), input);
            out = advance(out, activation2.forward(out), input);
            
            if (useSe && sePool != null) {
                out = advance(out, squeezeExcite(out), input);
            }
            
            out = advance(out, conv3.forward(out), input);
            out = advance(out, bn3.forward(out), input);
            
            Tensor identity = useProjection && projection != null ? projectShortcut(input) : input;
            try {
                out = advance(out, TensorOps.add(out, identity), input);
            } finally {
                release(identity, input);
            }
            return advance(out, finalActivation.forward(out), input);
        } catch (RuntimeException e) {
            release(out, input);
            throw e;
        }
    }

    /**
     * {@code features * sigmoid(fc2(relu(fc1(avgpool(features)))))} with the gate broadcast over the spatial dims,
     * as a new tensor; {@code features} stays open.
     */
    private Tensor squeezeExcite(Tensor features) {
        Shape shape = features.shape();
        int batchSize = shape.dim(0);
        int channels = shape.dim(1);
        
        Tensor gate;
        try (Tensor pooled = sePool.forward(features)) {
            gate = pooled.reshape(Shape.matrix(batchSize, channels));
        }
        try {
            gate = advance(gate, seFc1.forward(gate), features);
            gate = advance(gate, seActivation.forward(gate), features);
            gate = advance(gate, seFc2.forward(gate), features);
            gate = advance(gate, seSigmoid.forward(gate), features);
            try (Tensor scale = gate.reshape(Shape.of(batchSize, channels, 1, 1))) {
                return TensorOps.mul(features, scale);
            }
        } finally {
            release(gate, features);
        }
    }

    /** The projected shortcut {@code projectionBn(projection(input))} as a new tensor. */
    private Tensor projectShortcut(Tensor input) {
        Tensor projected = projection.forward(input);
        try {
            return advance(projected, projectionBn.forward(projected), input);
        } catch (RuntimeException e) {
            release(projected, input);
            throw e;
        }
    }

    /** {@code next}, after closing {@code current} unless it is {@code next} itself or the caller's input. */
    private static Tensor advance(Tensor current, Tensor next, Tensor input) {
        if (current != next) {
            release(current, input);
        }
        return next;
    }

    private static void release(Tensor t, Tensor input) {
        if (t != input) {
            t.close();
        }
    }

    private void ensureInitialized() {
        if (conv1.weight() == null) {
            throw new IllegalStateException("BottleneckBlock not initialized. Call initialize(backend) first.");
        }
    }

    public int inChannels() {
        return inChannels;
    }

    public int midChannels() {
        return midChannels;
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

    public boolean hasSe() {
        return useSe;
    }

    public int seReduction() {
        return seReduction;
    }

    public Conv2d conv1() {
        return conv1;
    }

    public Conv2d conv2() {
        return conv2;
    }

    public Conv2d conv3() {
        return conv3;
    }

    public Conv2d projection() {
        return projection;
    }

    @Override
    public String toString() {
        return "BottleneckBlock{" +
            "inChannels=" + inChannels +
            ", midChannels=" + midChannels +
            ", outChannels=" + outChannels +
            ", stride=[" + stride[0] + ", " + stride[1] + "]" +
            ", useProjection=" + useProjection +
            ", useSe=" + useSe +
            '}';
    }
}

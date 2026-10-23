package com.eonmux.neuralfree.nn.vision;

import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.ModuleList;
import com.eonmux.neuralfree.nn.layers.Conv2d;
import com.eonmux.neuralfree.nn.layers.MaxPool2d;
import com.eonmux.neuralfree.nn.layers.AdaptiveAvgPool2d;
import com.eonmux.neuralfree.nn.layers.Linear;
import com.eonmux.neuralfree.nn.norm.BatchNorm2d;
import com.eonmux.neuralfree.nn.activations.ReLU;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;

import java.util.Objects;
import java.util.Random;

/**
 * ResNet-18/34 (basic residual blocks) and ResNet-50/101/152 (bottleneck blocks) over {@code [batch, 3, H, W]}
 * input: a 7x7 strided stem with batch normalization, ReLU and 3x3 max pooling, four stages of blocks, global average
 * pooling ({@link AdaptiveAvgPool2d} 1x1) and an optional linear classifier. The pooled {@code [batch, channels, 1, 1]}
 * features are flattened by a view, so the forward pass stays on the device. Every step is a recorded module or view,
 * so in grad mode the output carries the graph back to every parameter and to the input. Every intermediate is closed
 * once consumed: closing releases only this handle, since the graph routes gradients by node and each recorded
 * operation keeps its own handles of what its backward reads.
 */
public final class ResNet extends Module {
    public static final int[] RESNET18_DEPTHS = {2, 2, 2, 2};
    public static final int[] RESNET34_DEPTHS = {3, 4, 6, 3};
    public static final int[] RESNET50_DEPTHS = {3, 4, 6, 3};
    public static final int[] RESNET101_DEPTHS = {3, 4, 23, 3};
    public static final int[] RESNET152_DEPTHS = {3, 8, 36, 3};
    
    public static final int[] RESNET18_CHANNELS = {64, 64, 128, 256, 512};
    public static final int[] RESNET50_CHANNELS = {64, 256, 512, 1024, 2048};
    
    private final int numClasses;
    private final boolean useBottleneck;
    private final int[] layerDepths;
    
    private final Conv2d conv1;
    private final BatchNorm2d bn1;
    private final ReLU activation;
    private final MaxPool2d maxpool;
    private final ModuleList layers;
    private final AdaptiveAvgPool2d avgpool;
    private final Linear fc;

    private ResNet(int[] layerDepths, int numClasses, boolean useBottleneck) {
        if (layerDepths == null || layerDepths.length != 4) {
            throw new IllegalArgumentException("layerDepths must be int[4]");
        }
        if (numClasses < 0) {
            throw new IllegalArgumentException("numClasses must be non-negative: " + numClasses);
        }
        
        this.layerDepths = layerDepths.clone();
        this.numClasses = numClasses;
        this.useBottleneck = useBottleneck;
        
        this.conv1 = new Conv2d(3, 64, new int[]{7, 7}, new int[]{2, 2}, new int[]{3, 3});
        this.bn1 = new BatchNorm2d(64);
        this.activation = new ReLU();
        this.maxpool = new MaxPool2d(new int[]{3, 3}, new int[]{2, 2}, new int[]{1, 1});
        
        this.layers = new ModuleList();
        
        this.avgpool = new AdaptiveAvgPool2d(1, 1);
        this.fc = numClasses > 0 ? new Linear(useBottleneck ? 2048 : 512, numClasses) : null;
        
        registerModule("conv1", conv1);
        registerModule("bn1", bn1);
        registerModule("maxpool", maxpool);
        registerModule("layers", layers);
        registerModule("avgpool", avgpool);
        if (fc != null) {
            registerModule("fc", fc);
        }
    }

    public static ResNet resnet18(int numClasses) {
        return new ResNet(RESNET18_DEPTHS, numClasses, false);
    }

    public static ResNet resnet34(int numClasses) {
        return new ResNet(RESNET34_DEPTHS, numClasses, false);
    }

    public static ResNet resnet50(int numClasses) {
        return new ResNet(RESNET50_DEPTHS, numClasses, true);
    }

    public static ResNet resnet101(int numClasses) {
        return new ResNet(RESNET101_DEPTHS, numClasses, true);
    }

    public static ResNet resnet152(int numClasses) {
        return new ResNet(RESNET152_DEPTHS, numClasses, true);
    }

    public void initialize(ComputeBackend backend) {
        initialize(backend, new Random());
    }

    /**
     * Initializes the stem and classifier and builds the residual stages into the module list registered at
     * construction.
     *
     * @throws IllegalStateException when the network is already initialized
     */
    public void initialize(ComputeBackend backend, Random random) {
        Objects.requireNonNull(backend, "Backend cannot be null");
        Objects.requireNonNull(random, "Random cannot be null");
        if (!layers.isEmpty()) {
            throw new IllegalStateException("ResNet is already initialized");
        }
        
        conv1.initialize(backend, random);
        bn1.initialize(backend);
        
        buildLayers(backend, random);
        
        if (fc != null) {
            fc.initialize(backend, random);
        }
    }

    private void buildLayers(ComputeBackend backend, Random random) {
        int inChannels = 64;
        int[] outChannels = useBottleneck ? 
            new int[]{256, 512, 1024, 2048} : 
            new int[]{64, 128, 256, 512};
        int[] midChannels = useBottleneck ? 
            new int[]{64, 128, 256, 512} : 
            new int[]{64, 128, 256, 512};
        
        for (int layerIdx = 0; layerIdx < 4; layerIdx++) {
            int depth = layerDepths[layerIdx];
            int outCh = outChannels[layerIdx];
            int midCh = midChannels[layerIdx];
            int[] stride = (layerIdx == 0) ? new int[]{1, 1} : new int[]{2, 2};
            
            for (int blockIdx = 0; blockIdx < depth; blockIdx++) {
                int[] blockStride = (blockIdx == 0) ? stride : new int[]{1, 1};
                
                Module block;
                if (useBottleneck) {
                    BottleneckBlock bottleneck = new BottleneckBlock(
                        inChannels, midCh, outCh, blockStride, false, 16);
                    bottleneck.initialize(backend, random);
                    block = bottleneck;
                } else {
                    ResidualBlock residual = new ResidualBlock(
                        inChannels, outCh, blockStride);
                    residual.initialize(backend, random);
                    block = residual;
                }
                
                layers.add(block);
                inChannels = outCh;
            }
        }
    }

    /**
     * Returns a new {@code [batch, numClasses]} tensor of logits, or the pooled {@code [batch, channels, 1, 1]}
     * features when the network has no classifier. Intermediates are closed and {@code input} stays open.
     */
    @Override
    public Tensor forward(Tensor input) {
        ensureInitialized();
        Objects.requireNonNull(input, "Input cannot be null");
        
        Shape inputShape = input.shape();
        if (inputShape.rank() != 4) {
            throw new IllegalArgumentException("ResNet expects 4D input [B, C, H, W], got rank " + inputShape.rank());
        }
        
        Tensor out = conv1.forward(input);
        try {
            out = advance(out, bn1.forward(out), input);
            out = advance(out, activation.forward(out), input);
            out = advance(out, maxpool.forward(out), input);
            
            for (Module layer : layers) {
                out = advance(out, layer.forward(out), input);
            }
            
            out = advance(out, avgpool.forward(out), input);
            
            if (fc != null) {
                Shape outShape = out.shape();
                out = advance(out, out.reshape(Shape.matrix(outShape.dim(0), outShape.dim(1))), input);
                out = advance(out, fc.forward(out), input);
            }
            
            return out;
        } catch (RuntimeException e) {
            release(out, input);
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
            throw new IllegalStateException("ResNet not initialized. Call initialize(backend) first.");
        }
    }

    public int numClasses() {
        return numClasses;
    }

    public boolean usesBottleneck() {
        return useBottleneck;
    }

    public int[] layerDepths() {
        return layerDepths.clone();
    }

    public Conv2d conv1() {
        return conv1;
    }

    public BatchNorm2d bn1() {
        return bn1;
    }

    public MaxPool2d maxpool() {
        return maxpool;
    }

    public ModuleList layers() {
        return layers;
    }

    public AdaptiveAvgPool2d avgpool() {
        return avgpool;
    }

    public Linear fc() {
        return fc;
    }

    public int outputChannels() {
        return useBottleneck ? 2048 : 512;
    }

    @Override
    public String toString() {
        String variant;
        if (useBottleneck) {
            if (layerDepths[0] == 3 && layerDepths[1] == 4 && layerDepths[2] == 6 && layerDepths[3] == 3) {
                variant = "ResNet-50";
            } else if (layerDepths[2] == 23) {
                variant = "ResNet-101";
            } else {
                variant = "ResNet-152";
            }
        } else {
            if (layerDepths[0] == 2 && layerDepths[1] == 2 && layerDepths[2] == 2 && layerDepths[3] == 2) {
                variant = "ResNet-18";
            } else {
                variant = "ResNet-34";
            }
        }
        
        return variant + "{" +
            "numClasses=" + numClasses +
            ", useBottleneck=" + useBottleneck +
            ", layers=" + layers.size() +
            '}';
    }
}

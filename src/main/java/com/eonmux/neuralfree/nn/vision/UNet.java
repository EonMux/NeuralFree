package com.eonmux.neuralfree.nn.vision;

import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.ModuleList;
import com.eonmux.neuralfree.nn.Sequential;
import com.eonmux.neuralfree.nn.layers.Conv2d;
import com.eonmux.neuralfree.nn.layers.ConvTranspose2d;
import com.eonmux.neuralfree.nn.layers.MaxPool2d;
import com.eonmux.neuralfree.nn.norm.BatchNorm2d;
import com.eonmux.neuralfree.nn.activations.ReLU;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.Random;

/**
 * U-Net over {@code [batch, inChannels, H, W]} input: an encoder of double-convolution blocks separated by 2x2 max
 * pooling, a decoder that upsamples with 2x2 transposed convolutions and concatenates the center-cropped encoder
 * output of the same level along the channel axis before its block, and a final 1x1 convolution. Crops are slice
 * views and concatenation is {@link TensorOps#cat}, so the forward pass stays on the device. Every step is a recorded
 * module, view or operation, so in grad mode the output carries the graph back to every parameter and to the input,
 * the skip connections included. Every intermediate is closed once consumed: closing releases only this handle, since
 * the graph routes gradients by node and each recorded operation keeps its own handles of what its backward reads.
 */
public final class UNet extends Module {
    private final int inChannels;
    private final int outChannels;
    private final int[] channelDepths;
    private final boolean useBatchNorm;
    
    private final ModuleList encoderBlocks;
    private final ModuleList encoderPools;
    private final ModuleList decoderBlocks;
    private final ModuleList decoderUpsamples;
    private final Conv2d finalConv;

    public UNet(int inChannels, int outChannels) {
        this(inChannels, outChannels, new int[]{64, 128, 256, 512, 1024}, true);
    }

    public UNet(int inChannels, int outChannels, int[] channelDepths, boolean useBatchNorm) {
        if (inChannels <= 0) {
            throw new IllegalArgumentException("inChannels must be positive: " + inChannels);
        }
        if (outChannels <= 0) {
            throw new IllegalArgumentException("outChannels must be positive: " + outChannels);
        }
        if (channelDepths == null || channelDepths.length < 2) {
            throw new IllegalArgumentException("channelDepths must have at least 2 elements");
        }
        
        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.channelDepths = channelDepths.clone();
        this.useBatchNorm = useBatchNorm;
        
        this.encoderBlocks = new ModuleList();
        this.encoderPools = new ModuleList();
        this.decoderBlocks = new ModuleList();
        this.decoderUpsamples = new ModuleList();
        this.finalConv = new Conv2d(channelDepths[0], outChannels, new int[]{1, 1});
        
        registerModule("encoder_blocks", encoderBlocks);
        registerModule("encoder_pools", encoderPools);
        registerModule("decoder_blocks", decoderBlocks);
        registerModule("decoder_upsamples", decoderUpsamples);
        registerModule("final_conv", finalConv);
    }

    public void initialize(ComputeBackend backend) {
        initialize(backend, new Random());
    }

    /**
     * Builds and initializes the encoder and decoder into the module lists registered at construction.
     *
     * @throws IllegalStateException when the network is already initialized
     */
    public void initialize(ComputeBackend backend, Random random) {
        Objects.requireNonNull(backend, "Backend cannot be null");
        Objects.requireNonNull(random, "Random cannot be null");
        if (!encoderBlocks.isEmpty()) {
            throw new IllegalStateException("UNet is already initialized");
        }
        
        buildEncoder(backend, random);
        buildDecoder(backend, random);
        finalConv.initialize(backend, random);
    }

    private void buildEncoder(ComputeBackend backend, Random random) {
        int inCh = inChannels;
        for (int i = 0; i < channelDepths.length; i++) {
            int outCh = channelDepths[i];
            
            Sequential block = createConvBlock(inCh, outCh);
            initializeConvBlock(block, backend, random);
            encoderBlocks.add(block);
            
            if (i < channelDepths.length - 1) {
                MaxPool2d pool = new MaxPool2d(new int[]{2, 2}, new int[]{2, 2}, new int[]{0, 0});
                encoderPools.add(pool);
            }
            
            inCh = outCh;
        }
    }

    private void buildDecoder(ComputeBackend backend, Random random) {
        int currentChannels = channelDepths[channelDepths.length - 1];
        for (int i = channelDepths.length - 2; i >= 0; i--) {
            int skipChannels = channelDepths[i];
            
            ConvTranspose2d upsample = new ConvTranspose2d(
                currentChannels, skipChannels, new int[]{2, 2}, new int[]{2, 2}, new int[]{0, 0});
            upsample.initialize(backend, random);
            decoderUpsamples.add(upsample);
            
            Sequential block = createConvBlock(skipChannels * 2, skipChannels);
            initializeConvBlock(block, backend, random);
            decoderBlocks.add(block);
            
            currentChannels = skipChannels;
        }
    }

    private Sequential createConvBlock(int inChannels, int outChannels) {
        Sequential block = new Sequential();
        
        Conv2d conv1 = new Conv2d(inChannels, outChannels, new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1});
        block.add(conv1);
        
        if (useBatchNorm) {
            BatchNorm2d bn1 = new BatchNorm2d(outChannels);
            block.add(bn1);
        }
        
        ReLU relu1 = new ReLU();
        block.add(relu1);
        
        Conv2d conv2 = new Conv2d(outChannels, outChannels, new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1});
        block.add(conv2);
        
        if (useBatchNorm) {
            BatchNorm2d bn2 = new BatchNorm2d(outChannels);
            block.add(bn2);
        }
        
        ReLU relu2 = new ReLU();
        block.add(relu2);
        
        return block;
    }

    private void initializeConvBlock(Sequential block, ComputeBackend backend, Random random) {
        for (Module module : block.layers()) {
            if (module instanceof Conv2d) {
                ((Conv2d) module).initialize(backend, random);
            } else if (module instanceof BatchNorm2d) {
                ((BatchNorm2d) module).initialize(backend);
            }
        }
    }

    /**
     * Returns a new {@code [batch, outChannels, H', W']} tensor. Encoder outputs are kept as skip connections until
     * their decoder level consumes them; every intermediate is closed and {@code input} stays open.
     */
    @Override
    public Tensor forward(Tensor input) {
        ensureInitialized();
        Objects.requireNonNull(input, "Input cannot be null");
        
        Shape inputShape = input.shape();
        if (inputShape.rank() != 4) {
            throw new IllegalArgumentException("UNet expects 4D input [B, C, H, W], got rank " + inputShape.rank());
        }
        
        List<Tensor> skipConnections = new ArrayList<>();
        Tensor x = input;
        try {
            for (int i = 0; i < encoderBlocks.size(); i++) {
                x = advance(x, encoderBlocks.get(i).forward(x), input, skipConnections);
                if (i < encoderBlocks.size() - 1) {
                    skipConnections.add(x);
                    x = encoderPools.get(i).forward(x);
                }
            }
            
            for (int i = 0; i < decoderBlocks.size(); i++) {
                x = advance(x, decoderUpsamples.get(i).forward(x), input, skipConnections);
                Tensor skip = skipConnections.remove(skipConnections.size() - 1);
                try {
                    x = advance(x, centerCropAndConcat(x, skip), input, skipConnections);
                } finally {
                    release(skip, input);
                }
                x = advance(x, decoderBlocks.get(i).forward(x), input, skipConnections);
            }
            
            return advance(x, finalConv.forward(x), input, skipConnections);
        } catch (RuntimeException e) {
            if (!isSkip(x, skipConnections)) {
                release(x, input);
            }
            for (Tensor skip : skipConnections) {
                release(skip, input);
            }
            throw e;
        }
    }

    /**
     * {@code [x | crop(skip)]} along the channel axis as a new tensor, where the crop is the {@code H x W} window of
     * {@code skip} centered at offset {@code ((skipH - H) / 2, (skipW - W) / 2)} (a slice view).
     *
     * @throws IllegalArgumentException when the batch sizes differ or {@code skip} is smaller than {@code x}
     */
    private static Tensor centerCropAndConcat(Tensor x, Tensor skip) {
        Shape xShape = x.shape();
        Shape skipShape = skip.shape();
        int xH = xShape.dim(2);
        int xW = xShape.dim(3);
        int skipH = skipShape.dim(2);
        int skipW = skipShape.dim(3);
        if (xShape.dim(0) != skipShape.dim(0)) {
            throw new IllegalArgumentException("Skip connection batch " + skipShape.dim(0) + " does not match "
                + xShape.dim(0));
        }
        if (skipH < xH || skipW < xW) {
            throw new IllegalArgumentException("Skip connection " + skipShape + " is smaller than the upsampled "
                + xShape);
        }
        
        int startH = (skipH - xH) / 2;
        int startW = (skipW - xW) / 2;
        try (Tensor rows = skip.slice(2, startH, startH + xH);
             Tensor cropped = rows.slice(3, startW, startW + xW)) {
            return TensorOps.cat(Arrays.asList(x, cropped), 1);
        }
    }

    /** {@code next}, after closing {@code current} unless it is {@code next}, the caller's input or a kept skip. */
    private static Tensor advance(Tensor current, Tensor next, Tensor input, List<Tensor> skips) {
        if (current != next && !isSkip(current, skips)) {
            release(current, input);
        }
        return next;
    }

    private static boolean isSkip(Tensor t, List<Tensor> skips) {
        for (Tensor skip : skips) {
            if (skip == t) {
                return true;
            }
        }
        return false;
    }

    private static void release(Tensor t, Tensor input) {
        if (t != input) {
            t.close();
        }
    }

    private void ensureInitialized() {
        if (encoderBlocks.isEmpty()) {
            throw new IllegalStateException("UNet not initialized. Call initialize(backend) first.");
        }
    }

    public int inChannels() {
        return inChannels;
    }

    public int outChannels() {
        return outChannels;
    }

    public int[] channelDepths() {
        return channelDepths.clone();
    }

    public boolean usesBatchNorm() {
        return useBatchNorm;
    }

    public ModuleList encoderBlocks() {
        return encoderBlocks;
    }

    public ModuleList decoderBlocks() {
        return decoderBlocks;
    }

    public Conv2d finalConv() {
        return finalConv;
    }

    public int depth() {
        return channelDepths.length;
    }

    @Override
    public String toString() {
        return "UNet{" +
            "inChannels=" + inChannels +
            ", outChannels=" + outChannels +
            ", channelDepths=" + java.util.Arrays.toString(channelDepths) +
            ", useBatchNorm=" + useBatchNorm +
            ", depth=" + channelDepths.length +
            '}';
    }
}

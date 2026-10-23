package com.eonmux.neuralfree.nn.layers;

import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.tensor.ConvOps;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;

import java.util.Objects;

public final class AdaptiveAvgPool2d extends Module {
    private final int[] outputSize;

    public AdaptiveAvgPool2d(int outputSize) {
        this(outputSize, outputSize);
    }

    public AdaptiveAvgPool2d(int outputH, int outputW) {
        if (outputH <= 0) {
            throw new IllegalArgumentException("outputH must be positive: " + outputH);
        }
        if (outputW <= 0) {
            throw new IllegalArgumentException("outputW must be positive: " + outputW);
        }
        this.outputSize = new int[]{outputH, outputW};
    }

    public AdaptiveAvgPool2d(int[] outputSize) {
        if (outputSize == null || outputSize.length != 2 || outputSize[0] <= 0 || outputSize[1] <= 0) {
            throw new IllegalArgumentException("outputSize must be int[2] with positive values");
        }
        this.outputSize = outputSize.clone();
    }

    /**
     * Adaptive average pooling of {@code input [B, C, H, W]} to {@code [B, C, outputH, outputW]} through
     * {@link ConvOps#adaptiveAvgPool2d} (PyTorch {@code AdaptiveAvgPool2d} windows: output row {@code oh} averages
     * input rows {@code [floor(oh*H/outputH), ceil((oh+1)*H/outputH))}, columns likewise, so neighbouring windows
     * overlap when the output size does not divide the input size). In grad mode the op records
     * {@code AdaptiveAvgPool2dGrad}, which spreads each output gradient over its window divided by the window's size;
     * an input element in several windows sums their shares.
     *
     * @throws IllegalArgumentException when the input is not 4-D
     */
    @Override
    public Tensor forward(Tensor input) {
        Objects.requireNonNull(input, "Input cannot be null");
        
        Shape inputShape = input.shape();
        if (inputShape.rank() != 4) {
            throw new IllegalArgumentException("AdaptiveAvgPool2d expects 4D input [B, C, H, W], got rank " + inputShape.rank());
        }
        
        return ConvOps.adaptiveAvgPool2d(input, outputSize[0], outputSize[1]);
    }

    public int[] outputSize() {
        return outputSize.clone();
    }

    public int outputH() {
        return outputSize[0];
    }

    public int outputW() {
        return outputSize[1];
    }

    public Shape computeOutputShape(Shape inputShape) {
        if (inputShape.rank() != 4) {
            throw new IllegalArgumentException("Input must be 4D [B, C, H, W]");
        }
        return Shape.of(inputShape.dim(0), inputShape.dim(1), outputSize[0], outputSize[1]);
    }

    @Override
    public String toString() {
        return "AdaptiveAvgPool2d{" +
            "outputSize=[" + outputSize[0] + ", " + outputSize[1] + "]" +
            '}';
    }
}

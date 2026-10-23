package com.eonmux.neuralfree.nn.layers;

import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.tensor.ConvOps;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;

import java.util.Objects;

public final class MaxPool2d extends Module {
    private final int[] kernelSize;
    private final int[] stride;
    private final int[] padding;
    private final int[] dilation;
    private final boolean ceilMode;

    public MaxPool2d(int[] kernelSize) {
        this(kernelSize, null, null);
    }

    public MaxPool2d(int[] kernelSize, int[] stride, int[] padding) {
        this(kernelSize, stride, padding, new int[]{1, 1}, false);
    }

    public MaxPool2d(int[] kernelSize, int[] stride, int[] padding, int[] dilation, boolean ceilMode) {
        if (kernelSize == null || kernelSize.length != 2 || kernelSize[0] <= 0 || kernelSize[1] <= 0) {
            throw new IllegalArgumentException("kernelSize must be int[2] with positive values");
        }
        
        if (stride == null) {
            stride = kernelSize.clone();
        } else if (stride.length != 2 || stride[0] <= 0 || stride[1] <= 0) {
            throw new IllegalArgumentException("stride must be int[2] with positive values");
        }
        
        if (padding == null) {
            padding = new int[]{0, 0};
        } else if (padding.length != 2 || padding[0] < 0 || padding[1] < 0) {
            throw new IllegalArgumentException("padding must be int[2] with non-negative values");
        }
        
        if (dilation == null) {
            dilation = new int[]{1, 1};
        } else if (dilation.length != 2 || dilation[0] <= 0 || dilation[1] <= 0) {
            throw new IllegalArgumentException("dilation must be int[2] with positive values");
        }
        
        this.kernelSize = kernelSize.clone();
        this.stride = stride.clone();
        this.padding = padding.clone();
        this.dilation = dilation.clone();
        this.ceilMode = ceilMode;
    }

    /**
     * Max pooling of {@code input [B, C, H, W]} through {@link ConvOps#maxPool2d} (PyTorch {@code MaxPool2d}
     * semantics); the output shape is {@link #computeOutputShape}. In grad mode the op records {@code MaxPool2dGrad},
     * which routes each output gradient to the first in-bounds tap that holds its window's maximum.
     *
     * @throws IllegalArgumentException when the input is not 4-D, a padding exceeds half the effective kernel
     *                                  {@code dilation*(kernel-1)+1}, or an output size is below 1
     */
    @Override
    public Tensor forward(Tensor input) {
        Objects.requireNonNull(input, "Input cannot be null");
        
        Shape inputShape = input.shape();
        if (inputShape.rank() != 4) {
            throw new IllegalArgumentException("MaxPool2d expects 4D input [B, C, H, W], got rank " + inputShape.rank());
        }
        
        return ConvOps.maxPool2d(input, kernelSize, stride, padding, dilation, ceilMode);
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

    public int[] dilation() {
        return dilation.clone();
    }

    public boolean isCeilMode() {
        return ceilMode;
    }

    /**
     * The forward's output shape {@code [B, C, OH, OW]} with {@code OH = ConvOps.poolOutputSize(H, kernelSize[0],
     * stride[0], padding[0], dilation[0], ceilMode)} and its W twin (PyTorch's rule: in ceil mode a last window that
     * would start inside the right padding is dropped).
     *
     * @throws IllegalArgumentException when the shape is not 4-D or the parameters fail PyTorch's checks
     */
    public Shape computeOutputShape(Shape inputShape) {
        if (inputShape.rank() != 4) {
            throw new IllegalArgumentException("Input must be 4D [B, C, H, W]");
        }
        int outH = ConvOps.poolOutputSize(inputShape.dim(2), kernelSize[0], stride[0], padding[0], dilation[0],
                ceilMode);
        int outW = ConvOps.poolOutputSize(inputShape.dim(3), kernelSize[1], stride[1], padding[1], dilation[1],
                ceilMode);
        return Shape.of(inputShape.dim(0), inputShape.dim(1), outH, outW);
    }

    @Override
    public String toString() {
        return "MaxPool2d{" +
            "kernelSize=[" + kernelSize[0] + ", " + kernelSize[1] + "]" +
            ", stride=[" + stride[0] + ", " + stride[1] + "]" +
            ", padding=[" + padding[0] + ", " + padding[1] + "]" +
            ", dilation=[" + dilation[0] + ", " + dilation[1] + "]" +
            ", ceilMode=" + ceilMode +
            '}';
    }
}

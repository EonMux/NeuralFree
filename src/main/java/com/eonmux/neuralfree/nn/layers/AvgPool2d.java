package com.eonmux.neuralfree.nn.layers;

import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.tensor.ConvOps;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;

import java.util.Objects;

public final class AvgPool2d extends Module {
    private final int[] kernelSize;
    private final int[] stride;
    private final int[] padding;
    private final boolean ceilMode;
    private final boolean countIncludePad;

    public AvgPool2d(int[] kernelSize) {
        this(kernelSize, null, null);
    }

    public AvgPool2d(int[] kernelSize, int[] stride, int[] padding) {
        this(kernelSize, stride, padding, false, true);
    }

    public AvgPool2d(int[] kernelSize, int[] stride, int[] padding, boolean ceilMode, boolean countIncludePad) {
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
        
        this.kernelSize = kernelSize.clone();
        this.stride = stride.clone();
        this.padding = padding.clone();
        this.ceilMode = ceilMode;
        this.countIncludePad = countIncludePad;
    }

    /**
     * Average pooling of {@code input [B, C, H, W]} through {@link ConvOps#avgPool2d} (PyTorch {@code AvgPool2d}
     * semantics without a divisor override): each window is clipped to the padded extent, and its in-bounds sum is
     * divided by the clipped window size when {@code countIncludePad}, otherwise by its in-bounds element count. The
     * output shape is {@link #computeOutputShape}. In grad mode the op records {@code AvgPool2dGrad}, which spreads
     * each output gradient over the window's in-bounds elements with the same divisor.
     *
     * @throws IllegalArgumentException when the input is not 4-D, a padding exceeds half the kernel, or an output
     *                                  size is below 1
     */
    @Override
    public Tensor forward(Tensor input) {
        Objects.requireNonNull(input, "Input cannot be null");
        
        Shape inputShape = input.shape();
        if (inputShape.rank() != 4) {
            throw new IllegalArgumentException("AvgPool2d expects 4D input [B, C, H, W], got rank " + inputShape.rank());
        }
        
        return ConvOps.avgPool2d(input, kernelSize, stride, padding, ceilMode, countIncludePad);
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

    public boolean isCeilMode() {
        return ceilMode;
    }

    public boolean isCountIncludePad() {
        return countIncludePad;
    }

    /**
     * The forward's output shape {@code [B, C, OH, OW]} with {@code OH = ConvOps.poolOutputSize(H, kernelSize[0],
     * stride[0], padding[0], 1, ceilMode)} and its W twin (PyTorch's rule: in ceil mode a last window that would
     * start inside the right padding is dropped).
     *
     * @throws IllegalArgumentException when the shape is not 4-D or the parameters fail PyTorch's checks
     */
    public Shape computeOutputShape(Shape inputShape) {
        if (inputShape.rank() != 4) {
            throw new IllegalArgumentException("Input must be 4D [B, C, H, W]");
        }
        int outH = ConvOps.poolOutputSize(inputShape.dim(2), kernelSize[0], stride[0], padding[0], 1, ceilMode);
        int outW = ConvOps.poolOutputSize(inputShape.dim(3), kernelSize[1], stride[1], padding[1], 1, ceilMode);
        return Shape.of(inputShape.dim(0), inputShape.dim(1), outH, outW);
    }

    @Override
    public String toString() {
        return "AvgPool2d{" +
            "kernelSize=[" + kernelSize[0] + ", " + kernelSize[1] + "]" +
            ", stride=[" + stride[0] + ", " + stride[1] + "]" +
            ", padding=[" + padding[0] + ", " + padding[1] + "]" +
            ", ceilMode=" + ceilMode +
            ", countIncludePad=" + countIncludePad +
            '}';
    }
}

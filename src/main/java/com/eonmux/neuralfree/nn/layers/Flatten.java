package com.eonmux.neuralfree.nn.layers;

import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;

import java.util.Objects;

public final class Flatten extends Module {
    private final int startDim;
    private final int endDim;

    public Flatten() {
        this(1, -1);
    }

    public Flatten(int startDim) {
        this(startDim, -1);
    }

    public Flatten(int startDim, int endDim) {
        this.startDim = startDim;
        this.endDim = endDim;
    }

    /**
     * Merges dimensions {@code startDim..endDim} (inclusive; negative values count from the end, and a range reaching
     * past either end of the input's rank is cut at it) of {@code input} into one and keeps every other dimension, as
     * PyTorch's {@code nn.Flatten}: with the defaults {@code [B, C, H, W]} becomes {@code [B, C*H*W]}, also for
     * {@code B = 1}. The result is {@link Tensor#reshape} of the input, which records {@code ReshapeGrad} in grad mode.
     *
     * @throws IllegalArgumentException when the dimension range selects no dimension of the input, or the merged size
     *                                  exceeds {@code 2^31-1}
     */
    @Override
    public Tensor forward(Tensor input) {
        Objects.requireNonNull(input, "Input cannot be null");

        Shape inputShape = input.shape();
        int rank = inputShape.rank();

        int effectiveStart = startDim < 0 ? rank + startDim : startDim;
        int effectiveEnd = endDim < 0 ? rank + endDim : endDim;

        if (effectiveStart < 0) effectiveStart = 0;
        if (effectiveEnd >= rank) effectiveEnd = rank - 1;

        if (effectiveStart >= rank || effectiveEnd < 0 || effectiveStart > effectiveEnd) {
            throw new IllegalArgumentException("Invalid flatten dimensions: startDim=" + startDim +
                ", endDim=" + endDim + " for input rank " + rank);
        }

        long mergedSize = 1;
        for (int i = effectiveStart; i <= effectiveEnd; i++) {
            mergedSize *= inputShape.dim(i);
        }
        if (mergedSize > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("Flattened dimension of " + mergedSize + " elements exceeds 2^31-1 for"
                + " input shape " + inputShape);
        }

        int[] outDims = new int[rank - (effectiveEnd - effectiveStart)];
        int o = 0;
        for (int i = 0; i < effectiveStart; i++) {
            outDims[o++] = inputShape.dim(i);
        }
        outDims[o++] = (int) mergedSize;
        for (int i = effectiveEnd + 1; i < rank; i++) {
            outDims[o++] = inputShape.dim(i);
        }
        return input.reshape(Shape.of(outDims));
    }

    public int startDim() {
        return startDim;
    }

    public int endDim() {
        return endDim;
    }

    @Override
    public String toString() {
        return "Flatten{startDim=" + startDim + ", endDim=" + endDim + "}";
    }
}

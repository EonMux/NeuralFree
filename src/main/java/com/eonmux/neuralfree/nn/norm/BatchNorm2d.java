package com.eonmux.neuralfree.nn.norm;

import com.eonmux.neuralfree.tensor.Shape;

/**
 * Batch normalization of an input {@code [B, C, H, W]} per channel: the statistics of each channel run over its
 * {@code B * H * W} elements. State, batch counter and running update as {@link AbstractBatchNorm} describes.
 */
public final class BatchNorm2d extends AbstractBatchNorm {

    /** Epsilon {@code 1e-5}, momentum {@code 0.1}, affine. */
    public BatchNorm2d(int numFeatures) {
        this(numFeatures, DEFAULT_EPSILON, DEFAULT_MOMENTUM, true);
    }

    /**
     * @param momentum the weight of the batch statistic in the running update, in {@code [0, 1]}
     * @throws IllegalArgumentException for a non-positive {@code numFeatures} or {@code epsilon}, or a momentum
     *                                  outside {@code [0, 1]}
     */
    public BatchNorm2d(int numFeatures, float epsilon, float momentum, boolean affine) {
        super(numFeatures, epsilon, Float.valueOf(momentum), affine);
    }

    /**
     * @param momentum the weight of the batch statistic in the running update, in {@code [0, 1]}, or null to keep
     *                 the cumulative moving average of every training batch (PyTorch's {@code momentum=None})
     * @throws IllegalArgumentException for a non-positive {@code numFeatures} or {@code epsilon}, or a momentum
     *                                  outside {@code [0, 1]}
     */
    public BatchNorm2d(int numFeatures, float epsilon, Float momentum, boolean affine) {
        super(numFeatures, epsilon, momentum, affine);
    }

    @Override
    void checkInput(Shape inputShape) {
        if (inputShape.rank() != 4) {
            throw new IllegalArgumentException("BatchNorm2d expects 4D input [B,C,H,W], got rank " + inputShape.rank());
        }
        int channels = inputShape.dim(1);
        if (channels != numFeatures()) {
            throw new IllegalArgumentException("Expected " + numFeatures() + " channels, got " + channels);
        }
    }
}

package com.eonmux.neuralfree.nn.norm;

import com.eonmux.neuralfree.tensor.Shape;

/**
 * Batch normalization of an input {@code [B, F]} or {@code [B, F, L]} per feature: the statistics of each feature run
 * over its {@code B * L} elements. State, batch counter and running update as {@link AbstractBatchNorm} describes.
 */
public final class BatchNorm1d extends AbstractBatchNorm {

    /** Epsilon {@code 1e-5}, momentum {@code 0.1}, affine. */
    public BatchNorm1d(int numFeatures) {
        this(numFeatures, DEFAULT_EPSILON, DEFAULT_MOMENTUM, true);
    }

    /**
     * @param momentum the weight of the batch statistic in the running update, in {@code [0, 1]}
     * @throws IllegalArgumentException for a non-positive {@code numFeatures} or {@code epsilon}, or a momentum
     *                                  outside {@code [0, 1]}
     */
    public BatchNorm1d(int numFeatures, float epsilon, float momentum, boolean affine) {
        super(numFeatures, epsilon, Float.valueOf(momentum), affine);
    }

    /**
     * @param momentum the weight of the batch statistic in the running update, in {@code [0, 1]}, or null to keep
     *                 the cumulative moving average of every training batch (PyTorch's {@code momentum=None})
     * @throws IllegalArgumentException for a non-positive {@code numFeatures} or {@code epsilon}, or a momentum
     *                                  outside {@code [0, 1]}
     */
    public BatchNorm1d(int numFeatures, float epsilon, Float momentum, boolean affine) {
        super(numFeatures, epsilon, momentum, affine);
    }

    @Override
    void checkInput(Shape inputShape) {
        if (inputShape.rank() < 2 || inputShape.rank() > 3) {
            throw new IllegalArgumentException("BatchNorm1d expects 2D or 3D input, got rank " + inputShape.rank());
        }
        int features = inputShape.dim(1);
        if (features != numFeatures()) {
            throw new IllegalArgumentException("Expected " + numFeatures() + " features, got " + features);
        }
    }
}

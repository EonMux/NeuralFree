package com.eonmux.neuralfree.nn.loss;

import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;

import java.util.Objects;

/**
 * Huber loss: with {@code d = |prediction - target|}, {@code 0.5 * d * d} where {@code d <= delta} and
 * {@code delta * (d - 0.5 * delta)} elsewhere, per element, then the reduction. The target pairs with the prediction
 * element by element in row-major order and takes the prediction's shape.
 *
 * <p>Composed from {@link TensorOps} (sub, abs, mul, add, gt, where), so it runs on the device of the operands and
 * nothing is read back. Per element it is in the single-rounding class, and a SUM or MEAN result in the reduction
 * class ({@linkplain com.eonmux.neuralfree.tensor.gpu numerical contract}).</p>
 */
public final class HuberLoss implements LossFunction {

    private static final float DEFAULT_DELTA = 1.0f;

    private final Reduction reduction;
    private final float delta;

    public HuberLoss() {
        this(Reduction.MEAN, DEFAULT_DELTA);
    }

    public HuberLoss(Reduction reduction) {
        this(reduction, DEFAULT_DELTA);
    }

    public HuberLoss(Reduction reduction, float delta) {
        this.reduction = Objects.requireNonNull(reduction, "Reduction cannot be null");
        if (delta <= 0) {
            throw new IllegalArgumentException("Delta must be positive, got: " + delta);
        }
        this.delta = delta;
    }

    /**
     * @throws IllegalArgumentException when the target's element count differs from the prediction's, or the
     *                                  tensors live on different backends
     */
    @Override
    public Tensor compute(Tensor prediction, Tensor target) {
        Objects.requireNonNull(prediction, "Prediction cannot be null");
        Objects.requireNonNull(target, "Target cannot be null");

        Tensor loss;
        try (Tensor t = LossTensors.alignTarget(prediction, target);
             Tensor diff = TensorOps.sub(prediction, t);
             Tensor absDiff = TensorOps.abs(diff);
             Tensor halfDiff = TensorOps.mul(absDiff, 0.5f);
             Tensor quadratic = TensorOps.mul(halfDiff, absDiff);
             Tensor shifted = TensorOps.add(absDiff, -(0.5f * delta));
             Tensor linear = TensorOps.mul(shifted, delta);
             Tensor deltaValue = LossTensors.constant(prediction.backend(), delta);
             Tensor beyond = TensorOps.gt(absDiff, deltaValue)) {
            loss = TensorOps.where(beyond, linear, quadratic);
        }
        return LossTensors.reduce(loss, reduction);
    }

    @Override
    public String name() {
        return "HuberLoss";
    }

    @Override
    public Reduction reduction() {
        return reduction;
    }

    public float delta() {
        return delta;
    }

    @Override
    public String toString() {
        return "HuberLoss{reduction=" + reduction + ", delta=" + delta + "}";
    }
}

package com.eonmux.neuralfree.nn.loss;

import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;

import java.util.Objects;

/**
 * Smooth L1 loss: with {@code d = |prediction - target|}, {@code 0.5 * d * d / beta^2} where {@code d < beta} and
 * {@code d / beta - 0.5} elsewhere, per element, then the reduction. The target pairs with the prediction element by
 * element in row-major order and takes the prediction's shape.
 *
 * <p>Composed from {@link TensorOps} (sub, abs, mul, div, add, lt, where), so it runs on the device of the operands
 * and nothing is read back. Per element it is in the division class, and a SUM or MEAN result in the reduction class
 * ({@linkplain com.eonmux.neuralfree.tensor.gpu numerical contract}).</p>
 */
public final class SmoothL1Loss implements LossFunction {

    private static final float BETA = 1.0f;

    private final Reduction reduction;
    private final float beta;

    public SmoothL1Loss() {
        this(Reduction.MEAN, BETA);
    }

    public SmoothL1Loss(Reduction reduction) {
        this(reduction, BETA);
    }

    public SmoothL1Loss(Reduction reduction, float beta) {
        this.reduction = Objects.requireNonNull(reduction, "Reduction cannot be null");
        if (beta <= 0) {
            throw new IllegalArgumentException("Beta must be positive, got: " + beta);
        }
        this.beta = beta;
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
             Tensor squaredHalf = TensorOps.mul(halfDiff, absDiff);
             Tensor betaSquared = LossTensors.constant(prediction.backend(), beta * beta);
             Tensor quadratic = TensorOps.div(squaredHalf, betaSquared);
             Tensor betaValue = LossTensors.constant(prediction.backend(), beta);
             Tensor scaled = TensorOps.div(absDiff, betaValue);
             Tensor linear = TensorOps.add(scaled, -0.5f);
             Tensor within = TensorOps.lt(absDiff, betaValue)) {
            loss = TensorOps.where(within, quadratic, linear);
        }
        return LossTensors.reduce(loss, reduction);
    }

    @Override
    public String name() {
        return "SmoothL1Loss";
    }

    @Override
    public Reduction reduction() {
        return reduction;
    }

    public float beta() {
        return beta;
    }

    @Override
    public String toString() {
        return "SmoothL1Loss{reduction=" + reduction + ", beta=" + beta + "}";
    }
}

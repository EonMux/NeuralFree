package com.eonmux.neuralfree.nn.loss;

import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;

import java.util.Objects;

/**
 * Binary cross-entropy on probabilities: with {@code p = clamp(prediction, 1e-7, 1 - 1e-7)},
 * {@code -(t * log(p) + (1 - t) * log(1 - p))} per element, then the reduction. The target pairs with the prediction
 * element by element in row-major order and takes the prediction's shape.
 *
 * <p>Composed from {@link TensorOps} (clamp, log, sub, mul, add, neg), so it runs on the device of the operands and
 * nothing is read back. Per element it is in the transcendental class, and a SUM or MEAN result in the reduction
 * class ({@linkplain com.eonmux.neuralfree.tensor.gpu numerical contract}).</p>
 */
public final class BCELoss implements LossFunction {

    private static final float EPSILON = 1e-7f;

    private final Reduction reduction;

    public BCELoss() {
        this(Reduction.MEAN);
    }

    public BCELoss(Reduction reduction) {
        this.reduction = Objects.requireNonNull(reduction, "Reduction cannot be null");
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
             Tensor one = LossTensors.constant(prediction.backend(), 1.0f);
             Tensor p = TensorOps.clamp(prediction, EPSILON, 1.0f - EPSILON);
             Tensor logP = TensorOps.log(p);
             Tensor oneMinusP = TensorOps.sub(one, p);
             Tensor logOneMinusP = TensorOps.log(oneMinusP);
             Tensor oneMinusT = TensorOps.sub(one, t);
             Tensor positive = TensorOps.mul(t, logP);
             Tensor negative = TensorOps.mul(oneMinusT, logOneMinusP);
             Tensor sum = TensorOps.add(positive, negative)) {
            loss = TensorOps.neg(sum);
        }
        return LossTensors.reduce(loss, reduction);
    }

    @Override
    public String name() {
        return "BCELoss";
    }

    @Override
    public Reduction reduction() {
        return reduction;
    }

    @Override
    public String toString() {
        return "BCELoss{reduction=" + reduction + "}";
    }
}

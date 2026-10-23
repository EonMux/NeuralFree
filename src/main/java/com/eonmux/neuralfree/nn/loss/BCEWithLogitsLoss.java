package com.eonmux.neuralfree.nn.loss;

import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;

import java.util.Objects;

/**
 * Binary cross-entropy on logits {@code x} with targets {@code t}, per element, then the reduction. The target pairs
 * with the prediction element by element in row-major order and takes the prediction's shape. With
 * {@code sp(z) = log(1 + exp(z))}:
 *
 * <ul>
 *   <li>Without a positive weight ({@code posWeight <= 0}): {@code sp(x) - t * x}, which is
 *   {@code -(t * log(sigmoid(x)) + (1 - t) * log(1 - sigmoid(x)))}.</li>
 *   <li>With a positive weight {@code w > 0}: {@code -(w * t * log(sigmoid(x)) + (1 - t) * log(1 - sigmoid(x)))},
 *   which is {@code w * t * sp(-x) + (1 - t) * sp(x)}.</li>
 * </ul>
 *
 * <p>{@code sp(z)} is evaluated as {@code z > 0 ? z + log(1 + exp(-z)) : log(1 + exp(z))}, so the loss stays finite
 * for every finite logit, and its recorded gradient is {@code sigmoid(z)} everywhere, {@code z = 0} included (the
 * gradient of the loss is PyTorch's {@code sigmoid(x) - t} without a positive weight). Composed from
 * {@link TensorOps} (gt, neg, softplus, add, where, mul, sub), so it runs on the device of the operands and nothing is
 * read back; every differentiable step records its autograd node, so the backward reaches the logits and a target
 * that requires grad. Per element it is in the transcendental class, and a SUM or MEAN result in the reduction class
 * ({@linkplain com.eonmux.neuralfree.tensor.gpu numerical contract}).</p>
 */
public final class BCEWithLogitsLoss implements LossFunction {

    private final Reduction reduction;
    private final float posWeight;

    public BCEWithLogitsLoss() {
        this(Reduction.MEAN, -1.0f);
    }

    public BCEWithLogitsLoss(Reduction reduction) {
        this(reduction, -1.0f);
    }

    public BCEWithLogitsLoss(Reduction reduction, float posWeight) {
        this.reduction = Objects.requireNonNull(reduction, "Reduction cannot be null");
        this.posWeight = posWeight;
    }

    /**
     * @throws IllegalArgumentException when the target's element count differs from the prediction's, or the
     *                                  tensors live on different backends
     */
    @Override
    public Tensor compute(Tensor prediction, Tensor target) {
        Objects.requireNonNull(prediction, "Prediction (logits) cannot be null");
        Objects.requireNonNull(target, "Target cannot be null");

        Tensor loss;
        try (Tensor t = LossTensors.alignTarget(prediction, target);
             Tensor softplusX = softplus(prediction)) {
            loss = posWeight > 0
                    ? weighted(prediction, t, softplusX)
                    : unweighted(prediction, t, softplusX);
        }
        return LossTensors.reduce(loss, reduction);
    }

    /** {@code sp(x) - t * x}. */
    private static Tensor unweighted(Tensor x, Tensor t, Tensor softplusX) {
        try (Tensor tx = TensorOps.mul(t, x)) {
            return TensorOps.sub(softplusX, tx);
        }
    }

    /** {@code w * t * sp(-x) + (1 - t) * sp(x)}. */
    private Tensor weighted(Tensor x, Tensor t, Tensor softplusX) {
        try (Tensor negX = TensorOps.neg(x);
             Tensor softplusNegX = softplus(negX);
             Tensor weightedT = TensorOps.mul(t, posWeight);
             Tensor positiveLoss = TensorOps.mul(weightedT, softplusNegX);
             Tensor one = LossTensors.constant(x.backend(), 1.0f);
             Tensor oneMinusT = TensorOps.sub(one, t);
             Tensor negativeLoss = TensorOps.mul(oneMinusT, softplusX)) {
            return TensorOps.add(positiveLoss, negativeLoss);
        }
    }

    /**
     * {@code log(1 + exp(z))} as {@code where(z > 0, z + softplus(-z), softplus(z))}: the selected branch is finite
     * for every finite {@code z}, and the gradient is {@code sigmoid(z)} on both sides of 0 and at 0. An unselected
     * branch that overflows gets a zero gradient from the selection, so it never turns the gradient into NaN.
     */
    private static Tensor softplus(Tensor z) {
        try (Tensor zero = LossTensors.constant(z.backend(), 0.0f);
             Tensor positive = TensorOps.gt(z, zero);
             Tensor negZ = TensorOps.neg(z);
             Tensor tail = TensorOps.softplus(negZ);
             Tensor shifted = TensorOps.add(z, tail);
             Tensor direct = TensorOps.softplus(z)) {
            return TensorOps.where(positive, shifted, direct);
        }
    }

    @Override
    public String name() {
        return "BCEWithLogitsLoss";
    }

    @Override
    public Reduction reduction() {
        return reduction;
    }

    public float posWeight() {
        return posWeight;
    }

    @Override
    public String toString() {
        return "BCEWithLogitsLoss{reduction=" + reduction +
               (posWeight > 0 ? ", posWeight=" + posWeight : "") + "}";
    }
}

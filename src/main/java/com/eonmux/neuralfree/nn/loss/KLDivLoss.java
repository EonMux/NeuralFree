package com.eonmux.neuralfree.nn.loss;

import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;

import java.util.Objects;

/**
 * Kullback-Leibler divergence loss with PyTorch's {@code kl_div} convention: the prediction (input) holds
 * <em>log-probabilities</em> {@code x} (for example a {@code logSoftmax} output) and the target holds probabilities
 * {@code t}, or log-probabilities {@code lt} with {@code logTarget}. Per element:
 * <ul>
 *   <li>{@code xlogy(t, t) - t * x}, where {@code xlogy(t, t)} is {@code t * log(t)} and exactly 0 where {@code t == 0}
 *   ({@code 0 * log 0 = 0}), so a zero target contributes {@code 0 - 0 * x};</li>
 *   <li>with {@code logTarget}: {@code exp(lt) * (lt - x)}.</li>
 * </ul>
 * Nothing is clamped: a negative target gives NaN and a zero target meeting {@code x = -inf} gives {@code 0 * -inf =
 * NaN}, as in PyTorch. The target pairs with the prediction element by element in row-major order and takes the
 * prediction's shape.
 *
 * <p><b>Reductions.</b> {@link #KLDivLoss()} and {@link #batchMean(boolean)} use PyTorch's {@code batchmean}, the
 * reduction that gives the mathematical KL divergence per sample: the sum of all elements divided by the batch size
 * (the prediction's first dimension; a scalar prediction is not divided). {@link LossFunction.Reduction#SUM} sums all
 * elements, {@link LossFunction.Reduction#MEAN} divides that sum by the element count (PyTorch {@code mean}, which is
 * not the KL divergence when there are several classes) and {@link LossFunction.Reduction#NONE} keeps the per-element
 * loss. {@link #reduction()} reports {@code MEAN} for {@code batchmean} (a mean over the batch); {@link #isBatchMean()}
 * tells the two apart.</p>
 *
 * <p>Composed from {@link TensorOps} (eq, where, log, exp, mul, sub, sum, mean, div), so it runs on the device of the
 * operands, nothing is read back, and it records when grad mode is on and an operand requires grad. The gradient with
 * respect to the prediction is {@code -t} ({@code -exp(lt)} with {@code logTarget}) times the reduction's scale; the
 * zero-target guard keeps the target's gradient finite at {@code t == 0}. Per element it is in the transcendental
 * class, and a reduced result in the reduction class ({@linkplain com.eonmux.neuralfree.tensor.gpu numerical
 * contract}).</p>
 */
public final class KLDivLoss implements LossFunction {

    private final Reduction reduction;
    private final boolean batchMean;
    private final boolean logTarget;

    /** {@code batchmean} reduction, target given as probabilities. */
    public KLDivLoss() {
        this(Reduction.MEAN, true, false);
    }

    /** The element-wise {@code reduction} ({@code none}, {@code mean} or {@code sum}), target as probabilities. */
    public KLDivLoss(Reduction reduction) {
        this(reduction, false);
    }

    /** The element-wise {@code reduction}; with {@code logTarget} the target holds log-probabilities. */
    public KLDivLoss(Reduction reduction, boolean logTarget) {
        this(Objects.requireNonNull(reduction, "Reduction cannot be null"), false, logTarget);
    }

    private KLDivLoss(Reduction reduction, boolean batchMean, boolean logTarget) {
        this.reduction = reduction;
        this.batchMean = batchMean;
        this.logTarget = logTarget;
    }

    /** The {@code batchmean} reduction; with {@code logTarget} the target holds log-probabilities. */
    public static KLDivLoss batchMean(boolean logTarget) {
        return new KLDivLoss(Reduction.MEAN, true, logTarget);
    }

    /**
     * @param prediction log-probabilities
     * @param target     probabilities, or log-probabilities with {@code logTarget}
     * @throws IllegalArgumentException when the target's element count differs from the prediction's, or the
     *                                  tensors live on different backends
     */
    @Override
    public Tensor compute(Tensor prediction, Tensor target) {
        Objects.requireNonNull(prediction, "Prediction cannot be null");
        Objects.requireNonNull(target, "Target cannot be null");

        Tensor loss;
        try (Tensor t = LossTensors.alignTarget(prediction, target)) {
            loss = logTarget ? logTargetDivergence(t, prediction) : divergence(t, prediction);
        }
        if (!batchMean) {
            return LossTensors.reduce(loss, reduction);
        }
        Tensor sum = LossTensors.reduce(loss, Reduction.SUM);
        Shape shape = prediction.shape();
        if (shape.rank() == 0) {
            return sum;
        }
        try (Tensor summed = sum;
             Tensor batch = LossTensors.constant(prediction.backend(), (float) shape.dim(0))) {
            return TensorOps.div(summed, batch);
        }
    }

    /**
     * {@code xlogy(t, t) - t * x} per element, with {@code xlogy(t, t) = t * log(t == 0 ? 1 : t)}: exactly
     * {@code t * log(t)} where {@code t != 0} and {@code +0} where {@code t == 0}.
     */
    private static Tensor divergence(Tensor t, Tensor x) {
        try (Tensor zero = LossTensors.constant(t.backend(), 0.0f);
             Tensor isZero = TensorOps.eq(t, zero);
             Tensor zeros = Tensor.zeros(t.backend(), t.shape());
             Tensor ones = TensorOps.add(zeros, 1.0f);
             Tensor safe = TensorOps.where(isZero, ones, t);
             Tensor logSafe = TensorOps.log(safe);
             Tensor entropyTerm = TensorOps.mul(t, logSafe);
             Tensor crossTerm = TensorOps.mul(t, x)) {
            return TensorOps.sub(entropyTerm, crossTerm);
        }
    }

    /** {@code exp(lt) * (lt - x)} per element. */
    private static Tensor logTargetDivergence(Tensor logT, Tensor x) {
        try (Tensor probT = TensorOps.exp(logT);
             Tensor logRatio = TensorOps.sub(logT, x)) {
            return TensorOps.mul(probT, logRatio);
        }
    }

    @Override
    public String name() {
        return "KLDivLoss";
    }

    /** The reduction; {@code MEAN} for {@code batchmean} as well ({@link #isBatchMean()}). */
    @Override
    public Reduction reduction() {
        return reduction;
    }

    /** Whether the loss uses PyTorch's {@code batchmean}: the element sum divided by the batch size. */
    public boolean isBatchMean() {
        return batchMean;
    }

    public boolean logTarget() {
        return logTarget;
    }

    @Override
    public String toString() {
        String mode = batchMean ? "batchmean" : reduction.toString();
        return "KLDivLoss{reduction=" + mode + ", logTarget=" + logTarget + "}";
    }
}

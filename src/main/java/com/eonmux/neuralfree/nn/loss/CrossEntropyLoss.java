package com.eonmux.neuralfree.nn.loss;

import com.eonmux.neuralfree.tensor.DType;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;

import java.util.Objects;

/**
 * Cross-entropy on logits along the class dim {@code dim} (negative counts from the end), per sample (the prediction
 * shape without the class dim; a scalar for rank 1), then the reduction. With {@code lp = logSoftmax(prediction, dim)},
 * {@code C} classes and the label smoothing {@code ls}:
 *
 * <ul>
 *   <li>a FLOAT32 target of the prediction's shape is a class distribution: {@code -sum_c t_c * lp_c} over the
 *   entries with {@code t_c > 0}, or with {@code ls > 0}, {@code -sum_c (t_c * (1 - ls) + ls / C) * lp_c} over all
 *   classes; the reduction is the plain SUM or MEAN over the samples;</li>
 *   <li>any other target holds one class index per sample in row-major sample order, as PyTorch's int64 targets do:
 *   INT32, INT64 (narrowed with Java {@code (int)}) or, for compatibility, FLOAT32 values read as
 *   {@code Math.round(t)}. The loss is {@code -lp[class]}, or with {@code ls > 0},
 *   {@code (1 - ls) * -lp[class] + ls * -sum_c lp_c / C}. A sample whose class is {@code ignoreIndex} (default -100)
 *   has loss 0 and is left out of the MEAN denominator, which counts the other samples (NaN when every sample is
 *   ignored). Any other class must lie in {@code [0, C)}: on the CPU path an out-of-range class throws
 *   {@link IndexOutOfBoundsException}; on the device the gather that reads it raises that exception, with the
 *   gather's message, at the next host synchronization. Integer targets support at most {@code 2^24}
 *   classes.</li>
 * </ul>
 *
 * <p>An empty class dim reads no class and no target value: the per-sample loss is the empty sum, 0, reduced over
 * every sample.</p>
 *
 * <p>Composed from {@link TensorOps} (logSoftmax, mul, add, gt, where, sum, sub, div, and round, cast, eq and gather
 * for class indices), so it runs on the device of the operands and nothing is read back. Every differentiable step
 * records its autograd node: the loss requires grad exactly when grad mode is on and the prediction (or a
 * class-distribution target) requires grad, and its backward reaches them through logSoftmax, gather and where. Class
 * indices and the ignore mask are computed under no-grad, so they get no gradient. The result is in the reduction
 * class ({@linkplain com.eonmux.neuralfree.tensor.gpu numerical contract}).</p>
 */
public final class CrossEntropyLoss implements LossFunction {

    /** PyTorch's default {@code ignore_index}. */
    public static final int DEFAULT_IGNORE_INDEX = LossTensors.DEFAULT_IGNORE_INDEX;

    private final Reduction reduction;
    private final int dim;
    private final float labelSmoothing;
    private final int ignoreIndex;

    public CrossEntropyLoss() {
        this(Reduction.MEAN, -1, 0.0f);
    }

    public CrossEntropyLoss(Reduction reduction) {
        this(reduction, -1, 0.0f);
    }

    public CrossEntropyLoss(Reduction reduction, int dim, float labelSmoothing) {
        this(reduction, dim, labelSmoothing, DEFAULT_IGNORE_INDEX);
    }

    /**
     * @param ignoreIndex the class index of samples that contribute nothing, in {@code (-2^24, 2^24)}
     * @throws IllegalArgumentException when {@code labelSmoothing} is outside {@code [0, 1)} or {@code ignoreIndex}
     *                                  outside {@code (-2^24, 2^24)}
     */
    public CrossEntropyLoss(Reduction reduction, int dim, float labelSmoothing, int ignoreIndex) {
        this.reduction = Objects.requireNonNull(reduction, "Reduction cannot be null");
        this.dim = dim;
        if (!(labelSmoothing >= 0 && labelSmoothing < 1)) {
            throw new IllegalArgumentException("Label smoothing must be in [0, 1), got: " + labelSmoothing);
        }
        this.labelSmoothing = labelSmoothing;
        this.ignoreIndex = LossTensors.requireIgnoreIndex(ignoreIndex);
    }

    /**
     * @throws IllegalArgumentException  when {@code dim} is outside the prediction's rank, a class-index target does
     *                                   not hold one element per sample or has another dtype than INT32, INT64 or
     *                                   FLOAT32, an integer target meets more than {@code 2^24} classes, or the
     *                                   tensors live on different backends
     * @throws IndexOutOfBoundsException on the CPU path, when a class index that is not ignored lies outside
     *                                   {@code [0, C)}
     */
    @Override
    public Tensor compute(Tensor prediction, Tensor target) {
        Objects.requireNonNull(prediction, "Prediction (logits) cannot be null");
        Objects.requireNonNull(target, "Target cannot be null");

        Shape predShape = prediction.shape();
        int axis = LossTensors.classAxis(predShape, dim);
        if (predShape.dim(axis) == 0) {
            return LossTensors.reduce(LossTensors.emptyClassLoss(prediction, axis), reduction);
        }
        boolean distribution = target.dtype() == DType.FLOAT32 && target.shape().equals(predShape);
        try (Tensor logProbs = TensorOps.logSoftmax(prediction, axis)) {
            if (!distribution) {
                return LossTensors.classIndexLoss(logProbs, axis, target, ignoreIndex, reduction,
                        picked -> classLoss(logProbs, axis, picked));
            }
            Tensor loss = LossTensors.reshapeOwned(distributionLoss(logProbs, axis, target),
                    LossTensors.withoutAxis(predShape, axis));
            return LossTensors.reduce(loss, reduction);
        }
    }

    /**
     * {@code -picked}, or with label smoothing {@code -((1 - ls) * picked + (ls / C) * sum_c lp_c)}, for the picked
     * log-probabilities (the class axis of size 1).
     */
    private Tensor classLoss(Tensor logProbs, int axis, Tensor picked) {
        if (labelSmoothing == 0) {
            return TensorOps.neg(picked);
        }
        float smoothFactor = labelSmoothing / logProbs.shape().dim(axis);
        try (Tensor classSum = TensorOps.sum(logProbs, axis);
             Tensor classSumKept = classSum.reshape(picked.shape());
             Tensor smooth = TensorOps.mul(classSumKept, smoothFactor);
             Tensor hard = TensorOps.mul(picked, 1.0f - labelSmoothing);
             Tensor both = TensorOps.add(hard, smooth)) {
            return TensorOps.neg(both);
        }
    }

    /** {@code -sum_c w_c * lp_c} over the class axis, with the weights of the class-distribution target. */
    private Tensor distributionLoss(Tensor logProbs, int axis, Tensor target) {
        if (labelSmoothing > 0) {
            float smoothFactor = labelSmoothing / logProbs.shape().dim(axis);
            try (Tensor scaled = TensorOps.mul(target, 1.0f - labelSmoothing);
                 Tensor weights = TensorOps.add(scaled, smoothFactor);
                 Tensor terms = TensorOps.mul(weights, logProbs)) {
                return LossTensors.negatedSum(terms, axis);
            }
        }
        try (Tensor weighted = TensorOps.mul(target, logProbs);
             Tensor terms = LossTensors.positiveOnly(target, weighted)) {
            return LossTensors.negatedSum(terms, axis);
        }
    }

    @Override
    public String name() {
        return "CrossEntropyLoss";
    }

    @Override
    public Reduction reduction() {
        return reduction;
    }

    public int dim() {
        return dim;
    }

    public float labelSmoothing() {
        return labelSmoothing;
    }

    public int ignoreIndex() {
        return ignoreIndex;
    }

    @Override
    public String toString() {
        return "CrossEntropyLoss{reduction=" + reduction + ", dim=" + dim +
               ", labelSmoothing=" + labelSmoothing + ", ignoreIndex=" + ignoreIndex + "}";
    }
}

package com.eonmux.neuralfree.nn.loss;

import com.eonmux.neuralfree.tensor.DType;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;

import java.util.Objects;

/**
 * Focal loss on logits along the class dim {@code dim} (negative counts from the end), per sample (the prediction
 * shape without the class dim; a scalar for rank 1), then the reduction. With {@code lp = logSoftmax(prediction, dim)},
 * {@code p = exp(lp)} and the focal weight {@code f = (1 - p)^gamma}:
 *
 * <ul>
 *   <li>a FLOAT32 target with the prediction's element count pairs with it element by element (row-major) as a class
 *   distribution: {@code -sum_c a_c * f_c * t_c * lp_c} over the entries with {@code t_c > 0}, where
 *   {@code a_c = alpha} when {@code t_c > 0.5} and {@code 1 - alpha} otherwise; the reduction is the plain SUM or
 *   MEAN over the samples;</li>
 *   <li>any other target holds one class index per sample in row-major sample order: INT32, INT64 (narrowed with Java
 *   {@code (int)}) or, for compatibility, FLOAT32 values read as {@code Math.round(t)}. The loss is
 *   {@code -alpha * f[class] * lp[class]}. A sample whose class is {@code ignoreIndex} (default -100, as in PyTorch's
 *   class-index losses) has loss 0 and is left out of the MEAN denominator, which counts the other samples (NaN when
 *   every sample is ignored). Any other class must lie in {@code [0, C)}: on the CPU path an out-of-range class
 *   throws {@link IndexOutOfBoundsException}; on the device the gather that reads it raises that exception, with the
 *   gather's message, at the next host synchronization. Integer targets support at most
 *   {@code 2^24} classes.</li>
 * </ul>
 *
 * <p>An empty class dim reads no class and no target value: the per-sample loss is the empty sum, 0, reduced over
 * every sample.</p>
 *
 * <p>Composed from {@link TensorOps} (logSoftmax, exp, sub, pow, gt, where, mul, sum, div, and round, cast, eq and
 * gather for class indices), so it runs on the device of the operands and nothing is read back. Every differentiable
 * step records its autograd node, so the loss requires grad exactly when grad mode is on and the prediction (or a
 * class-distribution target) requires grad, and the focal weight is differentiated with the log-probabilities, as in
 * PyTorch's composition. Class indices and the ignore mask are computed under no-grad. The result is in the reduction
 * class ({@linkplain com.eonmux.neuralfree.tensor.gpu numerical contract}).</p>
 */
public final class FocalLoss implements LossFunction {

    /** The default {@code ignore_index} of the class-index losses, PyTorch's -100. */
    public static final int DEFAULT_IGNORE_INDEX = LossTensors.DEFAULT_IGNORE_INDEX;

    private static final float DEFAULT_ALPHA = 0.25f;
    private static final float DEFAULT_GAMMA = 2.0f;

    private final float alpha;
    private final float gamma;
    private final Reduction reduction;
    private final int dim;
    private final int ignoreIndex;

    public FocalLoss() {
        this(DEFAULT_ALPHA, DEFAULT_GAMMA, Reduction.MEAN, -1);
    }

    public FocalLoss(float gamma) {
        this(DEFAULT_ALPHA, gamma, Reduction.MEAN, -1);
    }

    public FocalLoss(float alpha, float gamma) {
        this(alpha, gamma, Reduction.MEAN, -1);
    }

    public FocalLoss(float alpha, float gamma, Reduction reduction, int dim) {
        this(alpha, gamma, reduction, dim, DEFAULT_IGNORE_INDEX);
    }

    /**
     * @param ignoreIndex the class index of samples that contribute nothing, in {@code (-2^24, 2^24)}
     * @throws IllegalArgumentException when {@code alpha} is outside {@code [0, 1]}, {@code gamma} is negative or
     *                                  {@code ignoreIndex} is outside {@code (-2^24, 2^24)}
     */
    public FocalLoss(float alpha, float gamma, Reduction reduction, int dim, int ignoreIndex) {
        if (alpha < 0 || alpha > 1) {
            throw new IllegalArgumentException("Alpha must be in [0, 1], got: " + alpha);
        }
        if (gamma < 0) {
            throw new IllegalArgumentException("Gamma must be non-negative, got: " + gamma);
        }
        this.alpha = alpha;
        this.gamma = gamma;
        this.reduction = Objects.requireNonNull(reduction, "Reduction cannot be null");
        this.dim = dim;
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
        boolean distribution = target.dtype() == DType.FLOAT32 && target.numElements() == prediction.numElements();
        try (Tensor logProbs = TensorOps.logSoftmax(prediction, axis)) {
            if (!distribution) {
                return LossTensors.classIndexLoss(logProbs, axis, target, ignoreIndex, reduction, this::classLoss);
            }
            Tensor loss = LossTensors.reshapeOwned(distributionLoss(logProbs, axis, target),
                    LossTensors.withoutAxis(predShape, axis));
            return LossTensors.reduce(loss, reduction);
        }
    }

    /** {@code (1 - exp(lp))^gamma} per element. */
    private Tensor focalWeight(Tensor logProbs) {
        try (Tensor prob = TensorOps.exp(logProbs);
             Tensor one = LossTensors.constant(logProbs.backend(), 1.0f);
             Tensor oneMinusProb = TensorOps.sub(one, prob)) {
            return TensorOps.pow(oneMinusProb, gamma);
        }
    }

    /** {@code -alpha * f * lp} for the picked log-probabilities. */
    private Tensor classLoss(Tensor picked) {
        try (Tensor focal = focalWeight(picked);
             Tensor scaled = TensorOps.mul(focal, -alpha)) {
            return TensorOps.mul(scaled, picked);
        }
    }

    /** {@code -sum_c a_c * f_c * t_c * lp_c} over the entries with {@code t_c > 0}. */
    private Tensor distributionLoss(Tensor logProbs, int axis, Tensor target) {
        Shape shape = logProbs.shape();
        try (Tensor t = LossTensors.alignTarget(logProbs, target);
             Tensor focal = focalWeight(logProbs);
             Tensor half = LossTensors.constant(logProbs.backend(), 0.5f);
             Tensor confident = TensorOps.gt(t, half);
             Tensor alphaValue = LossTensors.constant(logProbs.backend(), alpha);
             Tensor alphaFull = alphaValue.expand(shape);
             Tensor complementValue = LossTensors.constant(logProbs.backend(), 1.0f - alpha);
             Tensor complementFull = complementValue.expand(shape);
             Tensor alphaWeight = TensorOps.where(confident, alphaFull, complementFull);
             Tensor weighted = TensorOps.mul(alphaWeight, focal);
             Tensor withTarget = TensorOps.mul(weighted, t);
             Tensor products = TensorOps.mul(withTarget, logProbs);
             Tensor terms = LossTensors.positiveOnly(t, products)) {
            return LossTensors.negatedSum(terms, axis);
        }
    }

    @Override
    public String name() {
        return "FocalLoss";
    }

    @Override
    public Reduction reduction() {
        return reduction;
    }

    public float alpha() {
        return alpha;
    }

    public float gamma() {
        return gamma;
    }

    public int dim() {
        return dim;
    }

    public int ignoreIndex() {
        return ignoreIndex;
    }

    @Override
    public String toString() {
        return "FocalLoss{alpha=" + alpha + ", gamma=" + gamma +
               ", reduction=" + reduction + ", dim=" + dim + ", ignoreIndex=" + ignoreIndex + "}";
    }
}

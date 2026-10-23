package com.eonmux.neuralfree.nn.loss;

import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;

import java.util.Arrays;
import java.util.Objects;

/**
 * Triplet margin loss over a batch. The first dim of {@code anchor} is the batch {@code B}; anchor, positive and
 * negative hold {@code B * F} elements each, read row-major as {@code [B, F]}. Per sample, with PyTorch's pairwise
 * p-norm distance {@code dist(a, b) = (sum_f |a - b + eps|^p)^(1/p)}, {@code eps = 1e-6}:
 * {@code max(0, dist(anchor, positive) - dist(anchor, negative) + margin)}, optionally scaled by a per-sample weight,
 * then the reduction over the {@code [B]} result. The {@code eps} keeps the distance away from 0 for identical rows,
 * where the root's derivative would be infinite and the gradient NaN.
 *
 * <p>Composed from {@link TensorOps} (sub, add, abs, pow, sum, relu, mul), so it runs on the device of the operands
 * and nothing is read back; every step records its autograd node, so the backward reaches whichever of the three
 * inputs require grad. The result is in the reduction class ({@linkplain com.eonmux.neuralfree.tensor.gpu numerical
 * contract}); the powers are taken in float arithmetic, with the root's exponent {@code 1/p} rounded to float.</p>
 */
public final class TripletMarginLoss implements LossFunction {

    private static final float DEFAULT_MARGIN = 1.0f;
    private static final float DISTANCE_EPSILON = 1e-6f;

    private final float margin;
    private final float p;
    private final Reduction reduction;

    public TripletMarginLoss() {
        this(DEFAULT_MARGIN, 2.0f, Reduction.MEAN);
    }

    public TripletMarginLoss(float margin) {
        this(margin, 2.0f, Reduction.MEAN);
    }

    public TripletMarginLoss(float margin, float p, Reduction reduction) {
        if (margin < 0) {
            throw new IllegalArgumentException("Margin must be non-negative, got: " + margin);
        }
        if (p <= 0) {
            throw new IllegalArgumentException("p must be positive, got: " + p);
        }
        this.margin = margin;
        this.p = p;
        this.reduction = Objects.requireNonNull(reduction, "Reduction cannot be null");
    }

    /**
     * @throws IllegalArgumentException when the three tensors differ in element count, the batch is empty, or they
     *                                  live on different backends
     */
    public Tensor compute(Tensor anchor, Tensor positive, Tensor negative) {
        return compute(anchor, positive, negative, null);
    }

    @Override
    public Tensor compute(Tensor prediction, Tensor target) {
        throw new UnsupportedOperationException(
            "TripletMarginLoss requires three inputs: compute(anchor, positive, negative)");
    }

    /**
     * The loss with sample {@code b} scaled by {@code weights[b]}; samples beyond the end of {@code weights}, or all
     * samples when it is null, keep weight 1. The weights are uploaded to the anchor's backend.
     *
     * @throws IllegalArgumentException when the three tensors differ in element count, the batch is empty, or they
     *                                  live on different backends
     */
    public Tensor compute(Tensor anchor, Tensor positive, Tensor negative, float[] weights) {
        Objects.requireNonNull(anchor, "Anchor cannot be null");
        Objects.requireNonNull(positive, "Positive cannot be null");
        Objects.requireNonNull(negative, "Negative cannot be null");

        long elements = anchor.numElements();
        if (positive.numElements() != elements || negative.numElements() != elements) {
            throw new IllegalArgumentException("Anchor, positive, and negative must have the same shape");
        }
        if (anchor.shape().rank() == 0 || anchor.shape().dim(0) == 0) {
            throw new IllegalArgumentException("Anchor needs a non-empty batch dim, got shape " + anchor.shape());
        }
        int batchSize = anchor.shape().dim(0);
        int featureSize = (int) (elements / batchSize);

        Tensor loss;
        try (Tensor a = anchor.reshape(batchSize, featureSize);
             Tensor pos = positive.reshape(batchSize, featureSize);
             Tensor neg = negative.reshape(batchSize, featureSize);
             Tensor distPos = distance(a, pos);
             Tensor distNeg = distance(a, neg);
             Tensor gap = TensorOps.sub(distPos, distNeg);
             Tensor shifted = TensorOps.add(gap, margin)) {
            loss = TensorOps.relu(shifted);
        }
        if (weights != null) {
            loss = weighted(loss, weights, batchSize);
        }
        return LossTensors.reduce(loss, reduction);
    }

    /** {@code (sum_f |a - b + eps|^p)^(1/p)} per row of {@code [B, F]} operands. */
    private Tensor distance(Tensor a, Tensor b) {
        try (Tensor diff = TensorOps.sub(a, b);
             Tensor shifted = TensorOps.add(diff, DISTANCE_EPSILON);
             Tensor absDiff = TensorOps.abs(shifted);
             Tensor powered = TensorOps.pow(absDiff, p);
             Tensor sum = TensorOps.sum(powered, 1)) {
            return TensorOps.pow(sum, 1.0f / p);
        }
    }

    /** {@code loss * w} with {@code w[b] = weights[b]}, or 1 past the end of {@code weights}; consumes {@code loss}. */
    private static Tensor weighted(Tensor loss, float[] weights, int batchSize) {
        float[] sampleWeights = new float[batchSize];
        Arrays.fill(sampleWeights, 1.0f);
        System.arraycopy(weights, 0, sampleWeights, 0, Math.min(weights.length, batchSize));
        try (Tensor lossOwned = loss;
             Tensor w = Tensor.fromArray(loss.backend(), sampleWeights)) {
            return TensorOps.mul(lossOwned, w);
        }
    }

    @Override
    public String name() {
        return "TripletMarginLoss";
    }

    @Override
    public Reduction reduction() {
        return reduction;
    }

    public float margin() {
        return margin;
    }

    public float p() {
        return p;
    }

    @Override
    public String toString() {
        return "TripletMarginLoss{margin=" + margin + ", p=" + p + ", reduction=" + reduction + "}";
    }
}

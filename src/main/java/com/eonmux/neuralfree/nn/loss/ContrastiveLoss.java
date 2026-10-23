package com.eonmux.neuralfree.nn.loss;

import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;

import java.util.Objects;

/**
 * Contrastive loss over a batch of embedding pairs. The first dim of {@code embedding1} is the batch {@code B}; both
 * embeddings hold {@code B * F} elements read row-major as {@code [B, F]}, and {@code label} holds one element per
 * sample. Per sample, with {@code d2 = sum_f (e1 - e2)^2} and {@code d = sqrt(d2 + 1e-8)}:
 * {@code (1 - label) * d2 + label * max(0, margin - d)^2}, then the reduction over the {@code [B]} result.
 *
 * <p>Composed from {@link TensorOps} (sub, mul, sum, add, sqrt, relu), so it runs on the device of the operands and
 * nothing is read back. The result is in the reduction class ({@linkplain com.eonmux.neuralfree.tensor.gpu numerical
 * contract}); the {@code 1e-8} is added in float arithmetic.</p>
 */
public final class ContrastiveLoss implements LossFunction {

    private static final float DEFAULT_MARGIN = 1.0f;
    private static final float DISTANCE_EPSILON = 1e-8f;

    private final float margin;
    private final Reduction reduction;

    public ContrastiveLoss() {
        this(DEFAULT_MARGIN, Reduction.MEAN);
    }

    public ContrastiveLoss(float margin) {
        this(margin, Reduction.MEAN);
    }

    public ContrastiveLoss(float margin, Reduction reduction) {
        if (margin < 0) {
            throw new IllegalArgumentException("Margin must be non-negative, got: " + margin);
        }
        this.margin = margin;
        this.reduction = Objects.requireNonNull(reduction, "Reduction cannot be null");
    }

    /**
     * @throws IllegalArgumentException when the embeddings differ in element count, the batch is empty, the label
     *                                  does not hold one element per sample, or the tensors live on different
     *                                  backends
     */
    public Tensor compute(Tensor embedding1, Tensor embedding2, Tensor label) {
        Objects.requireNonNull(embedding1, "Embedding1 cannot be null");
        Objects.requireNonNull(embedding2, "Embedding2 cannot be null");
        Objects.requireNonNull(label, "Label cannot be null");

        if (embedding1.numElements() != embedding2.numElements()) {
            throw new IllegalArgumentException("Embeddings must have the same shape");
        }
        if (embedding1.shape().rank() == 0 || embedding1.shape().dim(0) == 0) {
            throw new IllegalArgumentException("Embeddings need a non-empty batch dim, got shape "
                    + embedding1.shape());
        }
        int batchSize = embedding1.shape().dim(0);
        int featureSize = (int) (embedding1.numElements() / batchSize);
        if (label.numElements() != batchSize) {
            throw new IllegalArgumentException("Label has " + label.numElements() + " elements for a batch of "
                    + batchSize);
        }

        Tensor loss;
        try (Tensor e1 = embedding1.reshape(batchSize, featureSize);
             Tensor e2 = embedding2.reshape(batchSize, featureSize);
             Tensor y = label.reshape(batchSize);
             Tensor diff = TensorOps.sub(e1, e2);
             Tensor squared = TensorOps.mul(diff, diff);
             Tensor distSquared = TensorOps.sum(squared, 1);
             Tensor shifted = TensorOps.add(distSquared, DISTANCE_EPSILON);
             Tensor dist = TensorOps.sqrt(shifted);
             Tensor marginValue = LossTensors.constant(embedding1.backend(), margin);
             Tensor gap = TensorOps.sub(marginValue, dist);
             Tensor hinge = TensorOps.relu(gap);
             Tensor hingeSquared = TensorOps.mul(hinge, hinge);
             Tensor one = LossTensors.constant(embedding1.backend(), 1.0f);
             Tensor oneMinusY = TensorOps.sub(one, y);
             Tensor similar = TensorOps.mul(oneMinusY, distSquared);
             Tensor dissimilar = TensorOps.mul(y, hingeSquared)) {
            loss = TensorOps.add(similar, dissimilar);
        }
        return LossTensors.reduce(loss, reduction);
    }

    @Override
    public Tensor compute(Tensor prediction, Tensor target) {
        throw new UnsupportedOperationException(
            "ContrastiveLoss requires three inputs: compute(embedding1, embedding2, label)");
    }

    @Override
    public String name() {
        return "ContrastiveLoss";
    }

    @Override
    public Reduction reduction() {
        return reduction;
    }

    public float margin() {
        return margin;
    }

    @Override
    public String toString() {
        return "ContrastiveLoss{margin=" + margin + ", reduction=" + reduction + "}";
    }
}

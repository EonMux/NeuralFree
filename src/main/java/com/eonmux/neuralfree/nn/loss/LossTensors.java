package com.eonmux.neuralfree.nn.loss;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.tensor.DType;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;

import java.util.function.UnaryOperator;

/**
 * Building blocks shared by the loss compositions. Every helper composes {@link TensorOps} and Tensor views, so a loss
 * runs on the train kernels where its operands' backend has them and on the CPU reference otherwise, nothing is read
 * back to the host, and each differentiable step records its autograd node: a loss requires grad exactly when grad
 * mode is on and an operand it is computed from requires grad, and its backward reaches those operands. Constants
 * (zero fills, scalar constants, comparison masks and class indices) never record. Helpers return new tensors that
 * the caller owns; they never close the tensors they are given unless the method says so. Closing an intermediate
 * result does not cut the recorded graph: the nodes keep the handles their backward needs.
 */
final class LossTensors {

    /** PyTorch's default {@code ignore_index} of the class-index losses. */
    static final int DEFAULT_IGNORE_INDEX = -100;

    /**
     * {@code 2^24}: the bound below which every int is exact in FLOAT32. Integer class-index targets support at most
     * this many classes, and an ignore index must lie strictly between its negation and itself.
     */
    static final int MAX_INTEGER_TARGET_CLASSES = 1 << 24;

    /**
     * The log-probability the sample loss of an ignored sample is evaluated at in place of the gathered one: finite
     * and below 0, so every sample loss and its derivative are finite there. An ignored sample's gradient is then
     * exactly 0 even when its gathered log-probability is 0 or {@code -inf}, where a sample loss's derivative can be
     * infinite and a zero upstream gradient would turn into NaN.
     */
    private static final float IGNORED_LOG_PROB = -1.0f;

    private LossTensors() {
    }

    /**
     * {@code perElement} reduced as {@code reduction} asks: itself for {@link LossFunction.Reduction#NONE}, otherwise
     * the scalar sum or mean of all its elements. {@code perElement} is consumed: for SUM and MEAN it is closed, even
     * when the reduction throws.
     */
    static Tensor reduce(Tensor perElement, LossFunction.Reduction reduction) {
        switch (reduction) {
            case NONE:
                return perElement;
            case SUM:
                try {
                    return TensorOps.sum(perElement);
                } finally {
                    perElement.close();
                }
            case MEAN:
            default:
                try {
                    return TensorOps.mean(perElement);
                } finally {
                    perElement.close();
                }
        }
    }

    /**
     * A FLOAT32 scalar holding exactly {@code value} (for {@code -0.0f} it holds {@code +0.0f}) on {@code backend}: a
     * zero fill plus a scalar add, so where the train kernels are available it is made on the device without a host
     * upload.
     */
    static Tensor constant(ComputeBackend backend, float value) {
        try (Tensor zero = Tensor.zeros(backend, Shape.scalar())) {
            return TensorOps.add(zero, value);
        }
    }

    /**
     * A view of {@code target} with {@code prediction}'s shape, pairing the elements of the two in row-major order as
     * an element-by-element loss does.
     *
     * @throws IllegalArgumentException when the element counts differ
     */
    static Tensor alignTarget(Tensor prediction, Tensor target) {
        if (target.numElements() != prediction.numElements()) {
            throw new IllegalArgumentException("Target has " + target.numElements() + " elements but prediction "
                    + prediction.shape() + " has " + prediction.numElements());
        }
        return target.reshape(prediction.shape());
    }

    /** {@code 0 - sum(terms, axis)}: the per-sample loss of {@code sampleLoss -= term} over the class axis. */
    static Tensor negatedSum(Tensor terms, int axis) {
        try (Tensor sum = TensorOps.sum(terms, axis);
             Tensor zero = constant(terms.backend(), 0.0f)) {
            return TensorOps.sub(zero, sum);
        }
    }

    /**
     * {@code where(mask > 0, values, 0)}, the sum terms of entries whose target weight is positive (a NaN or
     * non-positive weight contributes nothing, and a zero weight never multiplies an infinite log-probability).
     */
    static Tensor positiveOnly(Tensor mask, Tensor values) {
        try (Tensor zero = constant(mask.backend(), 0.0f);
             Tensor positive = TensorOps.gt(mask, zero);
             Tensor zeros = Tensor.zeros(values.backend(), values.shape())) {
            return TensorOps.where(positive, values, zeros);
        }
    }

    // ------------------------------------------------------------------ class axis

    /**
     * The per-sample loss of a prediction whose class axis is empty: {@code sum(prediction, axis)}, the empty sum,
     * which is 0 for every sample. It is recorded like any sum, so the loss stays connected to the prediction (whose
     * gradient then has no elements), as a sum over no classes is in PyTorch.
     */
    static Tensor emptyClassLoss(Tensor prediction, int axis) {
        return TensorOps.sum(prediction, axis);
    }

    /**
     * {@code dim} as an axis of {@code shape}, counting from the end when negative.
     *
     * @throws IllegalArgumentException when the shape is a scalar or {@code dim} is outside {@code [-rank, rank)}
     */
    static int classAxis(Shape shape, int dim) {
        int rank = shape.rank();
        int axis = dim < 0 ? rank + dim : dim;
        if (rank == 0 || axis < 0 || axis >= rank) {
            throw new IllegalArgumentException("Class dim " + dim + " is out of range for prediction shape " + shape);
        }
        return axis;
    }

    /** {@code shape} without {@code axis}; a scalar shape for rank 1. */
    static Shape withoutAxis(Shape shape, int axis) {
        int rank = shape.rank();
        if (rank == 1) {
            return Shape.scalar();
        }
        int[] dims = new int[rank - 1];
        int j = 0;
        for (int i = 0; i < rank; i++) {
            if (i != axis) {
                dims[j++] = shape.dim(i);
            }
        }
        return Shape.of(dims);
    }

    /** {@code t} as a tensor of {@code shape}, closing {@code t} when a reshaped view replaces it. */
    static Tensor reshapeOwned(Tensor t, Shape shape) {
        if (t.shape().equals(shape)) {
            return t;
        }
        try {
            return t.reshape(shape);
        } finally {
            t.close();
        }
    }

    /**
     * The reduced loss of class-index targets. Sample {@code s} reads its class from {@code target[s]} (targets in
     * row-major order of the prediction shape without {@code axis}):
     *
     * <ul>
     *   <li>INT32 values as they are, INT64 values narrowed with Java {@code (int)} (the low word, on the device with
     *   {@code TR_Cast_I64_I32}, as {@code nn/layers/Embedding.java} reads its indices);</li>
     *   <li>FLOAT32 values as {@code Math.round(target[s])}: {@link TensorOps#round} followed by a cast to INT32,
     *   NaN becoming class 0 and out-of-int-range values saturating.</li>
     * </ul>
     *
     * <p>A sample whose class equals {@code ignoreIndex} is ignored: its loss is 0 and it does not count towards
     * the MEAN denominator, which is the number of samples that are not ignored (0/0, NaN, when every sample is
     * ignored, as in PyTorch). Every other sample's loss is {@code sampleLoss(logProbs[class])}; its class must lie in
     * {@code [0, classes)}, and the gather that reads it validates the range: on the CPU path it throws
     * {@link IndexOutOfBoundsException} at once, on the device the gather writes 0 for that sample and raises the
     * exception, with the gather's message, at the next host synchronization of the backend. The
     * class values and the ignore mask are constants: they are computed without recording and get no gradient. An
     * ignored sample never reads its gathered log-probability: {@code sampleLoss} sees {@code -1} in its place, and
     * the sample's gradient is exactly 0.</p>
     *
     * @param logProbs    log-probabilities over the class axis
     * @param axis        the class axis of {@code logProbs}, of size at least 1
     * @param target      one class value per sample, INT32, INT64 or FLOAT32
     * @param ignoreIndex the class value of ignored samples, validated by {@link #requireIgnoreIndex}
     * @param reduction   the reduction of the per-sample losses
     * @param sampleLoss  maps the picked log-probabilities (prediction shape with the class axis of size 1, -1 for the
     *                    ignored samples) to the per-sample loss of the same shape; it must not close its argument
     * @return the per-sample loss with the class axis removed for NONE, otherwise the scalar SUM or MEAN
     * @throws IllegalArgumentException  when {@code target} does not hold one element per sample, its dtype is not
     *                                   INT32, INT64 or FLOAT32, an integer target meets more than
     *                                   {@link #MAX_INTEGER_TARGET_CLASSES} classes, or the tensors live on different
     *                                   backends
     * @throws IndexOutOfBoundsException on the CPU path, when a class that is not ignored lies outside
     *                                   {@code [0, classes)}
     */
    static Tensor classIndexLoss(Tensor logProbs, int axis, Tensor target, int ignoreIndex,
                                 LossFunction.Reduction reduction, UnaryOperator<Tensor> sampleLoss) {
        Shape shape = logProbs.shape();
        int classes = shape.dim(axis);
        int[] keepDims = shape.dims();
        keepDims[axis] = 1;
        Shape keepShape = Shape.of(keepDims);
        DType dtype = target.dtype();
        if (dtype != DType.FLOAT32 && dtype != DType.INT32 && dtype != DType.INT64) {
            throw new IllegalArgumentException("Class-index targets must be INT32, INT64 or FLOAT32, got " + dtype);
        }
        if (dtype != DType.FLOAT32 && classes > MAX_INTEGER_TARGET_CLASSES) {
            throw new IllegalArgumentException(dtype + " class-index targets support at most "
                    + MAX_INTEGER_TARGET_CLASSES + " classes, the prediction has " + classes);
        }
        if (target.numElements() != keepShape.numElements()) {
            throw new IllegalArgumentException("Class-index target has " + target.numElements()
                    + " elements but prediction " + shape + " has " + keepShape.numElements() + " samples");
        }
        try (ClassTargets targets = ClassTargets.resolve(target, keepShape, ignoreIndex)) {
            Tensor perSample;
            try (Tensor gathered = TensorOps.gather(logProbs, axis, targets.index);
                 Tensor zeros = Tensor.zeros(logProbs.backend(), keepShape);
                 Tensor filler = TensorOps.add(zeros, IGNORED_LOG_PROB);
                 Tensor picked = TensorOps.where(targets.ignored, filler, gathered);
                 Tensor loss = sampleLoss.apply(picked)) {
                perSample = TensorOps.where(targets.ignored, zeros, loss);
            }
            perSample = reshapeOwned(perSample, withoutAxis(shape, axis));
            return reduction == LossFunction.Reduction.MEAN
                    ? meanOverKept(perSample, targets.ignored)
                    : reduce(perSample, reduction);
        }
    }

    /**
     * {@code ignoreIndex} itself when every class value that can equal it compares exactly in FLOAT32, where the
     * ignore mask is computed: {@code |ignoreIndex| < 2^24}.
     *
     * @throws IllegalArgumentException otherwise
     */
    static int requireIgnoreIndex(int ignoreIndex) {
        if (ignoreIndex <= -MAX_INTEGER_TARGET_CLASSES || ignoreIndex >= MAX_INTEGER_TARGET_CLASSES) {
            throw new IllegalArgumentException("Ignore index must lie in (-" + MAX_INTEGER_TARGET_CLASSES + ", "
                    + MAX_INTEGER_TARGET_CLASSES + "), got: " + ignoreIndex);
        }
        return ignoreIndex;
    }

    /** {@code sum(perSample) / count(not ignored)}; consumes {@code perSample}. */
    private static Tensor meanOverKept(Tensor perSample, Tensor ignored) {
        try (Tensor total = TensorOps.sum(perSample);
             Tensor count = keptCount(ignored)) {
            return TensorOps.div(total, count);
        } finally {
            perSample.close();
        }
    }

    /** The number of samples that are not ignored, as a FLOAT32 scalar constant. */
    private static Tensor keptCount(Tensor ignored) {
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad();
             Tensor ignoredValue = ignored.to(DType.FLOAT32);
             Tensor one = constant(ignored.backend(), 1.0f);
             Tensor kept = TensorOps.sub(one, ignoredValue)) {
            return TensorOps.sum(kept);
        }
    }

    /**
     * Class-index targets resolved for the gather: {@code index}, the INT32 class of every sample with 0 in place of
     * the ignored ones, and {@code ignored}, the BOOL mask of the ignored samples, both of the prediction shape with
     * the class axis of size 1.
     */
    private static final class ClassTargets implements AutoCloseable {
        private final Tensor index;
        private final Tensor ignored;

        private ClassTargets(Tensor index, Tensor ignored) {
            this.index = index;
            this.ignored = ignored;
        }

        /**
         * Resolves {@code target} without recording. The class values pass through FLOAT32 for the mask and the
         * substitution of ignored samples: FLOAT32 targets are whole floats already, and integer targets are exact
         * there up to {@code 2^24} in magnitude; a larger integer is out of range for at most 2^24 classes and stays
         * out of range after the conversion, so the gather still reports it.
         */
        static ClassTargets resolve(Tensor target, Shape keepShape, int ignoreIndex) {
            ComputeBackend backend = target.backend();
            Tensor ignored = null;
            try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad();
                 Tensor t = target.reshape(keepShape);
                 Tensor classValue = classValues(t);
                 Tensor ignoreValue = constant(backend, ignoreIndex)) {
                ignored = TensorOps.eq(classValue, ignoreValue);
                try (Tensor zeros = Tensor.zeros(backend, keepShape);
                     Tensor safe = TensorOps.where(ignored, zeros, classValue)) {
                    return new ClassTargets(safe.to(DType.INT32), ignored);
                }
            } catch (RuntimeException e) {
                if (ignored != null) {
                    ignored.close();
                }
                throw e;
            }
        }

        /** The whole-number FLOAT32 class value of every element of {@code t}, as a new tensor. */
        private static Tensor classValues(Tensor t) {
            switch (t.dtype()) {
                case FLOAT32:
                    try (Tensor rounded = TensorOps.round(t);
                         Tensor roundedInt = rounded.to(DType.INT32)) {
                        return roundedInt.to(DType.FLOAT32);
                    }
                case INT64:
                    try (Tensor narrowed = t.to(DType.INT32)) {
                        return narrowed.to(DType.FLOAT32);
                    }
                case INT32:
                    return t.to(DType.FLOAT32);
                default:
                    throw new IllegalStateException("unreachable class-index dtype " + t.dtype());
            }
        }

        @Override
        public void close() {
            try {
                index.close();
            } finally {
                ignored.close();
            }
        }
    }
}

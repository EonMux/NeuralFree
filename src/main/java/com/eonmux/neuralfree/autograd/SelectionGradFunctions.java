package com.eonmux.neuralfree.autograd;

import com.eonmux.neuralfree.tensor.BroadcastEngine;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * Backward functions of the selection, reduction and scatter ops of {@link TensorOps} that {@link GradFunctions} does
 * not cover: {@code where}, {@code min}, {@code var}, {@code sumToShape} and {@code scatterAdd}. Every backward is a
 * composition of {@link TensorOps} calls and tensor views, so it runs on the device when the train library is
 * available on the gradient's backend and on the CPU reference path otherwise, and follows the ownership contract of
 * {@link GradFunctions}: {@code gradOutput} is never closed, every temporary is closed, and each returned gradient is
 * owned by the caller unless it is {@code gradOutput} itself, which the graph then takes a new reference of.
 *
 * <p>{@code supportsCreateGraph()} is true for every class: each backward consists of recorded ops only (the backward
 * primitive {@code maxBackward} records its second-order function inside a create-graph pass) and never takes a
 * detached handle; saved tensors are read through {@code SavingGradFn.saved}, which routes second-order gradients to
 * the forward tensors inside a create-graph pass.</p>
 */
public final class SelectionGradFunctions {

    private SelectionGradFunctions() {
    }

    /**
     * {@code where(condition, a, b)} over one shape, recorded with the inputs {@code (condition, a, b)}:
     * {@code ga = where(condition, gy, 0)}, {@code gb = where(condition, 0, gy)}; the condition gets no gradient.
     * Saved: the condition. Tolerance: bit-identical (data movement).
     */
    public static class WhereGrad extends GradFunctions.SavingGradFn {
        private final boolean needA;
        private final boolean needB;

        public WhereGrad() {
            this(true, true);
        }

        /** {@code needA}/{@code needB}: whether the operand requires a gradient (else its entry is null). */
        public WhereGrad(boolean needA, boolean needB) {
            this.needA = needA;
            this.needB = needB;
        }

        /** Saves the condition (FLOAT32 or BOOL, the shape of both operands). */
        public void saveForBackward(Tensor condition) {
            save(condition);
        }

        @Override
        public List<Tensor> backward(Tensor gradOutput) {
            Tensor condition = saved(0);
            if (!condition.shape().equals(gradOutput.shape())) {
                throw new IllegalArgumentException("where gradient " + gradOutput.shape()
                        + " does not match the condition shape " + condition.shape());
            }
            if (!needA && !needB) {
                return Arrays.asList(null, null, null);
            }
            try (Tensor zeros = Tensor.zeros(gradOutput.backend(), gradOutput.shape(), gradOutput.dtype())) {
                Tensor ga = needA ? TensorOps.where(condition, gradOutput, zeros) : null;
                try {
                    Tensor gb = needB ? TensorOps.where(condition, zeros, gradOutput) : null;
                    return Arrays.asList(null, ga, gb);
                } catch (RuntimeException e) {
                    throw closeAfterFailure(ga, e);
                }
            }
        }

        @Override
        public boolean supportsCreateGraph() {
            return true;
        }
    }

    /**
     * {@code min(x, dim)}: zeros of the input's shape except {@code gy} at the first index along {@code dim} whose
     * value equals the saved minimum ({@link TensorOps#maxBackward}, whose equality routing is independent of the
     * extremum's direction; ties route to the first index, a NaN minimum routes nowhere). Saved: the input and the
     * output. Tolerance: bit-identical. Supports create-graph: {@code maxBackward} records
     * {@link SecondOrderGradFunctions.MaxBackwardGrad} inside a create-graph pass.
     */
    public static class MinGrad extends GradFunctions.SavingGradFn {
        private final int dim;
        private final Shape inputShape;

        public MinGrad(int dim, Shape inputShape) {
            this.dim = dim;
            this.inputShape = Objects.requireNonNull(inputShape, "inputShape");
        }

        public void saveForBackward(Tensor input, Tensor output) {
            if (!input.shape().equals(inputShape)) {
                throw new IllegalArgumentException("saved input shape " + input.shape() + " does not match "
                        + inputShape);
            }
            save(input, output);
        }

        @Override
        public List<Tensor> backward(Tensor gradOutput) {
            return Collections.singletonList(TensorOps.maxBackward(saved(0), saved(1), gradOutput, dim));
        }

        @Override
        public boolean supportsCreateGraph() {
            return true;
        }
    }

    /**
     * Population variance {@code var(x, dims)} ({@code null} or no dims: every element):
     * {@code gx = ((x - mean(x, dims, keep)) * (2 / n)) * gy_b}, with {@code n} the number of reduced elements
     * ({@code 2 / n} in float) and {@code gy_b} the gradient viewed with the reduced dims kept as 1 and broadcast over
     * them. Saved: the input. Tolerance: reduction, K = n (the recomputed mean).
     */
    public static class VarGrad extends GradFunctions.SavingGradFn {
        private final Shape inputShape;
        private final int[] meanDims;
        private final Shape keptShape;
        private final float scale;

        public VarGrad(Shape inputShape, int[] dims) {
            this.inputShape = Objects.requireNonNull(inputShape, "inputShape");
            int[] reduced = reducedDims(inputShape, dims);
            this.meanDims = dims == null || dims.length == 0 ? new int[0] : reduced;
            this.keptShape = BroadcastEngine.reduceShape(inputShape, reduced, true);
            long n = 1;
            for (int d : reduced) {
                n *= inputShape.dim(d);
            }
            this.scale = 2.0f / n;
        }

        public void saveForBackward(Tensor input) {
            if (!input.shape().equals(inputShape)) {
                throw new IllegalArgumentException("saved input shape " + input.shape() + " does not match "
                        + inputShape);
            }
            save(input);
        }

        @Override
        public List<Tensor> backward(Tensor gradOutput) {
            Tensor x = saved(0);
            try (Tensor mean = TensorOps.mean(x, meanDims);
                 Tensor meanKept = mean.reshape(keptShape);
                 Tensor centered = TensorOps.sub(x, meanKept);
                 Tensor scaled = TensorOps.mul(centered, scale);
                 Tensor gyKept = gradOutput.reshape(keptShape)) {
                return Collections.singletonList(TensorOps.mul(scaled, gyKept));
            }
        }

        @Override
        public boolean supportsCreateGraph() {
            return true;
        }

        /** {@code dims} normalized against {@code shape}; null or empty means every dim. */
        private static int[] reducedDims(Shape shape, int[] dims) {
            if (dims == null || dims.length == 0) {
                int[] all = new int[shape.rank()];
                for (int i = 0; i < all.length; i++) {
                    all[i] = i;
                }
                return all;
            }
            int[] normalized = BroadcastEngine.normalizeAxes(dims, shape.rank());
            boolean[] seen = new boolean[shape.rank()];
            for (int d : normalized) {
                if (seen[d]) {
                    throw new IllegalArgumentException("dim " + d + " is reduced twice in " + Arrays.toString(dims));
                }
                seen[d] = true;
            }
            return normalized;
        }
    }

    /**
     * {@code sumToShape(t, target)}, classified as the forward does. Broadcasting case ({@code target} broadcasts to
     * {@code t}'s shape and differs from it; the prepended dims and the dims where {@code target} has 1 and {@code t}
     * more were summed): {@code gy} viewed with the summed dims kept as 1, expanded to {@code t}'s shape and made
     * contiguous. Rearranging case (equal element counts, nothing summed): {@code gy.reshape(t.shape)}. Saved: nothing.
     * Tolerance: bit-identical (data movement).
     */
    public static class SumToShapeGrad implements GradientFunction {
        private final Shape sourceShape;
        private final Shape keptShape;

        /**
         * @param sourceShape the shape of the summed tensor {@code t}
         * @param target      the forward's target shape
         * @throws IllegalArgumentException when {@code target} neither broadcasts to {@code sourceShape} nor has its
         *                                  element count
         */
        public SumToShapeGrad(Shape sourceShape, Shape target) {
            this.sourceShape = Objects.requireNonNull(sourceShape, "sourceShape");
            Objects.requireNonNull(target, "target");
            if (!sourceShape.equals(target) && broadcastsTo(target, sourceShape)) {
                int[] kept = new int[sourceShape.rank()];
                Arrays.fill(kept, 1);
                int offset = sourceShape.rank() - target.rank();
                for (int i = 0; i < target.rank(); i++) {
                    kept[offset + i] = target.dim(i);
                }
                this.keptShape = Shape.of(kept);
            } else if (sourceShape.numElements() == target.numElements()) {
                this.keptShape = null;
            } else {
                throw new IllegalArgumentException("target shape " + target + " neither broadcasts to "
                        + sourceShape + " nor has its " + sourceShape.numElements() + " elements");
            }
        }

        @Override
        public List<Tensor> backward(Tensor gradOutput) {
            if (keptShape == null) {
                return Collections.singletonList(gradOutput.reshape(sourceShape));
            }
            try (Tensor keptView = gradOutput.reshape(keptShape)) {
                Tensor expanded = keptView.expand(sourceShape);
                Tensor dense;
                try {
                    dense = expanded.contiguous();
                } catch (RuntimeException e) {
                    throw closeAfterFailure(expanded, e);
                }
                if (dense != expanded) {
                    expanded.close();
                }
                return Collections.singletonList(dense);
            }
        }

        @Override
        public List<Tensor> savedTensors() {
            return Collections.emptyList();
        }

        @Override
        public boolean supportsCreateGraph() {
            return true;
        }

        /**
         * Whether broadcasting {@code target} against {@code shape} gives {@code shape} (the forward's broadcasting
         * test): no higher rank, and every aligned dim equal or 1 where {@code shape} has more than 1.
         */
        private static boolean broadcastsTo(Shape target, Shape shape) {
            if (target.rank() > shape.rank()) {
                return false;
            }
            int offset = shape.rank() - target.rank();
            for (int i = 0; i < target.rank(); i++) {
                int t = target.dim(i);
                int s = shape.dim(offset + i);
                if (t != s && !(t == 1 && s > 1)) {
                    return false;
                }
            }
            return true;
        }
    }

    /**
     * {@code scatterAdd(input, dim, index, src)}, recorded with the inputs {@code (input, index, src)}: the input gets
     * {@code gy} itself, the source gets {@code gather(gy, dim, index)}; the index gets no gradient. Saved: the index.
     * Tolerance: bit-identical (data movement).
     */
    public static class ScatterAddGrad extends GradFunctions.SavingGradFn {
        private final int dim;
        private final boolean needInput;
        private final boolean needSrc;

        public ScatterAddGrad(int dim) {
            this(dim, true, true);
        }

        /** {@code needInput}/{@code needSrc}: whether the operand requires a gradient (else its entry is null). */
        public ScatterAddGrad(int dim, boolean needInput, boolean needSrc) {
            this.dim = dim;
            this.needInput = needInput;
            this.needSrc = needSrc;
        }

        /** Saves the index; {@code src} must have the index's shape, which is the shape of its gradient. */
        public void saveForBackward(Tensor index, Tensor src) {
            if (!index.shape().equals(src.shape())) {
                throw new IllegalArgumentException("Index and source must have the same shape");
            }
            save(index);
        }

        @Override
        public List<Tensor> backward(Tensor gradOutput) {
            Tensor index = saved(0);
            Tensor gSrc = needSrc ? TensorOps.gather(gradOutput, dim, index) : null;
            return Arrays.asList(needInput ? gradOutput : null, null, gSrc);
        }

        @Override
        public boolean supportsCreateGraph() {
            return true;
        }
    }

    /** Closes {@code t} when non-null, attaching a close failure to {@code failure}, and returns {@code failure}. */
    private static RuntimeException closeAfterFailure(Tensor t, RuntimeException failure) {
        if (t != null) {
            try {
                t.close();
            } catch (RuntimeException e) {
                failure.addSuppressed(e);
            }
        }
        return failure;
    }
}

package com.eonmux.neuralfree.autograd;

import com.eonmux.neuralfree.tensor.GpuReduceOps;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;

import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * Backward functions of the row-index, segment and dropout ops of {@link TensorOps}: {@code indexSelectRows},
 * {@code segmentReduce} and {@code maskScale}. Each backward is a {@link TensorOps} call, so it runs on the device when
 * the train library is available on the gradient's backend and on the CPU reference path otherwise, and follows the
 * ownership contract of {@link GradFunctions}: {@code gradOutput} is never closed, every temporary is closed, and each
 * returned gradient is a new tensor owned by the caller. Index operands, the CSR arrays and the dropout mask are not
 * differentiable: their entries are null. Every class supports create-graph (rule G4): the backward primitives it
 * calls record their {@link SecondOrderGradFunctions} inside a create-graph pass.
 */
public final class IndexGradFunctions {

    private IndexGradFunctions() {
    }

    /**
     * {@code indexSelectRows(weight, indices, clamp, paddingIdx)}, recorded with the inputs {@code (weight, indices)}:
     * {@code gW = indexAddRows(gy viewed as [n, dim], indices, rows, clamp, paddingIdx)}, the sum of the gradient rows
     * of every selection of a table row, added in ascending selection order; the padding row (when
     * {@code paddingIdx >= 0}) and, without {@code clamp}, out-of-range indices contribute nothing. Saved: the
     * indices. Tolerance: single-rounding (the index-add of {@link TensorOps#indexAddRows}). Supports create-graph:
     * {@code indexAddRows} records {@link SecondOrderGradFunctions.IndexAddRowsGrad} inside a create-graph pass.
     */
    public static class IndexSelectRowsGrad extends GradFunctions.SavingGradFn {
        private final int rows;
        private final int dim;
        private final boolean clamp;
        private final int paddingIdx;

        /**
         * @param weightShape the {@code [rows, dim]} table shape
         * @param clamp       whether the forward clamped indices into the table
         * @param paddingIdx  the row that gets no gradient, or a negative value for none
         * @throws IllegalArgumentException when {@code weightShape} is not 2-D
         */
        public IndexSelectRowsGrad(Shape weightShape, boolean clamp, int paddingIdx) {
            Objects.requireNonNull(weightShape, "weightShape");
            if (weightShape.rank() != 2) {
                throw new IllegalArgumentException("indexSelectRows needs a 2-D table [rows, dim], got "
                        + weightShape);
            }
            this.rows = weightShape.dim(0);
            this.dim = weightShape.dim(1);
            this.clamp = clamp;
            this.paddingIdx = paddingIdx;
        }

        /** Saves the INT32 or INT64 indices of the forward. */
        public void saveForBackward(Tensor indices) {
            save(indices);
        }

        @Override
        public List<Tensor> backward(Tensor gradOutput) {
            Tensor indices = saved(0);
            int n = Math.toIntExact(indices.numElements());
            try (Tensor rowsView = gradOutput.reshape(Shape.of(n, dim))) {
                return Arrays.asList(TensorOps.indexAddRows(rowsView, indices, rows, clamp, paddingIdx), null);
            }
        }

        @Override
        public boolean supportsCreateGraph() {
            return true;
        }
    }

    /**
     * {@code segmentReduce(src, csrPtr, csrEdges, numNodes, op, emptyValue)}, recorded with the inputs
     * {@code (src, csrPtr, csrEdges)}: {@code gsrc = segmentReduceBackward(src, csrPtr, csrEdges, out, gy, op)}. SUM
     * gives every source row of a segment the segment's gradient row, MEAN that row divided by the segment's edge
     * count, MAX gives it to the first source row (ascending edge order) whose value equals the forward result; rows
     * no segment names get 0. Precondition: each source row appears at most once in {@code csrEdges} (GraphData
     * CSRs). Saved: the CSR arrays, the source and, for MAX, the output. Tolerance: SUM and MAX bit-identical, MEAN
     * division. Supports create-graph: {@code segmentReduceBackward} records
     * {@link SecondOrderGradFunctions.SegmentReduceBackwardGrad} inside a create-graph pass.
     */
    public static class SegmentReduceGrad extends GradFunctions.SavingGradFn {
        private final GpuReduceOps.SegmentOp op;

        public SegmentReduceGrad(GpuReduceOps.SegmentOp op) {
            this.op = Objects.requireNonNull(op, "op");
        }

        /**
         * Saves the forward's operands; {@code output} (the forward result) is saved only for
         * {@link GpuReduceOps.SegmentOp#MAX} and may be null otherwise.
         */
        public void saveForBackward(Tensor src, Tensor csrPtr, Tensor csrEdges, Tensor output) {
            if (op == GpuReduceOps.SegmentOp.MAX) {
                save(src, csrPtr, csrEdges, Objects.requireNonNull(output, "MAX saves the forward output"));
            } else {
                save(src, csrPtr, csrEdges);
            }
        }

        @Override
        public List<Tensor> backward(Tensor gradOutput) {
            Tensor out = op == GpuReduceOps.SegmentOp.MAX ? saved(3) : null;
            Tensor gsrc = TensorOps.segmentReduceBackward(saved(0), saved(1), saved(2), out, gradOutput, op);
            return Arrays.asList(gsrc, null, null);
        }

        @Override
        public boolean supportsCreateGraph() {
            return true;
        }
    }

    /**
     * {@code maskScale(x, mask, scale)} (dropout), recorded with the inputs {@code (x, mask)}:
     * {@code gx = maskScale(gy, mask, scale)}, the kernel formula {@code mask == +-0 ? +0 : (gy * mask) * scale} in
     * that evaluation order on both paths; the mask gets no gradient. Saved: the mask. Tolerance: single-rounding.
     * Supports create-graph: the backward is one recorded {@code maskScale}, linear in {@code gy}.
     */
    public static class MaskScaleGrad extends GradFunctions.SavingGradFn {
        private final float scale;

        public MaskScaleGrad(float scale) {
            this.scale = scale;
        }

        /** Saves the mask of the forward (the shape of {@code x} and of the gradient). */
        public void saveForBackward(Tensor mask) {
            save(mask);
        }

        @Override
        public List<Tensor> backward(Tensor gradOutput) {
            return Arrays.asList(TensorOps.maskScale(gradOutput, saved(0), scale), null);
        }

        @Override
        public boolean supportsCreateGraph() {
            return true;
        }
    }
}

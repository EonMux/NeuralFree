package com.eonmux.neuralfree.tensor;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.tensor.gpu.StridedView;
import com.eonmux.neuralfree.tensor.gpu.TrainGpu;
import com.eonmux.neuralfree.tensor.gpu.TrainOps;
import com.eonmux.neuralfree.tensor.gpu.TrainOps.Kernels;

/**
 * The backward primitives of row index-select and CSR segment reduction on the train kernels
 * ({@code TR_IndexAddRows_F32} in {@code kernels/train/norm.cl}, {@code TR_SegmentReduceBackward_F32} in
 * {@code kernels/train/segment.cl}, INT64 indices narrowed by {@code TR_Cast_I64_I32}). Results are new contiguous
 * FLOAT32 tensors on the gradient's backend that never require grad; nothing is read back to the host.
 *
 * <p><b>Dispatcher contract.</b> Every method returns {@code null} exactly when the train library is unavailable on
 * the gradient's backend ({@link TrainGpu#forBackend}) or an operand's dtype is not the documented one; that is the
 * caller's CPU path ({@link CpuIndexing}), and nothing has been enqueued then. Invalid arguments throw the exceptions
 * of the CPU reference whether or not the library is available (the checks are shared). Once the library is
 * available, kernel and backend failures propagate.</p>
 *
 * <p><b>Determinism.</b> Every output element has one writer that adds or copies in the CPU reference's order, and
 * no atomics are used: {@link #indexAddRows} is bit-identical to the sequential sum for normal values
 * (single-rounding class); {@link #segmentReduceBackward} is bit-identical for SUM and MAX and in the division class
 * for MEAN ({@linkplain com.eonmux.neuralfree.tensor.gpu numerical contract}).</p>
 */
public final class GpuIndexGradOps {

    /** Upper bound of the row blocks of {@code TR_IndexAddRows_F32} (the norm.cl host shape rule). */
    private static final int MAX_ROW_BLOCKS = 64;
    /** Work-items {@code dim * row_blocks} the row-block rule aims for. */
    private static final int TARGET_ITEMS = 16384;

    private GpuIndexGradOps() {
    }

    /**
     * Backward of {@link GpuNormOps#indexSelectRows}: {@code gw[k, :] += gy[i, :]} for {@code i} ascending into a
     * zero-filled {@code [rows, dim]} table, {@code k = indices[i]} (clamped into the table with {@code clamp}); an
     * out-of-range index without {@code clamp} (which the forward reported) and {@code k == skipRow} contribute
     * nothing. {@code TR_IndexAddRows_F32} with {@code row_blocks = min(rows, max(1, min(64, ceil(16384/dim))))};
     * nothing is launched when the table or the gradient has no elements.
     *
     * @param gy      FLOAT32 gradient {@code [n, dim]} of the selected rows, any view
     * @param indices n INT32 or INT64 indices of any shape and strides (INT64 narrowed by Java {@code (int)})
     * @param rows    the table's row count
     * @param clamp   whether the forward clamped indices
     * @param skipRow the row that gets no gradient (Embedding's padding row), negative for none
     * @return a new {@code [rows, dim]} FLOAT32 tensor, or null (dispatcher contract)
     * @throws IllegalArgumentException      as {@link CpuIndexing#checkIndexAddRows}, or when the table or the
     *                                       gradient exceeds {@code 2^31-1} elements
     * @throws UnsupportedOperationException when the indices are neither INT32 nor INT64
     */
    public static Tensor indexAddRows(Tensor gy, Tensor indices, int rows, boolean clamp, int skipRow) {
        CpuIndexing.checkIndexAddRows(gy, indices, rows, skipRow);
        int n = gy.shape().dim(0);
        int dim = gy.shape().dim(1);
        Shape outShape = Shape.of(rows, dim);
        requireKernelSize(outShape, "indexAddRows table");
        requireKernelSize(gy.shape(), "indexAddRows gradient");
        TrainGpu g = TrainGpu.forBackend(gy.backend());
        if (g == null) {
            return null;
        }
        Tensor gw = zeros(g, gy, outShape);
        if (gw.numElements() == 0 || n == 0) {
            return gw;
        }
        int rowBlocks = Math.min(rows, Math.max(1, Math.min(MAX_ROW_BLOCKS, ceilDiv(TARGET_ITEMS, dim))));
        int mode = clamp ? TrainOps.TR_IDX_OOB_CLAMP : TrainOps.TR_IDX_OOB_ERROR;
        try (GpuNormOps.Staging s = new GpuNormOps.Staging(g)) {
            StridedView idx = s.int32Indices(indices);
            StridedView grad = s.dense(gy);
            g.launch1d(Kernels.INDEX_ADD_ROWS_F32, (long) dim * rowBlocks,
                    n, dim, rows, rowBlocks, mode, skipRow, grad.offset(), idx.offset(), 0,
                    idx.buffer(), grad.buffer(), gw.storage().gpuBuffer());
            return gw;
        } catch (RuntimeException e) {
            throw closeAfterFailure(gw, e);
        }
    }

    /**
     * Backward of {@link GpuReduceOps#segmentReduce} with respect to {@code src}: a zero-filled FLOAT32 tensor of
     * {@code src}'s shape where, for every segment with edges and feature {@code f} ({@code g = gy[t, f]}), SUM writes
     * {@code g} at each source row of the segment, MEAN writes {@code g / (float) count}, and MAX writes {@code g} at
     * the first source row (ascending edge order) whose value is Java {@code ==} {@code out[t, f]}, the forward result
     * after the empty substitution (a NaN result routes nowhere). {@code TR_SegmentReduceBackward_F32} over
     * {@code numNodes * feat} work-items; {@code src} and {@code out} are bound only for MAX.
     *
     * <p>Precondition (not re-checked on the device): the CSR arrays satisfy the forward's precondition and each
     * source row appears at most once in {@code csrEdges} over {@code [csrPtr[0], csrPtr[numNodes])}, so every
     * gradient element has one writer. The CSR producers of {@code nn/graph/GraphData} build permutations of the edge
     * and node ids and satisfy it; the CPU reference validates it and throws.</p>
     *
     * @param src      the forward's source, rows along dim 0 (FLOAT32; read only for MAX)
     * @param csrPtr   {@code numNodes + 1} INT32 or INT64 offsets
     * @param csrEdges INT32 or INT64 source rows per edge slot
     * @param out      the forward result, required for MAX and ignored otherwise
     * @param gy       FLOAT32 gradient of the forward result
     * @param op       the forward's aggregation
     * @return a new FLOAT32 tensor of {@code src}'s shape, or null (dispatcher contract)
     * @throws IllegalArgumentException as {@link CpuIndexing#checkSegmentBackward}, or when the source or the gradient
     *                                  exceeds {@code 2^31-1} elements
     */
    public static Tensor segmentReduceBackward(Tensor src, Tensor csrPtr, Tensor csrEdges, Tensor out, Tensor gy,
                                               GpuReduceOps.SegmentOp op) {
        int numNodes = CpuIndexing.checkSegmentBackward(src, csrPtr, csrEdges, out, gy, op);
        requireKernelSize(src.shape(), "segmentReduceBackward source");
        requireKernelSize(gy.shape(), "segmentReduceBackward gradient");
        boolean max = op == GpuReduceOps.SegmentOp.MAX;
        if (src.dtype() != DType.FLOAT32 || !isIndexType(csrPtr) || !isIndexType(csrEdges)) {
            return null;
        }
        TrainGpu g = TrainGpu.forBackend(gy.backend());
        if (g == null) {
            return null;
        }
        Tensor gx = zeros(g, gy, src.shape());
        int items = (int) gy.numElements();
        if (items == 0) {
            return gx;
        }
        int feat = items / numNodes;
        try (GpuNormOps.Staging s = new GpuNormOps.Staging(g)) {
            StridedView ptr = s.int32Indices(csrPtr);
            StridedView eid = s.int32Indices(csrEdges);
            StridedView grad = s.dense(gy);
            StridedView values = max ? s.dense(src) : null;
            StridedView result = max ? s.dense(out) : null;
            g.launch1d(Kernels.SEGMENT_REDUCE_BACKWARD_F32, items,
                    op.id(), numNodes, feat, ptr.offset(), eid.offset(), offset(values), offset(result),
                    grad.offset(), 0,
                    ptr.buffer(), eid.buffer(), buffer(g, values), buffer(g, result), grad.buffer(),
                    gx.storage().gpuBuffer());
            return gx;
        } catch (RuntimeException e) {
            throw closeAfterFailure(gx, e);
        }
    }

    // ------------------------------------------------------------------ helpers

    /** A zero-filled contiguous FLOAT32 device tensor of {@code shape} on {@code like}'s backend. */
    private static Tensor zeros(TrainGpu g, Tensor like, Shape shape) {
        Tensor z = GpuCoreOps.zeros(like.backend(), shape, DType.FLOAT32);
        if (z == null) {
            throw new IllegalStateException("train kernels unavailable on " + g.backend().backendType()
                    + " while allocating a gradient");
        }
        return z;
    }

    private static boolean isIndexType(Tensor t) {
        return t.dtype() == DType.INT32 || t.dtype() == DType.INT64;
    }

    /** The train kernel ABI addresses at most {@code 2^31-1} elements. */
    private static void requireKernelSize(Shape shape, String what) {
        if (shape.numElements() > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(what + " " + shape + " exceeds the 2^31-1 elements of the train "
                    + "kernel ABI");
        }
    }

    private static int offset(StridedView v) {
        return v == null ? 0 : v.offset();
    }

    private static GpuBuffer buffer(TrainGpu g, StridedView v) {
        return v == null ? g.placeholder() : v.buffer();
    }

    private static int ceilDiv(int a, int b) {
        return a / b + (a % b == 0 ? 0 : 1);
    }

    private static RuntimeException closeAfterFailure(Tensor t, RuntimeException failure) {
        try {
            t.close();
        } catch (RuntimeException e) {
            failure.addSuppressed(e);
        }
        return failure;
    }
}

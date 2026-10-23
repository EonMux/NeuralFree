package com.eonmux.neuralfree.tensor;

import com.eonmux.neuralfree.tensor.gpu.TrainGpu;

import java.nio.ByteBuffer;
import java.util.Objects;

/**
 * CPU reference of {@link TensorOps}' index ops: gather, scatter (overwrite and add), row selection by index and
 * CSR segment reductions, and the backward primitives of the last two (row index-add and segment backward), which
 * compute every element in the order of their train kernels. Index tensors are INT32, or INT64 narrowed by Java
 * {@code (int)} ({@link Tensor#toIntArray()}). Results never require grad.
 */
final class CpuIndexing {

    private CpuIndexing() {
    }

    // ------------------------------------------------------------------ gather and scatter

    /**
     * {@code out[c] = input[c with c[dim] = index[c]]} over the index shape.
     *
     * @throws IllegalArgumentException  when the backends or ranks differ, or an index dim other than {@code dim}
     *                                   exceeds the input's
     * @throws IndexOutOfBoundsException when {@code dim} is outside the rank, or with
     *                                   {@link TrainGpu#gatherIndexMessage} (the device path's text) when an index is
     *                                   outside {@code [0, input.dim(dim))}
     */
    static Tensor gather(Tensor input, int dim, Tensor index) {
        CpuShapeOps.requireSameBackend(input, index);
        Shape inputShape = input.shape();
        Shape indexShape = index.shape();
        int d = CpuShapeOps.checkIndexShapes(inputShape, dim, indexShape);
        int rank = inputShape.rank();

        float[] inputData = input.toFloatArray();
        int[] indexData = index.toIntArray();
        float[] result = new float[indexData.length];
        int[] inputStrides = CpuShapeOps.contiguousStrides(inputShape);
        int[] indices = new int[rank];

        for (int i = 0; i < indexData.length; i++) {
            CpuShapeOps.unravelIndex(i, indexShape, indices);
            int gatherIdx = indexData[i];
            if (gatherIdx < 0 || gatherIdx >= inputShape.dim(d)) {
                throw new IndexOutOfBoundsException(TrainGpu.gatherIndexMessage(gatherIdx, d, inputShape.dim(d)));
            }
            result[i] = inputData[linear(indices, d, gatherIdx, inputStrides)];
        }
        return CpuShapeOps.floats(input.backend(), indexShape, result, false);
    }

    /**
     * A copy of {@code input} with {@code out[c with c[dim] = index[c]] = src[c]} in row-major index order, so the
     * last write wins ({@code add == false}), or {@code += src[c]} in the same order ({@code add == true}, the
     * accumulation of {@code GradFunctions.GatherGrad}).
     *
     * @throws IllegalArgumentException  when the backends differ, the index and source shapes differ, the ranks
     *                                   differ, or an index dim other than {@code dim} exceeds the input's
     * @throws IndexOutOfBoundsException when {@code dim} is outside the rank or an index is outside
     *                                   {@code [0, input.dim(dim))}
     */
    static Tensor scatter(Tensor input, int dim, Tensor index, Tensor src, boolean add) {
        CpuShapeOps.requireSameBackend(input, index);
        CpuShapeOps.requireSameBackend(input, src);
        Shape inputShape = input.shape();
        Shape indexShape = index.shape();
        if (!indexShape.equals(src.shape())) {
            throw new IllegalArgumentException("Index and source must have the same shape");
        }
        int d = CpuShapeOps.checkIndexShapes(inputShape, dim, indexShape);
        int rank = inputShape.rank();

        float[] result = input.toFloatArray();
        int[] indexData = index.toIntArray();
        float[] srcData = src.toFloatArray();
        int[] inputStrides = CpuShapeOps.contiguousStrides(inputShape);
        int[] indices = new int[rank];

        for (int i = 0; i < indexData.length; i++) {
            CpuShapeOps.unravelIndex(i, indexShape, indices);
            int scatterIdx = indexData[i];
            if (scatterIdx < 0 || scatterIdx >= inputShape.dim(d)) {
                throw new IndexOutOfBoundsException(TrainGpu.scatterIndexMessage(scatterIdx, d, inputShape.dim(d)));
            }
            int dst = linear(indices, d, scatterIdx, inputStrides);
            if (add) {
                result[dst] += srcData[i];
            } else {
                result[dst] = srcData[i];
            }
        }
        return CpuShapeOps.floats(input.backend(), inputShape, result, false);
    }

    /** Linear index of {@code indices} with coordinate {@code dim} replaced by {@code value}. */
    private static int linear(int[] indices, int dim, int value, int[] strides) {
        int idx = 0;
        for (int j = 0; j < indices.length; j++) {
            idx += (j == dim ? value : indices[j]) * strides[j];
        }
        return idx;
    }

    // ------------------------------------------------------------------ row selection

    /**
     * {@code out[i, :] = weight[indices[i], :]} with output shape {@code indices.shape + [weight.dim(1)]} and the
     * table's dtype. With {@code clamp} each index is clamped to {@code [0, rows - 1]}
     * ({@code nn/graph/MessagePassing.java} gatherNodes); without it an index outside {@code [0, rows)} throws
     * ({@code nn/layers/Embedding.java} forward). Indices are checked by the rows they copy, as on the device, so an
     * output without elements (a table with {@code dim == 0}) checks none.
     *
     * @throws IllegalArgumentException      when the table is not 2-D or the operands live on different backends
     * @throws UnsupportedOperationException when the indices are neither INT32 nor INT64
     * @throws IndexOutOfBoundsException     for an out-of-range index without {@code clamp}, or when clamping into an
     *                                       empty table
     */
    static Tensor indexSelectRows(Tensor weight, Tensor indices, boolean clamp) {
        if (weight.shape().rank() != 2) {
            throw new IllegalArgumentException("indexSelectRows needs a 2-D table [rows, dim], got " + weight.shape());
        }
        if (indices.backend() != weight.backend()) {
            throw new IllegalArgumentException("indices live on a different backend than the table");
        }
        if (indices.dtype() != DType.INT32 && indices.dtype() != DType.INT64) {
            throw new UnsupportedOperationException("indexSelectRows needs INT32 or INT64 indices, got "
                    + indices.dtype());
        }
        int rows = weight.shape().dim(0);
        int dim = weight.shape().dim(1);
        int elementBytes = weight.dtype().byteSize();
        long total = indices.numElements() * dim;
        if (clamp && rows == 0 && total > 0) {
            throw new IndexOutOfBoundsException("cannot clamp " + indices.numElements()
                    + " indices into an empty table of shape " + weight.shape());
        }
        int[] outDims = new int[indices.shape().rank() + 1];
        System.arraycopy(indices.shape().dims(), 0, outDims, 0, indices.shape().rank());
        outDims[outDims.length - 1] = dim;
        byte[] out = new byte[CpuShapeOps.hostLength(total * elementBytes)];
        if (total == 0) {
            return CpuShapeOps.bytes(weight.backend(), Shape.of(outDims), weight.dtype(), out);
        }
        int[] idx = indices.toIntArray();
        ByteBuffer table = weight.toByteBuffer();
        int rowBytes = dim * elementBytes;
        for (int i = 0; i < idx.length; i++) {
            int row = idx[i];
            if (clamp) {
                row = Math.max(0, Math.min(row, rows - 1));
            } else if (row < 0 || row >= rows) {
                throw new IndexOutOfBoundsException(TrainGpu.embeddingIndexMessage(row));
            }
            ByteBuffer source = table.duplicate();
            source.position(row * rowBytes);
            source.get(out, i * rowBytes, rowBytes);
        }
        return CpuShapeOps.bytes(weight.backend(), Shape.of(outDims), weight.dtype(), out);
    }

    /**
     * Backward of {@link #indexSelectRows}: {@code gw[k, :] += gy[i, :]} for {@code i = 0 .. n-1} in ascending order
     * into a zero {@code [rows, dim]} FLOAT32 table, with {@code k = indices[i]} clamped into {@code [0, rows - 1]}
     * when {@code clamp}; an index outside {@code [0, rows)} and {@code k == skipRow} contribute nothing. Every element
     * of {@code gw} is a float sum from {@code +0} in ascending {@code i}, the order of {@code TR_IndexAddRows_F32}.
     *
     * @throws IllegalArgumentException      as {@link #checkIndexAddRows}
     * @throws UnsupportedOperationException when the indices are neither INT32 nor INT64
     */
    static Tensor indexAddRows(Tensor gy, Tensor indices, int rows, boolean clamp, int skipRow) {
        checkIndexAddRows(gy, indices, rows, skipRow);
        int n = gy.shape().dim(0);
        int dim = gy.shape().dim(1);
        float[] gw = new float[CpuShapeOps.hostLength((long) rows * dim)];
        if (rows > 0 && dim > 0 && n > 0) {
            float[] g = gy.toFloatArray();
            int[] idx = indices.toIntArray();
            for (int i = 0; i < n; i++) {
                int k = idx[i];
                if (clamp) {
                    k = Math.max(0, Math.min(k, rows - 1));
                }
                if (k < 0 || k >= rows || k == skipRow) {
                    continue;
                }
                for (int j = 0; j < dim; j++) {
                    gw[k * dim + j] += g[i * dim + j];
                }
            }
        }
        return CpuShapeOps.floats(gy.backend(), Shape.of(rows, dim), gw, false);
    }

    /**
     * The argument checks of {@link #indexAddRows}, shared with its device dispatcher so both paths reject the same
     * arguments: {@code gy} a FLOAT32 {@code [n, dim]} gradient, {@code indices} n INT32 or INT64 values on its
     * backend, {@code rows} non-negative and {@code skipRow} below {@code rows} (negative for none).
     *
     * @throws IllegalArgumentException      when a check fails
     * @throws UnsupportedOperationException when the indices are neither INT32 nor INT64
     */
    static void checkIndexAddRows(Tensor gy, Tensor indices, int rows, int skipRow) {
        Objects.requireNonNull(gy, "gradient must not be null");
        Objects.requireNonNull(indices, "indices must not be null");
        if (gy.shape().rank() != 2) {
            throw new IllegalArgumentException("indexAddRows needs a 2-D gradient [n, dim], got " + gy.shape());
        }
        if (gy.dtype() != DType.FLOAT32) {
            throw new IllegalArgumentException("indexAddRows needs a FLOAT32 gradient, got " + gy.dtype());
        }
        if (indices.backend() != gy.backend()) {
            throw new IllegalArgumentException("indices live on a different backend than the gradient");
        }
        if (indices.dtype() != DType.INT32 && indices.dtype() != DType.INT64) {
            throw new UnsupportedOperationException("indexAddRows needs INT32 or INT64 indices, got "
                    + indices.dtype());
        }
        if (indices.numElements() != gy.shape().dim(0)) {
            throw new IllegalArgumentException(indices.numElements() + " indices for a gradient of "
                    + gy.shape().dim(0) + " rows");
        }
        if (rows < 0) {
            throw new IllegalArgumentException("rows must be non-negative, got " + rows);
        }
        if (skipRow >= rows) {
            throw new IllegalArgumentException("skipRow " + skipRow + " must be below the " + rows
                    + " table rows (negative for none)");
        }
    }

    // ------------------------------------------------------------------ segment reductions

    /**
     * Per-segment aggregation of the rows (dim 0) of {@code src} over a CSR edge list: segment {@code t} aggregates
     * the rows {@code csrEdges[e]} for {@code e} in {@code [csrPtr[t], csrPtr[t + 1])} in ascending {@code e}. SUM
     * adds from {@code 0.0f} ({@code MessagePassing.scatterAdd}), MEAN divides that sum once by the edge count
     * ({@code scatterMean}), MAX folds {@code Math.max} from {@code -inf}. A segment without edges, and for MAX any
     * feature whose maximum is {@code -inf}, gets {@code emptyValue}. A source row may appear in several segments;
     * only {@link #segmentReduceBackward} requires each source row at most once.
     *
     * @throws IllegalArgumentException when the backends differ, {@code numNodes} is negative, {@code src} is a
     *                                  scalar, {@code csrPtr} does not hold {@code numNodes + 1} offsets, or the CSR
     *                                  arrays break the precondition (offsets from 0 or above, non-decreasing, at
     *                                  most the edge count; edges naming rows of {@code src})
     */
    static Tensor segmentReduce(Tensor src, Tensor csrPtr, Tensor csrEdges, int numNodes, GpuReduceOps.SegmentOp op,
                                float emptyValue) {
        CpuShapeOps.requireSameBackend(src, csrPtr);
        CpuShapeOps.requireSameBackend(src, csrEdges);
        if (numNodes < 0) {
            throw new IllegalArgumentException("numNodes must be non-negative, got " + numNodes);
        }
        if (src.shape().rank() == 0) {
            throw new IllegalArgumentException("segmentReduce needs rows along dim 0, got a scalar source");
        }
        if (csrPtr.numElements() != numNodes + 1L) {
            throw new IllegalArgumentException("CSR offsets must hold numNodes + 1 = " + (numNodes + 1L)
                    + " values, got " + csrPtr.numElements());
        }
        int[] outDims = src.shape().dims();
        int rows = outDims[0];
        outDims[0] = numNodes;
        Shape outShape = Shape.of(outDims);
        int feat = CpuShapeOps.product(src.shape(), 1, src.shape().rank());
        int[] ptr = csrPtr.toIntArray();
        int[] edges = csrEdges.toIntArray();
        checkCsr(ptr, edges, rows);

        float[] data = src.toFloatArray();
        float[] result = new float[CpuShapeOps.hostLength(outShape.numElements())];
        for (int t = 0; t < numNodes; t++) {
            int count = ptr[t + 1] - ptr[t];
            for (int f = 0; f < feat; f++) {
                result[t * feat + f] = segment(op, data, edges, ptr[t], ptr[t + 1], feat, f, count, emptyValue);
            }
        }
        return CpuShapeOps.floats(src.backend(), outShape, result, false);
    }

    private static float segment(GpuReduceOps.SegmentOp op, float[] data, int[] edges, int e0, int e1, int feat, int f,
                                 int count, float emptyValue) {
        if (op == GpuReduceOps.SegmentOp.MAX) {
            float m = Float.NEGATIVE_INFINITY;
            for (int e = e0; e < e1; e++) {
                m = Math.max(m, data[edges[e] * feat + f]);
            }
            return m == Float.NEGATIVE_INFINITY ? emptyValue : m;
        }
        if (count == 0) {
            return emptyValue;
        }
        float s = 0.0f;
        for (int e = e0; e < e1; e++) {
            s += data[edges[e] * feat + f];
        }
        return op == GpuReduceOps.SegmentOp.MEAN ? s / count : s;
    }

    /**
     * Backward of {@link #segmentReduce} with respect to {@code src}: a FLOAT32 tensor of {@code src}'s shape, zero
     * except at the source rows the segments name. For each segment {@code t} with edges and feature {@code f}, with
     * {@code g = gy[t, f]}: SUM writes {@code g} at every source row of the segment, MEAN writes
     * {@code g / (float) count}, MAX writes {@code g} at the first source row, in ascending edge order, whose value is
     * Java {@code ==} {@code out[t, f]} (the forward result after the empty substitution; a NaN result matches
     * nothing). Segments without edges contribute nothing. This is the per-element computation of
     * {@code TR_SegmentReduceBackward_F32}.
     *
     * <p>Precondition, validated here: each source row appears at most once in {@code csrEdges} over
     * {@code [csrPtr[0], csrPtr[numNodes])}, so every gradient element has at most one writer. The CSR producers of
     * {@code nn/graph/GraphData} build permutations of the edge and node ids and satisfy it.</p>
     *
     * @throws IllegalArgumentException as {@link #checkSegmentBackward}, when the CSR arrays break the forward's
     *                                  precondition, or when a source row appears twice in {@code csrEdges}
     */
    static Tensor segmentReduceBackward(Tensor src, Tensor csrPtr, Tensor csrEdges, Tensor out, Tensor gy,
                                        GpuReduceOps.SegmentOp op) {
        int numNodes = checkSegmentBackward(src, csrPtr, csrEdges, out, gy, op);
        int rows = src.shape().dim(0);
        int feat = CpuShapeOps.product(src.shape(), 1, src.shape().rank());
        int[] ptr = csrPtr.toIntArray();
        int[] edges = csrEdges.toIntArray();
        checkCsr(ptr, edges, rows);
        checkInjective(ptr, edges, rows);

        float[] gx = new float[CpuShapeOps.hostLength(src.numElements())];
        if (feat > 0 && numNodes > 0) {
            boolean max = op == GpuReduceOps.SegmentOp.MAX;
            float[] g = gy.toFloatArray();
            float[] values = max ? src.toFloatArray() : null;
            float[] results = max ? out.toFloatArray() : null;
            for (int t = 0; t < numNodes; t++) {
                int e0 = ptr[t];
                int e1 = ptr[t + 1];
                if (e1 <= e0) {
                    continue;
                }
                for (int f = 0; f < feat; f++) {
                    int i = t * feat + f;
                    if (max) {
                        routeToMax(gx, values, edges, e0, e1, feat, f, results[i], g[i]);
                    } else {
                        float v = op == GpuReduceOps.SegmentOp.MEAN ? g[i] / (float) (e1 - e0) : g[i];
                        for (int e = e0; e < e1; e++) {
                            gx[edges[e] * feat + f] = v;
                        }
                    }
                }
            }
        }
        return CpuShapeOps.floats(src.backend(), src.shape(), gx, false);
    }

    /** MAX routing: {@code g} to the first source row of {@code [e0, e1)} whose feature {@code f} equals {@code m}. */
    private static void routeToMax(float[] gx, float[] values, int[] edges, int e0, int e1, int feat, int f, float m,
                                   float g) {
        for (int e = e0; e < e1; e++) {
            int r = edges[e] * feat + f;
            if (values[r] == m) {
                gx[r] = g;
                return;
            }
        }
    }

    /**
     * The argument checks of {@link #segmentReduceBackward}, shared with its device dispatcher so both paths reject
     * the same arguments, and returns the segment count {@code csrPtr.numElements() - 1}: non-null operands on one
     * backend ({@code out} only for MAX), a source of rank at least 1, CSR offsets with at least one value, a FLOAT32
     * gradient of the forward result's shape ({@code src}'s shape with dim 0 replaced by the segment count) and, for
     * MAX, a FLOAT32 source and an output of that shape.
     *
     * @throws IllegalArgumentException when a check fails
     */
    static int checkSegmentBackward(Tensor src, Tensor csrPtr, Tensor csrEdges, Tensor out, Tensor gy,
                                    GpuReduceOps.SegmentOp op) {
        Objects.requireNonNull(src, "source must not be null");
        Objects.requireNonNull(csrPtr, "CSR offsets must not be null");
        Objects.requireNonNull(csrEdges, "CSR edges must not be null");
        Objects.requireNonNull(gy, "gradient must not be null");
        Objects.requireNonNull(op, "segment op must not be null");
        boolean max = op == GpuReduceOps.SegmentOp.MAX;
        if (max) {
            Objects.requireNonNull(out, "MAX routing needs the forward output");
        }
        CpuShapeOps.requireSameBackend(src, csrPtr);
        CpuShapeOps.requireSameBackend(src, csrEdges);
        CpuShapeOps.requireSameBackend(src, gy);
        if (src.shape().rank() == 0) {
            throw new IllegalArgumentException("segmentReduceBackward needs rows along dim 0, got a scalar source");
        }
        if (csrPtr.numElements() < 1 || csrPtr.numElements() - 1 > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("CSR offsets must hold numNodes + 1 values (1 to 2^31), got "
                    + csrPtr.numElements());
        }
        int numNodes = (int) (csrPtr.numElements() - 1);
        int[] outDims = src.shape().dims();
        outDims[0] = numNodes;
        Shape outShape = Shape.of(outDims);
        if (!gy.shape().equals(outShape)) {
            throw new IllegalArgumentException("gradient " + gy.shape() + " does not match the segment result shape "
                    + outShape);
        }
        if (gy.dtype() != DType.FLOAT32) {
            throw new IllegalArgumentException("segmentReduceBackward needs a FLOAT32 gradient, got " + gy.dtype());
        }
        if (max) {
            CpuShapeOps.requireSameBackend(src, out);
            if (!out.shape().equals(outShape)) {
                throw new IllegalArgumentException("forward output " + out.shape()
                        + " does not match the segment result shape " + outShape);
            }
            if (src.dtype() != DType.FLOAT32 || out.dtype() != DType.FLOAT32) {
                throw new IllegalArgumentException("MAX routing needs a FLOAT32 source and output, got "
                        + src.dtype() + " and " + out.dtype());
            }
        }
        return numNodes;
    }

    /**
     * Throws when a source row appears more than once among the edges {@code [ptr[0], ptr[last])} (the slots the
     * segments own), the precondition of {@link #segmentReduceBackward}.
     */
    private static void checkInjective(int[] ptr, int[] edges, int rows) {
        boolean[] seen = new boolean[rows];
        for (int e = ptr[0]; e < ptr[ptr.length - 1]; e++) {
            int r = edges[e];
            if (seen[r]) {
                throw new IllegalArgumentException("CSR edge " + e + " names source row " + r + " a second time; "
                        + "segmentReduceBackward needs each source row at most once in the CSR edges (GraphData "
                        + "CSRs satisfy this)");
            }
            seen[r] = true;
        }
    }

    private static void checkCsr(int[] ptr, int[] edges, int rows) {
        if (ptr[0] < 0) {
            throw new IllegalArgumentException("CSR offsets must start at 0 or above, got " + ptr[0]);
        }
        for (int t = 1; t < ptr.length; t++) {
            if (ptr[t] < ptr[t - 1]) {
                throw new IllegalArgumentException("CSR offsets must not decrease: offset " + t + " is " + ptr[t]
                        + " after " + ptr[t - 1]);
            }
        }
        if (ptr[ptr.length - 1] > edges.length) {
            throw new IllegalArgumentException("CSR offsets end at " + ptr[ptr.length - 1] + ", beyond the "
                    + edges.length + " edges");
        }
        for (int e = ptr[0]; e < ptr[ptr.length - 1]; e++) {
            if (edges[e] < 0 || edges[e] >= rows) {
                throw new IllegalArgumentException("CSR edge " + e + " names row " + edges[e] + " of a source with "
                        + rows + " rows");
            }
        }
    }
}

package com.eonmux.neuralfree.tensor;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.tensor.gpu.StridedView;
import com.eonmux.neuralfree.tensor.gpu.TrainGpu;
import com.eonmux.neuralfree.tensor.gpu.TrainOps;
import com.eonmux.neuralfree.tensor.gpu.TrainOps.Kernels;

import java.util.Arrays;
import java.util.Objects;
import java.util.function.Consumer;

/**
 * Reductions, arg-reductions, softmax and segment reductions on the train kernels ({@code reduce.cl},
 * {@code softmax.cl}, {@code segment.cl}). Results are new contiguous tensors that stay on the device; nothing is
 * read back.
 *
 * <p><b>Dispatch.</b> Every method returns {@code null} exactly when the train library is unavailable on the
 * tensor's backend ({@link TrainGpu#forBackend}) or a dtype is not the documented one (FLOAT32 data, INT32 CSR
 * arrays); that is the caller's CPU path. Once the library is available, failures propagate as exceptions. Output
 * shapes follow {@link TensorOps}: reduced dims are removed (no keepdim); reducing every dim gives a scalar. Results
 * never require grad, whatever the inputs' flags; only the recording facades mark a result as requiring grad.</p>
 *
 * <p><b>Layout.</b> The kernels reduce the last dim of a dense {@code [rows, cols]} block. A tensor that already is
 * one is read in place; otherwise its reduced dims are moved to the end, in ascending order, by one strided copy into
 * {@link TrainGpu#scratch}, which keeps the CPU's ascending accumulation order per output. A SUM over a leading run
 * of dims that keeps a dim uses the column kernel (ascending row order per column). Softmax and the backward ops
 * write a block that is copied back when the dim was moved. Variants and chunks depend only on the shape:
 * rows of {@code cols <= 64} use one work-item per row, longer rows one 256-item group; with
 * {@code rows < 64 && cols > 65536} each row is split into {@code min(1024, ceil(cols/8192))} chunks and a second
 * pass reduces the partials (MAX and MIN keep their op, sums use SUM, the divisor applies on that pass only); column
 * sums over {@code rows > 4096} use {@code min(256, ceil(rows/1024))} row chunks and a second column pass.</p>
 *
 * <p><b>Tolerance classes</b> ({@linkplain com.eonmux.neuralfree.tensor.gpu numerical contract}) against the CPU path:
 * bit-identical for {@link #max}, {@link #min}, {@link #argmax}, {@link #argmin}, {@link #maxBackward},
 * {@link SegmentOp#MAX} and an unsummed {@link #sumToShape}; reductions (double-precision reference) for {@link #sum},
 * {@link #mean} (then one division), {@link #var}, {@link #sumToShape} and {@link #sumColumns}; single-rounding for
 * {@link SegmentOp#SUM} (the CPU's ascending edge order) and division for {@link SegmentOp#MEAN}; transcendentals for
 * {@link #softmax}, {@link #logSoftmax}, {@link #maskedSoftmax} and {@link #softmaxBackward}.</p>
 */
public final class GpuReduceOps {

    /** Aggregation of {@link #segmentReduce}, with the {@code TR_SEG_*} op id the kernel selects it by. */
    public enum SegmentOp {
        /** Sum of the source rows in ascending edge order ({@code MessagePassing.scatterAdd}). */
        SUM(TrainOps.TR_SEG_SUM),
        /** The same sum, then one division by the edge count ({@code MessagePassing.scatterMean}). */
        MEAN(TrainOps.TR_SEG_MEAN),
        /** Java {@code Math.max} fold from {@code -inf}; a {@code -inf} result becomes the empty value. */
        MAX(TrainOps.TR_SEG_MAX);

        private final int id;

        SegmentOp(int id) {
            this.id = id;
        }

        /** The kernel's op id. */
        public int id() {
            return id;
        }
    }

    /** Longest row reduced by one work-item ({@code TR_ReduceRowsSmall_F32}). */
    private static final int SMALL_ROW_MAX_COLS = 64;
    /** Row chunking: below PARTIAL_ROWS_BELOW rows of more than PARTIAL_COLS_ABOVE columns. */
    private static final int PARTIAL_ROWS_BELOW = 64, PARTIAL_COLS_ABOVE = 65536;
    private static final int PARTIAL_MAX_CHUNKS = 1024, PARTIAL_CHUNK_COLS = 8192;
    /** Column chunking: above COL_CHUNK_ROWS_ABOVE rows. */
    private static final int COL_CHUNK_ROWS_ABOVE = 4096, COL_MAX_CHUNKS = 256, COL_CHUNK_ROWS = 1024;

    /** Scratch slots, each used once per {@link TrainGpu#exclusive} sequence. */
    private static final int SLOT_IN = 0, SLOT_IN2 = 1, SLOT_IN3 = 2, SLOT_MEAN = 3, SLOT_PARTIAL = 4, SLOT_OUT = 5;

    private GpuReduceOps() {
    }

    /**
     * Sum over {@code dims} ({@code TensorOps.sum}); no dims sums every element into a scalar.
     *
     * @throws IndexOutOfBoundsException when a dim is outside the rank
     * @throws IllegalArgumentException  when a dim repeats
     */
    public static Tensor sum(Tensor a, int... dims) {
        return sumOrMean(a, dims, false);
    }

    /**
     * Mean over {@code dims} ({@code TensorOps.mean}): the sum, then one division by the reduction length (an empty
     * reduction gives NaN, as {@code 0f/0}).
     *
     * @throws IndexOutOfBoundsException when a dim is outside the rank
     * @throws IllegalArgumentException  when a dim repeats
     */
    public static Tensor mean(Tensor a, int... dims) {
        return sumOrMean(a, dims, true);
    }

    /** Java {@code Math.max} along {@code dim} from {@code -inf} ({@code TensorOps.max}): NaN wins, +0 above -0. */
    public static Tensor max(Tensor a, int dim) {
        return extreme(a, dim, TrainOps.TR_R_MAX);
    }

    /** Java {@code Math.min} along {@code dim} from {@code +inf} ({@code TensorOps.min}): NaN wins, -0 below +0. */
    public static Tensor min(Tensor a, int dim) {
        return extreme(a, dim, TrainOps.TR_R_MIN);
    }

    /**
     * INT32 index of the maximum along {@code dim} with the CPU's rule ({@code TensorOps.argmax}): the first index
     * of a value strictly above every earlier one, starting from {@code -inf}; NaN is never selected and a row with
     * no such value gives 0.
     */
    public static Tensor argmax(Tensor a, int dim) {
        return argReduce(a, dim, TrainOps.TR_R_ARGMAX);
    }

    /** INT32 index of the minimum along {@code dim}; {@link #argmax} with the comparison reversed. */
    public static Tensor argmin(Tensor a, int dim) {
        return argReduce(a, dim, TrainOps.TR_R_ARGMIN);
    }

    /**
     * Population variance over {@code dims} (no Bessel correction, as {@code TensorOps.var}): the mean of the squared
     * deviations from the mean over the same dims, taken per output element for any choice of dims; no dims reduces
     * every element into a scalar.
     *
     * @throws IndexOutOfBoundsException when a dim is outside the rank
     * @throws IllegalArgumentException  when a dim repeats
     */
    public static Tensor var(Tensor a, int... dims) {
        Objects.requireNonNull(dims, "dims must not be null");
        TrainGpu g = floatGpu(a);
        if (g == null) {
            return null;
        }
        Layout l = new Layout(a.shape(), reducedFlags(a.shape(), dims));
        float divisor = (float) l.cols;
        return produce(g, a, reducedShape(a.shape(), dims), DType.FLOAT32, out -> {
            if (l.rows > 0) {
                Block in = rowsBlock(g, a, l.perm, SLOT_IN);
                GpuBuffer mean = g.scratch(SLOT_MEAN, (long) l.rows * Float.BYTES);
                reduceRows(g, TrainOps.TR_R_SUM, in, none(g), l.rows, l.cols, divisor, mean, 0);
                reduceRows(g, TrainOps.TR_R_SQDEV, in, new Block(mean, 0), l.rows, l.cols, divisor, buffer(out), 0);
            }
        });
    }

    /**
     * Column sums of a 2-D tensor, shape {@code [cols]}: {@link #sum sum(m, 0)}, which sums each column in ascending
     * row order (above 4096 rows within each row chunk, then over the chunk sums in chunk order).
     *
     * @throws IllegalArgumentException when {@code m} is not 2-D (whether or not the library is available)
     */
    public static Tensor sumColumns(Tensor m) {
        if (Objects.requireNonNull(m, "tensor must not be null").shape().rank() != 2) {
            throw new IllegalArgumentException("sumColumns needs a 2-D tensor, got shape " + m.shape());
        }
        return sum(m, 0);
    }

    /**
     * {@code t} summed down to {@code target}, the shape of an operand that was broadcast to {@code t}'s shape, as
     * the CPU's {@code CpuReductions.sumToShape}: the prepended dims and the dims where {@code target} has 1
     * and {@code t} more are summed. When the shapes are equal, or {@code target} only rearranges {@code t}'s
     * elements, the result is a contiguous copy of {@code t} with shape {@code target}. Always a new tensor.
     *
     * @throws IllegalArgumentException when the shapes are not broadcast-compatible, or {@code target} neither
     *                                  broadcasts to {@code t}'s shape nor has its element count
     */
    public static Tensor sumToShape(Tensor t, Shape target) {
        Objects.requireNonNull(target, "target shape must not be null");
        TrainGpu g = floatGpu(t);
        if (g == null) {
            return null;
        }
        Shape shape = t.shape();
        if (!shape.equals(target)) {
            boolean[] reduced = BroadcastEngine.broadcastShape(target, shape).equals(shape)
                    ? broadcastAxes(target, shape) : new boolean[shape.rank()];
            for (boolean r : reduced) {
                if (r) {
                    return reduce(g, t, reduced, TrainOps.TR_R_SUM, false, target);
                }
            }
            shape.reshape(target.dims());
        }
        return produce(g, t, target, DType.FLOAT32, out -> {
            if (out.numElements() > 0) {
                copyDense(g, StridedView.of(t), buffer(out), true);
            }
        });
    }

    /**
     * Gradient of {@link #max} ({@code GradFunctions.MaxGrad}): zeros of {@code x}'s shape, except {@code gy[row]}
     * at the first index along {@code dim} whose value {@code ==} {@code maxOut[row]} (a NaN max routes nowhere).
     * {@code maxOut} and {@code gy} hold one value per reduced row, in row-major order of the kept dims (any shape
     * with that element count, such as the {@link #max} result).
     *
     * @throws IllegalArgumentException when the backends differ or {@code maxOut} or {@code gy} has the wrong
     *                                  element count
     */
    public static Tensor maxBackward(Tensor x, Tensor maxOut, Tensor gy, int dim) {
        Objects.requireNonNull(x, "input must not be null");
        Objects.requireNonNull(maxOut, "max output must not be null");
        Objects.requireNonNull(gy, "gradient must not be null");
        TrainGpu g = maxOut.dtype() == DType.FLOAT32 && gy.dtype() == DType.FLOAT32 ? floatGpu(x) : null;
        if (g == null) {
            return null;
        }
        requireSameBackend(x, maxOut, gy);
        Layout l = new Layout(x.shape(), reducedFlags(x.shape(), new int[]{dim}));
        requireCount("max output", maxOut, l.rows);
        requireCount("gradient", gy, l.rows);
        return produce(g, x, x.shape(), DType.FLOAT32, out -> writeRows(g, out, l, (ob, oo) -> {
            Block xb = rowsBlock(g, x, l.perm, SLOT_IN);
            Block mb = denseBlock(g, maxOut, SLOT_IN2);
            Block gb = denseBlock(g, gy, SLOT_IN3);
            g.launchRows(Kernels.MAX_BACKWARD_ROWS_F32, l.rows, l.rows, l.cols, xb.offset, mb.offset, gb.offset, oo,
                    xb.buffer, mb.buffer, gb.buffer, ob);
        }));
    }

    /** Softmax along {@code dim} ({@code TensorOps.softmax}): max-subtract, exp, sum, divide. */
    public static Tensor softmax(Tensor a, int dim) {
        return softmaxRows(a, dim, 0);
    }

    /** Log-softmax along {@code dim} ({@code TensorOps.logSoftmax}): {@code x - (max + log(sum))}. */
    public static Tensor logSoftmax(Tensor a, int dim) {
        return softmaxRows(a, dim, TrainOps.TR_SM_LOG);
    }

    /**
     * Softmax along {@code dim} where {@code -inf} entries are masked out ({@code MultiHeadAttention}'s attention
     * softmax): they are left out of the sum and get exactly 0, and a row of only {@code -inf} gives all 0.
     */
    public static Tensor maskedSoftmax(Tensor a, int dim) {
        return softmaxRows(a, dim, TrainOps.TR_SM_MASKED);
    }

    /**
     * Gradient of {@link #softmax} ({@code log == false}, {@code GradFunctions.SoftmaxGrad}:
     * {@code gx = y * (gy - sum(y * gy))}) or of {@link #logSoftmax} ({@code log == true}, {@code LogSoftmaxGrad}:
     * {@code gx = gy - exp(y) * sum(gy)}) along {@code dim}, from the forward output {@code y} and its gradient.
     *
     * @throws IllegalArgumentException when the backends or shapes differ
     */
    public static Tensor softmaxBackward(Tensor y, Tensor gy, int dim, boolean log) {
        Objects.requireNonNull(y, "softmax output must not be null");
        Objects.requireNonNull(gy, "gradient must not be null");
        TrainGpu g = gy.dtype() == DType.FLOAT32 ? floatGpu(y) : null;
        if (g == null) {
            return null;
        }
        requireSameBackend(y, gy);
        if (!y.shape().equals(gy.shape())) {
            throw new IllegalArgumentException("softmax output " + y.shape() + " and gradient " + gy.shape()
                    + " must have the same shape");
        }
        Layout l = new Layout(y.shape(), reducedFlags(y.shape(), new int[]{dim}));
        return produce(g, y, y.shape(), DType.FLOAT32, out -> writeRows(g, out, l, (ob, oo) -> {
            Block yb = rowsBlock(g, y, l.perm, SLOT_IN);
            Block gb = rowsBlock(g, gy, l.perm, SLOT_IN2);
            g.launchRows(Kernels.SOFTMAX_ROWS_BACKWARD_F32, l.rows, l.rows, l.cols, log ? 1 : 0, yb.offset, gb.offset,
                    oo, yb.buffer, gb.buffer, ob);
        }));
    }

    /**
     * Per-segment aggregation of source rows over a CSR edge list (graph message aggregation). {@code src} is
     * FLOAT32 with rows along dim 0; segment {@code t < numNodes} aggregates the rows {@code csrEdges[e]} for
     * {@code e} in {@code [csrPtr[t], csrPtr[t + 1])}, in ascending {@code e}. The result has {@code src}'s shape
     * with dim 0 replaced by {@code numNodes}. A segment without edges, and for {@link SegmentOp#MAX} any feature
     * whose maximum is {@code -inf}, gets {@code emptyValue}.
     *
     * <p>{@code csrPtr} (INT32, {@code numNodes + 1} values) and {@code csrEdges} (INT32) are built once per graph on
     * the host; the caller guarantees {@code 0 <= csrPtr[0]}, non-decreasing offsets ending at most
     * at the edge count, and edges that name rows of {@code src}. The kernel does not re-check them.</p>
     *
     * @throws IllegalArgumentException when the backends differ, {@code numNodes} is negative, {@code src} is a
     *                                  scalar, {@code csrPtr} does not hold {@code numNodes + 1} values, or the
     *                                  result exceeds {@code 2^31-1} elements
     */
    public static Tensor segmentReduce(Tensor src, Tensor csrPtr, Tensor csrEdges, int numNodes, SegmentOp op,
                                       float emptyValue) {
        Objects.requireNonNull(src, "source must not be null");
        Objects.requireNonNull(csrPtr, "CSR offsets must not be null");
        Objects.requireNonNull(csrEdges, "CSR edges must not be null");
        Objects.requireNonNull(op, "segment op must not be null");
        TrainGpu g = csrPtr.dtype() == DType.INT32 && csrEdges.dtype() == DType.INT32 ? floatGpu(src) : null;
        if (g == null) {
            return null;
        }
        requireSameBackend(src, csrPtr, csrEdges);
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
        outDims[0] = numNodes;
        Shape outShape = Shape.of(outDims);
        int items = requireInt(outShape.numElements(), "segmentReduce result elements");
        int feat = numNodes == 0 ? 0 : items / numNodes;
        return produce(g, src, outShape, DType.FLOAT32, out -> {
            if (items > 0) {
                Block s = denseBlock(g, src, SLOT_IN);
                Block p = denseBlock(g, csrPtr, SLOT_IN2);
                Block e = denseBlock(g, csrEdges, SLOT_IN3);
                g.launch1d(Kernels.SEGMENT_REDUCE_F32, items, op.id(), numNodes, feat, emptyValue, p.offset, e.offset,
                        s.offset, 0, p.buffer, e.buffer, s.buffer, buffer(out));
            }
        });
    }

    private static Tensor sumOrMean(Tensor a, int[] dims, boolean mean) {
        Objects.requireNonNull(dims, "dims must not be null");
        TrainGpu g = floatGpu(a);
        if (g == null) {
            return null;
        }
        return reduce(g, a, reducedFlags(a.shape(), dims), TrainOps.TR_R_SUM, mean, reducedShape(a.shape(), dims));
    }

    private static Tensor extreme(Tensor a, int dim, int op) {
        TrainGpu g = floatGpu(a);
        if (g == null) {
            return null;
        }
        int[] dims = {dim};
        return reduce(g, a, reducedFlags(a.shape(), dims), op, false, reducedShape(a.shape(), dims));
    }

    private static Tensor argReduce(Tensor a, int dim, int op) {
        TrainGpu g = floatGpu(a);
        if (g == null) {
            return null;
        }
        int[] dims = {dim};
        Layout l = new Layout(a.shape(), reducedFlags(a.shape(), dims));
        return produce(g, a, reducedShape(a.shape(), dims), DType.INT32, out -> {
            if (l.rows > 0) {
                Block in = rowsBlock(g, a, l.perm, SLOT_IN);
                g.launchRows(Kernels.ARG_REDUCE_ROWS_F32, l.rows, op, l.rows, l.cols, in.offset, 0, in.buffer,
                        buffer(out));
            }
        });
    }

    private static Tensor softmaxRows(Tensor a, int dim, int flags) {
        TrainGpu g = floatGpu(a);
        if (g == null) {
            return null;
        }
        Layout l = new Layout(a.shape(), reducedFlags(a.shape(), new int[]{dim}));
        return produce(g, a, a.shape(), DType.FLOAT32, out -> writeRows(g, out, l, (ob, oo) -> {
            Block in = rowsBlock(g, a, l.perm, SLOT_IN);
            g.launchRows(Kernels.SOFTMAX_ROWS_F32, l.rows, l.rows, l.cols, flags, in.offset, oo, in.buffer, ob);
        }));
    }

    /**
     * Reduces the flagged dims of {@code a} with a {@code TR_R_*} op into a new tensor of {@code outShape}, whose
     * row-major order is that of the kept dims; {@code divide} divides by the reduction length.
     */
    private static Tensor reduce(TrainGpu g, Tensor a, boolean[] reduced, int op, boolean divide, Shape outShape) {
        Layout l = new Layout(a.shape(), reduced);
        boolean columns = op == TrainOps.TR_R_SUM && !divide && l.leadingPrefix;
        float divisor = divide ? (float) l.cols : 1.0f;
        return produce(g, a, outShape, DType.FLOAT32, out -> {
            if (columns) {
                reduceCols(g, denseBlock(g, a, SLOT_IN), l.cols, l.rows, buffer(out));
            } else {
                reduceRows(g, op, rowsBlock(g, a, l.perm, SLOT_IN), none(g), l.rows, l.cols, divisor, buffer(out), 0);
            }
        });
    }

    /** Row reductions of a {@code [rows, cols]} block into {@code out[outOff .. outOff + rows)}. */
    private static void reduceRows(TrainGpu g, int op, Block in, Block mean, int rows, int cols, float divisor,
                                   GpuBuffer out, int outOff) {
        if (rows == 0) {
            return;
        }
        int chunks = rows < PARTIAL_ROWS_BELOW && cols > PARTIAL_COLS_ABOVE
                ? Math.min(PARTIAL_MAX_CHUNKS, ceilDiv(cols, PARTIAL_CHUNK_COLS)) : 1;
        if (chunks == 1) {
            launchRowReduce(g, op, in, mean, rows, cols, divisor, out, outOff);
            return;
        }
        GpuBuffer partial = g.scratch(SLOT_PARTIAL, (long) rows * chunks * Float.BYTES);
        g.launchRows(Kernels.REDUCE_ROWS_PARTIAL_F32, (long) rows * chunks, op, rows, cols, chunks, in.offset,
                mean.offset, in.buffer, mean.buffer, partial);
        int second = op == TrainOps.TR_R_MAX || op == TrainOps.TR_R_MIN ? op : TrainOps.TR_R_SUM;
        launchRowReduce(g, second, new Block(partial, 0), none(g), rows, chunks, divisor, out, outOff);
    }

    private static void launchRowReduce(TrainGpu g, int op, Block in, Block mean, int rows, int cols, float divisor,
                                        GpuBuffer out, int outOff) {
        Object[] args = {op, rows, cols, in.offset, mean.offset, outOff, divisor, in.buffer, mean.buffer, out};
        if (cols <= SMALL_ROW_MAX_COLS) {
            g.launch1d(Kernels.REDUCE_ROWS_SMALL_F32, rows, args);
        } else {
            g.launchRows(Kernels.REDUCE_ROWS_F32, rows, args);
        }
    }

    /** Column sums of a {@code [rows, cols]} block into {@code out[0 .. cols)}. */
    private static void reduceCols(TrainGpu g, Block in, int rows, int cols, GpuBuffer out) {
        if (cols == 0) {
            return;
        }
        int chunks = rows > COL_CHUNK_ROWS_ABOVE ? Math.min(COL_MAX_CHUNKS, ceilDiv(rows, COL_CHUNK_ROWS)) : 1;
        if (chunks == 1) {
            g.launch1d(Kernels.REDUCE_COLS_F32, cols, rows, cols, 1, in.offset, 0, in.buffer, out);
            return;
        }
        GpuBuffer partial = g.scratch(SLOT_PARTIAL, (long) chunks * cols * Float.BYTES);
        g.launch1d(Kernels.REDUCE_COLS_F32, (long) chunks * cols, rows, cols, chunks, in.offset, 0, in.buffer,
                partial);
        g.launch1d(Kernels.REDUCE_COLS_F32, cols, chunks, cols, 1, 0, 0, partial, out);
    }

    /**
     * Fills {@code out} (shaped like the input of layout {@code l}) through a kernel that writes the dense
     * {@code [rows, cols]} block of {@code l}: in place when that block is {@code out}'s own layout, else into
     * scratch and copied back.
     */
    private static void writeRows(TrainGpu g, Tensor out, Layout l, BlockWriter writer) {
        if (out.numElements() == 0) {
            return;
        }
        if (isIdentity(l.perm)) {
            writer.write(buffer(out), 0);
            return;
        }
        try (Tensor moved = out.permute(l.perm)) {
            StridedView view = StridedView.of(moved);
            if (view.contiguous()) {
                writer.write(view.buffer(), view.offset());
                return;
            }
            GpuBuffer block = g.scratch(SLOT_OUT, out.numElements() * Float.BYTES);
            writer.write(block, 0);
            copyDense(g, view, block, false);
        }
    }

    /**
     * {@code a} with its dims in {@code perm} order as a dense block: in place when already dense, else copied. The
     * permuted view of the operand is an internal temporary taken under {@link ComputationGraph#noGrad()}, so it never
     * records, whatever the caller's grad mode.
     */
    private static Block rowsBlock(TrainGpu g, Tensor a, int[] perm, int slot) {
        if (isIdentity(perm) || a.numElements() == 0) {
            return denseBlock(g, a, slot);
        }
        try (ComputationGraph.NoGradScope untracked = ComputationGraph.noGrad();
             Tensor moved = a.permute(perm)) {
            return denseBlock(g, StridedView.of(moved), slot);
        }
    }

    /** {@code t}'s elements in row-major order as a dense block; the placeholder when it has none. */
    private static Block denseBlock(TrainGpu g, Tensor t, int slot) {
        return t.numElements() == 0 ? none(g) : denseBlock(g, StridedView.of(t), slot);
    }

    private static Block denseBlock(TrainGpu g, StridedView v, int slot) {
        if (v.contiguous()) {
            return new Block(v.buffer(), v.offset());
        }
        GpuBuffer block = g.scratch(slot, (long) v.numElements() * Float.BYTES);
        copyDense(g, v, block, true);
        return new Block(block, 0);
    }

    /**
     * Copies between the 32-bit elements {@code v} addresses, in row-major order, and a dense block at element 0 of
     * {@code dense}: into the block when {@code toDense}, else from it into {@code v} (whose elements are then
     * distinct). TR_Copy_F32 when {@code v} coalesces to a dense run, else TR_Copy_Strided_F32 once per index of the
     * dims beyond {@link StridedView#MAX_RANK}.
     */
    private static void copyDense(TrainGpu g, StridedView v, GpuBuffer dense, boolean toDense) {
        StridedView c = StridedView.coalesce(v)[0];
        int n = c.numElements();
        if (n == 0) {
            return;
        }
        if (c.contiguous()) {
            g.launch1d(Kernels.COPY_F32, n, n, toDense ? c.offset() : 0, toDense ? 0 : c.offset(),
                    toDense ? c.buffer() : dense, toDense ? dense : c.buffer());
            return;
        }
        StridedView[] parts = c.splitLeading();
        int inner = parts[0].numElements();
        Object[] denseStrides = denseStrideArgs(parts[0]);
        for (int i = 0; i < parts.length; i++) {
            StridedView p = parts[i];
            int denseOff = i * inner;
            Object[] args = new Object[24];
            args[0] = inner;
            args[1] = p.rank();
            System.arraycopy(p.dimArgs(), 0, args, 2, StridedView.MAX_RANK);
            System.arraycopy(toDense ? p.strideArgs() : denseStrides, 0, args, 8, StridedView.MAX_RANK);
            System.arraycopy(toDense ? denseStrides : p.strideArgs(), 0, args, 14, StridedView.MAX_RANK);
            args[20] = toDense ? p.offset() : denseOff;
            args[21] = toDense ? denseOff : p.offset();
            args[22] = toDense ? p.buffer() : dense;
            args[23] = toDense ? dense : p.buffer();
            g.launch1d(Kernels.COPY_STRIDED_F32, inner, args);
        }
    }

    /** Row-major strides of {@code v}'s dims as the {@link StridedView#MAX_RANK} stride arguments (0 past the rank). */
    private static Object[] denseStrideArgs(StridedView v) {
        Object[] out = new Object[StridedView.MAX_RANK];
        int stride = 1;
        for (int k = StridedView.MAX_RANK - 1; k >= 0; k--) {
            out[k] = k < v.rank() ? stride : 0;
            stride *= k < v.rank() ? v.dim(k) : 1;
        }
        return out;
    }

    /**
     * Allocates a contiguous result that does not require grad on {@code like}'s backend and fills it under
     * {@link TrainGpu#exclusive} (the fill may use scratch); the result is closed when the fill fails.
     */
    private static Tensor produce(TrainGpu g, Tensor like, Shape shape, DType dtype, Consumer<Tensor> fill) {
        return g.exclusive(() -> {
            Tensor out = Tensor.wrap(shape, dtype, Strides.contiguous(shape), g.allocate(shape.numElements(), dtype),
                    like.backend(), false);
            try {
                fill.accept(out);
                return out;
            } catch (RuntimeException e) {
                out.close();
                throw e;
            }
        });
    }

    /** The context for a FLOAT32 tensor, or null for another dtype or when the library is unavailable. */
    private static TrainGpu floatGpu(Tensor a) {
        Objects.requireNonNull(a, "tensor must not be null");
        return a.dtype() == DType.FLOAT32 ? TrainGpu.forBackend(a.backend()) : null;
    }

    private static GpuBuffer buffer(Tensor fresh) {
        return fresh.storage().gpuBuffer();
    }

    /** The placeholder as an unused input block. */
    private static Block none(TrainGpu g) {
        return new Block(g.placeholder(), 0);
    }

    /**
     * Flags of the reduced dims: every dim for an empty {@code dims}, else the normalized dims.
     *
     * @throws IndexOutOfBoundsException when a dim is outside the rank (as {@link BroadcastEngine#normalizeAxes})
     * @throws IllegalArgumentException  when a dim repeats
     */
    private static boolean[] reducedFlags(Shape shape, int[] dims) {
        boolean[] reduced = new boolean[shape.rank()];
        if (dims.length == 0) {
            Arrays.fill(reduced, true);
            return reduced;
        }
        for (int d : BroadcastEngine.normalizeAxes(dims, shape.rank())) {
            if (reduced[d]) {
                throw new IllegalArgumentException("dim " + d + " is reduced twice in " + Arrays.toString(dims));
            }
            reduced[d] = true;
        }
        return reduced;
    }

    /** The CPU's result shape: a scalar for no dims, else {@code reduceShape} without keepdim. */
    private static Shape reducedShape(Shape shape, int[] dims) {
        return dims.length == 0 ? Shape.scalar() : BroadcastEngine.reduceShape(shape, dims, false);
    }

    /**
     * Flags of the dims of {@code larger} that a broadcast of {@code smaller} expanded: the prepended dims and the
     * dims where {@code smaller} has 1 and {@code larger} more (the CPU's {@code CpuReductions.broadcastAxes}).
     */
    private static boolean[] broadcastAxes(Shape smaller, Shape larger) {
        int rankDiff = larger.rank() - smaller.rank();
        boolean[] reduced = new boolean[larger.rank()];
        for (int k = 0; k < larger.rank(); k++) {
            reduced[k] = k < rankDiff || (smaller.dim(k - rankDiff) == 1 && larger.dim(k) > 1);
        }
        return reduced;
    }

    private static boolean isIdentity(int[] perm) {
        for (int i = 0; i < perm.length; i++) {
            if (perm[i] != i) {
                return false;
            }
        }
        return true;
    }

    private static void requireSameBackend(Tensor first, Tensor... others) {
        for (Tensor t : others) {
            if (t.backend() != first.backend()) {
                throw new IllegalArgumentException("tensors live on different backends: "
                        + first.backend().backendType() + " and " + t.backend().backendType());
            }
        }
    }

    private static void requireCount(String what, Tensor t, int expected) {
        if (t.numElements() != expected) {
            throw new IllegalArgumentException(what + " must hold one value per reduced row (" + expected
                    + "), got shape " + t.shape());
        }
    }

    private static int requireInt(long n, String what) {
        if (n > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(what + " = " + n + " is above the 32-bit kernel limit "
                    + Integer.MAX_VALUE);
        }
        return (int) n;
    }

    private static int ceilDiv(int a, int b) {
        return a / b + (a % b != 0 ? 1 : 0);
    }

    /** A dense row-major run of 32-bit elements on the device: a buffer and the element offset of its first element. */
    private static final class Block {
        final GpuBuffer buffer;
        final int offset;

        Block(GpuBuffer buffer, int offset) {
            this.buffer = buffer;
            this.offset = offset;
        }
    }

    /** Enqueues a kernel that writes a dense {@code [rows, cols]} block at element {@code offset} of {@code out}. */
    @FunctionalInterface
    private interface BlockWriter {
        void write(GpuBuffer out, int offset);
    }

    /**
     * A shape split into kept dims ({@code rows}, their product) and reduced dims ({@code cols}, their product);
     * {@code perm} lists the kept dims, then the reduced dims, each ascending. {@code leadingPrefix}: the reduced dims
     * are dims {@code 0..k-1} for some {@code 1 <= k < rank}.
     */
    private static final class Layout {
        final int[] perm;
        final int rows;
        final int cols;
        final boolean leadingPrefix;

        Layout(Shape shape, boolean[] reduced) {
            int rank = shape.rank();
            perm = new int[rank];
            int p = 0;
            int prefix = 0;
            while (prefix < rank && reduced[prefix]) {
                prefix++;
            }
            boolean leading = prefix > 0 && prefix < rank;
            long kept = 1;
            long length = 1;
            for (int k = 0; k < rank; k++) {
                if (!reduced[k]) {
                    perm[p++] = k;
                    kept *= shape.dim(k);
                }
            }
            for (int k = 0; k < rank; k++) {
                if (reduced[k]) {
                    perm[p++] = k;
                    length *= shape.dim(k);
                    leading &= k < prefix;
                }
            }
            rows = requireInt(kept, "kept elements");
            cols = requireInt(length, "reduction length");
            leadingPrefix = leading;
        }
    }
}

package com.eonmux.neuralfree.tensor;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.tensor.gpu.StridedView;
import com.eonmux.neuralfree.tensor.gpu.TrainGpu;
import com.eonmux.neuralfree.tensor.gpu.TrainOps;
import com.eonmux.neuralfree.tensor.gpu.TrainOps.Kernels;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.function.Consumer;

/**
 * Layer, RMS, batch and group normalization and row index-select on the train kernels ({@code kernels/train/norm.cl},
 * the row reductions of {@code reduce.cl}, {@code TR_Axpby_F32} and {@code TR_Cast_I64_I32}). Results are new
 * contiguous tensors on the input's backend; nothing is read back to the host.
 *
 * <p><b>Dispatcher contract.</b> Every method returns {@code null} exactly when the train library is unavailable on
 * the input's backend ({@link TrainGpu#forBackend}) or an operand's dtype is not the documented one; that is the
 * caller's CPU path, and nothing has been enqueued or modified then. Invalid arguments throw
 * {@link IllegalArgumentException} whether or not the library is available. Once the library is available, kernel
 * and backend failures propagate.</p>
 *
 * <p><b>Formulas.</b> Each op uses the CPU formula of the module it serves, cited per method: two-pass mean and
 * biased variance (sum, then one division by the count), {@code sdev = sqrt(var + eps)}, {@code (x - mean) / sdev},
 * and the affine step as a separate multiply and add ({@code gamma * y + beta}).</p>
 *
 * <p><b>Tolerance</b> against the CPU path. The reduction class is {@code |gpu - ref64| <= 1e-5 * sqrt(K) * sum|terms|
 * + K * FLT_MIN} against a double-precision reference with {@code K} terms; the division class is {@code |gpu - ref| <=
 * 4 ulp(ref)} or {@code <= FLT_MIN}; single-rounding is bit-identical for normal operands and results. Normalization
 * outputs and the batch statistics are in the reduction class: lane partials run in ascending index order, squares
 * accumulate with {@code fma}, and the lanes combine through the fixed 256-wide tree, where the CPU adds sequentially
 * with a separate multiply. The running-statistic update adds single-rounding arithmetic on top of those statistics.
 * {@link #indexSelectRows} is bit-identical.</p>
 *
 * <p><b>Operands.</b> Inputs may be any view: dense views (row-major, any offset) are bound in place, other views
 * are first copied contiguously on the device. The statistics, gamma and beta of {@code TR_ChannelNormApply_F32} are
 * read at element offset 0, so such operands that do not start at offset 0 of their buffer are staged the same
 * way.</p>
 */
public final class GpuNormOps {

    /** {@code TR_ReduceRowsSmall_F32} serves rows of at most this many columns. */
    private static final int SMALL_ROW_COLS = 64;
    /** Rows below this count with more than {@link #PARTIAL_MIN_COLS} columns are reduced in chunks. */
    private static final int PARTIAL_MAX_ROWS = 64;
    private static final int PARTIAL_MIN_COLS = 65536;
    private static final int PARTIAL_CHUNK_COLS = 8192;
    private static final int PARTIAL_MAX_CHUNKS = 1024;

    private GpuNormOps() {
    }

    // ------------------------------------------------------------------ row normalizations

    /**
     * Layer normalization over the trailing {@code normalizedSize} elements of each row: mean, biased variance,
     * {@code sdev = sqrt(var + eps)}, then normalize and the affine step, as the CPU reference
     * {@link CpuNorm#layerNorm} computes them.
     *
     * @param x              FLOAT32 input whose element count is a multiple of {@code normalizedSize}
     * @param normalizedSize product of the normalized trailing dims, positive
     * @param gamma          FLOAT32 scale of {@code normalizedSize} elements, or null for none
     * @param beta           FLOAT32 shift of {@code normalizedSize} elements, or null for none
     * @param eps            added to the variance before the square root
     * @return a new tensor of {@code x}'s shape, or null (dispatcher contract)
     * @throws IllegalArgumentException for a non-positive size, a size that does not divide the input, a parameter of
     *                                  the wrong length or on another backend, or an input above {@code 2^31-1}
     *                                  elements
     */
    public static Tensor layerNorm(Tensor x, int normalizedSize, Tensor gamma, Tensor beta, float eps) {
        Objects.requireNonNull(x, "input must not be null");
        if (normalizedSize <= 0) {
            throw new IllegalArgumentException("normalizedSize must be positive, got " + normalizedSize);
        }
        if (x.numElements() % normalizedSize != 0) {
            throw new IllegalArgumentException("input " + x.shape() + " does not split into rows of "
                    + normalizedSize + " elements");
        }
        requireKernelSize(x.shape());
        requireParameter(x, gamma, normalizedSize, "gamma");
        requireParameter(x, beta, normalizedSize, "beta");
        if (!allFloat32(x, gamma, beta)) {
            return null;
        }
        TrainGpu g = TrainGpu.forBackend(x.backend());
        if (g == null) {
            return null;
        }
        try (Staging s = new Staging(g)) {
            StridedView in = s.dense(x);
            StridedView gv = s.denseOrNull(gamma);
            StridedView bv = s.denseOrNull(beta);
            int rows = in.numElements() / normalizedSize;
            return output(g, x, x.shape(), DType.FLOAT32, out -> g.launchRows(Kernels.LAYER_NORM_ROWS_F32, rows,
                    rows, normalizedSize, eps, flag(gv), flag(bv), in.offset(), offset(gv), offset(bv), 0,
                    in.buffer(), buffer(g, gv), buffer(g, bv), out.storage().gpuBuffer()));
        }
    }

    /**
     * RMS normalization over the last dim: sum of squares, {@code rms = sqrt(sumSq / n + eps)}, then
     * {@code x / rms * weight}, as the CPU reference {@link CpuNorm#rmsNorm} computes them.
     *
     * @param x      FLOAT32 input of rank at least 1
     * @param weight FLOAT32 scale with as many elements as the last dim, or null for none
     * @param eps    added to the mean square before the square root
     * @return a new tensor of {@code x}'s shape, or null (dispatcher contract)
     * @throws IllegalArgumentException for a rank-0 input, a weight of the wrong length or on another backend, or an
     *                                  input above {@code 2^31-1} elements
     */
    public static Tensor rmsNorm(Tensor x, Tensor weight, float eps) {
        Objects.requireNonNull(x, "input must not be null");
        if (x.shape().rank() < 1) {
            throw new IllegalArgumentException("rmsNorm needs an input of rank at least 1, got " + x.shape());
        }
        requireKernelSize(x.shape());
        int cols = x.shape().dim(x.shape().rank() - 1);
        requireParameter(x, weight, cols, "weight");
        if (!allFloat32(x, weight)) {
            return null;
        }
        TrainGpu g = TrainGpu.forBackend(x.backend());
        if (g == null) {
            return null;
        }
        try (Staging s = new Staging(g)) {
            StridedView in = s.dense(x);
            StridedView wv = s.denseOrNull(weight);
            int rows = cols == 0 ? 0 : in.numElements() / cols;
            return output(g, x, x.shape(), DType.FLOAT32, out -> g.launchRows(Kernels.RMS_NORM_ROWS_F32, rows,
                    rows, cols, eps, flag(wv), in.offset(), offset(wv), 0,
                    in.buffer(), buffer(g, wv), out.storage().gpuBuffer()));
        }
    }

    // ------------------------------------------------------------------ channel normalizations

    /**
     * Batch normalization in training mode for an input {@code [B, C, *]} (BatchNorm1d on {@code [B, C]} and
     * {@code [B, C, L]}, BatchNorm2d on {@code [B, C, H, W]}): per-channel batch mean and biased variance over the
     * {@code B * prod(*)} elements of each channel, normalized with the batch statistics. The running statistics are
     * then updated in place on the device with {@code TR_Axpby_F32} as PyTorch's {@code _BatchNorm} does:
     * {@code running_mean = (1 - momentum) * running_mean + momentum * mean} and {@code running_var = (1 - momentum)
     * * running_var + momentum * var * n / (n - 1)}, the unbiased batch variance with {@code n = B * prod(*)}
     * ({@link CpuNorm#runningVarWeight}). These are the formulas of the training branch of
     * {@link CpuNorm#batchNorm}.
     *
     * @param x           FLOAT32 input of rank at least 2, channels on dim 1
     * @param gamma       FLOAT32 per-channel scale, or null; given exactly when {@code beta} is
     * @param beta        FLOAT32 per-channel shift, or null
     * @param runningMean FLOAT32 per-channel running mean, updated in place, or null; given exactly when
     *                    {@code runningVar} is
     * @param runningVar  FLOAT32 per-channel running variance, updated in place, or null
     * @param momentum    weight of the batch statistic in the running update
     * @param eps         added to the variance before the square root
     * @return a new tensor of {@code x}'s shape, or null (dispatcher contract; the running statistics are untouched)
     * @throws IllegalArgumentException for an input of rank below 2 or above {@code 2^31-1} elements, a single value
     *                                  per channel ({@link CpuNorm#requireBatchStatistics}), unpaired or mis-sized
     *                                  per-channel tensors, a running statistic whose view addresses an element
     *                                  twice, or operands on different backends
     */
    public static Tensor batchNormTrain(Tensor x, Tensor gamma, Tensor beta, Tensor runningMean, Tensor runningVar,
                                        float momentum, float eps) {
        ChannelGeometry geo = ChannelGeometry.of(x, 0);
        CpuNorm.requireBatchStatistics(x.shape(), geo.batch, geo.spatial);
        requireAffine(x, gamma, beta, geo.channels);
        requirePair(runningMean, runningVar, "runningMean", "runningVar");
        requireParameter(x, runningMean, geo.channels, "runningMean");
        requireParameter(x, runningVar, geo.channels, "runningVar");
        requireWritable(runningMean, "runningMean");
        requireWritable(runningVar, "runningVar");
        if (!allFloat32(x, gamma, beta, runningMean, runningVar)) {
            return null;
        }
        TrainGpu g = TrainGpu.forBackend(x.backend());
        if (g == null) {
            return null;
        }
        try (Staging s = new Staging(g)) {
            StridedView in = s.dense(x);
            StridedView gv = s.atOffsetZeroOrNull(gamma);
            StridedView bv = s.atOffsetZeroOrNull(beta);
            GpuBuffer mean = s.temp(geo.channels);
            GpuBuffer var = s.temp(geo.channels);
            return output(g, x, x.shape(), DType.FLOAT32, out -> {
                g.launchRows(Kernels.CHANNEL_STATS_F32, geo.channels,
                        geo.batch, geo.channels, geo.spatial, in.offset(), in.buffer(), mean, var);
                channelApply(g, geo, eps, in, mean, var, gv, bv, out);
                if (runningMean != null) {
                    long n = (long) geo.batch * geo.spatial;
                    updateRunning(g, s, runningMean, mean, momentum, momentum);
                    updateRunning(g, s, runningVar, var, momentum, CpuNorm.runningVarWeight(momentum, n));
                }
            });
        }
    }

    /**
     * Batch normalization in evaluation mode for an input {@code [B, C, *]}: normalized with the running statistics
     * (the evaluation branch of the CPU reference {@link CpuNorm#batchNorm}).
     *
     * @param x           FLOAT32 input of rank at least 2, channels on dim 1
     * @param gamma       FLOAT32 per-channel scale, or null; given exactly when {@code beta} is
     * @param beta        FLOAT32 per-channel shift, or null
     * @param runningMean FLOAT32 per-channel mean
     * @param runningVar  FLOAT32 per-channel variance
     * @param eps         added to the variance before the square root
     * @return a new tensor of {@code x}'s shape, or null (dispatcher contract)
     * @throws IllegalArgumentException for an input of rank below 2, unpaired or mis-sized per-channel tensors, or
     *                                  operands on different backends
     */
    public static Tensor batchNormEval(Tensor x, Tensor gamma, Tensor beta, Tensor runningMean, Tensor runningVar,
                                       float eps) {
        ChannelGeometry geo = ChannelGeometry.of(x, 0);
        requireAffine(x, gamma, beta, geo.channels);
        Objects.requireNonNull(runningMean, "runningMean must not be null");
        Objects.requireNonNull(runningVar, "runningVar must not be null");
        requireParameter(x, runningMean, geo.channels, "runningMean");
        requireParameter(x, runningVar, geo.channels, "runningVar");
        if (!allFloat32(x, gamma, beta, runningMean, runningVar)) {
            return null;
        }
        TrainGpu g = TrainGpu.forBackend(x.backend());
        if (g == null) {
            return null;
        }
        try (Staging s = new Staging(g)) {
            StridedView in = s.dense(x);
            StridedView gv = s.atOffsetZeroOrNull(gamma);
            StridedView bv = s.atOffsetZeroOrNull(beta);
            GpuBuffer mean = s.atOffsetZero(runningMean).buffer();
            GpuBuffer var = s.atOffsetZero(runningVar).buffer();
            return output(g, x, x.shape(), DType.FLOAT32,
                    out -> channelApply(g, geo, eps, in, mean, var, gv, bv, out));
        }
    }

    /**
     * Group normalization of an input {@code [B, C, *]} with {@code groups} groups of {@code C / groups} channels:
     * the mean and biased variance of each (batch, group) over its {@code (C / groups) * prod(*)} elements
     * (the CPU reference {@link CpuNorm#groupNorm}), then per-element normalization and the per-channel
     * affine step. The element of channel {@code c} at spatial index {@code s} is
     * {@code x[(b * C + c) * S + s]}; the statistics come from {@code TR_ReduceRows} over {@code B * groups} rows.
     *
     * @param x      FLOAT32 input of rank at least 2, channels on dim 1
     * @param groups number of groups, positive and dividing the channel count
     * @param gamma  FLOAT32 per-channel scale, or null; given exactly when {@code beta} is
     * @param beta   FLOAT32 per-channel shift, or null
     * @param eps    added to the variance before the square root
     * @return a new tensor of {@code x}'s shape, or null (dispatcher contract)
     * @throws IllegalArgumentException for an input of rank below 2, a group count that is not positive or does not
     *                                  divide the channels, unpaired or mis-sized per-channel tensors, or operands on
     *                                  different backends
     */
    public static Tensor groupNorm(Tensor x, int groups, Tensor gamma, Tensor beta, float eps) {
        if (groups <= 0) {
            throw new IllegalArgumentException("groups must be positive, got " + groups);
        }
        ChannelGeometry geo = ChannelGeometry.of(x, groups);
        requireAffine(x, gamma, beta, geo.channels);
        if (!allFloat32(x, gamma, beta)) {
            return null;
        }
        TrainGpu g = TrainGpu.forBackend(x.backend());
        if (g == null) {
            return null;
        }
        try (Staging s = new Staging(g)) {
            StridedView in = s.dense(x);
            StridedView gv = s.atOffsetZeroOrNull(gamma);
            StridedView bv = s.atOffsetZeroOrNull(beta);
            if (in.numElements() == 0) {
                return output(g, x, x.shape(), DType.FLOAT32, out -> { });
            }
            int rows = geo.batch * groups;
            int cols = (geo.channels / groups) * geo.spatial;
            GpuBuffer mean = s.temp(rows);
            GpuBuffer var = s.temp(rows);
            return output(g, x, x.shape(), DType.FLOAT32, out -> {
                reduceRows(g, s, TrainOps.TR_R_SUM, rows, cols, in, g.placeholder(), mean);
                reduceRows(g, s, TrainOps.TR_R_SQDEV, rows, cols, in, mean, var);
                channelApply(g, geo, eps, in, mean, var, gv, bv, out);
            });
        }
    }

    // ------------------------------------------------------------------ index select

    /**
     * Rows of a table selected by an index tensor: {@code out[i, :] = weight[indices[i], :]}, with output shape
     * {@code indices.shape + [weight.dim(1)]}. INT32 indices are bound directly; INT64 indices are narrowed on the
     * device with Java {@code (int)} (low word, {@code TR_Cast_I64_I32}), as {@code nn/layers/Embedding.java}
     * reads them ({@link Tensor#toIntArray}).
     *
     * <p>With {@code clamp} each index is clamped to {@code [0, rows - 1]} ({@code nn/graph/MessagePassing.java}
     * {@code gatherNodes}). Without it an index outside {@code [0, rows)} is an error ({@code nn/layers/Embedding.java}
     * {@code forward}): the kernel writes zeros for that row and sets the index-select error slot, and the next host
     * synchronization on the backend (a readback, {@link TrainGpu#checkErrors}) throws
     * {@link IndexOutOfBoundsException}. The slot records {@code {flag, index, dim, size}} for the first out-of-range
     * index since the last read, so the deferred message starts with the CPU path's message for that index
     * ({@link TrainGpu#embeddingIndexMessage}) followed by the device-report suffix. Indices are
     * checked by the work-items that copy their row, so a table with {@code dim == 0} has no work-items and validates
     * none (Embedding requires a positive dim).</p>
     *
     * @param weight  FLOAT32 or INT32 table {@code [rows, dim]}
     * @param indices INT32 or INT64 indices of any shape
     * @param clamp   clamp out-of-range indices instead of reporting them
     * @return a new tensor of {@code weight}'s dtype, or null (dispatcher contract)
     * @throws IllegalArgumentException  when the table is not 2-D, the operands live on different backends, or the
     *                                   output exceeds {@code 2^31-1} elements
     * @throws IndexOutOfBoundsException when clamping into an empty table, which has no row to clamp to
     */
    public static Tensor indexSelectRows(Tensor weight, Tensor indices, boolean clamp) {
        Objects.requireNonNull(weight, "weight must not be null");
        Objects.requireNonNull(indices, "indices must not be null");
        if (weight.shape().rank() != 2) {
            throw new IllegalArgumentException("indexSelectRows needs a 2-D table [rows, dim], got " + weight.shape());
        }
        if (indices.backend() != weight.backend()) {
            throw new IllegalArgumentException("indices live on a different backend than the table");
        }
        int rows = weight.shape().dim(0);
        int dim = weight.shape().dim(1);
        long total = indices.numElements() * dim;
        if (total > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("index select output of " + indices.numElements() + " x " + dim
                    + " elements exceeds the 2^31-1 elements of the train kernel ABI");
        }
        if (clamp && rows == 0 && total > 0) {
            throw new IndexOutOfBoundsException("cannot clamp " + indices.numElements()
                    + " indices into an empty table of shape " + weight.shape());
        }
        boolean wordTable = weight.dtype() == DType.FLOAT32 || weight.dtype() == DType.INT32;
        boolean intIndices = indices.dtype() == DType.INT32 || indices.dtype() == DType.INT64;
        if (!wordTable || !intIndices) {
            return null;
        }
        TrainGpu g = TrainGpu.forBackend(weight.backend());
        if (g == null) {
            return null;
        }
        int[] outDims = new int[indices.shape().rank() + 1];
        System.arraycopy(indices.shape().dims(), 0, outDims, 0, indices.shape().rank());
        outDims[outDims.length - 1] = dim;
        int mode = clamp ? TrainOps.TR_IDX_OOB_CLAMP : TrainOps.TR_IDX_OOB_ERROR;
        try (Staging s = new Staging(g)) {
            StridedView w = s.dense(weight);
            StridedView idx = s.int32Indices(indices);
            return output(g, weight, Shape.of(outDims), weight.dtype(), out -> g.launch1d(
                    Kernels.INDEX_SELECT_ROWS_F32, total,
                    idx.numElements(), dim, rows, mode, w.offset(), idx.offset(), 0, TrainGpu.ERROR_SLOT_INDEX_SELECT,
                    w.buffer(), idx.buffer(), out.storage().gpuBuffer(), g.errors()));
        }
    }

    // ------------------------------------------------------------------ launches

    /** {@code TR_ChannelNormApply_F32} over the whole input into {@code out}; statistics and affine at offset 0. */
    static void channelApply(TrainGpu g, ChannelGeometry geo, float eps, StridedView in, GpuBuffer mean,
                             GpuBuffer var, StridedView gamma, StridedView beta, Tensor out) {
        g.launch1d(Kernels.CHANNEL_NORM_APPLY_F32, in.numElements(),
                geo.batch, geo.channels, geo.spatial, geo.groups, eps, flag(gamma), in.offset(), 0,
                in.buffer(), mean, var, buffer(g, gamma), buffer(g, beta), out.storage().gpuBuffer());
    }

    /**
     * {@code running = weight * batch + (1 - momentum) * running} with {@code TR_Axpby_F32} (x = the batch statistic
     * at offset 0, y = the running statistic); {@code weight} is {@code momentum} for the mean and
     * {@link CpuNorm#runningVarWeight} for the variance. A running tensor that is not dense is updated through a
     * contiguous copy written back with {@link GpuCoreOps#copyInto}.
     */
    private static void updateRunning(TrainGpu g, Staging s, Tensor running, GpuBuffer batch, float momentum,
                                      float weight) {
        StridedView rv = StridedView.of(running);
        if (rv.contiguous()) {
            axpbyRunning(g, rv.numElements(), momentum, weight, batch, rv.buffer(), rv.offset());
            return;
        }
        Tensor copy = s.keep(GpuCoreOps.contiguous(running));
        axpbyRunning(g, rv.numElements(), momentum, weight, batch, copy.storage().gpuBuffer(), 0);
        GpuCoreOps.copyInto(copy, running);
    }

    /**
     * {@code y[off + i] = (1 - momentum) * y[off + i] + weight * batch[i]} for {@code i < n}, as the CPU evaluates
     * it. {@code TR_Axpby_F32} does not read {@code y} when its beta is zero, so for {@code momentum == 1} the
     * CPU's {@code 0 * running} term (NaN for a non-finite running value, a signed zero otherwise) is first formed in
     * place ({@code y = 0 * y}, an exact in-place Axpby) and then added to {@code weight * batch}.
     */
    private static void axpbyRunning(TrainGpu g, int n, float momentum, float weight, GpuBuffer batch, GpuBuffer y,
                                     int off) {
        float keep = 1.0f - momentum;
        if (keep != 0.0f) {
            g.launch1d(Kernels.AXPBY_F32, n, n, weight, keep, 0, off, batch, y);
            return;
        }
        g.launch1d(Kernels.AXPBY_F32, n, n, 0.0f, 0.0f, off, off, y, y);
        g.launch1d(Kernels.AXPBY_F32, n, n, weight, 1.0f, 0, off, batch, y);
    }

    /**
     * {@code out[r] = reduce(op, row r) / cols} for the {@code rows} rows of {@code cols} elements of {@code in}, op
     * {@link TrainOps#TR_R_SUM} or {@link TrainOps#TR_R_SQDEV} (which reads {@code mean[r]}), written at offset 0 of
     * {@code out}. Variant and chunk choice depend only on the shape: {@code TR_ReduceRowsSmall_F32} for at most
     * {@link #SMALL_ROW_COLS} columns, chunked partials for few long rows, else one group per row.
     */
    static void reduceRows(TrainGpu g, Staging s, int op, int rows, int cols, StridedView in, GpuBuffer mean,
                           GpuBuffer out) {
        float divisor = (float) cols;
        int chunks = rows < PARTIAL_MAX_ROWS && cols > PARTIAL_MIN_COLS
                ? Math.min(PARTIAL_MAX_CHUNKS, ceilDiv(cols, PARTIAL_CHUNK_COLS)) : 1;
        if (chunks == 1) {
            reduceRowsOnce(g, op, rows, cols, in.buffer(), in.offset(), mean, out, divisor);
            return;
        }
        int groups = rows * chunks;
        GpuBuffer partial = s.temp(groups);
        g.launchRows(Kernels.REDUCE_ROWS_PARTIAL_F32, groups,
                op, rows, cols, chunks, in.offset(), 0, in.buffer(), mean, partial);
        reduceRowsOnce(g, TrainOps.TR_R_SUM, rows, chunks, partial, 0, g.placeholder(), out, divisor);
    }

    static void reduceRowsOnce(TrainGpu g, int op, int rows, int cols, GpuBuffer in, int inOffset,
                               GpuBuffer mean, GpuBuffer out, float divisor) {
        if (cols <= SMALL_ROW_COLS) {
            g.launch1d(Kernels.REDUCE_ROWS_SMALL_F32, rows, op, rows, cols, inOffset, 0, 0, divisor, in, mean, out);
        } else {
            g.launchRows(Kernels.REDUCE_ROWS_F32, rows, op, rows, cols, inOffset, 0, 0, divisor, in, mean, out);
        }
    }

    /** A new contiguous tensor filled by {@code launches}; closed again when they fail. */
    static Tensor output(TrainGpu g, Tensor like, Shape shape, DType dtype, Consumer<Tensor> launches) {
        Tensor out = Tensor.fromStorage(like.backend(), shape, dtype, g.allocate(shape.numElements(), dtype));
        try {
            launches.accept(out);
            return out;
        } catch (RuntimeException e) {
            try {
                out.close();
            } catch (RuntimeException closing) {
                e.addSuppressed(closing);
            }
            throw e;
        }
    }

    // ------------------------------------------------------------------ validation

    private static void requireParameter(Tensor x, Tensor p, long elements, String name) {
        if (p == null) {
            return;
        }
        if (p.backend() != x.backend()) {
            throw new IllegalArgumentException(name + " lives on a different backend than the input");
        }
        if (p.numElements() != elements) {
            throw new IllegalArgumentException(name + " must have " + elements + " elements, got shape " + p.shape());
        }
    }

    /** The train kernel ABI addresses at most {@code 2^31-1} elements. */
    private static void requireKernelSize(Shape shape) {
        if (shape.numElements() > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("input " + shape + " exceeds the 2^31-1 elements of the train kernel "
                    + "ABI");
        }
    }

    /** An in-place target must address each of its elements once (no stride-0 dim of size above 1). */
    private static void requireWritable(Tensor t, String name) {
        if (t == null) {
            return;
        }
        long[] strides = t.strides().strides();
        for (int k = 0; k < strides.length; k++) {
            if (strides[k] == 0 && t.shape().dim(k) > 1) {
                throw new IllegalArgumentException(name + " view " + t.shape() + " with strides " + t.strides()
                        + " addresses an element more than once and cannot be updated in place");
            }
        }
    }

    private static void requirePair(Tensor a, Tensor b, String aName, String bName) {
        if ((a == null) != (b == null)) {
            throw new IllegalArgumentException(aName + " and " + bName + " must both be given or both be null");
        }
    }

    private static void requireAffine(Tensor x, Tensor gamma, Tensor beta, int channels) {
        requirePair(gamma, beta, "gamma", "beta");
        requireParameter(x, gamma, channels, "gamma");
        requireParameter(x, beta, channels, "beta");
    }

    /** True when every non-null tensor is FLOAT32. */
    private static boolean allFloat32(Tensor... tensors) {
        for (Tensor t : tensors) {
            if (t != null && t.dtype() != DType.FLOAT32) {
                return false;
            }
        }
        return true;
    }

    private static int flag(StridedView v) {
        return v == null ? 0 : 1;
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

    // ------------------------------------------------------------------ helpers

    /**
     * The {@code [B, C, *]} geometry of a channel normalization: batch {@code B = dim 0}, channels {@code C = dim 1},
     * spatial {@code S = prod(dims 2..)}, and {@code groups} (0 for BatchNorm's per-channel statistics).
     */
    static final class ChannelGeometry {
        final int batch;
        final int channels;
        final int spatial;
        final int groups;

        private ChannelGeometry(int batch, int channels, int spatial, int groups) {
            this.batch = batch;
            this.channels = channels;
            this.spatial = spatial;
            this.groups = groups;
        }

        static ChannelGeometry of(Tensor x, int groups) {
            Objects.requireNonNull(x, "input must not be null");
            Shape shape = x.shape();
            if (shape.rank() < 2) {
                throw new IllegalArgumentException("channel normalization needs an input [B, C, *] of rank at least 2, "
                        + "got " + shape);
            }
            int channels = shape.dim(1);
            if (groups < 0 || (groups > 0 && channels % groups != 0)) {
                throw new IllegalArgumentException(channels + " channels do not split into " + groups + " groups");
            }
            long spatial = 1;
            for (int k = 2; k < shape.rank(); k++) {
                spatial *= shape.dim(k);
            }
            requireKernelSize(shape);
            if (spatial > Integer.MAX_VALUE) {
                throw new IllegalArgumentException("input " + shape + " has " + spatial + " spatial elements per "
                        + "channel, beyond the 2^31-1 elements of the train kernel ABI");
            }
            return new ChannelGeometry(shape.dim(0), channels, (int) spatial, groups);
        }
    }

    /** Device temporaries of one dispatch: staged operand copies and statistics buffers, closed together. */
    static final class Staging implements AutoCloseable {
        private final TrainGpu g;
        private final List<Tensor> owned = new ArrayList<>();

        Staging(TrainGpu g) {
            this.g = g;
        }

        Tensor keep(Tensor t) {
            owned.add(Objects.requireNonNull(t, "staged tensor must not be null"));
            return t;
        }

        /** The tensor as a dense view (row-major from its offset), copied contiguously first when it is not. */
        StridedView dense(Tensor t) {
            StridedView v = StridedView.of(t);
            return v.contiguous() ? v : StridedView.of(keep(GpuCoreOps.contiguous(t)));
        }

        StridedView denseOrNull(Tensor t) {
            return t == null ? null : dense(t);
        }

        /** The tensor as a dense view starting at element 0 of its buffer, copied first when it is not. */
        StridedView atOffsetZero(Tensor t) {
            StridedView v = StridedView.of(t);
            return v.contiguous() && v.offset() == 0 ? v : StridedView.of(keep(GpuCoreOps.contiguous(t)));
        }

        StridedView atOffsetZeroOrNull(Tensor t) {
            return t == null ? null : atOffsetZero(t);
        }

        /**
         * Indices as a dense INT32 view: INT32 tensors as {@link #dense}, INT64 tensors of any strides narrowed on
         * the device ({@link GpuCoreOps#cast}, which copies a non-dense source contiguously first).
         */
        StridedView int32Indices(Tensor indices) {
            if (indices.dtype() == DType.INT32) {
                return dense(indices);
            }
            Tensor narrowed = GpuCoreOps.cast(indices, DType.INT32);
            if (narrowed == null) {
                throw new IllegalStateException("the device cast of INT64 indices " + indices.shape()
                        + " to INT32 is unavailable");
            }
            return StridedView.of(keep(narrowed));
        }

        /** A fresh FLOAT32 buffer of {@code elements} elements, read and written at offset 0. */
        GpuBuffer temp(int elements) {
            Tensor t = keep(Tensor.fromStorage(g.backend(), Shape.vector(elements), DType.FLOAT32,
                    g.allocate(elements, DType.FLOAT32)));
            return t.storage().gpuBuffer();
        }

        @Override
        public void close() {
            RuntimeException failure = null;
            for (Tensor t : owned) {
                try {
                    t.close();
                } catch (RuntimeException e) {
                    if (failure == null) {
                        failure = e;
                    } else {
                        failure.addSuppressed(e);
                    }
                }
            }
            owned.clear();
            if (failure != null) {
                throw failure;
            }
        }
    }
}

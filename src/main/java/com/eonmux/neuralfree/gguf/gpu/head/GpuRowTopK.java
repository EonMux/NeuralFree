package com.eonmux.neuralfree.gguf.gpu.head;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.PreparedDispatch;
import com.eonmux.neuralfree.gguf.kernels.LlmKernelConstants;
import com.eonmux.neuralfree.gguf.ops.gemm.GemmKernels;

import java.util.Objects;

/**
 * Row-wise top-k readout of a logits buffer on the device ({@value GemmKernels#ROW_TOPK},
 * {@code kernels/llm/sampling_topk.cl}): per row the {@code k} best candidates in rank order (larger
 * value first, the lower index on ties, as {@code CpuOps.topK} ranks) and the row's softmax
 * denominator {@code sum_j exp(l_j - max)}, so a caller reads {@code k} values per row back instead
 * of the whole vocabulary. The logits rows sit at the buffer's element 0, {@code vocab} floats apart.
 *
 * <p>The dispatch is prepared at construction for up to {@code maxRows} rows of up to {@code k} candidates; a
 * step re-sets only the row count, the grid and the step's width, so a narrower readout ranks and packs only
 * its own candidates ({@code rows * width}) and its cost follows the step's width rather than the widest one
 * bound. A backend without the kernel binds nothing ({@link #bind} returns null), and its caller reads the rows
 * back and selects on the host. Not thread-safe.</p>
 */
public final class GpuRowTopK implements AutoCloseable {

    private final GpuOps ops;
    private final PreparedDispatch dispatch;
    private final GpuBuffer topVal;
    private final GpuBuffer topIdx;
    private final GpuBuffer sumExp;
    private final int vocab;
    private final int k;
    private final int maxRows;
    private int lastRows = -1;
    private int width;

    private GpuRowTopK(GpuOps ops, PreparedDispatch dispatch, GpuBuffer topVal, GpuBuffer topIdx,
                       GpuBuffer sumExp, int vocab, int k, int maxRows) {
        this.ops = ops;
        this.dispatch = dispatch;
        this.topVal = topVal;
        this.topIdx = topIdx;
        this.sumExp = sumExp;
        this.vocab = vocab;
        this.k = k;
        this.maxRows = maxRows;
        this.width = k;
    }

    /**
     * Binds the readout of {@code maxRows} logits rows of {@code vocab} floats over {@code logits},
     * reporting {@code k} candidates per row.
     *
     * @return null when the backend provides no {@value GemmKernels#ROW_TOPK} kernel at the launch's
     *         work-group size; the caller keeps its host readout
     * @throws IllegalArgumentException for non-positive sizes or a logits buffer smaller than
     *                                  {@code maxRows * vocab} floats
     */
    public static GpuRowTopK bind(GpuOps ops, ComputeBackend backend, GpuBuffer logits, int maxRows, int vocab,
                                  int k) {
        Objects.requireNonNull(ops, "ops");
        Objects.requireNonNull(backend, "backend");
        Objects.requireNonNull(logits, "logits");
        if (maxRows < 1 || vocab < 1) {
            throw new IllegalArgumentException("row top-k needs positive rows and vocabulary, got " + maxRows
                    + "/" + vocab);
        }
        if (k < 1) {
            throw new IllegalArgumentException("row top-k needs positive k, got " + k);
        }
        if (logits.sizeInBytes() < (long) maxRows * vocab * Float.BYTES) {
            throw new IllegalArgumentException("row top-k logits buffer holds " + logits.sizeInBytes()
                    + " bytes, " + maxRows + " rows of " + vocab + " need " + (long) maxRows * vocab * Float.BYTES);
        }
        int local = Math.min(ops.local(), LlmKernelConstants.LLM_DIMX);
        if (!ops.prewarm(GemmKernels.ROW_TOPK, new long[]{local})) {
            return null;
        }
        GpuBuffer topVal = allocate(backend, (long) maxRows * k * Float.BYTES);
        GpuBuffer topIdx = allocate(backend, (long) maxRows * k * Integer.BYTES);
        GpuBuffer sumExp = allocate(backend, (long) maxRows * Float.BYTES);
        try {
            Object[] args = new Object[GemmKernels.ROWTOPK_ARG_COUNT];
            args[GemmKernels.ROWTOPK_ARG_SIZE] = vocab;
            args[GemmKernels.ROWTOPK_ARG_K] = k;
            args[GemmKernels.ROWTOPK_ARG_N_ROWS] = maxRows;
            args[GemmKernels.ROWTOPK_ARG_LOGITS] = logits;
            args[GemmKernels.ROWTOPK_ARG_TOP_VAL] = topVal;
            args[GemmKernels.ROWTOPK_ARG_TOP_IDX] = topIdx;
            args[GemmKernels.ROWTOPK_ARG_SUM_EXP] = sumExp;
            PreparedDispatch dispatch = ops.prepare(GemmKernels.ROW_TOPK, new long[]{(long) maxRows * local},
                    new long[]{local}, args);
            return new GpuRowTopK(ops, dispatch, topVal, topIdx, sumExp, vocab, k, maxRows);
        } catch (RuntimeException e) {
            close(topVal, topIdx, sumExp, e);
            throw e;
        }
    }

    /** Enqueues the readout of the first {@code rows} logits rows at the bound width {@link #k()}. */
    public void run(int rows) {
        run(rows, k);
    }

    /**
     * Enqueues the readout of the first {@code rows} logits rows, {@code candidates} each, packed
     * {@code candidates} apart from element 0 of {@link #topVal()} and {@link #topIdx()}.
     *
     * @throws IllegalArgumentException when {@code rows} is outside {@code [1, maxRows]} or {@code candidates}
     *                                  outside {@code [1, k()]}
     */
    public void run(int rows, int candidates) {
        if (rows < 1 || rows > maxRows) {
            throw new IllegalArgumentException("row top-k of " + rows + " rows outside [1, " + maxRows + "]");
        }
        if (candidates < 1 || candidates > k) {
            throw new IllegalArgumentException("row top-k of " + candidates + " candidates outside [1, " + k + "]");
        }
        if (rows != lastRows) {
            dispatch.setInt(GemmKernels.ROWTOPK_ARG_N_ROWS, rows);
            dispatch.setGlobal((long) rows * dispatchLocal());
            lastRows = rows;
        }
        if (candidates != width) {
            dispatch.setInt(GemmKernels.ROWTOPK_ARG_K, candidates);
            width = candidates;
        }
        dispatch.dispatch();
    }

    /** Most candidates per row the readout holds (its bound width). */
    public int k() {
        return k;
    }

    /** Candidates per row of the last enqueued readout, the packing stride of its rows. */
    public int width() {
        return width;
    }

    /** The F32 candidate values, {@code rows * width()}. */
    public GpuBuffer topVal() {
        return topVal;
    }

    /** The int32 candidate indices, {@code rows * width()}. */
    public GpuBuffer topIdx() {
        return topIdx;
    }

    /** The F32 softmax denominator of each row, {@code rows}. */
    public GpuBuffer sumExp() {
        return sumExp;
    }

    private int dispatchLocal() {
        return Math.min(ops.local(), LlmKernelConstants.LLM_DIMX);
    }

    private static GpuBuffer allocate(ComputeBackend backend, long bytes) {
        return backend.allocateBuffer(Math.max(bytes, Integer.BYTES));
    }

    private static void close(GpuBuffer topVal, GpuBuffer topIdx, GpuBuffer sumExp, RuntimeException failure) {
        RuntimeException first = failure;
        for (GpuBuffer b : new GpuBuffer[]{topIdx, sumExp, topVal}) {
            try {
                b.close();
            } catch (RuntimeException e) {
                if (first == null) {
                    first = e;
                } else {
                    first.addSuppressed(e);
                }
            }
        }
        if (first != null && first != failure) {
            throw first;
        }
    }

    /** Releases the prepared dispatch and the candidate buffers. Idempotent. */
    @Override
    public void close() {
        dispatch.close();
        close(topVal, topIdx, sumExp, null);
    }
}

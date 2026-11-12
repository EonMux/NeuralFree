package com.eonmux.neuralfree.gguf.gpu.head;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;

import java.util.Objects;

/**
 * The device greedy selection of logits rows ({@code kernels/llm/sampling.cl}, GLSL twin {@code argmax.comp}): the
 * index of each row's first maximum, NaN never selected, an all-NaN row yielding 0.
 *
 * <p>A row of at most {@code UNROLL * local} elements runs {@value #ROW}, one work-group per row. A longer row, a
 * decode step's vocabulary row, would leave one work-group streaming the whole row (151936 floats on Qwen3, a single
 * multiprocessor's bandwidth: 70 us per step on an RTX 5080), so it runs split: {@value #SPLIT} cuts each row into
 * {@link #chunks} chunks of {@code UNROLL * local} elements, one work-group each with all of its loads in flight,
 * writing each chunk's winner to a partials buffer, and {@value #COMBINE} folds a row's chunk winners. The candidate
 * order is total, so both forms give the same ids. A backend whose kernel set lacks the split pair keeps the
 * one-pass form.</p>
 *
 * <p>Instances are immutable launch plans for one row width and work-group size.</p>
 */
public final class GpuArgmax {

    /** The one-pass row argmax: one work-group per row. */
    public static final String ROW = "LLM_Argmax";
    /** The split argmax's chunk pass. */
    public static final String SPLIT = "LLM_ArgmaxSplit";
    /** The split argmax's per-row fold of the chunk winners. */
    public static final String COMBINE = "LLM_ArgmaxCombine";
    /** Elements each work-item of a {@value #SPLIT} work-group scans (the kernels' {@code NF_ARGMAX_UNROLL}). */
    public static final int UNROLL = 8;
    /** Ints of one chunk winner in the partials buffer: its rank key and its index. */
    static final int PARTIAL_INTS = 2;

    private final int size;
    private final int local;
    private final int chunks;

    private GpuArgmax(int size, int local, int chunks) {
        this.size = size;
        this.local = local;
        this.chunks = chunks;
    }

    /**
     * The launch plan for rows of {@code size} floats at work-group size {@code local}: the split form when a row
     * spans more than one chunk and {@code ops} prepares both split kernels, the one-pass form otherwise. Prepares
     * the kernels the plan launches.
     *
     * @throws IllegalArgumentException when {@code size} is outside {@code [1, 2^30]} or {@code local} is not positive
     */
    public static GpuArgmax plan(GpuOps ops, int size, int local) {
        Objects.requireNonNull(ops, "ops");
        int chunks = chunks(size, local);
        long[] l = {local};
        if (chunks > 1 && ops.prewarm(SPLIT, l) && ops.prewarm(COMBINE, l)) {
            return new GpuArgmax(size, local, chunks);
        }
        ops.prewarm(ROW, l);
        return new GpuArgmax(size, local, 1);
    }

    /**
     * Chunks of {@code UNROLL * local} elements a row of {@code size} elements spans.
     *
     * @throws IllegalArgumentException when {@code size} is outside {@code [1, 2^30]} or {@code local} is not positive
     */
    public static int chunks(int size, int local) {
        if (size < 1 || size > (1 << 30)) {
            throw new IllegalArgumentException("argmax row of " + size + " elements outside [1, 2^30]");
        }
        if (local < 1) {
            throw new IllegalArgumentException("argmax work-group size " + local + " is not positive");
        }
        long chunk = (long) UNROLL * local;
        return (int) ((size + chunk - 1) / chunk);
    }

    /** True when this plan runs the split pair. */
    public boolean split() {
        return chunks > 1;
    }

    /** Chunks per row of the split form; 1 for the one-pass form. */
    public int chunksPerRow() {
        return chunks;
    }

    /** Ints of the partials buffer a launch over {@code rows} rows needs; 0 for the one-pass form. */
    public long partialInts(int rows) {
        if (rows < 1) {
            throw new IllegalArgumentException("argmax over " + rows + " rows");
        }
        return split() ? (long) rows * chunks * PARTIAL_INTS : 0;
    }

    /**
     * Enqueues {@code ids[r] = argmax(input row r)} for the first {@code rows} rows of {@code input} (row stride =
     * the plan's size).
     *
     * @param partials at least {@link #partialInts(int)} ints; unused (may be null) for the one-pass form
     * @throws IllegalArgumentException when {@code rows < 1} or the split form gets no partials buffer
     */
    public void enqueue(GpuOps ops, GpuBuffer input, GpuBuffer ids, int rows, GpuBuffer partials) {
        Objects.requireNonNull(ops, "ops");
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(ids, "ids");
        if (rows < 1) {
            throw new IllegalArgumentException("argmax over " + rows + " rows");
        }
        long[] l = {local};
        if (!split()) {
            ops.enqueue(ROW, new long[]{(long) rows * local}, l, size, input, ids, rows);
            return;
        }
        Objects.requireNonNull(partials, "the split argmax's partials buffer");
        ops.enqueue(SPLIT, new long[]{(long) rows * chunks * local}, l, size, chunks, input, partials, rows);
        ops.enqueue(COMBINE, new long[]{(long) rows * local}, l, chunks, partials, ids, rows);
    }
}

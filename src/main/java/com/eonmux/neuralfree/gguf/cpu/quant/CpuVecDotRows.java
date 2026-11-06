package com.eonmux.neuralfree.gguf.cpu.quant;

import java.nio.ByteBuffer;

/**
 * Row-range validation and the shard-aware row walk shared by every {@link CpuVecDot} implementation.
 *
 * <p>A weight tensor may span several mapped shards ({@link CpuTensorRef#shardIndex}); the walk resolves each
 * shard's buffer once and advances by {@link CpuTensorRef#rowBytes()} inside it. Each row is copied into the
 * calling thread's row buffer and the kernels then read it as a packed {@code byte[]}: reading a mapped
 * {@code ByteBuffer} byte by byte costs a bounds-checked call per load, which dominated the vec-dot kernels
 * (llama.cpp reads the mapped rows with plain pointers instead).</p>
 */
final class CpuVecDotRows {

    /** Dot product of one packed weight row with the bound activations. */
    interface RowDot {
        float dot(byte[] row, int off);
    }

    /** Dot product of one packed weight row with the token's activations, told the row's index. */
    interface IndexedRowDot {
        void dot(byte[] row, int off, int index);
    }

    private CpuVecDotRows() {
    }

    private static final ThreadLocal<byte[]> ROWS = new ThreadLocal<>();

    /**
     * The calling thread's row buffers of a register tile, one per weight row of the tile. The array and its
     * buffers are reused by the next tile of the same thread; only the first {@code count} entries are grown.
     */
    private static final ThreadLocal<byte[][]> TILE_ROWS = new ThreadLocal<>();

    /** The calling thread's row cursor: the shard read last and its writable-position duplicate. */
    private static final ThreadLocal<Cursor> CURSORS = ThreadLocal.withInitial(Cursor::new);

    private static final class Cursor {
        ByteBuffer shard;
        ByteBuffer copy;
    }

    /**
     * Bytes of slack past the last row byte. A kernel may load a whole vector at the last byte of its row (a
     * 64-bit quant slice widened from the tail of an interleaved layout), so the buffer keeps the lanes of that
     * load inside the array; the values are never read because the widening that consumes them ignores the
     * lanes past the row.
     */
    private static final int SLACK = 8;

    /** The calling thread's row buffer, grown to hold one row of {@code w} plus {@link #SLACK} bytes. */
    static byte[] buffer(CpuTensorRef w) {
        byte[] buf = ROWS.get();
        int need = (int) Math.min(w.rowBytes(), Integer.MAX_VALUE);
        if (buf == null || buf.length - need < SLACK) {
            buf = new byte[Math.max((int) Math.min(need + (long) SLACK, Integer.MAX_VALUE), 1024)];
            ROWS.set(buf);
        }
        return buf;
    }

    /**
     * The calling thread's row buffer filled with {@code row} of {@code w}, starting at index 0. The buffer is
     * reused by the next call on this thread.
     */
    static byte[] copyRow(CpuTensorRef w, long row) {
        byte[] buf = buffer(w);
        copyRow(w, row, buf);
        return buf;
    }

    /**
     * The calling thread's {@code count} row buffers, each grown to hold one row of {@code w} plus
     * {@link #SLACK} bytes: the tile kernels copy the rows of one register tile into them and read the packed
     * arrays. Entries past {@code count} keep whatever they held.
     */
    static byte[][] buffers(CpuTensorRef w, int count) {
        byte[][] bufs = TILE_ROWS.get();
        if (bufs == null || bufs.length < count) {
            bufs = new byte[Math.max(count, 4)][];
            TILE_ROWS.set(bufs);
        }
        final int need = (int) Math.min(w.rowBytes(), Integer.MAX_VALUE);
        for (int i = 0; i < count; i++) {
            if (bufs[i] == null || bufs[i].length - need < SLACK) {
                bufs[i] = new byte[Math.max((int) Math.min(need + (long) SLACK, Integer.MAX_VALUE), 1024)];
            }
        }
        return bufs;
    }

    /** Copies {@code row} of {@code w} into {@code dst} at index 0. */
    static void copyRow(CpuTensorRef w, long row, byte[] dst) {
        int s = w.shardIndex(row);
        ByteBuffer shard = w.rowBuffer(row);
        Cursor c = CURSORS.get();
        if (c.shard != shard) {
            c.shard = shard;
            c.copy = shard.duplicate();
        }
        c.copy.position((int) ((row - w.shardRowStart(s)) * w.rowBytes()));
        c.copy.get(dst, 0, (int) w.rowBytes());
    }

    /**
     * Validates the arguments of {@link CpuVecDot#dotRows}.
     *
     * @throws IllegalArgumentException  when {@code y} is null
     * @throws IndexOutOfBoundsException when the rows or the output range are out of bounds
     */
    static void check(CpuTensorRef w, long rowStart, int rowCount, float[] y, int yOff) {
        if (y == null) {
            throw new IllegalArgumentException("output array is null");
        }
        if (rowCount < 0 || rowStart < 0 || rowStart > w.rows() - rowCount) {
            throw new IndexOutOfBoundsException("rows [" + rowStart + ", " + (rowStart + rowCount)
                    + ") are outside the " + w.rows() + "-row " + w.type() + " tensor");
        }
        if (yOff < 0 || yOff > y.length - rowCount) {
            throw new IndexOutOfBoundsException("output range [" + yOff + ", " + ((long) yOff + rowCount)
                    + ") exceeds array length " + y.length);
        }
    }

    /**
     * Validates the range, then writes {@code y[yOff + r] = dot(row rowStart + r)} for {@code r < rowCount}.
     *
     * @throws IllegalArgumentException  when {@code y} is null
     * @throws IndexOutOfBoundsException when the rows or the output range are out of bounds
     */
    static void forEachRow(CpuTensorRef w, long rowStart, int rowCount, float[] y, int yOff, RowDot dot) {
        check(w, rowStart, rowCount, y, yOff);
        forEachRow(w, rowStart, rowCount, (buf, off, r) -> y[yOff + r] = dot.dot(buf, off));
    }

    /**
     * Validates nothing, then hands each row of the range to {@code dot} as the calling thread's row buffer
     * filled from index 0.
     */
    static void forEachRow(CpuTensorRef w, long rowStart, int rowCount, IndexedRowDot dot) {
        final byte[] buf = buffer(w);
        for (int r = 0; r < rowCount; r++) {
            copyRow(w, rowStart + r, buf);
            dot.dot(buf, 0, r);
        }
    }

    /**
     * Validates the arguments of a batched product over {@code nTokens} activation rows: the activation array
     * and its tokens, each token's output range, and the row stride.
     *
     * @throws IllegalArgumentException  when the activation array, a token or {@code y} is null, or the stride
     *                                   is too small
     * @throws IndexOutOfBoundsException when {@code nTokens} exceeds the array or an output range is out of
     *                                   bounds
     */
    static void checkTokens(CpuTensorRef w, long rowStart, int rowCount, CpuVecDot.QuantizedActivations[] x,
                            int nTokens, float[] y, int yOff, int yStride) {
        if (x == null) {
            throw new IllegalArgumentException("activation array is null");
        }
        if (nTokens < 0 || nTokens > x.length) {
            throw new IndexOutOfBoundsException(nTokens + " tokens exceed the " + x.length
                    + "-element activation array");
        }
        if (nTokens > 1 && yStride < rowCount) {
            throw new IllegalArgumentException("output stride " + yStride + " is below the " + rowCount
                    + "-row tile for " + nTokens + " tokens");
        }
        for (int t = 0; t < nTokens; t++) {
            if (x[t] == null) {
                throw new IllegalArgumentException("activation vector of token " + t + " is null");
            }
            final long base = (long) yOff + (long) t * yStride;
            if (base < 0 || base > Integer.MAX_VALUE - rowCount) {
                throw new IndexOutOfBoundsException("output range of token " + t + " starts at " + base);
            }
            check(w, rowStart, rowCount, y, (int) base);
        }
    }

    /**
     * Validates the range, then writes {@code y[yOff + r] = kernel.dotRowBytes(row, 0, x)} for every row of it.
     * <p>
     * The loop calls the kernel through {@link CpuVecDot} instead of handing it a callback: measured on Q4_0
     * with Q8_0 activations and 2560-element rows, the callback form of the same kernel allocated 3.8 KB per
     * row dot and ran at a quarter of the speed, because the kernel reached through the callback loses its
     * Vector API intrinsics, while the interface call leaves it a compiled method of its own that keeps them
     * and allocates nothing.
     *
     * @throws IllegalArgumentException  when {@code y} is null
     * @throws IndexOutOfBoundsException when the rows or the output range are out of bounds
     */
    static void forEachRowBytes(CpuTensorRef w, long rowStart, int rowCount, CpuVecDot kernel,
                                CpuVecDot.QuantizedActivations x, float[] y, int yOff) {
        check(w, rowStart, rowCount, y, yOff);
        final byte[] buf = buffer(w);
        for (int r = 0; r < rowCount; r++) {
            copyRow(w, rowStart + r, buf);
            y[yOff + r] = kernel.dotRowBytes(buf, 0, x);
        }
    }

    /**
     * Validates the range, then writes {@code y[yOff + t * yStride + r] = kernel.dotRowBytes(row, 0, x[t])} for
     * every row {@code r} of the range and every token {@code t}: each weight row is read from the tensor once
     * and dotted with every token's activations. Like {@link #forEachRowBytes}, the loop reaches the kernel
     * through the interface call rather than a callback, so the kernels keep their Vector API intrinsics.
     *
     * @throws IllegalArgumentException  when the activation array, a token or {@code y} is null
     * @throws IndexOutOfBoundsException when the rows or a token's output range are out of bounds
     */
    static void forEachRowTokens(CpuTensorRef w, long rowStart, int rowCount, CpuVecDot kernel,
                                 CpuVecDot.QuantizedActivations[] x, int nTokens, float[] y, int yOff,
                                 int yStride) {
        if (x == null) {
            throw new IllegalArgumentException("activation array is null");
        }
        if (nTokens < 0 || nTokens > x.length) {
            throw new IndexOutOfBoundsException(nTokens + " tokens exceed the " + x.length
                    + "-element activation array");
        }
        for (int t = 0; t < nTokens; t++) {
            if (x[t] == null) {
                throw new IllegalArgumentException("activation vector of token " + t + " is null");
            }
            long base = (long) yOff + (long) t * yStride;
            if (base < 0 || base > Integer.MAX_VALUE - rowCount) {
                throw new IndexOutOfBoundsException("output range of token " + t + " starts at " + base);
            }
            check(w, rowStart, rowCount, y, (int) base);
        }
        final byte[] buf = buffer(w);
        for (int r = 0; r < rowCount; r++) {
            copyRow(w, rowStart + r, buf);
            for (int t = 0; t < nTokens; t++) {
                y[yOff + t * yStride + r] = kernel.dotRowBytes(buf, 0, x[t]);
            }
        }
    }
}

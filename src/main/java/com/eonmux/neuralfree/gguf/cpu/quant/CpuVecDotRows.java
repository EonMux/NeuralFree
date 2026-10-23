package com.eonmux.neuralfree.gguf.cpu.quant;

import java.nio.ByteBuffer;

/**
 * Row-range validation and the shard-aware row walk shared by every {@link CpuVecDot#dotRows} implementation.
 * A weight tensor may span several mapped shards ({@link CpuTensorRef#shardIndex}); the walk resolves each shard's
 * buffer once and advances by {@link CpuTensorRef#rowBytes()} inside it.
 */
final class CpuVecDotRows {

    /** Dot product of one weight row starting at byte {@code off} of {@code b} with the bound activations. */
    interface RowDot {
        float dot(ByteBuffer b, int off);
    }

    private CpuVecDotRows() {
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
        final long rowBytes = w.rowBytes();
        final long end = rowStart + rowCount;
        long row = rowStart;
        int yi = yOff;
        while (row < end) {
            final int s = w.shardIndex(row);
            final long shardEnd = Math.min(end, w.shardRowStart(s) + w.shardRows(s));
            final ByteBuffer b = w.rowBuffer(row);
            long off = w.rowOffset(row);
            for (; row < shardEnd; row++, off += rowBytes) {
                y[yi++] = dot.dot(b, (int) off);
            }
        }
    }
}

package com.eonmux.neuralfree.gguf.cpu.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.quant.BlockLayout;
import com.eonmux.neuralfree.gguf.quant.CpuBlockDecoders;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * Immutable view of a 2-D weight tensor stored in place (typically in mmap'd GGUF data), read row by row
 * by the CPU matvec and decode paths. A 3-D tensor (for example MoE experts) is viewed with
 * {@code rows = ne1 * ne2}.
 * <p>
 * Storage contract:
 * <ul>
 *   <li>Row {@code r} starts at tensor byte {@code r * rowBytes}; each row holds {@code cols} elements of
 *       {@code type} packed in {@code BlockLayout.of(type).rowBytes(cols)} bytes, and {@code rowBytes} is at
 *       least that (a larger value describes a strided view).</li>
 *   <li>The tensor bytes are split into consecutive shards. {@code shardByteOff[i]} is the tensor byte at
 *       which {@code shards[i]} begins, and index 0 of {@code shards[i]} is that byte. {@code shardByteOff[0]}
 *       is 0, offsets increase strictly, and every offset is a multiple of {@code rowBytes}, so no row
 *       straddles two shards.</li>
 * </ul>
 * The view keeps read-only little-endian duplicates of the shards; it never changes the caller's buffers.
 * Readers must use absolute indexing on the buffers returned by {@link #rowBuffer(long)}.
 */
public final class CpuTensorRef {

    private final GGMLType type;
    private final ByteBuffer[] shards;
    private final long[] shardByteOff;
    private final long[] shardRowStart;
    private final long rows;
    private final long cols;
    private final long rowBytes;

    /**
     * @throws IllegalArgumentException when the type has no layout, the geometry is inconsistent, or a shard
     *                                  is too small for the rows it must hold
     */
    public CpuTensorRef(GGMLType type, ByteBuffer[] shards, long[] shardByteOff, long rows, long cols,
                        long rowBytes) {
        long packedRow = checkGeometry(type, rows, cols, rowBytes);
        this.type = type;
        this.rows = rows;
        this.cols = cols;
        this.rowBytes = rowBytes;
        this.shardByteOff = checkShardOffsets(shards, shardByteOff, rowBytes);
        this.shardRowStart = new long[shards.length + 1];
        this.shards = new ByteBuffer[shards.length];
        for (int i = 0; i < shards.length; i++) {
            this.shardRowStart[i] = this.shardByteOff[i] / rowBytes;
            this.shards[i] = shards[i].asReadOnlyBuffer().order(ByteOrder.LITTLE_ENDIAN);
        }
        this.shardRowStart[shards.length] = rows;
        checkShardCapacity(packedRow);
    }

    /** Validates shape and stride; returns the packed byte size of one row. */
    private static long checkGeometry(GGMLType type, long rows, long cols, long rowBytes) {
        BlockLayout layout = BlockLayout.of(type);
        if (rows < 0 || cols <= 0) {
            throw new IllegalArgumentException("invalid " + type + " tensor shape rows=" + rows + " cols=" + cols);
        }
        long packedRow = layout.rowBytes(cols);
        if (rowBytes < packedRow) {
            throw new IllegalArgumentException("rowBytes " + rowBytes + " is smaller than the packed " + type
                    + " row of " + cols + " elements (" + packedRow + " bytes)");
        }
        return packedRow;
    }

    /** Validates the shard arrays and returns a copy of the offsets. */
    private static long[] checkShardOffsets(ByteBuffer[] shards, long[] shardByteOff, long rowBytes) {
        if (shards == null || shardByteOff == null || shards.length == 0 || shards.length != shardByteOff.length) {
            throw new IllegalArgumentException("shards and shardByteOff must be non-empty arrays of equal length");
        }
        long[] offsets = shardByteOff.clone();
        if (offsets[0] != 0) {
            throw new IllegalArgumentException("the first shard must start at tensor byte 0, got " + offsets[0]);
        }
        for (int i = 0; i < offsets.length; i++) {
            if (shards[i] == null) {
                throw new IllegalArgumentException("shard " + i + " is null");
            }
            if (i > 0 && offsets[i] <= offsets[i - 1]) {
                throw new IllegalArgumentException("shard offsets must increase strictly; shard " + i
                        + " starts at " + offsets[i] + " after " + offsets[i - 1]);
            }
            if (offsets[i] % rowBytes != 0) {
                throw new IllegalArgumentException("shard " + i + " starts at tensor byte " + offsets[i]
                        + ", which is not a multiple of rowBytes " + rowBytes);
            }
        }
        return offsets;
    }

    /** Validates that every shard holds at least one row and enough bytes for all of its rows. */
    private void checkShardCapacity(long packedRow) {
        for (int i = 0; i < shards.length; i++) {
            long shardRows = shardRows(i);
            if (shardRows <= 0 && !(rows == 0 && shards.length == 1)) {
                throw new IllegalArgumentException("shard " + i + " holds no rows of the " + rows + "-row tensor");
            }
            long needed = shardRows <= 0 ? 0 : (shardRows - 1) * rowBytes + packedRow;
            if (needed > shards[i].limit()) {
                throw new IllegalArgumentException("shard " + i + " holds " + shards[i].limit()
                        + " bytes but its " + shardRows + " rows need " + needed);
            }
        }
    }

    /** A single-shard view of densely packed rows starting at index 0 of {@code data}. */
    public static CpuTensorRef of(GGMLType type, ByteBuffer data, long rows, long cols) {
        long rowBytes = BlockLayout.of(type).rowBytes(cols);
        return new CpuTensorRef(type, new ByteBuffer[]{data}, new long[]{0L}, rows, cols, rowBytes);
    }

    public GGMLType type() {
        return type;
    }

    public long rows() {
        return rows;
    }

    public long cols() {
        return cols;
    }

    public long rowBytes() {
        return rowBytes;
    }

    public int shardCount() {
        return shards.length;
    }

    /** Tensor byte at which shard {@code i} begins. */
    public long shardByteOff(int i) {
        return shardByteOff[i];
    }

    /** First row held by shard {@code i}. */
    public long shardRowStart(int i) {
        return shardRowStart[i];
    }

    /** Number of rows held by shard {@code i}. */
    public long shardRows(int i) {
        return shardRowStart[i + 1] - shardRowStart[i];
    }

    /** Index of the shard holding {@code row}. */
    public int shardIndex(long row) {
        checkRow(row);
        int lo = 0;
        int hi = shards.length - 1;
        while (lo < hi) {
            int mid = (lo + hi + 1) >>> 1;
            if (shardRowStart[mid] <= row) {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        return lo;
    }

    /** The read-only little-endian shard buffer holding {@code row}; read it with absolute indexing only. */
    public ByteBuffer rowBuffer(long row) {
        return shards[shardIndex(row)];
    }

    /** Absolute index of the first byte of {@code row} inside {@link #rowBuffer(long)}. */
    public int rowOffset(long row) {
        int s = shardIndex(row);
        return (int) ((row - shardRowStart[s]) * rowBytes);
    }

    /** Decodes row {@code row} to F32 into {@code dst[dstOff .. dstOff + cols)}. */
    public void decodeRow(long row, float[] dst, int dstOff) {
        int s = shardIndex(row);
        long off = (row - shardRowStart[s]) * rowBytes;
        CpuBlockDecoders.decodeElements(type, shards[s], off, dst, dstOff, cols);
    }

    /** Decodes {@code count} consecutive rows to F32, row-major, into {@code dst} starting at {@code dstOff}. */
    public void decodeRows(long rowStart, int count, float[] dst, int dstOff) {
        if (count < 0 || rowStart < 0 || rowStart + count > rows) {
            throw new IndexOutOfBoundsException("rows [" + rowStart + ", " + (rowStart + count)
                    + ") are outside the " + rows + "-row " + type + " tensor");
        }
        if ((long) count * cols > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(count + " rows of " + cols + " elements exceed one float array");
        }
        for (int r = 0; r < count; r++) {
            decodeRow(rowStart + r, dst, dstOff + (int) (r * cols));
        }
    }

    private void checkRow(long row) {
        if (row < 0 || row >= rows) {
            throw new IndexOutOfBoundsException("row " + row + " is outside the " + rows + "-row " + type + " tensor");
        }
    }

    @Override
    public String toString() {
        return "CpuTensorRef{" + type + ", rows=" + rows + ", cols=" + cols + ", rowBytes=" + rowBytes
                + ", shards=" + shards.length + "}";
    }
}

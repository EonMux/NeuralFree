package com.eonmux.neuralfree.gguf.cpu.repack;

import com.eonmux.neuralfree.gguf.quant.BlockLayout;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * The interleaved {@code block_q8_0x4} weight layout of llama.cpp ggml/src/ggml-cpu/repack.cpp
 * ({@code make_block_q8_0x4}, repack.cpp:3022-3037, and {@code block<8, 4>} repack.h:26-42): four consecutive
 * 32-element Q8_0 blocks of one row form one 136-byte group that keeps the four half deltas first
 * ({@code d[4]}, 8 bytes) and then the 128 quant bytes interleaved in four-byte slices.
 * <p>
 * The layout is byte-permuted in place: a repacked row has the same byte count as its Q8_0 row, and the dot of
 * the {@code Q8_0x4VecDot} kernels over a repacked row equals {@code Q8VecDot.Q8_0} over the original.
 */
public final class Q8_0x4 {

    /** Q8_0 blocks per group ({@code ncols_interleaved}). */
    public static final int BLOCKS = 4;
    /** Elements per group (4 blocks x 32 elements). */
    public static final int ELEMENTS = BLOCKS * BlockLayout.QK8_0;
    /** Bytes per group: 4 half deltas plus 4 x 32 quant bytes. */
    public static final int BYTES = BLOCKS * BlockLayout.Q8_0_BYTES;
    /** Offset of the 8 delta bytes ({@code d[4]}). */
    public static final int D = 0;
    /** Offset of the 128 interleaved quant bytes ({@code qs}). */
    public static final int QS = 8;

    /** Four-byte slices of each source block. */
    private static final int INTERLEAVE = 4;

    private Q8_0x4() {
    }

    /** True when a row of {@code cols} elements splits into whole groups. */
    public static boolean fits(long cols) {
        return cols > 0 && cols % ELEMENTS == 0;
    }

    /**
     * Repacks the {@code groups} consecutive Q8_0 groups of one row from {@code src[srcOff ...]} into
     * {@code dst[dstOff ...]}, {@value #BYTES} bytes per group.
     *
     * @throws IllegalArgumentException when {@code dst} cannot hold the groups
     */
    public static void repackRows(ByteBuffer src, int srcOff, byte[] dst, int dstOff, int groups) {
        int bytes = checkGroups(src, srcOff, dst, dstOff, groups);
        byte[] in = new byte[bytes];
        ByteBuffer row = src.duplicate();
        row.position(srcOff);
        row.get(in, 0, bytes);
        repackRow(in, 0, dst, dstOff, groups);
    }

    /**
     * Repacks {@code groups} groups already in {@code src}, the bulk form {@link #repackRows} uses so a resident
     * row is read once instead of byte by byte.
     */
    static void repackRow(byte[] src, int srcOff, byte[] dst, int dstOff, int groups) {
        int in = srcOff;
        int out = dstOff;
        final int slices = BlockLayout.QK8_0 / INTERLEAVE;
        for (int g = 0; g < groups; g++, in += BLOCKS * BlockLayout.Q8_0_BYTES, out += BYTES) {
            for (int i = 0; i < BLOCKS; i++) {
                int block = in + i * BlockLayout.Q8_0_BYTES;
                dst[out + D + 2 * i] = src[block + BlockLayout.Q8_0_D];
                dst[out + D + 2 * i + 1] = src[block + BlockLayout.Q8_0_D + 1];
            }
            for (int i = 0; i < BLOCKS * slices; i++) {
                int block = in + (i % BLOCKS) * BlockLayout.Q8_0_BYTES + BlockLayout.Q8_0_QS
                        + (i / BLOCKS) * INTERLEAVE;
                System.arraycopy(src, block, dst, out + QS + i * INTERLEAVE, INTERLEAVE);
            }
        }
    }

    /** Validates the group counts and returns the number of source bytes the groups occupy. */
    private static int checkGroups(ByteBuffer src, int srcOff, byte[] dst, int dstOff, int groups) {
        if (groups < 0 || srcOff < 0 || dstOff < 0) {
            throw new IllegalArgumentException("negative group count or offset: " + groups + ", " + srcOff + ", "
                    + dstOff);
        }
        long source = (long) groups * BLOCKS * BlockLayout.Q8_0_BYTES;
        long target = (long) groups * BYTES;
        if (target > dst.length - dstOff) {
            throw new IllegalArgumentException(groups + " groups of " + BYTES + " bytes exceed the "
                    + Math.max(0, dst.length - dstOff) + " destination bytes at " + dstOff);
        }
        if (source > src.limit() - srcOff) {
            throw new IllegalArgumentException(groups + " groups of " + BLOCKS * BlockLayout.Q8_0_BYTES
                    + " source bytes exceed the " + Math.max(0, src.limit() - srcOff) + " bytes at " + srcOff);
        }
        return (int) source;
    }

    /** A little-endian buffer over freshly packed {@code bytes}. */
    public static ByteBuffer buffer(byte[] bytes) {
        return ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
    }

    @Override
    public String toString() {
        return "Q8_0x4{" + BLOCKS + " blocks, " + BYTES + " bytes per group}";
    }
}

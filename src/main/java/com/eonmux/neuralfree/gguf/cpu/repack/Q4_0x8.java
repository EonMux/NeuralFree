package com.eonmux.neuralfree.gguf.cpu.repack;

import com.eonmux.neuralfree.gguf.quant.BlockLayout;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * The interleaved {@code block_q4_0x8} weight layout of llama.cpp ggml/src/ggml-cpu/repack.cpp
 * ({@code make_block_q4_0x8}, repack.cpp:3128-3150, and {@code block<4, 8>} repack.h:26-42): eight consecutive
 * 32-element Q4_0 blocks of one row form one 144-byte group that keeps the eight half deltas first
 * ({@code d[8]}, 16 bytes) and then the 128 quant bytes interleaved in eight-byte slices
 * ({@code blck_size_interleave = 8}), each byte XORed with {@code 0x88} so a nibble's signed reading is
 * {@code value - 8}: the vec-dot recovers it with {@code (byte) (qs << 4) >> 4} on the low nibble and
 * {@code (qs & 0xF0) >> 4} on the high one.
 * <p>
 * The layout is byte-permuted in place: a repacked row has the same byte count as its Q4_0 row, and the dot of
 * the {@code Q4_0x8VecDot} kernels over a repacked row equals {@code LegacyVecDot.Q4_0} over the original.
 */
public final class Q4_0x8 {

    /** Q4_0 blocks per group ({@code ncols_interleaved}). */
    public static final int BLOCKS = 8;
    /** Elements per group (8 blocks x 32 elements). */
    public static final int ELEMENTS = BLOCKS * BlockLayout.QK4_0;
    /** Bytes per group: 8 half deltas plus 8 x 16 quant bytes. */
    public static final int BYTES = BLOCKS * BlockLayout.Q4_0_BYTES;
    /** Offset of the 16 delta bytes ({@code d[8]}). */
    public static final int D = 0;
    /** Offset of the 128 interleaved quant bytes ({@code qs}). */
    public static final int QS = 16;

    /** {@code 0x88} in every byte: turns each stored nibble {@code n} into {@code n + 8}. */
    static final long XOR_MASK = 0x8888888888888888L;

    private Q4_0x8() {
    }

    /** True when a row of {@code cols} elements splits into whole groups. */
    public static boolean fits(long cols) {
        return cols > 0 && cols % ELEMENTS == 0;
    }

    /**
     * Repacks the {@code groups} consecutive Q4_0 groups of one row from {@code src[srcOff ...]} into
     * {@code dst[dstOff ...]}, 144 bytes per group.
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
        for (int g = 0; g < groups; g++, in += BLOCKS * BlockLayout.Q4_0_BYTES, out += BYTES) {
            for (int i = 0; i < BLOCKS; i++) {
                int block = in + i * BlockLayout.Q4_0_BYTES;
                dst[out + D + 2 * i] = src[block + BlockLayout.Q4_0_D];
                dst[out + D + 2 * i + 1] = src[block + BlockLayout.Q4_0_D + 1];
            }
            for (int i = 0; i < 2 * BLOCKS; i++) {
                int block = in + (i % BLOCKS) * BlockLayout.Q4_0_BYTES + BlockLayout.Q4_0_QS;
                int slice = (i / BLOCKS) * 8;
                long v = getLongLE(src, block + slice) ^ XOR_MASK;
                dst[out + QS + i * 8] = (byte) v;
                dst[out + QS + i * 8 + 1] = (byte) (v >>> 8);
                dst[out + QS + i * 8 + 2] = (byte) (v >>> 16);
                dst[out + QS + i * 8 + 3] = (byte) (v >>> 24);
                dst[out + QS + i * 8 + 4] = (byte) (v >>> 32);
                dst[out + QS + i * 8 + 5] = (byte) (v >>> 40);
                dst[out + QS + i * 8 + 6] = (byte) (v >>> 48);
                dst[out + QS + i * 8 + 7] = (byte) (v >>> 56);
            }
        }
    }

    /** Validates the group counts and returns the number of source bytes the groups occupy. */
    private static int checkGroups(ByteBuffer src, int srcOff, byte[] dst, int dstOff, int groups) {
        if (groups < 0 || srcOff < 0 || dstOff < 0) {
            throw new IllegalArgumentException("negative group count or offset: " + groups + ", " + srcOff + ", "
                    + dstOff);
        }
        long source = (long) groups * BLOCKS * BlockLayout.Q4_0_BYTES;
        long target = (long) groups * BYTES;
        if (target > dst.length - dstOff) {
            throw new IllegalArgumentException(groups + " groups of " + BYTES + " bytes exceed the "
                    + Math.max(0, dst.length - dstOff) + " destination bytes at " + dstOff);
        }
        if (source > src.limit() - srcOff) {
            throw new IllegalArgumentException(groups + " groups of " + BLOCKS * BlockLayout.Q4_0_BYTES
                    + " source bytes exceed the " + Math.max(0, src.limit() - srcOff) + " bytes at " + srcOff);
        }
        return (int) source;
    }

    /** Little-endian 64-bit read of eight array bytes. */
    private static long getLongLE(byte[] a, int off) {
        return (a[off] & 0xFFL)
                | (a[off + 1] & 0xFFL) << 8
                | (a[off + 2] & 0xFFL) << 16
                | (a[off + 3] & 0xFFL) << 24
                | (a[off + 4] & 0xFFL) << 32
                | (a[off + 5] & 0xFFL) << 40
                | (a[off + 6] & 0xFFL) << 48
                | (a[off + 7] & 0xFFL) << 56;
    }

    /** A little-endian buffer over freshly packed {@code bytes}. */
    public static ByteBuffer buffer(byte[] bytes) {
        return ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
    }

    @Override
    public String toString() {
        return "Q4_0x8{" + BLOCKS + " blocks, " + BYTES + " bytes per group}";
    }
}

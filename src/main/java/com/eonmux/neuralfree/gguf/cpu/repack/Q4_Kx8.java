package com.eonmux.neuralfree.gguf.cpu.repack;

import com.eonmux.neuralfree.gguf.quant.BlockLayout;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * The interleaved {@code block_q4_Kx8} weight layout of llama.cpp ggml/src/ggml-cpu/repack.cpp
 * ({@code make_block_q4_Kx8}, repack.cpp:3177-3252, and the struct of repack.h:57-62): eight consecutive
 * 256-element Q4_K super-blocks of one row form one 1152-byte group that keeps the eight {@code d} and eight
 * {@code dmin} halfs first, then the 96 bytes of scales and mins regrouped per sub-block across the eight
 * super-blocks, then the 1024 quant bytes interleaved in eight-byte slices.
 * <p>
 * The layout is byte-permuted in place: a repacked row has the same byte count as its Q4_K row, and the dot of
 * the {@code Q4_Kx8VecDot} kernels over a repacked row equals {@code KQuantVecDot.Q4_K} over the original. The
 * scale regrouping follows the packing of {@code make_block_q4_Kx8}; {@link #scalesMins} reads one
 * super-block's eight 6-bit scales and mins back out.
 */
public final class Q4_Kx8 {

    /** Q4_K super-blocks per group ({@code ncols_interleaved}). */
    public static final int BLOCKS = 8;
    /** Elements per group (8 super-blocks x 256 elements). */
    public static final int ELEMENTS = BLOCKS * BlockLayout.QK_K;
    /** Bytes per group: 8 d + 8 dmin + 96 scales + 1024 quant bytes. */
    public static final int BYTES = BLOCKS * BlockLayout.Q4_K_BYTES;
    /** Offset of the eight {@code d} halfs. */
    public static final int D = 0;
    /** Offset of the eight {@code dmin} halfs. */
    public static final int DMIN = 16;
    /** Offset of the regrouped 96 scale and min bytes. */
    public static final int SCALES = 32;
    /** Offset of the 1024 interleaved quant bytes. */
    public static final int QS = 128;

    /** Eight-byte slices of each source super-block. */
    private static final int INTERLEAVE = 8;

    private Q4_Kx8() {
    }

    /** True when a row of {@code cols} elements splits into whole groups. */
    public static boolean fits(long cols) {
        return cols > 0 && cols % ELEMENTS == 0;
    }

    /**
     * Repacks the {@code groups} consecutive Q4_K groups of one row from {@code src[srcOff ...]} into
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
        final int[] s = new int[BLOCKS];
        final int[] m = new int[BLOCKS];
        final int slices = BlockLayout.QK_K * 4 / INTERLEAVE;
        int in = srcOff;
        int out = dstOff;
        for (int g = 0; g < groups; g++, in += BLOCKS * BlockLayout.Q4_K_BYTES, out += BYTES) {
            for (int i = 0; i < BLOCKS; i++) {
                int block = in + i * BlockLayout.Q4_K_BYTES;
                dst[out + D + 2 * i] = src[block + BlockLayout.Q4_K_D];
                dst[out + D + 2 * i + 1] = src[block + BlockLayout.Q4_K_D + 1];
                dst[out + DMIN + 2 * i] = src[block + BlockLayout.Q4_K_DMIN];
                dst[out + DMIN + 2 * i + 1] = src[block + BlockLayout.Q4_K_DMIN + 1];
            }
            for (int i = 0; i < slices; i++) {
                int block = in + (i % BLOCKS) * BlockLayout.Q4_K_BYTES + BlockLayout.Q4_K_QS
                        + (i / BLOCKS) * INTERLEAVE;
                System.arraycopy(src, block, dst, out + QS + i * INTERLEAVE, INTERLEAVE);
            }
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < BLOCKS; j++) {
                    int sc = in + j * BlockLayout.Q4_K_BYTES + BlockLayout.Q4_K_SCALES;
                    s[j] = src[sc + i] & 63;
                    m[j] = src[sc + i + 4] & 63;
                }
                packScales(dst, out + SCALES + i * 12, s, m);
            }
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < BLOCKS; j++) {
                    int sc = in + j * BlockLayout.Q4_K_BYTES + BlockLayout.Q4_K_SCALES;
                    s[j] = ((src[sc + i] & 192) >> 2) | (src[sc + i + 8] & 15);
                    m[j] = ((src[sc + i + 4] & 192) >> 2) | ((src[sc + i + 8] & 240) >> 4);
                }
                packScales(dst, out + SCALES + 48 + i * 12, s, m);
            }
        }
    }

    /** Validates the group counts and returns the number of source bytes the groups occupy. */
    private static int checkGroups(ByteBuffer src, int srcOff, byte[] dst, int dstOff, int groups) {
        if (groups < 0 || srcOff < 0 || dstOff < 0) {
            throw new IllegalArgumentException("negative group count or offset: " + groups + ", " + srcOff + ", "
                    + dstOff);
        }
        long source = (long) groups * BLOCKS * BlockLayout.Q4_K_BYTES;
        long target = (long) groups * BYTES;
        if (target > dst.length - dstOff) {
            throw new IllegalArgumentException(groups + " groups of " + BYTES + " bytes exceed the "
                    + Math.max(0, dst.length - dstOff) + " destination bytes at " + dstOff);
        }
        if (source > src.limit() - srcOff) {
            throw new IllegalArgumentException(groups + " groups of " + BLOCKS * BlockLayout.Q4_K_BYTES
                    + " source bytes exceed the " + Math.max(0, src.limit() - srcOff) + " bytes at " + srcOff);
        }
        return (int) source;
    }

    /** The 12-byte scale block {@code make_block_q4_Kx8} writes for one sub-block across the eight blocks. */
    private static void packScales(byte[] dst, int at, int[] s, int[] m) {
        dst[at] = (byte) ((s[0] & 63) + ((s[4] & 48) << 2));
        dst[at + 1] = (byte) ((s[1] & 63) + ((s[5] & 48) << 2));
        dst[at + 2] = (byte) ((s[2] & 63) + ((s[6] & 48) << 2));
        dst[at + 3] = (byte) ((s[3] & 63) + ((s[7] & 48) << 2));
        dst[at + 4] = (byte) ((m[0] & 63) + ((m[4] & 48) << 2));
        dst[at + 5] = (byte) ((m[1] & 63) + ((m[5] & 48) << 2));
        dst[at + 6] = (byte) ((m[2] & 63) + ((m[6] & 48) << 2));
        dst[at + 7] = (byte) ((m[3] & 63) + ((m[7] & 48) << 2));
        dst[at + 8] = (byte) ((s[4] & 15) + ((m[4] & 15) << 4));
        dst[at + 9] = (byte) ((s[5] & 15) + ((m[5] & 15) << 4));
        dst[at + 10] = (byte) ((s[6] & 15) + ((m[6] & 15) << 4));
        dst[at + 11] = (byte) ((s[7] & 15) + ((m[7] & 15) << 4));
    }

    /**
     * Reads the eight 6-bit scales and mins of super-block {@code block} (0-7) of the group at {@code groupOff}
     * into {@code scales[0..8)} and {@code mins[0..8)}.
     */
    public static void scalesMins(byte[] b, int groupOff, int block, int[] scales, int[] mins) {
        for (int sb = 0; sb < 8; sb++) {
            final int at = groupOff + SCALES + (sb < 4 ? sb * 12 : 48 + (sb - 4) * 12);
            if (block < 4) {
                scales[sb] = b[at + block] & 63;
                mins[sb] = b[at + 4 + block] & 63;
            } else {
                final int up = block - 4;
                final int low = b[at + 8 + up] & 0xFF;
                scales[sb] = (low & 0x0F) | (((b[at + up] & 0xFF) >> 6 & 3) << 4);
                mins[sb] = ((low >> 4) & 0x0F) | (((b[at + 4 + up] & 0xFF) >> 6 & 3) << 4);
            }
        }
    }

    /** {@link #scalesMins(ByteBuffer, int, int, int[], int[])} over a little-endian byte array. */
    public static void scalesMins(ByteBuffer b, int groupOff, int block, int[] scales, int[] mins) {
        for (int sb = 0; sb < 8; sb++) {
            final int at = groupOff + SCALES + (sb < 4 ? sb * 12 : 48 + (sb - 4) * 12);
            if (block < 4) {
                scales[sb] = b.get(at + block) & 63;
                mins[sb] = b.get(at + 4 + block) & 63;
            } else {
                final int up = block - 4;
                final int low = b.get(at + 8 + up) & 0xFF;
                scales[sb] = (low & 0x0F) | (((b.get(at + up) & 0xFF) >> 6 & 3) << 4);
                mins[sb] = ((low >> 4) & 0x0F) | (((b.get(at + 4 + up) & 0xFF) >> 6 & 3) << 4);
            }
        }
    }

    /** A little-endian buffer over freshly packed {@code bytes}. */
    public static ByteBuffer buffer(byte[] bytes) {
        return ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
    }

    @Override
    public String toString() {
        return "Q4_Kx8{" + BLOCKS + " blocks, " + BYTES + " bytes per group}";
    }
}

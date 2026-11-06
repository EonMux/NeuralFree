package com.eonmux.neuralfree.gguf.quant;

import com.eonmux.neuralfree.gguf.GGMLType;

import static com.eonmux.neuralfree.gguf.quant.BlockLayout.TQ1_0_D;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.TQ1_0_QH;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.TQ1_0_QH_BYTES;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.TQ1_0_QS;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.TQ1_0_QS_BYTES;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.TQ2_0_D;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.TQ2_0_QS;
import static com.eonmux.neuralfree.gguf.quant.FpBlockDecoders.fp32ToFp16;

/**
 * The reference quantizers of the ternary BitNet types, ports of llama.cpp's {@code quantize_row_tq1_0_ref} and
 * {@code quantize_row_tq2_0_ref} (ggml-quants.c): one block of {@code QK_K = 256} values scales by its maximum
 * absolute value ({@code d} as fp16, {@code 0} for an all-zero block), rounds {@code x / d} to the trits
 * {-1, 0, 1} half away from zero ({@code lroundf}) and packs them: TQ1_0 five trits per base-3 byte of
 * {@code qs} (32 bytes over strides of 32, then 16 bytes over strides of 16) and four per byte of {@code qh};
 * TQ2_0 two bits per element over strides of 32. The decoders they pair with are {@link TernaryBlockDecoders}.
 */
public final class TernaryBlockQuantizers {

    /** The block element count both ternary types share ({@code QK_K}). */
    private static final int QK_K = GGMLType.TQ1_0.getBlockSize();

    /** {@code 3^5}: the divisor of the ceiling division that spreads five trits over a byte. */
    private static final int POW3_5 = 243;

    /** The {@code qs} bytes TQ1_0 fills over strides of 32 elements ({@code sizeof(qs) - sizeof(qs) % 32}). */
    private static final int TQ1_0_QS_WIDE = TQ1_0_QS_BYTES - TQ1_0_QS_BYTES % 32;

    /** The {@code qs} bytes of a TQ2_0 block. */
    private static final int TQ2_0_QS_BYTES = TQ2_0_D - TQ2_0_QS;

    private TernaryBlockQuantizers() {
    }

    /**
     * Quantizes {@code x[xOff, xOff + 256)} into the TQ1_0 block at {@code y[yOff, yOff + 54)}.
     *
     * @throws IllegalArgumentException when either array does not hold a whole block at its offset
     */
    public static void quantizeTq1_0(float[] x, int xOff, byte[] y, int yOff) {
        checkBlock(x, xOff, y, yOff, GGMLType.TQ1_0);
        float d = absMax(x, xOff);
        float id = d != 0.0f ? 1.0f / d : 0.0f;
        int p = xOff;
        // 5 elements per byte, along the first 32 bytes of qs.
        for (int m = 0; m < 32; m++) {
            y[yOff + TQ1_0_QS + m] = (byte) spread(trits(x, p + m, 32, 5, id));
        }
        p += 5 * 32;
        // 5 elements per byte, along the last 16 bytes of qs.
        int tail = TQ1_0_QS_BYTES - TQ1_0_QS_WIDE;
        for (int m = 0; m < tail; m++) {
            y[yOff + TQ1_0_QS + TQ1_0_QS_WIDE + m] = (byte) spread(trits(x, p + m, tail, 5, id));
        }
        p += 5 * tail;
        // 4 elements per byte of qh, shifted one trit up so the first sits in the most significant position.
        for (int j = 0; j < TQ1_0_QH_BYTES; j++) {
            y[yOff + TQ1_0_QH + j] = (byte) spread(trits(x, p + j, TQ1_0_QH_BYTES, 4, id) * 3);
        }
        putF16(y, yOff + TQ1_0_D, d);
    }

    /**
     * Quantizes {@code x[xOff, xOff + 256)} into the TQ2_0 block at {@code y[yOff, yOff + 66)}.
     *
     * @throws IllegalArgumentException when either array does not hold a whole block at its offset
     */
    public static void quantizeTq2_0(float[] x, int xOff, byte[] y, int yOff) {
        checkBlock(x, xOff, y, yOff, GGMLType.TQ2_0);
        float d = absMax(x, xOff);
        float id = d != 0.0f ? 1.0f / d : 0.0f;
        int p = xOff;
        for (int j = 0; j < TQ2_0_QS_BYTES; j += 32) {
            for (int m = 0; m < 32; m++) {
                int q = 0;
                for (int n = 0; n < 4; n++) {
                    q |= ((lroundf(x[p + m + n * 32] * id) + 1) & 3) << (2 * n);
                }
                y[yOff + TQ2_0_QS + j + m] = (byte) q;
            }
            p += 4 * 32;
        }
        putF16(y, yOff + TQ2_0_D, d);
    }

    /** The quantizer of {@code t}, TQ1_0 or TQ2_0. @throws IllegalArgumentException for any other type */
    public static void quantize(GGMLType t, float[] x, int xOff, byte[] y, int yOff) {
        switch (t) {
            case TQ1_0:
                quantizeTq1_0(x, xOff, y, yOff);
                return;
            case TQ2_0:
                quantizeTq2_0(x, xOff, y, yOff);
                return;
            default:
                throw new IllegalArgumentException(t + " is not a ternary block type");
        }
    }

    /**
     * {@code lroundf(v)}: the nearest integer, halves away from zero. The half is added in double precision, so a
     * float just below a half (such as {@code 0.49999997f}) does not round up the way a float addition would.
     */
    static int lroundf(float v) {
        int r = (int) Math.floor(Math.abs((double) v) + 0.5);
        return v < 0 ? -r : r;
    }

    /** The largest {@code |x|} of the block at {@code off}, {@code 0} for an all-zero block. */
    private static float absMax(float[] x, int off) {
        float amax = 0.0f;
        for (int j = 0; j < QK_K; j++) {
            amax = Math.max(amax, Math.abs(x[off + j]));
        }
        return amax;
    }

    /**
     * The base-3 number of {@code count} trits read at {@code x[p + n * stride]}, each {@code lroundf(x * id)}
     * shifted from {-1, 0, 1} to the digit {0, 1, 2}, the first read the most significant.
     */
    private static int trits(float[] x, int p, int stride, int count, float id) {
        int q = 0;
        for (int n = 0; n < count; n++) {
            q = q * 3 + lroundf(x[p + n * stride] * id) + 1;
        }
        return q;
    }

    /** {@code ceil(q * 256 / 243)}: the five-trit value {@code q} spread over a byte. */
    private static int spread(int q) {
        return (q * 256 + POW3_5 - 1) / POW3_5;
    }

    /** Refuses offsets at which either array does not hold one whole block. */
    private static void checkBlock(float[] x, int xOff, byte[] y, int yOff, GGMLType t) {
        if (xOff < 0 || xOff > x.length - QK_K) {
            throw new IllegalArgumentException("values [" + xOff + ", " + (xOff + QK_K) + ") are not a whole " + t
                    + " block of an array of " + x.length);
        }
        if (yOff < 0 || yOff > y.length - t.getTypeSize()) {
            throw new IllegalArgumentException("bytes [" + yOff + ", " + (yOff + t.getTypeSize()) + ") are not a"
                    + " whole " + t + " block of an array of " + y.length);
        }
    }

    /** {@code v} as little-endian fp16 at {@code y[off]}. */
    private static void putF16(byte[] y, int off, float v) {
        int h = fp32ToFp16(v);
        y[off] = (byte) h;
        y[off + 1] = (byte) (h >>> 8);
    }
}

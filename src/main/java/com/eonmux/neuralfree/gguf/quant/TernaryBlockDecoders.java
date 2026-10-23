package com.eonmux.neuralfree.gguf.quant;

import java.nio.ByteBuffer;

import static com.eonmux.neuralfree.gguf.quant.BlockLayout.*;
import static com.eonmux.neuralfree.gguf.quant.FpBlockDecoders.fp16At;

/**
 * CPU decoders for the ternary BitNet types TQ1_0 (base-3 packing, 5 trits per byte) and TQ2_0
 * (2 bits per element), plus the trit extraction helper shared with the CPU vec-dot kernels.
 * <p>
 * Kernels are package-private: they expect a little-endian buffer and a range already validated by
 * {@link CpuBlockDecoders}.
 */
public final class TernaryBlockDecoders {

    /** Powers of three applied to a packed TQ1_0 byte to rotate trit {@code n} into the top position. */
    private static final int[] POW3 = {1, 3, 9, 27, 81, 243};

    private TernaryBlockDecoders() {
    }

    /**
     * Trit {@code n} (0-4, most significant first) of a TQ1_0 byte, in {0, 1, 2}: {@code q = (u8)(byte * 3^n)},
     * {@code xi = (q * 3) >> 8} (llama.cpp ggml-quants.c:2442-2443).
     */
    public static int tq1Trit(int packedByte, int n) {
        final int q = ((packedByte & 0xFF) * POW3[n]) & 0xFF;
        return (q * 3) >>> 8;
    }

    /** {@code dequantize_row_tq1_0} (llama.cpp ggml-quants.c:2429-2466): {@code y = (trit - 1) * d}. */
    static void decodeTq1_0(ByteBuffer b, int off, float[] y, int yOff, int nb) {
        final int full = TQ1_0_QS_BYTES - TQ1_0_QS_BYTES % 32;
        for (int i = 0; i < nb; i++, off += TQ1_0_BYTES) {
            final float d = fp16At(b, off + TQ1_0_D);
            final int qs = off + TQ1_0_QS;
            for (int j = 0; j < full; j += 32) {
                for (int n = 0; n < 5; ++n) {
                    for (int m = 0; m < 32; ++m) {
                        y[yOff++] = (float) (tq1Trit(b.get(qs + j + m), n) - 1) * d;
                    }
                }
            }
            for (int j = full; j < TQ1_0_QS_BYTES; j += 16) {
                for (int n = 0; n < 5; ++n) {
                    for (int m = 0; m < 16; ++m) {
                        y[yOff++] = (float) (tq1Trit(b.get(qs + j + m), n) - 1) * d;
                    }
                }
            }
            final int qh = off + TQ1_0_QH;
            for (int n = 0; n < 4; ++n) {
                for (int j = 0; j < TQ1_0_QH_BYTES; ++j) {
                    y[yOff++] = (float) (tq1Trit(b.get(qh + j), n) - 1) * d;
                }
            }
        }
    }

    /** {@code dequantize_row_tq2_0} (llama.cpp ggml-quants.c:2468-2485): {@code y = (q2 - 1) * d}. */
    static void decodeTq2_0(ByteBuffer b, int off, float[] y, int yOff, int nb) {
        final int qsBytes = QK_K / 4;
        for (int i = 0; i < nb; i++, off += TQ2_0_BYTES) {
            final float d = fp16At(b, off + TQ2_0_D);
            final int qs = off + TQ2_0_QS;
            for (int j = 0; j < qsBytes; j += 32) {
                for (int l = 0; l < 4; ++l) {
                    for (int m = 0; m < 32; ++m) {
                        final int q = ((b.get(qs + j + m) & 0xFF) >>> (l * 2)) & 3;
                        y[yOff++] = (float) (q - 1) * d;
                    }
                }
            }
        }
    }
}

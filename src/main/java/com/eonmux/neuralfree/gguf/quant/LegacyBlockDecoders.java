package com.eonmux.neuralfree.gguf.quant;

import java.nio.ByteBuffer;

import static com.eonmux.neuralfree.gguf.quant.BlockLayout.*;
import static com.eonmux.neuralfree.gguf.quant.FpBlockDecoders.fp16At;

/**
 * CPU decoders for the legacy 32-element block types Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 and Q8_1.
 * <p>
 * The 4- and 5-bit types use ggml's split nibble layout: byte {@code j} of {@code qs} holds element
 * {@code j} in its low nibble and element {@code j + 16} in its high nibble; for the 5-bit types bit
 * {@code j} of {@code qh} is the fifth bit of element {@code j} and bit {@code j + 16} that of element
 * {@code j + 16}.
 * <p>
 * Kernels are package-private: they expect a little-endian buffer and a range already validated by
 * {@link CpuBlockDecoders}.
 */
public final class LegacyBlockDecoders {

    private LegacyBlockDecoders() {
    }

    /** {@code dequantize_row_q4_0} (llama.cpp ggml-quants.c:459-477): {@code y = (q - 8) * d}. */
    static void decodeQ4_0(ByteBuffer b, int off, float[] y, int yOff, int nb) {
        final int half = QK4_0 / 2;
        for (int i = 0; i < nb; i++, off += Q4_0_BYTES, yOff += QK4_0) {
            final float d = fp16At(b, off + Q4_0_D);
            final int qs = off + Q4_0_QS;
            for (int j = 0; j < half; j++) {
                final int q = b.get(qs + j) & 0xFF;
                final int x0 = (q & 0x0F) - 8;
                final int x1 = (q >>> 4) - 8;
                y[yOff + j] = x0 * d;
                y[yOff + j + half] = x1 * d;
            }
        }
    }

    /** {@code dequantize_row_q4_1} (llama.cpp ggml-quants.c:479-498): {@code y = q * d + m}. */
    static void decodeQ4_1(ByteBuffer b, int off, float[] y, int yOff, int nb) {
        final int half = QK4_1 / 2;
        for (int i = 0; i < nb; i++, off += Q4_1_BYTES, yOff += QK4_1) {
            final float d = fp16At(b, off + Q4_1_D);
            final float m = fp16At(b, off + Q4_1_M);
            final int qs = off + Q4_1_QS;
            for (int j = 0; j < half; j++) {
                final int q = b.get(qs + j) & 0xFF;
                final int x0 = q & 0x0F;
                final int x1 = q >>> 4;
                y[yOff + j] = x0 * d + m;
                y[yOff + j + half] = x1 * d + m;
            }
        }
    }

    /** {@code dequantize_row_q5_0} (llama.cpp ggml-quants.c:500-524): {@code y = (q5 - 16) * d}. */
    static void decodeQ5_0(ByteBuffer b, int off, float[] y, int yOff, int nb) {
        final int half = QK5_0 / 2;
        for (int i = 0; i < nb; i++, off += Q5_0_BYTES, yOff += QK5_0) {
            final float d = fp16At(b, off + Q5_0_D);
            final int qh = b.getInt(off + Q5_0_QH);
            final int qs = off + Q5_0_QS;
            for (int j = 0; j < half; j++) {
                final int xh0 = ((qh >>> j) << 4) & 0x10;
                final int xh1 = (qh >>> (j + 12)) & 0x10;
                final int q = b.get(qs + j) & 0xFF;
                final int x0 = ((q & 0x0F) | xh0) - 16;
                final int x1 = ((q >>> 4) | xh1) - 16;
                y[yOff + j] = x0 * d;
                y[yOff + j + half] = x1 * d;
            }
        }
    }

    /** {@code dequantize_row_q5_1} (llama.cpp ggml-quants.c:526-551): {@code y = q5 * d + m}. */
    static void decodeQ5_1(ByteBuffer b, int off, float[] y, int yOff, int nb) {
        final int half = QK5_1 / 2;
        for (int i = 0; i < nb; i++, off += Q5_1_BYTES, yOff += QK5_1) {
            final float d = fp16At(b, off + Q5_1_D);
            final float m = fp16At(b, off + Q5_1_M);
            final int qh = b.getInt(off + Q5_1_QH);
            final int qs = off + Q5_1_QS;
            for (int j = 0; j < half; j++) {
                final int xh0 = ((qh >>> j) << 4) & 0x10;
                final int xh1 = (qh >>> (j + 12)) & 0x10;
                final int q = b.get(qs + j) & 0xFF;
                final int x0 = (q & 0x0F) | xh0;
                final int x1 = (q >>> 4) | xh1;
                y[yOff + j] = x0 * d + m;
                y[yOff + j + half] = x1 * d + m;
            }
        }
    }

    /** {@code dequantize_row_q8_0} (llama.cpp ggml-quants.c:553-567): {@code y = q * d}. */
    static void decodeQ8_0(ByteBuffer b, int off, float[] y, int yOff, int nb) {
        for (int i = 0; i < nb; i++, off += Q8_0_BYTES, yOff += QK8_0) {
            final float d = fp16At(b, off + Q8_0_D);
            final int qs = off + Q8_0_QS;
            for (int j = 0; j < QK8_0; j++) {
                y[yOff + j] = b.get(qs + j) * d;
            }
        }
    }

    /**
     * Q8_1: {@code y = q * d}. ggml defines no {@code dequantize_row_q8_1}; this applies the block
     * semantics of {@code block_q8_1} (llama.cpp ggml-common.h:259-268, {@code s = d * sum(qs)} is redundant
     * for decoding) in the form of {@code dequantize_row_q8_0}.
     */
    static void decodeQ8_1(ByteBuffer b, int off, float[] y, int yOff, int nb) {
        for (int i = 0; i < nb; i++, off += Q8_1_BYTES, yOff += QK8_1) {
            final float d = fp16At(b, off + Q8_1_D);
            final int qs = off + Q8_1_QS;
            for (int j = 0; j < QK8_1; j++) {
                y[yOff + j] = b.get(qs + j) * d;
            }
        }
    }
}

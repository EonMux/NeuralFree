package com.eonmux.neuralfree.gguf.quant;

import java.nio.ByteBuffer;

import static com.eonmux.neuralfree.gguf.quant.BlockLayout.*;

/**
 * CPU decoders for the NeuralFree extension block types ({@code GGMLType.isExtension()}): F8_E4M3, F8_E4M3_B128,
 * MXFP8, I8_B128, NF4 and FP4_BNB. These carry quantized checkpoint formats exactly, with every scale inside the
 * block; the field offsets are the {@link BlockLayout} constants and the element codecs those of
 * {@link FpBlockDecoders} and {@link QuantTables}.
 * <p>
 * Kernels are package-private: they expect a little-endian buffer and a range already validated by
 * {@link CpuBlockDecoders}.
 */
public final class ExtBlockDecoders {

    /** Value of every E4M3FN code, from {@link FpBlockDecoders#e4m3ToFp32(int)}; codes 0x7F / 0xFF are NaN. */
    private static final float[] E4M3 = buildE4m3();

    private ExtBlockDecoders() {
    }

    private static float[] buildE4m3() {
        float[] t = new float[256];
        for (int i = 0; i < t.length; i++) {
            t[i] = FpBlockDecoders.e4m3ToFp32(i);
        }
        return t;
    }

    /** F8_E4M3 {@code {u8 qs[32]}}: {@code y = e4m3(qs[j])}; the tensor's {@code .scale} applies after the matmul. */
    static void decodeF8E4m3(ByteBuffer b, int off, float[] y, int yOff, int nb) {
        final int n = nb * QK_F8_E4M3;
        final int qs = off + F8_E4M3_QS;
        for (int j = 0; j < n; j++) {
            y[yOff + j] = E4M3[b.get(qs + j) & 0xFF];
        }
    }

    /** F8_E4M3_B128 {@code {f32 d; u8 qs[128]}}: {@code y = d * e4m3(qs[j])}. */
    static void decodeF8E4m3B128(ByteBuffer b, int off, float[] y, int yOff, int nb) {
        for (int i = 0; i < nb; i++, off += F8B128_BYTES, yOff += QK_F8B128) {
            final float d = b.getFloat(off + F8B128_D);
            final int qs = off + F8B128_QS;
            for (int j = 0; j < QK_F8B128; j++) {
                y[yOff + j] = d * E4M3[b.get(qs + j) & 0xFF];
            }
        }
    }

    /** MXFP8 {@code {u8 e; u8 qs[32]}}: {@code y = 2^(e-127) * e4m3(qs[j])}; {@code e == 0xFF} decodes to NaN. */
    static void decodeMxfp8(ByteBuffer b, int off, float[] y, int yOff, int nb) {
        for (int i = 0; i < nb; i++, off += MXFP8_BYTES, yOff += QK_MXFP8) {
            final float d = FpBlockDecoders.e8m0ToFp32(b.get(off + MXFP8_E));
            final int qs = off + MXFP8_QS;
            for (int j = 0; j < QK_MXFP8; j++) {
                y[yOff + j] = d * E4M3[b.get(qs + j) & 0xFF];
            }
        }
    }

    /** I8_B128 {@code {f32 d; i8 qs[128]}}: {@code y = d * qs[j]}. */
    static void decodeI8B128(ByteBuffer b, int off, float[] y, int yOff, int nb) {
        for (int i = 0; i < nb; i++, off += I8B128_BYTES, yOff += QK_I8B128) {
            final float d = b.getFloat(off + I8B128_D);
            final int qs = off + I8B128_QS;
            for (int j = 0; j < QK_I8B128; j++) {
                y[yOff + j] = d * b.get(qs + j);
            }
        }
    }

    /** NF4 {@code {f32 absmax; u8 qs[32]}}: {@code y = nf4Code(q) * absmax}. */
    static void decodeNf4(ByteBuffer b, int off, float[] y, int yOff, int nb) {
        decodeBnb4(b, off, y, yOff, nb, false);
    }

    /** FP4_BNB {@code {f32 absmax; u8 qs[32]}}: {@code y = fp4BnbCode(q) * absmax}. */
    static void decodeFp4Bnb(ByteBuffer b, int off, float[] y, int yOff, int nb) {
        decodeBnb4(b, off, y, yOff, nb, true);
    }

    /**
     * NF4 / FP4_BNB: byte {@code j} of {@code qs} holds element {@code j} in its low nibble and element
     * {@code j + 32} in its high nibble.
     */
    private static void decodeBnb4(ByteBuffer b, int off, float[] y, int yOff, int nb, boolean fp4) {
        final int half = QK_BNB4 / 2;
        for (int i = 0; i < nb; i++, off += BNB4_BYTES, yOff += QK_BNB4) {
            final float absmax = b.getFloat(off + BNB4_D);
            final int qs = off + BNB4_QS;
            for (int j = 0; j < half; j++) {
                final int q = b.get(qs + j) & 0xFF;
                y[yOff + j] = code(q & 0x0F, fp4) * absmax;
                y[yOff + j + half] = code(q >>> 4, fp4) * absmax;
            }
        }
    }

    private static float code(int index, boolean fp4) {
        return fp4 ? QuantTables.fp4BnbCode(index) : QuantTables.nf4Code(index);
    }
}

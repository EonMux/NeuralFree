package com.eonmux.neuralfree.gguf.quant;

import java.nio.ByteBuffer;

import static com.eonmux.neuralfree.gguf.quant.BlockLayout.*;

/**
 * CPU decoders for the NeuralFree extension block types ({@code GGMLType.isExtension()}): F8_E4M3, F8_E5M2,
 * F8_E4M3_B128, MXFP8, MXFP6_E3M2, MXFP6_E2M3, I8_B128, NF4 and FP4_BNB. These carry quantized checkpoint formats exactly, with every scale inside the
 * block; the field offsets are the {@link BlockLayout} constants and the element codecs those of
 * {@link FpBlockDecoders} and {@link QuantTables}.
 * <p>
 * Kernels are package-private: they expect a little-endian buffer and a range already validated by
 * {@link CpuBlockDecoders}.
 */
public final class ExtBlockDecoders {

    /** Value of every E4M3FN code, from {@link FpBlockDecoders#e4m3ToFp32(int)}; codes 0x7F / 0xFF are NaN. */
    private static final float[] E4M3 = buildE4m3();

    /** Value of every E5M2 code, from {@link FpBlockDecoders#e5m2ToFp32(int)} (IEEE half high bytes). */
    private static final float[] E5M2 = buildE5m2();

    /** Value of every OCP-MX FP6 E2M3 code, from {@link FpBlockDecoders#e2m3ToFp32(int)}. */
    private static final float[] E2M3 = buildE2m3();

    /** Value of every OCP-MX FP6 E3M2 code, from {@link FpBlockDecoders#e3m2ToFp32(int)}. */
    private static final float[] E3M2 = buildE3m2();

    private ExtBlockDecoders() {
    }

    private static float[] buildE4m3() {
        float[] t = new float[256];
        for (int i = 0; i < t.length; i++) {
            t[i] = FpBlockDecoders.e4m3ToFp32(i);
        }
        return t;
    }

    private static float[] buildE5m2() {
        float[] t = new float[256];
        for (int i = 0; i < t.length; i++) {
            t[i] = FpBlockDecoders.e5m2ToFp32(i);
        }
        return t;
    }

    private static float[] buildE2m3() {
        float[] t = new float[64];
        for (int i = 0; i < t.length; i++) {
            t[i] = FpBlockDecoders.e2m3ToFp32(i);
        }
        return t;
    }

    private static float[] buildE3m2() {
        float[] t = new float[64];
        for (int i = 0; i < t.length; i++) {
            t[i] = FpBlockDecoders.e3m2ToFp32(i);
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

    /** F8_E5M2 {@code {u8 qs[32]}}: {@code y = e5m2(qs[j])}; the tensor's {@code .scale} applies after the matmul. */
    static void decodeF8E5m2(ByteBuffer b, int off, float[] y, int yOff, int nb) {
        final int n = nb * QK_F8_E4M3;
        final int qs = off + F8_E5M2_QS;
        for (int j = 0; j < n; j++) {
            y[yOff + j] = E5M2[b.get(qs + j) & 0xFF];
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

    /**
     * MXFP6 (OCP MX): {@code {u8 e; u8 qs[24]}}: the value of {@code code[j]} (element {@code j}, four 6-bit
     * codes per three bytes, little-endian bit order) times {@code 2^(e-127)}. {@code e == 0xFF} is the NaN E8M0
     * reservation and is rejected at load.
     *
     * @param e3m2 true for {@code MXFP6_E3M2} (2 mantissa bits), false for {@code MXFP6_E2M3} (3 mantissa bits)
     * @throws IllegalArgumentException when a block's E8M0 exponent byte is 0xFF
     */
    static void decodeMxfp6(ByteBuffer b, int off, float[] y, int yOff, int nb, boolean e3m2) {
        final float[] lut = e3m2 ? E3M2 : E2M3;
        for (int i = 0; i < nb; i++, off += MXFP6_BYTES, yOff += QK_MXFP6) {
            final int e = b.get(off + MXFP6_E) & 0xFF;
            if (e == 0xFF) {
                throw new IllegalArgumentException("MXFP6 block " + i + ": E8M0 exponent byte 0xFF is the NaN"
                        + " reservation" + (e3m2 ? " (MXFP6_E3M2)" : " (MXFP6_E2M3)")
                        + ", rejected at load");
            }
            final float d = FpBlockDecoders.e8m0ToFp32(e);
            final int qs = off + MXFP6_QS;
            for (int j = 0; j < QK_MXFP6; j++) {
                final int b0 = qs + (j >> 2) * 3;
                final int word = (b.get(b0) & 0xFF) | (b.get(b0 + 1) & 0xFF) << 8 | (b.get(b0 + 2) & 0xFF) << 16;
                y[yOff + j] = d * lut[(word >>> (6 * (j & 3))) & 0x3F];
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

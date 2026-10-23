package com.eonmux.neuralfree.gguf.quant;

import java.nio.ByteBuffer;

import static com.eonmux.neuralfree.gguf.quant.BlockLayout.*;

/**
 * CPU decoders for the plain element types (F32, F16, BF16, F64, I8, I16, I32, I64), the FP4 block
 * types (MXFP4, NVFP4) and the 1-/2-bit types Q1_0 and Q2_0, plus the scalar format conversions of
 * llama.cpp ggml/src/ggml-impl.h that the decoders and the activation quantizer share.
 * <p>
 * The per-type {@code decode*} kernels are package-private: they expect a little-endian buffer and a
 * range already validated by {@link CpuBlockDecoders}. The conversion helpers are public and bit-exact.
 */
public final class FpBlockDecoders {

    private FpBlockDecoders() {
    }

    // ------------------------------------------------------------------ scalar conversions

    /**
     * IEEE half to float, bit-exact port of {@code ggml_compute_fp16_to_fp32} (llama.cpp ggml-impl.h:396-417),
     * served from a 65536-entry table.
     */
    public static float fp16ToFp32(int h) {
        return Fp16Table.TABLE[h & 0xFFFF];
    }

    /** Reads the little-endian half at absolute index {@code i} of a little-endian buffer and widens it. */
    public static float fp16At(ByteBuffer le, int i) {
        return Fp16Table.TABLE[le.getShort(i) & 0xFFFF];
    }

    /** Direct computation behind {@link #fp16ToFp32(int)} (llama.cpp ggml-impl.h:396-417). */
    static float computeFp16ToFp32(int h) {
        final int w = (h & 0xFFFF) << 16;
        final int sign = w & 0x80000000;
        final int twoW = w + w;
        final int expOffset = 0xE0 << 23;
        final float expScale = 0x1.0p-112f;
        final float normalized = Float.intBitsToFloat((twoW >>> 4) + expOffset) * expScale;
        final int magicMask = 126 << 23;
        final float magicBias = 0.5f;
        final float denormalized = Float.intBitsToFloat((twoW >>> 17) | magicMask) - magicBias;
        final int denormalizedCutoff = 1 << 27;
        final int result = sign | (Integer.compareUnsigned(twoW, denormalizedCutoff) < 0
                ? Float.floatToRawIntBits(denormalized)
                : Float.floatToRawIntBits(normalized));
        return Float.intBitsToFloat(result);
    }

    /**
     * Float to IEEE half bits (round to nearest even), bit-exact port of {@code ggml_compute_fp32_to_fp16}
     * (llama.cpp ggml-impl.h:419-443). Returns the 16-bit pattern in the low bits of the int.
     */
    public static int fp32ToFp16(float f) {
        final float scaleToInf = 0x1.0p+112f;
        final float scaleToZero = 0x1.0p-110f;
        final float absF = Float.intBitsToFloat(Float.floatToRawIntBits(f) & 0x7FFFFFFF);
        float base = (absF * scaleToInf) * scaleToZero;

        final int w = Float.floatToRawIntBits(f);
        final int shl1W = w + w;
        final int sign = w & 0x80000000;
        int bias = shl1W & 0xFF000000;
        if (Integer.compareUnsigned(bias, 0x71000000) < 0) {
            bias = 0x71000000;
        }

        base = Float.intBitsToFloat((bias >>> 1) + 0x07800000) + base;
        final int bits = Float.floatToRawIntBits(base);
        final int expBits = (bits >>> 13) & 0x00007C00;
        final int mantissaBits = bits & 0x00000FFF;
        final int nonsign = expBits + mantissaBits;
        return ((sign >>> 16) | (Integer.compareUnsigned(shl1W, 0xFF000000) > 0 ? 0x7E00 : nonsign)) & 0xFFFF;
    }

    /** Rounds {@code f} through half precision and back, as a value stored in a ggml half field reads. */
    public static float roundToFp16(float f) {
        return fp16ToFp32(fp32ToFp16(f));
    }

    /** bfloat16 bits to float ({@code ggml_compute_bf16_to_fp32}, llama.cpp ggml-impl.h:606-613). */
    public static float bf16ToFp32(int bits) {
        return Float.intBitsToFloat((bits & 0xFFFF) << 16);
    }

    /**
     * E8M0 exponent byte to half of its power-of-two scale ({@code ggml_e8m0_to_fp32_half},
     * llama.cpp ggml-impl.h:489-507); pairs with the doubled {@link QuantTables#kvalueFp4(int)} codebook.
     */
    public static float e8m0ToFp32Half(int e) {
        final int x = e & 0xFF;
        final int bits = x < 2 ? 0x00200000 << x : (x - 1) << 23;
        return Float.intBitsToFloat(bits);
    }

    /**
     * Unsigned E4M3 scale byte to float, halved for the doubled FP4 codebook ({@code ggml_ue4m3_to_fp32},
     * llama.cpp ggml-impl.h:514-527). Bytes 0x00 and 0x7F decode to 0.
     */
    public static float ue4m3ToFp32(int x) {
        final int u = x & 0xFF;
        if (u == 0 || u == 0x7F) {
            return 0.0f;
        }
        final int exp = (u >>> 3) & 0xF;
        final int man = u & 0x7;
        final float raw = exp == 0
                ? Math.scalb((float) man, -9)
                : Math.scalb(1.0f + (float) man / 8.0f, exp - 7);
        return raw * 0.5f;
    }

    // ------------------------------------------------------------------ plain element types

    /** F32: one little-endian float per element. */
    static void decodeF32(ByteBuffer b, int off, float[] y, int yOff, int n) {
        for (int i = 0; i < n; i++) {
            y[yOff + i] = b.getFloat(off + 4 * i);
        }
    }

    /** F16: {@code GGML_FP16_TO_FP32} per element ({@code ggml_fp16_to_fp32_row}, llama.cpp ggml.c:480-484). */
    static void decodeF16(ByteBuffer b, int off, float[] y, int yOff, int n) {
        for (int i = 0; i < n; i++) {
            y[yOff + i] = Fp16Table.TABLE[b.getShort(off + 2 * i) & 0xFFFF];
        }
    }

    /** BF16: {@code GGML_BF16_TO_FP32} per element ({@code ggml_bf16_to_fp32_row}, llama.cpp ggml.c:493-498). */
    static void decodeBf16(ByteBuffer b, int off, float[] y, int yOff, int n) {
        for (int i = 0; i < n; i++) {
            y[yOff + i] = Float.intBitsToFloat((b.getShort(off + 2 * i) & 0xFFFF) << 16);
        }
    }

    /** F64: narrowed to float with round-to-nearest. */
    static void decodeF64(ByteBuffer b, int off, float[] y, int yOff, int n) {
        for (int i = 0; i < n; i++) {
            y[yOff + i] = (float) b.getDouble(off + 8 * i);
        }
    }

    /** I8: signed byte converted to float. */
    static void decodeI8(ByteBuffer b, int off, float[] y, int yOff, int n) {
        for (int i = 0; i < n; i++) {
            y[yOff + i] = b.get(off + i);
        }
    }

    /** I16: signed little-endian short converted to float. */
    static void decodeI16(ByteBuffer b, int off, float[] y, int yOff, int n) {
        for (int i = 0; i < n; i++) {
            y[yOff + i] = b.getShort(off + 2 * i);
        }
    }

    /** I32: signed little-endian int converted to float (round to nearest). */
    static void decodeI32(ByteBuffer b, int off, float[] y, int yOff, int n) {
        for (int i = 0; i < n; i++) {
            y[yOff + i] = (float) b.getInt(off + 4 * i);
        }
    }

    /** I64: signed little-endian long converted to float (round to nearest). */
    static void decodeI64(ByteBuffer b, int off, float[] y, int yOff, int n) {
        for (int i = 0; i < n; i++) {
            y[yOff + i] = (float) b.getLong(off + 8 * i);
        }
    }

    // ------------------------------------------------------------------ FP4 and 1-/2-bit block types

    /** {@code dequantize_row_mxfp4} (llama.cpp ggml-quants.c:569-587): split nibbles, E8M0 half scale. */
    static void decodeMxfp4(ByteBuffer b, int off, float[] y, int yOff, int nb) {
        final int half = QK_MXFP4 / 2;
        for (int i = 0; i < nb; i++, off += MXFP4_BYTES, yOff += QK_MXFP4) {
            final float d = e8m0ToFp32Half(b.get(off + MXFP4_E));
            final int qs = off + MXFP4_QS;
            for (int j = 0; j < half; j++) {
                final int q = b.get(qs + j) & 0xFF;
                y[yOff + j] = QuantTables.kvalueFp4(q & 0x0F) * d;
                y[yOff + j + half] = QuantTables.kvalueFp4(q >>> 4) * d;
            }
        }
    }

    /**
     * {@code dequantize_row_nvfp4} (llama.cpp ggml-quants.c:589-612): four 16-element sub-blocks, each with a
     * UE4M3 scale and split nibbles over its own 8 bytes.
     */
    static void decodeNvfp4(ByteBuffer b, int off, float[] y, int yOff, int nb) {
        final int nSub = QK_NVFP4 / QK_NVFP4_SUB;
        final int subHalf = QK_NVFP4_SUB / 2;
        for (int i = 0; i < nb; i++, off += NVFP4_BYTES, yOff += QK_NVFP4) {
            for (int s = 0; s < nSub; s++) {
                final float d = ue4m3ToFp32(b.get(off + NVFP4_D + s));
                final int qs = off + NVFP4_QS + s * subHalf;
                final int yb = yOff + s * QK_NVFP4_SUB;
                for (int j = 0; j < subHalf; j++) {
                    final int q = b.get(qs + j) & 0xFF;
                    y[yb + j] = QuantTables.kvalueFp4(q & 0x0F) * d;
                    y[yb + j + subHalf] = QuantTables.kvalueFp4(q >>> 4) * d;
                }
            }
        }
    }

    /**
     * {@code dequantize_row_q1_0} (llama.cpp ggml-quants.c:419-437): bit j of the block selects {@code +d} or
     * {@code -d}.
     */
    static void decodeQ1_0(ByteBuffer b, int off, float[] y, int yOff, int nb) {
        for (int i = 0; i < nb; i++, off += Q1_0_BYTES, yOff += QK1_0) {
            final float d = fp16At(b, off + Q1_0_D);
            final float negD = -d;
            final int qs = off + Q1_0_QS;
            for (int j = 0; j < QK1_0; j++) {
                final int bit = ((b.get(qs + (j >>> 3)) & 0xFF) >>> (j & 7)) & 1;
                y[yOff + j] = bit != 0 ? d : negD;
            }
        }
    }

    /** {@code dequantize_row_q2_0} (llama.cpp ggml-quants.c:439-457): 2-bit codes {0,1,2,3} map to {-1,0,+1,+2}. */
    static void decodeQ2_0(ByteBuffer b, int off, float[] y, int yOff, int nb) {
        for (int i = 0; i < nb; i++, off += Q2_0_BYTES, yOff += QK2_0) {
            final float d = fp16At(b, off + Q2_0_D);
            final int qs = off + Q2_0_QS;
            for (int j = 0; j < QK2_0; j++) {
                final int q = ((b.get(qs + (j >>> 2)) & 0xFF) >>> ((j & 3) * 2)) & 0x03;
                y[yOff + j] = (q - 1) * d;
            }
        }
    }

    /** Lazily built table of every half value, so the 256 KiB table exists only once something decodes F16. */
    private static final class Fp16Table {
        static final float[] TABLE = build();

        private static float[] build() {
            float[] t = new float[1 << 16];
            for (int h = 0; h < t.length; h++) {
                t[h] = computeFp16ToFp32(h);
            }
            return t;
        }
    }
}

package com.eonmux.neuralfree.gguf.quant;

import java.nio.ByteBuffer;

import static com.eonmux.neuralfree.gguf.quant.BlockLayout.*;
import static com.eonmux.neuralfree.gguf.quant.FpBlockDecoders.fp16At;

/**
 * CPU decoders for the 256-element K-quant super-blocks Q2_K, Q3_K, Q4_K, Q5_K, Q6_K and Q8_K, plus the
 * public scale-unpacking helpers that the CPU vec-dot kernels share with them.
 * <p>
 * Kernels are package-private: they expect a little-endian buffer and a range already validated by
 * {@link CpuBlockDecoders}.
 */
public final class KQuantBlockDecoders {

    private static final int KMASK1 = 0x03030303;
    private static final int KMASK2 = 0x0f0f0f0f;

    private KQuantBlockDecoders() {
    }

    /**
     * 6-bit scale and min {@code j} (0-7) of a Q4_K / Q5_K block, {@code get_scale_min_k4}
     * (llama.cpp ggml-quants.c:880-887). Returns {@code scale | (min << 8)}.
     *
     * @param scalesOff absolute index of the 12-byte {@code scales} field
     */
    public static int scaleMinK4(int j, ByteBuffer le, int scalesOff) {
        final int sc;
        final int m;
        if (j < 4) {
            sc = le.get(scalesOff + j) & 63;
            m = le.get(scalesOff + j + 4) & 63;
        } else {
            final int qj4 = le.get(scalesOff + j + 4) & 0xFF;
            sc = (qj4 & 0xF) | (((le.get(scalesOff + j - 4) & 0xFF) >>> 6) << 4);
            m = (qj4 >>> 4) | (((le.get(scalesOff + j) & 0xFF) >>> 6) << 4);
        }
        return sc | (m << 8);
    }

    /**
     * Unpacks the sixteen signed 6-bit Q3_K scales, as {@code dequantize_row_q3_K} does
     * (llama.cpp ggml-quants.c:1324-1329). {@code out[k]} receives the raw value; the caller subtracts 32.
     *
     * @param scalesOff absolute index of the 12-byte {@code scales} field
     * @param out       destination of length at least 16
     */
    public static void q3kScales(ByteBuffer le, int scalesOff, byte[] out) {
        final int a0 = le.getInt(scalesOff);
        final int a1 = le.getInt(scalesOff + 4);
        final int tmp = le.getInt(scalesOff + 8);
        final int w2 = ((a0 >>> 4) & KMASK2) | (((tmp >>> 4) & KMASK1) << 4);
        final int w3 = ((a1 >>> 4) & KMASK2) | (((tmp >>> 6) & KMASK1) << 4);
        final int w0 = (a0 & KMASK2) | ((tmp & KMASK1) << 4);
        final int w1 = (a1 & KMASK2) | (((tmp >>> 2) & KMASK1) << 4);
        unpackWord(w0, out, 0);
        unpackWord(w1, out, 4);
        unpackWord(w2, out, 8);
        unpackWord(w3, out, 12);
    }

    private static void unpackWord(int w, byte[] out, int at) {
        out[at] = (byte) w;
        out[at + 1] = (byte) (w >>> 8);
        out[at + 2] = (byte) (w >>> 16);
        out[at + 3] = (byte) (w >>> 24);
    }

    /** {@code dequantize_row_q2_K} (llama.cpp ggml-quants.c:961-991): {@code y = d*sc*q - dmin*m}. */
    static void decodeQ2_K(ByteBuffer b, int off, float[] y, int yOff, int nb) {
        for (int i = 0; i < nb; i++, off += Q2_K_BYTES) {
            final float d = fp16At(b, off + Q2_K_D);
            final float min = fp16At(b, off + Q2_K_DMIN);
            int q = off + Q2_K_QS;
            int is = 0;
            for (int n = 0; n < QK_K; n += 128) {
                int shift = 0;
                for (int j = 0; j < 4; ++j) {
                    int sc = b.get(off + Q2_K_SCALES + is++) & 0xFF;
                    float dl = d * (sc & 0xF);
                    float ml = min * (sc >>> 4);
                    for (int l = 0; l < 16; ++l) {
                        y[yOff++] = dl * (((b.get(q + l) & 0xFF) >>> shift) & 3) - ml;
                    }
                    sc = b.get(off + Q2_K_SCALES + is++) & 0xFF;
                    dl = d * (sc & 0xF);
                    ml = min * (sc >>> 4);
                    for (int l = 0; l < 16; ++l) {
                        y[yOff++] = dl * (((b.get(q + l + 16) & 0xFF) >>> shift) & 3) - ml;
                    }
                    shift += 2;
                }
                q += 32;
            }
        }
    }

    /** {@code dequantize_row_q3_K} (llama.cpp ggml-quants.c:1306-1354): {@code y = d*(sc-32)*(q2 - (hbit ? 0 : 4))}. */
    static void decodeQ3_K(ByteBuffer b, int off, float[] y, int yOff, int nb) {
        final byte[] scales = new byte[16];
        for (int i = 0; i < nb; i++, off += Q3_K_BYTES) {
            final float dAll = fp16At(b, off + Q3_K_D);
            final int hm = off + Q3_K_HMASK;
            int q = off + Q3_K_QS;
            int m = 1;
            q3kScales(b, off + Q3_K_SCALES, scales);
            int is = 0;
            for (int n = 0; n < QK_K; n += 128) {
                int shift = 0;
                for (int j = 0; j < 4; ++j) {
                    float dl = dAll * (scales[is++] - 32);
                    for (int l = 0; l < 16; ++l) {
                        final int v = (((b.get(q + l) & 0xFF) >>> shift) & 3) - ((b.get(hm + l) & m) != 0 ? 0 : 4);
                        y[yOff++] = dl * v;
                    }
                    dl = dAll * (scales[is++] - 32);
                    for (int l = 0; l < 16; ++l) {
                        final int v = (((b.get(q + l + 16) & 0xFF) >>> shift) & 3)
                                - ((b.get(hm + l + 16) & m) != 0 ? 0 : 4);
                        y[yOff++] = dl * v;
                    }
                    shift += 2;
                    m <<= 1;
                }
                q += 32;
            }
        }
    }

    /** {@code dequantize_row_q4_K} (llama.cpp ggml-quants.c:1530-1552): {@code y = d*sc*q - dmin*m}. */
    static void decodeQ4_K(ByteBuffer b, int off, float[] y, int yOff, int nb) {
        for (int i = 0; i < nb; i++, off += Q4_K_BYTES) {
            final float d = fp16At(b, off + Q4_K_D);
            final float min = fp16At(b, off + Q4_K_DMIN);
            final int scalesOff = off + Q4_K_SCALES;
            int q = off + Q4_K_QS;
            int is = 0;
            for (int j = 0; j < QK_K; j += 64) {
                final int sm1 = scaleMinK4(is, b, scalesOff);
                final float d1 = d * (sm1 & 0xFF);
                final float m1 = min * (sm1 >>> 8);
                final int sm2 = scaleMinK4(is + 1, b, scalesOff);
                final float d2 = d * (sm2 & 0xFF);
                final float m2 = min * (sm2 >>> 8);
                for (int l = 0; l < 32; ++l) {
                    y[yOff++] = d1 * (b.get(q + l) & 0xF) - m1;
                }
                for (int l = 0; l < 32; ++l) {
                    y[yOff++] = d2 * ((b.get(q + l) & 0xFF) >>> 4) - m2;
                }
                q += 32;
                is += 2;
            }
        }
    }

    /** {@code dequantize_row_q5_K} (llama.cpp ggml-quants.c:1732-1757): {@code y = d*sc*(q4 + 16*hbit) - dmin*m}. */
    static void decodeQ5_K(ByteBuffer b, int off, float[] y, int yOff, int nb) {
        for (int i = 0; i < nb; i++, off += Q5_K_BYTES) {
            final float d = fp16At(b, off + Q5_K_D);
            final float min = fp16At(b, off + Q5_K_DMIN);
            final int scalesOff = off + Q5_K_SCALES;
            final int qh = off + Q5_K_QH;
            int ql = off + Q5_K_QS;
            int is = 0;
            int u1 = 1;
            int u2 = 2;
            for (int j = 0; j < QK_K; j += 64) {
                final int sm1 = scaleMinK4(is, b, scalesOff);
                final float d1 = d * (sm1 & 0xFF);
                final float m1 = min * (sm1 >>> 8);
                final int sm2 = scaleMinK4(is + 1, b, scalesOff);
                final float d2 = d * (sm2 & 0xFF);
                final float m2 = min * (sm2 >>> 8);
                for (int l = 0; l < 32; ++l) {
                    y[yOff++] = d1 * ((b.get(ql + l) & 0xF) + ((b.get(qh + l) & u1) != 0 ? 16 : 0)) - m1;
                }
                for (int l = 0; l < 32; ++l) {
                    y[yOff++] = d2 * (((b.get(ql + l) & 0xFF) >>> 4) + ((b.get(qh + l) & u2) != 0 ? 16 : 0)) - m2;
                }
                ql += 32;
                is += 2;
                u1 <<= 2;
                u2 <<= 2;
            }
        }
    }

    /** {@code dequantize_row_q6_K} (llama.cpp ggml-quants.c:1940-1969): {@code y = d*sc*(q6 - 32)}. */
    static void decodeQ6_K(ByteBuffer b, int off, float[] y, int yOff, int nb) {
        for (int i = 0; i < nb; i++, off += Q6_K_BYTES) {
            final float d = fp16At(b, off + Q6_K_D);
            int ql = off + Q6_K_QL;
            int qh = off + Q6_K_QH;
            int sc = off + Q6_K_SCALES;
            for (int n = 0; n < QK_K; n += 128) {
                for (int l = 0; l < 32; ++l) {
                    final int is = l / 16;
                    final int l0 = b.get(ql + l) & 0xFF;
                    final int l32 = b.get(ql + l + 32) & 0xFF;
                    final int h = b.get(qh + l) & 0xFF;
                    final int q1 = ((l0 & 0xF) | ((h & 3) << 4)) - 32;
                    final int q2 = ((l32 & 0xF) | (((h >>> 2) & 3) << 4)) - 32;
                    final int q3 = ((l0 >>> 4) | (((h >>> 4) & 3) << 4)) - 32;
                    final int q4 = ((l32 >>> 4) | (((h >>> 6) & 3) << 4)) - 32;
                    y[yOff + l] = d * b.get(sc + is) * q1;
                    y[yOff + l + 32] = d * b.get(sc + is + 2) * q2;
                    y[yOff + l + 64] = d * b.get(sc + is + 4) * q3;
                    y[yOff + l + 96] = d * b.get(sc + is + 6) * q4;
                }
                yOff += 128;
                ql += 64;
                qh += 32;
                sc += 8;
            }
        }
    }

    /** {@code dequantize_row_q8_K} (llama.cpp ggml-quants.c:2808-2817): {@code y = d * q} with a float {@code d}. */
    static void decodeQ8_K(ByteBuffer b, int off, float[] y, int yOff, int nb) {
        for (int i = 0; i < nb; i++, off += Q8_K_BYTES, yOff += QK_K) {
            final float d = b.getFloat(off + Q8_K_D);
            final int qs = off + Q8_K_QS;
            for (int j = 0; j < QK_K; ++j) {
                y[yOff + j] = d * b.get(qs + j);
            }
        }
    }
}

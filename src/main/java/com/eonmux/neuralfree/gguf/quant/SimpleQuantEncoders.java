package com.eonmux.neuralfree.gguf.quant;

import com.eonmux.neuralfree.gguf.GGMLType;

import java.nio.ByteBuffer;

import static com.eonmux.neuralfree.gguf.quant.BlockLayout.MXFP4_BYTES;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.MXFP4_E;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.MXFP4_QS;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q1_0_BYTES;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q1_0_D;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q1_0_QS;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q2_0_BYTES;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q2_0_D;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q2_0_QS;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.QK1_0;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.QK2_0;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.QK_K;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.QK_MXFP4;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q8_K_BYTES;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q8_K_BSUMS;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q8_K_D;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q8_K_QS;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.TQ1_0_BYTES;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.TQ1_0_D;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.TQ1_0_QH;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.TQ1_0_QH_BYTES;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.TQ1_0_QS;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.TQ1_0_QS_BYTES;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.TQ2_0_BYTES;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.TQ2_0_D;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.TQ2_0_QS;
import static com.eonmux.neuralfree.gguf.quant.FpBlockDecoders.e8m0ToFp32Half;
import static com.eonmux.neuralfree.gguf.quant.FpBlockDecoders.fp32ToFp16;
import static com.eonmux.neuralfree.gguf.quant.QuantTables.kvalueFp4;

/**
 * Row encoders of the remaining ggml types the quantize tool writes: the experimental 1-bit and 2-bit
 * blocks Q1_0 and Q2_0, MXFP4, the ternary TQ1_0 and TQ2_0 (BitNet b1.58 / TriLMs), the internal Q8_K and
 * the float rows F16, BF16 and F32. All are the {@code quantize_row_*_ref} functions of llama.cpp
 * ggml-quants.c ({@code q1_0} :37, {@code q2_0} :71, {@code mxfp4} :347 with {@code best_index_mxfp4}
 * :334, {@code tq1_0} :2332, {@code tq2_0} :2398, {@code q8_K} :2784); the quantize entry points of these
 * types run the reference either way, importance weights included. The float rows are the row conversions
 * {@code ggml_fp32_to_fp16_row}, {@code ggml_fp32_to_bf16_row_ref} and the F32 copy of
 * {@code ggml_quantize_chunk}.
 * <p>
 * Package seam: {@link RowQuantizers}.
 */
public final class SimpleQuantEncoders {

    private SimpleQuantEncoders() {
    }

    /** True when {@link #quantizeRow} converts {@code type}. */
    public static boolean supports(GGMLType type) {
        switch (type) {
            case F32:
            case F16:
            case BF16:
            case Q1_0:
            case Q2_0:
            case MXFP4:
            case TQ1_0:
            case TQ2_0:
            case Q8_K:
                return true;
            default:
                return false;
        }
    }

    /**
     * Quantizes one row of {@code nPerRow} values at {@code xOff} into {@code dst} at {@code dstOff};
     * {@code weights} is ignored, as the reference entry points of these types ignore it.
     *
     * @throws IllegalArgumentException when the row is not a whole number of blocks of {@code type}
     */
    public static void quantizeRow(GGMLType type, float[] x, int xOff, byte[] dst, int dstOff, int nPerRow,
                                   float[] weights) {
        switch (type) {
            case F32:
                f32Row(x, xOff, dst, dstOff, nPerRow);
                return;
            case F16:
                f16Row(x, xOff, dst, dstOff, nPerRow);
                return;
            case BF16:
                bf16Row(x, xOff, dst, dstOff, nPerRow);
                return;
            case Q1_0:
                q1_0Row(x, xOff, dst, dstOff, nPerRow);
                return;
            case Q2_0:
                q2_0Row(x, xOff, dst, dstOff, nPerRow);
                return;
            case MXFP4:
                mxfp4Row(x, xOff, dst, dstOff, nPerRow);
                return;
            case TQ1_0:
                tq1_0Row(x, xOff, dst, dstOff, nPerRow);
                return;
            case TQ2_0:
                tq2_0Row(x, xOff, dst, dstOff, nPerRow);
                return;
            case Q8_K:
                q8KRow(x, xOff, dst, dstOff, nPerRow);
                return;
            default:
                throw new IllegalArgumentException(type + " has no simple row encoder");
        }
    }

    /** The float row of F32: little-endian values in order. */
    private static void f32Row(float[] x, int xOff, byte[] dst, int dstOff, int n) {
        ByteBuffer b = ByteBuffer.wrap(dst, dstOff, n * 4).order(java.nio.ByteOrder.LITTLE_ENDIAN);
        for (int j = 0; j < n; j++) {
            b.putFloat(x[xOff + j]);
        }
    }

    /** The float row of F16 ({@code ggml_fp32_to_fp16_row}). */
    private static void f16Row(float[] x, int xOff, byte[] dst, int dstOff, int n) {
        for (int j = 0; j < n; j++) {
            int h = fp32ToFp16(x[xOff + j]);
            dst[dstOff + 2 * j] = (byte) h;
            dst[dstOff + 2 * j + 1] = (byte) (h >>> 8);
        }
    }

    /** The float row of BF16 ({@code ggml_fp32_to_bf16_row_ref}). */
    private static void bf16Row(float[] x, int xOff, byte[] dst, int dstOff, int n) {
        for (int j = 0; j < n; j++) {
            int h = com.eonmux.neuralfree.gguf.memory.KvBlockCodec.fp32ToBf16(x[xOff + j]);
            dst[dstOff + 2 * j] = (byte) h;
            dst[dstOff + 2 * j + 1] = (byte) (h >>> 8);
        }
    }

    private static void checkBlocks(int nPerRow, int block, GGMLType type) {
        if (nPerRow < 0 || nPerRow % block != 0) {
            throw new IllegalArgumentException("a row of " + nPerRow + " values is not a whole number of "
                    + block + "-value " + type + " blocks");
        }
    }

    /** One Q1_0 block of 128 values: the mean |x| scale and one sign bit per value. */
    private static void q1_0Row(float[] x, int xOff, byte[] dst, int dstOff, int nPerRow) {
        checkBlocks(nPerRow, QK1_0, GGMLType.Q1_0);
        for (int i = 0; i < nPerRow / QK1_0; i++) {
            int off = dstOff + i * Q1_0_BYTES;
            float sumAbs = 0.0f;
            for (int j = 0; j < QK1_0; j++) {
                sumAbs += Math.abs(x[xOff + i * QK1_0 + j]);
            }
            float d = sumAbs / QK1_0;
            KQuantEncoders.putHalf(dst, off + Q1_0_D, d);

            // Clear all bits first, then store the sign of each weight directly (no normalization)
            for (int j = 0; j < QK1_0 / 8; ++j) {
                dst[off + Q1_0_QS + j] = 0;
            }
            for (int j = 0; j < QK1_0; ++j) {
                if (x[xOff + i * QK1_0 + j] >= 0.0f) {
                    dst[off + Q1_0_QS + j / 8] |= (byte) (1 << (j % 8));
                }
            }
        }
    }

    /** One Q2_0 block of 64 values: the max-|x| scale and 2-bit {@code round(w/d) + 1} codes. */
    private static void q2_0Row(float[] x, int xOff, byte[] dst, int dstOff, int nPerRow) {
        checkBlocks(nPerRow, QK2_0, GGMLType.Q2_0);
        for (int i = 0; i < nPerRow / QK2_0; i++) {
            int off = dstOff + i * Q2_0_BYTES;
            float amax = 0.0f;
            for (int j = 0; j < QK2_0; j++) {
                float a = Math.abs(x[xOff + i * QK2_0 + j]);
                if (a > amax) {
                    amax = a;
                }
            }
            float d = amax;
            float id = d > 0.0f ? 1.0f / d : 0.0f;
            KQuantEncoders.putHalf(dst, off + Q2_0_D, d);

            for (int j = 0; j < QK2_0 / 4; ++j) {
                dst[off + Q2_0_QS + j] = 0;
            }

            // 00 (-1) = -scale, 01 (0) = 0, 10 (+1) = +scale, 11 (+2) = 2*scale
            for (int j = 0; j < QK2_0; ++j) {
                float w = x[xOff + i * QK2_0 + j];
                int q = lround(w * id) + 1;
                if (q < 0) {
                    q = 0;
                }
                if (q > 3) {
                    q = 3;
                }
                dst[off + Q2_0_QS + j / 4] |= (byte) (q << ((j % 4) * 2));
            }
        }
    }

    /** One MXFP4 block of 32 values: the E8M0 group scale and the nearest E2M1 code. */
    private static void mxfp4Row(float[] x, int xOff, byte[] dst, int dstOff, int nPerRow) {
        checkBlocks(nPerRow, QK_MXFP4, GGMLType.MXFP4);
        for (int i = 0; i < nPerRow / QK_MXFP4; i++) {
            int off = dstOff + i * MXFP4_BYTES;
            float amax = 0.0f; // absolute max
            for (int j = 0; j < QK_MXFP4; j++) {
                float v = x[xOff + i * QK_MXFP4 + j];
                if (amax < Math.abs(v)) {
                    amax = Math.abs(v);
                }
            }
            // floor(log2(amax)): the unbiased exponent, exact for normal values like the correctly
            // rounded log2f the reference floors
            byte e = amax > 0.0f ? (byte) (Math.getExponent(amax) - 2 + 127) : 0;
            float d = e8m0ToFp32Half(e & 0xFF);
            dst[off + MXFP4_E] = e;
            for (int j = 0; j < QK_MXFP4 / 2; ++j) {
                int x0 = bestIndexMxfp4(x[xOff + i * QK_MXFP4 + j], d);
                int x1 = bestIndexMxfp4(x[xOff + i * QK_MXFP4 + QK_MXFP4 / 2 + j], d);
                dst[off + MXFP4_QS + j] = (byte) (x0 | (x1 << 4));
            }
        }
    }

    /** {@code best_index_mxfp4}: the nearest {@code kvalues_fp4} value, first strict minimum wins. */
    private static int bestIndexMxfp4(float x, float e) {
        int bestIndex = 0;
        float bestErr = Math.abs(kvalueFp4(0) * e - x);
        for (int i = 1; i < 16; i++) {
            float err = Math.abs(kvalueFp4(i) * e - x);
            if (err < bestErr) {
                bestIndex = i;
                bestErr = err;
            }
        }
        return bestIndex;
    }

    /** One TQ1_0 super-block: 256 ternary values in base-3 packed bytes ({@code quantize_row_tq1_0_ref}). */
    private static void tq1_0Row(float[] x, int xOff, byte[] dst, int dstOff, int nPerRow) {
        checkBlocks(nPerRow, QK_K, GGMLType.TQ1_0);
        for (int i = 0; i < nPerRow / QK_K; i++) {
            int off = dstOff + i * TQ1_0_BYTES;
            float amax = 0.0f; // absolute max
            for (int j = 0; j < QK_K; j++) {
                float v = x[xOff + j];
                amax = Math.max(amax, Math.abs(v));
            }
            float d = amax;
            float id = d != 0 ? 1.0f / d : 0.0f;
            KQuantEncoders.putHalf(dst, off + TQ1_0_D, d);

            // 5 elements per byte, along 32 bytes
            int p = xOff;
            int bs = TQ1_0_QS_BYTES - TQ1_0_QS_BYTES % 32;
            for (int j = 0; j < bs; j += 32) {
                for (int m = 0; m < 32; ++m) {
                    int q = 0;
                    for (int n = 0; n < 5; ++n) {
                        int xi = lround(x[p + m + n * 32] * id) + 1; // -1, 0, 1 -> 0, 1, 2
                        q *= 3;
                        q += xi;
                    }
                    // ceiling division (243 == pow(3, 5))
                    q = (q * 256 + (243 - 1)) / 243;
                    dst[off + TQ1_0_QS + j + m] = (byte) q;
                }
                p += 5 * 32;
            }
            // along 16 bytes
            for (int j = bs; j < TQ1_0_QS_BYTES; j += 16) {
                for (int m = 0; m < 16; ++m) {
                    int q = 0;
                    for (int n = 0; n < 5; ++n) {
                        int xi = lround(x[p + m + n * 16] * id) + 1; // -1, 0, 1 -> 0, 1, 2
                        q *= 3;
                        q += xi;
                    }
                    // ceiling division (243 == pow(3, 5))
                    q = (q * 256 + (243 - 1)) / 243;
                    dst[off + TQ1_0_QS + j + m] = (byte) q;
                }
                p += 5 * 16;
            }
            // 4 elements per byte
            for (int j = 0; j < TQ1_0_QH_BYTES; ++j) {
                int q = 0;
                for (int m = 0; m < 4; ++m) {
                    // -1, 0, 1 -> 0, 1, 2
                    int xi = lround(x[p + j + m * TQ1_0_QH_BYTES] * id) + 1;
                    q *= 3;
                    q += xi;
                }
                // shift the first value to the most significant trit
                q *= 3;
                // ceiling division (243 == pow(3, 5))
                q = (q * 256 + (243 - 1)) / 243;
                dst[off + TQ1_0_QH + j] = (byte) q;
            }
            xOff += QK_K;
        }
    }

    /** One TQ2_0 super-block: 256 ternary values as 2-bit codes ({@code quantize_row_tq2_0_ref}). */
    private static void tq2_0Row(float[] x, int xOff, byte[] dst, int dstOff, int nPerRow) {
        checkBlocks(nPerRow, QK_K, GGMLType.TQ2_0);
        for (int i = 0; i < nPerRow / QK_K; i++) {
            int off = dstOff + i * TQ2_0_BYTES;
            float amax = 0.0f; // absolute max
            for (int j = 0; j < QK_K; j++) {
                float v = x[xOff + j];
                amax = Math.max(amax, Math.abs(v));
            }
            float d = amax;
            float id = d != 0 ? 1.0f / d : 0.0f;
            KQuantEncoders.putHalf(dst, off + TQ2_0_D, d);

            for (int j = 0; j < TQ2_0_D; j += 32) {
                for (int m = 0; m < 32; ++m) {
                    int q = 0;
                    for (int n = 0; n < 4; ++n) {
                        // -1, 0, 1 -> 0, 1, 2
                        int xi = lround(x[xOff + m + n * 32] * id) + 1;
                        q += (xi & 3) << (2 * n);
                    }
                    dst[off + TQ2_0_QS + j + m] = (byte) q;
                }
                xOff += 4 * 32;
            }
        }
    }

    /** One Q8_K super-block: the float scale and 8-bit levels with the per-16 sums ({@code bsums}). */
    private static void q8KRow(float[] x, int xOff, byte[] dst, int dstOff, int nPerRow) {
        checkBlocks(nPerRow, QK_K, GGMLType.Q8_K);
        ByteBuffer b = ByteBuffer.wrap(dst).order(java.nio.ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < nPerRow / QK_K; i++) {
            int off = dstOff + i * Q8_K_BYTES;
            float max = 0;
            float amax = 0;
            for (int j = 0; j < QK_K; ++j) {
                float ax = Math.abs(x[xOff + i * QK_K + j]);
                if (ax > amax) {
                    amax = ax;
                    max = x[xOff + i * QK_K + j];
                }
            }
            if (amax == 0) {
                b.putFloat(off + Q8_K_D, 0);
                for (int j = 0; j < QK_K; j++) {
                    dst[off + Q8_K_QS + j] = 0;
                }
                for (int j = 0; j < QK_K / 16; j++) {
                    b.putShort(off + Q8_K_BSUMS + 2 * j, (short) 0);
                }
                continue;
            }
            // We need this change for IQ2_XXS, else the AVX implementation becomes very awkward
            float iscale = -127.f / max;
            for (int j = 0; j < QK_K; ++j) {
                int v = QuantizeSearch.nearestInt(iscale * x[xOff + i * QK_K + j]);
                dst[off + Q8_K_QS + j] = (byte) Math.min(127, v);
            }
            for (int j = 0; j < QK_K / 16; ++j) {
                int sum = 0;
                for (int ii = 0; ii < 16; ++ii) {
                    sum += dst[off + Q8_K_QS + j * 16 + ii];
                }
                b.putShort(off + Q8_K_BSUMS + 2 * j, (short) sum);
            }
            b.putFloat(off + Q8_K_D, 1 / iscale);
        }
    }

    /** C {@code lroundf}: round to nearest, halves away from zero. */
    private static int lround(float v) {
        double a = Math.floor(Math.abs((double) v) + 0.5);
        return (int) (v < 0 ? -a : a);
    }
}

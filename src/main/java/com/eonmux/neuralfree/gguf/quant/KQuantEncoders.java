package com.eonmux.neuralfree.gguf.quant;

import com.eonmux.neuralfree.gguf.GGMLType;

import java.nio.ByteBuffer;

import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q2_K_BYTES;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q2_K_D;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q2_K_DMIN;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q2_K_QS;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q2_K_SCALES;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q3_K_BYTES;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q3_K_D;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q3_K_HMASK;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q3_K_QS;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q3_K_SCALES;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q4_K_BYTES;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q4_K_D;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q4_K_DMIN;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q4_K_QS;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q4_K_SCALES;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q5_K_BYTES;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q5_K_D;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q5_K_DMIN;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q5_K_QH;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q5_K_QS;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q5_K_SCALES;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q6_K_BYTES;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q6_K_D;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q6_K_QH;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q6_K_QL;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q6_K_SCALES;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.QK_K;
import static com.eonmux.neuralfree.gguf.quant.FpBlockDecoders.fp16ToFp32;
import static com.eonmux.neuralfree.gguf.quant.FpBlockDecoders.fp32ToFp16;

/**
 * Row encoders of the K-quant super-blocks Q2_K, Q3_K, Q4_K, Q5_K and Q6_K: bit-faithful ports of the
 * {@code quantize_row_*_ref} and {@code quantize_row_*_impl} functions of llama.cpp ggml-quants.c
 * ({@code quantize_q2_K} :1211, {@code quantize_q3_K} :1447, {@code quantize_q4_K} :1632,
 * {@code quantize_q5_K} :1856, {@code quantize_q6_K} :2076). Without importance-matrix weights the
 * reference path runs; with them the weighted path runs, whose per-group weights combine the matrix with
 * the row's energy the way llama.cpp feeds an imatrix in ({@code qw * sqrt(sigma2 + x*x)}, sigma2 = twice
 * the row mean square; Q2_K uses the plain mean square). The searches live in {@link QuantizeSearch}.
 * <p>
 * Rows are one or more whole {@value QK_K}-value super-blocks; {@code weights}, when not null, holds one
 * importance value per input value of the row. Package seam: {@link RowQuantizers}.
 */
public final class KQuantEncoders {

    private KQuantEncoders() {
    }

    /** True when {@link #quantizeRow} quantizes {@code type}. */
    public static boolean supports(GGMLType type) {
        switch (type) {
            case Q2_K:
            case Q3_K:
            case Q4_K:
            case Q5_K:
            case Q6_K:
                return true;
            default:
                return false;
        }
    }

    /**
     * Quantizes one row of {@code nPerRow} values at {@code xOff} into {@code dst} at {@code dstOff},
     * {@code nPerRow / 256} super-blocks; {@code weights} (one value per input value, or null) selects the
     * weighted path of the type.
     *
     * @throws IllegalArgumentException when {@code nPerRow} is not a multiple of {@value QK_K} or the type
     *                                  is not a K-quant
     */
    public static void quantizeRow(GGMLType type, float[] x, int xOff, byte[] dst, int dstOff, int nPerRow,
                                   float[] weights) {
        quantizeRow(type, x, xOff, dst, dstOff, nPerRow, weights, 0);
    }

    /**
     * {@link #quantizeRow} of a row whose importance slice starts at {@code weightsOff} of the shared
     * {@code weights} array (one expert matrix after the other, as the tool passes them).
     *
     * @throws IllegalArgumentException when {@code nPerRow} is not a multiple of {@value QK_K} or the type
     *                                  is not a K-quant
     */
    public static void quantizeRow(GGMLType type, float[] x, int xOff, byte[] dst, int dstOff, int nPerRow,
                                   float[] weights, int weightsOff) {
        if (nPerRow < 0 || nPerRow % QK_K != 0) {
            throw new IllegalArgumentException("a row of " + nPerRow + " values is not a whole number of "
                    + QK_K + "-value K-quant super-blocks");
        }
        int nb = nPerRow / QK_K;
        for (int i = 0; i < nb; i++) {
            int xb = xOff + i * QK_K;
            int yb = dstOff + i * rowBytes(type);
            int wb = weightsOff + i * QK_K;
            switch (type) {
                case Q2_K:
                    if (weights == null) {
                        q2KRef(x, xb, dst, yb);
                    } else {
                        q2KImpl(x, xb, dst, yb, weights, wb);
                    }
                    break;
                case Q3_K:
                    if (weights == null) {
                        q3KRef(x, xb, dst, yb);
                    } else {
                        q3KImpl(x, xb, dst, yb, weights, wb);
                    }
                    break;
                case Q4_K:
                    if (weights == null) {
                        q4KRef(x, xb, dst, yb);
                    } else {
                        q4KImpl(x, xb, dst, yb, weights, wb);
                    }
                    break;
                case Q5_K:
                    if (weights == null) {
                        q5KRef(x, xb, dst, yb);
                    } else {
                        q5KImpl(x, xb, dst, yb, weights, wb);
                    }
                    break;
                case Q6_K:
                    if (weights == null) {
                        q6KRef(x, xb, dst, yb);
                    } else {
                        q6KImpl(x, xb, dst, yb, weights, wb);
                    }
                    break;
                default:
                    throw new IllegalArgumentException(type + " is not a K-quant row type");
            }
        }
    }

    /** The encoded bytes of one super-block row of {@code type}. */
    private static int rowBytes(GGMLType type) {
        switch (type) {
            case Q2_K: return Q2_K_BYTES;
            case Q3_K: return Q3_K_BYTES;
            case Q4_K: return Q4_K_BYTES;
            case Q5_K: return Q5_K_BYTES;
            case Q6_K: return Q6_K_BYTES;
            default: throw new IllegalArgumentException(type + " is not a K-quant row type");
        }
    }

    static void putHalf(byte[] y, int off, float v) {
        int h = fp32ToFp16(v);
        y[off] = (byte) h;
        y[off + 1] = (byte) (h >>> 8);
    }

    static float halfAt(byte[] y, int off) {
        return fp16ToFp32((y[off] & 0xFF) | (y[off + 1] & 0xFF) << 8);
    }

    // ------------------------------------------------------------------ Q2_K

    /** One Q2_K super-block without weights ({@code quantize_row_q2_K_ref}). */
    private static void q2KRef(float[] x, int xOff, byte[] y, int yOff) {
        byte[] L = new byte[QK_K];
        byte[] Laux = new byte[16];
        float[] weights = new float[16];
        float[] mins = new float[QK_K / 16];
        float[] scales = new float[QK_K / 16];
        float[] minOut = new float[1];

        final float q4scale = 15.f;

        float maxScale = 0; // as we are deducting the min, scales are always positive
        float maxMin = 0;
        for (int j = 0; j < QK_K / 16; ++j) {
            for (int l = 0; l < 16; l++) {
                weights[l] = Math.abs(x[xOff + 16 * j + l]);
            }
            scales[j] = QuantizeSearch.makeQkx2Quants(16, 3, x, xOff + 16 * j, weights, L, 16 * j, minOut,
                    Laux, 0, -0.5f, 0.1f, 15, true);
            mins[j] = minOut[0];
            float scale = scales[j];
            if (scale > maxScale) {
                maxScale = scale;
            }
            float min = mins[j];
            if (min > maxMin) {
                maxMin = min;
            }
        }

        if (maxScale > 0) {
            float iscale = q4scale / maxScale;
            for (int j = 0; j < QK_K / 16; ++j) {
                int l = QuantizeSearch.nearestInt(iscale * scales[j]);
                y[yOff + Q2_K_SCALES + j] = (byte) l;
            }
            putHalf(y, yOff + Q2_K_D, maxScale / q4scale);
        } else {
            for (int j = 0; j < QK_K / 16; ++j) {
                y[yOff + Q2_K_SCALES + j] = 0;
            }
            putHalf(y, yOff + Q2_K_D, 0.f);
        }
        if (maxMin > 0) {
            float iscale = q4scale / maxMin;
            for (int j = 0; j < QK_K / 16; ++j) {
                int l = QuantizeSearch.nearestInt(iscale * mins[j]);
                y[yOff + Q2_K_SCALES + j] |= (l << 4);
            }
            putHalf(y, yOff + Q2_K_DMIN, maxMin / q4scale);
        } else {
            putHalf(y, yOff + Q2_K_DMIN, 0.f);
        }
        for (int j = 0; j < QK_K / 16; ++j) {
            float d = halfAt(y, yOff + Q2_K_D) * (y[yOff + Q2_K_SCALES + j] & 0xF);
            if (d == 0) {
                continue;
            }
            float dm = halfAt(y, yOff + Q2_K_DMIN) * ((y[yOff + Q2_K_SCALES + j] & 0xFF) >>> 4);
            for (int ii = 0; ii < 16; ++ii) {
                int l = QuantizeSearch.nearestInt((x[xOff + 16 * j + ii] + dm) / d);
                l = Math.max(0, Math.min(3, l));
                L[16 * j + ii] = (byte) l;
            }
        }

        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) {
                y[yOff + Q2_K_QS + j / 4 + l] = (byte) (L[j + l] | (L[j + l + 32] << 2) | (L[j + l + 64] << 4)
                        | (L[j + l + 96] << 6));
            }
        }
    }

    /** One Q2_K super-block with importance weights ({@code quantize_row_q2_K_impl}). */
    private static void q2KImpl(float[] x, int xOff, byte[] y, int yOff, float[] qw, int qwOff) {
        // requantize stays true, as in the reference
        byte[] L = new byte[QK_K];
        byte[] Laux = new byte[16];
        float[] mins = new float[QK_K / 16];
        float[] scales = new float[QK_K / 16];
        float[] sw = new float[QK_K / 16];
        float[] weight = new float[16];
        byte[] Ls = new byte[QK_K / 16];
        byte[] Lm = new byte[QK_K / 16];
        float[] minOut = new float[1];

        float sumx2 = 0;
        for (int j = 0; j < QK_K; ++j) {
            sumx2 += x[xOff + j] * x[xOff + j];
        }
        float sigma2 = sumx2 / QK_K;
        for (int j = 0; j < QK_K / 16; ++j) {
            int base = qwOff + 16 * j;
            for (int l = 0; l < 16; ++l) {
                weight[l] = qw[base + l] * (float) Math.sqrt(sigma2 + x[xOff + 16 * j + l] * x[xOff + 16 * j + l]);
            }
            for (int l = 0; l < QK_K / 16; ++l) {
                sw[j] += weight[l];
            }
            scales[j] = QuantizeSearch.makeQkx3Quants(16, 3, x, xOff + 16 * j, weight, L, 16 * j, minOut,
                    Laux, 0, -0.9f, 0.05f, 36, false);
            mins[j] = minOut[0];
        }

        float dm = QuantizeSearch.makeQpQuants(QK_K / 16, 15, scales, 0, Ls, 0, sw, 0);
        float mm = QuantizeSearch.makeQpQuants(QK_K / 16, 15, mins, 0, Lm, 0, sw, 0);

        putHalf(y, yOff + Q2_K_D, dm);
        putHalf(y, yOff + Q2_K_DMIN, mm);
        dm = halfAt(y, yOff + Q2_K_D);
        mm = halfAt(y, yOff + Q2_K_DMIN);

        for (int j = 0; j < QK_K / 16; ++j) {
            y[yOff + Q2_K_SCALES + j] = (byte) ((Ls[j] & 0xFF) | ((Lm[j] & 0xFF) << 4));
        }

        for (int j = 0; j < QK_K / 16; ++j) {
            float d = dm * (y[yOff + Q2_K_SCALES + j] & 0xF);
            if (d == 0) {
                continue;
            }
            float m = mm * ((y[yOff + Q2_K_SCALES + j] & 0xFF) >>> 4);
            for (int ii = 0; ii < 16; ++ii) {
                int l = QuantizeSearch.nearestInt((x[xOff + 16 * j + ii] + m) / d);
                l = Math.max(0, Math.min(3, l));
                L[16 * j + ii] = (byte) l;
            }
        }

        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) {
                y[yOff + Q2_K_QS + j / 4 + l] = (byte) (L[j + l] | (L[j + l + 32] << 2) | (L[j + l + 64] << 4)
                        | (L[j + l + 96] << 6));
            }
        }
    }

    // ------------------------------------------------------------------ Q3_K

    /** One Q3_K super-block without weights ({@code quantize_row_q3_K_ref}). */
    private static void q3KRef(float[] x, int xOff, byte[] y, int yOff) {
        byte[] L = new byte[QK_K];
        float[] scales = new float[QK_K / 16];

        float maxScale = 0;
        float amax = 0;
        for (int j = 0; j < QK_K / 16; ++j) {
            scales[j] = QuantizeSearch.makeQ3Quants(16, 4, x, xOff + 16 * j, L, 16 * j, true);
            float scale = Math.abs(scales[j]);
            if (scale > amax) {
                amax = scale;
                maxScale = scales[j];
            }
        }

        for (int j = 0; j < 12; j++) {
            y[yOff + Q3_K_SCALES + j] = 0;
        }
        if (maxScale != 0) {
            float iscale = -32.f / maxScale;
            for (int j = 0; j < QK_K / 16; ++j) {
                int l = QuantizeSearch.nearestInt(iscale * scales[j]);
                l = Math.max(-32, Math.min(31, l)) + 32;
                if (j < 8) {
                    y[yOff + Q3_K_SCALES + j] = (byte) (l & 0xF);
                } else {
                    y[yOff + Q3_K_SCALES + j - 8] |= ((l & 0xF) << 4);
                }
                l >>= 4;
                y[yOff + Q3_K_SCALES + j % 4 + 8] |= (byte) (l << (2 * (j / 4)));
            }
            putHalf(y, yOff + Q3_K_D, 1 / iscale);
        } else {
            putHalf(y, yOff + Q3_K_D, 0.f);
        }

        for (int j = 0; j < QK_K / 16; ++j) {
            int sc = j < 8 ? y[yOff + Q3_K_SCALES + j] & 0xF : (y[yOff + Q3_K_SCALES + j - 8] & 0xFF) >>> 4;
            sc = ((sc | (((y[yOff + Q3_K_SCALES + 8 + j % 4] >>> (2 * (j / 4))) & 3) << 4)) & 0xFF) - 32;
            float d = halfAt(y, yOff + Q3_K_D) * sc;
            if (d == 0) {
                continue;
            }
            for (int ii = 0; ii < 16; ++ii) {
                int l = QuantizeSearch.nearestInt(x[xOff + 16 * j + ii] / d);
                l = Math.max(-4, Math.min(3, l));
                L[16 * j + ii] = (byte) (l + 4);
            }
        }

        for (int j = 0; j < QK_K / 8; j++) {
            y[yOff + Q3_K_HMASK + j] = 0;
        }
        // We put the high-bit for the 1st 8 quants into bit 0, the next 8 into bit 1, etc.
        int m = 0;
        int hm = 1;
        for (int j = 0; j < QK_K; ++j) {
            if (L[j] > 3) {
                y[yOff + Q3_K_HMASK + m] |= (byte) hm;
                L[j] -= 4;
            }
            if (++m == QK_K / 8) {
                m = 0;
                hm <<= 1;
            }
        }
        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) {
                y[yOff + Q3_K_QS + j / 4 + l] = (byte) (L[j + l] | (L[j + l + 32] << 2) | (L[j + l + 64] << 4)
                        | (L[j + l + 96] << 6));
            }
        }
    }

    /** One Q3_K super-block, weighted or not ({@code quantize_row_q3_K_impl}). */
    private static void q3KImpl(float[] x, int xOff, byte[] y, int yOff, float[] qw, int qwOff) {
        byte[] L = new byte[QK_K];
        float[] scales = new float[QK_K / 16];
        float[] weight = new float[16];
        float[] sw = new float[QK_K / 16];
        byte[] Ls = new byte[QK_K / 16];

        float sumx2 = 0;
        for (int j = 0; j < QK_K; ++j) {
            sumx2 += x[xOff + j] * x[xOff + j];
        }
        float sigma2 = 2 * sumx2 / QK_K;

        for (int j = 0; j < QK_K / 16; ++j) {
            if (qw != null) {
                int base = qwOff + 16 * j;
                for (int l = 0; l < 16; ++l) {
                    weight[l] = qw[base + l] * (float) Math.sqrt(sigma2 + x[xOff + 16 * j + l] * x[xOff + 16 * j + l]);
                }
            } else {
                for (int l = 0; l < 16; ++l) {
                    weight[l] = x[xOff + 16 * j + l] * x[xOff + 16 * j + l];
                }
            }
            float sumw = 0;
            for (int l = 0; l < 16; ++l) {
                sumw += weight[l];
            }
            sw[j] = sumw;

            scales[j] = QuantizeSearch.makeQxQuants(16, 4, x, xOff + 16 * j, L, 16 * j, 1, weight, 0);
        }

        for (int j = 0; j < 12; j++) {
            y[yOff + Q3_K_SCALES + j] = 0;
        }

        float dBlock = QuantizeSearch.makeQxQuants(QK_K / 16, 32, scales, 0, Ls, 0, 1, sw, 0);
        for (int j = 0; j < QK_K / 16; ++j) {
            int l = Ls[j] & 0xFF;
            if (j < 8) {
                y[yOff + Q3_K_SCALES + j] = (byte) (l & 0xF);
            } else {
                y[yOff + Q3_K_SCALES + j - 8] |= ((l & 0xF) << 4);
            }
            l >>= 4;
            y[yOff + Q3_K_SCALES + j % 4 + 8] |= (byte) (l << (2 * (j / 4)));
        }
        putHalf(y, yOff + Q3_K_D, dBlock);

        for (int j = 0; j < QK_K / 16; ++j) {
            int sc = j < 8 ? y[yOff + Q3_K_SCALES + j] & 0xF : (y[yOff + Q3_K_SCALES + j - 8] & 0xFF) >>> 4;
            sc = ((sc | (((y[yOff + Q3_K_SCALES + 8 + j % 4] >>> (2 * (j / 4))) & 3) << 4)) & 0xFF) - 32;
            float d = halfAt(y, yOff + Q3_K_D) * sc;
            if (d == 0) {
                continue;
            }
            for (int ii = 0; ii < 16; ++ii) {
                int l = QuantizeSearch.nearestInt(x[xOff + 16 * j + ii] / d);
                l = Math.max(-4, Math.min(3, l));
                L[16 * j + ii] = (byte) (l + 4);
            }
        }

        for (int j = 0; j < QK_K / 8; j++) {
            y[yOff + Q3_K_HMASK + j] = 0;
        }
        int m = 0;
        int hm = 1;
        for (int j = 0; j < QK_K; ++j) {
            if (L[j] > 3) {
                y[yOff + Q3_K_HMASK + m] |= (byte) hm;
                L[j] -= 4;
            }
            if (++m == QK_K / 8) {
                m = 0;
                hm <<= 1;
            }
        }
        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) {
                y[yOff + Q3_K_QS + j / 4 + l] = (byte) (L[j + l] | (L[j + l + 32] << 2) | (L[j + l + 64] << 4)
                        | (L[j + l + 96] << 6));
            }
        }
    }

    // ------------------------------------------------------------------ Q4_K / Q5_K shared scale packing

    /** Packs the 6-bit scale/min levels of Q4_K / Q5_K ({@code y.scales} of the reference). */
    private static void packKscales(byte[] y, int scalesOff, byte[] Ls, byte[] Lm, int n) {
        for (int j = 0; j < n; ++j) {
            int ls = Ls[j] & 0xFF;
            int lm = Lm[j] & 0xFF;
            if (j < 4) {
                y[scalesOff + j] = (byte) ls;
                y[scalesOff + j + 4] = (byte) lm;
            } else {
                y[scalesOff + j + 4] = (byte) ((ls & 0xF) | ((lm & 0xF) << 4));
                y[scalesOff + j - 4] |= (byte) ((ls >> 4) << 6);
                y[scalesOff + j] |= (byte) ((lm >> 4) << 6);
            }
        }
    }

    // ------------------------------------------------------------------ Q4_K

    /** One Q4_K super-block without weights ({@code quantize_row_q4_K_ref}). */
    private static void q4KRef(float[] x, int xOff, byte[] y, int yOff) {
        byte[] L = new byte[QK_K];
        byte[] Laux = new byte[32];
        float[] weights = new float[32];
        float[] mins = new float[QK_K / 32];
        float[] scales = new float[QK_K / 32];
        float[] minOut = new float[1];

        float maxScale = 0; // as we are deducting the min, scales are always positive
        float maxMin = 0;
        for (int j = 0; j < QK_K / 32; ++j) {
            float sumX2 = 0;
            for (int l = 0; l < 32; ++l) {
                sumX2 += x[xOff + 32 * j + l] * x[xOff + 32 * j + l];
            }
            float avX = (float) Math.sqrt(sumX2 / 32);
            for (int l = 0; l < 32; ++l) {
                weights[l] = avX + Math.abs(x[xOff + 32 * j + l]);
            }
            scales[j] = QuantizeSearch.makeQkx2Quants(32, 15, x, xOff + 32 * j, weights, L, 32 * j, minOut,
                    Laux, 0, -1.f, 0.1f, 20, false);
            mins[j] = minOut[0];
            float scale = scales[j];
            if (scale > maxScale) {
                maxScale = scale;
            }
            float min = mins[j];
            if (min > maxMin) {
                maxMin = min;
            }
        }

        float invScale = maxScale > 0 ? 63.f / maxScale : 0.f;
        float invMin = maxMin > 0 ? 63.f / maxMin : 0.f;
        byte[] Ls = new byte[QK_K / 32];
        byte[] Lm = new byte[QK_K / 32];
        for (int j = 0; j < QK_K / 32; ++j) {
            int ls = QuantizeSearch.nearestInt(invScale * scales[j]);
            int lm = QuantizeSearch.nearestInt(invMin * mins[j]);
            ls = Math.min(63, ls);
            lm = Math.min(63, lm);
            Ls[j] = (byte) ls;
            Lm[j] = (byte) lm;
        }
        packKscales(y, yOff + Q4_K_SCALES, Ls, Lm, QK_K / 32);
        putHalf(y, yOff + Q4_K_D, maxScale / 63.f);
        putHalf(y, yOff + Q4_K_DMIN, maxMin / 63.f);

        requantizeK4(x, xOff, y, yOff, L, 15);

        int q = yOff + Q4_K_QS;
        for (int j = 0; j < QK_K; j += 64) {
            for (int l = 0; l < 32; ++l) {
                y[q + l] = (byte) (L[j + l] | (L[j + l + 32] << 4));
            }
            q += 32;
        }
    }

    /** One Q4_K super-block with importance weights ({@code quantize_row_q4_K_impl}). */
    private static void q4KImpl(float[] x, int xOff, byte[] y, int yOff, float[] qw, int qwOff) {
        byte[] L = new byte[QK_K];
        byte[] Laux = new byte[32];
        byte[] Ls = new byte[QK_K / 32];
        byte[] Lm = new byte[QK_K / 32];
        float[] weights = new float[32];
        float[] sw = new float[QK_K / 32];
        float[] mins = new float[QK_K / 32];
        float[] scales = new float[QK_K / 32];
        float[] minOut = new float[1];

        float sumX2 = 0;
        for (int l = 0; l < QK_K; ++l) {
            sumX2 += x[xOff + l] * x[xOff + l];
        }
        float sigma2 = 2 * sumX2 / QK_K;
        float avX = (float) Math.sqrt(sigma2);

        for (int j = 0; j < QK_K / 32; ++j) {
            int base = qwOff + 32 * j;
            if (qw != null) {
                for (int l = 0; l < 32; ++l) {
                    weights[l] = qw[base + l] * (float) Math.sqrt(sigma2 + x[xOff + 32 * j + l] * x[xOff + 32 * j + l]);
                }
            } else {
                for (int l = 0; l < 32; ++l) {
                    weights[l] = avX + Math.abs(x[xOff + 32 * j + l]);
                }
            }
            float sumw = 0;
            for (int l = 0; l < 32; ++l) {
                sumw += weights[l];
            }
            sw[j] = sumw;
            scales[j] = QuantizeSearch.makeQkx3Quants(32, 15, x, xOff + 32 * j, weights, L, 32 * j, minOut,
                    Laux, 0, -0.9f, 0.05f, 36, false);
            mins[j] = minOut[0];
        }

        float dBlock = QuantizeSearch.makeQpQuants(QK_K / 32, 63, scales, 0, Ls, 0, sw, 0);
        float mBlock = QuantizeSearch.makeQpQuants(QK_K / 32, 63, mins, 0, Lm, 0, sw, 0);
        packKscales(y, yOff + Q4_K_SCALES, Ls, Lm, QK_K / 32);
        putHalf(y, yOff + Q4_K_D, dBlock);
        putHalf(y, yOff + Q4_K_DMIN, mBlock);

        requantizeK4(x, xOff, y, yOff, L, 15);

        int q = yOff + Q4_K_QS;
        for (int j = 0; j < QK_K; j += 64) {
            for (int l = 0; l < 32; ++l) {
                y[q + l] = (byte) (L[j + l] | (L[j + l + 32] << 4));
            }
            q += 32;
        }
    }

    /** The Q4_K / Q5_K requantize pass: levels from the packed scales ({@code max} is 15 or 31). */
    private static void requantizeK4(float[] x, int xOff, byte[] y, int yOff, byte[] L, int max) {
        ByteBuffer bb = ByteBuffer.wrap(y);
        for (int j = 0; j < QK_K / 32; ++j) {
            int sm = KQuantBlockDecoders.scaleMinK4(j, bb, yOff + Q4_K_SCALES);
            int sc = sm & 0xFF;
            int m = sm >>> 8;
            float d = halfAt(y, yOff + Q4_K_D) * sc;
            if (d == 0) {
                continue;
            }
            float dm = halfAt(y, yOff + Q4_K_DMIN) * m;
            for (int ii = 0; ii < 32; ++ii) {
                int l = QuantizeSearch.nearestInt((x[xOff + 32 * j + ii] + dm) / d);
                l = Math.max(0, Math.min(max, l));
                L[32 * j + ii] = (byte) l;
            }
        }
    }

    // ------------------------------------------------------------------ Q5_K

    /** One Q5_K super-block without weights ({@code quantize_row_q5_K_ref}). */
    private static void q5KRef(float[] x, int xOff, byte[] y, int yOff) {
        byte[] L = new byte[QK_K];
        float[] mins = new float[QK_K / 32];
        float[] scales = new float[QK_K / 32];
        float[] weights = new float[32];
        byte[] Laux = new byte[32];
        float[] minOut = new float[1];

        float maxScale = 0; // as we are deducting the min, scales are always positive
        float maxMin = 0;
        for (int j = 0; j < QK_K / 32; ++j) {
            float sumX2 = 0;
            for (int l = 0; l < 32; ++l) {
                sumX2 += x[xOff + 32 * j + l] * x[xOff + 32 * j + l];
            }
            float avX = (float) Math.sqrt(sumX2 / 32);
            for (int l = 0; l < 32; ++l) {
                weights[l] = avX + Math.abs(x[xOff + 32 * j + l]);
            }
            scales[j] = QuantizeSearch.makeQkx2Quants(32, 31, x, xOff + 32 * j, weights, L, 32 * j, minOut,
                    Laux, 0, -0.5f, 0.1f, 15, false);
            mins[j] = minOut[0];
            float scale = scales[j];
            if (scale > maxScale) {
                maxScale = scale;
            }
            float min = mins[j];
            if (min > maxMin) {
                maxMin = min;
            }
        }

        float invScale = maxScale > 0 ? 63.f / maxScale : 0.f;
        float invMin = maxMin > 0 ? 63.f / maxMin : 0.f;
        byte[] Ls = new byte[QK_K / 32];
        byte[] Lm = new byte[QK_K / 32];
        for (int j = 0; j < QK_K / 32; ++j) {
            int ls = QuantizeSearch.nearestInt(invScale * scales[j]);
            int lm = QuantizeSearch.nearestInt(invMin * mins[j]);
            ls = Math.min(63, ls);
            lm = Math.min(63, lm);
            Ls[j] = (byte) ls;
            Lm[j] = (byte) lm;
        }
        packKscales(y, yOff + Q5_K_SCALES, Ls, Lm, QK_K / 32);
        putHalf(y, yOff + Q5_K_D, maxScale / 63.f);
        putHalf(y, yOff + Q5_K_DMIN, maxMin / 63.f);

        requantizeK4(x, xOff, y, yOff, L, 31);

        packQ5Levels(y, yOff, L);
    }

    /** One Q5_K super-block with importance weights ({@code quantize_row_q5_K_impl}). */
    private static void q5KImpl(float[] x, int xOff, byte[] y, int yOff, float[] qw, int qwOff) {
        byte[] L = new byte[QK_K];
        byte[] Laux = new byte[32];
        byte[] Ls = new byte[QK_K / 32];
        byte[] Lm = new byte[QK_K / 32];
        float[] mins = new float[QK_K / 32];
        float[] scales = new float[QK_K / 32];
        float[] sw = new float[QK_K / 32];
        float[] weights = new float[32];
        float[] minOut = new float[1];

        float sumX2 = 0;
        for (int l = 0; l < QK_K; ++l) {
            sumX2 += x[xOff + l] * x[xOff + l];
        }
        float sigma2 = 2 * sumX2 / QK_K;
        float avX = (float) Math.sqrt(sigma2);

        for (int j = 0; j < QK_K / 32; ++j) {
            int base = qwOff + 32 * j;
            if (qw != null) {
                for (int l = 0; l < 32; ++l) {
                    weights[l] = qw[base + l] * (float) Math.sqrt(sigma2 + x[xOff + 32 * j + l] * x[xOff + 32 * j + l]);
                }
            } else {
                for (int l = 0; l < 32; ++l) {
                    weights[l] = avX + Math.abs(x[xOff + 32 * j + l]);
                }
            }
            float sumw = 0;
            for (int l = 0; l < 32; ++l) {
                sumw += weights[l];
            }
            sw[j] = sumw;

            scales[j] = QuantizeSearch.makeQkx3Quants(32, 31, x, xOff + 32 * j, weights, L, 32 * j, minOut,
                    Laux, 0, -0.9f, 0.05f, 36, false);
            mins[j] = minOut[0];
        }

        float dBlock = QuantizeSearch.makeQpQuants(QK_K / 32, 63, scales, 0, Ls, 0, sw, 0);
        float mBlock = QuantizeSearch.makeQpQuants(QK_K / 32, 63, mins, 0, Lm, 0, sw, 0);

        for (int j = 0; j < QK_K / 32; ++j) {
            int ls = Ls[j] & 0xFF;
            int lm = Lm[j] & 0xFF;
            ls = Math.min(63, ls);
            lm = Math.min(63, lm);
            Ls[j] = (byte) ls;
            Lm[j] = (byte) lm;
        }
        packKscales(y, yOff + Q5_K_SCALES, Ls, Lm, QK_K / 32);
        putHalf(y, yOff + Q5_K_D, dBlock);
        putHalf(y, yOff + Q5_K_DMIN, mBlock);

        requantizeK4(x, xOff, y, yOff, L, 31);

        packQ5Levels(y, yOff, L);
    }

    /** The Q5_K qh/ql packing of the levels ({@code l - 16} with the 5th bit in qh). */
    private static void packQ5Levels(byte[] y, int yOff, byte[] L) {
        int qh = yOff + Q5_K_QH;
        int ql = yOff + Q5_K_QS;
        for (int j = 0; j < QK_K / 8; j++) {
            y[qh + j] = 0;
        }
        int m1 = 1;
        int m2 = 2;
        for (int n = 0; n < QK_K; n += 64) {
            for (int j = 0; j < 32; ++j) {
                int l1 = L[n + j];
                if (l1 > 15) {
                    l1 -= 16;
                    y[qh + j] |= (byte) m1;
                }
                int l2 = L[n + j + 32];
                if (l2 > 15) {
                    l2 -= 16;
                    y[qh + j] |= (byte) m2;
                }
                y[ql + j] = (byte) (l1 | (l2 << 4));
            }
            m1 <<= 2;
            m2 <<= 2;
            ql += 32;
        }
    }

    // ------------------------------------------------------------------ Q6_K

    /** One Q6_K super-block, weighted or not ({@code quantize_row_q6_K_ref} / {@code _impl}). */
    private static void q6KRow(float[] x, int xOff, byte[] y, int yOff, float[] qw, int qwOff) {
        byte[] L = new byte[QK_K];
        float[] scales = new float[QK_K / 16];

        float maxScale = 0;
        float maxAbsScale = 0;

        for (int ib = 0; ib < QK_K / 16; ++ib) {
            float scale;
            if (qw != null) {
                int base = qwOff + 16 * ib;
                scale = QuantizeSearch.makeQxQuants(16, 32, x, xOff + 16 * ib, L, 16 * ib, 1, qw, base);
            } else {
                scale = QuantizeSearch.makeQxQuants(16, 32, x, xOff + 16 * ib, L, 16 * ib, 1, null, 0);
            }
            scales[ib] = scale;

            float absScale = Math.abs(scale);
            if (absScale > maxAbsScale) {
                maxAbsScale = absScale;
                maxScale = scale;
            }
        }

        if (maxAbsScale < QuantizeSearch.GROUP_MAX_EPS) {
            for (int j = 0; j < Q6_K_BYTES; j++) {
                y[yOff + j] = 0;
            }
            putHalf(y, yOff + Q6_K_D, 0.f);
            return;
        }

        float iscale = -128.f / maxScale;
        putHalf(y, yOff + Q6_K_D, 1 / iscale);
        for (int ib = 0; ib < QK_K / 16; ++ib) {
            y[yOff + Q6_K_SCALES + ib] = (byte) Math.min(127, QuantizeSearch.nearestInt(iscale * scales[ib]));
        }

        for (int j = 0; j < QK_K / 16; ++j) {
            float d = halfAt(y, yOff + Q6_K_D) * y[yOff + Q6_K_SCALES + j];
            if (d == 0) {
                continue;
            }
            for (int ii = 0; ii < 16; ++ii) {
                int l = QuantizeSearch.nearestInt(x[xOff + 16 * j + ii] / d);
                l = Math.max(-32, Math.min(31, l));
                L[16 * j + ii] = (byte) (l + 32);
            }
        }

        int ql = yOff + Q6_K_QL;
        int qh = yOff + Q6_K_QH;
        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) {
                int q1 = L[j + l] & 0xF;
                int q2 = L[j + l + 32] & 0xF;
                int q3 = L[j + l + 64] & 0xF;
                int q4 = L[j + l + 96] & 0xF;
                y[ql + l] = (byte) (q1 | (q3 << 4));
                y[ql + l + 32] = (byte) (q2 | (q4 << 4));
                y[qh + l] = (byte) ((L[j + l] >> 4) | ((L[j + l + 32] >> 4) << 2) | ((L[j + l + 64] >> 4) << 4)
                        | ((L[j + l + 96] >> 4) << 6));
            }
            ql += 64;
            qh += 32;
        }
    }

    /** One Q6_K super-block without weights ({@code quantize_row_q6_K_ref}). */
    private static void q6KRef(float[] x, int xOff, byte[] y, int yOff) {
        q6KRow(x, xOff, y, yOff, null, 0);
    }

    /** One Q6_K super-block with importance weights ({@code quantize_row_q6_K_impl}). */
    private static void q6KImpl(float[] x, int xOff, byte[] y, int yOff, float[] qw, int qwOff) {
        q6KRow(x, xOff, y, yOff, qw, qwOff);
    }
}

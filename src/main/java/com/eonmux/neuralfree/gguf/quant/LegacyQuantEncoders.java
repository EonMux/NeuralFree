package com.eonmux.neuralfree.gguf.quant;

import com.eonmux.neuralfree.gguf.GGMLType;

import static com.eonmux.neuralfree.gguf.quant.BlockLayout.IQ4_NL_BYTES;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.IQ4_NL_D;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.IQ4_NL_QS;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.IQ4_XS_BYTES;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.IQ4_XS_D;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.IQ4_XS_QS;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.IQ4_XS_SCALES_H;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.IQ4_XS_SCALES_L;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q4_0_BYTES;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q4_0_D;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q4_0_QS;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q4_1_BYTES;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q4_1_D;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q4_1_M;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q4_1_QS;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q5_0_BYTES;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q5_0_D;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q5_0_QH;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q5_0_QS;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q5_1_BYTES;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q5_1_D;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q5_1_M;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q5_1_QH;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q5_1_QS;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.QK4_NL;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.QK_K;

/**
 * Row encoders of the 32-value legacy blocks Q4_0, Q4_1, Q5_0 and Q5_1 and of the non-linear 4-bit blocks
 * IQ4_NL and IQ4_XS: ports of the imatrix-weighted {@code quantize_row_q4_0_impl} / {@code q4_1} /
 * {@code q5_0} / {@code q5_1} functions of llama.cpp ggml-quants.c (:2147-2424) and of
 * {@code quantize_row_iq4_nl_impl} (:4938-5047), the one quantizer behind both IQ4_NL and IQ4_XS
 * ({@code quantize_iq4_nl} :5065, {@code quantize_iq4_xs} :5103). The unweighted legacy paths are the ggml
 * reference quantizers {@link com.eonmux.neuralfree.gguf.memory.KvBlockCodec} already stores; the IQ4 paths
 * always run the {@code ntry = 7} search of the reference quantize entry points, weighted or not.
 * <p>
 * Package seam: {@link RowQuantizers}.
 */
public final class LegacyQuantEncoders {

    private LegacyQuantEncoders() {
    }

    /** True when {@link #quantizeRow} quantizes {@code type}. */
    public static boolean supports(GGMLType type) {
        switch (type) {
            case Q4_0:
            case Q4_1:
            case Q5_0:
            case Q5_1:
            case IQ4_NL:
            case IQ4_XS:
                return true;
            default:
                return false;
        }
    }

    /**
     * Quantizes one row of {@code nPerRow} values at {@code xOff} into {@code dst} at {@code dstOff};
     * {@code weights} (one value per input value, or null) selects the weighted path of the legacy types
     * and is the imatrix slice the IQ4 searches weight with either way.
     *
     * @throws IllegalArgumentException when the row is not a whole number of blocks of {@code type}
     */
    public static void quantizeRow(GGMLType type, float[] x, int xOff, byte[] dst, int dstOff, int nPerRow,
                                   float[] weights) {
        quantizeRow(type, x, xOff, dst, dstOff, nPerRow, weights, 0);
    }

    /**
     * {@link #quantizeRow} of a row whose importance slice starts at {@code weightsOff} of the shared
     * {@code weights} array (one expert matrix after the other, as the tool passes them).
     */
    public static void quantizeRow(GGMLType type, float[] x, int xOff, byte[] dst, int dstOff, int nPerRow,
                                   float[] weights, int weightsOff) {
        switch (type) {
            case Q4_0:
            case Q4_1:
            case Q5_0:
            case Q5_1:
                quantizeLegacyRow(type, x, xOff, dst, dstOff, nPerRow, weights, weightsOff);
                return;
            case IQ4_NL:
                quantizeIq4Grouped(type, x, xOff, dst, dstOff, nPerRow, weights, weightsOff, QK4_NL,
                        QK4_NL, IQ4_NL_BYTES);
                return;
            case IQ4_XS:
                quantizeIq4Grouped(type, x, xOff, dst, dstOff, nPerRow, weights, weightsOff, QK_K,
                        QK4_NL, IQ4_XS_BYTES);
                return;
            default:
                throw new IllegalArgumentException(type + " is not a legacy or IQ4 row type");
        }
    }

    /** One legacy row: the reference blocks without weights, the {@code _impl} row pass with them. */
    private static void quantizeLegacyRow(GGMLType type, float[] x, int xOff, byte[] dst, int dstOff,
                                          int nPerRow, float[] weights, int weightsOff) {
        if (nPerRow < 0 || nPerRow % 32 != 0) {
            throw new IllegalArgumentException("a row of " + nPerRow + " values is not a whole number of "
                    + "32-value " + type + " blocks");
        }
        if (weights == null) {
            for (int b = 0; b < nPerRow; b += 32) {
                legacyRef(type, x, xOff + b, dst, dstOff + b / 32 * type.getTypeSize());
            }
            return;
        }
        switch (type) {
            case Q4_0:
                q4_0Impl(x, xOff, dst, dstOff, nPerRow, weights, weightsOff);
                break;
            case Q4_1:
                q4_1Impl(x, xOff, dst, dstOff, nPerRow, weights, weightsOff);
                break;
            case Q5_0:
                q5_0Impl(x, xOff, dst, dstOff, nPerRow, weights, weightsOff);
                break;
            case Q5_1:
                q5_1Impl(x, xOff, dst, dstOff, nPerRow, weights, weightsOff);
                break;
            default:
                throw new IllegalArgumentException(type + " is not a legacy row type");
        }
    }

    /** One IQ4_NL or IQ4_XS row, one super-block group at a time ({@code quantize_iq4_nl} / {@code _xs}). */
    private static void quantizeIq4Grouped(GGMLType type, float[] x, int xOff, byte[] dst, int dstOff,
                                           int nPerRow, float[] weights, int weightsOff, int superBlock,
                                           int blockSize, int groupBytes) {
        if (nPerRow < 0 || nPerRow % superBlock != 0) {
            throw new IllegalArgumentException("a row of " + nPerRow + " values is not a whole number of "
                    + superBlock + "-value " + type + " blocks");
        }
        for (int g = 0; g < nPerRow; g += superBlock) {
            iq4Impl(x, xOff + g, dst, dstOff + g / superBlock * groupBytes, superBlock, blockSize,
                    weights, weightsOff + g, 7);
        }
    }

    /** One legacy block through {@link com.eonmux.neuralfree.gguf.memory.KvBlockCodec}'s reference quantizer. */
    private static void legacyRef(GGMLType type, float[] x, int xOff, byte[] y, int yOff) {
        float[] block = new float[32];
        System.arraycopy(x, xOff, block, 0, 32);
        byte[] out = new byte[type.getTypeSize()];
        com.eonmux.neuralfree.gguf.memory.KvBlockCodec.quantize(
                com.eonmux.neuralfree.gguf.memory.KvDType.valueOf(type.name()), block, 0, out, 0);
        System.arraycopy(out, 0, y, yOff, type.getTypeSize());
    }

    /** Row mean square over the whole row ({@code sigma2 = sum_x2 / n_per_row} of the legacy impls). */
    private static float rowSigma2(float[] x, int xOff, int nPerRow) {
        float sumX2 = 0;
        for (int j = 0; j < nPerRow; ++j) {
            sumX2 += x[xOff + j] * x[xOff + j];
        }
        return sumX2 / nPerRow;
    }

    /** One weighted Q4_0 row ({@code quantize_row_q4_0_impl}). */
    private static void q4_0Impl(float[] x, int xOff, byte[] y, int yOff, int nPerRow, float[] qw, int qwOff) {
        float sigma2 = rowSigma2(x, xOff, nPerRow);
        float[] weight = new float[32];
        byte[] L = new byte[32];

        int nb = nPerRow / 32;
        for (int ib = 0; ib < nb; ++ib) {
            int b = xOff + 32 * ib;
            int wb = qwOff + 32 * ib;
            for (int j = 0; j < 32; ++j) {
                weight[j] = qw[wb + j] * (float) Math.sqrt(sigma2 + x[b + j] * x[b + j]);
            }
            float d = QuantizeSearch.makeQxQuants(32, 8, x, b, L, 0, 1, weight, 0);
            KQuantEncoders.putHalf(y, yOff + ib * Q4_0_BYTES + Q4_0_D, d);
            for (int j = 0; j < 16; ++j) {
                y[yOff + ib * Q4_0_BYTES + Q4_0_QS + j] = (byte) (L[j] | (L[j + 16] << 4));
            }
        }
    }

    /** One weighted Q4_1 block ({@code quantize_row_q4_1_impl}). */
    private static void q4_1Impl(float[] x, int xOff, byte[] y, int yOff, int nPerRow, float[] qw, int qwOff) {
        float sigma2 = rowSigma2(x, xOff, nPerRow);
        float[] weight = new float[32];
        byte[] L = new byte[32];
        byte[] Laux = new byte[32];
        float[] minOut = new float[1];

        int nb = nPerRow / 32;
        for (int ib = 0; ib < nb; ++ib) {
            int b = xOff + 32 * ib;
            int wb = qwOff + 32 * ib;
            for (int j = 0; j < 32; ++j) {
                weight[j] = qw[wb + j] * (float) Math.sqrt(sigma2 + x[b + j] * x[b + j]);
            }
            float d = QuantizeSearch.makeQkx3Quants(32, 15, x, b, weight, L, 0, minOut, Laux, 0,
                    -0.9f, 0.05f, 36, false);
            int off = yOff + ib * Q4_1_BYTES;
            KQuantEncoders.putHalf(y, off + Q4_1_D, d);
            KQuantEncoders.putHalf(y, off + Q4_1_M, -minOut[0]);
            for (int j = 0; j < 16; ++j) {
                y[off + Q4_1_QS + j] = (byte) (L[j] | (L[j + 16] << 4));
            }
        }
    }

    /** One weighted Q5_0 block ({@code quantize_row_q5_0_impl}). */
    private static void q5_0Impl(float[] x, int xOff, byte[] y, int yOff, int nPerRow, float[] qw, int qwOff) {
        float sigma2 = rowSigma2(x, xOff, nPerRow);
        float[] weight = new float[32];
        byte[] L = new byte[32];

        int nb = nPerRow / 32;
        for (int ib = 0; ib < nb; ++ib) {
            int b = xOff + 32 * ib;
            int wb = qwOff + 32 * ib;
            for (int j = 0; j < 32; ++j) {
                weight[j] = qw[wb + j] * (float) Math.sqrt(sigma2 + x[b + j] * x[b + j]);
            }
            float d = QuantizeSearch.makeQxQuants(32, 16, x, b, L, 0, 1, weight, 0);
            int off = yOff + ib * Q5_0_BYTES;
            KQuantEncoders.putHalf(y, off + Q5_0_D, d);

            int qh = 0;
            for (int j = 0; j < 16; ++j) {
                int xi0 = L[j] & 0xFF;
                int xi1 = L[j + 16] & 0xFF;
                y[off + Q5_0_QS + j] = (byte) ((xi0 & 0x0F) | ((xi1 & 0x0F) << 4));

                // get the 5-th bit and store it in qh at the right position
                qh |= ((xi0 & 0x10) >>> 4) << (j + 0);
                qh |= ((xi1 & 0x10) >>> 4) << (j + 16);
            }
            putU32(y, off + Q5_0_QH, qh);
        }
    }

    /** One weighted Q5_1 block ({@code quantize_row_q5_1_impl}). */
    private static void q5_1Impl(float[] x, int xOff, byte[] y, int yOff, int nPerRow, float[] qw, int qwOff) {
        float sigma2 = rowSigma2(x, xOff, nPerRow);
        float[] weight = new float[32];
        byte[] L = new byte[32];
        byte[] Laux = new byte[32];
        float[] minOut = new float[1];

        int nb = nPerRow / 32;
        for (int ib = 0; ib < nb; ++ib) {
            int b = xOff + 32 * ib;
            int wb = qwOff + 32 * ib;
            for (int j = 0; j < 32; ++j) {
                weight[j] = qw[wb + j] * (float) Math.sqrt(sigma2 + x[b + j] * x[b + j]);
            }
            float d = QuantizeSearch.makeQkx3Quants(32, 31, x, b, weight, L, 0, minOut, Laux, 0,
                    -0.9f, 0.05f, 36, false);
            int off = yOff + ib * Q5_1_BYTES;
            KQuantEncoders.putHalf(y, off + Q5_1_D, d);
            KQuantEncoders.putHalf(y, off + Q5_1_M, -minOut[0]);

            int qh = 0;
            for (int j = 0; j < 16; ++j) {
                int xi0 = L[j] & 0xFF;
                int xi1 = L[j + 16] & 0xFF;
                y[off + Q5_1_QS + j] = (byte) ((xi0 & 0x0F) | ((xi1 & 0x0F) << 4));
                // get the 5-th bit and store it in qh at the right position
                qh |= ((xi0 & 0x10) >>> 4) << (j + 0);
                qh |= ((xi1 & 0x10) >>> 4) << (j + 16);
            }
            putU32(y, off + Q5_1_QH, qh);
        }
    }

    /**
     * One IQ4 group through {@code quantize_row_iq4_nl_impl}: {@code superBlock} input values (32 for
     * IQ4_NL, 256 for IQ4_XS) over {@code blockSize} sub-blocks, the {@code ntry} scale grid search, the
     * shared super-block scale for IQ4_XS ({@code superBlock > blockSize}) and the packed nibbles.
     * {@code qw} is the importance slice of this group, or null.
     */
    private static void iq4Impl(float[] x, int xOff, byte[] y, int yOff, int superBlock, int blockSize,
                                float[] qw, int qwOff, int ntry) {
        float[] values = IQ4_VALUES;

        float sigma2 = 0;
        for (int j = 0; j < superBlock; ++j) {
            sigma2 += x[xOff + j] * x[xOff + j];
        }
        sigma2 *= 2.f / superBlock;

        int nGroups = superBlock / blockSize;
        int q4Bytes = superBlock / 2;
        int dhOff = yOff + (superBlock == blockSize ? IQ4_NL_D : IQ4_XS_D);
        int q4Off = yOff + (superBlock == blockSize ? IQ4_NL_QS : IQ4_XS_QS);
        int scalesHOff = yOff + IQ4_XS_SCALES_H;
        int scalesLOff = yOff + IQ4_XS_SCALES_L;

        for (int j = 0; j < q4Bytes; ++j) {
            y[q4Off + j] = 0;
        }
        KQuantEncoders.putHalf(y, dhOff, 0.f);

        float[] mid = new float[15];
        float[] fval = new float[16];
        for (int k = 0; k < 15; ++k) {
            mid[k] = 0.5f * (values[k] + values[k + 1]);
        }
        for (int k = 0; k < 16; ++k) {
            fval[k] = values[k];
        }

        float[] weight = new float[blockSize];
        byte[] L = new byte[superBlock];
        float[] scales = new float[nGroups];

        float maxScale = 0;
        float amaxScale = 0;
        for (int ib = 0; ib < nGroups; ++ib) {
            int b = xOff + ib * blockSize;
            int lb = ib * blockSize;
            if (qw != null) {
                int wb = qwOff + ib * blockSize;
                for (int j = 0; j < blockSize; ++j) {
                    weight[j] = qw[wb + j] * (float) Math.sqrt(sigma2 + x[b + j] * x[b + j]);
                }
            } else {
                for (int j = 0; j < blockSize; ++j) {
                    weight[j] = x[b + j] * x[b + j];
                }
            }
            float amax = 0;
            float max = 0;
            for (int j = 0; j < blockSize; ++j) {
                float ax = Math.abs(x[b + j]);
                if (ax > amax) {
                    amax = ax;
                    max = x[b + j];
                }
            }
            if (amax < QuantizeSearch.GROUP_MAX_EPS) {
                scales[ib] = 0;
                continue;
            }
            float d = ntry > 0 ? -max / values[0] : max / values[0];
            // lane 0 is the initial guess, lanes 1..nt-1 are itry = -ntry..ntry; each lane sums over j in order
            int nt = ntry >= 0 ? 2 * ntry + 2 : 1;
            float[] idt = new float[16];
            float[] sqx = new float[16];
            float[] sq2 = new float[16];
            idt[0] = 1 / d;
            for (int t = 1; t < nt; ++t) {
                idt[t] = (t - 1 - ntry + values[0]) / max;
            }
            for (int j = 0; j < blockSize; ++j) {
                L[lb + j] = (byte) bestIndexIq4nl(mid, idt[0] * x[b + j]);
            }
            for (int j = 0; j < blockSize; ++j) {
                float w = weight[j];
                for (int t = 0; t < nt; ++t) {
                    float q = fval[bestIndexIq4nl(mid, idt[t] * x[b + j])];
                    sqx[t] += w * q * x[b + j];
                    sq2[t] += w * q * q;
                }
            }
            d = sq2[0] > 0 ? sqx[0] / sq2[0] : 0.f;
            float best = d * sqx[0];
            for (int t = 1; t < nt; ++t) {
                if (sq2[t] > 0 && sqx[t] * sqx[t] > best * sq2[t]) {
                    d = sqx[t] / sq2[t];
                    best = d * sqx[t];
                }
            }
            scales[ib] = d;
            float absD = Math.abs(d);
            if (absD > amaxScale) {
                amaxScale = absD;
                maxScale = d;
            }
        }

        if (nGroups > 1) {
            int scalesHWords = (nGroups + 7) / 8;
            for (int i = 0; i < scalesHWords * 2; ++i) {
                y[scalesHOff + i] = 0;
            }
            for (int i = 0; i < nGroups / 2; ++i) {
                y[scalesLOff + i] = 0;
            }
            float d = -maxScale / 32;
            KQuantEncoders.putHalf(y, dhOff, d);
            float id = d != 0 ? 1 / d : 0.f;
            for (int ib = 0; ib < nGroups; ++ib) {
                int l = QuantizeSearch.nearestInt(id * scales[ib]);
                l = Math.max(-32, Math.min(31, l));
                float dl = d * l;
                float idl = dl != 0 ? 1 / dl : 0.f;
                int b = xOff + ib * blockSize;
                int lb = ib * blockSize;
                for (int j = 0; j < blockSize; ++j) {
                    L[lb + j] = (byte) bestIndexIq4nl(mid, idl * x[b + j]);
                }
                l += 32;
                int lL = l & 0xf;
                int lH = l >> 4;
                if (ib % 2 == 0) {
                    y[scalesLOff + ib / 2] = (byte) lL;
                } else {
                    y[scalesLOff + ib / 2] |= (byte) (lL << 4);
                }
                // scales_h is one little-endian uint16 per 8 sub-blocks; the 2-bit code sits at bit
                // 2*(ib%8) of that word
                int word = ib / 8;
                int shift = 2 * (ib % 8);
                if (shift < 8) {
                    y[scalesHOff + 2 * word] |= (byte) (lH << shift);
                } else {
                    y[scalesHOff + 2 * word + 1] |= (byte) (lH << (shift - 8));
                }
            }
        } else {
            KQuantEncoders.putHalf(y, dhOff, scales[0]);
            if (ntry > 0) {
                float id = scales[0] != 0 ? 1 / scales[0] : 0;
                for (int j = 0; j < superBlock; ++j) {
                    L[j] = (byte) bestIndexIq4nl(mid, id * x[xOff + j]);
                }
            }
        }

        for (int i = 0; i < superBlock / 32; ++i) {
            for (int j = 0; j < 16; ++j) {
                y[q4Off + 16 * i + j] = (byte) (L[32 * i + j] | (L[32 * i + 16 + j] << 4));
            }
        }
    }

    /**
     * {@code best_index_iq4nl} (llama.cpp ggml-quants.c:30-35): the nearest {@code kvalues_iq4nl} index
     * over the midpoints, counting how many midpoints {@code x} stays below.
     */
    private static int bestIndexIq4nl(float[] mid, float x) {
        int l = 15;
        for (int k = 0; k < 15; ++k) {
            l -= x < mid[k] ? 1 : 0;
        }
        return l;
    }

    static void putU32(byte[] y, int off, int v) {
        y[off] = (byte) v;
        y[off + 1] = (byte) (v >>> 8);
        y[off + 2] = (byte) (v >>> 16);
        y[off + 3] = (byte) (v >>> 24);
    }

    /** {@code kvalues_iq4nl} as floats, the values array the IQ4 search reads. */
    private static final float[] IQ4_VALUES = {
        -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
    };
}

package com.eonmux.neuralfree.gguf.quant;

import com.eonmux.neuralfree.gguf.GGMLType;

import java.util.Arrays;

import static com.eonmux.neuralfree.gguf.quant.BlockLayout.IQ1_M_BYTES;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.IQ1_M_QH;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.IQ1_M_QS;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.IQ1_M_SCALES;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.IQ1_S_BYTES;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.IQ1_S_D;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.IQ1_S_QH;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.IQ1_S_QS;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.QK_K;
import static com.eonmux.neuralfree.gguf.quant.FpBlockDecoders.fp32ToFp16;
import static com.eonmux.neuralfree.gguf.quant.IqGridSearch.sumSquares;
import static com.eonmux.neuralfree.gguf.quant.KQuantEncoders.putHalf;
import static com.eonmux.neuralfree.gguf.quant.QuantizeSearch.nearestInt;

/**
 * Row encoders of the ternary-grid types IQ1_S and IQ1_M: ports of {@code quantize_iq1_s} (llama.cpp
 * ggml-quants.c :4524-4706) and {@code quantize_iq1_m} (:4708-4949). Each sub-block solves the weighted
 * three-level problem exactly — the values sorted, every pair of split points scored from prefix sums
 * over both signs of the {@code IQ1S_DELTA} shift — then snaps each 8-value group to the shared 2048-point
 * grid of {@link IqGrid}, off-grid vectors to their best neighbour ({@code iq1_find_best_neighbour2}
 * :4459). The output decodes through {@link IQuantBlockDecoders}.
 * <p>
 * IQ1_S needs the importance weights ({@code ggml_quantize_requires_imatrix}); IQ1_M falls back to the
 * reference's unweighted importance {@code x^2} without them.
 * <p>
 * Package seam: {@link RowQuantizers}. Thread-safe and stateless; each row call owns its scratch.
 */
public final class Iq1Encoders {

    private static final float GROUP_MAX_EPS_IQ1_S = 1e-12f;
    private static final float GROUP_MAX_EPS_IQ1_M = 1e-7f;
    private static final float DELTA = QuantTables.IQ1S_DELTA;
    private static final float[] X_P = {-1 + DELTA, DELTA, 1 + DELTA};
    private static final float[] X_M = {-1 - DELTA, -DELTA, 1 - DELTA};
    private static final float CP1 = X_P[0] * X_P[0] - X_P[1] * X_P[1];
    private static final float CP2 = X_P[1] * X_P[1] - X_P[2] * X_P[2];
    private static final float CM1 = X_M[0] * X_M[0] - X_M[1] * X_M[1];
    private static final float CM2 = X_M[1] * X_M[1] - X_M[2] * X_M[2];
    private static final int[] IQ1M_SHIFT_MASKS = {0x00, 0x80, 0x08, 0x88};

    private Iq1Encoders() {
    }

    /** True when {@link #quantizeRow} quantizes {@code type}. */
    public static boolean supports(GGMLType type) {
        return type == GGMLType.IQ1_S || type == GGMLType.IQ1_M;
    }

    /** True when {@code type}'s quantizer cannot run without importance weights (IQ1_S). */
    public static boolean requiresWeights(GGMLType type) {
        return type == GGMLType.IQ1_S;
    }

    /**
     * Quantizes one row of {@code nPerRow} values at {@code xOff} into {@code dst} at {@code dstOff};
     * {@code weights} is the importance matrix, read from {@code weightsOff} on (one value per column),
     * or null.
     *
     * @throws IllegalArgumentException when {@code type} is not IQ1_S or IQ1_M, the row is not a whole
     *                                  number of 256-value super-blocks, or IQ1_S has no weights
     */
    public static void quantizeRow(GGMLType type, float[] x, int xOff, byte[] dst, int dstOff, int nPerRow,
                                   float[] weights, int weightsOff) {
        if (!supports(type)) {
            throw new IllegalArgumentException(type + " is not an IQ1 type");
        }
        if (nPerRow < 0 || nPerRow % QK_K != 0) {
            throw new IllegalArgumentException("a row of " + nPerRow + " values is not a whole number of "
                    + QK_K + "-value " + type + " super-blocks");
        }
        if (weights == null && requiresWeights(type)) {
            throw new IllegalArgumentException(type + " quantization needs importance weights (an imatrix)"
                    + " for every row");
        }
        IqGrid grid = IqGrid.of(type);
        int nb = nPerRow / QK_K;
        for (int i = 0; i < nb; i++) {
            if (type == GGMLType.IQ1_S) {
                iq1S(grid, x, xOff + i * QK_K, dst, dstOff + i * IQ1_S_BYTES, weights, weightsOff + i * QK_K);
            } else {
                iq1M(grid, x, xOff + i * QK_K, dst, dstOff + i * IQ1_M_BYTES, weights, weightsOff + i * QK_K);
            }
        }
    }

    // ------------------------------------------------------------------ IQ1_S

    /** The winning split of one sub-block: the levels, the scale and the delta sign (or combination). */
    private static final class Split {
        float scale;
        int shift;
        final byte[] levels;
        final int[] index;

        Split(int n) {
            levels = new byte[n];
            index = new int[n / 8];
        }
    }

    /** One IQ1_S super-block: 8 sub-blocks of 32, one 3-bit scale and one delta sign each. */
    private static void iq1S(IqGrid grid, float[] x, int xOff, byte[] y, int yOff, float[] qw, int qwOff) {
        final int bs = 32;
        Arrays.fill(y, yOff, yOff + IQ1_S_BYTES, (byte) 0);
        float[] scales = new float[QK_K / bs];
        int[] shifts = new int[QK_K / bs];
        float[] weight = new float[bs];
        int[] idx = new int[bs];
        Split split = new Split(bs);
        int[] qh = new int[QK_K / bs];
        float sigma2 = 2 * sumSquares(x, xOff, QK_K) / QK_K;
        float maxScale = 0;
        for (int ib = 0; ib < QK_K / bs; ++ib) {
            int xb = xOff + bs * ib;
            for (int i = 0; i < bs; ++i) {
                weight[i] = qw[qwOff + bs * ib + i] * (float) Math.sqrt(sigma2 + x[xb + i] * x[xb + i]);
            }
            shifts[ib] = 1;
            if (maxAbs(x, xb, bs) < GROUP_MAX_EPS_IQ1_S || !bestSplitS(x, xb, weight, idx, split)) {
                continue;
            }
            float[] xx = split.shift == 1 ? X_P : X_M;
            float scale = snapGroups(grid, x, xb, weight, split, xx, xx);
            for (int k = 0; k < bs / 8; ++k) {
                y[yOff + IQ1_S_QS + (bs / 8) * ib + k] = (byte) split.index[k];
                qh[ib] |= (split.index[k] >>> 8) << (3 * k);
            }
            scales[ib] = scale;
            shifts[ib] = split.shift;
            maxScale = Math.max(maxScale, scale);
        }
        if (maxScale != 0) {
            float d = maxScale / 15;
            putHalf(y, yOff + IQ1_S_D, d * 1.125f);
            float id = 1 / d;
            for (int ib = 0; ib < QK_K / bs; ++ib) {
                int l = Math.max(0, Math.min(7, nearestInt(0.5f * (id * scales[ib] - 1))));
                if (shifts[ib] == -1) {
                    l |= 8;
                }
                qh[ib] |= l << 12;
            }
        }
        for (int ib = 0; ib < QK_K / bs; ++ib) {
            y[yOff + IQ1_S_QH + 2 * ib] = (byte) qh[ib];
            y[yOff + IQ1_S_QH + 2 * ib + 1] = (byte) (qh[ib] >>> 8);
        }
    }

    /**
     * The exact weighted three-level fit of one IQ1_S sub-block over both delta signs; fills {@code split}
     * with levels in [0, 2], a positive scale and the shift (+1 or -1). Returns false when no split has a
     * positive fit.
     */
    private static boolean bestSplitS(float[] x, int xb, float[] weight, int[] idx, Split split) {
        int n = idx.length;
        sortByValue(x, xb, idx);
        float[] sumx = new float[n + 1];
        float[] sumw = new float[n + 1];
        for (int j = 0; j < n; ++j) {
            int i = idx[j];
            sumx[j + 1] = sumx[j] + weight[i] * x[xb + i];
            sumw[j + 1] = sumw[j] + weight[i];
        }
        float bestScore = -Float.MAX_VALUE;
        float scale = 0;
        int besti1 = -1;
        int besti2 = -1;
        int bestShift = 0;
        for (int i1 = 0; i1 <= n; ++i1) {
            for (int i2 = i1; i2 <= n; ++i2) {
                for (int sign = 1; sign >= -1; sign -= 2) {
                    float[] v = sign == 1 ? X_P : X_M;
                    float sumqx = (sumx[i1] - sumx[0]) * v[0] + (sumx[i2] - sumx[i1]) * v[1]
                            + (sumx[n] - sumx[i2]) * v[2];
                    float sumq2 = (sumw[i1] - sumw[0]) * v[0] * v[0] + (sumw[i2] - sumw[i1]) * v[1] * v[1]
                            + (sumw[n] - sumw[i2]) * v[2] * v[2];
                    if (sumq2 > 0 && sumqx * sumqx > bestScore * sumq2) {
                        scale = sumqx / sumq2;
                        bestScore = scale * sumqx;
                        besti1 = i1;
                        besti2 = i2;
                        bestShift = sign;
                    }
                }
            }
        }
        if (besti1 < 0 || besti2 < 0 || bestShift == 0) {
            return false;
        }
        assignLevels(idx, besti1, besti2, split.levels);
        if (scale < 0) {
            for (int j = 0; j < n; ++j) {
                split.levels[j] = (byte) (2 - split.levels[j]);
            }
            scale = -scale;
            bestShift = -bestShift;
        }
        split.scale = scale;
        split.shift = bestShift;
        return true;
    }

    // ------------------------------------------------------------------ IQ1_M

    /** One IQ1_M super-block: 16 sub-blocks of 16, a 3-bit scale each and one delta sign per 8 values. */
    private static void iq1M(IqGrid grid, float[] x, int xOff, byte[] y, int yOff, float[] qw, int qwOff) {
        final int bs = 16;
        Arrays.fill(y, yOff, yOff + IQ1_M_BYTES, (byte) 0);
        float[] scales = new float[QK_K / bs];
        int[] shifts = new int[QK_K / bs];
        float[] weight = new float[bs];
        int[] idx = new int[bs];
        Split split = new Split(bs);
        float sigma2 = 2 * sumSquares(x, xOff, QK_K) / QK_K;
        float maxScale = 0;
        for (int ib = 0; ib < QK_K / bs; ++ib) {
            int xb = xOff + bs * ib;
            iq1mWeights(x, xb, qw, qwOff + bs * ib, sigma2, weight);
            if (maxAbs(x, xb, bs) < GROUP_MAX_EPS_IQ1_M || !bestSplitM(x, xb, weight, idx, split)) {
                continue;
            }
            float scale = snapGroups(grid, x, xb, weight, split,
                    split.shift < 2 ? X_P : X_M, split.shift % 2 == 0 ? X_P : X_M);
            y[yOff + IQ1_M_QS + 2 * ib] = (byte) split.index[0];
            y[yOff + IQ1_M_QS + 2 * ib + 1] = (byte) split.index[1];
            y[yOff + IQ1_M_QH + ib] = (byte) ((split.index[0] >>> 8) | ((split.index[1] >>> 8) << 4));
            scales[ib] = scale;
            shifts[ib] = split.shift;
            maxScale = Math.max(maxScale, scale);
        }
        if (maxScale == 0) {
            return;
        }
        iq1mScales(grid, x, xOff, y, yOff, qw, qwOff, sigma2, scales, shifts, maxScale);
    }

    /** The IQ1_M importance of one sub-block: {@code qw * sqrt(sigma2 + x^2)}, or {@code x^2} without weights. */
    private static void iq1mWeights(float[] x, int xb, float[] qw, int qwOff, float sigma2, float[] weight) {
        for (int i = 0; i < weight.length; ++i) {
            float v = x[xb + i];
            weight[i] = qw != null ? qw[qwOff + i] * (float) Math.sqrt(sigma2 + v * v) : v * v;
        }
    }

    /**
     * The IQ1_M scale pass: the 3-bit sub-block scales and delta masks, then the super-block scale refitted
     * by least squares over the whole block and stored in the high nibbles of the four scale words.
     */
    private static void iq1mScales(IqGrid grid, float[] x, int xOff, byte[] y, int yOff, float[] qw, int qwOff,
                                   float sigma2, float[] scales, int[] shifts, float maxScale) {
        final int bs = 16;
        int[] sc = new int[4];
        float[] weight = new float[bs];
        float d = maxScale / 15;
        float id = 1 / d;
        float sumqx = 0;
        float sumq2 = 0;
        for (int ib = 0; ib < QK_K / bs; ++ib) {
            int l = Math.max(0, Math.min(7, nearestInt(0.5f * (id * scales[ib] - 1))));
            sc[ib / 4] |= l << (3 * (ib % 4));
            int qh = (y[yOff + IQ1_M_QH + ib] & 0xFF) | IQ1M_SHIFT_MASKS[shifts[ib]];
            y[yOff + IQ1_M_QH + ib] = (byte) qh;
            int xb = xOff + bs * ib;
            iq1mWeights(x, xb, qw, qwOff + bs * ib, sigma2, weight);
            for (int k = 0; k < bs / 8; ++k) {
                float[] xx = k == 0 ? (shifts[ib] < 2 ? X_P : X_M) : (shifts[ib] % 2 == 0 ? X_P : X_M);
                int g = (y[yOff + IQ1_M_QS + 2 * ib + k] & 0xFF) + ((qh << (8 - 4 * k)) & 0x700);
                for (int j = 0; j < 8; ++j) {
                    float w = weight[8 * k + j];
                    float q = xx[grid.level(g, j)] * (2 * l + 1);
                    sumqx += w * q * x[xb + 8 * k + j];
                    sumq2 += w * q * q;
                }
            }
        }
        if (sumq2 > 0) {
            d = sumqx / sumq2;
        }
        int s = fp32ToFp16(d * 1.1125f);
        sc[0] |= (s & 0x000f) << 12;
        sc[1] |= (s & 0x00f0) << 8;
        sc[2] |= (s & 0x0f00) << 4;
        sc[3] |= s & 0xf000;
        for (int i = 0; i < 4; ++i) {
            y[yOff + IQ1_M_SCALES + 2 * i] = (byte) sc[i];
            y[yOff + IQ1_M_SCALES + 2 * i + 1] = (byte) (sc[i] >>> 8);
        }
    }

    /**
     * The exact weighted three-level fit of one IQ1_M sub-block, each 8-value half free to take either delta
     * sign; prefix sums run per half. Fills {@code split} with levels, a positive scale and the sign
     * combination (0: ++, 1: +-, 2: -+, 3: --). Returns false when no split has a positive fit.
     */
    private static boolean bestSplitM(float[] x, int xb, float[] weight, int[] idx, Split split) {
        int n = idx.length;
        sortByValue(x, xb, idx);
        float[][] px = new float[2][n + 1];
        float[][] pw = new float[2][n + 1];
        for (int j = 0; j < n; ++j) {
            int i = idx[j];
            int h = i < n / 2 ? 0 : 1;
            px[h][j + 1] = px[h][j] + weight[i] * x[xb + i];
            px[1 - h][j + 1] = px[1 - h][j];
            pw[h][j + 1] = pw[h][j] + weight[i];
            pw[1 - h][j + 1] = pw[1 - h][j];
        }
        SplitScore fit = new SplitScore();
        for (int i1 = 0; i1 <= n; ++i1) {
            for (int i2 = i1; i2 <= n; ++i2) {
                scoreSplitM(px, pw, n, i1, i2, fit);
            }
        }
        if (fit.i1 < 0 || fit.i2 < 0 || fit.k < 0) {
            return false;
        }
        float scale = fit.scale;
        int bestK = fit.k;
        assignLevels(idx, fit.i1, fit.i2, split.levels);
        if (scale < 0) {
            for (int j = 0; j < n; ++j) {
                split.levels[j] = (byte) (2 - split.levels[j]);
            }
            scale = -scale;
            bestK = 3 - bestK;
        }
        split.scale = scale;
        split.shift = bestK;
        return true;
    }

    /** The best IQ1_M split so far: its score, scale, split points and sign combination. */
    private static final class SplitScore {
        float score = -Float.MAX_VALUE;
        float scale;
        int i1 = -1;
        int i2 = -1;
        int k = -1;

        /** Takes the fit {@code sumqx / sumq2} of combination {@code k} when it explains more. */
        void offer(float sumqx, float sumq2, int i1, int i2, int k) {
            if (sumq2 > 0 && sumqx * sumqx > score * sumq2) {
                scale = sumqx / sumq2;
                score = scale * sumqx;
                this.i1 = i1;
                this.i2 = i2;
                this.k = k;
            }
        }
    }

    /**
     * Scores the split at {@code (i1, i2)} for the four sign combinations from the per-half prefix sums:
     * with level values {@code v0 - v1 = v1 - v2 = -1}, a half's three-group sum is
     * {@code total * v2 - px[i1] - px[i2]}.
     */
    private static void scoreSplitM(float[][] px, float[][] pw, int n, int i1, int i2, SplitScore fit) {
        float sx0 = px[0][i1] + px[0][i2];
        float sx1 = px[1][i1] + px[1][i2];
        float qxP0 = px[0][n] * X_P[2] - sx0;
        float qxM0 = px[0][n] * X_M[2] - sx0;
        float qxP1 = px[1][n] * X_P[2] - sx1;
        float qxM1 = px[1][n] * X_M[2] - sx1;
        float q2P0 = pw[0][n] * X_P[2] * X_P[2] + pw[0][i1] * CP1 + pw[0][i2] * CP2;
        float q2M0 = pw[0][n] * X_M[2] * X_M[2] + pw[0][i1] * CM1 + pw[0][i2] * CM2;
        float q2P1 = pw[1][n] * X_P[2] * X_P[2] + pw[1][i1] * CP1 + pw[1][i2] * CP2;
        float q2M1 = pw[1][n] * X_M[2] * X_M[2] + pw[1][i1] * CM1 + pw[1][i2] * CM2;
        fit.offer(qxP0 + qxP1, q2P0 + q2P1, i1, i2, 0);
        fit.offer(qxP0 + qxM1, q2P0 + q2M1, i1, i2, 1);
        fit.offer(qxM0 + qxP1, q2M0 + q2P1, i1, i2, 2);
        fit.offer(qxM0 + qxM1, q2M0 + q2M1, i1, i2, 3);
    }

    // ------------------------------------------------------------------ shared

    /**
     * Snaps the 8-value groups of {@code split} to the grid (the first group reads its level values from
     * {@code xxFirst}, the others from {@code xxRest}), off-grid vectors to their best neighbour, into
     * {@code split.index}; refits the scale when any group moved. Returns the scale.
     */
    private static float snapGroups(IqGrid grid, float[] x, int xb, float[] weight, Split split,
                                    float[] xxFirst, float[] xxRest) {
        boolean allOnGrid = true;
        for (int k = 0; k < split.index.length; ++k) {
            float[] xx = k == 0 ? xxFirst : xxRest;
            int u = 0;
            for (int j = 0; j < 8; ++j) {
                u |= split.levels[8 * k + j] << (2 * j);
            }
            int g = grid.indexOf(u);
            if (g < 0) {
                allOnGrid = false;
                g = bestNeighbour(grid, grid.neighbours(u), x, xb + 8 * k, weight, 8 * k, split.scale, xx);
            }
            split.index[k] = g;
        }
        float scale = split.scale;
        if (!allOnGrid) {
            float sumqx = 0;
            float sumq2 = 0;
            for (int k = 0; k < split.index.length; ++k) {
                float[] xx = k == 0 ? xxFirst : xxRest;
                for (int j = 0; j < 8; ++j) {
                    float w = weight[8 * k + j];
                    float q = xx[grid.level(split.index[k], j)];
                    sumqx += w * q * x[xb + 8 * k + j];
                    sumq2 += w * q * q;
                }
            }
            if (sumqx > 0 && sumq2 > 0) {
                scale = sumqx / sumq2;
            }
        }
        return scale;
    }

    /** {@code iq1_find_best_neighbour2}: the neighbour with the least weighted squared error at {@code scale}. */
    private static int bestNeighbour(IqGrid grid, int[] neighbours, float[] x, int xOff, float[] weight,
                                     int wOff, float scale, float[] xx) {
        float bestScore = Float.MAX_VALUE;
        int gridIndex = -1;
        for (int g : neighbours) {
            float d2 = 0;
            for (int i = 0; i < 8; ++i) {
                float q = xx[grid.level(g, i)];
                float diff = scale * q - x[xOff + i];
                d2 += weight[wOff + i] * diff * diff;
            }
            if (d2 < bestScore) {
                bestScore = d2;
                gridIndex = g;
            }
        }
        if (gridIndex < 0) {
            throw new IllegalStateException("no IQ1 grid neighbour fits the group; the values or weights are"
                    + " not finite");
        }
        return gridIndex;
    }

    /** Levels 0 / 1 / 2 for the sorted positions below {@code i1}, in {@code [i1, i2)} and from {@code i2} on. */
    private static void assignLevels(int[] idx, int i1, int i2, byte[] levels) {
        for (int j = 0; j < idx.length; ++j) {
            levels[idx[j]] = (byte) (j < i1 ? 0 : j < i2 ? 1 : 2);
        }
    }

    /** The positions of the sub-block at {@code x[xb]} sorted by value, equal values in position order. */
    private static void sortByValue(float[] x, int xb, int[] idx) {
        for (int j = 0; j < idx.length; ++j) {
            idx[j] = j;
        }
        for (int i = 1; i < idx.length; ++i) {
            int v = idx[i];
            float key = x[xb + v];
            int j = i - 1;
            while (j >= 0 && x[xb + idx[j]] > key) {
                idx[j + 1] = idx[j];
                --j;
            }
            idx[j + 1] = v;
        }
    }

    private static float maxAbs(float[] x, int off, int n) {
        float max = Math.abs(x[off]);
        for (int i = 1; i < n; ++i) {
            max = Math.max(max, Math.abs(x[off + i]));
        }
        return max;
    }
}

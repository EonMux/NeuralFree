package com.eonmux.neuralfree.gguf.quant;

import com.eonmux.neuralfree.gguf.GGMLType;

import java.util.Arrays;

import static com.eonmux.neuralfree.gguf.quant.BlockLayout.IQ2_S_BYTES;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.IQ2_S_D;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.IQ2_S_QH;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.IQ2_S_QS;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.IQ2_S_SCALES;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.IQ2_S_SIGNS;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.IQ2_XS_BYTES;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.IQ2_XS_D;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.IQ2_XS_QS;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.IQ2_XS_SCALES;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.IQ2_XXS_BYTES;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.IQ2_XXS_D;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.IQ2_XXS_QS;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.IQ3_S_BYTES;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.IQ3_S_D;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.IQ3_S_QH;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.IQ3_S_QS;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.IQ3_S_SCALES;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.IQ3_S_SIGNS;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.IQ3_XXS_BYTES;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.IQ3_XXS_D;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.IQ3_XXS_QS;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.IQ3_XXS_SAS;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.QK_K;
import static com.eonmux.neuralfree.gguf.quant.IqGridSearch.sumSquares;
import static com.eonmux.neuralfree.gguf.quant.KQuantEncoders.putHalf;
import static com.eonmux.neuralfree.gguf.quant.LegacyQuantEncoders.putU32;
import static com.eonmux.neuralfree.gguf.quant.QuantizeSearch.GROUP_MAX_EPS;
import static com.eonmux.neuralfree.gguf.quant.QuantizeSearch.makeQpQuants;
import static com.eonmux.neuralfree.gguf.quant.QuantizeSearch.nearestInt;

/**
 * Row encoders of the grid-search types IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS and IQ3_S: ports of
 * {@code quantize_iq2_xxs} (llama.cpp ggml-quants.c :3310-3486), {@code quantize_iq2_xs} (:3488-3666),
 * {@code quantize_iq2_s} (:5130-5309), {@code quantize_iq3_xxs} (:3954-4178) and {@code quantize_iq3_s}
 * (:4185-4389) over the lattices of {@link IqGrid} and the sub-block search of {@link IqGridSearch}. The
 * output decodes through {@link IQuantBlockDecoders}.
 * <p>
 * IQ2_XXS and IQ2_XS need the importance weights ({@code ggml_quantize_requires_imatrix}); IQ2_S, IQ3_XXS
 * and IQ3_S fall back to the reference's unweighted importance without them. Every IQ3_S sub-block's
 * grid bytes and signs sit at that sub-block's offsets, also after an all-zero sub-block.
 * <p>
 * Package seam: {@link RowQuantizers}. Thread-safe and stateless; each row call owns its scratch.
 */
public final class IqGridEncoders {

    private static final int KMAXQ_IQ2 = 3;
    private static final int KMAXQ_IQ3 = 8;
    private static final float GROUP_MAX_EPS_IQ2_S = 1e-8f;
    private static final float GROUP_MAX_EPS_IQ3_XXS = 1e-8f;

    private IqGridEncoders() {
    }

    /** True when {@link #quantizeRow} quantizes {@code type}. */
    public static boolean supports(GGMLType type) {
        switch (type) {
            case IQ2_XXS:
            case IQ2_XS:
            case IQ2_S:
            case IQ3_XXS:
            case IQ3_S:
                return true;
            default:
                return false;
        }
    }

    /** True when {@code type}'s quantizer cannot run without importance weights (IQ2_XXS and IQ2_XS). */
    public static boolean requiresWeights(GGMLType type) {
        return type == GGMLType.IQ2_XXS || type == GGMLType.IQ2_XS;
    }

    /**
     * Quantizes one row of {@code nPerRow} values at {@code xOff} into {@code dst} at {@code dstOff};
     * {@code weights} is the importance matrix, read from {@code weightsOff} on (one value per column),
     * or null.
     *
     * @throws IllegalArgumentException when {@code type} is not one of these types, the row is not a whole
     *                                  number of 256-value super-blocks, or IQ2_XXS / IQ2_XS has no weights
     */
    public static void quantizeRow(GGMLType type, float[] x, int xOff, byte[] dst, int dstOff, int nPerRow,
                                   float[] weights, int weightsOff) {
        if (!supports(type)) {
            throw new IllegalArgumentException(type + " is not an IQ2/IQ3 grid type");
        }
        if (nPerRow < 0 || nPerRow % QK_K != 0) {
            throw new IllegalArgumentException("a row of " + nPerRow + " values is not a whole number of "
                    + QK_K + "-value " + type + " super-blocks");
        }
        if (weights == null && requiresWeights(type)) {
            throw new IllegalArgumentException(type + " quantization needs importance weights (an imatrix)"
                    + " for every row");
        }
        int nb = nPerRow / QK_K;
        IqGrid grid = IqGrid.of(type);
        switch (type) {
            case IQ2_XXS: {
                IqGridSearch s = new IqGridSearch(grid, 32, KMAXQ_IQ2);
                for (int i = 0; i < nb; i++) {
                    iq2Xxs(s, x, xOff + i * QK_K, dst, dstOff + i * IQ2_XXS_BYTES, weights, weightsOff + i * QK_K);
                }
                return;
            }
            case IQ2_XS: {
                IqGridSearch s = new IqGridSearch(grid, 16, KMAXQ_IQ2);
                for (int i = 0; i < nb; i++) {
                    iq2Xs(s, x, xOff + i * QK_K, dst, dstOff + i * IQ2_XS_BYTES, weights, weightsOff + i * QK_K);
                }
                return;
            }
            case IQ2_S: {
                IqGridSearch s = new IqGridSearch(grid, 16, KMAXQ_IQ2);
                for (int i = 0; i < nb; i++) {
                    iq2S(s, x, xOff + i * QK_K, dst, dstOff + i * IQ2_S_BYTES, weights, weightsOff + i * QK_K);
                }
                return;
            }
            case IQ3_XXS: {
                IqGridSearch s = new IqGridSearch(grid, 32, KMAXQ_IQ3);
                for (int i = 0; i < nb; i++) {
                    iq3Xxs(s, x, xOff + i * QK_K, dst, dstOff + i * IQ3_XXS_BYTES, weights, weightsOff + i * QK_K);
                }
                return;
            }
            default: {
                IqGridSearch s = new IqGridSearch(grid, 32, KMAXQ_IQ3);
                for (int i = 0; i < nb; i++) {
                    iq3S(s, x, xOff + i * QK_K, dst, dstOff + i * IQ3_S_BYTES, weights, weightsOff + i * QK_K);
                }
            }
        }
    }

    /** The 4-bit sub-block scale index {@code nearest_int(0.5 * (id * scale - 1))} clamped to [0, 15]. */
    private static int scaleIndex(float id, float scale) {
        return Math.max(0, Math.min(15, nearestInt(0.5f * (id * scale - 1))));
    }

    // ------------------------------------------------------------------ IQ2_XXS

    /** One IQ2_XXS super-block: 8 sub-blocks of 32, four 8-lane grid bytes and a 32-bit sign/scale word each. */
    private static void iq2Xxs(IqGridSearch s, float[] x, int xOff, byte[] y, int yOff, float[] qw, int qwOff) {
        int[] q2 = new int[2 * (QK_K / 32)];
        float[] scales = new float[QK_K / 32];
        int[] signs = new int[4];
        float sigma2 = sumSquares(x, xOff, QK_K) / QK_K;
        float maxScale = 0;
        for (int ib = 0; ib < QK_K / 32; ++ib) {
            int xb = xOff + 32 * ib;
            s.weighted(x, xb, qw, qwOff + 32 * ib, sigma2);
            for (int k = 0; k < 4; ++k) {
                signs[k] = s.foldSigns(x, xb, k, true);
            }
            if (s.max() < GROUP_MAX_EPS) {
                continue;
            }
            float scale = makeQpQuants(32, KMAXQ_IQ2 + 1, s.xval, 0, s.levels, 0, s.weight, 0);
            float effMax = scale * KMAXQ_IQ2;
            if (effMax <= 0) {
                continue;
            }
            scale = s.scan(effMax, scale, 6, 0.1f, true);
            scale = s.refit(scale, IqGridSearch.Refit.ALL);
            if (scale < 0) {
                scale = -scale;
                flipSigns(signs, 127);
            }
            s.gridIndices();
            for (int k = 0; k < 4; ++k) {
                q2[2 * ib] |= s.index[k] << (8 * k);
                q2[2 * ib + 1] |= signs[k] << (7 * k);
            }
            scales[ib] = scale;
            maxScale = Math.max(maxScale, scale);
        }
        Arrays.fill(y, yOff, yOff + IQ2_XXS_BYTES, (byte) 0);
        if (maxScale == 0) {
            return;
        }
        float d = maxScale / 31;
        putHalf(y, yOff + IQ2_XXS_D, d);
        float id = 1 / d;
        for (int ib = 0; ib < QK_K / 32; ++ib) {
            q2[2 * ib + 1] |= scaleIndex(id, scales[ib]) << 28;
        }
        for (int i = 0; i < q2.length; ++i) {
            putU32(y, yOff + IQ2_XXS_QS + 4 * i, q2[i]);
        }
    }

    // ------------------------------------------------------------------ IQ2_XS

    /** One IQ2_XS super-block: 16 sub-blocks of 16, a 9-bit grid index and 7 sign bits per 8 values. */
    private static void iq2Xs(IqGridSearch s, float[] x, int xOff, byte[] y, int yOff, float[] qw, int qwOff) {
        int[] q2 = new int[2 * (QK_K / 16)];
        float[] scales = new float[QK_K / 16];
        int[] signs = new int[2];
        float sigma2 = sumSquares(x, xOff, QK_K) / QK_K;
        float maxScale = 0;
        for (int ib = 0; ib < QK_K / 16; ++ib) {
            int xb = xOff + 16 * ib;
            s.weighted(x, xb, qw, qwOff + 16 * ib, sigma2);
            for (int k = 0; k < 2; ++k) {
                signs[k] = s.foldSigns(x, xb, k, true);
            }
            float max = s.max();
            s.clearLevels();
            if (max < GROUP_MAX_EPS) {
                continue;
            }
            float scale = s.scan(max, max / (2 * KMAXQ_IQ2 - 1), 9, 0.1f, true);
            scale = s.refit(scale, IqGridSearch.Refit.OFF_GRID);
            if (scale < 0) {
                scale = -scale;
                flipSigns(signs, 127);
            }
            s.gridIndices();
            for (int k = 0; k < 2; ++k) {
                q2[2 * ib + k] = s.index[k] | (signs[k] << 9);
            }
            scales[ib] = scale;
            maxScale = Math.max(maxScale, scale);
        }
        Arrays.fill(y, yOff, yOff + IQ2_XS_BYTES, (byte) 0);
        if (maxScale == 0) {
            return;
        }
        float d = maxScale / 31;
        putHalf(y, yOff + IQ2_XS_D, d);
        float id = 1 / d;
        for (int ib = 0; ib < QK_K / 16; ++ib) {
            y[yOff + IQ2_XS_SCALES + ib / 2] |= (byte) (scaleIndex(id, scales[ib]) << (4 * (ib % 2)));
        }
        for (int i = 0; i < q2.length; ++i) {
            y[yOff + IQ2_XS_QS + 2 * i] = (byte) q2[i];
            y[yOff + IQ2_XS_QS + 2 * i + 1] = (byte) (q2[i] >>> 8);
        }
    }

    // ------------------------------------------------------------------ IQ2_S

    /** One IQ2_S super-block: 16 sub-blocks of 16, a 10-bit grid index and 8 sign bits per 8 values. */
    private static void iq2S(IqGridSearch s, float[] x, int xOff, byte[] y, int yOff, float[] qw, int qwOff) {
        Arrays.fill(y, yOff, yOff + IQ2_S_BYTES, (byte) 0);
        float[] scales = new float[QK_K / 16];
        int[] signs = new int[2];
        float sigma2 = 2 * sumSquares(x, xOff, QK_K) / QK_K;
        float maxScale = 0;
        for (int ib = 0; ib < QK_K / 16; ++ib) {
            int xb = xOff + 16 * ib;
            if (qw != null) {
                s.weighted(x, xb, qw, qwOff + 16 * ib, sigma2);
            } else {
                s.unweighted(x, xb, 0.25f * sigma2);
            }
            for (int k = 0; k < 2; ++k) {
                signs[k] = s.foldSigns(x, xb, k, false);
            }
            float max = s.max();
            s.clearLevels();
            if (max < GROUP_MAX_EPS_IQ2_S) {
                continue;
            }
            float scale = s.scan(max, max / (2 * KMAXQ_IQ2 - 1), 9, 0.1f, true);
            scale = s.refit(scale, IqGridSearch.Refit.OFF_GRID);
            if (scale < 0) {
                scale = -scale;
                flipSigns(signs, 255);
            }
            s.gridIndices();
            for (int k = 0; k < 2; ++k) {
                int i8 = 2 * ib + k;
                y[yOff + IQ2_S_QS + i8] = (byte) s.index[k];
                y[yOff + IQ2_S_QH + i8 / 4] |= (byte) ((s.index[k] >>> 8) << (2 * (i8 % 4)));
                y[yOff + IQ2_S_SIGNS + i8] = (byte) signs[k];
            }
            scales[ib] = scale;
            maxScale = Math.max(maxScale, scale);
        }
        if (maxScale == 0) {
            return;
        }
        float d = maxScale / 31;
        putHalf(y, yOff + IQ2_S_D, d * 0.9875f);
        float id = 1 / d;
        for (int ib = 0; ib < QK_K / 16; ++ib) {
            y[yOff + IQ2_S_SCALES + ib / 2] |= (byte) (scaleIndex(id, scales[ib]) << (4 * (ib % 2)));
        }
    }

    // ------------------------------------------------------------------ IQ3_XXS

    /** One IQ3_XXS super-block: 8 sub-blocks of 32, eight 4-lane grid bytes and a 32-bit sign/scale word each. */
    private static void iq3Xxs(IqGridSearch s, float[] x, int xOff, byte[] y, int yOff, float[] qw, int qwOff) {
        byte[] q3 = new byte[QK_K / 4];
        int[] sas = new int[QK_K / 32];
        float[] scales = new float[QK_K / 32];
        int[] signs = new int[4];
        float sigma2 = 2 * sumSquares(x, xOff, QK_K) / QK_K;
        float maxScale = 0;
        for (int ib = 0; ib < QK_K / 32; ++ib) {
            int xb = xOff + 32 * ib;
            if (qw != null) {
                s.weighted(x, xb, qw, qwOff + 32 * ib, sigma2);
            } else {
                s.unweighted(x, xb, 0f);
            }
            for (int k = 0; k < 4; ++k) {
                signs[k] = s.foldSigns(x, xb, k, true);
            }
            float max = s.max();
            s.clearLevels();
            if (max < GROUP_MAX_EPS_IQ3_XXS) {
                continue;
            }
            float scale = s.scan(max, max / (2 * KMAXQ_IQ3 - 1), 15, 0.2f, true);
            scale = s.refit(scale, IqGridSearch.Refit.OFF_GRID);
            if (scale < 0) {
                scale = -scale;
                flipSigns(signs, 127);
            }
            s.gridIndices();
            for (int k = 0; k < 8; ++k) {
                q3[8 * ib + k] = (byte) s.index[k];
            }
            sas[ib] = signs[0] | (signs[1] << 7) | (signs[2] << 14) | (signs[3] << 21);
            scales[ib] = scale;
            maxScale = Math.max(maxScale, scale);
        }
        Arrays.fill(y, yOff, yOff + IQ3_XXS_BYTES, (byte) 0);
        if (maxScale == 0) {
            return;
        }
        float d = maxScale / 31;
        putHalf(y, yOff + IQ3_XXS_D, d * 1.0125f);
        float id = 1 / d;
        System.arraycopy(q3, 0, y, yOff + IQ3_XXS_QS, q3.length);
        for (int ib = 0; ib < QK_K / 32; ++ib) {
            putU32(y, yOff + IQ3_XXS_SAS + 4 * ib, sas[ib] | (scaleIndex(id, scales[ib]) << 28));
        }
    }

    // ------------------------------------------------------------------ IQ3_S

    /** One IQ3_S super-block: 8 sub-blocks of 32, a 9-bit grid index per 4 values and 8 sign bits per 8. */
    private static void iq3S(IqGridSearch s, float[] x, int xOff, byte[] y, int yOff, float[] qw, int qwOff) {
        Arrays.fill(y, yOff, yOff + IQ3_S_BYTES, (byte) 0);
        float[] scales = new float[QK_K / 32];
        int[] signs = new int[4];
        float sigma2 = 2 * sumSquares(x, xOff, QK_K) / QK_K;
        float maxScale = 0;
        for (int ib = 0; ib < QK_K / 32; ++ib) {
            int xb = xOff + 32 * ib;
            if (qw != null) {
                s.weighted(x, xb, qw, qwOff + 32 * ib, sigma2);
            } else {
                s.unweighted(x, xb, 0f);
            }
            for (int k = 0; k < 4; ++k) {
                signs[k] = s.foldSigns(x, xb, k, false);
            }
            float max = s.max();
            s.clearLevels();
            if (max == 0) {
                continue;
            }
            float scale = s.scan(max, max / (2 * KMAXQ_IQ3 - 1), 9, 0.2f, false);
            scale = s.refit(scale, IqGridSearch.Refit.ALL_WHEN_OFF_GRID);
            if (scale < 0) {
                scale = -scale;
                flipSigns(signs, 255);
            }
            s.gridIndices();
            for (int k = 0; k < 8; ++k) {
                y[yOff + IQ3_S_QS + 8 * ib + k] = (byte) s.index[k];
                y[yOff + IQ3_S_QH + ib] |= (byte) ((s.index[k] >>> 8) << k);
            }
            for (int k = 0; k < 4; ++k) {
                y[yOff + IQ3_S_SIGNS + 4 * ib + k] = (byte) signs[k];
            }
            scales[ib] = scale;
            maxScale = Math.max(maxScale, scale);
        }
        if (maxScale == 0) {
            return;
        }
        float d = maxScale / 31;
        putHalf(y, yOff + IQ3_S_D, d * 1.033f);
        float id = 1 / d;
        for (int ib = 0; ib < QK_K / 32; ib += 2) {
            int l1 = scaleIndex(id, scales[ib]);
            int l2 = scaleIndex(id, scales[ib + 1]);
            y[yOff + IQ3_S_SCALES + ib / 2] = (byte) (l1 | (l2 << 4));
        }
    }

    /** Inverts the sign patterns within {@code mask}: the sign flip of a negative fitted scale. */
    private static void flipSigns(int[] signs, int mask) {
        for (int k = 0; k < signs.length; ++k) {
            signs[k] = ~signs[k] & mask;
        }
    }
}

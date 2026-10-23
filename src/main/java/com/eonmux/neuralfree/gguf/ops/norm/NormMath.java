package com.eonmux.neuralfree.gguf.ops.norm;

import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.ops.norm.NormKernels.MixMode;

import java.util.Objects;

/**
 * CPU reference of the normalizations, residual mixes and row scales: the Java twin of {@code kernels/llm/norm_ext.cl}
 * with the same index formulas, and the single home of the norm math every CPU block uses.
 *
 * <p>Arithmetic follows the llama.cpp CPU ops in single precision with double accumulation, as ggml does:</p>
 * <ul>
 *   <li>RMS (ggml-cpu/ops.cpp:4191): {@code mean = (float)(sum(x*x) / n)}, {@code scale = 1/sqrtf(mean + eps)},
 *       {@code y = x * scale};</li>
 *   <li>LAYER (ops.cpp:4094): {@code mean = (float) sum(x) / n}, {@code var = (float)(sum((x - mean)^2) / n)},
 *       {@code y = (x - mean) * (1/sqrtf(var + eps))};</li>
 *   <li>L2_CLAMP (ops.cpp:4570): {@code y = x * (1 / fmaxf(sqrtf(sum(x*x)), eps))};</li>
 *   <li>L2_GDN (src/models/models.h:14 {@code build_gdn_l2_norm}): RMS with {@code eps / n}, then
 *       {@code * 1/sqrtf(n)}, i.e. {@code x / sqrt(sum(x*x) + eps)};</li>
 *   <li>weight and bias afterwards, {@code y * w + b} (llama-graph.cpp:1757 {@code build_norm});</li>
 *   <li>GroupNorm (ops.cpp:4475) over channel groups of a channel-major {@code [C][L]} tensor, two passes.</li>
 * </ul>
 * The GPU kernels reduce in float, in a different order, so results agree to float rounding. Stateless and
 * thread-safe; callers shard rows for parallelism.
 */
public final class NormMath {

    private NormMath() {
    }

    // ============================================================== row norms

    /**
     * Normalizes one row: {@code y[yOff + i] = norm(x[xOff ..])[i] * w[wOff + i] + b[bOff + i]} for {@code i < len},
     * the weight and bias applied when non-null. {@code y} may be {@code x} at the same offset (in place).
     *
     * @throws IllegalArgumentException for a non-row-norm kind, a negative or non-finite eps, a range outside its
     *                                  array, or partially overlapping x / y ranges
     */
    public static void rowNorm(NormKind kind, float eps, float[] x, int xOff, float[] y, int yOff, int len,
                               float[] w, int wOff, float[] b, int bOff) {
        checkRowNorm(kind, eps);
        checkPositive(len, "row length");
        checkRange("norm input", x, xOff, len);
        checkRange("norm output", y, yOff, len);
        checkSameOrDisjoint("norm", x, xOff, y, yOff, len);
        if (w != null) {
            checkRange("norm weight", w, wOff, len);
        }
        if (b != null) {
            checkRange("norm bias", b, bOff, len);
        }
        normalizeRow(kind, eps, x, xOff, y, yOff, len, w, wOff, b, bOff);
    }

    /**
     * CPU twin of {@code LLM_RowNorm}: row {@code r < nRows} of {@code rowLen} floats at {@code x[xOff + r*rowStride]}
     * is normalized into {@code y[yOff + r*rowStride]}, then multiplied by row {@code r % wRows} of {@code w} (a
     * {@code [wRows][rowLen]} array from index 0) and offset by the same row of {@code b}, each when non-null.
     * {@code wRows = 1} shares one vector; {@code wRows = n_head} gives per-head QK norms. In place when
     * {@code y == x} and {@code yOff == xOff}.
     *
     * @throws IllegalArgumentException for a non-row-norm kind, invalid eps or shape, a stride below the row length
     *                                  with several rows, an out-of-range view, partially overlapping x / y views,
     *                                  or a weight / bias shorter than {@code wRows * rowLen}
     */
    public static void rowNorm(NormKind kind, float eps, int nRows, int rowLen, int rowStride, int wRows,
                               float[] x, int xOff, float[] w, float[] b, float[] y, int yOff) {
        checkRowNorm(kind, eps);
        checkNonNegative(nRows, "row count");
        checkPositive(rowLen, "row length");
        if (wRows < 1) {
            throw new IllegalArgumentException("weight rows must be at least 1, got " + wRows);
        }
        long ext = extent(nRows, rowStride, rowLen, "norm");
        checkRange("norm input", x, xOff, ext);
        checkRange("norm output", y, yOff, ext);
        checkSameOrDisjoint("norm", x, xOff, y, yOff, ext);
        long wLen = (long) wRows * rowLen;
        if (w != null) {
            checkRange("norm weight", w, 0, wLen);
        }
        if (b != null) {
            checkRange("norm bias", b, 0, wLen);
        }
        for (int r = 0; r < nRows; r++) {
            int wb = (r % wRows) * rowLen;
            normalizeRow(kind, eps, x, xOff + r * rowStride, y, yOff + r * rowStride, rowLen, w, wb, b, wb);
        }
    }

    /**
     * The factor that multiplies a centered row for {@code kind}: {@code 1/sqrtf(mean(x^2) + eps)} (RMS),
     * {@code 1/sqrtf(var + eps)} (LAYER), {@code 1/fmaxf(sqrtf(sum(x^2)), eps)} (L2_CLAMP),
     * {@code 1/sqrtf(mean(x^2) + eps/n)} (L2_GDN, before its {@code 1/sqrtf(n)} post-scale). {@code mean} is the
     * row mean for LAYER and ignored otherwise.
     */
    static float rowScale(NormKind kind, float eps, float[] x, int off, int len, float mean) {
        switch (kind) {
            case RMS:
                return 1.0f / sqrtf((float) (sumSquares(x, off, len) / len) + eps);
            case LAYER: {
                double s = 0.0;
                for (int i = 0; i < len; i++) {
                    float v = x[off + i] - mean;
                    s += v * v;
                }
                float var = (float) (s / len);
                return 1.0f / sqrtf(var + eps);
            }
            case L2_CLAMP:
                return 1.0f / Math.max(sqrtf((float) sumSquares(x, off, len)), eps);
            case L2_GDN:
                return 1.0f / sqrtf((float) (sumSquares(x, off, len) / len) + eps / (float) len);
            default:
                throw new IllegalArgumentException(kind + " is not a row norm");
        }
    }

    /** {@code (float) sum(x) / n} as ggml_vec_sum_f32 then the float division of ggml_compute_forward_norm_f32. */
    static float rowMean(float[] x, int off, int len) {
        double s = 0.0;
        for (int i = 0; i < len; i++) {
            s += x[off + i];
        }
        return (float) s / (float) len;
    }

    private static void normalizeRow(NormKind kind, float eps, float[] x, int xOff, float[] y, int yOff, int len,
                                     float[] w, int wOff, float[] b, int bOff) {
        float mean = kind == NormKind.LAYER ? rowMean(x, xOff, len) : 0.0f;
        float scale = rowScale(kind, eps, x, xOff, len, mean);
        float post = kind == NormKind.L2_GDN ? 1.0f / sqrtf((float) len) : 1.0f;
        boolean gdn = kind == NormKind.L2_GDN;
        for (int i = 0; i < len; i++) {
            float v = (x[xOff + i] - mean) * scale;
            if (gdn) {
                v *= post;
            }
            if (w != null) {
                v *= w[wOff + i];
            }
            if (b != null) {
                v += b[bOff + i];
            }
            y[yOff + i] = v;
        }
    }

    // ============================================================== grouped norms

    /**
     * CPU twin of {@code LLM_GroupRms} (mamba2-family {@code ssm_norm}): token {@code t < nTokens}, group
     * {@code g < nGroup}, {@code gs = dInner / nGroup}, {@code i < gs}:
     * {@code y[yOff + t*dInner + g*gs + i] = x[xOff + t*xStride + g*gs + i] * rsqrt(mean_g(x^2) + eps) * w[g*gs + i]}.
     * {@code y} is dense; it may be {@code x} when both views coincide ({@code yOff == xOff},
     * {@code xStride == dInner}).
     *
     * @throws IllegalArgumentException when {@code dInner} is not a positive multiple of {@code nGroup}, eps is
     *                                  invalid, a range lies outside its array, or x and y overlap otherwise
     */
    public static void groupRms(int nTokens, int dInner, int nGroup, float eps, float[] x, int xOff, int xStride,
                                float[] w, float[] y, int yOff) {
        checkNonNegative(nTokens, "token count");
        checkPositive(nGroup, "group count");
        if (dInner <= 0 || dInner % nGroup != 0) {
            throw new IllegalArgumentException("d_inner " + dInner + " is not a positive multiple of the " + nGroup
                    + " groups");
        }
        checkEps(eps);
        long xExt = extent(nTokens, xStride, dInner, "grouped norm");
        long yExt = (long) nTokens * dInner;
        checkRange("grouped norm input", x, xOff, xExt);
        checkRange("grouped norm output", y, yOff, yExt);
        checkRange("grouped norm weight", Objects.requireNonNull(w, "grouped norm weight"), 0, dInner);
        boolean coincide = x == y && xOff == yOff && (xStride == dInner || nTokens <= 1);
        if (x == y && !coincide && overlaps(xOff, xExt, yOff, yExt)) {
            throw new IllegalArgumentException("grouped norm output aliases its input with a different layout");
        }
        int gs = dInner / nGroup;
        for (int t = 0; t < nTokens; t++) {
            for (int g = 0; g < nGroup; g++) {
                int xb = xOff + t * xStride + g * gs;
                int yb = yOff + t * dInner + g * gs;
                int wb = g * gs;
                float scale = rowScale(NormKind.RMS, eps, x, xb, gs, 0.0f);
                for (int i = 0; i < gs; i++) {
                    y[yb + i] = x[xb + i] * scale * w[wb + i];
                }
            }
        }
    }

    /**
     * CPU twin of {@code LLM_GroupNorm} over a dense channel-major {@code [nCh][len]} tensor (wavtokenizer posnet):
     * {@code cpg = ceil(nCh / nGroups)}; group {@code g} covers channels {@code [g*cpg, min(g*cpg + cpg, nCh))} and
     * {@code y[c*len + t] = (x[c*len + t] - mean_g) * (1/sqrtf(var_g + eps)) * w[c] + b[c]}, mean and biased variance
     * over the whole group (ggml-cpu/ops.cpp:4475-4547). {@code y} must not overlap {@code x}.
     *
     * @throws IllegalArgumentException for a non-positive shape, invalid eps, a range outside its array, or
     *                                  overlapping x / y
     */
    public static void groupNorm(int nCh, int len, int nGroups, float eps, float[] x, int xOff,
                                 float[] w, float[] b, float[] y, int yOff) {
        checkPositive(nCh, "channel count");
        checkPositive(len, "length");
        checkPositive(nGroups, "group count");
        checkEps(eps);
        long n = (long) nCh * len;
        checkRange("group norm input", x, xOff, n);
        checkRange("group norm output", y, yOff, n);
        checkRange("group norm weight", Objects.requireNonNull(w, "group norm weight"), 0, nCh);
        checkRange("group norm bias", Objects.requireNonNull(b, "group norm bias"), 0, nCh);
        if (x == y && overlaps(xOff, n, yOff, n)) {
            throw new IllegalArgumentException("group norm output overlaps its input");
        }
        int cpg = (nCh + nGroups - 1) / nGroups;
        for (int g = 0; g < nGroups; g++) {
            int c0 = g * cpg;
            int c1 = Math.min(c0 + cpg, nCh);
            if (c1 <= c0) {
                continue;
            }
            long cnt = (long) (c1 - c0) * len;
            double sum = 0.0;
            for (int c = c0; c < c1; c++) {
                double sumr = 0.0;
                for (int t = 0; t < len; t++) {
                    sumr += x[xOff + c * len + t];
                }
                sum += sumr;
            }
            float mean = (float) (sum / cnt);
            double sum2 = 0.0;
            for (int c = c0; c < c1; c++) {
                double sumr = 0.0;
                for (int t = 0; t < len; t++) {
                    float v = x[xOff + c * len + t] - mean;
                    sumr += v * v;
                }
                sum2 += sumr;
            }
            float variance = (float) (sum2 / cnt);
            float scale = 1.0f / sqrtf(variance + eps);
            for (int c = c0; c < c1; c++) {
                float wc = w[c];
                float bc = b[c];
                for (int t = 0; t < len; t++) {
                    int e = c * len + t;
                    y[yOff + e] = (x[xOff + e] - mean) * scale * wc + bc;
                }
            }
        }
    }

    // ============================================================== residuals and scales

    /**
     * CPU twin of {@code LLM_ResidualMix}: for {@code i < n}, {@code a = inp[inpOff + i]}, {@code c = cur[curOff + i]},
     * {@code out[outOff + i]} = {@code a + c} (ADD), {@code a + s*c} (SCALED), {@code a*alpha[i % abLen] +
     * c*beta[i % abLen]} (ALPHA_BETA, alpha / beta from index 0) or {@code c + s*a} (NORMED_INPUT_SCALED).
     * {@code out} may be {@code inp} or {@code cur} at the same offset.
     *
     * @throws IllegalArgumentException when a range lies outside its array, out partially overlaps an input, or
     *                                  ALPHA_BETA lacks vectors of {@code abLen >= 1} floats
     */
    public static void residualMix(MixMode mode, int n, float s, float[] inp, int inpOff, float[] cur, int curOff,
                                   float[] alpha, float[] beta, int abLen, float[] out, int outOff) {
        Objects.requireNonNull(mode, "mode");
        checkNonNegative(n, "element count");
        checkRange("residual input", inp, inpOff, n);
        checkRange("residual branch", cur, curOff, n);
        checkRange("residual output", out, outOff, n);
        checkSameOrDisjoint("residual input", inp, inpOff, out, outOff, n);
        checkSameOrDisjoint("residual branch", cur, curOff, out, outOff, n);
        switch (mode) {
            case ADD:
                for (int i = 0; i < n; i++) {
                    out[outOff + i] = inp[inpOff + i] + cur[curOff + i];
                }
                break;
            case SCALED:
                for (int i = 0; i < n; i++) {
                    out[outOff + i] = inp[inpOff + i] + s * cur[curOff + i];
                }
                break;
            case ALPHA_BETA:
                checkPositive(abLen, "alpha/beta length");
                checkRange("residual alpha", Objects.requireNonNull(alpha, "residual alpha"), 0, abLen);
                checkRange("residual beta", Objects.requireNonNull(beta, "residual beta"), 0, abLen);
                for (int i = 0; i < n; i++) {
                    int k = i % abLen;
                    out[outOff + i] = inp[inpOff + i] * alpha[k] + cur[curOff + i] * beta[k];
                }
                break;
            case NORMED_INPUT_SCALED:
                for (int i = 0; i < n; i++) {
                    out[outOff + i] = cur[curOff + i] + s * inp[inpOff + i];
                }
                break;
            default:
                throw new IllegalArgumentException("unhandled residual mix mode " + mode);
        }
    }

    /**
     * CPU twin of {@code LLM_MulRowScale}: {@code x[xOff + r*rowLen + c] *= scale[scaleOff + (scaleLen == 1 ? 0 : c)]}
     * for {@code r < nRows}, {@code c < rowLen}.
     *
     * @throws IllegalArgumentException when {@code scaleLen} is neither 1 nor {@code rowLen} or a range lies outside
     *                                  its array
     */
    public static void mulRowScale(int nRows, int rowLen, float[] scale, int scaleOff, int scaleLen,
                                   float[] x, int xOff) {
        checkNonNegative(nRows, "row count");
        checkPositive(rowLen, "row length");
        if (scaleLen != 1 && scaleLen != rowLen) {
            throw new IllegalArgumentException("scale length " + scaleLen + " must be 1 or the row length " + rowLen);
        }
        checkRange("row scale", scale, scaleOff, scaleLen);
        checkRange("row-scaled tensor", x, xOff, (long) nRows * rowLen);
        for (int r = 0; r < nRows; r++) {
            int base = xOff + r * rowLen;
            if (scaleLen == 1) {
                float s = scale[scaleOff];
                for (int c = 0; c < rowLen; c++) {
                    x[base + c] *= s;
                }
            } else {
                for (int c = 0; c < rowLen; c++) {
                    x[base + c] *= scale[scaleOff + c];
                }
            }
        }
    }

    // ============================================================== helpers

    /** {@code sum((double)(x*x))}, the float product accumulated in double as ggml does. */
    private static double sumSquares(float[] x, int off, int len) {
        double s = 0.0;
        for (int i = 0; i < len; i++) {
            float v = x[off + i];
            s += v * v;
        }
        return s;
    }

    /** C {@code sqrtf}: the correctly rounded single-precision square root. */
    private static float sqrtf(float v) {
        return (float) Math.sqrt(v);
    }

    private static void checkRowNorm(NormKind kind, float eps) {
        Objects.requireNonNull(kind, "kind");
        if (!kind.isRowNorm()) {
            throw new IllegalArgumentException(kind + " is not a row norm; group norms have their own entry points");
        }
        checkEps(eps);
    }

    private static void checkEps(float eps) {
        if (!(eps >= 0.0f) || Float.isInfinite(eps)) {
            throw new IllegalArgumentException("norm eps must be finite and non-negative, got " + eps);
        }
    }

    private static void checkPositive(int v, String what) {
        if (v <= 0) {
            throw new IllegalArgumentException(what + " must be positive, got " + v);
        }
    }

    private static void checkNonNegative(int v, String what) {
        if (v < 0) {
            throw new IllegalArgumentException(what + " must be non-negative, got " + v);
        }
    }

    /** Floats spanned by {@code rows} rows of {@code cols} at {@code stride}; the stride must cover a row. */
    private static long extent(int rows, int stride, int cols, String what) {
        if (rows > 1 && stride < cols) {
            throw new IllegalArgumentException(what + " stride " + stride + " is below the row length " + cols);
        }
        return rows == 0 ? 0 : (long) (rows - 1) * stride + cols;
    }

    private static void checkRange(String name, float[] a, long off, long n) {
        Objects.requireNonNull(a, name);
        if (n < 0 || off < 0 || off + n > a.length) {
            throw new IllegalArgumentException(name + " range [" + off + ", " + (off + n)
                    + ") is outside an array of length " + a.length);
        }
    }

    private static boolean overlaps(long aOff, long aLen, long bOff, long bLen) {
        return aLen > 0 && bLen > 0 && aOff < bOff + bLen && bOff < aOff + aLen;
    }

    /** Rejects two ranges of one array that overlap without coinciding. */
    private static void checkSameOrDisjoint(String what, float[] a, long aOff, float[] b, long bOff, long len) {
        if (a == b && aOff != bOff && overlaps(aOff, len, bOff, len)) {
            throw new IllegalArgumentException(what + " ranges overlap at offsets " + aOff + " and " + bOff
                    + " without coinciding");
        }
    }
}

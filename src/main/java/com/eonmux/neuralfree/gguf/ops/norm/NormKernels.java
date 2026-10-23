package com.eonmux.neuralfree.gguf.ops.norm;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.arch.ResidualKind;
import com.eonmux.neuralfree.gguf.kernels.LlmKernelConstants;
import com.eonmux.neuralfree.gguf.ops.act.ActKernels;

import java.util.Objects;

/**
 * Names, argument indices, argument lists and launch geometry of the norm-extension kernels in
 * {@code kernels/llm/norm_ext.cl} (GLSL {@code norm_ext.comp}); the CPU reference is {@link NormMath}:
 * <pre>
 * LLM_RowNorm    (i n_rows(dyn), i row_len, i row_stride, i mode, f eps, i flags, i w_rows, i x_off, i y_off,
 *                 Rn x, R w, R b, Wn y)
 * LLM_GroupRms   (i n_tokens(dyn), i d_inner, i n_group, f eps, i x_stride, i x_off, Rn x, R w, Wn y)
 * LLM_GroupNorm  (i n_ch, i len, i n_groups, f eps, R x, R w, R b, W y)
 * LLM_ResidualMix(i n(dyn), i mode, f s, i inp_off, i cur_off, i out_off, i ab_len,
 *                 Rn inp, Rn cur, R alpha, R beta, Wn out)
 * LLM_MulRowScale(i n_rows(dyn), i row_len, i scale_len, R scale, RWn x)
 * </pre>
 * Offsets and strides are in floats. {@code LLM_ResidualMix} carries {@code out_off} and the alpha/beta broadcast
 * period {@code ab_len}: the muse-spark alpha/beta vectors hold one row of
 * {@code n_embd} and must broadcast over every token of a prefill step.
 *
 * <p>Dispatch: the row reductions ({@code LLM_RowNorm}, {@code LLM_GroupRms}, {@code LLM_GroupNorm}) run one
 * work-group of {@code L} work-items per normalized vector, {@code global = groups * L}, {@code local = {L}}
 * ({@link #reduceGlobal}), where {@code L} is a power of two no larger than {@link LlmKernelConstants#LLM_DIMX}
 * and a multiple of 32 ({@link #checkReduceLocal}); the element-wise kernels ({@code LLM_ResidualMix},
 * {@code LLM_MulRowScale}) run one work-item per element, {@code global = roundUp(n, L)}
 * ({@link ActKernels#global1d}).
 */
public final class NormKernels {

    public static final String ROW_NORM = "LLM_RowNorm";
    public static final String GROUP_RMS = "LLM_GroupRms";
    public static final String GROUP_NORM = "LLM_GroupNorm";
    public static final String RESIDUAL_MIX = "LLM_ResidualMix";
    public static final String MUL_ROW_SCALE = "LLM_MulRowScale";

    /** {@code LLM_RowNorm} flag: a weight is bound. */
    public static final int FLAG_HAS_W = 1;
    /** {@code LLM_RowNorm} flag: a bias is bound. */
    public static final int FLAG_HAS_B = 2;

    public static final int ARG_ROW_NORM_N_ROWS = 0;
    public static final int ARG_ROW_NORM_ROW_LEN = 1;
    public static final int ARG_ROW_NORM_ROW_STRIDE = 2;
    public static final int ARG_ROW_NORM_MODE = 3;
    public static final int ARG_ROW_NORM_EPS = 4;
    public static final int ARG_ROW_NORM_FLAGS = 5;
    public static final int ARG_ROW_NORM_W_ROWS = 6;
    public static final int ARG_ROW_NORM_X_OFF = 7;
    public static final int ARG_ROW_NORM_Y_OFF = 8;
    public static final int ARG_ROW_NORM_X = 9;
    public static final int ARG_ROW_NORM_W = 10;
    public static final int ARG_ROW_NORM_B = 11;
    public static final int ARG_ROW_NORM_Y = 12;
    public static final int ROW_NORM_ARG_COUNT = 13;

    public static final int ARG_GROUP_RMS_N_TOKENS = 0;
    public static final int ARG_GROUP_RMS_D_INNER = 1;
    public static final int ARG_GROUP_RMS_N_GROUP = 2;
    public static final int ARG_GROUP_RMS_EPS = 3;
    public static final int ARG_GROUP_RMS_X_STRIDE = 4;
    public static final int ARG_GROUP_RMS_X_OFF = 5;
    public static final int ARG_GROUP_RMS_X = 6;
    public static final int ARG_GROUP_RMS_W = 7;
    public static final int ARG_GROUP_RMS_Y = 8;
    public static final int GROUP_RMS_ARG_COUNT = 9;

    public static final int ARG_GROUP_NORM_N_CH = 0;
    public static final int ARG_GROUP_NORM_LEN = 1;
    public static final int ARG_GROUP_NORM_N_GROUPS = 2;
    public static final int ARG_GROUP_NORM_EPS = 3;
    public static final int ARG_GROUP_NORM_X = 4;
    public static final int ARG_GROUP_NORM_W = 5;
    public static final int ARG_GROUP_NORM_B = 6;
    public static final int ARG_GROUP_NORM_Y = 7;
    public static final int GROUP_NORM_ARG_COUNT = 8;

    public static final int ARG_MIX_N = 0;
    public static final int ARG_MIX_MODE = 1;
    public static final int ARG_MIX_S = 2;
    public static final int ARG_MIX_INP_OFF = 3;
    public static final int ARG_MIX_CUR_OFF = 4;
    public static final int ARG_MIX_OUT_OFF = 5;
    public static final int ARG_MIX_AB_LEN = 6;
    public static final int ARG_MIX_INP = 7;
    public static final int ARG_MIX_CUR = 8;
    public static final int ARG_MIX_ALPHA = 9;
    public static final int ARG_MIX_BETA = 10;
    public static final int ARG_MIX_OUT = 11;
    public static final int RESIDUAL_MIX_ARG_COUNT = 12;

    public static final int ARG_ROW_SCALE_N_ROWS = 0;
    public static final int ARG_ROW_SCALE_ROW_LEN = 1;
    public static final int ARG_ROW_SCALE_SCALE_LEN = 2;
    public static final int ARG_ROW_SCALE_SCALE = 3;
    public static final int ARG_ROW_SCALE_X = 4;
    public static final int MUL_ROW_SCALE_ARG_COUNT = 5;

    /** Smallest local size of the row reductions: one full subgroup / CUDA warp. */
    public static final int MIN_REDUCE_LOCAL = 32;

    /** Residual combination of {@code LLM_ResidualMix}; codes are the kernel's {@code mode}. */
    public enum MixMode {
        /** {@code out = inp + cur}. */
        ADD(0),
        /** {@code out = inp + s * cur} (granite, minicpm residual scale). */
        SCALED(1),
        /** {@code out = inp * alpha + cur * beta} with {n_embd} vectors (muse-spark). */
        ALPHA_BETA(2),
        /** {@code out = cur + s * inp}, {@code inp} = the normed block input (minimax-01). */
        NORMED_INPUT_SCALED(3);

        private final int code;

        MixMode(int code) {
            this.code = code;
        }

        /** The kernel's {@code mode} argument. */
        public int code() {
            return code;
        }

        /**
         * The mode of a kernel code.
         *
         * @throws IllegalArgumentException when no mode has that code
         */
        public static MixMode fromCode(int code) {
            for (MixMode m : values()) {
                if (m.code == code) {
                    return m;
                }
            }
            throw new IllegalArgumentException("no residual mix mode has code " + code);
        }

        /**
         * The mix applying a residual of {@code kind} after one sub-block (the mixer or the FFN); the
         * muse-spark alpha/beta residual is {@link #ALPHA_BETA}, chosen from {@code ResidualSpec.alphaBeta()}.
         *
         * @throws IllegalArgumentException for PARALLEL and HYPER_CONNECTIONS, which the executor combines itself
         */
        public static MixMode forResidual(ResidualKind kind) {
            switch (Objects.requireNonNull(kind, "kind")) {
                case STANDARD:
                    return ADD;
                case SCALED:
                    return SCALED;
                case NORMED_INPUT_SCALED:
                    return NORMED_INPUT_SCALED;
                default:
                    throw new IllegalArgumentException(kind + " residuals are not a single residual mix");
            }
        }
    }

    private NormKernels() {
    }

    /**
     * Arguments of {@code LLM_RowNorm} in signature order: row {@code r < nRows} of {@code rowLen} floats is read at
     * {@code x[xOff + r*rowStride]} and written at {@code y[yOff + r*rowStride]}, normalized by {@code kind}, then
     * multiplied by weight row {@code r % wRows} of {@code w} and offset by the same row of {@code b} when those are
     * non-null. Absent operands bind {@code placeholder}. {@code y} may coincide with {@code x}.
     *
     * @throws IllegalArgumentException for a non-row-norm kind, a negative or non-finite eps, an invalid shape, a
     *                                  view outside its buffer, partially overlapping x/y views, or a weight / bias
     *                                  smaller than {@code wRows * rowLen}
     */
    public static Object[] rowNormArgs(NormKind kind, float eps, int nRows, int rowLen, int rowStride,
                                       GpuBuffer x, int xOff, GpuBuffer y, int yOff,
                                       GpuBuffer w, GpuBuffer b, int wRows, GpuBuffer placeholder) {
        Objects.requireNonNull(kind, "kind");
        if (!kind.isRowNorm()) {
            throw new IllegalArgumentException(kind + " is not an LLM_RowNorm mode");
        }
        checkEps(eps);
        checkPositive(rowLen, "row length");
        checkNonNegative(nRows, "row count");
        if (wRows < 1) {
            throw new IllegalArgumentException("weight rows must be at least 1, got " + wRows);
        }
        ActKernels.elementCount(nRows, rowStride);
        ActKernels.checkFloatView("norm input", x, xOff, rowStride, nRows, rowLen);
        ActKernels.checkFloatView("norm output", y, yOff, rowStride, nRows, rowLen);
        checkSameOrDisjoint("norm", x, xOff, y, yOff, extent(nRows, rowStride, rowLen));
        int flags = 0;
        if (w != null) {
            ActKernels.checkFloatView("norm weight", w, 0, rowLen, wRows, rowLen);
            flags |= FLAG_HAS_W;
        }
        if (b != null) {
            ActKernels.checkFloatView("norm bias", b, 0, rowLen, wRows, rowLen);
            flags |= FLAG_HAS_B;
        }
        GpuBuffer wArg = w != null ? w : Objects.requireNonNull(placeholder, "placeholder for the absent weight");
        GpuBuffer bArg = b != null ? b : Objects.requireNonNull(placeholder, "placeholder for the absent bias");
        return new Object[]{nRows, rowLen, rowStride, kind.rowNormMode(), eps, flags, wRows, xOff, yOff,
                x, wArg, bArg, y};
    }

    /**
     * Arguments of {@code LLM_GroupRms} in signature order: token {@code t < nTokens}, group {@code g < nGroup},
     * {@code gs = dInner / nGroup}: {@code y[t*dInner + g*gs + i] = rms(x[xOff + t*xStride + g*gs + i]) * w[g*gs + i]}.
     * {@code y} is dense; it may coincide with {@code x} only when {@code xOff == 0} and {@code xStride == dInner}.
     *
     * @throws IllegalArgumentException when {@code dInner} is not a positive multiple of {@code nGroup}, eps is
     *                                  invalid, a view lies outside its buffer, x and y alias otherwise, or the
     *                                  weight holds fewer than {@code dInner} floats
     */
    public static Object[] groupRmsArgs(int nTokens, int dInner, int nGroup, float eps,
                                        GpuBuffer x, int xOff, int xStride, GpuBuffer w, GpuBuffer y) {
        checkNonNegative(nTokens, "token count");
        checkPositive(nGroup, "group count");
        if (dInner <= 0 || dInner % nGroup != 0) {
            throw new IllegalArgumentException("d_inner " + dInner + " is not a positive multiple of the "
                    + nGroup + " groups");
        }
        checkEps(eps);
        ActKernels.elementCount(nTokens, Math.max(xStride, dInner));
        ActKernels.checkFloatView("grouped norm input", x, xOff, xStride, nTokens, dInner);
        ActKernels.checkFloatView("grouped norm output", y, 0, dInner, nTokens, dInner);
        ActKernels.checkFloatView("grouped norm weight", w, 0, dInner, 1, dInner);
        if (x == y && nTokens > 0 && !(xOff == 0 && (xStride == dInner || nTokens == 1))) {
            long xExt = extent(nTokens, xStride, dInner);
            if (overlaps(xOff, xExt, 0, (long) nTokens * dInner)) {
                throw new IllegalArgumentException("grouped norm output aliases its input with a different layout");
            }
        }
        return new Object[]{nTokens, dInner, nGroup, eps, xStride, xOff, x, w, y};
    }

    /**
     * Arguments of {@code LLM_GroupNorm} in signature order over a dense channel-major {@code [nCh][len]} tensor:
     * {@code nGroups} groups of {@code ceil(nCh / nGroups)} channels, each normalized over all its elements, then
     * {@code * w[c] + b[c]}. {@code x} and {@code y} are distinct buffers.
     *
     * @throws IllegalArgumentException for a non-positive shape, invalid eps, a buffer too small, or x == y
     */
    public static Object[] groupNormArgs(int nCh, int len, int nGroups, float eps,
                                         GpuBuffer x, GpuBuffer w, GpuBuffer b, GpuBuffer y) {
        checkPositive(nCh, "channel count");
        checkPositive(len, "length");
        checkPositive(nGroups, "group count");
        checkEps(eps);
        ActKernels.elementCount(nCh, len);
        ActKernels.checkFloatView("group norm input", x, 0, len, nCh, len);
        ActKernels.checkFloatView("group norm output", y, 0, len, nCh, len);
        ActKernels.checkFloatView("group norm weight", w, 0, nCh, 1, nCh);
        ActKernels.checkFloatView("group norm bias", b, 0, nCh, 1, nCh);
        if (x == y) {
            throw new IllegalArgumentException("group norm cannot run in place");
        }
        return new Object[]{nCh, len, nGroups, eps, x, w, b, y};
    }

    /**
     * Arguments of {@code LLM_ResidualMix} in signature order: for {@code i < n},
     * {@code out[outOff + i] = mix(inp[inpOff + i], cur[curOff + i])} per {@link MixMode}; ALPHA_BETA reads
     * {@code alpha[i % abLen]} and {@code beta[i % abLen]}. Other modes bind {@code placeholder} for alpha and beta.
     * {@code out} may coincide with {@code inp} or {@code cur} (same buffer and offset).
     *
     * @throws IllegalArgumentException when a view lies outside its buffer, out partially overlaps an input,
     *                                  {@code s} is not finite, or ALPHA_BETA lacks vectors of {@code abLen} floats
     *                                  with {@code n % abLen == 0}
     */
    public static Object[] residualMixArgs(MixMode mode, int n, float s,
                                           GpuBuffer inp, int inpOff, GpuBuffer cur, int curOff,
                                           GpuBuffer alpha, GpuBuffer beta, int abLen,
                                           GpuBuffer out, int outOff, GpuBuffer placeholder) {
        Objects.requireNonNull(mode, "mode");
        checkNonNegative(n, "element count");
        if (Float.isNaN(s) || Float.isInfinite(s)) {
            throw new IllegalArgumentException("residual scale must be finite, got " + s);
        }
        ActKernels.checkFloatView("residual input", inp, inpOff, n, 1, n);
        ActKernels.checkFloatView("residual branch", cur, curOff, n, 1, n);
        ActKernels.checkFloatView("residual output", out, outOff, n, 1, n);
        checkSameOrDisjoint("residual input", inp, inpOff, out, outOff, n);
        checkSameOrDisjoint("residual branch", cur, curOff, out, outOff, n);
        GpuBuffer a;
        GpuBuffer bt;
        int period;
        if (mode == MixMode.ALPHA_BETA) {
            checkPositive(abLen, "alpha/beta length");
            if (n % abLen != 0) {
                throw new IllegalArgumentException("element count " + n + " is not a multiple of the alpha/beta length "
                        + abLen);
            }
            ActKernels.checkFloatView("residual alpha", alpha, 0, abLen, 1, abLen);
            ActKernels.checkFloatView("residual beta", beta, 0, abLen, 1, abLen);
            a = alpha;
            bt = beta;
            period = abLen;
        } else {
            a = Objects.requireNonNull(placeholder, "placeholder for alpha and beta");
            bt = placeholder;
            period = 1;
        }
        return new Object[]{n, mode.code(), s, inpOff, curOff, outOff, period, inp, cur, a, bt, out};
    }

    /**
     * Arguments of {@code LLM_MulRowScale} in signature order: {@code x[r*rowLen + c] *= scale[scaleLen == 1 ? 0 : c]}
     * over a dense {@code [nRows][rowLen]} tensor, in place.
     *
     * @throws IllegalArgumentException when {@code scaleLen} is neither 1 nor {@code rowLen}, or a buffer is too
     *                                  small
     */
    public static Object[] mulRowScaleArgs(int nRows, int rowLen, GpuBuffer scale, int scaleLen, GpuBuffer x) {
        checkNonNegative(nRows, "row count");
        checkPositive(rowLen, "row length");
        if (scaleLen != 1 && scaleLen != rowLen) {
            throw new IllegalArgumentException("scale length " + scaleLen + " must be 1 or the row length " + rowLen);
        }
        ActKernels.elementCount(nRows, rowLen);
        ActKernels.checkFloatView("row scale", scale, 0, scaleLen, 1, scaleLen);
        ActKernels.checkFloatView("row-scaled tensor", x, 0, rowLen, nRows, rowLen);
        return new Object[]{nRows, rowLen, scaleLen, scale, x};
    }

    /**
     * Global size of a row reduction: {@code groups} work-groups of {@code local} work-items, at least one group.
     *
     * @throws IllegalArgumentException for an invalid local size, a negative group count, or more than 2^31 - 1
     *                                  work-items
     */
    public static long[] reduceGlobal(long groups, int local) {
        checkReduceLocal(local);
        if (groups < 0) {
            throw new IllegalArgumentException("work-group count must be non-negative, got " + groups);
        }
        long g = Math.max(1L, groups);
        if (g > Integer.MAX_VALUE / local) {
            throw new IllegalArgumentException(g + " work-groups of " + local + " exceed 2^31 - 1 work-items");
        }
        return new long[]{g * local};
    }

    /** Work-groups of {@code LLM_RowNorm}: one per row. */
    public static long rowNormGroups(int nRows) {
        return nRows;
    }

    /** Work-groups of {@code LLM_GroupRms}: one per (token, group). */
    public static long groupRmsGroups(int nTokens, int nGroup) {
        return (long) nTokens * nGroup;
    }

    /** Work-groups of {@code LLM_GroupNorm}: one per channel group. */
    public static long groupNormGroups(int nGroups) {
        return nGroups;
    }

    /**
     * Validates the local size of a row reduction: a power of two in
     * {@code [MIN_REDUCE_LOCAL, LlmKernelConstants.LLM_DIMX]} (the kernels' {@code NF_LOCAL partial[LLM_DIMX]} and
     * full CUDA warps).
     *
     * @throws IllegalArgumentException otherwise
     */
    public static void checkReduceLocal(int local) {
        if (local < MIN_REDUCE_LOCAL || local > LlmKernelConstants.LLM_DIMX || Integer.bitCount(local) != 1) {
            throw new IllegalArgumentException("row-reduce local size must be a power of two in [" + MIN_REDUCE_LOCAL
                    + ", " + LlmKernelConstants.LLM_DIMX + "], got " + local);
        }
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

    /** Floats spanned by {@code rows} rows of {@code cols} at {@code stride}. */
    private static long extent(long rows, long stride, long cols) {
        return rows == 0 ? 0 : (rows - 1) * stride + cols;
    }

    private static boolean overlaps(long aOff, long aLen, long bOff, long bLen) {
        return aLen > 0 && bLen > 0 && aOff < bOff + bLen && bOff < aOff + aLen;
    }

    /** Rejects two views of one buffer that overlap without coinciding. */
    private static void checkSameOrDisjoint(String what, GpuBuffer a, long aOff, GpuBuffer b, long bOff, long len) {
        if (a == b && aOff != bOff && overlaps(aOff, len, bOff, len)) {
            throw new IllegalArgumentException(what + " views overlap at offsets " + aOff + " and " + bOff
                    + " without coinciding");
        }
    }
}

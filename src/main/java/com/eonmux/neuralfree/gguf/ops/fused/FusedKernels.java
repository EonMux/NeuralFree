package com.eonmux.neuralfree.gguf.ops.fused;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.ops.norm.NormKernels;
import com.eonmux.neuralfree.gguf.ops.rope.RopeKernels;
import com.eonmux.neuralfree.gguf.ops.rope.RopeMath;
import com.eonmux.neuralfree.gguf.spec.RopeSpec;

import java.util.Objects;

/**
 * Names, argument indices, argument lists and launch geometry of the fused decode-path kernels in
 * {@code kernels/llm/fused.cl} (GLSL {@code fused.comp}) and the persistent multi-task kernel
 * {@code kernels/llm/multitask.cl}. Each fused kernel replaces a pair of kernels with the same arithmetic, except the
 * two store forms, which fold the normalization, the rotation and the cache-row write into one kernel:
 *
 * <pre>
 * LLM_RmsNormRope (i n_tokens(dyn), i n_heads, i head_stride, i row_stride, i row_len, i mode, i n_dims, i n_offs,
 *                  i pos_stride(dyn), i has_ff, i flags, i w_rows, i x_off, i y_off, i sect0..i sect3,
 *                  f eps, f theta_scale, f freq_scale, f ext_factor, f attn_factor, f corr_lo, f corr_hi,
 *                  R&lt;int&gt; pos, R ff, R w, R b, R x, RW y)
 * LLM_GdnNormGate (i n_tokens(dyn), i n_heads, i head_dim, i row_stride, i x_off, i z_off, i out_off, i w_rows,
 *                  f eps, R x, R w, R z, RW out)
 * LLM_HcPostNorm  (i n_tokens(dyn), i n_embd, i hc, i has_comb, i post_act, i has_w, i w_rows, i scope, i y_off,
 *                  i x_off, i post_off, i comb_off, i xout_off, i xn_off, f post_in_scale, f post_mag, f eps,
 *                  R y, R X, R post, R comb, W Xout, R gamma, W xn)
 * LLM_NormRopeStore (i n_tokens(dyn), i n_heads, i head_stride, i row_stride, i row_len, i mode, i n_dims, i n_offs,
 *                  i pos_stride(dyn), i has_ff, i flags, i w_rows, i x_off, i dst_head_elems, i dst_row_stride,
 *                  i dst_type, i sect0..i sect3, f eps, f theta_scale..f corr_hi,
 *                  R pos, R ff, R w, R b, R x, R int cell_idx, W dst)
 * LLM_RopeStore   (i n_tokens(dyn), i n_heads, i head_dim, i row_stride, i mode, i n_dims, i n_offs,
 *                  i pos_stride(dyn), i has_ff, i x_off, i dst_head_elems, i dst_row_stride, i dst_type,
 *                  i sect0..i sect3, f theta_scale..f corr_hi, R pos, R ff, R x, R int cell_idx, W dst)
 * </pre>
 *
 * Dispatch: the fused kernels run one workgroup of {@code L} work-items per row (or per token for
 * {@code LLM_HcPostNorm}), {@code global = rows * L}, {@code local = {L}}; {@code L} is the row-reduce local size
 * ({@link com.eonmux.neuralfree.gguf.gpu.forward.GpuOps#local()}).
 */
public final class FusedKernels {

    /** RMS norm followed by RoPE on the same rows ({@code LLM_RowNorm} + {@code LLM_RoPE_Ext}). */
    public static final String RMS_NORM_ROPE = "LLM_RmsNormRope";
    /** Gated DeltaNet output RMS norm and SiLU gate ({@code LLM_RowNorm} + {@code LLM_GLU}). */
    public static final String GDN_NORM_GATE = "LLM_GdnNormGate";
    /** RMS norm followed by RoPE and the KV-cache row write ({@code LLM_RowNorm} + {@code LLM_RoPE_Ext} + store). */
    public static final String NORM_ROPE_STORE = "LLM_NormRopeStore";
    /** RoPE followed by the KV-cache row write ({@code LLM_RoPE_Ext} + store). */
    public static final String ROPE_STORE = "LLM_RopeStore";
    /** Hyper-connection expansion followed by the row norm of each expanded stream ({@code LLM_HC_Post} + norm). */
    public static final String HC_POST_NORM = "LLM_HcPostNorm";
    /** Persistent kernel running independent small tasks of one step in one launch. */
    public static final String MULTI_TASK = "LLM_MultiTask";
    /** Tasks one {@link #MULTI_TASK} launch records ({@code NF_MT_MAX_TASKS}). */
    public static final int MULTI_TASK_MAX_TASKS = 4;
    /** Floats per task descriptor ({@code NF_MT_WORDS}). */
    public static final int MULTI_TASK_WORDS = 48;
    /** Buffer arguments of one {@link #MULTI_TASK} launch ({@code NF_MT_BUFS}). */
    public static final int MULTI_TASK_BUFFERS = 11;

    /** {@code LLM_RmsNormRope}: first per-step argument. */
    public static final int ROPE_ARG_N_TOKENS = 0;
    /** {@code LLM_RmsNormRope}: section stride of the positions (the step's token count). */
    public static final int ROPE_ARG_POS_STRIDE = 8;
    public static final int ROPE_ARG_N_HEADS = 1;
    public static final int ROPE_ARG_HEAD_STRIDE = 2;
    public static final int ROPE_ARG_ROW_STRIDE = 3;
    public static final int ROPE_ARG_ROW_LEN = 4;

    /** {@code LLM_GdnNormGate}: first per-step argument. */
    public static final int GATE_ARG_N_TOKENS = 0;
    public static final int GATE_ARG_N_HEADS = 1;
    public static final int GATE_ARG_HEAD_DIM = 2;
    public static final int GATE_ARG_ROW_STRIDE = 3;
    public static final int GATE_ARG_X_OFF = 4;
    public static final int GATE_ARG_Z_OFF = 5;
    public static final int GATE_ARG_OUT_OFF = 6;
    public static final int GATE_ARG_W_ROWS = 7;
    public static final int GATE_ARG_EPS = 8;
    public static final int GATE_ARG_X = 9;
    public static final int GATE_ARG_W = 10;
    public static final int GATE_ARG_Z = 11;
    public static final int GATE_ARG_OUT = 12;
    public static final int GATE_ARG_COUNT = 13;

    /** {@code LLM_HcPostNorm}: first per-step argument. */
    public static final int HC_ARG_N_TOKENS = 0;
    public static final int HC_ARG_N_EMBD = 1;
    public static final int HC_ARG_HC = 2;
    public static final int HC_ARG_HAS_COMB = 3;
    public static final int HC_ARG_POST_ACT = 4;
    public static final int HC_ARG_HAS_W = 5;
    public static final int HC_ARG_W_ROWS = 6;
    /** 0: one norm axis per stream row (gated site); 1: one axis per token over the whole expanded row. */
    public static final int HC_ARG_SCOPE = 7;
    public static final int HC_ARG_Y_OFF = 8;
    public static final int HC_ARG_X_OFF = 9;
    public static final int HC_ARG_POST_OFF = 10;
    public static final int HC_ARG_COMB_OFF = 11;
    public static final int HC_ARG_XOUT_OFF = 12;
    public static final int HC_ARG_XN_OFF = 13;
    public static final int HC_ARG_POST_IN_SCALE = 14;
    public static final int HC_ARG_POST_MAG = 15;
    public static final int HC_ARG_EPS = 16;
    public static final int HC_ARG_Y = 17;
    public static final int HC_ARG_X = 18;
    public static final int HC_ARG_POST = 19;
    public static final int HC_ARG_COMB = 20;
    public static final int HC_ARG_XOUT = 21;
    public static final int HC_ARG_GAMMA = 22;
    public static final int HC_ARG_XN = 23;
    public static final int HC_ARG_COUNT = 24;

    private FusedKernels() {
    }

    /** The epsilon of a norm, which must be a finite non-negative value. */
    private static void checkEps(String kernel, float eps) {
        if (!(eps >= 0f) || Float.isInfinite(eps)) {
            throw new IllegalArgumentException(kernel + " needs a finite non-negative epsilon, got " + eps);
        }
    }

    // ------------------------------------------------------------------ LLM_RmsNormRope

    /**
     * Arguments of {@link #RMS_NORM_ROPE} for {@code nTokens} rows of {@code nHeads} heads. Each head row of
     * {@code rowLen} values is normalized with {@code eps} and then rotated by {@code rope}; {@code flags} selects
     * the norm weight and bias ({@link NormKernels#FLAG_HAS_W}, {@link NormKernels#FLAG_HAS_B}) and {@code wRows}
     * their row count (one shared row when {@code wRows <= 1}). {@code x} and {@code y} may coincide.
     *
     * @throws IllegalArgumentException when the row geometry or the rotation does not fit
     */
    public static Object[] rmsNormRopeArgs(RopeSpec rope, int nTokens, int nHeads, int headStride, int rowStride,
                                           int rowLen, int nOffs, int posStride, int xOff, int yOff, int flags,
                                           int wRows, float eps, boolean hasFf, GpuBuffer pos, GpuBuffer ff,
                                           GpuBuffer w, GpuBuffer b, GpuBuffer x, GpuBuffer y) {
        Objects.requireNonNull(rope, "rope");
        if (nHeads <= 0 || rowLen <= 0 || rowStride <= 0 || headStride <= 0) {
            throw new IllegalArgumentException(RMS_NORM_ROPE + " needs positive head geometry, got heads "
                    + nHeads + ", row length " + rowLen + ", head stride " + headStride + ", row stride "
                    + rowStride);
        }
        RopeKernels.validateRopeExt(rope.type(), rope.nDims(), nOffs, nTokens, nHeads, headStride, rowStride,
                posStride, xOff);
        if ((long) nOffs + RopeMath.rotatedWidth(rope.type(), rope.nDims()) > rowLen) {
            throw new IllegalArgumentException(RMS_NORM_ROPE + ": rotated dims [" + nOffs + ", "
                    + (nOffs + RopeMath.rotatedWidth(rope.type(), rope.nDims()))
                    + ") exceed the normalized row of " + rowLen);
        }
        final int[] s = rope.sections();
        checkEps(RMS_NORM_ROPE, eps);
        return new Object[]{
                nTokens, nHeads, headStride, rowStride, rowLen, rope.type().ggmlMode(), rope.nDims(), nOffs,
                posStride, hasFf ? 1 : 0, flags, wRows, xOff, yOff, s[0], s[1], s[2], s[3], eps,
                rope.thetaScale(), rope.freqScale(), rope.extFactor(), rope.attnFactor(), rope.corrLo(),
                rope.corrHi(), Objects.requireNonNull(pos, "pos"), Objects.requireNonNull(ff, "ff"),
                Objects.requireNonNull(w, "w"), Objects.requireNonNull(b, "b"), Objects.requireNonNull(x, "x"),
                Objects.requireNonNull(y, "y")};
    }

    /** Work-items of {@link #RMS_NORM_ROPE}: one workgroup per (token, head). */
    public static long rmsNormRopeGlobal(int nTokens, int nHeads, int local) {
        return (long) nTokens * nHeads * local;
    }

    // ------------------------------------------------------------------ LLM_NormRopeStore / LLM_RopeStore

    /** Destination type of a fused KV store: one 32-bit float per cache element. */
    public static final int KV_STORE_F32 = 0;
    /** Destination type of a fused KV store: one F16 per cache element. */
    public static final int KV_STORE_F16 = 1;

    /** {@code LLM_NormRopeStore}: first per-step argument. */
    public static final int STORE_ARG_N_TOKENS = 0;
    /** {@code LLM_NormRopeStore}: section stride of the positions (the step's token count). */
    public static final int STORE_ARG_POS_STRIDE = 8;
    /** {@code cell_idx} of {@link #NORM_ROPE_STORE} (the destination cache row per token). */
    public static final int STORE_ARG_CELL = 32;
    /** {@code dst} (the cache buffer) of {@link #NORM_ROPE_STORE}. */
    public static final int STORE_ARG_DST = 33;
    /** {@code LLM_RopeStore}: first per-step argument. */
    public static final int ROPE_STORE_ARG_N_TOKENS = 0;
    /** {@code LLM_RopeStore}: section stride of the positions (the step's token count). */
    public static final int ROPE_STORE_ARG_POS_STRIDE = 7;
    /** {@code cell_idx} of {@link #ROPE_STORE} (the destination cache row per token). */
    public static final int ROPE_STORE_ARG_CELL = 26;
    /** {@code dst} (the cache buffer) of {@link #ROPE_STORE}. */
    public static final int ROPE_STORE_ARG_DST = 27;

    /**
     * Arguments of {@link #NORM_ROPE_STORE}: the rows of {@link #rmsNormRopeArgs} normalized and rotated straight
     * into cache row {@code cellIdx[t]} at {@code dstHeadElems} per head and {@code dstRowStride} per row.
     *
     * @param dstType {@link #KV_STORE_F32} or {@link #KV_STORE_F16}
     * @throws IllegalArgumentException when the geometry, the rotation or the destination type is invalid
     */
    public static Object[] normRopeStoreArgs(RopeSpec rope, int nTokens, int nHeads, int headStride, int rowStride,
                                             int rowLen, int nOffs, int posStride, int xOff, int flags, int wRows,
                                             float eps, boolean hasFf, int dstHeadElems, int dstRowStride,
                                             int dstType, GpuBuffer pos, GpuBuffer ff, GpuBuffer w, GpuBuffer b,
                                             GpuBuffer x, GpuBuffer cellIdx, GpuBuffer dst) {
        Objects.requireNonNull(rope, "rope");
        if (nHeads <= 0 || rowLen <= 0 || rowStride <= 0 || headStride <= 0) {
            throw new IllegalArgumentException(NORM_ROPE_STORE + " needs positive head geometry, got heads "
                    + nHeads + ", row length " + rowLen + ", head stride " + headStride + ", row stride "
                    + rowStride);
        }
        RopeKernels.validateRopeExt(rope.type(), rope.nDims(), nOffs, nTokens, nHeads, headStride, rowStride,
                posStride, xOff);
        if ((long) nOffs + RopeMath.rotatedWidth(rope.type(), rope.nDims()) > rowLen) {
            throw new IllegalArgumentException(NORM_ROPE_STORE + ": rotated dims [" + nOffs + ", "
                    + (nOffs + RopeMath.rotatedWidth(rope.type(), rope.nDims()))
                    + ") exceed the normalized row of " + rowLen);
        }
        checkStoreGeometry(NORM_ROPE_STORE, dstHeadElems, dstRowStride, rowLen, dstType);
        final int[] s = rope.sections();
        checkEps(NORM_ROPE_STORE, eps);
        return new Object[]{
                nTokens, nHeads, headStride, rowStride, rowLen, rope.type().ggmlMode(), rope.nDims(), nOffs,
                posStride, hasFf ? 1 : 0, flags, wRows, xOff, dstHeadElems, dstRowStride, dstType, s[0], s[1],
                s[2], s[3], eps, rope.thetaScale(), rope.freqScale(), rope.extFactor(), rope.attnFactor(),
                rope.corrLo(), rope.corrHi(), Objects.requireNonNull(pos, "pos"), Objects.requireNonNull(ff, "ff"),
                Objects.requireNonNull(w, "w"), Objects.requireNonNull(b, "b"), Objects.requireNonNull(x, "x"),
                Objects.requireNonNull(cellIdx, "cellIdx"), Objects.requireNonNull(dst, "dst")};
    }

    /**
     * Arguments of {@link #ROPE_STORE}: the rows of {@code LLM_RoPE_Ext} rotated straight into cache row
     * {@code cellIdx[t]} at {@code dstHeadElems} per head and {@code dstRowStride} per row.
     *
     * @param dstType {@link #KV_STORE_F32} or {@link #KV_STORE_F16}
     * @throws IllegalArgumentException when the geometry, the rotation or the destination type is invalid
     */
    public static Object[] ropeStoreArgs(RopeSpec rope, int nTokens, int nHeads, int headDim, int rowStride,
                                         int nOffs, int posStride, int xOff, boolean hasFf, int dstHeadElems,
                                         int dstRowStride, int dstType, GpuBuffer pos, GpuBuffer ff, GpuBuffer x,
                                         GpuBuffer cellIdx, GpuBuffer dst) {
        Objects.requireNonNull(rope, "rope");
        if (nHeads <= 0 || headDim <= 0 || rowStride <= 0) {
            throw new IllegalArgumentException(ROPE_STORE + " needs positive head geometry, got heads " + nHeads
                    + ", head dim " + headDim + ", row stride " + rowStride);
        }
        RopeKernels.validateRopeExt(rope.type(), rope.nDims(), nOffs, nTokens, nHeads, headDim, rowStride,
                posStride, xOff);
        if ((long) nOffs + RopeMath.rotatedWidth(rope.type(), rope.nDims()) > headDim) {
            throw new IllegalArgumentException(ROPE_STORE + ": rotated dims [" + nOffs + ", "
                    + (nOffs + RopeMath.rotatedWidth(rope.type(), rope.nDims()))
                    + ") exceed the rotated row of " + headDim);
        }
        checkStoreGeometry(ROPE_STORE, dstHeadElems, dstRowStride, headDim, dstType);
        final int[] s = rope.sections();
        return new Object[]{
                nTokens, nHeads, headDim, rowStride, rope.type().ggmlMode(), rope.nDims(), nOffs, posStride,
                hasFf ? 1 : 0, xOff, dstHeadElems, dstRowStride, dstType, s[0], s[1], s[2], s[3],
                rope.thetaScale(), rope.freqScale(), rope.extFactor(), rope.attnFactor(), rope.corrLo(),
                rope.corrHi(), Objects.requireNonNull(pos, "pos"), Objects.requireNonNull(ff, "ff"),
                Objects.requireNonNull(x, "x"), Objects.requireNonNull(cellIdx, "cellIdx"),
                Objects.requireNonNull(dst, "dst")};
    }

    /** Work-items of {@link #NORM_ROPE_STORE} and {@link #ROPE_STORE}: one workgroup per (token, head). */
    public static long normRopeStoreGlobal(int nTokens, int nHeads, int local) {
        return (long) nTokens * nHeads * local;
    }

    private static void checkStoreGeometry(String kernel, int dstHeadElems, int dstRowStride, int rowLen,
                                           int dstType) {
        if (dstHeadElems < rowLen) {
            throw new IllegalArgumentException(kernel + ": destination head width " + dstHeadElems
                    + " is below the row of " + rowLen);
        }
        if (dstRowStride < dstHeadElems) {
            throw new IllegalArgumentException(kernel + ": destination row stride " + dstRowStride
                    + " is below the head width " + dstHeadElems);
        }
        if (dstType != KV_STORE_F32 && dstType != KV_STORE_F16) {
            throw new IllegalArgumentException(kernel + " stores F32 or F16 cache rows, got destination type "
                    + dstType);
        }
    }

    // ------------------------------------------------------------------ LLM_GdnNormGate

    /**
     * Arguments of {@link #GDN_NORM_GATE}: {@code nTokens * nHeads} rows of {@code headDim} values, each row
     * starting at {@code t*rowStride + h*headDim} in {@code x} and {@code z}, with the output at the same row plus
     * {@code outOff}. {@code x} and {@code out} may coincide.
     *
     * @throws IllegalArgumentException when a count is not positive or the weight does not tile the rows
     */
    public static Object[] gdnNormGateArgs(int nTokens, int nHeads, int headDim, int rowStride, int xOff, int zOff,
                                           int outOff, int wRows, float eps, GpuBuffer x, GpuBuffer w,
                                           GpuBuffer z, GpuBuffer out) {
        if (nHeads <= 0 || headDim <= 0 || rowStride < headDim) {
            throw new IllegalArgumentException(GDN_NORM_GATE + " needs positive heads and head dim and a row "
                    + "stride of at least the head dim, got " + nHeads + " x " + headDim + ", row stride "
                    + rowStride);
        }
        if (wRows < 1) {
            throw new IllegalArgumentException(GDN_NORM_GATE + " needs at least one weight row, got " + wRows);
        }
        checkEps(GDN_NORM_GATE, eps);
        return new Object[]{nTokens, nHeads, headDim, rowStride, xOff, zOff, outOff, wRows, eps,
                Objects.requireNonNull(x, "x"), Objects.requireNonNull(w, "w"), Objects.requireNonNull(z, "z"),
                Objects.requireNonNull(out, "out")};
    }

    /** Work-items of {@link #GDN_NORM_GATE}: one workgroup per (token, head). */
    public static long gdnNormGateGlobal(int nTokens, int nHeads, int local) {
        return (long) nTokens * nHeads * local;
    }

    // ------------------------------------------------------------------ LLM_HcPostNorm

    /** Whether a hyper-connection expansion can be fused with its following row norm. */
    public static boolean hcPostNormUsable(int hc) {
        return hc >= 1 && hc <= 8;
    }

    /**
     * Arguments of {@link #HC_POST_NORM} for the hyper-connection post map of {@code modes} and the row norm of
     * each expanded stream. The norm weight {@code gamma} is bound when {@code wRows > 0}; {@code gammaRows} rows
     * of {@code nEmbd} values, row {@code dst % gammaRows}, multiply the normed rows.
     *
     * @throws IllegalArgumentException when a width or the stream count is outside the kernel's bounds
     */
    public static Object[] hcPostNormArgs(int nTokens, int nEmbd, int hc, boolean hasComb, int postAct,
                                          float postInScale, float postMag, float eps, int gammaRows, int scope,
                                          int yOff, int xOff, int postOff, int combOff, int xoutOff, int xnOff,
                                          GpuBuffer y, GpuBuffer x, GpuBuffer post, GpuBuffer comb, GpuBuffer xout,
                                          GpuBuffer gamma, GpuBuffer xn) {
        if (nEmbd <= 0) {
            throw new IllegalArgumentException(HC_POST_NORM + " needs a positive embedding width, got " + nEmbd);
        }
        if (!hcPostNormUsable(hc)) {
            throw new IllegalArgumentException(HC_POST_NORM + " needs 1 <= hc <= 8, got " + hc);
        }
        int hasW = gammaRows > 0 ? 1 : 0;
        checkEps(HC_POST_NORM, eps);
        return new Object[]{nTokens, nEmbd, hc, hasComb ? 1 : 0, postAct, hasW, Math.max(1, gammaRows), scope, yOff,
                xOff, postOff, combOff, xoutOff, xnOff, postInScale, postMag, eps, Objects.requireNonNull(y, "y"),
                Objects.requireNonNull(x, "x"), Objects.requireNonNull(post, "post"),
                Objects.requireNonNull(comb, "comb"), Objects.requireNonNull(xout, "xout"),
                Objects.requireNonNull(gamma, "gamma"), Objects.requireNonNull(xn, "xn")};
    }

    /** Work-items of {@link #HC_POST_NORM}: one workgroup per token. */
    public static long hcPostNormGlobal(int nTokens, int local) {
        return (long) nTokens * local;
    }
}

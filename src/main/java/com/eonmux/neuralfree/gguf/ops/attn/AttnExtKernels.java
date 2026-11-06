package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.arch.AttnGateKind;
import com.eonmux.neuralfree.gguf.ops.act.ActKernels;
import com.eonmux.neuralfree.gguf.spec.AttentionSpec;

import java.util.Objects;

/**
 * Names, argument indices and argument lists of the attention-extra kernels in
 * {@code kernels/llm/attn_ext.cl} (GLSL {@code attn_ext.comp}); signatures:
 * <pre>
 * LLM_AttnTempScale(i n_tokens, i n_heads, i head_dim, i row_stride, i q_off, i pow_form,
 *                   f floor_scale, f temp_scale, f offset, R&lt;int&gt; pos, RWn q)
 * LLM_AttnOutGate(i n_tokens, i n_head, i head_dim, i attn_stride, i attn_off, i gate_stride,
 *                 i gate_off, i kind, i has_bias, R gate, R bias, RWn attn)
 * LLM_AttnSsmaxScale(i n_tokens, i n_heads, i head_dim, i head_stride, i row_stride, i x_off,
 *                    R scale, R bias, R pos, RWn x)
 * LLM_InklingQueryScale(i n_tokens, i n_heads, i head_dim, i row_stride, i q_off, i log_floor,
 *                       f log_alpha, R&lt;int&gt; pos, RWn q)
 * LLM_InklingRelLogits(i n_tokens, i n_heads, i d_rel, i rel_extent, i log_floor, f log_alpha,
 *                      R r, R proj, R&lt;int&gt; pos, Wn out)
 * LLM_AttnValueEmbd(i n_tokens, i v_dim, i n_head_kv, i head_dim, i has_gate, R ve, R gate, RWn v)
 * LLM_AttnXsa(i n_tokens, i n_head_kv, i n_group, i head_dim, i v_dim, f eps, R alpha, R v, RWn attn)
 * </pre>
 * All run on token-major views; the query and output transforms are in place, the value embedding adds to V and the
 * exclusive self attention rewrites the attention rows. The math is {@link AttnExtMath} and the CPU twins are in
 * {@code CpuAttentionMixer}. {@code n_tokens} is the per-step argument.
 *
 * <p>Dispatch: one work-item per element of the {@code [n_tokens][n_heads * head_dim]} view, 1-D
 * {@code global = roundUp(n_tokens * n_heads * head_dim, L)}, {@code local = {L}} ({@link #global}); the exclusive
 * self attention launches one work-item per (token, KV head) pair ({@link #xsaGlobal}).
 */
public final class AttnExtKernels {

    public static final String TEMP_SCALE = "LLM_AttnTempScale";
    public static final String OUT_GATE = "LLM_AttnOutGate";
    public static final String SSMAX_SCALE = "LLM_AttnSsmaxScale";
    /** Inkling's per-token query log scaling (vLLM {@code compute_log_scaling_tau}). */
    public static final String INKLING_QSCALE = "LLM_InklingQueryScale";
    /** Inkling's query-side relative-logit table (vLLM {@code qkvr_rel_proj}). */
    public static final String INKLING_REL = "LLM_InklingRelLogits";
    /** limite value embeddings: {@code v += value_embd[id] * sigmoid(attn_ve_gate @ x)} per KV head. */
    public static final String VALUE_EMBEDDINGS = "LLM_AttnValueEmbd";
    /** limite exclusive self attention: removes each head's projection on the token's own value. */
    public static final String XSA = "LLM_AttnXsa";

    public static final int ARG_TEMP_N_TOKENS = 0;
    public static final int ARG_TEMP_N_HEADS = 1;
    public static final int ARG_TEMP_HEAD_DIM = 2;
    public static final int ARG_TEMP_ROW_STRIDE = 3;
    public static final int ARG_TEMP_Q_OFF = 4;
    public static final int ARG_TEMP_POW_FORM = 5;
    public static final int ARG_TEMP_FLOOR_SCALE = 6;
    public static final int ARG_TEMP_TEMP_SCALE = 7;
    public static final int ARG_TEMP_OFFSET = 8;
    public static final int ARG_TEMP_POS = 9;
    public static final int ARG_TEMP_Q = 10;
    public static final int TEMP_ARG_COUNT = 11;

    public static final int ARG_GATE_N_TOKENS = 0;
    public static final int ARG_GATE_N_HEAD = 1;
    public static final int ARG_GATE_HEAD_DIM = 2;
    public static final int ARG_GATE_ATTN_STRIDE = 3;
    public static final int ARG_GATE_ATTN_OFF = 4;
    public static final int ARG_GATE_GATE_STRIDE = 5;
    public static final int ARG_GATE_GATE_OFF = 6;
    public static final int ARG_GATE_KIND = 7;
    public static final int ARG_GATE_HAS_BIAS = 8;
    public static final int ARG_GATE_GATE = 9;
    public static final int ARG_GATE_BIAS = 10;
    public static final int ARG_GATE_ATTN = 11;
    public static final int GATE_ARG_COUNT = 12;

    public static final int ARG_SSMAX_N_TOKENS = 0;
    public static final int ARG_SSMAX_N_HEADS = 1;
    public static final int ARG_SSMAX_HEAD_DIM = 2;
    public static final int ARG_SSMAX_HEAD_STRIDE = 3;
    public static final int ARG_SSMAX_ROW_STRIDE = 4;
    public static final int ARG_SSMAX_X_OFF = 5;
    public static final int SSMAX_ARG_COUNT = 6;

    public static final int ARG_QSCALE_N_TOKENS = 0;
    public static final int ARG_QSCALE_N_HEADS = 1;
    public static final int ARG_QSCALE_HEAD_DIM = 2;
    public static final int ARG_QSCALE_ROW_STRIDE = 3;
    public static final int ARG_QSCALE_Q_OFF = 4;
    public static final int ARG_QSCALE_LOG_FLOOR = 5;
    public static final int ARG_QSCALE_LOG_ALPHA = 6;
    public static final int ARG_QSCALE_POS = 7;
    public static final int ARG_QSCALE_Q = 8;
    public static final int QSCALE_ARG_COUNT = 9;

    public static final int ARG_REL_N_TOKENS = 0;
    public static final int ARG_REL_N_HEADS = 1;
    public static final int ARG_REL_D_REL = 2;
    public static final int ARG_REL_EXTENT = 3;
    public static final int ARG_REL_LOG_FLOOR = 4;
    public static final int ARG_REL_LOG_ALPHA = 5;
    public static final int ARG_REL_R = 6;
    public static final int ARG_REL_PROJ = 7;
    public static final int ARG_REL_POS = 8;
    public static final int ARG_REL_OUT = 9;
    public static final int REL_ARG_COUNT = 10;

    public static final int ARG_VE_N_TOKENS = 0;
    public static final int ARG_VE_V_DIM = 1;
    public static final int ARG_VE_N_HEAD_KV = 2;
    public static final int ARG_VE_HEAD_DIM = 3;
    public static final int ARG_VE_HAS_GATE = 4;
    public static final int VE_ARG_COUNT = 8;

    public static final int ARG_XSA_N_TOKENS = 0;
    public static final int ARG_XSA_N_HEAD_KV = 1;
    public static final int ARG_XSA_N_GROUP = 2;
    public static final int ARG_XSA_HEAD_DIM = 3;
    public static final int ARG_XSA_V_DIM = 4;
    public static final int ARG_XSA_EPS = 5;
    public static final int XSA_ARG_COUNT = 9;

    private AttnExtKernels() {
    }

    /**
     * Arguments of {@code LLM_AttnTempScale} in signature order. {@code pos} holds one query position per
     * token ({@code int}); {@code q} is scaled in place.
     *
     * @throws IllegalArgumentException when the temperature is disabled or a view exceeds its buffer
     */
    public static Object[] tempScaleArgs(AttentionSpec.Temperature t, int nTokens, int nHeads, int headDim,
                                         GpuBuffer q, int qOff, int rowStride, GpuBuffer pos) {
        Objects.requireNonNull(t, "temperature");
        if (!t.enabled()) {
            throw new IllegalArgumentException("attention temperature is disabled; skip " + TEMP_SCALE);
        }
        int width = ActKernels.elementCount(nHeads, headDim);
        ActKernels.elementCount(nTokens, width);
        ActKernels.checkFloatView("q", q, qOff, rowStride, nTokens, width);
        checkIntBuffer("pos", pos, nTokens);
        return new Object[]{nTokens, nHeads, headDim, rowStride, qOff, t.pow() ? 1 : 0,
                (float) t.floorScale(), t.scale(), t.offset(), pos, q};
    }

    /**
     * Arguments of {@code LLM_AttnOutGate} in signature order. {@code bias} is the per-head gate bias when
     * {@code hasBias}; otherwise it must be a read-only placeholder buffer, which the kernel never reads.
     *
     * @throws IllegalArgumentException for NONE, PER_HEAD_SIGMOID_BIAS without a bias, or a view that
     *                                  exceeds its buffer
     */
    public static Object[] outGateArgs(AttnGateKind kind, int nTokens, int nHead, int headDim,
                                       GpuBuffer attn, int attnOff, int attnStride,
                                       GpuBuffer gate, int gateOff, int gateStride,
                                       GpuBuffer bias, boolean hasBias) {
        int code = AttnExtMath.gateKernelKind(kind);
        if (!hasBias && AttnExtMath.gateRequiresBias(kind)) {
            throw new IllegalArgumentException("attention gate " + kind + " requires a per-head bias");
        }
        int width = ActKernels.elementCount(nHead, headDim);
        ActKernels.elementCount(nTokens, width);
        ActKernels.checkFloatView("attn", attn, attnOff, attnStride, nTokens, width);
        ActKernels.checkFloatView("gate", gate, gateOff, gateStride, nTokens,
                AttnExtMath.gateWidth(kind, nHead, headDim));
        Objects.requireNonNull(bias, "bias (bind a read-only placeholder when there is no bias)");
        if (hasBias) {
            ActKernels.checkFloatView("bias", bias, 0, nHead, 1, nHead);
        }
        return new Object[]{nTokens, nHead, headDim, attnStride, attnOff, gateStride, gateOff, code,
                hasBias ? 1 : 0, gate, bias, attn};
    }

    /** 1-D global size of either kernel: {@code roundUp(nTokens * nHeads * headDim, local)}. */
    public static long[] global(int nTokens, int nHeads, int headDim, int local) {
        return ActKernels.global1d((long) nTokens * nHeads * headDim, local);
    }

    /**
     * Arguments of {@code LLM_AttnSsmaxScale} in signature order. Token {@code t} at {@code pos[t]} scales head
     * {@code h} of its {@code [t][h][headStride]} view by {@code s[h] * log(pos[t] + 1) + b[h]} (mistral-large4's
     * SSMax, mistral-large4.cpp:181-190, 381, 240).
     *
     * @param scale  one value per head
     * @param bias   one value per head
     * @param pos    one query position per token
     * @param x      scaled in place
     * @throws IllegalArgumentException when the shape is not positive or a view exceeds its buffer
     */
    public static Object[] ssmaxArgs(int nTokens, int nHeads, int headDim, int headStride, int rowStride,
                                     int xOff, GpuBuffer scale, GpuBuffer bias, GpuBuffer pos, GpuBuffer x) {
        if (nTokens < 0 || nHeads < 0 || headDim < 0 || headStride < headDim || rowStride < 0 || xOff < 0) {
            throw new IllegalArgumentException("SSMax needs non-negative counts, headStride >= headDim, and a"
                    + " non-negative offset/row stride, got " + nTokens + "/" + nHeads + "/" + headDim + "/"
                    + headStride + "/" + rowStride + "/" + xOff);
        }
        checkFloatView("x", x, xOff, rowStride, nTokens, nHeads, headStride, headDim);
        checkFloatBuffer("ssmax.weight", scale, nHeads);
        checkFloatBuffer("ssmax.bias", bias, nHeads);
        checkIntBuffer("pos", pos, nTokens);
        return new Object[]{nTokens, nHeads, headDim, headStride, rowStride, xOff, scale, bias, pos, x};
    }

    /** 1-D global size of {@code LLM_AttnSsmaxScale}: {@code roundUp(nTokens * nHeads * headDim, local)}. */
    public static long[] ssmaxGlobal(int nTokens, int nHeads, int headDim, int local) {
        return ActKernels.global1d((long) nTokens * nHeads * headDim, local);
    }

    /**
     * Arguments of {@code LLM_InklingQueryScale} in signature order: {@code q[t][h][d] *= tau(pos[t])} with
     * {@code tau = 1 + alpha * log(max(1, (pos + 1) / floor))}, the log factor of Inkling's full-attention layers
     * (vLLM {@code compute_log_scaling_tau}, {@code fused_qkvr_prep}). {@code q} is scaled in place.
     *
     * @param logFloor the log-scaling floor; {@code <= 0} makes every factor 1 (the kernel still runs)
     * @param pos      one query position per token
     * @throws IllegalArgumentException when a view exceeds its buffer
     */
    public static Object[] inklingScaleArgs(int nTokens, int nHeads, int headDim, int rowStride, int qOff,
                                            int logFloor, float logAlpha, GpuBuffer pos, GpuBuffer q) {
        int width = ActKernels.elementCount(nHeads, headDim);
        ActKernels.elementCount(nTokens, width);
        ActKernels.checkFloatView("q", q, qOff, rowStride, nTokens, width);
        checkIntBuffer("pos", pos, nTokens);
        return new Object[]{nTokens, nHeads, headDim, rowStride, qOff, logFloor, logAlpha, pos, q};
    }

    /** 1-D global size of {@code LLM_InklingQueryScale}: {@code roundUp(nTokens * nHeads * headDim, local)}. */
    public static long[] inklingScaleGlobal(int nTokens, int nHeads, int headDim, int local) {
        return ActKernels.global1d((long) nTokens * nHeads * headDim, local);
    }

    /**
     * Arguments of {@code LLM_InklingRelLogits} in signature order: the relative-logit table
     * {@code out[(t * nHeads + h) * relExtent + e] = tau(pos[t]) * sum_j r[t][h][j] * proj[j][e]} of Inkling's
     * {@code RelLogitsProj}.
     *
     * @param r        the {@code attn_rel} output, {@code [nTokens][nHeads][dRel]}
     * @param proj     the shared {@code [dRel][relExtent]} projection
     * @param logFloor log-scaling floor; {@code <= 0} leaves the table unscaled
     * @param out      the table, {@code [nTokens][nHeads][relExtent]}
     * @throws IllegalArgumentException when a count is not positive or a buffer is too small
     */
    public static Object[] inklingRelArgs(int nTokens, int nHeads, int dRel, int relExtent, int logFloor,
                                          float logAlpha, GpuBuffer r, GpuBuffer proj, GpuBuffer pos,
                                          GpuBuffer out) {
        if (nTokens < 0 || nHeads <= 0 || dRel <= 0 || relExtent <= 0) {
            throw new IllegalArgumentException("the relative-logit table needs non-negative tokens and positive head"
                    + " counts, got " + nTokens + " tokens, " + nHeads + " heads, d_rel " + dRel + ", extent "
                    + relExtent);
        }
        checkFloatBuffer("attn_rel output", r, (long) nTokens * nHeads * dRel);
        checkFloatBuffer("attn_rel_proj", proj, (long) dRel * relExtent);
        checkFloatBuffer("relative-logit table", out, (long) nTokens * nHeads * relExtent);
        checkIntBuffer("pos", pos, nTokens);
        return new Object[]{nTokens, nHeads, dRel, relExtent, logFloor, logAlpha, r, proj, pos, out};
    }

    /** 1-D global size of {@code LLM_InklingRelLogits}: {@code roundUp(nTokens * nHeads * relExtent, local)}. */
    public static long[] inklingRelGlobal(int nTokens, int nHeads, int relExtent, int local) {
        return ActKernels.global1d((long) nTokens * nHeads * relExtent, local);
    }

    /**
     * Arguments of {@code LLM_AttnValueEmbd} in signature order: {@code v[t][h][d] += ve[t][h][d] * g} with
     * {@code g = sigmoid(gate[t][h])} when {@code hasGate}, else 1 (limite.cpp:87-89, 149-155). {@code v} is dense
     * {@code [nTokens][vDim]}; {@code ve} has the same layout and {@code gate} is {@code [nTokens][nHeadKv]}.
     *
     * @throws IllegalArgumentException when a dimension is negative or a view exceeds its buffer
     */
    public static Object[] valueEmbeddingArgs(int nTokens, int vDim, int nHeadKv, int headDim, boolean hasGate,
                                              GpuBuffer ve, GpuBuffer gate, GpuBuffer v) {
        checkNonNegative("n_tokens", nTokens);
        checkNonNegative("v_dim", vDim);
        checkNonNegative("n_head_kv", nHeadKv);
        checkNonNegative("head_dim", headDim);
        if (vDim > 0 && nHeadKv > 0 && headDim * nHeadKv != vDim) {
            throw new IllegalArgumentException("value embeddings need v_dim " + vDim + " = n_head_kv * head_dim, got "
                    + nHeadKv + " x " + headDim);
        }
        long rows = (long) nTokens * vDim;
        checkFloatBuffer("value embeddings", ve, rows);
        checkFloatBuffer("value embedding gate", gate, hasGate ? (long) nTokens * nHeadKv : 1);
        checkFloatBuffer("value rows", v, rows);
        return new Object[]{nTokens, vDim, nHeadKv, headDim, hasGate ? 1 : 0, ve, gate, v};
    }

    /**
     * Arguments of {@code LLM_AttnXsa} in signature order: token {@code t}'s KV head {@code h} loses its projection
     * on the head's L2-normalized value from each of its {@code nGroup} query heads, scaled by
     * {@code alpha[h*nGroup + g]} (limite.cpp:177-191). {@code v} is dense {@code [nTokens][vDim]} and {@code attn}
     * is {@code [nTokens][nHeadKv*nGroup*headDim]}.
     *
     * @throws IllegalArgumentException when a dimension or the epsilon is invalid, or a view exceeds its buffer
     */
    public static Object[] xsaArgs(int nTokens, int nHeadKv, int nGroup, int headDim, int vDim, float eps,
                                   GpuBuffer alpha, GpuBuffer v, GpuBuffer attn) {
        checkNonNegative("n_tokens", nTokens);
        checkNonNegative("n_head_kv", nHeadKv);
        checkNonNegative("n_group", nGroup);
        checkNonNegative("head_dim", headDim);
        if (vDim < 0 || vDim != (long) nHeadKv * headDim) {
            throw new IllegalArgumentException("exclusive self attention needs v_dim " + ((long) nHeadKv * headDim)
                    + " = n_head_kv * head_dim, got " + vDim);
        }
        if (!Float.isFinite(eps) || eps < 0f) {
            throw new IllegalArgumentException("exclusive self attention epsilon must be finite and non-negative, got "
                    + eps);
        }
        checkFloatBuffer("xsa alpha", alpha, (long) nHeadKv * nGroup);
        checkFloatBuffer("xsa values", v, (long) nTokens * vDim);
        checkFloatBuffer("xsa attention", attn, (long) nTokens * nHeadKv * nGroup * headDim);
        return new Object[]{nTokens, nHeadKv, nGroup, headDim, vDim, eps, alpha, v, attn};
    }

    /** 1-D global size of {@code LLM_AttnXsa}: {@code roundUp(nTokens * nHeadKv, local)}. */
    public static long[] xsaGlobal(int nTokens, int nHeadKv, int local) {
        return ActKernels.global1d((long) nTokens * nHeadKv, local);
    }

    private static void checkNonNegative(String name, int value) {
        if (value < 0) {
            throw new IllegalArgumentException(name + " must be non-negative, got " + value);
        }
    }

    private static void checkFloatBuffer(String name, GpuBuffer buf, long n) {
        Objects.requireNonNull(buf, name);
        if (n < 0 || n * Float.BYTES > buf.sizeInBytes()) {
            throw new IllegalArgumentException(name + ": " + n + " floats exceed a buffer of " + buf.sizeInBytes()
                    + " bytes");
        }
    }

    /** A {@code [rows] x [heads] x [headDim]} view with the given head and row strides lies inside {@code a}. */
    private static void checkFloatView(String name, GpuBuffer a, int off, int rowStride, int rows, int heads,
                                       int headStride, int headDim) {
        Objects.requireNonNull(a, name);
        if (rows == 0 || heads == 0 || headDim == 0) {
            return;
        }
        long end = (long) off + (long) (rows - 1) * rowStride + (long) (heads - 1) * headStride + headDim;
        if (end * Float.BYTES > a.sizeInBytes()) {
            throw new IllegalArgumentException(name + " view ending at " + end + " floats exceeds a buffer of "
                    + a.sizeInBytes() / Float.BYTES);
        }
    }

    private static void checkIntBuffer(String name, GpuBuffer buf, int n) {
        Objects.requireNonNull(buf, name);
        if ((long) n * Integer.BYTES > buf.sizeInBytes()) {
            throw new IllegalArgumentException(name + ": " + n + " ints exceed a buffer of " + buf.sizeInBytes() + " bytes");
        }
    }
}

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
 * </pre>
 * Both run in place on token-major views (offsets and strides in floats); the math is
 * {@link AttnExtMath}. {@code n_tokens} is the per-step argument.
 *
 * <p>Dispatch: one work-item per element of the {@code [n_tokens][n_heads * head_dim]} view, 1-D
 * {@code global = roundUp(n_tokens * n_heads * head_dim, L)}, {@code local = {L}} ({@link #global}).
 */
public final class AttnExtKernels {

    public static final String TEMP_SCALE = "LLM_AttnTempScale";
    public static final String OUT_GATE = "LLM_AttnOutGate";

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

    private static void checkIntBuffer(String name, GpuBuffer buf, int n) {
        Objects.requireNonNull(buf, name);
        if ((long) n * Integer.BYTES > buf.sizeInBytes()) {
            throw new IllegalArgumentException(name + ": " + n + " ints exceed a buffer of " + buf.sizeInBytes() + " bytes");
        }
    }
}

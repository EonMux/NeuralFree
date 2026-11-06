package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;

import java.util.Objects;

/**
 * Name and arguments of the gated mixed attention kernel in {@code kernels/llm/attn_ext.cl} (GLSL
 * {@code attn_ext.comp}), signature:
 * <pre>
 * LLM_LoopAttnMix(i n_tokens, i n_heads, i head_dim, i q_stride, i q_off, i attn_stride, i attn_off,
 *                 R q, R gate_w, R gate_b, R global, RWn local)
 * </pre>
 * Token {@code t}, head {@code h} computes {@code g = sigmoid(dot(q[t][h], gate_w[h]) + gate_b[h])} and mixes the
 * global attention rows into the local ones: {@code local = local * (1 - g) + global * g}. {@code gate_w} is
 * {@code [n_heads][head_dim]} and {@code gate_b} one value per head; offsets and strides are in floats. One
 * work-item per element of the {@code [n_tokens][n_heads][head_dim]} view.
 */
public final class LoopAttnKernels {

    public static final String MIX = "LLM_LoopAttnMix";

    public static final int ARG_N_TOKENS = 0;
    public static final int ARG_N_HEADS = 1;
    public static final int ARG_HEAD_DIM = 2;
    public static final int ARG_Q_STRIDE = 3;
    public static final int ARG_Q_OFF = 4;
    public static final int ARG_ATTN_STRIDE = 5;
    public static final int ARG_ATTN_OFF = 6;
    public static final int ARG_Q = 7;
    public static final int ARG_GATE_W = 8;
    public static final int ARG_GATE_B = 9;
    public static final int ARG_GLOBAL = 10;
    public static final int ARG_LOCAL = 11;
    public static final int ARG_COUNT = 12;

    private LoopAttnKernels() {
    }

    /**
     * Arguments in signature order.
     *
     * @param q      token-major query rows, {@code q_off + t * q_stride}
     * @param local  the local attention rows, mixed in place
     * @param global the global attention rows
     * @throws NullPointerException     when a buffer is missing
     * @throws IllegalArgumentException for non-positive head counts or a negative offset/stride
     */
    public static Object[] mixArgs(int nTokens, int nHeads, int headDim, int qStride, int qOff, int attnStride,
                                   int attnOff, GpuBuffer q, GpuBuffer gateW, GpuBuffer gateB, GpuBuffer global,
                                   GpuBuffer local) {
        if (nTokens < 0 || nHeads <= 0 || headDim <= 0) {
            throw new IllegalArgumentException("loop attention mix needs nTokens >= 0 and positive head counts, got "
                    + nTokens + "/" + nHeads + "/" + headDim);
        }
        if (qStride < 0 || qOff < 0 || attnStride < 0 || attnOff < 0) {
            throw new IllegalArgumentException("loop attention mix needs non-negative offsets and strides, got "
                    + qOff + "/" + qStride + "/" + attnOff + "/" + attnStride);
        }
        Objects.requireNonNull(q, "q");
        Objects.requireNonNull(gateW, "gateW");
        Objects.requireNonNull(gateB, "gateB");
        Objects.requireNonNull(global, "global");
        Objects.requireNonNull(local, "local");
        return new Object[]{nTokens, nHeads, headDim, qStride, qOff, attnStride, attnOff, q, gateW, gateB, global,
                local};
    }
}

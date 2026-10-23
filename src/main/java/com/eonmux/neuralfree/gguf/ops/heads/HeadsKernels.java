package com.eonmux.neuralfree.gguf.ops.heads;

import com.eonmux.neuralfree.gguf.kernels.LlmKernelConstants;

import java.util.Collections;
import java.util.List;

/**
 * Names, argument layout and launch geometry of the heads-family kernels ({@code kernels/llm/heads.cl} and the
 * GLSL twin {@code lora_switch.comp}). Offsets and strides are in floats; every scalar is a 32-bit {@code int};
 * buffers follow the scalars in the order listed.
 *
 * <ul>
 *   <li>{@link #SWITCHED_LORA_ADD} {@code (n_tokens*, in_dim, out_dim, rank, stage*, n_slots, x_stride, x_off,
 *       y_stride, y_off, tmp_off; slots, A, B, x, tmp, y)}: graniteswitch per-token LoRA delta
 *       {@code y[t] += B[s_t](A[s_t] x[t])} in two dispatches, {@link #LORA_STAGE_DOWN} then {@link #LORA_STAGE_UP}
 *       on one queue; geometry from {@link #loraGlobal}. CPU twin {@link CpuPooling#switchedLoraAdd}.</li>
 * </ul>
 * {@code *} marks the arguments re-set per dispatch, at the matching {@code ARG_*} index.
 */
public final class HeadsKernels {

    public static final String SWITCHED_LORA_ADD = "LLM_SwitchedLoraAdd";

    /** Every kernel of the family, for prewarming. */
    public static final List<String> ALL = Collections.singletonList(SWITCHED_LORA_ADD);

    /** {@link #SWITCHED_LORA_ADD} {@code n_tokens}. */
    public static final int ARG_LORA_N_TOKENS = 0;
    /** {@link #SWITCHED_LORA_ADD} {@code stage}. */
    public static final int ARG_LORA_STAGE = 4;
    /** {@link #SWITCHED_LORA_ADD} argument count (11 scalars, 6 buffers). */
    public static final int LORA_ARG_COUNT = 17;

    /** Stage 0: {@code tmp[t][r] = A[s_t][r] . x[t]}, one workgroup per (token, rank row). */
    public static final int LORA_STAGE_DOWN = 0;
    /** Stage 1: {@code y[t][o] += B[s_t][o] . tmp[t]}, one work-item per (token, output row). */
    public static final int LORA_STAGE_UP = 1;

    /**
     * Local size of both {@link #SWITCHED_LORA_ADD} stages: a power of two, a multiple of 32 and at most
     * {@code LLM_DIMX}, as the stage-0 workgroup reduction requires.
     */
    public static final int LORA_LOCAL = LlmKernelConstants.LLM_DIMX;

    private HeadsKernels() {
    }

    /**
     * Global size of one {@link #SWITCHED_LORA_ADD} stage with local size {@link #LORA_LOCAL}:
     * {@code n_tokens * rank * LORA_LOCAL} for {@link #LORA_STAGE_DOWN} (one workgroup per token and rank row),
     * {@code roundUp(n_tokens * out_dim, LORA_LOCAL)} for {@link #LORA_STAGE_UP}.
     *
     * @throws IllegalArgumentException for an unknown stage or a non-positive dimension
     */
    public static long loraGlobal(int stage, int nTokens, int rank, int outDim) {
        if (nTokens <= 0 || rank <= 0 || outDim <= 0) {
            throw new IllegalArgumentException(SWITCHED_LORA_ADD + " needs positive n_tokens, rank and out_dim: n_tokens="
                    + nTokens + " rank=" + rank + " out_dim=" + outDim);
        }
        if (stage == LORA_STAGE_DOWN) {
            return (long) nTokens * rank * LORA_LOCAL;
        }
        if (stage == LORA_STAGE_UP) {
            return roundUp((long) nTokens * outDim, LORA_LOCAL);
        }
        throw new IllegalArgumentException(SWITCHED_LORA_ADD + " stage must be " + LORA_STAGE_DOWN + " or "
                + LORA_STAGE_UP + ", got " + stage);
    }

    /**
     * Validates a switched-LoRA shape for the kernel and {@link CpuPooling#switchedLoraAdd}: positive dimensions and
     * stacked tensors {@code A {inDim, rank, nSlots}}, {@code B {rank, outDim, nSlots}} whose element counts fit the
     * 32-bit index range of the GLSL twin.
     *
     * @throws IllegalArgumentException with the offending values
     */
    public static void checkLoraShape(int inDim, int outDim, int rank, int nSlots) {
        if (inDim <= 0 || outDim <= 0 || rank <= 0 || nSlots <= 0) {
            throw new IllegalArgumentException("switched LoRA dimensions must be positive: in_dim=" + inDim
                    + " out_dim=" + outDim + " rank=" + rank + " n_slots=" + nSlots);
        }
        if ((long) inDim * rank * nSlots > Integer.MAX_VALUE || (long) outDim * rank * nSlots > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("switched LoRA tensors of " + nSlots + " slots x rank " + rank + " x ("
                    + inDim + ", " + outDim + ") exceed the 32-bit index range");
        }
    }

    static long roundUp(long n, int local) {
        return (n + local - 1) / local * local;
    }
}

package com.eonmux.neuralfree.gguf.kernels;

/**
 * Java mirror of the model-independent compile-time constants of the LLM kernels
 * ({@code kernels/llm/common/config.h}) and of the engine's kernel-selection thresholds.
 */
public final class LlmKernelConstants {

    /** Largest work-group size of every row-reduce kernel (a power of two); {@code LLM_DIMX}. */
    public static final int LLM_DIMX = 256;
    /** Work-group size of the flash-attention kernels; {@code NF_FA_WG}. */
    public static final int FA_WG = 128;
    /** Output accumulators per work-item in flash kernels ({@code g_tile * d_v <= FA_WG * FA_ACC_MAX}); {@code NF_FA_ACC_MAX}. */
    public static final int FA_ACC_MAX = 32;
    /** Largest number of query heads per flash-decode work-group; {@code NF_FD_MAX_G}. */
    public static final int FD_MAX_G = 16;
    /** Smallest token count for which projections use the tiled GEMM kernels instead of batched matvec. */
    public static final int TILED_MIN_TOKENS = 8;

    private LlmKernelConstants() {
    }
}

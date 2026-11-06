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
    /**
     * Output accumulators per work-item in flash kernels ({@code g_tile * d_v <= FA_WG * FA_ACC_MAX} for the tiled
     * and MMA kernels, {@code <= (FA_WG / (FA_WG / 32)) * FA_ACC_MAX} for the decode kernel, whose pairs spread
     * over each lane group's 32 items); {@code NF_FA_ACC_MAX}.
     */
    public static final int FA_ACC_MAX = 32;
    /** Largest number of query heads per flash-decode work-group; {@code NF_FD_MAX_G}. */
    public static final int FD_MAX_G = 16;
    /**
     * Smallest token count for which projections of a weight without a fast GEMM lane use the tiled GEMM
     * kernels instead of batched matvec. Weights with a fast lane cross over earlier, at
     * {@link #FAST_GEMM_MIN_TOKENS}.
     */
    public static final int TILED_MIN_TOKENS = 8;
    /**
     * Largest token-row count the multi-column integer-dot matvec serves in one launch
     * ({@code {T}_MatVecN_Q8}); {@code NF_MVQN_MAX}, whose unrolled token loop it bounds.
     */
    public static final int MATVEC_N_MAX_TOKENS = 16;
    /**
     * Largest token-row count the projection dispatch routes to the multi-column integer-dot matvec for a
     * weight without a fast GEMM lane; groups above it take the tiled GEMM lanes, which stay flat in the
     * row count while the matvec pays its integer dots per row. The kernel itself serves up to
     * {@link #MATVEC_N_MAX_TOKENS} rows. The cap is the measured crossover of that older dispatch: on an
     * RTX 5080 Laptop over Qwen3-Embedding-4B Q8_0, one 9-row step costs 40.3 ms on the matvec at 8 rows
     * against 43.3 ms on the tiled GEMM, and 45.1 ms at 9 rows against 43.6 ms, while at 16 rows the
     * matvec costs 74.6 ms against 44.9 ms.
     */
    public static final int MATVEC_N_DISPATCH_MAX = 8;
    /**
     * Smallest token count the fast GEMM lanes serve; with a lane present the multi-column matvec keeps
     * only the groups below it and the single row keeps the one-row matvec. The cap is the measured
     * crossover: the GEMM lanes' cost is flat in the row count while the matvec pays its integer dots per
     * row. On an RTX 5080 Laptop over Qwen3-Embedding-4B Q8_0 with the shared-memory stage pipeline, one
     * projection-forward step through the GEMM lanes costs 27.5 ms at 2 rows and stays at 27.0 to 30.0 ms
     * from 4 to 16 rows, against 17.4 ms on the multi-column matvec at 2 rows and 29.6 ms at 4 rows — the
     * matvec wins at 2 rows, loses at 4, and interpolating its per-row cost puts the tie at 3, so 4 is
     * the first count the lane serves.
     */
    public static final int FAST_GEMM_MIN_TOKENS = 4;

    private LlmKernelConstants() {
    }
}

package com.eonmux.neuralfree.gguf.kernels;

import com.eonmux.neuralfree.gguf.GGMLType;

import java.util.EnumSet;
import java.util.Set;

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
     * Head-dim limit of the flash-decode shuffle fast path (one coalesced row float4 per lane covers
     * {@code 32 * 4} dims); {@code NF_FA_FAST_DIM}.
     */
    public static final int FA_FAST_DIM = 128;
    /**
     * Smallest token count for which projections of a weight without a fast GEMM lane use the tiled GEMM
     * kernels instead of batched matvec. Weights with a fast lane cross over at the row count
     * {@link com.eonmux.neuralfree.gguf.gpu.forward.SmallBatchCrossover} measured for their device kind,
     * type and projection size.
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
     * Largest token-row count the counted multi-column integer-dot matvec forms instantiate
     * ({@code NF_MVQN_NC_TOP} of {@code common/matvec_family.h}); every type instantiates
     * 2..{@link #MATVEC_N_DISPATCH_MAX} by default, and {@link #countedMaxTokens} answers which
     * types go past it.
     */
    public static final int MATVEC_N_COUNTED_MAX = 16;

    /** The types whose counted multi-column forms instantiate past {@link #MATVEC_N_DISPATCH_MAX}. */
    private static final Set<GGMLType> COUNTED_PAST_DISPATCH_MAX = EnumSet.of(GGMLType.Q8_0,
            GGMLType.Q4_0, GGMLType.Q4_K, GGMLType.Q5_K, GGMLType.Q6_K, GGMLType.IQ4_XS);

    /**
     * The widest exact token count {@code type}'s counted multi-column matvec forms instantiate
     * ({@code {T}_MatVecN<n>_Q8} / {@code {T}_MatVec2N<n>_Q8}): {@link #MATVEC_N_COUNTED_MAX} for
     * the types whose measured dispatch serves exact counts above {@link #MATVEC_N_DISPATCH_MAX}
     * (Q8_0, and the K-quant, Q4_0 and IQ4_XS types whose kernels instantiate
     * {@code IQ_MVQN_NC_HIGH}), {@link #MATVEC_N_DISPATCH_MAX} for every other type.
     */
    public static int countedMaxTokens(GGMLType type) {
        return COUNTED_PAST_DISPATCH_MAX.contains(type) ? MATVEC_N_COUNTED_MAX : MATVEC_N_DISPATCH_MAX;
    }

    /**
     * The exact-token-count twin of a multi-column matvec kernel: the token count inserted after the
     * {@code MatVecN} name segment ({@code {T}_MatVecN_Q8} at four tokens gives {@code {T}_MatVecN4_Q8}).
     * Returns {@code kernel} unchanged when it has no {@code MatVecN} segment or {@code nTokens} lies
     * outside the instantiated range; the projection dispatch additionally serves the twin only when
     * the library instantiates it and it is ready at the matvec work-group size.
     */
    public static String countedMatvecNName(String kernel, int nTokens) {
        if (nTokens < 2 || nTokens > MATVEC_N_MAX_TOKENS) {
            return kernel;
        }
        int at = kernel.indexOf("MatVecN");
        if (at < 0) {
            return kernel;
        }
        return kernel.substring(0, at + "MatVecN".length()) + nTokens
                + kernel.substring(at + "MatVecN".length());
    }

    private LlmKernelConstants() {
    }
}

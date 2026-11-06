package com.eonmux.neuralfree.gguf.ops.sparse;

import com.eonmux.neuralfree.gguf.arch.SwaType;
import com.eonmux.neuralfree.gguf.kernels.LlmKernelConstants;
import com.eonmux.neuralfree.gguf.memory.KvDType;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * Kernel names, argument indices, launch geometry and preconditions of the DeepSeek sparse-attention kernels
 * ({@code kernels/llm/sparse/dsa.cl}, {@code topk_rows.cl}, {@code sparse_mla.cl} and their GLSL twins).
 *
 * <p>Argument lists ({@code (dyn)} marks the per-step arguments with an {@code ARG_*} constant):
 * <pre>
 * LLM_DSA_IndexerScore_&lt;KV&gt; (i n_tokens(dyn), i n_kv(dyn), i n_idx_heads, i idx_dim, i causal, i swa_type,
 *                       i n_swa, i key_stride, i key_off, R iq, R w, R&lt;KV&gt; ik_cache, R&lt;int&gt; q_pos,
 *                       R&lt;int&gt; kv_pos, W scores)
 * LLM_DSA_BlockScore   (i n_tokens(dyn), i n_kv(dyn), i block_size, i ratio, R&lt;int&gt; q_pos, R scores,
 *                       W block_scores)
 * LLM_DSA_RestrictBlocks (i n_tokens(dyn), i n_kv(dyn), i block_size, i blocks_off(dyn), i blocks_stride(dyn),
 *                       R&lt;int&gt; blocks, RW scores)
 * LLM_TopK_Rows        (i n_rows(dyn), i n_cols(dyn), i k(dyn), R scores, W&lt;int&gt; idx_out)
 * LLM_MLA_SparseDecode_&lt;KV&gt; (i n_head, i kv_lora, i rope_dim, i n_sel(dyn), f kq_scale, f softcap,
 *                       i has_sinks, i q_off(dyn), R q_abs, R&lt;KV&gt; kv_cache, R&lt;int&gt; sel_idx, R sinks,
 *                       W out_latent)
 * LLM_MLA_SparsePrefill_&lt;KV&gt;(i n_tokens(dyn), i n_head, i kv_lora, i rope_dim, i n_sel(dyn), f kq_scale,
 *                       f softcap, i has_sinks, i q_stride, R q_abs, R&lt;KV&gt; kv_cache, R&lt;int&gt; sel_idx,
 *                       R sinks, W out_latent)
 * </pre>
 * {@code <KV>} ({@link KvDType#kernelSuffix()}, every KV type with a sparse row decoder) is the latent cache's
 * {@link KvDType}
 * ({@link #sparseDecode}, {@link #sparsePrefill}) or the indexer cache's ({@link #indexerScore}); both follow the
 * context's K type, as llama.cpp creates the indexer cache with {@code type_k} (llama.cpp
 * src/llama-kv-cache-dsa.cpp:51).
 * {@code swa_type} is {@code SwaType.code()}; {@code k} of the top-k is {@link #selectedCount(int, int)}; the top-k
 * output row stride is {@code k}, which is also {@code n_sel} of the sparse MLA kernels. Unused {@code sinks} are a
 * read-only placeholder with {@code has_sinks = 0}. The math of each kernel is reproduced by {@link CpuDsaIndexer},
 * {@link CpuTopK} and {@link CpuSparseMla}.
 */
public final class SparseKernels {

    /** Name prefixes of the per-KV-type variants. */
    public static final String DSA_INDEXER_SCORE_PREFIX = "LLM_DSA_IndexerScore_";
    public static final String MLA_SPARSE_DECODE_PREFIX = "LLM_MLA_SparseDecode_";
    public static final String MLA_SPARSE_PREFILL_PREFIX = "LLM_MLA_SparsePrefill_";

    public static final String TOPK_ROWS = "LLM_TopK_Rows";
    /** DeepSeek-V4.1 candidate blocks: each block's best row score. */
    public static final String BLOCK_SCORE = "LLM_DSA_BlockScore";
    /** DeepSeek-V4.1 candidate blocks: mask rows outside a token's blocks. */
    public static final String RESTRICT_BLOCKS = "LLM_DSA_RestrictBlocks";

    /** Every kernel of this family. */
    public static final List<String> ALL;

    static {
        List<String> all = new ArrayList<>(KvDType.kernelVariants(DSA_INDEXER_SCORE_PREFIX, t -> !t.perTokenHead() && t.hasSparseKernel()));
        all.add(TOPK_ROWS);
        all.add(BLOCK_SCORE);
        all.add(RESTRICT_BLOCKS);
        all.addAll(KvDType.kernelVariants(MLA_SPARSE_DECODE_PREFIX, t -> !t.perTokenHead() && t.hasSparseKernel()));
        all.addAll(KvDType.kernelVariants(MLA_SPARSE_PREFILL_PREFIX, t -> !t.perTokenHead() && t.hasSparseKernel()));
        ALL = Collections.unmodifiableList(all);
    }

    /** {@code LLM_DSA_IndexerScore}: argument count and per-step argument indices. */
    public static final int SCORE_ARG_COUNT = 15;
    public static final int ARG_SCORE_N_TOKENS = 0;
    public static final int ARG_SCORE_N_KV = 1;
    public static final int ARG_SCORE_CAUSAL = 4;
    public static final int ARG_SCORE_KEY_STRIDE = 7;
    public static final int ARG_SCORE_KEY_OFF = 8;
    public static final int ARG_SCORE_IQ = 9;
    public static final int ARG_SCORE_W = 10;
    public static final int ARG_SCORE_KEYS = 11;
    public static final int ARG_SCORE_Q_POS = 12;
    public static final int ARG_SCORE_KV_POS = 13;

    /** {@code LLM_DSA_BlockScore}: argument count and the per-step argument. */
    public static final int BLOCK_SCORE_ARG_COUNT = 7;
    public static final int ARG_BLOCK_SCORE_N_TOKENS = 0;
    public static final int ARG_BLOCK_SCORE_N_KV = 1;
    public static final int ARG_BLOCK_SCORE_Q_POS = 4;

    /** {@code LLM_DSA_RestrictBlocks}: argument count and the per-step arguments. */
    public static final int RESTRICT_ARG_COUNT = 7;
    public static final int ARG_RESTRICT_N_TOKENS = 0;
    public static final int ARG_RESTRICT_N_KV = 1;
    public static final int ARG_RESTRICT_BLOCK_SIZE = 2;
    public static final int ARG_RESTRICT_BLOCKS_OFF = 3;
    public static final int ARG_RESTRICT_BLOCKS_STRIDE = 4;
    public static final int ARG_RESTRICT_BLOCKS = 5;

    /** {@code LLM_TopK_Rows}: argument count and per-step argument indices. */
    public static final int TOPK_ARG_COUNT = 5;
    public static final int ARG_TOPK_N_ROWS = 0;
    public static final int ARG_TOPK_N_COLS = 1;
    public static final int ARG_TOPK_K = 2;
    public static final int ARG_TOPK_SCORES = 3;
    public static final int ARG_TOPK_OUT = 4;

    /** {@code LLM_MLA_SparseDecode}: argument count and per-step argument indices. */
    public static final int DECODE_ARG_COUNT = 13;
    public static final int ARG_DECODE_N_SEL = 3;
    public static final int ARG_DECODE_Q_OFF = 7;

    /** {@code LLM_MLA_SparsePrefill}: argument count and per-step argument indices. */
    public static final int PREFILL_ARG_COUNT = 14;
    public static final int ARG_PREFILL_N_TOKENS = 0;
    public static final int ARG_PREFILL_N_SEL = 4;

    /** Largest top-k width and sparse-MLA selection ({@code NF_TOPK_MAX}, {@code NF_SMLA_MAX_SEL}). */
    public static final int TOPK_MAX = 2048;
    /** Largest indexer head size ({@code NF_DSA_QSTAGE}: one head must fit the query stage). */
    public static final int INDEXER_MAX_HEAD_DIM = 2048;

    /** Recommended work-group size of {@code LLM_DSA_IndexerScore} (any size up to {@code LLM_DIMX} is valid). */
    public static final int INDEXER_LOCAL = 128;
    /** Recommended work-group size of {@code LLM_TopK_Rows} (a power of two up to {@code LLM_DIMX}). */
    public static final int TOPK_LOCAL = 256;
    /** Recommended work-group size of the sparse MLA kernels (a power of two in [32, LLM_DIMX]). */
    public static final int SPARSE_MLA_LOCAL = 128;

    private SparseKernels() {
    }

    /**
     * Lightning-indexer score kernel for the indexer cache's element type.
     *
     * @throws UnsupportedOperationException naming the type when it has no sparse row decoder (the vendor formats)
     */
    public static String indexerScore(KvDType cache) {
        return requireSparse("indexer", cache).kernel(DSA_INDEXER_SCORE_PREFIX);
    }

    /**
     * Sparse MLA decode kernel for the latent cache's element type.
     *
     * @throws UnsupportedOperationException naming the type when it has no sparse row decoder (the vendor formats)
     */
    public static String sparseDecode(KvDType latent) {
        return requireSparse("latent", latent).kernel(MLA_SPARSE_DECODE_PREFIX);
    }

    /**
     * Sparse MLA prefill kernel for the latent cache's element type.
     *
     * @throws UnsupportedOperationException naming the type when it has no sparse row decoder (the vendor formats)
     */
    public static String sparsePrefill(KvDType latent) {
        return requireSparse("latent", latent).kernel(MLA_SPARSE_PREFILL_PREFIX);
    }

    /** The type of a sparse cache, refused by name when the family has no decoder for it. */
    private static KvDType requireSparse(String cache, KvDType type) {
        Objects.requireNonNull(type, cache + " cache type");
        if (!type.hasSparseKernel()) {
            throw new UnsupportedOperationException("the sparse MLA kernels decode their " + cache + " cache rows"
                    + " themselves and have no " + type.name().toLowerCase(java.util.Locale.ROOT) + " variant; keep the"
                    + " cache on a type with a sparse row decoder (f16, q8_0, ...), or run the model on the CPU"
                    + " backend");
        }
        return type;
    }

    /** Number of cells the indexer selects: {@code min(nKv, topK)} (llama.cpp src/models/deepseek32.cpp:314). */
    public static int selectedCount(int nKv, int topK) {
        if (nKv < 0 || topK < 1) {
            throw new IllegalArgumentException("DSA selection needs nKv >= 0 and topK >= 1, got " + nKv + "/" + topK);
        }
        return Math.min(nKv, topK);
    }

    /** 1-D global size of {@code LLM_DSA_IndexerScore}: one work-group per (token, block of {@code local} rows). */
    public static long[] indexerGlobal(int nTokens, int nKv, int local) {
        checkIndexerLocal(local);
        checkPositive("indexer n_tokens", nTokens);
        checkPositive("indexer n_kv", nKv);
        long blocks = ((long) nKv + local - 1) / local;
        return new long[]{(long) nTokens * blocks * local};
    }

    /** 1-D global size of {@code LLM_TopK_Rows}: one work-group per row. */
    public static long[] topKGlobal(int nRows, int local) {
        checkTopKLocal(local);
        checkPositive("top-k n_rows", nRows);
        return new long[]{(long) nRows * local};
    }

    /** Blocks of {@code nKv} rows, {@code ceil(nKv / blockSize)}. */
    public static int blockCount(int nKv, int blockSize) {
        if (blockSize < 1 || nKv < 0) {
            throw new IllegalArgumentException("block selection needs a positive block size and non-negative rows,"
                    + " got " + blockSize + " / " + nKv);
        }
        return (nKv + blockSize - 1) / blockSize;
    }

    /** 1-D global size of {@code LLM_DSA_BlockScore}: one work-item per (token, block). */
    public static long[] blockScoreGlobal(int nTokens, int nKv, int blockSize, int local) {
        if (local < 1) {
            throw new IllegalArgumentException("block score local size must be positive, got " + local);
        }
        checkPositive("block score n_tokens", nTokens);
        long n = (long) nTokens * blockCount(nKv, blockSize);
        return new long[]{(n + local - 1) / local * local};
    }

    /** 1-D global size of {@code LLM_DSA_RestrictBlocks}: one work-item per (token, row). */
    public static long[] restrictGlobal(int nTokens, int nKv, int local) {
        if (local < 1) {
            throw new IllegalArgumentException("block restriction local size must be positive, got " + local);
        }
        checkPositive("block restriction n_tokens", nTokens);
        long n = (long) nTokens * nKv;
        return new long[]{(n + local - 1) / local * local};
    }

    /** 1-D global size of {@code LLM_MLA_SparseDecode}: one work-group per head. */
    public static long[] sparseDecodeGlobal(int nHead, int local) {
        checkSparseMlaLocal(local);
        checkPositive("sparse MLA n_head", nHead);
        return new long[]{(long) nHead * local};
    }

    /** 1-D global size of {@code LLM_MLA_SparsePrefill}: one work-group per (token, head). */
    public static long[] sparsePrefillGlobal(int nTokens, int nHead, int local) {
        checkSparseMlaLocal(local);
        checkPositive("sparse MLA n_tokens", nTokens);
        checkPositive("sparse MLA n_head", nHead);
        return new long[]{(long) nTokens * nHead * local};
    }

    /** Local size array {@code {local}}. */
    public static long[] local(int local) {
        return new long[]{local};
    }

    /**
     * Checks the indexer shape the score kernel supports.
     *
     * @throws IllegalArgumentException for a non-positive head count or a head size outside
     *                                  {@code [1, INDEXER_MAX_HEAD_DIM]}
     */
    public static void checkIndexer(int nIdxHeads, int idxDim) {
        checkPositive("DSA indexer head count", nIdxHeads);
        if (idxDim < 1 || idxDim > INDEXER_MAX_HEAD_DIM) {
            throw new IllegalArgumentException("DSA indexer head size " + idxDim + " is outside [1, "
                    + INDEXER_MAX_HEAD_DIM + "] supported by LLM_DSA_IndexerScore");
        }
    }

    /**
     * Checks that indexer cache rows of {@code idxDim} values can be stored and scored in {@code cache}: a row of a
     * block type must be whole 32-value blocks, as the memory planner stores it.
     *
     * @throws IllegalArgumentException for a block-type row that is not whole blocks
     */
    public static void checkIndexerCache(KvDType cache, int idxDim) {
        Objects.requireNonNull(cache, "indexer cache type");
        if (!cache.supportsWidth(idxDim)) {
            throw new IllegalArgumentException("DSA indexer rows of " + idxDim + " values are not whole " + cache
                    + " blocks of " + cache.blockSize());
        }
    }

    /**
     * Checks the indexer mask parameters: every SWA type other than NONE divides or compares by the window, so it
     * needs {@code nSwa >= 1} (a zero window would divide by zero in the CHUNKED kernel path).
     *
     * @throws IllegalArgumentException for a null type or a non-positive window with a windowed type
     */
    public static void checkIndexerMask(SwaType swaType, int nSwa) {
        if (swaType == null) {
            throw new IllegalArgumentException("DSA indexer SWA type must not be null");
        }
        if (swaType != SwaType.NONE && nSwa < 1) {
            throw new IllegalArgumentException("DSA indexer SWA type " + swaType + " needs a positive window, got "
                    + nSwa);
        }
    }

    /** @throws IllegalArgumentException for a top-k width outside {@code [1, TOPK_MAX]} */
    public static void checkTopK(int k) {
        if (k < 1 || k > TOPK_MAX) {
            throw new IllegalArgumentException("DSA top-k " + k + " is outside [1, " + TOPK_MAX + "] supported by "
                    + TOPK_ROWS + " and the sparse MLA kernels");
        }
    }

    /**
     * Checks the sparse MLA shape.
     *
     * @throws IllegalArgumentException for non-positive heads or latent rank, a negative rope dim, or a selection
     *                                  width outside {@code [1, TOPK_MAX]}
     */
    public static void checkSparseMla(int nHead, int kvLora, int ropeDim, int nSel) {
        checkPositive("sparse MLA head count", nHead);
        checkPositive("sparse MLA kv_lora", kvLora);
        if (ropeDim < 0) {
            throw new IllegalArgumentException("sparse MLA rope dim must be non-negative, got " + ropeDim);
        }
        checkTopK(nSel);
    }

    /**
     * Checks that the latent cache rows ({@code kvLora + ropeDim} values) can be read by the sparse MLA kernel of
     * their type: a row of a block type must be whole 32-value blocks, as the memory planner stores it.
     *
     * @throws IllegalArgumentException for a block-type row that is not whole blocks
     */
    public static void checkLatentCache(KvDType latent, int kvLora, int ropeDim) {
        Objects.requireNonNull(latent, "latent");
        long width = (long) kvLora + ropeDim;
        if (!latent.supportsWidth(width)) {
            throw new IllegalArgumentException("sparse MLA latent rows of " + width + " values are not whole "
                    + latent + " blocks of " + latent.blockSize());
        }
    }

    /** @throws IllegalArgumentException unless {@code 1 <= local <= LLM_DIMX} */
    public static void checkIndexerLocal(int local) {
        if (local < 1 || local > LlmKernelConstants.LLM_DIMX) {
            throw new IllegalArgumentException("LLM_DSA_IndexerScore work-group size " + local + " is outside [1, "
                    + LlmKernelConstants.LLM_DIMX + "]");
        }
    }

    /** @throws IllegalArgumentException unless {@code local} is a power of two {@code <= LLM_DIMX} */
    public static void checkTopKLocal(int local) {
        if (local < 1 || local > LlmKernelConstants.LLM_DIMX || Integer.bitCount(local) != 1) {
            throw new IllegalArgumentException(TOPK_ROWS + " work-group size " + local
                    + " must be a power of two <= " + LlmKernelConstants.LLM_DIMX);
        }
    }

    /** @throws IllegalArgumentException unless {@code local} is a power of two in {@code [32, LLM_DIMX]} */
    public static void checkSparseMlaLocal(int local) {
        if (local < 32 || local > LlmKernelConstants.LLM_DIMX || Integer.bitCount(local) != 1) {
            throw new IllegalArgumentException("sparse MLA work-group size " + local
                    + " must be a power of two in [32, " + LlmKernelConstants.LLM_DIMX + "]");
        }
    }

    private static void checkPositive(String what, int v) {
        if (v < 1) {
            throw new IllegalArgumentException(what + " must be positive, got " + v);
        }
    }
}

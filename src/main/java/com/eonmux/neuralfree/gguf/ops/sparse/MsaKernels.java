package com.eonmux.neuralfree.gguf.ops.sparse;

import com.eonmux.neuralfree.gguf.kernels.LlmKernelConstants;
import com.eonmux.neuralfree.gguf.memory.KvDType;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * Kernel names, argument indices, codes, launch geometry and preconditions of the MiniMax sparse-attention (MSA)
 * and QSA k-pool kernels ({@code kernels/llm/sparse/msa.cl}, {@code sparse/qsa.cl} and their GLSL twins
 * {@code msa.comp}, {@code qsa.comp}).
 *
 * <p>Argument lists ({@code (dyn)} marks the per-step arguments with an {@code ARG_*} constant; {@code <KV>}
 * buffers hold rows of any {@link KvDType} selected by a {@link #kvTypeCode(KvDType)} scalar, or, for the QSA
 * kernels, by the {@code _<KV>} variant of the indexer cache's type, {@link #kpoolUpdate(KvDType)} and
 * {@link #kpoolScore(KvDType)}):
 * <pre>
 * LLM_MSA_BlockScore  (i n_tokens(dyn), i n_pos(dyn), i n_kv_heads, i idx_dim, i blk, i local_blocks, i kv_rows,
 *                      i ik_type, i iq_off(dyn), R iq, R&lt;KV&gt; ik_cache, R&lt;int&gt; q_pos, R&lt;int&gt; kv_pos,
 *                      W block_scores)
 * LLM_MSA_GatherAttn  (i n_tokens(dyn), i n_head, i n_kv_heads, i hd, i blk, i K, f kq_scale, i sel_heads, i tail,
 *                      i kv_rows, i k_row_stride, i v_row_stride, i k_type, i v_type, i q_stride, i q_off(dyn),
 *                      i out_stride, i out_off(dyn), R q, R&lt;KV&gt; k_cache, R&lt;KV&gt; v_cache, R&lt;int&gt; q_pos,
 *                      R&lt;int&gt; kv_pos, R&lt;int&gt; blocks, W out)
 * LLM_QSA_KpoolUpdate_&lt;KV&gt; (i n_new(dyn), i first_pool(dyn), i stage(dyn), i mode, i kpool, i idx_dim,
 *                      i row_width, i gate_off, i pooled_off, i kv_rows, R ape, RWn&lt;KV&gt; idx_cache, RWn pooled)
 * LLM_QSA_KpoolScore_&lt;KV&gt;  (i n_tokens(dyn), i n_pool(dyn), i n_idx_heads, i idx_dim, i kpool, i row_width,
 *                      i pooled_off, i kv_rows, i has_w, f w_scale, i iq_off(dyn), i w_off(dyn), R iq, R w,
 *                      R&lt;KV&gt; idx_cache, R&lt;int&gt; q_pos, R&lt;int&gt; kv_pos, W scores)
 * </pre>
 * Cells are addressed through the KV group's {@code kv_pos} plus {@code kv_rows} (row of position {@code p} is
 * {@code p % kv_rows}, present when {@code kv_pos[row] == p}; the memory provides {@code kv_pos}, see
 * {@code GpuMsaCache}); the gather kernel takes {@code q_pos}/{@code kv_pos} for causal visibility, run-time KV type
 * codes, row strides and q/out views, and {@code sel_heads}/{@code tail} so the QSA k-pool selection (glm5-next,
 * qwen4exp) runs on the same kernel. The QSA indexer cache has the context's K type (llama.cpp
 * src/llama-memory-hybrid-idx.cpp:71); a block-type cache keeps every {@code idx_dim} slot whole 32-value blocks
 * ({@link #checkKpoolCache}), as llama.cpp's {@code ggml_row_size} slot views do (llama-memory-hybrid-idx.cpp:575-598).
 *
 * <p>MSA step (minimax-m3 sparse layer): {@code LLM_MSA_BlockScore} &rarr; {@code LLM_TopK_Rows(n_rows =
 * n_tokens*n_kv_heads, n_cols = blockCount(n_pos, blk), k = K)} &rarr; {@code LLM_MSA_GatherAttn(sel_heads =
 * n_kv_heads, tail = 0)}, with {@code K = topKBlocks}: the top-k pads with -1 when fewer blocks are visible, which
 * selects the same visible cells as llama.cpp's {@code min(topk_blocks, nblk)} (src/models/minimax-m3.cpp:384).
 *
 * <p>QSA step (full-indexer layer): store the step's key (| gate) rows, then {@code LLM_QSA_KpoolUpdate} stage
 * POOL for the {@link #newPoolCount new pools} (qwen4exp then applies the RMS norm and RoPE of the pooled key to
 * the pooled rows), stage STORE; {@code LLM_QSA_KpoolScore} over the {@link #completePools complete pools};
 * {@code LLM_TopK_Rows(n_tokens, n_pool, selectedPools)}; {@code LLM_MSA_GatherAttn(blk = kpool, K =
 * selectedPools, sel_heads = 1, tail = kpool_select_tail)}. When no pool is complete the score and top-k are
 * skipped and the block list is filled with -1. The CPU twins are {@link CpuMsa} and {@link CpuQsa}.
 */
public final class MsaKernels {

    public static final String MSA_BLOCK_SCORE = "LLM_MSA_BlockScore";
    public static final String MSA_GATHER_ATTN = "LLM_MSA_GatherAttn";
    /** Name prefixes of the per-KV-type QSA variants. */
    public static final String QSA_KPOOL_UPDATE_PREFIX = "LLM_QSA_KpoolUpdate_";
    public static final String QSA_KPOOL_SCORE_PREFIX = "LLM_QSA_KpoolScore_";

    /** Every kernel of this family. */
    public static final List<String> ALL;

    static {
        List<String> all = new ArrayList<>();
        all.add(MSA_BLOCK_SCORE);
        all.add(MSA_GATHER_ATTN);
        all.addAll(KvDType.kernelVariants(QSA_KPOOL_UPDATE_PREFIX));
        all.addAll(KvDType.kernelVariants(QSA_KPOOL_SCORE_PREFIX));
        ALL = Collections.unmodifiableList(all);
    }

    /** {@code LLM_MSA_BlockScore}: argument count and per-step argument indices. */
    public static final int BLOCK_SCORE_ARG_COUNT = 14;
    public static final int ARG_BS_N_TOKENS = 0;
    public static final int ARG_BS_N_POS = 1;
    public static final int ARG_BS_IQ_OFF = 8;

    /** {@code LLM_MSA_GatherAttn}: argument count and per-step argument indices. */
    public static final int GATHER_ARG_COUNT = 25;
    public static final int ARG_GA_N_TOKENS = 0;
    public static final int ARG_GA_Q_OFF = 15;
    public static final int ARG_GA_OUT_OFF = 17;

    /** {@code LLM_QSA_KpoolUpdate}: argument count and per-step argument indices. */
    public static final int KPOOL_UPDATE_ARG_COUNT = 13;
    public static final int ARG_KU_N_NEW = 0;
    public static final int ARG_KU_FIRST_POOL = 1;
    public static final int ARG_KU_STAGE = 2;

    /** {@code LLM_QSA_KpoolScore}: argument count and per-step argument indices. */
    public static final int KPOOL_SCORE_ARG_COUNT = 18;
    public static final int ARG_KS_N_TOKENS = 0;
    public static final int ARG_KS_N_POOL = 1;
    public static final int ARG_KS_IQ_OFF = 10;
    public static final int ARG_KS_W_OFF = 11;

    /** {@code LLM_QSA_KpoolUpdate} pooling modes ({@code NF_QSA_MODE_*}). */
    public static final int KPOOL_MODE_GATED = 0;
    public static final int KPOOL_MODE_MEAN = 1;
    /** {@code LLM_QSA_KpoolUpdate} stages ({@code NF_QSA_STAGE_*}). */
    public static final int KPOOL_STAGE_POOL = 0;
    public static final int KPOOL_STAGE_STORE = 1;

    /** Cached indexer row layouts ({@code indexer_kpool_row}): key | gate | pooled, and raw key | pooled. */
    public static final int KPOOL_ROW_GATED = 3;
    public static final int KPOOL_ROW_MEAN = 2;

    /** Local-force bias added to the most recent blocks ({@code NF_MSA_LOCAL_BIAS}, minimax-m3.cpp:116). */
    public static final float LOCAL_BIAS = 1e30f;
    /** Largest index-key or indexer head size ({@code NF_MSA_QSTAGE}, {@code NF_QSA_QSTAGE}). */
    public static final int MAX_IDX_DIM = 2048;
    /** Largest attention head size of the gather kernel ({@code NF_MSA_HD_MAX}). */
    public static final int GATHER_MAX_HEAD_DIM = 1024;
    /** Output values per work-item of the gather kernel ({@code NF_MSA_ACC}): {@code hd <= local * GATHER_ACC}. */
    public static final int GATHER_ACC = 8;
    /** Largest selection width {@code K}: the {@code LLM_TopK_Rows} limit ({@code NF_TOPK_MAX}). */
    public static final int MAX_SELECTION = 2048;

    /** Recommended work-group size of {@code LLM_MSA_GatherAttn}. */
    public static final int GATHER_LOCAL = 128;
    /** Recommended work-group size of {@code LLM_QSA_KpoolScore}. */
    public static final int KPOOL_SCORE_LOCAL = 128;
    /** Recommended work-group size of {@code LLM_QSA_KpoolUpdate}. */
    public static final int KPOOL_UPDATE_LOCAL = 256;

    private MsaKernels() {
    }

    /** {@code NF_KV_*} code of a KV element type ({@link KvDType#code()}). */
    public static int kvTypeCode(KvDType type) {
        return Objects.requireNonNull(type, "type").code();
    }

    /** {@code LLM_QSA_KpoolUpdate} variant for an indexer cache of {@code type}. */
    public static String kpoolUpdate(KvDType type) {
        return Objects.requireNonNull(type, "indexer cache type").kernel(QSA_KPOOL_UPDATE_PREFIX);
    }

    /** {@code LLM_QSA_KpoolScore} variant for an indexer cache of {@code type}. */
    public static String kpoolScore(KvDType type) {
        return Objects.requireNonNull(type, "indexer cache type").kernel(QSA_KPOOL_SCORE_PREFIX);
    }

    /** Position blocks covering positions {@code [0, nPos)}: {@code ceil(nPos / blk)}. */
    public static int blockCount(int nPos, int blk) {
        checkPositive("MSA position count", nPos);
        checkPositive("MSA block size", blk);
        return (int) (((long) nPos + blk - 1) / blk);
    }

    /**
     * Recommended work-group size of {@code LLM_MSA_BlockScore}: the smallest power of two covering one block,
     * clamped to {@code [32, LLM_DIMX]}.
     */
    public static int blockScoreLocal(int blk) {
        checkPositive("MSA block size", blk);
        int l = 32;
        while (l < blk && l < LlmKernelConstants.LLM_DIMX) {
            l <<= 1;
        }
        return l;
    }

    /** 1-D global size of {@code LLM_MSA_BlockScore}: one work-group per (token, position block). */
    public static long[] blockScoreGlobal(int nTokens, int nPos, int blk, int local) {
        checkReduceLocal(MSA_BLOCK_SCORE, local);
        checkPositive("MSA n_tokens", nTokens);
        return new long[]{(long) nTokens * blockCount(nPos, blk) * local};
    }

    /** 1-D global size of {@code LLM_MSA_GatherAttn}: one work-group per (token, query head). */
    public static long[] gatherGlobal(int nTokens, int nHead, int local) {
        checkReduceLocal(MSA_GATHER_ATTN, local);
        checkPositive("MSA gather n_tokens", nTokens);
        checkPositive("MSA gather n_head", nHead);
        return new long[]{(long) nTokens * nHead * local};
    }

    /** 1-D global size of {@code LLM_QSA_KpoolUpdate}: {@code nNew * idxDim} work-items rounded up to {@code local}. */
    public static long[] kpoolUpdateGlobal(int nNew, int idxDim, int local) {
        checkAnyLocal("LLM_QSA_KpoolUpdate", local);
        checkPositive("QSA new pools", nNew);
        checkPositive("QSA indexer head size", idxDim);
        long n = (long) nNew * idxDim;
        return new long[]{(n + local - 1) / local * local};
    }

    /** 1-D global size of {@code LLM_QSA_KpoolScore}: one work-group per (token, block of {@code local} pools). */
    public static long[] kpoolScoreGlobal(int nTokens, int nPool, int local) {
        checkAnyLocal("LLM_QSA_KpoolScore", local);
        checkPositive("QSA n_tokens", nTokens);
        checkPositive("QSA pool count", nPool);
        long blocks = ((long) nPool + local - 1) / local;
        return new long[]{(long) nTokens * blocks * local};
    }

    /** Local size array {@code {local}}. */
    public static long[] local(int local) {
        return new long[]{local};
    }

    /** Pooling mode of a cached indexer row layout ({@code indexer_kpool_row}). */
    public static int kpoolMode(int kpoolRow) {
        if (kpoolRow == KPOOL_ROW_GATED) {
            return KPOOL_MODE_GATED;
        }
        if (kpoolRow == KPOOL_ROW_MEAN) {
            return KPOOL_MODE_MEAN;
        }
        throw new IllegalArgumentException("QSA k-pool row of " + kpoolRow + " head slots is not supported; expected "
                + KPOOL_ROW_GATED + " (key | gate | pooled, glm5-next) or " + KPOOL_ROW_MEAN
                + " (raw key | pooled, qwen4exp)");
    }

    /** {@code gate_off}: value offset of the gate inside a cached row (0, unused, in mode MEAN). */
    public static int kpoolGateOffset(int kpoolRow, int idxDim) {
        checkPositive("QSA indexer head size", idxDim);
        return kpoolMode(kpoolRow) == KPOOL_MODE_GATED ? idxDim : 0;
    }

    /** {@code pooled_off}: value offset of the pooled key inside a cached row ({@code kpool_access}: after the per-token part). */
    public static int kpoolPooledOffset(int kpoolRow, int idxDim) {
        checkPositive("QSA indexer head size", idxDim);
        kpoolMode(kpoolRow);
        return (kpoolRow - 1) * idxDim;
    }

    /** Complete pools among positions {@code [0, nPos)}: {@code nPos / kpool}. */
    public static int completePools(int nPos, int kpool) {
        checkKpoolSize(kpool);
        if (nPos < 0) {
            throw new IllegalArgumentException("QSA position count must be non-negative, got " + nPos);
        }
        return nPos / kpool;
    }

    /** First pool completed by a step at positions {@code [pos0, pos0 + nTokens)}: {@code pos0 / kpool}. */
    public static int firstNewPool(int pos0, int kpool) {
        checkKpoolSize(kpool);
        if (pos0 < 0) {
            throw new IllegalArgumentException("QSA step position must be non-negative, got " + pos0);
        }
        return pos0 / kpool;
    }

    /**
     * Pools whose last member the step at positions {@code [pos0, pos0 + nTokens)} writes, i.e. the pools
     * llama.cpp re-pools for that ubatch ({@code kpool_build_state}, src/llama-memory-hybrid-idx.cpp:684-709):
     * {@code (pos0 + nTokens) / kpool - pos0 / kpool}.
     */
    public static int newPoolCount(int pos0, int nTokens, int kpool) {
        checkPositive("QSA step tokens", nTokens);
        return completePools(pos0 + nTokens, kpool) - firstNewPool(pos0, kpool);
    }

    /**
     * Pools selected per token: {@code floor(top_k / kpool)} (src/models/glm5-next.cpp:975, qwen4exp.cpp:812). The
     * {@code LLM_TopK_Rows} width; it may exceed the complete pool count, the top-k then pads with -1, which keeps the
     * same cells as llama.cpp's {@code min(n_pool, top_k / kpool)}. Zero when {@code top_k < kpool}: only the tail
     * block is attended.
     */
    public static int selectedPools(int topK, int kpool) {
        checkKpoolSize(kpool);
        checkPositive("QSA indexer top-k", topK);
        return topK / kpool;
    }

    /**
     * Checks the {@code LLM_MSA_BlockScore} configuration.
     *
     * @throws IllegalArgumentException for a shape or local size the kernel cannot run
     */
    public static void checkBlockScore(int nKvHeads, int idxDim, int blk, int localBlocks, int kvRows, KvDType ikType,
                                       int local) {
        checkPositive("MSA index heads", nKvHeads);
        checkIdxDim("MSA index key", idxDim);
        checkPositive("MSA block size", blk);
        checkPositive("MSA KV rows", kvRows);
        if (localBlocks < 0) {
            throw new IllegalArgumentException("MSA local block count must be non-negative, got " + localBlocks);
        }
        Objects.requireNonNull(ikType, "ikType");
        if (!ikType.supportsWidth(idxDim)) {
            throw new IllegalArgumentException("MSA index keys of " + idxDim + " values cannot be stored as " + ikType);
        }
        checkReduceLocal(MSA_BLOCK_SCORE, local);
    }

    /**
     * Checks the {@code LLM_MSA_GatherAttn} configuration; strides are in values.
     *
     * @throws IllegalArgumentException for a shape, selection or local size the kernel cannot run
     */
    public static void checkGatherAttn(int nHead, int nKvHeads, int hd, int blk, int k, int selHeads, int kvRows,
                                       int kRowStride, int vRowStride, KvDType kType, KvDType vType, int local) {
        checkPositive("MSA gather n_head", nHead);
        checkPositive("MSA gather n_kv_heads", nKvHeads);
        if (nHead % nKvHeads != 0) {
            throw new IllegalArgumentException("MSA gather n_head " + nHead + " is not a multiple of n_kv_heads " + nKvHeads);
        }
        if (selHeads != 1 && selHeads != nKvHeads) {
            throw new IllegalArgumentException("MSA gather selection heads must be 1 or n_kv_heads (" + nKvHeads + "), got "
                    + selHeads);
        }
        if (hd < 4 || hd % 4 != 0 || hd > GATHER_MAX_HEAD_DIM) {
            throw new IllegalArgumentException("MSA gather head size " + hd + " must be a multiple of 4 in [4, "
                    + GATHER_MAX_HEAD_DIM + "]");
        }
        checkReduceLocal(MSA_GATHER_ATTN, local);
        if (hd > local * GATHER_ACC) {
            throw new IllegalArgumentException("MSA gather head size " + hd + " exceeds " + GATHER_ACC
                    + " values per work-item of a " + local + "-wide work-group");
        }
        checkPositive("MSA gather block size", blk);
        if (k < 0 || k > MAX_SELECTION) {
            throw new IllegalArgumentException("MSA gather selection width " + k + " is outside [0, " + MAX_SELECTION + "]");
        }
        checkPositive("MSA gather KV rows", kvRows);
        checkRowStride("K", kRowStride, nKvHeads, hd, kType);
        checkRowStride("V", vRowStride, nKvHeads, hd, vType);
    }

    /**
     * Checks the {@code LLM_QSA_KpoolUpdate} configuration.
     *
     * @throws IllegalArgumentException for a pool size, row layout or mode the kernel cannot run
     */
    public static void checkKpool(int kpool, int idxDim, int rowWidth, int gateOff, int pooledOff, int kvRows, int mode) {
        checkKpoolSize(kpool);
        checkIdxDim("QSA indexer", idxDim);
        checkPositive("QSA KV rows", kvRows);
        if (kpool > kvRows) {
            throw new IllegalArgumentException("QSA pool of " + kpool + " positions exceeds the " + kvRows + " cache rows");
        }
        if (mode != KPOOL_MODE_GATED && mode != KPOOL_MODE_MEAN) {
            throw new IllegalArgumentException("unknown QSA pooling mode " + mode);
        }
        checkSlot("pooled", pooledOff, idxDim, rowWidth);
        if (mode == KPOOL_MODE_GATED) {
            checkSlot("gate", gateOff, idxDim, rowWidth);
        }
    }

    /**
     * Checks that an indexer cache of {@code type} can hold the k-pool row layout: rows of {@code rowWidth} values
     * whose key, pooled and (GATED) gate slots of {@code idxDim} values start at block boundaries, so a block-type
     * cache keeps every slot whole 32-value blocks and the pooled store rewrites whole blocks.
     *
     * @throws IllegalArgumentException for a slot that is not whole blocks of {@code type}
     */
    public static void checkKpoolCache(KvDType type, int idxDim, int rowWidth, int gateOff, int pooledOff, int mode) {
        Objects.requireNonNull(type, "indexer cache type");
        boolean whole = type.supportsWidth(idxDim) && type.supportsWidth(rowWidth) && type.supportsWidth(pooledOff)
                && (mode != KPOOL_MODE_GATED || type.supportsWidth(gateOff));
        if (!whole) {
            throw new IllegalArgumentException("QSA indexer rows of " + rowWidth + " values with " + idxDim
                    + "-value slots at gate " + gateOff + " and pooled " + pooledOff + " are not whole " + type
                    + " blocks of " + type.blockSize());
        }
    }

    /**
     * Checks the {@code LLM_QSA_KpoolScore} configuration.
     *
     * @throws IllegalArgumentException for a shape, row layout or local size the kernel cannot run
     */
    public static void checkKpoolScore(int nIdxHeads, int idxDim, int kpool, int rowWidth, int pooledOff, int kvRows,
                                       float wScale, int local) {
        checkPositive("QSA indexer heads", nIdxHeads);
        checkIdxDim("QSA indexer", idxDim);
        checkKpoolSize(kpool);
        checkPositive("QSA KV rows", kvRows);
        checkSlot("pooled", pooledOff, idxDim, rowWidth);
        if (!(wScale > 0f) || Float.isInfinite(wScale)) {
            throw new IllegalArgumentException("QSA head-weight scale must be positive and finite, got " + wScale);
        }
        checkAnyLocal("LLM_QSA_KpoolScore", local);
    }

    /** @throws IllegalArgumentException unless {@code local} is a power of two in {@code [32, LLM_DIMX]} */
    static void checkReduceLocal(String kernel, int local) {
        if (local < 32 || local > LlmKernelConstants.LLM_DIMX || Integer.bitCount(local) != 1) {
            throw new IllegalArgumentException(kernel + " work-group size " + local + " must be a power of two in [32, "
                    + LlmKernelConstants.LLM_DIMX + "]");
        }
    }

    /** @throws IllegalArgumentException unless {@code 1 <= local <= LLM_DIMX} */
    static void checkAnyLocal(String kernel, int local) {
        if (local < 1 || local > LlmKernelConstants.LLM_DIMX) {
            throw new IllegalArgumentException(kernel + " work-group size " + local + " is outside [1, "
                    + LlmKernelConstants.LLM_DIMX + "]");
        }
    }

    private static void checkRowStride(String which, int stride, int nKvHeads, int hd, KvDType type) {
        Objects.requireNonNull(type, which + " type");
        if ((long) stride < (long) nKvHeads * hd) {
            throw new IllegalArgumentException("MSA gather " + which + " row stride " + stride + " is below "
                    + nKvHeads + " heads of " + hd);
        }
        if (!type.supportsWidth(stride)) {
            throw new IllegalArgumentException("MSA gather " + which + " rows of " + stride + " values cannot be " + type);
        }
    }

    private static void checkSlot(String what, int off, int idxDim, int rowWidth) {
        if (off < 0 || (long) off + idxDim > rowWidth) {
            throw new IllegalArgumentException("QSA " + what + " slot [" + off + ", " + ((long) off + idxDim)
                    + ") lies outside the cached row of " + rowWidth + " values");
        }
    }

    private static void checkIdxDim(String what, int idxDim) {
        if (idxDim < 1 || idxDim > MAX_IDX_DIM) {
            throw new IllegalArgumentException(what + " head size " + idxDim + " is outside [1, " + MAX_IDX_DIM + "]");
        }
    }

    static void checkKpoolSize(int kpool) {
        if (kpool < 1) {
            throw new IllegalArgumentException("QSA pool size must be positive, got " + kpool);
        }
    }

    private static void checkPositive(String what, int v) {
        if (v < 1) {
            throw new IllegalArgumentException(what + " must be positive, got " + v);
        }
    }
}

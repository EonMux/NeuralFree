package com.eonmux.neuralfree.gguf.ops.sparse;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.memory.KvDType;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * Names, argument lists, launch geometry and preconditions of the Step5 sparse region attention kernels
 * ({@code kernels/llm/sparse/step5.cl} and their GLSL twins {@code shaders/llm/step5_*.comp}). The CPU semantic
 * reference is {@link CpuStep5Sparse}, {@link CpuStep5RegionCache} and {@link CpuStep5Attn}.
 *
 * <p>Argument lists:
 * <pre>
 * LLM_Step5_Cast          (i n(dyn), i mode, R x, W y)
 * LLM_Step5_Summary       (i n_tokens(dyn), i region_block, i capacity, i proxy, R key, R z, R&lt;int&gt; q_pos,
 *                          W ring, W summary)
 * LLM_Step5_RegionScores  (i n_tokens(dyn), i n_kv, i n_idx_heads, i heads_per_group, i proxy, i regions,
 *                          i region_block, R iq, R iw, R summary, R&lt;int&gt; q_pos, W scores)
 * LLM_Step5_SelectRegions (i n_rows(dyn), i k, i n_kv, i region_block, i regions, R&lt;int&gt; topk,
 *                          R&lt;int&gt; q_pos, W&lt;int&gt; sel)
 * LLM_Step5_Attn_&lt;KV&gt;     (i n_tokens(dyn), i n_head, i n_kv, i head_dim, i region_block, i slot_count,
 *                          i mode, i window, f scale, i kv_rows, i k_row_stride, i v_row_stride, R q,
 *                          R&lt;KV&gt; k_cache, R&lt;KV&gt; v_cache, R&lt;int&gt; kv_pos, R&lt;int&gt; sel,
 *                          R&lt;int&gt; q_pos, R gate, W out)
 * </pre>
 * {@code <KV>} is the attention cache's {@link KvDType} ({@link #attn(KvDType)}); the per-token-head types are not
 * served because the kernel does not apply their row scales.</p>
 */
public final class Step5Kernels {

    public static final String CAST = "LLM_Step5_Cast";
    public static final String SUMMARY = "LLM_Step5_Summary";
    public static final String REGION_SCORES = "LLM_Step5_RegionScores";
    public static final String SELECT_REGIONS = "LLM_Step5_SelectRegions";
    /** Name prefix of the per-KV-type attention variants. */
    public static final String ATTN_PREFIX = "LLM_Step5_Attn_";

    /** Every kernel of this family (every attention variant included). */
    public static final List<String> ALL;

    static {
        List<String> all = new ArrayList<>();
        all.add(CAST);
        all.add(SUMMARY);
        all.add(REGION_SCORES);
        all.add(SELECT_REGIONS);
        all.addAll(KvDType.kernelVariants(ATTN_PREFIX, t -> !t.perTokenHead() && t.hasSparseKernel()));
        ALL = Collections.unmodifiableList(all);
    }

    /** {@code LLM_Step5_Cast}: argument count and the per-step argument indices. */
    public static final int CAST_ARG_COUNT = 4;
    public static final int ARG_CAST_N = 0;

    /** {@code LLM_Step5_Summary}: argument count and the per-step argument. */
    public static final int SUMMARY_ARG_COUNT = 9;
    public static final int ARG_SUMMARY_N_TOKENS = 0;

    /** {@code LLM_Step5_RegionScores}: argument count and the per-step argument. */
    public static final int REGION_SCORES_ARG_COUNT = 12;
    public static final int ARG_REGION_SCORES_N_TOKENS = 0;

    /** {@code LLM_Step5_SelectRegions}: argument count and the per-step argument. */
    public static final int SELECT_ARG_COUNT = 8;
    public static final int ARG_SELECT_N_ROWS = 0;

    /** {@code LLM_Step5_Attn_<KV>}: argument count and the per-step arguments. */
    public static final int ATTN_ARG_COUNT = 20;
    public static final int ARG_ATTN_N_TOKENS = 0;
    public static final int ARG_ATTN_SLOT_COUNT = 5;
    public static final int ARG_ATTN_MODE = 6;
    public static final int ARG_ATTN_WINDOW = 7;

    /** Masking mode of {@code LLM_Step5_Attn}: the selected regions, or the sliding window. */
    public static final int ATTN_MODE_SPARSE = 0;
    public static final int ATTN_MODE_WINDOW = 1;

    /** Rounding mode of {@code LLM_Step5_Cast}. */
    public static final int CAST_MODE_BF16 = 0;
    public static final int CAST_MODE_E4M3 = 1;

    /** Recommended work-group size of the element-wise kernels. */
    public static final int ELEMENT_LOCAL = 256;

    private Step5Kernels() {
    }

    /** {@code LLM_Step5_Attn_<KV>} for the attention cache's element type. */
    public static String attn(KvDType cache) {
        Objects.requireNonNull(cache, "cache type");
        if (cache.perTokenHead()) {
            throw new UnsupportedOperationException("Step5 sparse attention has no kernel for the per-token-head KV"
                    + " type " + cache + ": its row scales are not applied");
        }
        return cache.kernel(ATTN_PREFIX);
    }

    /** Arguments of {@code LLM_Step5_Cast}. */
    public static Object[] castArgs(int n, int mode, GpuBuffer x, GpuBuffer y) {
        checkNonNegative(n, "cast n");
        if (mode != CAST_MODE_BF16 && mode != CAST_MODE_E4M3) {
            throw new IllegalArgumentException("Step5 cast mode " + mode + " is neither BF16 nor E4M3");
        }
        Objects.requireNonNull(x, "x");
        Objects.requireNonNull(y, "y");
        return new Object[]{n, mode, x, y};
    }

    /** Arguments of {@code LLM_Step5_Summary}. */
    public static Object[] summaryArgs(int nTokens, int regionBlock, int capacity, int proxy, GpuBuffer key,
                                       GpuBuffer z, GpuBuffer qPos, GpuBuffer ring, GpuBuffer summary) {
        checkPositive(regionBlock, "summary region block");
        checkPositive(capacity, "summary ring capacity");
        checkPositive(proxy, "summary proxy width");
        checkNonNegative(nTokens, "summary token count");
        return new Object[]{nTokens, regionBlock, capacity, proxy, Objects.requireNonNull(key, "key"),
                Objects.requireNonNull(z, "z"), Objects.requireNonNull(qPos, "qPos"),
                Objects.requireNonNull(ring, "ring"), Objects.requireNonNull(summary, "summary")};
    }

    /** Arguments of {@code LLM_Step5_RegionScores}. */
    public static Object[] regionScoreArgs(int nTokens, int nKv, int nIdxHeads, int headsPerGroup, int proxy,
                                           int regions, int regionBlock, GpuBuffer iq, GpuBuffer iw,
                                           GpuBuffer summary, GpuBuffer qPos, GpuBuffer scores) {
        checkPositive(nKv, "region score KV groups");
        checkPositive(nIdxHeads, "region score indexer heads");
        checkPositive(headsPerGroup, "region score heads per group");
        checkPositive(proxy, "region score proxy width");
        checkPositive(regions, "region score region count");
        checkPositive(regionBlock, "region score region block");
        checkNonNegative(nTokens, "region score token count");
        return new Object[]{nTokens, nKv, nIdxHeads, headsPerGroup, proxy, regions, regionBlock,
                Objects.requireNonNull(iq, "iq"), Objects.requireNonNull(iw, "iw"),
                Objects.requireNonNull(summary, "summary"), Objects.requireNonNull(qPos, "qPos"),
                Objects.requireNonNull(scores, "scores")};
    }

    /** Arguments of {@code LLM_Step5_SelectRegions}. */
    public static Object[] selectArgs(int nRows, int k, int nKv, int regionBlock, int regions, GpuBuffer topK,
                                      GpuBuffer qPos, GpuBuffer sel) {
        checkPositive(k, "selection width");
        checkPositive(nKv, "selection KV groups");
        checkPositive(regionBlock, "selection region block");
        checkPositive(regions, "selection region count");
        checkNonNegative(nRows, "selection row count");
        return new Object[]{nRows, k, nKv, regionBlock, regions, Objects.requireNonNull(topK, "topK"),
                Objects.requireNonNull(qPos, "qPos"), Objects.requireNonNull(sel, "sel")};
    }

    /** Arguments of {@code LLM_Step5_Attn_<KV>}. */
    public static Object[] attnArgs(int nTokens, int nHead, int nKv, int headDim, int regionBlock, int slotCount,
                                    int mode, int window, float scale, int kvRows, int kRowStride, int vRowStride,
                                    GpuBuffer q, GpuBuffer kCache, GpuBuffer vCache, GpuBuffer kvPos, GpuBuffer sel,
                                    GpuBuffer qPos, GpuBuffer gate, GpuBuffer out) {
        checkPositive(nHead, "attention head count");
        checkPositive(nKv, "attention KV head count");
        checkPositive(headDim, "attention head width");
        checkPositive(regionBlock, "attention region block");
        checkNonNegative(slotCount, "attention selection slots");
        if (nHead % nKv != 0) {
            throw new IllegalArgumentException(nHead + " query heads do not divide into " + nKv + " KV heads");
        }
        if (mode != ATTN_MODE_SPARSE && mode != ATTN_MODE_WINDOW) {
            throw new IllegalArgumentException("Step5 attention mode " + mode + " is neither sparse nor windowed");
        }
        if (window < 0) {
            throw new IllegalArgumentException("Step5 attention window " + window + " is negative");
        }
        checkPositive(kvRows, "attention cache rows");
        if (kRowStride < 1 || vRowStride < 1 || kRowStride % headDim != 0 || vRowStride % headDim != 0) {
            throw new IllegalArgumentException("Step5 attention row strides " + kRowStride + "/" + vRowStride
                    + " must be positive multiples of the head width " + headDim);
        }
        checkNonNegative(nTokens, "attention token count");
        return new Object[]{nTokens, nHead, nKv, headDim, regionBlock, slotCount, mode, window, scale, kvRows,
                kRowStride, vRowStride, Objects.requireNonNull(q, "q"), Objects.requireNonNull(kCache, "kCache"),
                Objects.requireNonNull(vCache, "vCache"), Objects.requireNonNull(kvPos, "kvPos"),
                Objects.requireNonNull(sel, "sel"), Objects.requireNonNull(qPos, "qPos"),
                Objects.requireNonNull(gate, "gate"), Objects.requireNonNull(out, "out")};
    }

    /** 1-D global size of the element-wise kernels: {@code roundUp(n, local)}. */
    public static long[] elementGlobal(long n, int local) {
        checkLocal(local);
        checkNonNegative(n, "element count");
        return new long[]{(Math.max(1L, n) + local - 1) / local * local};
    }

    /** 1-D global size of {@code LLM_Step5_Summary}: one work-group per token. */
    public static long[] summaryGlobal(int nTokens, int local) {
        checkLocal(local);
        checkNonNegative(nTokens, "summary token count");
        return new long[]{(long) Math.max(1, nTokens) * local};
    }

    /** Local-size array {@code {local}}. */
    public static long[] local(int local) {
        checkLocal(local);
        return new long[]{local};
    }

    private static void checkLocal(int local) {
        if (local < 1) {
            throw new IllegalArgumentException("work-group size " + local + " is not positive");
        }
    }

    private static void checkPositive(int v, String what) {
        if (v < 1) {
            throw new IllegalArgumentException(what + " must be positive, got " + v);
        }
    }

    private static void checkNonNegative(int v, String what) {
        if (v < 0) {
            throw new IllegalArgumentException(what + " must be non-negative, got " + v);
        }
    }

    private static void checkNonNegative(long v, String what) {
        if (v < 0) {
            throw new IllegalArgumentException(what + " must be non-negative, got " + v);
        }
    }
}

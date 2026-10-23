/* sparse/topk_rows.cl - LLM_TopK_Rows per-row top-k selection. Requires: common/dialect.h, common/config.h */
/*
 * LLM_TopK_Rows(i n_rows, i n_cols, i k, R scores, W<int> idx_out)
 *
 * For each row r of scores [n_rows][n_cols], writes to idx_out[r*k .. r*k + k) the column indices of the
 * kk = min(k, n_valid) largest eligible scores, then -1 in the remaining k - kk slots. A score is
 * eligible when it is greater than -INF (NaN is never eligible): the indexer writes -INF for rows its
 * mask hides, so llama.cpp's ggml_top_k + mask (llama-graph.cpp:3402-3478) and this selection keep
 * the same visible cells. The selection is the set ggml_top_k returns (llama.cpp src/models/deepseek32.cpp:314-315),
 * which is order-agnostic for masking; ties at the k-th value go to the lowest column indices.
 * Output order: every column whose score is above the k-th value in increasing column order, then the
 * columns equal to it in increasing column order. Equal floats are equal keys (+0 and -0 included).
 *
 * Method: radix select on the order-preserving unsigned key of each score, three digit passes of
 * 11, 11 and 10 bits over a 2048-bin local histogram, find the k-th largest key T; then one ordered
 * compaction pass with a workgroup prefix scan writes the columns above T and the first (kk - #above)
 * columns equal to T.
 *
 * Contract: k <= NF_TOPK_MAX (the selection width of the sparse MLA kernels; validated on the host).
 *
 * Launch: 1-D, one workgroup per row; global = n_rows*L, local = L, a power of two <= LLM_DIMX
 * (256 recommended).
 */

#define NF_TOPK_MAX  2048
#define NF_TOPK_BINS 2048

/* Order-preserving key: a > b as floats  <=>  key(a) > key(b) as unsigned; -0 maps to the key of +0. */
NF_INLINE uint nf_topk_key(float x) {
    const uint u = (x == 0.0f) ? 0u : as_uint(x);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

NF_KERNEL void LLM_TopK_Rows(
    const int n_rows, const int n_cols, const int k,
    NF_GLOBAL const float* NF_RESTRICT scores,
    NF_GLOBAL int* NF_RESTRICT idx_out) {
    NF_LOCAL uint hist[NF_TOPK_BINS];
    NF_LOCAL uint part[LLM_DIMX];
    NF_LOCAL uint pick[2];
    const int row = get_group_id(0);
    /* Uniform per workgroup: every work-item of a dead group leaves before any barrier. */
    if (row >= n_rows) return;
    const int lid = get_local_id(0), lsize = get_local_size(0);
    const nf_i64 sbase = (nf_i64)row * n_cols;
    const int kc = max(k, 0);
    const nf_i64 obase = (nf_i64)row * kc;
    const int per = NF_TOPK_BINS / lsize;

    uint prefix = 0u, pmask = 0u, kk = 0u, remaining = 0u;
    for (int pass = 0; pass < 3; ++pass) {
        const int shift = pass == 0 ? 21 : (pass == 1 ? 10 : 0);
        const uint dmask = pass == 2 ? 0x3FFu : 0x7FFu;
        for (int b = lid; b < NF_TOPK_BINS; b += lsize) hist[b] = 0u;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int c = lid; c < n_cols; c += lsize) {
            const float x = scores[sbase + c];
            if (x > -NF_INF) {
                const uint key = nf_topk_key(x);
                if ((key & pmask) == prefix) atomic_inc(&hist[(key >> shift) & dmask]);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        uint mine = 0u;
        for (int i = 0; i < per; ++i) mine += hist[lid * per + i];
        part[lid] = mine;
        barrier(CLK_LOCAL_MEM_FENCE);
        /* Inclusive suffix sums: part[i] = candidates in the bins of work-items i .. L-1. */
        for (int off = 1; off < lsize; off <<= 1) {
            const uint v = part[lid] + (lid + off < lsize ? part[lid + off] : 0u);
            barrier(CLK_LOCAL_MEM_FENCE);
            part[lid] = v;
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (pass == 0) {
            kk = min((uint)kc, part[0]);
            remaining = kk;
        }
        /* kk is uniform: nothing eligible or k == 0 leaves every slot -1. */
        if (kk == 0u) break;
        /* The k-th largest key lies in the digit d with #(digits > d) < remaining <= #(digits >= d). */
        uint above = lid + 1 < lsize ? part[lid + 1] : 0u;
        if (above < remaining && remaining <= above + mine) {
            for (int i = per - 1; i >= 0; --i) {
                const uint h = hist[lid * per + i];
                if (above + h >= remaining) {
                    pick[0] = (uint)(lid * per + i);
                    pick[1] = remaining - above;
                    break;
                }
                above += h;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        prefix |= pick[0] << shift;
        pmask |= dmask << shift;
        remaining = pick[1];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (kk > 0u) {
        /* prefix is the k-th largest key T; remaining = how many columns equal to T are selected. */
        const uint n_above = kk - remaining;
        uint above_written = 0u, equal_seen = 0u;
        for (int c0 = 0; c0 < n_cols; c0 += lsize) {
            const int c = c0 + lid;
            uint flag = 0u;
            if (c < n_cols) {
                const float x = scores[sbase + c];
                if (x > -NF_INF) {
                    const uint key = nf_topk_key(x);
                    flag = key > prefix ? 1u : (key == prefix ? 0x10000u : 0u);
                }
            }
            part[lid] = flag;
            barrier(CLK_LOCAL_MEM_FENCE);
            /* Inclusive prefix sums of the packed (above | equal << 16) flags. */
            for (int off = 1; off < lsize; off <<= 1) {
                const uint v = part[lid] + (lid >= off ? part[lid - off] : 0u);
                barrier(CLK_LOCAL_MEM_FENCE);
                part[lid] = v;
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            const uint excl = part[lid] - flag;
            const uint tot = part[lsize - 1];
            if (flag == 1u) {
                idx_out[obase + above_written + (excl & 0xFFFFu)] = c;
            } else if (flag == 0x10000u) {
                const uint e = equal_seen + (excl >> 16);
                if (e < remaining) idx_out[obase + n_above + e] = c;
            }
            above_written += tot & 0xFFFFu;
            equal_seen += tot >> 16;
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
    for (int i = (int)kk + lid; i < kc; i += lsize) idx_out[obase + i] = -1;
}

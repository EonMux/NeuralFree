/* sparse/topk_rows.cl - LLM_TopK_Rows per-row top-k selection. Requires: common/dialect.h, common/config.h,
 * common/topk_select.h */
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
 * Method: nf_topk_select (common/topk_select.h) finds the k-th largest key T by a radix select of three digit
 * passes; nf_topk_compact writes the columns above T and the first (kk - #above) columns equal to T in column order,
 * each work-item compacting one contiguous column range after one scan over the work-items' counts.
 *
 * Contract: k <= NF_TOPK_MAX (the selection width of the sparse MLA kernels; validated on the host).
 *
 * Launch: 1-D, one workgroup per row; global = n_rows*L, local = L, a power of two <= LLM_DIMX
 * (256 recommended).
 */

#define NF_TOPK_MAX  2048

NF_KERNEL void LLM_TopK_Rows(
    const int n_rows, const int n_cols, const int k,
    NF_GLOBAL const float* NF_RESTRICT scores,
    NF_GLOBAL int* NF_RESTRICT idx_out) {
    NF_LOCAL uint hist[NF_TOPK_BINS];
    NF_LOCAL uint part_a[LLM_DIMX];
    NF_LOCAL uint part_e[LLM_DIMX];
    NF_LOCAL uint pick[2];
    NF_LOCAL int sel[NF_TOPK_MAX];
    const int row = get_group_id(0);
    /* Uniform per workgroup: every work-item of a dead group leaves before any barrier. */
    if (row >= n_rows) return;
    const int lid = get_local_id(0), lsize = get_local_size(0);
    NF_GLOBAL const float* srow = scores + (nf_i64)row * n_cols;
    const int kc = min(max(k, 0), NF_TOPK_MAX);
    const nf_i64 obase = (nf_i64)row * max(k, 0);

    uint key_k, kk, n_eq;
    nf_topk_select(srow, n_cols, kc, 0, hist, part_a, pick, &key_k, &kk, &n_eq);
    if (kk > 0u) {
        nf_topk_compact(srow, n_cols, 0, key_k, kk - n_eq, n_eq, part_a, part_e, sel);
    }
    for (int i = lid; i < max(k, 0); i += lsize) {
        idx_out[obase + i] = i < (int)kk ? sel[i] : -1;
    }
}

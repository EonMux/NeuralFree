/* cp_merge.cl - LLM_AttnCpMerge context-parallel attention merge.
 * Requires: common/dialect.h, common/config.h, common/attn_common.h */
/*
 * Merges the normalized attention outputs of n_rank context-parallel ranks into the one softmax over the
 * concatenated key ranges (vLLM cp_lse_ag_out_ar, vllm/v1/attention/ops/dcp.py). Each rank attended over its
 * context shard and wrote out_r = o_r / l_r, plus the merged log-sum-exp lse_r = m_r + log(l_r) of
 * LLM_FlashAttn_Lse; the combined result is
 *   M      = max_r lse_r
 *   w_r    = e^(lse_r - M)                       (0 for a rank that saw no key, lse_r = -INF)
 *   out[d] = sum_r w_r * out_r[d] / sum_r w_r,   and 0 when every rank saw no key.
 * Because e^lse_r = l_r * e^m_r, this is exactly the one softmax over every rank's keys.
 *
 * Layout: outs[r * n_rows * d_v + row * d_v + d], lses[r * n_rows + row], out[row * d_v + d]; rows are
 * (token, head) pairs in the order LLM_FlashAttn_Combine writes them.
 * Grid: global {n_rows * 128}, local {128} (one workgroup per row; any local size is valid, no barriers).
 */
NF_KERNEL void LLM_AttnCpMerge(
    const int n_rows,
    const int d_v,
    const int n_rank,
    NF_GLOBAL const float* NF_RESTRICT outs,
    NF_GLOBAL const float* NF_RESTRICT lses,
    NF_GLOBAL float* NF_RESTRICT out)
{
    const int row = (int)get_group_id(0);
    if (row >= n_rows || n_rank <= 0 || d_v <= 0) {
        return;
    }
    const int lid = (int)get_local_id(0);
    const int lsz = (int)get_local_size(0);
    NF_GLOBAL float* dst = out + (nf_i64)row * d_v;

    float m = -NF_INF;
    for (int r = 0; r < n_rank; r++) {
        m = fmax(m, lses[(nf_i64)r * n_rows + row]);
    }
    if (m == -NF_INF) {
        for (int d = lid; d < d_v; d += lsz) {
            dst[d] = 0.0f;
        }
        return;
    }
    float denom = 0.0f;
    for (int r = 0; r < n_rank; r++) {
        denom += nf_attn_rescale(lses[(nf_i64)r * n_rows + row], m);
    }
    for (int d = lid; d < d_v; d += lsz) {
        float o = 0.0f;
        for (int r = 0; r < n_rank; r++) {
            const float w = nf_attn_rescale(lses[(nf_i64)r * n_rows + row], m);
            o += w * outs[((nf_i64)r * n_rows + row) * d_v + d];
        }
        dst[d] = denom > 0.0f ? o / denom : 0.0f;
    }
}

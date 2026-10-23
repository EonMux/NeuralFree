/* flash_combine.cl - LLM_FlashAttn_Combine split-K merge. Requires: common/dialect.h, common/config.h, common/reduce.h, common/attn_common.h */
/*
 * Merges the n_split partial results (o[d_v], m, l) of LLM_FlashDecode_* / LLM_FlashAttn_* called with
 * direct_out = 0, as ggml-cuda/fattn-common.cuh:924 flash_attn_combine_results, plus the head's sink
 * logit, which the split kernels leave out (ggml applies sinks once, ggml-cpu/ops.cpp:9130-9145):
 *   m = max(max_s m_s, sink)
 *   l = sum_s l_s * e^(m_s - m) + e^(sink - m)
 *   out[d] = (sum_s o_s[d] * e^(m_s - m)) / l, and 0 when l == 0 (every key masked, no sink).
 * Splits whose keys were all masked carry m_s = -INF and contribute nothing.
 * GLSL twin: shaders/llm/flash_combine.comp. Java reference: gguf.ops.attn.CpuAttentionCore.
 *
 * Layout: partial[(r * n_split + s) * (d_v + 2) + d], m at + d_v, l at + d_v + 1; row r has head
 * r % n_head and writes out[out_off + r * d_v + d].
 * Grid: global {n_rows * 128}, local {128} (one workgroup per row; any local size is valid, no barriers).
 */
NF_KERNEL void LLM_FlashAttn_Combine(
    const int n_rows,
    const int d_v,
    const int n_split,
    const int n_head,
    const int has_sinks,
    const int out_off,
    NF_GLOBAL const float* NF_RESTRICT partial,
    NF_GLOBAL const float* NF_RESTRICT sinks,
    NF_GLOBAL float* NF_RESTRICT out)
{
    const int row = (int)get_group_id(0);
    if (row >= n_rows) {
        return;
    }
    const int lid = (int)get_local_id(0);
    const int lsz = (int)get_local_size(0);
    const int stride = d_v + 2;
    NF_GLOBAL const float* pr = partial + (nf_i64)row * n_split * stride;
    NF_GLOBAL float* dst = out + (nf_i64)out_off + (nf_i64)row * d_v;

    const float sink = has_sinks ? sinks[row % n_head] : -NF_INF;
    float m = sink;
    for (int s = 0; s < n_split; s++) {
        m = fmax(m, pr[(nf_i64)s * stride + d_v]);
    }
    if (m == -NF_INF) {
        for (int d = lid; d < d_v; d += lsz) {
            dst[d] = 0.0f;
        }
        return;
    }
    float l = nf_attn_rescale(sink, m);
    for (int s = 0; s < n_split; s++) {
        NF_GLOBAL const float* ps = pr + (nf_i64)s * stride;
        l += ps[d_v + 1] * nf_attn_rescale(ps[d_v], m);
    }
    for (int d = lid; d < d_v; d += lsz) {
        float o = 0.0f;
        for (int s = 0; s < n_split; s++) {
            NF_GLOBAL const float* ps = pr + (nf_i64)s * stride;
            o += ps[d] * nf_attn_rescale(ps[d_v], m);
        }
        dst[d] = l > 0.0f ? o / l : 0.0f;
    }
}

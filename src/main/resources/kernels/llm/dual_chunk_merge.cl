/* dual_chunk_merge.cl - LLM_DualChunkMerge. Requires: common/dialect.h, common/config.h, common/attn_common.h */
/*
 * Merges the three (o, m, l) partial results of a token's dual-chunk stages (Qwen2.5-1M, vLLM
 * _dual_chunk_flash_attn_prefill_func: intra, succ and inter are scored separately and share one softmax over the
 * concatenated logits):
 *   m = max_s m_s
 *   l = sum_s l_s * e^(m_s - m)
 *   out[d] = (sum_s o_s[d] * e^(m_s - m)) / l, and 0 when l == 0 (no visible key in any stage).
 * A stage with no visible key carries m_s = -INF and contributes nothing, so an absent stage can be launched over
 * an empty key range. The stage kernels are LLM_FlashAttn_* / LLM_FlashAttnPaged_* with direct_out = 0, n_split = 1
 * and one partial buffer per stage.
 * GLSL twin: shaders/llm/dual_chunk_merge.comp. Java reference: gguf.ops.attn.CpuDualChunkAttention.
 *
 * Layout: stage buffer s holds partial[(r * n_head + h) * (d_v + 2) + d], m at + d_v, l at + d_v + 1, where r is the
 * query row within the launch; the three buffers hold the same (row, head) rows. Row r of head h writes
 * out[out_off + r * out_row_stride + h * d_v + d].
 * Grid: global {n_rows * n_head * 128}, local {128} (one workgroup per (row, head); any local size is valid, no
 * barriers).
 */
NF_KERNEL void LLM_DualChunkMerge(
    const int n_rows,
    const int n_head,
    const int d_v,
    const int out_row_stride,
    const int out_off,
    NF_GLOBAL const float* NF_RESTRICT p0,
    NF_GLOBAL const float* NF_RESTRICT p1,
    NF_GLOBAL const float* NF_RESTRICT p2,
    NF_GLOBAL float* NF_RESTRICT out)
{
    const int row = (int)get_group_id(0);
    if (n_rows <= 0 || n_head <= 0 || d_v <= 0 || row >= n_rows * n_head) {
        return;
    }
    const int lid = (int)get_local_id(0);
    const int lsz = (int)get_local_size(0);
    const int t = row / n_head;
    const int h = row - t * n_head;
    const nf_i64 at = (nf_i64)row * (d_v + 2);

    const float m = fmax(fmax(p0[at + d_v], p1[at + d_v]), p2[at + d_v]);
    NF_GLOBAL float* dst = out + (nf_i64)out_off + (nf_i64)t * out_row_stride + (nf_i64)h * d_v;
    if (m == -NF_INF) {
        for (int d = lid; d < d_v; d += lsz) {
            dst[d] = 0.0f;
        }
        return;
    }
    const float w0 = nf_attn_rescale(p0[at + d_v], m);
    const float w1 = nf_attn_rescale(p1[at + d_v], m);
    const float w2 = nf_attn_rescale(p2[at + d_v], m);
    const float l = p0[at + d_v + 1] * w0 + p1[at + d_v + 1] * w1 + p2[at + d_v + 1] * w2;
    for (int d = lid; d < d_v; d += lsz) {
        const float o = p0[at + d] * w0 + p1[at + d] * w1 + p2[at + d] * w2;
        dst[d] = l > 0.0f ? o / l : 0.0f;
    }
}

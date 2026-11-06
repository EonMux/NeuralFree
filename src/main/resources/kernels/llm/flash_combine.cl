/* flash_combine.cl - LLM_FlashAttn_Combine split-K merge, LLM_FlashAttn_Lse merged log-sum-exp.
 * Requires: common/dialect.h, common/config.h, common/reduce.h, common/attn_common.h */
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

/*
 * LLM_FlashAttn_Lse - the merged log-sum-exp of the split-K partials the way
 * LLM_FlashAttn_Combine merges them, written once per row:
 *   m  = max(max_s m_s, sink)
 *   l  = sum_s l_s * e^(m_s - m) + e^(sink - m)
 *   lse = m + log(l), or -INF when l == 0 (every key masked, no sink).
 * The log is base e, so e^lse is the softmax denominator of the row. Context-parallel attention combines the
 * per-rank outputs with these values (LLM_AttnCpMerge); a row that saw no key contributes nothing.
 *
 * Layout as LLM_FlashAttn_Combine: partial[(r * n_split + s) * (d_v + 2) + d], m at + d_v, l at + d_v + 1; row r
 * has head r % n_head and writes lse[out_off + r]. Grid: global {n_rows * 128}, local {128} (one workgroup per
 * row; only local id 0 writes, no barriers).
 */
NF_KERNEL void LLM_FlashAttn_Lse(
    const int n_rows,
    const int d_v,
    const int n_split,
    const int n_head,
    const int has_sinks,
    const int out_off,
    NF_GLOBAL const float* NF_RESTRICT partial,
    NF_GLOBAL const float* NF_RESTRICT sinks,
    NF_GLOBAL float* NF_RESTRICT lse)
{
    const int row = (int)get_group_id(0);
    if (row >= n_rows || (int)get_local_id(0) != 0) {
        return;
    }
    const int stride = d_v + 2;
    NF_GLOBAL const float* pr = partial + (nf_i64)row * n_split * stride;
    const float sink = has_sinks ? sinks[row % n_head] : -NF_INF;
    float m = sink;
    for (int s = 0; s < n_split; s++) {
        m = fmax(m, pr[(nf_i64)s * stride + d_v]);
    }
    if (m == -NF_INF) {
        lse[out_off + row] = -NF_INF;
        return;
    }
    float l = nf_attn_rescale(sink, m);
    for (int s = 0; s < n_split; s++) {
        NF_GLOBAL const float* ps = pr + (nf_i64)s * stride;
        l += ps[d_v + 1] * nf_attn_rescale(ps[d_v], m);
    }
    lse[out_off + row] = l > 0.0f ? m + log(l) : -NF_INF;
}

/*
 * LLM_AttnStaticSinkPartial - the split-K partial of the learned static sink tokens of vLLM's
 * StaticSinkAttention (openPangu): one row of the (o, m, l) layout of LLM_FlashDecode_* / LLM_FlashAttn_* for
 * every (token, head), written at split index `split_index` so the unchanged LLM_FlashAttn_Combine merges it with
 * the cached-key splits.
 *
 * The sink tokens precede every cached key: vLLM adds sink_len to seq_lens and to the block table, so key index r
 * of the sinks maps to the query's absolute position plus n_sink under the causal and sliding-window mask. The
 * sink keys are already normalized by the layer's K norm and carry no RoPE, and they take no position bias. A
 * split whose sinks are all masked carries m = -INF and contributes nothing.
 *
 * Row r = t * n_head + h; token t reads q[q_off + t * q_row_stride + h * d_k], its position is
 * q_pos[pos_off + t * pos_stride] (pos_stride == 0: fixed_pos for every row); the sinks are dense F32 of
 * [n_sink][n_head_kv][d_k] / [n_sink][n_head_kv][d_v]. The partial row is
 * partial[(row * n_split + split_index) * (d_v + 2)].
 * Grid: global {roundUp(n_rows, WG)}, local {WG} (one work-item per row, no barriers).
 */
NF_KERNEL void LLM_AttnStaticSinkPartial(
    const int n_rows,
    const int n_head,
    const int n_head_kv,
    const int d_k,
    const int d_v,
    const int n_sink,
    const int q_off,
    const int q_row_stride,
    const int pos_off,
    const int pos_stride,
    const int fixed_pos,
    const int n_split,
    const int split_index,
    const float scale,
    const float softcap,
    const int causal,
    const int swa_type,
    const int n_swa,
    NF_GLOBAL const float* NF_RESTRICT q,
    NF_GLOBAL const int* NF_RESTRICT q_pos,
    NF_GLOBAL const float* NF_RESTRICT sink_k,
    NF_GLOBAL const float* NF_RESTRICT sink_v,
    NF_GLOBAL float* partial)
{
    const int row = (int)get_global_id(0);
    if (row >= n_rows || n_head <= 0 || n_head_kv <= 0 || n_head % n_head_kv != 0 || d_k <= 0 || d_v <= 0
        || n_sink <= 0 || n_split <= 0) {
        return;
    }
    const int t = row / n_head;
    const int h = row - t * n_head;
    const int gq = n_head / n_head_kv;
    const int kvh = h / gq;
    const int p_q = pos_stride == 0 ? fixed_pos : q_pos[pos_off + t * pos_stride];
    /* vLLM aligns the query against the right of the key sequence that carries the sink tokens. */
    const int p_k = p_q + n_sink;
    const nf_i64 k_step = (nf_i64)n_head_kv * d_k;
    const nf_i64 v_step = (nf_i64)n_head_kv * d_v;
    NF_GLOBAL const float* qr = q + (nf_i64)q_off + (nf_i64)t * q_row_stride + (nf_i64)h * d_k;

    float m = -NF_INF;
    for (int r = 0; r < n_sink; r++) {
        const int vis = causal == 0 ? 1 : nf_attn_visible(p_k, r, 1, swa_type, n_swa);
        if (vis) {
            NF_GLOBAL const float* kr = sink_k + (nf_i64)r * k_step + (nf_i64)kvh * d_k;
            float dot = 0.0f;
            for (int d = 0; d < d_k; d++) {
                dot += qr[d] * kr[d];
            }
            m = fmax(m, nf_attn_score(dot, scale, softcap));
        }
    }

    NF_GLOBAL float* pr = partial + (((nf_i64)row * n_split + split_index) * (d_v + 2));
    for (int d = 0; d < d_v; d++) {
        pr[d] = 0.0f;
    }
    float l = 0.0f;
    if (m != -NF_INF) {
        for (int r = 0; r < n_sink; r++) {
            const int vis = causal == 0 ? 1 : nf_attn_visible(p_k, r, 1, swa_type, n_swa);
            if (!vis) {
                continue;
            }
            NF_GLOBAL const float* kr = sink_k + (nf_i64)r * k_step + (nf_i64)kvh * d_k;
            float dot = 0.0f;
            for (int d = 0; d < d_k; d++) {
                dot += qr[d] * kr[d];
            }
            const float p = nf_attn_rescale(nf_attn_score(dot, scale, softcap), m);
            l += p;
            NF_GLOBAL const float* vr = sink_v + (nf_i64)r * v_step + (nf_i64)kvh * d_v;
            for (int d = 0; d < d_v; d++) {
                pr[d] += p * vr[d];
            }
        }
    }
    pr[d_v] = m;
    pr[d_v + 1] = l;
}

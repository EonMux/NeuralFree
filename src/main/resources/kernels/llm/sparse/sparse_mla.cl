/* sparse/sparse_mla.cl - LLM_MLA_SparseDecode_<KV> and LLM_MLA_SparsePrefill_<KV> sparse MLA attention. Requires: common/dialect.h, common/config.h, common/reduce.h, common/kv_types.h, common/attn_common.h */
/*
 * Absorbed multi-head latent attention restricted to the cells chosen by the DSA indexer
 * (llama-graph.cpp:3402-3478: the kq mask is -INF except at the top-k cells, then
 * build_attn_mha with the latent cache as K and its first kv_lora values as V, MQA over all heads).
 * For one query row q (d_k = kv_lora + rope_dim values, [absorbed nope | rope] as the latent cache row
 * [kv_lora | rope], llama.cpp src/models/deepseek32.cpp:368-388) and the selected rows r_j = sel[j], j < n_sel:
 *     s_j  = softcap > 0 ? softcap * tanh(q.k_j * (kq_scale / softcap)) : q.k_j * kq_scale
 *            (llama.cpp ggml-cpu/ops.cpp:8997-9071 flash_attn_ext: scale /= softcap; s = softcap*tanh(s*scale))
 *     m    = max(max_j s_j, sink)                 sink = sinks[h] when has_sinks, else absent
 *     l    = sum_j exp(s_j - m) + exp(sink - m)
 *     out[d] = (1/l) * sum_j exp(s_j - m) * v_j[d],  v_j = k_j[0 .. kv_lora),  d < kv_lora
 * Entries with r_j < 0 (LLM_TopK_Rows padding) are skipped; a row with nothing selected and no sink
 * writes zeros (l == 0). The latent cache holds rows of d_k values, row r at value r*d_k, of the kernel's
 * KV element type (the layer's KvDType, read through the common/kv_types.h loaders).
 *
 * Preconditions (validated on the host by SparseKernels): 1 <= n_sel <= NF_SMLA_MAX_SEL, every
 * non-negative sel entry is a valid cache row, kv_lora >= 1, rope_dim >= 0; for a block type, d_k is a
 * multiple of 32 (whole ggml blocks per row, KvDType.supportsWidth).
 *
 * Launch: 1-D, local = L (power of two, 32 <= L <= LLM_DIMX, multiple of 32 on CUDA; 128 recommended).
 *   LLM_MLA_SparseDecode_<KV> : one workgroup per head h; global = n_head*L.
 *   LLM_MLA_SparsePrefill_<KV>: one workgroup per (token t, head h), group g = t*n_head + h;
 *                               global = n_tokens*n_head*L.
 */

#define NF_SMLA_MAX_SEL 2048

/*
 * One (token, head) attention row. Every work-item calls it in uniform control flow (it contains
 * barriers). q: d_k query values; sel: n_sel cache rows; out: kv_lora outputs. sc holds n_sel
 * floats, red is the reduction buffer of common/reduce.h.
 */
NF_INLINE void nf_smla_row(
    const int kv_lora, const int d_k, const int n_sel, const float kq_scale, const float softcap,
    const int has_sinks, const float sink,
    NF_GLOBAL const float* q, NF_GLOBAL const uchar* kv_cache, const int kv_type, NF_GLOBAL const int* sel,
    NF_GLOBAL float* out, NF_LOCAL_PTR float* sc, NF_LOCAL_PTR float* red) {
    const int lid = get_local_id(0), lsize = get_local_size(0);

    /* Logits: one selected cell per work-item and step. */
    float m_loc = -NF_INF;
    for (int j = lid; j < n_sel; j += lsize) {
        const int r = sel[j];
        float s = -NF_INF;
        if (r >= 0) {
            const nf_i64 krow = (nf_i64)r * d_k;
            float acc = 0.0f;
            int d = 0;
            for (; d + 4 <= d_k; d += 4) acc += dot(vload4(0, q + d), nf_kv_load4(kv_cache, krow + d, kv_type));
            for (; d < d_k; ++d) acc += q[d] * nf_kv_load1(kv_cache, krow + d, kv_type);
            s = nf_attn_score(acc, kq_scale, softcap);
        }
        sc[j] = s;
        m_loc = fmax(m_loc, s);
    }
    float m = nf_wg_max(red, m_loc);
    if (has_sinks) m = fmax(m, sink);

    /* Probabilities (unnormalized) replace the logits in sc. */
    float l_loc = 0.0f;
    for (int j = lid; j < n_sel; j += lsize) {
        const float s = sc[j];
        const float p = s == -NF_INF ? 0.0f : exp(s - m);
        sc[j] = p;
        l_loc += p;
    }
    float l = nf_wg_sum(red, l_loc);
    if (has_sinks && m > -NF_INF) l += exp(sink - m);
    barrier(CLK_LOCAL_MEM_FENCE);

    /* Values: one latent dim per work-item and step, consecutive dims read consecutive values. */
    const float inv_l = l > 0.0f ? 1.0f / l : 0.0f;
    for (int d = lid; d < kv_lora; d += lsize) {
        float acc = 0.0f;
        for (int j = 0; j < n_sel; ++j) {
            const float p = sc[j];
            if (p != 0.0f) acc += p * nf_kv_load1(kv_cache, (nf_i64)sel[j] * d_k + d, kv_type);
        }
        out[d] = acc * inv_l;
    }
}

/*
 * LLM_MLA_SparseDecode_<KV>(8 scalars, 5 buffers) and LLM_MLA_SparsePrefill_<KV>(9 scalars, 5 buffers);
 * KV is a cache element type of common/kv_types.h. sinks is a read-only placeholder when has_sinks == 0.
 */
#define NF_SMLA_KERNELS(SFX, KV_T, KV_CODE)                                                  \
NF_KERNEL void LLM_MLA_SparseDecode_##SFX(                                                   \
    const int n_head, const int kv_lora, const int rope_dim, const int n_sel,                \
    const float kq_scale, const float softcap, const int has_sinks, const int q_off,         \
    NF_GLOBAL const float* NF_RESTRICT q_abs,                                                \
    NF_GLOBAL const KV_T* NF_RESTRICT kv_cache,                                              \
    NF_GLOBAL const int* NF_RESTRICT sel_idx,                                                \
    NF_GLOBAL const float* NF_RESTRICT sinks,                                                \
    NF_GLOBAL float* NF_RESTRICT out_latent) {                                               \
    NF_LOCAL float sc[NF_SMLA_MAX_SEL];                                                      \
    NF_LOCAL float red[LLM_DIMX];                                                            \
    const int h = get_group_id(0);                                                           \
    /* Uniform per workgroup: every work-item of a dead group leaves before any barrier. */ \
    if (h >= n_head) return;                                                                 \
    const int d_k = kv_lora + rope_dim;                                                      \
    const float sink = has_sinks ? sinks[h] : 0.0f;                                          \
    nf_smla_row(kv_lora, d_k, min(n_sel, NF_SMLA_MAX_SEL), kq_scale, softcap, has_sinks, sink, \
                q_abs + (nf_i64)q_off + (nf_i64)h * d_k, (NF_GLOBAL const uchar*)kv_cache,    \
                KV_CODE, sel_idx, out_latent + (nf_i64)h * kv_lora, sc, red);                \
}                                                                                            \
                                                                                             \
NF_KERNEL void LLM_MLA_SparsePrefill_##SFX(                                                  \
    const int n_tokens, const int n_head, const int kv_lora, const int rope_dim, const int n_sel, \
    const float kq_scale, const float softcap, const int has_sinks, const int q_stride,      \
    NF_GLOBAL const float* NF_RESTRICT q_abs,                                                \
    NF_GLOBAL const KV_T* NF_RESTRICT kv_cache,                                              \
    NF_GLOBAL const int* NF_RESTRICT sel_idx,                                                \
    NF_GLOBAL const float* NF_RESTRICT sinks,                                                \
    NF_GLOBAL float* NF_RESTRICT out_latent) {                                               \
    NF_LOCAL float sc[NF_SMLA_MAX_SEL];                                                      \
    NF_LOCAL float red[LLM_DIMX];                                                            \
    const int g = get_group_id(0);                                                           \
    const int t = g / n_head;                                                                \
    /* Uniform per workgroup: every work-item of a dead group leaves before any barrier. */ \
    if (t >= n_tokens) return;                                                               \
    const int h = g - t * n_head;                                                            \
    const int d_k = kv_lora + rope_dim;                                                      \
    const float sink = has_sinks ? sinks[h] : 0.0f;                                          \
    nf_smla_row(kv_lora, d_k, min(n_sel, NF_SMLA_MAX_SEL), kq_scale, softcap, has_sinks, sink, \
                q_abs + (nf_i64)t * q_stride + (nf_i64)h * d_k, (NF_GLOBAL const uchar*)kv_cache, \
                KV_CODE, sel_idx + (nf_i64)t * n_sel,                                        \
                out_latent + ((nf_i64)t * n_head + h) * kv_lora, sc, red);                   \
}

NF_SMLA_KERNELS(F32, float, NF_KV_F32)
NF_SMLA_KERNELS(F16, NF_HALF, NF_KV_F16)
NF_SMLA_KERNELS(Q8_0, uchar, NF_KV_Q8_0)
NF_SMLA_KERNELS(BF16, ushort, NF_KV_BF16)
NF_SMLA_KERNELS(Q4_0, uchar, NF_KV_Q4_0)
NF_SMLA_KERNELS(Q4_1, uchar, NF_KV_Q4_1)
NF_SMLA_KERNELS(IQ4_NL, uchar, NF_KV_IQ4_NL)
NF_SMLA_KERNELS(Q5_0, uchar, NF_KV_Q5_0)
NF_SMLA_KERNELS(Q5_1, uchar, NF_KV_Q5_1)

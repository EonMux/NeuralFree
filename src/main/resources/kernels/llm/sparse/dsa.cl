/* sparse/dsa.cl - LLM_DSA_IndexerScore_<KV> DeepSeek sparse-attention lightning indexer. Requires: common/dialect.h, common/config.h, common/kv_types.h, common/attn_common.h */
/*
 * LLM_DSA_IndexerScore_<KV>(i n_tokens, i n_kv, i n_idx_heads, i idx_dim, i causal, i swa_type,
 *                      i n_swa, R iq, R w, R<KV> ik_cache, R<int> q_pos, R<int> kv_pos, W scores)
 *
 * Lightning-indexer score of query token t against indexer-cache row s (llama.cpp ggml-cpu/ops.cpp:12531-12614,
 * ggml_compute_forward_lightning_indexer; graph llama.cpp src/models/deepseek32.cpp:220-316):
 *     scores[t*n_kv + s] = sum_h max(iq[t][h] . ik[s], 0) * w[t][h]   when row s is visible to token t
 *                        = -INF                                       otherwise (the -INF entries of the
 *                                                                      indexer kq mask)
 * Layouts: iq [n_tokens][n_idx_heads][idx_dim] (rotated, roped indexer queries), w [n_tokens][n_idx_heads]
 * (head weights, already prescaled by (idx_dim*n_idx_heads)^-0.5 as llama.cpp deepseek32.cpp:269 does),
 * ik_cache rows of idx_dim values in the kernel's KV element type (the indexer cache follows the context's
 * type_k, llama-kv-cache-dsa.cpp:51 and llama-kv-cache-dsv4.cpp:1308-1312; read through the
 * common/kv_types.h loaders), q_pos [n_tokens] and kv_pos [rows] (position held by each row, -1 when
 * empty). Visibility is the llama.cpp KV mask (llama-kv-cache.cpp:1664-1710): the row is not empty,
 * kv <= q when causal, and llama_hparams::is_masked_swa (llama-hparams.h:506-540) is false for
 * swa_type (0 NONE, 1 STANDARD, 2 CHUNKED, 3 SYMMETRIC) and window n_swa.
 *
 * Preconditions (validated on the host by SparseKernels): 1 <= idx_dim <= NF_DSA_QSTAGE, n_idx_heads >= 1;
 * for a block type, idx_dim is a multiple of 32 (whole ggml blocks per row, KvDType.supportsWidth).
 *
 * Launch: 1-D, one workgroup per (token, block of L rows), group g = t*ceil(n_kv/L) + b, row
 * s = b*L + lid; global = n_tokens*ceil(n_kv/L)*L, local = L (any size <= LLM_DIMX). The token's queries
 * and weights are staged in local memory in head chunks; each work-item scores one row with four
 * head accumulators per pass over its key row.
 */

#define NF_DSA_QSTAGE 2048   /* floats of staged indexer queries per head chunk */
#define NF_DSA_WSTAGE 256    /* head weights staged per head chunk */

/* llama.cpp KV-mask visibility of a row holding position p_kv for a query at position p_q. */
NF_INLINE int nf_dsa_visible(int causal, int swa_type, int n_swa, int p_kv, int p_q) {
    if (p_kv < 0) return 0;
    if (causal && p_kv > p_q) return 0;
    if (swa_type == 1) {
        if (p_q - p_kv >= n_swa) return 0;
    } else if (swa_type == 2) {
        if (p_kv < (p_q / n_swa) * n_swa) return 0;
    } else if (swa_type == 3) {
        const int half_n_swa = n_swa / 2;
        const int diff = p_q - p_kv;
        if (diff < -half_n_swa || diff > half_n_swa) return 0;
    }
    return 1;
}

/*
 * Score of token t against row s for every work-item of the group (see the file comment). ik_cache is the
 * cache of element type kv_type (NF_KV_*); qs and ws are the local query and weight stages. Every work-item
 * calls it in uniform control flow (it contains barriers).
 */
NF_INLINE void nf_dsa_score(
    const int n_tokens, const int n_kv, const int n_idx_heads, const int idx_dim,
    const int causal, const int swa_type, const int n_swa,
    NF_GLOBAL const float* iq, NF_GLOBAL const float* w, NF_GLOBAL const uchar* ik_cache, const int kv_type,
    NF_GLOBAL const int* q_pos, NF_GLOBAL const int* kv_pos, NF_GLOBAL float* scores,
    NF_LOCAL_PTR float* qs, NF_LOCAL_PTR float* ws) {
    const int lid = get_local_id(0), lsize = get_local_size(0);
    const int nblk = (n_kv + lsize - 1) / lsize;
    const int g = get_group_id(0);
    const int t = g / nblk;
    /* Uniform per workgroup: every work-item of a dead group (or of a launch that violates the head-size
     * precondition) leaves before any barrier. */
    if (t >= n_tokens || idx_dim < 1 || idx_dim > NF_DSA_QSTAGE) return;
    const int s = (g - t * nblk) * lsize + lid;

    int vis = 0;
    if (s < n_kv) {
        vis = nf_dsa_visible(causal, swa_type, n_swa, kv_pos[s], q_pos[t]);
    }
    const nf_i64 krow = (nf_i64)(vis ? s : 0) * idx_dim;
    const nf_i64 qbase = (nf_i64)t * n_idx_heads * idx_dim;
    const nf_i64 wbase = (nf_i64)t * n_idx_heads;
    const int hs = min(NF_DSA_WSTAGE, NF_DSA_QSTAGE / idx_dim);

    float score = 0.0f;
    for (int h0 = 0; h0 < n_idx_heads; h0 += hs) {
        const int nh = min(hs, n_idx_heads - h0);
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int i = lid; i < nh * idx_dim; i += lsize) qs[i] = iq[qbase + (nf_i64)h0 * idx_dim + i];
        for (int i = lid; i < nh; i += lsize) ws[i] = w[wbase + h0 + i];
        barrier(CLK_LOCAL_MEM_FENCE);
        if (vis) {
            for (int hg = 0; hg < nh; hg += 4) {
                /* Heads past nh repeat the last head; their sums are computed and dropped. */
                NF_LOCAL_PTR const float* q0 = qs + hg * idx_dim;
                NF_LOCAL_PTR const float* q1 = qs + min(hg + 1, nh - 1) * idx_dim;
                NF_LOCAL_PTR const float* q2 = qs + min(hg + 2, nh - 1) * idx_dim;
                NF_LOCAL_PTR const float* q3 = qs + min(hg + 3, nh - 1) * idx_dim;
                float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
                int d = 0;
                for (; d + 4 <= idx_dim; d += 4) {
                    const float4 k4 = nf_kv_load4(ik_cache, krow + d, kv_type);
                    a0 += dot(vload4(0, q0 + d), k4);
                    a1 += dot(vload4(0, q1 + d), k4);
                    a2 += dot(vload4(0, q2 + d), k4);
                    a3 += dot(vload4(0, q3 + d), k4);
                }
                for (; d < idx_dim; ++d) {
                    const float kd = nf_kv_load1(ik_cache, krow + d, kv_type);
                    a0 += q0[d] * kd;
                    a1 += q1[d] * kd;
                    a2 += q2[d] * kd;
                    a3 += q3[d] * kd;
                }
                score += fmax(a0, 0.0f) * ws[hg];
                if (hg + 1 < nh) score += fmax(a1, 0.0f) * ws[hg + 1];
                if (hg + 2 < nh) score += fmax(a2, 0.0f) * ws[hg + 2];
                if (hg + 3 < nh) score += fmax(a3, 0.0f) * ws[hg + 3];
            }
        }
    }
    if (s < n_kv) {
        scores[(nf_i64)t * n_kv + s] = vis ? score : -NF_INF;
    }
}

/* LLM_DSA_IndexerScore_<KV>(7 scalars, 6 buffers); KV is a cache element type of common/kv_types.h. */
#define NF_DSA_SCORE_KERNEL(SFX, KV_T, KV_CODE)                                                \
NF_KERNEL void LLM_DSA_IndexerScore_##SFX(                                                     \
    const int n_tokens, const int n_kv, const int n_idx_heads, const int idx_dim,              \
    const int causal, const int swa_type, const int n_swa,                                     \
    NF_GLOBAL const float* NF_RESTRICT iq,                                                     \
    NF_GLOBAL const float* NF_RESTRICT w,                                                      \
    NF_GLOBAL const KV_T* NF_RESTRICT ik_cache,                                                \
    NF_GLOBAL const int* NF_RESTRICT q_pos,                                                    \
    NF_GLOBAL const int* NF_RESTRICT kv_pos,                                                   \
    NF_GLOBAL float* NF_RESTRICT scores) {                                                     \
    NF_LOCAL float qs[NF_DSA_QSTAGE];                                                          \
    NF_LOCAL float ws[NF_DSA_WSTAGE];                                                          \
    nf_dsa_score(n_tokens, n_kv, n_idx_heads, idx_dim, causal, swa_type, n_swa, iq, w,         \
                 (NF_GLOBAL const uchar*)ik_cache, KV_CODE, q_pos, kv_pos, scores, qs, ws);    \
}

NF_DSA_SCORE_KERNEL(F32, float, NF_KV_F32)
NF_DSA_SCORE_KERNEL(F16, NF_HALF, NF_KV_F16)
NF_DSA_SCORE_KERNEL(Q8_0, uchar, NF_KV_Q8_0)
NF_DSA_SCORE_KERNEL(BF16, ushort, NF_KV_BF16)
NF_DSA_SCORE_KERNEL(Q4_0, uchar, NF_KV_Q4_0)
NF_DSA_SCORE_KERNEL(Q4_1, uchar, NF_KV_Q4_1)
NF_DSA_SCORE_KERNEL(IQ4_NL, uchar, NF_KV_IQ4_NL)
NF_DSA_SCORE_KERNEL(Q5_0, uchar, NF_KV_Q5_0)
NF_DSA_SCORE_KERNEL(Q5_1, uchar, NF_KV_Q5_1)

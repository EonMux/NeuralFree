/* sparse/qsa.cl - LLM_QSA_KpoolUpdate_<KV> and LLM_QSA_KpoolScore_<KV>: pooled indexer keys and pool scores of the QSA k-pool indexer (glm5-next, qwen4exp). Requires: common/dialect.h, common/config.h, common/kv_types.h, common/attn_common.h */
/*
 * GLSL twin: shaders/llm/qsa.comp. Java: gguf.ops.sparse.MsaKernels (names, arg indices, codes, launch
 * geometry, preconditions) and gguf.ops.sparse.CpuQsa (reference math, same index formulas).
 *
 * Pools (llama_memory_hybrid_idx::kpool_layout_update, llama.cpp src/llama-memory-hybrid-idx.cpp:389-474). The single
 * sequence starts at position 0 and holds consecutive positions, so position order and cache order agree and
 * pool i is positions [i*kpool, (i+1)*kpool), complete once position (i+1)*kpool - 1 is cached. Position p
 * lives in indexer row p % kv_rows. The pooled key of a pool is stored in the row of its last member
 * ("rep", :652, :852-854).
 *
 * Indexer cache rows have the context's type_k (llama.cpp src/llama-memory-hybrid-idx.cpp:71), the variant's element
 * type <KV> (a cache element type of common/kv_types.h, read and written through its loaders and stores), and
 * row_width = idx_dim * kpool_row values (kpool_access, :575-598):
 *   GATED (glm5-next, kpool_row 3): [key | gate | pooled], gate_off = idx_dim, pooled_off = 2*idx_dim
 *   MEAN  (qwen4exp,  kpool_row 2): [raw key | pooled],     pooled_off = idx_dim
 * Element offsets are in values. For a block type, idx_dim, row_width and the offsets are multiples of 32, so every
 * slot is whole ggml blocks (the ggml_row_size views of kpool_access).
 */

#define NF_QSA_MODE_GATED  0
#define NF_QSA_MODE_MEAN   1
#define NF_QSA_STAGE_POOL  0
#define NF_QSA_STAGE_STORE 1

#define NF_QSA_QSTAGE 2048   /* floats of staged indexer queries per head chunk */
#define NF_QSA_WSTAGE 256    /* head weights staged per head chunk */

/*
 * LLM_QSA_KpoolUpdate_<KV>(i n_new, i first_pool, i stage, i mode, i kpool, i idx_dim, i row_width,
 *                     i gate_off, i pooled_off, i kv_rows, R ape, RWn<KV> idx_cache, RWn pooled)
 *
 * Re-pools the n_new pools first_pool .. first_pool + n_new - 1 (the pools whose last member the step wrote).
 * Work-item (j, d) handles value d of pool i = first_pool + j; member k is row r_k = (i*kpool + k) % kv_rows,
 * key_k = idx_cache[r_k*row_width + d].
 *   stage POOL, mode GATED (llama.cpp src/models/glm5-next.cpp:926-941):
 *     x_k = idx_cache[r_k*row_width + gate_off + d] + ape[k*idx_dim + d]
 *     pooled[j*idx_dim + d] = sum_k (e^(x_k - max_k x) * (1 / sum_k e^(x_k - max_k x))) * key_k
 *   stage POOL, mode MEAN (llama.cpp src/models/qwen4exp.cpp:770-780):
 *     pooled[j*idx_dim + d] = (key_0 + key_1 + ... + key_{kpool-1}) * (1 / kpool)
 *     (the caller then applies the RMS norm and the RoPE of the pooled key to `pooled`, qwen4exp.cpp:781-787)
 *   stage STORE (scatter_pooled, glm5-next.cpp:946, qwen4exp.cpp:792; ggml_set_rows into the pooled view):
 *     idx_cache[r_{kpool-1}*row_width + pooled_off + d] = KV(pooled[j*idx_dim + d])
 *     F32 as is, F16 / BF16 rounded to nearest even, a block type by the work-items with d % 32 == 0, one block
 *     each, by its ggml reference quantizer.
 * ape is a read-only placeholder in mode MEAN. pooled holds n_new rows of idx_dim.
 *
 * Preconditions (MsaKernels.checkKpool): idx_dim >= 1, 1 <= kpool <= kv_rows, first_pool >= 0,
 * offsets inside the row; block-type slots whole blocks. Launch: 1-D, global = roundUp(n_new*idx_dim, L), any local
 * size L <= LLM_DIMX.
 */
NF_INLINE void nf_qsa_kpool_update(
    const int n_new, const int first_pool, const int stage, const int mode, const int kpool,
    const int idx_dim, const int row_width, const int gate_off, const int pooled_off, const int kv_rows,
    NF_GLOBAL const float* ape, NF_GLOBAL uchar* idx_cache, const int kv_type, NF_GLOBAL float* pooled)
{
    const nf_i64 i = (nf_i64)get_global_id(0);
    if (n_new < 1 || idx_dim < 1 || kpool < 1 || kv_rows < 1 || first_pool < 0
        || i >= (nf_i64)n_new * idx_dim) {
        return;
    }
    const int j = (int)(i / idx_dim);
    const int d = (int)(i - (nf_i64)j * idx_dim);
    const nf_i64 pos0 = ((nf_i64)first_pool + j) * kpool;

    if (stage == NF_QSA_STAGE_STORE) {
        const nf_i64 rep = (pos0 + kpool - 1) % kv_rows;
        const nf_i64 e = rep * row_width + pooled_off + d;
        if (!nf_kv_is_block(kv_type)) {
            nf_kv_store1(idx_cache, e, pooled[i], kv_type);
        } else if ((d & (NF_KV_BLOCK - 1)) == 0) {
            nf_kv_store_global_block(idx_cache, e, pooled + i, kv_type);
        }
        return;
    }
    if (mode == NF_QSA_MODE_GATED) {
        float mx = -NF_INF;
        for (int k = 0; k < kpool; k++) {
            const nf_i64 r = (pos0 + k) % kv_rows;
            const float x = nf_kv_load1(idx_cache, r * row_width + gate_off + d, kv_type) + ape[(nf_i64)k * idx_dim + d];
            mx = fmax(mx, x);
        }
        float sum = 0.0f;
        for (int k = 0; k < kpool; k++) {
            const nf_i64 r = (pos0 + k) % kv_rows;
            const float x = nf_kv_load1(idx_cache, r * row_width + gate_off + d, kv_type) + ape[(nf_i64)k * idx_dim + d];
            sum += exp(x - mx);
        }
        const float inv = 1.0f / sum;
        float acc = 0.0f;
        for (int k = 0; k < kpool; k++) {
            const nf_i64 r = (pos0 + k) % kv_rows;
            const float x = nf_kv_load1(idx_cache, r * row_width + gate_off + d, kv_type) + ape[(nf_i64)k * idx_dim + d];
            acc += (exp(x - mx) * inv) * nf_kv_load1(idx_cache, r * row_width + d, kv_type);
        }
        pooled[i] = acc;
    } else {
        float acc = 0.0f;
        for (int k = 0; k < kpool; k++) {
            const nf_i64 r = (pos0 + k) % kv_rows;
            acc += nf_kv_load1(idx_cache, r * row_width + d, kv_type);
        }
        pooled[i] = acc * (1.0f / (float)kpool);
    }
}

#define NF_QSA_UPDATE_KERNEL(SFX, KV_T, KV_CODE)                                                               \
NF_KERNEL void LLM_QSA_KpoolUpdate_##SFX(                                                                      \
    const int n_new, const int first_pool, const int stage, const int mode, const int kpool,                   \
    const int idx_dim, const int row_width, const int gate_off, const int pooled_off, const int kv_rows,       \
    NF_GLOBAL const float* NF_RESTRICT ape,                                                                    \
    NF_GLOBAL KV_T* idx_cache,                                                                                 \
    NF_GLOBAL float* pooled)                                                                                   \
{                                                                                                              \
    nf_qsa_kpool_update(n_new, first_pool, stage, mode, kpool, idx_dim, row_width, gate_off, pooled_off,       \
                        kv_rows, ape, (NF_GLOBAL uchar*)idx_cache, KV_CODE, pooled);                           \
}

NF_QSA_UPDATE_KERNEL(F32, float, NF_KV_F32)
NF_QSA_UPDATE_KERNEL(F16, NF_HALF, NF_KV_F16)
NF_QSA_UPDATE_KERNEL(Q8_0, uchar, NF_KV_Q8_0)
NF_QSA_UPDATE_KERNEL(BF16, ushort, NF_KV_BF16)
NF_QSA_UPDATE_KERNEL(Q4_0, uchar, NF_KV_Q4_0)
NF_QSA_UPDATE_KERNEL(Q4_1, uchar, NF_KV_Q4_1)
NF_QSA_UPDATE_KERNEL(IQ4_NL, uchar, NF_KV_IQ4_NL)
NF_QSA_UPDATE_KERNEL(Q5_0, uchar, NF_KV_Q5_0)
NF_QSA_UPDATE_KERNEL(Q5_1, uchar, NF_KV_Q5_1)

/*
 * LLM_QSA_KpoolScore_<KV>(i n_tokens, i n_pool, i n_idx_heads, i idx_dim, i kpool, i row_width,
 *                    i pooled_off, i kv_rows, i has_w, f w_scale, i iq_off, i w_off, R iq, R w, R<KV> idx_cache,
 *                    R<int> q_pos, R<int> kv_pos, W scores)
 *
 * Lightning-indexer score of token t against pooled key i (ggml_lightning_indexer over pool_cells with
 * pool_mask; llama.cpp src/models/glm5-next.cpp:951-973, qwen4exp.cpp:797-810):
 *   e_i = (i+1)*kpool - 1 (last member), r_i = e_i % kv_rows
 *   scores[t*n_pool + i] = sum_h max(iq[t][h] . pooled_i, 0) * w_h     when e_i <= q_pos[t] and kv_pos[r_i] == e_i
 *                        = -INF                                         otherwise (pool_mask, :921-943 of
 *                                                                        llama-memory-hybrid-idx.cpp)
 *   pooled_i = idx_cache[r_i*row_width + pooled_off + d], d < idx_dim
 *   w_h = (has_w ? w[w_off + t*n_idx_heads + h] : 1) * w_scale
 * glm5-next: has_w = 1, w = indexer_proj . x, w_scale = (idx_dim*n_idx_heads)^-0.5 (glm5-next.cpp:953-954);
 * qwen4exp: has_w = 0 (w a read-only placeholder), w_scale = idx_dim^-0.5 (qwen4exp.cpp:807).
 * iq[iq_off + (t*n_idx_heads + h)*idx_dim + d]. Pools are scored for i < n_pool, the complete pools of the
 * cached positions; the selection then takes LLM_TopK_Rows(n_tokens, n_pool, top_k / kpool).
 *
 * Preconditions (MsaKernels.checkKpoolScore): n_pool >= 1, n_idx_heads >= 1, 1 <= idx_dim <= NF_QSA_QSTAGE,
 * kpool >= 1, kv_rows >= 1; block-type pooled slots whole blocks. Launch: 1-D, one workgroup per (token, block of L
 * pools), group g = t*ceil(n_pool/L) + b, pool i = b*L + lid; global = n_tokens*ceil(n_pool/L)*L,
 * local = L <= LLM_DIMX.
 */
NF_INLINE void nf_qsa_kpool_score(
    const int n_tokens, const int n_pool, const int n_idx_heads, const int idx_dim, const int kpool,
    const int row_width, const int pooled_off, const int kv_rows, const int has_w, const float w_scale,
    const int iq_off, const int w_off,
    NF_GLOBAL const float* iq, NF_GLOBAL const float* w, NF_GLOBAL const uchar* idx_cache, const int kv_type,
    NF_GLOBAL const int* q_pos, NF_GLOBAL const int* kv_pos, NF_GLOBAL float* scores,
    NF_LOCAL_PTR float* qs, NF_LOCAL_PTR float* ws)
{
    /* Uniform per workgroup: scalars and the group id only, before any barrier. */
    if (n_pool < 1 || n_idx_heads < 1 || idx_dim < 1 || idx_dim > NF_QSA_QSTAGE || kpool < 1 || kv_rows < 1) {
        return;
    }
    const int lid = (int)get_local_id(0);
    const int lsize = (int)get_local_size(0);
    const int nblk = (n_pool + lsize - 1) / lsize;
    const int g = (int)get_group_id(0);
    const int t = g / nblk;
    if (t >= n_tokens) {
        return;
    }
    const int pool = (g - t * nblk) * lsize + lid;

    int vis = 0;
    nf_i64 krow = 0;
    if (pool < n_pool) {
        const nf_i64 e = ((nf_i64)pool + 1) * kpool - 1;
        const nf_i64 r = e % kv_rows;
        vis = e <= q_pos[t] && kv_pos[r] == e;
        krow = r * row_width + pooled_off;
    }
    const nf_i64 kp = vis ? krow : 0;
    const nf_i64 qbase = (nf_i64)iq_off + (nf_i64)t * n_idx_heads * idx_dim;
    const nf_i64 wbase = (nf_i64)w_off + (nf_i64)t * n_idx_heads;
    const int hs = min(NF_QSA_WSTAGE, NF_QSA_QSTAGE / idx_dim);

    float score = 0.0f;
    for (int h0 = 0; h0 < n_idx_heads; h0 += hs) {
        const int nh = min(hs, n_idx_heads - h0);
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int i = lid; i < nh * idx_dim; i += lsize) {
            qs[i] = iq[qbase + (nf_i64)h0 * idx_dim + i];
        }
        for (int i = lid; i < nh; i += lsize) {
            ws[i] = (has_w ? w[wbase + h0 + i] : 1.0f) * w_scale;
        }
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
                    const float4 k4 = nf_kv_load4(idx_cache, kp + d, kv_type);
                    a0 += dot(vload4(0, q0 + d), k4);
                    a1 += dot(vload4(0, q1 + d), k4);
                    a2 += dot(vload4(0, q2 + d), k4);
                    a3 += dot(vload4(0, q3 + d), k4);
                }
                for (; d < idx_dim; ++d) {
                    const float kd = nf_kv_load1(idx_cache, kp + d, kv_type);
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
    if (pool < n_pool) {
        scores[(nf_i64)t * n_pool + pool] = vis ? score : -NF_INF;
    }
}

#define NF_QSA_SCORE_KERNEL(SFX, KV_T, KV_CODE)                                                                \
NF_KERNEL void LLM_QSA_KpoolScore_##SFX(                                                                       \
    const int n_tokens, const int n_pool, const int n_idx_heads, const int idx_dim, const int kpool,           \
    const int row_width, const int pooled_off, const int kv_rows, const int has_w, const float w_scale,        \
    const int iq_off, const int w_off,                                                                         \
    NF_GLOBAL const float* NF_RESTRICT iq,                                                                     \
    NF_GLOBAL const float* NF_RESTRICT w,                                                                      \
    NF_GLOBAL const KV_T* NF_RESTRICT idx_cache,                                                               \
    NF_GLOBAL const int* NF_RESTRICT q_pos,                                                                    \
    NF_GLOBAL const int* NF_RESTRICT kv_pos,                                                                   \
    NF_GLOBAL float* NF_RESTRICT scores)                                                                       \
{                                                                                                              \
    NF_LOCAL float qs[NF_QSA_QSTAGE];                                                                          \
    NF_LOCAL float ws[NF_QSA_WSTAGE];                                                                          \
    nf_qsa_kpool_score(n_tokens, n_pool, n_idx_heads, idx_dim, kpool, row_width, pooled_off, kv_rows, has_w,   \
                       w_scale, iq_off, w_off, iq, w, (NF_GLOBAL const uchar*)idx_cache, KV_CODE, q_pos,       \
                       kv_pos, scores, qs, ws);                                                                \
}

NF_QSA_SCORE_KERNEL(F32, float, NF_KV_F32)
NF_QSA_SCORE_KERNEL(F16, NF_HALF, NF_KV_F16)
NF_QSA_SCORE_KERNEL(Q8_0, uchar, NF_KV_Q8_0)
NF_QSA_SCORE_KERNEL(BF16, ushort, NF_KV_BF16)
NF_QSA_SCORE_KERNEL(Q4_0, uchar, NF_KV_Q4_0)
NF_QSA_SCORE_KERNEL(Q4_1, uchar, NF_KV_Q4_1)
NF_QSA_SCORE_KERNEL(IQ4_NL, uchar, NF_KV_IQ4_NL)
NF_QSA_SCORE_KERNEL(Q5_0, uchar, NF_KV_Q5_0)
NF_QSA_SCORE_KERNEL(Q5_1, uchar, NF_KV_Q5_1)

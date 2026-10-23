/* sparse/msa.cl - LLM_MSA_BlockScore and LLM_MSA_GatherAttn block-sparse attention (MiniMax sparse attention; also the attention of the QSA k-pool selection). Requires: common/dialect.h, common/config.h, common/reduce.h, common/kv_types.h, common/attn_common.h */
/*
 * GLSL twin: shaders/llm/msa.comp. Java: gguf.ops.sparse.MsaKernels (names, arg indices, launch geometry,
 * preconditions) and gguf.ops.sparse.CpuMsa (reference math, same index formulas).
 *
 * Addressing. Positions are mapped to KV rows by row(p) = p % kv_rows (KvLayout.rowOf for a ring and for a
 * plain cache, whose rows exceed every position). Position p is present when kv_pos[row(p)] == p and is
 * visible to a query at q_pos when it is also p <= q_pos. This is the per-position mask llama.cpp builds in
 * llama_kv_cache_msa_context::set_input_pos_mask (llama.cpp src/llama-kv-cache-msa.cpp:348-395) together with the
 * pos -> cell map of set_input_pos_slot (:304-346); MiniMax-M3 has no sliding window.
 *
 * KV element types (ik_type, k_type, v_type) are the NF_KV_* codes of common/kv_types.h (block-type row widths
 * are multiples of 32 values).
 */

#define NF_MSA_QSTAGE     2048     /* floats of staged index queries (one head must fit) */
#define NF_MSA_LOCAL_BIAS 1e30f    /* local-force bias of llm_graph_input_msa (llama.cpp models/minimax-m3.cpp:116) */
#define NF_MSA_HD_MAX     1024     /* largest attention head size of LLM_MSA_GatherAttn */
#define NF_MSA_ACC        8        /* output values per work-item of LLM_MSA_GatherAttn: hd <= local * NF_MSA_ACC */

/* Whether position p is cached (kv_pos[row(p)] == p) and causally visible to a query at p_q. */
NF_INLINE int nf_msa_visible(int p, int p_q, int kv_rows, NF_GLOBAL const int* kv_pos) {
    return p >= 0 && p <= p_q && kv_pos[p % kv_rows] == p;
}

/*
 * LLM_MSA_BlockScore(i n_tokens, i n_pos, i n_kv_heads, i idx_dim, i blk, i local_blocks, i kv_rows, i ik_type,
 *                    i iq_off, R iq, R<KV> ik_cache, R<int> q_pos, R<int> kv_pos, W block_scores)
 *
 * Position-block scores of the MSA indexer (llama.cpp src/models/minimax-m3.cpp:388-406 decode, :473-488 batch):
 *   sc[t][h][p]  = iq[t][h] . ik[row(p)]       when p is visible to token t, -INF otherwise
 *   bs[t][h][b]  = max_{p in [b*blk, min((b+1)*blk, n_pos))} sc[t][h][p]          (ggml_pool_2d MAX)
 *   block_scores[(t*n_kv_heads + h)*nblk + b] = bs + 1e30 when b is one of the `local_blocks` most recent blocks of
 *                  the token (L - local_blocks < b <= L, L = q_pos[t] / blk; minimax-m3.cpp:107-121), bs otherwise
 * with nblk = ceil(n_pos / blk). -INF + 1e30 stays -INF, as in the graph. n_kv_heads is the number of index
 * heads, one per GQA group (minimax-m3.cpp:382). Layouts: iq[iq_off + (t*n_kv_heads + h)*idx_dim + d] (normed,
 * roped index queries), ik_cache rows of idx_dim values (one index key per KV row).
 *
 * Preconditions (MsaKernels.checkBlockScore): n_pos >= 1, blk >= 1, local_blocks >= 0, kv_rows >= 1, n_kv_heads >= 1,
 * 1 <= idx_dim <= NF_MSA_QSTAGE; local size a power of two in [32, LLM_DIMX]. A violation writes nothing.
 *
 * Launch: 1-D, one workgroup per (token, block): group g = t*nblk + b; global = n_tokens*nblk*L, local = L.
 * Work-item lid scores positions b*blk + lid, + L, ...; the block maximum of each head is a workgroup
 * reduction (nf_wg_max). Subgroups: nf_wg_max uses subgroup reductions under USE_SUBGROUPS; L is a multiple
 * of 32.
 */
NF_KERNEL void LLM_MSA_BlockScore(
    const int n_tokens, const int n_pos, const int n_kv_heads, const int idx_dim, const int blk,
    const int local_blocks, const int kv_rows, const int ik_type, const int iq_off,
    NF_GLOBAL const float* NF_RESTRICT iq,
    NF_GLOBAL const uchar* NF_RESTRICT ik_cache,
    NF_GLOBAL const int* NF_RESTRICT q_pos,
    NF_GLOBAL const int* NF_RESTRICT kv_pos,
    NF_GLOBAL float* NF_RESTRICT block_scores)
{
    NF_LOCAL float qs[NF_MSA_QSTAGE];
    NF_LOCAL float red[LLM_DIMX];
    const int lid = (int)get_local_id(0);
    const int lsize = (int)get_local_size(0);
    /* Every condition below depends only on scalars, the local size and the group id: the returns are uniform
     * per workgroup and happen before any barrier. */
    if (n_pos < 1 || blk < 1 || local_blocks < 0 || kv_rows < 1 || n_kv_heads < 1 || idx_dim < 1
        || idx_dim > NF_MSA_QSTAGE || lsize > LLM_DIMX) {
        return;
    }
    const int nblk = (n_pos + blk - 1) / blk;
    const int g = (int)get_group_id(0);
    const int t = g / nblk;
    if (t >= n_tokens) {
        return;
    }
    const int b = g - t * nblk;
    const int p_q = q_pos[t];
    const int p_begin = b * blk;
    const int p_end = min(p_begin + blk, n_pos);
    const int l_blk = p_q / blk;
    const float bias = (b <= l_blk && b > l_blk - local_blocks) ? NF_MSA_LOCAL_BIAS : 0.0f;
    const int hs = min(n_kv_heads, NF_MSA_QSTAGE / idx_dim);
    const nf_i64 out_row = (nf_i64)t * n_kv_heads;

    for (int h0 = 0; h0 < n_kv_heads; h0 += hs) {
        const int nh = min(hs, n_kv_heads - h0);
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int i = lid; i < nh * idx_dim; i += lsize) {
            qs[i] = iq[(nf_i64)iq_off + (out_row + h0) * idx_dim + i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int hg = 0; hg < nh; hg += 4) {
            /* Heads past nh repeat the last head; their maxima are computed and dropped. */
            NF_LOCAL_PTR const float* q0 = qs + hg * idx_dim;
            NF_LOCAL_PTR const float* q1 = qs + min(hg + 1, nh - 1) * idx_dim;
            NF_LOCAL_PTR const float* q2 = qs + min(hg + 2, nh - 1) * idx_dim;
            NF_LOCAL_PTR const float* q3 = qs + min(hg + 3, nh - 1) * idx_dim;
            float m0 = -NF_INF, m1 = -NF_INF, m2 = -NF_INF, m3 = -NF_INF;
            for (int p = p_begin + lid; p < p_end; p += lsize) {
                if (nf_msa_visible(p, p_q, kv_rows, kv_pos)) {
                    const nf_i64 e0 = (nf_i64)(p % kv_rows) * idx_dim;
                    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
                    int d = 0;
                    for (; d + 4 <= idx_dim; d += 4) {
                        const float4 k4 = nf_kv_load4(ik_cache, e0 + d, ik_type);
                        a0 += dot(vload4(0, q0 + d), k4);
                        a1 += dot(vload4(0, q1 + d), k4);
                        a2 += dot(vload4(0, q2 + d), k4);
                        a3 += dot(vload4(0, q3 + d), k4);
                    }
                    for (; d < idx_dim; ++d) {
                        const float kd = nf_kv_load1(ik_cache, e0 + d, ik_type);
                        a0 += q0[d] * kd;
                        a1 += q1[d] * kd;
                        a2 += q2[d] * kd;
                        a3 += q3[d] * kd;
                    }
                    m0 = fmax(m0, a0);
                    m1 = fmax(m1, a1);
                    m2 = fmax(m2, a2);
                    m3 = fmax(m3, a3);
                }
            }
            /* nh and hg are uniform, so every reduction below is reached by the whole workgroup. */
            const float r0 = nf_wg_max(red, m0);
            if (lid == 0) {
                block_scores[(out_row + h0 + hg) * nblk + b] = r0 + bias;
            }
            if (hg + 1 < nh) {
                const float r1 = nf_wg_max(red, m1);
                if (lid == 0) {
                    block_scores[(out_row + h0 + hg + 1) * nblk + b] = r1 + bias;
                }
            }
            if (hg + 2 < nh) {
                const float r2 = nf_wg_max(red, m2);
                if (lid == 0) {
                    block_scores[(out_row + h0 + hg + 2) * nblk + b] = r2 + bias;
                }
            }
            if (hg + 3 < nh) {
                const float r3 = nf_wg_max(red, m3);
                if (lid == 0) {
                    block_scores[(out_row + h0 + hg + 3) * nblk + b] = r3 + bias;
                }
            }
        }
    }
}

/*
 * LLM_MSA_GatherAttn(i n_tokens, i n_head, i n_kv_heads, i hd, i blk, i K, f kq_scale, i sel_heads, i tail,
 *                    i kv_rows, i k_row_stride, i v_row_stride, i k_type, i v_type, i q_stride, i q_off,
 *                    i out_stride, i out_off, R q, R<KV> k_cache, R<KV> v_cache, R<int> q_pos, R<int> kv_pos,
 *                    R<int> blocks, W out)
 *
 * Attention of query head h of token t over the positions of its selected position blocks:
 *   kvh  = h / (n_head / n_kv_heads),  sh = sel_heads == 1 ? 0 : kvh
 *   slot s in [0, K):  block blocks[(t*sel_heads + sh)*K + s] (-1 = unused slot)
 *   slot K (tail == 1): the block q_pos[t] / blk when (q_pos[t] + 1) % blk != 0 (the incomplete pool that
 *                       holds the query), else unused
 *   cells: p = block*blk + o, o in [0, blk), visible per nf_msa_visible; row r = p % kv_rows
 *   s_p  = (q[t][h] . K[r][kvh]) * kq_scale;  out[t][h] = sum_p softmax(s)_p * V[r][kvh]   (0 when no cell)
 * MSA (llama.cpp src/models/minimax-m3.cpp:388-447 decode gather, :490-522 batch mask): sel_heads = n_kv_heads (one
 * selection per GQA group), tail = 0; the local blocks are already forced into the top-K by the score bias.
 * QSA attention (llama.cpp src/models/qwen4exp.cpp:812-880, glm5-next.cpp:975-1042): blocks are the k-pools
 * (blk = kpool), sel_heads = 1, tail = indexer_kpool_select_tail (always 1 for qwen4exp, qwen4exp.cpp:83; read
 * from the GGUF for glm5-next, glm5-next.cpp:44); the tail block's visible positions are the
 * tail_idxs cells of llama_memory_hybrid_idx_context::set_input_kpool (llama.cpp
 * src/llama-memory-hybrid-idx.cpp:946-966).
 * The tail block is never among the selected blocks: its pool is incomplete, so its score is -INF and
 * LLM_TopK_Rows does not select it.
 * An MLA latent cache (glm5-next) binds one buffer to k_cache and v_cache with n_kv_heads = 1 and
 * v_row_stride = k_row_stride.
 * Layouts (elements): q[q_off + t*q_stride + h*hd + d]; K[r][kvh][d] at r*k_row_stride + kvh*hd + d (same for V
 * with v_row_stride); out[out_off + t*out_stride + h*hd + d].
 *
 * Preconditions (MsaKernels.checkGatherAttn): n_head % n_kv_heads == 0, sel_heads in {1, n_kv_heads},
 * 4 <= hd <= NF_MSA_HD_MAX, hd % 4 == 0, hd <= local * NF_MSA_ACC, blk >= 1, K >= 0, kv_rows >= 1, local size
 * a power of two in [32, LLM_DIMX]. A violation writes NaN over the group's output row.
 *
 * Launch: 1-D, one workgroup per (token, head): group g = t*n_head + h; global = n_tokens*n_head*L, local = L.
 * Algorithm: q is staged in local memory; the (K + tail)*blk candidate cells are processed in tiles of L:
 * work-item lid scores cell c0 + lid, the tile maximum and sum update an online softmax (m, l), and every
 * work-item then accumulates its output values d = lid + i*L over the tile's cells, reading V rows coalesced
 * along d. Subgroups: nf_wg_max / nf_wg_sum use subgroup reductions under USE_SUBGROUPS; L is a multiple of 32.
 */
NF_KERNEL void LLM_MSA_GatherAttn(
    const int n_tokens, const int n_head, const int n_kv_heads, const int hd, const int blk, const int K,
    const float kq_scale, const int sel_heads, const int tail, const int kv_rows,
    const int k_row_stride, const int v_row_stride, const int k_type, const int v_type,
    const int q_stride, const int q_off, const int out_stride, const int out_off,
    NF_GLOBAL const float* NF_RESTRICT q,
    NF_GLOBAL const uchar* k_cache,
    NF_GLOBAL const uchar* v_cache,
    NF_GLOBAL const int* NF_RESTRICT q_pos,
    NF_GLOBAL const int* NF_RESTRICT kv_pos,
    NF_GLOBAL const int* NF_RESTRICT blocks,
    NF_GLOBAL float* NF_RESTRICT out)
{
    NF_LOCAL float q_s[NF_MSA_HD_MAX];
    NF_LOCAL float p_s[LLM_DIMX];
    NF_LOCAL int r_s[LLM_DIMX];
    NF_LOCAL float red[LLM_DIMX];
    if (n_head < 1) {
        return;
    }
    const int lid = (int)get_local_id(0);
    const int lsize = (int)get_local_size(0);
    const int g = (int)get_group_id(0);
    const int t = g / n_head;
    /* Uniform per workgroup: depends on scalars and the group id only, before any barrier. */
    if (t >= n_tokens) {
        return;
    }
    const int h = g - t * n_head;
    const nf_i64 o_base = (nf_i64)out_off + (nf_i64)t * out_stride + (nf_i64)h * hd;
    if (n_kv_heads < 1 || n_head % n_kv_heads != 0 || (sel_heads != 1 && sel_heads != n_kv_heads)
        || hd < 4 || hd > NF_MSA_HD_MAX || (hd & 3) != 0 || hd > lsize * NF_MSA_ACC || lsize > LLM_DIMX
        || blk < 1 || K < 0 || kv_rows < 1) {
        for (int d = lid; d < hd && d < NF_MSA_HD_MAX; d += lsize) {
            out[o_base + d] = NF_NAN;
        }
        return;
    }
    const int kvh = h / (n_head / n_kv_heads);
    const int sh = sel_heads == 1 ? 0 : kvh;
    const int p_q = q_pos[t];
    NF_GLOBAL const int* sel = blocks + ((nf_i64)t * sel_heads + sh) * K;
    const int tail_blk = (tail != 0 && (p_q + 1) % blk != 0) ? p_q / blk : -1;
    const int n_slots = K + (tail != 0 ? 1 : 0);
    const int n_cells = n_slots * blk;
    const nf_i64 k_head = (nf_i64)kvh * hd;
    const nf_i64 v_head = (nf_i64)kvh * hd;

    const nf_i64 q_base = (nf_i64)q_off + (nf_i64)t * q_stride + (nf_i64)h * hd;
    for (int d = lid; d < hd; d += lsize) {
        q_s[d] = q[q_base + d];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    float m = -NF_INF;
    float l = 0.0f;
    float o[NF_MSA_ACC];
#pragma unroll
    for (int i = 0; i < NF_MSA_ACC; i++) {
        o[i] = 0.0f;
    }

    for (int c0 = 0; c0 < n_cells; c0 += lsize) {
        const int c = c0 + lid;
        int row = -1;
        float s = -NF_INF;
        if (c < n_cells) {
            const int slot = c / blk;
            const int bk = slot < K ? sel[slot] : tail_blk;
            if (bk >= 0) {
                const int p = bk * blk + (c - slot * blk);
                if (nf_msa_visible(p, p_q, kv_rows, kv_pos)) {
                    row = p % kv_rows;
                    const nf_i64 e0 = (nf_i64)row * k_row_stride + k_head;
                    float acc = 0.0f;
                    for (int d = 0; d < hd; d += 4) {
                        acc += dot(vload4(0, q_s + d), nf_kv_load4(k_cache, e0 + d, k_type));
                    }
                    s = acc * kq_scale;
                }
            }
        }
        const float m_new = fmax(m, nf_wg_max(red, s));
        const float alpha = nf_attn_rescale(m, m_new);
        const float pw = nf_attn_rescale(s, m_new);
        p_s[lid] = pw;
        r_s[lid] = row;
        l = l * alpha + nf_wg_sum(red, pw);   /* its barriers also publish p_s and r_s */
#pragma unroll
        for (int i = 0; i < NF_MSA_ACC; i++) {
            o[i] *= alpha;
        }
        const int n_here = min(lsize, n_cells - c0);
        for (int j = 0; j < n_here; j++) {
            const float pj = p_s[j];
            if (pj != 0.0f) {
                const nf_i64 ev = (nf_i64)r_s[j] * v_row_stride + v_head;
#pragma unroll
                for (int i = 0; i < NF_MSA_ACC; i++) {
                    const int d = lid + i * lsize;
                    if (d < hd) {
                        o[i] += pj * nf_kv_load1(v_cache, ev + d, v_type);
                    }
                }
            }
        }
        m = m_new;
        barrier(CLK_LOCAL_MEM_FENCE);   /* p_s and r_s are rewritten by the next tile */
    }

#pragma unroll
    for (int i = 0; i < NF_MSA_ACC; i++) {
        const int d = lid + i * lsize;
        if (d < hd) {
            out[o_base + d] = l > 0.0f ? o[i] / l : 0.0f;
        }
    }
}

/* flash_prefill.cl - LLM_FlashAttn_<KV>, LLM_FlashAttn_Mixed (identity pages) and LLM_FlashAttnPaged_<KV>,
 * LLM_FlashAttnPaged_Mixed (paged) tiled prefill attention. Requires: common/dialect.h, common/config.h,
 * common/reduce.h, common/math.h, common/kv_types.h, common/attn_common.h */
/*
 * Attention of n_tokens query rows against a key range (prefill through the KV cache, encoders and
 * non-causal passes over the current batch, speculative verification) with online softmax,
 * query/key tiling, GQA packing and head-dimension chunks of NF_FA_CHUNK (64).
 * GLSL twin: shaders/llm/flash_prefill.comp. Java: gguf.ops.attn.AttnKernels,
 * gguf.ops.attn.FlashAttnTiling (bq, bk, g_tile, grid), gguf.ops.attn.CpuAttentionCore (reference).
 *
 * Layouts (offsets in elements, 64-bit inside the kernel):
 *   q        q[q_off + t * q_row_stride + h * d_k + d], query position q_pos[t]
 *   K value  k[j * k_row_stride + kvh * k_head_stride + d], key position kv_pos[j] (-1 = empty);
 *            same for v with the v strides; K is read as k_type and V as v_type (the variant's type in
 *            LLM_FlashAttn_<KV>, the trailing kv_types argument of LLM_FlashAttn_Mixed; flash_decode.cl)
 *   out      out[out_off + t * out_row_stride + h * d_v + d]                      (direct_out = 1)
 *   partial  partial[((t * n_head + h) * n_split + s) * (d_v + 2) + d], m at + d_v, l at + d_v + 1
 *            (direct_out = 0; LLM_FlashAttn_Combine with rows = t * n_head + h writes dense rows
 *            out[out_off + (t * n_head + h) * d_v + d], so a split call needs out_row_stride == n_head * d_v)
 *
 * Grid: global {ceil(n_tokens / bq) * 128, n_head_kv * ceil(G / g_tile), n_split}, local {128, 1, 1}.
 * Workgroup (b, y, s) owns R = bq * g_tile query rows r = i * g_tile + g: token t = b * bq + i, head
 * h = h0 + g with h0 = kvh * G + gt * g_tile (kvh, gt from y as in flash_decode.cl), and keys
 * [kv_start + s * split_len, min(+ split_len, kv_end)) in tiles of bk keys.
 *
 * Shared memory (NF_FA_SMEM_FLOATS floats):
 *   qc[R * 64] | kvc[bk * 64] | S[R * bk] | m[R] | l[R] | alpha[R]
 * Plan constraints (FlashAttnTiling.prefill guarantees them; a violation writes NaN outputs):
 *   R * 64 + bk * 64 + R * bk + 3 * R <= NF_FA_SMEM_FLOATS, R * d_v <= NF_FA_WG * NF_FA_ACC_MAX
 *   (output accumulators stay in registers), g_tile <= NF_FD_MAX_G, d_k % 4 == 0, d_v % 4 == 0,
 *   split_len > 0, local size == NF_FA_WG, direct_out only with n_split == 1, k_type and v_type KV cache type codes.
 *
 * Per key tile:
 *   0. Uniform skip: a tile in which no (token, key) pair of the workgroup is visible is skipped
 *      (causal and window tiles outside every row's range cost one reduction).
 *   1. S = Q K^T accumulated over d_k in chunks of 64 (Q chunk and K chunk staged in qc / kvc;
 *      the dot loop starts at a per-key rotation, so the 64-float rows hit distinct banks).
 *   2. S = score + bias, -INF where not visible.
 *   3. Online softmax per row: m' = max(m, max S), alpha = e^(m - m'), P = e^(S - m'), l = l * alpha + sum P.
 *   4. O *= alpha; O += P V over d_v in chunks of 64 (V chunk staged in kvc); keys with P == 0 are skipped.
 * Finally rows apply sinks and 1 / l (l == 0 writes 0), or emit (o, m, l) partials.
 * Subgroups: the skip test uses nf_wg_max (common/reduce.h); local size 128 is a multiple of every subgroup size.
 */

/* Plan constraints of LLM_FlashAttn_*; see the header comment. */
NF_INLINE int nf_fa_plan_ok(int n_head, int n_head_kv, int d_k, int d_v, int bq, int bk, int g_tile,
                            int n_split, int split_len, int direct_out, int k_type, int v_type) {
    const int rows = bq * g_tile;
    return nf_attn_kv_type_ok(k_type) && nf_attn_kv_type_ok(v_type)
        && n_head > 0 && n_head_kv > 0 && n_head % n_head_kv == 0
        && g_tile >= 1 && g_tile <= NF_FD_MAX_G && bq >= 1 && bk >= 1
        && d_k > 0 && d_v > 0 && (d_k & 3) == 0 && (d_v & 3) == 0 && n_split >= 1 && split_len > 0
        && rows * NF_FA_CHUNK + bk * NF_FA_CHUNK + rows * bk + 3 * rows <= NF_FA_SMEM_FLOATS
        && rows * d_v <= NF_FA_WG * NF_FA_ACC_MAX
        && (direct_out == 0 || n_split == 1)
        && (int)get_local_size(0) == NF_FA_WG;
}

/* Writes NaN over the outputs of a call whose arguments violate the plan (workgroup (0, 0, 0) only). */
NF_INLINE void nf_fa_poison(int n_tokens, int n_head, int d_v, int n_split, int out_row_stride,
                            int direct_out, int out_off, NF_GLOBAL float* out, NF_GLOBAL float* partial) {
    if (get_group_id(0) != 0 || get_group_id(1) != 0 || get_group_id(2) != 0
        || n_tokens <= 0 || n_head <= 0 || d_v <= 0 || n_split <= 0) {
        return;
    }
    const int lid = (int)get_local_id(0);
    const int lsz = (int)get_local_size(0);
    if (direct_out) {
        for (int t = 0; t < n_tokens; t++) {
            for (int i = lid; i < n_head * d_v; i += lsz) {
                out[(nf_i64)out_off + (nf_i64)t * out_row_stride + i] = NF_NAN;
            }
        }
    } else {
        const nf_i64 n = (nf_i64)n_tokens * n_head * n_split * (d_v + 2);
        for (nf_i64 i = lid; i < n; i += lsz) {
            partial[i] = NF_NAN;
        }
    }
}

NF_INLINE void nf_fa_run(
    const int n_tokens, const int n_head, const int n_head_kv, const int d_k, const int d_v,
    const int bq, const int bk, const int g_tile, const int n_split, const int split_len,
    const int kv_start, const int kv_end,
    const int q_row_stride, const int out_row_stride,
    const int k_row_stride, const int k_head_stride, const int v_row_stride, const int v_head_stride,
    const float scale, const float softcap, const float max_bias,
    const int causal, const int swa_type, const int n_swa,
    const int bias_mode, const int n_buckets, const int max_distance,
    const int has_sinks, const int direct_out, const int q_off, const int out_off,
    NF_GLOBAL const float* q, NF_GLOBAL const uchar* k, NF_GLOBAL const uchar* v,
    NF_GLOBAL const int* q_pos, NF_GLOBAL const int* kv_pos,
    NF_GLOBAL const float* sinks, NF_GLOBAL const float* rel_bias,
    NF_GLOBAL float* out, NF_GLOBAL float* partial, NF_GLOBAL const float* kv_scales,
    const int k_type, const int v_type, const int v_aliases_k,
    const int page_shift, NF_GLOBAL const int* pt, const int pt_off,
    const int tree, NF_GLOBAL const ulong* tree_mask, const int n_fresh,
    NF_LOCAL_PTR float* smem, NF_LOCAL_PTR float* red, NF_LOCAL_PTR float* slope_s)
{
    if (!nf_fa_plan_ok(n_head, n_head_kv, d_k, d_v, bq, bk, g_tile, n_split, split_len, direct_out, k_type,
                       v_type)) {
        nf_fa_poison(n_tokens, n_head, d_v, n_split, out_row_stride, direct_out, out_off, out, partial);
        return;   /* uniform across the workgroup: no barrier has been reached */
    }

    /* A vendor cache keeps its packed side planes and Lloyd-Max table in the kv_scales binding (common/kv_types.h
     * nf_kv_side_plane_base): the K planes of a row come first, the V planes after them unless V aliases K. */
    const int k_plane = nf_kv_is_vendor(k_type) ? nf_kv_vendor_plane_bytes(k_type, k_row_stride, n_head_kv) : 0;
    const int v_plane = !v_aliases_k && nf_kv_is_vendor(v_type)
                        ? nf_kv_vendor_plane_bytes(v_type, v_row_stride, n_head_kv) : 0;
    const int plane_stride = k_plane + v_plane;
    NF_GLOBAL const uchar* k_planes = (NF_GLOBAL const uchar*)kv_scales + nf_kv_side_plane_base(n_head_kv);
    NF_GLOBAL const uchar* v_planes = k_planes + (v_aliases_k ? 0 : k_plane);
    NF_GLOBAL const float* centroids = kv_scales + 2 * n_head_kv;

    const int lid = (int)get_local_id(0);
    const int split = (int)get_group_id(2);
    const int G = n_head / n_head_kv;
    const int n_gt = (G + g_tile - 1) / g_tile;
    const int kvh = (int)get_group_id(1) / n_gt;
    const int gt = (int)get_group_id(1) - kvh * n_gt;
    const int h0 = kvh * G + gt * g_tile;
    const int g_cnt = min(g_tile, G - gt * g_tile);
    const int t0 = (int)get_group_id(0) * bq;
    const int R = bq * g_tile;
    const int k_begin = kv_start + split * split_len;
    const int k_stop = split < n_split ? min(k_begin + split_len, kv_end) : k_begin;

    NF_LOCAL_PTR float* qc = smem;
    NF_LOCAL_PTR float* kvc = qc + R * NF_FA_CHUNK;
    NF_LOCAL_PTR float* sc = kvc + bk * NF_FA_CHUNK;
    NF_LOCAL_PTR float* m_s = sc + R * bk;
    NF_LOCAL_PTR float* l_s = m_s + R;
    NF_LOCAL_PTR float* a_s = l_s + R;

    for (int r = lid; r < R; r += NF_FA_WG) {
        m_s[r] = -NF_INF;
        l_s[r] = 0.0f;
        a_s[r] = 0.0f;
    }
    if (lid < g_tile) {
        slope_s[lid] = ((bias_mode == NF_ATTN_BIAS_ALIBI || bias_mode == NF_ATTN_BIAS_ALIBI_SQRT) && lid < g_cnt)
                       ? nf_alibi_slope(h0 + lid, n_head, max_bias) : 1.0f;
    }

    /* output pairs (row, d) owned by this item: pair lid + i * 128 of R * d_v */
    const int n_pairs = R * d_v;
    float o[NF_FA_ACC_MAX];
    int pr[NF_FA_ACC_MAX];
    int pd[NF_FA_ACC_MAX];
#pragma unroll
    for (int i = 0; i < NF_FA_ACC_MAX; i++) {
        const int pi = lid + i * NF_FA_WG;
        const int r = pi < n_pairs ? pi / d_v : 0;
        o[i] = 0.0f;
        pr[i] = r;
        pd[i] = pi - r * d_v;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const nf_i64 k_head = (nf_i64)kvh * k_head_stride;
    const nf_i64 v_head = (nf_i64)kvh * v_head_stride;
    /* The KV-cache scales of this KV head (common/kv_types.h; 1.0 without a scaled cache). */
    for (int kt = k_begin; kt < k_stop; kt += bk) {
        const int nk = min(bk, k_stop - kt);

        /* 0. uniform skip of tiles invisible to every row */
        float any = 0.0f;
        for (int idx = lid; idx < bq * nk; idx += NF_FA_WG) {
            const int i = idx / nk;
            const int tq = t0 + i;
            const int slot = kt + idx - i * nk;
            if (tq < n_tokens
                && nf_kv_page_ok(slot, page_shift, pt, pt_off)
                && nf_attn_tree_visible(kv_pos[nf_kv_row(slot, page_shift, pt, pt_off)],
                                        tree ? tree_mask[tq] : (ulong)0, kv_end, n_fresh,
                                        q_pos[tq], causal, swa_type, n_swa)) {
                any = 1.0f;
            }
        }
        any = nf_wg_max(red, any);
        if (any == 0.0f) {
            continue;
        }

        /* 1. S = Q K^T over head-dim chunks */
        for (int idx = lid; idx < R * bk; idx += NF_FA_WG) {
            sc[idx] = 0.0f;
        }
        for (int c = 0; c < d_k; c += NF_FA_CHUNK) {
            barrier(CLK_LOCAL_MEM_FENCE);
            for (int idx = lid; idx < R * (NF_FA_CHUNK / 4); idx += NF_FA_WG) {
                const int r = idx / (NF_FA_CHUNK / 4);
                const int c4 = idx - r * (NF_FA_CHUNK / 4);
                const int d = c + 4 * c4;
                const int i = r / g_tile;
                const int g = r - i * g_tile;
                const int tq = t0 + i;
                float4 val = NF_F4S(0.0f);
                if (tq < n_tokens && g < g_cnt && d < d_k) {
                    val = vload4(0, q + (nf_i64)q_off + (nf_i64)tq * q_row_stride + (nf_i64)(h0 + g) * d_k + d);
                }
                vstore4(val, 0, qc + r * NF_FA_CHUNK + 4 * c4);
            }
            for (int idx = lid; idx < bk * (NF_FA_CHUNK / 4); idx += NF_FA_WG) {
                const int kk = idx / (NF_FA_CHUNK / 4);
                const int c4 = idx - kk * (NF_FA_CHUNK / 4);
                const int d = c + 4 * c4;
                float4 val = NF_F4S(0.0f);
                if (kk < nk && d < d_k && nf_kv_page_ok(kt + kk, page_shift, pt, pt_off)) {
                    const int krow = nf_kv_row(kt + kk, page_shift, pt, pt_off);
                    const float ks = nf_kv_head_scale(k_type, kv_scales, krow, kvh, n_head_kv, 0);
                    val = nf_kv_attn_load4(k, (nf_i64)krow * k_row_stride + k_head + d, krow, k_type, k_planes,
                                           plane_stride, k_row_stride, d_k, centroids) * ks;
                }
                vstore4(val, 0, kvc + kk * NF_FA_CHUNK + 4 * c4);
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            for (int idx = lid; idx < R * bk; idx += NF_FA_WG) {
                const int r = idx / bk;
                const int kk = idx - r * bk;
                NF_LOCAL_PTR const float* qrow = qc + r * NF_FA_CHUNK;
                NF_LOCAL_PTR const float* krow = kvc + kk * NF_FA_CHUNK;
                float acc = 0.0f;
                for (int dd = 0; dd < NF_FA_CHUNK; dd++) {
                    const int x = (dd + kk) & (NF_FA_CHUNK - 1);
                    acc += qrow[x] * krow[x];
                }
                sc[idx] += acc;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        /* 2. scores, bias and mask */
        for (int idx = lid; idx < R * bk; idx += NF_FA_WG) {
            const int r = idx / bk;
            const int kk = idx - r * bk;
            const int i = r / g_tile;
            const int g = r - i * g_tile;
            const int tq = t0 + i;
            float s = -NF_INF;
            if (kk < nk && tq < n_tokens && g < g_cnt) {
                const int slot = kt + kk;
                const int p_q = q_pos[tq];
                /* A masked page's row is not a row, so kv_pos is read only after the page test; an empty
                 * cell's kv_pos stays -1 and is never visible. */
                int p_k = -1;
                if (nf_kv_page_ok(slot, page_shift, pt, pt_off)) {
                    p_k = kv_pos[nf_kv_row(slot, page_shift, pt, pt_off)];
                }
                if (nf_attn_tree_visible(p_k, tree ? tree_mask[tq] : (ulong)0, kv_end, n_fresh,
                                         p_q, causal, swa_type, n_swa)) {
                    float dot = sc[idx];
                    if (nf_kv_is_tq_mse(k_type)) {
                        /* The key decodes to its centroid vector: scale by norm / |c| around the rotated query. */
                        const int krow = nf_kv_row(slot, page_shift, pt, pt_off);
                        dot *= nf_kv_vendor_tq_score(k, (nf_i64)krow * k_row_stride + k_head, krow, k_type,
                                                     k_planes, plane_stride, k_row_stride, d_k, centroids);
                    }
                    s = nf_attn_score(dot, scale, softcap)
                      + nf_attn_bias(bias_mode, h0 + g, n_head, p_q, p_k, slope_s[g], rel_bias,
                                     n_buckets, max_distance, tq);
                }
            }
            sc[idx] = s;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        /* 3. online softmax update per row */
        for (int r = lid; r < R; r += NF_FA_WG) {
            NF_LOCAL_PTR float* srow = sc + r * bk;
            const int rot = r % bk;
            const float m_old = m_s[r];
            float m_new = m_old;
            for (int j = 0; j < bk; j++) {
                int kk = j + rot;
                kk = kk >= bk ? kk - bk : kk;
                m_new = fmax(m_new, srow[kk]);
            }
            float sum = 0.0f;
            for (int j = 0; j < bk; j++) {
                int kk = j + rot;
                kk = kk >= bk ? kk - bk : kk;
                const float p = nf_attn_rescale(srow[kk], m_new);
                srow[kk] = p;
                sum += p;
            }
            const float alpha = nf_attn_rescale(m_old, m_new);
            l_s[r] = l_s[r] * alpha + sum;
            m_s[r] = m_new;
            a_s[r] = alpha;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        /* 4. O = O * alpha + P V over value chunks */
#pragma unroll
        for (int i = 0; i < NF_FA_ACC_MAX; i++) {
            if (lid + i * NF_FA_WG < n_pairs) {
                o[i] *= a_s[pr[i]];
            }
        }
        for (int c = 0; c < d_v; c += NF_FA_CHUNK) {
            for (int idx = lid; idx < bk * (NF_FA_CHUNK / 4); idx += NF_FA_WG) {
                const int kk = idx / (NF_FA_CHUNK / 4);
                const int c4 = idx - kk * (NF_FA_CHUNK / 4);
                const int d = c + 4 * c4;
                float4 val = NF_F4S(0.0f);
                if (kk < nk && d < d_v && nf_kv_page_ok(kt + kk, page_shift, pt, pt_off)) {
                    const int vrow = nf_kv_row(kt + kk, page_shift, pt, pt_off);
                    const float vs = nf_kv_head_scale(v_type, kv_scales, vrow, kvh, n_head_kv, 1);
                    val = nf_kv_attn_load4(v, (nf_i64)vrow * v_row_stride + v_head + d, vrow, v_type, v_planes,
                                           plane_stride, v_row_stride, d_v, centroids) * vs;
                }
                vstore4(val, 0, kvc + kk * NF_FA_CHUNK + 4 * c4);
            }
            barrier(CLK_LOCAL_MEM_FENCE);
#pragma unroll
            for (int i = 0; i < NF_FA_ACC_MAX; i++) {
                const int dd = pd[i] - c;
                if (lid + i * NF_FA_WG < n_pairs && dd >= 0 && dd < NF_FA_CHUNK) {
                    NF_LOCAL_PTR const float* prow = sc + pr[i] * bk;
                    float acc = 0.0f;
                    for (int kk = 0; kk < nk; kk++) {
                        const float p = prow[kk];
                        if (p != 0.0f) {
                            acc += p * kvc[kk * NF_FA_CHUNK + dd];
                        }
                    }
                    o[i] += acc;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    /* 5. output */
#pragma unroll
    for (int i = 0; i < NF_FA_ACC_MAX; i++) {
        const int r = pr[i];
        const int d = pd[i];
        const int ti = r / g_tile;
        const int g = r - ti * g_tile;
        const int tq = t0 + ti;
        if (lid + i * NF_FA_WG < n_pairs && tq < n_tokens && g < g_cnt && split < n_split) {
            const int h = h0 + g;
            const float m = m_s[r];
            const float l = l_s[r];
            if (direct_out) {
                const float2 fl = has_sinks ? nf_attn_sink(m, l, sinks[h]) : NF_F2(1.0f, l);
                out[(nf_i64)out_off + (nf_i64)tq * out_row_stride + (nf_i64)h * d_v + d] =
                    fl.y > 0.0f ? o[i] * fl.x / fl.y : 0.0f;
            } else {
                NF_GLOBAL float* pp = partial + (((nf_i64)tq * n_head + h) * n_split + split) * (d_v + 2);
                pp[d] = o[i];
                if (d == 0) {
                    pp[d_v] = m;
                    pp[d_v + 1] = l;
                }
            }
        }
    }
}

/*
 * LLM_FlashAttn_<KV>(29 scalars, 11 buffers); KV is a cache element type of common/kv_types.h. Encoders pass k/v of the
 * current batch and kv_pos == q_pos; k and v may be the same buffer; sinks, rel_bias and kv_scales are read-only
 * placeholders when unused (kv_scales holds n_head_kv K scales then n_head_kv V scales, all 1.0 without an FP8 cache,
 * or the interleaved per-(row, KV head) scales of a per-token-head cache). flags packs the causal, has_sinks, direct
 * and tree bits (common/attn_common.h NF_ATTN_FLAG_*) with n_fresh in bits 5-31 under the tree bit: the last n_fresh
 * slots of the key range take their visibility from tree_mask, one ulong per query row.
 */
#define NF_FLASH_PREFILL_KERNEL(SFX, KV_T, KV_CODE)                                          \
NF_KERNEL void LLM_FlashAttn_##SFX(                                                          \
    const int n_tokens, const int n_head, const int n_head_kv, const int d_k, const int d_v,  \
    const int bq, const int bk, const int g_tile, const int n_split, const int split_len,     \
    const int kv_start, const int kv_end,                                                    \
    const int q_row_stride, const int out_row_stride,                                        \
    const int k_row_stride, const int k_head_stride, const int v_row_stride,                  \
    const int v_head_stride,                                                                 \
    const float scale, const float softcap, const float max_bias,                            \
    const int flags, const int swa_type, const int n_swa,                                    \
    const int bias_mode, const int n_buckets, const int max_distance,                        \
    const int q_off, const int out_off,                                                      \
    NF_GLOBAL const float* q,                                                                \
    NF_GLOBAL const KV_T* k,                                                                 \
    NF_GLOBAL const KV_T* v,                                                                 \
    NF_GLOBAL const int* NF_RESTRICT q_pos,                                                  \
    NF_GLOBAL const int* NF_RESTRICT kv_pos,                                                 \
    NF_GLOBAL const float* NF_RESTRICT sinks,                                                \
    NF_GLOBAL const float* NF_RESTRICT rel_bias,                                             \
    NF_GLOBAL float* out,                                                                    \
    NF_GLOBAL float* partial,                                                                \
    NF_GLOBAL const float* NF_RESTRICT kv_scales,                                            \
    NF_GLOBAL const ulong* NF_RESTRICT tree_mask)                                             \
{                                                                                            \
    NF_LOCAL float smem[NF_FA_SMEM_FLOATS];                                                  \
    NF_LOCAL float red[NF_FA_WG];                                                            \
    NF_LOCAL float slope_s[NF_FD_MAX_G];                                                     \
    nf_fa_run(n_tokens, n_head, n_head_kv, d_k, d_v, bq, bk, g_tile, n_split, split_len,     \
              kv_start, kv_end, q_row_stride, out_row_stride, k_row_stride, k_head_stride,   \
              v_row_stride, v_head_stride, scale, softcap, max_bias,                          \
              flags & NF_ATTN_FLAG_CAUSAL, swa_type, n_swa, bias_mode, n_buckets,             \
              max_distance, flags & NF_ATTN_FLAG_SINKS, flags & NF_ATTN_FLAG_DIRECT, q_off,   \
              out_off, q, (NF_GLOBAL const uchar*)k, (NF_GLOBAL const uchar*)v, q_pos,       \
              kv_pos, sinks, rel_bias, out, partial, kv_scales, KV_CODE, KV_CODE,             \
              (flags & NF_ATTN_FLAG_V_ALIAS) != 0, -1, kv_pos, 0,                             \
              flags & NF_ATTN_FLAG_TREE, tree_mask, nf_attn_flag_fresh(flags), smem, red,     \
              slope_s);                                                                        \
}

NF_FLASH_PREFILL_KERNEL(F32, float, NF_KV_F32)
NF_FLASH_PREFILL_KERNEL(F16, NF_HALF, NF_KV_F16)
NF_FLASH_PREFILL_KERNEL(Q8_0, uchar, NF_KV_Q8_0)
NF_FLASH_PREFILL_KERNEL(BF16, ushort, NF_KV_BF16)
NF_FLASH_PREFILL_KERNEL(Q4_0, uchar, NF_KV_Q4_0)
NF_FLASH_PREFILL_KERNEL(Q4_1, uchar, NF_KV_Q4_1)
NF_FLASH_PREFILL_KERNEL(IQ4_NL, uchar, NF_KV_IQ4_NL)
NF_FLASH_PREFILL_KERNEL(Q5_0, uchar, NF_KV_Q5_0)
NF_FLASH_PREFILL_KERNEL(Q5_1, uchar, NF_KV_Q5_1)
NF_FLASH_PREFILL_KERNEL(FP8_E4M3, uchar, NF_KV_FP8_E4M3)
NF_FLASH_PREFILL_KERNEL(FP8_E5M2, uchar, NF_KV_FP8_E5M2)
NF_FLASH_PREFILL_KERNEL(FP8_PER_TOKEN_HEAD, uchar, NF_KV_FP8_PER_TOKEN_HEAD)
NF_FLASH_PREFILL_KERNEL(INT8_PER_TOKEN_HEAD, uchar, NF_KV_INT8_PER_TOKEN_HEAD)
NF_FLASH_PREFILL_KERNEL(INT4_PER_TOKEN_HEAD, uchar, NF_KV_INT4_PER_TOKEN_HEAD)

/* The vendor formats take the same arguments: their side planes and centroid table ride the kv_scales binding. */
NF_FLASH_PREFILL_KERNEL(NVFP4, uchar, NF_KV_NVFP4)
NF_FLASH_PREFILL_KERNEL(NVFP4_4OVER6, uchar, NF_KV_NVFP4_4OVER6)
NF_FLASH_PREFILL_KERNEL(TQ_K8_KEYS, uchar, NF_KV_TQ_K8_KEYS)
NF_FLASH_PREFILL_KERNEL(TQ_4BIT_KEYS, uchar, NF_KV_TQ_4BIT_KEYS)
NF_FLASH_PREFILL_KERNEL(TQ_3BIT_KEYS, uchar, NF_KV_TQ_3BIT_KEYS)
NF_FLASH_PREFILL_KERNEL(TQ_4BIT_VALUES, uchar, NF_KV_TQ_4BIT_VALUES)
NF_FLASH_PREFILL_KERNEL(TQ_3BIT_VALUES, uchar, NF_KV_TQ_3BIT_VALUES)
NF_FLASH_PREFILL_KERNEL(FP8_DS_MLA, uchar, NF_KV_FP8_DS_MLA)
NF_FLASH_PREFILL_KERNEL(NVFP4_DS_MLA, uchar, NF_KV_NVFP4_DS_MLA)

/*
 * LLM_FlashAttn_Mixed(30 scalars, 11 buffers): LLM_FlashAttn_<KV> over a K cache and a V cache of different types. The
 * arguments are those of LLM_FlashAttn_<KV> followed by kv_types (common/attn_common.h); k and v are byte buffers.
 * A malformed kv_types violates the plan and writes NaN outputs.
 */
NF_KERNEL void LLM_FlashAttn_Mixed(
    const int n_tokens, const int n_head, const int n_head_kv, const int d_k, const int d_v,
    const int bq, const int bk, const int g_tile, const int n_split, const int split_len,
    const int kv_start, const int kv_end,
    const int q_row_stride, const int out_row_stride,
    const int k_row_stride, const int k_head_stride, const int v_row_stride,
    const int v_head_stride,
    const float scale, const float softcap, const float max_bias,
    const int flags, const int swa_type, const int n_swa,
    const int bias_mode, const int n_buckets, const int max_distance,
    const int q_off, const int out_off,
    NF_GLOBAL const float* q,
    NF_GLOBAL const uchar* k,
    NF_GLOBAL const uchar* v,
    NF_GLOBAL const int* NF_RESTRICT q_pos,
    NF_GLOBAL const int* NF_RESTRICT kv_pos,
    NF_GLOBAL const float* NF_RESTRICT sinks,
    NF_GLOBAL const float* NF_RESTRICT rel_bias,
    NF_GLOBAL float* out,
    NF_GLOBAL float* partial,
    NF_GLOBAL const float* NF_RESTRICT kv_scales,
    NF_GLOBAL const ulong* NF_RESTRICT tree_mask,
    const int kv_types)
{
    NF_LOCAL float smem[NF_FA_SMEM_FLOATS];
    NF_LOCAL float red[NF_FA_WG];
    NF_LOCAL float slope_s[NF_FD_MAX_G];
    nf_fa_run(n_tokens, n_head, n_head_kv, d_k, d_v, bq, bk, g_tile, n_split, split_len,
              kv_start, kv_end, q_row_stride, out_row_stride, k_row_stride, k_head_stride,
              v_row_stride, v_head_stride, scale, softcap, max_bias,
              flags & NF_ATTN_FLAG_CAUSAL, swa_type, n_swa, bias_mode, n_buckets,
              max_distance, flags & NF_ATTN_FLAG_SINKS, flags & NF_ATTN_FLAG_DIRECT, q_off,
              out_off, q, k, v, q_pos, kv_pos, sinks, rel_bias, out, partial, kv_scales,
              nf_attn_kv_type_k(kv_types), nf_attn_kv_type_v(kv_types),
              (flags & NF_ATTN_FLAG_V_ALIAS) != 0, -1, kv_pos, 0,
              flags & NF_ATTN_FLAG_TREE, tree_mask, nf_attn_flag_fresh(flags), smem, red,
              slope_s);
}

/*
 * Paged prefill over one paginated KV segment: the caller resolves kv_start / kv_end in the segment's slots
 * and the tokens' positions come from q_pos; K, V and kv_pos are read at the physical row nf_kv_row(t)
 * (common/attn_common.h; a page id below 0 masks the keys of its pages). One launch per
 * multi-token segment, so the grid and every other argument are those of LLM_FlashAttn_<KV>.
 */

/* Plan constraints of LLM_FlashAttnPaged_*: those of LLM_FlashAttn_* plus the page geometry (page_shift is the
 * log2 page size, bounded so the physical row stays an int; pt_off is the page table's element offset; the page
 * table and kv_start / kv_end are validated on the host before the call). */
NF_INLINE int nf_fa_paged_ok(int n_head, int n_head_kv, int d_k, int d_v, int bq, int bk, int g_tile,
                             int n_split, int split_len, int direct_out, int k_type, int v_type,
                             int page_shift, int pt_off) {
    return nf_fa_plan_ok(n_head, n_head_kv, d_k, d_v, bq, bk, g_tile, n_split, split_len, direct_out, k_type,
                         v_type)
        && page_shift >= 0 && page_shift <= 20 && pt_off >= 0;
}

/*
 * LLM_FlashAttnPaged_<KV>(31 scalars, 12 buffers): the arguments are those of LLM_FlashAttn_<KV> (kv_start / kv_end
 * count slots), then page_shift, pt_off and buffer page_table. k and v may be the same buffer (MLA latent cache);
 * sinks, rel_bias and kv_scales are read-only placeholders when unused.
 * Grid: the 3-D grid of LLM_FlashAttn_<KV>, one launch per multi-token segment.
 */
#define NF_FLASH_ATTN_PAGED_KERNEL(SFX, KV_T, KV_CODE)                                        \
NF_KERNEL void LLM_FlashAttnPaged_##SFX(                                                      \
    const int n_tokens, const int n_head, const int n_head_kv, const int d_k, const int d_v,   \
    const int bq, const int bk, const int g_tile, const int n_split, const int split_len,     \
    const int kv_start, const int kv_end,                                                    \
    const int q_row_stride, const int out_row_stride,                                        \
    const int k_row_stride, const int k_head_stride, const int v_row_stride,                  \
    const int v_head_stride,                                                                 \
    const float scale, const float softcap, const float max_bias,                            \
    const int flags, const int swa_type, const int n_swa,                                     \
    const int bias_mode, const int n_buckets, const int max_distance,                         \
    const int q_off, const int out_off,                                                       \
    const int page_shift, const int pt_off,                                                  \
    NF_GLOBAL const float* q,                                                                \
    NF_GLOBAL const KV_T* k,                                                                 \
    NF_GLOBAL const KV_T* v,                                                                 \
    NF_GLOBAL const int* NF_RESTRICT q_pos,                                                  \
    NF_GLOBAL const int* NF_RESTRICT kv_pos,                                                 \
    NF_GLOBAL const float* NF_RESTRICT sinks,                                                \
    NF_GLOBAL const float* NF_RESTRICT rel_bias,                                             \
    NF_GLOBAL float* out,                                                                    \
    NF_GLOBAL float* partial,                                                                \
    NF_GLOBAL const int* NF_RESTRICT page_table,                                             \
    NF_GLOBAL const float* NF_RESTRICT kv_scales,                                            \
    NF_GLOBAL const ulong* NF_RESTRICT tree_mask)                                             \
{                                                                                             \
    NF_LOCAL float smem[NF_FA_SMEM_FLOATS];                                                   \
    NF_LOCAL float red[NF_FA_WG];                                                             \
    NF_LOCAL float slope_s[NF_FD_MAX_G];                                                      \
    if (!nf_fa_paged_ok(n_head, n_head_kv, d_k, d_v, bq, bk, g_tile, n_split, split_len,       \
                        flags & NF_ATTN_FLAG_DIRECT, KV_CODE, KV_CODE, page_shift, pt_off)) {  \
        nf_fa_poison(n_tokens, n_head, d_v, n_split, out_row_stride,                           \
                     flags & NF_ATTN_FLAG_DIRECT, out_off, out, partial);                      \
        return;   /* uniform across the workgroup: no barrier has been reached */              \
    }                                                                                         \
    nf_fa_run(n_tokens, n_head, n_head_kv, d_k, d_v, bq, bk, g_tile, n_split, split_len,      \
              kv_start, kv_end, q_row_stride, out_row_stride, k_row_stride, k_head_stride,    \
              v_row_stride, v_head_stride, scale, softcap, max_bias,                          \
              flags & NF_ATTN_FLAG_CAUSAL, swa_type, n_swa, bias_mode, n_buckets,             \
              max_distance, flags & NF_ATTN_FLAG_SINKS, flags & NF_ATTN_FLAG_DIRECT, q_off,   \
              out_off, q, (NF_GLOBAL const uchar*)k, (NF_GLOBAL const uchar*)v, q_pos,        \
              kv_pos, sinks, rel_bias, out, partial, kv_scales, KV_CODE, KV_CODE,             \
              (flags & NF_ATTN_FLAG_V_ALIAS) != 0, page_shift, page_table, pt_off,            \
              flags & NF_ATTN_FLAG_TREE, tree_mask, nf_attn_flag_fresh(flags), smem, red,     \
              slope_s);                                                                        \
}

NF_FLASH_ATTN_PAGED_KERNEL(F32, float, NF_KV_F32)
NF_FLASH_ATTN_PAGED_KERNEL(F16, NF_HALF, NF_KV_F16)
NF_FLASH_ATTN_PAGED_KERNEL(Q8_0, uchar, NF_KV_Q8_0)
NF_FLASH_ATTN_PAGED_KERNEL(BF16, ushort, NF_KV_BF16)
NF_FLASH_ATTN_PAGED_KERNEL(Q4_0, uchar, NF_KV_Q4_0)
NF_FLASH_ATTN_PAGED_KERNEL(Q4_1, uchar, NF_KV_Q4_1)
NF_FLASH_ATTN_PAGED_KERNEL(IQ4_NL, uchar, NF_KV_IQ4_NL)
NF_FLASH_ATTN_PAGED_KERNEL(Q5_0, uchar, NF_KV_Q5_0)
NF_FLASH_ATTN_PAGED_KERNEL(Q5_1, uchar, NF_KV_Q5_1)
NF_FLASH_ATTN_PAGED_KERNEL(FP8_E4M3, uchar, NF_KV_FP8_E4M3)
NF_FLASH_ATTN_PAGED_KERNEL(FP8_E5M2, uchar, NF_KV_FP8_E5M2)
NF_FLASH_ATTN_PAGED_KERNEL(FP8_PER_TOKEN_HEAD, uchar, NF_KV_FP8_PER_TOKEN_HEAD)
NF_FLASH_ATTN_PAGED_KERNEL(INT8_PER_TOKEN_HEAD, uchar, NF_KV_INT8_PER_TOKEN_HEAD)
NF_FLASH_ATTN_PAGED_KERNEL(INT4_PER_TOKEN_HEAD, uchar, NF_KV_INT4_PER_TOKEN_HEAD)

/* The vendor formats take the same arguments: their side planes and centroid table ride the kv_scales binding. */
NF_FLASH_ATTN_PAGED_KERNEL(NVFP4, uchar, NF_KV_NVFP4)
NF_FLASH_ATTN_PAGED_KERNEL(NVFP4_4OVER6, uchar, NF_KV_NVFP4_4OVER6)
NF_FLASH_ATTN_PAGED_KERNEL(TQ_K8_KEYS, uchar, NF_KV_TQ_K8_KEYS)
NF_FLASH_ATTN_PAGED_KERNEL(TQ_4BIT_KEYS, uchar, NF_KV_TQ_4BIT_KEYS)
NF_FLASH_ATTN_PAGED_KERNEL(TQ_3BIT_KEYS, uchar, NF_KV_TQ_3BIT_KEYS)
NF_FLASH_ATTN_PAGED_KERNEL(TQ_4BIT_VALUES, uchar, NF_KV_TQ_4BIT_VALUES)
NF_FLASH_ATTN_PAGED_KERNEL(TQ_3BIT_VALUES, uchar, NF_KV_TQ_3BIT_VALUES)
NF_FLASH_ATTN_PAGED_KERNEL(FP8_DS_MLA, uchar, NF_KV_FP8_DS_MLA)
NF_FLASH_ATTN_PAGED_KERNEL(NVFP4_DS_MLA, uchar, NF_KV_NVFP4_DS_MLA)

/*
 * LLM_FlashAttnPaged_Mixed(32 scalars, 12 buffers): LLM_FlashAttnPaged_<KV> over a K cache and a V cache of different
 * types. The arguments are those of LLM_FlashAttnPaged_<KV> followed by kv_types (common/attn_common.h) after the
 * buffers; k and v are byte buffers. A malformed kv_types violates the plan and writes NaN outputs.
 */
NF_KERNEL void LLM_FlashAttnPaged_Mixed(
    const int n_tokens, const int n_head, const int n_head_kv, const int d_k, const int d_v,
    const int bq, const int bk, const int g_tile, const int n_split, const int split_len,
    const int kv_start, const int kv_end,
    const int q_row_stride, const int out_row_stride,
    const int k_row_stride, const int k_head_stride, const int v_row_stride,
    const int v_head_stride,
    const float scale, const float softcap, const float max_bias,
    const int flags, const int swa_type, const int n_swa,
    const int bias_mode, const int n_buckets, const int max_distance,
    const int q_off, const int out_off,
    const int page_shift, const int pt_off,
    NF_GLOBAL const float* q,
    NF_GLOBAL const uchar* k,
    NF_GLOBAL const uchar* v,
    NF_GLOBAL const int* NF_RESTRICT q_pos,
    NF_GLOBAL const int* NF_RESTRICT kv_pos,
    NF_GLOBAL const float* NF_RESTRICT sinks,
    NF_GLOBAL const float* NF_RESTRICT rel_bias,
    NF_GLOBAL float* out,
    NF_GLOBAL float* partial,
    NF_GLOBAL const int* NF_RESTRICT page_table,
    NF_GLOBAL const float* NF_RESTRICT kv_scales,
    NF_GLOBAL const ulong* NF_RESTRICT tree_mask,
    const int kv_types)
{
    NF_LOCAL float smem[NF_FA_SMEM_FLOATS];
    NF_LOCAL float red[NF_FA_WG];
    NF_LOCAL float slope_s[NF_FD_MAX_G];
    if (!nf_fa_paged_ok(n_head, n_head_kv, d_k, d_v, bq, bk, g_tile, n_split, split_len,
                        flags & NF_ATTN_FLAG_DIRECT, nf_attn_kv_type_k(kv_types),
                        nf_attn_kv_type_v(kv_types), page_shift, pt_off)) {
        nf_fa_poison(n_tokens, n_head, d_v, n_split, out_row_stride,
                     flags & NF_ATTN_FLAG_DIRECT, out_off, out, partial);
        return;   /* uniform across the workgroup: no barrier has been reached */
    }
    nf_fa_run(n_tokens, n_head, n_head_kv, d_k, d_v, bq, bk, g_tile, n_split, split_len,
              kv_start, kv_end, q_row_stride, out_row_stride, k_row_stride, k_head_stride,
              v_row_stride, v_head_stride, scale, softcap, max_bias,
              flags & NF_ATTN_FLAG_CAUSAL, swa_type, n_swa, bias_mode, n_buckets,
              max_distance, flags & NF_ATTN_FLAG_SINKS, flags & NF_ATTN_FLAG_DIRECT, q_off,
              out_off, q, k, v, q_pos, kv_pos, sinks, rel_bias, out, partial, kv_scales,
              nf_attn_kv_type_k(kv_types), nf_attn_kv_type_v(kv_types),
              (flags & NF_ATTN_FLAG_V_ALIAS) != 0,
              page_shift, page_table, pt_off,
              flags & NF_ATTN_FLAG_TREE, tree_mask, nf_attn_flag_fresh(flags), smem, red,
              slope_s);
}

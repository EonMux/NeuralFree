/* flash_prefill_sg.cl - LLM_FlashAttnSg_<KV> and LLM_FlashAttnPagedSg_<KV> subgroup prefill attention (OpenCL only). Requires: common/dialect.h, common/config.h, common/math.h, common/kv_types.h, common/attn_common.h, llm/flash_prefill.cl, llm/flash_decode_sg.cl */
/*
 * The prefill attention of LLM_FlashAttn_<KV> / LLM_FlashAttnPaged_<KV> (flash_prefill.cl) for plain F32, F16 and
 * BF16 caches, with the same argument lists, layouts, masking, biases, sinks, tree masks and partial outputs, under a
 * work mapping built for the Intel Xe GPUs (no local-memory tiles, so no residency limit and no barrier in the key
 * loop):
 *
 * - A work-group is NF_FAS_SUBGROUPS subgroups of NF_SG_ROW_SIZE lanes and owns R = bq * g_tile == NF_FAS_WG_ROWS
 *   query rows of one KV head (r = i * g_tile + g: token t0 + i, head h0 + g, as in flash_prefill.cl); subgroup s
 *   serves rows 2s and 2s + 1, so with g_tile >= 2 a subgroup's two rows are one token's two heads and share every
 *   K and V load and the visibility.
 * - The subgroup scans its key range in blocks of NF_SG_ROW_SIZE keys. Scores: lane j takes key j of the block and
 *   computes both rows' full dot products itself (the query rows are read as uniform values from local memory), so a
 *   score needs no cross-lane reduction. Softmax: one subgroup max and one subgroup sum per row and block. Values:
 *   lane l holds the NF_FDS_DPL output dims [l * NF_FDS_DPL, +NF_FDS_DPL) of both rows and accumulates each visible
 *   key's V row (one vector load per lane) weighted by the key's broadcast probabilities.
 * - A block in which no key is visible to either row costs its visibility test only (causal and window blocks past
 *   the rows' ranges).
 *
 * Grid: global {ceil(n_tokens / bq) * NF_FAS_WG, n_head_kv * ceil(G / g_tile), n_split}, local {NF_FAS_WG, 1, 1}.
 * Plan constraints (FlashAttnTiling's subgroup prefill form; a violation writes NaN outputs): g_tile in {1, 2, 4, 8}
 * with bq * g_tile == NF_FAS_WG_ROWS, d_k == d_v a positive multiple of NF_FDS_DPL of at most NF_FDS_MAX_D,
 * K and V row and head strides multiples of NF_FDS_DPL, n_split >= 1, split_len > 0, direct_out only with n_split == 1, the work-group and subgroup sizes above. Results
 * are deterministic (fixed key order within a row).
 */
#if defined(USE_SUBGROUPS) && defined(cl_intel_required_subgroup_size) && defined(NF_SG_ROW_SIZE)

#define NF_FAS_ATTR __attribute__((intel_reqd_sub_group_size(NF_SG_ROW_SIZE)))
/* Subgroups per work-group and query rows per subgroup (the host mirrors both). */
#define NF_FAS_SUBGROUPS 4
#define NF_FAS_ROWS 2
#define NF_FAS_WG (NF_FAS_SUBGROUPS * NF_SG_ROW_SIZE)
#define NF_FAS_WG_ROWS (NF_FAS_SUBGROUPS * NF_FAS_ROWS)
/* The work-group's query rows staged for the uniform reads of the score loop. */
#define NF_FAS_Q_FLOATS (NF_FAS_WG_ROWS * NF_FDS_MAX_D)

NF_INLINE int nf_fas_plan_ok(int n_head, int n_head_kv, int d_k, int d_v, int bq, int g_tile, int n_split,
                             int split_len, int direct_out, int k_row_stride, int k_head_stride, int v_row_stride,
                             int v_head_stride) {
    return nf_fds_strides_ok(k_row_stride, k_head_stride, v_row_stride, v_head_stride)
        && n_head > 0 && n_head_kv > 0 && n_head % n_head_kv == 0
        && (g_tile == 1 || g_tile == 2 || g_tile == 4 || g_tile == 8) && bq * g_tile == NF_FAS_WG_ROWS
        && d_k == d_v && d_k > 0 && d_k % NF_FDS_DPL == 0 && d_k <= NF_FDS_MAX_D
        && n_split >= 1 && split_len > 0
        && (direct_out == 0 || n_split == 1)
        && (int)get_local_size(0) == NF_FAS_WG
        && (int)get_max_sub_group_size() == NF_SG_ROW_SIZE;
}

/*
 * The shared prefill body. page_shift < 0 is the identity row map of the un-paged entry point (pt unused); the
 * paged entry point passes its page table and the segment's offset in it, as nf_fa_run takes them.
 */
NF_INLINE void nf_fas_run(
    const int n_tokens, const int n_head, const int n_head_kv, const int d,
    const int bq, const int g_tile, const int n_split, const int split_len,
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
    NF_GLOBAL float* out, NF_GLOBAL float* partial, NF_GLOBAL const float* kv_scales, const int kv_type,
    const int page_shift, NF_GLOBAL const int* pt, const int pt_off,
    const int tree, NF_GLOBAL const ulong* tree_mask, const int n_fresh,
    NF_LOCAL_PTR float* qs)
{
    const int lane = (int)get_sub_group_local_id();
    const int sg = (int)get_sub_group_id();
    const int split = (int)get_group_id(2);
    const int G = n_head / n_head_kv;
    const int n_gt = (G + g_tile - 1) / g_tile;
    const int kvh = (int)get_group_id(1) / n_gt;
    const int gt = (int)get_group_id(1) - kvh * n_gt;
    const int h0 = kvh * G + gt * g_tile;
    const int g_cnt = min(g_tile, G - gt * g_tile);
    const int t0 = (int)get_group_id(0) * bq;
    const int k_begin = kv_start + split * split_len;
    const int k_stop = min(k_begin + split_len, kv_end);

    /* Stage the work-group's query rows (zeros for rows past the tokens or the head tile). */
    for (int idx = (int)get_local_id(0); idx < NF_FAS_WG_ROWS * d; idx += NF_FAS_WG) {
        const int r = idx / d;
        const int c = idx - r * d;
        const int i = r / g_tile;
        const int g = r - i * g_tile;
        const int t = t0 + i;
        qs[r * NF_FDS_MAX_D + c] = t < n_tokens && g < g_cnt
            ? q[(nf_i64)q_off + (nf_i64)t * q_row_stride + (nf_i64)(h0 + g) * d + c] : 0.0f;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    /* This subgroup's two rows. */
    const int ra = sg * NF_FAS_ROWS;
    const int rb = ra + 1;
    const int ia = ra / g_tile;
    const int ga = ra - ia * g_tile;
    const int ib = rb / g_tile;
    const int gb = rb - ib * g_tile;
    const int ta = t0 + ia;
    const int tb = t0 + ib;
    const int va = ta < n_tokens && ga < g_cnt;
    const int vb = tb < n_tokens && gb < g_cnt;
    if (!va && !vb) {
        return;   /* no barrier follows */
    }
    const int ha = h0 + ga;
    const int hb = h0 + gb;
    const int pqa = va ? q_pos[ta] : 0;
    const int pqb = vb ? q_pos[tb] : 0;
    const ulong tma = tree && va ? tree_mask[ta] : (ulong)0;
    const ulong tmb = tree && vb ? tree_mask[tb] : (ulong)0;
    const int alibi = bias_mode == NF_ATTN_BIAS_ALIBI || bias_mode == NF_ATTN_BIAS_ALIBI_SQRT;
    const float slope_a = alibi ? nf_alibi_slope(ha, n_head, max_bias) : 1.0f;
    const float slope_b = alibi ? nf_alibi_slope(hb, n_head, max_bias) : 1.0f;
    const float ks = nf_kv_head_scale(kv_type, kv_scales, 0, kvh, n_head_kv, 0);
    const float vs = nf_kv_head_scale(kv_type, kv_scales, 0, kvh, n_head_kv, 1);
    const int esize = nf_fds_esize(kv_type);
    NF_GLOBAL const uchar* k_base = k + (nf_i64)kvh * k_head_stride * esize;
    NF_GLOBAL const uchar* v_base = v + (nf_i64)kvh * v_head_stride * esize;
    const nf_i64 k_row_bytes = (nf_i64)k_row_stride * esize;
    const nf_i64 v_row_bytes = (nf_i64)v_row_stride * esize;
    NF_LOCAL_PTR const float* qa = qs + ra * NF_FDS_MAX_D;
    NF_LOCAL_PTR const float* qb = qs + rb * NF_FDS_MAX_D;
    const int dim0 = lane * NF_FDS_DPL;
    const int live = dim0 < d;

    float m_a = -NF_INF, l_a = 0.0f, m_b = -NF_INF, l_b = 0.0f;
    float8 acc_a = (float8)(0.0f);
    float8 acc_b = (float8)(0.0f);

    for (int blk = k_begin; blk < k_stop; blk += NF_SG_ROW_SIZE) {
        /* Lane j: key blk + j, its row, position and visibility to both rows. */
        const int slot = blk + lane;
        int row_l = 0;
        int vis_a = 0;
        int vis_b = 0;
        int p_k = -1;
        if (slot < k_stop && nf_kv_page_ok(slot, page_shift, pt, pt_off)) {
            row_l = nf_kv_row(slot, page_shift, pt, pt_off);
            p_k = kv_pos[row_l];
            vis_a = va && nf_attn_tree_visible(p_k, tma, kv_end, n_fresh, pqa, causal, swa_type, n_swa);
            vis_b = vb && nf_attn_tree_visible(p_k, tmb, kv_end, n_fresh, pqb, causal, swa_type, n_swa);
        }
        const uint vis_mask = sub_group_reduce_add((vis_a | vis_b) ? (1u << lane) : 0u);
        if (vis_mask == 0u) {
            continue;
        }

        /* Scores: the lane's key against both rows, full dot products in the lane. */
        float sa = 0.0f;
        float sb = 0.0f;
        if (vis_a | vis_b) {
            NF_GLOBAL const uchar* krow = k_base + row_l * k_row_bytes;
            for (int c = 0; c < d; c += NF_FDS_DPL) {
                const float8 k8 = nf_fds_load(krow, c, kv_type);
                sa += nf_fds_dot(vload8(0, qa + c), k8);
                sb += nf_fds_dot(vload8(0, qb + c), k8);
            }
        }
        sa = vis_a ? nf_attn_score(sa * ks, scale, softcap)
                     + nf_attn_bias(bias_mode, ha, n_head, pqa, p_k, slope_a, rel_bias, n_buckets, max_distance, ta)
                   : -NF_INF;
        sb = vis_b ? nf_attn_score(sb * ks, scale, softcap)
                     + nf_attn_bias(bias_mode, hb, n_head, pqb, p_k, slope_b, rel_bias, n_buckets, max_distance, tb)
                   : -NF_INF;

        /* Online softmax per row over the block. */
        const float mn_a = fmax(m_a, sub_group_reduce_max(sa));
        const float mn_b = fmax(m_b, sub_group_reduce_max(sb));
        const float al_a = nf_attn_rescale(m_a, mn_a);
        const float al_b = nf_attn_rescale(m_b, mn_b);
        const float pa = nf_attn_rescale(sa, mn_a);
        const float pb = nf_attn_rescale(sb, mn_b);
        l_a = l_a * al_a + sub_group_reduce_add(pa);
        l_b = l_b * al_b + sub_group_reduce_add(pb);
        m_a = mn_a;
        m_b = mn_b;
        acc_a *= al_a;
        acc_b *= al_b;

        /* Values: every visible key's V row, weighted by its broadcast probabilities. */
        uint todo = vis_mask;
        while (todo != 0u) {
            const int j = nf_fds_first(todo);
            todo &= todo - 1u;
            const int row = sub_group_broadcast(row_l, (uint)j);
            const float wa = sub_group_broadcast(pa, (uint)j) * vs;
            const float wb = sub_group_broadcast(pb, (uint)j) * vs;
            if (live) {
                const float8 v8 = nf_fds_load(v_base + row * v_row_bytes, dim0, kv_type);
                acc_a += wa * v8;
                acc_b += wb * v8;
            }
        }
    }

    /* Output: the normalized rows (direct_out) or their (o, m, l) partials. */
    for (int s = 0; s < NF_FAS_ROWS; s++) {
        const int valid = s == 0 ? va : vb;
        if (!valid) {
            continue;
        }
        const int t = s == 0 ? ta : tb;
        const int h = s == 0 ? ha : hb;
        const float m = s == 0 ? m_a : m_b;
        const float l = s == 0 ? l_a : l_b;
        const float8 o = s == 0 ? acc_a : acc_b;
        if (direct_out) {
            const float2 fl = has_sinks ? nf_attn_sink(m, l, sinks[h]) : NF_F2(1.0f, l);
            if (live) {
                vstore8(fl.y > 0.0f ? o * (fl.x / fl.y) : (float8)(0.0f), 0,
                        out + (nf_i64)out_off + (nf_i64)t * out_row_stride + (nf_i64)h * d + dim0);
            }
        } else {
            NF_GLOBAL float* pp = partial + (((nf_i64)t * n_head + h) * n_split + split) * (d + 2);
            if (live) {
                vstore8(o, 0, pp + dim0);
            }
            if (lane == 0) {
                pp[d] = m;
                pp[d + 1] = l;
            }
        }
    }
}

/* LLM_FlashAttnSg_<KV>: the arguments of LLM_FlashAttn_<KV> (flash_prefill.cl); bk is not read. */
#define NF_FLASH_PREFILL_SG_KERNEL(SFX, KV_T, KV_CODE)                                        \
NF_FAS_ATTR NF_KERNEL void LLM_FlashAttnSg_##SFX(                                             \
    const int n_tokens, const int n_head, const int n_head_kv, const int d_k, const int d_v,   \
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
    NF_GLOBAL const ulong* NF_RESTRICT tree_mask)                                            \
{                                                                                            \
    NF_LOCAL float qs[NF_FAS_Q_FLOATS];                                                      \
    if (!nf_fas_plan_ok(n_head, n_head_kv, d_k, d_v, bq, g_tile, n_split, split_len,          \
                        flags & NF_ATTN_FLAG_DIRECT, k_row_stride, k_head_stride,            \
                        v_row_stride, v_head_stride)) {                                      \
        nf_fa_poison(n_tokens, n_head, d_v, n_split, out_row_stride,                          \
                     flags & NF_ATTN_FLAG_DIRECT, out_off, out, partial);                     \
        return;   /* uniform across the work-group: no barrier has been reached */            \
    }                                                                                        \
    nf_fas_run(n_tokens, n_head, n_head_kv, d_k, bq, g_tile, n_split, split_len,              \
               kv_start, kv_end, q_row_stride, out_row_stride, k_row_stride, k_head_stride,  \
               v_row_stride, v_head_stride, scale, softcap, max_bias,                         \
               flags & NF_ATTN_FLAG_CAUSAL, swa_type, n_swa, bias_mode, n_buckets,            \
               max_distance, flags & NF_ATTN_FLAG_SINKS, flags & NF_ATTN_FLAG_DIRECT, q_off,  \
               out_off, q, (NF_GLOBAL const uchar*)k, (NF_GLOBAL const uchar*)v, q_pos,       \
               kv_pos, sinks, rel_bias, out, partial, kv_scales, KV_CODE, -1, kv_pos, 0,       \
               flags & NF_ATTN_FLAG_TREE, tree_mask, nf_attn_flag_fresh(flags), qs);          \
}

/* LLM_FlashAttnPagedSg_<KV>: the arguments of LLM_FlashAttnPaged_<KV> (flash_prefill.cl); bk is not read. */
#define NF_FLASH_ATTN_PAGED_SG_KERNEL(SFX, KV_T, KV_CODE)                                     \
NF_FAS_ATTR NF_KERNEL void LLM_FlashAttnPagedSg_##SFX(                                        \
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
    NF_LOCAL float qs[NF_FAS_Q_FLOATS];                                                       \
    if (!(page_shift >= 0 && page_shift <= 20 && pt_off >= 0                                  \
          && nf_fas_plan_ok(n_head, n_head_kv, d_k, d_v, bq, g_tile, n_split, split_len,      \
                            flags & NF_ATTN_FLAG_DIRECT, k_row_stride, k_head_stride,         \
                            v_row_stride, v_head_stride))) {                                  \
        nf_fa_poison(n_tokens, n_head, d_v, n_split, out_row_stride,                           \
                     flags & NF_ATTN_FLAG_DIRECT, out_off, out, partial);                      \
        return;   /* uniform across the work-group: no barrier has been reached */             \
    }                                                                                         \
    nf_fas_run(n_tokens, n_head, n_head_kv, d_k, bq, g_tile, n_split, split_len,               \
               kv_start, kv_end, q_row_stride, out_row_stride, k_row_stride, k_head_stride,   \
               v_row_stride, v_head_stride, scale, softcap, max_bias,                          \
               flags & NF_ATTN_FLAG_CAUSAL, swa_type, n_swa, bias_mode, n_buckets,             \
               max_distance, flags & NF_ATTN_FLAG_SINKS, flags & NF_ATTN_FLAG_DIRECT, q_off,   \
               out_off, q, (NF_GLOBAL const uchar*)k, (NF_GLOBAL const uchar*)v, q_pos,        \
               kv_pos, sinks, rel_bias, out, partial, kv_scales, KV_CODE, page_shift,          \
               page_table, pt_off, flags & NF_ATTN_FLAG_TREE, tree_mask,                       \
               nf_attn_flag_fresh(flags), qs);                                                \
}

NF_FLASH_PREFILL_SG_KERNEL(F32, float, NF_KV_F32)
NF_FLASH_PREFILL_SG_KERNEL(F16, NF_HALF, NF_KV_F16)
NF_FLASH_PREFILL_SG_KERNEL(BF16, ushort, NF_KV_BF16)
NF_FLASH_ATTN_PAGED_SG_KERNEL(F32, float, NF_KV_F32)
NF_FLASH_ATTN_PAGED_SG_KERNEL(F16, NF_HALF, NF_KV_F16)
NF_FLASH_ATTN_PAGED_SG_KERNEL(BF16, ushort, NF_KV_BF16)

#endif /* USE_SUBGROUPS && cl_intel_required_subgroup_size && NF_SG_ROW_SIZE */

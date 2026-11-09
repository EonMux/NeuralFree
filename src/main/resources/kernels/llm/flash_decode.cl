/* flash_decode.cl - LLM_FlashDecode_<KV>, LLM_FlashDecode_Mixed (identity pages) and LLM_FlashDecodePaged_<KV>,
 * LLM_FlashDecodePaged_Mixed (paged) split-K decode attention. Requires: common/dialect.h, common/config.h,
 * common/reduce.h, common/math.h, common/kv_types.h, common/attn_common.h */
/*
 * Attention of one query token against a KV cache (decode, speculative per-row verify, MLA latent
 * cache, T5 cross-attention) with GQA packing and an optional split over the key range.
 * GLSL twin: shaders/llm/flash_decode.comp. Java: gguf.ops.attn.AttnKernels (arg indices),
 * gguf.ops.attn.FlashAttnTiling (g_tile, split_len, grid), gguf.ops.attn.CpuAttentionCore (reference).
 *
 * Layouts (all offsets in elements, 64-bit inside the kernel):
 *   q        q[q_off + h * d_k + d]
 *   K value  k[t * k_row_stride + kvh * k_head_stride + d]   (same for v with the v strides); KV rows
 *            hold one position each, kv_pos[t] is that position (-1 = empty cell); MLA latent caches
 *            bind one buffer to k and v with v_row_stride == k_row_stride
 *   out      out[out_off + h * d_v + d]                       (direct_out = 1, n_split == 1)
 *   partial  partial[(h * n_split + s) * (d_v + 2) + d], then m at + d_v and l at + d_v + 1
 *            (direct_out = 0; merged by LLM_FlashAttn_Combine with rows = heads)
 * The paged entry points (below) address K, V and kv_pos at the physical row nf_kv_row(t, page_shift, pt, pt_off)
 * (common/attn_common.h): the un-paged entry points pass page_shift = -1, the identity map, so every index above
 * is exactly row t for them.
 * Query head h reads KV head kvh = h / (n_head / n_head_kv) (ggml broadcast rk2 = neq2 / nek2).
 * K is read as k_type and V as v_type: both are the variant's type in LLM_FlashDecode_<KV>; LLM_FlashDecode_Mixed
 * takes them from its trailing kv_types argument (common/attn_common.h), as ggml's flash attention reads K and V of
 * different cache types (llama.cpp ggml-cpu/ops.cpp:9007-9013 reads each through its own type traits).
 *
 * Grid: global {n_split * 128, n_head_kv * ceil(G / g_tile)}, local {128, 1}, G = n_head / n_head_kv.
 * Workgroup (s, y) handles keys [kv_start + s * split_len, min(+ split_len, kv_end)) for the
 * g_tile query heads h0 = kvh * G + gt * g_tile .. of KV head kvh = y / ceil(G/g_tile), gt = y % ceil(G/g_tile).
 *
 * Shared memory (NF_FA_SMEM_FLOATS floats): q_s[g_tile * d_k] | s_s[g_tile * split_len]
 * | o_s[NF_FA_LG * g_tile * d_v] pass-2 lane-group partials | the shuffle fast path's row staging
 * (2 * split_len ints: the split's physical rows and their kv_pos, page_shift >= 0 or not).
 * Plan constraints (FlashAttnTiling.decode guarantees them; a violation writes NaN outputs):
 *   g_tile <= NF_FD_MAX_G, g_tile * (d_k + split_len) + NF_FA_LG * g_tile * d_v
 *   + (the fast path's 2 * split_len staging) <= NF_FA_SMEM_FLOATS,
 *   g_tile * d_v <= (NF_FA_WG / NF_FA_LG) * NF_FA_ACC_MAX, d_k % 4 == 0, d_v % 4 == 0, local size == NF_FA_WG,
 *   direct_out only with n_split == 1, k_type and v_type KV cache type codes.
 *
 * Algorithm (no per-key barriers):
 *   1. q of the g_tile heads is staged in q_s; ALiBi slopes per head in stats.
 *   2. Pass 1: work-item j scores keys j, j + 128, ... of the split; one K row read feeds the g_tile
 *      dot products; masked keys score -INF without touching K.
 *      NF_HAS_SG_SHUFFLE fast path (plain F32/F16/BF16 caches, no row-relative bias table, split_len
 *      <= NF_FA_WG, g_tile <= 4): the workgroup first stages the split's rows cooperatively — the
 *      page id -> physical row -> kv_pos chain resolved for every key in parallel instead of once per
 *      serial scan step — then warp w scans the 32 keys [w * 32, +32) of the split, each lane
 *      holding one float4 of the key's K row (one coalesced row read per key, the only dependent
 *      load of the step) and the per-head dot is
 *      reduced with the subgroup butterfly (nf_seg_sum at width 32, register form); lane 0 of the warp
 *      stores the score.
 *   3. Per head: m = workgroup max, p = e^(s - m) in place, l = workgroup sum.
 *   4. Pass 2: the workgroup's NF_FA_WG/32 lane groups own interleaved keys (j = lane group, step
 *      NF_FA_WG/32); each accumulates every o[g][d] = sum_t p[g][t] * V[t][d] pair over its keys with V
 *      read coalesced along d (lane l owning pairs l, l + 32, ...) and stages its accumulators in o_s; each
 *      work-item then sums the staged partials of the pairs it writes. A row wider than
 *      32 * NF_FA_ACC_MAX pairs runs as one group over every key. Keys with p == 0 are not read.
 *      The fast path takes the same lane-group keys with float4 accumulators: lane l holds dims
 *      l * 4 of every o[g] and one coalesced V row float4 per key, staged to the same o_s slots.
 *   5. direct_out: sinks merged (nf_attn_sink) and o / l written; l == 0 (fully masked row, no sink)
 *      writes 0. Otherwise (o, m, l) go to partial.
 * Subgroups: reductions use nf_wg_max / nf_wg_sum (common/reduce.h); local size 128 is a multiple of 32
 * and of every subgroup size, as the subgroup reductions require.
 */

/* Plan constraints of LLM_FlashDecode_*; see the header comment. stage_floats is the fast path's row
 * staging (2 * split_len when it runs, else 0), which rides the same shared budget: the decode plans
 * FlashAttnTiling generates stage one query head per workgroup (g_tile 1), whose tile plus the
 * staging stays under the 3072-float minimum budget at the fast path's head-dim and split bounds.
 */
NF_INLINE int nf_fd_plan_ok(int n_head, int n_head_kv, int d_k, int d_v, int g_tile, int n_split,
                            int split_len, int direct_out, int k_type, int v_type, int stage_floats) {
    return nf_attn_kv_type_ok(k_type) && nf_attn_kv_type_ok(v_type)
        && n_head > 0 && n_head_kv > 0 && n_head % n_head_kv == 0
        && g_tile >= 1 && g_tile <= NF_FD_MAX_G
        && d_k > 0 && d_v > 0 && (d_k & 3) == 0 && (d_v & 3) == 0
        && n_split >= 1 && split_len > 0
        && g_tile * (d_k + split_len) + NF_FA_LG * g_tile * d_v + stage_floats <= NF_FA_SMEM_FLOATS
        && g_tile * d_v <= (NF_FA_WG / NF_FA_LG) * NF_FA_ACC_MAX
        && (direct_out == 0 || n_split == 1)
        && (int)get_local_size(0) == NF_FA_WG;
}

#ifdef NF_HAS_SG_SHUFFLE
/*
 * The warp fast path's gate (NF_HAS_SG_SHUFFLE backends only; elsewhere the scalar body runs and the
 * plan below always generates its shape): one query head's K row is read as one coalesced float4 per
 * lane (the row spans exactly one warp instruction) and its dot is reduced with the subgroup butterfly,
 * so pass 1 needs no local-memory tree per key; pass 2 accumulates V the same coalesced way. Plain
 * cache element types only (F32/F16/BF16: the loaders are direct vector loads, no side planes or
 * centroids), no row-relative bias table, head dims of at most NF_FA_FAST_DIM (32 lanes cover exactly
 * 32 * 4 dims; a wider row would need the lanes to loop and only the scalar body is measured there),
 * and the split no longer than the workgroup (each warp scans 32 keys of it). d_k and d_v are
 * multiples of 4 by the plan; lanes past d_k/4 or d_v/4 contribute zeros.
 *
 * A step over at most one workgroup of keys reaches the fast path only through the plan's route: the
 * mixer routes those steps to the tensor-core prefill when the plan has one (paired CUDA benches over
 * 128-token runs put the prefill route 3-7% ahead of the decode route on every model tried), so the
 * scalar body below serves short steps only where no prefill kernel exists or the caches refuse the
 * fast path. The fast body measured faster than the scalar body at every depth on the decode route
 * (paired 0.6B Q8_0 runs: 337 t/s against 311 t/s over 128-token steps), so no key-count gate here.
 */
NF_INLINE int nf_fd_fast_ok(int k_type, int v_type, int d_k, int d_v, int split_len,
                            int n_buckets, int g_tile) {
    return d_k <= NF_FA_FAST_DIM && d_v <= NF_FA_FAST_DIM
        && split_len <= NF_FA_WG && n_buckets == 0 && g_tile <= NF_FA_LG
        && (k_type == NF_KV_F32 || k_type == NF_KV_F16 || k_type == NF_KV_BF16)
        && (v_type == NF_KV_F32 || v_type == NF_KV_F16 || v_type == NF_KV_BF16);
}
#endif /* NF_HAS_SG_SHUFFLE */

/* Writes NaN over the outputs of a call whose arguments violate the plan (workgroup (0, 0) only). */
NF_INLINE void nf_fd_poison(int n_head, int d_v, int n_split, int direct_out, int out_off,
                            NF_GLOBAL float* out, NF_GLOBAL float* partial) {
    if (get_group_id(0) != 0 || get_group_id(1) != 0 || n_head <= 0 || d_v <= 0 || n_split <= 0) {
        return;
    }
    const int lid = (int)get_local_id(0);
    const int lsz = (int)get_local_size(0);
    if (direct_out) {
        for (int i = lid; i < n_head * d_v; i += lsz) {
            out[(nf_i64)out_off + i] = NF_NAN;
        }
    } else {
        const nf_i64 n = (nf_i64)n_head * n_split * (d_v + 2);
        for (nf_i64 i = lid; i < n; i += lsz) {
            partial[i] = NF_NAN;
        }
    }
}

/*
 * The shared decode algorithm. The paged entry points pass page_shift >= 0, their page table (pt already offset to
 * the sequence's block, pt_off of nf_kv_row) and seq >= 0 (their sequence): K, V and kv_pos are then read at the
 * physical row nf_kv_row(t) (common/attn_common.h), page ids below 0 are masked, the sequence's q / out block
 * (q_off += seq * q_seq_stride, out_off += seq * out_seq_stride, resolved by the entry point) and partial rows at
 * (seq * n_head + h) are used, and the sequence's fold in front of grid dimension 1 is subtracted before the
 * KV head index. The un-paged entry points pass page_shift = -1 and seq = -1: identity rows, no grid change, no
 * partial offset, and pt (a placeholder, kv_pos) is never dereferenced.
 *
 * kv_scales holds the FP8 KV-cache scales as n_head_kv K values followed by n_head_kv V values (each the checkpoint
 * kv_scale of one KV head, 1.0 without an FP8 cache): the dequantized K of KV head kvh is fp8(code) * kv_scales[kvh]
 * and its V fp8(code) * kv_scales[n_head_kv + kvh], which the kernel applies per loaded element, as vLLM's flash
 * attention does for a scaled FP8 cache (quant_utils.cuh scaled_vec_conversion: fp8 * scale). A per-token-head cache
 * stores its per-(token, KV head) scales in the same buffer, interleaved by cache row; the kernels read them with
 * nf_kv_head_scale of common/kv_types.h.
 */
NF_INLINE void nf_fd_run(
    const int n_head, const int n_head_kv, const int d_k, const int d_v, const int g_tile,
    const int n_split, const int split_len, const int kv_start, const int kv_end, const int q_pos,
    const int k_row_stride, const int k_head_stride, const int v_row_stride, const int v_head_stride,
    const float scale, const float softcap, const float max_bias,
    const int causal, const int swa_type, const int n_swa,
    const int bias_mode, const int n_buckets, const int max_distance,
    const int has_sinks, const int direct_out, const int q_off, const int out_off,
    NF_GLOBAL const float* q, NF_GLOBAL const uchar* k, NF_GLOBAL const uchar* v,
    NF_GLOBAL const int* kv_pos, NF_GLOBAL const float* sinks, NF_GLOBAL const float* rel_bias,
    NF_GLOBAL float* out, NF_GLOBAL float* partial, NF_GLOBAL const float* kv_scales,
    const int k_type, const int v_type, const int v_aliases_k,
    const int page_shift, NF_GLOBAL const int* pt, const int pt_off, const int seq,
    const int tree, NF_GLOBAL const ulong* tree_mask, const int n_fresh,
    NF_LOCAL_PTR float* smem, NF_LOCAL_PTR float* red, NF_LOCAL_PTR float* stats)
{
    /* The fast path (both passes) runs wherever the gate says; on it the row staging below runs too,
     * so the plan check budgets it. Backends without the subgroup shuffle never stage. */
#ifdef NF_HAS_SG_SHUFFLE
    const int fd_fast = nf_fd_fast_ok(k_type, v_type, d_k, d_v, split_len, n_buckets, g_tile);
#else
    const int fd_fast = 0;
#endif
    if (!nf_fd_plan_ok(n_head, n_head_kv, d_k, d_v, g_tile, n_split, split_len, direct_out, k_type, v_type,
                       fd_fast ? 2 * split_len : 0)) {
        nf_fd_poison(n_head, d_v, n_split, direct_out, out_off, out, partial);
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
    const int split = (int)get_group_id(0);
    const int G = n_head / n_head_kv;
    const int n_gt = (G + g_tile - 1) / g_tile;
    int y = (int)get_group_id(1);
    if (seq > 0) {
        y -= seq * (n_head_kv * n_gt);   /* the paged grid folds the sequence in front of the KV head */
    }
    const int kvh = y / n_gt;
    const int gt = y - kvh * n_gt;
    const int h0 = kvh * G + gt * g_tile;
    const int g_cnt = min(g_tile, G - gt * g_tile);
    const int k_begin = kv_start + split * split_len;
    const int k_stop = split < n_split ? min(k_begin + split_len, kv_end) : k_begin;
    const int n_keys = max(0, k_stop - k_begin);

    NF_LOCAL_PTR float* q_s = smem;
    NF_LOCAL_PTR float* s_s = smem + g_tile * d_k;
    NF_LOCAL_PTR float* slope_s = stats + 2 * NF_FD_MAX_G;

    /* 1. stage q and the per-head ALiBi slopes */
    for (int i = lid; i < g_tile * d_k; i += NF_FA_WG) {
        const int g = i / d_k;
        q_s[i] = g < g_cnt ? q[(nf_i64)q_off + (nf_i64)(h0 + g) * d_k + (i - g * d_k)] : 0.0f;
    }
    if (lid < g_tile) {
        slope_s[lid] = ((bias_mode == NF_ATTN_BIAS_ALIBI || bias_mode == NF_ATTN_BIAS_ALIBI_SQRT) && lid < g_cnt)
                       ? nf_alibi_slope(h0 + lid, n_head, max_bias) : 1.0f;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

#ifdef NF_HAS_SG_SHUFFLE
    /* 1b. stage the fast path's rows: the split's physical rows and their kv_pos, resolved by the
     * whole workgroup at once (each work-item one key's page id -> row -> position chain in parallel)
     * instead of once per serial scan step of a warp below. A masked page stages row -1, which every
     * reader treats as the absent key the page test used to gate; the identity path (page_shift < 0)
     * stages its dense rows and positions the same way. */
    NF_LOCAL_PTR int* fd_rows = (NF_LOCAL_PTR int*)(smem + g_tile * d_k + g_tile * split_len
                                                   + NF_FA_LG * g_tile * d_v);
    NF_LOCAL_PTR int* fd_pos = fd_rows + split_len;
    if (fd_fast) {
        for (int j = lid; j < n_keys; j += NF_FA_WG) {
            if (nf_kv_page_ok(k_begin + j, page_shift, pt, pt_off)) {
                const int row = nf_kv_row(k_begin + j, page_shift, pt, pt_off);
                fd_rows[j] = row;
                fd_pos[j] = kv_pos[row];
            } else {
                fd_rows[j] = -1;
                fd_pos[j] = -1;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
#endif /* NF_HAS_SG_SHUFFLE */

    /* 2. pass 1: scores of the split's keys into s_s, per-item running max. Under NF_HAS_SG_SHUFFLE the
     * warp fast path runs first: warp w of the workgroup scans the 32 keys [w * 32, +32) of the split,
     * every lane holding one float4 of the key's K row (one coalesced row read per key) and the dot
     * reduced by the subgroup butterfly (nf_seg_sum at width 32, register form, no barrier). The scalar
     * body below is the portable form (OpenCL, exotic caches, splits above the workgroup). */
    float mx[NF_FD_MAX_G];
#pragma unroll
    for (int g = 0; g < NF_FD_MAX_G; g++) {
        mx[g] = -NF_INF;
    }
    const nf_i64 k_head = (nf_i64)kvh * k_head_stride;
#ifdef NF_HAS_SG_SHUFFLE
    if (fd_fast) {
        const int slane = lid & 31;
        const int w = lid >> 5;
        const int j0 = w * 32;
        const int dk4 = d_k >> 2;
        for (int t = 0; t < 32; t++) {
            const int j = j0 + t;
            if (j >= split_len) {
                break;   /* uniform per warp: every lane leaves together */
            }
            int p_k = -1;
            int vis = 0;
            int row = 0;
            nf_i64 e0 = 0;
            float ks = 1.0f;
            if (j < n_keys && fd_rows[j] >= 0) {
                /* the staged row and position: the page id -> row -> kv_pos chain ran in step 1b */
                row = fd_rows[j];
                p_k = fd_pos[j];
                vis = nf_attn_tree_visible(p_k, tree ? tree_mask[0] : (ulong)0, kv_end, n_fresh,
                                           q_pos, causal, swa_type, n_swa);
                if (vis) {
                    e0 = (nf_i64)row * k_row_stride + k_head;
                    ks = nf_kv_head_scale(k_type, kv_scales, row, kvh, n_head_kv, 0);
                }
            }
            float part[4];
#pragma unroll
            for (int g = 0; g < 4; g++) {
                part[g] = 0.0f;
            }
            if (vis && slane < dk4) {
                const float4 k4 = nf_kv_attn_load4(k, e0 + slane * 4, row, k_type, k_planes, plane_stride,
                                                   k_row_stride, d_k, centroids) * ks;
#pragma unroll
                for (int g = 0; g < 4; g++) {
                    if (g < g_tile) {
                        part[g] = dot(vload4(0, q_s + g * d_k + slane * 4), k4);
                    }
                }
            }
#pragma unroll
            for (int g = 0; g < 4; g++) {
                if (g < g_tile) {
                    float s = -NF_INF;
                    if (vis && g < g_cnt) {
                        s = nf_attn_score(nf_seg_sum(red, part[g], 32), scale, softcap)
                          + nf_attn_bias(bias_mode, h0 + g, n_head, q_pos, p_k, slope_s[g], rel_bias,
                                         n_buckets, max_distance, seq > 0 ? seq : 0);
                    }
                    if (slane == 0) {
                        s_s[g * split_len + j] = s;
                    }
                    mx[g] = fmax(mx[g], s);
                }
            }
        }
    } else
#endif /* NF_HAS_SG_SHUFFLE */
    {
    for (int j = lid; j < split_len; j += NF_FA_WG) {
        float acc[NF_FD_MAX_G];
#pragma unroll
        for (int g = 0; g < NF_FD_MAX_G; g++) {
            acc[g] = 0.0f;
        }
        int p_k = -1;
        int vis = 0;
        if (j < n_keys && nf_kv_page_ok(k_begin + j, page_shift, pt, pt_off)) {
            /* A masked page's row is not a row, so kv_pos is read only after the page test; an empty cell's
             * kv_pos stays -1 and is never visible. */
            const int row = nf_kv_row(k_begin + j, page_shift, pt, pt_off);
            p_k = kv_pos[row];
            /* One query row per launch: tree_mask[0] is the row's ancestor bitmask (the paged entry point offsets
             * the buffer to its sequence's word). */
            vis = nf_attn_tree_visible(p_k, tree ? tree_mask[0] : (ulong)0, kv_end, n_fresh,
                                       q_pos, causal, swa_type, n_swa);
            if (vis) {
                const nf_i64 e0 = (nf_i64)row * k_row_stride + k_head;
                float ks = nf_kv_head_scale(k_type, kv_scales, row, kvh, n_head_kv, 0);
                if (nf_kv_is_tq_mse(k_type)) {
                    /* The key decodes to its centroid vector: scale the dot with the rotated query by norm / |c|. */
                    ks *= nf_kv_vendor_tq_score(k, e0, row, k_type, k_planes, plane_stride, k_row_stride, d_k,
                                               centroids);
                }
                for (int d = 0; d < d_k; d += 4) {
                    const float4 k4 = nf_kv_attn_load4(k, e0 + d, row, k_type, k_planes, plane_stride,
                                                       k_row_stride, d_k, centroids) * ks;
#pragma unroll
                    for (int g = 0; g < NF_FD_MAX_G; g++) {
                        if (g < g_tile) {
                            acc[g] += dot(vload4(0, q_s + g * d_k + d), k4);
                        }
                    }
                }
            }
        }
#pragma unroll
        for (int g = 0; g < NF_FD_MAX_G; g++) {
            if (g < g_tile) {
                float s = -NF_INF;
                if (vis && g < g_cnt) {
                    s = nf_attn_score(acc[g], scale, softcap)
                      + nf_attn_bias(bias_mode, h0 + g, n_head, q_pos, p_k, slope_s[g], rel_bias,
                                     n_buckets, max_distance, seq > 0 ? seq : 0);
                }
                s_s[g * split_len + j] = s;
                mx[g] = fmax(mx[g], s);
            }
        }
    }
    }

    /* 3. softmax statistics per head: m (max), p = e^(s - m) in place, l (sum) */
    float ls[NF_FD_MAX_G];
#pragma unroll
    for (int g = 0; g < NF_FD_MAX_G; g++) {
        ls[g] = 0.0f;
        if (g < g_tile) {
            mx[g] = nf_wg_max(red, mx[g]);
        }
    }
    for (int j = lid; j < split_len; j += NF_FA_WG) {
#pragma unroll
        for (int g = 0; g < NF_FD_MAX_G; g++) {
            if (g < g_tile) {
                const float p = nf_attn_rescale(s_s[g * split_len + j], mx[g]);
                s_s[g * split_len + j] = p;
                ls[g] += p;
            }
        }
    }
#pragma unroll
    for (int g = 0; g < NF_FD_MAX_G; g++) {
        if (g < g_tile) {
            const float l = nf_wg_sum(red, ls[g]);
            if (lid == 0) {
                stats[g] = mx[g];
                stats[NF_FD_MAX_G + g] = l;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    /* 4. pass 2: o[g][d] = sum_t p[g][t] * V[t][d]. The workgroup runs as n_lg lane groups of gs work-items
     * that own interleaved keys (j = lg, step n_lg), so n_lg key streams run in parallel: every lane group
     * accumulates every (g, d) pair over its own keys, lane l of a group owning pairs l, l + gs, .... A row too
     * wide for one 32-lane group's accumulators (g_tile * d_v > 32 * NF_FA_ACC_MAX) runs as one group of
     * NF_FA_WG items over every key. The groups stage their accumulators in o_s; each work-item then sums the
     * partials of the pairs it writes in step 5 (pairs lid, lid + NF_FA_WG, ...). */
    const int n_pairs = g_tile * d_v;
    const int n_lg = n_pairs <= 32 * NF_FA_ACC_MAX ? NF_FA_WG / 32 : 1;
    const int gs = NF_FA_WG / n_lg;
    const int lg = lid / gs;
    const int lane = lid - lg * gs;
    float o[NF_FA_ACC_MAX];
    int pg[NF_FA_ACC_MAX];
    int pd[NF_FA_ACC_MAX];
    NF_LOCAL_PTR float* o_s = s_s + g_tile * split_len;
#ifdef NF_HAS_SG_SHUFFLE
    if (fd_fast && n_pairs <= 32 * NF_FA_ACC_MAX) {
        /* Fast form of the same accumulation: lane group lg owns keys lg, lg + 4, ..., and each lane
         * holds the float4 of dims lane * 4 of every o[g], so one coalesced V row load feeds the whole
         * group's lane slice per key. The staged partials land in the same o_s slots the step-5 sum
         * below reads. */
        const int dv4 = d_v >> 2;
        const nf_i64 v_head = (nf_i64)kvh * v_head_stride;
        float4 acc[4];
#pragma unroll
        for (int g = 0; g < 4; g++) {
            acc[g] = NF_F4S(0.0f);
        }
        for (int j = lg; j < n_keys; j += n_lg) {
            /* A key no head of the tile weights (masked, or on an unmapped page) is not read: its row
             * index is undefined off the page table. The test is uniform across the lane group. */
            int live = 0;
#pragma unroll
            for (int g = 0; g < 4; g++) {
                live |= g < g_tile && s_s[g * split_len + j] != 0.0f;
            }
            if (!live) {
                continue;
            }
            const int vrow = fd_rows[j];   /* staged with the scores' rows: live implies an on-page row */
            const nf_i64 e0 = (nf_i64)vrow * v_row_stride + v_head;
            const float v_scale = nf_kv_head_scale(v_type, kv_scales, vrow, kvh, n_head_kv, 1);
            float4 v4 = NF_F4S(0.0f);
            if (lane < dv4) {
                v4 = nf_kv_attn_load4(v, e0 + lane * 4, vrow, v_type, v_planes, plane_stride,
                                      v_row_stride, d_v, centroids) * v_scale;
            }
#pragma unroll
            for (int g = 0; g < 4; g++) {
                if (g < g_tile) {
                    const float p = s_s[g * split_len + j];
                    if (p != 0.0f) {
                        acc[g] += p * v4;
                    }
                }
            }
        }
#pragma unroll
        for (int g = 0; g < 4; g++) {
            if (g < g_tile && lane < dv4) {
                vstore4(acc[g], 0, o_s + lg * n_pairs + g * d_v + lane * 4);
            }
        }
    } else
#endif /* NF_HAS_SG_SHUFFLE */
    {
#pragma unroll
    for (int i = 0; i < NF_FA_ACC_MAX; i++) {
        const int pi = lane + i * gs;
        const int g = pi < n_pairs ? pi / d_v : 0;
        o[i] = 0.0f;
        pg[i] = g;
        pd[i] = pi - g * d_v;
    }
    const nf_i64 v_head = (nf_i64)kvh * v_head_stride;
    for (int j = lg; j < n_keys; j += n_lg) {
        const int vrow = nf_kv_row(k_begin + j, page_shift, pt, pt_off);
        const nf_i64 e0 = (nf_i64)vrow * v_row_stride + v_head;
        const float v_scale = nf_kv_head_scale(v_type, kv_scales, vrow, kvh, n_head_kv, 1);
#pragma unroll
        for (int i = 0; i < NF_FA_ACC_MAX; i++) {
            if (lane + i * gs < n_pairs) {
                const float p = s_s[pg[i] * split_len + j];
                if (p != 0.0f) {
                    o[i] += p * (nf_kv_attn_load1(v, e0 + pd[i], vrow, v_type, v_planes, plane_stride,
                                                  v_row_stride, d_v, centroids) * v_scale);
                }
            }
        }
    }
#pragma unroll
    for (int i = 0; i < NF_FA_ACC_MAX; i++) {
        const int pi = lane + i * gs;
        if (pi < n_pairs) {
            o_s[lg * n_pairs + pi] = o[i];
        }
    }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
#pragma unroll
    for (int i = 0; i < NF_FA_ACC_MAX; i++) {
        const int pi = lid + i * NF_FA_WG;
        const int g = pi < n_pairs ? pi / d_v : 0;
        float sum = 0.0f;
        if (pi < n_pairs) {
            for (int w = 0; w < n_lg; w++) {
                sum += o_s[w * n_pairs + pi];
            }
        }
        o[i] = sum;
        pg[i] = g;
        pd[i] = pi - g * d_v;
    }

    /* 5. output: normalized row (direct_out) or split partial */
#pragma unroll
    for (int i = 0; i < NF_FA_ACC_MAX; i++) {
        const int g = pg[i];
        const int d = pd[i];
        if (lid + i * NF_FA_WG < n_pairs && g < g_cnt && split < n_split) {
            const int h = h0 + g;
            const float m = stats[g];
            const float l = stats[NF_FD_MAX_G + g];
            if (direct_out) {
                const float2 fl = has_sinks ? nf_attn_sink(m, l, sinks[h]) : NF_F2(1.0f, l);
                out[(nf_i64)out_off + (nf_i64)h * d_v + d] = fl.y > 0.0f ? o[i] * fl.x / fl.y : 0.0f;
            } else {
                NF_GLOBAL float* pr = partial + (((nf_i64)max(seq, 0) * n_head + h) * n_split + split) * (d_v + 2);
                pr[d] = o[i];
                if (d == 0) {
                    pr[d_v] = m;
                    pr[d_v + 1] = l;
                }
            }
        }
    }
}

/*
 * LLM_FlashDecode_<KV>(25 scalars, 10 buffers); KV is a cache element type of common/kv_types.h. k and v may be the same
 * buffer (MLA latent cache); sinks, rel_bias and kv_scales are read-only placeholders when unused (kv_scales holds
 * n_head_kv K scales then n_head_kv V scales, all 1.0 without an FP8 cache, or the interleaved per-(row, KV head)
 * scales of a per-token-head cache). flags packs the causal, has_sinks, direct and tree bits
 * (common/attn_common.h NF_ATTN_FLAG_*), with n_fresh in bits 5-31 under the tree bit: the last n_fresh slots of the
 * key range take their visibility from tree_mask[0], the one query row's ancestor bitmask.
 */
#define NF_FLASH_DECODE_KERNEL(SFX, KV_T, KV_CODE)                                           \
NF_KERNEL void LLM_FlashDecode_##SFX(                                                        \
    const int n_head, const int n_head_kv, const int d_k, const int d_v, const int g_tile,    \
    const int n_split, const int split_len, const int kv_start, const int kv_end,            \
    const int q_pos,                                                                         \
    const int k_row_stride, const int k_head_stride, const int v_row_stride,                  \
    const int v_head_stride,                                                                 \
    const float scale, const float softcap, const float max_bias,                            \
    const int flags, const int swa_type, const int n_swa,                                    \
    const int bias_mode, const int n_buckets, const int max_distance,                        \
    const int q_off, const int out_off,                                                      \
    NF_GLOBAL const float* q,                                                                \
    NF_GLOBAL const KV_T* k,                                                                 \
    NF_GLOBAL const KV_T* v,                                                                 \
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
    NF_LOCAL float stats[3 * NF_FD_MAX_G];                                                   \
    nf_fd_run(n_head, n_head_kv, d_k, d_v, g_tile, n_split, split_len, kv_start, kv_end,     \
              q_pos, k_row_stride, k_head_stride, v_row_stride, v_head_stride,               \
              scale, softcap, max_bias, flags & NF_ATTN_FLAG_CAUSAL, swa_type, n_swa,        \
              bias_mode, n_buckets, max_distance, flags & NF_ATTN_FLAG_SINKS,                \
              flags & NF_ATTN_FLAG_DIRECT, q_off, out_off,                                   \
              q, (NF_GLOBAL const uchar*)k, (NF_GLOBAL const uchar*)v, kv_pos, sinks,         \
              rel_bias, out, partial, kv_scales, KV_CODE, KV_CODE,                              \
              (flags & NF_ATTN_FLAG_V_ALIAS) != 0, -1, kv_pos, 0, -1,                           \
              flags & NF_ATTN_FLAG_TREE, tree_mask, nf_attn_flag_fresh(flags), smem, red,       \
              stats);                                                                            \
}

NF_FLASH_DECODE_KERNEL(F32, float, NF_KV_F32)
NF_FLASH_DECODE_KERNEL(F16, NF_HALF, NF_KV_F16)
NF_FLASH_DECODE_KERNEL(Q8_0, uchar, NF_KV_Q8_0)
NF_FLASH_DECODE_KERNEL(BF16, ushort, NF_KV_BF16)
NF_FLASH_DECODE_KERNEL(Q4_0, uchar, NF_KV_Q4_0)
NF_FLASH_DECODE_KERNEL(Q4_1, uchar, NF_KV_Q4_1)
NF_FLASH_DECODE_KERNEL(IQ4_NL, uchar, NF_KV_IQ4_NL)
NF_FLASH_DECODE_KERNEL(Q5_0, uchar, NF_KV_Q5_0)
NF_FLASH_DECODE_KERNEL(Q5_1, uchar, NF_KV_Q5_1)
NF_FLASH_DECODE_KERNEL(FP8_E4M3, uchar, NF_KV_FP8_E4M3)
NF_FLASH_DECODE_KERNEL(FP8_E5M2, uchar, NF_KV_FP8_E5M2)
NF_FLASH_DECODE_KERNEL(FP8_PER_TOKEN_HEAD, uchar, NF_KV_FP8_PER_TOKEN_HEAD)
NF_FLASH_DECODE_KERNEL(INT8_PER_TOKEN_HEAD, uchar, NF_KV_INT8_PER_TOKEN_HEAD)
NF_FLASH_DECODE_KERNEL(INT4_PER_TOKEN_HEAD, uchar, NF_KV_INT4_PER_TOKEN_HEAD)

/* The vendor formats take the same arguments: their side planes and centroid table ride the kv_scales binding. */
NF_FLASH_DECODE_KERNEL(NVFP4, uchar, NF_KV_NVFP4)
NF_FLASH_DECODE_KERNEL(NVFP4_4OVER6, uchar, NF_KV_NVFP4_4OVER6)
NF_FLASH_DECODE_KERNEL(TQ_K8_KEYS, uchar, NF_KV_TQ_K8_KEYS)
NF_FLASH_DECODE_KERNEL(TQ_4BIT_KEYS, uchar, NF_KV_TQ_4BIT_KEYS)
NF_FLASH_DECODE_KERNEL(TQ_3BIT_KEYS, uchar, NF_KV_TQ_3BIT_KEYS)
NF_FLASH_DECODE_KERNEL(TQ_4BIT_VALUES, uchar, NF_KV_TQ_4BIT_VALUES)
NF_FLASH_DECODE_KERNEL(TQ_3BIT_VALUES, uchar, NF_KV_TQ_3BIT_VALUES)
NF_FLASH_DECODE_KERNEL(FP8_DS_MLA, uchar, NF_KV_FP8_DS_MLA)
NF_FLASH_DECODE_KERNEL(NVFP4_DS_MLA, uchar, NF_KV_NVFP4_DS_MLA)

/*
 * LLM_FlashDecode_Mixed(26 scalars, 10 buffers): LLM_FlashDecode_<KV> over a K cache and a V cache of different types.
 * The arguments are those of LLM_FlashDecode_<KV> followed by kv_types (common/attn_common.h); k and v are byte
 * buffers. A malformed kv_types violates the plan and writes NaN outputs.
 */
NF_KERNEL void LLM_FlashDecode_Mixed(
    const int n_head, const int n_head_kv, const int d_k, const int d_v, const int g_tile,
    const int n_split, const int split_len, const int kv_start, const int kv_end,
    const int q_pos,
    const int k_row_stride, const int k_head_stride, const int v_row_stride,
    const int v_head_stride,
    const float scale, const float softcap, const float max_bias,
    const int flags, const int swa_type, const int n_swa,
    const int bias_mode, const int n_buckets, const int max_distance,
    const int q_off, const int out_off,
    NF_GLOBAL const float* q,
    NF_GLOBAL const uchar* k,
    NF_GLOBAL const uchar* v,
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
    NF_LOCAL float stats[3 * NF_FD_MAX_G];
    nf_fd_run(n_head, n_head_kv, d_k, d_v, g_tile, n_split, split_len, kv_start, kv_end,
              q_pos, k_row_stride, k_head_stride, v_row_stride, v_head_stride,
              scale, softcap, max_bias, flags & NF_ATTN_FLAG_CAUSAL, swa_type, n_swa,
              bias_mode, n_buckets, max_distance, flags & NF_ATTN_FLAG_SINKS,
              flags & NF_ATTN_FLAG_DIRECT, q_off, out_off,
              q, k, v, kv_pos, sinks, rel_bias, out, partial, kv_scales,
              nf_attn_kv_type_k(kv_types), nf_attn_kv_type_v(kv_types),
              (flags & NF_ATTN_FLAG_V_ALIAS) != 0, -1, kv_pos, 0, -1,
              flags & NF_ATTN_FLAG_TREE, tree_mask, nf_attn_flag_fresh(flags), smem, red,
              stats);
}

/*
 * Paged decode over a paginated KV cache: every sequence's rows live behind a page table
 * page_table[s * maxPages .. ] (common/attn_common.h nf_kv_row), one page table per sequence.
 */

/* Plan constraints of LLM_FlashDecodePaged_*: those of LLM_FlashDecode_* plus the page geometry. page_shift is the
 * log2 page size (0 = one row per page), bounded so the physical row stays an int; the strides must be positive;
 * seq_kv_len[s] and the page table entries are validated on the host before the upload. */
NF_INLINE int nf_fd_paged_ok(int n_head, int n_head_kv, int d_k, int d_v, int g_tile, int n_split,
                             int split_len, int direct_out, int k_type, int v_type,
                             int page_shift, int max_pages, int n_seq, int q_seq_stride, int out_seq_stride) {
    return page_shift >= 0 && page_shift <= 20 && max_pages >= 1 && n_seq >= 1
        && q_seq_stride >= 0 && out_seq_stride >= 0
        /* stage_floats 0: this coarse gate lacks the bias the fast-path check derives it from, and the
         * shared nf_fd_run below re-checks the exact budget its own staging needs */
        && nf_fd_plan_ok(n_head, n_head_kv, d_k, d_v, g_tile, n_split, split_len, direct_out, k_type, v_type, 0);
}

/* NaN over the outputs of a paged call whose arguments violate the plan (workgroup (0, 0) only). The out stride of
 * the direct-out case is n_head * d_v (the plan guarantees out_seq_stride >= n_head * d_v when the host reuses
 * one dense output block per sequence). */
NF_INLINE void nf_fd_paged_poison(int n_seq, int n_head, int d_v, int n_split, int direct_out, int out_off,
                                  NF_GLOBAL float* out, NF_GLOBAL float* partial) {
    if (get_group_id(0) != 0 || get_group_id(1) != 0 || n_seq <= 0 || n_head <= 0 || d_v <= 0 || n_split <= 0) {
        return;
    }
    const int lid = (int)get_local_id(0);
    const int lsz = (int)get_local_size(0);
    if (direct_out) {
        for (int s = 0; s < n_seq; s++) {
            for (int i = lid; i < n_head * d_v; i += lsz) {
                out[(nf_i64)out_off + (nf_i64)s * n_head * d_v + i] = NF_NAN;
            }
        }
    } else {
        const nf_i64 n = (nf_i64)n_seq * n_head * n_split * (d_v + 2);
        for (nf_i64 i = lid; i < n; i += lsz) {
            partial[i] = NF_NAN;
        }
    }
}

/*
 * Full paged entry-point arguments: the LLM_FlashDecode_<KV> arguments in order (kv_start, kv_end and q_pos are
 * ignored; the host passes 0), then page_shift, max_pages, n_seq, q_seq_stride, out_seq_stride and buffers
 * page_table, seq_kv_len, q_pos_seq.
 */
#define NF_FLASH_DECODE_PAGED_RUN_ARGS_PAGED(KV_T)                                            \
    const int n_head, const int n_head_kv, const int d_k, const int d_v, const int g_tile,     \
    const int n_split, const int split_len, const int kv_start, const int kv_end,             \
    const int q_pos,                                                                          \
    const int k_row_stride, const int k_head_stride, const int v_row_stride,                  \
    const int v_head_stride,                                                                  \
    const float scale, const float softcap, const float max_bias,                             \
    const int flags, const int swa_type, const int n_swa,                                     \
    const int bias_mode, const int n_buckets, const int max_distance,                         \
    const int q_off, const int out_off,                                                       \
    const int page_shift, const int max_pages, const int n_seq,                               \
    const int q_seq_stride, const int out_seq_stride,                                         \
    NF_GLOBAL const float* q,                                                                 \
    NF_GLOBAL const KV_T* k,                                                                  \
    NF_GLOBAL const KV_T* v,                                                                  \
    NF_GLOBAL const int* NF_RESTRICT kv_pos,                                                  \
    NF_GLOBAL const float* NF_RESTRICT sinks,                                                 \
    NF_GLOBAL const float* NF_RESTRICT rel_bias,                                              \
    NF_GLOBAL float* out,                                                                     \
    NF_GLOBAL float* partial,                                                                 \
    NF_GLOBAL const int* NF_RESTRICT page_table,                                              \
    NF_GLOBAL const int* NF_RESTRICT seq_kv_len,                                              \
    NF_GLOBAL const int* NF_RESTRICT q_pos_seq,                                               \
    NF_GLOBAL const float* NF_RESTRICT kv_scales,                                             \
    NF_GLOBAL const ulong* NF_RESTRICT tree_mask

/*
 * Per workgroup: the sequence s, its slot range [0, kv_len) and its q / out blocks are resolved before
 * nf_fd_run, which is the shared algorithm with page_shift >= 0 (K, V and kv_pos read at nf_kv_row(t),
 * page ids below 0 masked, partial rows at s * n_head + h).
 * kv_len = seq_kv_len[s] clamped to the page table's slots (maxPages << page_shift unreadable entries would be a
 * host contract violation; the clamp keeps every page-table read in bounds).
 */
#define NF_FLASH_DECODE_PAGED_RUN_ARGS                                                       \
    const int n_head, const int n_head_kv, const int d_k, const int d_v, const int g_tile,    \
    const int n_split, const int split_len,                                                  \
    const int k_row_stride, const int k_head_stride, const int v_row_stride,                  \
    const int v_head_stride,                                                                 \
    const float scale, const float softcap, const float max_bias,                            \
    const int flags, const int swa_type, const int n_swa,                                    \
    const int bias_mode, const int n_buckets, const int max_distance,                        \
    const int q_off, const int out_off,                                                      \
    const int page_shift, const int max_pages, const int n_seq,                              \
    const int q_seq_stride, const int out_seq_stride,                                        \
    NF_GLOBAL const float* q,                                                                \
    NF_GLOBAL const uchar* k,                                                                \
    NF_GLOBAL const uchar* v,                                                                \
    NF_GLOBAL const int* NF_RESTRICT kv_pos,                                                 \
    NF_GLOBAL const float* NF_RESTRICT sinks,                                                \
    NF_GLOBAL const float* NF_RESTRICT rel_bias,                                             \
    NF_GLOBAL float* out,                                                                    \
    NF_GLOBAL float* partial,                                                                \
    NF_GLOBAL const int* NF_RESTRICT page_table,                                             \
    NF_GLOBAL const int* NF_RESTRICT seq_kv_len,                                             \
    NF_GLOBAL const int* NF_RESTRICT q_pos_seq,                                              \
    NF_GLOBAL const float* NF_RESTRICT kv_scales,                                            \
    NF_GLOBAL const ulong* NF_RESTRICT tree_mask

NF_INLINE void nf_fd_paged_run(
    NF_FLASH_DECODE_PAGED_RUN_ARGS, const int k_type, const int v_type,
    NF_LOCAL_PTR float* smem, NF_LOCAL_PTR float* red, NF_LOCAL_PTR float* stats)
{
    if (!nf_fd_paged_ok(n_head, n_head_kv, d_k, d_v, g_tile, n_split, split_len,
                        flags & NF_ATTN_FLAG_DIRECT, k_type, v_type,
                        page_shift, max_pages, n_seq, q_seq_stride, out_seq_stride)) {
        nf_fd_paged_poison(n_seq, n_head, d_v, n_split, flags & NF_ATTN_FLAG_DIRECT, out_off, out, partial);
        return;   /* uniform across the workgroup: no barrier has been reached */
    }

    const int G = n_head / n_head_kv;
    const int n_gt = (G + g_tile - 1) / g_tile;
    const int gid = (int)get_group_id(1);
    const int s = gid / (n_head_kv * n_gt);   /* the sequence is folded into dimension 1 in front */
    const nf_i64 kv_cap = (nf_i64)max_pages << page_shift;
    const int kv_len = seq_kv_len[s] < 0 ? 0
                       : ((nf_i64)seq_kv_len[s] > kv_cap ? (int)kv_cap : seq_kv_len[s]);
    /* The page table of sequence s starts at s * maxPages (PagedStep.pageTable(g), the host-validated
     * int[nSeq x maxPages]; n_seq * maxPages fits int by the upload budget), so every row lookup lands in it. */
    const int pt_off = s * max_pages;
    nf_fd_run(n_head, n_head_kv, d_k, d_v, g_tile, n_split, split_len,
              0, kv_len, q_pos_seq[s],
              k_row_stride, k_head_stride, v_row_stride, v_head_stride,
              scale, softcap, max_bias, flags & NF_ATTN_FLAG_CAUSAL, swa_type, n_swa,
              bias_mode, n_buckets, max_distance, flags & NF_ATTN_FLAG_SINKS,
              flags & NF_ATTN_FLAG_DIRECT, q_off + s * q_seq_stride, out_off + s * out_seq_stride,
              q, k, v, kv_pos, sinks, rel_bias, out, partial, kv_scales, k_type, v_type,
              (flags & NF_ATTN_FLAG_V_ALIAS) != 0, page_shift, page_table, pt_off, s,
              flags & NF_ATTN_FLAG_TREE, tree_mask + s, nf_attn_flag_fresh(flags), smem, red,
              stats);
}

/*
 * LLM_FlashDecodePaged_<KV>(30 scalars, 13 buffers): the arguments are those of LLM_FlashDecode_<KV>
 * (kv_start, kv_end and q_pos are ignored there; the host passes 0), then page_shift, max_pages, n_seq,
 * q_seq_stride, out_seq_stride and buffers page_table, seq_kv_len, q_pos_seq. k and v may be the same buffer
 * (MLA latent cache); sinks, rel_bias, kv_scales, q_pos_seq and out are placeholders when unused.
 * Grid: global {n_split * 128, n_seq * n_head_kv * ceil(G / g_tile)}, local {128, 1}.
 */
#define NF_FLASH_DECODE_PAGED_KERNEL(SFX, KV_T, KV_CODE)                                      \
NF_KERNEL void LLM_FlashDecodePaged_##SFX(                                                    \
    NF_FLASH_DECODE_PAGED_RUN_ARGS_PAGED(KV_T))                                               \
{                                                                                             \
    NF_LOCAL float smem[NF_FA_SMEM_FLOATS];                                                   \
    NF_LOCAL float red[NF_FA_WG];                                                             \
    NF_LOCAL float stats[3 * NF_FD_MAX_G];                                                    \
    nf_fd_paged_run(n_head, n_head_kv, d_k, d_v, g_tile, n_split, split_len,                   \
                    k_row_stride, k_head_stride, v_row_stride, v_head_stride,                  \
                    scale, softcap, max_bias, flags, swa_type, n_swa, bias_mode, n_buckets,    \
                    max_distance, q_off, out_off,                                              \
                    page_shift, max_pages, n_seq, q_seq_stride, out_seq_stride,                \
                    q, (NF_GLOBAL const uchar*)k, (NF_GLOBAL const uchar*)v, kv_pos,            \
                    sinks, rel_bias, out, partial, page_table, seq_kv_len, q_pos_seq,          \
                    kv_scales, tree_mask, KV_CODE, KV_CODE, smem, red, stats);                  \
}

NF_FLASH_DECODE_PAGED_KERNEL(F32, float, NF_KV_F32)
NF_FLASH_DECODE_PAGED_KERNEL(F16, NF_HALF, NF_KV_F16)
NF_FLASH_DECODE_PAGED_KERNEL(Q8_0, uchar, NF_KV_Q8_0)
NF_FLASH_DECODE_PAGED_KERNEL(BF16, ushort, NF_KV_BF16)
NF_FLASH_DECODE_PAGED_KERNEL(Q4_0, uchar, NF_KV_Q4_0)
NF_FLASH_DECODE_PAGED_KERNEL(Q4_1, uchar, NF_KV_Q4_1)
NF_FLASH_DECODE_PAGED_KERNEL(IQ4_NL, uchar, NF_KV_IQ4_NL)
NF_FLASH_DECODE_PAGED_KERNEL(Q5_0, uchar, NF_KV_Q5_0)
NF_FLASH_DECODE_PAGED_KERNEL(Q5_1, uchar, NF_KV_Q5_1)
NF_FLASH_DECODE_PAGED_KERNEL(FP8_E4M3, uchar, NF_KV_FP8_E4M3)
NF_FLASH_DECODE_PAGED_KERNEL(FP8_E5M2, uchar, NF_KV_FP8_E5M2)
NF_FLASH_DECODE_PAGED_KERNEL(FP8_PER_TOKEN_HEAD, uchar, NF_KV_FP8_PER_TOKEN_HEAD)
NF_FLASH_DECODE_PAGED_KERNEL(INT8_PER_TOKEN_HEAD, uchar, NF_KV_INT8_PER_TOKEN_HEAD)
NF_FLASH_DECODE_PAGED_KERNEL(INT4_PER_TOKEN_HEAD, uchar, NF_KV_INT4_PER_TOKEN_HEAD)

/* The vendor formats take the same arguments: their side planes and centroid table ride the kv_scales binding. */
NF_FLASH_DECODE_PAGED_KERNEL(NVFP4, uchar, NF_KV_NVFP4)
NF_FLASH_DECODE_PAGED_KERNEL(NVFP4_4OVER6, uchar, NF_KV_NVFP4_4OVER6)
NF_FLASH_DECODE_PAGED_KERNEL(TQ_K8_KEYS, uchar, NF_KV_TQ_K8_KEYS)
NF_FLASH_DECODE_PAGED_KERNEL(TQ_4BIT_KEYS, uchar, NF_KV_TQ_4BIT_KEYS)
NF_FLASH_DECODE_PAGED_KERNEL(TQ_3BIT_KEYS, uchar, NF_KV_TQ_3BIT_KEYS)
NF_FLASH_DECODE_PAGED_KERNEL(TQ_4BIT_VALUES, uchar, NF_KV_TQ_4BIT_VALUES)
NF_FLASH_DECODE_PAGED_KERNEL(TQ_3BIT_VALUES, uchar, NF_KV_TQ_3BIT_VALUES)
NF_FLASH_DECODE_PAGED_KERNEL(FP8_DS_MLA, uchar, NF_KV_FP8_DS_MLA)
NF_FLASH_DECODE_PAGED_KERNEL(NVFP4_DS_MLA, uchar, NF_KV_NVFP4_DS_MLA)

/*
 * LLM_FlashDecodePaged_Mixed(31 scalars, 13 buffers): LLM_FlashDecodePaged_<KV> over a K cache and a V cache of
 * different types. The arguments are those of LLM_FlashDecodePaged_<KV> followed by kv_types (common/attn_common.h)
 * after the buffers; k and v are byte buffers. A malformed kv_types violates the plan and writes NaN outputs.
 */
NF_KERNEL void LLM_FlashDecodePaged_Mixed(NF_FLASH_DECODE_PAGED_RUN_ARGS_PAGED(uchar), const int kv_types)
{
    NF_LOCAL float smem[NF_FA_SMEM_FLOATS];
    NF_LOCAL float red[NF_FA_WG];
    NF_LOCAL float stats[3 * NF_FD_MAX_G];
    nf_fd_paged_run(n_head, n_head_kv, d_k, d_v, g_tile, n_split, split_len,
                    k_row_stride, k_head_stride, v_row_stride, v_head_stride,
                    scale, softcap, max_bias, flags, swa_type, n_swa, bias_mode, n_buckets,
                    max_distance, q_off, out_off,
                    page_shift, max_pages, n_seq, q_seq_stride, out_seq_stride,
                    q, k, v, kv_pos, sinks, rel_bias, out, partial, page_table, seq_kv_len,
                    q_pos_seq, kv_scales, tree_mask,
                    nf_attn_kv_type_k(kv_types), nf_attn_kv_type_v(kv_types), smem, red, stats);
}

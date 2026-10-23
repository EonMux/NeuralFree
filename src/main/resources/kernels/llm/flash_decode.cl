/* flash_decode.cl - LLM_FlashDecode_<KV> and LLM_FlashDecode_Mixed split-K decode attention. Requires: common/dialect.h, common/config.h, common/reduce.h, common/math.h, common/kv_types.h, common/attn_common.h */
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
 * Query head h reads KV head kvh = h / (n_head / n_head_kv) (ggml broadcast rk2 = neq2 / nek2).
 * K is read as k_type and V as v_type: both are the variant's type in LLM_FlashDecode_<KV>; LLM_FlashDecode_Mixed
 * takes them from its trailing kv_types argument (common/attn_common.h), as ggml's flash attention reads K and V of
 * different cache types (llama.cpp ggml-cpu/ops.cpp:9007-9013 reads each through its own type traits).
 *
 * Grid: global {n_split * 128, n_head_kv * ceil(G / g_tile)}, local {128, 1}, G = n_head / n_head_kv.
 * Workgroup (s, y) handles keys [kv_start + s * split_len, min(+ split_len, kv_end)) for the
 * g_tile query heads h0 = kvh * G + gt * g_tile .. of KV head kvh = y / ceil(G/g_tile), gt = y % ceil(G/g_tile).
 *
 * Shared memory (NF_FA_SMEM_FLOATS floats): q_s[g_tile * d_k] | s_s[g_tile * split_len].
 * Plan constraints (FlashAttnTiling.decode guarantees them; a violation writes NaN outputs):
 *   g_tile <= NF_FD_MAX_G, g_tile * (d_k + split_len) <= NF_FA_SMEM_FLOATS,
 *   g_tile * d_v <= NF_FA_WG * NF_FA_ACC_MAX, d_k % 4 == 0, d_v % 4 == 0, local size == NF_FA_WG,
 *   direct_out only with n_split == 1, k_type and v_type KV cache type codes.
 *
 * Algorithm (no per-key barriers):
 *   1. q of the g_tile heads is staged in q_s; ALiBi slopes per head in stats.
 *   2. Pass 1: work-item j scores keys j, j + 128, ... of the split; one K row read feeds the g_tile
 *      dot products; masked keys score -INF without touching K.
 *   3. Per head: m = workgroup max, p = e^(s - m) in place, l = workgroup sum.
 *   4. Pass 2: work-item pairs (g, d) (g_tile * d_v / 128 per item) accumulate sum_t p[g][t] * V[t][d],
 *      V read coalesced along d; keys with p == 0 are not read.
 *   5. direct_out: sinks merged (nf_attn_sink) and o / l written; l == 0 (fully masked row, no sink)
 *      writes 0. Otherwise (o, m, l) go to partial.
 * Subgroups: reductions use nf_wg_max / nf_wg_sum (common/reduce.h); local size 128 is a multiple of 32
 * and of every subgroup size, as the subgroup reductions require.
 */

/* Plan constraints of LLM_FlashDecode_*; see the header comment. */
NF_INLINE int nf_fd_plan_ok(int n_head, int n_head_kv, int d_k, int d_v, int g_tile, int n_split,
                            int split_len, int direct_out, int k_type, int v_type) {
    return nf_attn_kv_type_ok(k_type) && nf_attn_kv_type_ok(v_type)
        && n_head > 0 && n_head_kv > 0 && n_head % n_head_kv == 0
        && g_tile >= 1 && g_tile <= NF_FD_MAX_G
        && d_k > 0 && d_v > 0 && (d_k & 3) == 0 && (d_v & 3) == 0
        && n_split >= 1 && split_len > 0
        && g_tile * (d_k + split_len) <= NF_FA_SMEM_FLOATS
        && g_tile * d_v <= NF_FA_WG * NF_FA_ACC_MAX
        && (direct_out == 0 || n_split == 1)
        && (int)get_local_size(0) == NF_FA_WG;
}

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
    NF_GLOBAL float* out, NF_GLOBAL float* partial, const int k_type, const int v_type,
    NF_LOCAL_PTR float* smem, NF_LOCAL_PTR float* red, NF_LOCAL_PTR float* stats)
{
    if (!nf_fd_plan_ok(n_head, n_head_kv, d_k, d_v, g_tile, n_split, split_len, direct_out, k_type, v_type)) {
        nf_fd_poison(n_head, d_v, n_split, direct_out, out_off, out, partial);
        return;   /* uniform across the workgroup: no barrier has been reached */
    }

    const int lid = (int)get_local_id(0);
    const int split = (int)get_group_id(0);
    const int G = n_head / n_head_kv;
    const int n_gt = (G + g_tile - 1) / g_tile;
    const int kvh = (int)get_group_id(1) / n_gt;
    const int gt = (int)get_group_id(1) - kvh * n_gt;
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
        slope_s[lid] = (bias_mode == NF_ATTN_BIAS_ALIBI && lid < g_cnt)
                       ? nf_alibi_slope(h0 + lid, n_head, max_bias) : 1.0f;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    /* 2. pass 1: scores of the split's keys into s_s, per-item running max */
    float mx[NF_FD_MAX_G];
#pragma unroll
    for (int g = 0; g < NF_FD_MAX_G; g++) {
        mx[g] = -NF_INF;
    }
    const nf_i64 k_head = (nf_i64)kvh * k_head_stride;
    for (int j = lid; j < split_len; j += NF_FA_WG) {
        float acc[NF_FD_MAX_G];
#pragma unroll
        for (int g = 0; g < NF_FD_MAX_G; g++) {
            acc[g] = 0.0f;
        }
        int p_k = -1;
        int vis = 0;
        if (j < n_keys) {
            const int t = k_begin + j;
            p_k = kv_pos[t];
            vis = nf_attn_visible(q_pos, p_k, causal, swa_type, n_swa);
            if (vis) {
                const nf_i64 e0 = (nf_i64)t * k_row_stride + k_head;
                for (int d = 0; d < d_k; d += 4) {
                    const float4 k4 = nf_kv_load4(k, e0 + d, k_type);
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
                                     n_buckets, max_distance);
                }
                s_s[g * split_len + j] = s;
                mx[g] = fmax(mx[g], s);
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

    /* 4. pass 2: o[g][d] = sum_t p[g][t] * V[t][d] over (g, d) pairs owned by this item */
    const int n_pairs = g_tile * d_v;
    float o[NF_FA_ACC_MAX];
    int pg[NF_FA_ACC_MAX];
    int pd[NF_FA_ACC_MAX];
#pragma unroll
    for (int i = 0; i < NF_FA_ACC_MAX; i++) {
        const int pi = lid + i * NF_FA_WG;
        const int g = pi < n_pairs ? pi / d_v : 0;
        o[i] = 0.0f;
        pg[i] = g;
        pd[i] = pi - g * d_v;
    }
    const nf_i64 v_head = (nf_i64)kvh * v_head_stride;
    for (int j = 0; j < n_keys; j++) {
        const nf_i64 e0 = (nf_i64)(k_begin + j) * v_row_stride + v_head;
#pragma unroll
        for (int i = 0; i < NF_FA_ACC_MAX; i++) {
            if (lid + i * NF_FA_WG < n_pairs) {
                const float p = s_s[pg[i] * split_len + j];
                if (p != 0.0f) {
                    o[i] += p * nf_kv_load1(v, e0 + pd[i], v_type);
                }
            }
        }
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
                NF_GLOBAL float* pr = partial + ((nf_i64)h * n_split + split) * (d_v + 2);
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
 * LLM_FlashDecode_<KV>(27 scalars, 8 buffers); KV is a cache element type of common/kv_types.h. k and v may be the same
 * buffer (MLA latent cache); sinks and rel_bias are read-only placeholders when unused.
 */
#define NF_FLASH_DECODE_KERNEL(SFX, KV_T, KV_CODE)                                           \
NF_KERNEL void LLM_FlashDecode_##SFX(                                                        \
    const int n_head, const int n_head_kv, const int d_k, const int d_v, const int g_tile,    \
    const int n_split, const int split_len, const int kv_start, const int kv_end,            \
    const int q_pos,                                                                         \
    const int k_row_stride, const int k_head_stride, const int v_row_stride,                  \
    const int v_head_stride,                                                                 \
    const float scale, const float softcap, const float max_bias,                            \
    const int causal, const int swa_type, const int n_swa,                                   \
    const int bias_mode, const int n_buckets, const int max_distance,                        \
    const int has_sinks, const int direct_out, const int q_off, const int out_off,           \
    NF_GLOBAL const float* q,                                                                \
    NF_GLOBAL const KV_T* k,                                                                 \
    NF_GLOBAL const KV_T* v,                                                                 \
    NF_GLOBAL const int* NF_RESTRICT kv_pos,                                                 \
    NF_GLOBAL const float* NF_RESTRICT sinks,                                                \
    NF_GLOBAL const float* NF_RESTRICT rel_bias,                                             \
    NF_GLOBAL float* out,                                                                    \
    NF_GLOBAL float* partial)                                                                \
{                                                                                            \
    NF_LOCAL float smem[NF_FA_SMEM_FLOATS];                                                  \
    NF_LOCAL float red[NF_FA_WG];                                                            \
    NF_LOCAL float stats[3 * NF_FD_MAX_G];                                                   \
    nf_fd_run(n_head, n_head_kv, d_k, d_v, g_tile, n_split, split_len, kv_start, kv_end,     \
              q_pos, k_row_stride, k_head_stride, v_row_stride, v_head_stride,               \
              scale, softcap, max_bias, causal, swa_type, n_swa, bias_mode, n_buckets,       \
              max_distance, has_sinks, direct_out, q_off, out_off,                           \
              q, (NF_GLOBAL const uchar*)k, (NF_GLOBAL const uchar*)v, kv_pos, sinks,         \
              rel_bias, out, partial, KV_CODE, KV_CODE, smem, red, stats);                   \
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

/*
 * LLM_FlashDecode_Mixed(28 scalars, 8 buffers): LLM_FlashDecode_<KV> over a K cache and a V cache of different types.
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
    const int causal, const int swa_type, const int n_swa,
    const int bias_mode, const int n_buckets, const int max_distance,
    const int has_sinks, const int direct_out, const int q_off, const int out_off,
    NF_GLOBAL const float* q,
    NF_GLOBAL const uchar* k,
    NF_GLOBAL const uchar* v,
    NF_GLOBAL const int* NF_RESTRICT kv_pos,
    NF_GLOBAL const float* NF_RESTRICT sinks,
    NF_GLOBAL const float* NF_RESTRICT rel_bias,
    NF_GLOBAL float* out,
    NF_GLOBAL float* partial,
    const int kv_types)
{
    NF_LOCAL float smem[NF_FA_SMEM_FLOATS];
    NF_LOCAL float red[NF_FA_WG];
    NF_LOCAL float stats[3 * NF_FD_MAX_G];
    nf_fd_run(n_head, n_head_kv, d_k, d_v, g_tile, n_split, split_len, kv_start, kv_end,
              q_pos, k_row_stride, k_head_stride, v_row_stride, v_head_stride,
              scale, softcap, max_bias, causal, swa_type, n_swa, bias_mode, n_buckets,
              max_distance, has_sinks, direct_out, q_off, out_off,
              q, k, v, kv_pos, sinks, rel_bias, out, partial,
              nf_attn_kv_type_k(kv_types), nf_attn_kv_type_v(kv_types), smem, red, stats);
}

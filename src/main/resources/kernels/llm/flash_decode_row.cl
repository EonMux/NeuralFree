/* flash_decode_row.cl - LLM_FlashDecodeRow_F16, the one-row decode attention over a dense F16 cache.
 * Requires: common/dialect.h, common/config.h, common/attn_common.h */
/*
 * CUDA-only twin of LLM_FlashDecode_F16 (flash_decode.cl) for the plain one-row step, with the same argument
 * list, grid, split partials and direct output, so the host swaps the kernel name and nothing else: the decode
 * patcher, the split plan (gguf.ops.attn.FlashAttnTiling.decodeSplit) and LLM_FlashAttn_Combine stay as they
 * are. The generic kernel serves every cache type, the paged and tree forms, sinks and the position biases in
 * one body, which compiles to tens of thousands of instructions and pays several dependent global round trips
 * before its first key; a one-row step over a short cache is bound by exactly that latency. This body keeps
 * only what the plain step needs, the shape of llama.cpp's fattn-vec kernel for one column:
 *
 *   Work-group (split s, head h): four warps over the split's keys [kv_start + s * split_len, + split_len)
 *   (clamped to kv_end), warp w taking keys w, w + 4, ...; lane l holds the query and output elements
 *   2 (32 c + l) and 2 (32 c + l) + 1 of every 64-element chunk c, so a key's K and V rows are d / 64
 *   coalesced 128-byte half2 loads per warp. Each warp issues the K row, the V row and the position of
 *   NF_FDR_KB keys before using the first, reduces each score by the 32-lane butterfly, and keeps an online
 *   softmax (running max m, sum l, output o) in registers: one rescale per batch. The four warps' states
 *   merge once through local memory at the end.
 *
 * Output as LLM_FlashDecode_F16: direct_out writes o / l (0 when l == 0) at out[out_off + h * d_v + d], else
 * the split partial partial[(h * n_split + s) * (d_v + 2) + d] = o with m at + d_v and l at + d_v + 1, an
 * empty split writing (0, -inf, 0).
 *
 * Plan constraints (gguf.ops.attn.FlashPlan selects this kernel only under them; a violation writes NaN
 * outputs): local size NF_FA_WG, g_tile == 1, d_k == d_v, d_k a multiple of 64 of at most NF_FDR_MAX_D,
 * even row and head strides, a dense cache (kv_pos[row] for cache row row), bias_mode NF_ATTN_BIAS_NONE, and
 * none of the sink, V-alias and tree flags. Softcap, sliding windows and kv_start > 0 are served.
 */

#if defined(__CUDACC__) && !defined(__HIP__)

#define NF_FDR_MAX_D 256                  /* largest head dim */
#define NF_FDR_MAX_C (NF_FDR_MAX_D / 64)  /* half2 chunks per lane */
#define NF_FDR_KB    4                    /* keys per warp in flight */
#define NF_FDR_WARPS (NF_FA_WG / 32)

/* One lane's chunk c of a row: the half2 at element 2 (32 c + lane) of an F16 row. */
NF_INLINE float2 nf_fdr_load(NF_GLOBAL const NF_HALF* row, int c, int lane) {
    const __half2 h = *(const __half2*)(row + 2 * (32 * c + lane));
    return __half22float2(h);
}

NF_INLINE float nf_fdr_warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        v += nf_sg_shuffle_xor(v, o);
    }
    return v;
}

/* NaN outputs of a launch outside the plan constraints (the whole launch's rows, from work-group 0 only). */
NF_INLINE void nf_fdr_poison(int n_head, int d_v, int n_split, int direct_out, int out_off,
                             NF_GLOBAL float* out, NF_GLOBAL float* partial) {
    if (get_group_id(0) != 0 || get_group_id(1) != 0 || n_head <= 0 || d_v <= 0 || n_split <= 0) {
        return;
    }
    const int lid = (int)get_local_id(0);
    if (direct_out) {
        for (int i = lid; i < n_head * d_v; i += NF_FA_WG) {
            out[(nf_i64)out_off + i] = NF_NAN;
        }
    } else {
        const nf_i64 n = (nf_i64)n_head * n_split * (d_v + 2);
        for (nf_i64 i = lid; i < n; i += NF_FA_WG) {
            partial[i] = NF_NAN;
        }
    }
}

NF_KERNEL void LLM_FlashDecodeRow_F16(
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
    NF_GLOBAL const NF_HALF* k,
    NF_GLOBAL const NF_HALF* v,
    NF_GLOBAL const int* NF_RESTRICT kv_pos,
    NF_GLOBAL const float* NF_RESTRICT sinks,
    NF_GLOBAL const float* NF_RESTRICT rel_bias,
    NF_GLOBAL float* out,
    NF_GLOBAL float* partial,
    NF_GLOBAL const float* NF_RESTRICT kv_scales,
    NF_GLOBAL const ulong* NF_RESTRICT tree_mask)
{
    NF_LOCAL float o_s[NF_FDR_WARPS * NF_FDR_MAX_D];
    NF_LOCAL float m_s[NF_FDR_WARPS];
    NF_LOCAL float l_s[NF_FDR_WARPS];

    const int direct_out = (flags & NF_ATTN_FLAG_DIRECT) != 0;
    const int unsupported = NF_ATTN_FLAG_SINKS | NF_ATTN_FLAG_V_ALIAS | NF_ATTN_FLAG_TREE;
    if (g_tile != 1 || d_k != d_v || d_k <= 0 || d_k % 64 != 0 || d_k > NF_FDR_MAX_D || n_head_kv <= 0
            || n_head % n_head_kv != 0 || split_len <= 0 || bias_mode != NF_ATTN_BIAS_NONE
            || (flags & unsupported) != 0 || ((k_row_stride | k_head_stride | v_row_stride | v_head_stride) & 1)
            || (direct_out && n_split != 1)) {
        nf_fdr_poison(n_head, d_v, n_split, direct_out, out_off, out, partial);
        return;   /* uniform: no barrier reached */
    }

    const int lid = (int)get_local_id(0);
    const int lane = lid & 31;
    const int w = lid >> 5;
    const int split = (int)get_group_id(0);
    const int h = (int)get_group_id(1);
    const int kvh = h / (n_head / n_head_kv);
    const int chunks = d_k >> 6;
    const int causal = flags & NF_ATTN_FLAG_CAUSAL;
    const int k_begin = kv_start + split * split_len;
    const int k_stop = split < n_split ? min(k_begin + split_len, kv_end) : k_begin;
    const int n_keys = max(0, k_stop - k_begin);

    float2 qv[NF_FDR_MAX_C];
    float2 ov[NF_FDR_MAX_C];
#pragma unroll
    for (int c = 0; c < NF_FDR_MAX_C; c++) {
        qv[c] = NF_F2(0.0f, 0.0f);
        ov[c] = NF_F2(0.0f, 0.0f);
        if (c < chunks) {
            const nf_i64 qi = (nf_i64)q_off + (nf_i64)h * d_k + 2 * (32 * c + lane);
            qv[c] = NF_F2(q[qi], q[qi + 1]);
        }
    }
    float m = -NF_INF;
    float l = 0.0f;
    NF_GLOBAL const NF_HALF* k_head = k + (nf_i64)kvh * k_head_stride;
    NF_GLOBAL const NF_HALF* v_head = v + (nf_i64)kvh * v_head_stride;

    for (int j0 = w; j0 < n_keys; j0 += NF_FDR_WARPS * NF_FDR_KB) {
        float2 kb[NF_FDR_KB][NF_FDR_MAX_C];
        float2 vb[NF_FDR_KB][NF_FDR_MAX_C];
        int pk[NF_FDR_KB];
#pragma unroll
        for (int b = 0; b < NF_FDR_KB; b++) {
            const int j = j0 + b * NF_FDR_WARPS;
            const int row = k_begin + j;
            pk[b] = j < n_keys ? kv_pos[row] : -1;
#pragma unroll
            for (int c = 0; c < NF_FDR_MAX_C; c++) {
                kb[b][c] = NF_F2(0.0f, 0.0f);
                vb[b][c] = NF_F2(0.0f, 0.0f);
                if (j < n_keys && c < chunks) {
                    kb[b][c] = nf_fdr_load(k_head + (nf_i64)row * k_row_stride, c, lane);
                    vb[b][c] = nf_fdr_load(v_head + (nf_i64)row * v_row_stride, c, lane);
                }
            }
        }
        float s[NF_FDR_KB];
        float mb = m;
#pragma unroll
        for (int b = 0; b < NF_FDR_KB; b++) {
            float dot = 0.0f;
#pragma unroll
            for (int c = 0; c < NF_FDR_MAX_C; c++) {
                dot += qv[c].x * kb[b][c].x + qv[c].y * kb[b][c].y;
            }
            dot = nf_fdr_warp_sum(dot);
            /* the key's visibility is uniform across the warp: one position per key */
            const int vis = nf_attn_visible(q_pos, pk[b], causal, swa_type, n_swa);
            s[b] = vis ? nf_attn_score(dot, scale, softcap) : -NF_INF;
            mb = fmax(mb, s[b]);
        }
        if (mb != -NF_INF) {
            const float corr = m == -NF_INF ? 0.0f : exp(m - mb);
            l *= corr;
#pragma unroll
            for (int c = 0; c < NF_FDR_MAX_C; c++) {
                ov[c].x *= corr;
                ov[c].y *= corr;
            }
#pragma unroll
            for (int b = 0; b < NF_FDR_KB; b++) {
                if (s[b] != -NF_INF) {
                    const float p = exp(s[b] - mb);
                    l += p;
#pragma unroll
                    for (int c = 0; c < NF_FDR_MAX_C; c++) {
                        ov[c].x += p * vb[b][c].x;
                        ov[c].y += p * vb[b][c].y;
                    }
                }
            }
            m = mb;
        }
    }

    /* merge the four warps' (m, l, o) */
#pragma unroll
    for (int c = 0; c < NF_FDR_MAX_C; c++) {
        if (c < chunks) {
            const int d = 2 * (32 * c + lane);
            o_s[w * NF_FDR_MAX_D + d] = ov[c].x;
            o_s[w * NF_FDR_MAX_D + d + 1] = ov[c].y;
        }
    }
    if (lane == 0) {
        m_s[w] = m;
        l_s[w] = l;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (split >= n_split) {
        return;   /* after the only barrier */
    }
    float mt = -NF_INF;
#pragma unroll
    for (int i = 0; i < NF_FDR_WARPS; i++) {
        mt = fmax(mt, m_s[i]);
    }
    float f[NF_FDR_WARPS];
    float lt = 0.0f;
#pragma unroll
    for (int i = 0; i < NF_FDR_WARPS; i++) {
        f[i] = m_s[i] == -NF_INF ? 0.0f : exp(m_s[i] - mt);
        lt += f[i] * l_s[i];
    }
    NF_GLOBAL float* pr = partial + ((nf_i64)h * n_split + split) * (d_v + 2);
    for (int d = lid; d < d_v; d += NF_FA_WG) {
        float o = 0.0f;
#pragma unroll
        for (int i = 0; i < NF_FDR_WARPS; i++) {
            o += f[i] * o_s[i * NF_FDR_MAX_D + d];
        }
        if (direct_out) {
            out[(nf_i64)out_off + (nf_i64)h * d_v + d] = lt > 0.0f ? o / lt : 0.0f;
        } else {
            pr[d] = o;
        }
    }
    if (!direct_out && lid == 0) {
        pr[d_v] = mt;
        pr[d_v + 1] = lt;
    }
}

#endif /* __CUDACC__ and not __HIP__: the OpenCL program, the HIP build and the Vulkan shader set define no row kernel */

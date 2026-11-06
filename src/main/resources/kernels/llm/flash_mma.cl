/* flash_mma.cl - LLM_FlashAttnMma_F16 and LLM_FlashAttnMmaPaged_F16 tensor-core tiled prefill attention.
 * Requires: common/dialect.h, common/config.h, common/math.h, common/attn_common.h */
/*
 * CUDA-only tensor-core twin of the tiled prefill kernel (flash_prefill.cl): the Q K^T and P V
 * products run on the matrix unit through mma.sync.aligned.m16n8k16 fragments, so long prompts,
 * speculative verification and batched decode reach matrix-unit throughput as llama.cpp's
 * fattn-mma-f16 does. Every other semantic matches LLM_FlashAttn_F16: same argument list, same
 * grid, same online softmax, split-K partials, GQA packing, sinks, softcap, ALiBi, T5 bias, SWA
 * and the paged KV addressing of flash_prefill.cl. K and V caches are F16; Q, the score state and
 * the output stay F32, P and the staged tiles are F16 (as in ggml-cuda/fattn-mma-f16.cuh). The
 * OpenCL program and the Vulkan shader set define no mma kernels: devices reached through them
 * have no matrix-unit access here, so the host runs the scalar kernels (gguf.ops.attn
 * .FlashAttnTiling.Plan.MMA is selected only on a backend that reports DeviceFeature
 * .COOPERATIVE_MATRIX and this kernel).
 *
 * Layouts (offsets in elements, 64-bit inside the kernel): as flash_prefill.cl, with
 *   q        q[q_off + t * q_row_stride + h * d_k + d], query position q_pos[t]
 *   K value  k[j * k_row_stride + kvh * k_head_stride + d] (F16), key position kv_pos[j]
 *   out      out[out_off + t * out_row_stride + h * d_v + d]                      (direct_out = 1)
 *   partial  partial[((t * n_head + h) * n_split + s) * (d_v + 2) + d], m at + d_v, l at + d_v + 1
 *
 * Grid: global {ceil(n_tokens / bq) * 128, n_head_kv * ceil(G / g_tile), n_split}, local {128, 1, 1}
 * with R = bq * g_tile == 64. Workgroup (b, y, s): four warps, warp w owns the 16 query rows
 * r = 16 w .. 16 w + 15 (token t = b * bq + r / g_tile, head h = h0 + r % g_tile,
 * h0 = kvh * G + gt * g_tile) and keys [kv_start + s * split_len, min(+ split_len, kv_end)) in
 * tiles of NF_FM_BN = 32.
 *
 * Shared memory: Q tile 64 x (d_k + 8) halves, K tile 32 x (d_k + 8) halves, V tile
 * d_v x (32 + 8) halves (transposed, key pairs at even halves), per-tile key positions.
 *
 * Per key tile: uniform skip of tiles invisible to every row; K and V staged to shared memory;
 * S = Q K^T accumulated on the mma over d_k in steps of 16; score, position bias and -INF masking
 * on the F32 fragments; online softmax per row (fragment max/sum reduced over the row's quad);
 * P converted in the fragments to F16 and O = O * alpha + P V accumulated on the mma over the
 * tile in two 16-key chunks. Finally rows apply sinks and 1 / l (l == 0 writes 0) or emit
 * (o, m, l) partials, as flash_prefill.cl does.
 *
 * Plan constraints (gguf.ops.attn.FlashAttnTiling.mma guarantees them; a violation writes NaN
 * outputs): d_k and d_v positive multiples of 16 of at most NF_FM_MAX_D, R == 64,
 * g_tile in 1..NF_FD_MAX_G, bq >= 1, split_len > 0, local size == NF_FM_WG, direct_out only with
 * n_split == 1, paged page_shift in 0..20.
 */

#if defined(__CUDACC__) && !defined(__HIP__)
/* Kernel bodies need mma.sync.m16n8k16 (compute capability 8.0+); NF_CUDA_PREPROCESS is the build's
 * nvcc -E inventory pass, which sees the kernels so the index and inventory list them. */
#if defined(NF_CUDA_PREPROCESS) || !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800

/* Tile shape of the mma kernel. */
#define NF_FM_WG     128   /* work-group size: four warps of 32 lanes */
#define NF_FM_BM     64    /* query rows per work-group (bq * g_tile == NF_FM_BM) */
#define NF_FM_BN     32    /* keys per tile */
#define NF_FM_D2     8     /* half2 padding columns of the Q and K tiles */
#define NF_FM_MAX_D  128   /* largest head dim (d_k and d_v) */
#define NF_FM_GROUPS 4     /* NF_FM_BN / 8: 16 x 8 D fragments of one score tile */

/* mma.sync.aligned.m16n8k16.f32.f16.f16.f32: D = A(16 x 16) * B(16 x 8) + C over F16 fragments,
 * F32 accumulate. Fragment layouts (lane 0..31, group = lane / 4, tig = lane % 4):
 *   A a0..a3 (.f16x2): a0 row group,      k 2 tig, 2 tig + 1;  a1 row group + 8, same k;
 *                      a2 row group,      k 2 tig + 8 + {0, 1}; a3 row group + 8, same k;
 *   B b0..b1 (.f16x2): b0 k 2 tig, 2 tig + 1, col group; b1 k 2 tig + 8 + {0, 1}, col group;
 *   C/D c0..c3 (F32):  c0 row group, col 2 tig; c1 row group, col 2 tig + 1;
 *                      c2 row group + 8, same cols.
 * The C fragment pairs map onto the A fragment pairs, so P is converted to the F16 A operand of
 * the P V product in the registers that hold it (probed against the reference layouts). */
NF_INLINE void nf_fm_mma(const unsigned& a0, const unsigned& a1, const unsigned& a2, const unsigned& a3,
                         const unsigned& b0, const unsigned& b1,
                         float& d0, float& d1, float& d2, float& d3) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
          "f"(d0), "f"(d1), "f"(d2), "f"(d3));
}

/* F16 pair of a shared-memory half row as one fragment register. */
NF_INLINE unsigned nf_fm_h2(NF_LOCAL_PTR const NF_HALF* p) {
    __half2 v;
    v.x = __ushort_as_half(p[0]);
    v.y = __ushort_as_half(p[1]);
    return *reinterpret_cast<const unsigned*>(&v);
}

/* F16 pair of two F32 probabilities, in the fragment register order. */
NF_INLINE unsigned nf_fm_p2(float a, float b) {
    __half2 v;
    v.x = __float2half_rn(a);
    v.y = __float2half_rn(b);
    return *reinterpret_cast<unsigned*>(&v);
}

/* Plan constraints of the mma kernels; see the header comment. */
NF_INLINE int nf_fm_plan_ok(int n_head, int n_head_kv, int d_k, int d_v, int bq, int g_tile,
                            int n_split, int split_len, int direct_out) {
    return n_head > 0 && n_head_kv > 0 && n_head % n_head_kv == 0
        && g_tile >= 1 && g_tile <= NF_FD_MAX_G && bq >= 1 && bq * g_tile == NF_FM_BM
        && d_k > 0 && d_v > 0 && d_k <= NF_FM_MAX_D && d_v <= NF_FM_MAX_D
        && (d_k & 15) == 0 && (d_v & 15) == 0
        && n_split >= 1 && split_len > 0 && (direct_out == 0 || n_split == 1)
        && (int)get_local_size(0) == NF_FM_WG;
}

/* Shared-memory floats the mma kernel uses for d_k and d_v (FaSmem counts 4-byte floats). */
NF_INLINE long nf_fm_smem_floats(int d_k, int d_v) {
    long halves = (long)NF_FM_BM * (d_k + NF_FM_D2) + (long)NF_FM_BN * (d_k + NF_FM_D2)
        + (long)d_v * (NF_FM_BN + NF_FM_D2);
    return (halves + 2 * NF_FM_BN + NF_FM_WG) / 2;
}

/* Writes NaN over the outputs of a call whose arguments violate the plan (block (0, 0, 0) only). */
NF_INLINE void nf_fm_poison(int n_tokens, int n_head, int d_v, int n_split, int out_row_stride,
                            int direct_out, int out_off, NF_GLOBAL float* out, NF_GLOBAL float* partial) {
    if (blockIdx.x != 0 || blockIdx.y != 0 || blockIdx.z != 0
        || n_tokens <= 0 || n_head <= 0 || d_v <= 0 || n_split <= 0) {
        return;
    }
    const int tid = (int)threadIdx.x;
    const int nt = (int)blockDim.x;
    if (direct_out) {
        for (int t = 0; t < n_tokens; t++) {
            for (int i = tid; i < n_head * d_v; i += nt) {
                out[(nf_i64)out_off + (nf_i64)t * out_row_stride + i] = NF_NAN;
            }
        }
    } else {
        const nf_i64 n = (nf_i64)n_tokens * n_head * n_split * (d_v + 2);
        for (nf_i64 i = tid; i < n; i += nt) {
            partial[i] = NF_NAN;
        }
    }
}

/*
 * The mma tiled attention body; the arguments are those of nf_fa_run (flash_prefill.cl). k_type and
 * v_type are NF_KV_F16, Q positions come from q_pos, cached keys from kv_pos over the page map.
 */
NF_INLINE void nf_fm_run(
    const int n_tokens, const int n_head, const int n_head_kv, const int d_k, const int d_v,
    const int bq, const int bk, const int g_tile, const int n_split, const int split_len,
    const int kv_start, const int kv_end,
    const int q_row_stride, const int out_row_stride,
    const int k_row_stride, const int k_head_stride, const int v_row_stride, const int v_head_stride,
    const float scale, const float softcap, const float max_bias,
    const int causal, const int swa_type, const int n_swa,
    const int bias_mode, const int n_buckets, const int max_distance,
    const int has_sinks, const int direct_out, const int q_off, const int out_off,
    NF_GLOBAL const float* q, NF_GLOBAL const NF_HALF* k, NF_GLOBAL const NF_HALF* v,
    NF_GLOBAL const int* NF_RESTRICT q_pos, NF_GLOBAL const int* NF_RESTRICT kv_pos,
    NF_GLOBAL const float* NF_RESTRICT sinks, NF_GLOBAL const float* NF_RESTRICT rel_bias,
    NF_GLOBAL float* out, NF_GLOBAL float* partial,
    const int page_shift, NF_GLOBAL const int* NF_RESTRICT pt, const int pt_off,
    const int tree, NF_GLOBAL const ulong* NF_RESTRICT tree_mask, const int n_fresh)
{
    if (!nf_fm_plan_ok(n_head, n_head_kv, d_k, d_v, bq, g_tile, n_split, split_len, direct_out)
        || bk != NF_FM_BN) {
        nf_fm_poison(n_tokens, n_head, d_v, n_split, out_row_stride, direct_out, out_off, out, partial);
        return;   /* uniform across the work-group: no barrier has been reached */
    }

    NF_LOCAL NF_HALF q_s[NF_FM_BM * (NF_FM_MAX_D + NF_FM_D2)];
    NF_LOCAL NF_HALF k_s[NF_FM_BN * (NF_FM_MAX_D + NF_FM_D2)];
    NF_LOCAL NF_HALF v_s[NF_FM_MAX_D * (NF_FM_BN + NF_FM_D2)];
    NF_LOCAL int kp_s[NF_FM_BN];
    NF_LOCAL int skip_s;

    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int group = lane >> 2;
    const int tig = lane & 3;
    const int split = (int)blockIdx.z;
    const int G = n_head / n_head_kv;
    const int n_gt = (G + g_tile - 1) / g_tile;
    const int kvh = (int)blockIdx.y / n_gt;
    const int gt = (int)blockIdx.y - kvh * n_gt;
    const int h0 = kvh * G + gt * g_tile;
    const int t0 = (int)blockIdx.x * bq;
    const int k_begin = kv_start + split * split_len;
    const int k_stop = split < n_split ? min(k_begin + split_len, kv_end) : k_begin;
    const int kd2 = d_k >> 1;

    /* Rows this lane's fragment pairs cover and their (token, head) identity. */
    const int r0 = warp * 16 + group;
    const int r1 = r0 + 8;
    const int i0 = r0 / g_tile, g0 = r0 - i0 * g_tile;
    const int i1 = r1 / g_tile, g1 = r1 - i1 * g_tile;
    const int tq0 = t0 + i0, tq1 = t0 + i1;
    const int hq0 = h0 + g0, hq1 = h0 + g1;
    const int ok0 = tq0 < n_tokens && g0 < min(g_tile, G - gt * g_tile);
    const int ok1 = tq1 < n_tokens && g1 < min(g_tile, G - gt * g_tile);
    const int pq0 = ok0 ? q_pos[tq0] : -1;
    const int pq1 = ok1 ? q_pos[tq1] : -1;
    const float slope0 = (bias_mode == NF_ATTN_BIAS_ALIBI || bias_mode == NF_ATTN_BIAS_ALIBI_SQRT) && ok0
        ? nf_alibi_slope(hq0, n_head, max_bias) : 0.0f;
    const float slope1 = (bias_mode == NF_ATTN_BIAS_ALIBI || bias_mode == NF_ATTN_BIAS_ALIBI_SQRT) && ok1
        ? nf_alibi_slope(hq1, n_head, max_bias) : 0.0f;

    /* 0. Stage the Q tile as F16 (invalid rows are zeros; their outputs are never written). */
    for (int idx = tid; idx < NF_FM_BM * (d_k / 4); idx += NF_FM_WG) {
        const int r = idx / (d_k / 4);
        const int c4 = idx - r * (d_k / 4);
        const int i = r / g_tile;
        const int g = r - i * g_tile;
        const int tq = t0 + i;
        const int hg = h0 + g;
        float4 f = NF_F4S(0.0f);
        if (tq < n_tokens && g < min(g_tile, G - gt * g_tile)) {
            /* d_k is a positive multiple of 16, so the 4-wide loads never leave the row. */
            f = vload4(0, q + (nf_i64)q_off + (nf_i64)tq * q_row_stride + (nf_i64)hg * d_k + 4 * c4);
        }
        NF_LOCAL_PTR NF_HALF* dst = q_s + r * (d_k + NF_FM_D2) + 4 * c4;
        dst[0] = __half_as_ushort(__float2half_rn(f.x));
        dst[1] = __half_as_ushort(__float2half_rn(f.y));
        dst[2] = __half_as_ushort(__float2half_rn(f.z));
        dst[3] = __half_as_ushort(__float2half_rn(f.w));
    }
    __syncthreads();

    /* Running softmax state: this lane's rows (r0, r1). */
    float m0 = -NF_INF, m1 = -NF_INF, l0 = 0.0f, l1 = 0.0f;
    /* Output accumulators: per 8-column group, this lane holds rows r0 (c0, c1) and r1 (c2, c3). */
    float vkq[NF_FM_MAX_D / 8][4];
#pragma unroll
    for (int dg = 0; dg < NF_FM_MAX_D / 8; dg++) {
        vkq[dg][0] = vkq[dg][1] = vkq[dg][2] = vkq[dg][3] = 0.0f;
    }
    const int vgroups = d_v >> 3;

    const nf_i64 k_head = (nf_i64)kvh * k_head_stride;
    const nf_i64 v_head = (nf_i64)kvh * v_head_stride;
    for (int kt = k_begin; kt < k_stop; kt += NF_FM_BN) {
        const int nk = min(NF_FM_BN, k_stop - kt);

        /* 1. Uniform skip of tiles invisible to every row (as flash_prefill.cl step 0). */
        int any = 0;
        for (int idx = tid; idx < NF_FM_BM * NF_FM_BN; idx += NF_FM_WG) {
            const int r = idx / NF_FM_BN;
            const int j = idx - r * NF_FM_BN;
            const int i = r / g_tile;
            const int g = r - i * g_tile;
            if (t0 + i < n_tokens && g < min(g_tile, G - gt * g_tile) && j < nk
                && nf_kv_page_ok(kt + j, page_shift, pt, pt_off)
                && nf_attn_tree_visible(kv_pos[nf_kv_row(kt + j, page_shift, pt, pt_off)],
                                        tree ? tree_mask[t0 + i] : (ulong)0, kv_end, n_fresh,
                                        q_pos[t0 + i], causal, swa_type, n_swa)) {
                any = 1;
            }
        }
        skip_s = __syncthreads_or(any);
        if (!skip_s) {
            continue;
        }

        /* 2. Stage the key positions and the K tile (F16 rows; missing keys are zeros and their
         *    positions -1, so they are masked like empty cache rows). */
        if (tid < NF_FM_BN) {
            const int slot = kt + tid;
            kp_s[tid] = tid < nk && nf_kv_page_ok(slot, page_shift, pt, pt_off)
                ? kv_pos[nf_kv_row(slot, page_shift, pt, pt_off)] : -1;
        }
        for (int idx = tid; idx < NF_FM_BN * kd2; idx += NF_FM_WG) {
            const int j = idx / kd2;
            const int c2 = idx - j * kd2;
            float2 f = NF_F2(0.0f, 0.0f);
            if (j < nk && nf_kv_page_ok(kt + j, page_shift, pt, pt_off)) {
                const nf_i64 row = nf_kv_row(kt + j, page_shift, pt, pt_off);
                f.x = vload_half(0, k + row * k_row_stride + k_head + 2 * c2);
                f.y = vload_half(0, k + row * k_row_stride + k_head + 2 * c2 + 1);
            }
            NF_LOCAL_PTR NF_HALF* dst = k_s + j * (d_k + NF_FM_D2) + 2 * c2;
            dst[0] = __half_as_ushort(__float2half_rn(f.x));
            dst[1] = __half_as_ushort(__float2half_rn(f.y));
        }
        for (int idx = tid; idx < d_v * (NF_FM_BN >> 1); idx += NF_FM_WG) {
            const int d = idx / (NF_FM_BN >> 1);
            const int p = idx - d * (NF_FM_BN >> 1);
            float2 f = NF_F2(0.0f, 0.0f);
            if (2 * p < nk) {
                if (nf_kv_page_ok(kt + 2 * p, page_shift, pt, pt_off)) {
                    const nf_i64 row0 = nf_kv_row(kt + 2 * p, page_shift, pt, pt_off);
                    f.x = vload_half(0, v + row0 * v_row_stride + v_head + d);
                }
                if (2 * p + 1 < nk && nf_kv_page_ok(kt + 2 * p + 1, page_shift, pt, pt_off)) {
                    const nf_i64 row1 = nf_kv_row(kt + 2 * p + 1, page_shift, pt, pt_off);
                    f.y = vload_half(0, v + row1 * v_row_stride + v_head + d);
                }
            }
            /* key pairs at even half positions: v_s[d][h] is V[h][d] */
            NF_LOCAL_PTR NF_HALF* dst = v_s + d * (NF_FM_BN + NF_FM_D2) + 2 * p;
            dst[0] = __half_as_ushort(__float2half_rn(f.x));
            dst[1] = __half_as_ushort(__float2half_rn(f.y));
        }
        __syncthreads();

        /* 3. S = Q K^T on the mma, then score, bias and mask on the F32 fragments. */
        float s[NF_FM_GROUPS][4];
#pragma unroll
        for (int grp = 0; grp < NF_FM_GROUPS; grp++) {
            s[grp][0] = s[grp][1] = s[grp][2] = s[grp][3] = 0.0f;
        }
        for (int ks = 0; ks < d_k; ks += 16) {
#pragma unroll
            for (int grp = 0; grp < NF_FM_GROUPS; grp++) {
                /* B fragment: K^T k 16 x 8 keys of this group, from the K tile rows. */
                NF_LOCAL_PTR const NF_HALF* krow = k_s + (grp * 8 + group) * (d_k + NF_FM_D2);
                const unsigned b0 = nf_fm_h2(krow + ks + 2 * tig);
                const unsigned b1 = nf_fm_h2(krow + ks + 2 * tig + 8);
                /* A fragment: the 16 rows of this warp, Q columns ks..ks + 15. */
                const unsigned a0 = nf_fm_h2(q_s + (r0) * (d_k + NF_FM_D2) + ks + 2 * tig);
                const unsigned a1 = nf_fm_h2(q_s + (r1) * (d_k + NF_FM_D2) + ks + 2 * tig);
                const unsigned a2 = nf_fm_h2(q_s + (r0) * (d_k + NF_FM_D2) + ks + 2 * tig + 8);
                const unsigned a3 = nf_fm_h2(q_s + (r1) * (d_k + NF_FM_D2) + ks + 2 * tig + 8);
                nf_fm_mma(a0, a1, a2, a3, b0, b1,
                          s[grp][0], s[grp][1], s[grp][2], s[grp][3]);
            }
        }
#pragma unroll
        for (int grp = 0; grp < NF_FM_GROUPS; grp++) {
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const int g = e < 2 ? g0 : g1;
                const int pq = e < 2 ? pq0 : pq1;
                const int tq = e < 2 ? tq0 : tq1;
                const int j = grp * 8 + 2 * tig + (e & 1);
                float sv = -NF_INF;
                if (j < nk && pq >= 0 && kp_s[j] >= 0
                    && nf_attn_tree_visible(kp_s[j], tree ? tree_mask[tq] : (ulong)0, kv_end, n_fresh,
                                            pq, causal, swa_type, n_swa)) {
                    const float slope = e < 2 ? slope0 : slope1;
                    sv = nf_attn_score(s[grp][e], scale, softcap)
                       + nf_attn_bias(bias_mode, h0 + g, n_head, pq, kp_s[j], slope, rel_bias,
                                      n_buckets, max_distance, tq);
                }
                s[grp][e] = sv;
            }
        }

        /* 4. Online softmax per row: the row's fragment values live in the lanes of its quad. */
        float rm0 = -NF_INF, rm1 = -NF_INF, rs0 = 0.0f, rs1 = 0.0f;
#pragma unroll
        for (int grp = 0; grp < NF_FM_GROUPS; grp++) {
            rm0 = fmaxf(rm0, fmaxf(s[grp][0], s[grp][1]));
            rm1 = fmaxf(rm1, fmaxf(s[grp][2], s[grp][3]));
        }
#pragma unroll
        for (int o = 1; o <= 2; o <<= 1) {
            rm0 = fmaxf(rm0, __shfl_xor_sync(0xffffffffu, rm0, o));
            rm1 = fmaxf(rm1, __shfl_xor_sync(0xffffffffu, rm1, o));
        }
        const float mn0 = fmaxf(m0, rm0);
        const float mn1 = fmaxf(m1, rm1);
        const float al0 = nf_attn_rescale(m0, mn0);
        const float al1 = nf_attn_rescale(m1, mn1);
#pragma unroll
        for (int grp = 0; grp < NF_FM_GROUPS; grp++) {
            s[grp][0] = nf_attn_rescale(s[grp][0], mn0);
            s[grp][1] = nf_attn_rescale(s[grp][1], mn0);
            s[grp][2] = nf_attn_rescale(s[grp][2], mn1);
            s[grp][3] = nf_attn_rescale(s[grp][3], mn1);
            rs0 += s[grp][0] + s[grp][1];
            rs1 += s[grp][2] + s[grp][3];
        }
#pragma unroll
        for (int o = 1; o <= 2; o <<= 1) {
            rs0 += __shfl_xor_sync(0xffffffffu, rs0, o);
            rs1 += __shfl_xor_sync(0xffffffffu, rs1, o);
        }
        l0 = l0 * al0 + rs0;
        l1 = l1 * al1 + rs1;
        m0 = mn0;
        m1 = mn1;

        /* 5. O = O * alpha + P V: P converts in the fragments to the F16 A operand. */
#pragma unroll
        for (int dg = 0; dg < NF_FM_MAX_D / 8; dg++) {
            if (dg < vgroups) {
                vkq[dg][0] *= al0;
                vkq[dg][1] *= al0;
                vkq[dg][2] *= al1;
                vkq[dg][3] *= al1;
            }
        }
#pragma unroll
        for (int c = 0; c < NF_FM_GROUPS / 2; c++) {
            /* A fragment of P: k chunk c covers key groups 2 c and 2 c + 1. */
            const unsigned a0 = nf_fm_p2(s[2 * c][0], s[2 * c][1]);
            const unsigned a1 = nf_fm_p2(s[2 * c][2], s[2 * c][3]);
            const unsigned a2 = nf_fm_p2(s[2 * c + 1][0], s[2 * c + 1][1]);
            const unsigned a3 = nf_fm_p2(s[2 * c + 1][2], s[2 * c + 1][3]);
#pragma unroll
            for (int dg = 0; dg < NF_FM_MAX_D / 8; dg++) {
                if (dg < vgroups) {
                    /* B fragment: V of the chunk's 16 keys at columns 8 dg..8 dg + 7, from the
                     * transposed tile row of the group's column (key pairs at even halves). */
                    NF_LOCAL_PTR const NF_HALF* vrow = v_s + (dg * 8 + group) * (NF_FM_BN + NF_FM_D2);
                    const unsigned b0 = nf_fm_h2(vrow + 2 * (c * 8 + tig));
                    const unsigned b1 = nf_fm_h2(vrow + 2 * (c * 8 + tig + 4));
                    nf_fm_mma(a0, a1, a2, a3, b0, b1,
                              vkq[dg][0], vkq[dg][1], vkq[dg][2], vkq[dg][3]);
                }
            }
        }
        __syncthreads();
    }

    /* 6. Output: normalize with the sink, or emit (o, m, l) partials — as flash_prefill.cl step 5. */
    if (!direct_out) {
#pragma unroll
        for (int dg = 0; dg < NF_FM_MAX_D / 8; dg++) {
            if (dg >= vgroups) {
                continue;
            }
            const int d = dg * 8 + 2 * tig;
            if (ok0 && split < n_split) {
                NF_GLOBAL float* pp = partial
                    + (((nf_i64)tq0 * n_head + hq0) * n_split + split) * (d_v + 2);
                pp[d] = vkq[dg][0];
                pp[d + 1] = vkq[dg][1];
                if (tig == 0 && d == 0) {
                    pp[d_v] = m0;
                    pp[d_v + 1] = l0;
                }
            }
            if (ok1 && split < n_split) {
                NF_GLOBAL float* pp = partial
                    + (((nf_i64)tq1 * n_head + hq1) * n_split + split) * (d_v + 2);
                pp[d] = vkq[dg][2];
                pp[d + 1] = vkq[dg][3];
                if (tig == 0 && d == 0) {
                    pp[d_v] = m1;
                    pp[d_v + 1] = l1;
                }
            }
        }
        return;
    }
#pragma unroll
    for (int dg = 0; dg < NF_FM_MAX_D / 8; dg++) {
        if (dg >= vgroups) {
            continue;
        }
        const int d = dg * 8 + 2 * tig;
        if (ok0) {
            const float2 fl = has_sinks ? nf_attn_sink(m0, l0, sinks[hq0]) : NF_F2(1.0f, l0);
            NF_GLOBAL float* o = out + (nf_i64)out_off + (nf_i64)tq0 * out_row_stride
                + (nf_i64)hq0 * d_v + d;
            o[0] = fl.y > 0.0f ? vkq[dg][0] * fl.x / fl.y : 0.0f;
            o[1] = fl.y > 0.0f ? vkq[dg][1] * fl.x / fl.y : 0.0f;
        }
        if (ok1) {
            const float2 fl = has_sinks ? nf_attn_sink(m1, l1, sinks[hq1]) : NF_F2(1.0f, l1);
            NF_GLOBAL float* o = out + (nf_i64)out_off + (nf_i64)tq1 * out_row_stride
                + (nf_i64)hq1 * d_v + d;
            o[0] = fl.y > 0.0f ? vkq[dg][2] * fl.x / fl.y : 0.0f;
            o[1] = fl.y > 0.0f ? vkq[dg][3] * fl.x / fl.y : 0.0f;
        }
    }
}

/*
 * LLM_FlashAttnMma_F16(29 scalars, 10 buffers): the LLM_FlashAttn_F16 arguments; K and V caches are
 * F16. Grid and per-step arguments are those of LLM_FlashAttn_F16 with bq * g_tile == 64. Sinks and
 * rel_bias are read-only placeholders when unused.
 */
NF_KERNEL void LLM_FlashAttnMma_F16(
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
    NF_GLOBAL const NF_HALF* k,
    NF_GLOBAL const NF_HALF* v,
    NF_GLOBAL const int* NF_RESTRICT q_pos,
    NF_GLOBAL const int* NF_RESTRICT kv_pos,
    NF_GLOBAL const float* NF_RESTRICT sinks,
    NF_GLOBAL const float* NF_RESTRICT rel_bias,
    NF_GLOBAL float* out,
    NF_GLOBAL float* partial,
    NF_GLOBAL const ulong* NF_RESTRICT tree_mask)
{
    nf_fm_run(n_tokens, n_head, n_head_kv, d_k, d_v, bq, bk, g_tile, n_split, split_len,
              kv_start, kv_end, q_row_stride, out_row_stride, k_row_stride, k_head_stride,
              v_row_stride, v_head_stride, scale, softcap, max_bias,
              flags & NF_ATTN_FLAG_CAUSAL, swa_type, n_swa, bias_mode, n_buckets,
              max_distance, flags & NF_ATTN_FLAG_SINKS, flags & NF_ATTN_FLAG_DIRECT, q_off,
              out_off, q, k, v, q_pos, kv_pos, sinks, rel_bias, out, partial, -1, kv_pos, 0,
              flags & NF_ATTN_FLAG_TREE, tree_mask, nf_attn_flag_fresh(flags));
}

/*
 * LLM_FlashAttnMmaPaged_F16(31 scalars, 11 buffers): the LLM_FlashAttnPaged_F16 arguments; K and V
 * caches are F16, K, V and kv_pos are read at the physical row nf_kv_row (common/attn_common.h).
 */
NF_KERNEL void LLM_FlashAttnMmaPaged_F16(
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
    NF_GLOBAL const NF_HALF* k,
    NF_GLOBAL const NF_HALF* v,
    NF_GLOBAL const int* NF_RESTRICT q_pos,
    NF_GLOBAL const int* NF_RESTRICT kv_pos,
    NF_GLOBAL const float* NF_RESTRICT sinks,
    NF_GLOBAL const float* NF_RESTRICT rel_bias,
    NF_GLOBAL float* out,
    NF_GLOBAL float* partial,
    NF_GLOBAL const int* NF_RESTRICT page_table,
    NF_GLOBAL const ulong* NF_RESTRICT tree_mask)
{
    if (page_shift < 0 || page_shift > 20 || pt_off < 0) {
        nf_fm_poison(n_tokens, n_head, d_v, n_split, out_row_stride,
                     flags & NF_ATTN_FLAG_DIRECT, out_off, out, partial);
        return;   /* uniform across the work-group: no barrier has been reached */
    }
    nf_fm_run(n_tokens, n_head, n_head_kv, d_k, d_v, bq, bk, g_tile, n_split, split_len,
              kv_start, kv_end, q_row_stride, out_row_stride, k_row_stride, k_head_stride,
              v_row_stride, v_head_stride, scale, softcap, max_bias,
              flags & NF_ATTN_FLAG_CAUSAL, swa_type, n_swa, bias_mode, n_buckets,
              max_distance, flags & NF_ATTN_FLAG_SINKS, flags & NF_ATTN_FLAG_DIRECT, q_off,
              out_off, q, k, v, q_pos, kv_pos, sinks, rel_bias, out, partial, page_shift,
              page_table, pt_off, flags & NF_ATTN_FLAG_TREE, tree_mask, nf_attn_flag_fresh(flags));
}

#endif /* NF_CUDA_PREPROCESS, or mma.sync.m16n8k16 on compute capability 8.0+ */

#endif /* __CUDACC__ and not __HIP__: the OpenCL program, the HIP build and the Vulkan shader set define no mma kernels */

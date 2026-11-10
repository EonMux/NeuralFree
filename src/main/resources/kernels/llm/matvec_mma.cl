/* matvec_mma.cl - {T}_MatVecMma{8,16}_Q8: the int8 tensor-core multi-column matvec of small decode batches.
 * Requires: common/dialect.h, common/config.h, common/quant_blocks.h */
/*
 * CUDA-only (sm_80+) projection of 2..16 token rows through a quantized weight, the decode-batch shape
 * llama.cpp serves with its mmq_x = 8 / 16 tiles: every weight byte is read from DRAM exactly once, straight
 * into the mma.sync A fragments (no shared-memory staging, no barrier in the stream), and every token's Q8_1
 * activation block is the B fragment of the same instruction, so the arithmetic per weight byte stays one
 * int8 MAC per token on the tensor cores and the kernel runs at the weight-stream rate for every count.
 *
 *   {T}_MatVecMma{NT}_Q8(i rows, i cols, i n_tokens, R uchar weight, R uchar xq, W float out)
 *   out[t * rows + r] = sum_k W[r][k] * x[t][k],  r < rows, t < n_tokens <= NT
 *
 * xq holds n_tokens consecutive Q8_1 rows (ggml block_q8_1: d f16 @0, s = d * sum(qs) f16 @2, qs[32] @4;
 * cols / 32 blocks per row), the LLM_QuantizeQ8_1 output the counted matvec lane reads. T is Q8_0, Q4_K, Q5_K
 * or Q6_K; NT is 8 or 16 (one or two n8 token tiles).
 *
 * Geometry: global {ceil(rows / (16 * MT)) * L}, local {L}, L = 32 * nw with nw in 1..NF_MVM_MAX_WARPS. A
 * work-group owns the 16 * MT rows [r0, r0 + 16 * MT) (MT m16 tiles); its nw warps split the reduction
 * into contiguous step ranges (a step is the type's load unit below) and the work-group sums the warps'
 * partial tiles through shared memory at the end, so a row's reduction order is fixed by (cols, nw) and
 * never by the token count. Rows past `rows` load the last row (clamped, never stored); tokens past
 * n_tokens load zero activations.
 *
 * Fragments (PTX mma.m16n8k32 / m16n8k16 .s8, lane = 4 * g + q): A registers hold rows g and g + 8 of the
 * m16 tile, B registers token g of the n8 tile, C/D rows g (+8) at tokens 2q, 2q + 1. The dot over a
 * 32-element group is order-free, so each type picks the lane-to-element map that makes its weight bytes
 * contiguous per lane, and the B fragment follows the same map:
 *   - Q8_0, Q4_K, Q5_K: lane q takes elements [8q, 8q + 8) of the group (one 8-byte run of Q8_0 codes, the low
 *     or high nibbles of one 8-byte Q4_K or Q5_K run, a Q5_K code's fifth bit from the qh bytes [8q, 8q + 8)),
 *     so B is the 8 bytes [8q, 8q + 8) of the Q8_1 block.
 *   - Q6_K: one k16 product per 16-element scale group, lane q takes elements [4q, 4q + 4) and
 *     [16 + 4q, 16 + 4q + 4) (the standard layout), so B is the words at 4q and 16 + 4q.
 * Scales apply per group after the int32 product: Q8_0 c * dw * dx, Q4_K and Q5_K c * (d sc) * dx - (dmin m) * s,
 * Q6_K (c0 sc0 + c1 sc1) * d * dx with the integer scale sums exact in int32.
 *
 * Steps: Q8_0 two blocks (64 elements; a block pair is word aligned, its first block's codes start two
 * bytes past a word boundary, so they are rebuilt with a funnel shift), the K-quants half a super-block
 * (128 elements). Host plan constraints (gguf.ops.gemm.MatVecMmaPlan): cols a multiple of 64 (Q8_0) or 512
 * (Q4_K, Q5_K, Q6_K: an even block count keeps every row word aligned), 2 <= n_tokens <= NT, the weight rows
 * contiguous at cols / BS * BB bytes. A violation writes NaN over the work-group's outputs.
 */

#if defined(__CUDACC__) && !defined(__HIP__)
/* The s8 mma.sync fragments need compute capability 8.0+; NF_CUDA_PREPROCESS is the build's nvcc -E
 * inventory pass, which sees the kernels so the index and inventory list them. */
#if defined(NF_CUDA_PREPROCESS) || !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800

#define NF_MVM_MAX_WARPS 8
#define NF_MVM_Q8_0 0
#define NF_MVM_Q4_K 1
#define NF_MVM_Q6_K 2
#define NF_MVM_Q5_K 3

/* d = A * B over one 16 x 8 x 32 int8 tile, zero accumulator. */
NF_INLINE void nf_mvm_k32(int d[4], uint a0, uint a1, uint a2, uint a3, uint b0, uint b1) {
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(0));
}

/* d = A * B over one 16 x 8 x 16 int8 tile, zero accumulator. */
NF_INLINE void nf_mvm_k16(int d[4], uint a0, uint a1, uint b0) {
    asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%7,%7,%7};"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a0), "r"(a1), "r"(b0), "r"(0));
}

/* Aligned 32-bit word. */
NF_INLINE uint nf_mvm_w(NF_GLOBAL const uchar* p) {
    return *(NF_GLOBAL const uint*)p;
}

/* The 32-bit word at p, a two-byte aligned address whose alignment `odd` (p % 4 == 2) is uniform across
 * the warp: one load when aligned, else the two covering words and a funnel shift (no byte past the word). */
NF_INLINE uint nf_mvm_w2(NF_GLOBAL const uchar* p, int odd) {
    if (!odd) {
        return nf_mvm_w(p);
    }
    NF_GLOBAL const uint* w = (NF_GLOBAL const uint*)(p - 2);
    return __funnelshift_r(w[0], w[1], 16);
}

NF_INLINE float nf_mvm_h(uint bits) {
    return __half2float(__ushort_as_half((ushort)bits));
}

/*
 * One step's weight registers of one row for this lane (lane q = tq):
 *   Q8_0: w[0..1] block 2p codes [8q, 8q+8), w[2..3] block 2p+1 codes, w[4] the two block scales (f16 pair)
 *   Q4_K: w[0..3] the two 8-byte runs [16 + 32j + 8q] of the half's sub-block pairs j = 2h, 2h + 1,
 *         w[4..6] the 12 scale bytes and w[7] (d, dmin)
 *   Q5_K: w[0..7] as Q4_K from the code runs [48 + 32j + 8q], w[8..9] the qh bytes [16 + 8q, 16 + 8q + 8)
 *   Q6_K: w[0..3] ql words at 64h + {0, 16, 32, 48} + 4q, w[4..5] qh words at 128 + 32h + {0, 16} + 4q,
 *         w[6..7] the half's eight int8 scales at 192 + 8h, w[8] d (f16 bits)
 */
#define NF_MVM_WREGS 10

template <int T>
NF_INLINE void nf_mvm_load(NF_GLOBAL const uchar* row, int step, int tq, uint w[NF_MVM_WREGS]) {
    if (T == NF_MVM_Q8_0) {
        NF_GLOBAL const uchar* p = row + step * 68;           /* block pair, word aligned */
        const uint w0 = nf_mvm_w(p + 8 * tq);
        const uint w1 = nf_mvm_w(p + 8 * tq + 4);
        const uint w2 = nf_mvm_w(p + 8 * tq + 8);
        w[0] = __funnelshift_r(w0, w1, 16);                   /* codes start at byte 2 */
        w[1] = __funnelshift_r(w1, w2, 16);
        w[2] = nf_mvm_w(p + 36 + 8 * tq);                     /* second block's codes, aligned */
        w[3] = nf_mvm_w(p + 40 + 8 * tq);
        const uint s0 = nf_mvm_w(p);                          /* d0 | codes 0, 1 */
        const uint s1 = nf_mvm_w(p + 32);                     /* codes 30, 31 | d1 */
        w[4] = (s0 & 0xFFFFu) | (s1 & 0xFFFF0000u);
    } else if (T == NF_MVM_Q4_K || T == NF_MVM_Q5_K) {
        const int q5 = T == NF_MVM_Q5_K;
        NF_GLOBAL const uchar* b = row + (step >> 1) * (q5 ? 176 : 144);
        const int qs = q5 ? 48 : 16;
        const int h = step & 1;
        const uint2 r0 = *(NF_GLOBAL const uint2*)(b + qs + 64 * h + 8 * tq);
        const uint2 r1 = *(NF_GLOBAL const uint2*)(b + qs + 32 + 64 * h + 8 * tq);
        const uint4 hd = *(NF_GLOBAL const uint4*)b;
        w[0] = r0.x; w[1] = r0.y; w[2] = r1.x; w[3] = r1.y;
        w[4] = hd.y; w[5] = hd.z; w[6] = hd.w; w[7] = hd.x;
        if (q5) {
            const uint2 qh = *(NF_GLOBAL const uint2*)(b + 16 + 8 * tq);
            w[8] = qh.x; w[9] = qh.y;
        }
    } else {
        NF_GLOBAL const uchar* b = row + (step >> 1) * 210;
        const int h = step & 1;
        const int odd = (step >> 1) & 1;                     /* block 2i+1 starts two bytes past a word */
        NF_GLOBAL const uchar* ql = b + 64 * h + 4 * tq;
        w[0] = nf_mvm_w2(ql, odd);
        w[1] = nf_mvm_w2(ql + 16, odd);
        w[2] = nf_mvm_w2(ql + 32, odd);
        w[3] = nf_mvm_w2(ql + 48, odd);
        w[4] = nf_mvm_w2(b + 128 + 32 * h + 4 * tq, odd);
        w[5] = nf_mvm_w2(b + 144 + 32 * h + 4 * tq, odd);
        w[6] = nf_mvm_w2(b + 192 + 8 * h, odd);
        w[7] = nf_mvm_w2(b + 196 + 8 * h, odd);
        w[8] = (uint)(*(NF_GLOBAL const ushort*)(b + 208));       /* d, two-byte aligned */
    }
}

/* ggml get_scale_min_k4 of sub-block s from the 12 scale bytes in words s4[0..2]. */
NF_INLINE void nf_mvm_k4_scale(const uint s4[3], int s, int* sc, int* m) {
    const uint q0 = (s4[0] >> (8 * (s & 3))) & 0xFFu;          /* scales[s & 3] */
    const uint q1 = (s4[1] >> (8 * (s & 3))) & 0xFFu;          /* scales[4 + (s & 3)] */
    const uint q2 = (s4[2] >> (8 * (s & 3))) & 0xFFu;          /* scales[8 + (s & 3)] */
    if (s < 4) {
        *sc = (int)(q0 & 63u);
        *m = (int)(q1 & 63u);
    } else {
        *sc = (int)((q2 & 0xFu) | ((q0 >> 6) << 4));
        *m = (int)((q2 >> 4) | ((q1 >> 6) << 4));
    }
}

/* Q6_K codes of one 16-element group word: low or high nibbles of ql with the matching qh bit pair, minus 32
 * per byte (codes 0..63 -> -32..31, exact in int8). */
NF_INLINE uint nf_mvm_q6(uint ql, uint qh, int hi, int sh) {
    const uint lo = (ql >> (4 * hi)) & 0x0F0F0F0Fu;
    const uint up = ((qh >> sh) & 0x03030303u) << 4;
    return __vsub4(lo | up, 0x20202020u);
}

/*
 * The activation words of Q8_1 block sb for this lane, per n tile j: x[j][0..1] the B fragment (token 8j + g;
 * Q8_0 / Q4_K / Q5_K elements [8q, 8q + 8), Q6_K the words at 4q and 16 + 4q), x[j][2..3] the d|s words of the C
 * tokens 8j + 2q and 8j + 2q + 1. Dead tokens read nothing and give zero.
 */
template <int T, int NJ>
NF_INLINE void nf_mvm_loadx(NF_GLOBAL const uchar* const xb[NJ], NF_GLOBAL const uchar* const xc[NJ][2],
                            const int lb[NJ], const int lc[NJ][2], int sb, int tq, uint x[NJ][4]) {
#pragma unroll
    for (int j = 0; j < NJ; j++) {
        NF_GLOBAL const uchar* xbb = xb[j] + (nf_i64)sb * Q8_1_BLOCK_BYTES + Q8_1_QS_OFFSET;
        if (T == NF_MVM_Q6_K) {
            x[j][0] = lb[j] ? nf_mvm_w(xbb + 4 * tq) : 0u;
            x[j][1] = lb[j] ? nf_mvm_w(xbb + 16 + 4 * tq) : 0u;
        } else {
            x[j][0] = lb[j] ? nf_mvm_w(xbb + 8 * tq) : 0u;
            x[j][1] = lb[j] ? nf_mvm_w(xbb + 8 * tq + 4) : 0u;
        }
#pragma unroll
        for (int k = 0; k < 2; k++) {
            x[j][2 + k] = lc[j][k] ? nf_mvm_w(xc[j][k] + (nf_i64)sb * Q8_1_BLOCK_BYTES) : 0u;
        }
    }
}

/*
 * The shared body. acc[i][j][c]: m tile i, n tile j, C register c (row g + 8 (c >> 1), token 2q + (c & 1)).
 */
template <int T, int NT, int MT, int PB>
NF_INLINE void nf_mvm_run(const int rows, const int cols, const int n_tokens,
                          NF_GLOBAL const uchar* NF_RESTRICT weight, NF_GLOBAL const uchar* NF_RESTRICT xq,
                          NF_GLOBAL float* NF_RESTRICT out, NF_LOCAL_PTR float* red) {
    const int NJ = NT / 8;
    const int TR = 16 * MT;                                     /* rows of the work-group */
    const int lid = (int)get_local_id(0);
    const int nw = (int)get_local_size(0) >> 5;
    const int warp = lid >> 5;
    const int lane = lid & 31;
    const int g = lane >> 2;
    const int tq = lane & 3;
    const int r0 = (int)get_group_id(0) * TR;
    const int bs = T == NF_MVM_Q8_0 ? 32 : 256;
    const int bb = T == NF_MVM_Q8_0 ? 34 : T == NF_MVM_Q4_K ? 144 : T == NF_MVM_Q5_K ? 176 : 210;
    const int align = T == NF_MVM_Q8_0 ? 64 : 512;
    const int ok = rows > 0 && cols > 0 && cols % align == 0 && n_tokens >= 1 && n_tokens <= NT
                   && nw >= 1 && nw <= NF_MVM_MAX_WARPS && ((int)get_local_size(0) & 31) == 0;
    if (!ok) {
        for (int i = lid; i < TR * NT; i += (int)get_local_size(0)) {
            const int r = r0 + i % TR;
            const int t = i / TR;
            if (r < rows && t < max(n_tokens, 0) && t < NT) {
                out[(nf_i64)t * rows + r] = NF_NAN;
            }
        }
        return;
    }
    const nf_i64 row_bytes = (nf_i64)(cols / bs) * bb;
    const int nsb = cols / 32;                                  /* Q8_1 blocks per token row */
    NF_GLOBAL const uchar* wrow[2 * MT];
#pragma unroll
    for (int i = 0; i < MT; i++) {
        wrow[2 * i] = weight + (nf_i64)min(r0 + 16 * i + g, rows - 1) * row_bytes;
        wrow[2 * i + 1] = weight + (nf_i64)min(r0 + 16 * i + g + 8, rows - 1) * row_bytes;
    }
    /* B fragment base of token 8j + g and the two C tokens 8j + 2q (+1) of every n tile; dead tokens
     * read nothing (their registers stay zero). */
    NF_GLOBAL const uchar* xb[NJ];
    NF_GLOBAL const uchar* xc[NJ][2];
    int lb[NJ], lc[NJ][2];
#pragma unroll
    for (int j = 0; j < NJ; j++) {
        const int tb = 8 * j + g;
        lb[j] = tb < n_tokens;
        xb[j] = xq + (nf_i64)min(tb, n_tokens - 1) * nsb * Q8_1_BLOCK_BYTES;
#pragma unroll
        for (int k = 0; k < 2; k++) {
            const int tc = 8 * j + 2 * tq + k;
            lc[j][k] = tc < n_tokens;
            xc[j][k] = xq + (nf_i64)min(tc, n_tokens - 1) * nsb * Q8_1_BLOCK_BYTES;
        }
    }

    float acc[MT][NJ][4];
#pragma unroll
    for (int i = 0; i < MT; i++) {
#pragma unroll
        for (int j = 0; j < NJ; j++) {
#pragma unroll
            for (int c = 0; c < 4; c++) {
                acc[i][j][c] = 0.0f;
            }
        }
    }

    const int groups = T == NF_MVM_Q8_0 ? 2 : 4;               /* Q8_1 blocks (32 elements) per step */
    const int steps = cols / (32 * groups);
    const int s_begin = (int)(((nf_i64)steps * warp) / nw);
    const int s_end = (int)(((nf_i64)steps * (warp + 1)) / nw);
    uint wc[2 * MT][NF_MVM_WREGS];
    uint wn[2 * MT][NF_MVM_WREGS];
    /* PB: the step's activation words (B fragments and the C tokens' d|s words) load one step ahead with the
     * weights, so no step waits on the activation rows' cache round trip */
    const int XG = PB ? groups : 1;
    uint xbc[XG][NJ][4], xbn[XG][NJ][4];
    if (s_begin < s_end) {
#pragma unroll
        for (int r = 0; r < 2 * MT; r++) {
            nf_mvm_load<T>(wrow[r], s_begin, tq, wc[r]);
        }
        if (PB) {
#pragma unroll
            for (int u = 0; u < XG; u++) {
                nf_mvm_loadx<T, NJ>(xb, xc, lb, lc, s_begin * groups + u, tq, xbc[u]);
            }
        }
    }
    for (int s = s_begin; s < s_end; s++) {
        if (s + 1 < s_end) {
#pragma unroll
            for (int r = 0; r < 2 * MT; r++) {
                nf_mvm_load<T>(wrow[r], s + 1, tq, wn[r]);
            }
            if (PB) {
#pragma unroll
                for (int u = 0; u < XG; u++) {
                    nf_mvm_loadx<T, NJ>(xb, xc, lb, lc, (s + 1) * groups + u, tq, xbn[u]);
                }
            }
        }
#pragma unroll
        for (int u = 0; u < groups; u++) {
            const int sb = s * groups + u;                      /* Q8_1 block index */
            uint xw[NJ][4];
            if (PB) {
#pragma unroll
                for (int j = 0; j < NJ; j++) {
#pragma unroll
                    for (int c = 0; c < 4; c++) {
                        xw[j][c] = xbc[PB ? u : 0][j][c];
                    }
                }
            } else {
                nf_mvm_loadx<T, NJ>(xb, xc, lb, lc, sb, tq, xw);
            }
            uint b[NJ][2];
            float dx[NJ][2], sx[NJ][2];
#pragma unroll
            for (int j = 0; j < NJ; j++) {
                b[j][0] = xw[j][0];
                b[j][1] = xw[j][1];
#pragma unroll
                for (int k = 0; k < 2; k++) {
                    dx[j][k] = nf_mvm_h(xw[j][2 + k] & 0xFFFFu);
                    sx[j][k] = nf_mvm_h(xw[j][2 + k] >> 16);
                }
            }
#pragma unroll
            for (int i = 0; i < MT; i++) {
                const uint* wa = wc[2 * i];
                const uint* wb = wc[2 * i + 1];
                if (T == NF_MVM_Q8_0) {
                    const float da = nf_mvm_h(u == 0 ? (wa[4] & 0xFFFFu) : (wa[4] >> 16));
                    const float db = nf_mvm_h(u == 0 ? (wb[4] & 0xFFFFu) : (wb[4] >> 16));
#pragma unroll
                    for (int j = 0; j < NJ; j++) {
                        int d[4];
                        nf_mvm_k32(d, wa[2 * u], wb[2 * u], wa[2 * u + 1], wb[2 * u + 1], b[j][0], b[j][1]);
                        acc[i][j][0] += (float)d[0] * (da * dx[j][0]);
                        acc[i][j][1] += (float)d[1] * (da * dx[j][1]);
                        acc[i][j][2] += (float)d[2] * (db * dx[j][0]);
                        acc[i][j][3] += (float)d[3] * (db * dx[j][1]);
                    }
                } else if (T == NF_MVM_Q4_K || T == NF_MVM_Q5_K) {
                    /* sub-block 4h + u of the super-block: pair j = u >> 1 of the step, nibble u & 1; a Q5_K
                     * code's fifth bit is bit `sub` of its qh byte */
                    const int sub = 4 * (s & 1) + u;
                    const int hi = u & 1;
                    const int pr = u >> 1;
                    uint a0 = (wa[2 * pr] >> (4 * hi)) & 0x0F0F0F0Fu;
                    uint a2 = (wa[2 * pr + 1] >> (4 * hi)) & 0x0F0F0F0Fu;
                    uint a1 = (wb[2 * pr] >> (4 * hi)) & 0x0F0F0F0Fu;
                    uint a3 = (wb[2 * pr + 1] >> (4 * hi)) & 0x0F0F0F0Fu;
                    if (T == NF_MVM_Q5_K) {
                        a0 |= ((wa[8] >> sub) & 0x01010101u) << 4;
                        a2 |= ((wa[9] >> sub) & 0x01010101u) << 4;
                        a1 |= ((wb[8] >> sub) & 0x01010101u) << 4;
                        a3 |= ((wb[9] >> sub) & 0x01010101u) << 4;
                    }
                    int sca, ma, scb, mb;
                    nf_mvm_k4_scale(wa + 4, sub, &sca, &ma);
                    nf_mvm_k4_scale(wb + 4, sub, &scb, &mb);
                    const float dsa = nf_mvm_h(wa[7] & 0xFFFFu) * (float)sca;
                    const float dma = nf_mvm_h(wa[7] >> 16) * (float)ma;
                    const float dsb = nf_mvm_h(wb[7] & 0xFFFFu) * (float)scb;
                    const float dmb = nf_mvm_h(wb[7] >> 16) * (float)mb;
#pragma unroll
                    for (int j = 0; j < NJ; j++) {
                        int d[4];
                        nf_mvm_k32(d, a0, a1, a2, a3, b[j][0], b[j][1]);
                        acc[i][j][0] += (float)d[0] * (dsa * dx[j][0]) - dma * sx[j][0];
                        acc[i][j][1] += (float)d[1] * (dsa * dx[j][1]) - dma * sx[j][1];
                        acc[i][j][2] += (float)d[2] * (dsb * dx[j][0]) - dmb * sx[j][0];
                        acc[i][j][3] += (float)d[3] * (dsb * dx[j][1]) - dmb * sx[j][1];
                    }
                } else {
                    /* sub-block 4h + u: ql words (u & 1) * 2 (+1) low/high nibble by u >> 1, qh shift 2u */
                    const int hi = u >> 1;
                    const int lo = u & 1;
                    const int sh = 2 * u;
                    const uint a0 = nf_mvm_q6(wa[2 * lo], wa[4], hi, sh);
                    const uint a2 = nf_mvm_q6(wa[2 * lo + 1], wa[5], hi, sh);
                    const uint a1 = nf_mvm_q6(wb[2 * lo], wb[4], hi, sh);
                    const uint a3 = nf_mvm_q6(wb[2 * lo + 1], wb[5], hi, sh);
                    /* the half's scale bytes 2u, 2u + 1 (groups 0..15 of the half: 8h + 2u + k) */
                    const uint swa = wa[6 + (u >> 1)] >> (16 * (u & 1));
                    const uint swb = wb[6 + (u >> 1)] >> (16 * (u & 1));
                    const int sa0 = (int)(nf_i8)(swa & 0xFFu), sa1 = (int)(nf_i8)((swa >> 8) & 0xFFu);
                    const int sb0 = (int)(nf_i8)(swb & 0xFFu), sb1 = (int)(nf_i8)((swb >> 8) & 0xFFu);
                    const float da = nf_mvm_h(wa[8] & 0xFFFFu);
                    const float db = nf_mvm_h(wb[8] & 0xFFFFu);
#pragma unroll
                    for (int j = 0; j < NJ; j++) {
                        int d0[4], d1[4];
                        nf_mvm_k16(d0, a0, a1, b[j][0]);
                        nf_mvm_k16(d1, a2, a3, b[j][1]);
                        acc[i][j][0] += (float)(d0[0] * sa0 + d1[0] * sa1) * (da * dx[j][0]);
                        acc[i][j][1] += (float)(d0[1] * sa0 + d1[1] * sa1) * (da * dx[j][1]);
                        acc[i][j][2] += (float)(d0[2] * sb0 + d1[2] * sb1) * (db * dx[j][0]);
                        acc[i][j][3] += (float)(d0[3] * sb0 + d1[3] * sb1) * (db * dx[j][1]);
                    }
                }
            }
        }
#pragma unroll
        for (int r = 0; r < 2 * MT; r++) {
#pragma unroll
            for (int k = 0; k < NF_MVM_WREGS; k++) {
                wc[r][k] = wn[r][k];
            }
        }
        if (PB) {
#pragma unroll
            for (int u = 0; u < XG; u++) {
#pragma unroll
                for (int j = 0; j < NJ; j++) {
#pragma unroll
                    for (int c = 0; c < 4; c++) {
                        xbc[u][j][c] = xbn[u][j][c];
                    }
                }
            }
        }
    }

    /* The warps' partial tiles through shared memory: red[warp][row][token], then summed in warp order. */
#pragma unroll
    for (int i = 0; i < MT; i++) {
#pragma unroll
        for (int j = 0; j < NJ; j++) {
#pragma unroll
            for (int c = 0; c < 4; c++) {
                const int rr = 16 * i + g + 8 * (c >> 1);
                const int tt = 8 * j + 2 * tq + (c & 1);
                red[(warp * TR + rr) * NT + tt] = acc[i][j][c];
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int e = lid; e < TR * NT; e += (int)get_local_size(0)) {
        const int rr = e % TR;
        const int tt = e / TR;
        const int r = r0 + rr;
        if (r < rows && tt < n_tokens) {
            float v = 0.0f;
            for (int w = 0; w < nw; w++) {
                v += red[(w * TR + rr) * NT + tt];
            }
            out[(nf_i64)tt * rows + r] = v;
        }
    }
}

/* MINB: the resident work-groups of the widest (NF_MVM_MAX_WARPS) launch the register budget must allow. */
#define NF_MVM_KERNEL(NAME, T, NT, MT, PB, MINB)                                                \
NF_KERNEL __launch_bounds__(NF_MVM_MAX_WARPS * 32, MINB)                                          \
void NAME(const int rows, const int cols, const int n_tokens,                                     \
                    NF_GLOBAL const uchar* NF_RESTRICT weight,                                    \
                    NF_GLOBAL const uchar* NF_RESTRICT xq,                                        \
                    NF_GLOBAL float* NF_RESTRICT out)                                             \
{                                                                                                 \
    NF_LOCAL float red[NF_MVM_MAX_WARPS * 16 * (MT) * (NT)];                                      \
    nf_mvm_run<T, NT, MT, PB>(rows, cols, n_tokens, weight, xq, out, red);                        \
}

/* The (MT, PB) of every (type, token tile): the forms the RTX 5080 Laptop measured fastest from DRAM
 * (gpu.forward.MatVecMmaMeasureTest). MatVecMmaPlan.tileRows mirrors the MT column. The Q8_0 forms fit two
 * 8-warp work-groups per SM in 128 registers without spills (the 16-token form compiles to 130 otherwise, which
 * leaves one); the K-quant forms keep their unbounded allocation, which a bound of two would spill. */
NF_MVM_KERNEL(Q8_0_MatVecMma8_Q8, NF_MVM_Q8_0, 8, 2, 0, 2)
NF_MVM_KERNEL(Q8_0_MatVecMma16_Q8, NF_MVM_Q8_0, 16, 1, 1, 2)
NF_MVM_KERNEL(Q4_K_MatVecMma8_Q8, NF_MVM_Q4_K, 8, 2, 1, 1)
NF_MVM_KERNEL(Q4_K_MatVecMma16_Q8, NF_MVM_Q4_K, 16, 2, 1, 1)
NF_MVM_KERNEL(Q5_K_MatVecMma8_Q8, NF_MVM_Q5_K, 8, 2, 1, 1)
NF_MVM_KERNEL(Q5_K_MatVecMma16_Q8, NF_MVM_Q5_K, 16, 2, 1, 1)
NF_MVM_KERNEL(Q6_K_MatVecMma8_Q8, NF_MVM_Q6_K, 8, 2, 0, 1)
NF_MVM_KERNEL(Q6_K_MatVecMma16_Q8, NF_MVM_Q6_K, 16, 2, 0, 1)

#endif /* sm_80+ */
#endif /* CUDA */

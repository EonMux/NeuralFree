/* matvec_row.cl - {T}_MatVecRow_Q8, {T}_MatVecAddRow_Q8 and {T}_MatVecFusedGateUpRow_Q8: the one-row
 * integer-dot matvec of the CUDA decode step. Requires: common/dialect.h, common/config.h, common/math.h */
/*
 * CUDA-only twins of the family's single-token Q8_1-activation forms (common/matvec_family.h) for Q8_0, Q4_K, Q5_K
 * and Q6_K, with their argument lists:
 *
 *   {T}_MatVecRow_Q8(i rows, i cols, R uchar weight, R uchar xq, W float out)       out[r] = W[r] . x
 *   {T}_MatVecAddRow_Q8(i rows, i cols, R uchar weight, R uchar xq, RW float acc)   acc[r] += W[r] . x
 *   {T}_MatVecFusedGateUpRow_Q8(i rows, i cols, R uchar gate, R uchar up, R uchar xq, W float out)
 *                                                                  out[r] = silu(G[r] . x) * (U[r] . x)
 *
 * xq is one Q8_1 row (ggml block_q8_1: d f16 @0, s f16 @2, qs[32] @4; cols / 32 blocks), the norm's emitted
 * blocks or LLM_QuantizeQ8_1's output. The family forms run one 64-lane work-group per row whose lanes each own
 * 8 elements of a 32-element group; a K-quant lane then reads its 6-bit or 4-bit codes as fragments of words its
 * neighbours read again for the other nibble or the other 2-bit plane, and every row ends in a barrier tree.
 * These forms take the shape of llama.cpp's mmvq (ggml-cuda/mmvq.cu) with one warp per row:
 *
 *   Warp w of work-group g computes row g * W + w, W = get_local_size(0) / 32 (1, 2 or 4 warps; the host picks
 *   per type and shape, gpu.forward.WarpRowMatvec). A row's units (a Q8_0 block, a K-quant super-block) are split
 *   over lane groups of LPU lanes (4 for Q8_0, 16 for the K-quants); group k of the warp takes units k,
 *   k + 32 / LPU, ... so consecutive lanes read consecutive bytes. Each lane accumulates its dot in a float
 *   register and the warp reduces it by the 32-lane butterfly: no shared memory, no barrier.
 *
 * Per lane and unit:
 *   - Q8_0: the 8 codes [8q, 8q + 8) of block u (two u16-assembled words: the codes start two bytes into the
 *     34-byte block), dotted with the same 8 activation codes; times d_w * d_x.
 *   - Q4_K: lane q of 16 takes byte run [8 (q & 3), + 8) of sub-block pair j = q >> 2 (one aligned 8-byte load:
 *     the 144-byte super-blocks keep 16-byte alignment): the low nibbles are sub-block 2j, the high ones 2j + 1.
 *     Both dot their activation block's 8 codes; d * sc and dmin * m apply per sub-block, the minimum through the
 *     exact code sum of the lane's activation codes.
 *   - Q5_K: the Q4_K lane map over the 176-byte super-block (qs at 48, 16-byte aligned like Q4_K's): lane q also
 *     loads the 8 qh bytes [16 + 8 (q & 3), + 8), whose bits 2j and 2j + 1 are the fifth bits of the same
 *     positions in sub-blocks 2j and 2j + 1, and ORs them above each nibble before the dot; the scale and minimum
 *     terms are Q4_K's.
 *   - Q6_K: lane q of 16 takes half h = q >> 3 and the 4 positions l = 4 (q & 7): the ql words at 64h + l and
 *     64h + 32 + l and the qh word at 128 + 32h + l hold all four 2-bit planes of those positions, i.e. elements
 *     128h + l + {0, 32, 64, 96} + [0, 4), so every weight byte is read once; the 6-bit code minus 32 dots its
 *     activation block through dp4a, the -32 offset entering as the dp4a of the activation word with four -32
 *     bytes; times the int8 scale, d_x and d. The fused gate/up form loads each activation word, its scale and
 *     that offset sum once for both rows, which keeps its state in registers.
 * Integer dots are exact; scales apply in float per 32-element group as in the family forms, so the result
 * differs from theirs only in float summation order.
 *
 * Geometry: local {32 W} for W in {1, 2, 4}, global {ceil(rows / W) * 32 W}. Warps past `rows` return at once.
 * cols is a multiple of the type's block (32 or 256 elements); the weight rows are contiguous.
 */

#if defined(__CUDACC__) && !defined(__HIP__)

#define NF_MVR_MAX_WARPS 4
#define NF_MVR_Q8_0  0
#define NF_MVR_Q4_K  1
#define NF_MVR_Q6_K  2
#define NF_MVR_Q5_K  3

/* Lanes per unit, elements per unit and bytes per unit of type T. */
#define NF_MVR_LPU(T) ((T) == NF_MVR_Q8_0 ? 4 : 16)
#define NF_MVR_BS(T)  ((T) == NF_MVR_Q8_0 ? 32 : 256)
#define NF_MVR_BB(T)  ((T) == NF_MVR_Q8_0 ? 34 : (T) == NF_MVR_Q4_K ? 144 : (T) == NF_MVR_Q5_K ? 176 : 210)

#define NF_MVR_FORM_PLAIN  0
#define NF_MVR_FORM_ADD    1
#define NF_MVR_FORM_GATEUP 2

/* The 32-bit word at the two-byte aligned p. */
NF_INLINE int nf_mvr_w2(NF_GLOBAL const uchar* p) {
    const ushort* s = (const ushort*)p;
    return (int)((uint)s[0] | ((uint)s[1] << 16));
}

NF_INLINE float nf_mvr_h(NF_GLOBAL const uchar* p) {
    return __half2float(*(const __half*)p);
}

/* Scale d and the code word at byte off of Q8_1 block b of the activation row. */
NF_INLINE float nf_mvr_xd(NF_GLOBAL const uchar* xq, int b) {
    return nf_mvr_h(xq + b * Q8_1_BLOCK_BYTES);
}

NF_INLINE int nf_mvr_xw(NF_GLOBAL const uchar* xq, int b, int off) {
    return *(const int*)(xq + b * Q8_1_BLOCK_BYTES + Q8_1_QS_OFFSET + off);
}

/* ggml get_scale_min_k4 of sub-block s from the 12 scale bytes at sc. */
NF_INLINE void nf_mvr_k4(NF_GLOBAL const uchar* sc, int s, int* d, int* m) {
    if (s < 4) {
        *d = sc[s] & 63;
        *m = sc[s + 4] & 63;
    } else {
        *d = (sc[s + 4] & 0xF) | ((sc[s - 4] >> 6) << 4);
        *m = (sc[s + 4] >> 4) | ((sc[s] >> 6) << 4);
    }
}

/* Lane q's share of the dot of unit u of the Q8_0, Q4_K or Q5_K weight row at row with the activation row xq. */
template <int T>
NF_INLINE float nf_mvr_dot(NF_GLOBAL const uchar* row, NF_GLOBAL const uchar* xq, int u, int q) {
    if (T == NF_MVR_Q8_0) {
        NF_GLOBAL const uchar* bp = row + u * 34;
        const int w0 = nf_mvr_w2(bp + 2 + 8 * q);
        const int w1 = nf_mvr_w2(bp + 6 + 8 * q);
        const int s = nf_dp4a(w1, nf_mvr_xw(xq, u, 8 * q + 4), nf_dp4a(w0, nf_mvr_xw(xq, u, 8 * q), 0));
        return nf_mvr_h(bp) * nf_mvr_xd(xq, u) * (float)s;
    } else if (T == NF_MVR_Q4_K || T == NF_MVR_Q5_K) {
        NF_GLOBAL const uchar* bp = row + u * NF_MVR_BB(T);
        const int j = q >> 2, l0 = (q & 3) * 8;
        const uint2 w = *(const uint2*)(bp + (T == NF_MVR_Q5_K ? 48 : 16) + 32 * j + l0);
        uint lo0 = w.x & 0x0F0F0F0Fu, lo1 = w.y & 0x0F0F0F0Fu;
        uint hi0 = (w.x >> 4) & 0x0F0F0F0Fu, hi1 = (w.y >> 4) & 0x0F0F0F0Fu;
        if (T == NF_MVR_Q5_K) {
            const uint2 h = *(const uint2*)(bp + 16 + l0);
            lo0 |= ((h.x >> (2 * j)) & 0x01010101u) << 4;
            lo1 |= ((h.y >> (2 * j)) & 0x01010101u) << 4;
            hi0 |= ((h.x >> (2 * j + 1)) & 0x01010101u) << 4;
            hi1 |= ((h.y >> (2 * j + 1)) & 0x01010101u) << 4;
        }
        const int b0 = u * 8 + 2 * j;
        const int x0 = nf_mvr_xw(xq, b0, l0), x1 = nf_mvr_xw(xq, b0, l0 + 4);
        const int y0 = nf_mvr_xw(xq, b0 + 1, l0), y1 = nf_mvr_xw(xq, b0 + 1, l0 + 4);
        const int dlo = nf_dp4a((int)lo1, x1, nf_dp4a((int)lo0, x0, 0));
        const int dhi = nf_dp4a((int)hi1, y1, nf_dp4a((int)hi0, y0, 0));
        const int slo = nf_dp4a(0x01010101, x1, nf_dp4a(0x01010101, x0, 0));
        const int shi = nf_dp4a(0x01010101, y1, nf_dp4a(0x01010101, y0, 0));
        int s0, m0, s1, m1;
        nf_mvr_k4(bp + 4, 2 * j, &s0, &m0);
        nf_mvr_k4(bp + 4, 2 * j + 1, &s1, &m1);
        const float dx0 = nf_mvr_xd(xq, b0), dx1 = nf_mvr_xd(xq, b0 + 1);
        return nf_mvr_h(bp) * (dx0 * (float)(s0 * dlo) + dx1 * (float)(s1 * dhi))
             - nf_mvr_h(bp + 2) * (dx0 * (float)(m0 * slo) + dx1 * (float)(m1 * shi));
    }
    return 0.0f;
}

/* Code word k (0..3) of a Q6_K lane: the 2-bit plane k of qh above the low or high nibbles of ql0 or ql1. */
NF_INLINE int nf_mvr_q6_code(int ql0, int ql1, int qh, int k) {
    const int ql = (k & 1) ? ql1 : ql0;
    const int hi = k == 0 ? qh << 4 : k == 1 ? qh << 2 : k == 2 ? qh : qh >> 2;
    return ((k & 2 ? ql >> 4 : ql) & 0x0F0F0F0F) | (hi & 0x30303030);
}

/*
 * Lane q's share of unit u of the Q6_K weight row r0, and when NR == 2 of the row r1 at the same offset of a
 * second matrix, dotted with the activation row xq and added to *a (and *b). Each activation word, its scale and
 * its -32 offset sum load once for both rows.
 */
template <int NR>
NF_INLINE void nf_mvr_q6k(NF_GLOBAL const uchar* r0, NF_GLOBAL const uchar* r1, NF_GLOBAL const uchar* xq, int u,
                          int q, float* a, float* b) {
    const int h = q >> 3, l = (q & 7) * 4;
    const int b0 = u * 8 + 4 * h;
    const int wo = 64 * h + l, ho = 128 + 32 * h + l, so = 192 + 8 * h + (l >> 4);
    NF_GLOBAL const uchar* p0 = r0 + u * 210;
    NF_GLOBAL const uchar* p1 = r1 + u * 210;
    const int ql0a = nf_mvr_w2(p0 + wo), ql1a = nf_mvr_w2(p0 + wo + 32), qha = nf_mvr_w2(p0 + ho);
    const int ql0b = NR == 2 ? nf_mvr_w2(p1 + wo) : 0;
    const int ql1b = NR == 2 ? nf_mvr_w2(p1 + wo + 32) : 0;
    const int qhb = NR == 2 ? nf_mvr_w2(p1 + ho) : 0;
    const int m32 = (int)0xE0E0E0E0;      /* four int8 -32: the code offset */
    float sa = 0.0f, sb = 0.0f;
#pragma unroll
    for (int k = 0; k < 4; k++) {
        const int x = nf_mvr_xw(xq, b0 + k, l);
        const float dx = nf_mvr_xd(xq, b0 + k);
        const int off = nf_dp4a(m32, x, 0);
        sa += dx * (float)(nf_dp4a(nf_mvr_q6_code(ql0a, ql1a, qha, k), x, off)
                           * ((NF_GLOBAL const signed char*)p0)[so + 2 * k]);
        if (NR == 2) {
            sb += dx * (float)(nf_dp4a(nf_mvr_q6_code(ql0b, ql1b, qhb, k), x, off)
                               * ((NF_GLOBAL const signed char*)p1)[so + 2 * k]);
        }
    }
    *a += nf_mvr_h(p0 + 208) * sa;
    if (NR == 2) {
        *b += nf_mvr_h(p1 + 208) * sb;
    }
}

NF_INLINE float nf_mvr_warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, o);
    }
    return v;
}

template <int T, int FORM>
NF_INLINE void nf_mvr_run(const int rows, const int cols, NF_GLOBAL const uchar* NF_RESTRICT w,
                          NF_GLOBAL const uchar* NF_RESTRICT w2, NF_GLOBAL const uchar* NF_RESTRICT xq,
                          NF_GLOBAL float* NF_RESTRICT out) {
    const int lpu = NF_MVR_LPU(T);
    const int lane = (int)get_local_id(0) & 31;
    const int row = (int)get_group_id(0) * ((int)get_local_size(0) >> 5) + ((int)get_local_id(0) >> 5);
    if (row >= rows) {
        return;
    }
    const int units = cols / NF_MVR_BS(T);
    const nf_i64 off = (nf_i64)row * units * NF_MVR_BB(T);
    const int q = lane % lpu;
    float a = 0.0f, b = 0.0f;
    for (int u = lane / lpu; u < units; u += 32 / lpu) {
        if (T == NF_MVR_Q6_K) {
            nf_mvr_q6k<FORM == NF_MVR_FORM_GATEUP ? 2 : 1>(w + off, w2 + off, xq, u, q, &a, &b);
        } else {
            a += nf_mvr_dot<T>(w + off, xq, u, q);
            if (FORM == NF_MVR_FORM_GATEUP) {
                b += nf_mvr_dot<T>(w2 + off, xq, u, q);
            }
        }
    }
    a = nf_mvr_warp_sum(a);
    if (FORM == NF_MVR_FORM_GATEUP) {
        b = nf_mvr_warp_sum(b);
    }
    if (lane == 0) {
        if (FORM == NF_MVR_FORM_PLAIN) {
            out[row] = a;
        } else if (FORM == NF_MVR_FORM_ADD) {
            out[row] += a;
        } else {
            out[row] = nf_silu(a) * b;
        }
    }
}

#define NF_MVR_KERNELS(N, T)                                                                              \
NF_KERNEL __launch_bounds__(NF_MVR_MAX_WARPS * 32)                                                       \
void N##_MatVecRow_Q8(const int rows, const int cols, NF_GLOBAL const uchar* NF_RESTRICT weight,         \
                      NF_GLOBAL const uchar* NF_RESTRICT xq, NF_GLOBAL float* NF_RESTRICT out) {         \
    nf_mvr_run<T, NF_MVR_FORM_PLAIN>(rows, cols, weight, weight, xq, out);                               \
}                                                                                                        \
NF_KERNEL __launch_bounds__(NF_MVR_MAX_WARPS * 32)                                                       \
void N##_MatVecAddRow_Q8(const int rows, const int cols, NF_GLOBAL const uchar* NF_RESTRICT weight,      \
                         NF_GLOBAL const uchar* NF_RESTRICT xq, NF_GLOBAL float* NF_RESTRICT acc) {      \
    nf_mvr_run<T, NF_MVR_FORM_ADD>(rows, cols, weight, weight, xq, acc);                                 \
}                                                                                                        \
NF_KERNEL __launch_bounds__(NF_MVR_MAX_WARPS * 32)                                                       \
void N##_MatVecFusedGateUpRow_Q8(const int rows, const int cols, NF_GLOBAL const uchar* NF_RESTRICT gate, \
                                 NF_GLOBAL const uchar* NF_RESTRICT up, NF_GLOBAL const uchar* NF_RESTRICT xq, \
                                 NF_GLOBAL float* NF_RESTRICT out) {                                     \
    nf_mvr_run<T, NF_MVR_FORM_GATEUP>(rows, cols, gate, up, xq, out);                                    \
}

NF_MVR_KERNELS(Q8_0, NF_MVR_Q8_0)
NF_MVR_KERNELS(Q4_K, NF_MVR_Q4_K)
NF_MVR_KERNELS(Q5_K, NF_MVR_Q5_K)
NF_MVR_KERNELS(Q6_K, NF_MVR_Q6_K)

#endif /* CUDA */

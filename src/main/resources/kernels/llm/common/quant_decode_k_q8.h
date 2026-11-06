/* common/quant_decode_k_q8.h - integer-dot (Q8_1 activation) decoders of the K-quant super-blocks. Requires: common/dialect.h, common/config.h, common/quant_blocks.h, common/quant_decode_k.h, common/q8_dot.h */
/*
 * T_decode_q8 / T_decode_q8_dual for Q2_K, Q3_K, Q4_K, Q5_K, Q6_K and Q8_K, with the argument lists
 * of common/matvec_family.h's int8 forms:
 *
 *   NF_INLINE float T_decode_q8(NF_GLOBAL const uchar* bp, NF_GLOBAL const uchar* xq,
 *                               int blk, int ltid)
 *       Contribution of work-item ltid's 8 elements of 256-element block blk (bytes at bp) to the
 *       dot of the decoded weight row with the Q8_1 activations at xq[blk * 8 .. blk * 8 + 8): the
 *       exact decoded values times d8 * q8, the Q8_1 block scale and int8 code.
 *   NF_INLINE void T_decode_q8_dual(... gate block gbp, up block ubp ...).
 *
 * Work-item ltid owns elements 8*ltid .. 8*ltid+7, which lie in one 32-element group, hence in one
 * Q8_1 block (blk * 8 + ltid / 4) and its (ltid % 4)-th 8-byte slice. The 8 weight codes are read
 * as two packed words and dotted with the activation codes by NF_DP4A; the decoded scale and
 * minimum stay in float, so the result equals the float decoder's dot product over the same
 * dequantized activations.
 */
#ifndef NF_QUANT_DECODE_K_Q8_H
#define NF_QUANT_DECODE_K_Q8_H

#ifndef NF_Q8_DOT_H
#error "common/quant_decode_k_q8.h requires common/q8_dot.h"
#endif

/* Q8_1 block and byte slice of work-item ltid's group of 256-element block blk. */
#define NF_KQ_Q8_BLOCK(blk, ltid) ((blk) * 8 + ((ltid) >> 2))
#define NF_KQ_Q8_BYTE(ltid)       (((ltid) & 3) * 8)

/* ---- Q2_K: y = d*(sc & 15) * code - dmin*(sc >> 4), code in [0, 4). ---- */
NF_INLINE float Q2_K_decode_q8(NF_GLOBAL const uchar* bp, NF_GLOBAL const uchar* xq, int blk, int ltid) {
    const uint sc = bp[NF_Q2K_SCALES + (ltid >> 1)];
    const uint shift = (uint)((ltid & 15) >> 2) * 2u;
    NF_GLOBAL const uchar* q = bp + NF_Q2K_QS + (ltid >> 4) * 32 + ((ltid >> 1) & 1) * 16 + (ltid & 1) * 8;
    const int qb = NF_KQ_Q8_BLOCK(blk, ltid);
    const int bo = NF_KQ_Q8_BYTE(ltid);
    const int u0 = nf_q8_word(xq, qb, bo);
    const int u1 = nf_q8_word(xq, qb, bo + 4);
    const int c0 = (int)((nf_read_u32(q) >> shift) & 0x03030303u);
    const int c1 = (int)((nf_read_u32(q + 4) >> shift) & 0x03030303u);
    const int idot = NF_DP4A(c0, u0, 0) + NF_DP4A(c1, u1, 0);
    const int sumq = NF_DP4A(NF_Q8_ONES, u0, 0) + NF_DP4A(NF_Q8_ONES, u1, 0);
    const float dl = nf_kq_half(bp, NF_Q2K_D) * (float)(sc & 0xFu);
    const float ml = nf_kq_half(bp, NF_Q2K_DMIN) * (float)(sc >> 4);
    return nf_q8_scale(xq, qb) * (dl * (float)idot - ml * (float)sumq);
}

/* ---- Q3_K: y = d*scale * code, code in [-4, 4). ---- */
NF_INLINE float Q3_K_decode_q8(NF_GLOBAL const uchar* bp, NF_GLOBAL const uchar* xq, int blk, int ltid) {
    const float dl = nf_kq_half(bp, NF_Q3K_D) * (float)nf_q3k_scale(bp, ltid >> 1);
    const int hf = ltid >> 4, j = (ltid & 15) >> 2;
    const uint shift = (uint)j * 2u, mbit = (uint)(hf * 4 + j);
    NF_GLOBAL const uchar* q = bp + NF_Q3K_QS + hf * 32 + ((ltid >> 1) & 1) * 16 + (ltid & 1) * 8;
    NF_GLOBAL const uchar* hm = bp + NF_Q3K_HMASK + (ltid & 3) * 8;
    const int qb = NF_KQ_Q8_BLOCK(blk, ltid);
    const int bo = NF_KQ_Q8_BYTE(ltid);
    const int u0 = nf_q8_word(xq, qb, bo);
    const int u1 = nf_q8_word(xq, qb, bo + 4);
    const int c0 = (int)(((nf_read_u32_u16(q) >> shift) & 0x03030303u)
                       | (((nf_read_u32_u16(hm) >> mbit) & 0x01010101u) << 2));
    const int c1 = (int)(((nf_read_u32_u16(q + 4) >> shift) & 0x03030303u)
                       | (((nf_read_u32_u16(hm + 4) >> mbit) & 0x01010101u) << 2));
    const int idot = NF_DP4A(c0, u0, 0) + NF_DP4A(c1, u1, 0)
                   - 4 * (NF_DP4A(NF_Q8_ONES, u0, 0) + NF_DP4A(NF_Q8_ONES, u1, 0));
    return nf_q8_scale(xq, qb) * dl * (float)idot;
}

/* ---- Q4_K: y = d*sc * code - dmin*m, code in [0, 16). ---- */
NF_INLINE float Q4_K_decode_q8(NF_GLOBAL const uchar* bp, NF_GLOBAL const uchar* xq, int blk, int ltid) {
    const int si = ltid >> 2;
    const float2 sm = k4_scale_min(bp + NF_Q4K_SCALES, si);
    const uint sh = (uint)(si & 1) * 4u;
    NF_GLOBAL const uchar* q = bp + NF_Q4K_QS + (si >> 1) * 32 + (ltid & 3) * 8;
    const int qb = NF_KQ_Q8_BLOCK(blk, ltid);
    const int bo = NF_KQ_Q8_BYTE(ltid);
    const int u0 = nf_q8_word(xq, qb, bo);
    const int u1 = nf_q8_word(xq, qb, bo + 4);
    const int c0 = (int)((nf_read_u32(q) >> sh) & 0x0F0F0F0Fu);
    const int c1 = (int)((nf_read_u32(q + 4) >> sh) & 0x0F0F0F0Fu);
    const int idot = NF_DP4A(c0, u0, 0) + NF_DP4A(c1, u1, 0);
    const int sumq = NF_DP4A(NF_Q8_ONES, u0, 0) + NF_DP4A(NF_Q8_ONES, u1, 0);
    const float d = nf_kq_half(bp, NF_Q4K_D), dm = nf_kq_half(bp, NF_Q4K_DMIN);
    return nf_q8_scale(xq, qb) * (d * sm.x * (float)idot - dm * sm.y * (float)sumq);
}

/* ---- Q5_K: y = d*sc * code - dmin*m, code in [0, 32). ---- */
NF_INLINE float Q5_K_decode_q8(NF_GLOBAL const uchar* bp, NF_GLOBAL const uchar* xq, int blk, int ltid) {
    const int si = ltid >> 2;
    const float2 sm = k4_scale_min(bp + NF_Q5K_SCALES, si);
    const uint sh = (uint)(si & 1) * 4u;
    NF_GLOBAL const uchar* q = bp + NF_Q5K_QS + (si >> 1) * 32 + (ltid & 3) * 8;
    NF_GLOBAL const uchar* qh = bp + NF_Q5K_QH + (ltid & 3) * 8;
    const int qb = NF_KQ_Q8_BLOCK(blk, ltid);
    const int bo = NF_KQ_Q8_BYTE(ltid);
    const int u0 = nf_q8_word(xq, qb, bo);
    const int u1 = nf_q8_word(xq, qb, bo + 4);
    const int c0 = (int)(((nf_read_u32(q) >> sh) & 0x0F0F0F0Fu)
                       | (((nf_read_u32(qh) >> si) & 0x01010101u) << 4));
    const int c1 = (int)(((nf_read_u32(q + 4) >> sh) & 0x0F0F0F0Fu)
                       | (((nf_read_u32(qh + 4) >> si) & 0x01010101u) << 4));
    const int idot = NF_DP4A(c0, u0, 0) + NF_DP4A(c1, u1, 0);
    const int sumq = NF_DP4A(NF_Q8_ONES, u0, 0) + NF_DP4A(NF_Q8_ONES, u1, 0);
    const float d = nf_kq_half(bp, NF_Q5K_D), dm = nf_kq_half(bp, NF_Q5K_DMIN);
    return nf_q8_scale(xq, qb) * (d * sm.x * (float)idot - dm * sm.y * (float)sumq);
}

/* ---- Q6_K: y = d*scale * code, code in [-32, 32). ---- */
NF_INLINE float Q6_K_decode_q8(NF_GLOBAL const uchar* bp, NF_GLOBAL const uchar* xq, int blk, int ltid) {
    const int hf = ltid >> 4, t = (ltid >> 2) & 3, l0 = (ltid & 3) * 8;
    const int sc = ((NF_GLOBAL const nf_i8*)(bp + NF_Q6K_SCALES))[hf * 8 + t * 2 + (l0 >> 4)];
    const float dl = nf_kq_half(bp, NF_Q6K_D) * (float)sc;
    const uint qlsh = (uint)(t >> 1) * 4u, qhsh = (uint)t * 2u;
    NF_GLOBAL const uchar* ql = bp + NF_Q6K_QL + hf * 64 + (t & 1) * 32 + l0;
    NF_GLOBAL const uchar* qh = bp + NF_Q6K_QH + hf * 32 + l0;
    const int qb = NF_KQ_Q8_BLOCK(blk, ltid);
    const int bo = NF_KQ_Q8_BYTE(ltid);
    const int u0 = nf_q8_word(xq, qb, bo);
    const int u1 = nf_q8_word(xq, qb, bo + 4);
    const int c0 = (int)(((nf_read_u32_u16(ql) >> qlsh) & 0x0F0F0F0Fu)
                       | (((nf_read_u32_u16(qh) >> qhsh) & 0x03030303u) << 4));
    const int c1 = (int)(((nf_read_u32_u16(ql + 4) >> qlsh) & 0x0F0F0F0Fu)
                       | (((nf_read_u32_u16(qh + 4) >> qhsh) & 0x03030303u) << 4));
    const int idot = NF_DP4A(c0, u0, 0) + NF_DP4A(c1, u1, 0)
                   - 32 * (NF_DP4A(NF_Q8_ONES, u0, 0) + NF_DP4A(NF_Q8_ONES, u1, 0));
    return nf_q8_scale(xq, qb) * dl * (float)idot;
}

/* ---- Q8_K: y = d * qs[e], qs int8. ---- */
NF_INLINE float Q8_K_decode_q8(NF_GLOBAL const uchar* bp, NF_GLOBAL const uchar* xq, int blk, int ltid) {
    const float d = as_float(nf_read_u32(bp + NF_Q8K_D));
    NF_GLOBAL const uchar* qs = bp + NF_Q8K_QS + ltid * NF_KQ_EPT;
    const int qb = NF_KQ_Q8_BLOCK(blk, ltid);
    const int bo = NF_KQ_Q8_BYTE(ltid);
    const int idot = NF_DP4A((int)nf_read_u32(qs), nf_q8_word(xq, qb, bo), 0)
                   + NF_DP4A((int)nf_read_u32(qs + 4), nf_q8_word(xq, qb, bo + 4), 0);
    return nf_q8_scale(xq, qb) * d * (float)idot;
}

/* T_decode_q8_dual of the fused gate/up form: both rows over the same activation group. */
#define NF_KQ_Q8_DUAL(N) \
NF_INLINE void N##_decode_q8_dual(NF_GLOBAL const uchar* gbp, NF_GLOBAL const uchar* ubp, \
                                  NF_GLOBAL const uchar* xq, int blk, int ltid, \
                                  float* gs, float* us) { \
    *gs += N##_decode_q8(gbp, xq, blk, ltid); \
    *us += N##_decode_q8(ubp, xq, blk, ltid); \
}

NF_KQ_Q8_DUAL(Q2_K)
NF_KQ_Q8_DUAL(Q3_K)
NF_KQ_Q8_DUAL(Q4_K)
NF_KQ_Q8_DUAL(Q5_K)
NF_KQ_Q8_DUAL(Q6_K)
NF_KQ_Q8_DUAL(Q8_K)

#endif /* NF_QUANT_DECODE_K_Q8_H */

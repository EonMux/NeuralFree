/* common/quant_decode_legacy_q8.h - integer-dot (Q8_1 activation) decoders of Q4_0, Q4_1, Q5_0, Q5_1, IQ4_NL, MXFP4 and IQ4_XS. Requires: common/dialect.h, common/config.h, common/quant_blocks.h, common/quant_decode_legacy.h, common/q8_dot.h */
/*
 * T_decode_q8 for the 32-element legacy types and for the two i-quants with 32-element split nibble
 * blocks, with the argument list of common/matvec_family.h's int8 forms. Every type has 32-element
 * (sub-)blocks that map one to one onto the Q8_1 blocks, and the four work-items that share a block
 * (TPS = 4) own its qs bytes 4*ltid .. 4*ltid+3, i.e. elements 4*ltid .. +3 and their high-nibble
 * partners 4*ltid+16 .. +3:
 *
 *   Q4_0, Q5_0   y = d * (code - bias): d8 * d * (sum(code*q8) - bias * sum(q8))
 *   Q4_1, Q5_1   y = d * code + m:      d8 * (d * sum(code*q8) + m * sum(q8))
 *   IQ4_NL       y = d * kv[nibble], and the same split applies to IQ4_XS's 32-element sub-blocks.
 *
 * The weight codes are packed as int8 and dotted with the Q8_1 codes (NF_DP4A); the bias/min terms use
 * the sum of the owned Q8_1 codes, so the work-items of a block add up to the block's s term.
 */
#ifndef NF_QUANT_DECODE_LEGACY_Q8_H
#define NF_QUANT_DECODE_LEGACY_Q8_H

#ifndef NF_Q8_DOT_H
#error "common/quant_decode_legacy_q8.h requires common/q8_dot.h"
#endif

#ifndef NF_QUANT_DECODE_LEGACY_H
#error "common/quant_decode_legacy_q8.h requires common/quant_decode_legacy.h"
#endif

/* Q8_1 words and code sum of the 8 elements work-item ltid owns in a 32-element block. */
#define NF_QL_Q8_LO(q) ((int)((q) & 0x0F0F0F0Fu))
#define NF_QL_Q8_HI(q) ((int)(((q) >> 4) & 0x0F0F0F0Fu))

/* Fifth bits (as 0x10) of the four elements e0 .. e0+3 of a Q5_0/Q5_1 qh word. */
NF_INLINE int nf_ql_q8_hbits(uint qh, int e0) {
    const uint b = (qh >> e0) & 0xFu;
    const uint packed = ((b >> 0) & 1u) | (((b >> 1) & 1u) << 8)
                      | (((b >> 2) & 1u) << 16) | (((b >> 3) & 1u) << 24);
    return (int)(packed << 4);
}

/* ---- Q4_0: y = d*(code - 8), code the 4-bit nibble. ---- */
NF_INLINE float Q4_0_decode_q8(NF_GLOBAL const uchar* bp, NF_GLOBAL const uchar* xq, int blk, int ltid) {
    const uint q = read_u32_le(bp + 2 + 4 * ltid);
    const int bo = ltid * 4;
    const int u0 = nf_q8_word(xq, blk, bo);
    const int u1 = nf_q8_word(xq, blk, bo + 16);
    const int idot = NF_DP4A(NF_QL_Q8_LO(q), u0, 0) + NF_DP4A(NF_QL_Q8_HI(q), u1, 0);
    const int sumq = NF_DP4A(NF_Q8_ONES, u0, 0) + NF_DP4A(NF_Q8_ONES, u1, 0);
    return nf_q8_scale(xq, blk) * nf_ql_half(bp, 0) * (float)(idot - 8 * sumq);
}

/* ---- Q4_1: y = d*code + m. ---- */
NF_INLINE float Q4_1_decode_q8(NF_GLOBAL const uchar* bp, NF_GLOBAL const uchar* xq, int blk, int ltid) {
    const uint q = read_u32_le(bp + 4 + 4 * ltid);
    const int bo = ltid * 4;
    const int u0 = nf_q8_word(xq, blk, bo);
    const int u1 = nf_q8_word(xq, blk, bo + 16);
    const int idot = NF_DP4A(NF_QL_Q8_LO(q), u0, 0) + NF_DP4A(NF_QL_Q8_HI(q), u1, 0);
    const int sumq = NF_DP4A(NF_Q8_ONES, u0, 0) + NF_DP4A(NF_Q8_ONES, u1, 0);
    return nf_q8_scale(xq, blk) * (nf_ql_half(bp, 0) * (float)idot + nf_ql_half(bp, 2) * (float)sumq);
}

/* ---- Q5_0: y = d*(code - 16), code the 4-bit nibble with its qh bit as the 0x10 bit. ---- */
NF_INLINE float Q5_0_decode_q8(NF_GLOBAL const uchar* bp, NF_GLOBAL const uchar* xq, int blk, int ltid) {
    const uint qh = read_u32_le(bp + 2);
    const uint q = read_u32_le(bp + 6 + 4 * ltid);
    const int bo = ltid * 4;
    const int u0 = nf_q8_word(xq, blk, bo);
    const int u1 = nf_q8_word(xq, blk, bo + 16);
    const int c0 = NF_QL_Q8_LO(q) | nf_ql_q8_hbits(qh, bo);
    const int c1 = NF_QL_Q8_HI(q) | nf_ql_q8_hbits(qh, bo + 16);
    const int idot = NF_DP4A(c0, u0, 0) + NF_DP4A(c1, u1, 0);
    const int sumq = NF_DP4A(NF_Q8_ONES, u0, 0) + NF_DP4A(NF_Q8_ONES, u1, 0);
    return nf_q8_scale(xq, blk) * nf_ql_half(bp, 0) * (float)(idot - 16 * sumq);
}

/* ---- Q5_1: y = d*code + m, code as in Q5_0. ---- */
NF_INLINE float Q5_1_decode_q8(NF_GLOBAL const uchar* bp, NF_GLOBAL const uchar* xq, int blk, int ltid) {
    const uint qh = read_u32_le(bp + 4);
    const uint q = read_u32_le(bp + 8 + 4 * ltid);
    const int bo = ltid * 4;
    const int u0 = nf_q8_word(xq, blk, bo);
    const int u1 = nf_q8_word(xq, blk, bo + 16);
    const int c0 = NF_QL_Q8_LO(q) | nf_ql_q8_hbits(qh, bo);
    const int c1 = NF_QL_Q8_HI(q) | nf_ql_q8_hbits(qh, bo + 16);
    const int idot = NF_DP4A(c0, u0, 0) + NF_DP4A(c1, u1, 0);
    const int sumq = NF_DP4A(NF_Q8_ONES, u0, 0) + NF_DP4A(NF_Q8_ONES, u1, 0);
    return nf_q8_scale(xq, blk) * (nf_ql_half(bp, 0) * (float)idot + nf_ql_half(bp, 2) * (float)sumq);
}

/* ---- MXFP4: y = s * kv[nibble], kv the doubled E2M1 codes and s the half e8m0 scale. ---- */
NF_INLINE float MXFP4_decode_q8(NF_GLOBAL const uchar* bp, NF_GLOBAL const uchar* xq, int blk, int ltid) {
    const uint q = read_u32_le(bp + 1 + 4 * ltid);
    const int bo = ltid * 4;
    const int idot = NF_DP4A(nf_q8_fp4_word(q & 0x0F0F0F0Fu), nf_q8_word(xq, blk, bo), 0)
                   + NF_DP4A(nf_q8_fp4_word((q >> 4) & 0x0F0F0F0Fu), nf_q8_word(xq, blk, bo + 16), 0);
    return nf_q8_scale(xq, blk) * mxfp4_scale(bp[0]) * (float)idot;
}

/* Four kvalues_iq4nl codes packed as int8, nibble k in byte k. */
NF_INLINE int nf_q8_iq4nl_word(uint nibbles) {
    return nf_q8_pack4((int)IQ4_NL_KVALUES[nibbles & 0xFu],
                       (int)IQ4_NL_KVALUES[(nibbles >> 8) & 0xFu],
                       (int)IQ4_NL_KVALUES[(nibbles >> 16) & 0xFu],
                       (int)IQ4_NL_KVALUES[(nibbles >> 24) & 0xFu]);
}

/* ---- IQ4_NL: one 32-element block is one Q8_1 block; work-item ltid owns qs bytes 4*ltid .. +3. ---- */
NF_INLINE float IQ4_NL_decode_q8(NF_GLOBAL const uchar* bp, NF_GLOBAL const uchar* xq, int blk, int ltid) {
    const uint q = read_u32_le(bp + 2 + 4 * ltid);
    const int lo = nf_q8_iq4nl_word(q & 0x0F0F0F0Fu);
    const int hi = nf_q8_iq4nl_word((q >> 4) & 0x0F0F0F0Fu);
    const int bo = ltid * 4;
    const int idot = NF_DP4A(lo, nf_q8_word(xq, blk, bo), 0)
                   + NF_DP4A(hi, nf_q8_word(xq, blk, bo + 16), 0);
    return nf_q8_scale(xq, blk) * vload_half(0, (NF_GLOBAL const NF_HALF*)bp) * (float)idot;
}

/* ---- IQ4_XS: work-item ltid owns sub-block ltid/4 and its qs bytes 4*(ltid%4) .. +3. ---- */
NF_INLINE float IQ4_XS_decode_q8(NF_GLOBAL const uchar* bp, NF_GLOBAL const uchar* xq, int blk, int ltid) {
    const int sb = ltid / 4, bo = (ltid % 4) * 4;
    const uint q = read_u32_le(bp + IQ4_XS_HEADER_BYTES + sb * IQ4_XS_SUB_BLOCK_QUANT_BYTES + bo);
    const int lo = nf_q8_iq4nl_word(q & 0x0F0F0F0Fu);
    const int hi = nf_q8_iq4nl_word((q >> 4) & 0x0F0F0F0Fu);
    const int qb = blk * 8 + sb;
    const int idot = NF_DP4A(lo, nf_q8_word(xq, qb, bo), 0)
                   + NF_DP4A(hi, nf_q8_word(xq, qb, bo + 16), 0);
    return nf_q8_scale(xq, qb) * iq4xs_sub_scale(bp, sb) * (float)idot;
}

/* T_decode_q8_dual of the fused gate/up form: both rows over the same activation group. */
#define NF_QL_Q8_DUAL(N) \
NF_INLINE void N##_decode_q8_dual(NF_GLOBAL const uchar* gbp, NF_GLOBAL const uchar* ubp, \
                                  NF_GLOBAL const uchar* xq, int blk, int ltid, \
                                  float* gs, float* us) { \
    *gs += N##_decode_q8(gbp, xq, blk, ltid); \
    *us += N##_decode_q8(ubp, xq, blk, ltid); \
}

NF_QL_Q8_DUAL(Q4_0)
NF_QL_Q8_DUAL(Q4_1)
NF_QL_Q8_DUAL(Q5_0)
NF_QL_Q8_DUAL(Q5_1)
NF_QL_Q8_DUAL(MXFP4)

NF_INLINE void IQ4_NL_decode_q8_dual(NF_GLOBAL const uchar* gbp, NF_GLOBAL const uchar* ubp,
                                     NF_GLOBAL const uchar* xq, int blk, int ltid,
                                     float* gs, float* us) {
    *gs += IQ4_NL_decode_q8(gbp, xq, blk, ltid);
    *us += IQ4_NL_decode_q8(ubp, xq, blk, ltid);
}

NF_INLINE void IQ4_XS_decode_q8_dual(NF_GLOBAL const uchar* gbp, NF_GLOBAL const uchar* ubp,
                                     NF_GLOBAL const uchar* xq, int blk, int ltid,
                                     float* gs, float* us) {
    *gs += IQ4_XS_decode_q8(gbp, xq, blk, ltid);
    *us += IQ4_XS_decode_q8(ubp, xq, blk, ltid);
}

#endif /* NF_QUANT_DECODE_LEGACY_Q8_H */

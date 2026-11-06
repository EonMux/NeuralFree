/* common/quant_decode_iq_q8.h - integer-dot (Q8_1 activation) decoders of the i-quant super-blocks. Requires: common/dialect.h, common/config.h, common/quant_blocks.h, common/quant_decode_iq.h, common/q8_dot.h */
/*
 * T_decode_q8 / T_decode_q8_dual for IQ3_XXS, IQ3_S, IQ3_K, IQ3_KS, IQ2_XXS, IQ2_XS, IQ2_S, IQ1_S and
 * IQ1_M, with the argument list of common/matvec_family.h's int8 forms. The per-type group builders
 * of common/quant_decode_iq.h already reduce the 8 elements of work-item ltid to integer codes:
 *
 *   sign-magnitude (IQ2_*, IQ3_XXS, IQ3_S): code_j = magnitude_j * (sign_j ? -1 : 1), y = dl * code
 *   iq3nl (IQ3_K, IQ3_KS):                  code_j = IQ3NL_KVALUES[code_j + grid], y = dl * code
 *   ternary (IQ1_S, IQ1_M):                 y = dl * (code_j + delta), code_j in {-1, 0, 1}
 *
 * The 8 codes are packed as int8 and dotted with the Q8_1 codes of the work-item's group (elements
 * 8*ltid .. 8*ltid+7, in Q8_1 block blk * 8 + ltid / 4). The ternary delta term uses the sum of the
 * Q8_1 codes, which the same packed dot yields against NF_Q8_ONES.
 */
#ifndef NF_QUANT_DECODE_IQ_Q8_H
#define NF_QUANT_DECODE_IQ_Q8_H

#ifndef NF_Q8_DOT_H
#error "common/quant_decode_iq_q8.h requires common/q8_dot.h"
#endif

/* Q8_1 block and byte slice of work-item ltid's 8-element group of 256-element block blk. */
#define NF_IQD_Q8_BLOCK(blk, ltid) ((blk) * 8 + ((ltid) >> 2))
#define NF_IQD_Q8_BYTE(ltid)       (((ltid) & 3) * 8)

/* Four 3-bit iq3nl codes (code k at bits 3k) as IQ3NL_KVALUES codes, packed as int8. */
NF_INLINE int nf_q8_iq3nl_word(uint codes, uint grid) {
    return nf_q8_pack4((int)IQ3NL_KVALUES[((codes >> 0) & 7u) + grid],
                       (int)IQ3NL_KVALUES[((codes >> 3) & 7u) + grid],
                       (int)IQ3NL_KVALUES[((codes >> 6) & 7u) + grid],
                       (int)IQ3NL_KVALUES[((codes >> 9) & 7u) + grid]);
}

/* Dot of a sign-magnitude group with the Q8_1 activations of block blk. */
NF_INLINE float nf_q8_sgroup_dot(nf_iqd_sgroup g, NF_GLOBAL const uchar* xq, int blk, int ltid) {
    const int qb = NF_IQD_Q8_BLOCK(blk, ltid);
    const int bo = NF_IQD_Q8_BYTE(ltid);
    const int w0 = nf_q8_signed_mags(g.lo, g.signs);
    const int w1 = nf_q8_signed_mags(g.hi, g.signs >> 4);
    const int idot = NF_DP4A(w0, nf_q8_word(xq, qb, bo), 0)
                   + NF_DP4A(w1, nf_q8_word(xq, qb, bo + 4), 0);
    return nf_q8_scale(xq, qb) * g.dl * (float)idot;
}

/* Dot of an iq3nl group with the Q8_1 activations of block blk. */
NF_INLINE float nf_q8_kgroup_dot(nf_iqd_kgroup g, NF_GLOBAL const uchar* xq, int blk, int ltid) {
    const int qb = NF_IQD_Q8_BLOCK(blk, ltid);
    const int bo = NF_IQD_Q8_BYTE(ltid);
    const int idot = NF_DP4A(nf_q8_iq3nl_word(g.lo, g.grid), nf_q8_word(xq, qb, bo), 0)
                   + NF_DP4A(nf_q8_iq3nl_word(g.hi, g.grid), nf_q8_word(xq, qb, bo + 4), 0);
    return nf_q8_scale(xq, qb) * g.dl * (float)idot;
}

/* Dot of a ternary group with the Q8_1 activations of block blk: the delta term is delta * sum(q8). */
NF_INLINE float nf_q8_tgroup_dot(nf_iqd_tgroup g, NF_GLOBAL const uchar* xq, int blk, int ltid) {
    const int qb = NF_IQD_Q8_BLOCK(blk, ltid);
    const int bo = NF_IQD_Q8_BYTE(ltid);
    const int u0 = nf_q8_word(xq, qb, bo);
    const int u1 = nf_q8_word(xq, qb, bo + 4);
    const int idot = NF_DP4A((int)g.lo, u0, 0) + NF_DP4A((int)g.hi, u1, 0);
    const int sumq = NF_DP4A(NF_Q8_ONES, u0, 0) + NF_DP4A(NF_Q8_ONES, u1, 0);
    return nf_q8_scale(xq, qb) * g.dl * ((float)idot + g.delta * (float)sumq);
}

#define NF_IQD_Q8_TRIPLE(N, BUILD, KIND) \
NF_INLINE float N##_decode_q8(NF_GLOBAL const uchar* bp, NF_GLOBAL const uchar* xq, int blk, int ltid) { \
    return nf_q8_##KIND##group_dot(BUILD(bp, ltid >> 2, ltid & 3), xq, blk, ltid); \
} \
NF_INLINE void N##_decode_q8_dual(NF_GLOBAL const uchar* gbp, NF_GLOBAL const uchar* ubp, \
                                  NF_GLOBAL const uchar* xq, int blk, int ltid, \
                                  float* gs, float* us) { \
    *gs += nf_q8_##KIND##group_dot(BUILD(gbp, ltid >> 2, ltid & 3), xq, blk, ltid); \
    *us += nf_q8_##KIND##group_dot(BUILD(ubp, ltid >> 2, ltid & 3), xq, blk, ltid); \
}

NF_IQD_Q8_TRIPLE(IQ3_XXS, nf_iqd_iq3xxs, s)
NF_IQD_Q8_TRIPLE(IQ3_S,   nf_iqd_iq3s,   s)
NF_IQD_Q8_TRIPLE(IQ3_K,   nf_iqd_iq3k,   k)
NF_IQD_Q8_TRIPLE(IQ3_KS,  nf_iqd_iq3ks,  k)
NF_IQD_Q8_TRIPLE(IQ2_XXS, nf_iqd_iq2xxs, s)
NF_IQD_Q8_TRIPLE(IQ2_XS,  nf_iqd_iq2xs,  s)
NF_IQD_Q8_TRIPLE(IQ2_S,   nf_iqd_iq2s,   s)
NF_IQD_Q8_TRIPLE(IQ1_S,   nf_iqd_iq1s,   t)
NF_IQD_Q8_TRIPLE(IQ1_M,   nf_iqd_iq1m,   t)

#endif /* NF_QUANT_DECODE_IQ_Q8_H */

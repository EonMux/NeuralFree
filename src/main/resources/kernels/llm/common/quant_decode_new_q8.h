/* common/quant_decode_new_q8.h - integer-dot (Q8_1 activation) decoders of NVFP4. Requires: common/dialect.h, common/config.h, common/quant_blocks.h, common/quant_decode_new.h, common/q8_dot.h */
/*
 * T_decode_q8 / T_decode_q8_dual for NVFP4 with the argument list of common/matvec_family.h's int8
 * forms. An NVFP4 block is 64 elements, four 16-element sub-blocks with their own UE4M3 scale; the
 * eight work-items that share a block (TPS = 8) own 8 elements each: sub-block ltid/2, its qs bytes
 * 4*(ltid%2) .. +3, i.e. the four low-nibble elements and their high-nibble partners 8 elements on.
 * Both nibble groups lie in one 32-element Q8_1 block, so the value is the doubled E2M1 code times
 * d8 * the halved sub-block scale, dotted as int8 codes by NF_DP4A.
 */
#ifndef NF_QUANT_DECODE_NEW_Q8_H
#define NF_QUANT_DECODE_NEW_Q8_H

#ifndef NF_Q8_DOT_H
#error "common/quant_decode_new_q8.h requires common/q8_dot.h"
#endif

#ifndef NF_QUANT_DECODE_NEW_H
#error "common/quant_decode_new_q8.h requires common/quant_decode_new.h"
#endif

/* ---- NVFP4: y = ue4m3_half(d[s]) * kvalues_fp4[nibble] of the sub-block s of the element. ---- */
NF_INLINE float NVFP4_decode_q8(NF_GLOBAL const uchar* bp, NF_GLOBAL const uchar* xq, int blk, int ltid) {
    const int sb = ltid >> 1, j0 = (ltid & 1) * 4;
    const uint q = read_u32_le(bp + NF_NVFP4_QS + sb * 8 + j0);
    const int qb = blk * 2 + (ltid >> 2);
    const int bo = (sb & 1) * 16 + j0;
    const int idot = NF_DP4A(nf_q8_fp4_word(q & 0x0F0F0F0Fu), nf_q8_word(xq, qb, bo), 0)
                   + NF_DP4A(nf_q8_fp4_word((q >> 4) & 0x0F0F0F0Fu), nf_q8_word(xq, qb, bo + 8), 0);
    return nf_q8_scale(xq, qb) * nf_nq_ue4m3_half((uint)bp[NF_NVFP4_D + sb]) * (float)idot;
}

NF_INLINE void NVFP4_decode_q8_dual(NF_GLOBAL const uchar* gbp, NF_GLOBAL const uchar* ubp,
                                    NF_GLOBAL const uchar* xq, int blk, int ltid,
                                    float* gs, float* us) {
    *gs += NVFP4_decode_q8(gbp, xq, blk, ltid);
    *us += NVFP4_decode_q8(ubp, xq, blk, ltid);
}

#endif /* NF_QUANT_DECODE_NEW_Q8_H */

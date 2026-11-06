/* common/quant_new_q8.glsl - integer-dot (Q8_1 activation) decoder of NVFP4.
 * GLSL twin of kernels/llm/common/quant_decode_new_q8.h. Requires: common/quant_blocks.glsl,
 * common/q8_dot.glsl, and the QT_BS/QT_BB/QT_TPS geometry of QTYPE == QT_NVFP4.
 *
 * A 64-element NVFP4 block has four 16-element sub-blocks with their own UE4M3 scale; the eight
 * invocations that share a block (QT_TPS = 8) own 8 elements each: sub-block ltid/2, its qs bytes
 * 4*(ltid%2) .. +3, i.e. the four low-nibble elements and their high-nibble partners 8 elements on.
 * Both nibble groups lie in one 32-element Q8_1 block, so the value is the doubled E2M1 code times
 * d8 * the halved sub-block scale, dotted as int8 codes with dotPacked4x8EXT.
 */
#ifndef NF_QUANT_NEW_Q8_GLSL
#define NF_QUANT_NEW_Q8_GLSL

#ifndef NF_Q8_DOT_GLSL
#error "common/quant_new_q8.glsl requires common/q8_dot.glsl"
#endif

#ifdef QTYPE
#if QTYPE == QT_NVFP4

#define QT_BS  NF_NVFP4_BS
#define QT_BB  NF_NVFP4_BB
#define QT_TPS 8

/* Eight elements of sub-block s = ltid/2: elements 16*s + j0 + k (low nibbles) and +8 (high). */
float qt_dot_q8(uint ws, uint bp, uint blk, uint ltid) {
    const uint sb = ltid >> 1, j0 = (ltid & 1u) * 4u;
    const uint q = nf_w_u32(ws, bp + 4u + sb * 8u + j0);
    const uint qb = blk * 2u + (ltid >> 2);
    const uint bo = (sb & 1u) * 16u + j0;
    const int idot = nf_q8_dp4(nf_q8_fp4_word(q & 0x0F0F0F0Fu), nf_q8_word(qb, bo))
                   + nf_q8_dp4(nf_q8_fp4_word((q >> 4) & 0x0F0F0F0Fu), nf_q8_word(qb, bo + 8u));
    return nf_q8_scale_of(qb) * nf_ue4m3_half(nf_w_u8(ws, bp + sb)) * float(idot);
}

void qt_dot_q8_dual(uint gbp, uint ubp, uint blk, uint ltid, inout float gs, inout float us) {
    gs += qt_dot_q8(NF_WS_A, gbp, blk, ltid);
    us += qt_dot_q8(NF_WS_B, ubp, blk, ltid);
}

#endif /* QT_NVFP4 */
#endif /* QTYPE */

#endif /* NF_QUANT_NEW_Q8_GLSL */

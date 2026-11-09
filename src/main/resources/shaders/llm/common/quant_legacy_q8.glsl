/* common/quant_legacy_q8.glsl - integer-dot (Q8_1 activation) decoders of Q4_0, Q4_1, Q5_0, Q5_1,
 * IQ4_NL, MXFP4 and IQ4_XS. GLSL twin of kernels/llm/common/quant_decode_legacy_q8.h. Requires:
 * common/quant_blocks.glsl, common/q8_dot.glsl, and the QT_BS/QT_BB/QT_TPS geometry of the selected
 * QTYPE.
 *
 * Every type has 32-element (sub-)blocks that map one to one onto the Q8_1 blocks, and the four
 * invocations that share a block (QT_TPS = 4) own its qs bytes 4*ltid .. 4*ltid+3, i.e. elements
 * 4*ltid .. +3 and their high-nibble partners 4*ltid+16 .. +3. The direct-value types dot the packed
 * int8 codes with the Q8_1 codes; the bias/min types (Q4_0, Q5_0, Q4_1, Q5_1) add the sum of the owned
 * Q8_1 codes, so the invocations of a block add up to the block's s term.
 *
 * The term functions are precise: the float chain they return is computed exactly as written
 * (SPIR-V NoContraction), so every kernel that inlines a term computes the same operations and the
 * same rounding, whatever mul-add fusion the driver would otherwise pick per inlined copy. A
 * token's {T}_MatVecN dot is therefore its stacked {T}_MatVec dots bit for bit.
 */
#ifndef NF_QUANT_LEGACY_Q8_GLSL
#define NF_QUANT_LEGACY_Q8_GLSL

#ifndef NF_Q8_DOT_GLSL
#error "common/quant_legacy_q8.glsl requires common/q8_dot.glsl"
#endif

#ifdef QTYPE
#if QTYPE == QT_Q4_0 || QTYPE == QT_Q4_1 || QTYPE == QT_Q5_0 || QTYPE == QT_Q5_1 \
 || QTYPE == QT_IQ4_NL || QTYPE == QT_MXFP4 || QTYPE == QT_IQ4_XS

/* Block geometry of the integer-dot forms (matvec_q8.comp). */
#if QTYPE == QT_IQ4_NL
#define QT_BS  32
#define QT_BB  NF_IQ4_NL_BB
#define QT_TPS 4
#elif QTYPE == QT_Q4_0
#define QT_BS  NF_Q4_0_BS
#define QT_BB  NF_Q4_0_BB
#define QT_TPS 4
#elif QTYPE == QT_Q4_1
#define QT_BS  NF_Q4_1_BS
#define QT_BB  NF_Q4_1_BB
#define QT_TPS 4
#elif QTYPE == QT_Q5_0
#define QT_BS  NF_Q5_0_BS
#define QT_BB  NF_Q5_0_BB
#define QT_TPS 4
#elif QTYPE == QT_Q5_1
#define QT_BS  NF_Q5_1_BS
#define QT_BB  NF_Q5_1_BB
#define QT_TPS 4
#elif QTYPE == QT_MXFP4
#define QT_BS  NF_MXFP4_BS
#define QT_BB  NF_MXFP4_BB
#define QT_TPS 4
#else
#define QT_BS  256
#define QT_BB  NF_IQ4_XS_BB
#define QT_TPS 32
#endif

#if QTYPE == QT_IQ4_NL || QTYPE == QT_IQ4_XS || QTYPE == QT_MXFP4

/* Four kvalues_iq4nl codes packed as int8, nibble k in byte k. */
int nf_q8_iq4nl_word(uint nibbles) {
    return nf_q8_pack4(nf_kvalue_iq4nl(nibbles & 0xFu),
                       nf_kvalue_iq4nl((nibbles >> 8) & 0xFu),
                       nf_kvalue_iq4nl((nibbles >> 16) & 0xFu),
                       nf_kvalue_iq4nl((nibbles >> 24) & 0xFu));
}

#endif

#if QTYPE == QT_Q4_0
/* Eight elements: y = d*(code - 8), the sampled nibbles in r0/r1. */
precise float qt_dot_q8(uint ws, uint bp, uint blk, uint ltid) {
    const uint q = nf_w_u32_even(ws, bp + 2u + 4u * ltid);
    const uint bo = ltid * 4u;
    const int u0 = nf_q8_word(blk, bo);
    const int u1 = nf_q8_word(blk, bo + 16u);
    const int idot = nf_q8_dp4(int(q & 0x0F0F0F0Fu), u0) + nf_q8_dp4(int((q >> 4) & 0x0F0F0F0Fu), u1);
    const int sumq = nf_q8_sum4(u0) + nf_q8_sum4(u1);
    return nf_q8_scale_of(blk) * nf_w_f16(ws, bp) * float(idot - 8 * sumq);
}

#elif QTYPE == QT_Q4_1
/* Eight elements: y = d*code + m. */
precise float qt_dot_q8(uint ws, uint bp, uint blk, uint ltid) {
    const uint q = nf_w_u32(ws, bp + 4u + 4u * ltid);
    const uint bo = ltid * 4u;
    const int u0 = nf_q8_word(blk, bo);
    const int u1 = nf_q8_word(blk, bo + 16u);
    const int idot = nf_q8_dp4(int(q & 0x0F0F0F0Fu), u0) + nf_q8_dp4(int((q >> 4) & 0x0F0F0F0Fu), u1);
    const int sumq = nf_q8_sum4(u0) + nf_q8_sum4(u1);
    return nf_q8_scale_of(blk) * (nf_w_f16(ws, bp) * float(idot) + nf_w_f16(ws, bp + 2u) * float(sumq));
}

#elif QTYPE == QT_Q5_0 || QTYPE == QT_Q5_1
/* Fifth bits (as 0x10) of the four elements e0 .. e0+3 of a Q5_0/Q5_1 qh word. */
int nf_q8_q5_hbits(uint qh, uint e0) {
    const uint b = (qh >> e0) & 0xFu;
    const uint packed = ((b >> 0) & 1u) | (((b >> 1) & 1u) << 8)
                      | (((b >> 2) & 1u) << 16) | (((b >> 3) & 1u) << 24);
    return int(packed << 4);
}

/* Eight elements: y = d*(code - 16) for Q5_0, y = d*code + m for Q5_1. */
precise float qt_dot_q8(uint ws, uint bp, uint blk, uint ltid) {
    const bool q5_1 = QTYPE == QT_Q5_1;
    /* Q5_0 blocks are 22 bytes, so qh at bp+2 is only 2-aligned; Q5_1's qh at bp+4 is 4-aligned. */
    const uint qh = q5_1 ? nf_w_u32(ws, bp + 4u) : nf_w_u32_even(ws, bp + 2u);
    const uint q = q5_1 ? nf_w_u32(ws, bp + 8u + 4u * ltid) : nf_w_u32_even(ws, bp + 6u + 4u * ltid);
    const uint bo = ltid * 4u;
    const int u0 = nf_q8_word(blk, bo);
    const int u1 = nf_q8_word(blk, bo + 16u);
    const int c0 = int(q & 0x0F0F0F0Fu) | nf_q8_q5_hbits(qh, bo);
    const int c1 = int((q >> 4) & 0x0F0F0F0Fu) | nf_q8_q5_hbits(qh, bo + 16u);
    const int idot = nf_q8_dp4(c0, u0) + nf_q8_dp4(c1, u1);
    const int sumq = nf_q8_sum4(u0) + nf_q8_sum4(u1);
    if (q5_1) {
        return nf_q8_scale_of(blk) * (nf_w_f16(ws, bp) * float(idot) + nf_w_f16(ws, bp + 2u) * float(sumq));
    }
    return nf_q8_scale_of(blk) * nf_w_f16(ws, bp) * float(idot - 16 * sumq);
}

#elif QTYPE == QT_MXFP4
/* Eight elements: y = s * kv[nibble], kv the doubled E2M1 codes and s the half e8m0 scale. */
precise float qt_dot_q8(uint ws, uint bp, uint blk, uint ltid) {
    const uint q = nf_w_u32_any(ws, bp + 1u + 4u * ltid);
    const uint bo = ltid * 4u;
    const int idot = nf_q8_dp4(nf_q8_fp4_word(q & 0x0F0F0F0Fu), nf_q8_word(blk, bo))
                   + nf_q8_dp4(nf_q8_fp4_word((q >> 4) & 0x0F0F0F0Fu), nf_q8_word(blk, bo + 16u));
    return nf_q8_scale_of(blk) * nf_e8m0_half(nf_w_u8(ws, bp)) * float(idot);
}

#elif QTYPE == QT_IQ4_NL
/* One 32-element block is one Q8_1 block; invocation ltid owns qs bytes 4*ltid .. +3. */
precise float qt_dot_q8(uint ws, uint bp, uint blk, uint ltid) {
    const uint q = nf_w_u32_even(ws, bp + 2u + 4u * ltid);
    const int lo = nf_q8_iq4nl_word(q & 0x0F0F0F0Fu);
    const int hi = nf_q8_iq4nl_word((q >> 4) & 0x0F0F0F0Fu);
    const uint bo = ltid * 4u;
    const int idot = nf_q8_dp4(lo, nf_q8_word(blk, bo))
                   + nf_q8_dp4(hi, nf_q8_word(blk, bo + 16u));
    return nf_q8_scale_of(blk) * nf_w_f16(ws, bp) * float(idot);
}

#else /* QT_IQ4_XS */
/* Invocation ltid owns sub-block ltid/4 and its qs bytes 4*(ltid%4) .. +3. */
precise float qt_dot_q8(uint ws, uint bp, uint blk, uint ltid) {
    const uint sb = ltid >> 2, bo = (ltid & 3u) * 4u;
    const uint q = nf_w_u32(ws, bp + 8u + sb * 16u + bo);
    const int lo = nf_q8_iq4nl_word(q & 0x0F0F0F0Fu);
    const int hi = nf_q8_iq4nl_word((q >> 4) & 0x0F0F0F0Fu);
    const uint qb = blk * 8u + sb;
    const int idot = nf_q8_dp4(lo, nf_q8_word(qb, bo))
                   + nf_q8_dp4(hi, nf_q8_word(qb, bo + 16u));
    return nf_q8_scale_of(qb) * nf_iq4xs_sub_scale(ws, bp, sb) * float(idot);
}
#endif

void qt_dot_q8_dual(uint gbp, uint ubp, uint blk, uint ltid, inout float gs, inout float us) {
    gs += qt_dot_q8(NF_WS_A, gbp, blk, ltid);
    us += qt_dot_q8(NF_WS_B, ubp, blk, ltid);
}

#endif /* legacy QTYPE */
#endif /* QTYPE */

#endif /* NF_QUANT_LEGACY_Q8_GLSL */

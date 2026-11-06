/* common/quant_k_q8.glsl - integer-dot (Q8_1 activation) decoders of the K-quant super-blocks.
 * GLSL twin of kernels/llm/common/quant_decode_k_q8.h. Requires: common/quant_blocks.glsl,
 * common/q8_dot.glsl.
 *
 * For the selected QTYPE the shader gets
 *     float qt_dot_q8(uint ws, uint bp, uint blk, uint ltid)
 *         contribution of invocation ltid's 8 elements of 256-element block blk (bytes at bp of
 *         weight stream ws) to the dot with the Q8_1 activations of block blk, and
 *     void qt_dot_q8_dual(uint gbp, uint ubp, uint blk, uint ltid, inout float gs, inout float us)
 * with the same work split as qt_dot (8 contiguous elements per invocation, one 32-element group,
 * hence one Q8_1 block blk * 8 + ltid / 4).
 */
#ifndef NF_QUANT_K_Q8_GLSL
#define NF_QUANT_K_Q8_GLSL

#ifndef NF_Q8_DOT_GLSL
#error "common/quant_k_q8.glsl requires common/q8_dot.glsl"
#endif

#ifdef QTYPE
#if QTYPE == QT_Q2_K || QTYPE == QT_Q3_K || QTYPE == QT_Q4_K || QTYPE == QT_Q5_K || QTYPE == QT_Q6_K || QTYPE == QT_Q8_K

/* Block geometry of the integer-dot forms (matvec_q8.comp). */
#define QT_BS  256
#define QT_TPS 32

/* Q8_1 block and byte slice of invocation ltid's group of block blk. */
uint nf_kq_q8_block(uint blk, uint ltid) {
    return blk * 8u + (ltid >> 2);
}

uint nf_kq_q8_byte(uint ltid) {
    return (ltid & 3u) * 8u;
}

#if QTYPE == QT_Q2_K
#define QT_BB NF_Q2_K_BB
/* y = d*(sc & 15) * code - dmin*(sc >> 4), code in [0, 4). */
float nf_kq_q8_part(uint ws, uint bp, uint blk, uint ltid) {
    const uint sc = nf_w_u8(ws, bp + 0u + (ltid >> 1));
    const uint shift = ((ltid & 15u) >> 2) * 2u;
    const uint q = bp + 16u + (ltid >> 4) * 32u + ((ltid >> 1) & 1u) * 16u + (ltid & 1u) * 8u;
    const uint qb = nf_kq_q8_block(blk, ltid), bo = nf_kq_q8_byte(ltid);
    const int u0 = nf_q8_word(qb, bo), u1 = nf_q8_word(qb, bo + 4u);
    const int c0 = int((nf_w_u32(ws, q) >> shift) & 0x03030303u);
    const int c1 = int((nf_w_u32(ws, q + 4u) >> shift) & 0x03030303u);
    const float dl = nf_w_f16(ws, bp + 80u) * float(sc & 0xFu);
    const float ml = nf_w_f16(ws, bp + 82u) * float(sc >> 4);
    const int idot = nf_q8_dp4(c0, u0) + nf_q8_dp4(c1, u1);
    const int sumq = nf_q8_sum4(u0) + nf_q8_sum4(u1);
    return nf_q8_scale_of(qb) * (dl * float(idot) - ml * float(sumq));
}

#elif QTYPE == QT_Q3_K
#define QT_BB NF_Q3_K_BB
/* Signed 6-bit scale si (nf_q3k_scale of common/quant_k.glsl, repeated here so this file does not
 * require the float decoders). */
int nf_kq_q8_q3k_scale(uint ws, uint bp, uint si) {
    const uint lo = (nf_w_u8(ws, bp + 96u + (si & 7u)) >> ((si >> 3) * 4u)) & 0xFu;
    const uint hi = (nf_w_u8(ws, bp + 104u + (si & 3u)) >> ((si >> 2) * 2u)) & 3u;
    return int(lo | (hi << 4)) - 32;
}

/* y = d*scale * code, code in [-4, 4). */
float nf_kq_q8_part(uint ws, uint bp, uint blk, uint ltid) {
    const float dl = nf_w_f16(ws, bp + 108u) * float(nf_kq_q8_q3k_scale(ws, bp, ltid >> 1));
    const uint hf = ltid >> 4, j = (ltid & 15u) >> 2;
    const uint shift = j * 2u, mbit = hf * 4u + j;
    const uint q = bp + 32u + hf * 32u + ((ltid >> 1) & 1u) * 16u + (ltid & 1u) * 8u;
    const uint hm = bp + 0u + (ltid & 3u) * 8u;
    const uint qb = nf_kq_q8_block(blk, ltid), bo = nf_kq_q8_byte(ltid);
    const int u0 = nf_q8_word(qb, bo), u1 = nf_q8_word(qb, bo + 4u);
    const int c0 = int(((nf_w_u32_even(ws, q) >> shift) & 0x03030303u)
                     | (((nf_w_u32_even(ws, hm) >> mbit) & 0x01010101u) << 2));
    const int c1 = int(((nf_w_u32_even(ws, q + 4u) >> shift) & 0x03030303u)
                     | (((nf_w_u32_even(ws, hm + 4u) >> mbit) & 0x01010101u) << 2));
    const int idot = nf_q8_dp4(c0, u0) + nf_q8_dp4(c1, u1) - 4 * (nf_q8_sum4(u0) + nf_q8_sum4(u1));
    return nf_q8_scale_of(qb) * dl * float(idot);
}

#elif QTYPE == QT_Q4_K
#define QT_BB NF_Q4_K_BB
/* y = d*sc * code - dmin*m, code in [0, 16). */
float nf_kq_q8_part(uint ws, uint bp, uint blk, uint ltid) {
    const uint si = ltid >> 2;
    const vec2 sm = nf_k4_scale_min(ws, bp + 4u, si);
    const uint sh = (si & 1u) * 4u;
    const uint q = bp + 16u + (si >> 1) * 32u + (ltid & 3u) * 8u;
    const uint qb = nf_kq_q8_block(blk, ltid), bo = nf_kq_q8_byte(ltid);
    const int u0 = nf_q8_word(qb, bo), u1 = nf_q8_word(qb, bo + 4u);
    const int c0 = int((nf_w_u32(ws, q) >> sh) & 0x0F0F0F0Fu);
    const int c1 = int((nf_w_u32(ws, q + 4u) >> sh) & 0x0F0F0F0Fu);
    const float d = nf_w_f16(ws, bp + 0u);
    const float dm = nf_w_f16(ws, bp + 2u);
    const int idot = nf_q8_dp4(c0, u0) + nf_q8_dp4(c1, u1);
    const int sumq = nf_q8_sum4(u0) + nf_q8_sum4(u1);
    return nf_q8_scale_of(qb) * (d * sm.x * float(idot) - dm * sm.y * float(sumq));
}

#elif QTYPE == QT_Q5_K
#define QT_BB NF_Q5_K_BB
/* y = d*sc * code - dmin*m, code in [0, 32). */
float nf_kq_q8_part(uint ws, uint bp, uint blk, uint ltid) {
    const uint si = ltid >> 2;
    const vec2 sm = nf_k4_scale_min(ws, bp + 4u, si);
    const uint sh = (si & 1u) * 4u;
    const uint q = bp + 48u + (si >> 1) * 32u + (ltid & 3u) * 8u;
    const uint qh = bp + 16u + (ltid & 3u) * 8u;
    const uint qb = nf_kq_q8_block(blk, ltid), bo = nf_kq_q8_byte(ltid);
    const int u0 = nf_q8_word(qb, bo), u1 = nf_q8_word(qb, bo + 4u);
    const int c0 = int(((nf_w_u32(ws, q) >> sh) & 0x0F0F0F0Fu)
                     | (((nf_w_u32(ws, qh) >> si) & 0x01010101u) << 4));
    const int c1 = int(((nf_w_u32(ws, q + 4u) >> sh) & 0x0F0F0F0Fu)
                     | (((nf_w_u32(ws, qh + 4u) >> si) & 0x01010101u) << 4));
    const float d = nf_w_f16(ws, bp + 0u);
    const float dm = nf_w_f16(ws, bp + 2u);
    const int idot = nf_q8_dp4(c0, u0) + nf_q8_dp4(c1, u1);
    const int sumq = nf_q8_sum4(u0) + nf_q8_sum4(u1);
    return nf_q8_scale_of(qb) * (d * sm.x * float(idot) - dm * sm.y * float(sumq));
}

#elif QTYPE == QT_Q6_K
#define QT_BB NF_Q6_K_BB
/* y = d*scale * code, code in [-32, 32). */
float nf_kq_q8_part(uint ws, uint bp, uint blk, uint ltid) {
    const uint hf = ltid >> 4, t = (ltid >> 2) & 3u, l0 = (ltid & 3u) * 8u;
    const int sc = nf_w_i8(ws, bp + 192u + hf * 8u + t * 2u + (l0 >> 4));
    const float dl = nf_w_f16(ws, bp + 208u) * float(sc);
    const uint qlsh = (t >> 1) * 4u, qhsh = t * 2u;
    const uint ql = bp + 0u + hf * 64u + (t & 1u) * 32u + l0;
    const uint qh = bp + 128u + hf * 32u + l0;
    const uint qb = nf_kq_q8_block(blk, ltid), bo = nf_kq_q8_byte(ltid);
    const int u0 = nf_q8_word(qb, bo), u1 = nf_q8_word(qb, bo + 4u);
    const int c0 = int(((nf_w_u32_even(ws, ql) >> qlsh) & 0x0F0F0F0Fu)
                     | (((nf_w_u32_even(ws, qh) >> qhsh) & 0x03030303u) << 4));
    const int c1 = int(((nf_w_u32_even(ws, ql + 4u) >> qlsh) & 0x0F0F0F0Fu)
                     | (((nf_w_u32_even(ws, qh + 4u) >> qhsh) & 0x03030303u) << 4));
    const int idot = nf_q8_dp4(c0, u0) + nf_q8_dp4(c1, u1)
                   - 32 * (nf_q8_sum4(u0) + nf_q8_sum4(u1));
    return nf_q8_scale_of(qb) * dl * float(idot);
}

#elif QTYPE == QT_Q8_K
#define QT_BB NF_Q8_K_BB
/* y = d * qs[e], qs int8. */
float nf_kq_q8_part(uint ws, uint bp, uint blk, uint ltid) {
    const float d = nf_w_f32(ws, bp + 0u);
    const uint qs = bp + 4u + ltid * 8u;
    const uint qb = nf_kq_q8_block(blk, ltid), bo = nf_kq_q8_byte(ltid);
    const int idot = nf_q8_dp4(int(nf_w_u32(ws, qs)), nf_q8_word(qb, bo))
                   + nf_q8_dp4(int(nf_w_u32(ws, qs + 4u)), nf_q8_word(qb, bo + 4u));
    return nf_q8_scale_of(qb) * d * float(idot);
}
#endif

float qt_dot_q8(uint ws, uint bp, uint blk, uint ltid) {
    return nf_kq_q8_part(ws, bp, blk, ltid);
}

void qt_dot_q8_dual(uint gbp, uint ubp, uint blk, uint ltid, inout float gs, inout float us) {
    gs += nf_kq_q8_part(NF_WS_A, gbp, blk, ltid);
    us += nf_kq_q8_part(NF_WS_B, ubp, blk, ltid);
}

#endif /* K-quant QTYPE */
#endif /* QTYPE */

#endif /* NF_QUANT_K_Q8_GLSL */

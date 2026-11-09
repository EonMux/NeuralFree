/* common/quant_iq_q8.glsl - integer-dot (Q8_1 activation) decoders of the i-quant super-blocks.
 * GLSL twin of kernels/llm/common/quant_decode_iq_q8.h. Requires: common/iq_tables.glsl,
 * common/quant_iq.glsl, common/q8_dot.glsl.
 *
 * Uses the per-type nf_iqd_group(ws, bp, sb, grp) builders of common/quant_iq.glsl: the 8 elements
 * of invocation ltid are its group (sb = ltid / 4, grp = ltid % 4), in Q8_1 block blk * 8 + sb.
 *   sign-magnitude (IQ2_*, IQ3_XXS, IQ3_S): code = magnitude with its sign bit applied
 *   iq3nl (IQ3_K, IQ3_KS):                 code = iq3nl_values[3-bit code + grid]
 *   ternary (IQ1_*, IQ1_M):                code in {-1, 0, 1}; the delta term uses sum(q8)
 *
 * The term functions are precise: the float chain they return is computed exactly as written
 * (SPIR-V NoContraction), so every kernel that inlines a term computes the same operations and the
 * same rounding, whatever mul-add fusion the driver would otherwise pick per inlined copy. A
 * token's {T}_MatVecN dot is therefore its stacked {T}_MatVec dots bit for bit.
 */
#ifndef NF_QUANT_IQ_Q8_GLSL
#define NF_QUANT_IQ_Q8_GLSL

#ifndef NF_Q8_DOT_GLSL
#error "common/quant_iq_q8.glsl requires common/q8_dot.glsl"
#endif

#ifdef QTYPE
#if defined(NF_IQD_SIGNED) || defined(NF_IQD_TERNARY) || defined(NF_IQD_KNL)

/* Four 3-bit iq3nl codes (code k at bits 3k) looked up on the shifted grid, packed as int8. */
int nf_q8_iq3nl_word(uint codes, uint grid) {
    return nf_q8_pack4(nf_kvalue_iq3nl((codes & 7u) + grid),
                       nf_kvalue_iq3nl(((codes >> 3) & 7u) + grid),
                       nf_kvalue_iq3nl(((codes >> 6) & 7u) + grid),
                       nf_kvalue_iq3nl(((codes >> 9) & 7u) + grid));
}

#if defined(NF_IQD_SIGNED)
precise float nf_q8_group_dot(NfIqdGroup g, uint blk, uint ltid) {
    const uint qb = blk * 8u + (ltid >> 2);
    const uint bo = (ltid & 3u) * 8u;
    const int w0 = nf_q8_signed_mags(g.lo, g.signs);
    const int w1 = nf_q8_signed_mags(g.hi, g.signs >> 4);
    const int idot = nf_q8_dp4(w0, nf_q8_word(qb, bo)) + nf_q8_dp4(w1, nf_q8_word(qb, bo + 4u));
    return nf_q8_scale_of(qb) * g.dl * float(idot);
}
#elif defined(NF_IQD_KNL)
precise float nf_q8_group_dot(NfIqdGroup g, uint blk, uint ltid) {
    const uint qb = blk * 8u + (ltid >> 2);
    const uint bo = (ltid & 3u) * 8u;
    const int idot = nf_q8_dp4(nf_q8_iq3nl_word(g.lo, g.grid), nf_q8_word(qb, bo))
                   + nf_q8_dp4(nf_q8_iq3nl_word(g.hi, g.grid), nf_q8_word(qb, bo + 4u));
    return nf_q8_scale_of(qb) * g.dl * float(idot);
}
#else
/* Four ternary values of a packed grid entry half as int8 codes. */
int nf_q8_tvals_word(uint vals, uint h) {
    const vec4 v = nf_iqd_tvals4(vals, h);
    return nf_q8_pack4(int(v.x), int(v.y), int(v.z), int(v.w));
}

precise float nf_q8_group_dot(NfIqdGroup g, uint blk, uint ltid) {
    const uint qb = blk * 8u + (ltid >> 2);
    const uint bo = (ltid & 3u) * 8u;
    const int u0 = nf_q8_word(qb, bo), u1 = nf_q8_word(qb, bo + 4u);
    const int idot = nf_q8_dp4(nf_q8_tvals_word(g.vals, 0u), u0)
                   + nf_q8_dp4(nf_q8_tvals_word(g.vals, 1u), u1);
    return nf_q8_scale_of(qb) * g.dl * (float(idot) + g.delta * float(nf_q8_sum4(u0) + nf_q8_sum4(u1)));
}
#endif

float qt_dot_q8(uint ws, uint bp, uint blk, uint ltid) {
    return nf_q8_group_dot(nf_iqd_group(ws, bp, ltid >> 2, ltid & 3u), blk, ltid);
}

void qt_dot_q8_dual(uint gbp, uint ubp, uint blk, uint ltid, inout float gs, inout float us) {
    gs += nf_q8_group_dot(nf_iqd_group(NF_WS_A, gbp, ltid >> 2, ltid & 3u), blk, ltid);
    us += nf_q8_group_dot(nf_iqd_group(NF_WS_B, ubp, ltid >> 2, ltid & 3u), blk, ltid);
}

#endif /* i-quant QTYPE */
#endif /* QTYPE */

#endif /* NF_QUANT_IQ_Q8_GLSL */

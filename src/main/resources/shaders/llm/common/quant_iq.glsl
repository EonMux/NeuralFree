/* common/quant_iq.glsl - QT_DECODE_* decoders for IQ3_XXS, IQ3_S, IQ3_K, IQ3_KS, IQ2_XXS, IQ2_XS, IQ2_S,
 * IQ1_S and IQ1_M. GLSL twin of kernels/llm/common/quant_decode_iq.h. Requires: common/nf_common.glsl,
 * common/quant_blocks.glsl, common/iq_tables.glsl.
 *
 * Math: llama.cpp ggml/src/ggml-quants.c dequantize_row_iq2_xxs (2489), _iq2_xs (2517), _iq2_s (2544),
 * _iq3_xxs (2576), _iq3_s (2608), _iq1_s (2651) and _iq1_m (2676); IQ3_K and IQ3_KS follow Avalon
 * ggml/src/ggml-quants.c dequantize_row_iq3_k (5381) and _iq3_ks (5613).
 *
 * Work split (QT_TPS = 32): invocation ltid of a block owns the 8-element group ltid = 4*sb + grp,
 * sb in [0, 8) the 32-element sub-block and grp in [0, 4) the group inside it, i.e. block elements
 * 8*ltid .. 8*ltid + 7. Each type reduces a group to an NfIqdGroup:
 *   sign-magnitude kinds (IQ2_*, IQ3_*): y[j] = dl * mag[j] * (bit j of signs ? -1 : 1),
 *                                        mag[j] = byte j of the grid bytes (lo: 0-3, hi: 4-7)
 *   ternary kinds (IQ1_*):               y[j] = dl * (val[j] + delta), val[j] in {-1, 0, 1} is the
 *                                        2-bit field j of the packed nf_iq1s grid entry
 *   iq3nl kinds (IQ3_K, IQ3_KS):         y[j] = dl * iq3nl_values[code j], a 3-bit grid code
 * Weights are read only through nf_w_*(ws, off) (every block starts at an even byte offset, so
 * nf_w_u16 / nf_w_u32_even are used at even in-block offsets), activations through qt_x4, and
 * grids through the nf_iq* accessors of common/iq_tables.glsl.
 */
#ifndef NF_QUANT_IQ_GLSL
#define NF_QUANT_IQ_GLSL

#ifndef NF_IQ_TABLES_GLSL
#error "common/quant_iq.glsl requires common/iq_tables.glsl"
#endif

#ifdef QTYPE

#if QTYPE == QT_IQ1_S || QTYPE == QT_IQ1_M
#define NF_IQD_TERNARY 1
#elif QTYPE == QT_IQ3_XXS || QTYPE == QT_IQ3_S || QTYPE == QT_IQ2_XXS || QTYPE == QT_IQ2_XS || QTYPE == QT_IQ2_S
#define NF_IQD_SIGNED 1
#elif QTYPE == QT_IQ3_K || QTYPE == QT_IQ3_KS
#define NF_IQD_KNL 1
#endif

/* ---- group descriptors ---- */

#ifdef NF_IQD_SIGNED
struct NfIqdGroup {
    float dl;     /* group scale */
    uint lo;      /* grid magnitude bytes 0-3, byte j at bits 8j */
    uint hi;      /* grid magnitude bytes 4-7 */
    uint signs;   /* bit j set: element j is negative */
};

/* sum_k x[k] * mag[k] * sign[k]; signs bit k belongs to element k. */
float nf_iqd_sdot4(vec4 x, uint mag, uint signs) {
    const vec4 m = vec4(uvec4(mag, mag >> 8, mag >> 16, mag >> 24) & 0xFFu);
    const vec4 s = vec4(1.0) - 2.0 * vec4(uvec4(signs, signs >> 1, signs >> 2, signs >> 3) & 1u);
    return dot(x, m * s);
}

/* Dot of a group with its activations x0 (elements 0-3) and x1 (elements 4-7). */
float nf_iqd_dot(NfIqdGroup g, vec4 x0, vec4 x1) {
    return g.dl * (nf_iqd_sdot4(x0, g.lo, g.signs) + nf_iqd_sdot4(x1, g.hi, g.signs >> 4));
}

/* Element e in [0, 8) of a group. */
float nf_iqd_elem(NfIqdGroup g, uint e) {
    const uint mag = bitfieldExtract(e < 4u ? g.lo : g.hi, int((e & 3u) * 8u), 8);
    const float sgn = ((g.signs >> e) & 1u) != 0u ? -1.0 : 1.0;
    return g.dl * float(mag) * sgn;
}

/* Group from one 8-byte grid entry (x: bytes 0-3, y: bytes 4-7). */
NfIqdGroup nf_iqd_group64(float dl, uvec2 grid, uint signs) {
    return NfIqdGroup(dl, grid.x, grid.y, signs);
}
#endif /* NF_IQD_SIGNED */

#ifdef NF_IQD_TERNARY
const float NF_IQD_IQ1_DELTA = 0.125;   /* IQ1S_DELTA == IQ1M_DELTA (llama.cpp ggml-common.h:1132) */

struct NfIqdGroup {
    float dl;     /* group scale */
    float delta;  /* +-NF_IQD_IQ1_DELTA added to every value */
    uint vals;    /* 16-bit grid entry: 2-bit two's complement field j = value j */
};

/* Values 4h .. 4h+3 of a packed entry, as floats in {-1, 0, 1}. */
vec4 nf_iqd_tvals4(uint vals, uint h) {
    const int v = int(vals >> (8u * h));
    return vec4(bitfieldExtract(v, 0, 2), bitfieldExtract(v, 2, 2),
                bitfieldExtract(v, 4, 2), bitfieldExtract(v, 6, 2));
}

/* Dot of a group with its activations x0 (elements 0-3) and x1 (elements 4-7). */
float nf_iqd_dot(NfIqdGroup g, vec4 x0, vec4 x1) {
    return g.dl * (dot(x0, nf_iqd_tvals4(g.vals, 0u) + g.delta)
                 + dot(x1, nf_iqd_tvals4(g.vals, 1u) + g.delta));
}

/* Element e in [0, 8) of a group. */
float nf_iqd_elem(NfIqdGroup g, uint e) {
    return g.dl * (float(bitfieldExtract(int(g.vals), int(2u * e), 2)) + g.delta);
}

/* Delta of a group: negative when the sign flag is set. */
float nf_iqd_iq1_delta(bool negative) {
    return negative ? -NF_IQD_IQ1_DELTA : NF_IQD_IQ1_DELTA;
}
#endif /* NF_IQD_TERNARY */

#ifdef NF_IQD_KNL
struct NfIqdGroup {
    float dl;     /* group scale */
    uint lo;      /* base codes 0-3, code j at bits 3j */
    uint hi;      /* base codes 4-7, code j at bits 3(j-4) */
    uint grid;    /* 0 or 8: the shifted-grid offset of the group */
};

/* Values 0-3 of a packed base-code word, on the iq3nl grid at offset grid (0 or 8). */
vec4 nf_iqd_kvals4(uint codes, uint grid) {
    return vec4(nf_kvalue_iq3nl((codes & 7u) + grid),
                nf_kvalue_iq3nl(((codes >> 3) & 7u) + grid),
                nf_kvalue_iq3nl(((codes >> 6) & 7u) + grid),
                nf_kvalue_iq3nl(((codes >> 9) & 7u) + grid));
}

/* Dot of a group with its activations x0 (elements 0-3) and x1 (elements 4-7). */
float nf_iqd_dot(NfIqdGroup g, vec4 x0, vec4 x1) {
    return g.dl * (dot(x0, nf_iqd_kvals4(g.lo, g.grid)) + dot(x1, nf_iqd_kvals4(g.hi, g.grid)));
}

/* Element e in [0, 8) of a group. */
float nf_iqd_elem(NfIqdGroup g, uint e) {
    const uint codes = e < 4u ? g.lo : g.hi;
    return g.dl * float(nf_kvalue_iq3nl(((codes >> (3u * (e & 3u))) & 7u) + g.grid));
}

/* Base codes of the 8 elements grp*8+k of sub-block sb: two bits per element from qs byte
 * 32*(sb/4) + grp*8 + k at shift 2*(sb%4), bit sb of qh byte grp*8 + k as the third. The caller
 * stores the shifted-grid offset (0 or 8) in the group's grid field. */
uvec2 nf_iqd_kcodes(uint ws, uint qs, uint qh, uint sb, uint grp) {
    const uint shift_l = 2u * (sb & 3u);
    const uint shift_h = sb & 7u;
    uvec2 c = uvec2(0u);
    for (uint k = 0u; k < 4u; ++k) {
        const uint c0 = ((nf_w_u8(ws, qs + k) >> shift_l) & 3u)
                      | (((nf_w_u8(ws, qh + k) >> shift_h) & 1u) << 2);
        const uint c1 = ((nf_w_u8(ws, qs + k + 4u) >> shift_l) & 3u)
                      | (((nf_w_u8(ws, qh + k + 4u) >> shift_h) & 1u) << 2);
        c.x |= c0 << (3u * k);
        c.y |= c1 << (3u * k);
    }
    return c;
}
#endif /* NF_IQD_KNL */

/* ---- per-type decoders: nf_iqd_group(ws, bp, sb, grp) = group grp of sub-block sb ---- */

#if QTYPE == QT_IQ3_XXS
#define QT_BS  NF_IQ3_XXS_BS
#define QT_BB  NF_IQ3_XXS_BB
#define QT_TPS 32
#define QT_DECODE_DOT  1
#define QT_DECODE_DUAL 1
#define QT_DECODE_ELEM 1
/* d(f16) | qs[64] grid indices (8 per sub-block) | 8 x u32: bits 28-31 scale, bits 7l..7l+6 the
 * sign index of group l (llama.cpp ggml-quants.c:2589-2603). */
NfIqdGroup nf_iqd_group(uint ws, uint bp, uint sb, uint grp) {
    const uint aux = nf_w_u32_even(ws, bp + 66u + 4u * sb);
    const uint q = nf_w_u16(ws, bp + 2u + 8u * sb + 2u * grp);
    const float dl = nf_w_f16(ws, bp) * (0.5 + float(aux >> 28)) * 0.5;
    return NfIqdGroup(dl, nf_iq3xxs(q & 0xFFu), nf_iq3xxs(q >> 8),
                      nf_iq2_ksigns((aux >> (7u * grp)) & 127u));
}
#endif

#if QTYPE == QT_IQ3_S
#define QT_BS  NF_IQ3_S_BS
#define QT_BB  NF_IQ3_S_BB
#define QT_TPS 32
#define QT_DECODE_DOT  1
#define QT_DECODE_DUAL 1
#define QT_DECODE_ELEM 1
/* d(f16) | qs[64] | qh[8] (index bit 8) | signs[32] | scales[4]; sub-block sb uses qs[8sb..],
 * qh[sb], signs[4sb..] and nibble sb&1 of scales[sb/2] (llama.cpp ggml-quants.c:2618-2645). */
NfIqdGroup nf_iqd_group(uint ws, uint bp, uint sb, uint grp) {
    const uint qh = nf_w_u8(ws, bp + 66u + sb);
    const uint q = nf_w_u16(ws, bp + 2u + 8u * sb + 2u * grp);
    const uint i0 = (q & 0xFFu) | ((qh << (8u - 2u * grp)) & 256u);
    const uint i1 = (q >> 8) | ((qh << (7u - 2u * grp)) & 256u);
    return NfIqdGroup(nf_iq3s_scale(ws, bp, sb), nf_iq3s(i0), nf_iq3s(i1),
                      nf_w_u8(ws, bp + 74u + 4u * sb + grp));
}
#endif

#if QTYPE == QT_IQ3_K
#define QT_BS  NF_IQ3_K_BS
#define QT_BB  NF_IQ3_K_BB
#define QT_TPS 32
#define QT_DECODE_DOT  1
#define QT_DECODE_DUAL 1
#define QT_DECODE_ELEM 1
/* d(f16) | extra(u16 grid-select bits) | scales_h(u16 sign bits) | scales_l[8] | qs[64] | qh[32].
 * The 16-element half h of sub-block sb (ib = 2*sb + h) has the scale 2*ls + 1 from nibble h of
 * scales_l[sb], the sign bit ib of scales_h and the grid-select bit ib of extra
 * (Avalon ggml-quants.c:5395-5406). */
NfIqdGroup nf_iqd_group(uint ws, uint bp, uint sb, uint grp) {
    const uint extra = nf_w_u16(ws, bp + 2u);
    const uint sh = nf_w_u16(ws, bp + 4u);
    const uint sl = nf_w_u8(ws, bp + 6u + sb);
    const uint ib = 2u * sb + (grp >> 1u);
    const float sgn = ((sh >> ib) & 1u) != 0u ? -1.0 : 1.0;
    const float dl = nf_w_f16(ws, bp) * float(2u * ((sl >> (4u * (grp >> 1u))) & 0xFu) + 1u) * sgn;
    const uvec2 c = nf_iqd_kcodes(ws, bp + 14u + 32u * (sb >> 2u) + 8u * grp,
                                  bp + 78u + 8u * grp, sb, grp);
    return NfIqdGroup(dl, c.x, c.y, ((extra >> ib) & 1u) != 0u ? 8u : 0u);
}
#endif

#if QTYPE == QT_IQ3_KS
#define QT_BS  NF_IQ3_KS_BS
#define QT_BB  NF_IQ3_KS_BB
#define QT_TPS 32
#define QT_DECODE_DOT  1
#define QT_DECODE_DUAL 1
#define QT_DECODE_ELEM 1
/* d(f16) | extra(u16: bits 0-7 the scales' bit 4, bits 8-15 the grid-select bits) | scales[4]
 * | qs[64] | qh[32]. Sub-block sb has the signed 5-bit scale (nibble sb/4 of scales[sb%4] | bit sb
 * of extra << 4) - 16 and the grid-select bit 8 + sb of extra (Avalon ggml-quants.c:5619-5629). */
NfIqdGroup nf_iqd_group(uint ws, uint bp, uint sb, uint grp) {
    const uint extra = nf_w_u16(ws, bp + 2u);
    const uint ls = (((nf_w_u8(ws, bp + 4u + (sb & 3u)) >> (4u * (sb >> 2u))) & 0xFu)
                   | (((extra >> sb) & 1u) << 4)) - 16u;
    const uvec2 c = nf_iqd_kcodes(ws, bp + 8u + 32u * (sb >> 2u) + 8u * grp,
                                  bp + 72u + 8u * grp, sb, grp);
    return NfIqdGroup(nf_w_f16(ws, bp) * float(int(ls)), c.x, c.y,
                      ((extra >> (8u + sb)) & 1u) != 0u ? 8u : 0u);
}
#endif

#if QTYPE == QT_IQ2_XXS
#define QT_BS  NF_IQ2_XXS_BS
#define QT_BB  NF_IQ2_XXS_BB
#define QT_TPS 32
#define QT_DECODE_DOT  1
#define QT_DECODE_DUAL 1
#define QT_DECODE_ELEM 1
/* d(f16) | 8 sub-blocks of 8 bytes: 4 grid indices, then a u32 with bits 28-31 the scale and bits
 * 7l..7l+6 the sign index of group l (llama.cpp ggml-quants.c:2499-2510). */
NfIqdGroup nf_iqd_group(uint ws, uint bp, uint sb, uint grp) {
    const uint q = bp + 2u + 8u * sb;
    const uint aux = nf_w_u32_even(ws, q + 4u);
    const float dl = nf_w_f16(ws, bp) * (0.5 + float(aux >> 28)) * 0.25;
    return nf_iqd_group64(dl, nf_iq2xxs(nf_w_u8(ws, q + grp)),
                          nf_iq2_ksigns((aux >> (7u * grp)) & 127u));
}
#endif

#if QTYPE == QT_IQ2_XS
#define QT_BS  NF_IQ2_XS_BS
#define QT_BB  NF_IQ2_XS_BB
#define QT_TPS 32
#define QT_DECODE_DOT  1
#define QT_DECODE_DUAL 1
#define QT_DECODE_ELEM 1
/* d(f16) | qs[32] u16 (bits 0-8 grid index, bits 9-15 sign index) | scales[8]; groups 0-1 use the
 * low scale nibble, groups 2-3 the high one (llama.cpp ggml-quants.c:2526-2537). */
NfIqdGroup nf_iqd_group(uint ws, uint bp, uint sb, uint grp) {
    const uint qv = nf_w_u16(ws, bp + 2u + 2u * (4u * sb + grp));
    const uint sc = nf_w_u8(ws, bp + 66u + sb);
    const float dl = nf_w_f16(ws, bp) * (0.5 + float(grp < 2u ? (sc & 0xFu) : (sc >> 4))) * 0.25;
    return nf_iqd_group64(dl, nf_iq2xs(qv & 511u), nf_iq2_ksigns(qv >> 9));
}
#endif

#if QTYPE == QT_IQ2_S
#define QT_BS  NF_IQ2_S_BS
#define QT_BB  NF_IQ2_S_BB
#define QT_TPS 32
#define QT_DECODE_DOT  1
#define QT_DECODE_DUAL 1
#define QT_DECODE_ELEM 1
/* d(f16) | qs[32] grid indices | signs[32] | qh[8] (index bits 8-9) | scales[8], nibbles as
 * IQ2_XS (llama.cpp ggml-quants.c:2553-2571). */
NfIqdGroup nf_iqd_group(uint ws, uint bp, uint sb, uint grp) {
    const uint qh = nf_w_u8(ws, bp + 66u + sb);
    const uint idx = nf_w_u8(ws, bp + 2u + 4u * sb + grp) | ((qh << (8u - 2u * grp)) & 0x300u);
    const uint sc = nf_w_u8(ws, bp + 74u + sb);
    const float dl = nf_w_f16(ws, bp) * (0.5 + float(grp < 2u ? (sc & 0xFu) : (sc >> 4))) * 0.25;
    return nf_iqd_group64(dl, nf_iq2s(idx), nf_w_u8(ws, bp + 34u + 4u * sb + grp));
}
#endif

#if QTYPE == QT_IQ1_S
#define QT_BS  NF_IQ1_S_BS
#define QT_BB  NF_IQ1_S_BB
#define QT_TPS 32
#define QT_DECODE_DOT  1
#define QT_DECODE_DUAL 1
#define QT_DECODE_ELEM 1
/* d(f16) | qs[32] index low bytes | qh[8] u16: bits 3l..3l+2 index bits 8-10 of group l, bits
 * 12-14 the sub-block scale, bit 15 the delta sign (llama.cpp ggml-quants.c:2660-2671). */
NfIqdGroup nf_iqd_group(uint ws, uint bp, uint sb, uint grp) {
    const uint qh = nf_w_u16(ws, bp + 34u + 2u * sb);
    const uint idx = nf_w_u8(ws, bp + 2u + 4u * sb + grp) | (((qh >> (3u * grp)) & 7u) << 8);
    const float dl = nf_w_f16(ws, bp) * float(2u * ((qh >> 12) & 7u) + 1u);
    return NfIqdGroup(dl, nf_iqd_iq1_delta((qh & 0x8000u) != 0u), nf_iq1s(idx));
}
#endif

#if QTYPE == QT_IQ1_M
#define QT_BS  NF_IQ1_M_BS
#define QT_BB  NF_IQ1_M_BB
#define QT_TPS 32
#define QT_DECODE_DOT  1
#define QT_DECODE_DUAL 1
#define QT_DECODE_ELEM 1
/* qs[32] | qh[16] | scales[8] (4 x u16). qh[2sb + grp/2] holds index bits 8-10 (low nibble for
 * even grp, high nibble for odd grp) and the delta sign (bit 3 / bit 7); the 3-bit scale of groups
 * 0-1 / 2-3 sits at bit 6(sb&1) / 6(sb&1)+3 of scale word sb/2; d is rebuilt from the top nibbles
 * of the four words (llama.cpp ggml-quants.c:2685-2721). */
NfIqdGroup nf_iqd_group(uint ws, uint bp, uint sb, uint grp) {
    const uint sc = nf_w_u16(ws, bp + 48u + 2u * (sb >> 1));
    const uint qh = nf_w_u8(ws, bp + 32u + 2u * sb + (grp >> 1));
    const bool odd = (grp & 1u) != 0u;
    const uint idx = nf_w_u8(ws, bp + 4u * sb + grp) | ((qh << (odd ? 4u : 8u)) & 0x700u);
    const float dl = nf_iq1m_d(ws, bp) * float(2u * ((sc >> (6u * (sb & 1u) + 3u * (grp >> 1))) & 7u) + 1u);
    return NfIqdGroup(dl, nf_iqd_iq1_delta((qh & (odd ? 0x80u : 0x08u)) != 0u), nf_iq1s(idx));
}
#endif

/* ---- decoder interface (common/quant_blocks.glsl) ----
 * A shader that only uses the group builders and its own qt_dot_q8 (common/quant_iq_q8.glsl) defines
 * NF_IQD_NO_FLOAT_DOT before including this file, since it has no F32 activation readers. */
#if (defined(NF_IQD_SIGNED) || defined(NF_IQD_TERNARY) || defined(NF_IQD_KNL)) && !defined(NF_IQD_NO_FLOAT_DOT)
float qt_dot(uint ws, uint bp, uint xoff, uint blk, uint ltid) {
    const NfIqdGroup g = nf_iqd_group(ws, bp, ltid >> 2, ltid & 3u);
    const uint xi = xoff + blk * uint(QT_BS) + 8u * ltid;
    return nf_iqd_dot(g, qt_x4(xi), qt_x4(xi + 4u));
}

void qt_dot_dual(uint gbp, uint ubp, uint xoff, uint blk, uint ltid, inout float gs, inout float us) {
    const NfIqdGroup g = nf_iqd_group(NF_WS_A, gbp, ltid >> 2, ltid & 3u);
    const NfIqdGroup u = nf_iqd_group(NF_WS_B, ubp, ltid >> 2, ltid & 3u);
    const uint xi = xoff + blk * uint(QT_BS) + 8u * ltid;
    const vec4 x0 = qt_x4(xi);
    const vec4 x1 = qt_x4(xi + 4u);
    gs += nf_iqd_dot(g, x0, x1);
    us += nf_iqd_dot(u, x0, x1);
}

float qt_elem(uint ws, uint rbase, uint i) {
    const uint w = i % uint(QT_BS);
    const NfIqdGroup g = nf_iqd_group(ws, rbase + (i / uint(QT_BS)) * uint(QT_BB), w >> 5, (w >> 3) & 3u);
    return nf_iqd_elem(g, w & 7u);
}
#endif

#endif /* QTYPE */

#endif /* NF_QUANT_IQ_GLSL */

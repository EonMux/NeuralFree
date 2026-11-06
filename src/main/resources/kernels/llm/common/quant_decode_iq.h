/* common/quant_decode_iq.h - IQ3_XXS, IQ3_S, IQ3_K, IQ3_KS, IQ2_XXS, IQ2_XS, IQ2_S, IQ1_S and IQ1_M decode
 * triples. Requires: common/dialect.h, common/quant_blocks.h, common/iq_tables.h */
/*
 * Decode triples T_decode / T_decode_dual / T_decode_elem for the 256-element i-quant super-blocks,
 * with the contract of common/matvec_family.h. The math is llama.cpp ggml/src/ggml-quants.c
 * dequantize_row_iq2_xxs (2489), _iq2_xs (2517), _iq2_s (2544), _iq3_xxs (2576), _iq3_s (2608),
 * _iq1_s (2651) and _iq1_m (2676); IQ3_K and IQ3_KS follow Avalon ggml/src/ggml-quants.c
 * dequantize_row_iq3_k (5381) and _iq3_ks (5613); block layouts are llama.cpp ggml/src/ggml-common.h
 * block_iq* and their Avalon block_iq3_k / block_iq3_ks.
 *
 * Work split (TPS = NF_IQD_TPS = 32): work-item ltid of a block owns the 8-element group
 * ltid = 4 * sb + grp, where sb in [0, 8) is the 32-element sub-block (ggml ib32) and grp in [0, 4)
 * the 8-element group inside it (ggml l). Its elements are block elements 8 * ltid .. 8 * ltid + 7.
 *
 * Every type reduces one group to a small descriptor:
 *   nf_iqd_sgroup (IQ2_*, IQ3_*):  y[j] = dl * mag[j] * (bit j of signs ? -1 : 1)
 *                                  mag[j] = unsigned byte j of the 8 grid bytes (lo: 0-3, hi: 4-7)
 *   nf_iqd_tgroup (IQ1_*):         y[j] = dl * (val[j] + delta)
 *                                  val[j] = signed byte j of the 8 grid bytes, in {-1, 0, 1}
 *   nf_iqd_kgroup (IQ3_K, IQ3_KS): y[j] = dl * IQ3NL_KVALUES[code j + grid]
 *                                  code j is a 3-bit base code, 4 per uint (lo: 0-3, hi: 4-7);
 *                                  grid is 0 or 8, the block's grid-select bit times 8
 * and the triples are generated from the per-type descriptor builders nf_iqd_<type>(bp, sb, grp).
 */
#ifndef NF_QUANT_DECODE_IQ_H
#define NF_QUANT_DECODE_IQ_H

#define NF_IQD_TPS 32   /* work-items per i-quant block: one per 8-element group */

/* ---- group descriptors and their dot / element helpers ---- */

/* Sign-magnitude group of the IQ2 and IQ3 grids. */
typedef struct {
    float dl;     /* group scale */
    uint lo;      /* grid magnitude bytes 0-3, byte j at bits 8j */
    uint hi;      /* grid magnitude bytes 4-7 */
    uint signs;   /* bit j set: element j is negative (ggml kmask_iq2xs[j] = 1 << j) */
} nf_iqd_sgroup;

/* Ternary group of the IQ1 grid. */
typedef struct {
    float dl;     /* group scale */
    float delta;  /* +-IQ1S_DELTA shift added to every value */
    uint lo;      /* signed grid bytes 0-3 (values -1, 0, 1) */
    uint hi;      /* signed grid bytes 4-7 */
} nf_iqd_tgroup;

/* iq3nl group of the IQ3_K / IQ3_KS grid: 8 three-bit base codes, 4 per word, plus the shared
 * grid offset (8 when the block's grid-select bit is set, else 0). */
typedef struct {
    float dl;     /* group scale */
    uint lo;      /* base codes 0-3, code j at bits 3j */
    uint hi;      /* base codes 4-7, code j at bits 3(j-4) */
    uint grid;    /* 0 or 8: the shifted-grid offset of the group */
} nf_iqd_kgroup;

/* Unsigned byte k of w. */
NF_INLINE float nf_iqd_ubyte(uint w, int k) {
    return (float)((w >> (8 * k)) & 0xFFu);
}

/* Signed (two's complement) byte k of w, without implementation-defined narrowing. */
NF_INLINE float nf_iqd_sbyte(uint w, int k) {
    const int b = (int)((w >> (8 * k)) & 0xFFu);
    return (float)(b - ((b & 0x80) << 1));
}

/* -1 when bit j of signs is set, otherwise 1. */
NF_INLINE float nf_iqd_sign(uint signs, int j) {
    return ((signs >> j) & 1u) ? -1.0f : 1.0f;
}

/* sum_k x[k] * mag[k] * sign[k] over 4 elements; signs bit k belongs to element k. */
NF_INLINE float nf_iqd_sdot4(float4 x, uint mag, uint signs) {
    return x.x * nf_iqd_ubyte(mag, 0) * nf_iqd_sign(signs, 0)
         + x.y * nf_iqd_ubyte(mag, 1) * nf_iqd_sign(signs, 1)
         + x.z * nf_iqd_ubyte(mag, 2) * nf_iqd_sign(signs, 2)
         + x.w * nf_iqd_ubyte(mag, 3) * nf_iqd_sign(signs, 3);
}

/* sum_k x[k] * (val[k] + delta) over 4 elements. */
NF_INLINE float nf_iqd_tdot4(float4 x, uint vals, float delta) {
    return x.x * (nf_iqd_sbyte(vals, 0) + delta)
         + x.y * (nf_iqd_sbyte(vals, 1) + delta)
         + x.z * (nf_iqd_sbyte(vals, 2) + delta)
         + x.w * (nf_iqd_sbyte(vals, 3) + delta);
}

/* Dot of a sign-magnitude group with its 8 activations x0 (elements 0-3) and x1 (4-7). */
NF_INLINE float nf_iqd_sgroup_dot(nf_iqd_sgroup g, float4 x0, float4 x1) {
    return g.dl * (nf_iqd_sdot4(x0, g.lo, g.signs) + nf_iqd_sdot4(x1, g.hi, g.signs >> 4));
}

/* Dot of a ternary group with its 8 activations. */
NF_INLINE float nf_iqd_tgroup_dot(nf_iqd_tgroup g, float4 x0, float4 x1) {
    return g.dl * (nf_iqd_tdot4(x0, g.lo, g.delta) + nf_iqd_tdot4(x1, g.hi, g.delta));
}

/* Code k in [0, 4) of a packed iq3nl group word (a base code in [0, 7)). */
NF_INLINE int nf_iqd_kcode(uint codes, int k) {
    return (int)((codes >> (3 * k)) & 7u);
}

/* sum_k x[k] * IQ3NL_KVALUES[codes[k] + grid] over 4 elements. */
NF_INLINE float nf_iqd_kdot4(float4 x, uint codes, uint grid) {
    return x.x * IQ3NL_KVALUES[nf_iqd_kcode(codes, 0) + grid]
         + x.y * IQ3NL_KVALUES[nf_iqd_kcode(codes, 1) + grid]
         + x.z * IQ3NL_KVALUES[nf_iqd_kcode(codes, 2) + grid]
         + x.w * IQ3NL_KVALUES[nf_iqd_kcode(codes, 3) + grid];
}

/* Dot of an iq3nl group with its 8 activations x0 (elements 0-3) and x1 (4-7). */
NF_INLINE float nf_iqd_kgroup_dot(nf_iqd_kgroup g, float4 x0, float4 x1) {
    return g.dl * (nf_iqd_kdot4(x0, g.lo, g.grid) + nf_iqd_kdot4(x1, g.hi, g.grid));
}

/* Element e in [0, 8) of an iq3nl group. */
NF_INLINE float nf_iqd_kgroup_elem(nf_iqd_kgroup g, int e) {
    const uint codes = e < 4 ? g.lo : g.hi;
    return g.dl * IQ3NL_KVALUES[nf_iqd_kcode(codes, e & 3) + g.grid];
}

/* Element e in [0, 8) of a sign-magnitude group. */
NF_INLINE float nf_iqd_sgroup_elem(nf_iqd_sgroup g, int e) {
    const uint mag = e < 4 ? g.lo : g.hi;
    return g.dl * nf_iqd_ubyte(mag, e & 3) * nf_iqd_sign(g.signs, e);
}

/* Element e in [0, 8) of a ternary group. */
NF_INLINE float nf_iqd_tgroup_elem(nf_iqd_tgroup g, int e) {
    const uint vals = e < 4 ? g.lo : g.hi;
    return g.dl * (nf_iqd_sbyte(vals, e & 3) + g.delta);
}

/* Super-block f16 scale d at byte 0 (every type here except IQ1_M). */
NF_INLINE float nf_iqd_d(NF_GLOBAL const uchar* bp) {
    return vload_half(0, (NF_GLOBAL const NF_HALF*)bp);
}

/* Sign-magnitude group from one 8-byte grid entry. */
NF_INLINE nf_iqd_sgroup nf_iqd_sgroup64(float dl, nf_u64 grid, uint signs) {
    nf_iqd_sgroup g;
    g.dl = dl;
    g.lo = (uint)grid;
    g.hi = (uint)(grid >> 32);
    g.signs = signs;
    return g;
}

/* Sign-magnitude group from two 4-byte grid entries. */
NF_INLINE nf_iqd_sgroup nf_iqd_sgroup32(float dl, uint grid0, uint grid1, uint signs) {
    nf_iqd_sgroup g;
    g.dl = dl;
    g.lo = grid0;
    g.hi = grid1;
    g.signs = signs;
    return g;
}

/* Ternary group from one 8-byte IQ1 grid entry. */
NF_INLINE nf_iqd_tgroup nf_iqd_tgroup64(float dl, float delta, nf_u64 grid) {
    nf_iqd_tgroup g;
    g.dl = dl;
    g.delta = delta;
    g.lo = (uint)grid;
    g.hi = (uint)(grid >> 32);
    return g;
}

/* ---- per-type group builders: group grp of sub-block sb of the block at bp ---- */

/* IQ3_XXS, 98 bytes: d(f16) | qs[64] grid indices (8 per sub-block) | 8 x u32 scale+signs words.
 * Word sb: bits 28-31 scale, bits 7l..7l+6 the sign index of group l (llama.cpp ggml-quants.c:2589-2603). */
NF_INLINE nf_iqd_sgroup nf_iqd_iq3xxs(NF_GLOBAL const uchar* bp, int sb, int grp) {
    const uint aux = read_u32_le(bp + 66 + 4 * sb);
    NF_GLOBAL const uchar* qs = bp + 2 + 8 * sb + 2 * grp;
    const float dl = nf_iqd_d(bp) * (0.5f + (float)(aux >> 28)) * 0.5f;
    return nf_iqd_sgroup32(dl, IQ3XXS_GRID_CL[qs[0]], IQ3XXS_GRID_CL[qs[1]],
                           KSIGNS_IQ2XS_CL[(aux >> (7 * grp)) & 127u]);
}

/* IQ3_S, 110 bytes: d(f16) | qs[64] | qh[8] (bit 8 of the grid indices) | signs[32] | scales[4].
 * Sub-block sb uses qs[8sb..], qh[sb], signs[4sb..] and nibble sb&1 of scales[sb/2]
 * (llama.cpp ggml-quants.c:2618-2645). */
NF_INLINE nf_iqd_sgroup nf_iqd_iq3s(NF_GLOBAL const uchar* bp, int sb, int grp) {
    const uint qh = bp[66 + sb];
    NF_GLOBAL const uchar* qs = bp + 2 + 8 * sb + 2 * grp;
    const uint i0 = (uint)qs[0] | ((qh << (8 - 2 * grp)) & 256u);
    const uint i1 = (uint)qs[1] | ((qh << (7 - 2 * grp)) & 256u);
    return nf_iqd_sgroup32(iq3s_scale(bp, sb), IQ3S_GRID_CL[i0], IQ3S_GRID_CL[i1],
                           bp[74 + 4 * sb + grp]);
}

/* Shared builder of the IQ3_K / IQ3_KS base codes: the 8 elements grp*8+k of sub-block sb take their
 * two low code bits from qs byte 32*(sb/4) + grp*8 + k at shift 2*(sb%4) and their high bit from bit
 * sb of qh byte grp*8 + k; the caller adds the shifted-grid offset at the value lookup. */
NF_INLINE nf_iqd_kgroup nf_iqd_iq3k_codes(NF_GLOBAL const uchar* qs, NF_GLOBAL const uchar* qh,
                                          int sb, int grp) {
    const int shift_l = 2 * (sb & 3);
    const int shift_h = sb & 7;
    uint lo = 0u, hi = 0u;
    for (int k = 0; k < 4; ++k) {
        const uint c0 = ((qs[k] >> shift_l) & 3u) | (((qh[k] >> shift_h) & 1u) << 2);
        const uint c1 = ((qs[k + 4] >> shift_l) & 3u) | (((qh[k + 4] >> shift_h) & 1u) << 2);
        lo |= c0 << (3 * k);
        hi |= c1 << (3 * k);
    }
    nf_iqd_kgroup g;
    g.lo = lo;
    g.hi = hi;
    return g;
}

/* IQ3_K, 110 bytes: d(f16) | extra(u16 grid-select bits) | scales_h(u16 sign bits) | scales_l[8]
 * | qs[64] | qh[32]. The 16-element half h of sub-block sb (ib = 2*sb + h) has scale (2*ls + 1)
 * from nibble h of scales_l[sb], sign bit ib of scales_h and grid-select bit ib of extra
 * (Avalon ggml-quants.c:5395-5406). */
NF_INLINE nf_iqd_kgroup nf_iqd_iq3k(NF_GLOBAL const uchar* bp, int sb, int grp) {
    const uint extra = (uint)bp[2] | ((uint)bp[3] << 8);
    const uint sh = (uint)bp[4] | ((uint)bp[5] << 8);
    const uint sl = bp[6 + sb];
    const int ib = 2 * sb + (grp >> 1);
    const float dl = nf_iqd_d(bp) * (float)(2u * ((sl >> (4 * (grp >> 1))) & 0xFu) + 1u)
                   * ((sh >> ib) & 1u ? -1.0f : 1.0f);
    nf_iqd_kgroup g = nf_iqd_iq3k_codes(bp + 14 + 32 * (sb >> 2) + 8 * grp, bp + 78 + 8 * grp,
                                        sb, grp);
    g.dl = dl;
    g.grid = ((extra >> ib) & 1u) ? 8u : 0u;
    return g;
}

/* IQ3_KS, 104 bytes: d(f16) | extra(u16: bits 0-7 the scales' bit 4, bits 8-15 the grid bits)
 * | scales[4] | qs[64] | qh[32]. Sub-block sb has the signed 5-bit scale
 * (nibble sb/4 of scales[sb%4] | bit sb of extra << 4) - 16 and the grid-select bit 8+sb of extra
 * (Avalon ggml-quants.c:5619-5629). */
NF_INLINE nf_iqd_kgroup nf_iqd_iq3ks(NF_GLOBAL const uchar* bp, int sb, int grp) {
    const uint extra = (uint)bp[2] | ((uint)bp[3] << 8);
    const uint ls = (((bp[4 + (sb & 3)] >> (4 * (sb >> 2))) & 0xFu) | ((extra >> sb) & 1u) << 4) - 16u;
    nf_iqd_kgroup g = nf_iqd_iq3k_codes(bp + 8 + 32 * (sb >> 2) + 8 * grp, bp + 72 + 8 * grp,
                                        sb, grp);
    g.dl = nf_iqd_d(bp) * (float)(int)ls;
    g.grid = ((extra >> (8 + sb)) & 1u) ? 8u : 0u;
    return g;
}

/* IQ2_XXS, 66 bytes: d(f16) | 8 sub-blocks of 8 bytes: 4 grid indices, then a u32 whose bits
 * 28-31 are the scale and bits 7l..7l+6 the sign index of group l (llama.cpp ggml-quants.c:2499-2510). */
NF_INLINE nf_iqd_sgroup nf_iqd_iq2xxs(NF_GLOBAL const uchar* bp, int sb, int grp) {
    NF_GLOBAL const uchar* q = bp + 2 + 8 * sb;
    const uint aux = read_u32_le(q + 4);
    const float dl = nf_iqd_d(bp) * (0.5f + (float)(aux >> 28)) * 0.25f;
    return nf_iqd_sgroup64(dl, IQ2XXS_GRID_CL[q[grp]], KSIGNS_IQ2XS_CL[(aux >> (7 * grp)) & 127u]);
}

/* IQ2_XS, 74 bytes: d(f16) | qs[32] u16 (bits 0-8 grid index, bits 9-15 sign index) | scales[8].
 * Groups 0-1 of a sub-block use the low scale nibble, groups 2-3 the high one
 * (llama.cpp ggml-quants.c:2526-2537). */
NF_INLINE nf_iqd_sgroup nf_iqd_iq2xs(NF_GLOBAL const uchar* bp, int sb, int grp) {
    NF_GLOBAL const uchar* q = bp + 2 + 2 * (4 * sb + grp);
    const uint qv = (uint)q[0] | ((uint)q[1] << 8);
    const uint sc = bp[66 + sb];
    const float dl = nf_iqd_d(bp) * (0.5f + (float)(grp < 2 ? (sc & 0xFu) : (sc >> 4))) * 0.25f;
    return nf_iqd_sgroup64(dl, IQ2XS_GRID_CL[qv & 511u], KSIGNS_IQ2XS_CL[qv >> 9]);
}

/* IQ2_S, 82 bytes: d(f16) | qs[32] grid indices | signs[32] | qh[8] (bits 8-9 of the indices) |
 * scales[8], nibbles as IQ2_XS (llama.cpp ggml-quants.c:2553-2571). */
NF_INLINE nf_iqd_sgroup nf_iqd_iq2s(NF_GLOBAL const uchar* bp, int sb, int grp) {
    const uint qh = bp[66 + sb];
    const uint idx = (uint)bp[2 + 4 * sb + grp] | ((qh << (8 - 2 * grp)) & 0x300u);
    const uint sc = bp[74 + sb];
    const float dl = nf_iqd_d(bp) * (0.5f + (float)(grp < 2 ? (sc & 0xFu) : (sc >> 4))) * 0.25f;
    return nf_iqd_sgroup64(dl, IQ2S_GRID_CL[idx], bp[34 + 4 * sb + grp]);
}

/* IQ1_S, 50 bytes: d(f16) | qs[32] grid index low bytes | qh[8] u16: bits 3l..3l+2 index bits
 * 8-10 of group l, bits 12-14 the sub-block scale, bit 15 the delta sign
 * (llama.cpp ggml-quants.c:2660-2671). */
NF_INLINE nf_iqd_tgroup nf_iqd_iq1s(NF_GLOBAL const uchar* bp, int sb, int grp) {
    const uint qh = (uint)bp[34 + 2 * sb] | ((uint)bp[35 + 2 * sb] << 8);
    const uint idx = (uint)bp[2 + 4 * sb + grp] | (((qh >> (3 * grp)) & 7u) << 8);
    const float dl = nf_iqd_d(bp) * (float)(2u * ((qh >> 12) & 7u) + 1u);
    const float delta = (qh & 0x8000u) ? -IQ1_S_DELTA : IQ1_S_DELTA;
    return nf_iqd_tgroup64(dl, delta, IQ1S_GRID_CL[idx]);
}

/* IQ1_M, 56 bytes: qs[32] | qh[16] | scales[8] (4 x u16). qh[2sb + grp/2] holds index bits 8-10
 * (low nibble for even grp, high nibble for odd grp) and the delta sign (bit 3 / bit 7). The
 * 3-bit scale of groups 0-1 / 2-3 sits at bit 6(sb&1) / 6(sb&1)+3 of u16 scale word sb/2; d is
 * rebuilt from the top nibbles of the four words (llama.cpp ggml-quants.c:2685-2721). */
NF_INLINE nf_iqd_tgroup nf_iqd_iq1m(NF_GLOBAL const uchar* bp, int sb, int grp) {
    NF_GLOBAL const uchar* scw = bp + 48 + 2 * (sb >> 1);
    const uint sc = (uint)scw[0] | ((uint)scw[1] << 8);
    const uint qh = bp[32 + 2 * sb + (grp >> 1)];
    const uint odd = (uint)(grp & 1);
    const uint idx = (uint)bp[4 * sb + grp] | ((qh << (odd ? 4u : 8u)) & 0x700u);
    const float dl = iq1m_reconstruct_d(bp)
                   * (float)(2u * ((sc >> (6 * (sb & 1) + 3 * (grp >> 1))) & 7u) + 1u);
    const float delta = (qh & (odd ? 0x80u : 0x08u)) ? -IQ1_M_DELTA : IQ1_M_DELTA;
    return nf_iqd_tgroup64(dl, delta, IQ1S_GRID_CL[idx]);
}

/* ---- triple generators ---- */

/*
 * NF_IQD_TRIPLE(N, BUILD, KIND, BB): N##_decode, N##_decode_dual and N##_decode_elem from the group
 * builder BUILD, which returns nf_iqd_<KIND>group; BB is the block size in bytes. Work-item ltid
 * reads its 8 activations input[off + blk*256 + 8*ltid ...] once (shared by gate and up in _dual).
 */
#define NF_IQD_TRIPLE(N, BUILD, KIND, BB) \
NF_INLINE float N##_decode(NF_GLOBAL const uchar* bp, NF_GLOBAL const float* input, \
                           int off, int blk, int ltid) { \
    const nf_iqd_##KIND##group g = BUILD(bp, ltid >> 2, ltid & 3); \
    NF_GLOBAL const float* x = input + off + blk * QK_K + 8 * ltid; \
    return nf_iqd_##KIND##group_dot(g, vload4(0, x), vload4(1, x)); \
} \
NF_INLINE void N##_decode_dual(NF_GLOBAL const uchar* gbp, NF_GLOBAL const uchar* ubp, \
                               NF_GLOBAL const float* input, int off, int blk, int ltid, \
                               float* gs, float* us) { \
    const nf_iqd_##KIND##group g = BUILD(gbp, ltid >> 2, ltid & 3); \
    const nf_iqd_##KIND##group u = BUILD(ubp, ltid >> 2, ltid & 3); \
    NF_GLOBAL const float* x = input + off + blk * QK_K + 8 * ltid; \
    const float4 x0 = vload4(0, x), x1 = vload4(1, x); \
    *gs += nf_iqd_##KIND##group_dot(g, x0, x1); \
    *us += nf_iqd_##KIND##group_dot(u, x0, x1); \
} \
NF_INLINE float N##_decode_elem(NF_GLOBAL const uchar* weight, int dim, int tok, int i) { \
    NF_GLOBAL const uchar* bp = weight + (nf_i64)tok * (dim / QK_K) * (BB) + (nf_i64)(i / QK_K) * (BB); \
    const int w = i % QK_K; \
    const nf_iqd_##KIND##group g = BUILD(bp, w >> 5, (w >> 3) & 3); \
    return nf_iqd_##KIND##group_elem(g, w & 7); \
}

NF_IQD_TRIPLE(IQ3_XXS, nf_iqd_iq3xxs, s, IQ3_XXS_BLOCK_BYTES)
NF_IQD_TRIPLE(IQ3_S,   nf_iqd_iq3s,   s, IQ3_S_BLOCK_BYTES)
NF_IQD_TRIPLE(IQ3_K,   nf_iqd_iq3k,   k, IQ3_K_BLOCK_BYTES)
NF_IQD_TRIPLE(IQ3_KS,  nf_iqd_iq3ks,  k, IQ3_KS_BLOCK_BYTES)
NF_IQD_TRIPLE(IQ2_XXS, nf_iqd_iq2xxs, s, IQ2_XXS_BLOCK_BYTES)
NF_IQD_TRIPLE(IQ2_XS,  nf_iqd_iq2xs,  s, IQ2_XS_BLOCK_BYTES)
NF_IQD_TRIPLE(IQ2_S,   nf_iqd_iq2s,   s, IQ2_S_BLOCK_BYTES)
NF_IQD_TRIPLE(IQ1_S,   nf_iqd_iq1s,   t, IQ1_S_BLOCK_BYTES)
NF_IQD_TRIPLE(IQ1_M,   nf_iqd_iq1m,   t, IQ1_M_BLOCK_BYTES)

#endif /* NF_QUANT_DECODE_IQ_H */

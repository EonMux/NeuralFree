/* matmul_mmq_types.h - the weight-code, (scale, offset) and offset-trait macros of the llama.cpp MMQ
 * weight types, shared by the MMQ GEMM instantiations of every backend (kernels/llm/matmul_mmq.cl)
 * and the CUDA stream-k instantiations (kernels/llm/matmul_mmq_stream.cl).
 * Requires: common/dialect.h, common/config.h, common/quant_blocks.h, common/iq_tables.h, common/quant_decode_legacy.h, common/quant_decode_k.h, common/matmul_mmq_family.h */
/*
 * For each MMQ type T: T_MMQ_CODE(bp, e) is the integer code of element e of block bp, T_MMQ_CODE4(bp,
 * e0) the four codes e0..e0+3 packed as int8 lanes, T_MMQ_SCALE(bp, k0) / T_MMQ_OFF(bp, k0) the scale
 * and offset of the group holding element k0 (value = scale * code + offset), T_MMQ_HAS_OFF whether
 * the offset can be non-zero, and T_MMQ_SCALE_NARROW whether every finite scale times 3 is exact in fp32 (a
 * half-precision value times integers of at most 11 bits in all, as the scale formulas below build them; not
 * the fp32 scale of I8_B128): the tensor tile folds the product conversion into the scaling on that trait
 * (nf_mmq_tc_prod in common/matmul_mmq_family.h), and T_MMQ_CODE_SETS how many chunks' code loads that tile
 * holds in registers at once on sm_120 (NF_MMQ_TC_STEP, NF_MMQ_TC_SETS): 2, except for the types whose two-set
 * form spills there.
 * The field offsets and value formulas follow the decode triples in
 * common/quant_decode_legacy.h, common/quant_decode_k.h and common/quant_decode_iq.h. The macros only
 * expand inside the generators of common/matmul_mmq_family.h.
 */

#define NF_MMQ_HBIT(qh, e) (((qh) >> (e)) & 1u)

/* Per-byte v - s of four packed bytes with every v and s below 128 and the results as int8 lanes: each byte is
 * offset by 128 before the subtraction so that no lane borrows from the next, and the offset flips back after. */
#define NF_MMQ_BYTES_SUB(v, s) ((((uint)(v) | 0x80808080u) - (uint)(s)) ^ 0x80808080u)

/* The four sign bits s (bit k for byte k) as 0xFF in each byte whose bit is set: bit k of s * 0x00204081 lands
 * at bit 8 k with no carries, as the four shifted copies of a 4-bit value occupy disjoint bits. */
#define NF_MMQ_SIGN_MASK4(s) ((((uint)(s) * 0x00204081u) & 0x01010101u) * 0xFFu)
/* A grid word's four magnitude bytes (each below 128) as int8 lanes, byte k negated when sign bit k is set:
 * the lanes NF_MMQ_CODE4_PACK builds from the four signed-magnitude codes. */
#define NF_MMQ_SIGNED4(word, s) \
    (((uint)(word) & ~NF_MMQ_SIGN_MASK4(s)) | NF_MMQ_BYTES_SUB(0u, (uint)(word) & NF_MMQ_SIGN_MASK4(s)))

/* Unsigned byte k of a grid word (the iq grid entries hold one value per byte). */
#define NF_MMQ_BYTE(w, k) (((w) >> (8u * (k))) & 0xFFu)
/* Signed magnitude code: the grid byte, negated when bit j of the sign byte is set. */
#define NF_MMQ_SM_CODE(word, j, sign_byte) \
    ((int)NF_MMQ_BYTE(word, j) * (1 - 2 * (int)((((uint)(sign_byte)) >> (j)) & 1u)))

/* ---- legacy 32-element blocks (common/quant_decode_legacy.h) ---- */

/* Q4_0: 18 B, d(f16)@0, qs[16]@2; value = (nibble - 8)*d. */
#define Q4_0_MMQ_CODE(bp, e) (int)((e) < 16 \
    ? ((uint)(bp)[2 + ((e) & 15)] & 0xFu) \
    : ((uint)(bp)[2 + ((e) & 15)] >> 4))
/* Codes e0..e0+3: the group's 4 qs bytes (element e = nibble (e&15) of byte e, one half per
 * 16-code run) as two aligned u16s (the 18-byte block is only 2-byte aligned). */
#define Q4_0_MMQ_CODE4(bp, e0) \
    NF_MMQ_CODE4_NIBBLES(nf_read_u32_u16((bp) + 2 + ((e0) & 15)), ((e0) & 16) ? 4 : 0)
#define Q4_0_MMQ_SCALE(bp, k0) nf_ql_half(bp, 0)
#define Q4_0_MMQ_OFF(bp, k0) (-8.0f * nf_ql_half(bp, 0))
#define Q4_0_MMQ_HAS_OFF 1
#define Q4_0_MMQ_SCALE_NARROW 1
#define Q4_0_MMQ_CODE_SETS 2

/* Q4_1: 20 B, d@0, m(f16)@2, qs[16]@4; value = nibble*d + m. */
#define Q4_1_MMQ_CODE(bp, e) (int)((e) < 16 \
    ? ((uint)(bp)[4 + ((e) & 15)] & 0xFu) \
    : ((uint)(bp)[4 + ((e) & 15)] >> 4))
#define Q4_1_MMQ_CODE4(bp, e0) \
    NF_MMQ_CODE4_NIBBLES(nf_read_u32((bp) + 4 + ((e0) & 15)), ((e0) & 16) ? 4 : 0)
#define Q4_1_MMQ_SCALE(bp, k0) nf_ql_half(bp, 0)
#define Q4_1_MMQ_OFF(bp, k0) nf_ql_half(bp, 2)
#define Q4_1_MMQ_HAS_OFF 1
#define Q4_1_MMQ_SCALE_NARROW 1
#define Q4_1_MMQ_CODE_SETS 2

/* Q5_0: 22 B, d@0, qh(u32 LE)@2, qs[16]@6; value = (nibble | hbit)*d - 16*d. */
#define Q5_0_MMQ_CODE(bp, e) (int)(((e) < 16 \
    ? ((uint)(bp)[6 + ((e) & 15)] & 0xFu) \
    : ((uint)(bp)[6 + ((e) & 15)] >> 4)) | (NF_MMQ_HBIT(read_u32_le((bp) + 2), (e)) << 4))
/* The nibbles of e0..e0+3 from the group's 4 qs bytes (two aligned u16s, the 22-byte block is
 * 2-byte aligned), the 5th bits from the block's qh word. */
#define Q5_0_MMQ_CODE4(bp, e0) \
    (NF_MMQ_CODE4_NIBBLES(nf_read_u32_u16((bp) + 6 + ((e0) & 15)), ((e0) & 16) ? 4 : 0) \
        | NF_MMQ_HBITS4(nf_read_u32_u16((bp) + 2), (e0)))
#define Q5_0_MMQ_SCALE(bp, k0) nf_ql_half(bp, 0)
#define Q5_0_MMQ_OFF(bp, k0) (-16.0f * nf_ql_half(bp, 0))
#define Q5_0_MMQ_HAS_OFF 1
#define Q5_0_MMQ_SCALE_NARROW 1
#define Q5_0_MMQ_CODE_SETS 1

/* Q5_1: 24 B, d@0, m@2, qh@4, qs[16]@8; value = (nibble | hbit)*d + m. */
#define Q5_1_MMQ_CODE(bp, e) (int)(((e) < 16 \
    ? ((uint)(bp)[8 + ((e) & 15)] & 0xFu) \
    : ((uint)(bp)[8 + ((e) & 15)] >> 4)) | (NF_MMQ_HBIT(read_u32_le((bp) + 4), (e)) << 4))
#define Q5_1_MMQ_CODE4(bp, e0) \
    (NF_MMQ_CODE4_NIBBLES(nf_read_u32((bp) + 8 + ((e0) & 15)), ((e0) & 16) ? 4 : 0) \
        | NF_MMQ_HBITS4(nf_read_u32((bp) + 4), (e0)))
#define Q5_1_MMQ_SCALE(bp, k0) nf_ql_half(bp, 0)
#define Q5_1_MMQ_OFF(bp, k0) nf_ql_half(bp, 2)
#define Q5_1_MMQ_HAS_OFF 1
#define Q5_1_MMQ_SCALE_NARROW 1
#define Q5_1_MMQ_CODE_SETS 1

/* Q8_0: 34 B, d@0, qs[32](int8)@2; value = qs*d. */
#define Q8_0_MMQ_CODE(bp, e) (int)((NF_GLOBAL const nf_i8*)(bp))[2 + (e)]
/* Codes e0..e0+3: the raw qs bytes as two aligned u16s (the 34-byte block is 2-byte aligned). */
#define Q8_0_MMQ_CODE4(bp, e0) nf_read_u32_u16((bp) + 2 + (e0))
#define Q8_0_MMQ_SCALE(bp, k0) q8_0_scale(bp)
#define Q8_0_MMQ_OFF(bp, k0) 0.0f
#define Q8_0_MMQ_HAS_OFF 0
#define Q8_0_MMQ_SCALE_NARROW 1
#define Q8_0_MMQ_CODE_SETS 2

/* I8_B128: d(f32)@0, qs[128](int8)@4; value = d*qs. The scale covers the whole 128-element block,
 * so every 32-element stage group reads the same d (SG 32). */
#define I8_B128_MMQ_CODE(bp, e) (int)((NF_GLOBAL const nf_i8*)(bp))[NF_I8_B128_QS + (e)]
/* Codes e0..e0+3: the raw qs bytes as one aligned u32 (the 132-byte block is 4-byte aligned). */
#define I8_B128_MMQ_CODE4(bp, e0) nf_read_u32((bp) + NF_I8_B128_QS + (e0))
#define I8_B128_MMQ_SCALE(bp, k0) as_float(read_u32_le((bp) + NF_I8_B128_D))
#define I8_B128_MMQ_OFF(bp, k0) 0.0f
#define I8_B128_MMQ_HAS_OFF 0
#define I8_B128_MMQ_SCALE_NARROW 0
#define I8_B128_MMQ_CODE_SETS 2

/* ---- 256-element K-quants (common/quant_decode_k.h) ---- */

/* Q2_K: scales[16]@0 (low nibble scale, high nibble min), qs[64]@16, d@80, dmin@82.
 * Sub-block j16 = e >> 4 holds the scale byte; the code is plane j = (e >> 5) & 3.
 * value = d*(sc & 15)*q - dmin*(sc >> 4). */
#define Q2_K_MMQ_CODE(bp, e) (int)(((uint)(bp)[NF_Q2K_QS + ((e) >> 7) * 32 + (((e) >> 4) & 1) * 16 + ((e) & 15)] \
    >> (2u * (((e) >> 5) & 3u))) & 3u)
/* Codes e0..e0+3: plane (e0 >> 5) & 3 of the group's 4 consecutive qs bytes, one aligned u32 (the 84-byte
 * block is 4-byte aligned). */
#define Q2_K_MMQ_CODE4(bp, e0) \
    ((nf_read_u32((bp) + NF_Q2K_QS + ((e0) >> 7) * 32 + (((e0) >> 4) & 1) * 16 + ((e0) & 15)) \
        >> (2u * (((e0) >> 5) & 3u))) & 0x03030303u)
#define Q2_K_MMQ_SCALE(bp, k0) (nf_kq_half(bp, NF_Q2K_D) \
    * (float)((bp)[NF_Q2K_SCALES + ((k0) % 256 >> 4)] & 0xFu))
#define Q2_K_MMQ_OFF(bp, k0) (-nf_kq_half(bp, NF_Q2K_DMIN) \
    * (float)((bp)[NF_Q2K_SCALES + ((k0) % 256 >> 4)] >> 4))
#define Q2_K_MMQ_HAS_OFF 1
#define Q2_K_MMQ_SCALE_NARROW 1
#define Q2_K_MMQ_CODE_SETS 2

/* Q3_K: hmask[32]@0, qs[64]@32, scales[12]@96, d@108; the code is the 2-bit plane minus 4 when the
 * hmask bit is clear, scale si = e >> 4 is the signed 6-bit aux repacking (nf_q3k_scale).
 * value = d * si * code. */
#define Q3_K_MMQ_CODE(bp, e) ((int)(((uint)(bp)[NF_Q3K_QS + ((e) >> 7) * 32 + (((e) >> 4) & 1) * 16 + ((e) & 15)] \
    >> (2u * (((e) >> 5) & 3u))) & 3u) - ((((uint)(bp)[NF_Q3K_HMASK + (((e) >> 4) & 1) * 16 + ((e) & 15)] \
    >> (4u * ((e) >> 7) + (((e) >> 5) & 3u))) & 1u) ? 0 : 4))
/* Codes e0..e0+3: the 2-bit plane of the group's 4 consecutive qs bytes and the hmask bits of its 4
 * consecutive hmask bytes (u16 pairs: the 110-byte block is 2-byte aligned) as v = plane | hbit << 2 per byte,
 * then v - 4 per byte (NF_MMQ_BYTES_SUB). */
#define Q3_K_MMQ_CODE4(bp, e0) \
    NF_MMQ_BYTES_SUB(((nf_read_u32_u16((bp) + NF_Q3K_QS + ((e0) >> 7) * 32 + (((e0) >> 4) & 1) * 16 + ((e0) & 15)) \
        >> (2u * (((e0) >> 5) & 3u))) & 0x03030303u) \
        | (((nf_read_u32_u16((bp) + NF_Q3K_HMASK + (((e0) >> 4) & 1) * 16 + ((e0) & 15)) \
        >> (4u * ((e0) >> 7) + (((e0) >> 5) & 3u))) & 0x01010101u) << 2), 0x04040404u)
#define Q3_K_MMQ_SCALE(bp, k0) (nf_kq_half(bp, NF_Q3K_D) * (float)nf_q3k_scale(bp, ((k0) % 256) >> 4))
#define Q3_K_MMQ_OFF(bp, k0) 0.0f
#define Q3_K_MMQ_HAS_OFF 0
#define Q3_K_MMQ_SCALE_NARROW 1
#define Q3_K_MMQ_CODE_SETS 2

/* Q4_K: d@0, dmin@2, scales[12]@4 (get_scale_min_k4), qs[128]@16. Group si = e >> 5 (32 elements)
 * holds the scale/min pair and the nibble shift; value = d*sc*q - dmin*m. */
#define Q4_K_MMQ_CODE(bp, e) (int)(((uint)(bp)[NF_Q4K_QS + (((e) >> 6) * 32) + ((e) & 31)] \
    >> ((((e) >> 5) & 1) * 4u)) & 0xFu)
/* Codes e0..e0+3: one aligned u32 of qs (the group of 4 stays inside one nibble half of the
 * 32-element scale group). */
#define Q4_K_MMQ_CODE4(bp, e0) \
    NF_MMQ_CODE4_NIBBLES(nf_read_u32((bp) + NF_Q4K_QS + (((e0) >> 6) * 32) + ((e0) & 31)), \
        (((e0) >> 5) & 1) * 4)
#define Q4_K_MMQ_SCALE(bp, k0) (nf_kq_half(bp, NF_Q4K_D) \
    * k4_scale_min(bp + NF_Q4K_SCALES, ((k0) % 256) >> 5).x)
#define Q4_K_MMQ_OFF(bp, k0) (-nf_kq_half(bp, NF_Q4K_DMIN) \
    * k4_scale_min(bp + NF_Q4K_SCALES, ((k0) % 256) >> 5).y)
#define Q4_K_MMQ_HAS_OFF 1
#define Q4_K_MMQ_SCALE_NARROW 1
#define Q4_K_MMQ_CODE_SETS 2

/* Q5_K: as Q4_K plus a fifth bit of element e from bit si of qh[e % 32] (qh[32]@16, qs@48). */
#define Q5_K_MMQ_CODE(bp, e) (int)((((uint)(bp)[NF_Q5K_QS + (((e) >> 6) * 32) + ((e) & 31)] \
    >> ((((e) >> 5) & 1) * 4u)) & 0xFu) | (NF_MMQ_HBIT((uint)(bp)[NF_Q5K_QH + ((e) & 31)], (e) >> 5) << 4))
/* Codes e0..e0+3: one aligned u32 of qs plus the group's qh u32 (4 consecutive qh bytes; the 5th
 * bit of every code in the group is bit (e0 >> 5) of its byte, constant within the group). */
#define Q5_K_MMQ_CODE4(bp, e0) \
    (NF_MMQ_CODE4_NIBBLES(nf_read_u32((bp) + NF_Q5K_QS + (((e0) >> 6) * 32) + ((e0) & 31)), \
        (((e0) >> 5) & 1) * 4) \
        | NF_MMQ_HBITS4_B(nf_read_u32((bp) + NF_Q5K_QH + ((e0) & 31)), ((e0) >> 5)))
#define Q5_K_MMQ_SCALE(bp, k0) (nf_kq_half(bp, NF_Q5K_D) \
    * k4_scale_min(bp + NF_Q5K_SCALES, ((k0) % 256) >> 5).x)
#define Q5_K_MMQ_OFF(bp, k0) (-nf_kq_half(bp, NF_Q5K_DMIN) \
    * k4_scale_min(bp + NF_Q5K_SCALES, ((k0) % 256) >> 5).y)
#define Q5_K_MMQ_HAS_OFF 1
#define Q5_K_MMQ_SCALE_NARROW 1
#define Q5_K_MMQ_CODE_SETS 1

/* Q6_K: ql[128]@0, qh[64]@128, scales[16](i8)@192, d@208. The 6-bit code is (lo | hi << 4) - 32
 * and value = d * scales[hf*8 + t*2 + (l >> 4)] * code. */
#define Q6_K_MMQ_CODE(bp, e) ((int)((((uint)(bp)[NF_Q6K_QL + ((e) >> 7) * 64 + (((e) >> 5) & 1) * 32 + ((e) & 31)] \
    >> ((((e) >> 6) & 1) * 4u)) & 0xFu) \
    | ((((uint)(bp)[NF_Q6K_QH + ((e) >> 7) * 32 + ((e) & 31)] \
    >> (2u * (((e) >> 5) & 3u))) & 3u) << 4)) - 32)
/* Codes e0..e0+3: the nibbles of the group's 4 consecutive ql bytes and the 2-bit plane of its 4 consecutive
 * qh bytes (u16 pairs: the 210-byte block is 2-byte aligned) as lo | hi << 4 per byte, then minus 32 per byte
 * (NF_MMQ_BYTES_SUB). */
#define Q6_K_MMQ_CODE4(bp, e0) \
    NF_MMQ_BYTES_SUB(NF_MMQ_CODE4_NIBBLES(nf_read_u32_u16((bp) + NF_Q6K_QL + ((e0) >> 7) * 64 \
        + (((e0) >> 5) & 1) * 32 + ((e0) & 31)), (((e0) >> 6) & 1) * 4) \
        | (((nf_read_u32_u16((bp) + NF_Q6K_QH + ((e0) >> 7) * 32 + ((e0) & 31)) \
        >> (2u * (((e0) >> 5) & 3u))) & 0x03030303u) << 4), 0x20202020u)
#define Q6_K_MMQ_SCALE(bp, k0) (nf_kq_half(bp, NF_Q6K_D) * (float)((NF_GLOBAL const nf_i8*)((bp) + NF_Q6K_SCALES))[ \
    (((k0) % 256) >> 7) * 8 + (((((k0) % 256) >> 5) & 3) * 2) + (((k0) % 32) >> 4)])
#define Q6_K_MMQ_OFF(bp, k0) 0.0f
#define Q6_K_MMQ_HAS_OFF 0
#define Q6_K_MMQ_SCALE_NARROW 1
#define Q6_K_MMQ_CODE_SETS 2

/* ---- i-quants (common/quant_decode_iq.h group builders, as per-element codes) ---- */

/* IQ3_XXS: d(f16)@0, qs[64] grid indices@2, 8 x u32 scale+sign words@66. Sub-block sb (32 elements)
 * holds the scale (bits 28-31 of its word) and, per 8-element group, a 7-bit sign index;
 * value = dl * mag * sign. */
NF_INLINE int nf_mmq_iq3xxs_code(NF_GLOBAL const uchar* bp, int e) {
    const int sb = e >> 5, grp = (e >> 3) & 3, j = e & 7;
    /* the group's two grid words hold elements 0-3 and 4-7: byte j & 3 of word (j >> 2) */
    const uint word = IQ3XXS_GRID_CL[bp[2 + 8 * sb + 2 * grp + (j >> 2)]];
    const uint signs = (uint)KSIGNS_IQ2XS_CL[(read_u32_le(bp + 66 + 4 * sb) >> (7u * grp)) & 0x7Fu];
    return NF_MMQ_BYTE(word, j & 3u) * (1 - 2 * (int)((signs >> j) & 1u));
}
#define IQ3_XXS_MMQ_CODE(bp, e) nf_mmq_iq3xxs_code(bp, e)
/* The four codes of e0 (a multiple of 4) share one grid word and four bits of the group's sign byte. */
NF_INLINE uint nf_mmq_iq3xxs_code4(NF_GLOBAL const uchar* bp, int e0) {
    const int sb = e0 >> 5, grp = (e0 >> 3) & 3, h = (e0 >> 2) & 1;
    const uint word = IQ3XXS_GRID_CL[bp[2 + 8 * sb + 2 * grp + h]];
    const uint signs = (uint)KSIGNS_IQ2XS_CL[(read_u32_le(bp + 66 + 4 * sb) >> (7u * grp)) & 0x7Fu];
    return NF_MMQ_SIGNED4(word, (signs >> (4 * h)) & 0xFu);
}
#define IQ3_XXS_MMQ_CODE4(bp, e0) nf_mmq_iq3xxs_code4(bp, e0)
#define IQ3_XXS_MMQ_SCALE(bp, k0) (nf_ql_half(bp, 0) \
    * (0.5f + (float)(read_u32_le((bp) + 66 + 4 * (((k0) % 256) >> 5)) >> 28)) * 0.5f)
#define IQ3_XXS_MMQ_OFF(bp, k0) 0.0f
#define IQ3_XXS_MMQ_HAS_OFF 0
#define IQ3_XXS_MMQ_SCALE_NARROW 1
#define IQ3_XXS_MMQ_CODE_SETS 1

/* IQ3_S: d(f16)@0, qs[64]@2, qh[8]@66 (index bit 8), signs[32]@74, scales[4]@106; per 32-element
 * sub-block a shared scale, per 8-element group its sign byte. value = dl * mag * sign. */
NF_INLINE int nf_mmq_iq3s_code(NF_GLOBAL const uchar* bp, int e) {
    const int sb = e >> 5, grp = (e >> 3) & 3, j = e & 7;
    const uint qh = bp[66 + sb];
    const uint idx = (uint)bp[2 + 8 * sb + 2 * grp + (j >> 2)]
        | ((qh << ((j & 4) ? (7u - 2u * grp) : (8u - 2u * grp))) & 256u);
    const uint word = IQ3S_GRID_CL[idx];
    /* the group's two grid words hold elements 0-3 and 4-7: byte j & 3 of word (j >> 2) */
    return NF_MMQ_BYTE(word, j & 3u) * (1 - 2 * (int)(((uint)bp[74 + 4 * sb + grp] >> j) & 1u));
}
#define IQ3_S_MMQ_CODE(bp, e) nf_mmq_iq3s_code(bp, e)
/* The four codes of e0 (a multiple of 4) share one grid index (with its qh bit), one grid word and four bits of
 * the group's sign byte. */
NF_INLINE uint nf_mmq_iq3s_code4(NF_GLOBAL const uchar* bp, int e0) {
    const int sb = e0 >> 5, grp = (e0 >> 3) & 3, h = (e0 >> 2) & 1;
    const uint qh = bp[66 + sb];
    const uint idx = (uint)bp[2 + 8 * sb + 2 * grp + h] | ((qh << (8u - 2u * grp - (uint)h)) & 256u);
    const uint word = IQ3S_GRID_CL[idx];
    return NF_MMQ_SIGNED4(word, ((uint)bp[74 + 4 * sb + grp] >> (4 * h)) & 0xFu);
}
#define IQ3_S_MMQ_CODE4(bp, e0) nf_mmq_iq3s_code4(bp, e0)
#define IQ3_S_MMQ_SCALE(bp, k0) iq3s_scale((bp), ((k0) % 256) >> 5)
#define IQ3_S_MMQ_OFF(bp, k0) 0.0f
#define IQ3_S_MMQ_HAS_OFF 0
#define IQ3_S_MMQ_SCALE_NARROW 1
#define IQ3_S_MMQ_CODE_SETS 2

/* IQ2_XXS: d(f16)@0, 8 sub-blocks of 8 bytes: 4 grid indices and a u32 with the scale (bits 28-31)
 * and the sign indexes; value = dl * mag * sign, dl per sub-block. */
NF_INLINE int nf_mmq_iq2xxs_code(NF_GLOBAL const uchar* bp, int e) {
    const int sb = e >> 5, grp = (e >> 3) & 3, j = e & 7;
    const nf_u64 word = IQ2XXS_GRID_CL[bp[2 + 8 * sb + grp]];
    const uint signs = (uint)KSIGNS_IQ2XS_CL[(read_u32_le(bp + 2 + 8 * sb + 4) >> (7u * grp)) & 0x7Fu];
    return NF_MMQ_SM_CODE(word, j, signs);
}
#define IQ2_XXS_MMQ_CODE(bp, e) nf_mmq_iq2xxs_code(bp, e)
/* The four codes of e0 (a multiple of 4) share one half of the group's grid word and four bits of its sign byte. */
NF_INLINE uint nf_mmq_iq2xxs_code4(NF_GLOBAL const uchar* bp, int e0) {
    const int sb = e0 >> 5, grp = (e0 >> 3) & 3, h = (e0 >> 2) & 1;
    const nf_u64 word = IQ2XXS_GRID_CL[bp[2 + 8 * sb + grp]];
    const uint signs = (uint)KSIGNS_IQ2XS_CL[(read_u32_le(bp + 2 + 8 * sb + 4) >> (7u * grp)) & 0x7Fu];
    return NF_MMQ_SIGNED4((uint)(word >> (32 * h)), (signs >> (4 * h)) & 0xFu);
}
#define IQ2_XXS_MMQ_CODE4(bp, e0) nf_mmq_iq2xxs_code4(bp, e0)
#define IQ2_XXS_MMQ_SCALE(bp, k0) (nf_ql_half(bp, 0) \
    * (0.5f + (float)(read_u32_le((bp) + 2 + (((k0) % 256) >> 5) * 8 + 4) >> 28)) * 0.25f)
#define IQ2_XXS_MMQ_OFF(bp, k0) 0.0f
#define IQ2_XXS_MMQ_HAS_OFF 0
#define IQ2_XXS_MMQ_SCALE_NARROW 1
#define IQ2_XXS_MMQ_CODE_SETS 1

/* IQ2_XS: d(f16)@0, qs[32] u16 (bits 0-8 grid index, bits 9-15 sign index)@2, scales[8]@66; groups
 * 0-1 of a sub-block use the low scale nibble, groups 2-3 the high one. 16-element scale groups. */
NF_INLINE uint nf_mmq_iq2xs_qv(NF_GLOBAL const uchar* bp, int e) {
    const int o = 2 + 2 * ((e >> 5) * 4 + ((e >> 3) & 3));
    return (uint)bp[o] | ((uint)bp[o + 1] << 8);
}
NF_INLINE int nf_mmq_iq2xs_code(NF_GLOBAL const uchar* bp, int e) {
    const uint qv = nf_mmq_iq2xs_qv(bp, e);
    const nf_u64 word = IQ2XS_GRID_CL[qv & 511u];
    const uint signs = (uint)KSIGNS_IQ2XS_CL[qv >> 9];
    return NF_MMQ_SM_CODE(word, e & 7, signs);
}
#define IQ2_XS_MMQ_CODE(bp, e) nf_mmq_iq2xs_code(bp, e)
/* The four codes of e0 (a multiple of 4) share the group's index word: one half of its grid word and four bits
 * of its sign byte. */
NF_INLINE uint nf_mmq_iq2xs_code4(NF_GLOBAL const uchar* bp, int e0) {
    const int h = (e0 >> 2) & 1;
    const uint qv = nf_mmq_iq2xs_qv(bp, e0);
    const nf_u64 word = IQ2XS_GRID_CL[qv & 511u];
    const uint signs = (uint)KSIGNS_IQ2XS_CL[qv >> 9];
    return NF_MMQ_SIGNED4((uint)(word >> (32 * h)), (signs >> (4 * h)) & 0xFu);
}
#define IQ2_XS_MMQ_CODE4(bp, e0) nf_mmq_iq2xs_code4(bp, e0)
#define IQ2_XS_MMQ_SCALE(bp, k0) (nf_ql_half(bp, 0) * (0.5f \
    + (float)((((k0) % 256) & 16u) != 0u ? ((uint)(bp)[66 + (((k0) % 256) >> 5)] >> 4) \
                                         : ((uint)(bp)[66 + (((k0) % 256) >> 5)] & 0xFu))) * 0.25f)
#define IQ2_XS_MMQ_OFF(bp, k0) 0.0f
#define IQ2_XS_MMQ_HAS_OFF 0
#define IQ2_XS_MMQ_SCALE_NARROW 1
#define IQ2_XS_MMQ_CODE_SETS 2

/* IQ2_S: d(f16)@0, qs[32] grid indices@2, signs[32]@34, qh[8]@66 (index bits 8-9), scales[8]@74;
 * scale nibbles as IQ2_XS. 16-element scale groups. */
NF_INLINE int nf_mmq_iq2s_code(NF_GLOBAL const uchar* bp, int e) {
    const int sb = e >> 5, grp = (e >> 3) & 3, j = e & 7;
    const uint qh = bp[66 + sb];
    const uint idx = (uint)bp[2 + 4 * sb + grp] | ((qh << (8u - 2u * grp)) & 0x300u);
    const nf_u64 word = IQ2S_GRID_CL[idx];
    return NF_MMQ_SM_CODE(word, j, bp[34 + 4 * sb + grp]);
}
#define IQ2_S_MMQ_CODE(bp, e) nf_mmq_iq2s_code(bp, e)
/* The four codes of e0 (a multiple of 4) share one grid index (with its qh bits), one half of its grid word and
 * four bits of the group's sign byte. */
NF_INLINE uint nf_mmq_iq2s_code4(NF_GLOBAL const uchar* bp, int e0) {
    const int sb = e0 >> 5, grp = (e0 >> 3) & 3, h = (e0 >> 2) & 1;
    const uint qh = bp[66 + sb];
    const uint idx = (uint)bp[2 + 4 * sb + grp] | ((qh << (8u - 2u * grp)) & 0x300u);
    const nf_u64 word = IQ2S_GRID_CL[idx];
    return NF_MMQ_SIGNED4((uint)(word >> (32 * h)), ((uint)bp[34 + 4 * sb + grp] >> (4 * h)) & 0xFu);
}
#define IQ2_S_MMQ_CODE4(bp, e0) nf_mmq_iq2s_code4(bp, e0)
#define IQ2_S_MMQ_SCALE(bp, k0) (nf_ql_half(bp, 0) * (0.5f \
    + (float)((((k0) % 256) & 16u) != 0u ? ((uint)(bp)[74 + (((k0) % 256) >> 5)] >> 4) \
                                         : ((uint)(bp)[74 + (((k0) % 256) >> 5)] & 0xFu))) * 0.25f)
#define IQ2_S_MMQ_OFF(bp, k0) 0.0f
#define IQ2_S_MMQ_HAS_OFF 0
#define IQ2_S_MMQ_SCALE_NARROW 1
#define IQ2_S_MMQ_CODE_SETS 2

/* IQ1_S: d(f16)@0, qs[32] grid index low bytes@2, qh[8] u16@34: bits 3l..3l+2 the index bits 8-10 of
 * group l, bits 12-14 the sub-block scale, bit 15 the delta sign. value = dl * (val + delta) over
 * the ternary grid bytes, dl per 32-element sub-block. */
NF_INLINE uint nf_mmq_iq1s_qh(NF_GLOBAL const uchar* bp, int sb) {
    return (uint)bp[34 + 2 * sb] | ((uint)bp[35 + 2 * sb] << 8);
}
NF_INLINE int nf_mmq_iq1s_code(NF_GLOBAL const uchar* bp, int e) {
    const int sb = e >> 5, grp = (e >> 3) & 3, j = e & 7;
    const uint qh = nf_mmq_iq1s_qh(bp, sb);
    const uint idx = (uint)bp[2 + 4 * sb + grp] | (((qh >> (3u * grp)) & 7u) << 8);
    const nf_u64 word = IQ1S_GRID_CL[idx];
    return (int)NF_MMQ_BYTE(word, j) - (int)(((word >> (8u * j)) & 0x80u) << 1);
}
#define IQ1_S_MMQ_CODE(bp, e) nf_mmq_iq1s_code(bp, e)
/* The four codes of e0 (a multiple of 4) are one half of the group's grid word, whose bytes already hold the
 * ternary values as int8. */
NF_INLINE uint nf_mmq_iq1s_code4(NF_GLOBAL const uchar* bp, int e0) {
    const int sb = e0 >> 5, grp = (e0 >> 3) & 3, h = (e0 >> 2) & 1;
    const uint qh = nf_mmq_iq1s_qh(bp, sb);
    const uint idx = (uint)bp[2 + 4 * sb + grp] | (((qh >> (3u * grp)) & 7u) << 8);
    return (uint)(IQ1S_GRID_CL[idx] >> (32 * h));
}
#define IQ1_S_MMQ_CODE4(bp, e0) nf_mmq_iq1s_code4(bp, e0)
#define IQ1_S_MMQ_DL(bp, k0) (nf_ql_half(bp, 0) \
    * (float)(2u * ((nf_mmq_iq1s_qh((bp), ((k0) % 256) >> 5) >> 12) & 7u) + 1u))
#define IQ1_S_MMQ_SCALE(bp, k0) IQ1_S_MMQ_DL(bp, k0)
#define IQ1_S_MMQ_OFF(bp, k0) (IQ1_S_MMQ_DL(bp, k0) \
    * (((nf_mmq_iq1s_qh((bp), ((k0) % 256) >> 5)) & 0x8000u) ? -IQ1_S_DELTA : IQ1_S_DELTA))
#define IQ1_S_MMQ_HAS_OFF 1
#define IQ1_S_MMQ_SCALE_NARROW 1
#define IQ1_S_MMQ_CODE_SETS 1

/* IQ4_NL: d(f16)@0, qs[16]@2; value = d * kvalues_iq4nl[nibble]. */
#define IQ4_NL_MMQ_CODE(bp, e) (int)(IQ4_NL_KVALUES[(e) < 16 \
    ? ((uint)(bp)[2 + ((e) & 15)] & 0xFu) \
    : ((uint)(bp)[2 + ((e) & 15)] >> 4)])
/* Codes e0..e0+3: the group's 4 qs bytes (two aligned u16s) through the codebook. */
NF_INLINE uint nf_mmq_iq4nl_code4(NF_GLOBAL const uchar* bp, int e0) {
    const uint word = nf_read_u32_u16(bp + 2 + (e0 & 15));
    const int shift = (e0 & 16) ? 4 : 0;
#if defined(__CUDA_ARCH__)
    return nf_iq4nl_codes4(NF_MMQ_CODE4_NIBBLES(word, shift));
#else
    return NF_MMQ_CODE4_PACK(IQ4_NL_KVALUES[(word >> shift) & 0xFu],
        IQ4_NL_KVALUES[(word >> (8 + shift)) & 0xFu],
        IQ4_NL_KVALUES[(word >> (16 + shift)) & 0xFu],
        IQ4_NL_KVALUES[(word >> (24 + shift)) & 0xFu]);
#endif
}
#define IQ4_NL_MMQ_CODE4(bp, e0) nf_mmq_iq4nl_code4((bp), (e0))
#define IQ4_NL_MMQ_SCALE(bp, k0) nf_ql_half(bp, 0)
#define IQ4_NL_MMQ_OFF(bp, k0) 0.0f
#define IQ4_NL_MMQ_HAS_OFF 0
#define IQ4_NL_MMQ_SCALE_NARROW 1
#define IQ4_NL_MMQ_CODE_SETS 2

/* IQ4_XS: d(f16)@0, scales_h/l@2 (iq4xs_sub_scale), qs[128]@8; 32-element sub-blocks carry the
 * scale, value = sub_scale * kvalues_iq4nl[nibble]. */
#define IQ4_XS_MMQ_CODE(bp, e) (int)(IQ4_NL_KVALUES[(((e) % 32) < 16 \
    ? ((uint)(bp)[IQ4_XS_HEADER_BYTES + ((e) >> 5) * IQ4_XS_SUB_BLOCK_QUANT_BYTES + ((e) & 15)] & 0xFu) \
    : ((uint)(bp)[IQ4_XS_HEADER_BYTES + ((e) >> 5) * IQ4_XS_SUB_BLOCK_QUANT_BYTES + ((e) & 15)] >> 4))])
/* Codes e0..e0+3: the group's qs bytes as one aligned u32 (the group stays inside one nibble half
 * of the sub-block), through the codebook. */
NF_INLINE uint nf_mmq_iq4xs_code4(NF_GLOBAL const uchar* bp, int e0) {
    const uint word = nf_read_u32(bp + IQ4_XS_HEADER_BYTES
        + (e0 >> 5) * IQ4_XS_SUB_BLOCK_QUANT_BYTES + (e0 & 15));
    const int shift = ((e0 % 32) < 16) ? 0 : 4;
#if defined(__CUDA_ARCH__)
    return nf_iq4nl_codes4(NF_MMQ_CODE4_NIBBLES(word, shift));
#else
    return NF_MMQ_CODE4_PACK(IQ4_NL_KVALUES[(word >> shift) & 0xFu],
        IQ4_NL_KVALUES[(word >> (8 + shift)) & 0xFu],
        IQ4_NL_KVALUES[(word >> (16 + shift)) & 0xFu],
        IQ4_NL_KVALUES[(word >> (24 + shift)) & 0xFu]);
#endif
}
#define IQ4_XS_MMQ_CODE4(bp, e0) nf_mmq_iq4xs_code4((bp), (e0))
#define IQ4_XS_MMQ_SCALE(bp, k0) iq4xs_sub_scale((bp), ((k0) % 256) >> 5)
#define IQ4_XS_MMQ_OFF(bp, k0) 0.0f
#define IQ4_XS_MMQ_HAS_OFF 0
#define IQ4_XS_MMQ_SCALE_NARROW 1
#define IQ4_XS_MMQ_CODE_SETS 2

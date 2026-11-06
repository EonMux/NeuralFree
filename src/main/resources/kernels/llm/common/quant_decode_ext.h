/* common/quant_decode_ext.h - F8_E5M2, F8_E4M3, F8_E4M3_B128, F8_E4M3_B32, MXFP8, MXFP6_E2M3, MXFP6_E3M2, I8_B128,
 * NF4 and FP4_BNB decode triples. Requires: common/dialect.h, common/config.h, common/quant_blocks.h,
 * common/quant_decode_new.h */
/*
 * Decode triples (_decode, _decode_dual, _decode_elem) of the NeuralFree extension block types
 * (gguf.GGMLType ids 1000-1009, field offsets of common/quant_blocks.h) for the IQ_ALL_KERNELS contract
 * of common/matvec_family.h; _decode and _decode_dual come from NF_NQ_DOT_PAIR (common/quant_decode_new.h).
 *
 * Element j of a block:
 *   F8_E5M2        e5m2(qs[j])                 (a per-tensor or per-expert .scale multiplies the product)
 *   F8_E4M3        e4m3(qs[j])                 (same .scale rule)
 *   F8_E4M3_B128   d * e4m3(qs[j])
 *   F8_E4M3_B32    d * e4m3(qs[j])
 *   MXFP8          2^(e - 127) * e4m3(qs[j])
 *   MXFP6_E2M3     2^(e - 127) * e2m3(code j)  (element j = code j, four 6-bit codes per three bytes)
 *   MXFP6_E3M2     2^(e - 127) * e3m2(code j)
 *   I8_B128        d * qs[j]                   (qs signed)
 *   NF4, FP4_BNB   absmax * CODE[q], q the low nibble of qs[j] for j < 32, the high nibble of qs[j - 32]
 *                  otherwise; CODE is NF4_CODE or FP4_BNB_CODE
 * with e4m3 = F8_E4M3_LUT and the E8M0 factor e8m0_scale; F8_E5M2 decodes its code as the high byte of an
 * IEEE half (half_bits_to_float), the exact value of every code including the infinities and NaNs.
 *
 * Work split: every type has BS / TPS = NF_NQ_EPT = 8, and work-item ltid owns the consecutive elements
 * [8*ltid, 8*ltid + 8) as the runs [8*ltid, 8*ltid + 4) and [8*ltid + 4, 8*ltid + 8):
 *   F8_E5M2, F8_E4M3, F8_E4M3_B32, MXFP8,
 *   MXFP6_E2M3, MXFP6_E3M2                         TPS = 4    qs bytes 8*ltid .. 8*ltid + 7
 *                                                               (MXFP6: the 48 bits at qs + 6*ltid)
 *   F8_E4M3_B128, I8_B128    TPS = 16   qs bytes 8*ltid .. 8*ltid + 7
 *   NF4, FP4_BNB             TPS = 8    qs bytes 8*(ltid%4) .. 8*(ltid%4) + 7, low nibbles for ltid < 4,
 *                                       high nibbles for ltid >= 4
 *
 * Hardware-FP8 fast paths. For each of the four FP8 code types (F8_E5M2, F8_E4M3, F8_E4M3_B128 and MXFP8)
 * the file defines a second triple T_Hw (kernel names {T}_Hw_* of GpuOps' hardware-FP8 set, selected when
 * the device reports the feature): the same decoder contract, with the codes converted in pairs by the
 * hardware FP8 conversions (CUDA sm_89+: cvt.rn.f16x2.e4m3x2 / .e5m2x2 of the PTX ISA; on other targets
 * the portable branch). Both paths give the identical value for every finite code, so a _Hw result is
 * bit-identical to its generic row; NaN codes run through the generic reader, which maps them to zero as
 * F8_E4M3_LUT does (the repackers reject the NaN codes). F8_E5M2's value is the IEEE-half upper byte also
 * on the portable branch, so there the two paths even share one reader.
 */
#ifndef NF_QUANT_DECODE_EXT_H
#define NF_QUANT_DECODE_EXT_H

#define F8_E5M2_TPS      4
#define F8_E4M3_TPS      4
#define F8_E4M3_B128_TPS 16
#define F8_E4M3_B32_TPS  4
#define MXFP8_TPS        4
#define I8_B128_TPS      16
#define NF4_TPS          8
#define FP4_BNB_TPS      8

/* Block-geometry aliases of the _Hw triples (NF_NQ_DOT_PAIR reads T##_BLOCK_SIZE; the block bytes are
 * those of the generic type, so are the _TPS values above). */
#define F8_E5M2_Hw_BLOCK_SIZE       F8_E5M2_BLOCK_SIZE
#define F8_E4M3_Hw_BLOCK_SIZE       F8_E4M3_BLOCK_SIZE
#define F8_E4M3_B128_Hw_BLOCK_SIZE  F8_E4M3_B128_BLOCK_SIZE
#define MXFP8_Hw_BLOCK_SIZE         MXFP8_BLOCK_SIZE

/* Distance from the first to the second 4-element run of a work-item. */
#define NF_XQ_X1 4

/* f32 field at byte offset off of block bp (4-byte aligned, see the file comment). */
NF_INLINE float nf_xq_f32(NF_GLOBAL const uchar* bp, int off) {
    return *(NF_GLOBAL const float*)(bp + off);
}

/* E4M3FN values of the 4 codes at p, element k from byte p[k]. */
NF_INLINE float4 nf_xq_e4m3x4(NF_GLOBAL const uchar* p) {
    const uchar4 b = vload4(0, p);
    return NF_F4(F8_E4M3_LUT[b.x], F8_E4M3_LUT[b.y], F8_E4M3_LUT[b.z], F8_E4M3_LUT[b.w]);
}

/* Unscaled E4M3FN values of the 8 elements work-item ltid owns, from the codes starting at qs. */
NF_INLINE void nf_xq_e4m3_w8(NF_GLOBAL const uchar* qs, int ltid, float4* w0, float4* w1) {
    NF_GLOBAL const uchar* q = qs + ltid * NF_NQ_EPT;
    *w0 = nf_xq_e4m3x4(q);
    *w1 = nf_xq_e4m3x4(q + NF_XQ_X1);
}

/* ---- F8_E4M3: qs[32] E4M3FN codes, no block scale ---- */
NF_INLINE void nf_f8_w8(NF_GLOBAL const uchar* bp, int ltid, float4* w0, float4* w1) {
    nf_xq_e4m3_w8(bp + NF_F8_E4M3_QS, ltid, w0, w1);
}
NF_INLINE float nf_f8_scale(NF_GLOBAL const uchar* bp, int ltid) {
    return 1.0f;
}
NF_NQ_DOT_PAIR(F8_E4M3, nf_f8_w8, nf_f8_scale, nf_nq_xo8, NF_XQ_X1)
NF_INLINE float F8_E4M3_decode_elem(NF_GLOBAL const uchar* w, int dim, int tok, int i) {
    NF_GLOBAL const uchar* bp = nf_nq_block(w, dim, tok, i, F8_E4M3_BLOCK_SIZE, F8_E4M3_BLOCK_BYTES);
    return F8_E4M3_LUT[bp[NF_F8_E4M3_QS + i % F8_E4M3_BLOCK_SIZE]];
}

/* ---- F8_E4M3_B128: f32 d, qs[128] E4M3FN codes ---- */
NF_INLINE void nf_f8b128_w8(NF_GLOBAL const uchar* bp, int ltid, float4* w0, float4* w1) {
    nf_xq_e4m3_w8(bp + NF_F8_E4M3_B128_QS, ltid, w0, w1);
}
NF_INLINE float nf_f8b128_scale(NF_GLOBAL const uchar* bp, int ltid) {
    return nf_xq_f32(bp, NF_F8_E4M3_B128_D);
}
NF_NQ_DOT_PAIR(F8_E4M3_B128, nf_f8b128_w8, nf_f8b128_scale, nf_nq_xo8, NF_XQ_X1)
NF_INLINE float F8_E4M3_B128_decode_elem(NF_GLOBAL const uchar* w, int dim, int tok, int i) {
    NF_GLOBAL const uchar* bp = nf_nq_block(w, dim, tok, i, F8_E4M3_B128_BLOCK_SIZE, F8_E4M3_B128_BLOCK_BYTES);
    return nf_xq_f32(bp, NF_F8_E4M3_B128_D)
         * F8_E4M3_LUT[bp[NF_F8_E4M3_B128_QS + i % F8_E4M3_B128_BLOCK_SIZE]];
}

/* ---- F8_E4M3_B32: f32 d, qs[32] E4M3FN codes ---- */
NF_INLINE void nf_f8b32_w8(NF_GLOBAL const uchar* bp, int ltid, float4* w0, float4* w1) {
    nf_xq_e4m3_w8(bp + NF_F8_E4M3_B32_QS, ltid, w0, w1);
}
NF_INLINE float nf_f8b32_scale(NF_GLOBAL const uchar* bp, int ltid) {
    return nf_xq_f32(bp, NF_F8_E4M3_B32_D);
}
NF_NQ_DOT_PAIR(F8_E4M3_B32, nf_f8b32_w8, nf_f8b32_scale, nf_nq_xo8, NF_XQ_X1)
NF_INLINE float F8_E4M3_B32_decode_elem(NF_GLOBAL const uchar* w, int dim, int tok, int i) {
    NF_GLOBAL const uchar* bp = nf_nq_block(w, dim, tok, i, F8_E4M3_B32_BLOCK_SIZE, F8_E4M3_B32_BLOCK_BYTES);
    return nf_xq_f32(bp, NF_F8_E4M3_B32_D)
         * F8_E4M3_LUT[bp[NF_F8_E4M3_B32_QS + i % F8_E4M3_B32_BLOCK_SIZE]];
}

/* ---- MXFP8: E8M0 e, qs[32] E4M3FN codes ---- */
NF_INLINE void nf_mxfp8_w8(NF_GLOBAL const uchar* bp, int ltid, float4* w0, float4* w1) {
    nf_xq_e4m3_w8(bp + NF_MXFP8_QS, ltid, w0, w1);
}
NF_INLINE float nf_mxfp8_scale(NF_GLOBAL const uchar* bp, int ltid) {
    return e8m0_scale((uint)bp[NF_MXFP8_E]);
}
NF_NQ_DOT_PAIR(MXFP8, nf_mxfp8_w8, nf_mxfp8_scale, nf_nq_xo8, NF_XQ_X1)
NF_INLINE float MXFP8_decode_elem(NF_GLOBAL const uchar* w, int dim, int tok, int i) {
    NF_GLOBAL const uchar* bp = nf_nq_block(w, dim, tok, i, MXFP8_BLOCK_SIZE, MXFP8_BLOCK_BYTES);
    return e8m0_scale((uint)bp[NF_MXFP8_E]) * F8_E4M3_LUT[bp[NF_MXFP8_QS + i % MXFP8_BLOCK_SIZE]];
}

/* ---- MXFP6_E2M3 and MXFP6_E3M2: E8M0 e, qs[24] packed 6-bit codes (Quark OCP-MX FP6, 6.25 bits/weight) ----
 * Element j is code j (four 6-bit codes per three bytes, little-endian bit order, the OCP-MX element
 * order); the value is 2^(e - 127) * the absolute value, the exact arithmetic of the Java
 * FpBlockDecoders.e2m3ToFp32 / e3m2ToFp32 and the GLSL twin nf_qe_mxfp6x4 (common/quant_ext.glsl):
 *   E2M3  sign 5, e bits 4-3 (bias 1), m bits 2-0: v = (e ? 8 + m : m) * 2^(max(e, 1) - 4)
 *   E3M2  sign 5, e bits 4-2 (bias 3), m bits 1-0: v = (e ? 4 + m : m) * 2^(max(e, 1) - 5)
 * The fp6 formats have no infinities and no NaNs (the largest magnitudes are 7.5 and 28 and the
 * smallest subnormals 2^-3 and 2^-4), so every one of the 64 codes is finite; the NaN E8M0
 * reservation e == 0xFF is rejected at load (gguf.quant.ExtBlockDecoders and the repackers).
 * NF_XQ_MXFP6(T, t, VAL4, VAL, E, QS) defines the W8/SCALE pair, NF_NQ_DOT_PAIR and T_decode_elem for
 * type T with helper prefix nf_<t>, the 4-code group of a 24-bit word VAL4 and the code VAL. Work-item
 * ltid (TPS = 4, NF_NQ_EPT = 8 elements) reads 48 bits = 6 bytes at qs + 6*ltid: the low 24 bits are
 * its elements 8*ltid .. +3, bits 24..47 (two u32 reads, the second shifted right by 8) elements
 * 8*ltid + 4 .. +7. */
#define MXFP6_E2M3_TPS 4
#define MXFP6_E3M2_TPS 4

#define NF_XQ_MXFP6(T, t, VAL4, VAL, E, QS) \
NF_INLINE void nf_##t##_w8(NF_GLOBAL const uchar* bp, int ltid, float4* w0, float4* w1) { \
    NF_GLOBAL const uchar* q = bp + (QS) + NF_MXFP6_BYTES_PER_WORKITEM * ltid; \
    *w0 = VAL4(read_u32_le(q)); \
    *w1 = VAL4(read_u32_le(q + 2) >> 8); \
} \
NF_INLINE float nf_##t##_scale(NF_GLOBAL const uchar* bp, int ltid) { \
    return e8m0_scale((uint)bp[(E)]); \
} \
NF_NQ_DOT_PAIR(T, nf_##t##_w8, nf_##t##_scale, nf_nq_xo8, NF_XQ_X1) \
NF_INLINE float T##_decode_elem(NF_GLOBAL const uchar* w, int dim, int tok, int i) { \
    NF_GLOBAL const uchar* bp = nf_nq_block(w, dim, tok, i, T##_BLOCK_SIZE, T##_BLOCK_BYTES); \
    const int e = i % T##_BLOCK_SIZE; \
    NF_GLOBAL const uchar* q = bp + (QS) + NF_MXFP6_BYTES_PER_WORKITEM * (e >> 2); \
    const int code = (int)((read_u32_le(q) >> (uint)(6 * (e & 3))) & 63u); \
    return e8m0_scale((uint)bp[(E)]) * VAL((uint)code); \
}

/* The value of an E2M3 code (sign 5, e bits 4-3, mantissa bits 2-0); 4 codes of a 24-bit little-endian word. */
NF_INLINE float nf_xq_e2m3(uint c) {
    const uint e = (c >> 3) & 3u, m = c & 7u;
    const float v = (float)(e == 0u ? m : 8u + m) * as_float(((e == 0u ? 1u : e) + 124u - 1u) << 23);
    return (c & 0x20u) != 0u ? -v : v;
}
NF_INLINE float4 nf_xq_e2m3x4(uint word) {
    return NF_F4(nf_xq_e2m3(word & 63u), nf_xq_e2m3((word >> 6) & 63u),
                 nf_xq_e2m3((word >> 12) & 63u), nf_xq_e2m3((word >> 18) & 63u));
}

/* The value of an E3M2 code (sign 5, e bits 4-2, mantissa bits 1-0); 4 codes of a 24-bit little-endian word. */
NF_INLINE float nf_xq_e3m2(uint c) {
    const uint e = (c >> 2) & 7u, m = c & 3u;
    const float v = (float)(e == 0u ? m : 4u + m) * as_float(((e == 0u ? 1u : e) + 124u - 2u) << 23);
    return (c & 0x20u) != 0u ? -v : v;
}
NF_INLINE float4 nf_xq_e3m2x4(uint word) {
    return NF_F4(nf_xq_e3m2(word & 63u), nf_xq_e3m2((word >> 6) & 63u),
                 nf_xq_e3m2((word >> 12) & 63u), nf_xq_e3m2((word >> 18) & 63u));
}

NF_XQ_MXFP6(MXFP6_E2M3, mxfp6e2m3, nf_xq_e2m3x4, nf_xq_e2m3, NF_MXFP6_E, NF_MXFP6_QS)
NF_XQ_MXFP6(MXFP6_E3M2, mxfp6e3m2, nf_xq_e3m2x4, nf_xq_e3m2, NF_MXFP6_E, NF_MXFP6_QS)

/* ---- I8_B128: f32 d, qs[128] int8 ---- */
NF_INLINE void nf_i8b128_w8(NF_GLOBAL const uchar* bp, int ltid, float4* w0, float4* w1) {
    NF_GLOBAL const nf_i8* q = (NF_GLOBAL const nf_i8*)(bp + NF_I8_B128_QS + ltid * NF_NQ_EPT);
    *w0 = convert_float4(vload4(0, q));
    *w1 = convert_float4(vload4(0, q + NF_XQ_X1));
}
NF_INLINE float nf_i8b128_scale(NF_GLOBAL const uchar* bp, int ltid) {
    return nf_xq_f32(bp, NF_I8_B128_D);
}
NF_NQ_DOT_PAIR(I8_B128, nf_i8b128_w8, nf_i8b128_scale, nf_nq_xo8, NF_XQ_X1)
NF_INLINE float I8_B128_decode_elem(NF_GLOBAL const uchar* w, int dim, int tok, int i) {
    NF_GLOBAL const uchar* bp = nf_nq_block(w, dim, tok, i, I8_B128_BLOCK_SIZE, I8_B128_BLOCK_BYTES);
    return nf_xq_f32(bp, NF_I8_B128_D)
         * (float)((NF_GLOBAL const nf_i8*)(bp + NF_I8_B128_QS))[i % I8_B128_BLOCK_SIZE];
}

/* ---- NF4 and FP4_BNB: f32 absmax, qs[32] codebook nibbles (bitsandbytes 4-bit, one 64-element block) ----
 * NF_XQ_BNB4(T, t, CODE, D, QS) defines, for type T with helper prefix nf_<t>, the codebook lookup of 4
 * nibbles, the W8 and SCALE pair, NF_NQ_DOT_PAIR and T_decode_elem. Work-item ltid reads the nibble
 * plane ltid / 4 (0 low, 1 high) of qs bytes 8*(ltid%4) .. +7, which are its elements 8*ltid .. +7. */
#define NF_XQ_BNB4(T, t, CODE, D, QS) \
NF_INLINE float4 nf_##t##_code4(NF_GLOBAL const uchar* p, uint sh) { \
    const uchar4 b = vload4(0, p); \
    return NF_F4(CODE[((uint)b.x >> sh) & 0xFu], CODE[((uint)b.y >> sh) & 0xFu], \
                 CODE[((uint)b.z >> sh) & 0xFu], CODE[((uint)b.w >> sh) & 0xFu]); \
} \
NF_INLINE void nf_##t##_w8(NF_GLOBAL const uchar* bp, int ltid, float4* w0, float4* w1) { \
    NF_GLOBAL const uchar* q = bp + (QS) + (ltid & 3) * NF_NQ_EPT; \
    const uint sh = (uint)(ltid >> 2) * 4u; \
    *w0 = nf_##t##_code4(q, sh); \
    *w1 = nf_##t##_code4(q + NF_XQ_X1, sh); \
} \
NF_INLINE float nf_##t##_scale(NF_GLOBAL const uchar* bp, int ltid) { \
    return nf_xq_f32(bp, (D)); \
} \
NF_NQ_DOT_PAIR(T, nf_##t##_w8, nf_##t##_scale, nf_nq_xo8, NF_XQ_X1) \
NF_INLINE float T##_decode_elem(NF_GLOBAL const uchar* w, int dim, int tok, int i) { \
    NF_GLOBAL const uchar* bp = nf_nq_block(w, dim, tok, i, T##_BLOCK_SIZE, T##_BLOCK_BYTES); \
    const int e = i % T##_BLOCK_SIZE; \
    const uint q = bp[(QS) + (e & (NF_BNB4_HI - 1))]; \
    return nf_xq_f32(bp, (D)) * CODE[e < NF_BNB4_HI ? (q & 0xFu) : (q >> 4)]; \
}

NF_XQ_BNB4(NF4, nf4, NF4_CODE, NF_NF4_D, NF_NF4_QS)
NF_XQ_BNB4(FP4_BNB, fp4bnb, FP4_BNB_CODE, NF_FP4_BNB_D, NF_FP4_BNB_QS)

/* ---- F8_E5M2: qs[32] E5M2 codes (IEEE half high bytes), no block scale ---- */
NF_INLINE float4 nf_xq_e5m2x4(NF_GLOBAL const uchar* p) {
    const uchar4 b = vload4(0, p);
    return NF_F4(half_bits_to_float((uint)b.x << 8), half_bits_to_float((uint)b.y << 8),
                 half_bits_to_float((uint)b.z << 8), half_bits_to_float((uint)b.w << 8));
}
NF_INLINE void nf_f8e5_w8(NF_GLOBAL const uchar* bp, int ltid, float4* w0, float4* w1) {
    NF_GLOBAL const uchar* q = bp + NF_F8_E5M2_QS + ltid * NF_NQ_EPT;
    *w0 = nf_xq_e5m2x4(q);
    *w1 = nf_xq_e5m2x4(q + NF_XQ_X1);
}
NF_INLINE float nf_f8e5_scale(NF_GLOBAL const uchar* bp, int ltid) {
    return 1.0f;
}
NF_NQ_DOT_PAIR(F8_E5M2, nf_f8e5_w8, nf_f8e5_scale, nf_nq_xo8, NF_XQ_X1)
NF_INLINE float F8_E5M2_decode_elem(NF_GLOBAL const uchar* w, int dim, int tok, int i) {
    NF_GLOBAL const uchar* bp = nf_nq_block(w, dim, tok, i, F8_E5M2_BLOCK_SIZE, F8_E5M2_BLOCK_BYTES);
    return half_bits_to_float((uint)bp[NF_F8_E5M2_QS + i % F8_E5M2_BLOCK_SIZE] << 8);
}

/* ---- hardware-FP8 pairs (NF_XQ_HW_CVT; see the file comment) ----
 * Without the hardware conversions the _Hw readers alias the generic ones, so the _Hw triples equal
 * the generic triples exactly and stay bit-identical on every target. With them (CUDA sm_89+) each
 * pair of codes is converted by cvt.rn.f16x2.e4m3x2 / .e5m2x2 to an f16x2 word, whose two halves carry
 * the generic values; NaN codes go through the generic reader. */
#if !defined(NF_XQ_HW_CVT)
#define NF_XQ_HW_CVT 0
#endif

#define NF_XQ_HW_W8(FN, HW4, OFF) \
NF_INLINE void FN(NF_GLOBAL const uchar* bp, int ltid, float4* w0, float4* w1) { \
    NF_GLOBAL const uchar* q = bp + (OFF) + ltid * NF_NQ_EPT; \
    *w0 = HW4(q); \
    *w1 = HW4(q + NF_XQ_X1); \
}

#if NF_XQ_HW_CVT
/* The f16x2 word of the hardware conversion of the code pair (c0 in its low byte, c1 above it). */
NF_INLINE uint nf_xq_cvt(uint c0, uint c1, int e5m2) {
    unsigned in = (unsigned)(c0 | (c1 << 8));
    unsigned h;
    if (e5m2) {
        asm("cvt.rn.f16x2.e5m2x2 %0, %1;" : "=r"(h) : "h"((unsigned short)in));
    } else {
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h) : "h"((unsigned short)in));
    }
    return h;
}

/* Hardware values of the 4 codes at p; a block holding a NaN code uses the generic reader
 * (its NaN values, zero for E4M3FN as the LUT and a NaN for E5M2 as the half decode). */
#define NF_XQ_HW4(NAME, CSVP, GENERIC) \
NF_INLINE float4 NAME(NF_GLOBAL const uchar* p) { \
    const uchar4 b = vload4(0, p); \
    if ((b.x & 0x7F) == 0x7F || (b.y & 0x7F) == 0x7F || (b.z & 0x7F) == 0x7F || (b.w & 0x7F) == 0x7F) { \
        return GENERIC; \
    } \
    const uint h01 = nf_xq_cvt(b.x, b.y, CSVP), h23 = nf_xq_cvt(b.z, b.w, CSVP); \
    return NF_F4(half_bits_to_float(h01 & 0xFFFFu), half_bits_to_float(h01 >> 16), \
                 half_bits_to_float(h23 & 0xFFFFu), half_bits_to_float(h23 >> 16)); \
}

NF_XQ_HW4(nf_xq_e4m3hw_x4, 0,
          NF_F4(F8_E4M3_LUT[b.x], F8_E4M3_LUT[b.y], F8_E4M3_LUT[b.z], F8_E4M3_LUT[b.w]))
NF_XQ_HW4(nf_xq_e5m2hw_x4, 1, nf_xq_e5m2x4(p))
#else /* portable: the hardware readers are the generic ones */
#define nf_xq_e4m3hw_x4 nf_xq_e4m3x4
#define nf_xq_e5m2hw_x4 nf_xq_e5m2x4
#endif /* NF_XQ_HW_CVT */

/* The four _Hw triples. */
NF_XQ_HW_W8(nf_f8e5hw_w8, nf_xq_e5m2hw_x4, NF_F8_E5M2_QS)
NF_NQ_DOT_PAIR(F8_E5M2_Hw, nf_f8e5hw_w8, nf_f8e5_scale, nf_nq_xo8, NF_XQ_X1)
NF_XQ_HW_W8(nf_f8hw_w8, nf_xq_e4m3hw_x4, NF_F8_E4M3_QS)
NF_NQ_DOT_PAIR(F8_E4M3_Hw, nf_f8hw_w8, nf_f8_scale, nf_nq_xo8, NF_XQ_X1)
NF_XQ_HW_W8(nf_f8b128hw_w8, nf_xq_e4m3hw_x4, NF_F8_E4M3_B128_QS)
NF_NQ_DOT_PAIR(F8_E4M3_B128_Hw, nf_f8b128hw_w8, nf_f8b128_scale, nf_nq_xo8, NF_XQ_X1)
NF_XQ_HW_W8(nf_mxfp8hw_w8, nf_xq_e4m3hw_x4, NF_MXFP8_QS)
NF_NQ_DOT_PAIR(MXFP8_Hw, nf_mxfp8hw_w8, nf_mxfp8_scale, nf_nq_xo8, NF_XQ_X1)

/* The per-element reader of a _Hw triple: the generic reader, which runs the NaN codes and agrees with every
 * converted pair (see the file comment), so a _Hw row's element output is the generic row's. */
NF_INLINE float F8_E5M2_Hw_decode_elem(NF_GLOBAL const uchar* w, int dim, int tok, int i) {
    return F8_E5M2_decode_elem(w, dim, tok, i);
}
NF_INLINE float F8_E4M3_Hw_decode_elem(NF_GLOBAL const uchar* w, int dim, int tok, int i) {
    return F8_E4M3_decode_elem(w, dim, tok, i);
}
NF_INLINE float F8_E4M3_B128_Hw_decode_elem(NF_GLOBAL const uchar* w, int dim, int tok, int i) {
    return F8_E4M3_B128_decode_elem(w, dim, tok, i);
}
NF_INLINE float MXFP8_Hw_decode_elem(NF_GLOBAL const uchar* w, int dim, int tok, int i) {
    return MXFP8_decode_elem(w, dim, tok, i);
}

#endif /* NF_QUANT_DECODE_EXT_H */

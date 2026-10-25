/* common/quant_decode_ext.h - F8_E4M3, F8_E4M3_B128, MXFP8, I8_B128, NF4 and FP4_BNB decode triples. Requires: common/dialect.h, common/config.h, common/quant_blocks.h, common/quant_decode_new.h */
/*
 * Decode triples (_decode, _decode_dual, _decode_elem) of the NeuralFree extension block types
 * (gguf.GGMLType ids 1000-1005, field offsets of common/quant_blocks.h) for the IQ_ALL_KERNELS contract
 * of common/matvec_family.h; _decode and _decode_dual come from NF_NQ_DOT_PAIR (common/quant_decode_new.h).
 *
 * Element j of a block:
 *   F8_E4M3        e4m3(qs[j])                 (a per-tensor or per-expert .scale multiplies the product)
 *   F8_E4M3_B128   d * e4m3(qs[j])
 *   MXFP8          2^(e - 127) * e4m3(qs[j])
 *   I8_B128        d * qs[j]                   (qs signed)
 *   NF4, FP4_BNB   absmax * CODE[q], q the low nibble of qs[j] for j < 32, the high nibble of qs[j - 32]
 *                  otherwise; CODE is NF4_CODE or FP4_BNB_CODE
 * with e4m3 = F8_E4M3_LUT and the E8M0 factor e8m0_scale.
 *
 * Work split: every type has BS / TPS = NF_NQ_EPT = 8, and work-item ltid owns the consecutive elements
 * [8*ltid, 8*ltid + 8) as the runs [8*ltid, 8*ltid + 4) and [8*ltid + 4, 8*ltid + 8):
 *   F8_E4M3, MXFP8          TPS = 4    qs bytes 8*ltid .. 8*ltid + 7
 *   F8_E4M3_B128, I8_B128   TPS = 16   qs bytes 8*ltid .. 8*ltid + 7
 *   NF4, FP4_BNB            TPS = 8    qs bytes 8*(ltid%4) .. 8*(ltid%4) + 7, low nibbles for ltid < 4,
 *                                      high nibbles for ltid >= 4
 *
 * The f32 d / absmax field sits at block offset 0 and is read as a float: the 132- and 36-byte blocks
 * are multiples of 4 bytes and rows hold whole blocks, so the field is 4-byte aligned whenever the
 * weight buffer is. MXFP8 blocks (33 bytes) are read bytewise.
 */
#ifndef NF_QUANT_DECODE_EXT_H
#define NF_QUANT_DECODE_EXT_H

#define F8_E4M3_TPS      4
#define F8_E4M3_B128_TPS 16
#define MXFP8_TPS        4
#define I8_B128_TPS      16
#define NF4_TPS          8
#define FP4_BNB_TPS      8

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

#endif /* NF_QUANT_DECODE_EXT_H */

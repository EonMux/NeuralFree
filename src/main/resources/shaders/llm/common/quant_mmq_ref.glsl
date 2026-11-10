/* common/quant_mmq_ref.glsl - the MMQ weight types as VK_NV_cooperative_matrix2 decode callbacks
 * (mmq_ref_decode): the tensor load positions a physical-storage-buffer reference at the element's
 * weight block and asks for element e (coordInBlock[1]) of it. The reference type NfMmqRef is one
 * block of 16-bit words, so its sizeof is the block's byte size: the tensor layout advances block
 * references by sizeof, and an unsized or wrongly sized record gives the layout the wrong block
 * stride. Every GGUF block size is even, so all fields are read through aligned 16-bit loads (two for
 * an fp32 scale), as llama.cpp's dequant_funcs_cm2.glsl reads its packed16 blocks. The value formulas
 * equal quant_mmq.glsl / quant_k.glsl; the result is rounded to fp16 once, as the KHR CM form rounds
 * its staged tile.
 * Requires: common/nf_common.glsl, common/quant_blocks.glsl (the QT_* ids); QTYPE selects the type.
 */
/* Portions derived from llama.cpp/ggml (MIT License); see THIRD-PARTY-NOTICES.md. */
#ifndef NF_QUANT_MMQ_REF_GLSL
#define NF_QUANT_MMQ_REF_GLSL

#if QTYPE == QT_Q4_0
#define MMQ_BS 32
#define MMQ_BB 18
#elif QTYPE == QT_Q4_1
#define MMQ_BS 32
#define MMQ_BB 20
#elif QTYPE == QT_Q5_0
#define MMQ_BS 32
#define MMQ_BB 22
#elif QTYPE == QT_Q5_1
#define MMQ_BS 32
#define MMQ_BB 24
#elif QTYPE == QT_Q8_0
#define MMQ_BS 32
#define MMQ_BB 34
#elif QTYPE == QT_Q2_K
#define MMQ_BS 256
#define MMQ_BB 84
#elif QTYPE == QT_Q3_K
#define MMQ_BS 256
#define MMQ_BB 110
#elif QTYPE == QT_Q4_K
#define MMQ_BS 256
#define MMQ_BB 144
#elif QTYPE == QT_Q5_K
#define MMQ_BS 256
#define MMQ_BB 176
#elif QTYPE == QT_Q6_K
#define MMQ_BS 256
#define MMQ_BB 210
#elif QTYPE == QT_I8_B128
#define MMQ_BS 128
#define MMQ_BB 132
#else
#error "quant_mmq_ref.glsl: QTYPE is not one of the cooperative-matrix2 MMQ weight types"
#endif

/* One weight block as 16-bit words: sizeof(NfMmqRef) == MMQ_BB. */
layout(buffer_reference, std430, buffer_reference_align = 2) buffer NfMmqRef { uint16_t h[MMQ_BB / 2]; };

/* The 16-bit word at even byte offset o. */
uint mmq_ref_w(const in NfMmqRef b, const in uint o) { return uint(b.h[o >> 1u]); }
/* The byte at offset o. */
uint mmq_ref_u8(const in NfMmqRef b, const in uint o) { return (uint(b.h[o >> 1u]) >> ((o & 1u) << 3u)) & 0xFFu; }
/* The fp16 value at even byte offset o. */
float mmq_ref_f16(const in NfMmqRef b, const in uint o) { return unpackHalf2x16(uint(b.h[o >> 1u])).x; }

/* get_scale_min_k4 of Q4_K/Q5_K (quant_blocks.glsl nf_k4_scale_min) over the 12 scale bytes at 4. */
vec2 mmq_ref_k4_scale_min(const in NfMmqRef b, const in uint j) {
    if (j < 4u) {
        return vec2(float(mmq_ref_u8(b, 4u + j) & 63u), float(mmq_ref_u8(b, 8u + j) & 63u));
    }
    const uint q4 = mmq_ref_u8(b, 4u + j + 4u);
    return vec2(float((q4 & 0xFu) | ((mmq_ref_u8(b, j) >> 6u) << 4u)),
                float((q4 >> 4u) | ((mmq_ref_u8(b, 4u + j) >> 6u) << 4u)));
}

#if QTYPE == QT_Q4_0
/* d@0, qs[16]@2: element e < 16 is the low nibble of qs[e], e >= 16 the high nibble of qs[e - 16]. */
float16_t mmq_ref_decode(const in NfMmqRef b, const in uint bc[2], const in uint e2[2]) {
    const uint e = e2[1];
    const uint code = (mmq_ref_u8(b, 2u + (e & 15u)) >> ((e & 16u) >> 2u)) & 0xFu;
    return float16_t((float(code) - 8.0) * mmq_ref_f16(b, 0u));
}
#elif QTYPE == QT_Q4_1
/* d@0, m@2, qs[16]@4. */
float16_t mmq_ref_decode(const in NfMmqRef b, const in uint bc[2], const in uint e2[2]) {
    const uint e = e2[1];
    const uint code = (mmq_ref_u8(b, 4u + (e & 15u)) >> ((e & 16u) >> 2u)) & 0xFu;
    return float16_t(mmq_ref_f16(b, 0u) * float(code) + mmq_ref_f16(b, 2u));
}
#elif QTYPE == QT_Q5_0
/* d@0, qh[4]@2, qs[16]@6. */
float16_t mmq_ref_decode(const in NfMmqRef b, const in uint bc[2], const in uint e2[2]) {
    const uint e = e2[1];
    const uint qh = mmq_ref_w(b, 2u) | (mmq_ref_w(b, 4u) << 16u);
    const uint code = ((mmq_ref_u8(b, 6u + (e & 15u)) >> ((e & 16u) >> 2u)) & 0xFu) | (((qh >> e) & 1u) << 4u);
    return float16_t((float(code) - 16.0) * mmq_ref_f16(b, 0u));
}
#elif QTYPE == QT_Q5_1
/* d@0, m@2, qh[4]@4, qs[16]@8. */
float16_t mmq_ref_decode(const in NfMmqRef b, const in uint bc[2], const in uint e2[2]) {
    const uint e = e2[1];
    const uint qh = mmq_ref_w(b, 4u) | (mmq_ref_w(b, 6u) << 16u);
    const uint code = ((mmq_ref_u8(b, 8u + (e & 15u)) >> ((e & 16u) >> 2u)) & 0xFu) | (((qh >> e) & 1u) << 4u);
    return float16_t(mmq_ref_f16(b, 0u) * float(code) + mmq_ref_f16(b, 2u));
}
#elif QTYPE == QT_Q8_0
/* d@0, qs[32] int8@2. */
float16_t mmq_ref_decode(const in NfMmqRef b, const in uint bc[2], const in uint e2[2]) {
    const uint e = e2[1];
    const int code = bitfieldExtract(int(mmq_ref_w(b, 2u + (e & ~1u))), int((e & 1u) << 3u), 8);
    return float16_t(float(code) * mmq_ref_f16(b, 0u));
}
#elif QTYPE == QT_Q2_K
/* scales[16]@0, qs[64]@16, d@80, dmin@82; 16-element sub-blocks. */
float16_t mmq_ref_decode(const in NfMmqRef b, const in uint bc[2], const in uint e2[2]) {
    const uint e = e2[1];
    const uint q = mmq_ref_u8(b, 16u + ((e >> 7u) << 5u) + (e & 31u));
    const uint code = (q >> (((e >> 5u) & 3u) << 1u)) & 3u;
    const uint sc = mmq_ref_u8(b, e >> 4u);
    return float16_t(mmq_ref_f16(b, 80u) * float(sc & 0xFu) * float(code) - mmq_ref_f16(b, 82u) * float(sc >> 4u));
}
#elif QTYPE == QT_Q3_K
/* hmask[32]@0, qs[64]@32, scales[12]@96, d@108. */
float16_t mmq_ref_decode(const in NfMmqRef b, const in uint bc[2], const in uint e2[2]) {
    const uint e = e2[1];
    const uint hf = e >> 7u, j = (e >> 5u) & 3u, l = e & 31u;
    const uint q = mmq_ref_u8(b, 32u + (hf << 5u) + l);
    const uint hm = mmq_ref_u8(b, l);
    int code = int((q >> (j << 1u)) & 3u);
    code -= ((hm >> ((hf << 2u) + j)) & 1u) != 0u ? 0 : 4;
    const uint si = e >> 4u;
    const uint lo = (mmq_ref_u8(b, 96u + (si & 7u)) >> ((si >> 3u) * 4u)) & 0xFu;
    const uint hi = (mmq_ref_u8(b, 104u + (si & 3u)) >> ((si >> 2u) * 2u)) & 3u;
    return float16_t(mmq_ref_f16(b, 108u) * float(int(lo | (hi << 4u)) - 32) * float(code));
}
#elif QTYPE == QT_Q4_K
/* d@0, dmin@2, scales[12]@4, qs[128]@16; 32-element sub-blocks. */
float16_t mmq_ref_decode(const in NfMmqRef b, const in uint bc[2], const in uint e2[2]) {
    const uint e = e2[1];
    const uint si = e >> 5u;
    const uint code = (mmq_ref_u8(b, 16u + ((si >> 1u) << 5u) + (e & 31u)) >> ((si & 1u) << 2u)) & 0xFu;
    const vec2 sm = mmq_ref_k4_scale_min(b, si);
    return float16_t(mmq_ref_f16(b, 0u) * sm.x * float(code) - mmq_ref_f16(b, 2u) * sm.y);
}
#elif QTYPE == QT_Q5_K
/* d@0, dmin@2, scales[12]@4, qh[32]@16, qs[128]@48. */
float16_t mmq_ref_decode(const in NfMmqRef b, const in uint bc[2], const in uint e2[2]) {
    const uint e = e2[1];
    const uint si = e >> 5u, l = e & 31u;
    const uint lo = (mmq_ref_u8(b, 48u + ((si >> 1u) << 5u) + l) >> ((si & 1u) << 2u)) & 0xFu;
    const uint code = lo | (((mmq_ref_u8(b, 16u + l) >> si) & 1u) << 4u);
    const vec2 sm = mmq_ref_k4_scale_min(b, si);
    return float16_t(mmq_ref_f16(b, 0u) * sm.x * float(code) - mmq_ref_f16(b, 2u) * sm.y);
}
#elif QTYPE == QT_Q6_K
/* ql[128]@0, qh[64]@128, scales[16] int8@192, d@208. */
float16_t mmq_ref_decode(const in NfMmqRef b, const in uint bc[2], const in uint e2[2]) {
    const uint e = e2[1];
    const uint hf = e >> 7u, t = (e >> 5u) & 3u, l = e & 31u;
    const uint lo = mmq_ref_u8(b, (hf << 6u) + ((t & 1u) << 5u) + l) >> ((t >> 1u) << 2u);
    const uint hi = (mmq_ref_u8(b, 128u + (hf << 5u) + l) >> (t << 1u)) & 3u;
    const int code = int((lo & 0xFu) | (hi << 4u)) - 32;
    const int sc = bitfieldExtract(int(mmq_ref_u8(b, 192u + (e >> 4u))), 0, 8);
    return float16_t(mmq_ref_f16(b, 208u) * float(sc) * float(code));
}
#elif QTYPE == QT_I8_B128
/* d (fp32)@0, qs[128] int8@4. */
float16_t mmq_ref_decode(const in NfMmqRef b, const in uint bc[2], const in uint e2[2]) {
    const uint e = e2[1];
    const float d = uintBitsToFloat(mmq_ref_w(b, 0u) | (mmq_ref_w(b, 2u) << 16u));
    const int code = bitfieldExtract(int(mmq_ref_w(b, 4u + (e & ~1u))), int((e & 1u) << 3u), 8);
    return float16_t(float(code) * d);
}
#endif

#endif /* NF_QUANT_MMQ_REF_GLSL */

/* common/quant_mmq_ref.glsl - the MMQ weight types as VK_NV_cooperative_matrix2 decode callbacks
 * (mmq_ref_decode): the tensor load positions a physical-storage-buffer reference at the element's
 * weight block and asks for element e (coordInBlock[1]) of it. The reference type NfMmqRef is one
 * block of 16-bit words, so its sizeof is the block's byte size: the tensor layout advances block
 * references by sizeof, and an unsized or wrongly sized record gives the layout the wrong block
 * stride. Every GGUF block size is even, so all fields are read through aligned 16-bit loads (two for
 * an fp32 scale), as llama.cpp's dequant_funcs_cm2.glsl reads its packed16 blocks. The value formulas
 * equal quant_mmq.glsl / quant_k.glsl; the result is rounded to fp16 once, as the KHR CM form rounds
 * its staged tile.
 *
 * Staged scales (MMQ_REF_STAGED: Q4_K and Q5_K when the shader defines MMQ_REF_SCALE_ROWS): the two types
 * share get_scale_min_k4's 6-bit packing, which costs the per-element decode three byte loads and the
 * unpacking. A shader that walks one K block of a tile of MMQ_REF_SCALE_ROWS rows at a time instead decodes
 * every row's eight (d * scale, dmin * min) pairs into workgroup memory once per block (mmq_ref_stage_fetch
 * into registers, then mmq_ref_stage_store between barriers: llama.cpp's fetch_scales / store_scales), and
 * the callback reads its pair from there by its row within the tile (blockCoords[0] % rows). Both products
 * are exact in fp32 (an fp16 times a 6-bit integer), so the decoded values equal the unstaged formula's; the
 * fetching invocations split a row's sub-blocks into equal parts. Q6_K keeps its per-element scale loads:
 * its sixteen int8 scales are one byte load each with nothing to unpack, so staging them only adds the
 * barriers and the shared-memory traffic.
 * Requires the shader's NF_BUFFER_VIEWS(0, w) (the weight buffer at binding 0, word view w_u32).
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

#if defined(MMQ_REF_SCALE_ROWS) && (QTYPE == QT_Q4_K || QTYPE == QT_Q5_K)
#define MMQ_REF_STAGED 1
/* Row stride of the staged scales: two words of padding keep the sub-block planes off one bank. */
#define MMQ_REF_STAGE_STRIDE (MMQ_REF_SCALE_ROWS + 2u)
#define MMQ_REF_STAGE_SUBS 8u
shared vec2 mmq_ref_stage[MMQ_REF_STAGE_SUBS * MMQ_REF_STAGE_STRIDE];

/* The first 16 bytes of the weight block of row `row`, K block `blk` (d, dmin and the 12 scale bytes, which
 * every part's sub-blocks draw on); every block starts on a 16-byte boundary (MMQ_BB is a multiple of 16). */
uvec4 mmq_ref_stage_fetch(const uint row, const uint blk, const uint blocks_per_row) {
    const uint w0 = (row * blocks_per_row + blk) * (uint(MMQ_BB) / 4u);
    return uvec4(w_u32[w0], w_u32[w0 + 1u], w_u32[w0 + 2u], w_u32[w0 + 3u]);
}

/* Stores sub-blocks part * 8 / parts on of the fetched block head v as the (d * scale, dmin * min) pairs of
 * tile row lrow. */
void mmq_ref_stage_store(const uvec4 v, const uint lrow, const uint part, const uint parts) {
    const vec2 dm = unpackHalf2x16(v.x);
    const uint sc_hi = (v.w & 0x0F0F0F0Fu) | ((v.y & 0xC0C0C0C0u) >> 2u);
    const uint mb_hi = ((v.w & 0xF0F0F0F0u) >> 4u) | ((v.z & 0xC0C0C0C0u) >> 2u);
    const uint count = MMQ_REF_STAGE_SUBS / parts;
    for (uint i = 0u; i < count; i++) {
        const uint j = part * count + i;
        const uint shift = (j & 3u) << 3u;
        const uint sc = ((j < 4u ? v.y : sc_hi) >> shift) & 63u;
        const uint mb = ((j < 4u ? v.z : mb_hi) >> shift) & 63u;
        mmq_ref_stage[j * MMQ_REF_STAGE_STRIDE + lrow] = vec2(dm.x * float(sc), dm.y * float(mb));
    }
}
#endif

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
#ifdef MMQ_REF_STAGED
    const vec2 s = mmq_ref_stage[si * MMQ_REF_STAGE_STRIDE + bc[0] % MMQ_REF_SCALE_ROWS];
    return float16_t(s.x * float(code) - s.y);
#else
    const vec2 sm = mmq_ref_k4_scale_min(b, si);
    return float16_t(mmq_ref_f16(b, 0u) * sm.x * float(code) - mmq_ref_f16(b, 2u) * sm.y);
#endif
}
#elif QTYPE == QT_Q5_K
/* d@0, dmin@2, scales[12]@4, qh[32]@16, qs[128]@48. */
float16_t mmq_ref_decode(const in NfMmqRef b, const in uint bc[2], const in uint e2[2]) {
    const uint e = e2[1];
    const uint si = e >> 5u, l = e & 31u;
    const uint lo = (mmq_ref_u8(b, 48u + ((si >> 1u) << 5u) + l) >> ((si & 1u) << 2u)) & 0xFu;
    const uint code = lo | (((mmq_ref_u8(b, 16u + l) >> si) & 1u) << 4u);
#ifdef MMQ_REF_STAGED
    const vec2 s = mmq_ref_stage[si * MMQ_REF_STAGE_STRIDE + bc[0] % MMQ_REF_SCALE_ROWS];
    return float16_t(s.x * float(code) - s.y);
#else
    const vec2 sm = mmq_ref_k4_scale_min(b, si);
    return float16_t(mmq_ref_f16(b, 0u) * sm.x * float(code) - mmq_ref_f16(b, 2u) * sm.y);
#endif
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

#ifdef NF_CM2_DECODE_VECTOR
/* Four-element decode callbacks (mmq_ref_decode4, GL_NV_cooperative_matrix_decode_vector): the tensor load asks
 * for elements e .. e+3 of a block, e a multiple of four, so one callback reads one 32-bit word of codes (two
 * aligned 16-bit loads) where mmq_ref_decode reads a byte per element; llama.cpp's dequantFunc*_v. The four
 * elements always share their sub-block, so the scale and offset load once. Each component takes the scalar
 * callback's formula in the same operation order, so both forms decode every element to the same fp16 value. */

/* The 32-bit word at even byte offset o. */
uint mmq_ref_u32(const in NfMmqRef b, const in uint o) { return mmq_ref_w(b, o) | (mmq_ref_w(b, o + 2u) << 16u); }
/* The four bytes of word v, unsigned. */
uvec4 mmq_ref_bytes(const uint v) { return uvec4(v, v >> 8u, v >> 16u, v >> 24u) & 0xFFu; }
/* The four bytes of word v as signed int8 values. */
ivec4 mmq_ref_sbytes(const uint v) {
    return ivec4(bitfieldExtract(int(v), 0, 8), bitfieldExtract(int(v), 8, 8), bitfieldExtract(int(v), 16, 8),
                 bitfieldExtract(int(v), 24, 8));
}

#if QTYPE == QT_Q4_0
f16vec4 mmq_ref_decode4(const in NfMmqRef b, const in uint bc[2], const in uint e2[2]) {
    const uint e = e2[1];
    const uvec4 code = mmq_ref_bytes((mmq_ref_u32(b, 2u + (e & 15u)) >> ((e & 16u) >> 2u)) & 0x0F0F0F0Fu);
    return f16vec4((vec4(code) - 8.0) * mmq_ref_f16(b, 0u));
}
#elif QTYPE == QT_Q4_1
f16vec4 mmq_ref_decode4(const in NfMmqRef b, const in uint bc[2], const in uint e2[2]) {
    const uint e = e2[1];
    const uvec4 code = mmq_ref_bytes((mmq_ref_u32(b, 4u + (e & 15u)) >> ((e & 16u) >> 2u)) & 0x0F0F0F0Fu);
    return f16vec4(mmq_ref_f16(b, 0u) * vec4(code) + mmq_ref_f16(b, 2u));
}
#elif QTYPE == QT_Q5_0
f16vec4 mmq_ref_decode4(const in NfMmqRef b, const in uint bc[2], const in uint e2[2]) {
    const uint e = e2[1];
    const uint qh = mmq_ref_u32(b, 2u) >> e;
    const uvec4 code = mmq_ref_bytes((mmq_ref_u32(b, 6u + (e & 15u)) >> ((e & 16u) >> 2u)) & 0x0F0F0F0Fu)
            | ((uvec4(qh, qh >> 1u, qh >> 2u, qh >> 3u) & 1u) << 4u);
    return f16vec4((vec4(code) - 16.0) * mmq_ref_f16(b, 0u));
}
#elif QTYPE == QT_Q5_1
f16vec4 mmq_ref_decode4(const in NfMmqRef b, const in uint bc[2], const in uint e2[2]) {
    const uint e = e2[1];
    const uint qh = mmq_ref_u32(b, 4u) >> e;
    const uvec4 code = mmq_ref_bytes((mmq_ref_u32(b, 8u + (e & 15u)) >> ((e & 16u) >> 2u)) & 0x0F0F0F0Fu)
            | ((uvec4(qh, qh >> 1u, qh >> 2u, qh >> 3u) & 1u) << 4u);
    return f16vec4(mmq_ref_f16(b, 0u) * vec4(code) + mmq_ref_f16(b, 2u));
}
#elif QTYPE == QT_Q8_0
f16vec4 mmq_ref_decode4(const in NfMmqRef b, const in uint bc[2], const in uint e2[2]) {
    return f16vec4(vec4(mmq_ref_sbytes(mmq_ref_u32(b, 2u + e2[1]))) * mmq_ref_f16(b, 0u));
}
#elif QTYPE == QT_Q2_K
f16vec4 mmq_ref_decode4(const in NfMmqRef b, const in uint bc[2], const in uint e2[2]) {
    const uint e = e2[1];
    const uint q = mmq_ref_u32(b, 16u + ((e >> 7u) << 5u) + (e & 31u));
    const uvec4 code = mmq_ref_bytes((q >> (((e >> 5u) & 3u) << 1u)) & 0x03030303u);
    const uint sc = mmq_ref_u8(b, e >> 4u);
    return f16vec4(mmq_ref_f16(b, 80u) * float(sc & 0xFu) * vec4(code) - mmq_ref_f16(b, 82u) * float(sc >> 4u));
}
#elif QTYPE == QT_Q3_K
f16vec4 mmq_ref_decode4(const in NfMmqRef b, const in uint bc[2], const in uint e2[2]) {
    const uint e = e2[1];
    const uint hf = e >> 7u, j = (e >> 5u) & 3u, l = e & 31u;
    const ivec4 low = ivec4(mmq_ref_bytes((mmq_ref_u32(b, 32u + (hf << 5u) + l) >> (j << 1u)) & 0x03030303u));
    const uvec4 high = mmq_ref_bytes((mmq_ref_u32(b, l) >> ((hf << 2u) + j)) & 0x01010101u);
    const ivec4 code = low - ivec4(4) * ivec4(uvec4(1u) - high);
    const uint si = e >> 4u;
    const uint lo = (mmq_ref_u8(b, 96u + (si & 7u)) >> ((si >> 3u) * 4u)) & 0xFu;
    const uint hi = (mmq_ref_u8(b, 104u + (si & 3u)) >> ((si >> 2u) * 2u)) & 3u;
    return f16vec4(mmq_ref_f16(b, 108u) * float(int(lo | (hi << 4u)) - 32) * vec4(code));
}
#elif QTYPE == QT_Q4_K
f16vec4 mmq_ref_decode4(const in NfMmqRef b, const in uint bc[2], const in uint e2[2]) {
    const uint e = e2[1];
    const uint si = e >> 5u;
    const vec4 code = vec4(mmq_ref_bytes((mmq_ref_u32(b, 16u + ((si >> 1u) << 5u) + (e & 31u)) >> ((si & 1u) << 2u))
            & 0x0F0F0F0Fu));
#ifdef MMQ_REF_STAGED
    const vec2 s = mmq_ref_stage[si * MMQ_REF_STAGE_STRIDE + bc[0] % MMQ_REF_SCALE_ROWS];
    return f16vec4(s.x * code - s.y);
#else
    const vec2 sm = mmq_ref_k4_scale_min(b, si);
    return f16vec4(mmq_ref_f16(b, 0u) * sm.x * code - mmq_ref_f16(b, 2u) * sm.y);
#endif
}
#elif QTYPE == QT_Q5_K
f16vec4 mmq_ref_decode4(const in NfMmqRef b, const in uint bc[2], const in uint e2[2]) {
    const uint e = e2[1];
    const uint si = e >> 5u, l = e & 31u;
    const uint lo = (mmq_ref_u32(b, 48u + ((si >> 1u) << 5u) + l) >> ((si & 1u) << 2u)) & 0x0F0F0F0Fu;
    const uint hi = ((mmq_ref_u32(b, 16u + l) >> si) & 0x01010101u) << 4u;
    const vec4 code = vec4(mmq_ref_bytes(lo | hi));
#ifdef MMQ_REF_STAGED
    const vec2 s = mmq_ref_stage[si * MMQ_REF_STAGE_STRIDE + bc[0] % MMQ_REF_SCALE_ROWS];
    return f16vec4(s.x * code - s.y);
#else
    const vec2 sm = mmq_ref_k4_scale_min(b, si);
    return f16vec4(mmq_ref_f16(b, 0u) * sm.x * code - mmq_ref_f16(b, 2u) * sm.y);
#endif
}
#elif QTYPE == QT_Q6_K
f16vec4 mmq_ref_decode4(const in NfMmqRef b, const in uint bc[2], const in uint e2[2]) {
    const uint e = e2[1];
    const uint hf = e >> 7u, t = (e >> 5u) & 3u, l = e & 31u;
    const uint lo = (mmq_ref_u32(b, (hf << 6u) + ((t & 1u) << 5u) + l) >> ((t >> 1u) << 2u)) & 0x0F0F0F0Fu;
    const uint hi = ((mmq_ref_u32(b, 128u + (hf << 5u) + l) >> (t << 1u)) & 0x03030303u) << 4u;
    const ivec4 code = ivec4(mmq_ref_bytes(lo | hi)) - 32;
    const int sc = bitfieldExtract(int(mmq_ref_u8(b, 192u + (e >> 4u))), 0, 8);
    return f16vec4(mmq_ref_f16(b, 208u) * float(sc) * vec4(code));
}
#elif QTYPE == QT_I8_B128
f16vec4 mmq_ref_decode4(const in NfMmqRef b, const in uint bc[2], const in uint e2[2]) {
    const float d = uintBitsToFloat(mmq_ref_u32(b, 0u));
    return f16vec4(vec4(mmq_ref_sbytes(mmq_ref_u32(b, 4u + e2[1]))) * d);
}
#endif
#endif /* NF_CM2_DECODE_VECTOR */

#endif /* NF_QUANT_MMQ_REF_GLSL */

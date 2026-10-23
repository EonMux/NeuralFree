/* common/quant_k.glsl - QT_DECODE_* decoders for Q2_K, Q3_K, Q4_K, Q5_K, Q6_K and Q8_K.
 * GLSL twin of kernels/llm/common/quant_decode_k.h (the semantic reference). Requires: common/quant_blocks.glsl.
 *
 * QT_TPS = 32 invocations cooperate on one 256-element super-block; invocation ltid owns the
 * NF_KQ_EPT = 8 consecutive elements [8 * ltid, 8 * ltid + 8). Per type, nf_kq_w8() dequantizes
 * those 8 elements with the same products in the same order as llama.cpp ggml-quants.c
 * dequantize_row_<t>; qt_dot / qt_dot_dual dot them with the activations read through qt_x4.
 *
 * Weight loads: Q2_K, Q4_K, Q5_K and Q8_K blocks are 84, 144, 176 and 292 bytes, all divisible by 4,
 * so every block of a binding starts 4-aligned and their 4-byte runs use nf_w_u32. Q3_K (110) and
 * Q6_K (210) blocks are only 2-aligned and use nf_w_u32_even.
 */
#ifndef NF_QUANT_K_GLSL
#define NF_QUANT_K_GLSL

#ifndef NF_QUANT_BLOCKS_GLSL
#error "common/quant_k.glsl requires common/quant_blocks.glsl"
#endif

#ifdef QTYPE
#if QTYPE == QT_Q2_K || QTYPE == QT_Q3_K || QTYPE == QT_Q4_K || QTYPE == QT_Q5_K || QTYPE == QT_Q6_K || QTYPE == QT_Q8_K

#define QT_BS 256
#define QT_TPS 32
#define QT_DECODE_DOT 1
#define QT_DECODE_DUAL 1
#define QT_DECODE_ELEM 1

const uint NF_KQ_EPT = 8u;    /* elements per invocation: QT_BS / QT_TPS */

/* (byte k of v >> sh) & mask for k = 0..3, as floats; sh + 24 < 32 and mask fits in 8 - sh bits. */
vec4 nf_kq_bits4(uint v, uint sh, uint mask) {
    return vec4(uvec4(v >> sh, v >> (sh + 8u), v >> (sh + 16u), v >> (sh + 24u)) & mask);
}

#if QTYPE == QT_Q2_K
/* ---- Q2_K (llama.cpp ggml-quants.c:961-989): scales[16] @0, qs[64] @16, d @80, dmin @82 ----
 * Element e = 128*hf + 32*j + 16*h + l: q = (qs[32*hf + 16*h + l] >> 2j) & 3 with scale byte
 * 8*hf + 2*j + h; y = d*(sc & 15) * q - dmin*(sc >> 4). Invocation ltid's scale byte is ltid/2. */
#define QT_BB NF_Q2_K_BB

void nf_kq_w8(uint ws, uint bp, uint ltid, out vec4 w0, out vec4 w1) {
    const uint sc = nf_w_u8(ws, bp + (ltid >> 1));
    const vec2 dm = unpackHalf2x16(nf_w_u32(ws, bp + 80u));
    const float dl = dm.x * float(sc & 0xFu);
    const float ml = dm.y * float(sc >> 4);
    const uint shift = ((ltid & 15u) >> 2) * 2u;
    const uint q = bp + 16u + (ltid >> 4) * 32u + ((ltid >> 1) & 1u) * 16u + (ltid & 1u) * 8u;
    w0 = dl * nf_kq_bits4(nf_w_u32(ws, q), shift, 3u) - ml;
    w1 = dl * nf_kq_bits4(nf_w_u32(ws, q + 4u), shift, 3u) - ml;
}

float qt_elem(uint ws, uint rbase, uint i) {
    const uint bp = rbase + (i / uint(QT_BS)) * uint(QT_BB);
    const uint e = i % uint(QT_BS);
    const uint hf = e >> 7, j = (e >> 5) & 3u, h = (e >> 4) & 1u, l = e & 15u;
    const uint sc = nf_w_u8(ws, bp + hf * 8u + j * 2u + h);
    const float dl = nf_w_f16(ws, bp + 80u) * float(sc & 0xFu);
    const float ml = nf_w_f16(ws, bp + 82u) * float(sc >> 4);
    const uint q = (nf_w_u8(ws, bp + 16u + hf * 32u + h * 16u + l) >> (2u * j)) & 3u;
    return dl * float(q) - ml;
}

#elif QTYPE == QT_Q3_K
/* ---- Q3_K (llama.cpp ggml-quants.c:1306-1352): hmask[32] @0, qs[64] @32, scales[12] @96, d @108 ----
 * Q2_K plane layout; the high bit of element e = 128*hf + 32*j + 16*h + l is bit 4*hf + j of
 * hmask[16*h + l] and a clear bit subtracts 4. Scale si is the signed 6-bit aux[] value: low
 * nibble from scales[si % 8] (high nibble when si >= 8), top bits from scales[8 + si % 4] >> 2*(si/4),
 * minus 32. */
#define QT_BB NF_Q3_K_BB

int nf_q3k_scale(uint ws, uint bp, uint si) {
    const uint lo = (nf_w_u8(ws, bp + 96u + (si & 7u)) >> ((si >> 3) * 4u)) & 0xFu;
    const uint hi = (nf_w_u8(ws, bp + 104u + (si & 3u)) >> ((si >> 2) * 2u)) & 3u;
    return int(lo | (hi << 4)) - 32;
}

void nf_kq_w8(uint ws, uint bp, uint ltid, out vec4 w0, out vec4 w1) {
    const float dl = nf_w_f16(ws, bp + 108u) * float(nf_q3k_scale(ws, bp, ltid >> 1));
    const uint hf = ltid >> 4, j = (ltid & 15u) >> 2;
    const uint shift = j * 2u, mbit = hf * 4u + j;
    const uint q = bp + 32u + hf * 32u + ((ltid >> 1) & 1u) * 16u + (ltid & 1u) * 8u;
    const uint hm = bp + (ltid & 3u) * 8u;
    w0 = dl * (nf_kq_bits4(nf_w_u32_even(ws, q), shift, 3u)
               + 4.0 * nf_kq_bits4(nf_w_u32_even(ws, hm), mbit, 1u) - 4.0);
    w1 = dl * (nf_kq_bits4(nf_w_u32_even(ws, q + 4u), shift, 3u)
               + 4.0 * nf_kq_bits4(nf_w_u32_even(ws, hm + 4u), mbit, 1u) - 4.0);
}

float qt_elem(uint ws, uint rbase, uint i) {
    const uint bp = rbase + (i / uint(QT_BS)) * uint(QT_BB);
    const uint e = i % uint(QT_BS);
    const uint hf = e >> 7, j = (e >> 5) & 3u, h = (e >> 4) & 1u, l = e & 15u;
    const float dl = nf_w_f16(ws, bp + 108u) * float(nf_q3k_scale(ws, bp, hf * 8u + j * 2u + h));
    int q = int((nf_w_u8(ws, bp + 32u + hf * 32u + h * 16u + l) >> (2u * j)) & 3u);
    q -= ((nf_w_u8(ws, bp + h * 16u + l) >> (hf * 4u + j)) & 1u) != 0u ? 0 : 4;
    return dl * float(q);
}

#elif QTYPE == QT_Q4_K
/* ---- Q4_K (llama.cpp ggml-quants.c:1530-1550): d @0, dmin @2, scales[12] @4, qs[128] @16 ----
 * 32-element group si = e / 32 uses scale/min pair si (nf_k4_scale_min) and the low (si even) or
 * high (si odd) nibbles of qs[32 * (si / 2) + e % 32]: y = d*sc * q - dmin*m. */
#define QT_BB NF_Q4_K_BB

void nf_kq_w8(uint ws, uint bp, uint ltid, out vec4 w0, out vec4 w1) {
    const uint si = ltid >> 2;
    const vec2 sm = nf_k4_scale_min(ws, bp + 4u, si);
    const vec2 dm = unpackHalf2x16(nf_w_u32(ws, bp));
    const float dl = dm.x * sm.x;
    const float ml = dm.y * sm.y;
    const uint sh = (si & 1u) * 4u;
    const uint q = bp + 16u + (si >> 1) * 32u + (ltid & 3u) * 8u;
    w0 = dl * nf_kq_bits4(nf_w_u32(ws, q), sh, 0xFu) - ml;
    w1 = dl * nf_kq_bits4(nf_w_u32(ws, q + 4u), sh, 0xFu) - ml;
}

float qt_elem(uint ws, uint rbase, uint i) {
    const uint bp = rbase + (i / uint(QT_BS)) * uint(QT_BB);
    const uint e = i % uint(QT_BS);
    const uint si = e >> 5, l = e & 31u;
    const vec2 sm = nf_k4_scale_min(ws, bp + 4u, si);
    const uint q = (nf_w_u8(ws, bp + 16u + (si >> 1) * 32u + l) >> ((si & 1u) * 4u)) & 0xFu;
    return nf_w_f16(ws, bp) * sm.x * float(q) - nf_w_f16(ws, bp + 2u) * sm.y;
}

#elif QTYPE == QT_Q5_K
/* ---- Q5_K (llama.cpp ggml-quants.c:1732-1756): d @0, dmin @2, scales[12] @4, qh[32] @16, qs[128] @48 ----
 * Q4_K plus a fifth bit: element e of group si = e / 32 adds 16 when bit si of qh[e % 32] is set. */
#define QT_BB NF_Q5_K_BB

void nf_kq_w8(uint ws, uint bp, uint ltid, out vec4 w0, out vec4 w1) {
    const uint si = ltid >> 2;
    const vec2 sm = nf_k4_scale_min(ws, bp + 4u, si);
    const vec2 dm = unpackHalf2x16(nf_w_u32(ws, bp));
    const float dl = dm.x * sm.x;
    const float ml = dm.y * sm.y;
    const uint sh = (si & 1u) * 4u;
    const uint q = bp + 48u + (si >> 1) * 32u + (ltid & 3u) * 8u;
    const uint qh = bp + 16u + (ltid & 3u) * 8u;
    w0 = dl * (nf_kq_bits4(nf_w_u32(ws, q), sh, 0xFu)
               + 16.0 * nf_kq_bits4(nf_w_u32(ws, qh), si, 1u)) - ml;
    w1 = dl * (nf_kq_bits4(nf_w_u32(ws, q + 4u), sh, 0xFu)
               + 16.0 * nf_kq_bits4(nf_w_u32(ws, qh + 4u), si, 1u)) - ml;
}

float qt_elem(uint ws, uint rbase, uint i) {
    const uint bp = rbase + (i / uint(QT_BS)) * uint(QT_BB);
    const uint e = i % uint(QT_BS);
    const uint si = e >> 5, l = e & 31u;
    const vec2 sm = nf_k4_scale_min(ws, bp + 4u, si);
    const uint q = ((nf_w_u8(ws, bp + 48u + (si >> 1) * 32u + l) >> ((si & 1u) * 4u)) & 0xFu)
                 + (((nf_w_u8(ws, bp + 16u + l) >> si) & 1u) << 4);
    return nf_w_f16(ws, bp) * sm.x * float(q) - nf_w_f16(ws, bp + 2u) * sm.y;
}

#elif QTYPE == QT_Q6_K
/* ---- Q6_K (llama.cpp ggml-quants.c:1940-1968): ql[128] @0, qh[64] @128, scales[16] (i8) @192, d @208 ----
 * Element e = 128*hf + 32*t + l is ((ql[64*hf + 32*(t%2) + l] >> 4*(t/2)) & 15)
 * | (((qh[32*hf + l] >> 2t) & 3) << 4), minus 32, times d * scales[8*hf + 2*t + l/16].
 * Invocation ltid covers hf = ltid/16, t = (ltid/4)%4, l = 8*(ltid%4) + 0..7. */
#define QT_BB NF_Q6_K_BB

void nf_kq_w8(uint ws, uint bp, uint ltid, out vec4 w0, out vec4 w1) {
    const uint hf = ltid >> 4, t = (ltid >> 2) & 3u, l0 = (ltid & 3u) * 8u;
    const float dl = nf_w_f16(ws, bp + 208u) * float(nf_w_i8(ws, bp + 192u + hf * 8u + t * 2u + (l0 >> 4)));
    const uint qlsh = (t >> 1) * 4u, qhsh = t * 2u;
    const uint ql = bp + hf * 64u + (t & 1u) * 32u + l0;
    const uint qh = bp + 128u + hf * 32u + l0;
    w0 = dl * (nf_kq_bits4(nf_w_u32_even(ws, ql), qlsh, 0xFu)
               + 16.0 * nf_kq_bits4(nf_w_u32_even(ws, qh), qhsh, 3u) - 32.0);
    w1 = dl * (nf_kq_bits4(nf_w_u32_even(ws, ql + 4u), qlsh, 0xFu)
               + 16.0 * nf_kq_bits4(nf_w_u32_even(ws, qh + 4u), qhsh, 3u) - 32.0);
}

float qt_elem(uint ws, uint rbase, uint i) {
    const uint bp = rbase + (i / uint(QT_BS)) * uint(QT_BB);
    const uint e = i % uint(QT_BS);
    const uint hf = e >> 7, t = (e >> 5) & 3u, l = e & 31u;
    const int sc = nf_w_i8(ws, bp + 192u + hf * 8u + t * 2u + (l >> 4));
    const uint lo = (nf_w_u8(ws, bp + hf * 64u + (t & 1u) * 32u + l) >> ((t >> 1) * 4u)) & 0xFu;
    const uint hi = (nf_w_u8(ws, bp + 128u + hf * 32u + l) >> (t * 2u)) & 3u;
    return nf_w_f16(ws, bp + 208u) * float(sc) * float(int(lo | (hi << 4)) - 32);
}

#else
/* ---- Q8_K (llama.cpp ggml-quants.c:2808-2816): d (f32) @0, qs[256] (i8) @4, bsums follow ----
 * y = d * qs[e]. */
#define QT_BB NF_Q8_K_BB

void nf_kq_w8(uint ws, uint bp, uint ltid, out vec4 w0, out vec4 w1) {
    const float d = nf_w_f32(ws, bp);
    const uint q = bp + 4u + ltid * NF_KQ_EPT;
    const int v0 = int(nf_w_u32(ws, q));
    const int v1 = int(nf_w_u32(ws, q + 4u));
    w0 = d * vec4(bitfieldExtract(v0, 0, 8), bitfieldExtract(v0, 8, 8),
                  bitfieldExtract(v0, 16, 8), bitfieldExtract(v0, 24, 8));
    w1 = d * vec4(bitfieldExtract(v1, 0, 8), bitfieldExtract(v1, 8, 8),
                  bitfieldExtract(v1, 16, 8), bitfieldExtract(v1, 24, 8));
}

float qt_elem(uint ws, uint rbase, uint i) {
    const uint bp = rbase + (i / uint(QT_BS)) * uint(QT_BB);
    return nf_w_f32(ws, bp) * float(nf_w_i8(ws, bp + 4u + i % uint(QT_BS)));
}

#endif /* K-quant type */

float qt_dot(uint ws, uint bp, uint xoff, uint blk, uint ltid) {
    vec4 w0, w1;
    nf_kq_w8(ws, bp, ltid, w0, w1);
    const uint x = xoff + blk * uint(QT_BS) + ltid * NF_KQ_EPT;
    return dot(w0, qt_x4(x)) + dot(w1, qt_x4(x + 4u));
}

void qt_dot_dual(uint gbp, uint ubp, uint xoff, uint blk, uint ltid, inout float gs, inout float us) {
    vec4 g0, g1, u0, u1;
    nf_kq_w8(NF_WS_A, gbp, ltid, g0, g1);
    nf_kq_w8(NF_WS_B, ubp, ltid, u0, u1);
    const uint x = xoff + blk * uint(QT_BS) + ltid * NF_KQ_EPT;
    const vec4 x0 = qt_x4(x);
    const vec4 x1 = qt_x4(x + 4u);
    gs += dot(g0, x0) + dot(g1, x1);
    us += dot(u0, x0) + dot(u1, x1);
}

#endif /* K-quant QTYPE */
#endif /* QTYPE */

#endif /* NF_QUANT_K_GLSL */

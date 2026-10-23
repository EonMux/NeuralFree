/* common/quant_new.glsl - QT_DECODE_* decoders for TQ1_0, TQ2_0, NVFP4, Q1_0 and Q2_0.
 * GLSL twin of kernels/llm/common/quant_decode_new.h (the semantic reference). Requires: common/quant_blocks.glsl.
 *
 * Decoder interface and include contract: common/quant_blocks.glsl. Weights are read only through
 * nf_w_*(ws, off) and activations only through qt_x/qt_x4.
 *
 * Work split (equal to the OpenCL triples): every invocation of a block owns NF_NQ_EPT = 8 elements,
 * in two runs of 4 that start at block elements nf_nq_xo(ltid) and nf_nq_xo(ltid) + NF_NQ_X1:
 *   TQ1_0, TQ2_0  QT_TPS = 32   elements [8*ltid, 8*ltid + 8)
 *   Q1_0          QT_TPS = 16   elements [8*ltid, 8*ltid + 8)  (qs byte ltid)
 *   Q2_0          QT_TPS = 8    elements [8*ltid, 8*ltid + 8)  (qs bytes 2*ltid, 2*ltid + 1)
 *   NVFP4         QT_TPS = 8    sub-block s = ltid/2, qs bytes 8*s + j0 .. +3 (j0 = 4*(ltid%2)):
 *                               elements 16*s + j0 + k (low nibbles), 16*s + j0 + k + 8 (high)
 * Per type, nf_nq_w8() yields the unscaled quant values of both runs and nf_nq_scale() the scale
 * shared by all 8 elements; qt_dot is scale * (dot(w0, run 0) + dot(w1, run 1)).
 *
 * Alignment: block offsets (bp, rbase) are multiples of QT_BB from the start of a 4-byte aligned
 * binding. NVFP4 blocks (36 bytes) are 4-aligned and use nf_w_u32; TQ1_0 (54), TQ2_0 (66), Q1_0 and
 * Q2_0 (18) blocks are 2-aligned and use nf_w_u32_even / nf_w_u16 / nf_w_u8.
 */
#ifndef NF_QUANT_NEW_GLSL
#define NF_QUANT_NEW_GLSL

#ifndef NF_QUANT_BLOCKS_GLSL
#error "common/quant_new.glsl requires common/quant_blocks.glsl"
#endif

#ifdef QTYPE
#if QTYPE == QT_TQ1_0 || QTYPE == QT_TQ2_0 || QTYPE == QT_NVFP4 || QTYPE == QT_Q1_0 || QTYPE == QT_Q2_0

#define QT_DECODE_DOT  1
#define QT_DECODE_DUAL 1
#define QT_DECODE_ELEM 1

const uint NF_NQ_EPT = 8u;    /* elements per invocation */

/* (byte k of v >> sh) & mask for k = 0..3, as floats; sh + 24 < 32 and mask fits in 8 - sh bits. */
vec4 nf_nq_bits4(uint v, uint sh, uint mask) {
    return vec4(uvec4(v >> sh, v >> (sh + 8u), v >> (sh + 16u), v >> (sh + 24u)) & mask);
}

/* Element offset of the first run for the consecutive 8-element split. */
uint nf_nq_xo8(uint ltid) {
    return ltid * NF_NQ_EPT;
}

#if QTYPE == QT_TQ1_0
/* ---- TQ1_0 (llama.cpp ggml-quants.c:2429-2466): qs[48] @0, qh[4] @48, d @52 ----
 * Element e is trit n of one byte, y = (trit - 1) * d, with trit = ((u8)(b * 3^n) * 3) >> 8:
 *   e <  160: byte qs[e % 32],              n = e / 32
 *   e <  240: byte qs[32 + (e - 160) % 16], n = (e - 160) / 16
 *   e >= 240: byte qh[(e - 240) % 4],       n = (e - 240) / 4 */
#define QT_BS  NF_TQ1_0_BS
#define QT_BB  NF_TQ1_0_BB
#define QT_TPS 32
const uint NF_NQ_X1 = 4u;
const uint NF_TQ1_POW3[5] = uint[5](1u, 3u, 9u, 27u, 81u);

/* trit - 1 of byte b at digit power pw. */
float nf_tq1_trit(uint b, uint pw) {
    const uint q = (b * pw) & 0xFFu;
    return float((q * 3u) >> 8) - 1.0;
}

/* nf_tq1_trit of the 4 bytes of v (byte k = component k). */
vec4 nf_tq1_trits4(uint v, uint pw) {
    return vec4(nf_tq1_trit(v & 0xFFu, pw), nf_tq1_trit((v >> 8) & 0xFFu, pw),
                nf_tq1_trit((v >> 16) & 0xFFu, pw), nf_tq1_trit(v >> 24, pw));
}

/* Invocation ltid < 20 reads qs[8*(ltid%4) ..] with n = ltid/4; ltid in [20, 30) reads
 * qs[32 + 8*((ltid-20)%2) ..] with n = (ltid-20)/2; ltid 30, 31 read qh[0..3] twice, with
 * n = 2*(ltid-30) and n + 1. */
void nf_nq_w8(uint ws, uint bp, uint ltid, out vec4 w0, out vec4 w1) {
    uint b0, b1, n0, n1;
    if (ltid < 20u) {
        b0 = 8u * (ltid & 3u);
        b1 = b0 + 4u;
        n0 = ltid >> 2;
        n1 = n0;
    } else if (ltid < 30u) {
        b0 = 32u + 8u * ((ltid - 20u) & 1u);
        b1 = b0 + 4u;
        n0 = (ltid - 20u) >> 1;
        n1 = n0;
    } else {
        b0 = 48u;
        b1 = 48u;
        n0 = 2u * (ltid - 30u);
        n1 = n0 + 1u;
    }
    w0 = nf_tq1_trits4(nf_w_u32_even(ws, bp + b0), NF_TQ1_POW3[n0]);
    w1 = nf_tq1_trits4(nf_w_u32_even(ws, bp + b1), NF_TQ1_POW3[n1]);
}

float nf_nq_scale(uint ws, uint bp, uint ltid) {
    return nf_w_f16(ws, bp + 52u);
}

uint nf_nq_xo(uint ltid) {
    return nf_nq_xo8(ltid);
}

float qt_elem(uint ws, uint rbase, uint i) {
    const uint bp = rbase + (i / uint(QT_BS)) * uint(QT_BB);
    const uint e = i % uint(QT_BS);
    uint off, n;
    if (e < 160u) {
        off = e & 31u;
        n = e >> 5;
    } else if (e < 240u) {
        off = 32u + ((e - 160u) & 15u);
        n = (e - 160u) >> 4;
    } else {
        off = 48u + ((e - 240u) & 3u);
        n = (e - 240u) >> 2;
    }
    return nf_tq1_trit(nf_w_u8(ws, bp + off), NF_TQ1_POW3[n]) * nf_w_f16(ws, bp + 52u);
}

#elif QTYPE == QT_TQ2_0
/* ---- TQ2_0 (llama.cpp ggml-quants.c:2468-2485): qs[64] @0, d @64 ----
 * Element e = 128*hf + 32*l + m is ((qs[32*hf + m] >> 2l) & 3) - 1, times d. Invocation ltid covers
 * hf = ltid/16, l = (ltid/4)%4, m = 8*(ltid%4) + 0..7. */
#define QT_BS  NF_TQ2_0_BS
#define QT_BB  NF_TQ2_0_BB
#define QT_TPS 32
const uint NF_NQ_X1 = 4u;

void nf_nq_w8(uint ws, uint bp, uint ltid, out vec4 w0, out vec4 w1) {
    const uint sh = ((ltid >> 2) & 3u) * 2u;
    const uint q = bp + (ltid >> 4) * 32u + (ltid & 3u) * 8u;
    w0 = nf_nq_bits4(nf_w_u32_even(ws, q), sh, 3u) - 1.0;
    w1 = nf_nq_bits4(nf_w_u32_even(ws, q + 4u), sh, 3u) - 1.0;
}

float nf_nq_scale(uint ws, uint bp, uint ltid) {
    return nf_w_f16(ws, bp + 64u);
}

uint nf_nq_xo(uint ltid) {
    return nf_nq_xo8(ltid);
}

float qt_elem(uint ws, uint rbase, uint i) {
    const uint bp = rbase + (i / uint(QT_BS)) * uint(QT_BB);
    const uint e = i % uint(QT_BS);
    const uint hf = e >> 7, l = (e >> 5) & 3u, m = e & 31u;
    const uint q = (nf_w_u8(ws, bp + hf * 32u + m) >> (2u * l)) & 3u;
    return (float(q) - 1.0) * nf_w_f16(ws, bp + 64u);
}

#elif QTYPE == QT_NVFP4
/* ---- NVFP4 (llama.cpp ggml-quants.c:589-611): d[4] (UE4M3) @0, qs[32] @4 ----
 * Sub-block s (16 elements) has scale ue4m3(d[s]); its element j < 8 is kvalues_fp4[qs[8s + j] & 15]
 * and element j + 8 is kvalues_fp4[qs[8s + j] >> 4]. */
#define QT_BS  NF_NVFP4_BS
#define QT_BB  NF_NVFP4_BB
#define QT_TPS 8
const uint NF_NQ_X1 = 8u;

/* UE4M3 scale byte to float, halved for the doubled E2M1 codebook (llama.cpp ggml-impl.h:514
 * ggml_ue4m3_to_fp32): 0 and 0x7F give 0; otherwise (e ? 8 + m : m) * 2^(max(e,1) - 11), exact. */
float nf_nq_ue4m3_half(uint x) {
    if (x == 0u || x == 0x7Fu) {
        return 0.0;
    }
    const uint e = (x >> 3) & 0xFu;
    const uint m = x & 7u;
    const float mant = float(e == 0u ? m : 8u + m);
    const uint ex = e == 0u ? 1u : e;
    return mant * uintBitsToFloat((ex + 116u) << 23);
}

vec4 nf_nq_fp4x4(uvec4 n) {
    return vec4(nf_kvalue_fp4(n.x), nf_kvalue_fp4(n.y), nf_kvalue_fp4(n.z), nf_kvalue_fp4(n.w));
}

void nf_nq_w8(uint ws, uint bp, uint ltid, out vec4 w0, out vec4 w1) {
    const uint q = nf_w_u32(ws, bp + 4u + (ltid >> 1) * 8u + (ltid & 1u) * 4u);
    w0 = nf_nq_fp4x4(uvec4(q, q >> 8, q >> 16, q >> 24) & 0xFu);
    w1 = nf_nq_fp4x4(uvec4(q >> 4, q >> 12, q >> 20, q >> 28) & 0xFu);
}

float nf_nq_scale(uint ws, uint bp, uint ltid) {
    return nf_nq_ue4m3_half(nf_w_u8(ws, bp + (ltid >> 1)));
}

uint nf_nq_xo(uint ltid) {
    return (ltid >> 1) * 16u + (ltid & 1u) * 4u;
}

float qt_elem(uint ws, uint rbase, uint i) {
    const uint bp = rbase + (i / uint(QT_BS)) * uint(QT_BB);
    const uint e = i % uint(QT_BS);
    const uint s = e >> 4, t = e & 15u;
    const uint q = nf_w_u8(ws, bp + 4u + s * 8u + (t & 7u));
    const uint nib = t < 8u ? (q & 0xFu) : (q >> 4);
    return float(nf_kvalue_fp4(nib)) * nf_nq_ue4m3_half(nf_w_u8(ws, bp + s));
}

#elif QTYPE == QT_Q1_0
/* ---- Q1_0 (llama.cpp ggml-quants.c:419-437): d @0, qs[16] @2 ----
 * Element j is +d when bit j%8 of qs[j/8] is set and -d otherwise. Invocation ltid owns qs[ltid]. */
#define QT_BS  NF_Q1_0_BS
#define QT_BB  NF_Q1_0_BB
#define QT_TPS 16
const uint NF_NQ_X1 = 4u;

void nf_nq_w8(uint ws, uint bp, uint ltid, out vec4 w0, out vec4 w1) {
    const uint b = nf_w_u8(ws, bp + 2u + ltid);
    w0 = 2.0 * vec4(uvec4(b, b >> 1, b >> 2, b >> 3) & 1u) - 1.0;
    w1 = 2.0 * vec4(uvec4(b >> 4, b >> 5, b >> 6, b >> 7) & 1u) - 1.0;
}

float nf_nq_scale(uint ws, uint bp, uint ltid) {
    return nf_w_f16(ws, bp);
}

uint nf_nq_xo(uint ltid) {
    return nf_nq_xo8(ltid);
}

float qt_elem(uint ws, uint rbase, uint i) {
    const uint bp = rbase + (i / uint(QT_BS)) * uint(QT_BB);
    const uint e = i % uint(QT_BS);
    const float d = nf_w_f16(ws, bp);
    return ((nf_w_u8(ws, bp + 2u + (e >> 3)) >> (e & 7u)) & 1u) != 0u ? d : -d;
}

#else /* QT_Q2_0 */
/* ---- Q2_0 (llama.cpp ggml-quants.c:439-457): d @0, qs[16] @2 ----
 * Element j is (((qs[j/4] >> 2*(j%4)) & 3) - 1) * d. Invocation ltid owns qs[2*ltid] (first run)
 * and qs[2*ltid + 1] (second run), read as one 16-bit value. */
#define QT_BS  NF_Q2_0_BS
#define QT_BB  NF_Q2_0_BB
#define QT_TPS 8
const uint NF_NQ_X1 = 4u;

void nf_nq_w8(uint ws, uint bp, uint ltid, out vec4 w0, out vec4 w1) {
    const uint v = nf_w_u16(ws, bp + 2u + 2u * ltid);
    w0 = vec4(uvec4(v, v >> 2, v >> 4, v >> 6) & 3u) - 1.0;
    w1 = vec4(uvec4(v >> 8, v >> 10, v >> 12, v >> 14) & 3u) - 1.0;
}

float nf_nq_scale(uint ws, uint bp, uint ltid) {
    return nf_w_f16(ws, bp);
}

uint nf_nq_xo(uint ltid) {
    return nf_nq_xo8(ltid);
}

float qt_elem(uint ws, uint rbase, uint i) {
    const uint bp = rbase + (i / uint(QT_BS)) * uint(QT_BB);
    const uint e = i % uint(QT_BS);
    const uint q = (nf_w_u8(ws, bp + 2u + (e >> 2)) >> (2u * (e & 3u))) & 3u;
    return (float(q) - 1.0) * nf_w_f16(ws, bp);
}

#endif /* per-type decoders */

float qt_dot(uint ws, uint bp, uint xoff, uint blk, uint ltid) {
    vec4 w0, w1;
    nf_nq_w8(ws, bp, ltid, w0, w1);
    const uint x = xoff + blk * uint(QT_BS) + nf_nq_xo(ltid);
    return nf_nq_scale(ws, bp, ltid) * (dot(w0, qt_x4(x)) + dot(w1, qt_x4(x + NF_NQ_X1)));
}

void qt_dot_dual(uint gbp, uint ubp, uint xoff, uint blk, uint ltid, inout float gs, inout float us) {
    vec4 g0, g1, u0, u1;
    nf_nq_w8(NF_WS_A, gbp, ltid, g0, g1);
    nf_nq_w8(NF_WS_B, ubp, ltid, u0, u1);
    const uint x = xoff + blk * uint(QT_BS) + nf_nq_xo(ltid);
    const vec4 x0 = qt_x4(x);
    const vec4 x1 = qt_x4(x + NF_NQ_X1);
    gs += nf_nq_scale(NF_WS_A, gbp, ltid) * (dot(g0, x0) + dot(g1, x1));
    us += nf_nq_scale(NF_WS_B, ubp, ltid) * (dot(u0, x0) + dot(u1, x1));
}

#endif /* new-quant QTYPE */
#endif /* QTYPE */

#endif /* NF_QUANT_NEW_GLSL */

/* common/quant_legacy.glsl - QT_DECODE_* decoders for Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, IQ4_NL, MXFP4
 * and IQ4_XS. GLSL twin of kernels/llm/common/quant_decode_legacy.h. Requires: common/quant_blocks.glsl.
 *
 * Decoder interface and include contract: common/quant_blocks.glsl. Weights are read only through
 * nf_w_*(ws, off) and activations only through qt_x/qt_x4.
 *
 * Element order is the ggml SPLIT layout (llama.cpp ggml-quants.c:458-588, 2726-2762): inside a 32-element
 * (sub-)block, qs byte k holds element k in its low nibble and element k + 16 in its high nibble;
 * the fifth bit of Q5_0/Q5_1 element e is bit e of the little-endian qh word.
 *
 * Work split (equal to the OpenCL triples): Q4_0, Q4_1, Q5_0, Q5_1, IQ4_NL and MXFP4 use
 * QT_TPS = 4, invocation ltid owning qs bytes 4*ltid .. 4*ltid+3 (elements k and k + 16); Q8_0 uses
 * QT_TPS = 4 with 8 consecutive elements per invocation; IQ4_XS uses QT_TPS = 32, invocation ltid
 * owning bytes 4*(ltid%4) .. +3 of sub-block ltid/4.
 *
 * Alignment: block offsets (bp, rbase) are multiples of QT_BB from the start of a 4-byte aligned
 * binding. 32-bit reads therefore use nf_w_u32 when QT_BB % 4 == 0 (Q4_1, Q5_1, IQ4_XS),
 * nf_w_u32_even for even QT_BB (Q4_0, Q5_0, IQ4_NL, Q8_0) and byte reads for MXFP4 (17 bytes).
 */
#ifndef NF_QUANT_LEGACY_GLSL
#define NF_QUANT_LEGACY_GLSL

#ifndef NF_QUANT_BLOCKS_GLSL
#error "common/quant_legacy.glsl requires common/quant_blocks.glsl"
#endif

#ifdef QTYPE
#if QTYPE == QT_Q4_0 || QTYPE == QT_Q4_1 || QTYPE == QT_Q5_0 || QTYPE == QT_Q5_1 \
 || QTYPE == QT_IQ4_NL || QTYPE == QT_Q8_0 || QTYPE == QT_MXFP4 || QTYPE == QT_IQ4_XS

/* Low / high nibbles of the four bytes of q; component k comes from byte k. */
uvec4 nf_ql_nib_lo(uint q) {
    return uvec4(q, q >> 8, q >> 16, q >> 24) & 0xFu;
}

uvec4 nf_ql_nib_hi(uint q) {
    return uvec4(q >> 4, q >> 12, q >> 20, q >> 28) & 0xFu;
}

/* Fifth bits (as 0x10) of elements e0 .. e0+3: bits e0 .. e0+3 of qh (ggml xh_0 / xh_1). */
uvec4 nf_ql_q5_hbits(uint qh, uint e0) {
    return ((uvec4(qh) >> (uvec4(0u, 1u, 2u, 3u) + e0)) & 1u) << 4;
}

/* Fifth bit (as 0x10) of element e. */
uint nf_ql_q5_hbit(uint qh, uint e) {
    return ((qh >> e) & 1u) << 4;
}

/* Nibble of element e (0..31) of a split-layout 32-element block whose qs bytes start at qs. */
uint nf_ql_nibble(uint ws, uint qs, uint e) {
    const uint q = nf_w_u8(ws, qs + (e & 15u));
    return e < 16u ? (q & 0xFu) : (q >> 4);
}

/* kvalues_iq4nl / kvalues_mxfp4 of four nibbles. */
vec4 nf_ql_kv_iq4nl(uvec4 n) {
    return vec4(nf_kvalue_iq4nl(n.x), nf_kvalue_iq4nl(n.y), nf_kvalue_iq4nl(n.z), nf_kvalue_iq4nl(n.w));
}

vec4 nf_ql_kv_fp4(uvec4 n) {
    return vec4(nf_kvalue_fp4(n.x), nf_kvalue_fp4(n.y), nf_kvalue_fp4(n.z), nf_kvalue_fp4(n.w));
}

#endif /* legacy QTYPE */

/* ================================ Q4_0 ================================
 * 18 B: d(f16)@0, qs[16]@2. y[k] = ((qs[k]&15) - 8)*d, y[k+16] = ((qs[k]>>4) - 8)*d. */
#if QTYPE == QT_Q4_0
#define QT_BS  NF_Q4_0_BS
#define QT_BB  NF_Q4_0_BB
#define QT_TPS 4
#define QT_DECODE_DOT  1
#define QT_DECODE_DUAL 1
#define QT_DECODE_ELEM 1

float nf_ql_part(uint ws, uint bp, uint ltid, vec4 xl, vec4 xh) {
    const uint q = nf_w_u32_even(ws, bp + 2u + 4u * ltid);
    return (dot(vec4(nf_ql_nib_lo(q)) - 8.0, xl) + dot(vec4(nf_ql_nib_hi(q)) - 8.0, xh)) * nf_w_f16(ws, bp);
}

float qt_elem(uint ws, uint rbase, uint i) {
    const uint bp = rbase + (i / uint(QT_BS)) * uint(QT_BB);
    return nf_w_f16(ws, bp) * float(int(nf_ql_nibble(ws, bp + 2u, i % uint(QT_BS))) - 8);
}

/* ================================ Q4_1 ================================
 * 20 B: d(f16)@0, m(f16)@2, qs[16]@4. y[k] = (qs[k]&15)*d + m, y[k+16] = (qs[k]>>4)*d + m
 * (llama.cpp ggml-quants.c:479-498); the dot is d*sum(q*x) + m*sum(x). */
#elif QTYPE == QT_Q4_1
#define QT_BS  NF_Q4_1_BS
#define QT_BB  NF_Q4_1_BB
#define QT_TPS 4
#define QT_DECODE_DOT  1
#define QT_DECODE_DUAL 1
#define QT_DECODE_ELEM 1

float nf_ql_part(uint ws, uint bp, uint ltid, vec4 xl, vec4 xh) {
    const uint q = nf_w_u32(ws, bp + 4u + 4u * ltid);
    const float qsum = dot(vec4(nf_ql_nib_lo(q)), xl) + dot(vec4(nf_ql_nib_hi(q)), xh);
    return nf_w_f16(ws, bp) * qsum + nf_w_f16(ws, bp + 2u) * dot(xl + xh, vec4(1.0));
}

float qt_elem(uint ws, uint rbase, uint i) {
    const uint bp = rbase + (i / uint(QT_BS)) * uint(QT_BB);
    return nf_w_f16(ws, bp) * float(nf_ql_nibble(ws, bp + 4u, i % uint(QT_BS))) + nf_w_f16(ws, bp + 2u);
}

/* ================================ Q5_0 ================================
 * 22 B: d(f16)@0, qh(u32 LE)@2, qs[16]@6.
 * y[k] = (((qs[k]&15) | hbit(k)) - 16)*d, y[k+16] = (((qs[k]>>4) | hbit(k+16)) - 16)*d
 * (llama.cpp ggml-quants.c:500-523). */
#elif QTYPE == QT_Q5_0
#define QT_BS  NF_Q5_0_BS
#define QT_BB  NF_Q5_0_BB
#define QT_TPS 4
#define QT_DECODE_DOT  1
#define QT_DECODE_DUAL 1
#define QT_DECODE_ELEM 1

float nf_ql_part(uint ws, uint bp, uint ltid, vec4 xl, vec4 xh) {
    const uint qh = nf_w_u32_even(ws, bp + 2u);
    const uint q = nf_w_u32_even(ws, bp + 6u + 4u * ltid);
    const uint k0 = 4u * ltid;
    const vec4 lo = vec4(nf_ql_nib_lo(q) | nf_ql_q5_hbits(qh, k0)) - 16.0;
    const vec4 hi = vec4(nf_ql_nib_hi(q) | nf_ql_q5_hbits(qh, k0 + 16u)) - 16.0;
    return (dot(lo, xl) + dot(hi, xh)) * nf_w_f16(ws, bp);
}

float qt_elem(uint ws, uint rbase, uint i) {
    const uint bp = rbase + (i / uint(QT_BS)) * uint(QT_BB);
    const uint e = i % uint(QT_BS);
    const uint v = nf_ql_nibble(ws, bp + 6u, e) | nf_ql_q5_hbit(nf_w_u32_even(ws, bp + 2u), e);
    return nf_w_f16(ws, bp) * float(int(v) - 16);
}

/* ================================ Q5_1 ================================
 * 24 B: d(f16)@0, m(f16)@2, qh(u32 LE)@4, qs[16]@8.
 * y[k] = ((qs[k]&15) | hbit(k))*d + m, y[k+16] = ((qs[k]>>4) | hbit(k+16))*d + m
 * (llama.cpp ggml-quants.c:525-548). */
#elif QTYPE == QT_Q5_1
#define QT_BS  NF_Q5_1_BS
#define QT_BB  NF_Q5_1_BB
#define QT_TPS 4
#define QT_DECODE_DOT  1
#define QT_DECODE_DUAL 1
#define QT_DECODE_ELEM 1

float nf_ql_part(uint ws, uint bp, uint ltid, vec4 xl, vec4 xh) {
    const uint qh = nf_w_u32(ws, bp + 4u);
    const uint q = nf_w_u32(ws, bp + 8u + 4u * ltid);
    const uint k0 = 4u * ltid;
    const vec4 lo = vec4(nf_ql_nib_lo(q) | nf_ql_q5_hbits(qh, k0));
    const vec4 hi = vec4(nf_ql_nib_hi(q) | nf_ql_q5_hbits(qh, k0 + 16u));
    return nf_w_f16(ws, bp) * (dot(lo, xl) + dot(hi, xh)) + nf_w_f16(ws, bp + 2u) * dot(xl + xh, vec4(1.0));
}

float qt_elem(uint ws, uint rbase, uint i) {
    const uint bp = rbase + (i / uint(QT_BS)) * uint(QT_BB);
    const uint e = i % uint(QT_BS);
    const uint v = nf_ql_nibble(ws, bp + 8u, e) | nf_ql_q5_hbit(nf_w_u32(ws, bp + 4u), e);
    return nf_w_f16(ws, bp) * float(v) + nf_w_f16(ws, bp + 2u);
}

/* =============================== IQ4_NL ===============================
 * 18 B: d(f16)@0, qs[16]@2. y[k] = d*kv[qs[k]&15], y[k+16] = d*kv[qs[k]>>4]
 * (llama.cpp ggml-quants.c:2726-2742, kv = kvalues_iq4nl). */
#elif QTYPE == QT_IQ4_NL
#define QT_BS  NF_IQ4_NL_BS
#define QT_BB  NF_IQ4_NL_BB
#define QT_TPS 4
#define QT_DECODE_DOT  1
#define QT_DECODE_DUAL 1
#define QT_DECODE_ELEM 1

float nf_ql_part(uint ws, uint bp, uint ltid, vec4 xl, vec4 xh) {
    const uint q = nf_w_u32_even(ws, bp + 2u + 4u * ltid);
    return (dot(nf_ql_kv_iq4nl(nf_ql_nib_lo(q)), xl) + dot(nf_ql_kv_iq4nl(nf_ql_nib_hi(q)), xh)) * nf_w_f16(ws, bp);
}

float qt_elem(uint ws, uint rbase, uint i) {
    const uint bp = rbase + (i / uint(QT_BS)) * uint(QT_BB);
    return nf_w_f16(ws, bp) * float(nf_kvalue_iq4nl(nf_ql_nibble(ws, bp + 2u, i % uint(QT_BS))));
}

/* ================================ MXFP4 ===============================
 * 17 B: e(e8m0)@0, qs[16]@1. y[k] = kv[qs[k]&15]*s, y[k+16] = kv[qs[k]>>4]*s with the doubled
 * e2m1 values kv and s = ggml_e8m0_to_fp32_half(e) (llama.cpp ggml-quants.c:569-586). */
#elif QTYPE == QT_MXFP4
#define QT_BS  NF_MXFP4_BS
#define QT_BB  NF_MXFP4_BB
#define QT_TPS 4
#define QT_DECODE_DOT  1
#define QT_DECODE_DUAL 1
#define QT_DECODE_ELEM 1

float nf_ql_part(uint ws, uint bp, uint ltid, vec4 xl, vec4 xh) {
    const uint q = nf_w_u32_any(ws, bp + 1u + 4u * ltid);
    const float sum = dot(nf_ql_kv_fp4(nf_ql_nib_lo(q)), xl) + dot(nf_ql_kv_fp4(nf_ql_nib_hi(q)), xh);
    return sum * nf_e8m0_half(nf_w_u8(ws, bp));
}

float qt_elem(uint ws, uint rbase, uint i) {
    const uint bp = rbase + (i / uint(QT_BS)) * uint(QT_BB);
    return nf_e8m0_half(nf_w_u8(ws, bp)) * float(nf_kvalue_fp4(nf_ql_nibble(ws, bp + 1u, i % uint(QT_BS))));
}

/* ================================ Q8_0 ================================
 * 34 B: d(f16)@0, qs[32](int8)@2. y[k] = qs[k]*d (llama.cpp ggml-quants.c:551-566). Invocation ltid
 * owns elements 8*ltid .. 8*ltid+7. */
#elif QTYPE == QT_Q8_0
#define QT_BS  NF_Q8_0_BS
#define QT_BB  NF_Q8_0_BB
#define QT_TPS 4
#define QT_DECODE_DOT  1
#define QT_DECODE_DUAL 1
#define QT_DECODE_ELEM 1

/* Block bp's share for invocation ltid, activations x0 (elements 8*ltid..+3) and x1 (+4..+7). */
float nf_ql_q8_0_part(uint ws, uint bp, uint ltid, vec4 x0, vec4 x1) {
    const uint qo = bp + 2u + 8u * ltid;
    const float s = dot(x0, vec4(nf_w_i8x4(ws, qo))) + dot(x1, vec4(nf_w_i8x4(ws, qo + 4u)));
    return s * nf_w_f16(ws, bp);
}

float qt_dot(uint ws, uint bp, uint xoff, uint blk, uint ltid) {
    const uint xb = xoff + blk * uint(QT_BS) + 8u * ltid;
    return nf_ql_q8_0_part(ws, bp, ltid, qt_x4(xb), qt_x4(xb + 4u));
}

void qt_dot_dual(uint gbp, uint ubp, uint xoff, uint blk, uint ltid, inout float gs, inout float us) {
    const uint xb = xoff + blk * uint(QT_BS) + 8u * ltid;
    const vec4 x0 = qt_x4(xb);
    const vec4 x1 = qt_x4(xb + 4u);
    gs += nf_ql_q8_0_part(NF_WS_A, gbp, ltid, x0, x1);
    us += nf_ql_q8_0_part(NF_WS_B, ubp, ltid, x0, x1);
}

/* A non-finite block scale (corrupt or padding block) decodes as 0, matching both Q8_0 embedding kernels. */
float qt_elem(uint ws, uint rbase, uint i) {
    const uint bp = rbase + (i / uint(QT_BS)) * uint(QT_BB);
    const float d = nf_w_f16(ws, bp);
    return float(nf_w_i8(ws, bp + 2u + i % uint(QT_BS))) * (nf_isfinite(d) ? d : 0.0);
}

/* =============================== IQ4_XS ===============================
 * 136 B: d(f16)@0, scales_h(u16)@2, scales_l[4]@4, qs[128]@8. Sub-block sb (32 elements, qs bytes
 * 8 + 16*sb ..) uses dl = d*(ls - 32) and the IQ4_NL split layout (llama.cpp ggml-quants.c:2744-2762). */
#elif QTYPE == QT_IQ4_XS
#define QT_BS  NF_IQ4_XS_BS
#define QT_BB  NF_IQ4_XS_BB
#define QT_TPS 32
#define QT_DECODE_DOT  1
#define QT_DECODE_DUAL 1
#define QT_DECODE_ELEM 1

/* Activation index of the first owned low-nibble element of invocation ltid in block blk. */
uint nf_ql_iq4xs_xbase(uint xoff, uint blk, uint ltid) {
    return xoff + blk * uint(QT_BS) + 32u * (ltid >> 2) + 4u * (ltid & 3u);
}

float nf_ql_part(uint ws, uint bp, uint ltid, vec4 xl, vec4 xh) {
    const uint sb = ltid >> 2;
    const uint q = nf_w_u32(ws, bp + 8u + 16u * sb + 4u * (ltid & 3u));
    const float sum = dot(nf_ql_kv_iq4nl(nf_ql_nib_lo(q)), xl) + dot(nf_ql_kv_iq4nl(nf_ql_nib_hi(q)), xh);
    return sum * nf_iq4xs_sub_scale(ws, bp, sb);
}

float qt_dot(uint ws, uint bp, uint xoff, uint blk, uint ltid) {
    const uint xb = nf_ql_iq4xs_xbase(xoff, blk, ltid);
    return nf_ql_part(ws, bp, ltid, qt_x4(xb), qt_x4(xb + 16u));
}

void qt_dot_dual(uint gbp, uint ubp, uint xoff, uint blk, uint ltid, inout float gs, inout float us) {
    const uint xb = nf_ql_iq4xs_xbase(xoff, blk, ltid);
    const vec4 xl = qt_x4(xb);
    const vec4 xh = qt_x4(xb + 16u);
    gs += nf_ql_part(NF_WS_A, gbp, ltid, xl, xh);
    us += nf_ql_part(NF_WS_B, ubp, ltid, xl, xh);
}

float qt_elem(uint ws, uint rbase, uint i) {
    const uint bp = rbase + (i / uint(QT_BS)) * uint(QT_BB);
    const uint w = i % uint(QT_BS);
    const uint sb = w >> 5;
    const uint n = nf_ql_nibble(ws, bp + 8u + 16u * sb, w & 31u);
    return nf_iq4xs_sub_scale(ws, bp, sb) * float(nf_kvalue_iq4nl(n));
}
#endif /* QTYPE selection */

/* Shared qt_dot / qt_dot_dual of the 32-element split-layout types (TPS = 4): invocation ltid
 * pairs activations xl = elements 4*ltid .. +3 and xh = elements 4*ltid+16 .. +19 of the block
 * with nf_ql_part(). */
#if QTYPE == QT_Q4_0 || QTYPE == QT_Q4_1 || QTYPE == QT_Q5_0 || QTYPE == QT_Q5_1 \
 || QTYPE == QT_IQ4_NL || QTYPE == QT_MXFP4
float qt_dot(uint ws, uint bp, uint xoff, uint blk, uint ltid) {
    const uint xb = xoff + blk * uint(QT_BS) + 4u * ltid;
    return nf_ql_part(ws, bp, ltid, qt_x4(xb), qt_x4(xb + 16u));
}

void qt_dot_dual(uint gbp, uint ubp, uint xoff, uint blk, uint ltid, inout float gs, inout float us) {
    const uint xb = xoff + blk * uint(QT_BS) + 4u * ltid;
    const vec4 xl = qt_x4(xb);
    const vec4 xh = qt_x4(xb + 16u);
    gs += nf_ql_part(NF_WS_A, gbp, ltid, xl, xh);
    us += nf_ql_part(NF_WS_B, ubp, ltid, xl, xh);
}
#endif

#endif /* QTYPE */

#endif /* NF_QUANT_LEGACY_GLSL */

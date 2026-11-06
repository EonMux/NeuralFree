/* common/quant_ext.glsl - QT_DECODE_* decoders for the NeuralFree extension weight types F8_E5M2,
 * F8_E4M3, F8_E4M3_B128, F8_E4M3_B32, MXFP8, MXFP6_E2M3, MXFP6_E3M2, I8_B128, NF4 and FP4_BNB. GLSL twin of
 * kernels/llm/common/quant_decode_ext.h (the semantic reference). Requires: common/quant_blocks.glsl.
 *
 * Decoder interface and include contract: common/quant_blocks.glsl. Weights are read only through
 * nf_w_*(ws, off) and activations only through qt_x/qt_x4.
 *
 * Block layouts (little-endian; Java GGMLType ids 1000-1009, BlockLayout offsets):
 *   F8_E5M2       32 B: qs[32] (E5M2) @0                y[j] = e5m2(qs[j])
 *   F8_E4M3       32 B: qs[32] (E4M3FN) @0              y[j] = e4m3(qs[j])
 *   F8_E4M3_B128 132 B: d (f32) @0, qs[128] (E4M3FN) @4  y[j] = d * e4m3(qs[j])
 *   F8_E4M3_B32   36 B: d (f32) @0, qs[32] (E4M3FN) @4   y[j] = d * e4m3(qs[j])
 *   MXFP8         33 B: e (E8M0) @0, qs[32] (E4M3FN) @1  y[j] = 2^(e - 127) * e4m3(qs[j])
 *   MXFP6_E2M3    25 B: e (E8M0) @0, qs[24] @1 (32 packed 6-bit codes)
 *   MXFP6_E3M2    25 B: e (E8M0) @0, qs[24] @1 (32 packed 6-bit codes)
 *                 y[j] = 2^(e - 127) * fp6(code j)
 *   I8_B128      132 B: d (f32) @0, qs[128] (int8) @4    y[j] = d * qs[j]
 *   NF4, FP4_BNB  36 B: absmax (f32) @0, qs[32] @4       y[j] = code[n_j] * absmax, n_j the low nibble
 *                                                        of qs[j] (j < 32) or the high nibble of qs[j - 32]
 * F8_E5M2 and F8_E4M3 carry no block scale; their per-tensor or per-expert `.scale` is applied by the
 * consumer after the matmul. An E5M2 code is the high byte of an IEEE half, decoded exactly with
 * nf_half (the infinities and the NaNs included, equal to the hardware-FP8 conversions on a float8
 * device); E4M3FN NaN codes (0x7F, 0xFF) decode to 0. The repackers reject the non-finite codes.
 *
 * Work split (equal to the OpenCL triples): every invocation of a block owns the NF_QE_EPT = 8
 * consecutive elements [8*ltid, 8*ltid + 8), in two runs of 4:
 *   F8_E5M2, F8_E4M3, F8_E4M3_B32, MXFP8       QT_TPS = 4  qs bytes 8*ltid .. +7
 *   MXFP6_E2M3, MXFP6_E3M2                     QT_TPS = 4  the 48 bits: qs bytes 6*ltid .. 6*ltid + 5
 *   F8_E4M3_B128, I8_B128                      QT_TPS = 16  qs bytes 8*ltid .. +7
 *   NF4, FP4_BNB                               QT_TPS = 8   low nibbles of qs bytes 8*ltid .. +7 (ltid < 4), high
 *                                                          nibbles of qs bytes 8*(ltid - 4) .. +7 (ltid >= 4)
 * Per type, nf_qe_w8() yields the unscaled values of both runs and nf_qe_scale() the block scale;
 * qt_dot is scale * (dot(w0, run 0) + dot(w1, run 1)).
 *
 * Fast paths. The {T}_Hw_* rows of variants/matvec_extquant.tsv and variants/gemm_tiled.tsv target
 * these decoders with the same QTYPE, so a _Hw variant is bit-identical to the generic row; GpuOps
 * selects them by name only when the device reports the FP8 decode feature (CUDA sm_89+; the CUDA
 * _Hw kernels convert the code pairs in hardware).
 *
 * Alignment: block offsets (bp, rbase) are multiples of QT_BB from the start of a 4-byte aligned
 * binding. F8_E5M2 and F8_E4M3 (32), F8_E4M3_B128 and I8_B128 (132), F8_E4M3_B32, NF4 and FP4_BNB (36) blocks are
 * 4-aligned and read their f32 scale and code words with nf_w_f32 / nf_w_u32; MXFP8 (33) and the MXFP6
 * types (25) read bytes (nf_w_u8 / nf_w_u32_any).
 */
#ifndef NF_QUANT_EXT_GLSL
#define NF_QUANT_EXT_GLSL

#ifndef NF_QUANT_BLOCKS_GLSL
#error "common/quant_ext.glsl requires common/quant_blocks.glsl"
#endif

#ifdef QTYPE
#if QTYPE == QT_F8_E5M2 || QTYPE == QT_F8_E4M3 || QTYPE == QT_F8_E4M3_B128 || QTYPE == QT_F8_E4M3_B32 \
 || QTYPE == QT_MXFP8 \
 || QTYPE == QT_MXFP6_E2M3 || QTYPE == QT_MXFP6_E3M2 \
 || QTYPE == QT_I8_B128 || QTYPE == QT_NF4 || QTYPE == QT_FP4_BNB

#define QT_DECODE_DOT  1
#define QT_DECODE_DUAL 1
#define QT_DECODE_ELEM 1

const uint NF_QE_EPT = 8u;    /* elements per invocation: QT_BS / QT_TPS */

/* The four bytes of v; component k comes from byte k. */
uvec4 nf_qe_bytes(uint v) {
    return uvec4(v, v >> 8, v >> 16, v >> 24) & 0xFFu;
}

/* E4M3FN code c (sign bit 7, exponent bits 6-3 with bias 7, mantissa bits 2-0) to float:
 * sign * (e == 0 ? m * 2^-9 : (1 + m/8) * 2^(e - 7)), computed exactly as (e ? 8 + m : m) *
 * 2^(max(e, 1) - 10), equal to Java Fp8Codec.E4M3FN and the C F8_E4M3_LUT on every finite code;
 * the NaN codes 0x7F and 0xFF give +0 as in F8_E4M3_LUT. */
float nf_qe_e4m3(uint c) {
    if ((c & 0x7Fu) == 0x7Fu) {
        return 0.0;
    }
    const uint e = (c >> 3) & 0xFu;
    const float v = float((c & 7u) | (min(e, 1u) << 3)) * uintBitsToFloat((max(e, 1u) + 117u) << 23);
    return (c & 0x80u) != 0u ? -v : v;
}

/* nf_qe_e4m3 of the four codes of c (each in [0, 256)). */
vec4 nf_qe_e4m3x4(uvec4 c) {
    const uvec4 e = (c >> 3) & 0xFu;
    const vec4 v = vec4((c & 7u) | (min(e, uvec4(1u)) << 3)) * uintBitsToFloat((max(e, uvec4(1u)) + 117u) << 23);
    return mix(vec4(0.0), (1.0 - 2.0 * vec4(c >> 7)) * v, notEqual(c & 0x7Fu, uvec4(0x7Fu)));
}

#if QTYPE == QT_F8_E5M2
/* ---- F8_E5M2: qs[32] @0, no block scale ---- */
#define QT_BS  NF_F8_E5M2_BS
#define QT_BB  NF_F8_E5M2_BB
#define QT_TPS 4

/* The unscaled E5M2 values of the four codes of c: the code is the high byte of an IEEE half
 * (Java Fp8Codec.E5M2 and the C half_bits_to_float), infinities and NaNs included. */
vec4 nf_qe_e5m2x4(uvec4 c) {
    return vec4(nf_half((c.x & 0xFFu) << 8), nf_half((c.y & 0xFFu) << 8),
                nf_half((c.z & 0xFFu) << 8), nf_half((c.w & 0xFFu) << 8));
}

void nf_qe_w8(uint ws, uint bp, uint ltid, out vec4 w0, out vec4 w1) {
    const uint q = bp + NF_QE_EPT * ltid;
    w0 = nf_qe_e5m2x4(nf_qe_bytes(nf_w_u32(ws, q)));
    w1 = nf_qe_e5m2x4(nf_qe_bytes(nf_w_u32(ws, q + 4u)));
}

float nf_qe_scale(uint ws, uint bp) {
    return 1.0;
}

float qt_elem(uint ws, uint rbase, uint i) {
    const uint bp = rbase + (i / uint(QT_BS)) * uint(QT_BB);
    return nf_half(uint(nf_w_u8(ws, bp + i % uint(QT_BS))) << 8);
}

#elif QTYPE == QT_F8_E4M3
/* ---- F8_E4M3: qs[32] @0, no block scale ---- */
#define QT_BS  NF_F8_E4M3_BS
#define QT_BB  NF_F8_E4M3_BB
#define QT_TPS 4

void nf_qe_w8(uint ws, uint bp, uint ltid, out vec4 w0, out vec4 w1) {
    const uint q = bp + NF_QE_EPT * ltid;
    w0 = nf_qe_e4m3x4(nf_qe_bytes(nf_w_u32(ws, q)));
    w1 = nf_qe_e4m3x4(nf_qe_bytes(nf_w_u32(ws, q + 4u)));
}

float nf_qe_scale(uint ws, uint bp) {
    return 1.0;
}

float qt_elem(uint ws, uint rbase, uint i) {
    const uint bp = rbase + (i / uint(QT_BS)) * uint(QT_BB);
    return nf_qe_e4m3(nf_w_u8(ws, bp + i % uint(QT_BS)));
}

#elif QTYPE == QT_F8_E4M3_B128
/* ---- F8_E4M3_B128: d (f32) @0, qs[128] @4 ---- */
#define QT_BS  NF_F8_E4M3_B128_BS
#define QT_BB  NF_F8_E4M3_B128_BB
#define QT_TPS 16

void nf_qe_w8(uint ws, uint bp, uint ltid, out vec4 w0, out vec4 w1) {
    const uint q = bp + 4u + NF_QE_EPT * ltid;
    w0 = nf_qe_e4m3x4(nf_qe_bytes(nf_w_u32(ws, q)));
    w1 = nf_qe_e4m3x4(nf_qe_bytes(nf_w_u32(ws, q + 4u)));
}

float nf_qe_scale(uint ws, uint bp) {
    return nf_w_f32(ws, bp);
}

float qt_elem(uint ws, uint rbase, uint i) {
    const uint bp = rbase + (i / uint(QT_BS)) * uint(QT_BB);
    return nf_w_f32(ws, bp) * nf_qe_e4m3(nf_w_u8(ws, bp + 4u + i % uint(QT_BS)));
}

#elif QTYPE == QT_F8_E4M3_B32
/* ---- F8_E4M3_B32: d (f32) @0, qs[32] @4 ---- */
#define QT_BS  NF_F8_E4M3_B32_BS
#define QT_BB  NF_F8_E4M3_B32_BB
#define QT_TPS 4

void nf_qe_w8(uint ws, uint bp, uint ltid, out vec4 w0, out vec4 w1) {
    const uint q = bp + 4u + NF_QE_EPT * ltid;
    w0 = nf_qe_e4m3x4(nf_qe_bytes(nf_w_u32(ws, q)));
    w1 = nf_qe_e4m3x4(nf_qe_bytes(nf_w_u32(ws, q + 4u)));
}

float nf_qe_scale(uint ws, uint bp) {
    return nf_w_f32(ws, bp);
}

float qt_elem(uint ws, uint rbase, uint i) {
    const uint bp = rbase + (i / uint(QT_BS)) * uint(QT_BB);
    return nf_w_f32(ws, bp) * nf_qe_e4m3(nf_w_u8(ws, bp + 4u + i % uint(QT_BS)));
}

#elif QTYPE == QT_MXFP8
/* ---- MXFP8 (OCP MX): e (E8M0) @0, qs[32] @1; block scale nf_e8m0(e) = 2^(e - 127) ---- */
#define QT_BS  NF_MXFP8_BS
#define QT_BB  NF_MXFP8_BB
#define QT_TPS 4

void nf_qe_w8(uint ws, uint bp, uint ltid, out vec4 w0, out vec4 w1) {
    const uint q = bp + 1u + NF_QE_EPT * ltid;
    w0 = nf_qe_e4m3x4(nf_qe_bytes(nf_w_u32_any(ws, q)));
    w1 = nf_qe_e4m3x4(nf_qe_bytes(nf_w_u32_any(ws, q + 4u)));
}

float nf_qe_scale(uint ws, uint bp) {
    return nf_e8m0(nf_w_u8(ws, bp));
}

float qt_elem(uint ws, uint rbase, uint i) {
    const uint bp = rbase + (i / uint(QT_BS)) * uint(QT_BB);
    return nf_qe_scale(ws, bp) * nf_qe_e4m3(nf_w_u8(ws, bp + 1u + i % uint(QT_BS)));
}

#elif QTYPE == QT_MXFP6_E2M3 || QTYPE == QT_MXFP6_E3M2
/* ---- MXFP6 (OCP MX FP6, Java GGMLType ids 1007 / 1008): e (E8M0) @0, qs[24] @1 ----
 * 32 packed 6-bit codes, element j = code j (four codes per three bytes, little-endian bit order; the
 * OCP-MX element order, matched by loops/FpBlockDecoders.java), times the block scale nf_e8m0(e) =
 * 2^(e - 127). The FP6 formats have no infinities and no NaNs (every one of the 64 codes is finite);
 * the NaN E8M0 reservation e == 0xFF is rejected at load. */
#define QT_BS  (QTYPE == QT_MXFP6_E2M3 ? NF_MXFP6_E2M3_BS : NF_MXFP6_E3M2_BS)
#define QT_BB  (QTYPE == QT_MXFP6_E2M3 ? NF_MXFP6_E2M3_BB : NF_MXFP6_E3M2_BB)
#define QT_TPS 4

#if QTYPE == QT_MXFP6_E2M3
/* The value of an E2M3 code (sign 5, e bits 4-3 with bias 1, m bits 2-0), exactly the arithmetic of
 * Java FpBlockDecoders.e2m3ToFp32: e == 0 the subnormal m * 2^-3, otherwise (8 + m) * 2^(e - 4). */
float nf_qe_mxfp6(uint c) {
    const uint e = (c >> 3u) & 3u;
    const float v = float(e == 0u ? (c & 7u) : 8u + (c & 7u)) * uintBitsToFloat((max(e, 1u) + 123u) << 23);
    return (c & 0x20u) != 0u ? -v : v;
}
#else
/* The value of an E3M2 code (sign 5, e bits 4-2 with bias 3, m bits 1-0), exactly the arithmetic of
 * Java FpBlockDecoders.e3m2ToFp32: e == 0 the subnormal m * 2^-4, otherwise (4 + m) * 2^(e - 5). */
float nf_qe_mxfp6(uint c) {
    const uint e = (c >> 2u) & 7u;
    const float v = float(e == 0u ? (c & 3u) : 4u + (c & 3u)) * uintBitsToFloat((max(e, 1u) + 122u) << 23);
    return (c & 0x20u) != 0u ? -v : v;
}
#endif

/* The four 6-bit codes of the 24-bit word: code k from bits 6k .. 6k + 5. */
vec4 nf_qe_mxfp6x4(uint word) {
    return vec4(nf_qe_mxfp6(word & 63u), nf_qe_mxfp6((word >> 6u) & 63u),
                nf_qe_mxfp6((word >> 12u) & 63u), nf_qe_mxfp6((word >> 18u) & 63u));
}

/* ltid owns elements 8*ltid .. +7: the 48 bits at qs + 6*ltid, the low 24 bits elements +0..+3
 * (word 1) and bits 24..47 elements +4..+7 (a u32 at qs + 6*ltid + 2 shifted right by 8). */
void nf_qe_w8(uint ws, uint bp, uint ltid, out vec4 w0, out vec4 w1) {
    const uint q = bp + 1u + 6u * ltid;
    w0 = nf_qe_mxfp6x4(nf_w_u32_any(ws, q));
    w1 = nf_qe_mxfp6x4(nf_w_u32_any(ws, q + 2u) >> 8);
}

float nf_qe_scale(uint ws, uint bp) {
    return nf_e8m0(nf_w_u8(ws, bp));
}

float qt_elem(uint ws, uint rbase, uint i) {
    const uint bp = rbase + (i / uint(QT_BS)) * uint(QT_BB);
    const uint e = i % uint(QT_BS);
    const uint code = (nf_w_u32_any(ws, bp + 1u + (e >> 2) * 3u) >> (6u * (e & 3u))) & 63u;
    return nf_qe_scale(ws, bp) * nf_qe_mxfp6(code);
}

#elif QTYPE == QT_I8_B128
/* ---- I8_B128: d (f32) @0, qs[128] (int8) @4 ---- */
#define QT_BS  NF_I8_B128_BS
#define QT_BB  NF_I8_B128_BB
#define QT_TPS 16

/* The four signed bytes of v; component k comes from byte k. */
vec4 nf_qe_i8x4(uint v) {
    const int s = int(v);
    return vec4(bitfieldExtract(s, 0, 8), bitfieldExtract(s, 8, 8), bitfieldExtract(s, 16, 8), bitfieldExtract(s, 24, 8));
}

void nf_qe_w8(uint ws, uint bp, uint ltid, out vec4 w0, out vec4 w1) {
    const uint q = bp + 4u + NF_QE_EPT * ltid;
    w0 = nf_qe_i8x4(nf_w_u32(ws, q));
    w1 = nf_qe_i8x4(nf_w_u32(ws, q + 4u));
}

float nf_qe_scale(uint ws, uint bp) {
    return nf_w_f32(ws, bp);
}

float qt_elem(uint ws, uint rbase, uint i) {
    const uint bp = rbase + (i / uint(QT_BS)) * uint(QT_BB);
    return nf_w_f32(ws, bp) * float(nf_w_i8(ws, bp + 4u + i % uint(QT_BS)));
}

#else /* QT_NF4, QT_FP4_BNB */
/* ---- NF4 / FP4_BNB (bitsandbytes 4-bit): absmax (f32) @0, qs[32] @4 ----
 * The codebooks are the fp32 output of bitsandbytes functional.get_4bit_type (functional.py:733-750
 * NF4; :764 FP4 divided by its maximum magnitude 12, :793-794), stored as IEEE bit patterns equal to
 * the C tables NF4_CODE / FP4_BNB_CODE (kernels/llm/common/quant_blocks.h) and Java QuantTables;
 * FP4 code 8 is -0, the sign-bit-only code (bitsandbytes kernels.cu dDequantizeFP4Tree). */
#if QTYPE == QT_NF4
#define QT_BS  NF_NF4_BS
#define QT_BB  NF_NF4_BB
const uint NF_QE_CODE_BITS[16] = uint[16](
    0xBF800000u, 0xBF3239B1u, 0xBF066B30u, 0xBECA32A0u, 0xBE91A24Du, 0xBE3D353Fu, 0xBDBA7871u, 0x00000000u,
    0x3DA2FAFFu, 0x3E24CAE3u, 0x3E7C04DDu, 0x3EAD033Au, 0x3EE1A4B8u, 0x3F1007ABu, 0x3F3913B3u, 0x3F800000u);
#else
#define QT_BS  NF_FP4_BNB_BS
#define QT_BB  NF_FP4_BNB_BB
const uint NF_QE_CODE_BITS[16] = uint[16](
    0x00000000u, 0x3BAAAAABu, 0x3F2AAAABu, 0x3F800000u, 0x3EAAAAABu, 0x3F000000u, 0x3E2AAAABu, 0x3E800000u,
    0x80000000u, 0xBBAAAAABu, 0xBF2AAAABu, 0xBF800000u, 0xBEAAAAABu, 0xBF000000u, 0xBE2AAAABu, 0xBE800000u);
#endif
#define QT_TPS 8

/* Codebook value of nibble n. */
float nf_qe_code(uint n) {
    return uintBitsToFloat(NF_QE_CODE_BITS[n]);
}

/* Codebook values of the nibbles at bit sh (0 or 4) of the four bytes of v. */
vec4 nf_qe_code4(uint v, uint sh) {
    return vec4(nf_qe_code((v >> sh) & 0xFu), nf_qe_code((v >> (sh + 8u)) & 0xFu),
                nf_qe_code((v >> (sh + 16u)) & 0xFu), nf_qe_code((v >> (sh + 24u)) & 0xFu));
}

void nf_qe_w8(uint ws, uint bp, uint ltid, out vec4 w0, out vec4 w1) {
    const uint q = bp + 4u + NF_QE_EPT * (ltid & 3u);
    const uint sh = (ltid >> 2) * 4u;
    w0 = nf_qe_code4(nf_w_u32(ws, q), sh);
    w1 = nf_qe_code4(nf_w_u32(ws, q + 4u), sh);
}

float nf_qe_scale(uint ws, uint bp) {
    return nf_w_f32(ws, bp);
}

float qt_elem(uint ws, uint rbase, uint i) {
    const uint bp = rbase + (i / uint(QT_BS)) * uint(QT_BB);
    const uint e = i % uint(QT_BS);
    const uint n = (nf_w_u8(ws, bp + 4u + (e & 31u)) >> ((e >> 5) * 4u)) & 0xFu;
    return nf_qe_code(n) * nf_w_f32(ws, bp);
}

#endif /* per-type decoders */

float qt_dot(uint ws, uint bp, uint xoff, uint blk, uint ltid) {
    vec4 w0, w1;
    nf_qe_w8(ws, bp, ltid, w0, w1);
    const uint x = xoff + blk * uint(QT_BS) + NF_QE_EPT * ltid;
    return nf_qe_scale(ws, bp) * (dot(w0, qt_x4(x)) + dot(w1, qt_x4(x + 4u)));
}

void qt_dot_dual(uint gbp, uint ubp, uint xoff, uint blk, uint ltid, inout float gs, inout float us) {
    vec4 g0, g1, u0, u1;
    nf_qe_w8(NF_WS_A, gbp, ltid, g0, g1);
    nf_qe_w8(NF_WS_B, ubp, ltid, u0, u1);
    const uint x = xoff + blk * uint(QT_BS) + NF_QE_EPT * ltid;
    const vec4 x0 = qt_x4(x);
    const vec4 x1 = qt_x4(x + 4u);
    gs += nf_qe_scale(NF_WS_A, gbp) * (dot(g0, x0) + dot(g1, x1));
    us += nf_qe_scale(NF_WS_B, ubp) * (dot(u0, x0) + dot(u1, x1));
}

#endif /* extension QTYPE */
#endif /* QTYPE */

#endif /* NF_QUANT_EXT_GLSL */

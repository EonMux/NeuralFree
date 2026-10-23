/* common/quant_blocks.glsl - weight-type ids (QT_*), block constants, byte/half readers over the
 * aliased weight views, scalar block helpers shared by several decoder files, the QT_DECODE_*
 * decoder interface and the dense (F32/F16/BF16) decoders.
 * GLSL twin of kernels/llm/common/quant_blocks.h. Requires: common/nf_common.glsl.
 *
 * Include contract. Before this file a shader declares its weight views and, for two weight
 * streams, the second set of views:
 *     NF_BUFFER_VIEWS(<binding>, w)                      weight stream A (NF_WS_A)
 *     #define NF_WEIGHT_STREAMS 2                         only for two weight streams
 *     NF_BUFFER_VIEWS(<binding>, u)                      weight stream B (NF_WS_B), e.g. the up
 *                                                        projection of a fused gate/up matvec
 * QTYPE (one of the QT_* ids, normally from a variants row: QTYPE=QT_Q4_K) selects the decoder;
 * it may be left undefined by shaders that only need the readers.
 *
 * Decoder interface (QT_DECODE_* plug-in points). For the selected QTYPE exactly one decoder file
 * defines:
 *     QT_BS, QT_BB, QT_TPS           elements per block, bytes per block, invocations that
 *                                    cooperate on one block (adjacent invocations, ltid 0..)
 *     QT_DECODE_DOT   + qt_dot()     partial dot product of one block with the activations:
 *                                    invocation ltid in [0, QT_TPS) of the block at byte offset
 *                                    bp of stream ws, with activation element e of the block read
 *                                    as qt_x(xoff + blk * QT_BS + e)
 *     QT_DECODE_ELEM  + qt_elem()    element i (0 <= i < row length) of the row whose first byte
 *                                    is rbase in stream ws
 *     QT_DECODE_DUAL  + qt_dot_dual  optional: qt_dot of stream A (gate, block at gbp) and stream B
 *                                    (up, block at ubp) sharing the activation loads; adds into
 *                                    gs and us. Without it templates issue two qt_dot calls.
 * A shader that calls qt_dot defines the activation readers qt_x()/qt_x4() (prototypes below).
 * All byte offsets are 32-bit: a weight binding never exceeds maxStorageBufferRange (GpuWeight
 * shards keep every binding below it).
 */
/* Portions derived from llama.cpp/ggml (MIT License); see THIRD-PARTY-NOTICES.md. */
#ifndef NF_QUANT_BLOCKS_GLSL
#define NF_QUANT_BLOCKS_GLSL

#ifndef NF_COMMON_GLSL
#error "common/quant_blocks.glsl requires common/nf_common.glsl"
#endif

/* ---- Weight-type ids (values equal ggml_type, llama.cpp ggml/include/ggml.h) ---- */
#define QT_F32      0
#define QT_F16      1
#define QT_Q4_0     2
#define QT_Q4_1     3
#define QT_Q5_0     6
#define QT_Q5_1     7
#define QT_Q8_0     8
#define QT_Q2_K     10
#define QT_Q3_K     11
#define QT_Q4_K     12
#define QT_Q5_K     13
#define QT_Q6_K     14
#define QT_Q8_K     15
#define QT_IQ2_XXS  16
#define QT_IQ2_XS   17
#define QT_IQ3_XXS  18
#define QT_IQ1_S    19
#define QT_IQ4_NL   20
#define QT_IQ3_S    21
#define QT_IQ2_S    22
#define QT_IQ4_XS   23
#define QT_IQ1_M    29
#define QT_BF16     30
#define QT_TQ1_0    34
#define QT_TQ2_0    35
#define QT_MXFP4    39
#define QT_NVFP4    40
#define QT_Q1_0     41
#define QT_Q2_0     42

/* ---- Block constants: elements (_BS) and bytes (_BB) per block (llama.cpp ggml-common.h) ---- */
#define NF_F32_BS      1
#define NF_F32_BB      4
#define NF_F16_BS      1
#define NF_F16_BB      2
#define NF_BF16_BS     1
#define NF_BF16_BB     2
#define NF_Q4_0_BS     32
#define NF_Q4_0_BB     18
#define NF_Q4_1_BS     32
#define NF_Q4_1_BB     20
#define NF_Q5_0_BS     32
#define NF_Q5_0_BB     22
#define NF_Q5_1_BS     32
#define NF_Q5_1_BB     24
#define NF_Q8_0_BS     32
#define NF_Q8_0_BB     34
#define NF_IQ4_NL_BS   32
#define NF_IQ4_NL_BB   18
#define NF_MXFP4_BS    32
#define NF_MXFP4_BB    17
#define NF_NVFP4_BS    64
#define NF_NVFP4_BB    36
#define NF_Q1_0_BS     128
#define NF_Q1_0_BB     18
#define NF_Q2_0_BS     64
#define NF_Q2_0_BB     18
#define NF_Q2_K_BS     256
#define NF_Q2_K_BB     84
#define NF_Q3_K_BS     256
#define NF_Q3_K_BB     110
#define NF_Q4_K_BS     256
#define NF_Q4_K_BB     144
#define NF_Q5_K_BS     256
#define NF_Q5_K_BB     176
#define NF_Q6_K_BS     256
#define NF_Q6_K_BB     210
#define NF_Q8_K_BS     256
#define NF_Q8_K_BB     292
#define NF_IQ2_XXS_BS  256
#define NF_IQ2_XXS_BB  66
#define NF_IQ2_XS_BS   256
#define NF_IQ2_XS_BB   74
#define NF_IQ2_S_BS    256
#define NF_IQ2_S_BB    82
#define NF_IQ3_XXS_BS  256
#define NF_IQ3_XXS_BB  98
#define NF_IQ3_S_BS    256
#define NF_IQ3_S_BB    110
#define NF_IQ1_S_BS    256
#define NF_IQ1_S_BB    50
#define NF_IQ1_M_BS    256
#define NF_IQ1_M_BB    56
#define NF_IQ4_XS_BS   256
#define NF_IQ4_XS_BB   136
#define NF_TQ1_0_BS    256
#define NF_TQ1_0_BB    54
#define NF_TQ2_0_BS    256
#define NF_TQ2_0_BB    66

/* ---- Weight streams and readers ---- */
#ifndef NF_WEIGHT_STREAMS
#define NF_WEIGHT_STREAMS 1
#endif
#if NF_WEIGHT_STREAMS != 1 && NF_WEIGHT_STREAMS != 2
#error "NF_WEIGHT_STREAMS must be 1 or 2"
#endif

const uint NF_WS_A = 0u;   /* views w_* (weight, or gate of a fused gate/up) */
const uint NF_WS_B = 1u;   /* views u_* (up of a fused gate/up); aliases stream A with one stream */

/* Byte at any offset. */
uint nf_w_u8(uint ws, uint off) {
#if NF_WEIGHT_STREAMS == 2
    return ws == NF_WS_A ? uint(w_u8[off]) : uint(u_u8[off]);
#else
    return uint(w_u8[off]);
#endif
}

/* Little-endian 16-bit value at an even offset. */
uint nf_w_u16(uint ws, uint off) {
#if NF_WEIGHT_STREAMS == 2
    return ws == NF_WS_A ? uint(w_u16[off >> 1]) : uint(u_u16[off >> 1]);
#else
    return uint(w_u16[off >> 1]);
#endif
}

/* Little-endian 32-bit value at an offset divisible by 4. */
uint nf_w_u32(uint ws, uint off) {
#if NF_WEIGHT_STREAMS == 2
    return ws == NF_WS_A ? w_u32[off >> 2] : u_u32[off >> 2];
#else
    return w_u32[off >> 2];
#endif
}

/* Signed byte at any offset. */
int nf_w_i8(uint ws, uint off) {
    return nf_i8(nf_w_u8(ws, off));
}

/* Little-endian 16-bit value at any offset. */
uint nf_w_u16_any(uint ws, uint off) {
    return nf_w_u8(ws, off) | (nf_w_u8(ws, off + 1u) << 8);
}

/* Little-endian 32-bit value at an even offset (two 16-bit loads). */
uint nf_w_u32_even(uint ws, uint off) {
    return nf_w_u16(ws, off) | (nf_w_u16(ws, off + 2u) << 16);
}

/* Little-endian 32-bit value at any offset (C read_u32_le). */
uint nf_w_u32_any(uint ws, uint off) {
    return nf_w_u8(ws, off) | (nf_w_u8(ws, off + 1u) << 8)
         | (nf_w_u8(ws, off + 2u) << 16) | (nf_w_u8(ws, off + 3u) << 24);
}

/* IEEE half at an even offset. */
float nf_w_f16(uint ws, uint off) {
    return nf_half(nf_w_u16(ws, off));
}

/* bfloat16 at an even offset. */
float nf_w_bf16(uint ws, uint off) {
    return nf_bf16(nf_w_u16(ws, off));
}

/* IEEE single at an offset divisible by 4. */
float nf_w_f32(uint ws, uint off) {
    return uintBitsToFloat(nf_w_u32(ws, off));
}

/* Four signed bytes at an even offset, element k from byte off + k. */
ivec4 nf_w_i8x4(uint ws, uint off) {
    const int v = int(nf_w_u32_even(ws, off));
    return ivec4(bitfieldExtract(v, 0, 8), bitfieldExtract(v, 8, 8),
                 bitfieldExtract(v, 16, 8), bitfieldExtract(v, 24, 8));
}

/* Four unsigned bytes at an even offset, element k from byte off + k. */
uvec4 nf_w_u8x4(uint ws, uint off) {
    const uint v = nf_w_u32_even(ws, off);
    return uvec4(v & 0xFFu, (v >> 8) & 0xFFu, (v >> 16) & 0xFFu, v >> 24);
}

/* ---- Scalar block helpers (C quant_blocks.h equivalents) ---- */

/* kvalues_iq4nl[q] for q in [0, 16) (llama.cpp ggml-common.h:1120), int8 values packed little-endian. */
const uvec4 NF_KVALUES_IQ4NL = uvec4(0xBFAD9881u, 0xF6EADDCFu, 0x26190D01u, 0x71594535u);
int nf_kvalue_iq4nl(uint q) {
    return bitfieldExtract(int(NF_KVALUES_IQ4NL[q >> 2]), int((q & 3u) * 8u), 8);
}

/* kvalues_fp4[q] for q in [0, 16): doubled e2m1 values shared by MXFP4 and NVFP4
 * (llama.cpp ggml-common.h:1126), int8 values packed little-endian. */
const uvec4 NF_KVALUES_FP4 = uvec4(0x03020100u, 0x0C080604u, 0xFDFEFF00u, 0xF4F8FAFCu);
int nf_kvalue_fp4(uint q) {
    return bitfieldExtract(int(NF_KVALUES_FP4[q >> 2]), int((q & 3u) * 8u), 8);
}

/* E8M0 exponent byte to 0.5 * 2^(e - 127), the scale paired with the doubled fp4 values
 * (llama.cpp ggml-impl.h:489 ggml_e8m0_to_fp32_half; C mxfp4_scale). */
float nf_e8m0_half(uint e) {
    return uintBitsToFloat(e < 2u ? (0x00200000u << e) : ((e - 1u) << 23));
}

/* 6-bit scale (x) and min (y) of sub-block j in [0, 8) of a Q4_K/Q5_K block whose 12 scale
 * bytes start at sc (llama.cpp ggml-quants.c:880 get_scale_min_k4; C k4_scale_min). */
vec2 nf_k4_scale_min(uint ws, uint sc, uint j) {
    if (j < 4u) {
        return vec2(float(nf_w_u8(ws, sc + j) & 63u), float(nf_w_u8(ws, sc + j + 4u) & 63u));
    }
    const uint q4 = nf_w_u8(ws, sc + j + 4u);
    return vec2(float((q4 & 0xFu) | ((nf_w_u8(ws, sc + j - 4u) >> 6) << 4)),
                float((q4 >> 4) | ((nf_w_u8(ws, sc + j) >> 6) << 4)));
}

/* d * (ls - 32) of sub-block sb in [0, 8) of the IQ4_XS block at bp
 * (llama.cpp ggml-quants.c:2755; C iq4xs_sub_scale). */
float nf_iq4xs_sub_scale(uint ws, uint bp, uint sb) {
    const float d = nf_w_f16(ws, bp);
    const uint scales_h = nf_w_u16(ws, bp + 2u);
    const uint sl = nf_w_u8(ws, bp + 4u + sb / 2u);
    const int ls = int(((sl >> (4u * (sb & 1u))) & 0xFu) | (((scales_h >> (2u * sb)) & 3u) << 4));
    return d * float(ls - 32);
}

/* d * (1 + 2 * nibble) of 32-element group sb in [0, 8) of the IQ3_S block at bp; the four scale
 * bytes start at byte 106 (llama.cpp ggml-quants.c:2620; C iq3s_scale). */
float nf_iq3s_scale(uint ws, uint bp, uint sb) {
    const float d = nf_w_f16(ws, bp);
    const uint s = nf_w_u8(ws, bp + 106u + sb / 2u);
    const uint nib = (sb & 1u) != 0u ? (s >> 4) : (s & 0xFu);
    return d * float(1u + 2u * nib);
}

/* Super-block scale of the IQ1_M block at bp, rebuilt from the top nibbles of its four 16-bit
 * scale words at byte 48 (llama.cpp ggml-quants.c:2688; C iq1m_reconstruct_d). */
float nf_iq1m_d(uint ws, uint bp) {
    const uint sc0 = nf_w_u16(ws, bp + 48u);
    const uint sc1 = nf_w_u16(ws, bp + 50u);
    const uint sc2 = nf_w_u16(ws, bp + 52u);
    const uint sc3 = nf_w_u16(ws, bp + 54u);
    return nf_half((sc0 >> 12) | ((sc1 >> 8) & 0xF0u) | ((sc2 >> 4) & 0xF00u) | (sc3 & 0xF000u));
}

/* ---- Decoder interface ---- */
float qt_x(uint i);            /* activation element i (defined by the shader that calls qt_dot) */
vec4  qt_x4(uint i);           /* activations i .. i+3 (any alignment) */
float qt_dot(uint ws, uint bp, uint xoff, uint blk, uint ltid);
void  qt_dot_dual(uint gbp, uint ubp, uint xoff, uint blk, uint ltid, inout float gs, inout float us);
float qt_elem(uint ws, uint rbase, uint i);

/* ---- Dense decoders: one element per block, one invocation per block ---- */
#ifdef QTYPE
#if QTYPE == QT_F32 || QTYPE == QT_F16 || QTYPE == QT_BF16

#define QT_TPS 1
#define QT_BS 1
#define QT_DECODE_DOT 1
#define QT_DECODE_ELEM 1

#if QTYPE == QT_F32
#define QT_BB NF_F32_BB
float nf_dense_load(uint ws, uint off) { return nf_w_f32(ws, off); }
#elif QTYPE == QT_F16
#define QT_BB NF_F16_BB
float nf_dense_load(uint ws, uint off) { return nf_w_f16(ws, off); }
#else
#define QT_BB NF_BF16_BB
float nf_dense_load(uint ws, uint off) { return nf_w_bf16(ws, off); }
#endif

float qt_dot(uint ws, uint bp, uint xoff, uint blk, uint ltid) {
    return nf_dense_load(ws, bp) * qt_x(xoff + blk);
}

float qt_elem(uint ws, uint rbase, uint i) {
    return nf_dense_load(ws, rbase + uint(QT_BB) * i);
}

#endif /* dense QTYPE */
#endif /* QTYPE */

#endif /* NF_QUANT_BLOCKS_GLSL */

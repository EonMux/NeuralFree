/* common/kv_types.glsl - KV cache element types (NF_KV_* codes) and element loaders over aliased byte views. Requires: common/nf_common.glsl */
/* Portions derived from llama.cpp/ggml (MIT License); see THIRD-PARTY-NOTICES.md. */
/*
 * GLSL twin of the codes and loaders of kernels/llm/common/kv_types.h (the semantic reference: layouts, decode formulas
 * and llama.cpp references); the stores are in common/kv_quant.glsl. The codes equal gguf.memory.KvDType.code().
 *
 * NF_KV_LOADERS(name), after NF_BUFFER_VIEWS(binding, name) or NF_BUFFER_VIEWS_RW(binding, name), defines
 *     vec4  name_kv_load4(uint e, int t)   values e .. e + 3 (e % 4 == 0) of a cache of type t (NF_KV_*)
 *     float name_kv_load1(uint e, int t)   value e
 * with e counted in values from the start of the binding. Every block starts at an even byte, so the scale, the
 * min and the (unaligned) Q5 qh word are read through the 16-bit view.
 */
#ifndef NF_KV_TYPES_GLSL
#define NF_KV_TYPES_GLSL

#ifndef NF_COMMON_GLSL
#error "common/kv_types.glsl requires common/nf_common.glsl"
#endif

#define NF_KV_F32    0
#define NF_KV_F16    1
#define NF_KV_Q8_0   2
#define NF_KV_BF16   3
#define NF_KV_Q4_0   4
#define NF_KV_Q4_1   5
#define NF_KV_IQ4_NL 6
#define NF_KV_Q5_0   7
#define NF_KV_Q5_1   8

/* Values per block of every block type. */
const uint NF_KV_BLOCK = 32u;

/* kvalues_iq4nl[q] for q in [0, 16) (llama.cpp ggml-common.h:1120), int8 values packed little-endian. */
const uvec4 NF_KV_IQ4NL_VALUES = uvec4(0xBFAD9881u, 0xF6EADDCFu, 0x26190D01u, 0x71594535u);

int nf_kv_iq4nl(uint q) {
    return bitfieldExtract(int(NF_KV_IQ4NL_VALUES[q >> 2]), int((q & 3u) * 8u), 8);
}

/* True for the 32-value block types. */
bool nf_kv_is_block(int t) {
    return t != NF_KV_F32 && t != NF_KV_F16 && t != NF_KV_BF16;
}

/* Bytes of one block (block types) or one value (F32, F16, BF16). */
uint nf_kv_unit_bytes(int t) {
    if (t == NF_KV_F32) return 4u;
    if (t == NF_KV_F16 || t == NF_KV_BF16) return 2u;
    if (t == NF_KV_Q8_0) return 34u;
    if (t == NF_KV_Q4_1) return 20u;
    if (t == NF_KV_Q5_0) return 22u;
    if (t == NF_KV_Q5_1) return 24u;
    return 18u; /* Q4_0, IQ4_NL */
}

/*
 * Value of the 4- or 5-bit code q (fifth bit included) of a block with scale d and min m: Q4_0 (q - 8) d, Q4_1
 * q d + m, IQ4_NL d kvalues[q], Q5_0 (q - 16) d, Q5_1 q d + m (dequantize_row_*, llama.cpp ggml-quants.c:328-548,
 * 2726-2743).
 */
float nf_kv_code_value(int t, uint q, float d, float m) {
    if (t == NF_KV_Q4_0) return float(int(q) - 8) * d;
    if (t == NF_KV_IQ4_NL) return d * float(nf_kv_iq4nl(q));
    if (t == NF_KV_Q5_0) return float(int(q) - 16) * d;
    return float(q) * d + m; /* Q4_1, Q5_1 */
}

#define NF_KV_LOADERS(name) \
    vec4 name##_kv_block4(uint e, int t, uint n) { \
        const uint base = (e >> 5u) * nf_kv_unit_bytes(t); \
        const uint j = e & 31u; \
        const float d = nf_half(uint(name##_u16[base >> 1u])); \
        vec4 v = vec4(0.0); \
        if (t == NF_KV_Q8_0) { \
            for (uint i = 0u; i < n; ++i) { \
                v[i] = d * float(nf_i8(uint(name##_u8[base + 2u + j + i]))); \
            } \
            return v; \
        } \
        const bool has_m = t == NF_KV_Q4_1 || t == NF_KV_Q5_1; \
        const float m = has_m ? nf_half(uint(name##_u16[(base >> 1u) + 1u])) : 0.0; \
        const uint qh_w = t == NF_KV_Q5_0 ? 1u : 2u; \
        const bool five = t == NF_KV_Q5_0 || t == NF_KV_Q5_1; \
        const uint qh = five ? (uint(name##_u16[(base >> 1u) + qh_w]) \
                                | (uint(name##_u16[(base >> 1u) + qh_w + 1u]) << 16)) : 0u; \
        const uint qs = base + (t == NF_KV_Q5_1 ? 8u : (t == NF_KV_Q5_0 ? 6u : (has_m ? 4u : 2u))); \
        const uint sh = (j >> 4u) << 2u; \
        for (uint i = 0u; i < n; ++i) { \
            const uint q = ((uint(name##_u8[qs + ((j + i) & 15u)]) >> sh) & 0xFu) | (((qh >> (j + i)) & 1u) << 4u); \
            v[i] = nf_kv_code_value(t, q, d, m); \
        } \
        return v; \
    } \
    vec4 name##_kv_load4(uint e, int t) { \
        if (t == NF_KV_F32) { \
            return vec4(uintBitsToFloat(name##_u32[e]), uintBitsToFloat(name##_u32[e + 1u]), \
                        uintBitsToFloat(name##_u32[e + 2u]), uintBitsToFloat(name##_u32[e + 3u])); \
        } \
        if (t == NF_KV_F16) { \
            return vec4(nf_half(uint(name##_u16[e])), nf_half(uint(name##_u16[e + 1u])), \
                        nf_half(uint(name##_u16[e + 2u])), nf_half(uint(name##_u16[e + 3u]))); \
        } \
        if (t == NF_KV_BF16) { \
            return vec4(nf_bf16(uint(name##_u16[e])), nf_bf16(uint(name##_u16[e + 1u])), \
                        nf_bf16(uint(name##_u16[e + 2u])), nf_bf16(uint(name##_u16[e + 3u]))); \
        } \
        return name##_kv_block4(e, t, 4u); \
    } \
    float name##_kv_load1(uint e, int t) { \
        if (t == NF_KV_F32) return uintBitsToFloat(name##_u32[e]); \
        if (t == NF_KV_F16) return nf_half(uint(name##_u16[e])); \
        if (t == NF_KV_BF16) return nf_bf16(uint(name##_u16[e])); \
        return name##_kv_block4(e, t, 1u).x; \
    }

#endif /* NF_KV_TYPES_GLSL */

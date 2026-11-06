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

#define NF_KV_F32      0
#define NF_KV_F16      1
#define NF_KV_Q8_0     2
#define NF_KV_BF16     3
#define NF_KV_Q4_0     4
#define NF_KV_Q4_1     5
#define NF_KV_IQ4_NL   6
#define NF_KV_Q5_0     7
#define NF_KV_Q5_1     8
#define NF_KV_FP8_E4M3 9
#define NF_KV_FP8_E5M2 10
#define NF_KV_FP8_PER_TOKEN_HEAD  11
#define NF_KV_INT8_PER_TOKEN_HEAD 12
#define NF_KV_INT4_PER_TOKEN_HEAD 13
#define NF_KV_NVFP4        14
#define NF_KV_NVFP4_4OVER6 15
#define NF_KV_TQ_K8_KEYS   16
#define NF_KV_TQ_4BIT_KEYS 17
#define NF_KV_TQ_3BIT_KEYS 18
#define NF_KV_TQ_4BIT_VALUES 19
#define NF_KV_TQ_3BIT_VALUES 20
#define NF_KV_FP8_DS_MLA   21
#define NF_KV_NVFP4_DS_MLA 22

/* Values per block of every block type. */
const uint NF_KV_BLOCK = 32u;

/* kvalues_iq4nl[q] for q in [0, 16) (llama.cpp ggml-common.h:1120), int8 values packed little-endian. */
const uvec4 NF_KV_IQ4NL_VALUES = uvec4(0xBFAD9881u, 0xF6EADDCFu, 0x26190D01u, 0x71594535u);

int nf_kv_iq4nl(uint q) {
    return bitfieldExtract(int(NF_KV_IQ4NL_VALUES[q >> 2]), int((q & 3u) * 8u), 8);
}

/* True for the per-token-head types: one code per value under a per-(token, head) scale row. */
bool nf_kv_is_per_token_head(int t) {
    return t == NF_KV_FP8_PER_TOKEN_HEAD || t == NF_KV_INT8_PER_TOKEN_HEAD || t == NF_KV_INT4_PER_TOKEN_HEAD;
}

/* True for the 32-value block types. */
bool nf_kv_is_block(int t) {
    return t != NF_KV_F32 && t != NF_KV_F16 && t != NF_KV_BF16
        && t != NF_KV_FP8_E4M3 && t != NF_KV_FP8_E5M2 && !nf_kv_is_per_token_head(t);
}

/* Bytes of one block (block types) or one value (F32, F16, BF16, FP8; INT4 packs two values per byte). */
uint nf_kv_unit_bytes(int t) {
    if (t == NF_KV_F32) return 4u;
    if (t == NF_KV_F16 || t == NF_KV_BF16) return 2u;
    if (t == NF_KV_FP8_E4M3 || t == NF_KV_FP8_E5M2) return 1u;
    if (nf_kv_is_per_token_head(t)) return 1u;
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

/* E4M3FN code c to float (F8_E4M3_LUT of common/quant_blocks.h; the NaN codes 0x7F and 0xFF give 0). */
float nf_kv_e4m3(uint c) {
    if ((c & 0x7Fu) == 0x7Fu) {
        return 0.0;
    }
    const uint e = (c >> 3) & 0xFu;
    const float v = float((c & 7u) | (min(e, 1u) << 3)) * uintBitsToFloat((max(e, 1u) + 117u) << 23);
    return (c & 0x80u) != 0u ? -v : v;
}

/* E5M2 code c to float: the code is the high byte of an IEEE half (Fp8Codec.E5M2), infinities included. */
float nf_kv_e5m2(uint c) {
    return nf_half((c & 0xFFu) << 8);
}

/* Value of one element of an FP8 cache (Fp8Codec decode; common/kv_types.h nf_kv_load1). */
float nf_kv_fp8_value(int t, uint c) {
    return t == NF_KV_FP8_E4M3 || t == NF_KV_FP8_PER_TOKEN_HEAD ? nf_kv_e4m3(c) : nf_kv_e5m2(c);
}

/* Unscaled value of a per-token-head code (attention multiplies it by the row's per-head scale). */
float nf_kv_pt_value(int t, uint c) {
    if (t == NF_KV_FP8_PER_TOKEN_HEAD) return nf_kv_e4m3(c);
    if (t == NF_KV_INT8_PER_TOKEN_HEAD) return float(nf_i8(c));
    const int n = int(c & 0xFu);
    return float((n & 8) != 0 ? n - 16 : n);
}

/* Largest code magnitude of a per-token-head type: 448 (E4M3FN), 127 (INT8) or 7 (INT4). */
float nf_kv_pt_qmax(int t) {
    if (t == NF_KV_FP8_PER_TOKEN_HEAD) return 448.0;
    return t == NF_KV_INT8_PER_TOKEN_HEAD ? 127.0 : 7.0;
}

/* ---- vendor formats (twin of the vendor section of common/kv_types.h) ---- */

bool nf_kv_is_vendor(int t) { return t >= NF_KV_NVFP4 && t <= NF_KV_NVFP4_DS_MLA; }
bool nf_kv_is_tq_mse(int t) { return t == NF_KV_TQ_4BIT_KEYS || t == NF_KV_TQ_3BIT_KEYS; }
bool nf_kv_is_tq_value(int t) { return t == NF_KV_TQ_4BIT_VALUES || t == NF_KV_TQ_3BIT_VALUES; }
int nf_kv_tq_bits(int t) {
    return (t == NF_KV_TQ_3BIT_KEYS || t == NF_KV_TQ_3BIT_VALUES) ? 3 : 4;
}

const uint NF_KV_NVFP4_GROUP = 16u;
const uint NF_KV_DS_MLA_NOPE = 512u;
const uint NF_KV_DS_MLA_ROPE = 64u;
const uint NF_KV_DS_MLA_TILE = 128u;
const uint NF_KV_TQ_MAX_DIM = 256u;
const uint NF_KV_VENDOR_MAX_ROW = 576u;
const uint NF_KV_SIDE_CTR_FLOATS = 16u;

/*
 * Byte offset of the packed plane rows in the side buffer of a vendor cache (common/kv_types.h
 * nf_kv_side_plane_base): the 2 * n_head_kv checkpoint KV scales, then the Lloyd-Max centroid table. The side buffer
 * is the flash kernels' kv_scales binding.
 */
uint nf_kv_side_plane_base(uint n_head_kv) {
    return 4u * (2u * n_head_kv + NF_KV_SIDE_CTR_FLOATS);
}

/* E2M1 code to its value (KvVendorCodec.e2m1Value). */
float nf_kv_e2m1_value(uint c) {
    const uint m = c & 7u;
    float v = 0.0;
    if (m == 1u) v = 0.5;
    else if (m == 2u) v = 1.0;
    else if (m == 3u) v = 1.5;
    else if (m == 4u) v = 2.0;
    else if (m == 5u) v = 3.0;
    else if (m == 6u) v = 4.0;
    else if (m == 7u) v = 6.0;
    return (c & 8u) != 0u ? -v : v;
}

/* E2M1 code of x, round to nearest with ties to even and saturation to 6 (KvVendorCodec.e2m1Code). */
uint nf_kv_e2m1_code(float x) {
    const float ax = abs(x);
    uint m;
    if (ax <= 0.25) m = 0u;
    else if (ax < 0.75) m = 1u;
    else if (ax <= 1.25) m = 2u;
    else if (ax < 1.75) m = 3u;
    else if (ax <= 2.5) m = 4u;
    else if (ax < 3.5) m = 5u;
    else if (ax <= 5.0) m = 6u;
    else m = 7u;
    return (floatBitsToUint(x) & 0x80000000u) != 0u ? m | 8u : m;
}

/* Bytes of one row of the packed side planes of a vendor cache of width row_width across heads KV heads. */
uint nf_kv_vendor_plane_bytes(int t, uint row_width, uint heads) {
    if (t == NF_KV_NVFP4 || t == NF_KV_NVFP4_4OVER6) return row_width / NF_KV_NVFP4_GROUP;
    if (t == NF_KV_TQ_4BIT_KEYS || t == NF_KV_TQ_3BIT_KEYS) return 2u * heads;
    if (t == NF_KV_TQ_4BIT_VALUES || t == NF_KV_TQ_3BIT_VALUES) return 4u * heads;
    if (t == NF_KV_FP8_DS_MLA) return 16u;
    if (t == NF_KV_NVFP4_DS_MLA) return 32u;
    return 0u;
}

/* Bytes of one row of a vendor cache of width row_width values (KvDType.rowBytes). */
uint nf_kv_vendor_row_bytes(int t, uint row_width) {
    if (t == NF_KV_NVFP4 || t == NF_KV_NVFP4_4OVER6) return row_width / 2u;
    if (t == NF_KV_FP8_DS_MLA) return 640u;
    if (t == NF_KV_NVFP4_DS_MLA) return 320u;
    if (t == NF_KV_TQ_K8_KEYS) return row_width;
    return row_width * uint(nf_kv_tq_bits(t)) / 8u;
}

/* Code at index of a packed TurboQuant stream whose bytes start at byte_base of the cache view. */
#define NF_KV_TQ_CODE(cache, byte_base, index, bits)     ((bits) == 4         ? (((index) & 1u) == 0u             ? (uint(cache##_u8[(byte_base) + ((index) >> 1u)]) & 0xFu)             : ((uint(cache##_u8[(byte_base) + ((index) >> 1u)]) >> 4u) & 0xFu))         : ((uint(cache##_u8[(byte_base) + (((index) * 3u) >> 3u)])             | (uint(cache##_u8[(byte_base) + (((index) * 3u) >> 3u) + 1u]) << 8))            >> (((index) * 3u) & 7u)) & 7u)

/*
 * F16 plane value at half index i of a byte-addressed plane view (SSBO runtime arrays cannot be function
 * parameters, so the buffer name is part of the macro). i is evaluated twice.
 */
#define NF_KV_PLANE_HALF(planes, i) \
    nf_half(uint(planes##_u8[2u * (i)]) | (uint(planes##_u8[2u * (i) + 1u]) << 8))

/*
 * Generates the vendor decode of one cache side: name##_vendor_load1(e, row, t, row_width, head_elems, plane_base,
 * plane_stride) is the value of element e of row `row`, decoded from the packed codes in the byte view `cache` and
 * the packed side planes in the byte view `side` at plane_base + row * plane_stride, with the Lloyd-Max table in the
 * float view `ctr` at 2 * heads (common/kv_types.h nf_kv_vendor_load1).
 */
#define NF_KV_VENDOR_DECODE(name, cache, side, ctr)\
    float name##_vendor_load1(uint e, int row, int t, uint row_width, uint head_elems, uint plane_base,\
                              uint plane_stride) {\
        const uint in_row = e - uint(row) * row_width;\
        const uint heads = row_width / head_elems;\
        const uint pl = plane_base + uint(row) * plane_stride;\
        const uint row_bytes = uint(row) * nf_kv_vendor_row_bytes(t, row_width);\
        if (t == NF_KV_NVFP4 || t == NF_KV_NVFP4_4OVER6) {\
            const uint b = uint(cache##_u8[row_bytes + (in_row >> 1u)]);\
            const uint code = (in_row & 1u) == 0u ? (b & 0xFu) : ((b >> 4u) & 0xFu);\
            return nf_kv_e2m1_value(code) * nf_kv_e4m3(uint(side##_u8[pl + (in_row / NF_KV_NVFP4_GROUP)]));\
        }\
        if (t == NF_KV_TQ_K8_KEYS) return nf_kv_e4m3(uint(cache##_u8[row_bytes + in_row]));\
        if (t == NF_KV_FP8_DS_MLA) {\
            if (in_row < NF_KV_DS_MLA_NOPE) {\
                const uint t4 = pl + 4u * (in_row / NF_KV_DS_MLA_TILE);\
                const float sc = uintBitsToFloat(uint(side##_u8[t4])\
                    | (uint(side##_u8[t4 + 1u]) << 8) | (uint(side##_u8[t4 + 2u]) << 16)\
                    | (uint(side##_u8[t4 + 3u]) << 24));\
                return nf_kv_e4m3(uint(cache##_u8[row_bytes + in_row])) * sc;\
            }\
            const uint at = row_bytes + NF_KV_DS_MLA_NOPE + 2u * (in_row - NF_KV_DS_MLA_NOPE);\
            return nf_bf16(uint(cache##_u8[at]) | (uint(cache##_u8[at + 1u]) << 8));\
        }\
        if (t == NF_KV_NVFP4_DS_MLA) {\
            if (in_row < NF_KV_DS_MLA_NOPE) {\
                const uint b = uint(cache##_u8[row_bytes + (in_row >> 1u)]);\
                const uint code = (in_row & 1u) == 0u ? (b & 0xFu) : ((b >> 4u) & 0xFu);\
                return nf_kv_e2m1_value(code) * nf_kv_e4m3(uint(side##_u8[pl + (in_row / NF_KV_NVFP4_GROUP)]));\
            }\
            return nf_kv_e4m3(uint(cache##_u8[row_bytes + NF_KV_DS_MLA_NOPE / 2u\
                + (in_row - NF_KV_DS_MLA_NOPE)]));\
        }\
        const uint bits = uint(nf_kv_tq_bits(t));\
        const uint code = NF_KV_TQ_CODE(cache, row_bytes, in_row, bits);\
        if (nf_kv_is_tq_mse(t)) return ctr[2u * heads + code];\
        const uint head = in_row / head_elems;\
        const float scale = NF_KV_PLANE_HALF(side, (pl >> 1u) + head);\
        const float zero = NF_KV_PLANE_HALF(side, (pl >> 1u) + heads + head);\
        return float(code) * scale + zero;\
    }\
    vec4 name##_vendor_load4(uint e, int row, int t, uint row_width, uint head_elems, uint plane_base,\
                             uint plane_stride) {\
        return vec4(name##_vendor_load1(e, row, t, row_width, head_elems, plane_base, plane_stride),\
                    name##_vendor_load1(e + 1u, row, t, row_width, head_elems, plane_base, plane_stride),\
                    name##_vendor_load1(e + 2u, row, t, row_width, head_elems, plane_base, plane_stride),\
                    name##_vendor_load1(e + 3u, row, t, row_width, head_elems, plane_base, plane_stride));\
    }\
    float name##_vendor_tq_score(uint e_head, int row, int t, uint row_width, uint head_elems, uint plane_base,\
                                 uint plane_stride) {\
        if (!nf_kv_is_tq_mse(t)) return 1.0;\
        const uint in_row = e_head - uint(row) * row_width;\
        const uint heads = row_width / head_elems;\
        const uint head = in_row / head_elems;\
        const uint pl = plane_base + uint(row) * plane_stride;\
        const float norm = NF_KV_PLANE_HALF(side, (pl >> 1u) + head);\
        const uint bits = uint(nf_kv_tq_bits(t));\
        const uint row_bytes = uint(row) * nf_kv_vendor_row_bytes(t, row_width);\
        const uint first = head * head_elems;\
        float sum = 0.0;\
        for (uint i = 0u; i < head_elems; ++i) {\
            const float c = ctr[2u * heads + NF_KV_TQ_CODE(cache, row_bytes, first + i, bits)];\
            sum += c * c;\
        }\
        return norm / sqrt(sum + 1e-16);\
    }

#define NF_KV_LOADERS(name) \
    vec4 name##_kv_fp8x4(uint e, int t) { \
        return vec4(nf_kv_fp8_value(t, uint(name##_u8[e])), nf_kv_fp8_value(t, uint(name##_u8[e + 1u])), \
                    nf_kv_fp8_value(t, uint(name##_u8[e + 2u])), nf_kv_fp8_value(t, uint(name##_u8[e + 3u]))); \
    } \
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
    vec4 name##_kv_pt4(uint e, int t) { \
        const uint b0 = uint(name##_u8[e >> 1u]); \
        const uint b1 = uint(name##_u8[(e >> 1u) + 1u]); \
        return vec4(nf_kv_pt_value(t, b0 & 0xFu), nf_kv_pt_value(t, (b0 >> 4) & 0xFu), \
                    nf_kv_pt_value(t, b1 & 0xFu), nf_kv_pt_value(t, (b1 >> 4) & 0xFu)); \
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
        if (t == NF_KV_FP8_E4M3 || t == NF_KV_FP8_E5M2 || t == NF_KV_FP8_PER_TOKEN_HEAD) { \
            return name##_kv_fp8x4(e, t); \
        } \
        if (t == NF_KV_INT8_PER_TOKEN_HEAD) { \
            return vec4(float(nf_i8(uint(name##_u8[e]))), float(nf_i8(uint(name##_u8[e + 1u]))), \
                        float(nf_i8(uint(name##_u8[e + 2u]))), float(nf_i8(uint(name##_u8[e + 3u])))); \
        } \
        if (t == NF_KV_INT4_PER_TOKEN_HEAD) { \
            return name##_kv_pt4(e, t); \
        } \
        return name##_kv_block4(e, t, 4u); \
    } \
    float name##_kv_load1(uint e, int t) { \
        if (t == NF_KV_F32) return uintBitsToFloat(name##_u32[e]); \
        if (t == NF_KV_F16) return nf_half(uint(name##_u16[e])); \
        if (t == NF_KV_BF16) return nf_bf16(uint(name##_u16[e])); \
        if (t == NF_KV_FP8_E4M3 || t == NF_KV_FP8_E5M2 || t == NF_KV_FP8_PER_TOKEN_HEAD) { \
            return nf_kv_fp8_value(t, uint(name##_u8[e])); \
        } \
        if (t == NF_KV_INT8_PER_TOKEN_HEAD) return float(nf_i8(uint(name##_u8[e]))); \
        if (t == NF_KV_INT4_PER_TOKEN_HEAD) { \
            const uint b = uint(name##_u8[e >> 1u]); \
            return nf_kv_pt_value(t, (e & 1u) == 0u ? (b & 0xFu) : ((b >> 4) & 0xFu)); \
        } \
        return name##_kv_block4(e, t, 1u).x; \
    }

#endif /* NF_KV_TYPES_GLSL */

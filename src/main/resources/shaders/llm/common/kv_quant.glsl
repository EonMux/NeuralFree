/* common/kv_quant.glsl - KV cache stores: F32 / F16 / BF16 values and the ggml reference quantizers of the KV block types. Requires: common/nf_common.glsl, common/rope_common.glsl, common/kv_types.glsl */
/* Portions derived from llama.cpp/ggml (MIT License); see THIRD-PARTY-NOTICES.md. */
/*
 * GLSL twin of the stores of kernels/llm/common/kv_types.h (the semantic reference, with the llama.cpp references of
 * each quantizer). F16 rounds to nearest even (nf_rope_f16_bits), BF16 as ggml_compute_fp32_to_bf16.
 *
 * nf_kv_encode_block(x, t, w) encodes the 32 values x of one block of type t as the 16-bit little-endian words
 * w[0, n) of the block's bytes and returns n (block bytes / 2). NF_KV_STORES(name), after
 * NF_BUFFER_VIEWS_RW(binding, name), defines
 *     void name_kv_store1(uint e, float v, int t)          value e of an F32, F16 or BF16 cache
 *     void name_kv_store_block(uint e, float x[32], int t)  the block holding value e (e % 32 == 0) of a block
 *                                                           cache, written as 16-bit words (every block starts at
 *                                                           an even byte)
 */
#ifndef NF_KV_QUANT_GLSL
#define NF_KV_QUANT_GLSL

#ifndef NF_COMMON_GLSL
#error "common/kv_quant.glsl requires common/nf_common.glsl"
#endif
#ifndef NF_ROPE_COMMON_GLSL
#error "common/kv_quant.glsl requires common/rope_common.glsl"
#endif
#ifndef NF_KV_TYPES_GLSL
#error "common/kv_quant.glsl requires common/kv_types.glsl"
#endif

/* Largest block of 16-bit words (Q8_0, 34 bytes). */
const uint NF_KV_MAX_WORDS = 17u;

/* f32 -> bfloat16 bits, round to nearest even, NaN kept quiet (ggml_compute_fp32_to_bf16,
 * llama.cpp ggml-impl.h:623-636). */
uint nf_kv_bf16_bits(float f) {
    const uint u = floatBitsToUint(f);
    if ((u & 0x7FFFFFFFu) > 0x7F800000u) {
        return ((u >> 16) | 64u) & 0xFFFFu;
    }
    return (u + (0x7FFFu + ((u >> 16) & 1u))) >> 16;
}

/* Index of the IQ4_NL value nearest x (best_index_int8, llama.cpp ggml-quants.c:28-37). */
uint nf_kv_iq4nl_index(float x) {
    if (x <= float(nf_kv_iq4nl(0u))) return 0u;
    if (x >= float(nf_kv_iq4nl(15u))) return 15u;
    uint ml = 0u;
    uint mu = 15u;
    while (mu - ml > 1u) {
        const uint mav = (ml + mu) / 2u;
        if (x < float(nf_kv_iq4nl(mav))) {
            mu = mav;
        } else {
            ml = mav;
        }
    }
    return x - float(nf_kv_iq4nl(mu - 1u)) < float(nf_kv_iq4nl(mu)) - x ? mu - 1u : mu;
}

/* Q8_0 (quantize_row_q8_0_ref): d = amax/127, q = round(x/d) half away from zero. */
uint nf_kv_encode_q8_0(float x[32], out uint w[NF_KV_MAX_WORDS]) {
    float amax = 0.0;
    for (uint j = 0u; j < 32u; ++j) {
        amax = max(amax, abs(x[j]));
    }
    const float d = amax / 127.0;
    const float id = d != 0.0 ? 1.0 / d : 0.0;
    w[0] = nf_rope_f16_bits(d);
    for (uint j = 0u; j < 32u; j += 2u) {
        const uint lo = uint(clamp(int(nf_roundf(x[j] * id)), -127, 127)) & 0xFFu;
        const uint hi = uint(clamp(int(nf_roundf(x[j + 1u] * id)), -127, 127)) & 0xFFu;
        w[1u + j / 2u] = lo | (hi << 8);
    }
    return 17u;
}

/* Q4_0 / Q5_0 (quantize_row_q4_0_ref / _q5_0_ref): d = max / -2^(bits-1) of the value with the largest |x|. */
uint nf_kv_encode_sym(float x[32], bool five, out uint w[NF_KV_MAX_WORDS]) {
    float amax = 0.0;
    float vmax = 0.0;
    for (uint j = 0u; j < 32u; ++j) {
        if (amax < abs(x[j])) {
            amax = abs(x[j]);
            vmax = x[j];
        }
    }
    const float d = vmax / (five ? -16.0 : -8.0);
    const float id = d != 0.0 ? 1.0 / d : 0.0;
    const float bias = five ? 16.5 : 8.5;
    const int top = five ? 31 : 15;
    const uint q0 = five ? 3u : 1u;
    w[0] = nf_rope_f16_bits(d);
    uint qh = 0u;
    for (uint j = 0u; j < 16u; j += 2u) {
        uint packed = 0u;
        for (uint k = 0u; k < 2u; ++k) {
            const uint xi0 = uint(min(top, int(x[j + k] * id + bias)));
            const uint xi1 = uint(min(top, int(x[j + k + 16u] * id + bias)));
            packed |= ((xi0 & 0xFu) | ((xi1 & 0xFu) << 4)) << (8u * k);
            qh |= ((xi0 & 0x10u) >> 4) << (j + k);
            qh |= ((xi1 & 0x10u) >> 4) << (j + k + 16u);
        }
        w[q0 + j / 2u] = packed;
    }
    if (five) {
        w[1] = qh & 0xFFFFu;
        w[2] = qh >> 16;
    }
    return five ? 11u : 9u;
}

/* Q4_1 / Q5_1 (quantize_row_q4_1_ref / _q5_1_ref): d = (max - min) / (2^bits - 1), m = min. */
uint nf_kv_encode_affine(float x[32], bool five, out uint w[NF_KV_MAX_WORDS]) {
    float vmin = x[0];
    float vmax = x[0];
    for (uint j = 1u; j < 32u; ++j) {
        vmin = x[j] < vmin ? x[j] : vmin;
        vmax = x[j] > vmax ? x[j] : vmax;
    }
    const float d = (vmax - vmin) / (five ? 31.0 : 15.0);
    const float id = d != 0.0 ? 1.0 / d : 0.0;
    const uint q0 = five ? 4u : 2u;
    w[0] = nf_rope_f16_bits(d);
    w[1] = nf_rope_f16_bits(vmin);
    uint qh = 0u;
    for (uint j = 0u; j < 16u; j += 2u) {
        uint packed = 0u;
        for (uint k = 0u; k < 2u; ++k) {
            const int r0 = int((x[j + k] - vmin) * id + 0.5);
            const int r1 = int((x[j + k + 16u] - vmin) * id + 0.5);
            const uint xi0 = five ? uint(r0) & 0xFFu : uint(min(15, r0));
            const uint xi1 = five ? uint(r1) & 0xFFu : uint(min(15, r1));
            packed |= ((xi0 & 0xFu) | ((xi1 & 0xFu) << 4)) << (8u * k);
            qh |= ((xi0 & 0x10u) >> 4) << (j + k);
            qh |= ((xi1 & 0x10u) >> 4) << (j + k + 16u);
        }
        w[q0 + j / 2u] = packed;
    }
    if (five) {
        w[2] = qh & 0xFFFFu;
        w[3] = qh >> 16;
    }
    return five ? 12u : 10u;
}

/*
 * IQ4_NL (quantize_row_iq4_nl_ref): indices of x / (max / kvalues[0]), then d = sum(w q x) / sum(w q^2) with
 * w = x^2 in j order; a block with max |x| < 1e-15 (GROUP_MAX_EPS) stores d = 0 and zero indices.
 */
uint nf_kv_encode_iq4_nl(float x[32], out uint w[NF_KV_MAX_WORDS]) {
    float amax = 0.0;
    float vmax = 0.0;
    for (uint j = 0u; j < 32u; ++j) {
        if (abs(x[j]) > amax) {
            amax = abs(x[j]);
            vmax = x[j];
        }
    }
    const bool live = amax >= 1e-15;
    const float id = live ? 1.0 / (vmax / float(nf_kv_iq4nl(0u))) : 0.0;
    float sumqx = 0.0;
    float sumq2 = 0.0;
    uint l[32];
    for (uint j = 0u; j < 32u; ++j) {
        l[j] = live ? nf_kv_iq4nl_index(id * x[j]) : 0u;
        const float q = float(nf_kv_iq4nl(l[j]));
        const float wt = x[j] * x[j];
        sumqx += wt * q * x[j];
        sumq2 += wt * q * q;
    }
    const float d = live && sumq2 > 0.0 ? sumqx / sumq2 : 0.0;
    w[0] = nf_rope_f16_bits(d);
    for (uint j = 0u; j < 16u; j += 2u) {
        w[1u + j / 2u] = (l[j] | (l[j + 16u] << 4)) | ((l[j + 1u] | (l[j + 17u] << 4)) << 8);
    }
    return 9u;
}

uint nf_kv_encode_block(float x[32], int t, out uint w[NF_KV_MAX_WORDS]) {
    if (t == NF_KV_Q8_0) return nf_kv_encode_q8_0(x, w);
    if (t == NF_KV_Q4_0 || t == NF_KV_Q5_0) return nf_kv_encode_sym(x, t == NF_KV_Q5_0, w);
    if (t == NF_KV_Q4_1 || t == NF_KV_Q5_1) return nf_kv_encode_affine(x, t == NF_KV_Q5_1, w);
    return nf_kv_encode_iq4_nl(x, w);
}

#define NF_KV_STORES(name) \
    void name##_kv_store1(uint e, float v, int t) { \
        if (t == NF_KV_F16) { \
            name##_u16[e] = uint16_t(nf_rope_f16_bits(v)); \
        } else if (t == NF_KV_BF16) { \
            name##_u16[e] = uint16_t(nf_kv_bf16_bits(v)); \
        } else { \
            name##_u32[e] = floatBitsToUint(v); \
        } \
    } \
    void name##_kv_store_block(uint e, float x[32], int t) { \
        uint w[NF_KV_MAX_WORDS]; \
        const uint n = nf_kv_encode_block(x, t, w); \
        const uint wb = ((e >> 5u) * nf_kv_unit_bytes(t)) >> 1u; \
        for (uint i = 0u; i < n; ++i) { \
            name##_u16[wb + i] = uint16_t(w[i]); \
        } \
    }

#endif /* NF_KV_QUANT_GLSL */

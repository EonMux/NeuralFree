/* common/kv_types.h - KV cache element types (NF_KV_* codes), element loaders and element / block stores for F32, F16, BF16, Q8_0, Q4_0, Q4_1, IQ4_NL, Q5_0 and Q5_1 caches. Requires: common/dialect.h, common/config.h, common/quant_blocks.h, common/quant_decode_legacy.h */
/* Portions derived from llama.cpp/ggml (MIT License); see THIRD-PARTY-NOTICES.md. */
/*
 * The cache types llama.cpp accepts for -ctk / -ctv (llama.cpp common/arg.cpp:304-314). The codes equal
 * gguf.memory.KvDType.code(). GLSL twins: shaders/llm/common/kv_types.glsl (loaders) and
 * shaders/llm/common/kv_quant.glsl (stores).
 *
 * A cache is a byte buffer; element e counts values from the start of the buffer:
 *   F32, F16, BF16   element e of a float, half or bfloat16 array
 *   block types      block e / 32 of the ggml block struct (Q8_0 34 B, Q4_0 18 B, Q4_1 20 B, IQ4_NL 18 B,
 *                    Q5_0 22 B, Q5_1 24 B; common/quant_blocks.h), value j = e % 32 in the ggml order of
 *                    dequantize_row_* (llama.cpp ggml-quants.c:328-548, 2726-2743): Q8_0 qs[j]; the 4/5-bit types hold
 *                    j < 16 in the low nibble of qs[j] and j >= 16 in the high nibble of qs[j - 16], and the
 *                    fifth bit of value j is bit j of the little-endian qh word.
 * Hosts keep every row, head and slot of a block cache whole blocks (KvDType.supportsWidth), so a four-value load
 * at e % 4 == 0 stays inside one block and one nibble half.
 *
 * Stores follow the ggml reference quantizers of each type (llama.cpp ggml-quants.c quantize_row_*_ref, which the CUDA
 * set_rows copies repeat in ggml-cuda/cpy-utils.cuh): Q8_0 :276-299, Q4_0 :113-147, Q4_1 :149-182, Q5_0 :184-222,
 * Q5_1 :224-260, IQ4_NL :5072-5085 (quantize_row_iq4_nl_impl :4938-5047 with ntry = -1); F16 rounds to nearest
 * even and BF16 by ggml_compute_fp32_to_bf16 (llama.cpp ggml-impl.h:623-636).
 */
#ifndef NF_KV_TYPES_H
#define NF_KV_TYPES_H

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
#define NF_KV_BLOCK 32

/* Smallest |x| of a block that IQ4_NL quantizes with a non-zero scale (GROUP_MAX_EPS, llama.cpp ggml-quants.c:20). */
#define NF_KV_IQ4_NL_EPS 1e-15f

/* 1 for the 32-value block types, 0 for F32, F16 and BF16. */
NF_INLINE int nf_kv_is_block(int kv_type) {
    return kv_type != NF_KV_F32 && kv_type != NF_KV_F16 && kv_type != NF_KV_BF16;
}

/* Bytes of one block (block types) or one value (F32, F16, BF16). */
NF_INLINE int nf_kv_unit_bytes(int kv_type) {
    if (kv_type == NF_KV_F32) return 4;
    if (kv_type == NF_KV_F16 || kv_type == NF_KV_BF16) return 2;
    if (kv_type == NF_KV_Q8_0) return Q8_0_BLOCK_BYTES;
    if (kv_type == NF_KV_Q4_1) return Q4_1_BLOCK_BYTES;
    if (kv_type == NF_KV_Q5_0) return Q5_0_BLOCK_BYTES;
    if (kv_type == NF_KV_Q5_1) return Q5_1_BLOCK_BYTES;
    return Q4_0_BLOCK_BYTES; /* Q4_0, IQ4_NL */
}

/* ---- loaders ---- */

/*
 * n (1 or 4) consecutive values e .. e + n - 1 of a block cache, the rest of the result 0. For n == 4,
 * e % 4 == 0 keeps the values in one block and one nibble half.
 */
NF_INLINE float4 nf_kv_block_load(NF_GLOBAL const uchar* p, nf_i64 e, int kv_type, int n) {
    NF_GLOBAL const uchar* bp = p + (e >> 5) * nf_kv_unit_bytes(kv_type);
    const int j = (int)(e & 31);
    const int k = j & 15;
    const int sh = (j >> 4) << 2;
    const float d = nf_ql_half(bp, 0);
    float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (kv_type == NF_KV_Q8_0) {
        NF_GLOBAL const nf_i8* qs = (NF_GLOBAL const nf_i8*)(bp + 2) + j;
        for (int i = 0; i < n; i++) v[i] = d * (float)qs[i];
    } else if (kv_type == NF_KV_Q4_0) {
        for (int i = 0; i < n; i++) v[i] = (float)((int)((bp[2 + k + i] >> sh) & 0xFu) - 8) * d;
    } else if (kv_type == NF_KV_IQ4_NL) {
        for (int i = 0; i < n; i++) v[i] = d * IQ4_NL_KVALUES[(bp[2 + k + i] >> sh) & 0xFu];
    } else if (kv_type == NF_KV_Q4_1) {
        const float m = nf_ql_half(bp, 2);
        for (int i = 0; i < n; i++) v[i] = (float)((bp[4 + k + i] >> sh) & 0xFu) * d + m;
    } else if (kv_type == NF_KV_Q5_0) {
        const uint qh = read_u32_le(bp + 2);
        for (int i = 0; i < n; i++) {
            const uint q = ((bp[6 + k + i] >> sh) & 0xFu) | nf_ql_q5_hbit(qh, j + i);
            v[i] = (float)((int)q - 16) * d;
        }
    } else {
        const float m = nf_ql_half(bp, 2);
        const uint qh = read_u32_le(bp + 4);
        for (int i = 0; i < n; i++) {
            const uint q = ((bp[8 + k + i] >> sh) & 0xFu) | nf_ql_q5_hbit(qh, j + i);
            v[i] = (float)q * d + m;
        }
    }
    return NF_F4(v[0], v[1], v[2], v[3]);
}

/* Four consecutive values e .. e + 3 (e % 4 == 0) of a cache of type kv_type (NF_KV_*). */
NF_INLINE float4 nf_kv_load4(NF_GLOBAL const uchar* p, nf_i64 e, int kv_type) {
    if (kv_type == NF_KV_F32) {
        return vload4(0, (NF_GLOBAL const float*)p + e);
    }
    if (kv_type == NF_KV_F16) {
        return vload_half4(0, (NF_GLOBAL const NF_HALF*)p + e);
    }
    if (kv_type == NF_KV_BF16) {
        NF_GLOBAL const ushort* b = (NF_GLOBAL const ushort*)p + e;
        return NF_F4(bf16_to_f32(b[0]), bf16_to_f32(b[1]), bf16_to_f32(b[2]), bf16_to_f32(b[3]));
    }
    return nf_kv_block_load(p, e, kv_type, 4);
}

/* One value e of a cache of type kv_type (NF_KV_*). */
NF_INLINE float nf_kv_load1(NF_GLOBAL const uchar* p, nf_i64 e, int kv_type) {
    if (kv_type == NF_KV_F32) {
        return ((NF_GLOBAL const float*)p)[e];
    }
    if (kv_type == NF_KV_F16) {
        return vload_half(0, (NF_GLOBAL const NF_HALF*)p + e);
    }
    if (kv_type == NF_KV_BF16) {
        return bf16_to_f32(((NF_GLOBAL const ushort*)p)[e]);
    }
    return nf_kv_block_load(p, e, kv_type, 1).x;
}

/* ---- stores ---- */

/* f32 -> bfloat16 bits, round to nearest even, NaN kept quiet (ggml_compute_fp32_to_bf16). */
NF_INLINE ushort nf_f32_to_bf16(float f) {
    const uint u = as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) {
        return (ushort)((u >> 16) | 64u);
    }
    return (ushort)((u + (0x7fffu + ((u >> 16) & 1u))) >> 16);
}

/* Stores value e of an F32, F16 or BF16 cache. */
NF_INLINE void nf_kv_store1(NF_GLOBAL uchar* p, nf_i64 e, float v, int kv_type) {
    if (kv_type == NF_KV_F16) {
        vstore_half(v, 0, (NF_GLOBAL NF_HALF*)p + e);
    } else if (kv_type == NF_KV_BF16) {
        ((NF_GLOBAL ushort*)p)[e] = nf_f32_to_bf16(v);
    } else {
        ((NF_GLOBAL float*)p)[e] = v;
    }
}

/* Index of the IQ4_NL value nearest x (best_index_int8 over kvalues_iq4nl, llama.cpp ggml-quants.c:28-37). */
NF_INLINE int nf_kv_iq4nl_index(float x) {
    if (x <= IQ4_NL_KVALUES[0]) return 0;
    if (x >= IQ4_NL_KVALUES[15]) return 15;
    int ml = 0;
    int mu = 15;
    while (mu - ml > 1) {
        const int mav = (ml + mu) / 2;
        if (x < IQ4_NL_KVALUES[mav]) {
            mu = mav;
        } else {
            ml = mav;
        }
    }
    return x - IQ4_NL_KVALUES[mu - 1] < IQ4_NL_KVALUES[mu] - x ? mu - 1 : mu;
}

/* Writes the little-endian u32 v at bp (any alignment). */
NF_INLINE void nf_kv_write_u32(NF_GLOBAL uchar* bp, uint v) {
    bp[0] = (uchar)(v & 0xFFu);
    bp[1] = (uchar)((v >> 8) & 0xFFu);
    bp[2] = (uchar)((v >> 16) & 0xFFu);
    bp[3] = (uchar)(v >> 24);
}

/* Q8_0 (quantize_row_q8_0_ref): d = amax/127, q = round(x/d) half away from zero. */
NF_INLINE void nf_kv_quantize_q8_0(const float* x, NF_GLOBAL uchar* bp) {
    float amax = 0.0f;
    for (int j = 0; j < NF_KV_BLOCK; j++) {
        amax = fmax(amax, fabs(x[j]));
    }
    const float d = amax / 127.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    vstore_half(d, 0, (NF_GLOBAL NF_HALF*)bp);
    for (int j = 0; j < NF_KV_BLOCK; j++) {
        bp[2 + j] = (uchar)(clamp((int)round(x[j] * id), -127, 127) & 0xFF);
    }
}

/* Q4_0 / Q5_0 (quantize_row_q4_0_ref / _q5_0_ref): d = max / -2^(bits-1) of the value with the largest |x|. */
NF_INLINE void nf_kv_quantize_sym(const float* x, NF_GLOBAL uchar* bp, int five) {
    float amax = 0.0f;
    float max = 0.0f;
    for (int j = 0; j < NF_KV_BLOCK; j++) {
        if (amax < fabs(x[j])) {
            amax = fabs(x[j]);
            max = x[j];
        }
    }
    const float d = max / (five ? -16.0f : -8.0f);
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    const float bias = five ? 16.5f : 8.5f;
    const int top = five ? 31 : 15;
    vstore_half(d, 0, (NF_GLOBAL NF_HALF*)bp);
    uint qh = 0u;
    NF_GLOBAL uchar* qs = bp + (five ? 6 : 2);
    for (int j = 0; j < NF_KV_BLOCK / 2; j++) {
        const uint xi0 = (uint)min(top, (int)(x[j] * id + bias));
        const uint xi1 = (uint)min(top, (int)(x[j + 16] * id + bias));
        qs[j] = (uchar)((xi0 & 0xFu) | ((xi1 & 0xFu) << 4));
        qh |= ((xi0 & 0x10u) >> 4) << j;
        qh |= ((xi1 & 0x10u) >> 4) << (j + 16);
    }
    if (five) {
        nf_kv_write_u32(bp + 2, qh);
    }
}

/* Q4_1 / Q5_1 (quantize_row_q4_1_ref / _q5_1_ref): d = (max - min) / (2^bits - 1), m = min. */
NF_INLINE void nf_kv_quantize_affine(const float* x, NF_GLOBAL uchar* bp, int five) {
    float vmin = x[0];
    float vmax = x[0];
    for (int j = 1; j < NF_KV_BLOCK; j++) {
        vmin = x[j] < vmin ? x[j] : vmin;
        vmax = x[j] > vmax ? x[j] : vmax;
    }
    const float d = (vmax - vmin) / (five ? 31.0f : 15.0f);
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    vstore_half(d, 0, (NF_GLOBAL NF_HALF*)bp);
    vstore_half(vmin, 1, (NF_GLOBAL NF_HALF*)bp);
    uint qh = 0u;
    NF_GLOBAL uchar* qs = bp + (five ? 8 : 4);
    for (int j = 0; j < NF_KV_BLOCK / 2; j++) {
        const int r0 = (int)((x[j] - vmin) * id + 0.5f);
        const int r1 = (int)((x[j + 16] - vmin) * id + 0.5f);
        const uint xi0 = five ? (uint)r0 & 0xFFu : (uint)min(15, r0);
        const uint xi1 = five ? (uint)r1 & 0xFFu : (uint)min(15, r1);
        qs[j] = (uchar)((xi0 & 0xFu) | ((xi1 & 0xFu) << 4));
        qh |= ((xi0 & 0x10u) >> 4) << j;
        qh |= ((xi1 & 0x10u) >> 4) << (j + 16);
    }
    if (five) {
        nf_kv_write_u32(bp + 4, qh);
    }
}

/*
 * IQ4_NL (quantize_row_iq4_nl_ref): d0 = max / kvalues[0] of the value with the largest |x|, indices
 * l_j = best_index(x_j / d0), then the weighted least-squares scale d = sum(w q x) / sum(w q^2) with w = x^2,
 * accumulated in j order. A block with max |x| < NF_KV_IQ4_NL_EPS stores d = 0 and zero indices (llama.cpp
 * leaves those indices unset; every index decodes to 0 under d = 0).
 */
NF_INLINE void nf_kv_quantize_iq4_nl(const float* x, NF_GLOBAL uchar* bp) {
    float amax = 0.0f;
    float max = 0.0f;
    for (int j = 0; j < NF_KV_BLOCK; j++) {
        if (fabs(x[j]) > amax) {
            amax = fabs(x[j]);
            max = x[j];
        }
    }
    const int live = amax >= NF_KV_IQ4_NL_EPS;
    const float id = live ? 1.0f / (max / IQ4_NL_KVALUES[0]) : 0.0f;
    float sumqx = 0.0f;
    float sumq2 = 0.0f;
    int l[NF_KV_BLOCK];
    for (int j = 0; j < NF_KV_BLOCK; j++) {
        l[j] = live ? nf_kv_iq4nl_index(id * x[j]) : 0;
        const float q = IQ4_NL_KVALUES[l[j]];
        const float w = x[j] * x[j];
        sumqx += w * q * x[j];
        sumq2 += w * q * q;
    }
    const float d = live && sumq2 > 0.0f ? sumqx / sumq2 : 0.0f;
    vstore_half(d, 0, (NF_GLOBAL NF_HALF*)bp);
    for (int j = 0; j < NF_KV_BLOCK / 2; j++) {
        bp[2 + j] = (uchar)(l[j] | (l[j + 16] << 4));
    }
}

/* Quantizes the 32 values x[0, 32) into the block at bp of a block cache of type kv_type. */
NF_INLINE void nf_kv_quantize_block(const float* x, NF_GLOBAL uchar* bp, int kv_type) {
    if (kv_type == NF_KV_Q8_0) {
        nf_kv_quantize_q8_0(x, bp);
    } else if (kv_type == NF_KV_Q4_0 || kv_type == NF_KV_Q5_0) {
        nf_kv_quantize_sym(x, bp, kv_type == NF_KV_Q5_0);
    } else if (kv_type == NF_KV_Q4_1 || kv_type == NF_KV_Q5_1) {
        nf_kv_quantize_affine(x, bp, kv_type == NF_KV_Q5_1);
    } else {
        nf_kv_quantize_iq4_nl(x, bp);
    }
}

/* Block holding element e (e % 32 == 0) of a block cache. */
NF_INLINE NF_GLOBAL uchar* nf_kv_block_ptr(NF_GLOBAL uchar* p, nf_i64 e, int kv_type) {
    return p + (e >> 5) * nf_kv_unit_bytes(kv_type);
}

/*
 * Work-group store of row[0, width) (local memory) as cache elements [e0, e0 + width): one work-item per value for
 * F32 / F16 / BF16, one per 32-value block for the block types (e0 and width multiples of 32). No barrier inside;
 * the caller synchronizes before reusing row.
 */
NF_INLINE void nf_kv_store_local_row(NF_GLOBAL uchar* p, nf_i64 e0, NF_LOCAL_PTR const float* row, int width,
                                     int kv_type) {
    const int lid = get_local_id(0);
    const int lsize = get_local_size(0);
    if (!nf_kv_is_block(kv_type)) {
        for (int c = lid; c < width; c += lsize) {
            nf_kv_store1(p, e0 + c, row[c], kv_type);
        }
        return;
    }
    for (int b = lid; b < width / NF_KV_BLOCK; b += lsize) {
        float x[NF_KV_BLOCK];
        for (int j = 0; j < NF_KV_BLOCK; j++) {
            x[j] = row[b * NF_KV_BLOCK + j];
        }
        nf_kv_quantize_block(x, nf_kv_block_ptr(p, e0 + (nf_i64)b * NF_KV_BLOCK, kv_type), kv_type);
    }
}

/* Store of the 32 global values src[0, 32) as the block holding element e (e % 32 == 0) of a block cache. */
NF_INLINE void nf_kv_store_global_block(NF_GLOBAL uchar* p, nf_i64 e, NF_GLOBAL const float* src, int kv_type) {
    float x[NF_KV_BLOCK];
    for (int j = 0; j < NF_KV_BLOCK; j++) {
        x[j] = src[j];
    }
    nf_kv_quantize_block(x, nf_kv_block_ptr(p, e, kv_type), kv_type);
}

#endif /* NF_KV_TYPES_H */

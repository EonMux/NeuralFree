/* common/kv_types.h - KV cache element types (NF_KV_* codes), element loaders and element / block stores for F32, F16, BF16, FP8_E4M3, FP8_E5M2, Q8_0, Q4_0, Q4_1, IQ4_NL, Q5_0 and Q5_1 caches. Requires: common/dialect.h, common/config.h, common/quant_blocks.h, common/quant_decode_legacy.h */
/* Portions derived from llama.cpp/ggml (MIT License); see THIRD-PARTY-NOTICES.md. */
/*
 * The cache types llama.cpp accepts for -ctk / -ctv (llama.cpp common/arg.cpp:304-314) and the FP8 KV cache of
 * vLLM (--kv-cache-dtype fp8_e4m3 / fp8_e5m2, vllm/config/cache.py CacheDType). The codes equal
 * gguf.memory.KvDType.code(). GLSL twins: shaders/llm/common/kv_types.glsl (loaders) and
 * shaders/llm/common/kv_quant.glsl (stores).
 *
 * A cache is a byte buffer; element e counts values from the start of the buffer:
 *   F32, F16, BF16   element e of a float, half or bfloat16 array
 *   FP8_E4M3, FP8_E5M2  element e, one OCP FP8 code (Fp8Codec.E4M3FN / .E5M2): E4M3 decodes through F8_E4M3_LUT
 *                    (the NaN codes give 0, as for the weight decoders; the store encoders never produce them),
 *                    E5M2 decodes as the high byte of an IEEE half (half_bits_to_float), infinities included
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
 *
 * FP8 stores encode the RNE saturating code of the value (nf_kv_fp8_e4m3_bits / nf_kv_fp8_e5m2_bits). The KV cache
 * divides by the layer's checkpoint KV scale before the store and attention multiplies the decoded values by it
 * (vLLM reshape_and_cache_fp8: cvt_float_to_fp8(a / scale, SATFINITE), and its flash attention reads
 * fp8(code) * scale; vllm csrc/cache_kernels.cu, quantization/w8a8/fp8/nvidia/quant_utils.cuh). The encoders repeat
 * ggml_fp32_to_f8_e4m3 (llama.cpp ggml-impl.h) with Fp8Codec's special cases: magnitudes at the largest finite value
 * and above (the infinities included) saturate to it, NaN goes to the NaN code with its sign.
 */
#ifndef NF_KV_TYPES_H
#define NF_KV_TYPES_H

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
#define NF_KV_BLOCK 32

/* Smallest |x| of a block that IQ4_NL quantizes with a non-zero scale (GROUP_MAX_EPS, llama.cpp ggml-quants.c:20). */
#define NF_KV_IQ4_NL_EPS 1e-15f

/* 1 for the per-token-head types: one code per value under a per-(token, head) scale row. */
NF_INLINE int nf_kv_is_per_token_head(int kv_type) {
    return kv_type == NF_KV_FP8_PER_TOKEN_HEAD || kv_type == NF_KV_INT8_PER_TOKEN_HEAD
        || kv_type == NF_KV_INT4_PER_TOKEN_HEAD;
}

/* 1 for the 32-value block types, 0 for F32, F16, BF16, the FP8 types and the per-token-head types. */
NF_INLINE int nf_kv_is_block(int kv_type) {
    return kv_type != NF_KV_F32 && kv_type != NF_KV_F16 && kv_type != NF_KV_BF16
        && kv_type != NF_KV_FP8_E4M3 && kv_type != NF_KV_FP8_E5M2 && !nf_kv_is_per_token_head(kv_type);
}

/* Bytes of one block (block types) or one value (F32, F16, BF16, FP8; INT4 packs two values per byte). */
NF_INLINE int nf_kv_unit_bytes(int kv_type) {
    if (kv_type == NF_KV_F32) return 4;
    if (kv_type == NF_KV_F16 || kv_type == NF_KV_BF16) return 2;
    if (kv_type == NF_KV_FP8_E4M3 || kv_type == NF_KV_FP8_E5M2) return 1;
    if (nf_kv_is_per_token_head(kv_type)) return 1;
    if (kv_type == NF_KV_Q8_0) return Q8_0_BLOCK_BYTES;
    if (kv_type == NF_KV_Q4_1) return Q4_1_BLOCK_BYTES;
    if (kv_type == NF_KV_Q5_0) return Q5_0_BLOCK_BYTES;
    if (kv_type == NF_KV_Q5_1) return Q5_1_BLOCK_BYTES;
    return Q4_0_BLOCK_BYTES; /* Q4_0, IQ4_NL */
}

/* ---- per-token-head codes ---- */

/* Largest code magnitude of a per-token-head type: 448 (E4M3FN), 127 (INT8) or 7 (INT4). */
NF_INLINE float nf_kv_pt_qmax(int kv_type) {
    if (kv_type == NF_KV_FP8_PER_TOKEN_HEAD) return 448.0f;
    return kv_type == NF_KV_INT8_PER_TOKEN_HEAD ? 127.0f : 7.0f;
}

/* Unscaled value of a per-token-head code (attention multiplies it by the row's per-head scale). */
NF_INLINE float nf_kv_pt_value(int kv_type, int code) {
    if (kv_type == NF_KV_FP8_PER_TOKEN_HEAD) {
        return F8_E4M3_LUT[code & 0xFF];
    }
    if (kv_type == NF_KV_INT8_PER_TOKEN_HEAD) {
        return (float)(nf_i8)(code & 0xFF);
    }
    const int n = code & 0xF;
    return (float)((n & 0x8) != 0 ? n - 16 : n);
}

/* One value of a per-token-head cache (INT4 values pack two per byte, value 2i in the low nibble). */
NF_INLINE float nf_kv_pt_load1(NF_GLOBAL const uchar* p, nf_i64 e, int kv_type) {
    if (kv_type == NF_KV_INT4_PER_TOKEN_HEAD) {
        const int b = p[e >> 1];
        return nf_kv_pt_value(kv_type, (e & 1) == 0 ? b & 0xF : (b >> 4) & 0xF);
    }
    return nf_kv_pt_value(kv_type, p[e]);
}

/* Four consecutive values e .. e + 3 of a per-token-head cache. */
NF_INLINE float4 nf_kv_pt_load4(NF_GLOBAL const uchar* p, nf_i64 e, int kv_type) {
    return NF_F4(nf_kv_pt_load1(p, e, kv_type), nf_kv_pt_load1(p, e + 1, kv_type),
                 nf_kv_pt_load1(p, e + 2, kv_type), nf_kv_pt_load1(p, e + 3, kv_type));
}

/* Stores code c of value e of a per-token-head cache (an INT4 write keeps the other nibble). */
NF_INLINE void nf_kv_pt_store1(NF_GLOBAL uchar* dst, nf_i64 e, int c, int kv_type) {
    if (kv_type == NF_KV_INT4_PER_TOKEN_HEAD) {
        const nf_i64 b = e >> 1;
        const int old = dst[b];
        dst[b] = (uchar)((e & 1) == 0 ? ((old & 0xF0) | (c & 0xF)) : ((old & 0x0F) | ((c & 0xF) << 4)));
        return;
    }
    dst[e] = (uchar)c;
}

/*
 * The scale of one (row, head) of K (v_side 0) or V (v_side 1): the row's per-(token, KV head) scale of a
 * per-token-head cache, or the checkpoint per-head scale of an FP8 cache (n_head_kv K values then n_head_kv V
 * values). A per-token-head cache stores its scales interleaved by row: [row0 K heads, row0 V heads, row1 ...].
 */
NF_INLINE float nf_kv_head_scale(int kv_type, NF_GLOBAL const float* kv_scales, int row, int head, int n_head_kv,
                                 int v_side) {
    if (nf_kv_is_per_token_head(kv_type)) {
        return kv_scales[((nf_i64)row * 2 + v_side) * n_head_kv + head];
    }
    return kv_scales[v_side * n_head_kv + head];
}

/* ---- vendor formats ---- */

/* 1 for the vLLM vendor formats (NVFP4, the TurboQuant streams and the DeepSeek MLA latent rows). */
NF_INLINE int nf_kv_is_vendor(int kv_type) {
    return kv_type >= NF_KV_NVFP4 && kv_type <= NF_KV_NVFP4_DS_MLA;
}

/* 1 for the TurboQuant key streams quantized to Lloyd-Max centroids. */
NF_INLINE int nf_kv_is_tq_mse(int kv_type) {
    return kv_type == NF_KV_TQ_4BIT_KEYS || kv_type == NF_KV_TQ_3BIT_KEYS;
}

/* 1 for the TurboQuant value streams (uniform codes under a per-head affine scale and zero). */
NF_INLINE int nf_kv_is_tq_value(int kv_type) {
    return kv_type == NF_KV_TQ_4BIT_VALUES || kv_type == NF_KV_TQ_3BIT_VALUES;
}

/* 1 for the DeepSeek MLA latent row formats. */
NF_INLINE int nf_kv_is_latent(int kv_type) {
    return kv_type == NF_KV_FP8_DS_MLA || kv_type == NF_KV_NVFP4_DS_MLA;
}

/* Code width of a TurboQuant stream (3 or 4); 4 for every other type. */
NF_INLINE int nf_kv_tq_bits(int kv_type) {
    return (kv_type == NF_KV_TQ_3BIT_KEYS || kv_type == NF_KV_TQ_3BIT_VALUES) ? 3 : 4;
}

/* E2M1 code to its value: magnitudes 0, 0.5, 1, 1.5, 2, 3, 4, 6, signed by bit 3 (KvVendorCodec.e2m1Value). */
NF_INLINE float nf_kv_e2m1_value(int c) {
    const int m = c & 7;
    float v = 0.0f;
    if (m == 1) v = 0.5f;
    else if (m == 2) v = 1.0f;
    else if (m == 3) v = 1.5f;
    else if (m == 4) v = 2.0f;
    else if (m == 5) v = 3.0f;
    else if (m == 6) v = 4.0f;
    else if (m == 7) v = 6.0f;
    return (c & 8) != 0 ? -v : v;
}

/* E2M1 code of x, round to nearest with ties to even and saturation to 6 (KvVendorCodec.e2m1Code). */
NF_INLINE int nf_kv_e2m1_code(float x) {
    const float ax = fabs(x);
    int m;
    if (ax <= 0.25f) m = 0;
    else if (ax < 0.75f) m = 1;
    else if (ax <= 1.25f) m = 2;
    else if (ax < 1.75f) m = 3;
    else if (ax <= 2.5f) m = 4;
    else if (ax < 3.5f) m = 5;
    else if (ax <= 5.0f) m = 6;
    else m = 7;
    return (as_uint(x) & 0x80000000u) != 0u ? m | 8 : m;
}

/* Values per NVFP4 scale group and per DeepSeek MLA NoPE tile of fp8_ds_mla. */
#define NF_KV_NVFP4_GROUP 16
#define NF_KV_DS_MLA_TILE 128
/* NoPE and RoPE values of one DeepSeek MLA latent row. */
#define NF_KV_DS_MLA_NOPE 512
#define NF_KV_DS_MLA_ROPE 64
/* Largest TurboQuant key head width a device store rotates (private work-item buffer). */
#define NF_KV_TQ_MAX_DIM 256
/* Floats of the Lloyd-Max centroid table at the head of a vendor side buffer (the 16 levels of a 4-bit stream). */
#define NF_KV_SIDE_CTR_FLOATS 16

/*
 * Byte offset of the packed plane rows in the side buffer of a vendor cache: the 2 * n_head_kv checkpoint KV scales
 * (K heads then V heads, the values attention multiplies the decoded rows by), then the Lloyd-Max centroid table.
 * The side buffer is the flash kernels' kv_scales binding, so the scales and the planes the attention needs travel
 * together and no kernel takes a second side-plane binding.
 */
NF_INLINE int nf_kv_side_plane_base(int n_head_kv) {
    return 4 * (2 * n_head_kv + NF_KV_SIDE_CTR_FLOATS);
}

/* Bytes of one row of the packed side planes of a vendor cache of width row_width across n_heads KV heads. */
NF_INLINE int nf_kv_vendor_plane_bytes(int kv_type, int row_width, int n_heads) {
    if (kv_type == NF_KV_NVFP4 || kv_type == NF_KV_NVFP4_4OVER6) {
        return row_width / NF_KV_NVFP4_GROUP;
    }
    if (kv_type == NF_KV_TQ_4BIT_KEYS || kv_type == NF_KV_TQ_3BIT_KEYS) {
        return 2 * n_heads;
    }
    if (kv_type == NF_KV_TQ_4BIT_VALUES || kv_type == NF_KV_TQ_3BIT_VALUES) {
        return 4 * n_heads;
    }
    if (kv_type == NF_KV_FP8_DS_MLA) {
        return 4 * (int)sizeof(float);
    }
    if (kv_type == NF_KV_NVFP4_DS_MLA) {
        return NF_KV_DS_MLA_NOPE / NF_KV_NVFP4_GROUP;
    }
    return 0;
}

/* Bytes of one row of a vendor cache of width row_width values (KvDType.rowBytes). */
NF_INLINE int nf_kv_vendor_row_bytes(int kv_type, int row_width) {
    if (kv_type == NF_KV_NVFP4 || kv_type == NF_KV_NVFP4_4OVER6) {
        return row_width / 2;
    }
    if (kv_type == NF_KV_FP8_DS_MLA) {
        return 640;
    }
    if (kv_type == NF_KV_NVFP4_DS_MLA) {
        return 320;
    }
    if (kv_type == NF_KV_TQ_K8_KEYS) {
        return row_width;
    }
    return row_width * nf_kv_tq_bits(kv_type) / 8;
}

/* Code at index of a 3- or 4-bit packed TurboQuant stream starting at byte_base (KvVendorCodec.turboQuantCode). */
NF_INLINE int nf_kv_tq_code(NF_GLOBAL const uchar* p, nf_i64 byte_base, int index, int bits) {
    if (bits == 4) {
        const int b = p[byte_base + (index >> 1)];
        return (index & 1) == 0 ? (b & 0xF) : ((b >> 4) & 0xF);
    }
    const int bit = index * 3;
    const nf_i64 at = byte_base + (bit >> 3);
    const int sh = bit & 7;
    int raw = (int)p[at];
    if (sh > 5) {
        raw |= (int)p[at + 1] << 8;
    }
    return (raw >> sh) & 0x7;
}

/* Writes code c at index of a packed TurboQuant stream (bits 3 or 4) starting at byte_base. */
NF_INLINE void nf_kv_tq_store_code(NF_GLOBAL uchar* p, nf_i64 byte_base, int index, int bits, int c) {
    if (bits == 4) {
        const nf_i64 at = byte_base + (index >> 1);
        const int old = p[at];
        p[at] = (uchar)((index & 1) == 0 ? ((old & 0xF0) | (c & 0xF)) : ((old & 0x0F) | ((c & 0xF) << 4)));
        return;
    }
    const int bit = index * 3;
    const nf_i64 at = byte_base + (bit >> 3);
    const int sh = bit & 7;
    int raw = (int)p[at];
    if (sh > 5) {
        raw |= (int)p[at + 1] << 8;
    }
    raw = (raw & ~(7 << sh)) | ((c & 7) << sh);
    p[at] = (uchar)(raw & 0xFF);
    if (sh > 5) {
        p[at + 1] = (uchar)((raw >> 8) & 0xFF);
    }
}

/* Unscaled value of element e (absolute, in values) of a vendor cache, from its row and side planes. */
NF_INLINE float nf_kv_vendor_load1(NF_GLOBAL const uchar* p, nf_i64 e, int row, int kv_type,
                                   NF_GLOBAL const uchar* planes, int plane_stride, int row_width, int head_elems,
                                   NF_GLOBAL const float* centroids) {
    const int in_row = (int)(e - (nf_i64)row * row_width);
    const int heads = row_width / head_elems;
    NF_GLOBAL const uchar* pl = planes + (nf_i64)row * plane_stride;
    const nf_i64 row_bytes = (nf_i64)row * nf_kv_vendor_row_bytes(kv_type, row_width);
    if (kv_type == NF_KV_NVFP4 || kv_type == NF_KV_NVFP4_4OVER6) {
        const int b = p[row_bytes + (in_row >> 1)];
        const int code = (in_row & 1) == 0 ? (b & 0xF) : ((b >> 4) & 0xF);
        return nf_kv_e2m1_value(code) * F8_E4M3_LUT[pl[in_row / NF_KV_NVFP4_GROUP]];
    }
    if (kv_type == NF_KV_TQ_K8_KEYS) {
        return F8_E4M3_LUT[p[row_bytes + in_row]];
    }
    if (kv_type == NF_KV_FP8_DS_MLA) {
        if (in_row < NF_KV_DS_MLA_NOPE) {
            const float scale = ((NF_GLOBAL const float*)pl)[in_row / NF_KV_DS_MLA_TILE];
            return F8_E4M3_LUT[p[row_bytes + in_row]] * scale;
        }
        const nf_i64 at = row_bytes + NF_KV_DS_MLA_NOPE + 2 * (in_row - NF_KV_DS_MLA_NOPE);
        return bf16_to_f32((ushort)((uint)p[at] | ((uint)p[at + 1] << 8)));
    }
    if (kv_type == NF_KV_NVFP4_DS_MLA) {
        if (in_row < NF_KV_DS_MLA_NOPE) {
            const int b = p[row_bytes + (in_row >> 1)];
            const int code = (in_row & 1) == 0 ? (b & 0xF) : ((b >> 4) & 0xF);
            return nf_kv_e2m1_value(code) * F8_E4M3_LUT[pl[in_row / NF_KV_NVFP4_GROUP]];
        }
        return F8_E4M3_LUT[p[row_bytes + NF_KV_DS_MLA_NOPE / 2 + (in_row - NF_KV_DS_MLA_NOPE)]];
    }
    const int bits = nf_kv_tq_bits(kv_type);
    const int code = nf_kv_tq_code(p, row_bytes, in_row, bits);
    if (nf_kv_is_tq_mse(kv_type)) {
        return centroids[code];
    }
    const int head = in_row / head_elems;
    const float scale = half_bits_to_float((uint)pl[2 * head] | ((uint)pl[2 * head + 1] << 8));
    const float zero = half_bits_to_float((uint)pl[2 * heads + 2 * head]
                                          | ((uint)pl[2 * heads + 2 * head + 1] << 8));
    return (float)code * scale + zero;
}

/* Four consecutive values e .. e + 3 (e % 4 == 0) of a vendor cache. */
NF_INLINE float4 nf_kv_vendor_load4(NF_GLOBAL const uchar* p, nf_i64 e, int row, int kv_type,
                                    NF_GLOBAL const uchar* planes, int plane_stride, int row_width, int head_elems,
                                    NF_GLOBAL const float* centroids) {
    return NF_F4(nf_kv_vendor_load1(p, e, row, kv_type, planes, plane_stride, row_width, head_elems, centroids),
                 nf_kv_vendor_load1(p, e + 1, row, kv_type, planes, plane_stride, row_width, head_elems, centroids),
                 nf_kv_vendor_load1(p, e + 2, row, kv_type, planes, plane_stride, row_width, head_elems, centroids),
                 nf_kv_vendor_load1(p, e + 3, row, kv_type, planes, plane_stride, row_width, head_elems, centroids));
}

/*
 * Score factor of one TurboQuant key row: the row's vector norm over the norm of its reconstructed centroid
 * vector (vLLM's NORM_CORRECTION path). The key is decoded as the centroid vector itself, so the attention must
 * rotate its query by the same normalized Hadamard matrix and multiply the resulting dot product by this factor.
 * 1 for every other type.
 */
NF_INLINE float nf_kv_vendor_tq_score(NF_GLOBAL const uchar* p, nf_i64 e_head, int row, int kv_type,
                                      NF_GLOBAL const uchar* planes, int plane_stride, int row_width,
                                      int head_elems, NF_GLOBAL const float* centroids) {
    if (!nf_kv_is_tq_mse(kv_type)) {
        return 1.0f;
    }
    const int in_row = (int)(e_head - (nf_i64)row * row_width);
    const int head = in_row / head_elems;
    const int heads = row_width / head_elems;
    NF_GLOBAL const uchar* pl = planes + (nf_i64)row * plane_stride;
    const float norm = half_bits_to_float((uint)pl[2 * head] | ((uint)pl[2 * head + 1] << 8));
    const int bits = nf_kv_tq_bits(kv_type);
    const nf_i64 row_bytes = (nf_i64)row * ((nf_i64)row_width * bits / 8);
    const int first = head * head_elems;
    float sum = 0.0f;
    for (int i = 0; i < head_elems; i++) {
        const float c = centroids[nf_kv_tq_code(p, row_bytes, first + i, bits)];
        sum += c * c;
    }
    return norm / sqrt(sum + 1e-16f);
}

/*
 * In-place normalized Sylvester Hadamard transform of a private row of n values (n a power of two): the same
 * butterfly and scale as nf_fwht_local (common/hadamard.h), which a work-item runs on its own registers.
 */
NF_INLINE void nf_kv_fwht_private(float* buf, int n) {
    for (int h = 1; h < n; h <<= 1) {
        for (int p = 0; p < (n >> 1); p++) {
            const int i = ((p & ~(h - 1)) << 1) | (p & (h - 1));
            const float a = buf[i];
            const float c = buf[i + h];
            buf[i] = a + c;
            buf[i + h] = a - c;
        }
    }
    const float scale = 1.0f / sqrt((float)n);
    for (int k = 0; k < n; k++) {
        buf[k] *= scale;
    }
}

/* Index of y among the sorted Lloyd-Max centroids: the count of midpoints not above y (KvVendorCodec.centroidIndex). */
NF_INLINE int nf_kv_centroid_index(float y, NF_GLOBAL const float* centroids, int levels) {
    int lo = 0;
    int hi = levels - 1;
    while (lo < hi) {
        const int mid = (lo + hi) >> 1;
        if (y >= 0.5f * (centroids[mid] + centroids[mid + 1])) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

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
    if (kv_type == NF_KV_FP8_E4M3) {
        NF_GLOBAL const uchar* c = p + e;
        return NF_F4(F8_E4M3_LUT[c[0]], F8_E4M3_LUT[c[1]], F8_E4M3_LUT[c[2]], F8_E4M3_LUT[c[3]]);
    }
    if (kv_type == NF_KV_FP8_E5M2) {
        NF_GLOBAL const uchar* c = p + e;
        return NF_F4(half_bits_to_float((uint)c[0] << 8), half_bits_to_float((uint)c[1] << 8),
                     half_bits_to_float((uint)c[2] << 8), half_bits_to_float((uint)c[3] << 8));
    }
    if (nf_kv_is_per_token_head(kv_type)) {
        return nf_kv_pt_load4(p, e, kv_type);
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
    if (kv_type == NF_KV_FP8_E4M3) {
        return F8_E4M3_LUT[p[e]];
    }
    if (kv_type == NF_KV_FP8_E5M2) {
        return half_bits_to_float((uint)p[e] << 8);
    }
    if (nf_kv_is_per_token_head(kv_type)) {
        return nf_kv_pt_load1(p, e, kv_type);
    }
    return nf_kv_block_load(p, e, kv_type, 1).x;
}

/* Element e of a KV cache as attention reads it: the vendor decode with its row's side planes, else the plain loader. */
NF_INLINE float nf_kv_attn_load1(NF_GLOBAL const uchar* p, nf_i64 e, int row, int kv_type,
                                 NF_GLOBAL const uchar* planes, int plane_stride, int row_width, int head_elems,
                                 NF_GLOBAL const float* centroids) {
    return nf_kv_is_vendor(kv_type)
        ? nf_kv_vendor_load1(p, e, row, kv_type, planes, plane_stride, row_width, head_elems, centroids)
        : nf_kv_load1(p, e, kv_type);
}

/* Four consecutive values e .. e + 3 (e % 4 == 0) of a KV cache as attention reads it. */
NF_INLINE float4 nf_kv_attn_load4(NF_GLOBAL const uchar* p, nf_i64 e, int row, int kv_type,
                                  NF_GLOBAL const uchar* planes, int plane_stride, int row_width, int head_elems,
                                  NF_GLOBAL const float* centroids) {
    return nf_kv_is_vendor(kv_type)
        ? nf_kv_vendor_load4(p, e, row, kv_type, planes, plane_stride, row_width, head_elems, centroids)
        : nf_kv_load4(p, e, kv_type);
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

/*
 * RNE saturating FP8 encodes (Fp8Codec.E4M3FN.encode / .E5M2.encode; the structure of ggml_fp32_to_f8_e4m3,
 * llama.cpp ggml-impl.h): NaN goes to the NaN code with its sign (0x7F / 0x7E), every other magnitude at the
 * largest finite value and above (the infinities included) saturates to it (0x7E / 0x7B), subnormals carry the
 * exponent of the smallest normal, and the mantissa units round to nearest even with the carry into the exponent.
 */
NF_INLINE uint nf_kv_fp8_e4m3_bits(float v) {
    const uint sign = (as_uint(v) >> 24) & 0x80u;
    const float m = fabs(v);
    if (isnan(m)) {
        return sign | 0x7Fu;
    }
    if (m >= 448.0f) {
        return sign | 0x7Eu;
    }
    if (m == 0.0f) {
        return sign;
    }
    const int ex0 = max((int)((as_uint(m) >> 23) & 0xFFu) - 127, -6);
    int units = (int)rint(ldexp(m, 3 - ex0));
    int ex = ex0;
    if (units < 8) {
        return sign | (uint)units;
    }
    if (units == 16) {
        units = 8;
        ex++;
    }
    return sign | (uint)min(((ex + 7) << 3) | (units - 8), 0x7E);
}

NF_INLINE uint nf_kv_fp8_e5m2_bits(float v) {
    const uint sign = (as_uint(v) >> 24) & 0x80u;
    const float m = fabs(v);
    if (isnan(m)) {
        return sign | 0x7Eu;
    }
    if (m >= 57344.0f) {
        return sign | 0x7Bu;
    }
    if (m == 0.0f) {
        return sign;
    }
    const int ex0 = max((int)((as_uint(m) >> 23) & 0xFFu) - 127, -14);
    int units = (int)rint(ldexp(m, 2 - ex0));
    int ex = ex0;
    if (units < 4) {
        return sign | (uint)units;
    }
    if (units == 8) {
        units = 4;
        ex++;
    }
    return sign | (uint)min(((ex + 15) << 2) | (units - 4), 0x7B);
}

/* Stores value e of an F32, F16, BF16, FP8_E4M3 or FP8_E5M2 cache. */
NF_INLINE void nf_kv_store1(NF_GLOBAL uchar* p, nf_i64 e, float v, int kv_type) {
    if (kv_type == NF_KV_F16) {
        vstore_half(v, 0, (NF_GLOBAL NF_HALF*)p + e);
    } else if (kv_type == NF_KV_BF16) {
        ((NF_GLOBAL ushort*)p)[e] = nf_f32_to_bf16(v);
    } else if (kv_type == NF_KV_FP8_E4M3) {
        p[e] = (uchar)nf_kv_fp8_e4m3_bits(v);
    } else if (kv_type == NF_KV_FP8_E5M2) {
        p[e] = (uchar)nf_kv_fp8_e5m2_bits(v);
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

/* Round-to-nearest-even code of a value already divided by its per-(token, head) scale. */
NF_INLINE int nf_kv_pt_encode(int kv_type, float v) {
    if (kv_type == NF_KV_FP8_PER_TOKEN_HEAD) {
        return (int)nf_kv_fp8_e4m3_bits(v);
    }
    if (kv_type == NF_KV_INT8_PER_TOKEN_HEAD) {
        return ((int)rint(fmax(-127.0f, fmin(127.0f, v)))) & 0xFF;
    }
    return ((int)rint(fmax(-7.0f, fmin(7.0f, v)))) & 0xF;
}


/* ---- vendor stores ---- */

/* Squared reconstruction error of one 16-value NVFP4 group under scale sf (KvVendorCodec.reconstructionError). */
NF_INLINE float nf_kv_nvfp4_error(NF_GLOBAL const float* x, int base, float sf) {
    const float out_scale = sf != 0.0f ? 1.0f / sf : 0.0f;
    const float dequant = out_scale != 0.0f ? 1.0f / out_scale : 0.0f;
    float err = 0.0f;
    for (int i = 0; i < NF_KV_NVFP4_GROUP; i++) {
        const float q = nf_kv_e2m1_value(nf_kv_e2m1_code(x[base + i] * out_scale)) * dequant;
        const float d = q - x[base + i];
        err += d * d;
    }
    return err;
}

/* Little-endian F16 store at half index i of a byte-addressed plane row. */
NF_INLINE void nf_kv_plane_set_half(NF_GLOBAL uchar* pl, int i, float v) {
    vstore_half(v, (size_t)i, (NF_GLOBAL NF_HALF*)pl);
}

/* Little-endian BF16 store at byte offset at of a byte-addressed row. */
NF_INLINE void nf_kv_store_bf16(NF_GLOBAL uchar* p, nf_i64 at, float v) {
    const ushort b = nf_f32_to_bf16(v);
    p[at] = (uchar)(b & 0xFF);
    p[at + 1] = (uchar)(b >> 8);
}

/*
 * Stores the head_elems values x[0, head_elems) of KV head `head` of one vendor cache row: the packed codes at
 * dst (the row's bytes), the head's share of the packed side planes at pl (the row's plane bytes), with in_row the
 * head's first value offset in the row and heads the row's KV head count. kv_scale is the layer's checkpoint KV
 * scale (NVFP4 only) and centroids the sorted Lloyd-Max centroids of a TurboQuant key stream. The formulas are those
 * of KvVendorCodec, shared with the host stores; a latent row is stored whole by its one head.
 */
NF_INLINE void nf_kv_vendor_store_head(NF_GLOBAL const float* x, int head_elems, int heads, NF_GLOBAL uchar* dst,
                                       NF_GLOBAL uchar* pl, int in_row, int head, int kv_type, float kv_scale,
                                       NF_GLOBAL const float* centroids) {
    if (kv_type == NF_KV_NVFP4 || kv_type == NF_KV_NVFP4_4OVER6) {
        const int groups = head_elems / NF_KV_NVFP4_GROUP;
        const float scale = kv_scale != 0.0f ? kv_scale : 1.0f;
        for (int g = 0; g < groups; g++) {
            const int base = g * NF_KV_NVFP4_GROUP;
            float amax = 0.0f;
            for (int i = 0; i < NF_KV_NVFP4_GROUP; i++) {
                amax = fmax(amax, fabs(x[base + i] / scale));
            }
            int code = (int)nf_kv_fp8_e4m3_bits(amax / 6.0f);
            if (kv_type == NF_KV_NVFP4_4OVER6) {
                const int code4 = (int)nf_kv_fp8_e4m3_bits(amax / 4.0f);
                const float sf6 = F8_E4M3_LUT[code];
                const float sf4 = F8_E4M3_LUT[code4];
                if (nf_kv_nvfp4_error(x, base, sf4) < nf_kv_nvfp4_error(x, base, sf6)) {
                    code = code4;
                }
            }
            pl[head * groups + g] = (uchar)code;
            const float sf = F8_E4M3_LUT[code];
            const float inv = sf != 0.0f ? 1.0f / sf : 0.0f;
            NF_GLOBAL uchar* hb = dst + (in_row >> 1) + g * (NF_KV_NVFP4_GROUP / 2);
            for (int i = 0; i < NF_KV_NVFP4_GROUP; i += 2) {
                const int lo = nf_kv_e2m1_code(x[base + i] / scale * inv);
                const int hi = nf_kv_e2m1_code(x[base + i + 1] / scale * inv);
                hb[i >> 1] = (uchar)(lo | (hi << 4));
            }
        }
        return;
    }
    if (kv_type == NF_KV_TQ_K8_KEYS) {
        for (int i = 0; i < head_elems; i++) {
            dst[in_row + i] = (uchar)nf_kv_fp8_e4m3_bits(x[i]);
        }
        return;
    }
    if (kv_type == NF_KV_TQ_4BIT_VALUES || kv_type == NF_KV_TQ_3BIT_VALUES) {
        const int bits = nf_kv_tq_bits(kv_type);
        const int levels = (1 << bits) - 1;
        float vmin = x[0];
        float vmax = x[0];
        for (int i = 1; i < head_elems; i++) {
            vmin = fmin(vmin, x[i]);
            vmax = fmax(vmax, x[i]);
        }
        float scale = (vmax - vmin) / (float)levels;
        scale = scale > 1e-8f ? scale : 1e-8f;
        nf_kv_plane_set_half(pl, head, scale);
        nf_kv_plane_set_half(pl, heads + head, vmin);
        const nf_i64 byte_base = (nf_i64)(in_row * bits / 8);
        for (int i = 0; i < head_elems; i++) {
            int q = (int)((x[i] - vmin) / scale + 0.5f);
            q = max(0, min(levels, q));
            nf_kv_tq_store_code(dst, byte_base, i, bits, q);
        }
        return;
    }
    if (kv_type == NF_KV_TQ_4BIT_KEYS || kv_type == NF_KV_TQ_3BIT_KEYS) {
        const int bits = nf_kv_tq_bits(kv_type);
        const int levels = 1 << bits;
        float sum = 0.0f;
        for (int i = 0; i < head_elems; i++) {
            sum += x[i] * x[i];
        }
        const float norm = sqrt(sum);
        float rot[NF_KV_TQ_MAX_DIM];
        for (int i = 0; i < head_elems; i++) {
            rot[i] = x[i] / (norm + 1e-8f);
        }
        nf_kv_fwht_private(rot, head_elems);
        nf_kv_plane_set_half(pl, head, norm);
        const nf_i64 byte_base = (nf_i64)(in_row * bits / 8);
        for (int i = 0; i < head_elems; i++) {
            nf_kv_tq_store_code(dst, byte_base, i, bits, nf_kv_centroid_index(rot[i], centroids, levels));
        }
        return;
    }
    if (kv_type == NF_KV_FP8_DS_MLA) {
        NF_GLOBAL float* tile = (NF_GLOBAL float*)pl;
        for (int t = 0; t < NF_KV_DS_MLA_NOPE / NF_KV_DS_MLA_TILE; t++) {
            const int base = t * NF_KV_DS_MLA_TILE;
            float amax = 0.0f;
            for (int i = 0; i < NF_KV_DS_MLA_TILE; i++) {
                amax = fmax(amax, fabs(x[base + i]));
            }
            float s = fmax(amax / 448.0f, 1e-4f);
            s = exp2(ceil(log2(s)));
            tile[t] = s;
            for (int i = 0; i < NF_KV_DS_MLA_TILE; i++) {
                dst[base + i] = (uchar)nf_kv_fp8_e4m3_bits(x[base + i] / s);
            }
        }
        for (int i = 0; i < NF_KV_DS_MLA_ROPE; i++) {
            nf_kv_store_bf16(dst, NF_KV_DS_MLA_NOPE + 2 * i, x[NF_KV_DS_MLA_NOPE + i]);
        }
        return;
    }
    for (int g = 0; g < NF_KV_DS_MLA_NOPE / NF_KV_NVFP4_GROUP; g++) {
        const int base = g * NF_KV_NVFP4_GROUP;
        float amax = 0.0f;
        for (int i = 0; i < NF_KV_NVFP4_GROUP; i++) {
            amax = fmax(amax, fabs(x[base + i]));
        }
        const int code = (int)nf_kv_fp8_e4m3_bits(fmax(amax / 6.0f, 0.001953125f));
        pl[g] = (uchar)code;
        const float sf = F8_E4M3_LUT[code];
        const float inv = sf != 0.0f ? 1.0f / sf : 0.0f;
        for (int i = 0; i < NF_KV_NVFP4_GROUP; i += 2) {
            const int lo = nf_kv_e2m1_code(x[base + i] * inv);
            const int hi = nf_kv_e2m1_code(x[base + i + 1] * inv);
            dst[g * (NF_KV_NVFP4_GROUP / 2) + (i >> 1)] = (uchar)(lo | (hi << 4));
        }
    }
    for (int i = 0; i < NF_KV_DS_MLA_ROPE; i++) {
        dst[NF_KV_DS_MLA_NOPE / 2 + i] = (uchar)nf_kv_fp8_e4m3_bits(x[NF_KV_DS_MLA_NOPE + i]);
    }
}


#endif /* NF_KV_TYPES_H */

/* common/attn_common.glsl - attention helpers (visibility, ALiBi, T5 buckets, softcap, online-softmax merge, kv_types decode, KV row loaders) shared by the flash shaders. Requires: common/nf_common.glsl, common/act.glsl, and common/kv_types.glsl (included first) for the kv_types decode and when KV is defined */
/*
 * GLSL twin of kernels/llm/common/attn_common.h plus the ALiBi slope and T5 bucket of kernels/llm/common/math.h; same
 * codes, formulas and llama.cpp references (see the C header). Softcap uses nf_tanh (common/act.glsl),
 * which stays finite for large arguments on every driver.
 *
 * KV loaders. KV (a variants-row define: one of the NF_KV_* codes of common/kv_types.glsl) selects the
 * element type of the cache bindings; NF_ATTN_KV_VIEWS(binding, name) declares the read-only views of one
 * binding and defines
 *     vec4  name_load4(uint e)   values e .. e + 3 (e % 4 == 0)
 *     float name_load1(uint e)   value e
 * with e counted in values from the start of the binding. F32, F16 and BF16 bindings are plain arrays; block
 * types read through the common/kv_types.glsl loaders.
 */
#ifndef NF_ATTN_COMMON_GLSL
#define NF_ATTN_COMMON_GLSL

#ifndef NF_COMMON_GLSL
#error "common/attn_common.glsl requires common/nf_common.glsl"
#endif
#ifndef NF_ACT_GLSL
#error "common/attn_common.glsl requires common/act.glsl"
#endif

/* swa_type codes: llama_swa_type == gguf.arch.SwaType.code() */
#define NF_SWA_NONE      0
#define NF_SWA_STANDARD  1
#define NF_SWA_CHUNKED   2
#define NF_SWA_SYMMETRIC 3

/* bias_mode codes == gguf.spec.AttentionSpec.BiasMode.code() */
#define NF_ATTN_BIAS_NONE      0
#define NF_ATTN_BIAS_ALIBI     1
#define NF_ATTN_BIAS_T5_CAUSAL 2
#define NF_ATTN_BIAS_T5_BIDIR  3

/* Head-dimension chunk of the tiled prefill shader. */
#define NF_FA_CHUNK 64

/* Causal mask plus llama_hparams::is_masked_swa (llama-hparams.h:506-540); p_k < 0 is an empty cell. */
int nf_attn_visible(int p_q, int p_k, int causal, int swa_type, int n_swa) {
    if (p_k < 0) {
        return 0;
    }
    if (causal != 0 && p_k > p_q) {
        return 0;
    }
    if (swa_type == NF_SWA_STANDARD) {
        return (p_q - p_k >= n_swa) ? 0 : 1;
    }
    if (swa_type == NF_SWA_CHUNKED) {
        return (p_k < (p_q / n_swa) * n_swa) ? 0 : 1;
    }
    if (swa_type == NF_SWA_SYMMETRIC) {
        const int half_n_swa = n_swa / 2;
        const int diff = p_q - p_k;
        return (diff < -half_n_swa || diff > half_n_swa) ? 0 : 1;
    }
    return 1;
}

/* Capped and scaled logit (ggml-cpu/ops.cpp:8997-8998, 9068-9072). */
float nf_attn_score(float dot_qk, float scale, float softcap) {
    if (softcap > 0.0) {
        return softcap * nf_tanh(dot_qk * (scale / softcap));
    }
    return dot_qk * scale;
}

/* ALiBi slope of head h (ggml-cpu/ops.cpp:9002-9005, 9024); 1 when max_bias <= 0. */
float nf_alibi_slope(int h, int n_head, float max_bias) {
    if (max_bias <= 0.0) {
        return 1.0;
    }
    int n_head_log2 = 1;
    while ((n_head_log2 << 1) <= n_head) {
        n_head_log2 <<= 1;
    }
    const float m0 = pow(2.0, -max_bias / float(n_head_log2));
    const float m1 = pow(2.0, -(max_bias / 2.0) / float(n_head_log2));
    const float base = h < n_head_log2 ? m0 : m1;
    const int exph = h < n_head_log2 ? h + 1 : 2 * (h - n_head_log2) + 1;
    return pow(base, float(exph));
}

/* llama_relative_position_bucket(x, y, n_buckets, bidirectional) (llama-graph.cpp:4316-4343). */
int nf_t5_bucket(int x, int y, int n_buckets, int max_distance, int bidirectional) {
    if (bidirectional != 0) {
        n_buckets >>= 1;
    }
    const int max_exact = n_buckets >> 1;
    int relative_position = x - y;
    int relative_bucket = 0;
    if (bidirectional != 0) {
        relative_bucket += (relative_position > 0) ? n_buckets : 0;
        relative_position = abs(relative_position);
    } else {
        relative_position = relative_position < 0 ? -relative_position : 0;
    }
    if (relative_position < max_exact) {
        return relative_bucket + relative_position;
    }
    const float scaled = log(float(relative_position) / float(max_exact)) * float(n_buckets - max_exact)
                         / log(float(max_distance) / float(max_exact));
    const int if_large = min(int(floor(float(max_exact) + scaled)), n_buckets - 1);
    return relative_bucket + if_large;
}

/* e^(x - m), exactly 0 when x is -INF; never evaluates exp of an infinity. */
float nf_attn_rescale(float x, float m) {
    return (x == -NF_INF) ? 0.0 : exp(x - m);
}

/* Sink merge of a finished softmax state: returns (factor for the unnormalized output, new sum). */
vec2 nf_attn_sink(float m, float l, float sink) {
    const float m2 = max(m, sink);
    const float a = nf_attn_rescale(m, m2);
    return vec2(a, l * a + exp(sink - m2));
}

#ifdef NF_KV_TYPES_GLSL
/*
 * kv_types argument of the mixed-type flash shaders (LLM_FlashDecode_Mixed, LLM_FlashAttn_Mixed): the K cache type in
 * bits 0-7 and the V cache type in bits 8-15, each an NF_KV_* code of common/kv_types.glsl; no other bit is set.
 */
#define NF_ATTN_KV_TYPE_BITS 8
#define NF_ATTN_KV_TYPE_MASK 0xFF

/* True when t is a KV cache type code (NF_KV_F32 .. NF_KV_Q5_1). */
bool nf_attn_kv_type_ok(int t) {
    return t >= NF_KV_F32 && t <= NF_KV_Q5_1;
}

/* K cache type of a kv_types argument, or -1 (rejected by nf_attn_kv_type_ok) when kv_types is malformed. */
int nf_attn_kv_type_k(int kv_types) {
    return (kv_types >> (2 * NF_ATTN_KV_TYPE_BITS)) == 0 ? (kv_types & NF_ATTN_KV_TYPE_MASK) : -1;
}

/* V cache type of a kv_types argument, or -1 (rejected by nf_attn_kv_type_ok) when kv_types is malformed. */
int nf_attn_kv_type_v(int kv_types) {
    return (kv_types >> (2 * NF_ATTN_KV_TYPE_BITS)) == 0
           ? ((kv_types >> NF_ATTN_KV_TYPE_BITS) & NF_ATTN_KV_TYPE_MASK) : -1;
}
#endif /* NF_KV_TYPES_GLSL */

#if defined(KV)

#ifndef NF_KV_TYPES_GLSL
#error "common/attn_common.glsl with KV requires common/kv_types.glsl"
#endif

#if KV == NF_KV_F32

#define NF_ATTN_KV_VIEWS(binding_index, name) \
    layout(std430, binding = binding_index) readonly buffer NfKvF32_##name { float name##_f32[]; }; \
    vec4 name##_load4(uint e) { \
        return vec4(name##_f32[e], name##_f32[e + 1u], name##_f32[e + 2u], name##_f32[e + 3u]); \
    } \
    float name##_load1(uint e) { return name##_f32[e]; }

#elif KV == NF_KV_F16

#define NF_ATTN_KV_VIEWS(binding_index, name) \
    layout(std430, binding = binding_index) readonly buffer NfKvF16_##name { uint16_t name##_u16[]; }; \
    vec4 name##_load4(uint e) { \
        return vec4(nf_half(uint(name##_u16[e])), nf_half(uint(name##_u16[e + 1u])), \
                    nf_half(uint(name##_u16[e + 2u])), nf_half(uint(name##_u16[e + 3u]))); \
    } \
    float name##_load1(uint e) { return nf_half(uint(name##_u16[e])); }

#elif KV == NF_KV_BF16

#define NF_ATTN_KV_VIEWS(binding_index, name) \
    layout(std430, binding = binding_index) readonly buffer NfKvBf16_##name { uint16_t name##_u16[]; }; \
    vec4 name##_load4(uint e) { \
        return vec4(nf_bf16(uint(name##_u16[e])), nf_bf16(uint(name##_u16[e + 1u])), \
                    nf_bf16(uint(name##_u16[e + 2u])), nf_bf16(uint(name##_u16[e + 3u]))); \
    } \
    float name##_load1(uint e) { return nf_bf16(uint(name##_u16[e])); }

#elif KV == NF_KV_Q8_0 || KV == NF_KV_Q4_0 || KV == NF_KV_Q4_1 || KV == NF_KV_IQ4_NL || KV == NF_KV_Q5_0 \
    || KV == NF_KV_Q5_1

#define NF_ATTN_KV_VIEWS(binding_index, name) \
    NF_BUFFER_VIEWS(binding_index, name) \
    NF_KV_LOADERS(name) \
    vec4 name##_load4(uint e) { return name##_kv_load4(e, KV); } \
    float name##_load1(uint e) { return name##_kv_load1(e, KV); }

#else
#error "common/attn_common.glsl: KV is not a KV cache type"
#endif

#endif /* KV */

#endif /* NF_ATTN_COMMON_GLSL */

/* common/attn_common.h - attention helpers: visibility, position bias, softcap, online-softmax merge and the packed K/V type argument. Requires: common/dialect.h, common/config.h, common/math.h, common/kv_types.h */
/*
 * Shared by the flash kernels (flash_decode.cl, flash_prefill.cl, flash_combine.cl) and by every
 * attention-like kernel family that scores cached keys; cached rows are read through the common/kv_types.h
 * loaders. GLSL twin: shaders/llm/common/attn_common.glsl, including the packed K/V type helpers
 * nf_attn_kv_type_ok, nf_attn_kv_type_k and nf_attn_kv_type_v (under NF_KV_TYPES_GLSL).
 * Java reference: gguf.ops.attn.CpuAttentionCore (same formulas, same order).
 *
 * Score of query row h against cached key t (llama.cpp ggml-cpu/ops.cpp:8997-9072):
 *   s = softcap > 0 ? softcap * tanh(dot * scale / softcap) : dot * scale
 *   s += bias(bias_mode)                                   (the ggml mask value times the ALiBi slope)
 *   s  = -INF when nf_attn_visible(q_pos, kv_pos[t]) == 0
 * then softmax over the visible keys with an optional per-head sink logit in the denominator.
 */
#ifndef NF_ATTN_COMMON_H
#define NF_ATTN_COMMON_H

/* swa_type codes: llama_swa_type (llama-hparams.h:24) == gguf.arch.SwaType.code() */
#define NF_SWA_NONE      0
#define NF_SWA_STANDARD  1
#define NF_SWA_CHUNKED   2
#define NF_SWA_SYMMETRIC 3

/* bias_mode codes == gguf.spec.AttentionSpec.BiasMode.code() */
#define NF_ATTN_BIAS_NONE      0
#define NF_ATTN_BIAS_ALIBI     1   /* slope(h) * -|p_q - p_k|          (llama-kv-cache.cpp:1712) */
#define NF_ATTN_BIAS_T5_CAUSAL 2   /* rel_bias[bucket(p_k, p_q, uni) * n_head + h]  (decoder self-attention) */
#define NF_ATTN_BIAS_T5_BIDIR  3   /* rel_bias[bucket(p_k, p_q, bidi) * n_head + h] (encoder) */

/* Head-dimension chunk of the tiled prefill kernel: K, V and Q tiles are staged 64 values at a time. */
#define NF_FA_CHUNK 64

/*
 * Whether query position p_q may attend key position p_k: llama.cpp causal mask plus
 * llama_hparams::is_masked_swa (llama-hparams.h:506-540). An empty cell (p_k < 0) is never visible.
 *   causal    p_k > p_q                      -> masked
 *   STANDARD  p_q - p_k >= n_swa             -> masked (a query sees n_swa positions including itself)
 *   CHUNKED   p_k < (p_q / n_swa) * n_swa    -> masked
 *   SYMMETRIC |p_q - p_k| > n_swa / 2        -> masked
 */
NF_INLINE int nf_attn_visible(int p_q, int p_k, int causal, int swa_type, int n_swa) {
    if (p_k < 0) {
        return 0;
    }
    if (causal && p_k > p_q) {
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

/* Capped and scaled logit (ggml-cpu/ops.cpp:8997-8998 scale /= cap; 9068-9072 s = cap * tanh(s * scale)). */
NF_INLINE float nf_attn_score(float dot, float scale, float softcap) {
    if (softcap > 0.0f) {
        return softcap * tanh(dot * (scale / softcap));
    }
    return dot * scale;
}

/*
 * Additive position bias of head h. slope is nf_alibi_slope(h, n_head, max_bias) (common/math.h),
 * precomputed per head by the caller; rel_bias is the T5 attn_rel_b table [n_buckets][n_head]
 * (llama-graph.cpp build_pos_bias: row = bucket, column = head).
 */
NF_INLINE float nf_attn_bias(int bias_mode, int h, int n_head, int p_q, int p_k, float slope,
                             NF_GLOBAL const float* rel_bias, int n_buckets, int max_distance) {
    if (bias_mode == NF_ATTN_BIAS_ALIBI) {
        const int dist = p_q > p_k ? p_q - p_k : p_k - p_q;
        return slope * (-(float)dist);
    }
    if (bias_mode == NF_ATTN_BIAS_T5_CAUSAL || bias_mode == NF_ATTN_BIAS_T5_BIDIR) {
        const int bucket = nf_t5_bucket(p_k, p_q, n_buckets, max_distance,
                                        bias_mode == NF_ATTN_BIAS_T5_BIDIR ? 1 : 0);
        return rel_bias[bucket * n_head + h];
    }
    return 0.0f;
}

/*
 * e^(x - m) for the online softmax, exactly 0 when x is -INF (a masked score or an empty
 * running maximum). With m >= x this never overflows, and it never evaluates -INF - -INF.
 */
NF_INLINE float nf_attn_rescale(float x, float m) {
    return (x == -NF_INF) ? 0.0f : exp(x - m);
}

/*
 * Adds the sink logit of a head to a finished softmax state (running max m, sum l), as ggml
 * (ggml-cpu/ops.cpp:9130-9145): m' = max(m, sink), l' = l * e^(m - m') + e^(sink - m').
 * Returns (factor for the unnormalized output, l').
 */
NF_INLINE float2 nf_attn_sink(float m, float l, float sink) {
    const float m2 = fmax(m, sink);
    const float a = nf_attn_rescale(m, m2);
    return NF_F2(a, l * a + exp(sink - m2));
}

/*
 * kv_types argument of the mixed-type flash kernels (LLM_FlashDecode_Mixed, LLM_FlashAttn_Mixed): the K cache type in
 * bits 0-7 and the V cache type in bits 8-15, each an NF_KV_* code of common/kv_types.h; no other bit is set
 * (gguf.ops.attn.AttnKernels.kvTypes). The same-type kernels fix both types at compile time.
 */
#define NF_ATTN_KV_TYPE_BITS 8
#define NF_ATTN_KV_TYPE_MASK 0xFF

/* 1 when t is a KV cache type code (NF_KV_F32 .. NF_KV_Q5_1). */
NF_INLINE int nf_attn_kv_type_ok(int t) {
    return t >= NF_KV_F32 && t <= NF_KV_Q5_1;
}

/* K cache type of a kv_types argument, or -1 (rejected by nf_attn_kv_type_ok) when kv_types is malformed. */
NF_INLINE int nf_attn_kv_type_k(int kv_types) {
    return (kv_types >> (2 * NF_ATTN_KV_TYPE_BITS)) == 0 ? (kv_types & NF_ATTN_KV_TYPE_MASK) : -1;
}

/* V cache type of a kv_types argument, or -1 (rejected by nf_attn_kv_type_ok) when kv_types is malformed. */
NF_INLINE int nf_attn_kv_type_v(int kv_types) {
    return (kv_types >> (2 * NF_ATTN_KV_TYPE_BITS)) == 0
           ? ((kv_types >> NF_ATTN_KV_TYPE_BITS) & NF_ATTN_KV_TYPE_MASK) : -1;
}

#endif /* NF_ATTN_COMMON_H */

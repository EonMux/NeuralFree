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
#define NF_SWA_CHUNKED_LOOKBACK 4

/* bias_mode codes == gguf.spec.AttentionSpec.BiasMode.code() */
#define NF_ATTN_BIAS_NONE      0
#define NF_ATTN_BIAS_ALIBI     1   /* slope(h) * -|p_q - p_k|          (llama-kv-cache.cpp:1712) */
#define NF_ATTN_BIAS_T5_CAUSAL 2   /* rel_bias[bucket(p_k, p_q, uni) * n_head + h]  (decoder self-attention) */
#define NF_ATTN_BIAS_T5_BIDIR  3   /* rel_bias[bucket(p_k, p_q, bidi) * n_head + h] (encoder) */
#define NF_ATTN_BIAS_ALIBI_SQRT 4  /* slope(h) * -sqrt(p_q - p_k), 0 for later keys (vLLM use_alibi_sqrt) */
#define NF_ATTN_BIAS_INKLING   5   /* rel_bias[(q_row * n_head + h) * rel_extent + (p_q - p_k)] (vLLM Inkling) */

/*
 * flags argument of the flash attention kernels (flash_decode.cl, flash_prefill.cl, flash_mma.cl): bit 0 causal
 * masking, bit 1 the sink logits of buffer sinks, bit 2 direct normalized output (n_split == 1) instead of
 * (o, m, l) partials (gguf.ops.attn.AttnKernels.FLAG_CAUSAL / FLAG_SINKS / FLAG_DIRECT). The clear bit is 0, so
 * each bit substitutes its scalar argument unchanged. Bits 5-31 carry n_fresh next to the tree bit (see below).
 */
#define NF_ATTN_FLAG_CAUSAL 1
#define NF_ATTN_FLAG_SINKS  2
#define NF_ATTN_FLAG_DIRECT 4
/* V of an MLA latent cache is read from the K rows, so it decodes through the K side planes. */
#define NF_ATTN_FLAG_V_ALIAS 8
/* A verify step's tree mask: the last n_fresh slots of [kv_start, kv_end) are the step's fresh rows, and query
 * row t sees the fresh slots its tree_mask[t] bit names; the slots before them keep the position rules. */
#define NF_ATTN_FLAG_TREE 16

/*
 * n_fresh rides in flags bits 5-31 next to NF_ATTN_FLAG_TREE (gguf.ops.attn.AttnKernels.treeFlags), keeping the
 * fresh-row count out of the flash kernels' scalar arguments: the count of the step's fresh rows when the tree bit
 * is set, 0 otherwise.
 */
#define NF_ATTN_FLAG_FRESH_SHIFT 5

/* The fresh-row count packed in flags: the bits 5-31 payload when the tree bit is set, 0 without it. */
NF_INLINE int nf_attn_flag_fresh(int flags) {
    return (flags & NF_ATTN_FLAG_TREE) != 0 ? (int)((unsigned int)flags >> NF_ATTN_FLAG_FRESH_SHIFT) : 0;
}

/* Head-dimension chunk of the tiled prefill kernel: K, V and Q tiles are staged 64 values at a time. */
#define NF_FA_CHUNK 64

/*
 * Whether query position p_q may attend key position p_k: llama.cpp causal mask plus
 * llama_hparams::is_masked_swa (llama-hparams.h:506-540). An empty cell (p_k < 0) is never visible.
 *   causal    p_k > p_q                      -> masked
 *   STANDARD  p_q - p_k >= n_swa             -> masked (a query sees n_swa positions including itself)
 *   CHUNKED   p_k < (p_q / n_swa) * n_swa    -> masked
 *   CHUNKED_LOOKBACK p_k < ((p_q / n_swa) - 1) * n_swa -> masked
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
    if (swa_type == NF_SWA_CHUNKED_LOOKBACK) {
        return (p_k < ((p_q / n_swa) - 1) * n_swa) ? 0 : 1;
    }
    if (swa_type == NF_SWA_SYMMETRIC) {
        const int half_n_swa = n_swa / 2;
        const int diff = p_q - p_k;
        return (diff < -half_n_swa || diff > half_n_swa) ? 0 : 1;
    }
    return 1;
}

/*
 * Whether query row t of a tree-masked step sees KV slot t_kv: fresh slots (the last n_fresh of the step's key
 * range) read the row's ancestor bitmask tree_bits = tree_mask[t], one bit per fresh slot; the slots before the
 * fresh block keep the position rules (nf_attn_visible). A step without fresh rows (n_fresh = 0) takes no tree
 * rule at all: a context-sharded cache counts kv_end over its own rows while its kv_pos holds global positions,
 * so a plain slot's position can sit at or above kv_end without being a fresh slot.
 * gguf.ops.attn.CpuAttentionCore.scoreBlock is the Java reference.
 */
NF_INLINE int nf_attn_tree_visible(int t_kv, ulong tree_bits, int kv_end, int n_fresh, int q_pos,
                                   int causal, int swa_type, int n_swa) {
    const int fslot = n_fresh > 0 ? t_kv - (kv_end - n_fresh) : -1;
    if (fslot >= 0) {
        return (int)((tree_bits >> fslot) & 1UL);
    }
    return nf_attn_visible(q_pos, t_kv, causal, swa_type, n_swa);
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
 * (llama-graph.cpp build_pos_bias: row = bucket, column = head), or the Inkling relative table
 * [q_row][n_head][n_buckets] holding one value per distance, n_buckets being rel_extent there. q_row is the
 * query token's row in the step (its index in the batch's q / q_pos arrays); the position-only modes ignore it.
 */
NF_INLINE float nf_attn_bias(int bias_mode, int h, int n_head, int p_q, int p_k, float slope,
                             NF_GLOBAL const float* rel_bias, int n_buckets, int max_distance, int q_row) {
    if (bias_mode == NF_ATTN_BIAS_ALIBI) {
        const int dist = p_q > p_k ? p_q - p_k : p_k - p_q;
        return slope * (-(float)dist);
    }
    if (bias_mode == NF_ATTN_BIAS_ALIBI_SQRT) {
        return p_k <= p_q ? slope * (-sqrt((float)(p_q - p_k))) : 0.0f;
    }
    if (bias_mode == NF_ATTN_BIAS_T5_CAUSAL || bias_mode == NF_ATTN_BIAS_T5_BIDIR) {
        const int bucket = nf_t5_bucket(p_k, p_q, n_buckets, max_distance,
                                        bias_mode == NF_ATTN_BIAS_T5_BIDIR ? 1 : 0);
        return rel_bias[bucket * n_head + h];
    }
    if (bias_mode == NF_ATTN_BIAS_INKLING) {
        /* Distances at or past rel_extent carry no bias: the table only spans the local window. */
        const int dist = p_q - p_k;
        if (dist < 0 || dist >= n_buckets) {
            return 0.0f;
        }
        return rel_bias[((nf_i64)q_row * n_head + h) * n_buckets + dist];
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
 * Physical KV row the cached row indexed by slot t reads, as PagedSequences maps slots:
 *   page_shift < 0:   contiguous rows, t is the row (identity map; the un-paged flash entry points pass
 *                     page_shift = -1, so their arguments and results are unchanged);
 *   page_shift >= 0:  page size P = 1 << page_shift; pt[pt_off + (t >> page_shift)] is the page id of the page t
 *                     falls on (one page table of page_size entries per sequence: page_table[seq * maxPages ..];
 *                     DevicePageTables), and the row is (page id << page_shift) | (t & (P - 1)).
 * The buffers k, v and kv_pos are laid out by physical row: the paged kernels read K, V and kv_pos at nf_kv_row(t).
 * pt is never dereferenced on the identity path (page_shift < 0), so un-paged entry points pass any int buffer.
 */
NF_INLINE int nf_kv_row(int t, int page_shift, NF_GLOBAL const int* pt, const int pt_off) {
    if (page_shift < 0) {
        return t;
    }
    const int page_id = pt[pt_off + (t >> page_shift)];
    return (page_id << page_shift) | (t & ((1 << page_shift) - 1));
}

/*
 * 1 when the page of slot t is valid: always on the identity path, otherwise the page id must be >= 0 (a page the
 * sequence never had; negative page ids are masked, never required: the prefix cache never
 * relies on a masked page for a required row).
 */
NF_INLINE int nf_kv_page_ok(int t, int page_shift, NF_GLOBAL const int* pt, const int pt_off) {
    return page_shift < 0 || pt[pt_off + (t >> page_shift)] >= 0;
}

/*
 * kv_types argument of the mixed-type flash kernels (LLM_FlashDecode_Mixed, LLM_FlashAttn_Mixed): the K cache type in
 * bits 0-7 and the V cache type in bits 8-15, each an NF_KV_* code of common/kv_types.h; no other bit is set
 * (gguf.ops.attn.AttnKernels.kvTypes). The same-type kernels fix both types at compile time.
 */
#define NF_ATTN_KV_TYPE_BITS 8
#define NF_ATTN_KV_TYPE_MASK 0xFF

/* 1 when t is a KV cache type code (NF_KV_F32 .. NF_KV_INT4_PER_TOKEN_HEAD). */
NF_INLINE int nf_attn_kv_type_ok(int t) {
    return t >= NF_KV_F32 && t <= NF_KV_NVFP4_DS_MLA;
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

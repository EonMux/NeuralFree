/* sparse/step5.cl - Step5 sparse region attention: dtype roundings, the region-summary ring, the weighted-ReLU
 * region scorer, the region selection and the sparse/windowed gather attention with the head gate.
 * Requires: common/dialect.h, common/config.h, common/math.h, common/kv_types.h */
/*
 * The CPU semantic reference is CpuStep5Sparse / CpuStep5RegionCache / CpuStep5Attn (crates of vLLM
 * vllm/models/step5/common/ops). A KV region is region_block consecutive token positions; a closed region's summary
 * is an online softmax over its indexer keys weighted by exp(index_z) per proxy dimension, rounded through BF16 then
 * FP8 E4M3 (CpuStep5RegionCache.summarize).
 *
 * LLM_Step5_Cast(i n, i mode, R x, W y)
 *     y[i] = bf16(x[i])                         mode 0
 *     y[i] = e4m3(bf16(x[i]))                   mode 1
 * LLM_Step5_Summary(i n_tokens, i region_block, i capacity, i proxy, R key, R z, R<int> q_pos, W ring, W summary)
 *     Every token stores its [key | z] row, rounded to BF16, in ring[row][*], row = q_pos[t] % capacity; a token
 *     closing its region rewrites that region's summary from the ring. One workgroup per token.
 * LLM_Step5_RegionScores(i n_tokens, i n_kv, i n_idx_heads, i heads_per_group, i proxy, i regions, i region_block,
 *                        R iq, R iw, R summary, R<int> q_pos, W scores)
 *     scores[t][g][r] = sum_h relu(iq[t][g][h] . summary[r]) * iw[t][g][h], -INF for a region still open.
 *     One work-item per (token, KV group, region).
 * LLM_Step5_SelectRegions(i n_rows, i k, i n_kv, i region_block, i regions, R<int> topk, R<int> q_pos, W<int> sel)
 *     Reorders a row's LLM_TopK_Rows selection ascending behind its left padding and appends the open region:
 *     sel[r][0 .. k) = -1 padding then the ascending closed regions, sel[r][k] = closed count. One work-item per row.
 * LLM_Step5_Attn_<KV>(i n_tokens, i n_head, i n_kv, i head_dim, i region_block, i slot_count, i mode, i window,
 *                     f scale, i kv_rows, i k_row_stride, i v_row_stride, R q, R<KV> k_cache, R<KV> v_cache,
 *                     R<int> kv_pos, R<int> sel, R<int> q_pos, R gate, W out)
 *     Sparse GQA over the selected regions up to the token's position (mode 0) or dense causal attention over the
 *     last window positions (mode 1), with the vLLM arithmetic: base-2 softmax, BF16 softmax weights, F32 value
 *     accumulation, bf16(bf16(acc * sum^-1) * sigmoid(gate)). A position lives in row p % kv_rows when
 *     kv_pos[row] == p (the contiguous cache's ring addressing). One work-item per (token, query head).
 */

#define NF_STEP5_LOG2E 1.4426950408889634f
#define NF_STEP5_MODE_BF16 0
#define NF_STEP5_MODE_E4M3 1

/* The vLLM roundings: BF16 round to nearest even, and the saturating E4M3 code widened back to F32. */
NF_INLINE float nf_step5_bf16(float x) {
    return bf16_to_f32(nf_f32_to_bf16(x));
}

NF_INLINE float nf_step5_e4m3(float x) {
    if (isnan(x)) {
        return x;
    }
    return nf_kv_pt_value(NF_KV_FP8_PER_TOKEN_HEAD, (int)nf_kv_fp8_e4m3_bits(x));
}

NF_KERNEL void LLM_Step5_Cast(
    const int n, const int mode,
    NF_GLOBAL const float* x,
    NF_GLOBAL float* y) {
    const nf_i64 i = (nf_i64)get_global_id(0);
    if (i >= n) {
        return;
    }
    const float rounded = nf_step5_bf16(x[i]);
    y[i] = mode == NF_STEP5_MODE_E4M3 ? nf_step5_e4m3(rounded) : rounded;
}

NF_KERNEL void LLM_Step5_Summary(
    const int n_tokens, const int region_block, const int capacity, const int proxy,
    NF_GLOBAL const float* NF_RESTRICT key,
    NF_GLOBAL const float* NF_RESTRICT z,
    NF_GLOBAL const int* NF_RESTRICT q_pos,
    NF_GLOBAL float* NF_RESTRICT ring,
    NF_GLOBAL float* NF_RESTRICT summary) {
    const int t = get_group_id(0);
    if (t >= n_tokens) {
        return;
    }
    const int lid = get_local_id(0), lsize = get_local_size(0);
    const int pos = q_pos[t];
    if (pos < 0) {
        return;
    }
    NF_GLOBAL float* dst = ring + (nf_i64)(pos % capacity) * 2 * proxy;
    NF_GLOBAL const float* krow = key + (nf_i64)t * proxy;
    NF_GLOBAL const float* zrow = z + (nf_i64)t * proxy;
    for (int d = lid; d < proxy; d += lsize) {
        dst[d] = nf_step5_bf16(krow[d]);
        dst[proxy + d] = nf_step5_bf16(zrow[d]);
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
    if ((pos + 1) % region_block != 0) {
        return;
    }
    const int region = pos / region_block;
    const int first = region * region_block;
    NF_GLOBAL float* out = summary + (nf_i64)region * proxy;
    for (int d = lid; d < proxy; d += lsize) {
        float numerator = 0.0f, denominator = 0.0f, maximum = -NF_INF;
        for (int p = first; p <= pos; ++p) {
            NF_GLOBAL const float* r = ring + (nf_i64)(p % capacity) * 2 * proxy;
            float updated = maximum;
            for (int j = 0; j < proxy; ++j) {
                updated = fmax(updated, r[proxy + j]);
            }
            const float scale = denominator > 0.0f ? exp(maximum - updated) : 0.0f;
            const float weight = exp(r[proxy + d] - updated);
            numerator = numerator * scale + weight * r[d];
            denominator = denominator * scale + weight;
            maximum = updated;
        }
        const float mean = numerator / fmax(denominator, 1e-20f);
        out[d] = nf_step5_e4m3(nf_step5_bf16(mean));
    }
}

NF_KERNEL void LLM_Step5_RegionScores(
    const int n_tokens, const int n_kv, const int n_idx_heads, const int heads_per_group,
    const int proxy, const int regions, const int region_block,
    NF_GLOBAL const float* NF_RESTRICT iq,
    NF_GLOBAL const float* NF_RESTRICT iw,
    NF_GLOBAL const float* NF_RESTRICT summary,
    NF_GLOBAL const int* NF_RESTRICT q_pos,
    NF_GLOBAL float* NF_RESTRICT scores) {
    const nf_i64 gid = (nf_i64)get_global_id(0);
    const nf_i64 total = (nf_i64)n_tokens * n_kv * regions;
    if (gid >= total) {
        return;
    }
    const int r = (int)(gid % regions);
    const nf_i64 rest = gid / regions;
    const int g = (int)(rest % n_kv);
    const int t = (int)(rest / n_kv);
    const int closed = min(q_pos[t] / region_block, regions);
    float score;
    if (r >= closed) {
        score = -NF_INF;
    } else {
        /* four independent FMA chains, left-associated like the CPU scorer */
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        NF_GLOBAL const float* srow = summary + (nf_i64)r * proxy;
        const nf_i64 qbase = ((nf_i64)t * n_kv + g) * heads_per_group * proxy;
        const nf_i64 wbase = (nf_i64)t * n_idx_heads + (nf_i64)g * heads_per_group;
        for (int h = 0; h < heads_per_group; ++h) {
            NF_GLOBAL const float* qh = iq + qbase + (nf_i64)h * proxy;
            float dot = 0.0f;
            for (int d = 0; d < proxy; ++d) {
                dot += qh[d] * srow[d];
            }
            const float v = fmax(dot, 0.0f) * iw[wbase + h];
            switch (h & 3) {
                case 0: a0 += v; break;
                case 1: a1 += v; break;
                case 2: a2 += v; break;
                default: a3 += v; break;
            }
        }
        score = ((a0 + a1) + a2) + a3;
    }
    scores[gid] = score;
}

NF_KERNEL void LLM_Step5_SelectRegions(
    const int n_rows, const int k, const int n_kv, const int region_block, const int regions,
    NF_GLOBAL const int* NF_RESTRICT topk,
    NF_GLOBAL const int* NF_RESTRICT q_pos,
    NF_GLOBAL int* NF_RESTRICT sel) {
    const int row = (int)get_global_id(0);
    if (row >= n_rows) {
        return;
    }
    const int closed = min(q_pos[row / n_kv] / region_block, regions);
    const int kk = min(k, closed);
    const int outBase = row * (k + 1);
    for (int i = 0; i < k; ++i) {
        sel[outBase + i] = -1;
    }
    const int first = k - kk;
    for (int i = 0; i < kk; ++i) {
        sel[outBase + first + i] = topk[(nf_i64)row * k + i];
    }
    /* insertion sort of the selected ids: ascending, as CpuStep5Sparse.selectRegions emits them */
    for (int i = 1; i < kk; ++i) {
        const int v = sel[outBase + first + i];
        int j = i - 1;
        while (j >= 0 && sel[outBase + first + j] > v) {
            sel[outBase + first + j + 1] = sel[outBase + first + j];
            --j;
        }
        sel[outBase + first + j + 1] = v;
    }
    sel[outBase + k] = closed;
}

/* The indexer score of one query head against the position p of a contiguous cache, -INF when it is not held. */
NF_INLINE float nf_step5_dot_at(
    NF_GLOBAL const float* qh, NF_GLOBAL const uchar* k_cache, const int kv_type,
    NF_GLOBAL const int* kv_pos, const int kv_rows, const int k_row_stride, const int group,
    const int head_dim, const int p, const float scale) {
    const int row = p % kv_rows;
    if (kv_pos[row] != p) {
        return -NF_INF;
    }
    const nf_i64 base = (nf_i64)row * k_row_stride + (nf_i64)group * head_dim;
    float dot = 0.0f;
    for (int d = 0; d < head_dim; ++d) {
        dot += qh[d] * nf_kv_load1(k_cache, base + d, kv_type);
    }
    return dot * scale * NF_STEP5_LOG2E;
}

/* The value of position p of a contiguous cache, -INF when it is not held (the same map as nf_step5_dot_at). */
NF_INLINE float nf_step5_value_at(
    NF_GLOBAL const uchar* v_cache, const int kv_type, NF_GLOBAL const int* kv_pos, const int kv_rows,
    const int v_row_stride, const int group, const int head_dim, const int d, const int p) {
    const int row = p % kv_rows;
    if (kv_pos[row] != p) {
        return 0.0f;
    }
    return nf_kv_load1(v_cache, (nf_i64)row * v_row_stride + (nf_i64)group * head_dim + d, kv_type);
}

/*
 * VISIT(PHASE) runs once per attended position p of the token's selection (mode 0) or window (mode 1). PHASE 0 keeps
 * the maximum score, 1 sums the exponentials, 2 accumulates the BF16 weights into the output.
 */
#define NF_STEP5_VISIT(PHASE)                                                                     \
    if (mode == 0) {                                                                              \
        for (int s = 0; s < slot_count; ++s) {                                                    \
            const int region = sel[gbase + s];                                                    \
            if (region < 0) continue;                                                             \
            const int first = region * region_block;                                              \
            const int last = min(first + region_block - 1, pos);                                  \
            for (int p = first; p <= last; ++p) { PHASE }                                         \
        }                                                                                         \
    } else {                                                                                      \
        const int first = window > 0 ? max(0, pos - window + 1) : 0;                               \
        for (int p = first; p <= pos; ++p) { PHASE }                                              \
    }

#define NF_STEP5_ATTN_KERNEL(SFX, KV_T, KV_CODE)                                                   \
NF_KERNEL void LLM_Step5_Attn_##SFX(                                                               \
    const int n_tokens, const int n_head, const int n_kv, const int head_dim,                      \
    const int region_block, const int slot_count, const int mode, const int window,                \
    const float scale, const int kv_rows, const int k_row_stride, const int v_row_stride,          \
    NF_GLOBAL const float* NF_RESTRICT q,                                                          \
    NF_GLOBAL const KV_T* NF_RESTRICT k_cache,                                                     \
    NF_GLOBAL const KV_T* NF_RESTRICT v_cache,                                                     \
    NF_GLOBAL const int* NF_RESTRICT kv_pos,                                                       \
    NF_GLOBAL const int* NF_RESTRICT sel,                                                          \
    NF_GLOBAL const int* NF_RESTRICT q_pos,                                                        \
    NF_GLOBAL const float* NF_RESTRICT gate,                                                       \
    NF_GLOBAL float* NF_RESTRICT out) {                                                            \
    const nf_i64 gid = (nf_i64)get_global_id(0);                                                   \
    const nf_i64 total = (nf_i64)n_tokens * n_head;                                                \
    if (gid >= total) {                                                                            \
        return;                                                                                    \
    }                                                                                              \
    const int h = (int)(gid % n_head);                                                             \
    const int t = (int)(gid / n_head);                                                             \
    const int pos = q_pos[t];                                                                      \
    const int group = h / (n_head / n_kv);                                                         \
    const nf_i64 gbase = (nf_i64)(t * n_kv + group) * slot_count;                                  \
    NF_GLOBAL const float* qh = q + ((nf_i64)t * n_head + h) * head_dim;                           \
    NF_GLOBAL const uchar* kc = (NF_GLOBAL const uchar*)k_cache;                                   \
    NF_GLOBAL const uchar* vc = (NF_GLOBAL const uchar*)v_cache;                                   \
    NF_GLOBAL float* oh = out + ((nf_i64)t * n_head + h) * head_dim;                               \
    float maxs = -NF_INF;                                                                          \
    NF_STEP5_VISIT({                                                                               \
        const float sc = nf_step5_dot_at(qh, kc, KV_CODE, kv_pos, kv_rows, k_row_stride, group,     \
                head_dim, p, scale);                                                               \
        maxs = fmax(maxs, sc);                                                                     \
    })                                                                                             \
    float sum = 0.0f;                                                                              \
    NF_STEP5_VISIT({                                                                               \
        const float sc = nf_step5_dot_at(qh, kc, KV_CODE, kv_pos, kv_rows, k_row_stride, group,     \
                head_dim, p, scale);                                                               \
        if (sc > -NF_INF) {                                                                        \
            sum += exp2(sc - maxs);                                                                \
        }                                                                                          \
    })                                                                                             \
    for (int d = 0; d < head_dim; ++d) {                                                           \
        oh[d] = 0.0f;                                                                              \
    }                                                                                              \
    NF_STEP5_VISIT({                                                                               \
        const float sc = nf_step5_dot_at(qh, kc, KV_CODE, kv_pos, kv_rows, k_row_stride, group,     \
                head_dim, p, scale);                                                               \
        if (sc > -NF_INF) {                                                                        \
            const float weight = nf_step5_bf16(exp2(sc - maxs));                                   \
            for (int d = 0; d < head_dim; ++d) {                                                   \
                oh[d] += weight * nf_step5_value_at(vc, KV_CODE, kv_pos, kv_rows, v_row_stride,     \
                        group, head_dim, d, p);                                                    \
            }                                                                                      \
        }                                                                                          \
    })                                                                                             \
    const float inverse = sum > 0.0f ? 1.0f / fmax(sum, 1e-30f) : 0.0f;                            \
    const float gateValue = nf_step5_bf16(1.0f / (1.0f + exp(-gate[(nf_i64)t * n_head + h])));        \
    for (int d = 0; d < head_dim; ++d) {                                                           \
        oh[d] = nf_step5_bf16(nf_step5_bf16(oh[d] * inverse) * gateValue);                         \
    }                                                                                              \
}

NF_STEP5_ATTN_KERNEL(F32, float, NF_KV_F32)
NF_STEP5_ATTN_KERNEL(F16, NF_HALF, NF_KV_F16)
NF_STEP5_ATTN_KERNEL(Q8_0, uchar, NF_KV_Q8_0)
NF_STEP5_ATTN_KERNEL(BF16, ushort, NF_KV_BF16)
NF_STEP5_ATTN_KERNEL(Q4_0, uchar, NF_KV_Q4_0)
NF_STEP5_ATTN_KERNEL(Q4_1, uchar, NF_KV_Q4_1)
NF_STEP5_ATTN_KERNEL(IQ4_NL, uchar, NF_KV_IQ4_NL)
NF_STEP5_ATTN_KERNEL(Q5_0, uchar, NF_KV_Q5_0)
NF_STEP5_ATTN_KERNEL(Q5_1, uchar, NF_KV_Q5_1)
NF_STEP5_ATTN_KERNEL(FP8_E4M3, uchar, NF_KV_FP8_E4M3)
NF_STEP5_ATTN_KERNEL(FP8_E5M2, uchar, NF_KV_FP8_E5M2)

/* moe.cl - MoE router, hash ids, weighted sum, add-id, bucketize and input-scale kernels. Requires: common/dialect.h, common/config.h, common/math.h, common/reduce.h */
/*
 * Routing follows llama.cpp build_moe_ffn (src/llama-graph.cpp:2189-2345); the Java twin is
 * gguf.ops.moe.MoeRouting (routing) and gguf.ops.moe.MoeCombine (the other kernels), with the same
 * selection order and the same per-token arithmetic order.
 *
 * Kernels and launch contracts (pair index p = t * k + j, j < k):
 *   LLM_MoE_Router       global = n_tokens * L, local = L (power of two, 32 <= L <= LLM_DIMX)
 *   LLM_MoE_HashIds      global >= n_tokens * k, 1-D
 *   LLM_MoE_WeightedSum  global >= n_tokens * n_embd, 1-D
 *   LLM_AddId            global >= n_sel * dim, 1-D
 *   LLM_MoE_Bucketize    global = local = L (one workgroup, power of two <= LLM_DIMX)
 *   LLM_MoE_ScaleInput   global >= n_tokens * k * n_embd, 1-D
 */

#define NF_MOE_MAX_EXPERTS 1024   /* router / bucketize experts per token (Java: MoeKernels.MAX_EXPERTS) */
#define NF_MOE_MAX_K       32     /* experts used per token (MoeKernels.MAX_K) */
#define NF_MOE_MAX_GROUPS  64     /* expert groups (MoeKernels.MAX_GROUPS) */

/* GatingFunc codes */
#define NF_MOE_GATING_SOFTMAX        0
#define NF_MOE_GATING_SIGMOID        1
#define NF_MOE_GATING_SOFTMAX_WEIGHT 2
#define NF_MOE_GATING_SQRT_SOFTPLUS  3

/* sel_mode codes: 0-2 equal MoeSpec.SelectionMode codes; 3 takes the ids already in `ids` */
#define NF_MOE_SEL_PROBS       0
#define NF_MOE_SEL_RAW_LOGITS  1
#define NF_MOE_SEL_SIGMOID     2
#define NF_MOE_SEL_PRESELECTED 3

/* LLM_MoE_WeightedSum flag bits */
#define NF_MOE_WS_ACCUMULATE 1
#define NF_MOE_WS_UNWEIGHTED 2

/* Smallest weight sum when normalizing: the smallest normal f16 (llama-graph.cpp:2339). */
#define NF_MOE_NORM_MIN 6.103515625e-5f

/* Selection key: NaN ranks as -INF, so the order is total and the pick does not depend on how the
 * work-items split the experts. */
NF_INLINE float nf_moe_key(float v) {
    return isnan(v) ? -NF_INF : v;
}

/* Whether candidate (cv, ci) ranks before (bv, bi): higher key, then lower index; an index < 0 is
 * the empty candidate. This is the stable descending order of ggml_argsort_top_k with ties to the
 * lower index. */
NF_INLINE int nf_moe_better(float cv, int ci, float bv, int bi) {
    return ci >= 0 && (bi < 0 || cv > bv || (cv == bv && ci < bi));
}

/* Workgroup arg-max over (v, i) candidates by nf_moe_better; every work-item receives the winning
 * index. rv / ri hold at least L entries; L is a power of two. Ends with a barrier. */
NF_INLINE int nf_moe_wg_argmax(NF_LOCAL_PTR float* rv, NF_LOCAL_PTR int* ri, float v, int i) {
    const int lid = get_local_id(0);
    rv[lid] = v;
    ri[lid] = i;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = get_local_size(0) >> 1; s > 0; s >>= 1) {
        if (lid < s && nf_moe_better(rv[lid + s], ri[lid + s], rv[lid], ri[lid])) {
            rv[lid] = rv[lid + s];
            ri[lid] = ri[lid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const int r = ri[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    return r;
}

/*
 * (int)((float)id * (1.0f / (float)div)): llama-graph.cpp:2312-2318 scales the selected ids by
 * 1 / n_group_experts in f32 and truncates. The reciprocal is rounded to nearest-even from an
 * integer long division, so the result does not depend on the device's division accuracy.
 * Requires 1 <= div < 2^24 and 0 <= id < 2^24.
 */
NF_INLINE int nf_moe_div_id(int id, int div) {
    const uint g = (uint)div;
    uint n = 0u;
    while ((2u << n) <= g) {
        n++;
    }
    float r;
    if ((g & (g - 1u)) == 0u) {
        r = as_float((127u - n) << 23);
    } else {
        uint rem = 1u << n;
        uint q = 0u;
        for (int b = 0; b < 24; b++) {
            rem <<= 1;
            q <<= 1;
            if (rem >= g) {
                rem -= g;
                q |= 1u;
            }
        }
        if (2u * rem > g || (2u * rem == g && (q & 1u) != 0u)) {
            q++;
        }
        r = q == (1u << 24) ? as_float((127u - n) << 23) : as_float(((126u - n) << 23) | (q & 0x7FFFFFu));
    }
    return (int)((float)id * r);
}

/* Probability of one router logit under gating code `gating` (softmax is finished by the caller). */
NF_INLINE float nf_moe_gate(int gating, float x, float m) {
    switch (gating) {
        case NF_MOE_GATING_SOFTMAX:        return exp(x - m);
        case NF_MOE_GATING_SIGMOID:        return nf_sigmoid(x);
        case NF_MOE_GATING_SOFTMAX_WEIGHT: return x;
        case NF_MOE_GATING_SQRT_SOFTPLUS:  return sqrt(nf_softplus(x));
        default:                           return NF_NAN;
    }
}

/*
 * One workgroup per token t:
 *   probs  = gating(logits[t]) over n_expert            (SOFTMAX: exp(x - max) * (1 / sum))
 *   sel    = sel_mode 0: probs (+ exp_probs_b when has_bias); 1: logits (llama4);
 *            2: sigmoid(logits) (grovemoe); 3: no selection, ids[t * k ..] are given (hash layers)
 *   groups = when n_groups > 1 and sel_mode < 3: group score = sum of the group's two largest sel;
 *            the n_group_used best groups are kept, the others' sel set to -INF (llama-graph.cpp:2281-2305)
 *   ids    = k rounds of arg-max over the unpicked sel (descending, ties to the lower index)
 *   id_div = when > 0 and sel_mode < 3, ids are divided as nf_moe_div_id (grovemoe chunk experts)
 *   w[j]   = probs[ids[j]]; SOFTMAX_WEIGHT: softmax over the k weights; norm_w: w / max(sum w,
 *            NF_MOE_NORM_MIN); w_scale not in {0, 1}: w * w_scale (llama-graph.cpp:2320-2345)
 * Writes ids[t * k + j] and weights[t * k + j]. Router logits must already include ffn_gate_inp.bias.
 * Host limits: n_expert <= min(NF_MOE_MAX_EXPERTS, 32 * L), k <= min(n_expert, NF_MOE_MAX_K),
 * n_groups <= NF_MOE_MAX_GROUPS with n_expert % n_groups == 0 and n_expert / n_groups >= 2,
 * n_group_used <= n_groups.
 */
NF_KERNEL void LLM_MoE_Router(
    const int n_tokens, const int n_expert, const int k, const int gating,
    const int n_groups, const int n_group_used, const int norm_w, const float w_scale,
    const int has_bias, const int sel_mode, const int id_div,
    NF_GLOBAL const float* NF_RESTRICT logits,
    NF_GLOBAL const float* NF_RESTRICT exp_probs_b,
    NF_GLOBAL int* NF_RESTRICT ids,
    NF_GLOBAL float* NF_RESTRICT weights) {
    const int t = get_group_id(0);
    if (t >= n_tokens) return;
    NF_LOCAL float nf_moe_probs[NF_MOE_MAX_EXPERTS];
    NF_LOCAL float nf_moe_sel[NF_MOE_MAX_EXPERTS];
    NF_LOCAL float nf_moe_rv[LLM_DIMX];
    NF_LOCAL int nf_moe_ri[LLM_DIMX];
    NF_LOCAL float nf_moe_gscore[NF_MOE_MAX_GROUPS];
    NF_LOCAL int nf_moe_gkeep[NF_MOE_MAX_GROUPS];
    NF_LOCAL int nf_moe_pick[NF_MOE_MAX_K];
    const int lid = get_local_id(0), lsize = get_local_size(0);
    NF_GLOBAL const float* x = logits + (nf_i64)t * n_expert;
    const nf_i64 row = (nf_i64)t * k;

    /* gating probabilities */
    float m = 0.0f;
    if (gating == NF_MOE_GATING_SOFTMAX) {
        float lm = -NF_INF;
        for (int e = lid; e < n_expert; e += lsize) {
            lm = fmax(lm, x[e]);
        }
        m = nf_wg_max(nf_moe_rv, lm);
    }
    float ls = 0.0f;
    for (int e = lid; e < n_expert; e += lsize) {
        const float p = nf_moe_gate(gating, x[e], m);
        nf_moe_probs[e] = p;
        ls += p;
    }
    if (gating == NF_MOE_GATING_SOFTMAX) {
        const float inv = 1.0f / nf_wg_sum(nf_moe_rv, ls);
        for (int e = lid; e < n_expert; e += lsize) {
            nf_moe_probs[e] *= inv;
        }
    }

    /* selection scores */
    if (sel_mode != NF_MOE_SEL_PRESELECTED) {
        for (int e = lid; e < n_expert; e += lsize) {
            float sv;
            if (sel_mode == NF_MOE_SEL_RAW_LOGITS) {
                sv = x[e];
            } else if (sel_mode == NF_MOE_SEL_SIGMOID) {
                sv = nf_sigmoid(x[e]);
            } else {
                sv = has_bias ? nf_moe_probs[e] + exp_probs_b[e] : nf_moe_probs[e];
            }
            nf_moe_sel[e] = sv;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    /* group-limited routing */
    if (sel_mode != NF_MOE_SEL_PRESELECTED && n_groups > 1) {
        const int n_per = n_expert / n_groups;
        for (int g = lid; g < n_groups; g += lsize) {
            float a = -NF_INF, b = -NF_INF;
            for (int i = 0; i < n_per; i++) {
                const float v = nf_moe_key(nf_moe_sel[g * n_per + i]);
                if (v > a) {
                    b = a;
                    a = v;
                } else if (v > b) {
                    b = v;
                }
            }
            nf_moe_gscore[g] = a + b;
            nf_moe_gkeep[g] = 0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid == 0) {
            for (int r = 0; r < n_group_used; r++) {
                float bv = -NF_INF;
                int bi = -1;
                for (int g = 0; g < n_groups; g++) {
                    const float v = nf_moe_key(nf_moe_gscore[g]);
                    if (!nf_moe_gkeep[g] && nf_moe_better(v, g, bv, bi)) {
                        bv = v;
                        bi = g;
                    }
                }
                if (bi >= 0) {
                    nf_moe_gkeep[bi] = 1;
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int e = lid; e < n_expert; e += lsize) {
            if (!nf_moe_gkeep[e / n_per]) {
                nf_moe_sel[e] = -NF_INF;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    /* top-k: work-item lid owns experts lid + c * L, c < 32, with bit c of `taken` marking a pick */
    if (sel_mode != NF_MOE_SEL_PRESELECTED) {
        uint taken = 0u;
        for (int j = 0; j < k; j++) {
            float bv = -NF_INF;
            int bi = -1;
            int c = 0;
            for (int e = lid; e < n_expert; e += lsize, c++) {
                if ((taken >> c) & 1u) {
                    continue;
                }
                const float v = nf_moe_key(nf_moe_sel[e]);
                if (nf_moe_better(v, e, bv, bi)) {
                    bv = v;
                    bi = e;
                }
            }
            const int pick = nf_moe_wg_argmax(nf_moe_rv, nf_moe_ri, bv, bi);
            if (pick >= 0 && pick % lsize == lid) {
                taken |= 1u << (pick / lsize);
            }
            if (lid == 0) {
                nf_moe_pick[j] = pick;
            }
        }
    } else {
        for (int j = lid; j < k; j += lsize) {
            nf_moe_pick[j] = ids[row + j];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    /* weights, in the sequential order of the Java twin */
    if (lid == 0) {
        float w[NF_MOE_MAX_K];
        for (int j = 0; j < k; j++) {
            int id = nf_moe_pick[j];
            if (sel_mode != NF_MOE_SEL_PRESELECTED && id_div > 0 && id >= 0) {
                id = nf_moe_div_id(id, id_div);
            }
            nf_moe_pick[j] = id;
            w[j] = (id >= 0 && id < n_expert) ? nf_moe_probs[id] : 0.0f;
        }
        if (gating == NF_MOE_GATING_SOFTMAX_WEIGHT) {
            float wm = -NF_INF;
            for (int j = 0; j < k; j++) {
                wm = fmax(wm, w[j]);
            }
            float s = 0.0f;
            for (int j = 0; j < k; j++) {
                w[j] = exp(w[j] - wm);
                s += w[j];
            }
            const float inv = 1.0f / s;
            for (int j = 0; j < k; j++) {
                w[j] *= inv;
            }
        }
        if (norm_w) {
            float s = 0.0f;
            for (int j = 0; j < k; j++) {
                s += w[j];
            }
            s = nf_clamp_ggml(s, NF_MOE_NORM_MIN, NF_INF);
            for (int j = 0; j < k; j++) {
                w[j] = w[j] / s;
            }
        }
        if (w_scale != 0.0f && w_scale != 1.0f) {
            for (int j = 0; j < k; j++) {
                w[j] *= w_scale;
            }
        }
        for (int j = 0; j < k; j++) {
            ids[row + j] = nf_moe_pick[j];
            weights[row + j] = w[j];
        }
    }
}

/* deepseek4 hash layers: ids[t * k + j] = tid2eid[tokens[t] * k + j] (models/deepseek4.cpp:1297-1299,
 * ffn_gate_tid2eid is [k, n_vocab] I32). The weights then come from LLM_MoE_Router with sel_mode 3,
 * because build_moe_ffn takes them from the gating probabilities of the given ids. */
NF_KERNEL void LLM_MoE_HashIds(
    const int n_tokens, const int k,
    NF_GLOBAL const int* NF_RESTRICT tokens,
    NF_GLOBAL const int* NF_RESTRICT tid2eid,
    NF_GLOBAL int* NF_RESTRICT ids) {
    const int gid = get_global_id(0);
    if (gid >= n_tokens * k) return;
    const int t = gid / k, j = gid % k;
    ids[gid] = tid2eid[(nf_i64)tokens[t] * k + j];
}

/*
 * dst[t][c] = sum_j w[t][j] * expert_out[t * k + j][c], summed j = 0, 1, .. k-1 as llama-graph.cpp:2541-2567
 * adds the expert views. flags: NF_MOE_WS_UNWEIGHTED takes w = 1 (llama4 weight-before-FFN, whose
 * inputs were already weighted by LLM_MoE_ScaleInput); NF_MOE_WS_ACCUMULATE adds the sum to dst.
 */
NF_KERNEL void LLM_MoE_WeightedSum(
    const int n_tokens, const int n_embd, const int k, const int flags,
    NF_GLOBAL const float* NF_RESTRICT expert_out,
    NF_GLOBAL const float* NF_RESTRICT weights,
    NF_GLOBAL float* dst) {
    const int gid = get_global_id(0);
    if (gid >= n_tokens * n_embd) return;
    const int t = gid / n_embd, c = gid % n_embd;
    const int unweighted = flags & NF_MOE_WS_UNWEIGHTED;
    float acc = 0.0f;
    for (int j = 0; j < k; j++) {
        const nf_i64 p = (nf_i64)t * k + j;
        const float v = expert_out[p * n_embd + c];
        acc += unweighted ? v : v * weights[p];
    }
    dst[gid] = (flags & NF_MOE_WS_ACCUMULATE) ? acc + dst[gid] : acc;
}

/* data[s][c] += bias[slots[s]][c] (ggml_add_id, expert biases, llama-graph.cpp:2381, 2397, 2410, 2525). */
NF_KERNEL void LLM_AddId(
    const int n_sel, const int dim,
    NF_GLOBAL const int* NF_RESTRICT slots,
    NF_GLOBAL const float* NF_RESTRICT bias,
    NF_GLOBAL float* data) {
    const int gid = get_global_id(0);
    if (gid >= n_sel * dim) return;
    const int s = gid / dim, c = gid % dim;
    data[gid] += bias[(nf_i64)slots[s] * dim + c];
}

/* Bucket of an id: the expert, or n_expert for an id out of range. */
NF_INLINE int nf_moe_bucket(int id, int n_expert) {
    return (id >= 0 && id < n_expert) ? id : n_expert;
}

/*
 * Stable counting sort of the n_tokens * k pairs by expert, one workgroup:
 *   counts[e]  = pairs routed to e; offsets[e] = exclusive prefix sum of counts
 *   sorted row i = offsets[e] + (rank of the pair among e's pairs in ascending pair order):
 *       row_token[i] = pair index p = t * k + j, row_slot[i] = e
 * Pairs whose id is outside [0, n_expert) follow all valid rows with row_slot = -1
 * ({T}_BatchMatMulId writes 0 for them). Each pass handles L pairs at a time; work-item lid ranks its
 * pair by scanning the chunk, and the chunk's last pair of a bucket advances that bucket's cursor, so
 * no atomics are used and the result is deterministic.
 */
NF_KERNEL void LLM_MoE_Bucketize(
    const int n_tokens, const int k, const int n_expert,
    NF_GLOBAL const int* NF_RESTRICT ids,
    NF_GLOBAL int* NF_RESTRICT counts,
    NF_GLOBAL int* NF_RESTRICT offsets,
    NF_GLOBAL int* NF_RESTRICT row_token,
    NF_GLOBAL int* NF_RESTRICT row_slot) {
    if (get_group_id(0) != 0) return;
    NF_LOCAL int nf_moe_cur[NF_MOE_MAX_EXPERTS + 1];
    NF_LOCAL int nf_moe_chunk[LLM_DIMX];
    const int lid = get_local_id(0), lsize = get_local_size(0);
    const int n_rows = n_tokens * k;
    const int nb = n_expert + 1;
    for (int b = lid; b < nb; b += lsize) {
        nf_moe_cur[b] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int base = 0; base < n_rows; base += lsize) {
        const int p = base + lid;
        const int b = p < n_rows ? nf_moe_bucket(ids[p], n_expert) : -1;
        nf_moe_chunk[lid] = b;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (b >= 0) {
            int rank = 0, last = 1;
            for (int l = 0; l < lsize; l++) {
                if (nf_moe_chunk[l] == b) {
                    if (l < lid) rank++;
                    else if (l > lid) last = 0;
                }
            }
            if (last) {
                nf_moe_cur[b] += rank + 1;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int b = lid; b < n_expert; b += lsize) {
        counts[b] = nf_moe_cur[b];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid == 0) {
        int run = 0;
        for (int b = 0; b < nb; b++) {
            const int c = nf_moe_cur[b];
            if (b < n_expert) {
                offsets[b] = run;
            }
            nf_moe_cur[b] = run;
            run += c;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int base = 0; base < n_rows; base += lsize) {
        const int p = base + lid;
        const int b = p < n_rows ? nf_moe_bucket(ids[p], n_expert) : -1;
        nf_moe_chunk[lid] = b;
        barrier(CLK_LOCAL_MEM_FENCE);
        int rank = 0, last = 1;
        if (b >= 0) {
            for (int l = 0; l < lsize; l++) {
                if (nf_moe_chunk[l] == b) {
                    if (l < lid) rank++;
                    else if (l > lid) last = 0;
                }
            }
            const int pos = nf_moe_cur[b] + rank;
            row_token[pos] = p;
            row_slot[pos] = b < n_expert ? b : -1;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (b >= 0 && last) {
            nf_moe_cur[b] += rank + 1;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

/* llama4 weight-before-FFN: out[t * k + j][c] = x[t][c] * w[t][j] (llama-graph.cpp:2360-2365). */
NF_KERNEL void LLM_MoE_ScaleInput(
    const int n_tokens, const int n_embd, const int k,
    NF_GLOBAL const float* NF_RESTRICT x,
    NF_GLOBAL const float* NF_RESTRICT weights,
    NF_GLOBAL float* NF_RESTRICT out) {
    const nf_i64 gid = (nf_i64)get_global_id(0);
    const nf_i64 total = (nf_i64)n_tokens * k * n_embd;
    if (gid >= total) return;
    const nf_i64 p = gid / n_embd;
    const int c = (int)(gid % n_embd);
    const nf_i64 t = p / k;
    out[gid] = x[t * n_embd + c] * weights[p];
}

/* moe.cl - MoE router, hash ids, weighted sum, add-id, bucketize, block-align and input-scale kernels. Requires: common/dialect.h, common/config.h, common/math.h, common/reduce.h, common/quant_blocks.h, common/matmul_tiled_family.h */
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
 *   LLM_MoE_BucketCount  global = ceil(n_tokens * k / chunk) * L, local = L (power of two <= LLM_DIMX)
 *   LLM_MoE_BucketScan   global = local = L (one workgroup, power of two <= LLM_DIMX)
 *   LLM_MoE_BucketScatter global = ceil(n_tokens * k / chunk) * L, local = L (power of two <= LLM_DIMX)
 *   LLM_MoE_AlignBlocks  global = G * L for any G >= 1, local = L (power of two <= LLM_DIMX)
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
 *   groups = when n_groups > 1 and sel_mode < 3 and !per_group: group score = sum of the group's two largest sel;
 *            the n_group_used best groups are kept, the others' sel set to -INF (llama-graph.cpp:2281-2305)
 *   per_group = when n_groups > 1 and sel_mode < 3: each group contributes k / n_groups of its own best experts,
 *            groups in index order (vLLM's InternS1-Pro router, interns1_pro.py:196-213)
 *   ids    = k rounds of arg-max over the unpicked sel (descending, ties to the lower index)
 *   id_div = when > 0 and sel_mode < 3, ids are divided as nf_moe_div_id (grovemoe chunk experts)
 *   w[j]   = probs[ids[j]]; SOFTMAX_WEIGHT: softmax over the k weights; norm_w: w / max(sum w,
 *            NF_MOE_NORM_MIN); w_scale not in {0, 1}: w * w_scale (llama-graph.cpp:2320-2345)
 * Writes ids[t * k + j] and weights[t * k + j]. Router logits must already include ffn_gate_inp.bias.
 * Host limits: n_expert <= min(NF_MOE_MAX_EXPERTS, 32 * L), k <= min(n_expert, NF_MOE_MAX_K),
 * n_groups <= NF_MOE_MAX_GROUPS with n_expert % n_groups == 0 and n_expert / n_groups >= 2,
 * n_group_used <= n_groups; per_group needs k divisible by n_groups.
 */
NF_KERNEL void LLM_MoE_Router(
    const int n_tokens, const int n_expert, const int k, const int gating,
    const int n_groups, const int n_group_used, const int norm_w, const float w_scale,
    const int has_bias, const int sel_mode, const int id_div, const int per_group,
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

    /* group-limited routing (not used by per-group selection) */
    if (sel_mode != NF_MOE_SEL_PRESELECTED && n_groups > 1 && !per_group) {
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
    if (sel_mode != NF_MOE_SEL_PRESELECTED && n_groups > 1 && per_group) {
        const int n_per = n_expert / n_groups;
        const int per_k = k / n_groups;
        uint taken = 0u;
        for (int g = 0; g < n_groups; g++) {
            const int lo = g * n_per;
            const int hi = lo + n_per;
            for (int j = 0; j < per_k; j++) {
                float bv = -NF_INF;
                int bi = -1;
                int c = 0;
                for (int e = lid; e < n_expert; e += lsize, c++) {
                    if (e < lo || e >= hi || ((taken >> c) & 1u)) {
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
                    nf_moe_pick[g * per_k + j] = pick;
                }
            }
        }
    } else if (sel_mode != NF_MOE_SEL_PRESELECTED) {
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

/*
 * LongCat-Flash identity (zero-compute) experts: ids >= n_real_experts select no expert FFN, so their weight is
 * summed per token into zero_weights and the slot is remapped to expert 0 with weight 0 before any expert kernel
 * reads it (vLLM's ZeroExpertRouter masks the same way, router/zero_expert_router.py:86-95). One work-item per
 * token; k <= NF_MOE_MAX_K.
 */
NF_KERNEL void LLM_MoE_ZeroExperts(
    const int n_tokens, const int k, const int n_real_experts,
    NF_GLOBAL int* NF_RESTRICT ids,
    NF_GLOBAL float* NF_RESTRICT weights,
    NF_GLOBAL float* NF_RESTRICT zero_weights) {
    const int t = get_global_id(0);
    if (t >= n_tokens) return;
    float zero = 0.0f;
    for (int j = 0; j < k; j++) {
        const nf_i64 p = (nf_i64)t * k + j;
        if (ids[p] >= n_real_experts) {
            zero += weights[p];
            ids[p] = 0;
            weights[p] = 0.0f;
        }
    }
    zero_weights[t] = zero;
}

/*
 * EPLB replica mapping: out[p] = logical_to_physical[e * map_stride + hash(t) % replica_count[e]] for the logical
 * expert e = ids[p] of pair p = t * k + j, or -1 for an id outside [0, n_logical). The token index goes through the
 * reference's Knuth multiplicative hash and its low bits choose one of the expert's physical replicas
 * (vllm/model_executor/layers/fused_moe/eplb_triton.py eplb_map_and_record), so a pair is computed by exactly one
 * rank even when the expert is replicated. Ids of any pair width k >= 1; one work-item per pair.
 */
NF_KERNEL void LLM_MoE_EplbMap(
    const int n_pairs, const int k, const int n_logical, const int map_stride,
    NF_GLOBAL const int* NF_RESTRICT ids,
    NF_GLOBAL const int* NF_RESTRICT replica_count,
    NF_GLOBAL const int* NF_RESTRICT logical_to_physical,
    NF_GLOBAL int* NF_RESTRICT out) {
    const int p = get_global_id(0);
    if (p >= n_pairs) return;
    const int e = ids[p];
    if (e < 0 || e >= n_logical) {
        out[p] = -1;
        return;
    }
    const int replicas = replica_count[e];
    const unsigned int token = (unsigned int)(p / k);
    const unsigned int hashed = token * 2654435769u;
    const int slot = (int)(hashed % (unsigned int)replicas);
    out[p] = logical_to_physical[e * map_stride + slot];
}

/*
 * dst[t][c] += x[t][c] * zero_weights[t]: the identity experts' contribution, which vLLM's
 * zero_experts_compute_triton accumulates from the expert input over the selected zero slots
 * (fused_moe.py:1051-1126). x and dst are element offsets; they may name the same buffer.
 */
NF_KERNEL void LLM_MoE_ZeroApply(
    const int n_tokens, const int n_embd, const int x_off, const int dst_off,
    NF_GLOBAL const float* NF_RESTRICT x,
    NF_GLOBAL const float* NF_RESTRICT zero_weights,
    NF_GLOBAL float* dst) {
    const nf_i64 gid = (nf_i64)get_global_id(0);
    if (gid >= (nf_i64)n_tokens * n_embd) return;
    const int t = (int)(gid / n_embd);
    dst[dst_off + gid] += x[x_off + gid] * zero_weights[t];
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
 * Rank of this work-item's bucket b among the earlier entries of the local chunk, and whether no later
 * entry of the chunk shares it (the work-item then advances the bucket's cursor).
 */
NF_INLINE int nf_moe_rank(NF_LOCAL_PTR const int* chunk, int b, int lid, int lsize, int* last) {
    int rank = 0;
    *last = 1;
    for (int l = 0; l < lsize; l++) {
        if (chunk[l] == b) {
            if (l < lid) rank++;
            else if (l > lid) *last = 0;
        }
    }
    return rank;
}

/*
 * Writes the sorted rows of the pairs [p0, p1) in L-pair passes: pair p of bucket b goes to
 * cur[b] + (its rank among b's pairs of the pass), and the pass's last pair of b advances cur[b], so the
 * rows of a bucket keep ascending pair order without atomics. Every work-item of the workgroup calls it
 * with the same range (the barriers inside are uniform).
 */
NF_INLINE void nf_moe_scatter_rows(NF_GLOBAL const int* NF_RESTRICT ids, int n_expert, int p0, int p1,
                                   NF_LOCAL_PTR int* cur, NF_LOCAL_PTR int* chunk, int lid, int lsize,
                                   NF_GLOBAL int* NF_RESTRICT row_token, NF_GLOBAL int* NF_RESTRICT row_slot) {
    for (int base = p0; base < p1; base += lsize) {
        const int p = base + lid;
        const int b = p < p1 ? nf_moe_bucket(ids[p], n_expert) : -1;
        chunk[lid] = b;
        barrier(CLK_LOCAL_MEM_FENCE);
        int rank = 0, last = 0;
        if (b >= 0) {
            rank = nf_moe_rank(chunk, b, lid, lsize, &last);
            const int pos = cur[b] + rank;
            row_token[pos] = p;
            row_slot[pos] = b < n_expert ? b : -1;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (b >= 0 && last) {
            cur[b] += rank + 1;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

/*
 * In-place exclusive prefix sum of v[0, n) over one workgroup, returning the total: work-item lid sums
 * the contiguous segment [lid * s, (lid + 1) * s) (s = ceil(n / L)), part[] scans the L segment sums
 * (Hillis-Steele), and each work-item rewrites its segment. Integer adds, so the result is exact and
 * independent of the work-group size. Every work-item must call it (it holds barriers).
 */
NF_INLINE int nf_moe_excl_scan(NF_LOCAL_PTR int* v, int n, NF_LOCAL_PTR int* part, int lid, int lsize) {
    const int seg = (n + lsize - 1) / lsize;
    const int b0 = lid * seg < n ? lid * seg : n;
    const int b1 = b0 + seg < n ? b0 + seg : n;
    int sum = 0;
    for (int b = b0; b < b1; b++) sum += v[b];
    part[lid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = 1; off < lsize; off <<= 1) {
        const int add = lid >= off ? part[lid - off] : 0;
        barrier(CLK_LOCAL_MEM_FENCE);
        part[lid] += add;
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    int run = lid == 0 ? 0 : part[lid - 1];
    const int total = part[lsize - 1];
    for (int b = b0; b < b1; b++) {
        const int c = v[b];
        v[b] = run;
        run += c;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    return total;
}

/*
 * Stable counting sort of the n_tokens * k pairs by expert, one workgroup:
 *   counts[e]  = pairs routed to e; offsets[e] = exclusive prefix sum of counts
 *   sorted row i = offsets[e] + (rank of the pair among e's pairs in ascending pair order):
 *       row_token[i] = pair index p = t * k + j, row_slot[i] = e
 * Pairs whose id is outside [0, n_expert) follow all valid rows with row_slot = -1
 * ({T}_BatchMatMulId writes 0 for them). Each pass handles L pairs at a time; work-item lid ranks its
 * pair by scanning the chunk, and the chunk's last pair of a bucket advances that bucket's cursor, so
 * no atomics are used and the result is deterministic. Large batches use the multi-workgroup
 * LLM_MoE_BucketCount / LLM_MoE_BucketScan / LLM_MoE_BucketScatter, which write the same output.
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
            int last = 0;
            const int rank = nf_moe_rank(nf_moe_chunk, b, lid, lsize, &last);
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
    nf_moe_scatter_rows(ids, n_expert, 0, n_rows, nf_moe_cur, nf_moe_chunk, lid, lsize, row_token, row_slot);
}

/*
 * Multi-workgroup stable counting sort, phase 1 of 3 (the output of all three equals
 * LLM_MoE_Bucketize). Workgroup g counts the buckets of the pairs [g * chunk, min((g + 1) * chunk,
 * n_tokens * k)) into hist[g * (n_expert + 1) + b], the out-of-range bucket b = n_expert last. The
 * counts do not depend on the order of the local atomic increments. The host launches
 * ceil(n_tokens * k / chunk) workgroups (Java: MoeKernels.bucketGroups).
 */
NF_KERNEL void LLM_MoE_BucketCount(
    const int n_tokens, const int k, const int n_expert, const int chunk,
    NF_GLOBAL const int* NF_RESTRICT ids,
    NF_GLOBAL int* NF_RESTRICT hist) {
    NF_LOCAL int nf_moe_cnt[NF_MOE_MAX_EXPERTS + 1];
    const int lid = get_local_id(0), lsize = get_local_size(0);
    const int g = get_group_id(0);
    const int n_rows = n_tokens * k;
    const int nb = n_expert + 1;
    for (int b = lid; b < nb; b += lsize) {
        nf_moe_cnt[b] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    const nf_i64 p0 = (nf_i64)g * chunk;
    const int p1 = p0 + chunk < n_rows ? (int)(p0 + chunk) : n_rows;
    for (nf_i64 p = p0 + lid; p < p1; p += lsize) {
        atomic_inc(&nf_moe_cnt[nf_moe_bucket(ids[p], n_expert)]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int b = lid; b < nb; b += lsize) {
        hist[(nf_i64)g * nb + b] = nf_moe_cnt[b];
    }
}

/*
 * Multi-workgroup stable counting sort, phase 2 of 3, one workgroup: from the per-group bucket counts
 * of LLM_MoE_BucketCount writes counts[e] and offsets[e] exactly as LLM_MoE_Bucketize does, and
 * rewrites hist[g * (n_expert + 1) + b] in place to the first sorted row of group g's pairs of bucket b:
 * offsets[b] (valid_total for the out-of-range bucket) plus the pairs of b in the groups before g.
 */
NF_KERNEL void LLM_MoE_BucketScan(
    const int n_tokens, const int k, const int n_expert, const int chunk,
    NF_GLOBAL int* NF_RESTRICT hist,
    NF_GLOBAL int* NF_RESTRICT counts,
    NF_GLOBAL int* NF_RESTRICT offsets) {
    if (get_group_id(0) != 0) return;
    NF_LOCAL int nf_moe_tot[NF_MOE_MAX_EXPERTS + 1];
    NF_LOCAL int nf_moe_part[LLM_DIMX];
    const int lid = get_local_id(0), lsize = get_local_size(0);
    const int n_rows = n_tokens * k;
    const int nb = n_expert + 1;
    const int n_groups = (n_rows + chunk - 1) / chunk;
    for (int b = lid; b < nb; b += lsize) {
        int run = 0;
        for (int g = 0; g < n_groups; g++) {
            const nf_i64 i = (nf_i64)g * nb + b;
            const int c = hist[i];
            hist[i] = run;
            run += c;
        }
        nf_moe_tot[b] = run;
        if (b < n_expert) {
            counts[b] = run;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    nf_moe_excl_scan(nf_moe_tot, nb, nf_moe_part, lid, lsize);
    for (int b = lid; b < nb; b += lsize) {
        const int start = nf_moe_tot[b];
        if (b < n_expert) {
            offsets[b] = start;
        }
        for (int g = 0; g < n_groups; g++) {
            hist[(nf_i64)g * nb + b] += start;
        }
    }
}

/*
 * Multi-workgroup stable counting sort, phase 3 of 3: workgroup g writes the sorted rows of its pairs
 * [g * chunk, min((g + 1) * chunk, n_tokens * k)) from the row starts LLM_MoE_BucketScan left in hist,
 * ranking them in L-pair passes as LLM_MoE_Bucketize does, so row_token and row_slot equal its output.
 */
NF_KERNEL void LLM_MoE_BucketScatter(
    const int n_tokens, const int k, const int n_expert, const int chunk,
    NF_GLOBAL const int* NF_RESTRICT ids,
    NF_GLOBAL const int* NF_RESTRICT hist,
    NF_GLOBAL int* NF_RESTRICT row_token,
    NF_GLOBAL int* NF_RESTRICT row_slot) {
    NF_LOCAL int nf_moe_cur[NF_MOE_MAX_EXPERTS + 1];
    NF_LOCAL int nf_moe_chunk[LLM_DIMX];
    const int lid = get_local_id(0), lsize = get_local_size(0);
    const int g = get_group_id(0);
    const int n_rows = n_tokens * k;
    const int nb = n_expert + 1;
    for (int b = lid; b < nb; b += lsize) {
        nf_moe_cur[b] = hist[(nf_i64)g * nb + b];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    const nf_i64 p0 = (nf_i64)g * chunk;
    if (p0 >= n_rows) return;
    const int p1 = p0 + chunk < n_rows ? (int)(p0 + chunk) : n_rows;
    nf_moe_scatter_rows(ids, n_expert, (int)p0, p1, nf_moe_cur, nf_moe_chunk, lid, lsize, row_token, row_slot);
}

/*
 * Block-aligned copy of the sorted rows for the grouped GEMM ({T}_GroupedGemm,
 * common/matmul_grouped_family.h; the layout of vLLM moe_align_block_size) over the counts, offsets
 * and sorted rows of LLM_MoE_Bucketize. n_tiles is the host's worst-case tile bound (Java:
 * MoeKernels.groupedTiles) and fixes the written length n_tiles * NF_GEMM_TN: bucket b's rows keep
 * their sorted order at positions [tile_base[b] * NF_GEMM_TN, ..), tile_base being the exclusive
 * prefix sum of the buckets' ceil(count / NF_GEMM_TN) tile counts, and the bucket's tile padding
 * holds gemm_token = gemm_slot = -1. A position's pair is gemm_token[i] (p = t * k + j) and its
 * expert gemm_slot[i] (the invalid-id bucket keeps -1), so tile q's expert is
 * gemm_slot[q * NF_GEMM_TN]: every tile's first position is a real row, and the tiles past the
 * aligned count hold only fillers. The sum-order and output positions of {T}_GroupedGemm equal
 * {T}_BatchMatMulId, whose row_token / row_slot this kernel consumes unchanged.
 *
 * Any number of workgroups: each computes tile_base in local memory, then every position is gathered
 * by exactly one work-item of the grid-stride loop (the bucket of its tile by binary search over
 * tile_base), so the output does not depend on the grid.
 */
NF_KERNEL void LLM_MoE_AlignBlocks(
    const int n_tokens, const int k, const int n_expert, const int n_tiles,
    NF_GLOBAL const int* NF_RESTRICT counts,
    NF_GLOBAL const int* NF_RESTRICT offsets,
    NF_GLOBAL const int* NF_RESTRICT row_token,
    NF_GLOBAL const int* NF_RESTRICT row_slot,
    NF_GLOBAL int* NF_RESTRICT gemm_token,
    NF_GLOBAL int* NF_RESTRICT gemm_slot) {
    NF_LOCAL int nf_moe_base[NF_MOE_MAX_EXPERTS + 1];
    NF_LOCAL int nf_moe_part[LLM_DIMX];
    const int lid = get_local_id(0), lsize = get_local_size(0);
    const int n_rows = n_tokens * k;
    const int nb = n_expert + 1;
    /* requires n_expert >= 1 (MoeKernels validates), so both arrays below are in range */
    const int valid_total = offsets[n_expert - 1] + counts[n_expert - 1];
    for (int b = lid; b < nb; b += lsize) {
        const int c = b < n_expert ? counts[b] : n_rows - valid_total;
        nf_moe_base[b] = (c + NF_GEMM_TN - 1) / NF_GEMM_TN;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    const int used = nf_moe_excl_scan(nf_moe_base, nb, nf_moe_part, lid, lsize);
    const nf_i64 n_pos = (nf_i64)n_tiles * NF_GEMM_TN;
    const nf_i64 stride = (nf_i64)get_global_size(0);
    for (nf_i64 q = (nf_i64)get_global_id(0); q < n_pos; q += stride) {
        const int tile = (int)(q / NF_GEMM_TN);
        int tok = -1, slot = -1;
        if (tile < used) {
            int lo = 0, hi = nb - 1;
            while (lo < hi) {
                const int mid = (lo + hi + 1) >> 1;
                if (nf_moe_base[mid] <= tile) lo = mid;
                else hi = mid - 1;
            }
            const int idx = (int)(q - (nf_i64)nf_moe_base[lo] * NF_GEMM_TN);
            const int cnt = lo < n_expert ? counts[lo] : n_rows - valid_total;
            if (idx < cnt) {
                const int start = lo < n_expert ? offsets[lo] : valid_total;
                tok = row_token[start + idx];
                slot = row_slot[start + idx];
            }
        }
        gemm_token[q] = tok;
        gemm_slot[q] = slot;
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

/* sampling_topk.cl - LLM_TopK_Partial partial top-k for sampling. Requires: common/dialect.h, common/config.h */
/*
 * Candidate reduction for the host sampler: instead of reading back the whole logits
 * vector, the host reads G*k candidates and finishes temperature, top-k, top-p, min-p and penalties
 * on them, as llama.cpp's CPU sampler chain does (llama.cpp src/llama-sampler.cpp:321
 * llama_sampler_top_k_impl keeps the k largest logits; at a tie on the boundary value the lower
 * index is kept here).
 *
 * Candidate order is the total order of LLM_Argmax (sampling.cl): the larger value wins and equal
 * values go to the lower index. NaN logits are never candidates.
 */

/* Index sentinel of a candidate slot that holds no element. */
#define NF_TKP_NONE 0x7fffffff

/* 1 when (v, i) ranks before (bv, bi): larger value first, then lower index. Non-NaN inputs. */
NF_INLINE int nf_tkp_before(float v, int i, float bv, int bi) {
    return v > bv || (v == bv && i < bi);
}

/*
 * LLM_TopK_Partial: workgroup g of G = get_num_groups(0) owns the slice
 * [g * S, min(n, (g + 1) * S)) of logits with S = ceil(n / G), and writes the k best elements of
 * its slice in rank order to cand_val[g*k + r] = logits[idx] * inv_temp and cand_idx[g*k + r] = idx
 * (r < k). Slots beyond the slice's non-NaN element count hold cand_val = -INF, cand_idx = -1.
 *
 * Algorithm: k rounds of a workgroup argmax restricted to the elements that rank after the previous
 * round's winner, so round r yields the r-th best element of the slice in the total order above.
 * inv_temp > 0 scales the reported values only; it never changes the order.
 *
 * Launch contract: 1-D, global = {G * L}, local = {L}, L a power of two <= LLM_DIMX (CUDA: a
 * multiple of 32), 1 <= n <= 2^30 (slice bounds and the strided scan stay within int),
 * 1 <= k <= 64 and G at most the device's 1-D group limit (never folded). Every work-item runs the
 * same number of rounds, so all reach every barrier.
 */
NF_KERNEL void LLM_TopK_Partial(
    const int n, const int k, const float inv_temp,
    NF_GLOBAL const float* NF_RESTRICT logits,
    NF_GLOBAL float* NF_RESTRICT cand_val,
    NF_GLOBAL int* NF_RESTRICT cand_idx) {
    NF_LOCAL float best_v[LLM_DIMX];
    NF_LOCAL int best_i[LLM_DIMX];
    const int lid = get_local_id(0);
    const int lsize = get_local_size(0);
    const int g = get_group_id(0);
    const int n_groups = get_num_groups(0);
    const int slice = (n + n_groups - 1) / n_groups;
    const int s0 = min(n, g * slice);
    const int s1 = min(n, s0 + slice);
    const nf_i64 out_base = (nf_i64)g * k;

    /* Previous winner; (+INF, -1) ranks before every element, so round 0 sees the whole slice. */
    float pv = NF_INF;
    int pi = -1;
    for (int r = 0; r < k; r++) {
        float bv = -NF_INF;
        int bi = NF_TKP_NONE;
        for (int i = s0 + lid; i < s1; i += lsize) {
            const float v = logits[i];
            if (isnan(v)) continue;
            if (nf_tkp_before(pv, pi, v, i) && nf_tkp_before(v, i, bv, bi)) {
                bv = v;
                bi = i;
            }
        }
        best_v[lid] = bv;
        best_i[lid] = bi;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int stride = lsize >> 1; stride > 0; stride >>= 1) {
            if (lid < stride) {
                const float ov = best_v[lid + stride];
                const int oi = best_i[lid + stride];
                if (nf_tkp_before(ov, oi, best_v[lid], best_i[lid])) {
                    best_v[lid] = ov;
                    best_i[lid] = oi;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        const float wv = best_v[0];
        const int wi = best_i[0];
        barrier(CLK_LOCAL_MEM_FENCE);
        if (wi == NF_TKP_NONE) {
            /* The slice is exhausted (uniform: every work-item read the same winner). */
            for (int q = r + lid; q < k; q += lsize) {
                cand_val[out_base + q] = -NF_INF;
                cand_idx[out_base + q] = -1;
            }
            break;
        }
        if (lid == 0) {
            cand_val[out_base + r] = wv * inv_temp;
            cand_idx[out_base + r] = wi;
        }
        pv = wv;
        pi = wi;
    }
}

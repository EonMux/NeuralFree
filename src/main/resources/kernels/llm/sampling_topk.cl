/* sampling_topk.cl - partial top-k, row top-k and pre-top-k logit edits for sampling.
 * Requires: common/dialect.h, common/config.h */
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

/*
 * LLM_Row_TopK: row-wise top-k in rank order with the row's softmax denominator, for the device
 * readouts that read a few values per row back instead of the whole row (the dflash drafter's
 * p_min / DSpark / LiLiCorr readouts and the backend-sampled verification rows).
 *
 * Workgroup `row` of n_rows scans logits[row * size .. + size) once for the maximum, once for
 * sum_exp[row] = sum_j exp(logits[j] - max), then k rounds of a workgroup argmax restricted to the
 * elements ranking after the previous round's winner (llama.cpp llama_sampler_top_k_impl keeps the
 * k largest logits; at a tie on the boundary value the lower index is kept, as the CPU CpuOps.topK
 * scan does). top_val[row * k + r] / top_idx[row * k + r] receive the r-th best value and its index
 * in the row's total order (larger value first, then lower index); rounds past the row's element
 * count hold top_val = -INF, top_idx = -1. NaN logits are never candidates and never counted.
 * Dispatch: global = {n_rows * L}, local = {L}, L a power of two <= LLM_DIMX (CUDA: a multiple of
 * 32), 1 <= size <= 2^30, 1 <= k <= 64. Every work-item runs the same number of rounds, so all
 * reach every barrier.
 */
NF_KERNEL void LLM_Row_TopK(
    const int size, const int k, const int n_rows,
    NF_GLOBAL const float* logits,
    NF_GLOBAL float* top_val,
    NF_GLOBAL int* top_idx,
    NF_GLOBAL float* sum_exp) {
    const int row = get_group_id(0);
    if (row >= n_rows) return;
    const int lid = get_local_id(0);
    const int lsize = get_local_size(0);
    NF_LOCAL float vals[LLM_DIMX];
    NF_LOCAL int idxs[LLM_DIMX];
    const nf_i64 base = (nf_i64)row * size;

    float my_max = -NF_INF;
    for (int i = lid; i < size; i += lsize) {
        const float v = logits[base + i];
        if (!isnan(v) && v > my_max) {
            my_max = v;
        }
    }
    vals[lid] = my_max;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = lsize >> 1; stride > 0; stride >>= 1) {
        if (lid < stride && vals[lid + stride] > vals[lid]) {
            vals[lid] = vals[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float row_max = vals[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    float my_sum = 0.0f;
    for (int i = lid; i < size; i += lsize) {
        const float v = logits[base + i];
        if (!isnan(v)) {
            my_sum += exp(v - row_max);
        }
    }
    vals[lid] = my_sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = lsize >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
            vals[lid] += vals[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) {
        sum_exp[row] = vals[0];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    float pv = NF_INF;
    int pi = -1;
    for (int r = 0; r < k; r++) {
        float bv = -NF_INF;
        int bi = NF_TKP_NONE;
        for (int i = lid; i < size; i += lsize) {
            const float v = logits[base + i];
            if (isnan(v)) continue;
            if (nf_tkp_before(pv, pi, v, i) && nf_tkp_before(v, i, bv, bi)) {
                bv = v;
                bi = i;
            }
        }
        vals[lid] = bv;
        idxs[lid] = bi;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int stride = lsize >> 1; stride > 0; stride >>= 1) {
            if (lid < stride && nf_tkp_before(vals[lid + stride], idxs[lid + stride], vals[lid], idxs[lid])) {
                vals[lid] = vals[lid + stride];
                idxs[lid] = idxs[lid + stride];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        const int wi = idxs[0];
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid == 0) {
            top_val[(nf_i64)row * k + r] = wi == NF_TKP_NONE ? -NF_INF : vals[0];
            top_idx[(nf_i64)row * k + r] = wi == NF_TKP_NONE ? -1 : wi;
        }
        pv = wi == NF_TKP_NONE ? -NF_INF : vals[0];
        pi = wi;
    }
}

/*
 * LLM_LogitsPrefilter: the pre-top-k logit edits of a truncated sampler chain, per row, over the whole
 * vocabulary row (llama.cpp backend sampling's logit-bias and penalties samplers run before its top-k):
 *
 *   logits[token] = NaN bias ? -INF : logits[token] + bias          (the fixed logit-bias stage)
 *   logits[token] = (l <= 0 ? l * repeat : l / repeat)
 *                   - (count * frequency + presence)                 (penalties, once per distinct token)
 *   logits[token] -= multiplier * base^(repeatLength - allowed)      (DRY)
 *
 * applied in the chain's order: penalties before DRY when meta[penalties_first] is 1, DRY before penalties
 * otherwise. Candidate 0 of every row is vocabulary element 0.
 *
 * Per-row layout (row r reads block r * NF_PF_META / r * NF_PF_FCFG):
 *   meta:  [0] bias offset   [1] bias count  [2] penalty offset  [3] penalty count  [4] penalties active
 *          [5] DRY offset    [6] DRY count   [7] DRY allowed     [8] DRY active      [9] breaker offset
 *          [10] breaker count [11] penalties-first flag
 *   fcfg:  [0] repeat  [1] frequency  [2] presence  [3] DRY multiplier  [4] DRY base
 * bias_ids / bias_vals hold the row's distinct biased tokens (a NaN value is the -INF ban); tokens holds
 * the penalty entries as (token, count) pairs and then the DRY window oldest first; br_meta holds four ints
 * per restart sequence (head, tail offset into br_tails, tail length, unused).
 *
 * Dispatch: global = {n_rows * L}, local = {L}, L a power of two <= LLM_DIMX (CUDA: a multiple of 32).
 * The DRY window is at most NF_PF_DRY_MAX tokens (the host path keeps wider windows). Every work-item of a
 * row reaches every barrier, so the row's edits are visible to the next stage.
 */
#define NF_PF_META     16
#define NF_PF_FCFG      8
#define NF_PF_BR        4
#define NF_PF_DRY_MAX 256

/* Penalties over the row's distinct window tokens: every distinct token once, with its window count. */
NF_INLINE void nf_pf_penalties(
    const int lid, const int lsize, const int logit_off, const int tok_off, const int count,
    NF_GLOBAL float* NF_RESTRICT logits, NF_GLOBAL const int* NF_RESTRICT tokens,
    const float repeat, const float frequency, const float presence) {
    for (int i = lid; i < count; i += lsize) {
        const int id = tokens[tok_off + i * 2];
        const int n = tokens[tok_off + i * 2 + 1];
        float l = logits[logit_off + id];
        l = l <= 0.0f ? l * repeat : l / repeat;
        logits[logit_off + id] = l - ((float) n * frequency + presence);
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
}

/* True when one of the token's restart sequences carries an empty tail (llama.cpp is_single_token_breaker). */
NF_INLINE int nf_pf_single_token_breaker(
    const int br_off, const int br_cnt, const int token,
    NF_GLOBAL const int* NF_RESTRICT br_meta) {
    for (int b = 0; b < br_cnt; b++) {
        if (br_meta[(br_off + b) * NF_PF_BR] == token && br_meta[(br_off + b) * NF_PF_BR + 2] == 0) {
            return 1;
        }
    }
    return 0;
}

/* DRY over the row's window (llama.cpp llama_sampler_dry_apply over rat(i) = window[N - 1 - i]). */
NF_INLINE void nf_pf_dry(
    const int lid, const int lsize, const int logit_off, const int tok_off, const int n_tok, const int allowed,
    const int br_off, const int br_cnt, const float multiplier, const float base,
    NF_GLOBAL float* NF_RESTRICT logits, NF_GLOBAL const int* NF_RESTRICT tokens,
    NF_GLOBAL const int* NF_RESTRICT br_meta, NF_GLOBAL const int* NF_RESTRICT br_tails,
    NF_LOCAL_PTR int* lens, NF_LOCAL_PTR int* rep_limit) {
    /* Step 1: a restart sequence caps the repeat length at the position that completes it. */
    if (lid == 0) {
        int limit = n_tok;
        for (int i = 0; i < n_tok; i++) {
            const int head = tokens[tok_off + (n_tok - 1 - i)];
            int longest = -1;
            for (int b = 0; b < br_cnt; b++) {
                const int bm = (br_off + b) * NF_PF_BR;
                if (br_meta[bm] != head) continue;
                const int tail_len = br_meta[bm + 2];
                if (tail_len > longest && tail_len <= i) {
                    const int tail_off = br_meta[bm + 1];
                    int match = 1;
                    for (int o = 0; o < tail_len; o++) {
                        if (br_tails[tail_off + o] != tokens[tok_off + (n_tok - 1 - (i - o - 1))]) {
                            match = 0;
                            break;
                        }
                    }
                    if (match) longest = tail_len;
                }
            }
            if (longest >= 0) {
                limit = i - longest;
                break;
            }
        }
        *rep_limit = limit;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    /* Step 2: the reverse Z lengths, in parallel over the window's suffixes. */
    for (int k = 1 + lid; k < n_tok; k += lsize) {
        const int lim = n_tok - k;
        int z = 0;
        while (z < lim && tokens[tok_off + (n_tok - 1 - z)] == tokens[tok_off + (n_tok - 1 - (z + k))]) {
            z++;
        }
        if (z > *rep_limit) z = *rep_limit;
        lens[k] = z;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    /* Step 3 and 4: the largest repeat length per token, then the penalty (llama.cpp llama-sampler.cpp:3533). */
    if (lid == 0) {
        const int limit = *rep_limit;
        if (limit >= allowed) {
            float max_exponent_float = 0.0f;
            if (base > 1.000001f) {
                max_exponent_float = 88.7228391f / log(base);
            }
            const int max_exponent = max_exponent_float >= 2147483648.0f ? 2147483647
                    : (max_exponent_float > 0.0f ? (int) max_exponent_float : 0);
            for (int k = 1; k < n_tok; k++) {
                const int len = lens[k];
                if (len < allowed) continue;
                const int token = tokens[tok_off + (n_tok - k)];
                int is_max = 1;
                for (int k2 = 1; k2 < n_tok; k2++) {
                    if (k2 == k) continue;
                    if (tokens[tok_off + (n_tok - k2)] != token) continue;
                    const int len2 = lens[k2];
                    if (len2 > len || (len2 == len && k2 < k)) {
                        is_max = 0;
                        break;
                    }
                }
                if (!is_max) continue;
                if (nf_pf_single_token_breaker(br_off, br_cnt, token, br_meta)) continue;
                int repeat_exp = len - allowed;
                if (max_exponent > 0 && repeat_exp > max_exponent) repeat_exp = max_exponent;
                logits[logit_off + token] -= multiplier * pow(base, (float) repeat_exp);
            }
        }
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
}

NF_KERNEL void LLM_LogitsPrefilter(
    const int vocab, const int n_rows,
    NF_GLOBAL float* NF_RESTRICT logits,
    NF_GLOBAL const int* NF_RESTRICT meta,
    NF_GLOBAL const float* NF_RESTRICT fcfg,
    NF_GLOBAL const int* NF_RESTRICT bias_ids,
    NF_GLOBAL const float* NF_RESTRICT bias_vals,
    NF_GLOBAL const int* NF_RESTRICT tokens,
    NF_GLOBAL const int* NF_RESTRICT br_meta,
    NF_GLOBAL const int* NF_RESTRICT br_tails) {
    const int row = get_group_id(0);
    if (row >= n_rows) return;
    const int lid = get_local_id(0);
    const int lsize = get_local_size(0);
    const int base_off = row * vocab;
    const int mb = row * NF_PF_META;
    const int fb = row * NF_PF_FCFG;

    const int bias_off = meta[mb + 0];
    const int bias_cnt = meta[mb + 1];
    const int pen_off = meta[mb + 2];
    const int pen_cnt = meta[mb + 3];
    const int pen_active = meta[mb + 4];
    const int dry_off = meta[mb + 5];
    const int dry_cnt = meta[mb + 6];
    const int dry_allowed = meta[mb + 7];
    const int dry_active = meta[mb + 8];
    const int br_off = meta[mb + 9];
    const int br_cnt = meta[mb + 10];
    const int penalties_first = meta[mb + 11];

    NF_LOCAL int lens[NF_PF_DRY_MAX];
    NF_LOCAL int rep_limit;

    /* The fixed logit-bias stage: a NaN value marks the -INF ban, any other value is added. */
    for (int i = lid; i < bias_cnt; i += lsize) {
        const int id = bias_ids[bias_off + i];
        if (id < 0 || id >= vocab) continue;
        const float b = bias_vals[bias_off + i];
        logits[base_off + id] = isnan(b) ? -NF_INF : logits[base_off + id] + b;
    }
    barrier(CLK_GLOBAL_MEM_FENCE);

    if (pen_active && penalties_first) {
        nf_pf_penalties(lid, lsize, base_off, pen_off, pen_cnt, logits, tokens,
                fcfg[fb + 0], fcfg[fb + 1], fcfg[fb + 2]);
    }
    if (dry_active) {
        nf_pf_dry(lid, lsize, base_off, dry_off, dry_cnt, dry_allowed, br_off, br_cnt,
                fcfg[fb + 3], fcfg[fb + 4], logits, tokens, br_meta, br_tails, lens, &rep_limit);
    }
    if (pen_active && !penalties_first) {
        nf_pf_penalties(lid, lsize, base_off, pen_off, pen_cnt, logits, tokens,
                fcfg[fb + 0], fcfg[fb + 1], fcfg[fb + 2]);
    }
}

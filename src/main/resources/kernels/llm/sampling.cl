/* sampling.cl - LLM_Softmax and LLM_Argmax. Requires: common/dialect.h, common/config.h, common/reduce.h */
/*
 * Local size L is a power of two <= LLM_DIMX (a multiple of 32 on CUDA). Every work-item reaches
 * every barrier; early returns depend on the group id alone.
 */

/*
 * Numerically stable softmax of one vector: output[i] = exp(x[i] - max) / sum_j exp(x[j] - max)
 * (ggml-cpu/vec.cpp ggml_vec_soft_max_f32, precise expf). input may equal output: every element
 * is read for the maximum before any is written, and each later pass reads and writes only the
 * work-item's own elements. Dispatch: global = local = {L} (exactly one workgroup).
 */
NF_KERNEL void LLM_Softmax(
    const int size,
    NF_GLOBAL const float* input,
    NF_GLOBAL float* output)
{
    if (get_group_id(0) != 0) return;
    const int lid = get_local_id(0);
    const int lsize = get_local_size(0);
    const int size4 = size & ~3;
    NF_LOCAL float partial[LLM_DIMX];

    float my_max = -NF_INF;
    for (int i = lid * 4; i < size4; i += lsize * 4) {
        const float4 v = vload4(0, input + i);
        my_max = fmax(my_max, fmax(fmax(v.x, v.y), fmax(v.z, v.w)));
    }
    for (int i = size4 + lid; i < size; i += lsize) {
        my_max = fmax(my_max, input[i]);
    }
    const float row_max = nf_wg_max(partial, my_max);

    float my_sum = 0.0f;
    for (int i = lid * 4; i < size4; i += lsize * 4) {
        const float4 v = vload4(0, input + i);
        const float4 e = NF_F4(exp(v.x - row_max), exp(v.y - row_max), exp(v.z - row_max), exp(v.w - row_max));
        vstore4(e, 0, output + i);
        my_sum += e.x + e.y + e.z + e.w;
    }
    for (int i = size4 + lid; i < size; i += lsize) {
        const float e = exp(input[i] - row_max);
        output[i] = e;
        my_sum += e;
    }
    const float inv_sum = 1.0f / nf_wg_sum(partial, my_sum);

    for (int i = lid * 4; i < size4; i += lsize * 4) {
        vstore4(vload4(0, output + i) * inv_sum, 0, output + i);
    }
    for (int i = size4 + lid; i < size; i += lsize) {
        output[i] *= inv_sum;
    }
}

/* Index sentinel of an argmax candidate that holds no element. */
#define NF_ARGMAX_NONE 0x7fffffff

/*
 * Total order of argmax candidates (v, i): a non-NaN value beats NaN; among non-NaN values the
 * larger wins; equal values (and two NaNs) go to the lower index. Returns 1 when (v, i) beats
 * (bv, bi). This makes the result the first maximum, as std::max_element and the llama.cpp greedy
 * sampler (llama-sampler.cpp llama_sampler_greedy_apply, strict '>' scan), and never selects NaN
 * unless the row holds nothing else.
 */
NF_INLINE int nf_argmax_better(float v, int i, float bv, int bi) {
    if (isnan(v)) {
        return isnan(bv) && i < bi;
    }
    if (isnan(bv)) {
        return 1;
    }
    return v > bv || (v == bv && i < bi);
}

/*
 * Row-wise argmax over n_rows rows of `size` floats (row stride = size): max_idx[row] is the index
 * of the first maximum of row `row`. NaN is never selected; a row of only NaN yields 0, and so does
 * size == 0. Dispatch: global = {n_rows * L}, local = {L}; one workgroup per row.
 */
NF_KERNEL void LLM_Argmax(
    const int size,
    NF_GLOBAL const float* input,
    NF_GLOBAL int* max_idx,
    const int n_rows)
{
    const int row = get_group_id(0);
    if (row >= n_rows) return;
    const int lid = get_local_id(0);
    const int lsize = get_local_size(0);
    NF_LOCAL float best_v[LLM_DIMX];
    NF_LOCAL int best_i[LLM_DIMX];

    const nf_i64 base = (nf_i64)row * size;
    float bv = NF_NAN;
    int bi = NF_ARGMAX_NONE;
    for (int i = lid; i < size; i += lsize) {
        const float v = input[base + i];
        if (nf_argmax_better(v, i, bv, bi)) {
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
            if (nf_argmax_better(ov, oi, best_v[lid], best_i[lid])) {
                best_v[lid] = ov;
                best_i[lid] = oi;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        const int r = best_i[0];
        max_idx[row] = (r == NF_ARGMAX_NONE) ? 0 : r;
    }
}

/*
 * The single-pass tree verify's acceptance walk on the device (llama.cpp token-match acceptance over
 * one masked batch): follows the accepted path through the proposal's parent array over the verify
 * rows' greedy ids, so the host reads back the accepted depth and path instead of every logits row.
 *
 * Per depth the workgroup scans the nodes of that depth whose parent is in the frontier — the set of
 * nodes the walk matched at the previous depth, the depth-1 roots when it is empty — and keeps the
 * ones whose token equals the pending sample `pred` (the greedy id of the predecessor row). The
 * matched set becomes the next frontier; the walk records its smallest index and continues from that
 * node's row. Per walk step every work-item stages its matches, one reduction folds the smallest
 * matched index and the matched bit set, and the walk advances uniformly: every work-item reads the
 * reduced values, so the next scan runs on the same frontier. The walk stops at the first depth with
 * no matched node.
 *
 * Arguments:
 *   n_nodes  the proposal's node count (at most 63: the verify batch's rows and the frontier bits
 *            name rows of one 64-bit mask)
 *   tree     3 ints per node: token, parent (-1 for a root), depth (1-based); a parent precedes its
 *            child
 *   argmax   the verify rows' greedy ids in row order: row 0 the pending token, row i + 1 node i
 *   out      [0] the accepted depth, [1] the first emission (row 0's greedy id), [2] the final draw
 *            (the greedy id of the last accepted row, row 0's when nothing was accepted), [3 ..]
 *            the accepted path's node indices in depth order
 *
 * A session stop the host replays can only cut the walk short: the emissions it produces are a
 * prefix of [out[1], the accepted nodes' tokens from the second match on, out[2]] — the first
 * match's token is out[1] itself — so the host stops that replay at the stop.
 *
 * Dispatch: global = local = {L} (exactly one workgroup).
 */
NF_KERNEL void LLM_TreeAccept(
    const int n_nodes,
    NF_GLOBAL const int* NF_RESTRICT tree,
    NF_GLOBAL const int* NF_RESTRICT argmax,
    NF_GLOBAL int* out)
{
    if (get_group_id(0) != 0) return;
    const int lid = get_local_id(0);
    const int lsize = get_local_size(0);
    NF_LOCAL int matched[LLM_DIMX];
    NF_LOCAL uint lo_set[LLM_DIMX];
    NF_LOCAL uint hi_set[LLM_DIMX];
    NF_LOCAL int stop;

    /* the frontier as a bit set over the node indices, empty before the first depth */
    uint lo = 0u;
    uint hi = 0u;
    int pred_row = 0;
    int accepted = 0;
    const int first = argmax[0];
    int pred = first;
    int depth = 1;

    for (;;) {
        int my_match = NF_ARGMAX_NONE;
        uint my_lo = 0u;
        uint my_hi = 0u;
        for (int i = lid; i < n_nodes; i += lsize) {
            if (tree[3 * i + 2] != depth) {
                continue;
            }
            const int parent = tree[3 * i + 1];
            const int in_frontier = (lo | hi) == 0u
                ? (int)(parent == -1)
                : (parent < 32 ? (int)((lo >> parent) & 1u) : (int)((hi >> (parent - 32)) & 1u));
            if (in_frontier == 0 || tree[3 * i] != pred) {
                continue;
            }
            if (my_match == NF_ARGMAX_NONE) {
                my_match = i; /* the scan ascends: the first hit is the smallest */
            }
            if (i < 32) {
                my_lo |= 1u << i;
            } else {
                my_hi |= 1u << (i - 32);
            }
        }
        matched[lid] = my_match;
        lo_set[lid] = my_lo;
        hi_set[lid] = my_hi;
        barrier(CLK_LOCAL_MEM_FENCE);

        /* ceiling halving: every lane's content reaches lane 0 for any work-group size (plain halving
         * drops the tail lane on a non-power-of-two lsize) */
        for (int live = lsize; live > 1; live = (live + 1) >> 1) {
            const int stride = (live + 1) >> 1;
            if (lid < stride && lid + stride < live) {
                if (matched[lid + stride] < matched[lid]) {
                    matched[lid] = matched[lid + stride];
                }
                lo_set[lid] |= lo_set[lid + stride];
                hi_set[lid] |= hi_set[lid + stride];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (lid == 0) {
            stop = matched[0] == NF_ARGMAX_NONE ? 1 : 0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (stop == 1) {
            break;
        }
        if (lid == 0) {
            out[3 + accepted] = matched[0];
        }
        pred_row = matched[0] + 1;
        pred = argmax[pred_row];
        lo = lo_set[0];
        hi = hi_set[0];
        accepted++;
        depth++;
        /* the reads of the reduced values above must finish before the next scan writes the arrays */
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        out[0] = accepted;
        out[1] = first;
        out[2] = argmax[pred_row];
    }
}

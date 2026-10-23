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

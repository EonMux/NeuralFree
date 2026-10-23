/* norm.cl - RMS and layer norm kernels. Requires: common/dialect.h, common/config.h, common/reduce.h */
/*
 * Row reductions with one workgroup per normalized row. Local size L is a power of two <= LLM_DIMX
 * (a multiple of 32 on CUDA); every work-item of a workgroup reaches every barrier, and the only
 * early returns depend on the group id alone.
 *
 * Aliasing: callers normalize in place (input == output), so input and output are never restrict.
 * In-place use is exact because every pass that writes output[i] runs after the reduction barriers
 * of the passes that read the whole row, and output[i] is written by the work-item that reads
 * input[i] in that pass. Weights and biases are model parameters and stay restrict.
 *
 * Math (ggml-cpu/ops.cpp):
 *   RMS    y = x * rsqrt(mean(x^2) + eps) * w                       (ggml_compute_forward_rms_norm_f32)
 *   layer  y = (x - mean) * rsqrt(mean((x - mean)^2) + eps) * w + b (ggml_compute_forward_norm_f32)
 *   L2 per head (the GDN form) y = x * rsqrt(sum(x^2) + eps) * w
 */

/* Sum of x[base + i]^2 over i in [0, n), split over the workgroup in float4 steps plus a scalar tail. */
NF_INLINE float nf_norm_row_sumsq(NF_GLOBAL const float* x, nf_i64 base, int n, int lid, int lsize) {
    const int n4 = n & ~3;
    float ss = 0.0f;
    for (int i = lid * 4; i < n4; i += lsize * 4) {
        const float4 v = vload4(0, x + base + i);
        ss += dot(v, v);
    }
    for (int i = n4 + lid; i < n; i += lsize) {
        const float v = x[base + i];
        ss += v * v;
    }
    return ss;
}

/* Sum of x[base + i] over i in [0, n). */
NF_INLINE float nf_norm_row_sum(NF_GLOBAL const float* x, nf_i64 base, int n, int lid, int lsize) {
    const int n4 = n & ~3;
    float s = 0.0f;
    for (int i = lid * 4; i < n4; i += lsize * 4) {
        const float4 v = vload4(0, x + base + i);
        s += v.x + v.y + v.z + v.w;
    }
    for (int i = n4 + lid; i < n; i += lsize) {
        s += x[base + i];
    }
    return s;
}

/* Sum of (x[base + i] - mean)^2 over i in [0, n). */
NF_INLINE float nf_norm_row_sqdev(NF_GLOBAL const float* x, nf_i64 base, int n, float mean, int lid, int lsize) {
    const int n4 = n & ~3;
    float s = 0.0f;
    for (int i = lid * 4; i < n4; i += lsize * 4) {
        const float4 d = vload4(0, x + base + i) - NF_F4S(mean);
        s += dot(d, d);
    }
    for (int i = n4 + lid; i < n; i += lsize) {
        const float d = x[base + i] - mean;
        s += d * d;
    }
    return s;
}

/* out[base + i] = x[base + i] * scale * w[i]; same work-item partition as nf_norm_row_sumsq. */
NF_INLINE void nf_norm_row_apply_rms(NF_GLOBAL const float* x, NF_GLOBAL const float* NF_RESTRICT w,
                                     NF_GLOBAL float* out, nf_i64 base, int n, float scale, int lid, int lsize) {
    const int n4 = n & ~3;
    for (int i = lid * 4; i < n4; i += lsize * 4) {
        const float4 v = vload4(0, x + base + i);
        const float4 w4 = vload4(0, w + i);
        vstore4(v * scale * w4, 0, out + base + i);
    }
    for (int i = n4 + lid; i < n; i += lsize) {
        out[base + i] = x[base + i] * scale * w[i];
    }
}

/* out[base + i] = (x[base + i] - mean) * rstd * w[i] + b[i]. */
NF_INLINE void nf_norm_row_apply_ln(NF_GLOBAL const float* x, NF_GLOBAL const float* NF_RESTRICT w,
                                    NF_GLOBAL const float* NF_RESTRICT b, NF_GLOBAL float* out, nf_i64 base,
                                    int n, float mean, float rstd, int lid, int lsize) {
    const int n4 = n & ~3;
    for (int i = lid * 4; i < n4; i += lsize * 4) {
        const float4 v = vload4(0, x + base + i);
        const float4 w4 = vload4(0, w + i);
        const float4 b4 = vload4(0, b + i);
        vstore4((v - NF_F4S(mean)) * rstd * w4 + b4, 0, out + base + i);
    }
    for (int i = n4 + lid; i < n; i += lsize) {
        out[base + i] = (x[base + i] - mean) * rstd * w[i] + b[i];
    }
}

/* Sum of x[base + i]^2 over one head, i in [0, head_dim). */
NF_INLINE float nf_norm_head_sumsq(NF_GLOBAL const float* x, nf_i64 base, int head_dim, int lid, int lsize) {
    float ss = 0.0f;
    for (int i = lid; i < head_dim; i += lsize) {
        const float v = x[base + i];
        ss += v * v;
    }
    return ss;
}

/* out[base + i] = x[base + i] * scale * w[w_base + i] over one head. */
NF_INLINE void nf_norm_head_apply(NF_GLOBAL const float* x, NF_GLOBAL const float* NF_RESTRICT w, nf_i64 w_base,
                                  NF_GLOBAL float* out, nf_i64 base, int head_dim, float scale, int lid, int lsize) {
    for (int i = lid; i < head_dim; i += lsize) {
        out[base + i] = x[base + i] * scale * w[w_base + i];
    }
}

/*
 * RMS norm of one vector of `size` floats. Dispatch: global = local = {L} (exactly one workgroup;
 * any further group exits so in-place use stays race-free).
 */
NF_KERNEL void LLM_RMSNorm(
    const int size,
    const float eps,
    NF_GLOBAL const float* input,
    NF_GLOBAL const float* NF_RESTRICT weight,
    NF_GLOBAL float* output)
{
    if (get_group_id(0) != 0) return;
    const int lid = get_local_id(0);
    const int lsize = get_local_size(0);
    NF_LOCAL float partial[LLM_DIMX];

    const float ss = nf_wg_sum(partial, nf_norm_row_sumsq(input, 0, size, lid, lsize));
    const float rstd = rsqrt(ss / (float)size + eps);
    nf_norm_row_apply_rms(input, weight, output, 0, size, rstd, lid, lsize);
}

/*
 * Per-head RMS norm with one weight vector shared by all heads (QK norm):
 * x[h * head_dim + i] *= rsqrt(mean_h(x^2) + eps) * weight[i]. total_size is informational.
 * Dispatch: global = {num_heads * L}, local = {L}; one workgroup per head.
 */
NF_KERNEL void LLM_RMSNormHeads(
    const int total_size,
    const int num_heads,
    const int head_dim,
    const float eps,
    NF_GLOBAL const float* input,
    NF_GLOBAL const float* NF_RESTRICT weight,
    NF_GLOBAL float* output)
{
    (void)total_size;
    const int head = get_group_id(0);
    if (head >= num_heads) return;
    const int lid = get_local_id(0);
    const int lsize = get_local_size(0);
    NF_LOCAL float partial[LLM_DIMX];

    const nf_i64 base = (nf_i64)head * head_dim;
    const float ss = nf_wg_sum(partial, nf_norm_head_sumsq(input, base, head_dim, lid, lsize));
    const float rstd = rsqrt(ss / (float)head_dim + eps);
    nf_norm_head_apply(input, weight, 0, output, base, head_dim, rstd, lid, lsize);
}

/*
 * Per-head RMS norm with a per-head weight slice, weight[h * head_dim + i] (Mamba2 grouped norm,
 * per-head q/k norms). Arguments and dispatch as LLM_RMSNormHeads.
 */
NF_KERNEL void LLM_RMSNormGrouped(
    const int total_size,
    const int num_heads,
    const int head_dim,
    const float eps,
    NF_GLOBAL const float* input,
    NF_GLOBAL const float* NF_RESTRICT weight,
    NF_GLOBAL float* output)
{
    (void)total_size;
    const int head = get_group_id(0);
    if (head >= num_heads) return;
    const int lid = get_local_id(0);
    const int lsize = get_local_size(0);
    NF_LOCAL float partial[LLM_DIMX];

    const nf_i64 base = (nf_i64)head * head_dim;
    const float ss = nf_wg_sum(partial, nf_norm_head_sumsq(input, base, head_dim, lid, lsize));
    const float rstd = rsqrt(ss / (float)head_dim + eps);
    nf_norm_head_apply(input, weight, base, output, base, head_dim, rstd, lid, lsize);
}

/*
 * Per-head L2 norm, x * rsqrt(sum_h(x^2) + eps) * weight[i] (no division by head_dim).
 * Arguments and dispatch as LLM_RMSNormHeads.
 */
NF_KERNEL void LLM_L2NormHeads(
    const int total_size,
    const int num_heads,
    const int head_dim,
    const float eps,
    NF_GLOBAL const float* input,
    NF_GLOBAL const float* NF_RESTRICT weight,
    NF_GLOBAL float* output)
{
    (void)total_size;
    const int head = get_group_id(0);
    if (head >= num_heads) return;
    const int lid = get_local_id(0);
    const int lsize = get_local_size(0);
    NF_LOCAL float partial[LLM_DIMX];

    const nf_i64 base = (nf_i64)head * head_dim;
    const float ss = nf_wg_sum(partial, nf_norm_head_sumsq(input, base, head_dim, lid, lsize));
    const float rstd = rsqrt(ss + eps);
    nf_norm_head_apply(input, weight, 0, output, base, head_dim, rstd, lid, lsize);
}

/*
 * RMS norm of every row of a [seq_len x size] matrix. Dispatch: global = {seq_len * L},
 * local = {L}; one workgroup per row.
 */
NF_KERNEL void LLM_Batch_RMSNorm(
    const int size,
    const int seq_len,
    const float eps,
    NF_GLOBAL const float* input,
    NF_GLOBAL const float* NF_RESTRICT weight,
    NF_GLOBAL float* output)
{
    const int s = get_group_id(0);
    if (s >= seq_len) return;
    const int lid = get_local_id(0);
    const int lsize = get_local_size(0);
    NF_LOCAL float partial[LLM_DIMX];

    const nf_i64 base = (nf_i64)s * size;
    const float ss = nf_wg_sum(partial, nf_norm_row_sumsq(input, base, size, lid, lsize));
    const float rstd = rsqrt(ss / (float)size + eps);
    nf_norm_row_apply_rms(input, weight, output, base, size, rstd, lid, lsize);
}

/*
 * LLM_RMSNormHeads over seq_len tokens laid out [seq_len x num_heads x head_dim].
 * Dispatch: global = {seq_len * num_heads * L}, local = {L}; workgroup g normalizes token
 * g / num_heads, head g % num_heads.
 */
NF_KERNEL void LLM_Batch_RMSNormHeads(
    const int total_size,
    const int num_heads,
    const int head_dim,
    const int seq_len,
    const float eps,
    NF_GLOBAL const float* input,
    NF_GLOBAL const float* NF_RESTRICT weight,
    NF_GLOBAL float* output)
{
    (void)total_size;
    const int wg = get_group_id(0);
    const int s = wg / num_heads;
    const int head = wg - s * num_heads;
    if (s >= seq_len) return;
    const int lid = get_local_id(0);
    const int lsize = get_local_size(0);
    NF_LOCAL float partial[LLM_DIMX];

    const nf_i64 base = ((nf_i64)s * num_heads + head) * head_dim;
    const float ss = nf_wg_sum(partial, nf_norm_head_sumsq(input, base, head_dim, lid, lsize));
    const float rstd = rsqrt(ss / (float)head_dim + eps);
    nf_norm_head_apply(input, weight, 0, output, base, head_dim, rstd, lid, lsize);
}

/* LLM_L2NormHeads over seq_len tokens; arguments and dispatch as LLM_Batch_RMSNormHeads. */
NF_KERNEL void LLM_Batch_L2NormHeads(
    const int total_size,
    const int num_heads,
    const int head_dim,
    const int seq_len,
    const float eps,
    NF_GLOBAL const float* input,
    NF_GLOBAL const float* NF_RESTRICT weight,
    NF_GLOBAL float* output)
{
    (void)total_size;
    const int wg = get_group_id(0);
    const int s = wg / num_heads;
    const int head = wg - s * num_heads;
    if (s >= seq_len) return;
    const int lid = get_local_id(0);
    const int lsize = get_local_size(0);
    NF_LOCAL float partial[LLM_DIMX];

    const nf_i64 base = ((nf_i64)s * num_heads + head) * head_dim;
    const float ss = nf_wg_sum(partial, nf_norm_head_sumsq(input, base, head_dim, lid, lsize));
    const float rstd = rsqrt(ss + eps);
    nf_norm_head_apply(input, weight, 0, output, base, head_dim, rstd, lid, lsize);
}

/*
 * Layer norm with weight and bias of one vector of `size` floats (two-pass mean / variance).
 * Dispatch: global = local = {L} (exactly one workgroup).
 */
NF_KERNEL void LLM_LayerNorm(
    const int size,
    const float eps,
    NF_GLOBAL const float* input,
    NF_GLOBAL const float* NF_RESTRICT weight,
    NF_GLOBAL const float* NF_RESTRICT bias,
    NF_GLOBAL float* output)
{
    if (get_group_id(0) != 0) return;
    const int lid = get_local_id(0);
    const int lsize = get_local_size(0);
    NF_LOCAL float partial[LLM_DIMX];

    const float mean = nf_wg_sum(partial, nf_norm_row_sum(input, 0, size, lid, lsize)) / (float)size;
    const float var = nf_wg_sum(partial, nf_norm_row_sqdev(input, 0, size, mean, lid, lsize)) / (float)size;
    const float rstd = rsqrt(var + eps);
    nf_norm_row_apply_ln(input, weight, bias, output, 0, size, mean, rstd, lid, lsize);
}

/*
 * Layer norm of every row of a [seq_len x size] matrix. Dispatch: global = {seq_len * L},
 * local = {L}; one workgroup per row.
 */
NF_KERNEL void LLM_Batch_LayerNorm(
    const int size,
    const int seq_len,
    const float eps,
    NF_GLOBAL const float* input,
    NF_GLOBAL const float* NF_RESTRICT weight,
    NF_GLOBAL const float* NF_RESTRICT bias,
    NF_GLOBAL float* output)
{
    const int s = get_group_id(0);
    if (s >= seq_len) return;
    const int lid = get_local_id(0);
    const int lsize = get_local_size(0);
    NF_LOCAL float partial[LLM_DIMX];

    const nf_i64 base = (nf_i64)s * size;
    const float mean = nf_wg_sum(partial, nf_norm_row_sum(input, base, size, lid, lsize)) / (float)size;
    const float var = nf_wg_sum(partial, nf_norm_row_sqdev(input, base, size, mean, lid, lsize)) / (float)size;
    const float rstd = rsqrt(var + eps);
    nf_norm_row_apply_ln(input, weight, bias, output, base, size, mean, rstd, lid, lsize);
}

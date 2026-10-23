/* gemma3n.cl - LLM_AltUp_*, LLM_MagnitudeMatch, LLM_GaussianTopK and LLM_StreamsMean Gemma3n kernels. Requires: common/dialect.h, common/config.h, common/reduce.h */
/*
 * Gemma3n AltUp stream mixing, magnitude matching, Gaussian top-k activation sparsity and the final
 * stream mean (llama.cpp src/models/gemma3n.cpp). Java: gguf.ops.gemma3n.Gemma3nKernels; CPU twins:
 * gguf.ops.gemma3n.CpuAltUp and gguf.ops.gemma3n.CpuActivationSparsity.
 *
 * Stream layout: the AltUp state of one ubatch is n_altup streams of n_tokens token-major rows of
 * n_embd floats; element (stream k, token t, column e) is at
 *     base + k*stream_stride + t*n_embd + e,
 * which is llama.cpp's [n_embd, n_tokens, n_altup] tensor with the stream dimension strided by
 * stream_stride >= n_tokens*n_embd, so one fixed-capacity buffer serves every ubatch. Offsets and
 * strides are in floats; addresses are formed in nf_i64. The host keeps n_tokens*n_embd below 2^31
 * and n_altup in [1, NF_G3N_MAX_ALTUP].
 *
 * Dispatch:
 *   element-wise kernels (LLM_AltUp_Predict, LLM_AltUp_Correct, LLM_StreamsMean): one work-item per
 *   (token, column), 1-D global = roundUp(n_tokens*n_embd, L), no barriers;
 *   row kernels (LLM_MagnitudeMatch, LLM_GaussianTopK): one workgroup per row, 1-D global = rows*L,
 *   local = L, L a power of two in [32, LLM_DIMX] (a multiple of 32 for CUDA's full-warp reductions).
 */

/* Largest stream count; the per-work-item stream registers are sized by it. */
#define NF_G3N_MAX_ALTUP 8

/*
 * AltUp predict (gemma3n.cpp:416-436):
 *   pred[j][t][e] = sum_{k<n_altup} streams[k][t][e] * coefs[t][j*n_altup + k] + streams[j][t][e]
 * coefs is the matvec output altup_predict_coef . modalities, n_altup*n_altup floats per token at
 * coefs_off + t*n_altup*n_altup (its reshape to [n_altup, n_altup, n_tokens] puts column k of output
 * stream j at j*n_altup + k). The sum runs over ascending k. Each work-item reads every stream of its
 * own (t, e) before writing, so pred may be streams in place (pred_off == streams_off).
 */
NF_KERNEL void LLM_AltUp_Predict(
    const int n_tokens,
    const int n_embd,
    const int n_altup,
    const int stream_stride,
    const int coefs_off,
    const int streams_off,
    const int pred_off,
    NF_GLOBAL const float* NF_RESTRICT coefs,
    NF_GLOBAL const float* streams,
    NF_GLOBAL float* pred)
{
    const uint idx = (uint)get_global_id(0);
    if (n_altup < 1 || n_altup > NF_G3N_MAX_ALTUP || idx >= (uint)n_tokens * (uint)n_embd) return;
    const uint t = idx / (uint)n_embd;
    /* Constant trip counts with n_altup guards keep s[] in registers (full unroll). */
    float s[NF_G3N_MAX_ALTUP];
    for (int k = 0; k < NF_G3N_MAX_ALTUP; k++) {
        s[k] = k < n_altup ? streams[(nf_i64)streams_off + (nf_i64)k * stream_stride + idx] : 0.0f;
    }
    NF_GLOBAL const float* c = coefs + (nf_i64)coefs_off + (nf_i64)t * (n_altup * n_altup);
    for (int j = 0; j < NF_G3N_MAX_ALTUP; j++) {
        if (j < n_altup) {
            float acc = 0.0f;
            for (int k = 0; k < NF_G3N_MAX_ALTUP; k++) {
                if (k < n_altup) {
                    acc += s[k] * c[j * n_altup + k];
                }
            }
            pred[(nf_i64)pred_off + (nf_i64)j * stream_stride + idx] = acc + s[j];
        }
    }
}

/*
 * AltUp correct (gemma3n.cpp:441-461):
 *   innov = activated[t][e] - pred[act][t][e]
 *   streams_out[j][t][e] = innov * (coefs[t][j] + 1) + pred[j][t][e]
 * coefs is the raw matvec output altup_correct_coef . modalities, n_altup floats per token at
 * coefs_off + t*n_altup; the +1 of ggml_scale_bias(coefs, 1, 1) is applied here. activated is a
 * token-major [n_tokens][n_embd] view at activated_off. Each work-item reads pred[act] before any write
 * and reads pred[j] before writing streams_out[j] of its own (t, e), so streams_out may be pred in place
 * (out_off == pred_off).
 */
NF_KERNEL void LLM_AltUp_Correct(
    const int n_tokens,
    const int n_embd,
    const int n_altup,
    const int act,
    const int stream_stride,
    const int coefs_off,
    const int activated_off,
    const int pred_off,
    const int out_off,
    NF_GLOBAL const float* NF_RESTRICT coefs,
    NF_GLOBAL const float* NF_RESTRICT activated,
    NF_GLOBAL const float* pred,
    NF_GLOBAL float* streams_out)
{
    const uint idx = (uint)get_global_id(0);
    if (n_altup < 1 || act < 0 || act >= n_altup || idx >= (uint)n_tokens * (uint)n_embd) return;
    const uint t = idx / (uint)n_embd;
    const float innov = activated[(nf_i64)activated_off + idx]
                      - pred[(nf_i64)pred_off + (nf_i64)act * stream_stride + idx];
    NF_GLOBAL const float* c = coefs + (nf_i64)coefs_off + (nf_i64)t * n_altup;
    for (int j = 0; j < n_altup; j++) {
        const float cj = c[j] + 1.0f;
        const float p = pred[(nf_i64)pred_off + (nf_i64)j * stream_stride + idx];
        streams_out[(nf_i64)out_off + (nf_i64)j * stream_stride + idx] = innov * cj + p;
    }
}

/*
 * Magnitude matching (gemma3n.cpp:117-123 and 269-278, calc_magnitude at :314-316): for every row
 * (x stream j < n_x, token t),
 *   x[j][t][e] = (x[j][t][e] * |ref[t]|) / |x[j][t]|,   |v| = sqrt(sum_e v[e]^2)
 * ref is a token-major [n_tokens][n_embd] view at ref_off; x row (j, t) starts at
 * x_off + j*x_stride + t*n_embd. Rows are indexed by group id = j*n_tokens + t. The x rows must not
 * overlap ref. A zero |x| row yields inf/NaN exactly as ggml_div does.
 */
NF_KERNEL void LLM_MagnitudeMatch(
    const int n_tokens,
    const int n_embd,
    const int n_x,
    const int x_stride,
    const int ref_off,
    const int x_off,
    NF_GLOBAL const float* NF_RESTRICT ref,
    NF_GLOBAL float* x)
{
    NF_LOCAL float nf_g3n_red[LLM_DIMX];
    const int row = (int)get_group_id(0);
    if (row >= n_tokens * n_x) return;               /* uniform across the workgroup */
    const int j = row / n_tokens;
    const int t = row - j * n_tokens;
    const int lid = (int)get_local_id(0);
    const int lsz = (int)get_local_size(0);
    NF_GLOBAL const float* r = ref + (nf_i64)ref_off + (nf_i64)t * n_embd;
    NF_GLOBAL float* y = x + (nf_i64)x_off + (nf_i64)j * x_stride + (nf_i64)t * n_embd;

    float sr = 0.0f;
    float sx = 0.0f;
    for (int i = lid; i < n_embd; i += lsz) {
        const float a = r[i];
        const float b = y[i];
        sr += a * a;
        sx += b * b;
    }
    sr = nf_wg_sum(nf_g3n_red, sr);
    sx = nf_wg_sum(nf_g3n_red, sx);
    const float target = sqrt(sr);
    const float current = sqrt(sx);
    for (int i = lid; i < n_embd; i += lsz) {
        y[i] = (y[i] * target) / current;
    }
}

/*
 * Gaussian top-k activation sparsity (gemma3n.cpp:389-395), in place on each row t < n_tokens of n
 * floats at x_off + t*x_stride:
 *   mean = sum(x) / n
 *   std  = sqrt(sum((x - mean)^2) * (1 / (n - 1)))      (unbiased, n - 1 denominator)
 *   x    = relu(x - (mean + std * k_std))                relu(v) = v > 0 ? v : 0 (ggml_vec_relu_f32)
 * One workgroup per row (group id = t); two reduction passes, then the in-place update.
 */
NF_KERNEL void LLM_GaussianTopK(
    const int n_tokens,
    const int n,
    const int x_stride,
    const int x_off,
    const float k_std,
    NF_GLOBAL float* x)
{
    NF_LOCAL float nf_g3n_red[LLM_DIMX];
    const int t = (int)get_group_id(0);
    if (t >= n_tokens) return;                       /* uniform across the workgroup */
    const int lid = (int)get_local_id(0);
    const int lsz = (int)get_local_size(0);
    NF_GLOBAL float* y = x + (nf_i64)x_off + (nf_i64)t * x_stride;

    float s = 0.0f;
    for (int i = lid; i < n; i += lsz) {
        s += y[i];
    }
    s = nf_wg_sum(nf_g3n_red, s);
    const float mean = s / (float)n;

    float q = 0.0f;
    for (int i = lid; i < n; i += lsz) {
        const float d = y[i] - mean;
        q += d * d;
    }
    q = nf_wg_sum(nf_g3n_red, q);
    const float std_dev = sqrt(q * (1.0f / (float)(n - 1)));
    const float cutoff = mean + std_dev * k_std;

    for (int i = lid; i < n; i += lsz) {
        const float v = y[i] - cutoff;
        y[i] = v > 0.0f ? v : 0.0f;
    }
}

/*
 * Stream mean (gemma3n.cpp:281-286):
 *   out[t][e] = (((streams[0][t][e] + streams[1][t][e]) + ...) + streams[n_altup-1][t][e]) * (1 / n_altup)
 * summed in ascending stream order, then scaled as ggml_scale does. out is a token-major
 * [n_tokens][n_embd] view at out_off; it may be stream 0 in place (out_off == streams_off).
 */
NF_KERNEL void LLM_StreamsMean(
    const int n_tokens,
    const int n_embd,
    const int n_altup,
    const int stream_stride,
    const int streams_off,
    const int out_off,
    NF_GLOBAL const float* streams,
    NF_GLOBAL float* out)
{
    const uint idx = (uint)get_global_id(0);
    if (n_altup < 1 || idx >= (uint)n_tokens * (uint)n_embd) return;
    float acc = streams[(nf_i64)streams_off + idx];
    for (int k = 1; k < n_altup; k++) {
        acc += streams[(nf_i64)streams_off + (nf_i64)k * stream_stride + idx];
    }
    out[(nf_i64)out_off + idx] = acc * (1.0f / (float)n_altup);
}

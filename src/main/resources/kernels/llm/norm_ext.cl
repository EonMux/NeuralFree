/* norm_ext.cl - LLM_RowNorm, LLM_GroupRms, LLM_GroupNorm, LLM_ResidualMix and row-scale kernels. Requires: common/dialect.h, common/config.h, common/reduce.h */
/*
 * Generic normalizations, residual mixes and row scales of the layer executor.
 * The CPU reference of every kernel is gguf.ops.norm.NormMath, with identical index formulas.
 *
 * Row reductions (LLM_RowNorm, LLM_GroupRms, LLM_GroupNorm) run one workgroup per normalized vector:
 * 1-D, local = {L} with L a power of two <= LLM_DIMX (a multiple of 32 on CUDA). Every work-item reaches
 * every barrier; the only early returns depend on the group id alone. Element i of a vector is read and
 * written by work-item i % L in every pass, and every write follows the reductions over the whole vector,
 * so in-place use (input and output views coincide) is exact.
 *
 * Element-wise kernels (LLM_ResidualMix, LLM_MulRowScale) run one work-item per element over the
 * flattened index get_global_id(0) + get_global_id(1) * get_global_size(0): 1-D global = roundUp(n, L).
 *
 * All offsets and strides are in floats; addresses are computed in nf_i64. Element counts stay below 2^31
 * (the host validates).
 */

/* LLM_RowNorm modes (gguf.arch.NormKind codes). */
#define NF_NORM_EXT_RMS       0
#define NF_NORM_EXT_LAYER     1
#define NF_NORM_EXT_L2_CLAMP  2
#define NF_NORM_EXT_L2_GDN    3

/* LLM_RowNorm flags. */
#define NF_NORM_EXT_HAS_W 1
#define NF_NORM_EXT_HAS_B 2

/* LLM_ResidualMix modes (gguf.ops.norm.NormKernels.MixMode codes). */
#define NF_MIX_ADD                 0
#define NF_MIX_SCALED              1
#define NF_MIX_ALPHA_BETA          2
#define NF_MIX_NORMED_INPUT_SCALED 3

/* Sum of x[base + i] over i in [0, n), strided over the workgroup. */
NF_INLINE float nf_norm_ext_sum(NF_GLOBAL const float* x, nf_i64 base, int n, int lid, int lsize) {
    float s = 0.0f;
    for (int i = lid; i < n; i += lsize) {
        s += x[base + i];
    }
    return s;
}

/* Sum of x[base + i]^2 over i in [0, n). */
NF_INLINE float nf_norm_ext_sumsq(NF_GLOBAL const float* x, nf_i64 base, int n, int lid, int lsize) {
    float s = 0.0f;
    for (int i = lid; i < n; i += lsize) {
        const float v = x[base + i];
        s += v * v;
    }
    return s;
}

/* Sum of (x[base + i] - mean)^2 over i in [0, n). */
NF_INLINE float nf_norm_ext_sqdev(NF_GLOBAL const float* x, nf_i64 base, int n, float mean, int lid, int lsize) {
    float s = 0.0f;
    for (int i = lid; i < n; i += lsize) {
        const float d = x[base + i] - mean;
        s += d * d;
    }
    return s;
}

/* Flattened work-item index of a 1-D or 2-D range. */
NF_INLINE uint nf_norm_ext_linear_id(void) {
    return (uint)get_global_id(0) + (uint)get_global_id(1) * (uint)get_global_size(0);
}

/*
 * Normalizes row r < n_rows of row_len floats: x row at x_off + r*row_stride, y row at y_off + r*row_stride.
 *   mode 0 RMS       y = x * rsqrt(sum(x^2)/n + eps)                       (ops.cpp:4191 rms_norm)
 *   mode 1 LAYER     y = (x - mean) * rsqrt(sum((x - mean)^2)/n + eps)     (ops.cpp:4094 norm)
 *   mode 2 L2_CLAMP  y = x / max(sqrt(sum(x^2)), eps)                      (ops.cpp:4570 l2_norm)
 *   mode 3 L2_GDN    y = x * rsqrt(sum(x^2)/n + eps/n) * rsqrt(n)          (models.h:14 build_gdn_l2_norm)
 * then y *= w[wr*row_len + c] (flags & 1) and y += b[wr*row_len + c] (flags & 2), with weight row
 * wr = r % w_rows (llama-graph.cpp:1757 build_norm). w_rows = 1 shares one weight vector; w_rows = n_head
 * gives per-head QK norms (w_rows <= 1 reads row 0). Absent w / b bind read-only placeholders, never read.
 * Dispatch: global = {n_rows * L}, local = {L}; workgroup r normalizes row r.
 */
NF_KERNEL void LLM_RowNorm(
    const int n_rows,
    const int row_len,
    const int row_stride,
    const int mode,
    const float eps,
    const int flags,
    const int w_rows,
    const int x_off,
    const int y_off,
    NF_GLOBAL const float* x,
    NF_GLOBAL const float* NF_RESTRICT w,
    NF_GLOBAL const float* NF_RESTRICT b,
    NF_GLOBAL float* y)
{
    const int row = get_group_id(0);
    if (row >= n_rows) return;
    const int lid = get_local_id(0);
    const int lsize = get_local_size(0);
    NF_LOCAL float partial[LLM_DIMX];

    const nf_i64 xb = (nf_i64)x_off + (nf_i64)row * row_stride;
    const nf_i64 yb = (nf_i64)y_off + (nf_i64)row * row_stride;
    const float n = (float)row_len;

    float mean = 0.0f;
    float scale;
    float post = 1.0f;
    if (mode == NF_NORM_EXT_LAYER) {
        mean = nf_wg_sum(partial, nf_norm_ext_sum(x, xb, row_len, lid, lsize)) / n;
        const float var = nf_wg_sum(partial, nf_norm_ext_sqdev(x, xb, row_len, mean, lid, lsize)) / n;
        scale = rsqrt(var + eps);
    } else {
        const float ss = nf_wg_sum(partial, nf_norm_ext_sumsq(x, xb, row_len, lid, lsize));
        if (mode == NF_NORM_EXT_L2_CLAMP) {
            scale = 1.0f / fmax(sqrt(ss), eps);
        } else if (mode == NF_NORM_EXT_L2_GDN) {
            scale = rsqrt(ss / n + eps / n);
            post = rsqrt(n);
        } else {
            scale = rsqrt(ss / n + eps);
        }
    }

    const int has_w = (flags & NF_NORM_EXT_HAS_W) != 0;
    const int has_b = (flags & NF_NORM_EXT_HAS_B) != 0;
    const nf_i64 wb = (nf_i64)(w_rows > 1 ? row % w_rows : 0) * row_len;
    for (int i = lid; i < row_len; i += lsize) {
        float v = (x[xb + i] - mean) * scale;
        if (mode == NF_NORM_EXT_L2_GDN) {
            v *= post;
        }
        if (has_w) {
            v *= w[wb + i];
        }
        if (has_b) {
            v += b[wb + i];
        }
        y[yb + i] = v;
    }
}

/*
 * Grouped RMS norm (mamba2 / nemotron-h / granite-hybrid / falcon-h1 ssm_norm, models/mamba-base.cpp:286-290):
 * token t < n_tokens, group g < n_group, gs = d_inner / n_group, i < gs:
 *   y[t*d_inner + g*gs + i] = x[x_off + t*x_stride + g*gs + i] * rsqrt(sum_g(x^2)/gs + eps) * w[g*gs + i].
 * y is dense [n_tokens][d_inner]; it may coincide with x when x_off == 0 and x_stride == d_inner.
 * Dispatch: global = {n_tokens * n_group * L}, local = {L}; workgroup t*n_group + g.
 */
NF_KERNEL void LLM_GroupRms(
    const int n_tokens,
    const int d_inner,
    const int n_group,
    const float eps,
    const int x_stride,
    const int x_off,
    NF_GLOBAL const float* x,
    NF_GLOBAL const float* NF_RESTRICT w,
    NF_GLOBAL float* y)
{
    const int wg = get_group_id(0);
    const int t = wg / n_group;
    const int g = wg - t * n_group;
    if (t >= n_tokens) return;
    const int lid = get_local_id(0);
    const int lsize = get_local_size(0);
    NF_LOCAL float partial[LLM_DIMX];

    const int gs = d_inner / n_group;
    const nf_i64 xb = (nf_i64)x_off + (nf_i64)t * x_stride + (nf_i64)g * gs;
    const nf_i64 yb = (nf_i64)t * d_inner + (nf_i64)g * gs;
    const nf_i64 wb = (nf_i64)g * gs;

    const float ss = nf_wg_sum(partial, nf_norm_ext_sumsq(x, xb, gs, lid, lsize));
    const float scale = rsqrt(ss / (float)gs + eps);
    for (int i = lid; i < gs; i += lsize) {
        y[yb + i] = x[xb + i] * scale * w[wb + i];
    }
}

/*
 * Group norm over channel groups of a channel-major [n_ch][len] tensor (ops.cpp:4475 group_norm through
 * llama-graph.cpp build_norm LLM_NORM_GROUP; wavtokenizer posnet). cpg = ceil(n_ch / n_groups); group g covers
 * channels [g*cpg, min(g*cpg + cpg, n_ch)), i.e. the contiguous elements [c0*len, c1*len):
 *   y[c*len + t] = (x[c*len + t] - mean_g) * rsqrt(var_g + eps) * w[c] + b[c]
 * with mean and (biased) variance over the group by two passes. A group without channels writes nothing.
 * Dispatch: global = {n_groups * L}, local = {L}; workgroup g.
 */
NF_KERNEL void LLM_GroupNorm(
    const int n_ch,
    const int len,
    const int n_groups,
    const float eps,
    NF_GLOBAL const float* NF_RESTRICT x,
    NF_GLOBAL const float* NF_RESTRICT w,
    NF_GLOBAL const float* NF_RESTRICT b,
    NF_GLOBAL float* NF_RESTRICT y)
{
    const int g = get_group_id(0);
    if (g >= n_groups) return;
    const int cpg = (n_ch + n_groups - 1) / n_groups;
    const int c0 = g * cpg;
    const int c1 = min(c0 + cpg, n_ch);
    if (c1 <= c0) return;
    const int lid = get_local_id(0);
    const int lsize = get_local_size(0);
    NF_LOCAL float partial[LLM_DIMX];

    const nf_i64 base = (nf_i64)c0 * len;
    const int cnt = (c1 - c0) * len;
    const float mean = nf_wg_sum(partial, nf_norm_ext_sum(x, base, cnt, lid, lsize)) / (float)cnt;
    const float var = nf_wg_sum(partial, nf_norm_ext_sqdev(x, base, cnt, mean, lid, lsize)) / (float)cnt;
    const float scale = rsqrt(var + eps);
    for (int c = c0; c < c1; c++) {
        const nf_i64 cb = (nf_i64)c * len;
        const float wc = w[c];
        const float bc = b[c];
        for (int t = lid; t < len; t += lsize) {
            y[cb + t] = (x[cb + t] - mean) * scale * wc + bc;
        }
    }
}

/*
 * Residual combination over i < n, a = inp[inp_off + i], c = cur[cur_off + i]:
 *   mode 0 ADD                  out = a + c
 *   mode 1 SCALED               out = a + s*c                (granite / minicpm, models/granite.cpp:248-252)
 *   mode 2 ALPHA_BETA           out = a*alpha[k] + c*beta[k], k = i % ab_len   (models/muse-spark.cpp:166-168)
 *   mode 3 NORMED_INPUT_SCALED  out = c + s*a, inp = the normed input           (models/minimax-01.cpp:428-431)
 * written to out[out_off + i]. out may coincide with inp or cur (same buffer and offset). alpha / beta are
 * read only in mode 2 (placeholders otherwise); ab_len is the broadcast period of alpha / beta (n_embd).
 * Dispatch: global = {roundUp(n, L)}, local = {L}.
 */
NF_KERNEL void LLM_ResidualMix(
    const int n,
    const int mode,
    const float s,
    const int inp_off,
    const int cur_off,
    const int out_off,
    const int ab_len,
    NF_GLOBAL const float* inp,
    NF_GLOBAL const float* cur,
    NF_GLOBAL const float* NF_RESTRICT alpha,
    NF_GLOBAL const float* NF_RESTRICT beta,
    NF_GLOBAL float* out)
{
    const uint i = nf_norm_ext_linear_id();
    if (i >= (uint)n) return;
    const float a = inp[(nf_i64)inp_off + i];
    const float c = cur[(nf_i64)cur_off + i];
    float r;
    if (mode == NF_MIX_SCALED) {
        r = a + s * c;
    } else if (mode == NF_MIX_ALPHA_BETA) {
        const uint k = ab_len > 1 ? i % (uint)ab_len : 0u;
        r = a * alpha[k] + c * beta[k];
    } else if (mode == NF_MIX_NORMED_INPUT_SCALED) {
        r = c + s * a;
    } else {
        r = a + c;
    }
    out[(nf_i64)out_off + i] = r;
}

/*
 * In-place row scale of a dense [n_rows][row_len] tensor: x[r*row_len + c] *= scale[scale_len == 1 ? 0 : c]
 * (layer_out_scale {1}, per-column output scales {row_len}; llama-graph.cpp:1641-1645).
 * Dispatch: global = {roundUp(n_rows * row_len, L)}, local = {L}.
 */
NF_KERNEL void LLM_MulRowScale(
    const int n_rows,
    const int row_len,
    const int scale_len,
    NF_GLOBAL const float* NF_RESTRICT scale,
    NF_GLOBAL float* x)
{
    const uint idx = nf_norm_ext_linear_id();
    if (row_len <= 0 || idx >= (uint)n_rows * (uint)row_len) return;
    const uint c = idx % (uint)row_len;
    x[idx] = x[idx] * scale[scale_len == 1 ? 0u : c];
}

/* norm_ext.cl - LLM_RowNorm, LLM_GroupRms, LLM_GroupNorm, LLM_ResidualMix, LLM_ResidualMixNorm, their
 * Q8_1-emitting forms LLM_RowNormQ8 and LLM_ResidualMixNormQ8, and row-scale kernels.
 * Requires: common/dialect.h, common/config.h, common/reduce.h */
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

/*
 * Elements a work-item loads before it uses the first of them in the strided row passes below. A row of
 * row_len floats gives each work-item ceil(row_len / L) elements (10 for a 2560-wide row at L = 256); a
 * pass that waited one load latency per element would chain them all, so the passes load NF_NORM_EXT_BATCH
 * elements per step first. Every pass still visits a work-item's elements in increasing index order, so
 * its per-work-item sums, and every result bit, do not depend on the batch.
 */
#define NF_NORM_EXT_BATCH 4

/* Loads a work-item's next batch of the strided elements i0, i0 + lsize, ... of x[base ..] into v (0 past n). */
#define NF_NORM_EXT_LOAD(v, x, base, n, i0, lsize)                 \
    for (int u_ = 0; u_ < NF_NORM_EXT_BATCH; u_++) {               \
        const int i_ = (i0) + u_ * (lsize);                        \
        (v)[u_] = i_ < (n) ? (x)[(base) + i_] : 0.0f;              \
    }

/* Sum of x[base + i] over i in [0, n), strided over the workgroup. */
NF_INLINE float nf_norm_ext_sum(NF_GLOBAL const float* x, nf_i64 base, int n, int lid, int lsize) {
    float s = 0.0f;
    for (int i0 = lid; i0 < n; i0 += NF_NORM_EXT_BATCH * lsize) {
        float v[NF_NORM_EXT_BATCH];
        NF_NORM_EXT_LOAD(v, x, base, n, i0, lsize)
#pragma unroll
        for (int u = 0; u < NF_NORM_EXT_BATCH; u++) {
            if (i0 + u * lsize < n) {
                s += v[u];
            }
        }
    }
    return s;
}

/* Sum of x[base + i]^2 over i in [0, n). */
NF_INLINE float nf_norm_ext_sumsq(NF_GLOBAL const float* x, nf_i64 base, int n, int lid, int lsize) {
    float s = 0.0f;
    for (int i0 = lid; i0 < n; i0 += NF_NORM_EXT_BATCH * lsize) {
        float v[NF_NORM_EXT_BATCH];
        NF_NORM_EXT_LOAD(v, x, base, n, i0, lsize)
#pragma unroll
        for (int u = 0; u < NF_NORM_EXT_BATCH; u++) {
            if (i0 + u * lsize < n) {
                s += v[u] * v[u];
            }
        }
    }
    return s;
}

/* Sum of (x[base + i] - mean)^2 over i in [0, n). */
NF_INLINE float nf_norm_ext_sqdev(NF_GLOBAL const float* x, nf_i64 base, int n, float mean, int lid, int lsize) {
    float s = 0.0f;
    for (int i0 = lid; i0 < n; i0 += NF_NORM_EXT_BATCH * lsize) {
        float v[NF_NORM_EXT_BATCH];
        NF_NORM_EXT_LOAD(v, x, base, n, i0, lsize)
#pragma unroll
        for (int u = 0; u < NF_NORM_EXT_BATCH; u++) {
            if (i0 + u * lsize < n) {
                const float d = v[u] - mean;
                s += d * d;
            }
        }
    }
    return s;
}

/* Flattened work-item index of a 1-D or 2-D range. */
NF_INLINE uint nf_norm_ext_linear_id(void) {
    return (uint)get_global_id(0) + (uint)get_global_id(1) * (uint)get_global_size(0);
}

/*
 * Element i of a normalized row's Q8_1 form, inside the final pass of a row-norm workgroup whose
 * work-item lid handles elements t * L + lid (L a multiple of 32), so the 32 elements of a block are
 * the 32 work-items of one aligned segment. Every work-item of the workgroup calls it once per pass
 * step (the local-memory segment reductions need it); a work-item past the row passes live = 0, and a
 * whole segment is live or not because row_len is a multiple of 32. The arithmetic is
 * LLM_QuantizeQ8_1's: amax = max |v| through fmax from 0 (a NaN drops out), d = amax / 127,
 * id = d != 0 ? 1/d : 0, q = clamp(round(v * id)), s = d * sum(q); the code sum lies in [-4064, 4064],
 * so its float reduction is exact in any order and the bits equal the separate quantizer's.
 */
NF_INLINE void nf_norm_ext_q8(NF_LOCAL_PTR float* buf, NF_GLOBAL uchar* q8row, int i, int live, float v) {
    const float amax = nf_seg_max(buf, live ? fmax(0.0f, fabs(v)) : 0.0f, Q8_1_BLOCK_SIZE);
    const float d = amax / 127.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    const int q = live ? clamp((int)round(v * id), -127, 127) : 0;
    const float sum = nf_seg_sum(buf, (float)q, Q8_1_BLOCK_SIZE);
    if (live) {
        NF_GLOBAL uchar* blk = q8row + (nf_i64)(i / Q8_1_BLOCK_SIZE) * Q8_1_BLOCK_BYTES;
        const int lane = i % Q8_1_BLOCK_SIZE;
        blk[Q8_1_QS_OFFSET + lane] = (uchar)(q & 0xFF);
        if (lane == 0) {
            vstore_half(d, 0, (NF_GLOBAL NF_HALF*)blk);
            vstore_half(sum * d, 1, (NF_GLOBAL NF_HALF*)blk);
        }
    }
}

/*
 * The final pass of the row norms: y[yb + i] = ((x[xb + i] - mean) * scale [* post]) [* w] [+ b] for every
 * element of the row, element i by work-item i % L in step i / L. With emit_q8 (uniform) every work-item
 * runs the same number of steps and the row's Q8_1 blocks go to q8row as well (nf_norm_ext_q8).
 */
NF_INLINE void nf_norm_ext_store(NF_GLOBAL const float* x, nf_i64 xb, NF_GLOBAL float* y, nf_i64 yb, int row_len,
                                 int mode, float mean, float scale, float post, int flags,
                                 NF_GLOBAL const float* w, NF_GLOBAL const float* b, nf_i64 wb,
                                 NF_GLOBAL uchar* q8row, int emit_q8, NF_LOCAL_PTR float* buf) {
    const int lid = get_local_id(0);
    const int lsize = get_local_size(0);
    const int has_w = (flags & NF_NORM_EXT_HAS_W) != 0;
    const int has_b = (flags & NF_NORM_EXT_HAS_B) != 0;
    /* emit_q8 runs every work-item through the same steps (the Q8_1 segment reductions are collective):
     * a step past the row is dead for the whole workgroup, past this work-item's last element for it. */
    const int span = emit_q8 ? (row_len + lsize - 1) / lsize * lsize : row_len;
    for (int i0 = lid; i0 - lid < span && (emit_q8 || i0 < row_len); i0 += NF_NORM_EXT_BATCH * lsize) {
        float xv[NF_NORM_EXT_BATCH];
        float wv[NF_NORM_EXT_BATCH];
        float bv[NF_NORM_EXT_BATCH];
        NF_NORM_EXT_LOAD(xv, x, xb, row_len, i0, lsize)
        if (has_w) {
            NF_NORM_EXT_LOAD(wv, w, wb, row_len, i0, lsize)
        }
        if (has_b) {
            NF_NORM_EXT_LOAD(bv, b, wb, row_len, i0, lsize)
        }
#pragma unroll
        for (int u = 0; u < NF_NORM_EXT_BATCH; u++) {
            const int i = i0 + u * lsize;
            const int live = i < row_len;
            float v = 0.0f;
            if (live) {
                v = (xv[u] - mean) * scale;
                if (mode == NF_NORM_EXT_L2_GDN) {
                    v *= post;
                }
                /* one rounding for the weight and bias step, as the per-variant loops always compiled it */
                if (has_w && has_b) {
                    v = fma(v, wv[u], bv[u]);
                } else if (has_w) {
                    v *= wv[u];
                } else if (has_b) {
                    v += bv[u];
                }
                y[yb + i] = v;
            }
            if (emit_q8 && i - lid < span) {
                nf_norm_ext_q8(buf, q8row, i, live, v);
            }
        }
    }
}

/* ADD or SCALED residual combination of one element, as LLM_ResidualMix modes 0 and 1: a + c, a + s*c. */
NF_INLINE float nf_rmn_mix(float a, float c, int mix_mode, float s) {
    return mix_mode == NF_MIX_SCALED ? a + s * c : a + c;
}

/* LLM_RowNorm's body; emit_q8 adds the Q8_1 blocks of LLM_RowNormQ8. */
NF_INLINE void nf_row_norm(const int n_rows, const int row_len, const int row_stride, const int mode, const float eps,
                           const int flags, const int w_rows, const int x_off, const int y_off,
                           NF_GLOBAL const float* x, NF_GLOBAL const float* w, NF_GLOBAL const float* b,
                           NF_GLOBAL float* y, NF_GLOBAL uchar* q8, const int emit_q8, NF_LOCAL_PTR float* partial) {
    const int row = get_group_id(0);
    if (row >= n_rows) return;
    const int lid = get_local_id(0);
    const int lsize = get_local_size(0);

    const nf_i64 xb = (nf_i64)x_off + (nf_i64)row * row_stride;
    const nf_i64 yb = (nf_i64)y_off + (nf_i64)row * row_stride;
    const nf_i64 wb = (nf_i64)(w_rows > 1 ? row % w_rows : 0) * row_len;
    NF_GLOBAL uchar* q8row = q8 + (nf_i64)row * (row_len / Q8_1_BLOCK_SIZE) * Q8_1_BLOCK_BYTES;
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
    nf_norm_ext_store(x, xb, y, yb, row_len, mode, mean, scale, post, flags, w, b, wb, q8row, emit_q8, partial);
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
    NF_LOCAL float partial[LLM_DIMX];
    nf_row_norm(n_rows, row_len, row_stride, mode, eps, flags, w_rows, x_off, y_off, x, w, b, y,
                (NF_GLOBAL uchar*)y, 0, partial);
}

/*
 * LLM_RowNorm that also writes each normalized row as Q8_1 blocks: row r's blocks at
 * q8 + r * (row_len / 32) * 36 bytes, the layout and the bits LLM_QuantizeQ8_1 writes for the same
 * values (nf_norm_ext_q8). The y bits equal LLM_RowNorm's. A projection of the row on the integer-dot
 * lane then reads these blocks instead of quantizing the row in a launch of its own.
 * Dispatch: as LLM_RowNorm, L a multiple of 32; row_len a multiple of 32.
 */
NF_KERNEL void LLM_RowNormQ8(
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
    NF_GLOBAL float* y,
    NF_GLOBAL uchar* NF_RESTRICT q8)
{
    NF_LOCAL float partial[LLM_DIMX];
    nf_row_norm(n_rows, row_len, row_stride, mode, eps, flags, w_rows, x_off, y_off, x, w, b, y, q8, 1, partial);
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

/* LLM_ResidualMixNorm's body; emit_q8 adds the Q8_1 blocks of LLM_ResidualMixNormQ8. */
NF_INLINE void nf_residual_mix_norm(const int n_rows, const int row_len, const int mode, const float eps,
                                    const int flags, const int w_rows, const int mix_mode, const float mix_s,
                                    const int x_off, const int add_off, const int y_off, NF_GLOBAL float* x,
                                    NF_GLOBAL const float* addend, NF_GLOBAL const float* w,
                                    NF_GLOBAL const float* b, NF_GLOBAL float* y, NF_GLOBAL uchar* q8,
                                    const int emit_q8, NF_LOCAL_PTR float* partial) {
    const int row = get_group_id(0);
    if (row >= n_rows) return;
    const int lid = get_local_id(0);
    const int lsize = get_local_size(0);

    const nf_i64 xb = (nf_i64)x_off + (nf_i64)row * row_len;
    const nf_i64 ab = (nf_i64)add_off + (nf_i64)row * row_len;
    const nf_i64 yb = (nf_i64)y_off + (nf_i64)row * row_len;
    const nf_i64 wb = (nf_i64)(w_rows > 1 ? row % w_rows : 0) * row_len;
    NF_GLOBAL uchar* q8row = q8 + (nf_i64)row * (row_len / Q8_1_BLOCK_SIZE) * Q8_1_BLOCK_BYTES;
    const float n = (float)row_len;

    float sum = 0.0f;
    float ss = 0.0f;
    for (int i0 = lid; i0 < row_len; i0 += NF_NORM_EXT_BATCH * lsize) {
        float xv[NF_NORM_EXT_BATCH];
        float av[NF_NORM_EXT_BATCH];
        NF_NORM_EXT_LOAD(xv, x, xb, row_len, i0, lsize)
        NF_NORM_EXT_LOAD(av, addend, ab, row_len, i0, lsize)
#pragma unroll
        for (int u = 0; u < NF_NORM_EXT_BATCH; u++) {
            const int i = i0 + u * lsize;
            if (i < row_len) {
                const float v = nf_rmn_mix(xv[u], av[u], mix_mode, mix_s);
                x[xb + i] = v;
                sum += v;
                ss += v * v;
            }
        }
    }

    float mean = 0.0f;
    float scale;
    float post = 1.0f;
    if (mode == NF_NORM_EXT_LAYER) {
        mean = nf_wg_sum(partial, sum) / n;
        scale = rsqrt(nf_wg_sum(partial, nf_norm_ext_sqdev(x, xb, row_len, mean, lid, lsize)) / n + eps);
    } else {
        const float tss = nf_wg_sum(partial, ss);
        if (mode == NF_NORM_EXT_L2_CLAMP) {
            scale = 1.0f / fmax(sqrt(tss), eps);
        } else if (mode == NF_NORM_EXT_L2_GDN) {
            scale = rsqrt(tss / n + eps / n);
            post = rsqrt(n);
        } else {
            scale = rsqrt(tss / n + eps);
        }
    }
    nf_norm_ext_store(x, xb, y, yb, row_len, mode, mean, scale, post, flags, w, b, wb, q8row, emit_q8, partial);
}

/*
 * The residual combination of LLM_ResidualMix (modes ADD and SCALED) and the row norm of LLM_RowNorm in
 * one workgroup per row: row r of the residual stream is rewritten with x = mix(x, addend) at
 * x[x_off + r*row_len], and its norm is written to y[y_off + r*row_len] with the same modes, scale, weight
 * and bias as LLM_RowNorm. A layer that adds a branch to the stream and normalizes the sum therefore
 * needs one launch instead of the residual mix's and the norm's. x and y must not overlap; addend may
 * coincide with neither. Every element is written by the work-item that read it, so the reductions only
 * need the workgroup barriers of nf_wg_sum.
 * Dispatch: global = {n_rows * L}, local = {L}; workgroup r handles row r.
 */
NF_KERNEL void LLM_ResidualMixNorm(
    const int n_rows,
    const int row_len,
    const int mode,
    const float eps,
    const int flags,
    const int w_rows,
    const int mix_mode,
    const float mix_s,
    const int x_off,
    const int add_off,
    const int y_off,
    NF_GLOBAL float* x,
    NF_GLOBAL const float* NF_RESTRICT addend,
    NF_GLOBAL const float* NF_RESTRICT w,
    NF_GLOBAL const float* NF_RESTRICT b,
    NF_GLOBAL float* y)
{
    NF_LOCAL float partial[LLM_DIMX];
    nf_residual_mix_norm(n_rows, row_len, mode, eps, flags, w_rows, mix_mode, mix_s, x_off, add_off, y_off,
                         x, addend, w, b, y, (NF_GLOBAL uchar*)y, 0, partial);
}

/*
 * LLM_ResidualMixNorm that also writes each normalized row as Q8_1 blocks at q8 + r * (row_len / 32) * 36
 * bytes, with the bits LLM_QuantizeQ8_1 writes for the same values (nf_norm_ext_q8); x and y get the bits
 * LLM_ResidualMixNorm writes. Dispatch: as LLM_ResidualMixNorm, L a multiple of 32; row_len a multiple
 * of 32.
 */
NF_KERNEL void LLM_ResidualMixNormQ8(
    const int n_rows,
    const int row_len,
    const int mode,
    const float eps,
    const int flags,
    const int w_rows,
    const int mix_mode,
    const float mix_s,
    const int x_off,
    const int add_off,
    const int y_off,
    NF_GLOBAL float* x,
    NF_GLOBAL const float* NF_RESTRICT addend,
    NF_GLOBAL const float* NF_RESTRICT w,
    NF_GLOBAL const float* NF_RESTRICT b,
    NF_GLOBAL float* y,
    NF_GLOBAL uchar* NF_RESTRICT q8)
{
    NF_LOCAL float partial[LLM_DIMX];
    nf_residual_mix_norm(n_rows, row_len, mode, eps, flags, w_rows, mix_mode, mix_s, x_off, add_off, y_off,
                         x, addend, w, b, y, q8, 1, partial);
}

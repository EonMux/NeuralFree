/* fused.cl - fused forms of decode-path kernel pairs and of the K-row path into the KV cache.
 * Requires: common/dialect.h, common/config.h, common/math.h, common/reduce.h, common/rope_common.h */
/*
 * Every kernel here replaces two kernels the layer executor would otherwise launch back to back, with the same
 * operands and the same arithmetic:
 *
 *   LLM_RmsNormRope   = LLM_RowNorm (RMS) then LLM_RoPE_Ext on the same rows (norm before rope).
 *   LLM_GdnNormGate   = LLM_RowNorm (RMS) then LLM_GLU (SWIGLU) of a gated DeltaNet layer's output.
 *   LLM_HcPostNorm    = LLM_HC_Post (expand one row into hc streams) then LLM_RowNorm (RMS) of each expanded
 *                       stream row, the combine-and-norm pair of ggml-cuda/qwen4-hc.cu.
 *   LLM_NormRopeStore = LLM_RowNorm (RMS), LLM_RoPE_Ext and the cache-row write of LLM_KV_Store (llama.cpp's
 *                       rms_norm + mul + rope + view + set_rows fusion).
 *   LLM_RopeStore     = LLM_RoPE_Ext and the cache-row write (llama.cpp's rope + view + set_rows fusion).
 *
 * The hosts pick them through the existing dispatch ({@code GpuOps}); the unfused pair remains the fallback for a
 * device that does not provide them (see {@code fused.cl}'s GLSL twin shaders/llm/fused.comp).
 */

/* LLM_RowNorm flags: weight and bias are bound (gguf.ops.norm.NormKernels). */
#define NF_FUSED_HAS_W 1
#define NF_FUSED_HAS_B 2

/* Largest stream count of LLM_HcPostNorm (the hc kernels' NF_HC_MAX). */
#define NF_FUSED_HC_MAX 8

/*
 * LLM_RmsNormRope: RMS-normalize n_heads rows of row_len values per token, then rotate the pairs of every row as
 * LLM_RoPE_Ext does. Row (t, h) starts at x[x_off + t*row_stride + h*head_stride]; y receives the normalized and
 * rotated row at the same indices plus y_off. The norm weight and bias, when bound, use row
 * (h % w_rows) * row_len. Rotation pair i of mode `mode` rotates the elements relative to n_offs selected by
 * common/rope_common.h, at position pos[t + section*pos_stride]. x and y may coincide.
 *
 * Dispatch: 1-D, one workgroup of L work-items per (token, head); global = n_tokens*n_heads*L, local = L, L a power
 * of two <= LLM_DIMX and a multiple of 32 on CUDA. Requires 1 <= n_dims <= head row and
 * n_offs + rotated width <= head row (the host validates). A pair of the rotation may be written by a different
 * work-item than the one that normalized it, so the barrier between the two phases also fences global memory.
 */
NF_KERNEL void LLM_RmsNormRope(
    const int n_tokens,
    const int n_heads,
    const int head_stride,
    const int row_stride,
    const int row_len,
    const int mode,
    const int n_dims,
    const int n_offs,
    const int pos_stride,
    const int has_ff,
    const int flags,
    const int w_rows,
    const int x_off,
    const int y_off,
    const int sect0,
    const int sect1,
    const int sect2,
    const int sect3,
    const float eps,
    const float theta_scale,
    const float freq_scale,
    const float ext_factor,
    const float attn_factor,
    const float corr_lo,
    const float corr_hi,
    NF_GLOBAL const int* NF_RESTRICT pos,
    NF_GLOBAL const float* NF_RESTRICT ff,
    NF_GLOBAL const float* NF_RESTRICT w,
    NF_GLOBAL const float* NF_RESTRICT b,
    NF_GLOBAL const float* x,
    NF_GLOBAL float* y)
{
    const nf_i64 grp = (nf_i64)get_group_id(0);
    if (grp >= (nf_i64)n_tokens * n_heads) {
        return;
    }
    const int h = (int)(grp % n_heads);
    const int t = (int)(grp / n_heads);
    const int lid = get_local_id(0);
    const int lsize = get_local_size(0);
    NF_LOCAL float partial[LLM_DIMX];

    const nf_i64 xb = (nf_i64)x_off + (nf_i64)t * row_stride + (nf_i64)h * head_stride;
    const nf_i64 yb = (nf_i64)y_off + (nf_i64)t * row_stride + (nf_i64)h * head_stride;

    float ss = 0.0f;
    for (int i = lid; i < row_len; i += lsize) {
        const float v = x[xb + i];
        ss += v * v;
    }
    const float rstd = rsqrt(nf_wg_sum(partial, ss) / (float)row_len + eps);

    const int has_w = (flags & NF_FUSED_HAS_W) != 0;
    const int has_b = (flags & NF_FUSED_HAS_B) != 0;
    const nf_i64 wrow = (nf_i64)(w_rows > 1 ? h % w_rows : 0) * row_len;
    for (int i = lid; i < row_len; i += lsize) {
        float v = x[xb + i] * rstd;
        if (has_w) {
            v *= w[wrow + i];
        }
        if (has_b) {
            v += b[wrow + i];
        }
        y[yb + i] = v;
    }
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    /* The rotation reads the normalized rows, so y is the input of this pass (in place when x == y). */
    const int pairs = nf_rope_pairs(mode, n_dims);
    for (int i = lid; i < pairs; i += lsize) {
        const int sec = nf_rope_section(mode, i, sect0, sect1, sect2, sect3);
        const int e = nf_rope_theta_exp(mode, i, sec, sect0, sect1, sect2, sect3);
        const int p = pos[t + sec * pos_stride];
        const float f = has_ff ? ff[i] : 1.0f;
        const float2 cs = nf_rope_cs(p, e, theta_scale, f, freq_scale, ext_factor, attn_factor, corr_lo, corr_hi,
                                     i);
        const nf_i64 a = yb + n_offs + nf_rope_first(mode, i);
        const nf_i64 c = a + nf_rope_span(mode, n_dims);
        const float y0 = y[a];
        const float y1 = y[c];
        y[a] = y0 * cs.x - y1 * cs.y;
        y[c] = y0 * cs.y + y1 * cs.x;
    }
}

/*
 * LLM_GdnNormGate: the gated DeltaNet output norm and gate in one launch. Row (t, h) of `x` (the scan output)
 * receives an RMS norm with the ssm_norm weight row (h % w_rows) * head_dim, then is multiplied by the SiLU of
 * the matching row of `z` (the attention-gate projection), written to `out`:
 *   out = silu(z) * rms_norm(x, weight)       (LLM_RowNorm RMS then LLM_GLU SWIGLU)
 * Rows are valued at x[x_off + t*row_stride + h*head_dim] and z[z_off + t*row_stride + h*head_dim]; `out` uses the
 * same row layout plus out_off. x and out may coincide.
 *
 * Dispatch: 1-D, one workgroup of L work-items per (token, value head); global = n_tokens*n_heads*L, local = L.
 */
NF_KERNEL void LLM_GdnNormGate(
    const int n_tokens,
    const int n_heads,
    const int head_dim,
    const int row_stride,
    const int x_off,
    const int z_off,
    const int out_off,
    const int w_rows,
    const float eps,
    NF_GLOBAL const float* x,
    NF_GLOBAL const float* NF_RESTRICT w,
    NF_GLOBAL const float* NF_RESTRICT z,
    NF_GLOBAL float* out)
{
    const nf_i64 grp = (nf_i64)get_group_id(0);
    if (grp >= (nf_i64)n_tokens * n_heads) {
        return;
    }
    const int h = (int)(grp % n_heads);
    const int t = (int)(grp / n_heads);
    const int lid = get_local_id(0);
    const int lsize = get_local_size(0);
    NF_LOCAL float partial[LLM_DIMX];

    const nf_i64 xb = (nf_i64)x_off + (nf_i64)t * row_stride + (nf_i64)h * head_dim;
    const nf_i64 zb = (nf_i64)z_off + (nf_i64)t * row_stride + (nf_i64)h * head_dim;
    const nf_i64 ob = (nf_i64)out_off + (nf_i64)t * row_stride + (nf_i64)h * head_dim;
    const nf_i64 wb = (nf_i64)(w_rows > 1 ? (int)(grp % w_rows) : 0) * head_dim;

    float ss = 0.0f;
    for (int i = lid; i < head_dim; i += lsize) {
        const float v = x[xb + i];
        ss += v * v;
    }
    const float rstd = rsqrt(nf_wg_sum(partial, ss) / (float)head_dim + eps);
    for (int i = lid; i < head_dim; i += lsize) {
        out[ob + i] = nf_silu(z[zb + i]) * (x[xb + i] * rstd * w[wb + i]);
    }
}

/*
 * LLM_HcPostNorm: expand one n_embd row into hc streams and normalize every expanded stream row (the combine and
 * norm pair of ggml-cuda/qwen4-hc.cu). For token t, destination stream dst and value i:
 *   v[dst][i] = y[t*n_embd + i] * p[dst]
 *             + (has_comb ? sum_src X[(t*hc + src)*n_embd + i] * comb[t*hc*hc + dst + hc*src]
 *                         : X[(t*hc + dst)*n_embd + i])
 *   p[dst]    = post_act ? post_mag * sigmoid(post_in_scale * post[t*hc + dst]) : post[t*hc + dst]
 *   Xout[(t*hc + dst)*n_embd + i] = v[dst][i]
 *   xn[(t*hc + dst)*n_embd + i]   = v[dst][i] * rsqrt(sum v^2 / n + eps)
 *                                   * (has_w ? gamma[(dst % w_rows)*n_embd + i] : 1)
 * The norm scope selects the summed values, matching the leading norm of the consuming site:
 *   scope 0 (gated site): one axis per stream row (n = n_embd), the weight row dst % w_rows;
 *   scope 1 (projected site): one axis per token over the whole expanded row (n = hc*n_embd), no weight.
 * Xout may equal X with xout_off == x_off: one workgroup handles one token, reads every stream of a value index
 * before writing any, so no other workgroup writes the values it still reads.
 *
 * Dispatch: 1-D, one workgroup of L work-items per token; global = n_tokens*L, local = L. Requires
 * 1 <= hc <= NF_HC_MAX.
 */
NF_KERNEL void LLM_HcPostNorm(
    const int n_tokens,
    const int n_embd,
    const int hc,
    const int has_comb,
    const int post_act,
    const int has_w,
    const int w_rows,
    const int scope,
    const int y_off,
    const int x_off,
    const int post_off,
    const int comb_off,
    const int xout_off,
    const int xn_off,
    const float post_in_scale,
    const float post_mag,
    const float eps,
    NF_GLOBAL const float* NF_RESTRICT y,
    NF_GLOBAL const float* X,
    NF_GLOBAL const float* NF_RESTRICT post,
    NF_GLOBAL const float* NF_RESTRICT comb,
    NF_GLOBAL float* Xout,
    NF_GLOBAL const float* NF_RESTRICT gamma,
    NF_GLOBAL float* xn)
{
    const nf_i64 t = (nf_i64)get_group_id(0);
    if (t >= n_tokens || hc < 1 || hc > NF_FUSED_HC_MAX) {
        return;
    }
    const int lid = get_local_id(0);
    const int lsize = get_local_size(0);
    NF_LOCAL float partial[LLM_DIMX];

    const nf_i64 yb = (nf_i64)y_off + t * n_embd;
    const nf_i64 xb = (nf_i64)x_off + t * hc * n_embd;
    const nf_i64 ob = (nf_i64)xout_off + t * hc * n_embd;
    const nf_i64 nb = (nf_i64)xn_off + t * hc * n_embd;
    const nf_i64 pb = (nf_i64)post_off + t * hc;
    const nf_i64 cb = (nf_i64)comb_off + t * hc * hc;

    float p[NF_FUSED_HC_MAX];
    for (int dst = 0; dst < hc; dst++) {
        if (post_act != 0) {
            p[dst] = post_mag * nf_sigmoid(post_in_scale * post[pb + dst]);
        } else {
            p[dst] = post[pb + dst];
        }
    }

    const int per_row = scope == 0;
    float ssq[NF_FUSED_HC_MAX];
    for (int dst = 0; dst < hc; dst++) {
        ssq[dst] = 0.0f;
    }
    float total = 0.0f;
    for (int i = lid; i < n_embd; i += lsize) {
        /* Every stream value of this column is read before any is written, so Xout may be X (has_comb). */
        float src[NF_FUSED_HC_MAX];
        for (int h = 0; h < hc; h++) {
            src[h] = X[xb + h * n_embd + i];
        }
        const float yv = y[yb + i];
        for (int dst = 0; dst < hc; dst++) {
            float base = src[dst];
            if (has_comb != 0) {
                base = 0.0f;
                for (int s = 0; s < hc; s++) {
                    base += src[s] * comb[cb + dst + hc * s];
                }
            }
            const float v = yv * p[dst] + base;
            if (per_row) {
                ssq[dst] += v * v;
            } else {
                total += v * v;
            }
        }
    }
    float rstd[NF_FUSED_HC_MAX];
    if (per_row) {
        for (int dst = 0; dst < hc; dst++) {
            rstd[dst] = rsqrt(nf_wg_sum(partial, ssq[dst]) / (float)n_embd + eps);
        }
    } else {
        const float r = rsqrt(nf_wg_sum(partial, total) / (float)(hc * n_embd) + eps);
        for (int dst = 0; dst < hc; dst++) {
            rstd[dst] = r;
        }
    }

    const int has_gamma = has_w != 0 && per_row;
    for (int i = lid; i < n_embd; i += lsize) {
        float src[NF_FUSED_HC_MAX];
        for (int h = 0; h < hc; h++) {
            src[h] = X[xb + h * n_embd + i];
        }
        const float yv = y[yb + i];
        for (int dst = 0; dst < hc; dst++) {
            float base = src[dst];
            if (has_comb != 0) {
                base = 0.0f;
                for (int s = 0; s < hc; s++) {
                    base += src[s] * comb[cb + dst + hc * s];
                }
            }
            const float v = yv * p[dst] + base;
            Xout[ob + dst * n_embd + i] = v;
            const int grow = w_rows > 1 ? dst % w_rows : 0;
            float nv = v * rstd[dst];
            if (has_gamma) {
                nv *= gamma[(nf_i64)grow * n_embd + i];
            }
            xn[nb + dst * n_embd + i] = nv;
        }
    }
}

/*
 * LLM_NormRopeStore and LLM_RopeStore: the K-row path of an attention layer up to the KV cache write, as one
 * launch (llama.cpp's rms_norm + mul + rope + view + set_rows and rope + view + set_rows fusions). They compute
 * the normalized (first form) and rotated row element by element and write it directly to the cache row
 * cell_idx[t], instead of materializing the rotated K rows and copying them in a second launch.
 *
 * dst_type selects the cache element type: 0 = F32, 1 = F16 (round to nearest even, as LLM_KV_Store_F16). The host
 * offers them only for those two cache types, as llama.cpp does. A negative cell index skips the token.
 *
 * Dispatch: 1-D, one workgroup of L work-items per (token, KV head), global = n_tokens*n_heads*L, local = L.
 */

/* LLM_NormRopeStore: RMS-normalize row (t, h) of x with the per-head weight (h % w_rows) * row_len and bias,
 * rotate its pairs, and store the result at dst[cell_idx[t]*dst_row_stride + h*dst_head_elems + e]. The size and
 * type of the destination come from dst_type; the row geometry is that of LLM_RmsNormRope. */
NF_KERNEL void LLM_NormRopeStore(
    const int n_tokens,
    const int n_heads,
    const int head_stride,
    const int row_stride,
    const int row_len,
    const int mode,
    const int n_dims,
    const int n_offs,
    const int pos_stride,
    const int has_ff,
    const int flags,
    const int w_rows,
    const int x_off,
    const int dst_head_elems,
    const int dst_row_stride,
    const int dst_type,
    const int sect0,
    const int sect1,
    const int sect2,
    const int sect3,
    const float eps,
    const float theta_scale,
    const float freq_scale,
    const float ext_factor,
    const float attn_factor,
    const float corr_lo,
    const float corr_hi,
    NF_GLOBAL const int* NF_RESTRICT pos,
    NF_GLOBAL const float* NF_RESTRICT ff,
    NF_GLOBAL const float* NF_RESTRICT w,
    NF_GLOBAL const float* NF_RESTRICT b,
    NF_GLOBAL const float* x,
    NF_GLOBAL const int* NF_RESTRICT cell_idx,
    NF_GLOBAL uchar* dst)
{
    const nf_i64 grp = (nf_i64)get_group_id(0);
    if (grp >= (nf_i64)n_tokens * n_heads) {
        return;
    }
    const int h = (int)(grp % n_heads);
    const int t = (int)(grp / n_heads);
    const int cell = cell_idx[t];
    if (cell < 0) {
        return;
    }
    const int lid = get_local_id(0);
    const int lsize = get_local_size(0);
    NF_LOCAL float partial[LLM_DIMX];

    const nf_i64 xb = (nf_i64)x_off + (nf_i64)t * row_stride + (nf_i64)h * head_stride;
    const nf_i64 db = (nf_i64)cell * dst_row_stride + (nf_i64)h * dst_head_elems;

    float ss = 0.0f;
    for (int i = lid; i < row_len; i += lsize) {
        const float v = x[xb + i];
        ss += v * v;
    }
    const float rstd = rsqrt(nf_wg_sum(partial, ss) / (float)row_len + eps);
    const int has_w = (flags & NF_FUSED_HAS_W) != 0;
    const int has_b = (flags & NF_FUSED_HAS_B) != 0;
    const nf_i64 wrow = (nf_i64)(w_rows > 1 ? h % w_rows : 0) * row_len;
    const int n_offs_end = n_offs + (mode == NF_ROPE_VISION ? 2 * n_dims : n_dims);
    const int span = nf_rope_span(mode, n_dims);
    for (int i = lid; i < row_len; i += lsize) {
        float v = x[xb + i] * rstd;
        if (has_w) {
            v *= w[wrow + i];
        }
        if (has_b) {
            v += b[wrow + i];
        }
        if (i >= n_offs && i < n_offs_end) {
            const int rel = i - n_offs;
            int pair;
            int second;
            if (mode == NF_ROPE_NORM) {
                pair = rel >> 1;
                second = rel & 1;
            } else if (mode == NF_ROPE_VISION) {
                pair = rel < n_dims ? rel : rel - n_dims;
                second = rel >= n_dims ? 1 : 0;
            } else {
                const int half_dims = n_dims >> 1;
                pair = rel < half_dims ? rel : rel - half_dims;
                second = rel >= half_dims ? 1 : 0;
            }
            const int sec = nf_rope_section(mode, pair, sect0, sect1, sect2, sect3);
            const int e = nf_rope_theta_exp(mode, pair, sec, sect0, sect1, sect2, sect3);
            const int p = pos[t + sec * pos_stride];
            const float f = has_ff != 0 ? ff[pair] : 1.0f;
            const float2 cs = nf_rope_cs(p, e, theta_scale, f, freq_scale, ext_factor, attn_factor, corr_lo,
                                         corr_hi, pair);
            const int other_i = second != 0 ? i - span : i + span;
            float other = x[xb + other_i] * rstd;
            if (has_w) {
                other *= w[wrow + other_i];
            }
            if (has_b) {
                other += b[wrow + other_i];
            }
            v = second != 0 ? other * cs.y + v * cs.x : v * cs.x - other * cs.y;
        }
        if (dst_type == 1) {
            vstore_half(v, (size_t)(db + i), (NF_GLOBAL NF_HALF*)dst);
        } else {
            ((NF_GLOBAL float*)dst)[db + i] = v;
        }
    }
}

/* LLM_RopeStore: rotate row (t, h) of x as LLM_RoPE_Ext does and store it at the same destination as
 * LLM_NormRopeStore. */
NF_KERNEL void LLM_RopeStore(
    const int n_tokens,
    const int n_heads,
    const int head_dim,
    const int row_stride,
    const int mode,
    const int n_dims,
    const int n_offs,
    const int pos_stride,
    const int has_ff,
    const int x_off,
    const int dst_head_elems,
    const int dst_row_stride,
    const int dst_type,
    const int sect0,
    const int sect1,
    const int sect2,
    const int sect3,
    const float theta_scale,
    const float freq_scale,
    const float ext_factor,
    const float attn_factor,
    const float corr_lo,
    const float corr_hi,
    NF_GLOBAL const int* NF_RESTRICT pos,
    NF_GLOBAL const float* NF_RESTRICT ff,
    NF_GLOBAL const float* x,
    NF_GLOBAL const int* NF_RESTRICT cell_idx,
    NF_GLOBAL uchar* dst)
{
    const nf_i64 grp = (nf_i64)get_group_id(0);
    if (grp >= (nf_i64)n_tokens * n_heads) {
        return;
    }
    const int h = (int)(grp % n_heads);
    const int t = (int)(grp / n_heads);
    const int cell = cell_idx[t];
    if (cell < 0) {
        return;
    }
    const int lid = get_local_id(0);
    const int lsize = get_local_size(0);

    const nf_i64 xb = (nf_i64)x_off + (nf_i64)t * row_stride + (nf_i64)h * head_dim;
    const nf_i64 db = (nf_i64)cell * dst_row_stride + (nf_i64)h * dst_head_elems;
    const int n_offs_end = n_offs + (mode == NF_ROPE_VISION ? 2 * n_dims : n_dims);
    const int span = nf_rope_span(mode, n_dims);
    for (int i = lid; i < head_dim; i += lsize) {
        float v = x[xb + i];
        if (i >= n_offs && i < n_offs_end) {
            const int rel = i - n_offs;
            int pair;
            int second;
            if (mode == NF_ROPE_NORM) {
                pair = rel >> 1;
                second = rel & 1;
            } else if (mode == NF_ROPE_VISION) {
                pair = rel < n_dims ? rel : rel - n_dims;
                second = rel >= n_dims ? 1 : 0;
            } else {
                const int half_dims = n_dims >> 1;
                pair = rel < half_dims ? rel : rel - half_dims;
                second = rel >= half_dims ? 1 : 0;
            }
            const int sec = nf_rope_section(mode, pair, sect0, sect1, sect2, sect3);
            const int e = nf_rope_theta_exp(mode, pair, sec, sect0, sect1, sect2, sect3);
            const int p = pos[t + sec * pos_stride];
            const float f = has_ff != 0 ? ff[pair] : 1.0f;
            const float2 cs = nf_rope_cs(p, e, theta_scale, f, freq_scale, ext_factor, attn_factor, corr_lo,
                                         corr_hi, pair);
            const float other = x[xb + (second != 0 ? i - span : i + span)];
            v = second != 0 ? other * cs.y + v * cs.x : v * cs.x - other * cs.y;
        }
        if (dst_type == 1) {
            vstore_half(v, (size_t)(db + i), (NF_GLOBAL NF_HALF*)dst);
        } else {
            ((NF_GLOBAL float*)dst)[db + i] = v;
        }
    }
}

/* rope.cl - LLM_RoPE_Ext rotary position embedding. Requires: common/dialect.h, common/config.h, common/math.h, common/rope_common.h */
/*
 * LLM_RoPE_Ext(i n_tokens, i n_heads, i head_stride, i row_stride, i n_dims, i n_offs, i mode, i pos_stride,
 *              i has_ff, f theta_scale, f freq_scale, f ext_factor, f attn_factor, f corr_lo, f corr_hi,
 *              i sect0, i sect1, i sect2, i sect3, i x_off, R<int> pos, R ff, RWn x)
 *
 * In-place ggml_rope_ext / ggml_rope_multi (llama.cpp ggml/src/ggml-cpu/ops.cpp:6368-6500, ggml-cuda/rope.cu) of
 * n_tokens rows of n_heads heads: head h of token t starts at x[x_off + t*row_stride + h*head_stride], and
 * dims [n_offs, n_offs + n_dims) of each head rotate (VISION: dims [0, 2*n_dims), n_offs == 0); all other
 * dims are untouched. mode is a ggml rope mode (common/rope_common.h); pair layouts, section selection and
 * angles are those of rope_common.h. Positions: NORM/NEOX read pos[t]; MROPE/IMROPE/VISION read section
 * k of token t at pos[t + k*pos_stride] (pos_stride = n_tokens of the step, layout of llama-batch.cpp:146).
 * has_ff != 0 divides each pair's angle by ff[i] (rope_freqs / long / short factors); otherwise ff is a
 * placeholder and never read. theta_scale = freq_base^(-2/n_dims); corr_lo/corr_hi are
 * ggml_rope_yarn_corr_dims (ggml.c:4496).
 *
 * Launch: 1-D, one work-item per rotation pair, work-item g = (t*n_heads + h)*n_pairs + i with
 * n_pairs = n_dims/2 (VISION: n_dims); global = roundUp(n_tokens*n_heads*n_pairs, L), local = L (any).
 * No barriers. Each pair is read and written by one work-item only, so x is safely updated in place.
 * GLSL twin: shaders/llm/rope.comp.
 */

NF_KERNEL void LLM_RoPE_Ext(
    const int n_tokens,
    const int n_heads,
    const int head_stride,
    const int row_stride,
    const int n_dims,
    const int n_offs,
    const int mode,
    const int pos_stride,
    const int has_ff,
    const float theta_scale,
    const float freq_scale,
    const float ext_factor,
    const float attn_factor,
    const float corr_lo,
    const float corr_hi,
    const int sect0,
    const int sect1,
    const int sect2,
    const int sect3,
    const int x_off,
    NF_GLOBAL const int* NF_RESTRICT pos,
    NF_GLOBAL const float* NF_RESTRICT ff,
    NF_GLOBAL float* x)
{
    const int n_pairs = nf_rope_pairs(mode, n_dims);
    const nf_i64 gid = (nf_i64)get_global_id(0);
    if (n_pairs <= 0 || gid >= (nf_i64)n_tokens * n_heads * n_pairs) return;

    const int i = (int)(gid % n_pairs);
    const nf_i64 th = gid / n_pairs;
    const int h = (int)(th % n_heads);
    const int t = (int)(th / n_heads);

    const int sec = nf_rope_section(mode, i, sect0, sect1, sect2, sect3);
    const int e = nf_rope_theta_exp(mode, i, sec, sect0, sect1, sect2, sect3);
    const int p = pos[t + sec * pos_stride];
    const float f = has_ff ? ff[i] : 1.0f;
    const float2 cs = nf_rope_cs(p, e, theta_scale, f, freq_scale, ext_factor, attn_factor, corr_lo, corr_hi, i);

    const nf_i64 a = (nf_i64)x_off + (nf_i64)t * row_stride + (nf_i64)h * head_stride + n_offs
                     + nf_rope_first(mode, i);
    const nf_i64 b = a + nf_rope_span(mode, n_dims);
    const float x0 = x[a];
    const float x1 = x[b];
    x[a] = x0 * cs.x - x1 * cs.y;
    x[b] = x0 * cs.y + x1 * cs.x;
}

/*
 * LLM_RoPE_FoPE(i n_tokens, i n_heads, i head_stride, i row_stride, i n_dims, i x_off,
 *               i n_inv, i n_coef_rows, i sep_head,
 *               R<int> pos, R cos_coef, R sin_coef, R inv_freq, RWn x)
 *
 * Fourier position embedding (vLLM FourierRotaryEmbedding._compute_cos_sin_cache, rotary_embedding/fope.py):
 * a NEOX rotation over the whole head whose cos/sin of pair i are the learned projections
 *   cos_i = sum_d cos(pos * inv_freq[d]) * cos_coef[row, d, i],  sin_i = ... (sin_coef)
 * for i < n_inv, and (1, 1) for the remaining pairs up to n_dims / 2. The coefficient matrix is stored in ggml ne
 * order [n_inv, n_inv, n_coef_rows], so its index is row * n_inv^2 + d * n_inv + i. Head h reads row h / group with
 * group = n_heads / n_coef_rows when sep_head, else row 0.
 *
 * Launch: 1-D, one work-item per (token, head), g = t * n_heads + h; global = roundUp(n_tokens*n_heads, L),
 * local = L (any). No barriers; every pair of a head is owned by one work-item.
 * GLSL twin: shaders/llm/fope_rope.comp.
 */
NF_KERNEL void LLM_RoPE_FoPE(
    const int n_tokens,
    const int n_heads,
    const int head_stride,
    const int row_stride,
    const int n_dims,
    const int x_off,
    const int n_inv,
    const int n_coef_rows,
    const int sep_head,
    NF_GLOBAL const int* NF_RESTRICT pos,
    NF_GLOBAL const float* NF_RESTRICT cos_coef,
    NF_GLOBAL const float* NF_RESTRICT sin_coef,
    NF_GLOBAL const float* NF_RESTRICT inv_freq,
    NF_GLOBAL float* x)
{
    const int pairs = n_dims >> 1;
    const nf_i64 gid = (nf_i64)get_global_id(0);
    if (pairs <= 0 || n_inv <= 0 || n_inv > pairs || n_coef_rows <= 0
            || gid >= (nf_i64)n_tokens * n_heads) {
        return;
    }
    const int h = (int)(gid % n_heads);
    const int t = (int)(gid / n_heads);
    const int group = sep_head ? (n_heads / n_coef_rows) : n_heads;
    const int row = sep_head ? (h / group) : 0;
    const nf_i64 row_off = (nf_i64)row * n_inv * n_inv;
    const int p = pos[t];
    const nf_i64 base = (nf_i64)x_off + (nf_i64)t * row_stride + (nf_i64)h * head_stride;
    for (int i = 0; i < pairs; i++) {
        float c = 1.0f;
        float s = 1.0f;
        if (i < n_inv) {
            c = 0.0f;
            s = 0.0f;
            for (int d = 0; d < n_inv; d++) {
                const float angle = (float)p * inv_freq[d];
                const nf_i64 off = row_off + (nf_i64)d * n_inv + i;
                c += cos(angle) * cos_coef[off];
                s += sin(angle) * sin_coef[off];
            }
        }
        const nf_i64 a = base + i;
        const nf_i64 b = a + pairs;
        const float x0 = x[a];
        const float x1 = x[b];
        x[a] = x0 * c - x1 * s;
        x[b] = x0 * s + x1 * c;
    }
}

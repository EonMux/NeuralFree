/* kv_cache.cl - LLM_KV_Store_<KV> and LLM_KV_RopeShift_<KV> KV cache kernels. Requires: common/dialect.h, common/config.h, common/math.h, common/quant_blocks.h, common/quant_decode_legacy.h, common/rope_common.h, common/kv_types.h, common/hadamard.h */
/*
 * KV rows are row_elems (store) or row_stride (shift) elements apart; element e of row r of a cache is cache
 * element r*dst_row_stride + e in the layout of common/kv_types.h (F32, F16 and BF16 arrays; the 32-value
 * ggml blocks of Q8_0, Q4_0, Q4_1, IQ4_NL, Q5_0 and Q5_1). Block-type row strides, head strides and store
 * widths are multiples of 32 (validated on the host), so every row and every head starts on a block boundary.
 *
 * Stores convert F16 and BF16 with round to nearest even and quantize block types with the ggml reference
 * quantizers (common/kv_types.h; ggml_set_rows converts with the type's from_float, llama.cpp
 * ggml-cpu/ops.cpp:5530-5598). Dequantization is that of dequantize_row_* (llama.cpp ggml-quants.c).
 *
 * GLSL twins: shaders/llm/kv_store.comp (LLM_KV_Store_*) and shaders/llm/kv_shift.comp (LLM_KV_RopeShift_*).
 */

/* ---------------------------------------------------------------------------------------------------
 * LLM_KV_Store_<KV>(i n_tokens, i row_elems, i src_row_stride, i src_off, i dst_row_stride,
 *                   R src, R<int> cell_idx, W<KV> dst)
 *
 * ggml_set_rows (llama-kv-cache.cpp:1353 cpy_k / cpy_v): row cell_idx[t] of dst <- elements
 * [0, row_elems) of src row t, which starts at src[src_off + t*src_row_stride]. A negative cell index
 * skips the token. Launch: 1-D; F32/F16/BF16 one work-item per element, g = t*row_elems + e, global =
 * roundUp(n_tokens*row_elems, L); block types one work-item per 32-element block, g = t*(row_elems/32) + b,
 * global = roundUp(n_tokens*row_elems/32, L); local = L (any). No barriers.
 * --------------------------------------------------------------------------------------------------- */

NF_KERNEL void LLM_KV_Store_F32(
    const int n_tokens,
    const int row_elems,
    const int src_row_stride,
    const int src_off,
    const int dst_row_stride,
    NF_GLOBAL const float* NF_RESTRICT src,
    NF_GLOBAL const int* NF_RESTRICT cell_idx,
    NF_GLOBAL float* NF_RESTRICT dst)
{
    const nf_i64 gid = (nf_i64)get_global_id(0);
    if (row_elems <= 0 || gid >= (nf_i64)n_tokens * row_elems) return;
    const int t = (int)(gid / row_elems);
    const int e = (int)(gid - (nf_i64)t * row_elems);
    const int cell = cell_idx[t];
    if (cell < 0) return;
    dst[(nf_i64)cell * dst_row_stride + e] = src[(nf_i64)src_off + (nf_i64)t * src_row_stride + e];
}

NF_KERNEL void LLM_KV_Store_F16(
    const int n_tokens,
    const int row_elems,
    const int src_row_stride,
    const int src_off,
    const int dst_row_stride,
    NF_GLOBAL const float* NF_RESTRICT src,
    NF_GLOBAL const int* NF_RESTRICT cell_idx,
    NF_GLOBAL NF_HALF* NF_RESTRICT dst)
{
    const nf_i64 gid = (nf_i64)get_global_id(0);
    if (row_elems <= 0 || gid >= (nf_i64)n_tokens * row_elems) return;
    const int t = (int)(gid / row_elems);
    const int e = (int)(gid - (nf_i64)t * row_elems);
    const int cell = cell_idx[t];
    if (cell < 0) return;
    const float v = src[(nf_i64)src_off + (nf_i64)t * src_row_stride + e];
    vstore_half(v, (size_t)((nf_i64)cell * dst_row_stride + e), dst);
}

NF_KERNEL void LLM_KV_Store_BF16(
    const int n_tokens,
    const int row_elems,
    const int src_row_stride,
    const int src_off,
    const int dst_row_stride,
    NF_GLOBAL const float* NF_RESTRICT src,
    NF_GLOBAL const int* NF_RESTRICT cell_idx,
    NF_GLOBAL ushort* NF_RESTRICT dst)
{
    const nf_i64 gid = (nf_i64)get_global_id(0);
    if (row_elems <= 0 || gid >= (nf_i64)n_tokens * row_elems) return;
    const int t = (int)(gid / row_elems);
    const int e = (int)(gid - (nf_i64)t * row_elems);
    const int cell = cell_idx[t];
    if (cell < 0) return;
    dst[(nf_i64)cell * dst_row_stride + e] = nf_f32_to_bf16(src[(nf_i64)src_off + (nf_i64)t * src_row_stride + e]);
}

/* Block-type store: one work-item quantizes one 32-value block of one token row. */
NF_INLINE void nf_kv_store_blocks(
    const int n_tokens, const int row_elems, const int src_row_stride, const int src_off, const int dst_row_stride,
    NF_GLOBAL const float* src, NF_GLOBAL const int* cell_idx, NF_GLOBAL uchar* dst, const int kv_type)
{
    const int nb = row_elems / NF_KV_BLOCK;
    const nf_i64 gid = (nf_i64)get_global_id(0);
    if (nb <= 0 || gid >= (nf_i64)n_tokens * nb) return;
    const int t = (int)(gid / nb);
    const int b = (int)(gid - (nf_i64)t * nb);
    const int cell = cell_idx[t];
    if (cell < 0) return;
    NF_GLOBAL const float* xs = src + (nf_i64)src_off + (nf_i64)t * src_row_stride + (nf_i64)b * NF_KV_BLOCK;
    nf_kv_store_global_block(dst, (nf_i64)cell * dst_row_stride + (nf_i64)b * NF_KV_BLOCK, xs, kv_type);
}

#define NF_KV_STORE_BLOCK_KERNEL(SFX, KV_CODE)                                                                  \
NF_KERNEL void LLM_KV_Store_##SFX(                                                                              \
    const int n_tokens,                                                                                         \
    const int row_elems,                                                                                        \
    const int src_row_stride,                                                                                   \
    const int src_off,                                                                                          \
    const int dst_row_stride,                                                                                   \
    NF_GLOBAL const float* NF_RESTRICT src,                                                                     \
    NF_GLOBAL const int* NF_RESTRICT cell_idx,                                                                  \
    NF_GLOBAL uchar* NF_RESTRICT dst)                                                                           \
{                                                                                                               \
    nf_kv_store_blocks(n_tokens, row_elems, src_row_stride, src_off, dst_row_stride, src, cell_idx, dst,        \
                       KV_CODE);                                                                                \
}

NF_KV_STORE_BLOCK_KERNEL(Q8_0, NF_KV_Q8_0)
NF_KV_STORE_BLOCK_KERNEL(Q4_0, NF_KV_Q4_0)
NF_KV_STORE_BLOCK_KERNEL(Q4_1, NF_KV_Q4_1)
NF_KV_STORE_BLOCK_KERNEL(IQ4_NL, NF_KV_IQ4_NL)
NF_KV_STORE_BLOCK_KERNEL(Q5_0, NF_KV_Q5_0)
NF_KV_STORE_BLOCK_KERNEL(Q5_1, NF_KV_Q5_1)


/* ---------------------------------------------------------------------------------------------------
 * LLM_KV_RopeShift_F32 | _F16 | _BF16(i n_cells, i n_heads, i head_stride, i row_stride, i n_dims, i n_offs,
 *                            i mode, i has_ff, f theta_scale, f freq_scale, f ext_factor, f attn_factor,
 *                            f corr_lo, f corr_hi, R<int> delta, R ff, RWn<KV> k_cache)
 * LLM_KV_RopeShift_<block type>(the same scalars, then i rot_k; the same buffers)
 *
 * Context shift of a K cache (llama-kv-cache.cpp:1945-1993 build_rope_shift): every cell c rotates dims
 * [n_offs, n_offs + n_dims) of each head h (row c at c*row_stride, head at h*head_stride) by RoPE at position
 * delta[c]. Cells with delta 0 go through the same RoPE, as llama.cpp shifts every cell of the cache with its
 * per-cell shift (0 for unshifted and empty cells, llama-kv-cache.cpp:1545-1556): the rotation is the identity and
 * only the YaRN magnitude scale (attn_factor, and the ext_factor mscale term) applies. mode 0 (NORM) rotates
 * (2i, 2i+1) pairs; every other mode rotates NEOX pairs (i, i + n_dims/2) with the single position delta[c]
 * and exponent i, as llama.cpp shifts MROPE/IMROPE caches with NEOX (llama-kv-cache.cpp:1962-1969).
 * has_ff != 0 divides pair i's angle by ff[i]; otherwise ff is a placeholder and never read.
 *
 * F32/F16/BF16 (rotated in place, ggml_rope_ext_inplace; F16 and BF16 round the results to nearest even) launch:
 * 1-D, one work-item per pair, g = (c*n_heads + h)*(n_dims/2) + i; global = roundUp(n_cells*n_heads*n_dims/2, L),
 * local = L (any). No barriers.
 *
 * Block types (dequantize, rotate, requantize: the ggml_is_quantized branch of build_rope_shift): one workgroup
 * per (cell, head), group g = c*n_heads + h;
 * global = n_cells*n_heads*L, local = L (a power of two <= LLM_DIMX; CUDA: a multiple of 32). The work window
 * of a head is [w0, w1) = the rotated dims [n_offs, n_offs + n_dims) widened outwards to multiples of the
 * granule G (nf_rope_shift_w0 / nf_rope_shift_w1): G = rot_k when rot_k > 0, else the 32-element block. The
 * window is dequantized into local memory, rotated, and requantized block by block with the reference quantizer of
 * the type; blocks outside the window keep their stored bits. rot_k > 0 states that
 * the cache holds K multiplied by the normalized Hadamard matrix in chunks of rot_k from the head start
 * (quantized-KV rotation, llama-kv-cache.cpp:315-361); the window is then un-rotated before and re-rotated after
 * RoPE, as build_rope_shift does with k_rot (H*H = I). Host preconditions: head_stride and row_stride multiples
 * of 32; rot_k == 0 or a power of two >= 32; w1 <= head_stride; w1 - w0 <= NF_ROPE_SHIFT_MAX_DIMS.
 * --------------------------------------------------------------------------------------------------- */

/* Element index of the first element of shift pair i from the first rotated dim (NORM or NEOX layout). */
NF_INLINE int nf_rope_shift_first(int mode, int i) {
    return nf_rope_first(mode == NF_ROPE_NORM ? NF_ROPE_NORM : NF_ROPE_NEOX, i);
}

/* Distance between the two elements of a shift pair (NORM or NEOX layout). */
NF_INLINE int nf_rope_shift_span(int mode, int n_dims) {
    return nf_rope_span(mode == NF_ROPE_NORM ? NF_ROPE_NORM : NF_ROPE_NEOX, n_dims);
}

/* Granule of the block-type shift window: the Hadamard chunk when the cache is rotated, else the 32-value block. */
NF_INLINE int nf_rope_shift_granule(int rot_k) {
    return rot_k > 0 ? rot_k : NF_KV_BLOCK;
}

/* Start w0 of the block-type shift window: n_offs rounded down to the granule (Java: RopeKernels.shiftWindowStart). */
NF_INLINE int nf_rope_shift_w0(int n_offs, int rot_k) {
    const int g = nf_rope_shift_granule(rot_k);
    return (n_offs / g) * g;
}

/* End w1 of the block-type shift window: n_offs + n_dims rounded up to the granule (RopeKernels.shiftWindowEnd). */
NF_INLINE int nf_rope_shift_w1(int n_dims, int n_offs, int rot_k) {
    const int g = nf_rope_shift_granule(rot_k);
    return ((n_offs + n_dims + g - 1) / g) * g;
}

NF_KERNEL void LLM_KV_RopeShift_F32(
    const int n_cells,
    const int n_heads,
    const int head_stride,
    const int row_stride,
    const int n_dims,
    const int n_offs,
    const int mode,
    const int has_ff,
    const float theta_scale,
    const float freq_scale,
    const float ext_factor,
    const float attn_factor,
    const float corr_lo,
    const float corr_hi,
    NF_GLOBAL const int* NF_RESTRICT delta,
    NF_GLOBAL const float* NF_RESTRICT ff,
    NF_GLOBAL float* k_cache)
{
    const int n_pairs = n_dims >> 1;
    const nf_i64 gid = (nf_i64)get_global_id(0);
    if (n_pairs <= 0 || gid >= (nf_i64)n_cells * n_heads * n_pairs) return;
    const int i = (int)(gid % n_pairs);
    const nf_i64 ch = gid / n_pairs;
    const int h = (int)(ch % n_heads);
    const int c = (int)(ch / n_heads);
    const int p = delta[c];

    const float f = has_ff ? ff[i] : 1.0f;
    const float2 cs = nf_rope_cs(p, i, theta_scale, f, freq_scale, ext_factor, attn_factor, corr_lo, corr_hi, i);
    const nf_i64 a = (nf_i64)c * row_stride + (nf_i64)h * head_stride + n_offs + nf_rope_shift_first(mode, i);
    const nf_i64 b = a + nf_rope_shift_span(mode, n_dims);
    const float x0 = k_cache[a];
    const float x1 = k_cache[b];
    k_cache[a] = x0 * cs.x - x1 * cs.y;
    k_cache[b] = x0 * cs.y + x1 * cs.x;
}

NF_KERNEL void LLM_KV_RopeShift_F16(
    const int n_cells,
    const int n_heads,
    const int head_stride,
    const int row_stride,
    const int n_dims,
    const int n_offs,
    const int mode,
    const int has_ff,
    const float theta_scale,
    const float freq_scale,
    const float ext_factor,
    const float attn_factor,
    const float corr_lo,
    const float corr_hi,
    NF_GLOBAL const int* NF_RESTRICT delta,
    NF_GLOBAL const float* NF_RESTRICT ff,
    NF_GLOBAL NF_HALF* k_cache)
{
    const int n_pairs = n_dims >> 1;
    const nf_i64 gid = (nf_i64)get_global_id(0);
    if (n_pairs <= 0 || gid >= (nf_i64)n_cells * n_heads * n_pairs) return;
    const int i = (int)(gid % n_pairs);
    const nf_i64 ch = gid / n_pairs;
    const int h = (int)(ch % n_heads);
    const int c = (int)(ch / n_heads);
    const int p = delta[c];

    const float f = has_ff ? ff[i] : 1.0f;
    const float2 cs = nf_rope_cs(p, i, theta_scale, f, freq_scale, ext_factor, attn_factor, corr_lo, corr_hi, i);
    const nf_i64 a = (nf_i64)c * row_stride + (nf_i64)h * head_stride + n_offs + nf_rope_shift_first(mode, i);
    const nf_i64 b = a + nf_rope_shift_span(mode, n_dims);
    const float x0 = vload_half((size_t)a, k_cache);
    const float x1 = vload_half((size_t)b, k_cache);
    vstore_half(x0 * cs.x - x1 * cs.y, (size_t)a, k_cache);
    vstore_half(x0 * cs.y + x1 * cs.x, (size_t)b, k_cache);
}

NF_KERNEL void LLM_KV_RopeShift_BF16(
    const int n_cells,
    const int n_heads,
    const int head_stride,
    const int row_stride,
    const int n_dims,
    const int n_offs,
    const int mode,
    const int has_ff,
    const float theta_scale,
    const float freq_scale,
    const float ext_factor,
    const float attn_factor,
    const float corr_lo,
    const float corr_hi,
    NF_GLOBAL const int* NF_RESTRICT delta,
    NF_GLOBAL const float* NF_RESTRICT ff,
    NF_GLOBAL ushort* k_cache)
{
    const int n_pairs = n_dims >> 1;
    const nf_i64 gid = (nf_i64)get_global_id(0);
    if (n_pairs <= 0 || gid >= (nf_i64)n_cells * n_heads * n_pairs) return;
    const int i = (int)(gid % n_pairs);
    const nf_i64 ch = gid / n_pairs;
    const int h = (int)(ch % n_heads);
    const int c = (int)(ch / n_heads);
    const int p = delta[c];

    const float f = has_ff ? ff[i] : 1.0f;
    const float2 cs = nf_rope_cs(p, i, theta_scale, f, freq_scale, ext_factor, attn_factor, corr_lo, corr_hi, i);
    const nf_i64 a = (nf_i64)c * row_stride + (nf_i64)h * head_stride + n_offs + nf_rope_shift_first(mode, i);
    const nf_i64 b = a + nf_rope_shift_span(mode, n_dims);
    const float x0 = bf16_to_f32(k_cache[a]);
    const float x1 = bf16_to_f32(k_cache[b]);
    k_cache[a] = nf_f32_to_bf16(x0 * cs.x - x1 * cs.y);
    k_cache[b] = nf_f32_to_bf16(x0 * cs.y + x1 * cs.x);
}

/*
 * Block-type shift of one (cell, head) window in the work-group's local buffer buf[NF_ROPE_SHIFT_MAX_DIMS]. Every
 * work-item calls it in uniform control flow (it contains barriers).
 */
NF_INLINE void nf_kv_rope_shift_blocks(
    const int n_cells, const int n_heads, const int head_stride, const int row_stride, const int n_dims,
    const int n_offs, const int mode, const int has_ff, const float theta_scale, const float freq_scale,
    const float ext_factor, const float attn_factor, const float corr_lo, const float corr_hi, const int rot_k,
    NF_GLOBAL const int* delta, NF_GLOBAL const float* ff, NF_GLOBAL uchar* k_cache, const int kv_type,
    NF_LOCAL_PTR float* buf)
{
    const int g = get_group_id(0);
    const int heads = n_heads > 0 ? n_heads : 1;
    const int c = g / heads;
    const int h = g - c * heads;
    const int w0 = nf_rope_shift_w0(n_offs, rot_k);
    const int width = nf_rope_shift_w1(n_dims, n_offs, rot_k) - w0;
    /* Uniform per workgroup: a dead group leaves before any barrier. */
    if (c >= n_cells || n_dims <= 0 || width > NF_ROPE_SHIFT_MAX_DIMS) return;
    const int p = delta[c];

    const int lid = get_local_id(0);
    const int lsize = get_local_size(0);
    /* First element of the window (row, head and w0 are multiples of 32); rotated dims start at buf[r0]. */
    const nf_i64 e0 = (nf_i64)c * row_stride + (nf_i64)h * head_stride + w0;
    const int r0 = n_offs - w0;

    for (int j = lid; j < width; j += lsize) {
        buf[j] = nf_kv_load1(k_cache, e0 + j, kv_type);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (rot_k > 1) {
        nf_fwht_local(buf, width, rot_k);
    }

    for (int i = lid; i < (n_dims >> 1); i += lsize) {
        const float f = has_ff ? ff[i] : 1.0f;
        const float2 cs = nf_rope_cs(p, i, theta_scale, f, freq_scale, ext_factor, attn_factor, corr_lo, corr_hi, i);
        const int a = r0 + nf_rope_shift_first(mode, i);
        const int b = a + nf_rope_shift_span(mode, n_dims);
        const float x0 = buf[a];
        const float x1 = buf[b];
        buf[a] = x0 * cs.x - x1 * cs.y;
        buf[b] = x0 * cs.y + x1 * cs.x;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (rot_k > 1) {
        nf_fwht_local(buf, width, rot_k);
    }

    nf_kv_store_local_row(k_cache, e0, buf, width, kv_type);
}

#define NF_KV_SHIFT_BLOCK_KERNEL(SFX, KV_CODE)                                                                  \
NF_KERNEL void LLM_KV_RopeShift_##SFX(                                                                          \
    const int n_cells,                                                                                          \
    const int n_heads,                                                                                          \
    const int head_stride,                                                                                      \
    const int row_stride,                                                                                       \
    const int n_dims,                                                                                           \
    const int n_offs,                                                                                           \
    const int mode,                                                                                             \
    const int has_ff,                                                                                           \
    const float theta_scale,                                                                                    \
    const float freq_scale,                                                                                     \
    const float ext_factor,                                                                                     \
    const float attn_factor,                                                                                    \
    const float corr_lo,                                                                                        \
    const float corr_hi,                                                                                        \
    const int rot_k,                                                                                            \
    NF_GLOBAL const int* NF_RESTRICT delta,                                                                     \
    NF_GLOBAL const float* NF_RESTRICT ff,                                                                      \
    NF_GLOBAL uchar* k_cache)                                                                                   \
{                                                                                                               \
    NF_LOCAL float buf[NF_ROPE_SHIFT_MAX_DIMS];                                                                 \
    nf_kv_rope_shift_blocks(n_cells, n_heads, head_stride, row_stride, n_dims, n_offs, mode, has_ff,           \
                            theta_scale, freq_scale, ext_factor, attn_factor, corr_lo, corr_hi, rot_k, delta,   \
                            ff, k_cache, KV_CODE, buf);                                                         \
}

NF_KV_SHIFT_BLOCK_KERNEL(Q8_0, NF_KV_Q8_0)
NF_KV_SHIFT_BLOCK_KERNEL(Q4_0, NF_KV_Q4_0)
NF_KV_SHIFT_BLOCK_KERNEL(Q4_1, NF_KV_Q4_1)
NF_KV_SHIFT_BLOCK_KERNEL(IQ4_NL, NF_KV_IQ4_NL)
NF_KV_SHIFT_BLOCK_KERNEL(Q5_0, NF_KV_Q5_0)
NF_KV_SHIFT_BLOCK_KERNEL(Q5_1, NF_KV_Q5_1)

/* hadamard.cl - LLM_HadamardRotate: Walsh-Hadamard rotation of quantized-KV and PRISM rows. Requires: common/dialect.h, common/config.h, common/hadamard.h */
/*
 * LLM_HadamardRotate(i n_rows, i width, i block, i has_signs, i perm_hd, i perm_nk, i perm_rep,
 *                    i x_off, i y_off, R signs, Rn x, Wn y)
 *
 * Rows r in [0, n_rows) of width elements start at x + x_off + r*width and y + y_off + r*width.
 * Each row is split into width / block blocks; block b of output row r is
 *     y[o] = scale(o, after) * (1/sqrt(block)) * sum_j (-1)^popcount(i & j) * v[base + j],
 *     v[o] = scale(o, before) * x[src(o)],  o = base + i, base = b * block, i, j in [0, block)
 * i.e. the orthonormal Sylvester Hadamard matrix of llama.cpp (llama-model.cpp:2300-2312 and
 * llama-kv-cache.cpp:23 ggml_gen_hadamard; H*H = I) applied by an in-place fast Walsh-Hadamard
 * transform in local memory.
 *
 * has_signs selects the +/-1 sign vector signs[0 .. width):
 *   NF_HD_SIGNS_NONE   (0) no signs (signs is a placeholder and never read)
 *   NF_HD_SIGNS_BEFORE (1) scale(o, before) = signs[o]: the folded-weight input transform
 *                          y = H * (D * x) of llama-graph.cpp:1620-1624
 *   NF_HD_SIGNS_AFTER  (2) scale(o, after) = signs[o]: the inverse-after-lookup transform
 *                          y = D * (H * z) of llama-graph.cpp:2685-2690
 * perm_rep > 1 gathers the input from tiled head order [perm_hd, perm_nk, perm_rep] into grouped order
 * [perm_hd, perm_rep, perm_nk] before the signs (llama-graph.cpp:1610-1618): for
 * o = h + perm_hd*(r + perm_rep*k), src(o) = h + perm_hd*(k + perm_nk*r). Otherwise src(o) = o.
 *
 * Preconditions (validated on the host): block is a power of two <= NF_HD_MAX_BLOCK, width % block == 0,
 * perm_hd*perm_nk*perm_rep == width when perm_rep > 1. y may alias x only in place (same buffer,
 * x_off == y_off) and only when perm_rep <= 1: every output block then reads only its own input block,
 * before the first barrier.
 *
 * Launch: 1-D, one workgroup per (row, block) pair, group g = r*(width/block) + b; global =
 * n_rows*(width/block)*L, local = L (a power of two <= LLM_DIMX; CUDA: a multiple of 32). Work-items
 * stride over the block, so any such L serves any block size.
 */

#define NF_HD_MAX_BLOCK    1024
#define NF_HD_SIGNS_NONE   0
#define NF_HD_SIGNS_BEFORE 1
#define NF_HD_SIGNS_AFTER  2

/* Input element of grouped position o (llama-graph.cpp:1610-1618 permute(0, 2, 1, 3)). */
NF_INLINE int nf_hd_src(int o, int perm_hd, int perm_nk, int perm_rep) {
    if (perm_rep <= 1) {
        return o;
    }
    const int h = o % perm_hd;
    const int t = o / perm_hd;
    const int r = t % perm_rep;
    const int k = t / perm_rep;
    return h + perm_hd * (k + perm_nk * r);
}

NF_KERNEL void LLM_HadamardRotate(
    const int n_rows, const int width, const int block, const int has_signs,
    const int perm_hd, const int perm_nk, const int perm_rep, const int x_off, const int y_off,
    NF_GLOBAL const float* NF_RESTRICT signs,
    NF_GLOBAL const float* x,
    NF_GLOBAL float* y) {
    NF_LOCAL float buf[NF_HD_MAX_BLOCK];
    const int nblk = width / block;
    const int g = get_group_id(0);
    const int row = g / nblk;
    /* Uniform per workgroup: every work-item of a dead group leaves before any barrier. */
    if (row >= n_rows) return;
    const int base = (g - row * nblk) * block;
    const int lid = get_local_id(0), lsize = get_local_size(0);
    const nf_i64 xrow = (nf_i64)x_off + (nf_i64)row * width;
    const nf_i64 yrow = (nf_i64)y_off + (nf_i64)row * width;

    for (int i = lid; i < block; i += lsize) {
        const int o = base + i;
        float v = x[xrow + nf_hd_src(o, perm_hd, perm_nk, perm_rep)];
        if (has_signs == NF_HD_SIGNS_BEFORE) v *= signs[o];
        buf[i] = v;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    nf_fwht_local(buf, block, block);

    for (int i = lid; i < block; i += lsize) {
        float v = buf[i];
        if (has_signs == NF_HD_SIGNS_AFTER) v *= signs[base + i];
        y[yrow + base + i] = v;
    }
}

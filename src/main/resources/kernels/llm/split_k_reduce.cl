/* split_k_reduce.cl - the split-K reduce: sums the slice planes of a split-K GEMM into the strided
 * output (llama.cpp mul_mat_split_k_reduce.comp). Requires: common/dialect.h, common/config.h */
/*
 * LLM_SplitK_Reduce(i ne, i splits, i M, i y_stride, i y_off, R float partial, W float y)
 *   y[y_off + (e / M) * y_stride + (e % M)] = sum_{s < splits} partial[s * ne + e],  e < ne
 *
 * `partial` holds `splits` consecutive planes of ne = M * N floats, laid out by the KSplit GEMM
 * kernels (plane element (t, r) at plane_start + t * M + r); the reduce scatters each summed element
 * to its row of the caller's strided output. Element-wise, so it serves every split-K GEMM form.
 *
 * Launch contract: 1-D, global = roundUp(ceil(ne / 4), L) work-items of local size L; each handles
 * four consecutive elements (as llama.cpp's vec4 reduce), guarded past ne.
 */
#define NF_SK_RED 4                     /* elements per work-item */

NF_KERNEL void LLM_SplitK_Reduce(
    const int ne, const int splits, const int M,
    const int y_stride, const int y_off,
    NF_GLOBAL const float* NF_RESTRICT partial,
    NF_GLOBAL float* NF_RESTRICT y) {
    const nf_i64 e0 = (nf_i64)get_global_id(0) * NF_SK_RED;
    for (int j = 0; j < NF_SK_RED; j++) {
        const nf_i64 e = e0 + j;
        if (e >= ne) return;
        float sum = 0.0f;
        for (int s = 0; s < splits; s++) {
            sum += partial[(nf_i64)s * ne + e];
        }
        y[(nf_i64)y_off + (e / M) * y_stride + (e % M)] = sum;
    }
}

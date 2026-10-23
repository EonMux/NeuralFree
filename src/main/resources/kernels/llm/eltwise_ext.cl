/* eltwise_ext.cl - LLM_Copy2D_F32 and LLM_Axpby_F32 offset-aware generic ops used by GpuOps. Requires: common/dialect.h, common/config.h */
/*
 * Generic element-wise helpers over offset views (offsets and strides in float elements). GpuOps
 * stages activations through them for kernels without offset arguments and uses them for residual
 * adds, scaled accumulation and per-tensor output scales.
 *
 * LLM_Copy2D_F32(i rows, i cols, i src_stride, i src_off, i dst_stride, i dst_off, R src, W dst)
 *     dst[dst_off + r*dst_stride + c] = src[src_off + r*src_stride + c],  r < rows, c < cols.
 *     src and dst are restrict: the copied element ranges must not overlap. They may lie in one
 *     buffer at disjoint offsets.
 *
 * LLM_Axpby_F32(i n, f alpha, f beta, i a_off, i b_off, i out_off, R a, Rn b, Wn out)
 *     out[out_off + i] = A + B,  i < n,  A = alpha != 0 ? alpha*a[a_off + i] : 0,
 *                                       B = beta  != 0 ? beta*b[b_off + i]  : 0.
 *     A zero coefficient skips the read of its operand (the BLAS beta == 0 convention), so a
 *     non-finite value there does not propagate and the operand may be a placeholder buffer.
 *     out may alias b element for element (same buffer, out_off == b_off): each work-item reads its
 *     own b element before writing it. a must not overlap out.
 *
 * Dispatch: 1-D, one work-item per element, global = roundUp(n, L) (Copy2D: n = rows*cols),
 * local = {L}. The host keeps n below 2^31; addresses are formed in nf_i64. No barriers.
 */

NF_KERNEL void LLM_Copy2D_F32(
    const int rows,
    const int cols,
    const int src_stride,
    const int src_off,
    const int dst_stride,
    const int dst_off,
    NF_GLOBAL const float* NF_RESTRICT src,
    NF_GLOBAL float* NF_RESTRICT dst)
{
    const nf_i64 idx = (nf_i64)get_global_id(0);
    if (cols <= 0 || idx >= (nf_i64)rows * cols) return;
    const nf_i64 r = idx / cols;
    const nf_i64 c = idx - r * cols;
    dst[(nf_i64)dst_off + r * dst_stride + c] = src[(nf_i64)src_off + r * src_stride + c];
}

NF_KERNEL void LLM_Axpby_F32(
    const int n,
    const float alpha,
    const float beta,
    const int a_off,
    const int b_off,
    const int out_off,
    NF_GLOBAL const float* NF_RESTRICT a,
    NF_GLOBAL const float* b,
    NF_GLOBAL float* out)
{
    const nf_i64 i = (nf_i64)get_global_id(0);
    if (i >= (nf_i64)n) return;
    const float av = alpha != 0.0f ? alpha * a[(nf_i64)a_off + i] : 0.0f;
    const float bv = beta != 0.0f ? beta * b[(nf_i64)b_off + i] : 0.0f;
    out[(nf_i64)out_off + i] = av + bv;
}

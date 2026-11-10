/* eltwise_ext.cl - LLM_Copy2D_F32, LLM_Axpby_F32 and LLM_MuddMix offset-aware generic ops. Requires: common/dialect.h, common/config.h */
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
 * LLM_MuddMix(i n_tokens, i n_embd, i n_taps, i bank_off, i bank_stride, i w_off, i taps_off, i cur_off,
 *             i resid_off, R bank, R w, R bias, R<int> taps, W cur, W resid)
 *     h_j = bank[bank_off + taps[taps_off + j]*bank_stride + t*n_embd + i],
 *     cur[cur_off + t*n_embd + i]     = sum_j (w[w_off + t*2*n_taps + j] + bias[j]) * h_j,
 *     resid[resid_off + t*n_embd + i] = sum_j (w[w_off + t*2*n_taps + n_taps + j] + bias[n_taps + j]) * h_j,
 *     j < n_taps, i < n_embd: the MUDD dense cross-layer mix of limite (models/limite.cpp:100-140), with
 *     the mudd_up bias added to the mix weights as they are read. cur and resid must not overlap the inputs
 *     or each other.
 *
 * Dispatch: 1-D, one work-item per element, global = roundUp(n, L) (MuddMix: n = n_tokens*n_embd),
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

NF_KERNEL void LLM_MuddMix(
    const int n_tokens,
    const int n_embd,
    const int n_taps,
    const int bank_off,
    const int bank_stride,
    const int w_off,
    const int taps_off,
    const int cur_off,
    const int resid_off,
    NF_GLOBAL const float* NF_RESTRICT bank,
    NF_GLOBAL const float* NF_RESTRICT w,
    NF_GLOBAL const float* NF_RESTRICT bias,
    NF_GLOBAL const int* NF_RESTRICT taps,
    NF_GLOBAL float* cur,
    NF_GLOBAL float* resid)
{
    const nf_i64 idx = (nf_i64)get_global_id(0);
    if (n_embd <= 0 || idx >= (nf_i64)n_tokens * n_embd) return;
    const nf_i64 t = idx / n_embd;
    const nf_i64 i = idx - t * n_embd;
    const nf_i64 w_base = (nf_i64)w_off + t * (nf_i64)2 * n_taps;
    float c = 0.0f;
    float r = 0.0f;
    for (int j = 0; j < n_taps; j++) {
        const float h = bank[(nf_i64)bank_off + (nf_i64)taps[taps_off + j] * bank_stride + t * n_embd + i];
        c += (w[w_base + j] + bias[j]) * h;
        r += (w[w_base + n_taps + j] + bias[n_taps + j]) * h;
    }
    cur[(nf_i64)cur_off + idx] = c;
    resid[(nf_i64)resid_off + idx] = r;
}

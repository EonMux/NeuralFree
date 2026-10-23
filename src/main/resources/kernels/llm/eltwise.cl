/* eltwise.cl - element-wise add, multiply, scale and copy kernels. Requires: common/dialect.h, common/config.h */
/*
 * One work-item per element (or per four elements for the float4 batch kernels), so the only
 * early returns are tail bounds checks and no kernel here has a barrier.
 *
 * Aliasing: callers run these kernels in place (out == a, dst == src, ...), so activation buffers
 * are never restrict; every work-item reads and writes only its own elements, which makes in-place
 * use exact. Model parameters (bias, pos_embed) and the two copy kernels' distinct src/dst stay
 * restrict.
 *
 * Dispatch: 1-D, global = roundUp(n, L) work-items with n = the element count, or
 * roundUp(ceil(n / 4), L) for the float4 batch kernels; local = {L}.
 */

/* out[i] = a[i] + b[i], i in [0, size). */
NF_KERNEL void LLM_Add(
    const int size,
    NF_GLOBAL const float* a,
    NF_GLOBAL const float* b,
    NF_GLOBAL float* out)
{
    const int i = get_global_id(0);
    if (i >= size) return;
    out[i] = a[i] + b[i];
}

/* dst[i] += bias[i], i in [0, size). */
NF_KERNEL void LLM_BiasAdd(
    const int size,
    NF_GLOBAL const float* NF_RESTRICT bias,
    NF_GLOBAL float* dst)
{
    const int i = get_global_id(0);
    if (i >= size) return;
    dst[i] += bias[i];
}

/* dst[i] += bias[src_offset + i]: one row of a [rows x size] bias table (per-expert biases). */
NF_KERNEL void LLM_BiasAddSlice(
    const int size,
    const int src_offset,
    NF_GLOBAL const float* NF_RESTRICT bias,
    NF_GLOBAL float* dst)
{
    const int i = get_global_id(0);
    if (i >= size) return;
    dst[i] += bias[(nf_i64)src_offset + i];
}

/* dst[dst_offset + i] += bias[i]: bias added to one row of a larger matrix. */
NF_KERNEL void LLM_BiasAddDstOffset(
    const int size,
    const int dst_offset,
    NF_GLOBAL const float* NF_RESTRICT bias,
    NF_GLOBAL float* dst)
{
    const int i = get_global_id(0);
    if (i >= size) return;
    dst[(nf_i64)dst_offset + i] += bias[i];
}

/* dst[i] = src[i], i in [0, size); src and dst are distinct buffers. */
NF_KERNEL void LLM_Copy(
    const int size,
    NF_GLOBAL const float* NF_RESTRICT src,
    NF_GLOBAL float* NF_RESTRICT dst)
{
    const int i = get_global_id(0);
    if (i >= size) return;
    dst[i] = src[i];
}

/* buf[i] = 0, i in [0, size). */
NF_KERNEL void LLM_FillZero(
    const int size,
    NF_GLOBAL float* buf)
{
    const int i = get_global_id(0);
    if (i >= size) return;
    buf[i] = 0.0f;
}

/* out[i] = a[i] + b[i], i in [0, total), four elements per work-item. */
NF_KERNEL void LLM_Batch_Add(
    const int total,
    NF_GLOBAL const float* a,
    NF_GLOBAL const float* b,
    NF_GLOBAL float* out)
{
    const int i = get_global_id(0) * 4;
    if (i >= total) return;
    if (i + 3 < total) {
        vstore4(vload4(0, a + i) + vload4(0, b + i), 0, out + i);
    } else {
        for (int j = i; j < total; j++) {
            out[j] = a[j] + b[j];
        }
    }
}

/* dst[i] = src[i], i in [0, total), four elements per work-item; src and dst are distinct. */
NF_KERNEL void LLM_Batch_Copy(
    const int total,
    NF_GLOBAL const float* NF_RESTRICT src,
    NF_GLOBAL float* NF_RESTRICT dst)
{
    const int i = get_global_id(0) * 4;
    if (i >= total) return;
    if (i + 3 < total) {
        vstore4(vload4(0, src + i), 0, dst + i);
    } else {
        for (int j = i; j < total; j++) {
            dst[j] = src[j];
        }
    }
}

/*
 * data[s * dim + d] += bias[d] for every row s in [0, seq_len). One work-item per column d, which
 * loads bias[d] once. Dispatch: global = roundUp(dim, L).
 */
NF_KERNEL void LLM_Batch_BiasAdd(
    const int dim,
    const int seq_len,
    NF_GLOBAL const float* NF_RESTRICT bias,
    NF_GLOBAL float* data)
{
    const int d = get_global_id(0);
    if (d >= dim) return;
    const float b = bias[d];
    for (int s = 0; s < seq_len; s++) {
        data[(nf_i64)s * dim + d] += b;
    }
}

/* dst[i] += weight * src[i], i in [0, size); src may equal dst. */
NF_KERNEL void LLM_ScaleAdd(
    const int size,
    const float weight,
    NF_GLOBAL const float* src,
    NF_GLOBAL float* dst)
{
    const int i = get_global_id(0);
    if (i >= size) return;
    dst[i] += weight * src[i];
}

/* data[i] *= scale, i in [0, size). */
NF_KERNEL void LLM_EmbedScale(
    const int size,
    const float scale,
    NF_GLOBAL float* data)
{
    const int i = get_global_id(0);
    if (i >= size) return;
    data[i] *= scale;
}

/* logits[i] = cap * tanh(logits[i] / cap), i in [0, size) (final logit soft-capping). */
NF_KERNEL void LLM_LogitSoftcap(
    const int size,
    const float cap,
    NF_GLOBAL float* logits)
{
    const int i = get_global_id(0);
    if (i >= size) return;
    logits[i] = cap * tanh(logits[i] / cap);
}

/* data[i] += pos_embed[pos * size + i]: absolute position embedding of position pos. */
NF_KERNEL void LLM_AddPosEmbed(
    const int size,
    const int pos,
    NF_GLOBAL const float* NF_RESTRICT pos_embed,
    NF_GLOBAL float* data)
{
    const int i = get_global_id(0);
    if (i >= size) return;
    data[i] += pos_embed[(nf_i64)pos * size + i];
}

/* out[i] = a[i] + b[i] + c[i], i in [0, size) (parallel residual). */
NF_KERNEL void LLM_Add3(
    const int size,
    NF_GLOBAL const float* a,
    NF_GLOBAL const float* b,
    NF_GLOBAL const float* c,
    NF_GLOBAL float* out)
{
    const int i = get_global_id(0);
    if (i >= size) return;
    out[i] = a[i] + b[i] + c[i];
}

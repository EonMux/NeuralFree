/* flow.cl - LLM_GatherRows: row selection for output rows (logits heads) and drafter selection. Requires: common/dialect.h, common/config.h */
/*
 * LLM_GatherRows(i n_rows, i row_len, i src_stride, i dst_stride, i src_off, i dst_off, R<int> rows, R src, W dst)
 *     dst[dst_off + r*dst_stride + c] = rows[r] >= 0 ? src[src_off + rows[r]*src_stride + c] : 0
 *     for r < n_rows, c < row_len (ggml_get_rows on F32 rows, ggml-cpu/ops.cpp:5414; used for the
 *     inp_out_ids selection of build_inp_out_ids, llama-graph.cpp:2824, and drafter row selection). A negative
 *     index yields a zero row (padding of a selection list). Indices are not range-checked against the
 *     source: the host builds them from positions it owns. Offsets and strides are in float elements;
 *     addresses are formed in nf_i64. dst must not overlap src.
 *     Dispatch: 1-D, one work-item per element (r*row_len + c), global = roundUp(n_rows*row_len, L),
 *     local = {L}. No barriers.
 * Java: gguf.ops.flow.FlowKernels; CPU twin: gguf.ops.heads.CpuPooling.gatherRows.
 */

NF_KERNEL void LLM_GatherRows(
    const int n_rows,
    const int row_len,
    const int src_stride,
    const int dst_stride,
    const int src_off,
    const int dst_off,
    NF_GLOBAL const int* NF_RESTRICT rows,
    NF_GLOBAL const float* NF_RESTRICT src,
    NF_GLOBAL float* NF_RESTRICT dst)
{
    const nf_i64 idx = (nf_i64)get_global_id(0);
    if (row_len <= 0 || idx >= (nf_i64)n_rows * row_len) return;
    const nf_i64 r = idx / row_len;
    const nf_i64 c = idx - r * row_len;
    const int s = rows[r];
    dst[(nf_i64)dst_off + r * dst_stride + c] = s >= 0 ? src[(nf_i64)src_off + (nf_i64)s * src_stride + c] : 0.0f;
}

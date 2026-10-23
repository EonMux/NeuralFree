/* segment.cl - CSR segment reductions for graph aggregation and their backward. Requires: common/dialect.h, common/train.h */
/*
 * Segment (target node) t owns the edge slots [ptr[ptr_off + t], ptr[ptr_off + t + 1]) of eid; slot e names
 * the source row eid[eid_off + e] of src, a [rows, feat] block at src_off. The host builds ptr and eid once
 * per graph (target-sorted, stable by edge id) as INT32 tensors, so every row index is in range and ptr is
 * non-decreasing. Offsets are element offsets.
 *
 * Ops (CPU reference tensor/CpuIndexing.java segmentReduce; callers nn/graph/MessagePassing.java scatterAdd,
 * scatterMean and scatterMax, GATConv.java, GlobalMeanPool.java and GlobalMaxPool.java forward):
 *   TR_SEG_SUM  0  sum of the source rows, ascending slot order
 *   TR_SEG_MEAN 1  the same sum, then one division by the slot count
 *   TR_SEG_MAX  2  Java Math.max fold from -inf (tr_jmax; NaN propagates, +0 above -0)
 *
 * TR_SegmentReduce_F32(i op, i nodes, i feat, f empty, i ptr_off, i eid_off, i src_off, i out_off,
 *                      R int ptr, R int eid, R src, W out)
 *     Work-item i < nodes*feat computes feature f = i % feat of segment t = i / feat and writes
 *     out[out_off + i]. A segment with no slots writes empty (SUM and MEAN). MAX writes empty whenever the
 *     result is -inf, which covers every empty segment and mirrors the CPU's replacement of a
 *     NEGATIVE_INFINITY result (CpuIndexing.java segment; MessagePassing.java scatterMax and GlobalMaxPool.java
 *     forward pass the empty value).
 *
 * TR_SegmentReduceBackward_F32(i op, i nodes, i feat, i ptr_off, i eid_off, i src_off, i out_off, i gy_off,
 *                              i gx_off, R int ptr, R int eid, R src, R out, R gy, W gx)
 *     The gradient of TR_SegmentReduce_F32 with respect to src. gx is a [rows, feat] block at gx_off that the
 *     host zero-fills; rows no slot names keep 0. Work-item i < nodes*feat handles feature f = i % feat of
 *     segment t = i / feat, with g = gy[gy_off + i]; a segment with no slots writes nothing.
 *       SUM:  gx[eid[e]*feat + f] = g for every slot e of t
 *       MEAN: gx[eid[e]*feat + f] = g / (float)(slot count) for every slot e of t
 *       MAX:  gx[eid[e]*feat + f] = g for the first slot e of t, ascending, whose source value is Java ==
 *             out[out_off + i] (tr_s_same; -0 == +0), the forward result as written (after the empty
 *             substitution); a NaN result matches nothing and routes nowhere.
 *     src and out are read only for MAX (otherwise the placeholder). Precondition: every source row appears
 *     at most once in eid[ptr[0] .. ptr[nodes]), so each gx element has at most one writer (the GraphData CSR
 *     producers build permutations; the CPU reference tensor/CpuIndexing.java segmentReduceBackward validates
 *     it and throws). SUM and MAX are bit-identical copies; MEAN is the division class.
 *
 * Dispatch: 1-D, local TR_LOCAL, nodes*feat items. Deterministic: one work-item owns each output element
 * and no atomics are used. Inputs and outputs never alias.
 */

#define TR_SEG_SUM  0
#define TR_SEG_MEAN 1
#define TR_SEG_MAX  2

NF_KERNEL void TR_SegmentReduce_F32(
    const int op,
    const int nodes,
    const int feat,
    const float empty,
    const int ptr_off,
    const int eid_off,
    const int src_off,
    const int out_off,
    NF_GLOBAL const int* NF_RESTRICT ptr,
    NF_GLOBAL const int* NF_RESTRICT eid,
    NF_GLOBAL const float* NF_RESTRICT src,
    NF_GLOBAL float* NF_RESTRICT out)
{
    const int i = (feat > 0) ? tr_gid(nodes * feat) : -1;
    if (i < 0) return;
    const int node = i / feat;
    const int f = i - node * feat;
    const int e0 = ptr[ptr_off + node];
    const int e1 = ptr[ptr_off + node + 1];
    float r;
    if (op == TR_SEG_MAX) {
        float acc = -NF_INF;
        for (int e = e0; e < e1; ++e) {
            acc = tr_jmax(acc, src[src_off + eid[eid_off + e] * feat + f]);
        }
        r = as_uint(acc) == 0xff800000u ? empty : acc;
    } else {
        float acc = 0.0f;
        for (int e = e0; e < e1; ++e) {
            acc = acc + src[src_off + eid[eid_off + e] * feat + f];
        }
        if (e1 <= e0) {
            r = empty;
        } else {
            r = op == TR_SEG_MEAN ? acc / (float)(e1 - e0) : acc;
        }
    }
    out[out_off + i] = r;
}

/* Java a == b for floats: false when either is NaN, true for -0 == +0, otherwise bit equality, which is
 * exact value equality for non-NaN floats even on devices that flush denormals in float compares. */
NF_INLINE int tr_s_same(const float a, const float b) {
    if (tr_isnan(a) || tr_isnan(b)) return 0;
    if (tr_iszero(a) && tr_iszero(b)) return 1;
    return as_uint(a) == as_uint(b);
}

NF_KERNEL void TR_SegmentReduceBackward_F32(
    const int op,
    const int nodes,
    const int feat,
    const int ptr_off,
    const int eid_off,
    const int src_off,
    const int out_off,
    const int gy_off,
    const int gx_off,
    NF_GLOBAL const int* NF_RESTRICT ptr,
    NF_GLOBAL const int* NF_RESTRICT eid,
    NF_GLOBAL const float* NF_RESTRICT src,
    NF_GLOBAL const float* NF_RESTRICT out,
    NF_GLOBAL const float* NF_RESTRICT gy,
    NF_GLOBAL float* NF_RESTRICT gx)
{
    const int i = (feat > 0) ? tr_gid(nodes * feat) : -1;
    if (i < 0) return;
    const int node = i / feat;
    const int f = i - node * feat;
    const int e0 = ptr[ptr_off + node];
    const int e1 = ptr[ptr_off + node + 1];
    if (e1 <= e0) return;
    const float g = gy[gy_off + i];
    if (op == TR_SEG_MAX) {
        const float m = out[out_off + i];
        for (int e = e0; e < e1; ++e) {
            const int r = eid[eid_off + e] * feat + f;
            if (tr_s_same(src[src_off + r], m)) {
                gx[gx_off + r] = g;
                break;
            }
        }
    } else {
        const float v = op == TR_SEG_MEAN ? g / (float)(e1 - e0) : g;
        for (int e = e0; e < e1; ++e) {
            gx[gx_off + eid[eid_off + e] * feat + f] = v;
        }
    }
}

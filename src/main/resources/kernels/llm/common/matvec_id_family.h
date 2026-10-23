/* common/matvec_id_family.h - {T}_MatVecId, {T}_MatVecIdGateUp and {T}_BatchMatMulId macro family for MoE expert matvecs. Requires: common/dialect.h, common/config.h, common/math.h, common/reduce.h, common/quant_blocks.h */
/*
 * NF_MOE_ID_KERNELS(N, D, BS, BB, TPS) instantiates, for weight type N with BS elements and BB bytes
 * per block and TPS work-items cooperating on one block, the three expert-indexed matvec kernels
 * below. D is the prefix of the decode pair the kernels call: D##_decode and D##_decode_dual with the
 * decoder contract of common/matvec_family.h (for the quant types D = N and the triples come from
 * common/quant_decode_*.h; F32, F16 and BF16 use the nf_moe_* pairs defined here).
 *
 * An expert tensor holds every expert ("slot") back to back: a 3-D ggml tensor [in_dim, rows, n_slot]
 * whose slot e row r is tensor row e * rows + r of row_bytes = in_dim / BS * BB bytes, which is the
 * layout ggml_mul_mat_id reads (llama-graph.cpp:1706-1734). A tensor larger than one device binding
 * is uploaded as shards of whole experts, so every kernel binds one shard: the experts
 * [expert_base, expert_base + expert_count) of the tensor, expert e (in that range) starting at byte
 * (e - expert_base) * rows * row_bytes of the bound weight. One dispatch per shard covers the tensor;
 * a single-shard tensor is one dispatch with expert_base = 0, expert_count = n_slot. A (slot, row)
 * output belongs to the dispatch whose range holds its expert and is written by that dispatch only,
 * so the shard dispatches of one product write disjoint outputs. Per-expert scales are indexed with
 * the global expert e.
 *
 * {T}_MatVecId(out_dim, in_dim, n_sel, x_stride, in_off, out_off, has_scale, expert_base,
 *              expert_count, slots, weight, input, scales, output)
 *     for s < n_sel, r < out_dim, e = slots[s] with expert_base <= e < expert_base + expert_count:
 *         y = dot(weight[e - expert_base][r][:], input[in_off + s * x_stride + 0 .. in_dim))
 *         output[out_off + s * out_dim + r] = has_scale ? y * scales[e] : y
 *     x_stride = 0 broadcasts one input row to every slot (decode gate/up), x_stride = in_dim gives
 *     each slot its own row (down projection, weight-before-FFN inputs, MLA absorption).
 *     scales is the per-expert *_exps.scale vector (build_lora_mm_id, llama-graph.cpp:1724-1730).
 *
 * {T}_MatVecIdGateUp(out_dim, in_dim, n_sel, x_stride, in_off, glu_op, limit, alpha, beta, fused,
 *                    has_scale, expert_base, expert_count, slots, gate_w, up_w, input, scales, output)
 *     For the slots whose expert lies in the bound range (gate_w and up_w bind the same experts),
 *     g = dot(gate row), u = dot(up row) over the same input row as {T}_MatVecId, then
 *         output[s * out_dim + r] = nf_glu(glu_op, g * sg, u * su, limit, alpha, alpha, beta)
 *     with the LLM_GLU parameter mapping (SITU: alpha = situ_beta, beta = situ_linear_beta).
 *     fused = 0: separate gate_exps / up_exps tensors, rows r of slot e in each.
 *     fused = 1: one gate_up_exps tensor bound to both gate_w and up_w; slot e holds 2 * out_dim rows,
 *                gate rows first (llama-graph.cpp:2372-2389 views at 0 and n_ff).
 *     has_scale: 0 sg = su = 1; 1 sg = su = scales[e] (the fused tensor scaled by up_exps_s,
 *                llama-graph.cpp:2372); 2 sg = scales[2e], su = scales[2e + 1] (separate gate_exps_s
 *                and up_exps_s interleaved by the host, 1 where a tensor has no scale).
 *     Expert biases are not applied here: with *_exps.bias the host runs two {T}_MatVecId, LLM_AddId
 *     and LLM_GLU instead, because the bias is added before the activation (llama-graph.cpp:2381-2418).
 *
 * {T}_BatchMatMulId(out_dim, in_dim, n_rows, x_div, in_off, has_scale, expert_base, expert_count,
 *                   row_token, row_slot, weight, input, scales, output)
 *     Prefill over the expert-sorted rows of LLM_MoE_Bucketize: sorted row i has pair index
 *     p = row_token[i] (p = t * k + j) and slot e = row_slot[i]; for e in the bound range it computes
 *         output[p * out_dim + r] = scale(e) * dot(weight[e - expert_base][r][:],
 *                                                  input[in_off + (p / x_div) * in_dim ..))
 *     so the output is in pair order whatever the sort. x_div = k reads the token row (gate/up),
 *     x_div = 1 the pair row (down). A row with e < 0 (an id LLM_MoE_Bucketize found out of range)
 *     writes 0, in the dispatch with expert_base = 0 only. Each workgroup handles one output row r for
 *     NF_MOE_BMM_ROWS consecutive sorted rows, which share their expert except at bucket boundaries,
 *     so the weight row stays in cache.
 *
 * Launch contract: local = {L, 1}, L a power of two <= LLM_DIMX and a multiple of TPS (CUDA: of 32).
 *     {T}_MatVecId, {T}_MatVecIdGateUp: global = {out_dim * L, n_sel}
 *     {T}_BatchMatMulId:                global = {out_dim * L, ceil(n_rows / NF_MOE_BMM_ROWS)}
 * The `row >= out_dim` style early returns depend only on group ids, and the expert-range returns
 * (and skipped sorted rows) only on the slot of the group, so they are uniform per workgroup and
 * every barrier inside nf_wg_sum is reached by all work-items of a live workgroup. Weight row offsets
 * are formed in nf_i64.
 */
#ifndef NF_MATVEC_ID_FAMILY_H
#define NF_MATVEC_ID_FAMILY_H

/* Sorted rows per {T}_BatchMatMulId workgroup (Java: MoeKernels.BMM_ROWS). */
#define NF_MOE_BMM_ROWS 8

/* ---- dense decode pairs: one element per block, one work-item per block (BS = TPS = 1) ---- */

NF_INLINE float nf_moe_f32_w(NF_GLOBAL const uchar* bp) {
    return *(NF_GLOBAL const float*)bp;
}

NF_INLINE float nf_moe_f16_w(NF_GLOBAL const uchar* bp) {
    return vload_half(0, (NF_GLOBAL const NF_HALF*)bp);
}

NF_INLINE float nf_moe_bf16_w(NF_GLOBAL const uchar* bp) {
    return bf16_to_f32(*(NF_GLOBAL const ushort*)bp);
}

#define NF_MOE_DENSE_PAIR(D, LOAD) \
NF_INLINE float D##_decode(NF_GLOBAL const uchar* bp, NF_GLOBAL const float* input, \
                           int off, int blk, int ltid) { \
    return LOAD(bp) * input[off + blk]; \
} \
NF_INLINE void D##_decode_dual(NF_GLOBAL const uchar* gbp, NF_GLOBAL const uchar* ubp, \
                               NF_GLOBAL const float* input, int off, int blk, int ltid, \
                               float* gs, float* us) { \
    const float x = input[off + blk]; \
    *gs += LOAD(gbp) * x; \
    *us += LOAD(ubp) * x; \
}

NF_MOE_DENSE_PAIR(nf_moe_f32, nf_moe_f32_w)
NF_MOE_DENSE_PAIR(nf_moe_f16, nf_moe_f16_w)
NF_MOE_DENSE_PAIR(nf_moe_bf16, nf_moe_bf16_w)

/* True when expert e lies in the bound shard [base, base + count); never overflows for base >= 0. */
NF_INLINE int nf_moe_in_shard(const int e, const int base, const int count) {
    return e >= base && e - base < count;
}

/* ---- kernel templates ---- */

#define NF_MOE_ID_MATVEC(N, D, BS, BB, TPS) \
NF_KERNEL void N##_MatVecId( \
    const int out_dim, const int in_dim, const int n_sel, const int x_stride, \
    const int in_off, const int out_off, const int has_scale, \
    const int expert_base, const int expert_count, \
    NF_GLOBAL const int* NF_RESTRICT slots, \
    NF_GLOBAL const uchar* NF_RESTRICT weight, \
    NF_GLOBAL const float* NF_RESTRICT input, \
    NF_GLOBAL const float* NF_RESTRICT scales, \
    NF_GLOBAL float* NF_RESTRICT output) { \
    const int row = get_group_id(0), s = get_group_id(1); \
    if (row >= out_dim || s >= n_sel) return; \
    const int e = slots[s]; \
    if (!nf_moe_in_shard(e, expert_base, expert_count)) return; \
    NF_LOCAL float nf_moe_red[LLM_DIMX]; \
    const int lid = get_local_id(0), lsize = get_local_size(0); \
    const int sblk = in_dim / (BS); \
    NF_GLOBAL const uchar* wrow = weight \
        + ((nf_i64)(e - expert_base) * out_dim + row) * ((nf_i64)sblk * (BB)); \
    NF_GLOBAL const float* x = input + in_off + (nf_i64)s * x_stride; \
    const int gpi = lsize / (TPS), gid = lid / (TPS), ltid = lid % (TPS); \
    float ps = 0.0f; \
    for (int b = gid; b < sblk; b += gpi) \
        ps += D##_decode(wrow + (nf_i64)b * (BB), x, 0, b, ltid); \
    const float acc = nf_wg_sum(nf_moe_red, ps); \
    if (lid == 0) \
        output[out_off + (nf_i64)s * out_dim + row] = has_scale ? acc * scales[e] : acc; \
}

#define NF_MOE_ID_GATEUP(N, D, BS, BB, TPS) \
NF_KERNEL void N##_MatVecIdGateUp( \
    const int out_dim, const int in_dim, const int n_sel, const int x_stride, const int in_off, \
    const int glu_op, const float limit, const float alpha, const float beta, \
    const int fused, const int has_scale, const int expert_base, const int expert_count, \
    NF_GLOBAL const int* NF_RESTRICT slots, \
    NF_GLOBAL const uchar* gate_w, \
    NF_GLOBAL const uchar* up_w, \
    NF_GLOBAL const float* NF_RESTRICT input, \
    NF_GLOBAL const float* NF_RESTRICT scales, \
    NF_GLOBAL float* NF_RESTRICT output) { \
    const int row = get_group_id(0), s = get_group_id(1); \
    if (row >= out_dim || s >= n_sel) return; \
    const int e = slots[s]; \
    if (!nf_moe_in_shard(e, expert_base, expert_count)) return; \
    NF_LOCAL float nf_moe_red[LLM_DIMX]; \
    const int lid = get_local_id(0), lsize = get_local_size(0); \
    const int sblk = in_dim / (BS); \
    const nf_i64 row_bytes = (nf_i64)sblk * (BB); \
    const nf_i64 le = (nf_i64)(e - expert_base); \
    const nf_i64 slot_rows = fused ? 2 * (nf_i64)out_dim : (nf_i64)out_dim; \
    NF_GLOBAL const uchar* grow = gate_w + (le * slot_rows + row) * row_bytes; \
    NF_GLOBAL const uchar* urow = up_w + (le * slot_rows + (fused ? out_dim : 0) + row) * row_bytes; \
    NF_GLOBAL const float* x = input + in_off + (nf_i64)s * x_stride; \
    const int gpi = lsize / (TPS), gid = lid / (TPS), ltid = lid % (TPS); \
    float gs = 0.0f, us = 0.0f; \
    for (int b = gid; b < sblk; b += gpi) \
        D##_decode_dual(grow + (nf_i64)b * (BB), urow + (nf_i64)b * (BB), x, 0, b, ltid, &gs, &us); \
    const float g = nf_wg_sum(nf_moe_red, gs); \
    const float u = nf_wg_sum(nf_moe_red, us); \
    if (lid == 0) { \
        float sg = 1.0f, su = 1.0f; \
        if (has_scale == 1) { sg = scales[e]; su = sg; } \
        else if (has_scale == 2) { sg = scales[2 * e]; su = scales[2 * e + 1]; } \
        output[(nf_i64)s * out_dim + row] = nf_glu(glu_op, g * sg, u * su, limit, alpha, alpha, beta); \
    } \
}

#define NF_MOE_ID_BATCH(N, D, BS, BB, TPS) \
NF_KERNEL void N##_BatchMatMulId( \
    const int out_dim, const int in_dim, const int n_rows, const int x_div, \
    const int in_off, const int has_scale, const int expert_base, const int expert_count, \
    NF_GLOBAL const int* NF_RESTRICT row_token, \
    NF_GLOBAL const int* NF_RESTRICT row_slot, \
    NF_GLOBAL const uchar* NF_RESTRICT weight, \
    NF_GLOBAL const float* NF_RESTRICT input, \
    NF_GLOBAL const float* NF_RESTRICT scales, \
    NF_GLOBAL float* NF_RESTRICT output) { \
    const int row = get_group_id(0); \
    const int i0 = (int)get_group_id(1) * NF_MOE_BMM_ROWS; \
    if (row >= out_dim || i0 >= n_rows) return; \
    NF_LOCAL float nf_moe_red[LLM_DIMX]; \
    const int lid = get_local_id(0), lsize = get_local_size(0); \
    const int sblk = in_dim / (BS); \
    const nf_i64 row_bytes = (nf_i64)sblk * (BB); \
    const int gpi = lsize / (TPS), gid = lid / (TPS), ltid = lid % (TPS); \
    const int i1 = min(i0 + NF_MOE_BMM_ROWS, n_rows); \
    for (int i = i0; i < i1; i++) { \
        const int p = row_token[i], e = row_slot[i]; \
        if (e < 0 ? expert_base != 0 : !nf_moe_in_shard(e, expert_base, expert_count)) continue; \
        float ps = 0.0f; \
        if (e >= 0) { \
            NF_GLOBAL const uchar* wrow = weight + ((nf_i64)(e - expert_base) * out_dim + row) * row_bytes; \
            NF_GLOBAL const float* x = input + in_off + (nf_i64)(p / x_div) * in_dim; \
            for (int b = gid; b < sblk; b += gpi) \
                ps += D##_decode(wrow + (nf_i64)b * (BB), x, 0, b, ltid); \
        } \
        const float acc = nf_wg_sum(nf_moe_red, ps); \
        if (lid == 0) \
            output[(nf_i64)p * out_dim + row] = e < 0 ? 0.0f : (has_scale ? acc * scales[e] : acc); \
    } \
}

#define NF_MOE_ID_KERNELS(N, D, BS, BB, TPS) \
    NF_MOE_ID_MATVEC(N, D, BS, BB, TPS) \
    NF_MOE_ID_GATEUP(N, D, BS, BB, TPS) \
    NF_MOE_ID_BATCH(N, D, BS, BB, TPS)

#endif /* NF_MATVEC_ID_FAMILY_H */

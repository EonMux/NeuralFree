/* common/matvec_family.h - IQ_* macro family that generates the 8 per-weight-type matvec/embedding kernels. Requires: common/dialect.h, common/config.h, common/math.h, common/reduce.h */
/*
 * IQ_ALL_KERNELS(T, BS, BB, TPS) instantiates, for weight type T with BS elements and BB bytes per
 * block and TPS threads cooperating on one block:
 *   T_MatVec, T_MatVecAdd, T_MatVecFusedGateUp, T_ComputeLogits, T_EmbeddingLookup,
 *   T_BatchEmbeddingLookup, T_BatchMatMul, T_BatchMatMul_2D
 * with one argument list per kernel form, shared by every weight type.
 *
 * Decoder contract the instantiating file must define before the instantiation:
 *   NF_INLINE float T_decode(NF_GLOBAL const uchar* bp, NF_GLOBAL const float* input,
 *                            int off, int blk, int ltid);
 *       Partial dot product of block blk (bytes at bp) with input[off + blk*BS ...], restricted to the
 *       elements thread ltid in [0, TPS) owns; the TPS partial sums of a block add to the full dot.
 *   NF_INLINE void T_decode_dual(NF_GLOBAL const uchar* gbp, NF_GLOBAL const uchar* ubp,
 *                                NF_GLOBAL const float* input, int off, int blk, int ltid,
 *                                float* gs, float* us);
 *       Same as T_decode for two weight rows sharing the input; accumulates into *gs and *us.
 *   NF_INLINE float T_decode_elem(NF_GLOBAL const uchar* weight, int dim, int tok, int i);
 *       Element i of row tok of a [rows x dim] weight; forms the row byte offset in nf_i64.
 *
 * Launch contract: matvec forms use global = rows * L, local = L (L a power of two <= LLM_DIMX and a
 * multiple of TPS; CUDA: multiple of 32); _2D uses global = {rows * L, seq_len}, local = {L, 1};
 * embedding forms are 1-D element grids. Row byte offsets and token-row element offsets are nf_i64.
 * The `row >= rows` and `s >= seq_len` early returns are uniform per workgroup, so every barrier
 * is reached by all work-items of a live workgroup.
 */
#ifndef NF_MATVEC_FAMILY_H
#define NF_MATVEC_FAMILY_H

#define IQ_MATVEC(N, BS, BB, TPS) \
NF_KERNEL void N##_MatVec( \
    const int out_dim, const int in_dim, \
    NF_GLOBAL const uchar* NF_RESTRICT weight, \
    NF_GLOBAL const float* NF_RESTRICT input, \
    NF_GLOBAL float* NF_RESTRICT output) { \
    const int row = get_group_id(0); \
    if (row >= out_dim) return; \
    const int lid = get_local_id(0), lsize = get_local_size(0); \
    const int sblk = in_dim / BS; \
    NF_GLOBAL const uchar* wrow = weight + (nf_i64)row * sblk * BB; \
    const int gpi = lsize / TPS, gid = lid / TPS, ltid = lid % TPS; \
    float ps = 0.0f; \
    for (int b = gid; b < sblk; b += gpi) \
        ps += N##_decode(wrow + b * BB, input, 0, b, ltid); \
    NF_LOCAL float sc[LLM_DIMX]; sc[lid] = ps; barrier(CLK_LOCAL_MEM_FENCE); \
    REDUCE_SUM(sc, lid); \
    if (lid == 0) output[row] = sc[0]; \
}

#define IQ_MATVECADD(N, BS, BB, TPS) \
NF_KERNEL void N##_MatVecAdd( \
    const int out_dim, const int in_dim, \
    NF_GLOBAL const uchar* NF_RESTRICT weight, \
    NF_GLOBAL const float* NF_RESTRICT input, \
    NF_GLOBAL float* NF_RESTRICT addend_output) { \
    const int row = get_group_id(0); \
    if (row >= out_dim) return; \
    const int lid = get_local_id(0), lsize = get_local_size(0); \
    const int sblk = in_dim / BS; \
    NF_GLOBAL const uchar* wrow = weight + (nf_i64)row * sblk * BB; \
    const int gpi = lsize / TPS, gid = lid / TPS, ltid = lid % TPS; \
    float ps = 0.0f; \
    for (int b = gid; b < sblk; b += gpi) \
        ps += N##_decode(wrow + b * BB, input, 0, b, ltid); \
    NF_LOCAL float sc[LLM_DIMX]; sc[lid] = ps; barrier(CLK_LOCAL_MEM_FENCE); \
    REDUCE_SUM(sc, lid); \
    if (lid == 0) addend_output[row] += sc[0]; \
}

#define IQ_FUSEDGATEUP(N, BS, BB, TPS) \
NF_KERNEL void N##_MatVecFusedGateUp( \
    const int out_dim, const int in_dim, \
    NF_GLOBAL const uchar* NF_RESTRICT gate_weight, \
    NF_GLOBAL const uchar* NF_RESTRICT up_weight, \
    NF_GLOBAL const float* NF_RESTRICT input, \
    NF_GLOBAL float* NF_RESTRICT output) { \
    const int row = get_group_id(0); \
    if (row >= out_dim) return; \
    const int lid = get_local_id(0), lsize = get_local_size(0); \
    const int sblk = in_dim / BS; \
    const nf_i64 rbase = (nf_i64)row * sblk * BB; \
    NF_GLOBAL const uchar* grow = gate_weight + rbase; \
    NF_GLOBAL const uchar* urow = up_weight + rbase; \
    const int gpi = lsize / TPS, gid = lid / TPS, ltid = lid % TPS; \
    float gs = 0.0f, us = 0.0f; \
    for (int b = gid; b < sblk; b += gpi) \
        N##_decode_dual(grow + b * BB, urow + b * BB, input, 0, b, ltid, &gs, &us); \
    NF_LOCAL float sc[LLM_DIMX]; \
    sc[lid] = gs; barrier(CLK_LOCAL_MEM_FENCE); REDUCE_SUM(sc, lid); \
    float gv = sc[0]; barrier(CLK_LOCAL_MEM_FENCE); \
    sc[lid] = us; barrier(CLK_LOCAL_MEM_FENCE); REDUCE_SUM(sc, lid); \
    if (lid == 0) output[row] = nf_silu(gv) * sc[0]; \
}

#define IQ_COMPUTELOGITS(N, BS, BB, TPS) \
NF_KERNEL void N##_ComputeLogits( \
    const int vocab_size, const int embed_dim, \
    NF_GLOBAL const uchar* NF_RESTRICT weight, \
    NF_GLOBAL const float* NF_RESTRICT hidden, \
    NF_GLOBAL float* NF_RESTRICT logits) { \
    const int row = get_group_id(0); \
    if (row >= vocab_size) return; \
    const int lid = get_local_id(0), lsize = get_local_size(0); \
    const int sblk = embed_dim / BS; \
    NF_GLOBAL const uchar* wrow = weight + (nf_i64)row * sblk * BB; \
    const int gpi = lsize / TPS, gid = lid / TPS, ltid = lid % TPS; \
    float ps = 0.0f; \
    for (int b = gid; b < sblk; b += gpi) \
        ps += N##_decode(wrow + b * BB, hidden, 0, b, ltid); \
    NF_LOCAL float sc[LLM_DIMX]; sc[lid] = ps; barrier(CLK_LOCAL_MEM_FENCE); \
    REDUCE_SUM(sc, lid); \
    if (lid == 0) logits[row] = sc[0]; \
}

#define IQ_EMBEDLOOKUP(N, BS, BB) \
NF_KERNEL void N##_EmbeddingLookup( \
    const int embed_dim, const int token_id, \
    NF_GLOBAL const uchar* NF_RESTRICT weight, \
    NF_GLOBAL float* NF_RESTRICT output) { \
    const int i = get_global_id(0); \
    if (i >= embed_dim) return; \
    output[i] = N##_decode_elem(weight, embed_dim, token_id, i); \
}

#define IQ_BATCHEMBED(N, BS, BB) \
NF_KERNEL void N##_BatchEmbeddingLookup( \
    const int embed_dim, const int seq_len, \
    NF_GLOBAL const int* NF_RESTRICT token_ids, \
    NF_GLOBAL const uchar* NF_RESTRICT weight, \
    NF_GLOBAL float* NF_RESTRICT output) { \
    const nf_i64 gid = (nf_i64)get_global_id(0); \
    if (gid >= (nf_i64)seq_len * embed_dim) return; \
    const int s = (int)(gid / embed_dim), i = (int)(gid % embed_dim); \
    output[gid] = N##_decode_elem(weight, embed_dim, token_ids[s], i); \
}

#define IQ_BATCHMATMUL(N, BS, BB, TPS) \
NF_KERNEL void N##_BatchMatMul( \
    const int out_dim, const int in_dim, const int seq_len, \
    NF_GLOBAL const uchar* NF_RESTRICT weight, \
    NF_GLOBAL const float* NF_RESTRICT input, \
    NF_GLOBAL float* NF_RESTRICT output) { \
    const int row = get_group_id(0); \
    if (row >= out_dim) return; \
    const int lid = get_local_id(0), lsize = get_local_size(0); \
    const int sblk = in_dim / BS; \
    NF_GLOBAL const uchar* wrow = weight + (nf_i64)row * sblk * BB; \
    const int gpi = lsize / TPS, gid = lid / TPS, ltid = lid % TPS; \
    NF_LOCAL float sc[LLM_DIMX]; \
    for (int s = 0; s < seq_len; s++) { \
        NF_GLOBAL const float* in_s = input + (nf_i64)s * in_dim; \
        float ps = 0.0f; \
        for (int b = gid; b < sblk; b += gpi) \
            ps += N##_decode(wrow + b * BB, in_s, 0, b, ltid); \
        sc[lid] = ps; barrier(CLK_LOCAL_MEM_FENCE); REDUCE_SUM(sc, lid); \
        if (lid == 0) output[(nf_i64)s * out_dim + row] = sc[0]; \
        barrier(CLK_LOCAL_MEM_FENCE); \
    } \
}

#define IQ_BATCHMATMUL2D(N, BS, BB, TPS) \
NF_KERNEL void N##_BatchMatMul_2D( \
    const int out_dim, const int in_dim, const int seq_len, \
    NF_GLOBAL const uchar* NF_RESTRICT weight, \
    NF_GLOBAL const float* NF_RESTRICT input, \
    NF_GLOBAL float* NF_RESTRICT output) { \
    const int row = get_group_id(0), s = get_global_id(1); \
    if (row >= out_dim || s >= seq_len) return; \
    const int lid = get_local_id(0), lsize = get_local_size(0); \
    const int sblk = in_dim / BS; \
    NF_GLOBAL const uchar* wrow = weight + (nf_i64)row * sblk * BB; \
    NF_GLOBAL const float* in_s = input + (nf_i64)s * in_dim; \
    const int gpi = lsize / TPS, gid = lid / TPS, ltid = lid % TPS; \
    float ps = 0.0f; \
    for (int b = gid; b < sblk; b += gpi) \
        ps += N##_decode(wrow + b * BB, in_s, 0, b, ltid); \
    NF_LOCAL float sc[LLM_DIMX]; sc[lid] = ps; barrier(CLK_LOCAL_MEM_FENCE); \
    REDUCE_SUM(sc, lid); \
    if (lid == 0) output[(nf_i64)s * out_dim + row] = sc[0]; \
}

#define IQ_ALL_KERNELS(N, BS, BB, TPS) \
    IQ_MATVEC(N, BS, BB, TPS) \
    IQ_MATVECADD(N, BS, BB, TPS) \
    IQ_FUSEDGATEUP(N, BS, BB, TPS) \
    IQ_COMPUTELOGITS(N, BS, BB, TPS) \
    IQ_EMBEDLOOKUP(N, BS, BB) \
    IQ_BATCHEMBED(N, BS, BB) \
    IQ_BATCHMATMUL(N, BS, BB, TPS) \
    IQ_BATCHMATMUL2D(N, BS, BB, TPS)

#endif /* NF_MATVEC_FAMILY_H */

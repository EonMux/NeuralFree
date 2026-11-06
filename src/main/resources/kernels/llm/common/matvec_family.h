/* common/matvec_family.h - IQ_* macro family that generates the 8 per-weight-type matvec/embedding kernels. Requires: common/dialect.h, common/config.h, common/math.h, common/reduce.h */
/*
 * IQ_ALL_KERNELS(T, BS, BB, TPS) instantiates, for weight type T with BS elements and BB bytes per
 * block and TPS threads cooperating on one block:
 *   T_MatVec, T_MatVec2, T_MatVecAdd, T_MatVecFusedGateUp, T_MatVec2FusedGateUp, T_ComputeLogits,
 *   T_EmbeddingLookup,
 *   T_BatchEmbeddingLookup, T_BatchMatMul, T_BatchMatMul_2D
 * with one argument list per kernel form, shared by every weight type. IQ_ALL_KERNELS_Q8(T, BS, BB, TPS)
 * instantiates the Q8_1-activation forms T_MatVec_Q8, T_MatVecN_Q8, T_MatVec2_Q8, T_MatVecAdd_Q8,
 * T_MatVecFusedGateUp_Q8 and T_MatVec2FusedGateUp_Q8.
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
 * multiple of TPS; CUDA: multiple of 32); T_MatVec2 and T_MatVec2FusedGateUp use global = ceil(rows / 2) * L; _2D uses
 * global = {rows * L, seq_len}, local = {L, 1}; embedding forms are 1-D element grids. The Q8_1-activation
 * forms keep the same geometry, including T_MatVec2_Q8 and T_MatVec2FusedGateUp_Q8 at ceil(rows / 2) * L.
 * Row byte
 * offsets and token-row element offsets are nf_i64. T_MatVecN_Q8 keeps global = rows * L and takes
 * n_tokens consecutive Q8_1 rows. The `row >= rows` and `s >= seq_len` early
 * returns are uniform per workgroup, so every barrier is reached by all work-items of a live
 * workgroup; IQ_MATVECN_Q8's `j < n_tokens` guards are uniform too (n_tokens is a kernel scalar).
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

/*
 * Two output rows per workgroup: one workgroup covers rows 2*p and 2*p+1, so the input row is read
 * once per two weight rows instead of once per row (the activations otherwise dominate L2 traffic:
 * every workgroup streams the whole input). decode_dual loads the input block once for both
 * accumulators. An odd row count repeats the last row's weights into the odd pair's second
 * accumulator and drops its store, so the tail workgroup stays barrier-uniform.
 */
#define IQ_MATVEC2(N, BS, BB, TPS) \
NF_KERNEL void N##_MatVec2( \
    const int out_dim, const int in_dim, \
    NF_GLOBAL const uchar* NF_RESTRICT weight, \
    NF_GLOBAL const float* NF_RESTRICT input, \
    NF_GLOBAL float* NF_RESTRICT output) { \
    const int pair = get_group_id(0); \
    const int row0 = pair * 2; \
    if (row0 >= out_dim) return; \
    const int row1 = min(row0 + 1, out_dim - 1); \
    const int lid = get_local_id(0), lsize = get_local_size(0); \
    const int sblk = in_dim / BS; \
    const nf_i64 rbase = (nf_i64)sblk * BB; \
    NF_GLOBAL const uchar* w0 = weight + (nf_i64)row0 * rbase; \
    NF_GLOBAL const uchar* w1 = weight + (nf_i64)row1 * rbase; \
    const int gpi = lsize / TPS, gid = lid / TPS, ltid = lid % TPS; \
    float ps0 = 0.0f, ps1 = 0.0f; \
    for (int b = gid; b < sblk; b += gpi) \
        N##_decode_dual(w0 + b * BB, w1 + b * BB, input, 0, b, ltid, &ps0, &ps1); \
    NF_LOCAL float sc[LLM_DIMX]; \
    sc[lid] = ps0; barrier(CLK_LOCAL_MEM_FENCE); REDUCE_SUM(sc, lid); \
    const float o0 = sc[0]; barrier(CLK_LOCAL_MEM_FENCE); \
    sc[lid] = ps1; barrier(CLK_LOCAL_MEM_FENCE); REDUCE_SUM(sc, lid); \
    if (lid == 0) { \
        output[row0] = o0; \
        if (row0 + 1 < out_dim) output[row0 + 1] = sc[0]; \
    } \
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

/*
 * The fused gate/up form over two output rows per workgroup: pair p covers rows 2p and 2p+1 of both
 * weights, so the input row is read once per two gate/up row pairs instead of once per pair (the
 * activations otherwise dominate L2 traffic). Each block calls decode_dual for the pair's gate/up rows
 * twice; the second call's input loads hit L1. An odd row count repeats the last row's weights into the
 * odd pair's second accumulators and drops its store, so the tail workgroup stays barrier-uniform.
 */
#define IQ_FUSEDGATEUP2(N, BS, BB, TPS) \
NF_KERNEL void N##_MatVec2FusedGateUp( \
    const int out_dim, const int in_dim, \
    NF_GLOBAL const uchar* NF_RESTRICT gate_weight, \
    NF_GLOBAL const uchar* NF_RESTRICT up_weight, \
    NF_GLOBAL const float* NF_RESTRICT input, \
    NF_GLOBAL float* NF_RESTRICT output) { \
    const int pair = get_group_id(0); \
    const int row0 = pair * 2; \
    if (row0 >= out_dim) return; \
    const int row1 = min(row0 + 1, out_dim - 1); \
    const int lid = get_local_id(0), lsize = get_local_size(0); \
    const int sblk = in_dim / BS; \
    const nf_i64 rbase = (nf_i64)sblk * BB; \
    NF_GLOBAL const uchar* g0 = gate_weight + (nf_i64)row0 * rbase; \
    NF_GLOBAL const uchar* u0 = up_weight + (nf_i64)row0 * rbase; \
    NF_GLOBAL const uchar* g1 = gate_weight + (nf_i64)row1 * rbase; \
    NF_GLOBAL const uchar* u1 = up_weight + (nf_i64)row1 * rbase; \
    const int gpi = lsize / TPS, gid = lid / TPS, ltid = lid % TPS; \
    float gs0 = 0.0f, us0 = 0.0f, gs1 = 0.0f, us1 = 0.0f; \
    for (int b = gid; b < sblk; b += gpi) { \
        N##_decode_dual(g0 + b * BB, u0 + b * BB, input, 0, b, ltid, &gs0, &us0); \
        N##_decode_dual(g1 + b * BB, u1 + b * BB, input, 0, b, ltid, &gs1, &us1); \
    } \
    NF_LOCAL float sc[LLM_DIMX]; \
    sc[lid] = gs0; barrier(CLK_LOCAL_MEM_FENCE); REDUCE_SUM(sc, lid); \
    const float gv0 = sc[0]; barrier(CLK_LOCAL_MEM_FENCE); \
    sc[lid] = us0; barrier(CLK_LOCAL_MEM_FENCE); REDUCE_SUM(sc, lid); \
    const float uv0 = sc[0]; barrier(CLK_LOCAL_MEM_FENCE); \
    sc[lid] = gs1; barrier(CLK_LOCAL_MEM_FENCE); REDUCE_SUM(sc, lid); \
    const float gv1 = sc[0]; barrier(CLK_LOCAL_MEM_FENCE); \
    sc[lid] = us1; barrier(CLK_LOCAL_MEM_FENCE); REDUCE_SUM(sc, lid); \
    if (lid == 0) { \
        output[row0] = nf_silu(gv0) * uv0; \
        if (row0 + 1 < out_dim) output[row1] = nf_silu(gv1) * sc[0]; \
    } \
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
    IQ_MATVEC2(N, BS, BB, TPS) \
    IQ_MATVECADD(N, BS, BB, TPS) \
    IQ_FUSEDGATEUP(N, BS, BB, TPS) \
    IQ_FUSEDGATEUP2(N, BS, BB, TPS) \
    IQ_COMPUTELOGITS(N, BS, BB, TPS) \
    IQ_EMBEDLOOKUP(N, BS, BB) \
    IQ_BATCHEMBED(N, BS, BB) \
    IQ_BATCHMATMUL(N, BS, BB, TPS) \
    IQ_BATCHMATMUL2D(N, BS, BB, TPS)

/*
 * Integer-dot forms: the activation argument is a Q8_1 row (one 36-byte block per 32 elements)
 * instead of F32, and the weight block is decoded through N##_decode_q8 / N##_decode_q8_dual
 * (common/quant_decode_*_q8.h). The argument lists and the launch contract equal the float forms
 * (the input buffer holds a whole Q8_1 row: in_dim / 32 * 36 bytes). IQ_MATVEC2_Q8 and
 * IQ_FUSEDGATEUP2_Q8 pair output rows exactly as IQ_MATVEC2 and IQ_FUSEDGATEUP2 do, reading the
 * single Q8_1 row once per row pair.
 */
#define IQ_MATVEC_Q8(N, BS, BB, TPS) \
NF_KERNEL void N##_MatVec_Q8( \
    const int out_dim, const int in_dim, \
    NF_GLOBAL const uchar* NF_RESTRICT weight, \
    NF_GLOBAL const uchar* NF_RESTRICT input, \
    NF_GLOBAL float* NF_RESTRICT output) { \
    const int row = get_group_id(0); \
    if (row >= out_dim) return; \
    const int lid = get_local_id(0), lsize = get_local_size(0); \
    const int sblk = in_dim / BS; \
    NF_GLOBAL const uchar* wrow = weight + (nf_i64)row * sblk * BB; \
    const int gpi = lsize / TPS, gid = lid / TPS, ltid = lid % TPS; \
    float ps = 0.0f; \
    for (int b = gid; b < sblk; b += gpi) \
        ps += N##_decode_q8(wrow + b * BB, input, b, ltid); \
    NF_LOCAL float sc[LLM_DIMX]; sc[lid] = ps; barrier(CLK_LOCAL_MEM_FENCE); \
    REDUCE_SUM(sc, lid); \
    if (lid == 0) output[row] = sc[0]; \
}

/*
 * Multi-column form: input holds n_tokens consecutive Q8_1 rows (row t's blocks at byte
 * t * in_dim / Q8_1_BLOCK_SIZE * Q8_1_BLOCK_BYTES), and one launch serves the whole token group
 * with the same one-workgroup-per-row geometry as IQ_MATVEC_Q8 (global = rows * L). The block loop
 * streams the weight row once and dots every block against all live activation blocks, instead of
 * the token count re-reads of one IQ_MATVEC_Q8 per row; the token loop is compile-time unrolled
 * over NF_MVQN_MAX with the uniform j < n_tokens guard, so the accumulators stay in registers and
 * out-of-range rows are never read. n_tokens is in [1, NF_MVQN_MAX]; token t's dot lands at
 * output[t * out_dim + row], and token t's reduction is the exact IQ_MATVEC_Q8 reduction.
 *
 * The whole token group is reduced in one pass (IQ_MVQN_TREE): a per-token reduction pays its
 * barrier chain for every token of every weight row, which from a few tokens up costs more than the
 * block loop it follows; reducing all tokens' slots in lockstep divides the barrier count by the
 * token count. Each token slot still takes the exact REDUCE_SUM steps in the same order, so the
 * per-token results equal the single-row kernel's. The CUDA and HIP warps are 32 lanes
 * (common/dialect.h), so their form stages the per-warp sums of every token in one leader slot
 * table (IQ_MVQN_LEADER_SLOTS entries per token) and trees over the slots; the other dialects
 * stage every lane's partial of every token and need NF_MVQN_MAX * LLM_DIMX floats of local
 * memory, within the 32 KiB the OpenCL spec guarantees.
 */
#if defined(USE_SUBGROUPS) && (defined(__CUDACC__) || defined(__HIP__))
#define IQ_MVQN_LEADER_SLOTS (LLM_DIMX / 32)
/* Stages the per-warp sums of every live token in one leader slot table, trees over the slots and
 * writes the tokens' outputs. Reads and rewrites the kernel's register partials `ps`; every lane of
 * every subgroup calls it. */
#define IQ_MVQN_TREE(ts, lid, lsize, n_tokens, out, out_dim, row) \
    { \
        const uint nsg = lsize / 32; \
        _Pragma("unroll") \
        for (int j = 0; j < NF_MVQN_MAX; j++) \
            if (j < n_tokens) ps[j] = sub_group_reduce_add(ps[j]); \
        if (get_sub_group_local_id() == 0) { \
            _Pragma("unroll") \
            for (int j = 0; j < NF_MVQN_MAX; j++) \
                if (j < n_tokens) ts[j * IQ_MVQN_LEADER_SLOTS + get_sub_group_id()] = ps[j]; \
        } \
        barrier(CLK_LOCAL_MEM_FENCE); \
        for (uint s = nsg >> 1; s > 0; s >>= 1) { \
            if (lid < s) { \
                _Pragma("unroll") \
                for (int j = 0; j < NF_MVQN_MAX; j++) \
                    if (j < n_tokens) ts[j * IQ_MVQN_LEADER_SLOTS + lid] += ts[j * IQ_MVQN_LEADER_SLOTS + lid + s]; \
            } \
            barrier(CLK_LOCAL_MEM_FENCE); \
        } \
        if (lid == 0) { \
            _Pragma("unroll") \
            for (int j = 0; j < NF_MVQN_MAX; j++) \
                if (j < n_tokens) out[(nf_i64)j * out_dim + row] = ts[j * IQ_MVQN_LEADER_SLOTS]; \
        } \
    }
#else
#define IQ_MVQN_LEADER_SLOTS LLM_DIMX
/* Stages every lane's partial of every live token, trees over the lanes and writes the tokens'
 * outputs. Reads the kernel's register partials `ps`. */
#define IQ_MVQN_TREE(ts, lid, lsize, n_tokens, out, out_dim, row) \
    { \
        _Pragma("unroll") \
        for (int j = 0; j < NF_MVQN_MAX; j++) \
            if (j < n_tokens) ts[j * LLM_DIMX + lid] = ps[j]; \
        barrier(CLK_LOCAL_MEM_FENCE); \
        for (int stride = lsize >> 1; stride > 0; stride >>= 1) { \
            if (lid < stride) { \
                _Pragma("unroll") \
                for (int j = 0; j < NF_MVQN_MAX; j++) \
                    if (j < n_tokens) ts[j * LLM_DIMX + lid] += ts[j * LLM_DIMX + lid + stride]; \
            } \
            barrier(CLK_LOCAL_MEM_FENCE); \
        } \
        if (lid == 0) { \
            _Pragma("unroll") \
            for (int j = 0; j < NF_MVQN_MAX; j++) \
                if (j < n_tokens) out[(nf_i64)j * out_dim + row] = ts[j * LLM_DIMX]; \
        } \
    }
#endif

/*
 * One workgroup per output row over n_tokens consecutive Q8_1 rows (n_tokens in [1, NF_MVQN_MAX]):
 * the block loop streams the weight row once and dots each block against every live activation
 * block (the token loop is compile-time unrolled over NF_MVQN_MAX with the uniform j < n_tokens
 * guard, so the accumulators stay in registers and out-of-range rows are never read), then
 * IQ_MVQN_TREE reduces the whole token group in one pass and writes token j's result to
 * output[j * out_dim + row].
 */
#define IQ_MATVECN_Q8(N, BS, BB, TPS) \
NF_KERNEL void N##_MatVecN_Q8( \
    const int out_dim, const int in_dim, const int n_tokens, \
    NF_GLOBAL const uchar* NF_RESTRICT weight, \
    NF_GLOBAL const uchar* NF_RESTRICT input, \
    NF_GLOBAL float* NF_RESTRICT output) { \
    const int row = get_group_id(0); \
    if (row >= out_dim) return; \
    const int lid = get_local_id(0), lsize = get_local_size(0); \
    const int sblk = in_dim / BS; \
    const nf_i64 qrow = (nf_i64)(in_dim / Q8_1_BLOCK_SIZE) * Q8_1_BLOCK_BYTES; \
    NF_GLOBAL const uchar* wrow = weight + (nf_i64)row * sblk * BB; \
    const int gpi = lsize / TPS, gid = lid / TPS, ltid = lid % TPS; \
    float ps[NF_MVQN_MAX]; \
    _Pragma("unroll") \
    for (int j = 0; j < NF_MVQN_MAX; j++) ps[j] = 0.0f; \
    for (int b = gid; b < sblk; b += gpi) { \
        NF_GLOBAL const uchar* bp = wrow + (nf_i64)b * BB; \
        _Pragma("unroll") \
        for (int j = 0; j < NF_MVQN_MAX; j++) \
            if (j < n_tokens) ps[j] += N##_decode_q8(bp, input + (nf_i64)j * qrow, b, ltid); \
    } \
    NF_LOCAL float ts[NF_MVQN_MAX * IQ_MVQN_LEADER_SLOTS]; \
    IQ_MVQN_TREE(ts, lid, lsize, n_tokens, output, out_dim, row) \
}


/*
 * Two output rows per workgroup over the Q8_1 activation row: the integer-dot twin of IQ_MATVEC2.
 * Pair p covers rows 2p and 2p+1, so the activation row is read once per two weight rows, and
 * decode_q8_dual decodes both weight blocks against the same activation group. An odd row count
 * repeats the last row's weights into the odd pair's second accumulator and drops its store, so the
 * tail workgroup stays barrier-uniform.
 */
#define IQ_MATVEC2_Q8(N, BS, BB, TPS) \
NF_KERNEL void N##_MatVec2_Q8( \
    const int out_dim, const int in_dim, \
    NF_GLOBAL const uchar* NF_RESTRICT weight, \
    NF_GLOBAL const uchar* NF_RESTRICT input, \
    NF_GLOBAL float* NF_RESTRICT output) { \
    const int pair = get_group_id(0); \
    const int row0 = pair * 2; \
    if (row0 >= out_dim) return; \
    const int row1 = min(row0 + 1, out_dim - 1); \
    const int lid = get_local_id(0), lsize = get_local_size(0); \
    const int sblk = in_dim / BS; \
    const nf_i64 rbase = (nf_i64)sblk * BB; \
    NF_GLOBAL const uchar* w0 = weight + (nf_i64)row0 * rbase; \
    NF_GLOBAL const uchar* w1 = weight + (nf_i64)row1 * rbase; \
    const int gpi = lsize / TPS, gid = lid / TPS, ltid = lid % TPS; \
    float ps0 = 0.0f, ps1 = 0.0f; \
    for (int b = gid; b < sblk; b += gpi) \
        N##_decode_q8_dual(w0 + b * BB, w1 + b * BB, input, b, ltid, &ps0, &ps1); \
    NF_LOCAL float sc[LLM_DIMX]; \
    sc[lid] = ps0; barrier(CLK_LOCAL_MEM_FENCE); REDUCE_SUM(sc, lid); \
    const float o0 = sc[0]; barrier(CLK_LOCAL_MEM_FENCE); \
    sc[lid] = ps1; barrier(CLK_LOCAL_MEM_FENCE); REDUCE_SUM(sc, lid); \
    if (lid == 0) { \
        output[row0] = o0; \
        if (row0 + 1 < out_dim) output[row0 + 1] = sc[0]; \
    } \
}

#define IQ_MATVECADD_Q8(N, BS, BB, TPS) \
NF_KERNEL void N##_MatVecAdd_Q8( \
    const int out_dim, const int in_dim, \
    NF_GLOBAL const uchar* NF_RESTRICT weight, \
    NF_GLOBAL const uchar* NF_RESTRICT input, \
    NF_GLOBAL float* NF_RESTRICT addend_output) { \
    const int row = get_group_id(0); \
    if (row >= out_dim) return; \
    const int lid = get_local_id(0), lsize = get_local_size(0); \
    const int sblk = in_dim / BS; \
    NF_GLOBAL const uchar* wrow = weight + (nf_i64)row * sblk * BB; \
    const int gpi = lsize / TPS, gid = lid / TPS, ltid = lid % TPS; \
    float ps = 0.0f; \
    for (int b = gid; b < sblk; b += gpi) \
        ps += N##_decode_q8(wrow + b * BB, input, b, ltid); \
    NF_LOCAL float sc[LLM_DIMX]; sc[lid] = ps; barrier(CLK_LOCAL_MEM_FENCE); \
    REDUCE_SUM(sc, lid); \
    if (lid == 0) addend_output[row] += sc[0]; \
}

#define IQ_FUSEDGATEUP_Q8(N, BS, BB, TPS) \
NF_KERNEL void N##_MatVecFusedGateUp_Q8( \
    const int out_dim, const int in_dim, \
    NF_GLOBAL const uchar* NF_RESTRICT gate_weight, \
    NF_GLOBAL const uchar* NF_RESTRICT up_weight, \
    NF_GLOBAL const uchar* NF_RESTRICT input, \
    NF_GLOBAL float* NF_RESTRICT output) { \
    const int row = get_group_id(0); \
    if (row >= out_dim) return; \
    const int lid = get_local_id(0), lsize = get_local_size(0); \
    const int sblk = in_dim / BS; \
    const nf_i64 rbase = (nf_i64)row * sblk * BB; \
    const int gpi = lsize / TPS, gid = lid / TPS, ltid = lid % TPS; \
    float gs = 0.0f, us = 0.0f; \
    for (int b = gid; b < sblk; b += gpi) \
        N##_decode_q8_dual(gate_weight + rbase + b * BB, up_weight + rbase + b * BB, \
                           input, b, ltid, &gs, &us); \
    NF_LOCAL float sc[LLM_DIMX]; \
    sc[lid] = gs; barrier(CLK_LOCAL_MEM_FENCE); REDUCE_SUM(sc, lid); \
    float gv = sc[0]; barrier(CLK_LOCAL_MEM_FENCE); \
    sc[lid] = us; barrier(CLK_LOCAL_MEM_FENCE); REDUCE_SUM(sc, lid); \
    if (lid == 0) output[row] = nf_silu(gv) * sc[0]; \
}

/*
 * The fused gate/up form over two output rows per workgroup over the Q8_1 activation row: pair p
 * covers rows 2p and 2p+1 of both weights, so the activation row is read once per two gate/up row
 * pairs. Each weight pair decodes through decode_q8_dual against the same activation group. An odd
 * row count repeats the last row's weights into the odd pair's second accumulators and drops its
 * store, so the tail workgroup stays barrier-uniform.
 */
#define IQ_FUSEDGATEUP2_Q8(N, BS, BB, TPS) \
NF_KERNEL void N##_MatVec2FusedGateUp_Q8( \
    const int out_dim, const int in_dim, \
    NF_GLOBAL const uchar* NF_RESTRICT gate_weight, \
    NF_GLOBAL const uchar* NF_RESTRICT up_weight, \
    NF_GLOBAL const uchar* NF_RESTRICT input, \
    NF_GLOBAL float* NF_RESTRICT output) { \
    const int pair = get_group_id(0); \
    const int row0 = pair * 2; \
    if (row0 >= out_dim) return; \
    const int row1 = min(row0 + 1, out_dim - 1); \
    const int lid = get_local_id(0), lsize = get_local_size(0); \
    const int sblk = in_dim / BS; \
    const nf_i64 rbase = (nf_i64)sblk * BB; \
    NF_GLOBAL const uchar* g0 = gate_weight + (nf_i64)row0 * rbase; \
    NF_GLOBAL const uchar* u0 = up_weight + (nf_i64)row0 * rbase; \
    NF_GLOBAL const uchar* g1 = gate_weight + (nf_i64)row1 * rbase; \
    NF_GLOBAL const uchar* u1 = up_weight + (nf_i64)row1 * rbase; \
    const int gpi = lsize / TPS, gid = lid / TPS, ltid = lid % TPS; \
    float gs0 = 0.0f, us0 = 0.0f, gs1 = 0.0f, us1 = 0.0f; \
    for (int b = gid; b < sblk; b += gpi) { \
        N##_decode_q8_dual(g0 + b * BB, u0 + b * BB, input, b, ltid, &gs0, &us0); \
        N##_decode_q8_dual(g1 + b * BB, u1 + b * BB, input, b, ltid, &gs1, &us1); \
    } \
    NF_LOCAL float sc[LLM_DIMX]; \
    sc[lid] = gs0; barrier(CLK_LOCAL_MEM_FENCE); REDUCE_SUM(sc, lid); \
    const float gv0 = sc[0]; barrier(CLK_LOCAL_MEM_FENCE); \
    sc[lid] = us0; barrier(CLK_LOCAL_MEM_FENCE); REDUCE_SUM(sc, lid); \
    const float uv0 = sc[0]; barrier(CLK_LOCAL_MEM_FENCE); \
    sc[lid] = gs1; barrier(CLK_LOCAL_MEM_FENCE); REDUCE_SUM(sc, lid); \
    const float gv1 = sc[0]; barrier(CLK_LOCAL_MEM_FENCE); \
    sc[lid] = us1; barrier(CLK_LOCAL_MEM_FENCE); REDUCE_SUM(sc, lid); \
    if (lid == 0) { \
        output[row0] = nf_silu(gv0) * uv0; \
        if (row0 + 1 < out_dim) output[row1] = nf_silu(gv1) * sc[0]; \
    } \
}

#define IQ_ALL_KERNELS_Q8(N, BS, BB, TPS) \
    IQ_MATVEC_Q8(N, BS, BB, TPS) \
    IQ_MATVECN_Q8(N, BS, BB, TPS) \
    IQ_MATVEC2_Q8(N, BS, BB, TPS) \
    IQ_MATVECADD_Q8(N, BS, BB, TPS) \
    IQ_FUSEDGATEUP_Q8(N, BS, BB, TPS) \
    IQ_FUSEDGATEUP2_Q8(N, BS, BB, TPS)

#endif /* NF_MATVEC_FAMILY_H */

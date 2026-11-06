/* common/matmul_grouped_family.h - {T}_GroupedGemm and {T}_GroupedGemmInd macro family: the expert-sorted tiled GEMM of the batched MoE expert products. Requires: common/dialect.h, common/config.h, common/quant_blocks.h and common/matmul_tiled_family.h (tile geometry, stage compute and dense loaders), which must come first in the program. */
/*
 * NF_GEMM_GROUPED(T, ELEM, BS, BB) defines T##_GroupedGemm (NF_GEMM_GROUPED_DIRECT) and
 * T##_GroupedGemmInd (NF_GEMM_GROUPED_IND): {T}_MatMul_Tiled over the block-aligned
 * expert-sorted rows of LLM_MoE_AlignBlocks (kernels/llm/moe.cl; the layout of vLLM
 * moe_align_block_size), for weight type T with BS elements and BB bytes per block (BS = 1 and
 * BB = element bytes for dense types):
 *
 *   T_GroupedGemm(i out_dim, i in_dim, i n_tiles, i x_div, i x_stride, i in_off, i out_off,
 *                 i has_scale, i expert_base, i expert_count, R<int> gemm_token, R<int> gemm_slot,
 *                 R uchar weight, R float input, R float scales, W float output)
 *
 * gemm_token / gemm_slot hold n_tiles * NF_GEMM_TN positions: tile q covers
 * [q * NF_GEMM_TN, (q + 1) * NF_GEMM_TN), whose positions all belong to one expert
 * e = gemm_slot[q * NF_GEMM_TN] (every tile's first position is a real row, so the first slot names
 * the tile's expert), and a filler position of the bucket's tile padding holds
 * gemm_token = gemm_slot = -1. For a position with pair p = gemm_token[i] >= 0 the tile computes
 *
 *   output[out_off + p * out_dim + r] =
 *       scale(e) * sum_{k < in_dim} W[e][r][k] * input[in_off + (p / x_div) * x_stride + k]
 *
 * for rows r < out_dim: the same pair outputs as {T}_BatchMatMulId (x_stride = in_dim, out_off = 0),
 * because the rows keep LLM_MoE_Bucketize's order inside their bucket. x_div = k reads the token row
 * (gate/up), x_div = 1 the pair row (down, and the per-head MLA products, whose pair rows are
 * x_stride >= in_dim apart). The weight layout is the id-matvec family's: expert e row r starts at byte
 * (e * out_dim + r) * row_bytes of the tensor (row_bytes = in_dim / BS * BB), or
 * (e - expert_base) * out_dim * row_bytes of a shard binding the experts
 * [expert_base, expert_base + expert_count); one dispatch per shard covers the tensor, and a tile
 * whose expert lies outside the bound range belongs to another shard's dispatch, whose workgroup
 * returns whole (uniform, so every barrier inside the K loop is reached or skipped by all of its
 * work-items). has_scale follows the id-matvec codes: 0 scales nothing, 1 multiplies by scales[e].
 * A pair whose id was outside [0, n_expert) sits in the last tiles with e = -1; the dispatch with
 * expert_base = 0 writes 0 to its outputs, a shard dispatch returns.
 *
 *   T_GroupedGemmInd(i out_dim, i in_dim, i n_tiles, i x_div, i x_stride, i in_off, i out_off,
 *                    i has_scale, R<int> gemm_token, R<int> gemm_slot, R<int> map,
 *                    R uchar weight, R float input, R float scales, W float output)
 *
 * is the same product over an expert cache bank, the indirection of {T}_BatchMatMulIdInd
 * (common/matvec_id_family.h) in place of the expert range: the tile of expert e reads local expert
 * l = map[e] of the bound bank (row r of l starts at byte (l * out_dim + r) * row_bytes), and a tile
 * with map[e] < 0 returns whole, leaving its pairs to the host that computes the cache misses. Scales
 * stay indexed with the global expert e; the tiles of out-of-range ids (e = -1) write 0.
 *
 * Launch contract: 2-D, global = {ceil(out_dim / NF_GEMM_TM) * NF_GEMM_WG, n_tiles}, local =
 * {NF_GEMM_WG, 1}. n_tiles is the host's worst-case bound (Java: MoeKernels.groupedTiles) and stays
 * constant across steps: the tiles past the aligned count hold only fillers, and the slot read makes
 * them exit before any barrier. in_dim must be a multiple of BS. Tiling, shared memory and the
 * per-element accumulation order equal {T}_MatMul_Tiled (matmul_tiled_family.h), so an expert bucket
 * of enough pairs produces the same sums as the dense tiled GEMM of its rows.
 */

#ifndef NF_MATMUL_GROUPED_FAMILY_H
#define NF_MATMUL_GROUPED_FAMILY_H

/* True when expert e lies in the bound shard [base, base + count); never overflows for base >= 0. */
NF_INLINE int nf_gg_in_shard(const int e, const int base, const int count) {
    return e >= base && e - base < count;
}

/* Writes 0 to the outputs of the real positions of a tile whose pairs had an out-of-range id. */
NF_INLINE void nf_gg_zero(NF_GLOBAL float* NF_RESTRICT output, NF_GLOBAL const int* NF_RESTRICT gemm_token,
                          const int tile, const int out_dim, const int out_off, const int m0, const int tx,
                          const int ty) {
    for (int j = 0; j < NF_GEMM_RN; j++) {
        const int p = gemm_token[tile * NF_GEMM_TN + ty + NF_GEMM_TX * j];
        if (p < 0) continue;
        for (int i = 0; i < NF_GEMM_RM; i++) {
            const int r = m0 + tx + NF_GEMM_TX * i;
            if (r < out_dim) output[(nf_i64)out_off + (nf_i64)p * out_dim + r] = 0.0f;
        }
    }
}

/* Opening of every grouped kernel: the shared tiles, the tile's expert e and the work-item lanes. */
#define NF_GG_PROLOGUE \
    NF_LOCAL float a_tile[NF_GEMM_TK * NF_GEMM_AS]; \
    NF_LOCAL float b_tile[NF_GEMM_TK * NF_GEMM_BS]; \
    const int tile = (int)get_group_id(1); \
    const int e = gemm_slot[tile * NF_GEMM_TN]; \
    const int lid = get_local_id(0); \
    const int m0 = (int)get_group_id(0) * NF_GEMM_TM; \
    const int tx = lid % NF_GEMM_TX, ty = lid / NF_GEMM_TX; \
    const int lk = lid % NF_GEMM_TK, lr = lid / NF_GEMM_TK;

/*
 * The tile product of a live tile (after NF_GG_PROLOGUE and the expert guards): the 64 rows of local
 * expert WE of the bound weight buffer against the tile's 32 positions, stored as the pair outputs
 * scaled by scales[e] (the global expert) when has_scale.
 */
#define NF_GG_TILE(ELEM, BS, BB, WE) \
    const nf_i64 row_bytes = (nf_i64)(in_dim / (BS)) * (BB); \
    const nf_i64 wbase = (nf_i64)(WE) * out_dim * row_bytes; \
    NF_GLOBAL const uchar* wrow[NF_GEMM_RM]; \
    int wok[NF_GEMM_RM]; \
    for (int q = 0; q < NF_GEMM_RM; q++) { \
        const int r = m0 + lr + NF_GEMM_TX * q; \
        wok[q] = r < out_dim; \
        wrow[q] = weight + wbase + (wok[q] ? (nf_i64)r * row_bytes : (nf_i64)0); \
    } \
    nf_i64 xrow[NF_GEMM_RN]; \
    int xok[NF_GEMM_RN]; \
    for (int q = 0; q < NF_GEMM_RN; q++) { \
        const int pos = tile * NF_GEMM_TN + lr + NF_GEMM_TX * q; \
        const int p = gemm_token[pos]; \
        xok[q] = p >= 0; \
        xrow[q] = (nf_i64)in_off + (xok[q] ? (nf_i64)(p / x_div) * x_stride : (nf_i64)0); \
    } \
    float acc[NF_GEMM_RM][NF_GEMM_RN]; \
    for (int i = 0; i < NF_GEMM_RM; i++) \
        for (int j = 0; j < NF_GEMM_RN; j++) \
            acc[i][j] = 0.0f; \
    for (int k0 = 0; k0 < in_dim; k0 += NF_GEMM_TK) { \
        const int k = k0 + lk; \
        const int kok = k < in_dim; \
        for (int q = 0; q < NF_GEMM_RM; q++) \
            a_tile[lk * NF_GEMM_AS + lr + NF_GEMM_TX * q] = \
                (wok[q] && kok) ? ELEM(wrow[q], in_dim, 0, k) : 0.0f; \
        for (int q = 0; q < NF_GEMM_RN; q++) \
            b_tile[lk * NF_GEMM_BS + lr + NF_GEMM_TX * q] = \
                (xok[q] && kok) ? input[xrow[q] + k] : 0.0f; \
        barrier(CLK_LOCAL_MEM_FENCE); \
        nf_gemm_stage_fma(a_tile, b_tile, tx, ty, acc); \
        barrier(CLK_LOCAL_MEM_FENCE); \
    } \
    for (int j = 0; j < NF_GEMM_RN; j++) { \
        const int pos = tile * NF_GEMM_TN + ty + NF_GEMM_TX * j; \
        const int p = gemm_token[pos]; \
        if (p < 0) continue; \
        const nf_i64 ybase = (nf_i64)out_off + (nf_i64)p * out_dim; \
        for (int i = 0; i < NF_GEMM_RM; i++) { \
            const int r = m0 + tx + NF_GEMM_TX * i; \
            if (r < out_dim) output[ybase + r] = has_scale ? acc[i][j] * scales[e] : acc[i][j]; \
        } \
    }

#define NF_GEMM_GROUPED_DIRECT(T, ELEM, BS, BB) \
NF_KERNEL void T##_GroupedGemm( \
    const int out_dim, const int in_dim, const int n_tiles, const int x_div, const int x_stride, \
    const int in_off, const int out_off, const int has_scale, const int expert_base, \
    const int expert_count, \
    NF_GLOBAL const int* NF_RESTRICT gemm_token, \
    NF_GLOBAL const int* NF_RESTRICT gemm_slot, \
    NF_GLOBAL const uchar* NF_RESTRICT weight, \
    NF_GLOBAL const float* NF_RESTRICT input, \
    NF_GLOBAL const float* NF_RESTRICT scales, \
    NF_GLOBAL float* NF_RESTRICT output) { \
    NF_GG_PROLOGUE \
    if (e < 0) { \
        if (expert_base == 0) nf_gg_zero(output, gemm_token, tile, out_dim, out_off, m0, tx, ty); \
        return; \
    } \
    if (!nf_gg_in_shard(e, expert_base, expert_count)) return; \
    NF_GG_TILE(ELEM, BS, BB, e - expert_base) \
}

#define NF_GEMM_GROUPED_IND(T, ELEM, BS, BB) \
NF_KERNEL void T##_GroupedGemmInd( \
    const int out_dim, const int in_dim, const int n_tiles, const int x_div, const int x_stride, \
    const int in_off, const int out_off, const int has_scale, \
    NF_GLOBAL const int* NF_RESTRICT gemm_token, \
    NF_GLOBAL const int* NF_RESTRICT gemm_slot, \
    NF_GLOBAL const int* NF_RESTRICT map, \
    NF_GLOBAL const uchar* NF_RESTRICT weight, \
    NF_GLOBAL const float* NF_RESTRICT input, \
    NF_GLOBAL const float* NF_RESTRICT scales, \
    NF_GLOBAL float* NF_RESTRICT output) { \
    NF_GG_PROLOGUE \
    if (e < 0) { \
        nf_gg_zero(output, gemm_token, tile, out_dim, out_off, m0, tx, ty); \
        return; \
    } \
    const int l = map[e]; \
    if (l < 0) return; \
    NF_GG_TILE(ELEM, BS, BB, l) \
}

#define NF_GEMM_GROUPED(T, ELEM, BS, BB) \
    NF_GEMM_GROUPED_DIRECT(T, ELEM, BS, BB) \
    NF_GEMM_GROUPED_IND(T, ELEM, BS, BB)

#endif /* NF_MATMUL_GROUPED_FAMILY_H */

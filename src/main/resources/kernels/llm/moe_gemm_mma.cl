/* moe_gemm_mma.cl - {T}_GroupedGemmMma and {T}_GroupedGemmMmaInd: the tensor-core forms of the MoE grouped GEMM (CUDA only).
 * Requires: common/dialect.h, common/config.h, common/quant_blocks.h, common/iq_tables.h, common/quant_decode_legacy.h, common/quant_decode_k.h, common/quant_decode_iq.h, common/quant_decode_new.h, common/quant_decode_ext.h, common/matmul_tiled_family.h, common/matmul_grouped_family.h, common/matmul_mma_family.h */
/*
 * NF_GEMM_GROUPED_MMA(T, ELEM, BS, BB, OP) defines T##_GroupedGemmMma and T##_GroupedGemmMmaInd with
 * the arguments, semantics, aligned-row layout and launch contract of T##_GroupedGemm and
 * T##_GroupedGemmInd (common/matmul_grouped_family.h):
 *
 *   T_GroupedGemmMma(i out_dim, i in_dim, i n_tiles, i x_div, i x_stride, i in_off, i out_off,
 *                    i has_scale, i expert_base, i expert_count, R<int> gemm_token, R<int> gemm_slot,
 *                    R uchar weight, R float input, R float scales, W float output)
 *   T_GroupedGemmMmaInd(arguments of T_GroupedGemmInd)
 *   global = {ceil(out_dim / 64) * 256, n_tiles}, local = {256, 1}
 *
 * T##_GroupedGemmMmaInd is the same tile over an expert cache bank, with the map indirection and
 * the host-computed misses of T##_GroupedGemmInd.
 *
 * The tile is the tensor-core GEMM's (common/matmul_mma_family.h: 64 weight rows of the tile's
 * expert x the 32 aligned positions, 16-deep stages, 8 warps of two 16x8 mma tiles): every stage
 * decodes its weight elements with the type's ELEM loader and converts them, and the activations of
 * the tile's pairs, to 16-bit operands in shared memory, then multiplies with
 * mma.sync.m16n8k16 fragments loaded by ldmatrix and accumulates in fp32, as vLLM's fused MoE
 * multiplies dequantized expert tiles on the matrix unit. OP picks the operand format: F16 for every
 * quantized type and F16 (fp16 operands, as F16_MatMul_MMA), BF16 for BF16 experts (bfloat16
 * operands, so the weights stay exact). F32 experts have no row and keep the scalar kernel.
 * Results differ from {T}_GroupedGemm by the operand rounding. Global loads are coalesced along the
 * reduction (work-item lid loads depth lid % 16 of rows lid / 16 + 16 i and positions
 * lid / 16 + 16 j).
 *
 * The kernels exist on CUDA sm_80+ only (the build's inventory pass sees them through
 * NF_CUDA_PREPROCESS); the OpenCL program has none, and the Vulkan shader set builds its own
 * cooperative-matrix twins from shaders/llm/moe_gemm_mma.comp. The host (gguf.ops.moe.GpuGroupedRows)
 * selects them on a CUDA or Vulkan device that reports DeviceFeature.COOPERATIVE_MATRIX and keeps
 * {T}_GroupedGemm / {T}_GroupedGemmInd everywhere else.
 */

#if defined(__CUDACC__) && !defined(__HIP__)
#if defined(NF_CUDA_PREPROCESS) || !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800

/* fp16 bits of x, round to nearest even. */
NF_INLINE NF_HALF nf_ggm_op_F16(float x) {
    return __half_as_ushort(__float2half_rn(x));
}

/* bfloat16 bits of x, round to nearest even; NaN stays a quiet NaN. */
NF_INLINE NF_HALF nf_ggm_op_BF16(float x) {
    uint u = __float_as_uint(x);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (NF_HALF)((u >> 16) | 0x40u);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (NF_HALF)(u >> 16);
}

#define NF_GGM_BF16_F16 0
#define NF_GGM_BF16_BF16 1

/*
 * The warp's mma fragments to the outputs of their aligned positions: fragment column c of the tile
 * is position tile * 32 + c, whose pair p = gemm_token[..] (a filler, p < 0, writes nothing) receives
 * output[out_off + p * out_dim + r] for rows r < out_dim, scaled by scales[e] when has_scale.
 */
NF_INLINE void nf_ggm_store(NF_GLOBAL float* NF_RESTRICT output, NF_GLOBAL const int* NF_RESTRICT gemm_token,
                            int tile, int out_dim, int out_off, int m0, int lid, float scale, const float d[8]) {
    const int warp = lid >> 5, wy = warp >> 2, wx = warp & 3, lane = lid & 31;
    const int c = wx * 8 + (lane & 3) * 2;
    const int p0 = gemm_token[tile * NF_MMA_TN + c];
    const int p1 = gemm_token[tile * NF_MMA_TN + c + 1];
    for (int mf = 0; mf < 2; mf++) {
        const int r = m0 + wy * 32 + mf * 16 + (lane >> 2);
        for (int h = 0; h < 2; h++) {
            const int rr = r + 8 * h;
            if (rr >= out_dim) continue;
            if (p0 >= 0) output[(nf_i64)out_off + (nf_i64)p0 * out_dim + rr] = d[mf * 4 + 2 * h] * scale;
            if (p1 >= 0) output[(nf_i64)out_off + (nf_i64)p1 * out_dim + rr] = d[mf * 4 + 2 * h + 1] * scale;
        }
    }
}

/* Zero outputs of a tile of pairs whose id was out of range (the expert_base = 0 dispatch writes them). */
NF_INLINE void nf_ggm_zero(NF_GLOBAL float* NF_RESTRICT output, NF_GLOBAL const int* NF_RESTRICT gemm_token,
                           int tile, int out_dim, int out_off, int m0, int lid) {
    const int lk = lid % NF_MMA_TK, lr = lid / NF_MMA_TK;
    for (int j = 0; j < NF_MMA_RN; j++) {
        const int p = gemm_token[tile * NF_MMA_TN + lr + NF_MMA_TX * j];
        if (p < 0) continue;
        for (int i = 0; i < NF_MMA_RM; i++) {
            const int r = m0 + lk + NF_MMA_TX * i;
            if (r < out_dim) output[(nf_i64)out_off + (nf_i64)p * out_dim + r] = 0.0f;
        }
    }
}

/* Opening of every tensor-core grouped kernel: the 16-bit shared tiles, the tile's expert e and the lanes. */
#define NF_GGM_PROLOGUE \
    __shared__ __align__(16) NF_HALF a_t[NF_MMA_TM * NF_MMA_AS]; \
    __shared__ __align__(16) NF_HALF b_t[NF_MMA_TK * NF_MMA_BS]; \
    const int tile = (int)get_group_id(1); \
    const int e = gemm_slot[tile * NF_MMA_TN]; \
    const int lid = get_local_id(0); \
    const int m0 = (int)get_group_id(0) * NF_MMA_TM; \
    const int lk = lid % NF_MMA_TK, lr = lid / NF_MMA_TK;

/*
 * The tensor-core tile product of a live tile (after NF_GGM_PROLOGUE and the expert guards): local
 * expert WE of the bound weight buffer against the tile's positions, stored scaled by scales[e].
 */
#define NF_GGM_TILE(ELEM, BS, BB, OP, WE) \
    const nf_i64 row_bytes = (nf_i64)(in_dim / (BS)) * (BB); \
    const nf_i64 wbase = (nf_i64)(WE) * out_dim * row_bytes; \
    NF_GLOBAL const uchar* wrow[NF_MMA_RM]; \
    int wok[NF_MMA_RM]; \
    for (int q = 0; q < NF_MMA_RM; q++) { \
        const int r = m0 + lr + NF_MMA_TX * q; \
        wok[q] = r < out_dim; \
        wrow[q] = weight + wbase + (wok[q] ? (nf_i64)r * row_bytes : (nf_i64)0); \
    } \
    nf_i64 xrow[NF_MMA_RN]; \
    int xok[NF_MMA_RN]; \
    for (int q = 0; q < NF_MMA_RN; q++) { \
        const int p = gemm_token[tile * NF_MMA_TN + lr + NF_MMA_TX * q]; \
        xok[q] = p >= 0; \
        xrow[q] = (nf_i64)in_off + (xok[q] ? (nf_i64)(p / x_div) * x_stride : (nf_i64)0); \
    } \
    float d[8]; \
    for (int i = 0; i < 8; i++) d[i] = 0.0f; \
    for (int k0 = 0; k0 < in_dim; k0 += NF_MMA_TK) { \
        const int k = k0 + lk; \
        const int kok = k < in_dim; \
        for (int q = 0; q < NF_MMA_RM; q++) \
            a_t[(lr + NF_MMA_TX * q) * NF_MMA_AS + lk] = \
                nf_ggm_op_##OP((wok[q] && kok) ? ELEM(wrow[q], in_dim, 0, k) : 0.0f); \
        for (int q = 0; q < NF_MMA_RN; q++) \
            b_t[lk * NF_MMA_BS + lr + NF_MMA_TX * q] = \
                nf_ggm_op_##OP((xok[q] && kok) ? input[xrow[q] + k] : 0.0f); \
        barrier(CLK_LOCAL_MEM_FENCE); \
        nf_mma_tensor_stage(a_t, b_t, lid, d, NF_GGM_BF16_##OP); \
        barrier(CLK_LOCAL_MEM_FENCE); \
    } \
    nf_ggm_store(output, gemm_token, tile, out_dim, out_off, m0, lid, has_scale ? scales[e] : 1.0f, d);

#define NF_GEMM_GROUPED_MMA_DIRECT(T, ELEM, BS, BB, OP) \
NF_KERNEL void T##_GroupedGemmMma( \
    const int out_dim, const int in_dim, const int n_tiles, const int x_div, const int x_stride, \
    const int in_off, const int out_off, const int has_scale, const int expert_base, \
    const int expert_count, \
    NF_GLOBAL const int* NF_RESTRICT gemm_token, \
    NF_GLOBAL const int* NF_RESTRICT gemm_slot, \
    NF_GLOBAL const uchar* NF_RESTRICT weight, \
    NF_GLOBAL const float* NF_RESTRICT input, \
    NF_GLOBAL const float* NF_RESTRICT scales, \
    NF_GLOBAL float* NF_RESTRICT output) { \
    NF_GGM_PROLOGUE \
    if (e < 0) { \
        if (expert_base == 0) nf_ggm_zero(output, gemm_token, tile, out_dim, out_off, m0, lid); \
        return; \
    } \
    if (!nf_gg_in_shard(e, expert_base, expert_count)) return; \
    NF_GGM_TILE(ELEM, BS, BB, OP, e - expert_base) \
}

#define NF_GEMM_GROUPED_MMA_IND(T, ELEM, BS, BB, OP) \
NF_KERNEL void T##_GroupedGemmMmaInd( \
    const int out_dim, const int in_dim, const int n_tiles, const int x_div, const int x_stride, \
    const int in_off, const int out_off, const int has_scale, \
    NF_GLOBAL const int* NF_RESTRICT gemm_token, \
    NF_GLOBAL const int* NF_RESTRICT gemm_slot, \
    NF_GLOBAL const int* NF_RESTRICT map, \
    NF_GLOBAL const uchar* NF_RESTRICT weight, \
    NF_GLOBAL const float* NF_RESTRICT input, \
    NF_GLOBAL const float* NF_RESTRICT scales, \
    NF_GLOBAL float* NF_RESTRICT output) { \
    NF_GGM_PROLOGUE \
    if (e < 0) { \
        nf_ggm_zero(output, gemm_token, tile, out_dim, out_off, m0, lid); \
        return; \
    } \
    const int l = map[e]; \
    if (l < 0) return; \
    NF_GGM_TILE(ELEM, BS, BB, OP, l) \
}

#define NF_GEMM_GROUPED_MMA(T, ELEM, BS, BB, OP) \
    NF_GEMM_GROUPED_MMA_DIRECT(T, ELEM, BS, BB, OP) \
    NF_GEMM_GROUPED_MMA_IND(T, ELEM, BS, BB, OP)

/* ---- dense (F32 keeps the scalar F32_GroupedGemm) ---- */
NF_GEMM_GROUPED_MMA(F16,  nf_gemm_f16_elem,  1, 2, F16)
NF_GEMM_GROUPED_MMA(BF16, nf_gemm_bf16_elem, 1, 2, BF16)

/* ---- legacy 32-element blocks, MXFP4 and IQ4_XS ---- */
NF_GEMM_GROUPED_MMA(Q4_0,   Q4_0_decode_elem,   Q4_0_BLOCK_SIZE,   Q4_0_BLOCK_BYTES,   F16)
NF_GEMM_GROUPED_MMA(Q4_1,   Q4_1_decode_elem,   Q4_1_BLOCK_SIZE,   Q4_1_BLOCK_BYTES,   F16)
NF_GEMM_GROUPED_MMA(Q5_0,   Q5_0_decode_elem,   Q5_0_BLOCK_SIZE,   Q5_0_BLOCK_BYTES,   F16)
NF_GEMM_GROUPED_MMA(Q5_1,   Q5_1_decode_elem,   Q5_1_BLOCK_SIZE,   Q5_1_BLOCK_BYTES,   F16)
NF_GEMM_GROUPED_MMA(Q8_0,   Q8_0_decode_elem,   Q8_0_BLOCK_SIZE,   Q8_0_BLOCK_BYTES,   F16)
NF_GEMM_GROUPED_MMA(IQ4_NL, IQ4_NL_decode_elem, IQ4_NL_BLOCK_SIZE, IQ4_NL_BLOCK_BYTES, F16)
NF_GEMM_GROUPED_MMA(IQ4_XS, IQ4_XS_decode_elem, IQ4_XS_BLOCK_SIZE, IQ4_XS_BLOCK_BYTES, F16)
NF_GEMM_GROUPED_MMA(MXFP4,  MXFP4_decode_elem,  MXFP4_BLOCK_SIZE,  MXFP4_BLOCK_BYTES,  F16)

/* ---- K-quants ---- */
NF_GEMM_GROUPED_MMA(Q2_K, Q2_K_decode_elem, Q2_K_BLOCK_SIZE, Q2_K_BLOCK_BYTES, F16)
NF_GEMM_GROUPED_MMA(Q3_K, Q3_K_decode_elem, Q3_K_BLOCK_SIZE, Q3_K_BLOCK_BYTES, F16)
NF_GEMM_GROUPED_MMA(Q4_K, Q4_K_decode_elem, Q4_K_BLOCK_SIZE, Q4_K_BLOCK_BYTES, F16)
NF_GEMM_GROUPED_MMA(Q5_K, Q5_K_decode_elem, Q5_K_BLOCK_SIZE, Q5_K_BLOCK_BYTES, F16)
NF_GEMM_GROUPED_MMA(Q6_K, Q6_K_decode_elem, Q6_K_BLOCK_SIZE, Q6_K_BLOCK_BYTES, F16)
NF_GEMM_GROUPED_MMA(Q8_K, Q8_K_decode_elem, Q8_K_BLOCK_SIZE, Q8_K_BLOCK_BYTES, F16)

/* ---- I-quants ---- */
NF_GEMM_GROUPED_MMA(IQ3_XXS, IQ3_XXS_decode_elem, IQ3_XXS_BLOCK_SIZE, IQ3_XXS_BLOCK_BYTES, F16)
NF_GEMM_GROUPED_MMA(IQ3_S,   IQ3_S_decode_elem,   IQ3_S_BLOCK_SIZE,   IQ3_S_BLOCK_BYTES,   F16)
NF_GEMM_GROUPED_MMA(IQ3_K,   IQ3_K_decode_elem,   IQ3_K_BLOCK_SIZE,   IQ3_K_BLOCK_BYTES,   F16)
NF_GEMM_GROUPED_MMA(IQ3_KS,  IQ3_KS_decode_elem,  IQ3_KS_BLOCK_SIZE,  IQ3_KS_BLOCK_BYTES,  F16)
NF_GEMM_GROUPED_MMA(IQ2_XXS, IQ2_XXS_decode_elem, IQ2_XXS_BLOCK_SIZE, IQ2_XXS_BLOCK_BYTES, F16)
NF_GEMM_GROUPED_MMA(IQ2_XS,  IQ2_XS_decode_elem,  IQ2_XS_BLOCK_SIZE,  IQ2_XS_BLOCK_BYTES,  F16)
NF_GEMM_GROUPED_MMA(IQ2_S,   IQ2_S_decode_elem,   IQ2_S_BLOCK_SIZE,   IQ2_S_BLOCK_BYTES,   F16)
NF_GEMM_GROUPED_MMA(IQ1_S,   IQ1_S_decode_elem,   IQ1_S_BLOCK_SIZE,   IQ1_S_BLOCK_BYTES,   F16)
NF_GEMM_GROUPED_MMA(IQ1_M,   IQ1_M_decode_elem,   IQ1_M_BLOCK_SIZE,   IQ1_M_BLOCK_BYTES,   F16)

/* ---- ternary, NVFP4 and low-bit ---- */
NF_GEMM_GROUPED_MMA(TQ1_0, TQ1_0_decode_elem, TQ1_0_BLOCK_SIZE, TQ1_0_BLOCK_BYTES, F16)
NF_GEMM_GROUPED_MMA(TQ2_0, TQ2_0_decode_elem, TQ2_0_BLOCK_SIZE, TQ2_0_BLOCK_BYTES, F16)
NF_GEMM_GROUPED_MMA(NVFP4, NVFP4_decode_elem, NVFP4_BLOCK_SIZE, NVFP4_BLOCK_BYTES, F16)
NF_GEMM_GROUPED_MMA(Q1_0,  Q1_0_decode_elem,  Q1_0_BLOCK_SIZE,  Q1_0_BLOCK_BYTES,  F16)
NF_GEMM_GROUPED_MMA(Q2_0,  Q2_0_decode_elem,  Q2_0_BLOCK_SIZE,  Q2_0_BLOCK_BYTES,  F16)

/* ---- NeuralFree extension types ---- */
NF_GEMM_GROUPED_MMA(F8_E5M2,      F8_E5M2_decode_elem,      F8_E5M2_BLOCK_SIZE,      F8_E5M2_BLOCK_BYTES,      F16)
NF_GEMM_GROUPED_MMA(F8_E4M3,      F8_E4M3_decode_elem,      F8_E4M3_BLOCK_SIZE,      F8_E4M3_BLOCK_BYTES,      F16)
NF_GEMM_GROUPED_MMA(F8_E4M3_B128, F8_E4M3_B128_decode_elem, F8_E4M3_B128_BLOCK_SIZE, F8_E4M3_B128_BLOCK_BYTES, F16)
NF_GEMM_GROUPED_MMA(F8_E4M3_B32,  F8_E4M3_B32_decode_elem,  F8_E4M3_B32_BLOCK_SIZE,  F8_E4M3_B32_BLOCK_BYTES,  F16)
NF_GEMM_GROUPED_MMA(MXFP8,        MXFP8_decode_elem,        MXFP8_BLOCK_SIZE,        MXFP8_BLOCK_BYTES,        F16)
NF_GEMM_GROUPED_MMA(MXFP6_E2M3,   MXFP6_E2M3_decode_elem,   MXFP6_E2M3_BLOCK_SIZE,   MXFP6_E2M3_BLOCK_BYTES,   F16)
NF_GEMM_GROUPED_MMA(MXFP6_E3M2,   MXFP6_E3M2_decode_elem,   MXFP6_E3M2_BLOCK_SIZE,   MXFP6_E3M2_BLOCK_BYTES,   F16)
NF_GEMM_GROUPED_MMA(I8_B128,      I8_B128_decode_elem,      I8_B128_BLOCK_SIZE,      I8_B128_BLOCK_BYTES,      F16)
NF_GEMM_GROUPED_MMA(NF4,          NF4_decode_elem,          NF4_BLOCK_SIZE,          NF4_BLOCK_BYTES,          F16)
NF_GEMM_GROUPED_MMA(FP4_BNB,      FP4_BNB_decode_elem,      FP4_BNB_BLOCK_SIZE,      FP4_BNB_BLOCK_BYTES,      F16)

#endif /* NF_CUDA_PREPROCESS, or mma.sync.m16n8k16 on compute capability 8.0+ */
#endif /* __CUDACC__ and not __HIP__: the OpenCL program, the HIP build and the Vulkan shader set define no mma kernels */

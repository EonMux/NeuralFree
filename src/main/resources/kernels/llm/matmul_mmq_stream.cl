/* matmul_mmq_stream.cl - T_MatMul_MMQ_StreamK, the stream-k form of the MMQ GEMM's tensor-core tile,
 * and LLM_MMQ_StreamK_Fixup, which adds its split tiles' partial sums ([cuda]: CUDA sm_80+ only).
 * Requires: common/dialect.h, common/config.h, common/quant_blocks.h, common/iq_tables.h, common/quant_decode_legacy.h, common/quant_decode_k.h, common/matmul_mmq_config.h, common/matmul_mmq_family.h, common/matmul_mmq_types.h */
/*
 * The stream-k kernels run the plain T_MatMul_MMQ tile of every MMQ weight type (kernels/llm/matmul_mmq.cl)
 * over the launch's sequence of (tile, chunk) steps split evenly across the device's resident blocks
 * (common/matmul_mmq_family.h, NF_MMQ_TC_STREAMK_NAMED); the host takes them for a staged launch of at
 * least one device tile of tokens whose tile grid would leave the last wave of blocks partly empty, and
 * keeps the plain grid under batch invariance (a split tile's sum depends on the grid). The file is
 * CUDA-only, and the kernels exist on the sm_80+ segments the tensor tile compiles for.
 */

#if NF_MMQ_TENSOR
NF_MMQ_GEMM_STREAMK(Q4_0, Q4_0_BLOCK_SIZE, Q4_0_BLOCK_BYTES, 32, Q4_0_MMQ_CODE, Q4_0_MMQ_SCALE, Q4_0_MMQ_OFF)
NF_MMQ_GEMM_STREAMK(Q4_1, Q4_1_BLOCK_SIZE, Q4_1_BLOCK_BYTES, 32, Q4_1_MMQ_CODE, Q4_1_MMQ_SCALE, Q4_1_MMQ_OFF)
NF_MMQ_GEMM_STREAMK(Q5_0, Q5_0_BLOCK_SIZE, Q5_0_BLOCK_BYTES, 32, Q5_0_MMQ_CODE, Q5_0_MMQ_SCALE, Q5_0_MMQ_OFF)
NF_MMQ_GEMM_STREAMK(Q5_1, Q5_1_BLOCK_SIZE, Q5_1_BLOCK_BYTES, 32, Q5_1_MMQ_CODE, Q5_1_MMQ_SCALE, Q5_1_MMQ_OFF)
NF_MMQ_GEMM_STREAMK(Q8_0, Q8_0_BLOCK_SIZE, Q8_0_BLOCK_BYTES, 32, Q8_0_MMQ_CODE, Q8_0_MMQ_SCALE, Q8_0_MMQ_OFF)
NF_MMQ_GEMM_STREAMK(I8_B128, I8_B128_BLOCK_SIZE, I8_B128_BLOCK_BYTES, 32, I8_B128_MMQ_CODE, I8_B128_MMQ_SCALE, I8_B128_MMQ_OFF)
NF_MMQ_GEMM_STREAMK(Q2_K, Q2_K_BLOCK_SIZE, Q2_K_BLOCK_BYTES, 16, Q2_K_MMQ_CODE, Q2_K_MMQ_SCALE, Q2_K_MMQ_OFF)
NF_MMQ_GEMM_STREAMK(Q3_K, Q3_K_BLOCK_SIZE, Q3_K_BLOCK_BYTES, 16, Q3_K_MMQ_CODE, Q3_K_MMQ_SCALE, Q3_K_MMQ_OFF)
NF_MMQ_GEMM_STREAMK(Q4_K, Q4_K_BLOCK_SIZE, Q4_K_BLOCK_BYTES, 32, Q4_K_MMQ_CODE, Q4_K_MMQ_SCALE, Q4_K_MMQ_OFF)
NF_MMQ_GEMM_STREAMK(Q5_K, Q5_K_BLOCK_SIZE, Q5_K_BLOCK_BYTES, 32, Q5_K_MMQ_CODE, Q5_K_MMQ_SCALE, Q5_K_MMQ_OFF)
NF_MMQ_GEMM_STREAMK(Q6_K, Q6_K_BLOCK_SIZE, Q6_K_BLOCK_BYTES, 16, Q6_K_MMQ_CODE, Q6_K_MMQ_SCALE, Q6_K_MMQ_OFF)
NF_MMQ_GEMM_STREAMK(IQ3_XXS, IQ3_XXS_BLOCK_SIZE, IQ3_XXS_BLOCK_BYTES, 32, IQ3_XXS_MMQ_CODE, IQ3_XXS_MMQ_SCALE, IQ3_XXS_MMQ_OFF)
NF_MMQ_GEMM_STREAMK(IQ3_S, IQ3_S_BLOCK_SIZE, IQ3_S_BLOCK_BYTES, 32, IQ3_S_MMQ_CODE, IQ3_S_MMQ_SCALE, IQ3_S_MMQ_OFF)
NF_MMQ_GEMM_STREAMK(IQ2_XXS, IQ2_XXS_BLOCK_SIZE, IQ2_XXS_BLOCK_BYTES, 32, IQ2_XXS_MMQ_CODE, IQ2_XXS_MMQ_SCALE, IQ2_XXS_MMQ_OFF)
NF_MMQ_GEMM_STREAMK(IQ2_XS, IQ2_XS_BLOCK_SIZE, IQ2_XS_BLOCK_BYTES, 16, IQ2_XS_MMQ_CODE, IQ2_XS_MMQ_SCALE, IQ2_XS_MMQ_OFF)
NF_MMQ_GEMM_STREAMK(IQ2_S, IQ2_S_BLOCK_SIZE, IQ2_S_BLOCK_BYTES, 16, IQ2_S_MMQ_CODE, IQ2_S_MMQ_SCALE, IQ2_S_MMQ_OFF)
NF_MMQ_GEMM_STREAMK(IQ1_S, IQ1_S_BLOCK_SIZE, IQ1_S_BLOCK_BYTES, 32, IQ1_S_MMQ_CODE, IQ1_S_MMQ_SCALE, IQ1_S_MMQ_OFF)
NF_MMQ_GEMM_STREAMK(IQ4_NL, IQ4_NL_BLOCK_SIZE, IQ4_NL_BLOCK_BYTES, 32, IQ4_NL_MMQ_CODE, IQ4_NL_MMQ_SCALE, IQ4_NL_MMQ_OFF)
NF_MMQ_GEMM_STREAMK(IQ4_XS, IQ4_XS_BLOCK_SIZE, IQ4_XS_BLOCK_BYTES, 32, IQ4_XS_MMQ_CODE, IQ4_XS_MMQ_SCALE, IQ4_XS_MMQ_OFF)

/* Outputs of a tile one work-item of the fixup adds: the tile's TM x TN outputs split into
 * NF_MMQ_TM * NF_MMQ_TN / (NF_MMQ_WG * NF_MMQ_SK_FIX_E) slices of contiguous plane elements, one
 * work-group each, so a split tile's planes are read with that many loads in flight per element. */
#define NF_MMQ_SK_FIX_E 4
#define NF_MMQ_SK_FIX_SLICES (NF_MMQ_TM * NF_MMQ_TN / (NF_MMQ_WG * NF_MMQ_SK_FIX_E))

/*
 * The stream-k fixup: global {nb * NF_MMQ_WG, NF_MMQ_SK_FIX_SLICES}, local {NF_MMQ_WG, 1} over the nb
 * blocks of the stream-k launch with the same M, K, N. Work-group (b, s) acts when block b's first
 * segment finished a tile it did not start: it sums slice s of the fixup planes of the preceding blocks
 * that ran the rest of that tile, nearest first (each one's last segment lies in the tile), and adds the
 * sum into the tile's outputs; the work-groups of the other blocks return at once.
 */
NF_KERNEL __launch_bounds__(NF_MMQ_WG) void LLM_MMQ_StreamK_Fixup(
    const int M, const int K, const int N, const int y_stride, const int y_off,
    NF_GLOBAL const float* NF_RESTRICT fixup,
    NF_GLOBAL float* NF_RESTRICT y) {
    const int lid = get_local_id(0);
    const int kc = K / NF_MMQ_TK;
    const int ntj = (N + NF_MMQ_TN - 1) / NF_MMQ_TN;
    const int total = (M + NF_MMQ_TM - 1) / NF_MMQ_TM * ntj * kc;
    const int b = (int)get_group_id(0);
    const int nb = (int)get_num_groups(0);
    const int k0 = nf_mmq_streamk_start(b, nb, total);
    const int k1 = nf_mmq_streamk_start(b + 1, nb, total);
    const int tile = k0 / kc;
    if (k0 == k1 || k0 % kc == 0 || (tile == k1 / kc && k1 % kc != 0)) {
        return;
    }
    const int e0 = (int)get_group_id(1) * NF_MMQ_SK_FIX_E * NF_MMQ_WG + lid;
    float sum[NF_MMQ_SK_FIX_E];
    for (int i = 0; i < NF_MMQ_SK_FIX_E; i++) {
        sum[i] = 0.0f;
    }
    for (int p = b - 1, stop = k0; p >= 0; p--) {
        const int pk = nf_mmq_streamk_start(p, nb, total);
        if (pk == stop) {
            continue;
        }
        const NF_GLOBAL float* plane = fixup + (nf_i64)p * NF_MMQ_TM * NF_MMQ_TN + e0;
        for (int i = 0; i < NF_MMQ_SK_FIX_E; i++) {
            sum[i] += plane[i * NF_MMQ_WG];
        }
        if (pk % kc == 0 || pk / kc < tile) {
            break;
        }
        stop = pk;
    }
    const int m0 = tile / ntj * NF_MMQ_TM;
    const int n0 = tile % ntj * NF_MMQ_TN;
    for (int i = 0; i < NF_MMQ_SK_FIX_E; i++) {
        const int e = e0 + i * NF_MMQ_WG;
        const int r = m0 + e % NF_MMQ_TM;
        const int c = n0 + e / NF_MMQ_TM;
        if (r < M && c < N) {
            y[(nf_i64)y_off + (nf_i64)c * y_stride + r] += sum[i];
        }
    }
}
#endif /* NF_MMQ_TENSOR */

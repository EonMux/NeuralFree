/* matmul_cm.cl - scalar twins of the quantized-weight cooperative-matrix GEMM: the same body and
 * numerics as the tiled family, instantiated as {T}_MatMul_CM and the split-K twin
 * {T}_MatMul_CM_KSplit for the llama.cpp MMQ weight types and the int8 checkpoint type I8_B128.
 * Requires: common/dialect.h, common/config.h, common/quant_blocks.h, common/iq_tables.h,
 * common/quant_decode_legacy.h, common/quant_decode_k.h, common/quant_decode_iq.h,
 * common/quant_decode_ext.h, common/matmul_tiled_family.h */
/*
 * The Vulkan backend serves {T}_MatMul_CM from shaders/llm/matmul_mmq_cm.comp (the tensor-core form
 * that dequantizes the weight into fp16 tiles) on devices with VK_KHR_cooperative_matrix; this file
 * provides the portable rows of the same kernel names so every backend's inventory is complete, as
 * matmul_mma.cl does for F16_MatMul_MMA. The dispatch selects {T}_MatMul_CM only where the
 * cooperative-matrix form runs; elsewhere the int8 MMQ GEMM (matmul_mmq.cl) or the tiled GEMM serves
 * the product. Arguments, semantics and launch contract: common/matmul_tiled_family.h (the
 * NF_GEMM_TILED_AS rows share the body, the dequantizing fp32 arithmetic and the
 * global = {ceil(M / 64) * 256, ceil(N / 32)} geometry; the NF_GEMM_TILED_KSPLIT_AS rows are the
 * split-K twins whose slice planes LLM_SplitK_Reduce sums).
 */
NF_GEMM_TILED_AS(Q4_0, MatMul_CM, Q4_0_decode_elem, Q4_0_BLOCK_SIZE, Q4_0_BLOCK_BYTES)
NF_GEMM_TILED_KSPLIT_AS(Q4_0, MatMul_CM, Q4_0_decode_elem, Q4_0_BLOCK_SIZE, Q4_0_BLOCK_BYTES)
NF_GEMM_TILED_AS(Q4_1, MatMul_CM, Q4_1_decode_elem, Q4_1_BLOCK_SIZE, Q4_1_BLOCK_BYTES)
NF_GEMM_TILED_KSPLIT_AS(Q4_1, MatMul_CM, Q4_1_decode_elem, Q4_1_BLOCK_SIZE, Q4_1_BLOCK_BYTES)
NF_GEMM_TILED_AS(Q5_0, MatMul_CM, Q5_0_decode_elem, Q5_0_BLOCK_SIZE, Q5_0_BLOCK_BYTES)
NF_GEMM_TILED_KSPLIT_AS(Q5_0, MatMul_CM, Q5_0_decode_elem, Q5_0_BLOCK_SIZE, Q5_0_BLOCK_BYTES)
NF_GEMM_TILED_AS(Q5_1, MatMul_CM, Q5_1_decode_elem, Q5_1_BLOCK_SIZE, Q5_1_BLOCK_BYTES)
NF_GEMM_TILED_KSPLIT_AS(Q5_1, MatMul_CM, Q5_1_decode_elem, Q5_1_BLOCK_SIZE, Q5_1_BLOCK_BYTES)
NF_GEMM_TILED_AS(Q8_0, MatMul_CM, Q8_0_decode_elem, Q8_0_BLOCK_SIZE, Q8_0_BLOCK_BYTES)
NF_GEMM_TILED_KSPLIT_AS(Q8_0, MatMul_CM, Q8_0_decode_elem, Q8_0_BLOCK_SIZE, Q8_0_BLOCK_BYTES)
NF_GEMM_TILED_AS(Q2_K, MatMul_CM, Q2_K_decode_elem, Q2_K_BLOCK_SIZE, Q2_K_BLOCK_BYTES)
NF_GEMM_TILED_KSPLIT_AS(Q2_K, MatMul_CM, Q2_K_decode_elem, Q2_K_BLOCK_SIZE, Q2_K_BLOCK_BYTES)
NF_GEMM_TILED_AS(Q3_K, MatMul_CM, Q3_K_decode_elem, Q3_K_BLOCK_SIZE, Q3_K_BLOCK_BYTES)
NF_GEMM_TILED_KSPLIT_AS(Q3_K, MatMul_CM, Q3_K_decode_elem, Q3_K_BLOCK_SIZE, Q3_K_BLOCK_BYTES)
NF_GEMM_TILED_AS(Q4_K, MatMul_CM, Q4_K_decode_elem, Q4_K_BLOCK_SIZE, Q4_K_BLOCK_BYTES)
NF_GEMM_TILED_KSPLIT_AS(Q4_K, MatMul_CM, Q4_K_decode_elem, Q4_K_BLOCK_SIZE, Q4_K_BLOCK_BYTES)
NF_GEMM_TILED_AS(Q5_K, MatMul_CM, Q5_K_decode_elem, Q5_K_BLOCK_SIZE, Q5_K_BLOCK_BYTES)
NF_GEMM_TILED_KSPLIT_AS(Q5_K, MatMul_CM, Q5_K_decode_elem, Q5_K_BLOCK_SIZE, Q5_K_BLOCK_BYTES)
NF_GEMM_TILED_AS(Q6_K, MatMul_CM, Q6_K_decode_elem, Q6_K_BLOCK_SIZE, Q6_K_BLOCK_BYTES)
NF_GEMM_TILED_KSPLIT_AS(Q6_K, MatMul_CM, Q6_K_decode_elem, Q6_K_BLOCK_SIZE, Q6_K_BLOCK_BYTES)
NF_GEMM_TILED_AS(IQ1_S, MatMul_CM, IQ1_S_decode_elem, IQ1_S_BLOCK_SIZE, IQ1_S_BLOCK_BYTES)
NF_GEMM_TILED_KSPLIT_AS(IQ1_S, MatMul_CM, IQ1_S_decode_elem, IQ1_S_BLOCK_SIZE, IQ1_S_BLOCK_BYTES)
NF_GEMM_TILED_AS(IQ2_XXS, MatMul_CM, IQ2_XXS_decode_elem, IQ2_XXS_BLOCK_SIZE, IQ2_XXS_BLOCK_BYTES)
NF_GEMM_TILED_KSPLIT_AS(IQ2_XXS, MatMul_CM, IQ2_XXS_decode_elem, IQ2_XXS_BLOCK_SIZE, IQ2_XXS_BLOCK_BYTES)
NF_GEMM_TILED_AS(IQ2_XS, MatMul_CM, IQ2_XS_decode_elem, IQ2_XS_BLOCK_SIZE, IQ2_XS_BLOCK_BYTES)
NF_GEMM_TILED_KSPLIT_AS(IQ2_XS, MatMul_CM, IQ2_XS_decode_elem, IQ2_XS_BLOCK_SIZE, IQ2_XS_BLOCK_BYTES)
NF_GEMM_TILED_AS(IQ2_S, MatMul_CM, IQ2_S_decode_elem, IQ2_S_BLOCK_SIZE, IQ2_S_BLOCK_BYTES)
NF_GEMM_TILED_KSPLIT_AS(IQ2_S, MatMul_CM, IQ2_S_decode_elem, IQ2_S_BLOCK_SIZE, IQ2_S_BLOCK_BYTES)
NF_GEMM_TILED_AS(IQ3_XXS, MatMul_CM, IQ3_XXS_decode_elem, IQ3_XXS_BLOCK_SIZE, IQ3_XXS_BLOCK_BYTES)
NF_GEMM_TILED_KSPLIT_AS(IQ3_XXS, MatMul_CM, IQ3_XXS_decode_elem, IQ3_XXS_BLOCK_SIZE, IQ3_XXS_BLOCK_BYTES)
NF_GEMM_TILED_AS(IQ3_S, MatMul_CM, IQ3_S_decode_elem, IQ3_S_BLOCK_SIZE, IQ3_S_BLOCK_BYTES)
NF_GEMM_TILED_KSPLIT_AS(IQ3_S, MatMul_CM, IQ3_S_decode_elem, IQ3_S_BLOCK_SIZE, IQ3_S_BLOCK_BYTES)
NF_GEMM_TILED_AS(IQ4_NL, MatMul_CM, IQ4_NL_decode_elem, IQ4_NL_BLOCK_SIZE, IQ4_NL_BLOCK_BYTES)
NF_GEMM_TILED_KSPLIT_AS(IQ4_NL, MatMul_CM, IQ4_NL_decode_elem, IQ4_NL_BLOCK_SIZE, IQ4_NL_BLOCK_BYTES)
NF_GEMM_TILED_AS(IQ4_XS, MatMul_CM, IQ4_XS_decode_elem, IQ4_XS_BLOCK_SIZE, IQ4_XS_BLOCK_BYTES)
NF_GEMM_TILED_KSPLIT_AS(IQ4_XS, MatMul_CM, IQ4_XS_decode_elem, IQ4_XS_BLOCK_SIZE, IQ4_XS_BLOCK_BYTES)
NF_GEMM_TILED_AS(I8_B128, MatMul_CM, I8_B128_decode_elem, I8_B128_BLOCK_SIZE, I8_B128_BLOCK_BYTES)
NF_GEMM_TILED_KSPLIT_AS(I8_B128, MatMul_CM, I8_B128_decode_elem, I8_B128_BLOCK_SIZE, I8_B128_BLOCK_BYTES)

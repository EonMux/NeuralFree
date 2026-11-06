/* moe_gemm.cl - matmul_grouped_family instantiations for every weight type (MoE grouped GEMM and its expert-cache form). Requires: common/dialect.h, common/config.h, common/quant_blocks.h, common/iq_tables.h, common/quant_decode_legacy.h, common/quant_decode_k.h, common/quant_decode_iq.h, common/quant_decode_new.h, common/quant_decode_ext.h, common/matmul_tiled_family.h, common/matmul_grouped_family.h */
/*
 * Each NF_GEMM_GROUPED line generates T_GroupedGemm and T_GroupedGemmInd (the expert-cache bank
 * form) with the arguments, aligned-row layout and launch contract of
 * common/matmul_grouped_family.h, for the 39 weight types with expert kernels
 * (MoeKernels.weightTypes): the same set and element loaders as matmul_tiled.cl, without the _Hw
 * hardware-FP8 rows, which no expert tensor uses. in_dim must be a multiple of the type's block size
 * (32, 64, 128 or 256 elements); the host falls back to {T}_BatchMatMulId (or
 * {T}_BatchMatMulIdInd) when it is not.
 */

/* ---- dense ---- */
NF_GEMM_GROUPED(F32,  nf_gemm_f32_elem,  1, 4)
NF_GEMM_GROUPED(F16,  nf_gemm_f16_elem,  1, 2)
NF_GEMM_GROUPED(BF16, nf_gemm_bf16_elem, 1, 2)

/* ---- legacy 32-element blocks, MXFP4 and IQ4_XS (common/quant_decode_legacy.h) ---- */
NF_GEMM_GROUPED(Q4_0,   Q4_0_decode_elem,   Q4_0_BLOCK_SIZE,   Q4_0_BLOCK_BYTES)
NF_GEMM_GROUPED(Q4_1,   Q4_1_decode_elem,   Q4_1_BLOCK_SIZE,   Q4_1_BLOCK_BYTES)
NF_GEMM_GROUPED(Q5_0,   Q5_0_decode_elem,   Q5_0_BLOCK_SIZE,   Q5_0_BLOCK_BYTES)
NF_GEMM_GROUPED(Q5_1,   Q5_1_decode_elem,   Q5_1_BLOCK_SIZE,   Q5_1_BLOCK_BYTES)
NF_GEMM_GROUPED(Q8_0,   Q8_0_decode_elem,   Q8_0_BLOCK_SIZE,   Q8_0_BLOCK_BYTES)
NF_GEMM_GROUPED(IQ4_NL, IQ4_NL_decode_elem, IQ4_NL_BLOCK_SIZE, IQ4_NL_BLOCK_BYTES)
NF_GEMM_GROUPED(IQ4_XS, IQ4_XS_decode_elem, IQ4_XS_BLOCK_SIZE, IQ4_XS_BLOCK_BYTES)
NF_GEMM_GROUPED(MXFP4,  MXFP4_decode_elem,  MXFP4_BLOCK_SIZE,  MXFP4_BLOCK_BYTES)

/* ---- K-quants (common/quant_decode_k.h) ---- */
NF_GEMM_GROUPED(Q2_K, Q2_K_decode_elem, Q2_K_BLOCK_SIZE, Q2_K_BLOCK_BYTES)
NF_GEMM_GROUPED(Q3_K, Q3_K_decode_elem, Q3_K_BLOCK_SIZE, Q3_K_BLOCK_BYTES)
NF_GEMM_GROUPED(Q4_K, Q4_K_decode_elem, Q4_K_BLOCK_SIZE, Q4_K_BLOCK_BYTES)
NF_GEMM_GROUPED(Q5_K, Q5_K_decode_elem, Q5_K_BLOCK_SIZE, Q5_K_BLOCK_BYTES)
NF_GEMM_GROUPED(Q6_K, Q6_K_decode_elem, Q6_K_BLOCK_SIZE, Q6_K_BLOCK_BYTES)
NF_GEMM_GROUPED(Q8_K, Q8_K_decode_elem, Q8_K_BLOCK_SIZE, Q8_K_BLOCK_BYTES)

/* ---- I-quants (common/quant_decode_iq.h) ---- */
NF_GEMM_GROUPED(IQ3_XXS, IQ3_XXS_decode_elem, IQ3_XXS_BLOCK_SIZE, IQ3_XXS_BLOCK_BYTES)
NF_GEMM_GROUPED(IQ3_S,   IQ3_S_decode_elem,   IQ3_S_BLOCK_SIZE,   IQ3_S_BLOCK_BYTES)
NF_GEMM_GROUPED(IQ3_K,   IQ3_K_decode_elem,   IQ3_K_BLOCK_SIZE,   IQ3_K_BLOCK_BYTES)
NF_GEMM_GROUPED(IQ3_KS,  IQ3_KS_decode_elem,  IQ3_KS_BLOCK_SIZE,  IQ3_KS_BLOCK_BYTES)
NF_GEMM_GROUPED(IQ2_XXS, IQ2_XXS_decode_elem, IQ2_XXS_BLOCK_SIZE, IQ2_XXS_BLOCK_BYTES)
NF_GEMM_GROUPED(IQ2_XS,  IQ2_XS_decode_elem,  IQ2_XS_BLOCK_SIZE,  IQ2_XS_BLOCK_BYTES)
NF_GEMM_GROUPED(IQ2_S,   IQ2_S_decode_elem,   IQ2_S_BLOCK_SIZE,   IQ2_S_BLOCK_BYTES)
NF_GEMM_GROUPED(IQ1_S,   IQ1_S_decode_elem,   IQ1_S_BLOCK_SIZE,   IQ1_S_BLOCK_BYTES)
NF_GEMM_GROUPED(IQ1_M,   IQ1_M_decode_elem,   IQ1_M_BLOCK_SIZE,   IQ1_M_BLOCK_BYTES)

/* ---- ternary, NVFP4 and low-bit (common/quant_decode_new.h) ---- */
NF_GEMM_GROUPED(TQ1_0, TQ1_0_decode_elem, TQ1_0_BLOCK_SIZE, TQ1_0_BLOCK_BYTES)
NF_GEMM_GROUPED(TQ2_0, TQ2_0_decode_elem, TQ2_0_BLOCK_SIZE, TQ2_0_BLOCK_BYTES)
NF_GEMM_GROUPED(NVFP4, NVFP4_decode_elem, NVFP4_BLOCK_SIZE, NVFP4_BLOCK_BYTES)
NF_GEMM_GROUPED(Q1_0,  Q1_0_decode_elem,  Q1_0_BLOCK_SIZE,  Q1_0_BLOCK_BYTES)
NF_GEMM_GROUPED(Q2_0,  Q2_0_decode_elem,  Q2_0_BLOCK_SIZE,  Q2_0_BLOCK_BYTES)

/* ---- NeuralFree extension types (common/quant_decode_ext.h) ---- */
NF_GEMM_GROUPED(F8_E5M2,      F8_E5M2_decode_elem,      F8_E5M2_BLOCK_SIZE,      F8_E5M2_BLOCK_BYTES)
NF_GEMM_GROUPED(F8_E4M3,      F8_E4M3_decode_elem,      F8_E4M3_BLOCK_SIZE,      F8_E4M3_BLOCK_BYTES)
NF_GEMM_GROUPED(F8_E4M3_B128, F8_E4M3_B128_decode_elem, F8_E4M3_B128_BLOCK_SIZE, F8_E4M3_B128_BLOCK_BYTES)
NF_GEMM_GROUPED(F8_E4M3_B32,  F8_E4M3_B32_decode_elem,  F8_E4M3_B32_BLOCK_SIZE,  F8_E4M3_B32_BLOCK_BYTES)
NF_GEMM_GROUPED(MXFP8,        MXFP8_decode_elem,        MXFP8_BLOCK_SIZE,        MXFP8_BLOCK_BYTES)
NF_GEMM_GROUPED(MXFP6_E2M3,   MXFP6_E2M3_decode_elem,   MXFP6_E2M3_BLOCK_SIZE,   MXFP6_E2M3_BLOCK_BYTES)
NF_GEMM_GROUPED(MXFP6_E3M2,   MXFP6_E3M2_decode_elem,   MXFP6_E3M2_BLOCK_SIZE,   MXFP6_E3M2_BLOCK_BYTES)
NF_GEMM_GROUPED(I8_B128,      I8_B128_decode_elem,      I8_B128_BLOCK_SIZE,      I8_B128_BLOCK_BYTES)
NF_GEMM_GROUPED(NF4,          NF4_decode_elem,          NF4_BLOCK_SIZE,          NF4_BLOCK_BYTES)
NF_GEMM_GROUPED(FP4_BNB,      FP4_BNB_decode_elem,      FP4_BNB_BLOCK_SIZE,      FP4_BNB_BLOCK_BYTES)

/* moe_matvec.cl - matvec_id_family instantiations for every weight type (MoE expert matvecs). Requires: common/dialect.h, common/config.h, common/math.h, common/reduce.h, common/quant_blocks.h, common/iq_tables.h, common/quant_decode_legacy.h, common/quant_decode_k.h, common/quant_decode_iq.h, common/quant_decode_new.h, common/matvec_id_family.h */
/*
 * Each NF_MOE_ID_KERNELS line generates T_MatVecId, T_MatVecIdGateUp and T_BatchMatMulId with the
 * arguments and launch contract of common/matvec_id_family.h, for the 29 supported weight types.
 * Threads per block (TPS): dense 1; Q4_0, Q4_1, Q5_0, Q5_1, IQ4_NL, Q8_0, MXFP4 4; NVFP4, Q2_0 8;
 * Q1_0 16; IQ4_XS, the K-quants, the I-quants, TQ1_0 and TQ2_0 32. The local size must be a
 * power of two and a multiple of TPS (Java: MoeKernels.threadsPerBlock).
 */

/* ---- dense ---- */
NF_MOE_ID_KERNELS(F32,  nf_moe_f32,  1, 4, 1)
NF_MOE_ID_KERNELS(F16,  nf_moe_f16,  1, 2, 1)
NF_MOE_ID_KERNELS(BF16, nf_moe_bf16, 1, 2, 1)

/* ---- legacy 32-element blocks (common/quant_decode_legacy.h) ---- */
NF_MOE_ID_KERNELS(Q4_0,   Q4_0,   Q4_0_BLOCK_SIZE,   Q4_0_BLOCK_BYTES,   Q4_0_TPS)
NF_MOE_ID_KERNELS(Q4_1,   Q4_1,   Q4_1_BLOCK_SIZE,   Q4_1_BLOCK_BYTES,   Q4_1_TPS)
NF_MOE_ID_KERNELS(Q5_0,   Q5_0,   Q5_0_BLOCK_SIZE,   Q5_0_BLOCK_BYTES,   Q5_0_TPS)
NF_MOE_ID_KERNELS(Q5_1,   Q5_1,   Q5_1_BLOCK_SIZE,   Q5_1_BLOCK_BYTES,   Q5_1_TPS)
NF_MOE_ID_KERNELS(Q8_0,   Q8_0,   Q8_0_BLOCK_SIZE,   Q8_0_BLOCK_BYTES,   Q8_0_TPS)
NF_MOE_ID_KERNELS(IQ4_NL, IQ4_NL, IQ4_NL_BLOCK_SIZE, IQ4_NL_BLOCK_BYTES, IQ4_NL_TPS)
NF_MOE_ID_KERNELS(IQ4_XS, IQ4_XS, IQ4_XS_BLOCK_SIZE, IQ4_XS_BLOCK_BYTES, IQ4_XS_TPS)
NF_MOE_ID_KERNELS(MXFP4,  MXFP4,  MXFP4_BLOCK_SIZE,  MXFP4_BLOCK_BYTES,  MXFP4_TPS)

/* ---- K-quant super-blocks (common/quant_decode_k.h) ---- */
NF_MOE_ID_KERNELS(Q2_K, Q2_K, Q2_K_BLOCK_SIZE, Q2_K_BLOCK_BYTES, NF_KQ_TPS)
NF_MOE_ID_KERNELS(Q3_K, Q3_K, Q3_K_BLOCK_SIZE, Q3_K_BLOCK_BYTES, NF_KQ_TPS)
NF_MOE_ID_KERNELS(Q4_K, Q4_K, Q4_K_BLOCK_SIZE, Q4_K_BLOCK_BYTES, NF_KQ_TPS)
NF_MOE_ID_KERNELS(Q5_K, Q5_K, Q5_K_BLOCK_SIZE, Q5_K_BLOCK_BYTES, NF_KQ_TPS)
NF_MOE_ID_KERNELS(Q6_K, Q6_K, Q6_K_BLOCK_SIZE, Q6_K_BLOCK_BYTES, NF_KQ_TPS)
NF_MOE_ID_KERNELS(Q8_K, Q8_K, Q8_K_BLOCK_SIZE, Q8_K_BLOCK_BYTES, NF_KQ_TPS)

/* ---- I-quant super-blocks (common/quant_decode_iq.h) ---- */
NF_MOE_ID_KERNELS(IQ3_XXS, IQ3_XXS, IQ3_XXS_BLOCK_SIZE, IQ3_XXS_BLOCK_BYTES, NF_IQD_TPS)
NF_MOE_ID_KERNELS(IQ3_S,   IQ3_S,   IQ3_S_BLOCK_SIZE,   IQ3_S_BLOCK_BYTES,   NF_IQD_TPS)
NF_MOE_ID_KERNELS(IQ2_XXS, IQ2_XXS, IQ2_XXS_BLOCK_SIZE, IQ2_XXS_BLOCK_BYTES, NF_IQD_TPS)
NF_MOE_ID_KERNELS(IQ2_XS,  IQ2_XS,  IQ2_XS_BLOCK_SIZE,  IQ2_XS_BLOCK_BYTES,  NF_IQD_TPS)
NF_MOE_ID_KERNELS(IQ2_S,   IQ2_S,   IQ2_S_BLOCK_SIZE,   IQ2_S_BLOCK_BYTES,   NF_IQD_TPS)
NF_MOE_ID_KERNELS(IQ1_S,   IQ1_S,   IQ1_S_BLOCK_SIZE,   IQ1_S_BLOCK_BYTES,   NF_IQD_TPS)
NF_MOE_ID_KERNELS(IQ1_M,   IQ1_M,   IQ1_M_BLOCK_SIZE,   IQ1_M_BLOCK_BYTES,   NF_IQD_TPS)

/* ---- ternary, NVFP4 and 1-/2-bit blocks (common/quant_decode_new.h) ---- */
NF_MOE_ID_KERNELS(TQ1_0, TQ1_0, TQ1_0_BLOCK_SIZE, TQ1_0_BLOCK_BYTES, TQ1_0_TPS)
NF_MOE_ID_KERNELS(TQ2_0, TQ2_0, TQ2_0_BLOCK_SIZE, TQ2_0_BLOCK_BYTES, TQ2_0_TPS)
NF_MOE_ID_KERNELS(NVFP4, NVFP4, NVFP4_BLOCK_SIZE, NVFP4_BLOCK_BYTES, NVFP4_TPS)
NF_MOE_ID_KERNELS(Q1_0,  Q1_0,  Q1_0_BLOCK_SIZE,  Q1_0_BLOCK_BYTES,  Q1_0_TPS)
NF_MOE_ID_KERNELS(Q2_0,  Q2_0,  Q2_0_BLOCK_SIZE,  Q2_0_BLOCK_BYTES,  Q2_0_TPS)

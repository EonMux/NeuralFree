/* matvec_iquant.cl - IQ_ALL_KERNELS and the integer-dot forms for IQ3_XXS, IQ3_S, IQ3_K, IQ3_KS,
 * IQ2_XXS, IQ2_XS, IQ2_S, IQ1_S and IQ1_M. Requires: common/dialect.h, common/config.h,
 * common/reduce.h, common/quant_blocks.h, common/iq_tables.h, common/q8_dot.h,
 * common/quant_decode_iq.h, common/quant_decode_iq_q8.h, common/matvec_family.h */
/*
 * Eight kernels per type (T_MatVec, T_MatVecAdd, T_MatVecFusedGateUp, T_ComputeLogits,
 * T_EmbeddingLookup, T_BatchEmbeddingLookup, T_BatchMatMul, T_BatchMatMul_2D) with the argument
 * lists and launch contract of common/matvec_family.h, plus the Q8_1-activation forms
 * (T_MatVec_Q8, T_MatVec2_Q8, T_MatVecAdd_Q8, T_MatVecFusedGateUp_Q8 and
 * T_MatVec2FusedGateUp_Q8). Every type has 256-element blocks decoded
 * by NF_IQD_TPS = 32 work-items, so matvec local sizes are multiples of 32.
 */

IQ_ALL_KERNELS(IQ3_XXS, IQ3_XXS_BLOCK_SIZE, IQ3_XXS_BLOCK_BYTES, NF_IQD_TPS)
IQ_ALL_KERNELS(IQ3_S,   IQ3_S_BLOCK_SIZE,   IQ3_S_BLOCK_BYTES,   NF_IQD_TPS)
IQ_ALL_KERNELS(IQ3_K,   IQ3_K_BLOCK_SIZE,   IQ3_K_BLOCK_BYTES,   NF_IQD_TPS)
IQ_ALL_KERNELS(IQ3_KS,  IQ3_KS_BLOCK_SIZE,  IQ3_KS_BLOCK_BYTES,  NF_IQD_TPS)
IQ_ALL_KERNELS(IQ2_XXS, IQ2_XXS_BLOCK_SIZE, IQ2_XXS_BLOCK_BYTES, NF_IQD_TPS)
IQ_ALL_KERNELS(IQ2_XS,  IQ2_XS_BLOCK_SIZE,  IQ2_XS_BLOCK_BYTES,  NF_IQD_TPS)
IQ_ALL_KERNELS(IQ2_S,   IQ2_S_BLOCK_SIZE,   IQ2_S_BLOCK_BYTES,   NF_IQD_TPS)
IQ_ALL_KERNELS(IQ1_S,   IQ1_S_BLOCK_SIZE,   IQ1_S_BLOCK_BYTES,   NF_IQD_TPS)
IQ_ALL_KERNELS(IQ1_M,   IQ1_M_BLOCK_SIZE,   IQ1_M_BLOCK_BYTES,   NF_IQD_TPS)

IQ_ALL_KERNELS_Q8(IQ3_XXS, IQ3_XXS_BLOCK_SIZE, IQ3_XXS_BLOCK_BYTES, NF_IQD_TPS)
IQ_ALL_KERNELS_Q8(IQ3_S,   IQ3_S_BLOCK_SIZE,   IQ3_S_BLOCK_BYTES,   NF_IQD_TPS)
IQ_ALL_KERNELS_Q8(IQ3_K,   IQ3_K_BLOCK_SIZE,   IQ3_K_BLOCK_BYTES,   NF_IQD_TPS)
IQ_ALL_KERNELS_Q8(IQ3_KS,  IQ3_KS_BLOCK_SIZE,  IQ3_KS_BLOCK_BYTES,  NF_IQD_TPS)
IQ_ALL_KERNELS_Q8(IQ2_XXS, IQ2_XXS_BLOCK_SIZE, IQ2_XXS_BLOCK_BYTES, NF_IQD_TPS)
IQ_ALL_KERNELS_Q8(IQ2_XS,  IQ2_XS_BLOCK_SIZE,  IQ2_XS_BLOCK_BYTES,  NF_IQD_TPS)
IQ_ALL_KERNELS_Q8(IQ2_S,   IQ2_S_BLOCK_SIZE,   IQ2_S_BLOCK_BYTES,   NF_IQD_TPS)
IQ_ALL_KERNELS_Q8(IQ1_S,   IQ1_S_BLOCK_SIZE,   IQ1_S_BLOCK_BYTES,   NF_IQD_TPS)
IQ_ALL_KERNELS_Q8(IQ1_M,   IQ1_M_BLOCK_SIZE,   IQ1_M_BLOCK_BYTES,   NF_IQD_TPS)

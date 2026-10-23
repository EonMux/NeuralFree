/* matvec_iquant.cl - IQ_ALL_KERNELS for IQ3_XXS, IQ3_S, IQ2_XXS, IQ2_XS, IQ2_S, IQ1_S and IQ1_M. Requires: common/dialect.h, common/config.h, common/reduce.h, common/quant_blocks.h, common/iq_tables.h, common/quant_decode_iq.h, common/matvec_family.h */
/*
 * Eight kernels per type (T_MatVec, T_MatVecAdd, T_MatVecFusedGateUp, T_ComputeLogits,
 * T_EmbeddingLookup, T_BatchEmbeddingLookup, T_BatchMatMul, T_BatchMatMul_2D) with the argument
 * lists and launch contract of common/matvec_family.h. Every type has 256-element blocks decoded by
 * NF_IQD_TPS = 32 work-items, so matvec local sizes are multiples of 32.
 */

IQ_ALL_KERNELS(IQ3_XXS, IQ3_XXS_BLOCK_SIZE, IQ3_XXS_BLOCK_BYTES, NF_IQD_TPS)
IQ_ALL_KERNELS(IQ3_S,   IQ3_S_BLOCK_SIZE,   IQ3_S_BLOCK_BYTES,   NF_IQD_TPS)
IQ_ALL_KERNELS(IQ2_XXS, IQ2_XXS_BLOCK_SIZE, IQ2_XXS_BLOCK_BYTES, NF_IQD_TPS)
IQ_ALL_KERNELS(IQ2_XS,  IQ2_XS_BLOCK_SIZE,  IQ2_XS_BLOCK_BYTES,  NF_IQD_TPS)
IQ_ALL_KERNELS(IQ2_S,   IQ2_S_BLOCK_SIZE,   IQ2_S_BLOCK_BYTES,   NF_IQD_TPS)
IQ_ALL_KERNELS(IQ1_S,   IQ1_S_BLOCK_SIZE,   IQ1_S_BLOCK_BYTES,   NF_IQD_TPS)
IQ_ALL_KERNELS(IQ1_M,   IQ1_M_BLOCK_SIZE,   IQ1_M_BLOCK_BYTES,   NF_IQD_TPS)

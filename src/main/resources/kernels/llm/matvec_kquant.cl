/* matvec_kquant.cl - IQ_ALL_KERNELS for Q2_K, Q3_K, Q4_K, Q5_K, Q6_K and Q8_K. Requires: common/dialect.h, common/config.h, common/reduce.h, common/quant_blocks.h, common/quant_decode_k.h, common/matvec_family.h */
/*
 * Generates, per K-quant type T, the 8 family kernels of common/matvec_family.h
 * with its argument lists: T_MatVec, T_MatVecAdd, T_MatVecFusedGateUp, T_ComputeLogits,
 * T_EmbeddingLookup, T_BatchEmbeddingLookup, T_BatchMatMul and T_BatchMatMul_2D.
 * Matvec forms require a local size that is a power of two, a multiple of NF_KQ_TPS (32) and at
 * most LLM_DIMX; in_dim (embed_dim) is a multiple of QK_K.
 */

IQ_ALL_KERNELS(Q2_K, Q2_K_BLOCK_SIZE, Q2_K_BLOCK_BYTES, NF_KQ_TPS)
IQ_ALL_KERNELS(Q3_K, Q3_K_BLOCK_SIZE, Q3_K_BLOCK_BYTES, NF_KQ_TPS)
IQ_ALL_KERNELS(Q4_K, Q4_K_BLOCK_SIZE, Q4_K_BLOCK_BYTES, NF_KQ_TPS)
IQ_ALL_KERNELS(Q5_K, Q5_K_BLOCK_SIZE, Q5_K_BLOCK_BYTES, NF_KQ_TPS)
IQ_ALL_KERNELS(Q6_K, Q6_K_BLOCK_SIZE, Q6_K_BLOCK_BYTES, NF_KQ_TPS)
IQ_ALL_KERNELS(Q8_K, Q8_K_BLOCK_SIZE, Q8_K_BLOCK_BYTES, NF_KQ_TPS)

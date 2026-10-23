/* matvec_newquant.cl - IQ_ALL_KERNELS for TQ1_0, TQ2_0, NVFP4, Q1_0 and Q2_0. Requires: common/dialect.h, common/config.h, common/reduce.h, common/quant_blocks.h, common/quant_decode_new.h, common/matvec_family.h */
/*
 * Generates, per type T, the 8 family kernels with the IQ_ALL_KERNELS argument lists of common/matvec_family.h:
 * T_MatVec, T_MatVecAdd, T_MatVecFusedGateUp, T_ComputeLogits, T_EmbeddingLookup,
 * T_BatchEmbeddingLookup, T_BatchMatMul and T_BatchMatMul_2D, decoding through the triples of
 * common/quant_decode_new.h.
 * Matvec forms require a local size that is a power of two, at most LLM_DIMX and a multiple of the
 * type's TPS (TQ1_0/TQ2_0 32, Q1_0 16, NVFP4/Q2_0 8; CUDA: a multiple of 32); in_dim (embed_dim) is a
 * multiple of the block size (TQ1_0/TQ2_0 256, Q1_0 128, NVFP4/Q2_0 64).
 */

IQ_ALL_KERNELS(TQ1_0, TQ1_0_BLOCK_SIZE, TQ1_0_BLOCK_BYTES, TQ1_0_TPS)
IQ_ALL_KERNELS(TQ2_0, TQ2_0_BLOCK_SIZE, TQ2_0_BLOCK_BYTES, TQ2_0_TPS)
IQ_ALL_KERNELS(NVFP4, NVFP4_BLOCK_SIZE, NVFP4_BLOCK_BYTES, NVFP4_TPS)
IQ_ALL_KERNELS(Q1_0, Q1_0_BLOCK_SIZE, Q1_0_BLOCK_BYTES, Q1_0_TPS)
IQ_ALL_KERNELS(Q2_0, Q2_0_BLOCK_SIZE, Q2_0_BLOCK_BYTES, Q2_0_TPS)

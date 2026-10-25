/* matvec_extquant.cl - IQ_ALL_KERNELS for F8_E4M3, F8_E4M3_B128, MXFP8, I8_B128, NF4 and FP4_BNB. Requires: common/dialect.h, common/config.h, common/reduce.h, common/quant_blocks.h, common/quant_decode_new.h, common/quant_decode_ext.h, common/matvec_family.h */
/*
 * Generates, per NeuralFree extension type T, the 8 family kernels with the IQ_ALL_KERNELS argument lists
 * of common/matvec_family.h: T_MatVec, T_MatVecAdd, T_MatVecFusedGateUp, T_ComputeLogits,
 * T_EmbeddingLookup, T_BatchEmbeddingLookup, T_BatchMatMul and T_BatchMatMul_2D, decoding through the
 * triples of common/quant_decode_ext.h.
 * Matvec forms require a local size that is a power of two, at most LLM_DIMX and a multiple of the
 * type's TPS (F8_E4M3/MXFP8 4, NF4/FP4_BNB 8, F8_E4M3_B128/I8_B128 16; CUDA: a multiple of 32); in_dim
 * (embed_dim) is a multiple of the block size (F8_E4M3/MXFP8 32, NF4/FP4_BNB 64, F8_E4M3_B128/I8_B128 128).
 */

IQ_ALL_KERNELS(F8_E4M3, F8_E4M3_BLOCK_SIZE, F8_E4M3_BLOCK_BYTES, F8_E4M3_TPS)
IQ_ALL_KERNELS(F8_E4M3_B128, F8_E4M3_B128_BLOCK_SIZE, F8_E4M3_B128_BLOCK_BYTES, F8_E4M3_B128_TPS)
IQ_ALL_KERNELS(MXFP8, MXFP8_BLOCK_SIZE, MXFP8_BLOCK_BYTES, MXFP8_TPS)
IQ_ALL_KERNELS(I8_B128, I8_B128_BLOCK_SIZE, I8_B128_BLOCK_BYTES, I8_B128_TPS)
IQ_ALL_KERNELS(NF4, NF4_BLOCK_SIZE, NF4_BLOCK_BYTES, NF4_TPS)
IQ_ALL_KERNELS(FP4_BNB, FP4_BNB_BLOCK_SIZE, FP4_BNB_BLOCK_BYTES, FP4_BNB_TPS)

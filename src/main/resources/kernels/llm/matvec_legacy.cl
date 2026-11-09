/* matvec_legacy.cl - IQ_ALL_KERNELS and the integer-dot forms for IQ4_NL, Q4_0, Q4_1, Q5_0, Q5_1 and the MXFP4 family completions. Requires: common/dialect.h, common/config.h, common/reduce.h, common/quant_blocks.h, common/q8_dot.h, common/quant_decode_legacy.h, common/quant_decode_legacy_q8.h, common/matvec_family.h */
/*
 * Each IQ_ALL_KERNELS line generates T_MatVec, T_MatVecAdd, T_MatVecFusedGateUp, T_ComputeLogits,
 * T_EmbeddingLookup, T_BatchEmbeddingLookup, T_BatchMatMul and T_BatchMatMul_2D with the arguments
 * and launch contract of common/matvec_family.h, decoding through the triples of
 * common/quant_decode_legacy.h (ggml split nibble layout). The Q8_1-activation forms
 * (IQ_ALL_KERNELS_Q8: T_MatVec_Q8, T_MatVec2_Q8, T_MatVecAdd_Q8, T_MatVecFusedGateUp_Q8 and
 * T_MatVec2FusedGateUp_Q8) come from
 * common/quant_decode_legacy_q8.h and exist for every type whose block is 32 elements.
 *
 * MXFP4_MatVec, MXFP4_MatVecAdd, MXFP4_MatVecFusedGateUp and LLM_MXFP4_EmbeddingLookup are the
 * hand-written kernels of matvec_mxfp4.cl; the four remaining family members are generated here.
 */

IQ_ALL_KERNELS(IQ4_NL, IQ4_NL_BLOCK_SIZE, IQ4_NL_BLOCK_BYTES, IQ4_NL_TPS)
IQ_ALL_KERNELS_Q8(IQ4_NL, IQ4_NL_BLOCK_SIZE, IQ4_NL_BLOCK_BYTES, IQ4_NL_TPS)
IQ_ALL_KERNELS(Q4_0, Q4_0_BLOCK_SIZE, Q4_0_BLOCK_BYTES, Q4_0_TPS)
IQ_ALL_KERNELS_Q8(Q4_0, Q4_0_BLOCK_SIZE, Q4_0_BLOCK_BYTES, Q4_0_TPS)
/* The counted forms above the dispatched range, for the type whose measured dispatch serves exact
 * counts to 16 (SmallBatchCrossover's entries on the CUDA and OpenCL NVIDIA kinds). */
IQ_MVQN_NC_HIGH(Q4_0, Q4_0_BLOCK_SIZE, Q4_0_BLOCK_BYTES, Q4_0_TPS)
IQ_ALL_KERNELS(Q4_1, Q4_1_BLOCK_SIZE, Q4_1_BLOCK_BYTES, Q4_1_TPS)
IQ_ALL_KERNELS_Q8(Q4_1, Q4_1_BLOCK_SIZE, Q4_1_BLOCK_BYTES, Q4_1_TPS)
IQ_ALL_KERNELS(Q5_0, Q5_0_BLOCK_SIZE, Q5_0_BLOCK_BYTES, Q5_0_TPS)
IQ_ALL_KERNELS_Q8(Q5_0, Q5_0_BLOCK_SIZE, Q5_0_BLOCK_BYTES, Q5_0_TPS)
IQ_ALL_KERNELS(Q5_1, Q5_1_BLOCK_SIZE, Q5_1_BLOCK_BYTES, Q5_1_TPS)
IQ_ALL_KERNELS_Q8(Q5_1, Q5_1_BLOCK_SIZE, Q5_1_BLOCK_BYTES, Q5_1_TPS)

IQ_COMPUTELOGITS(MXFP4, MXFP4_BLOCK_SIZE, MXFP4_BLOCK_BYTES, MXFP4_TPS)
IQ_BATCHEMBED(MXFP4, MXFP4_BLOCK_SIZE, MXFP4_BLOCK_BYTES)
IQ_BATCHMATMUL(MXFP4, MXFP4_BLOCK_SIZE, MXFP4_BLOCK_BYTES, MXFP4_TPS)
IQ_BATCHMATMUL2D(MXFP4, MXFP4_BLOCK_SIZE, MXFP4_BLOCK_BYTES, MXFP4_TPS)
IQ_ALL_KERNELS_Q8(MXFP4, MXFP4_BLOCK_SIZE, MXFP4_BLOCK_BYTES, MXFP4_TPS)

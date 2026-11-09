/* matvec_kquant.cl - IQ_ALL_KERNELS and the integer-dot forms for Q2_K, Q3_K, Q4_K, Q5_K, Q6_K and Q8_K. Requires: common/dialect.h, common/config.h, common/reduce.h, common/quant_blocks.h, common/q8_dot.h, common/quant_decode_k.h, common/quant_decode_k_q8.h, common/matvec_family.h */
/*
 * Generates, per K-quant type T, the 8 family kernels of common/matvec_family.h with its argument
 * lists (T_MatVec, T_MatVecAdd, T_MatVecFusedGateUp, T_ComputeLogits, T_EmbeddingLookup,
 * T_BatchEmbeddingLookup, T_BatchMatMul and T_BatchMatMul_2D) and the Q8_1-activation forms
 * (T_MatVec_Q8, T_MatVec2_Q8, T_MatVecAdd_Q8, T_MatVecFusedGateUp_Q8 and
 * T_MatVec2FusedGateUp_Q8), whose integer dots need
 * common/quant_decode_k_q8.h.
 * Matvec forms require a local size that is a power of two, a multiple of NF_KQ_TPS (32) and at
 * most LLM_DIMX; in_dim (embed_dim) is a multiple of QK_K.
 */

IQ_ALL_KERNELS(Q2_K, Q2_K_BLOCK_SIZE, Q2_K_BLOCK_BYTES, NF_KQ_TPS)
IQ_ALL_KERNELS(Q3_K, Q3_K_BLOCK_SIZE, Q3_K_BLOCK_BYTES, NF_KQ_TPS)
IQ_ALL_KERNELS(Q4_K, Q4_K_BLOCK_SIZE, Q4_K_BLOCK_BYTES, NF_KQ_TPS)
IQ_ALL_KERNELS(Q5_K, Q5_K_BLOCK_SIZE, Q5_K_BLOCK_BYTES, NF_KQ_TPS)
IQ_ALL_KERNELS(Q6_K, Q6_K_BLOCK_SIZE, Q6_K_BLOCK_BYTES, NF_KQ_TPS)
IQ_ALL_KERNELS(Q8_K, Q8_K_BLOCK_SIZE, Q8_K_BLOCK_BYTES, NF_KQ_TPS)

IQ_ALL_KERNELS_Q8(Q2_K, Q2_K_BLOCK_SIZE, Q2_K_BLOCK_BYTES, NF_KQ_TPS)
IQ_ALL_KERNELS_Q8(Q3_K, Q3_K_BLOCK_SIZE, Q3_K_BLOCK_BYTES, NF_KQ_TPS)
IQ_ALL_KERNELS_Q8(Q4_K, Q4_K_BLOCK_SIZE, Q4_K_BLOCK_BYTES, NF_KQ_TPS)
IQ_ALL_KERNELS_Q8(Q5_K, Q5_K_BLOCK_SIZE, Q5_K_BLOCK_BYTES, NF_KQ_TPS)
IQ_ALL_KERNELS_Q8(Q6_K, Q6_K_BLOCK_SIZE, Q6_K_BLOCK_BYTES, NF_KQ_TPS)
IQ_ALL_KERNELS_Q8(Q8_K, Q8_K_BLOCK_SIZE, Q8_K_BLOCK_BYTES, NF_KQ_TPS)

/* The counted forms above the dispatched range, for the types whose measured dispatch serves exact
 * counts to 16 (SmallBatchCrossover's entries on the CUDA and OpenCL NVIDIA kinds). */
IQ_MVQN_NC_HIGH(Q4_K, Q4_K_BLOCK_SIZE, Q4_K_BLOCK_BYTES, NF_KQ_TPS)
IQ_MVQN_NC_HIGH(Q5_K, Q5_K_BLOCK_SIZE, Q5_K_BLOCK_BYTES, NF_KQ_TPS)
IQ_MVQN_NC_HIGH(Q6_K, Q6_K_BLOCK_SIZE, Q6_K_BLOCK_BYTES, NF_KQ_TPS)

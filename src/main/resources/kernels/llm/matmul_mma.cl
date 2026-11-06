/* matmul_mma.cl - matmul_mma_family instantiations for the dense 16-bit weight types F16 and BF16
 * (tensor-core GEMM and its split-K twin).
 * Requires: common/dialect.h, common/config.h, common/quant_blocks.h, common/matmul_mma_family.h */
/*
 * Generates F16_MatMul_MMA and BF16_MatMul_MMA plus their KSplit twins: the tensor-core GEMM rows of
 * the dense 16-bit weight types. On CUDA sm_80+ the kernels multiply with mma.sync.m16n8k16 fragments
 * loaded by ldmatrix (BF16 on the bfloat16 instruction); on CUDA below sm_80, on HIP and on OpenCL
 * they are the scalar twins of the same names (see common/matmul_mma_family.h for the per-path
 * arithmetic).
 * Arguments, semantics and launch contract: common/matmul_mma_family.h. K may be any positive
 * length; partial stages load 0.
 */
NF_MMA_GEMM(F16, 0)
NF_MMA_GEMM(BF16, 1)
NF_MMA_GEMM_KSPLIT(F16, 0)
NF_MMA_GEMM_KSPLIT(BF16, 1)

/* matmul_tiled.cl - matmul_tiled_family instantiations for every weight type (prefill GEMM). Requires: common/dialect.h, common/config.h, common/quant_blocks.h, common/iq_tables.h, common/quant_decode_legacy.h, common/quant_decode_k.h, common/quant_decode_iq.h, common/quant_decode_new.h, common/matmul_tiled_family.h */
/*
 * Generates T_MatMul_Tiled for the 29 supported weight types: F32, F16 and BF16 through the
 * dense loaders of common/matmul_tiled_family.h, every quantized type through its T_decode_elem
 * (common/quant_decode_*.h). Arguments, semantics and launch contract: matmul_tiled_family.h.
 * K must be a multiple of the type's block size (32, 64, 128 or 256 elements).
 */

/* ---- dense ---- */
NF_GEMM_TILED(F32,  nf_gemm_f32_elem,  1, 4)
NF_GEMM_TILED(F16,  nf_gemm_f16_elem,  1, 2)
NF_GEMM_TILED(BF16, nf_gemm_bf16_elem, 1, 2)

/* ---- legacy 32-element blocks, MXFP4 and IQ4_XS (common/quant_decode_legacy.h) ---- */
NF_GEMM_TILED(Q4_0,   Q4_0_decode_elem,   Q4_0_BLOCK_SIZE,   Q4_0_BLOCK_BYTES)
NF_GEMM_TILED(Q4_1,   Q4_1_decode_elem,   Q4_1_BLOCK_SIZE,   Q4_1_BLOCK_BYTES)
NF_GEMM_TILED(Q5_0,   Q5_0_decode_elem,   Q5_0_BLOCK_SIZE,   Q5_0_BLOCK_BYTES)
NF_GEMM_TILED(Q5_1,   Q5_1_decode_elem,   Q5_1_BLOCK_SIZE,   Q5_1_BLOCK_BYTES)
NF_GEMM_TILED(Q8_0,   Q8_0_decode_elem,   Q8_0_BLOCK_SIZE,   Q8_0_BLOCK_BYTES)
NF_GEMM_TILED(IQ4_NL, IQ4_NL_decode_elem, IQ4_NL_BLOCK_SIZE, IQ4_NL_BLOCK_BYTES)
NF_GEMM_TILED(IQ4_XS, IQ4_XS_decode_elem, IQ4_XS_BLOCK_SIZE, IQ4_XS_BLOCK_BYTES)
NF_GEMM_TILED(MXFP4,  MXFP4_decode_elem,  MXFP4_BLOCK_SIZE,  MXFP4_BLOCK_BYTES)

/* ---- K-quants (common/quant_decode_k.h) ---- */
NF_GEMM_TILED(Q2_K, Q2_K_decode_elem, Q2_K_BLOCK_SIZE, Q2_K_BLOCK_BYTES)
NF_GEMM_TILED(Q3_K, Q3_K_decode_elem, Q3_K_BLOCK_SIZE, Q3_K_BLOCK_BYTES)
NF_GEMM_TILED(Q4_K, Q4_K_decode_elem, Q4_K_BLOCK_SIZE, Q4_K_BLOCK_BYTES)
NF_GEMM_TILED(Q5_K, Q5_K_decode_elem, Q5_K_BLOCK_SIZE, Q5_K_BLOCK_BYTES)
NF_GEMM_TILED(Q6_K, Q6_K_decode_elem, Q6_K_BLOCK_SIZE, Q6_K_BLOCK_BYTES)
NF_GEMM_TILED(Q8_K, Q8_K_decode_elem, Q8_K_BLOCK_SIZE, Q8_K_BLOCK_BYTES)

/* ---- I-quants (common/quant_decode_iq.h) ---- */
NF_GEMM_TILED(IQ3_XXS, IQ3_XXS_decode_elem, IQ3_XXS_BLOCK_SIZE, IQ3_XXS_BLOCK_BYTES)
NF_GEMM_TILED(IQ3_S,   IQ3_S_decode_elem,   IQ3_S_BLOCK_SIZE,   IQ3_S_BLOCK_BYTES)
NF_GEMM_TILED(IQ2_XXS, IQ2_XXS_decode_elem, IQ2_XXS_BLOCK_SIZE, IQ2_XXS_BLOCK_BYTES)
NF_GEMM_TILED(IQ2_XS,  IQ2_XS_decode_elem,  IQ2_XS_BLOCK_SIZE,  IQ2_XS_BLOCK_BYTES)
NF_GEMM_TILED(IQ2_S,   IQ2_S_decode_elem,   IQ2_S_BLOCK_SIZE,   IQ2_S_BLOCK_BYTES)
NF_GEMM_TILED(IQ1_S,   IQ1_S_decode_elem,   IQ1_S_BLOCK_SIZE,   IQ1_S_BLOCK_BYTES)
NF_GEMM_TILED(IQ1_M,   IQ1_M_decode_elem,   IQ1_M_BLOCK_SIZE,   IQ1_M_BLOCK_BYTES)

/* ---- ternary, NVFP4 and low-bit (common/quant_decode_new.h) ---- */
NF_GEMM_TILED(TQ1_0, TQ1_0_decode_elem, TQ1_0_BLOCK_SIZE, TQ1_0_BLOCK_BYTES)
NF_GEMM_TILED(TQ2_0, TQ2_0_decode_elem, TQ2_0_BLOCK_SIZE, TQ2_0_BLOCK_BYTES)
NF_GEMM_TILED(NVFP4, NVFP4_decode_elem, NVFP4_BLOCK_SIZE, NVFP4_BLOCK_BYTES)
NF_GEMM_TILED(Q1_0,  Q1_0_decode_elem,  Q1_0_BLOCK_SIZE,  Q1_0_BLOCK_BYTES)
NF_GEMM_TILED(Q2_0,  Q2_0_decode_elem,  Q2_0_BLOCK_SIZE,  Q2_0_BLOCK_BYTES)

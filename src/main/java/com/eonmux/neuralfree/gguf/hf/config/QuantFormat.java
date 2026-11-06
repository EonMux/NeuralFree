package com.eonmux.neuralfree.gguf.hf.config;

/**
 * The on-disk tensor layout of one quantized module, which selects the rewrite scheme. Each constant names the
 * checkpoint family whose layout it is; the layouts themselves are decoded by the quantized-weight rewriters.
 */
public enum QuantFormat {
    /** compressed-tensors {@code pack-quantized}: int32 words of 2-8 bit values, {@code weight_packed}. */
    INT_PACKED_CT,
    /** GPTQ {@code qweight}/{@code qzeros}/{@code scales}/{@code g_idx} (also AutoRound auto_gptq, moe_wna16). */
    INT_GPTQ,
    /** AutoAWQ GEMM {@code qweight}/{@code qzeros}/{@code scales} in AWQ order (also AutoRound auto_awq). */
    INT_AWQ,
    /** AMD Quark int4/uint4 packed weights. */
    INT_QUARK,
    /** MLX affine {@code weight}/{@code scales}/{@code biases}. */
    INT_MLX,
    /** HQQ {@code W_q}/{@code scale}/{@code zero}. */
    INT_HQQ,
    /** int4 values stored one per int8 (compressed-tensors W4A8 int, W4AFP8 experts). */
    INT4_IN_INT8,
    /**
     * torchao {@code Int4Tensor}/{@code Int4TilePackedTo4dTensor} weights: two signed int4 values per byte, a
     * float scale and a float zero point per group, as the tensor subclass stores them.
     */
    INT4_TORCHAO,
    /** int8 weights with per-tensor or per-channel scales. */
    INT8_CHANNEL,
    /** int8 weights with block scales ({@code blockwise_int8}). */
    INT8_BLOCK,
    /** bitsandbytes LLM.int8 ({@code SCB}). */
    INT8_BNB,
    /** BitNet b1.58 packed ternary weights. */
    BITNET,
    /** FP8 E4M3 weights with per-tensor, per-channel or block scales. */
    FP8,
    /** MXFP8: FP8 E4M3 weights with an E8M0 scale per 32 elements. */
    MXFP8,
    /** compressed-tensors NVFP4 ({@code weight_packed}, {@code weight_global_scale}). */
    NVFP4_CT,
    /** ModelOpt NVFP4 ({@code weight_scale_2}). */
    NVFP4_MODELOPT,
    /**
     * Quark NVFP4 from two-level weight and input specs: {@code weight}, E4M3 {@code weight_scale}, the global
     * {@code weight_scale_2} and the input global scale {@code input_scale_2} (V quark/schemes/quark_nvfp4.py).
     */
    NVFP4_QUARK,
    /** compressed-tensors / Quark MXFP4 ({@code weight_packed} plus E8M0 scales). */
    MXFP4_CT,
    /** gpt-oss MXFP4 {@code _blocks}/{@code _scales}. */
    MXFP4_GPTOSS,
    /** DeepSeek-V4 MXFP4 experts with an E8M0 {@code .scale}. */
    MXFP4_DSV4,
    /** Quark OCP-MX FP6 (E2M3 or E3M2). */
    MXFP6,
    /** bitsandbytes 4-bit NF4. */
    NF4_BNB,
    /** bitsandbytes 4-bit FP4. */
    FP4_BNB,
    /** fp_quant real-quantized MXFP4 with a Hadamard rotation. */
    FPQUANT_MXFP4,
    /** fp_quant real-quantized NVFP4 with a Hadamard rotation. */
    FPQUANT_NVFP4,
    /** fp_quant pseudo-quantized ({@code dqweight}) weights with a Hadamard rotation. */
    FPQUANT_PSEUDO
}

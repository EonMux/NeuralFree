package com.eonmux.neuralfree.gguf.ops.gemm;

import com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.kernels.LlmKernelConstants;
import com.eonmux.neuralfree.gguf.quant.ActQuantScheme;

import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/**
 * Kernel names and argument indices of the GEMM family ({@code kernels/llm/matmul_tiled.cl},
 * {@code kernels/llm/matmul_mmq.cl}, {@code kernels/llm/matmul_mmq_stream.cl}, {@code kernels/llm/matmul_mma.cl},
 * {@code kernels/llm/matmul_cm.cl}, {@code kernels/llm/split_k_reduce.cl}, {@code kernels/llm/sampling_topk.cl}; GLSL twins
 * {@code shaders/llm/matmul_tiled.comp}, {@code shaders/llm/matmul_mmq.comp},
 * {@code shaders/llm/matmul_mma.comp}, {@code shaders/llm/matmul_mmq_cm.comp},
 * {@code shaders/llm/split_k_reduce.comp}, {@code shaders/llm/topk_partial.comp}).
 *
 * <p>Argument lists (buffers and scalars in declared order):</p>
 * <ul>
 *   <li>{@code <T>_MatMul_Tiled(int M, int K, int N, int x_stride, int x_off, int y_stride, int y_off,
 *       GpuBuffer weight, GpuBuffer x, GpuBuffer y)}: {@code y[y_off + t*y_stride + r] =
 *       sum_k W[r][k] * x[x_off + t*x_stride + k]}; dispatch geometry in {@link TiledGemmPlan}.</li>
 *   <li>{@code <T>_MatMul_MMA(...)} and {@code <T>_MatMul_CM(...)}: the tiled argument list and
 *       semantics with their own dispatch geometry ({@link MmaGemmPlan}, {@link CmGemmPlan}).</li>
 *   <li>{@code <T>_MatMul_MMQ(int M, int K, int N, int q_stride, int q_off, int y_stride, int y_off,
 *       GpuBuffer weight, GpuBuffer q8, GpuBuffer y)}: the int8 MMQ GEMM over the activations staged
 *       by {@link #QUANT_MMQ_Q8_1}, or, for the FP8 types, the E4M3 GEMM over the E4M3 records of the
 *       {@code LLM_Quant_MMQ_Fp8*}/{@code Fp4*} stagings; {@code q_stride} / {@code q_off} are in 32-bit
 *       record units; dispatch geometry in {@link MmqGemmPlan}.</li>
 *   <li>{@code <T>_MatMul_MMQ_StreamK(int M, int K, int N, int q_stride, int q_off, int y_stride,
 *       int y_off, GpuBuffer weight, GpuBuffer q8, GpuBuffer y, GpuBuffer fixup)} and
 *       {@code LLM_MMQ_StreamK_Fixup(int M, int K, int N, int y_stride, int y_off, GpuBuffer fixup,
 *       GpuBuffer y)}: the stream-k form of the int8 MMQ GEMM on the CUDA tensor tile
 *       ({@code kernels/llm/matmul_mmq_stream.cl}) and the pass that adds its split tiles' partial sums;
 *       dispatch geometry in {@link StreamKGemmPlan}.</li>
 *   <li>{@code <T>_MatMul_MMQ_StreamK_GateUp(...)}: the stream-k kernel's arguments followed by
 *       {@code GpuBuffer weight_up}, the fused gate/up SwiGLU form: {@code y = silu(weight . x) * (weight_up . x)}
 *       over two weights of one type and shape ({@link StreamKGemmPlan#ofGateUp}), and
 *       {@code LLM_MMQ_StreamK_Fixup_GateUp}, the fixup's arguments, which activates its split tiles.</li>
 *   <li>{@code <T>_MatMul_W4A8(int M, int K, int N, int q_stride, int q_off, int y_stride, int y_off,
 *       GpuBuffer weight, GpuBuffer q8, GpuBuffer y)}: the E4M3 GEMM of a 4-bit weight type, same argument
 *       list and staged records as the MMQ form ({@link #MATMUL_W4A8_SUFFIX}).</li>
 *   <li>{@code <T>_MatMul_MMQ_S(...)} and {@code <T>_MatMul_W4A8_S(...)}: the small-token-tile twins
 *       of the two forms above over a 16-token tile, same argument list and staged records; the host
 *       serves staged launches of fewer tokens than the device tile with them
 *       ({@link MmqGemmPlan#smallOf}), so a multi-row step does not stream its weights through
 *       64-token tiles.</li>
 *   <li>{@code LLM_Quant_MMQ_Q8_1(int N, int K, int tk, int x_stride, int x_off, GpuBuffer x,
 *       GpuBuffer q8)}: quantizes every token row of a projection once, per {@code tk}-deep chunk
 *       (the device's {@link MmqConfig#stageDepth()}), into the records the MMQ GEMM kernels read,
 *       shared by all row tiles (llama.cpp quantize_mmq_q8_1); one work-item per chunk, a 1-D
 *       dispatch of {@code N * (K / tk)} work-items. {@code LLM_Quant_MMQ_Q8_1_Warp} (same arguments, CUDA)
 *       writes the same records from {@code N * K / 4} work-items, one float4 each.</li>
 *   <li>{@code LLM_Quant_MMQ_Int8Static(int N, int K, int tk, int x_stride, int x_off, float scale,
 *       GpuBuffer x, GpuBuffer q8)}, {@code LLM_Quant_MMQ_Int8Row(int N, int K, int tk, int x_stride,
 *       int x_off, GpuBuffer x, GpuBuffer q8)} and {@code LLM_Quant_MMQ_Int8Group128(...)}: the same
 *       records with the checkpoint activation scheme's own int8 codes ({@code INT8_STATIC},
 *       {@code INT8_TOKEN} and {@code INT8_GROUP128}), so the GEMM multiplies the checkpoint's int8
 *       activations (W8A8). The row and per-128-run forms launch one workgroup per run.</li>
 *   <li>{@code <T>_MatMul_MMA_KSplit(int M, int K, int N, int x_stride, int x_off, int k_slice,
 *       GpuBuffer weight, GpuBuffer x, GpuBuffer partial)}: the split-K twin of the MMA GEMM over the
 *       same tiles, writing per-slice planes; dispatch geometry in {@link SplitKGemmPlan}, whose
 *       reduce launches {@code LLM_SplitK_Reduce}.</li>
 *   <li>{@code <T>_MatMul_MMQ_KSplit(int M, int K, int N, int q_stride, int q_off, int k_slice,
 *       GpuBuffer weight, GpuBuffer q8, GpuBuffer partial)} and
 *       {@code <T>_MatMul_CM_KSplit(...)} with the tiled {@code x} list: the split-K twins of the
 *       MMQ and CM GEMM lanes, same plane layout; dispatch geometry in {@link SplitKGemmPlan}.</li>
 *   <li>{@code LLM_SplitK_Reduce(int ne, int splits, int M, int y_stride, int y_off,
 *       GpuBuffer partial, GpuBuffer y)}: sums the split-K slice planes into the strided output.</li>
 *   <li>{@code LLM_TopK_Partial(int n, int k, float inv_temp, GpuBuffer logits, GpuBuffer cand_val,
 *       GpuBuffer cand_idx)}: per-slice top-k candidates for the host sampler; dispatch geometry in
 *       {@link TopKPartialPlan}.</li>
 * </ul>
 */
public final class GemmKernels {

    /** Name suffix of the per-weight-type tiled GEMM kernels, {@code <T>_MatMul_Tiled}. */
    public static final String MATMUL_TILED_SUFFIX = "_MatMul_Tiled";
    /** Name suffix of the int8 MMQ GEMM kernels, {@code <T>_MatMul_MMQ}. */
    public static final String MATMUL_MMQ_SUFFIX = "_MatMul_MMQ";
    /** Name suffix of the split-K twins of the int8 MMQ GEMM kernels, {@code <T>_MatMul_MMQ_KSplit}. */
    public static final String MATMUL_MMQ_KSPLIT_SUFFIX = "_MatMul_MMQ_KSplit";
    /** Name suffix of the split-K twins of the subgroup-row MMQ GEMM kernels, {@code <T>_MatMulSg_MMQ_KSplit}
     *  (kernels/llm/matmul_mmq_sg.cl). */
    public static final String MATMUL_MMQ_SG_KSPLIT_SUFFIX = "_MatMulSg_MMQ_KSplit";
    /** Name suffix of the stream-k form of the int8 MMQ GEMM kernels, {@code <T>_MatMul_MMQ_StreamK}. */
    public static final String MATMUL_MMQ_STREAMK_SUFFIX = "_MatMul_MMQ_StreamK";
    /** Name suffix of the fused gate/up SwiGLU stream-k kernels, {@code <T>_MatMul_MMQ_StreamK_GateUp}. */
    public static final String MATMUL_MMQ_STREAMK_GATE_UP_SUFFIX = "_MatMul_MMQ_StreamK_GateUp";
    /** Name suffix of the small-token-tile twins of the int8 MMQ GEMM kernels, {@code <T>_MatMul_MMQ_S}. */
    public static final String MATMUL_MMQ_S_SUFFIX = "_MatMul_MMQ_S";
    /** Name suffix of the 64-token small-token-tile twins, {@code <T>_MatMul_MMQ_S64}. */
    public static final String MATMUL_MMQ_S64_SUFFIX = "_MatMul_MMQ_S64";
    /**
     * Name suffix of the split-K twins of the small-token-tile MMQ GEMM kernels,
     * {@code <T>_MatMul_MMQ_SKSplit} (also for the FP8 types): the plain kernel's split-K form over
     * the small tiles' 16-token tile and row tile.
     */
    public static final String MATMUL_MMQ_SKSPLIT_SUFFIX = "_MatMul_MMQ_SKSplit";
    /** Name suffix of the split-K twins of the 64-token small-token-tile MMQ GEMM kernels,
     *  {@code <T>_MatMul_MMQ_SKSplit64}. */
    public static final String MATMUL_MMQ_SKSPLIT64_SUFFIX = "_MatMul_MMQ_SKSplit64";
    /** Name suffix of the small-token-tile twins of the W4A8 GEMM kernels, {@code <T>_MatMul_W4A8_S}. */
    public static final String MATMUL_W4A8_S_SUFFIX = "_MatMul_W4A8_S";
    /** Name suffix of the 64-token small-token-tile W4A8 twins, {@code <T>_MatMul_W4A8_S64}. */
    public static final String MATMUL_W4A8_S64_SUFFIX = "_MatMul_W4A8_S64";
    /** Name suffix of the split-K twins of the small-token-tile W4A8 GEMM kernels,
     *  {@code <T>_MatMul_W4A8_SKSplit}. */
    public static final String MATMUL_W4A8_SKSPLIT_SUFFIX = "_MatMul_W4A8_SKSplit";
    /** Name suffix of the split-K twins of the 64-token small-token-tile W4A8 GEMM kernels,
     *  {@code <T>_MatMul_W4A8_SKSplit64}. */
    public static final String MATMUL_W4A8_SKSPLIT64_SUFFIX = "_MatMul_W4A8_SKSplit64";
    /** Name suffix of the dequantizing GEMM kernels of the MMQ types, {@code <T>_MatMul_CM}. */
    public static final String MATMUL_CM_SUFFIX = "_MatMul_CM";
    /** Name suffix of the split-K twins of the CM GEMM kernels, {@code <T>_MatMul_CM_KSplit}. */
    public static final String MATMUL_CM_KSPLIT_SUFFIX = "_MatMul_CM_KSplit";
    /** Name suffix of the workgroup-scope cooperative-matrix GEMM kernels, {@code <T>_MatMul_CM2}. */
    public static final String MATMUL_CM2_SUFFIX = "_MatMul_CM2";
    /** Name suffix of the split-K twins of the CM2 GEMM kernels, {@code <T>_MatMul_CM2_KSplit}. */
    public static final String MATMUL_CM2_KSPLIT_SUFFIX = "_MatMul_CM2_KSplit";
    /** Name suffix of the E4M3 GEMM kernels of the 4-bit weight types, {@code <T>_MatMul_W4A8}. */
    public static final String MATMUL_W4A8_SUFFIX = "_MatMul_W4A8";
    /** Name suffix of the split-K twins of the W4A8 GEMM kernels, {@code <T>_MatMul_W4A8_KSplit}. */
    public static final String MATMUL_W4A8_KSPLIT_SUFFIX = "_MatMul_W4A8_KSplit";
    /** Name suffix of the tensor-core MMA GEMM kernels, {@code <T>_MatMul_MMA}. */
    public static final String MATMUL_MMA_SUFFIX = "_MatMul_MMA";
    /** Name suffix of the split-K twins of the MMA GEMM kernels, {@code <T>_MatMul_MMA_KSplit}. */
    public static final String MATMUL_MMA_KSPLIT_SUFFIX = "_MatMul_MMA_KSplit";
    /**
     * The MMQ activation staging ({@code kernels/llm/quant_mmq_q8_1.cl}, llama.cpp
     * {@code quantize_mmq_q8_1}): quantizes every token row of a projection once, per stage-depth
     * chunk, into the records the {@code <T>_MatMul_MMQ} and {@code <T>_MatMul_MMQ_KSplit} kernels
     * read, shared by all row tiles.
     */
    public static final String QUANT_MMQ_Q8_1 = "LLM_Quant_MMQ_Q8_1";
    /**
     * The warp-cooperative form of {@link #QUANT_MMQ_Q8_1} ({@code kernels/llm/quant_mmq_warp.cl}, CUDA): the same
     * records, written by {@code tk / 4} lanes per record from one float4 load each, so the row is read and the
     * records are stored in contiguous warp spans (llama.cpp {@code quantize_mmq_q8_1}'s lane mapping), where
     * the per-record kernel's lanes store 48 bytes apart.
     */
    public static final String QUANT_MMQ_Q8_1_WARP = "LLM_Quant_MMQ_Q8_1_Warp";
    /**
     * The static int8 activation staging of an {@link ActQuantScheme#INT8_STATIC}
     * weight ({@code kernels/llm/quant_mmq_q8_1.cl}): the codes are {@code sat8(rne(x / input_scale))} and the
     * chunk scale is {@code input_scale}, so the int8 MMQ GEMM multiplies the checkpoint's own int8 codes.
     */
    public static final String QUANT_MMQ_INT8_STATIC = "LLM_Quant_MMQ_Int8Static";
    /**
     * The dynamic per-token int8 staging of an {@link ActQuantScheme#INT8_TOKEN}
     * weight: one row scale {@code amax / 127}, codes {@code sat8(rne(x * 127 / amax))}.
     */
    public static final String QUANT_MMQ_INT8_ROW = "LLM_Quant_MMQ_Int8Row";
    /**
     * The dynamic per-128-run int8 staging of an
     * {@link ActQuantScheme#INT8_GROUP128} weight.
     */
    public static final String QUANT_MMQ_INT8_GROUP128 = "LLM_Quant_MMQ_Int8Group128";
    /** The static asymmetric int8 staging of an {@link ActQuantScheme#INT8_STATIC_ASYM} weight: codes
     *  {@code sat8(rne(x / input_scale) + input_zero_point)}, and the record carries the zero point the
     *  GEMM subtracts per weight row. */
    public static final String QUANT_MMQ_INT8_ASYM = "LLM_Quant_MMQ_Int8Asym";
    /** The static E4M3 staging of an {@link ActQuantScheme#FP8_STATIC} weight. */
    public static final String QUANT_MMQ_FP8_STATIC = "LLM_Quant_MMQ_Fp8Static";
    /** The per-token E4M3 staging of an {@link ActQuantScheme#FP8_TOKEN} weight. */
    public static final String QUANT_MMQ_FP8_ROW = "LLM_Quant_MMQ_Fp8Row";
    /** The per-128-run E4M3 staging of an {@link ActQuantScheme#FP8_GROUP128} weight. */
    public static final String QUANT_MMQ_FP8_GROUP128 = "LLM_Quant_MMQ_Fp8Group128";
    /** The power-of-two per-run E4M3 staging of an {@link ActQuantScheme#MXFP8} weight. */
    public static final String QUANT_MMQ_FP8_MX = "LLM_Quant_MMQ_Fp8Mx";
    /** The E2M1 staging of an {@link ActQuantScheme#MXFP4} weight, as the exact E4M3 codes of the values. */
    public static final String QUANT_MMQ_FP4_MX = "LLM_Quant_MMQ_Fp4Mx";
    /** The E2M1 staging of an {@link ActQuantScheme#NVFP4} weight, re-encoded to E4M3 with a chunk scale. */
    public static final String QUANT_MMQ_FP4_NV = "LLM_Quant_MMQ_Fp4Nv";
    /** The split-K slice reduce. */
    public static final String SPLIT_K_REDUCE = "LLM_SplitK_Reduce";
    /** The stream-k fixup: adds the partial sums of the tiles a stream-k MMQ launch split into their outputs. */
    public static final String STREAM_K_FIXUP = "LLM_MMQ_StreamK_Fixup";
    /** The gate/up stream-k fixup: writes {@code silu(gate) * up} of the summed partials of the split tiles of a
     *  {@code <T>_MatMul_MMQ_StreamK_GateUp} launch. */
    public static final String STREAM_K_FIXUP_GATE_UP = "LLM_MMQ_StreamK_Fixup_GateUp";
    /** Partial top-k over vocabulary slices. */
    public static final String TOPK_PARTIAL = "LLM_TopK_Partial";
    /** Row-wise top-k in rank order with the row's softmax denominators. */
    public static final String ROW_TOPK = "LLM_Row_TopK";
    /** Per-row pre-top-k logit edits: the logit-bias table, penalties and DRY. */
    public static final String LOGITS_PREFILTER = "LLM_LogitsPrefilter";

    /** {@code <T>_MatMul_Tiled}: weight rows (output features); argument 0 of every GEMM form. */
    public static final int ARG_M = 0;
    /** {@code <T>_MatMul_Tiled}: reduction length (input features). */
    public static final int ARG_K = 1;
    /** {@code <T>_MatMul_Tiled}: token count; set per step. */
    public static final int ARG_N = 2;
    /** {@code <T>_MatMul_Tiled}: element stride between activation rows. */
    public static final int ARG_X_STRIDE = 3;
    /** {@code <T>_MatMul_Tiled}: element offset of the first activation row; set per step for views. */
    public static final int ARG_X_OFF = 4;
    /** {@code <T>_MatMul_Tiled}: element stride between output rows. */
    public static final int ARG_Y_STRIDE = 5;
    /** {@code <T>_MatMul_Tiled}: element offset of the first output row; set per step for views. */
    public static final int ARG_Y_OFF = 6;
    /** {@code <T>_MatMul_Tiled}: weight bytes. */
    public static final int ARG_WEIGHT = 7;
    /** {@code <T>_MatMul_Tiled}: F32 activations. */
    public static final int ARG_X = 8;
    /** {@code <T>_MatMul_Tiled}: F32 output. */
    public static final int ARG_Y = 9;
    /**
     * Number of arguments of every {@code <T>_MatMul_Tiled} / {@code <T>_MatMul_MMQ} /
     * {@code <T>_MatMul_MMA} / {@code <T>_MatMul_CM} launch (the GEMM forms share the argument list).
     */
    public static final int MATMUL_TILED_ARG_COUNT = 10;

    /** {@code <T>_MatMul_MMA_KSplit}: weight rows (output features). */
    public static final int KSPLIT_ARG_M = 0;
    /** {@code <T>_MatMul_MMA_KSplit}: reduction length (input features). */
    public static final int KSPLIT_ARG_K = 1;
    /** {@code <T>_MatMul_MMA_KSplit}: token count. */
    public static final int KSPLIT_ARG_N = 2;
    /** {@code <T>_MatMul_MMA_KSplit}: element stride between activation rows. */
    public static final int KSPLIT_ARG_X_STRIDE = 3;
    /** {@code <T>_MatMul_MMA_KSplit}: element offset of the first activation row. */
    public static final int KSPLIT_ARG_X_OFF = 4;
    /** {@code <T>_MatMul_MMA_KSplit}: k length of every slice (a multiple of the stage depth). */
    public static final int KSPLIT_ARG_K_SLICE = 5;
    /** {@code <T>_MatMul_MMA_KSplit}: weight bytes. */
    public static final int KSPLIT_ARG_WEIGHT = 6;
    /** {@code <T>_MatMul_MMA_KSplit}: F32 activations. */
    public static final int KSPLIT_ARG_X = 7;
    /** {@code <T>_MatMul_MMA_KSplit}: F32 slice planes, {@code splits * M * N}. */
    public static final int KSPLIT_ARG_PARTIAL = 8;
    /** Number of arguments of a {@code <T>_MatMul_MMA_KSplit} launch. */
    public static final int KSPLIT_ARG_COUNT = 9;

    /** {@code LLM_SplitK_Reduce}: plane length in elements ({@code M * N}). */
    public static final int REDUCE_ARG_NE = 0;
    /** {@code LLM_SplitK_Reduce}: slice planes. */
    public static final int REDUCE_ARG_SPLITS = 1;
    /** {@code LLM_SplitK_Reduce}: weight rows; plane element {@code e} sits at token {@code e / M}, row {@code e % M}. */
    public static final int REDUCE_ARG_M = 2;
    /** {@code LLM_SplitK_Reduce}: element stride between output rows. */
    public static final int REDUCE_ARG_Y_STRIDE = 3;
    /** {@code LLM_SplitK_Reduce}: element offset of the first output row. */
    public static final int REDUCE_ARG_Y_OFF = 4;
    /** {@code LLM_SplitK_Reduce}: F32 slice planes. */
    public static final int REDUCE_ARG_PARTIAL = 5;
    /** {@code LLM_SplitK_Reduce}: F32 output. */
    public static final int REDUCE_ARG_Y = 6;
    /** Number of arguments of an {@code LLM_SplitK_Reduce} launch. */
    public static final int SPLIT_K_REDUCE_ARG_COUNT = 7;

    /** {@code LLM_TopK_Partial}: logits length. */
    public static final int TOPK_ARG_N = 0;
    /** {@code LLM_TopK_Partial}: candidates per slice. */
    public static final int TOPK_ARG_K = 1;
    /** {@code LLM_TopK_Partial}: scale applied to the reported candidate values; set per step. */
    public static final int TOPK_ARG_INV_TEMP = 2;
    /** {@code LLM_TopK_Partial}: F32 logits. */
    public static final int TOPK_ARG_LOGITS = 3;
    /** {@code LLM_TopK_Partial}: F32 candidate values, {@code groups * k}. */
    public static final int TOPK_ARG_CAND_VAL = 4;
    /** {@code LLM_TopK_Partial}: int32 candidate indices, {@code groups * k}; -1 marks an empty slot. */
    public static final int TOPK_ARG_CAND_IDX = 5;
    /** Number of {@code LLM_TopK_Partial} arguments. */
    public static final int TOPK_ARG_COUNT = 6;

    /** {@code LLM_Row_TopK}: logits per row. */
    public static final int ROWTOPK_ARG_SIZE = 0;
    /** {@code LLM_Row_TopK}: candidates per row. */
    public static final int ROWTOPK_ARG_K = 1;
    /** {@code LLM_Row_TopK}: rows. */
    public static final int ROWTOPK_ARG_N_ROWS = 2;
    /** {@code LLM_Row_TopK}: F32 logits, {@code n_rows} rows of {@code size}. */
    public static final int ROWTOPK_ARG_LOGITS = 3;
    /** {@code LLM_Row_TopK}: F32 candidate values, {@code n_rows * k}. */
    public static final int ROWTOPK_ARG_TOP_VAL = 4;
    /** {@code LLM_Row_TopK}: int32 candidate indices, {@code n_rows * k}; -1 marks an empty slot. */
    public static final int ROWTOPK_ARG_TOP_IDX = 5;
    /** {@code LLM_Row_TopK}: F32 softmax denominator of each row, {@code n_rows}. */
    public static final int ROWTOPK_ARG_SUM_EXP = 6;
    /** Number of {@code LLM_Row_TopK} arguments. */
    public static final int ROWTOPK_ARG_COUNT = 7;

    /** Weight types with a {@code <T>_MatMul_Tiled} kernel: F32, F16, BF16 and every GPU quant type
     *  (the F8_E5M2, MXFP6_E3M2 and MXFP6_E2M3 extension types included). */
    public static final Set<GGMLType> TILED_TYPES = Collections.unmodifiableSet(EnumSet.of(
            GGMLType.F32, GGMLType.F16, GGMLType.BF16,
            GGMLType.Q4_0, GGMLType.Q4_1, GGMLType.Q5_0, GGMLType.Q5_1, GGMLType.Q8_0,
            GGMLType.IQ4_NL, GGMLType.IQ4_XS, GGMLType.MXFP4,
            GGMLType.Q2_K, GGMLType.Q3_K, GGMLType.Q4_K, GGMLType.Q5_K, GGMLType.Q6_K, GGMLType.Q8_K,
            GGMLType.IQ3_XXS, GGMLType.IQ3_S, GGMLType.IQ2_XXS, GGMLType.IQ2_XS, GGMLType.IQ2_S,
            GGMLType.IQ3_K, GGMLType.IQ3_KS,
            GGMLType.IQ1_S, GGMLType.IQ1_M,
            GGMLType.TQ1_0, GGMLType.TQ2_0, GGMLType.NVFP4, GGMLType.Q1_0, GGMLType.Q2_0,
            GGMLType.F8_E4M3, GGMLType.F8_E4M3_B128, GGMLType.F8_E4M3_B32, GGMLType.MXFP8, GGMLType.MXFP6_E3M2,
            GGMLType.MXFP6_E2M3, GGMLType.I8_B128, GGMLType.NF4,
            GGMLType.FP4_BNB, GGMLType.F8_E5M2));

    /** The E4M3 weight types of the FP8 W8A8 lane: the checkpoint's FP8 codes (per tensor, per 128 and per
     *  32 blocks) and the MXFP8 E8M0-scaled form. They join {@link #MMQ_TYPES}, whose staged-activation GEMM
     *  multiplies their codes by the scheme's E4M3 activation codes through the FP8 tensor cores. */
    public static final Set<GGMLType> FP8_TYPES = Collections.unmodifiableSet(EnumSet.of(
            GGMLType.F8_E4M3, GGMLType.F8_E4M3_B128, GGMLType.F8_E4M3_B32, GGMLType.MXFP8));

    /** The 4-bit weight types of the W4A8 lane: the MXFP4 and NVFP4 codebooks and the IQ4_NL / IQ4_XS ones,
     *  whose codes map to E4M3 codes the FP8 tensor cores multiply by the staged E4M3 activations. */
    public static final Set<GGMLType> W4A8_TYPES = Collections.unmodifiableSet(EnumSet.of(
            GGMLType.MXFP4, GGMLType.NVFP4, GGMLType.IQ4_NL, GGMLType.IQ4_XS));

    /** Weight types of the int8 MMQ GEMM: the set llama.cpp serves with MMQ (ggml-cuda/mmq.cu), the legacy
     *  32-element blocks, the 256-element K-quants and the i-quants iq1_s..iq4_xs, plus the int8 checkpoint type
     *  I8_B128 (a per-128 scale over int8 codes, the W8A8 int8 weight format). The FP8 types share the kernel
     *  name through {@link #FP8_TYPES} but carry the E4M3 datapath. */
    public static final Set<GGMLType> MMQ_TYPES = Collections.unmodifiableSet(EnumSet.of(
            GGMLType.Q4_0, GGMLType.Q4_1, GGMLType.Q5_0, GGMLType.Q5_1, GGMLType.Q8_0,
            GGMLType.Q2_K, GGMLType.Q3_K, GGMLType.Q4_K, GGMLType.Q5_K, GGMLType.Q6_K,
            GGMLType.IQ1_S, GGMLType.IQ2_XXS, GGMLType.IQ2_XS, GGMLType.IQ2_S,
            GGMLType.IQ3_XXS, GGMLType.IQ3_S, GGMLType.IQ4_NL, GGMLType.IQ4_XS,
            GGMLType.I8_B128));

    /** Weight types of the tensor-core MMA GEMM family: the dense 16-bit types F16 and BF16. */
    public static final Set<GGMLType> MMA_TYPES = Collections.unmodifiableSet(EnumSet.of(
            GGMLType.F16, GGMLType.BF16));

    /** Weight types of the workgroup-scope cooperative-matrix GEMM ({@code <T>_MatMul_CM2},
     *  VK_NV_cooperative_matrix2): the MMQ set whose blocks decode per element without the shared
     *  grid tables, i.e. every {@link #MMQ_TYPES} entry except the i-quants, which keep the KHR
     *  {@code <T>_MatMul_CM} lane, and the dense F16 type, whose halves the tensor loads read directly (BF16
     *  keeps the exact tiled GEMM on Vulkan, see {@link #MMA_TYPES}). */
    public static final Set<GGMLType> CM2_TYPES = Collections.unmodifiableSet(EnumSet.of(
            GGMLType.F16,
            GGMLType.Q4_0, GGMLType.Q4_1, GGMLType.Q5_0, GGMLType.Q5_1, GGMLType.Q8_0,
            GGMLType.Q2_K, GGMLType.Q3_K, GGMLType.Q4_K, GGMLType.Q5_K, GGMLType.Q6_K,
            GGMLType.I8_B128));

    private GemmKernels() {
    }

    /** Whether {@code type} has a tiled GEMM kernel. */
    public static boolean hasTiled(GGMLType type) {
        return type != null && TILED_TYPES.contains(type);
    }

    /**
     * Name of the tiled GEMM kernel of a weight type.
     *
     * @throws IllegalArgumentException when the type has no tiled GEMM kernel
     */
    public static String matmulTiled(GGMLType type) {
        if (!hasTiled(type)) {
            throw new IllegalArgumentException("no tiled GEMM kernel for weight type " + type
                    + "; supported: " + TILED_TYPES);
        }
        return type.name() + MATMUL_TILED_SUFFIX;
    }

    /**
     * Whether {@code type} has a staged-activation GEMM kernel of the {@code <T>_MatMul_MMQ} name: the int8 MMQ
     * types ({@link #MMQ_TYPES}) and the FP8 types ({@link #FP8_TYPES}, whose kernel carries the E4M3 datapath).
     */
    public static boolean hasMmq(GGMLType type) {
        return type != null && (MMQ_TYPES.contains(type) || FP8_TYPES.contains(type));
    }

    /** Whether {@code type} has the int8 MMQ datapath ({@link #MMQ_TYPES}). */
    public static boolean isInt8Mmq(GGMLType type) {
        return type != null && MMQ_TYPES.contains(type);
    }

    /**
     * The int8 activation staging kernel that applies an activation quantization scheme directly into the MMQ
     * record layout ({@link #QUANT_MMQ_INT8_STATIC}, {@link #QUANT_MMQ_INT8_ROW},
     * {@link #QUANT_MMQ_INT8_GROUP128} or {@link #QUANT_MMQ_INT8_ASYM}), or null for the schemes that only the
     * fake-quantization path serves.
     */
    public static String int8ActivationStaging(ActQuantScheme scheme) {
        if (scheme == null) {
            return null;
        }
        switch (scheme) {
            case INT8_STATIC:
                return QUANT_MMQ_INT8_STATIC;
            case INT8_TOKEN:
                return QUANT_MMQ_INT8_ROW;
            case INT8_GROUP128:
                return QUANT_MMQ_INT8_GROUP128;
            case INT8_STATIC_ASYM:
                return QUANT_MMQ_INT8_ASYM;
            default:
                return null;
        }
    }

    /**
     * Name of the int8 MMQ GEMM kernel of a weight type.
     *
     * @throws IllegalArgumentException when the type has no MMQ GEMM kernel
     */
    public static String matmulMmq(GGMLType type) {
        if (!hasMmq(type)) {
            throw new IllegalArgumentException("no MMQ GEMM kernel for weight type " + type
                    + "; supported: " + MMQ_TYPES);
        }
        return type.name() + MATMUL_MMQ_SUFFIX;
    }

    /**
     * Name of the split-K twin of the int8 MMQ GEMM kernel of a weight type
     * ({@code <T>_MatMul_MMQ_KSplit}).
     *
     * @throws IllegalArgumentException when the type has no MMQ GEMM kernel
     */
    public static String matmulMmqKSplit(GGMLType type) {
        if (!hasMmq(type)) {
            throw new IllegalArgumentException("no split-K MMQ GEMM kernel for weight type " + type
                    + "; supported: " + MMQ_TYPES);
        }
        return type.name() + MATMUL_MMQ_KSPLIT_SUFFIX;
    }

    /**
     * Name of the stream-k form of the int8 MMQ GEMM kernel of a weight type
     * ({@code <T>_MatMul_MMQ_StreamK}): CUDA sm_80+ only (the tensor tile), so callers select it through
     * {@code ops.optionalReady(name, workGroup, 1)}.
     *
     * @throws IllegalArgumentException when the type has no int8 MMQ GEMM kernel
     */
    public static String matmulMmqStreamK(GGMLType type) {
        if (!isInt8Mmq(type)) {
            throw new IllegalArgumentException("no stream-k MMQ GEMM kernel for weight type " + type
                    + "; supported: " + MMQ_TYPES);
        }
        return type.name() + MATMUL_MMQ_STREAMK_SUFFIX;
    }

    /**
     * Name of the fused gate/up SwiGLU stream-k kernel of a weight type ({@code <T>_MatMul_MMQ_StreamK_GateUp}):
     * CUDA sm_80+ only, like {@link #matmulMmqStreamK}.
     *
     * @throws IllegalArgumentException when the type has no int8 MMQ GEMM kernel
     */
    public static String matmulMmqStreamKGateUp(GGMLType type) {
        return matmulMmqStreamK(type).replace(MATMUL_MMQ_STREAMK_SUFFIX, MATMUL_MMQ_STREAMK_GATE_UP_SUFFIX);
    }

    /**
     * Name of the small-token-tile twin of the staged GEMM kernel of a weight type
     * ({@code <T>_MatMul_MMQ_S}, also for the FP8 types; llama.cpp picks the J tile per launch,
     * ggml_cuda_mmq_get_J_best). The twin exists on every device whose MMQ tile is wider than 16
     * tokens (the CUDA and HIP tiles, and the OpenCL forced tiles over 16); a 16-token device tile
     * has no twin and the plain kernel serves, so callers select the twin through
     * {@code ops.optionalReady(name, workGroup, 1)}.
     *
     * @throws IllegalArgumentException when the type has no MMQ GEMM kernel
     */
    public static String matmulMmqSmall(GGMLType type) {
        return matmulMmqSmall(type, MmqConfig.TOKEN_LANES);
    }

    /**
     * Name of the small-token-tile twin of the staged GEMM kernel of a weight type at the given
     * twin width: {@code <T>_MatMul_MMQ_S} at 16 tokens, {@code <T>_MatMul_MMQ_S64} at 64 (the
     * ladder row the 17..64-token launches run, llama.cpp's J=64).
     *
     * @throws IllegalArgumentException when the type has no MMQ GEMM kernel or {@code tokens} is
     *                                  not a twin width
     */
    public static String matmulMmqSmall(GGMLType type, int tokens) {
        if (!hasMmq(type)) {
            throw new IllegalArgumentException("no small-token-tile MMQ GEMM kernel for weight type " + type
                    + "; supported: " + MMQ_TYPES);
        }
        return type.name() + mmqSmallSuffix(tokens, MATMUL_MMQ_S_SUFFIX, MATMUL_MMQ_S64_SUFFIX);
    }

    /**
     * Name of the small-token-tile twin of the W4A8 GEMM kernel of a 4-bit weight type
     * ({@code <T>_MatMul_W4A8_S}).
     *
     * @throws IllegalArgumentException when the type has no W4A8 GEMM kernel
     */
    public static String matmulW4a8Small(GGMLType type) {
        return matmulW4a8Small(type, MmqConfig.TOKEN_LANES);
    }

    /**
     * Name of the small-token-tile twin of the W4A8 GEMM kernel of a 4-bit weight type at the
     * given twin width: {@code <T>_MatMul_W4A8_S} at 16 tokens, {@code <T>_MatMul_W4A8_S64} at 64.
     *
     * @throws IllegalArgumentException when the type has no W4A8 GEMM kernel or {@code tokens} is
     *                                  not a twin width
     */
    public static String matmulW4a8Small(GGMLType type, int tokens) {
        if (!isW4a8(type)) {
            throw new IllegalArgumentException("no small-token-tile W4A8 GEMM kernel for weight type " + type
                    + "; supported: " + W4A8_TYPES);
        }
        return type.name() + mmqSmallSuffix(tokens, MATMUL_W4A8_S_SUFFIX, MATMUL_W4A8_S64_SUFFIX);
    }

    /**
     * Name of the split-K twin of the small-token-tile MMQ GEMM kernel of a weight type
     * ({@code <T>_MatMul_MMQ_SKSplit}, also for the FP8 types): the partial kernel the staged
     * launches of fewer tokens than the device tile split with.
     *
     * @throws IllegalArgumentException when the type has no MMQ GEMM kernel
     */
    public static String matmulMmqSmallSplit(GGMLType type) {
        return matmulMmqSmallSplit(type, MmqConfig.TOKEN_LANES);
    }

    /**
     * Name of the split-K twin of the small-token-tile MMQ GEMM kernel of a weight type at the
     * given twin width: {@code <T>_MatMul_MMQ_SKSplit} at 16 tokens, {@code <T>_MatMul_MMQ_SKSplit64}
     * at 64.
     *
     * @throws IllegalArgumentException when the type has no MMQ GEMM kernel or {@code tokens} is
     *                                  not a twin width
     */
    public static String matmulMmqSmallSplit(GGMLType type, int tokens) {
        if (!hasMmq(type)) {
            throw new IllegalArgumentException("no small-token-tile split-K MMQ GEMM kernel for weight type " + type
                    + "; supported: " + MMQ_TYPES);
        }
        return type.name() + mmqSmallSuffix(tokens, MATMUL_MMQ_SKSPLIT_SUFFIX, MATMUL_MMQ_SKSPLIT64_SUFFIX);
    }

    /**
     * Name of the split-K twin of the small-token-tile W4A8 GEMM kernel of a 4-bit weight type
     * ({@code <T>_MatMul_W4A8_SKSplit}).
     *
     * @throws IllegalArgumentException when the type has no W4A8 GEMM kernel
     */
    public static String matmulW4a8SmallSplit(GGMLType type) {
        return matmulW4a8SmallSplit(type, MmqConfig.TOKEN_LANES);
    }

    /**
     * Name of the split-K twin of the small-token-tile W4A8 GEMM kernel of a 4-bit weight type at
     * the given twin width: {@code <T>_MatMul_W4A8_SKSplit} at 16, {@code <T>_MatMul_W4A8_SKSplit64}
     * at 64.
     *
     * @throws IllegalArgumentException when the type has no W4A8 GEMM kernel or {@code tokens} is
     *                                  not a twin width
     */
    public static String matmulW4a8SmallSplit(GGMLType type, int tokens) {
        if (!isW4a8(type)) {
            throw new IllegalArgumentException("no small-token-tile split-K W4A8 GEMM kernel for weight type " + type
                    + "; supported: " + W4A8_TYPES);
        }
        return type.name() + mmqSmallSuffix(tokens, MATMUL_W4A8_SKSPLIT_SUFFIX, MATMUL_W4A8_SKSPLIT64_SUFFIX);
    }

    /** Suffix of a twin kernel name at the twin width ({@link MmqConfig#twinTokens(int, int)}). */
    private static String mmqSmallSuffix(int tokens, String suffix16, String suffix64) {
        return tokens == MmqConfig.TOKEN_LANES ? suffix16
                : tokens == MmqConfig.TWIN_MAX_TOKENS ? suffix64
                : throwBadTwinWidth(tokens);
    }

    private static String throwBadTwinWidth(int tokens) {
        throw new IllegalArgumentException("MMQ twin kernels come in " + MmqConfig.TOKEN_LANES + " and "
                + MmqConfig.TWIN_MAX_TOKENS + " tokens, got " + tokens);
    }

    /**
     * Name of the tensor-core MMA GEMM kernel of a dense 16-bit weight type ({@code F16_MatMul_MMA},
     * {@code BF16_MatMul_MMA}).
     *
     * @throws IllegalArgumentException when the type has no MMA GEMM kernel
     */
    public static String matmulMma(GGMLType type) {
        if (!hasMma(type)) {
            throw new IllegalArgumentException("no MMA GEMM kernel for weight type " + type
                    + "; supported: " + MMA_TYPES);
        }
        return type.name() + MATMUL_MMA_SUFFIX;
    }

    /** Whether {@code type} has a tensor-core MMA GEMM kernel (the dense 16-bit types). */
    public static boolean hasMma(GGMLType type) {
        return type != null && MMA_TYPES.contains(type);
    }

    /**
     * Name of the split-K twin of the MMA GEMM kernel of a dense 16-bit weight type
     * ({@code <T>_MatMul_MMA_KSplit}).
     *
     * @throws IllegalArgumentException when the type has no MMA GEMM kernel
     */
    public static String matmulMmaKSplit(GGMLType type) {
        if (!hasMma(type)) {
            throw new IllegalArgumentException("no split-K MMA GEMM kernel for weight type " + type
                    + "; supported: " + MMA_TYPES);
        }
        return type.name() + MATMUL_MMA_KSPLIT_SUFFIX;
    }

    /**
     * Whether {@code type} has a dequantizing {@code <T>_MatMul_CM} GEMM kernel: the MMQ set except the FP8
     * types, whose cooperative-matrix form would need a float8 component type the shader toolchain does not
     * provide; they keep the exact tiled GEMM on Vulkan.
     */
    public static boolean hasCm(GGMLType type) {
        return hasMmq(type) && !FP8_TYPES.contains(type);
    }

    /**
     * Whether {@code type} has a workgroup-scope cooperative-matrix {@code <T>_MatMul_CM2} GEMM kernel
     * ({@link #CM2_TYPES}), the Vulkan form a device with {@code DeviceFeature.COOPERATIVE_MATRIX_2}
     * prefers over the KHR {@code <T>_MatMul_CM} one.
     */
    public static boolean hasCm2(GGMLType type) {
        return type != null && CM2_TYPES.contains(type);
    }

    /** Whether {@code type} is an E4M3 weight type of the FP8 lane ({@link #FP8_TYPES}). */
    public static boolean isFp8(GGMLType type) {
        return type != null && FP8_TYPES.contains(type);
    }

    /** Whether {@code type} is a 4-bit weight type of the W4A8 lane ({@link #W4A8_TYPES}). */
    public static boolean isW4a8(GGMLType type) {
        return type != null && W4A8_TYPES.contains(type);
    }

    /**
     * Name of the E4M3 GEMM kernel of a 4-bit weight type ({@code <T>_MatMul_W4A8}).
     *
     * @throws IllegalArgumentException when the type has no W4A8 GEMM kernel
     */
    public static String matmulW4a8(GGMLType type) {
        if (!isW4a8(type)) {
            throw new IllegalArgumentException("no W4A8 GEMM kernel for weight type " + type
                    + "; supported: " + W4A8_TYPES);
        }
        return type.name() + MATMUL_W4A8_SUFFIX;
    }

    /**
     * Name of the split-K twin of the W4A8 GEMM kernel of a 4-bit weight type
     * ({@code <T>_MatMul_W4A8_KSplit}).
     *
     * @throws IllegalArgumentException when the type has no W4A8 GEMM kernel
     */
    public static String matmulW4a8KSplit(GGMLType type) {
        if (!isW4a8(type)) {
            throw new IllegalArgumentException("no split-K W4A8 GEMM kernel for weight type " + type
                    + "; supported: " + W4A8_TYPES);
        }
        return type.name() + MATMUL_W4A8_KSPLIT_SUFFIX;
    }

    /**
     * The E4M3 activation staging kernel that applies an activation scheme directly into the staged record
     * layout ({@link #QUANT_MMQ_FP8_STATIC}, {@link #QUANT_MMQ_FP8_ROW}, {@link #QUANT_MMQ_FP8_GROUP128},
     * {@link #QUANT_MMQ_FP8_MX}, {@link #QUANT_MMQ_FP4_MX} or {@link #QUANT_MMQ_FP4_NV}), or null for the
     * schemes the E4M3 lanes do not stage.
     */
    public static String fp8ActivationStaging(ActQuantScheme scheme) {
        if (scheme == null) {
            return null;
        }
        switch (scheme) {
            case FP8_STATIC:
                return QUANT_MMQ_FP8_STATIC;
            case FP8_TOKEN:
                return QUANT_MMQ_FP8_ROW;
            case FP8_GROUP128:
                return QUANT_MMQ_FP8_GROUP128;
            case MXFP8:
                return QUANT_MMQ_FP8_MX;
            case MXFP4:
                return QUANT_MMQ_FP4_MX;
            case NVFP4:
                return QUANT_MMQ_FP4_NV;
            default:
                return null;
        }
    }

    /** Whether an activation scheme is staged in the E4M3 code form (the E4M3 lanes' record layout). */
    public static boolean isFp8Scheme(ActQuantScheme scheme) {
        return fp8ActivationStaging(scheme) != null;
    }

    /**
     * Name of the dequantizing GEMM kernel of an MMQ weight type ({@code <T>_MatMul_CM}).
     *
     * @throws IllegalArgumentException when the type has no MMQ GEMM kernel
     */
    public static String matmulCm(GGMLType type) {
        if (!hasCm(type)) {
            throw new IllegalArgumentException("no CM GEMM kernel for weight type " + type
                    + "; supported: " + MMQ_TYPES);
        }
        return type.name() + MATMUL_CM_SUFFIX;
    }

    /**
     * Name of the split-K twin of the dequantizing CM GEMM kernel of an MMQ weight type
     * ({@code <T>_MatMul_CM_KSplit}).
     *
     * @throws IllegalArgumentException when the type has no MMQ GEMM kernel
     */
    public static String matmulCmKSplit(GGMLType type) {
        if (!hasCm(type)) {
            throw new IllegalArgumentException("no split-K CM GEMM kernel for weight type " + type
                    + "; supported: " + MMQ_TYPES);
        }
        return type.name() + MATMUL_CM_KSPLIT_SUFFIX;
    }

    /**
     * Name of the workgroup-scope cooperative-matrix GEMM kernel of a CM2 weight type
     * ({@code <T>_MatMul_CM2}).
     *
     * @throws IllegalArgumentException when the type has no CM2 GEMM kernel
     */
    public static String matmulCm2(GGMLType type) {
        if (!hasCm2(type)) {
            throw new IllegalArgumentException("no CM2 GEMM kernel for weight type " + type
                    + "; supported: " + CM2_TYPES);
        }
        return type.name() + MATMUL_CM2_SUFFIX;
    }

    /**
     * Name of the split-K twin of the CM2 GEMM kernel of a CM2 weight type
     * ({@code <T>_MatMul_CM2_KSplit}).
     *
     * @throws IllegalArgumentException when the type has no CM2 GEMM kernel
     */
    public static String matmulCm2KSplit(GGMLType type) {
        if (!hasCm2(type)) {
            throw new IllegalArgumentException("no split-K CM2 GEMM kernel for weight type " + type
                    + "; supported: " + CM2_TYPES);
        }
        return type.name() + MATMUL_CM2_KSPLIT_SUFFIX;
    }

    /** Every kernel name of this family: tiled, MMQ, MMQ KSplit, MMQ S, MMQ S64, MMQ SKSplit,
     *  MMQ SKSplit64, CM and CM KSplit GEMM kernels in {@link GGMLType} order, the CM2 pair of the
     *  non-i-quant types and of F16 (after the MMA pair), the FP8 and W4A8 E4M3 GEMMs with their S, S64, SKSplit and SKSplit64
     *  twins, then the activation staging kernels, the reduce and the top-k. */
    public static List<String> kernelNames() {
        List<String> names = new ArrayList<>(TILED_TYPES.size() + 8 * MMQ_TYPES.size() + 2 * CM2_TYPES.size()
                + 6 * FP8_TYPES.size() + MMA_TYPES.size() + 6 * W4A8_TYPES.size() + 7);
        for (GGMLType type : TILED_TYPES) {
            names.add(matmulTiled(type));
        }
        for (GGMLType type : MMQ_TYPES) {
            names.add(matmulMmq(type));
            names.add(matmulMmqKSplit(type));
            names.add(matmulMmqSmall(type));
            names.add(matmulMmqSmall(type, MmqConfig.TWIN_MAX_TOKENS));
            names.add(matmulMmqSmallSplit(type));
            names.add(matmulMmqSmallSplit(type, MmqConfig.TWIN_MAX_TOKENS));
            names.add(matmulCm(type));
            names.add(matmulCmKSplit(type));
            if (hasCm2(type)) {
                names.add(matmulCm2(type));
                names.add(matmulCm2KSplit(type));
            }
        }
        for (GGMLType type : FP8_TYPES) {
            names.add(matmulMmq(type));
            names.add(matmulMmqKSplit(type));
            names.add(matmulMmqSmall(type));
            names.add(matmulMmqSmall(type, MmqConfig.TWIN_MAX_TOKENS));
            names.add(matmulMmqSmallSplit(type));
            names.add(matmulMmqSmallSplit(type, MmqConfig.TWIN_MAX_TOKENS));
        }
        for (GGMLType type : MMA_TYPES) {
            names.add(matmulMma(type));
            names.add(matmulMmaKSplit(type));
            if (hasCm2(type)) {
                names.add(matmulCm2(type));
                names.add(matmulCm2KSplit(type));
            }
        }
        for (GGMLType type : W4A8_TYPES) {
            names.add(matmulW4a8(type));
            names.add(matmulW4a8KSplit(type));
            names.add(matmulW4a8Small(type));
            names.add(matmulW4a8Small(type, MmqConfig.TWIN_MAX_TOKENS));
            names.add(matmulW4a8SmallSplit(type));
            names.add(matmulW4a8SmallSplit(type, MmqConfig.TWIN_MAX_TOKENS));
        }
        names.add(QUANT_MMQ_Q8_1);
        names.add(QUANT_MMQ_INT8_STATIC);
        names.add(QUANT_MMQ_INT8_ROW);
        names.add(QUANT_MMQ_INT8_GROUP128);
        names.add(QUANT_MMQ_INT8_ASYM);
        names.add(QUANT_MMQ_FP8_STATIC);
        names.add(QUANT_MMQ_FP8_ROW);
        names.add(QUANT_MMQ_FP8_GROUP128);
        names.add(QUANT_MMQ_FP8_MX);
        names.add(QUANT_MMQ_FP4_MX);
        names.add(QUANT_MMQ_FP4_NV);
        names.add(SPLIT_K_REDUCE);
        names.add(TOPK_PARTIAL);
        names.add(ROW_TOPK);
        names.add(LOGITS_PREFILTER);
        return Collections.unmodifiableList(names);
    }

    /**
     * Dispatch geometry of {@code LLM_TopK_Partial}: {@code groups} workgroups of {@code local}
     * work-items, each reducing one vocabulary slice of {@code ceil(n / groups)} logits to its
     * {@code k} best candidates, so the host reads {@code groups * k} candidates instead of {@code n}
     * logits.
     */
    public static final class TopKPartialPlan {

        /** Largest supported {@code k}. */
        public static final int MAX_K = 64;
        /** Largest logits length; keeps the kernel's slice bounds and strided scan within {@code int}. */
        public static final int MAX_N = 1 << 30;
        /** Logits per workgroup the group count aims for. */
        public static final int SLICE_TARGET = 4096;
        /** Largest group count; far below every device's 1-D group limit, so dispatches are never folded. */
        public static final int MAX_GROUPS = 1024;
        /** Smallest work-group size (one full CUDA warp). */
        public static final int MIN_LOCAL = 32;

        private final int n;
        private final int k;
        private final int groups;
        private final int local;

        private TopKPartialPlan(int n, int k, int groups, int local) {
            this.n = n;
            this.k = k;
            this.groups = groups;
            this.local = local;
        }

        /**
         * Plans a partial top-k over {@code n} logits.
         *
         * @param local work-group size: a power of two in [{@link #MIN_LOCAL}, {@code LLM_DIMX}]
         * @throws IllegalArgumentException for {@code n} outside [1, {@link #MAX_N}], {@code k} outside
         *                                  [1, {@link #MAX_K}] or an invalid work-group size
         */
        public static TopKPartialPlan of(int n, int k, int local) {
            if (n < 1) {
                throw new IllegalArgumentException("LLM_TopK_Partial needs at least one logit, got n=" + n);
            }
            if (n > MAX_N) {
                throw new IllegalArgumentException("LLM_TopK_Partial supports at most " + MAX_N + " logits, got n=" + n);
            }
            if (k < 1 || k > MAX_K) {
                throw new IllegalArgumentException("LLM_TopK_Partial supports 1 <= k <= " + MAX_K + ", got k=" + k);
            }
            if (local < MIN_LOCAL || local > LlmKernelConstants.LLM_DIMX || Integer.bitCount(local) != 1) {
                throw new IllegalArgumentException("LLM_TopK_Partial work-group size must be a power of two in ["
                        + MIN_LOCAL + ", " + LlmKernelConstants.LLM_DIMX + "], got " + local);
            }
            long wanted = ((long) n + SLICE_TARGET - 1) / SLICE_TARGET;
            int groups = (int) Math.max(1L, Math.min(MAX_GROUPS, wanted));
            return new TopKPartialPlan(n, k, groups, local);
        }

        public int n() {
            return n;
        }

        public int k() {
            return k;
        }

        /** Workgroups, one per vocabulary slice. */
        public int groups() {
            return groups;
        }

        /** Logits per slice, {@code ceil(n / groups)}; the last slices may be shorter or empty. */
        public int sliceLength() {
            return (int) (((long) n + groups - 1) / groups);
        }

        /** Candidate slots written: {@code groups * k} (the length of both candidate buffers). */
        public int candidateCount() {
            return groups * k;
        }

        /** {@code {groups * local}}. */
        public long[] globalWorkSize() {
            return new long[]{(long) groups * local};
        }

        /** {@code {local}}. */
        public long[] localWorkSize() {
            return new long[]{local};
        }

        /**
         * Whether the device can run this launch: the work-group size and the 1-D group count.
         *
         * @return null when it can, otherwise the reason
         */
        public String checkDevice(DeviceCapabilities caps) {
            Objects.requireNonNull(caps, "caps");
            long maxWg = caps.getMaxWorkGroupSize();
            if (maxWg > 0 && maxWg < local) {
                return TOPK_PARTIAL + " needs work-groups of " + local + " work-items; device "
                        + caps.getDeviceName() + " allows " + maxWg;
            }
            long maxGroups = caps.getMaxWorkGroupCount()[0];
            if (groups > maxGroups) {
                return TOPK_PARTIAL + " needs " + groups + " work-groups; device " + caps.getDeviceName()
                        + " allows " + maxGroups;
            }
            return null;
        }

        /**
         * Kernel arguments in declared order ({@link #TOPK_ARG_N} .. {@link #TOPK_ARG_CAND_IDX}), after
         * checking that the three buffers are distinct and cover {@code n} logits and
         * {@link #candidateCount()} candidate slots.
         *
         * @param invTemp inverse sampling temperature applied to the reported values; positive and finite
         * @throws IllegalArgumentException for a non-positive or non-finite {@code invTemp}, a shared
         *                                  buffer or a buffer smaller than the launch needs
         * @throws NullPointerException     for a null buffer
         */
        public Object[] args(float invTemp, GpuBuffer logits, GpuBuffer candVal, GpuBuffer candIdx) {
            if (!(invTemp > 0.0f) || Float.isInfinite(invTemp)) {
                throw new IllegalArgumentException("LLM_TopK_Partial inv_temp must be positive and finite, got " + invTemp);
            }
            Objects.requireNonNull(logits, "logits");
            Objects.requireNonNull(candVal, "candVal");
            Objects.requireNonNull(candIdx, "candIdx");
            if (logits == candVal || logits == candIdx || candVal == candIdx) {
                throw new IllegalArgumentException(TOPK_PARTIAL + ": logits, cand_val and cand_idx must be distinct buffers");
            }
            requireBytes("logits", logits, (long) n * Float.BYTES);
            requireBytes("cand_val", candVal, (long) candidateCount() * Float.BYTES);
            requireBytes("cand_idx", candIdx, (long) candidateCount() * Integer.BYTES);
            Object[] args = new Object[TOPK_ARG_COUNT];
            args[TOPK_ARG_N] = n;
            args[TOPK_ARG_K] = k;
            args[TOPK_ARG_INV_TEMP] = invTemp;
            args[TOPK_ARG_LOGITS] = logits;
            args[TOPK_ARG_CAND_VAL] = candVal;
            args[TOPK_ARG_CAND_IDX] = candIdx;
            return args;
        }

        private void requireBytes(String role, GpuBuffer buffer, long needed) {
            long size = buffer.sizeInBytes();
            if (size < needed) {
                throw new IllegalArgumentException(TOPK_PARTIAL + ": " + role + " buffer holds " + size
                        + " bytes but the launch (n=" + n + ", k=" + k + ", groups=" + groups + ") needs " + needed);
            }
        }

        @Override
        public String toString() {
            return "TopKPartialPlan{n=" + n + ", k=" + k + ", groups=" + groups + ", local=" + local + '}';
        }
    }
}

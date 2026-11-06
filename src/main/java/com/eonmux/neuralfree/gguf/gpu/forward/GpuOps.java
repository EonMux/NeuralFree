package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.GpuBackendException;
import com.eonmux.neuralfree.acceleration.GpuModelUnsupportedException;
import com.eonmux.neuralfree.acceleration.GpuOutOfMemoryException;
import com.eonmux.neuralfree.acceleration.UnsupportedKernelException;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities;
import com.eonmux.neuralfree.acceleration.backend.DeviceFeature;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.acceleration.backend.KernelHandle;
import com.eonmux.neuralfree.gguf.BatchInvariance;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.arch.GluOp;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.kernels.LlmKernelConstants;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuPlaceholders;
import com.eonmux.neuralfree.gguf.ops.act.ActKernels;
import com.eonmux.neuralfree.gguf.ops.fused.FusedKernels;
import com.eonmux.neuralfree.gguf.ops.gemm.GemmKernels;
import com.eonmux.neuralfree.gguf.ops.gemm.MmqConfig;
import com.eonmux.neuralfree.gguf.ops.gemm.MmqMode;
import com.eonmux.neuralfree.gguf.ops.gemm.SplitKGemmPlan;
import com.eonmux.neuralfree.gguf.ops.norm.NormKernels;
import com.eonmux.neuralfree.gguf.quant.ActQuantScheme;
import com.eonmux.neuralfree.gguf.quant.BlockLayout;
import com.eonmux.neuralfree.gguf.quant.HadamardSpec;
import com.eonmux.neuralfree.gguf.quant.InputTransform;
import com.eonmux.neuralfree.gguf.spec.ActParams;
import com.eonmux.neuralfree.utils.LoggingUtil;
import org.apache.logging.log4j.Logger;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.EnumMap;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * Typed dispatch helpers of the GPU forward path over one queue. Buffer arguments are never null; activations
 * are row-major {@code [t][width]} float views given as (buffer, element offset).
 *
 * <p><b>Projections.</b> {@link #matvec}, {@link #matvecAdd} and {@link #matmul} compute
 * {@code out[t][r] = scale * sum_k W[r][k] * x[t][k]} for {@code nTokens} rows and pick the kernel by weight type
 * ({@code {T}_MatVec}, {@code {T}_MatVec2}, {@code {T}_MatVecAdd}, {@code {T}_BatchMatMul[_2D]}, {@code {T}_MatMul_Tiled};
 * F32, Q8_0 and
 * MXFP4 keep their legacy names). They loop over the weight's row shards, writing each shard's rows at
 * {@code outOff + t*rows + shardRowStart}. The matvec kernels have no offset arguments, so an input at a nonzero
 * offset is first copied to a staging buffer, and an output at a nonzero offset or of a sharded weight is computed into
 * staging and scattered with {@value #COPY2D}. The tiled GEMM takes offsets and strides itself. A PRISM weight
 * ({@link GpuWeight#hadamard()}) first rotates its input into staging with {@value HadamardSpec#KERNEL}
 * (llama-graph.cpp:1598-1626); a weight with a full input transform
 * ({@link InputTransform}) runs its chain PRE_SCALE &rarr; HADAMARD &rarr; ACT_QUANT &rarr; PERMUTE into staging
 * &mdash; PRE_SCALE multiplying each column with {@value #SCALE_COLS}, HADAMARD with
 * {@value HadamardSpec#KERNEL} (or a copy of the plain input when no later part follows otherwise),
 * ACT_QUANT fake-quantizing in place with {@value #ACT_FAKE_QUANT}, and PERMUTE gathering out of place with
 * {@value #PERMUTE_COLS} into a second staging region; a weight with an output scale multiplies the result
 * (llama-graph.cpp:1643-1645).</p>
 *
 * <p><b>Hardware-FP8 fast paths.</b> The FP8 code types {@code F8_E5M2}, {@code F8_E4M3}, the per-element
 * {@code F8_E4M3FN}, {@code F8_E4M3_B128} and
 * {@code MXFP8} also have a {@code {T}_Hw_*} row set over the same argument lists, converting the code pairs with
 * the hardware float8 conversions (CUDA sm_89+, {@code cvt.rn.f16x2.e5m2x2}/{@code e4m3x2}); the results are
 * bit-identical to the generic rows. {@link #hardwareFp8Decode} picks the {@code _Hw} set for a device that
 * reports the feature; a missing feature, and OpenCL, selects the generic rows as the fallback. F8_E4M3FN shares
 * the F8_E4M3 rows, whose 32-element runs need whole-block rows (see {@code GpuWeightUploader}).</p>
 *
 * <p><b>Optional projection kernels.</b> The tiled GEMM, the fast quantized-weight GEMM, the tensor-core MMA GEMM and
 * its split-K twin, the two-row matvec and fused gate/up forms, the {@code _2D} batch and the in-kernel accumulate
 * ({@code {T}_MatVecAdd}) each
 * have a fallback:
 * the batched matvec or, between the GEMMs, the next slower form (fast quant GEMM &rarr; tiled &rarr; batched matvec;
 * MMA &rarr; tiled &rarr; batched matvec; split-K &rarr; its plain MMA kernel; two-row &rarr; single-row form). The
 * batched matvec serves the rest.
 * Such a kernel is prepared once for its work-group size; one the backend does not provide, or whose own work-group
 * limit (OpenCL {@code CL_KERNEL_WORK_GROUP_SIZE}, a register-limited CUDA block) is below that size, is never launched
 * and its fallback serves every product. The fast GEMMs take offsets and strides themselves:</p>
 * <ul>
 *   <li>the fast quantized-weight GEMM ({@code <T>_MatMul_MMQ}, the {@link MmqConfig} row and token tiles) over
 *       the llama.cpp MMQ types (the legacy 32-element blocks, the K-quants and the i-quants iq1_s..iq4_xs): the weight
 *       stays quantized and the activations are quantized once per projection by {@code LLM_Quant_MMQ_Q8_1} (llama.cpp
 *       quantize_mmq_q8_1) into 8-bit codes per chunk, as in llama.cpp ggml-cuda/mmq.cuh (int8 tensor cores on CUDA
 *       sm_80+, else dp4a / scalar integer dots), so results differ from the tiled GEMM by the 8-bit activation
 *       quantization error; a weight with an int8 activation scheme ({@link InputTransform.ActQuant}) stages the
 *       codes with that scheme's own scale instead ({@value GemmKernels#QUANT_MMQ_INT8_STATIC},
 *       {@value GemmKernels#QUANT_MMQ_INT8_ROW} or {@value GemmKernels#QUANT_MMQ_INT8_GROUP128}), so the GEMM
 *       multiplies the checkpoint's int8 activations and the ACT_QUANT fake-quantization pass is skipped. On Vulkan
 *       devices with cooperative matrices the same weight types run {@code <T>_MatMul_CM}
 *       (tile 64 rows x 32 tokens), which dequantizes the weight into fp16 tiles for the tensor cores (llama.cpp
 *       ggml-vulkan mul_mm_cm2.comp) and differs by fp16 rounding instead. {@link MmqMode} (property
 *       {@code neuralfree.gemm.mmq}) turns the lane off for callers that need the exact tiled GEMM.</li>
 *   <li>{@code <T>_MatMul_MMA} (tile 64 rows x 32 tokens) on devices with tensor cores
 *       ({@link DeviceFeature#COOPERATIVE_MATRIX}): F16 weights on CUDA sm_80+ and Vulkan, BF16 weights on CUDA sm_80+
 *       (bf16 {@code mma}; the Vulkan module stages fp16 tiles, so Vulkan keeps the exact tiled GEMM for BF16 until its
 *       native bf16 cooperative-matrix module exists); 16-bit products with an fp32 accumulator, as in llama.cpp
 *       ggml-cuda/mma.cuh. For K of 2048 and more and tile grids that fill less than about two thirds of the compute
 *       units (llama.cpp ggml-vulkan's split-K heuristic), the launch splits into at most 8 k-slices of
 *       {@code <T>_MatMul_MMA_KSplit} and one {@code LLM_SplitK_Reduce}; the MMQ and CM lanes split the same way through
 *       {@code <T>_MatMul_MMQ_KSplit} and {@code <T>_MatMul_CM_KSplit}.</li>
 * </ul>
 *
 * <p><b>Integer-dot decode lane.</b> A single-token projection of an integer-dot capable weight (the legacy Q4_0 /
 * Q4_1 / Q5_0 / Q5_1 / Q8_0, the MXFP4 and NVFP4 4-bit types, and the K-quants and i-quants) may run the Q8_1-activation
 * matvec ({@code {T}_MatVec_Q8}, {@code {T}_MatVecAdd_Q8}) instead of the F32 matvec: the
 * projection input is quantized once to Q8_1 blocks by {@value #QUANTIZE_Q8_1} and every 32-element weight group
 * contributes {@code d8 * sum(code * q8)} times its decoded block scale, the products packed into int8 dot
 * instructions (llama.cpp ggml-cuda/mmvq.cu / vecdotq.cuh and ggml-vulkan mul_mat_vecq.comp). Above the
 * two-row thread floor the single-token lane pairs rows in {@code {T}_MatVec2_Q8} (and the fused gate/up in
 * {@code {T}_MatVec2FusedGateUp_Q8}), reading the Q8_1 row once per two weight rows. A caller whose several
 * single-token projections read one activation row opens a {@link SharedQ8} scope
 * ({@link #shareQ8Activations}), so the row is quantized once for all of them instead of once per projection.
 * {@link Int8MatvecMode}
 * (property {@code neuralfree.matvec.int8}) selects it; the default {@link Int8MatvecMode#AUTO} uses it where the
 * device reports {@link DeviceFeature#INTEGER_DOT_PRODUCT}, the weight type measured faster with the lane on that
 * backend ({@link Int8MatvecTuning}), and the kernel is available, and falls back to the exact F32 matvec elsewhere;
 * a missing kernel falls back the same way. The batched (prefill) forms keep F32 activations; the int8 MMQ GEMM above
 * serves those.</p>
 *
 * <p><b>W12 weights.</b> The lossless type {@link GGMLType#W12} streams its projections through the
 * {@code W12_*} row kernels, which decode every row's plane codes in registers, apply the row's window scale
 * once and add its per-row escape corrections from the record's side tables: up to
 * {@value #W12_MATVEC_MAX_TOKENS} tokens take the matvec rows ({@code W12_MatVec}, {@code W12_MatVecAdd},
 * {@code W12_BatchMatMul[_2D]} and the fused gate/up), and a wider prefill decodes the whole tensor into its BF16
 * mirror ({@code W12_DecodeBF16}) and runs the BF16 family's GEMM lanes on the mirror. While a
 * {@link com.eonmux.neuralfree.gguf.gpu.forward.GpuOps.W12QView} scope is open, those projections read instead the
 * nested 8-bit view of the same code bytes through the {@code W12Q_*} row set (the fast first-level verification
 * pass of speculative decoding).</p>
 *
 * <p><b>Aliasing.</b> Views of one buffer either coincide (same offset, and same strides where strides apply) or are
 * disjoint; operations that allow in-place use say so, all others require disjoint views. Overlaps are rejected with
 * {@link IllegalArgumentException} before anything is enqueued.</p>
 *
 * <p><b>Launch geometry.</b> Row-reduce kernels (matvec families, {@code LLM_RowNorm}, Hadamard) run one work-group of
 * {@code L = LlmKernels.rowReduceLocal()} work-items per output row (per row block for Hadamard): global
 * {@code rows * L}, except the matvec families, whose workgroups hold {@code LlmKernels.matvecLocal()} work-items
 * (the two sizes differ on CUDA, where the matvec lanes measure faster at 64 and the row norms at 256).
 * {@code _2D} batches add the token dimension ({@code {rows * L, nTokens}}, local {@code {L, 1}})
 * and are used only while both group counts stay within the device's work-group-count limits; the one-dimensional
 * batch serves the rest. Element-wise kernels ({@code LLM_Unary}, {@code LLM_GLU}, embedding lookups,
 * {@value #COPY2D}, {@value #AXPBY}) use a 1-D grid of {@code roundUp(n, L)} work-items. The tiled GEMM uses
 * {@code {ceil(M/64)*256, ceil(N/32)}} with local {@code {256, 1}}.</p>
 *
 * <p>Staging buffers come from {@link GpuScratch} and the dummies bound to absent optional arguments from
 * {@link GpuPlaceholders}; the in-order queue orders the reuse of staging buffers. Argument lists of the activation and
 * normalization kernels are built by {@link ActKernels} and {@link NormKernels}, their single source. Not
 * thread-safe.</p>
 */
public final class GpuOps implements AutoCloseable {

    private static final Logger LOGGER = LoggingUtil.getLogger(GpuOps.class);

    /** Offset-aware 2-D copy ({@code kernels/llm/eltwise_ext.cl}). */
    public static final String COPY2D = "LLM_Copy2D_F32";
    /** {@code out = alpha*a + beta*b} ({@code kernels/llm/eltwise_ext.cl}). */
    public static final String AXPBY = "LLM_Axpby_F32";    /**
     * {@code data[s*dim + d] += bias[d]} for {@code s < seq_len} ({@code kernels/llm/eltwise.cl}); arguments
     * {@code (dim, seq_len, bias, data)}, global {@code roundUp(dim, local)}.
     */
    public static final String BATCH_BIAS_ADD = "LLM_Batch_BiasAdd";
    /** Generic activation. */
    public static final String UNARY = ActKernels.UNARY;
    /** Gated linear unit. */
    public static final String GLU = ActKernels.GLU;
    /** Row normalization. */
    public static final String ROW_NORM = NormKernels.ROW_NORM;
    /** Activation fake-quantization of runs of an input transform ({@code kernels/llm/act_quant.cl}). */
    public static final String ACT_FAKE_QUANT = "LLM_ActFakeQuant";
    /** Q8_1 activation quantization of the integer-dot matvec lane ({@code kernels/llm/quantize_q8_1.cl}). */
    public static final String QUANTIZE_Q8_1 = "LLM_QuantizeQ8_1";
    /** Input column gather of a weight stored in permuted input order ({@code kernels/llm/act_quant.cl}). */
    public static final String PERMUTE_COLS = "LLM_PermuteCols";
    /** Per-column activation pre-scale of a quantized weight ({@code kernels/llm/act_quant.cl}). */
    public static final String SCALE_COLS = "LLM_ScaleCols";

    /** Work-group size of the tiled GEMM ({@code NF_GEMM_WG}). */
    static final int TILED_WG = 256;
    /** Weight rows per tiled GEMM work-group ({@code NF_GEMM_TM}). */
    static final int TILED_ROWS = 64;
    /** Tokens per tiled GEMM work-group ({@code NF_GEMM_TN}). */
    static final int TILED_TOKENS = 32;
    /** Largest cooperating-thread count of any matvec family ({@code *_TPS}); the matvec local size is a multiple. */
    static final int MATVEC_MAX_TPS = 32;
    /**
     * Smallest number of live matvec threads for the two-row form: with fewer, halving the workgroups
     * starves the device more than the halved activation re-reads pay back.
     */
    static final int TWO_ROW_MIN_THREADS = 32768;
    /**
     * Largest token count the W12 matvec kernels serve: a wider prefill of a W12 weight decodes the weight into
     * its BF16 mirror and re-enters the projection on the mirror, so the BF16 family's GEMM lanes serve it.
     */
    static final int W12_MATVEC_MAX_TOKENS = 64;
    /** Elements per Q8_1 activation block ({@link BlockLayout#QK8_1}); the integer-dot lane's quantization unit. */
    static final int Q8_1_BLOCK_SIZE = BlockLayout.QK8_1;
    /** Floats of one Q8_1 block ({@link BlockLayout#Q8_1_BYTES}) in the Q8_1 staging slot. */
    static final int Q8_1_FLOATS = BlockLayout.Q8_1_BYTES / Float.BYTES;
    /** The kernels this class launches itself on a 1-D grid of local size {@code L}, apart from the weight families. */
    static final List<String> HELPER_KERNELS =
            Collections.unmodifiableList(Arrays.asList(ROW_NORM, UNARY, GLU, COPY2D, AXPBY, HadamardSpec.KERNEL,
                    ACT_FAKE_QUANT, PERMUTE_COLS, SCALE_COLS, QUANTIZE_Q8_1, NormKernels.RESIDUAL_MIX_NORM,
                    GemmKernels.QUANT_MMQ_INT8_STATIC,
                    GemmKernels.QUANT_MMQ_INT8_ROW, GemmKernels.QUANT_MMQ_INT8_GROUP128,
                    GemmKernels.QUANT_MMQ_INT8_ASYM, GemmKernels.QUANT_MMQ_FP8_STATIC,
                    GemmKernels.QUANT_MMQ_FP8_ROW, GemmKernels.QUANT_MMQ_FP8_GROUP128,
                    GemmKernels.QUANT_MMQ_FP8_MX, GemmKernels.QUANT_MMQ_FP4_MX, GemmKernels.QUANT_MMQ_FP4_NV));

    /** Kernel names of one weight type. */
    static final class WeightKernels {
        final String matvec;
        final String matvecAdd;
        final String batch;
        final String batch2d;
        final String tiled;
        final String embed;
        final String batchEmbed;
        /** Integer-dot (Q8_1 activation) single-token matvec, or null when the type has none. */
        final String matvecQ8;
        /** Integer-dot (Q8_1 activation) accumulating matvec, or null when the type has none. */
        final String matvecAddQ8;
        /** Integer-dot (Q8_1 activation) multi-column matvec of up to {@code NF_MVQN_MAX} token rows,
         * or null when the type has none. */
        final String matvecNQ8;

        WeightKernels(String matvec, String matvecAdd, String batch, String batch2d, String tiled, String embed,
                      String batchEmbed) {
            this(matvec, matvecAdd, batch, batch2d, tiled, embed, batchEmbed, null, null, null);
        }

        WeightKernels(String matvec, String matvecAdd, String batch, String batch2d, String tiled, String embed,
                      String batchEmbed, String matvecQ8, String matvecAddQ8) {
            this(matvec, matvecAdd, batch, batch2d, tiled, embed, batchEmbed, matvecQ8, matvecAddQ8, null);
        }

        WeightKernels(String matvec, String matvecAdd, String batch, String batch2d, String tiled, String embed,
                      String batchEmbed, String matvecQ8, String matvecAddQ8, String matvecNQ8) {
            this.matvec = matvec;
            this.matvecAdd = matvecAdd;
            this.batch = batch;
            this.batch2d = batch2d;
            this.tiled = tiled;
            this.embed = embed;
            this.batchEmbed = batchEmbed;
            this.matvecQ8 = matvecQ8;
            this.matvecAddQ8 = matvecAddQ8;
            this.matvecNQ8 = matvecNQ8;
        }

        static WeightKernels standard(String t) {
            return new WeightKernels(t + "_MatVec", t + "_MatVecAdd", t + "_BatchMatMul", t + "_BatchMatMul_2D",
                    t + "_MatMul_Tiled", t + "_EmbeddingLookup", t + "_BatchEmbeddingLookup");
        }

        /** {@link #standard} plus the {@code {T}_MatVec_Q8} integer-dot forms. */
        static WeightKernels int8(String t) {
            return int8(standard(t));
        }

        /** {@code f} with its {@code _MatVec_Q8}, {@code _MatVecAdd_Q8} and {@code _MatVecN_Q8} integer-dot
         * forms added; every other kernel name is {@code f}'s, so legacy-named families (Q8_0, MXFP4) keep their
         * batch kernels. */
        static WeightKernels int8(WeightKernels f) {
            return new WeightKernels(f.matvec, f.matvecAdd, f.batch, f.batch2d, f.tiled, f.embed, f.batchEmbed,
                    f.matvec + "_Q8", f.matvecAdd + "_Q8", f.matvec + "N_Q8");
        }
    }

    /** Weight types with a Q8_1-activation integer-dot matvec ({@code {T}_MatVec_Q8}). */
    private static final Set<GGMLType> INT8_MATVEC_TYPES = Collections.unmodifiableSet(EnumSet.of(
            GGMLType.Q4_0, GGMLType.Q4_1, GGMLType.Q5_0, GGMLType.Q5_1, GGMLType.Q8_0,
            GGMLType.MXFP4, GGMLType.NVFP4,
            GGMLType.Q2_K, GGMLType.Q3_K, GGMLType.Q4_K, GGMLType.Q5_K, GGMLType.Q6_K,
            GGMLType.Q8_K, GGMLType.IQ4_NL, GGMLType.IQ4_XS, GGMLType.IQ2_XXS, GGMLType.IQ2_XS, GGMLType.IQ2_S,
            GGMLType.IQ3_XXS, GGMLType.IQ3_S, GGMLType.IQ3_K, GGMLType.IQ3_KS, GGMLType.IQ1_S, GGMLType.IQ1_M));

    private static final Map<GGMLType, WeightKernels> FAMILIES = families();

    /** The W12 row set over the nested 8-bit view of the code bytes, serving an open {@link W12QView} scope. */
    private static final WeightKernels W12Q_FAMILY = new WeightKernels("W12Q_MatVec", "W12Q_MatVecAdd",
            "W12Q_BatchMatMul", "W12Q_BatchMatMul_2D", null, null, null);
    /** The whole-tensor BF16 decode behind a W12 weight's wide prefill. */
    private static final String W12_DECODE_BF16 = "W12_DecodeBF16";

    private static Map<GGMLType, WeightKernels> families() {
        Map<GGMLType, WeightKernels> m = new EnumMap<>(GGMLType.class);
        m.put(GGMLType.F32, new WeightKernels("Float_MatVec", "Float_MatVecAdd", "LLM_Batch_Float_MatMul",
                "LLM_Batch_Float_MatMul_2D", "F32_MatMul_Tiled", "LLM_EmbeddingLookup", "LLM_Batch_EmbeddingLookup"));
        m.put(GGMLType.Q8_0, new WeightKernels("Q8_0_MatVec", "Q8_0_MatVecAdd", "LLM_Batch_Q8_0_MatMul",
                "LLM_Batch_Q8_0_MatMul_2D", "Q8_0_MatMul_Tiled", "LLM_Q8_0_EmbeddingLookup",
                "LLM_Batch_Q8_0_EmbeddingLookup"));
        m.put(GGMLType.MXFP4, new WeightKernels("MXFP4_MatVec", "MXFP4_MatVecAdd", "MXFP4_BatchMatMul",
                "MXFP4_BatchMatMul_2D", "MXFP4_MatMul_Tiled", "LLM_MXFP4_EmbeddingLookup",
                "MXFP4_BatchEmbeddingLookup"));
        GGMLType[] standard = {
            GGMLType.F16, GGMLType.BF16,
            GGMLType.Q4_0, GGMLType.Q4_1, GGMLType.Q5_0, GGMLType.Q5_1, GGMLType.IQ4_NL, GGMLType.IQ4_XS,
            GGMLType.Q2_K, GGMLType.Q3_K, GGMLType.Q4_K, GGMLType.Q5_K, GGMLType.Q6_K, GGMLType.Q8_K,
            GGMLType.IQ3_XXS, GGMLType.IQ3_S, GGMLType.IQ2_XXS, GGMLType.IQ2_XS, GGMLType.IQ2_S, GGMLType.IQ1_S,
            GGMLType.IQ1_M,
            GGMLType.IQ3_K, GGMLType.IQ3_KS,
            GGMLType.TQ1_0, GGMLType.TQ2_0, GGMLType.NVFP4, GGMLType.Q1_0, GGMLType.Q2_0,
            GGMLType.F8_E4M3, GGMLType.F8_E5M2, GGMLType.F8_E4M3_B128, GGMLType.F8_E4M3_B32, GGMLType.MXFP8,
            GGMLType.MXFP6_E3M2, GGMLType.MXFP6_E2M3, GGMLType.I8_B128, GGMLType.NF4, GGMLType.FP4_BNB,
        };
        for (GGMLType t : standard) {
            m.put(t, WeightKernels.standard(t.name()));
        }
        // The integer-dot lane: the decode matvec forms that take a Q8_1 activation row, added to the
        // family the type already has (the legacy-named Q8_0 and MXFP4 families keep their batch kernels).
        for (GGMLType t : INT8_MATVEC_TYPES) {
            m.put(t, WeightKernels.int8(m.get(t)));
        }
        // ggml's per-element F8_E4M3FN (id 43) runs the F8_E4M3 rows: its code bytes equal the extension type's
        // blocks on rows that are whole 32-element runs (GpuWeightUploader keeps other rows off the device
        // kernels and decodes them to F32 on upload).
        m.put(GGMLType.F8_E4M3FN, WeightKernels.standard(GGMLType.F8_E4M3.name()));
        // W12: the row kernels decode the lossless 12-bit blocks in registers. A prefill wider than
        // W12_MATVEC_MAX_TOKENS decodes the tensor into its BF16 mirror and runs the BF16 family, so the type
        // has no tiled or lookup kernels of its own.
        m.put(GGMLType.W12, new WeightKernels("W12_MatVec", "W12_MatVecAdd", "W12_BatchMatMul",
                "W12_BatchMatMul_2D", null, null, null));
        return Collections.unmodifiableMap(m);
    }

    /** Weight types with hardware-FP8 decode rows ({@code *_Hw_*} kernels): the E4M3FN/E5M2 code types. */
    private static final Set<GGMLType> HW_FP8_TYPES = Collections.unmodifiableSet(EnumSet.of(
            GGMLType.F8_E4M3, GGMLType.F8_E4M3FN, GGMLType.F8_E4M3_B128, GGMLType.MXFP8, GGMLType.F8_E5M2));

    /** Lowest CUDA compute capability with the hardware FP8 conversions ({@code sm_89}: cvt.rn.f16x2.e4m3x2/e5m2x2). */
    private static final int HW_FP8_COMPUTE_CAPABILITY = 89;

    /**
     * Hardware-FP8 decode is selected on a device that reports the feature: CUDA sm_89 or higher
     * ({@link DeviceCapabilities#getComputeCapability()}), which converts the E4M3FN/E5M2 code pairs in hardware.
     * A missing feature, or a CPU/OpenCL/Vulkan device, selects the generic rows.
     */
    private static boolean hardwareFp8Decode(ComputeBackend.BackendType backendType, DeviceCapabilities caps) {
        if (backendType == null || caps == null) {
            return false;
        }
        return backendType == ComputeBackend.BackendType.CUDA
                && caps.getComputeCapability() >= HW_FP8_COMPUTE_CAPABILITY;
    }

    /**
     * The {@code {T}_Hw_*} fast-path kernels of one weight type (see the class notes); these are the same
     * argument lists and decoding contract as the generic rows, with the FP8 code pairs converted in hardware on
     * CUDA sm_89+, so the results are bit-identical.
     */
    private static WeightKernels hardware(String t) {
        return new WeightKernels(t + "_Hw_MatVec", t + "_Hw_MatVecAdd", t + "_Hw_BatchMatMul",
                t + "_Hw_BatchMatMul_2D", t + "_Hw_MatMul_Tiled", t + "_Hw_EmbeddingLookup",
                t + "_Hw_BatchEmbeddingLookup");
    }

    /** A float view (buffer, element offset). */
    private static final class View {
        final GpuBuffer buf;
        final int off;

        View(GpuBuffer buf, int off) {
            this.buf = buf;
            this.off = off;
        }
    }

    private final ComputeQueue queue;
    private final LlmKernels kernels;
    private final GpuScratch scratch;
    private final ComputeBackend backend;
    private final int local;
    private final int matvecLocal;
    private final long[] maxGroups;
    private final long maxWorkGroupSize;
    private final FastGemmLanes lanes;
    private final Set<PreparedDispatch> prepared = Collections.newSetFromMap(new IdentityHashMap<>());
    private final Map<String, float[]> signValues = new HashMap<>();
    private final Map<String, GpuBuffer> signBuffers = new HashMap<>();
    private final Map<HadamardSpec, GpuBuffer> signsBySpec = new IdentityHashMap<>();
    private final Map<String, Boolean> optionalStatus = new HashMap<>();
    private final GpuPlaceholders placeholders;
    private final Int8MatvecMode int8Mode;
    private final FusedMode fusedMode;
    private SharedQ8 activeShare;
    private W12QView activeW12Q;
    private boolean closed;

    /**
     * Binds the helpers to a queue and prepares the {@link #HELPER_KERNELS} for the work-group size they are launched
     * with ({@link LlmKernels#prewarm(String, long[])}), so a backend that builds one pipeline per work-group size
     * (Vulkan) does it while the model binds rather than during the first forward step.
     *
     * @param queue        queue every operation enqueues on
     * @param kernels      LLM kernels of the queue's backend
     * @param scratch      scratch providing the staging buffers
     * @param placeholders dummies for absent optional arguments, allocated on the queue's backend; not owned
     */
    public GpuOps(ComputeQueue queue, LlmKernels kernels, GpuScratch scratch, GpuPlaceholders placeholders) {
        this(queue, kernels, scratch, placeholders, Int8MatvecMode.configured());
    }

    /**
     * {@link #GpuOps(ComputeQueue, LlmKernels, GpuScratch, GpuPlaceholders)} with an explicit int8-activation
     * matvec mode (the tests pin it; production reads {@link Int8MatvecMode#configured()}).
     */
    public GpuOps(ComputeQueue queue, LlmKernels kernels, GpuScratch scratch, GpuPlaceholders placeholders,
                  Int8MatvecMode int8Mode) {
        this(queue, kernels, scratch, placeholders, int8Mode, FusedMode.configured());
    }

    /**
     * {@link #GpuOps(ComputeQueue, LlmKernels, GpuScratch, GpuPlaceholders, Int8MatvecMode)} with an explicit fused
     * kernel mode (the tests pin it; production reads {@link FusedMode#configured()}).
     */
    GpuOps(ComputeQueue queue, LlmKernels kernels, GpuScratch scratch, GpuPlaceholders placeholders,
           Int8MatvecMode int8Mode, FusedMode fusedMode) {
        this.queue = Objects.requireNonNull(queue, "queue");
        this.kernels = Objects.requireNonNull(kernels, "kernels");
        this.scratch = Objects.requireNonNull(scratch, "scratch");
        this.placeholders = Objects.requireNonNull(placeholders, "placeholders");
        this.int8Mode = Objects.requireNonNull(int8Mode, "int8Mode");
        this.fusedMode = Objects.requireNonNull(fusedMode, "fusedMode");
        this.backend = kernels.backend();
        this.local = kernels.rowReduceLocal();
        this.matvecLocal = kernels.matvecLocal();
        this.maxGroups = kernels.caps().getMaxWorkGroupCount();
        this.maxWorkGroupSize = kernels.caps().getMaxWorkGroupSize();
        if (local < 1 || Integer.bitCount(local) != 1) {
            throw new IllegalStateException("row-reduce work-group size " + local + " is not a power of two");
        }
        if (matvecLocal < 1 || Integer.bitCount(matvecLocal) != 1 || matvecLocal > local) {
            throw new IllegalStateException("matvec work-group size " + matvecLocal
                    + " is not a power of two of at most " + local);
        }
        this.lanes = new FastGemmLanes(this, queue, kernels, scratch);
        prewarmHelperKernels();
    }

    /** Prepares each of the {@link #HELPER_KERNELS} for local size {@code L} ({@link #prewarm(String, long[])}). */
    private void prewarmHelperKernels() {
        long[] size = {local};
        for (String kernel : HELPER_KERNELS) {
            prewarm(kernel, size);
        }
        prewarm(QUANTIZE_Q8_1, new long[]{matvecLocal});
    }

    /**
     * Prepares a kernel for launches at work-group size {@code local} ({@link LlmKernels#prewarm(String, long[])}),
     * so a backend that builds one pipeline per work-group size (Vulkan) builds it now. The owner of a kernel it
     * launches through {@link #enqueue} calls this while the model binds, once for each work-group size it launches
     * the kernel with, so that no forward step builds a pipeline. Preparing a size again is cheap.
     *
     * <p>A kernel the backend does not provide, cannot run at {@code local}, or fails to build a pipeline for is not
     * prepared. Its first launch then reports that failure as it would without this call, so a model that never
     * launches it is unaffected.</p>
     *
     * @param kernel kernel name
     * @param local  work-group size of the kernel's launches: one to three positive dimensions
     * @return true when the kernel is prepared for {@code local}
     * @throws IllegalArgumentException when the name is empty or {@code local} is malformed
     */
    public boolean prewarm(String kernel, long[] local) {
        ensureOpen();
        checkLocal(kernel, local);
        try {
            if (!kernels.has(kernel)) {
                LOGGER.debug("Kernel {} not prepared: the {} backend does not provide it", kernel,
                        backend.backendType());
                return false;
            }
            kernels.prewarm(kernel, local);
            return true;
        } catch (UnsupportedKernelException | IllegalArgumentException | GpuBackendException
                 | GpuOutOfMemoryException e) {
            LOGGER.debug("Kernel {} not prepared for work-group size {}: {}", kernel, Arrays.toString(local),
                    e.getMessage());
            return false;
        }
    }

    public ComputeQueue queue() { return queue; }
    public LlmKernels kernels() { return kernels; }
    public GpuScratch scratch() { return scratch; }
    /** Dummies bound to absent optional kernel arguments, as no kernel receives a null buffer. */
    public GpuPlaceholders placeholders() { return placeholders; }
    /** Work-group size {@code L} of the row-reduce and element-wise launches. */
    public int local() { return local; }
    /** Work-group size {@code L} of the matvec family launches. */
    public int matvecLocal() { return matvecLocal; }

    /** The configured fused-kernel selection. */
    public FusedMode fusedMode() { return fusedMode; }

    /**
     * Whether a fused kernel runs on this backend: the mode permits fusion and the kernel library provides it. In
     * {@link FusedMode#ON} a kernel the backend lacks is refused by name instead of falling back.
     *
     * @throws com.eonmux.neuralfree.acceleration.UnsupportedKernelException when the mode requires the kernel and
     *                                                                    the backend does not provide it
     */
    public boolean fused(String kernel) {
        ensureOpen();
        Objects.requireNonNull(kernel, "kernel");
        if (!fusedMode.permitsFusion()) {
            return false;
        }
        if (kernels.has(kernel)) {
            return true;
        }
        if (fusedMode.requiresFusion()) {
            throw new com.eonmux.neuralfree.acceleration.UnsupportedKernelException(kernel
                    + " is required by -D" + FusedMode.PROPERTY + "=on but the "
                    + backend.backendType() + " backend does not provide it");
        }
        return false;
    }

    /** True when GpuOps has projection and lookup kernels for weights of this type. */
    public static boolean supportsWeightType(GGMLType type) {
        return FAMILIES.containsKey(type);
    }

    /** The {@code {T}_MatVec} kernel name of a weight type ({@code Float_MatVec} for F32); null when the type has
     *  no GPU kernels. The name is the family's, so the types that share a family share the kernel rows. */
    public static String matVecKernel(GGMLType type) {
        WeightKernels k = FAMILIES.get(type);
        return k == null ? null : k.matvec;
    }

    /** The single-token embedding-lookup kernel name of a weight type; null when the type has no GPU kernels. */
    public static String lookupKernel(GGMLType type) {
        WeightKernels k = FAMILIES.get(type);
        return k == null ? null : k.embed;
    }

    // ================================================================== generic dispatch

    /**
     * Enqueues a kernel with all arguments. A kernel not prepared for {@code local} while the model bound
     * ({@link #prewarm(String, long[])}) has its pipeline built by this launch on a backend that builds one per
     * work-group size (Vulkan).
     *
     * @throws com.eonmux.neuralfree.acceleration.UnsupportedKernelException when the kernel is unavailable
     */
    public void enqueue(String kernel, long[] global, long[] local, Object... args) {
        ensureOpen();
        checkSizes(kernel, global, local);
        Objects.requireNonNull(args, "args");
        queue.enqueueKernelNoEvent(kernels.get(kernel), global.clone(), local.clone(), args.clone());
    }

    /**
     * A dispatch with sticky arguments bound once (see {@link PreparedDispatch}). The kernel is prepared for
     * {@code local} now ({@link LlmKernels#prewarm(String, long[])}), so a backend that builds one pipeline per
     * work-group size (Vulkan) does it while the model binds rather than at the first dispatch.
     *
     * @throws com.eonmux.neuralfree.acceleration.UnsupportedKernelException when the kernel is unavailable, or
     *                                                                       cannot run work-groups of this size
     */
    public PreparedDispatch prepare(String kernel, long[] global, long[] local, Object... args) {
        ensureOpen();
        checkSizes(kernel, global, local);
        Objects.requireNonNull(args, "args");
        for (int i = 0; i < args.length; i++) {
            if (args[i] == null) {
                throw new IllegalArgumentException(kernel + " argument " + i + " is null");
            }
        }
        kernels.prewarm(kernel, local);
        KernelHandle shared = kernels.get(kernel);
        KernelHandle clone = cloneOrNull(shared);
        PreparedDispatch d = null;
        if (clone != null) {
            try {
                d = new PreparedDispatch(this, kernel, clone, true, queue, global.clone(), local.clone(), args.clone());
            } catch (UnsupportedOperationException e) {
                clone.close();
            } catch (RuntimeException e) {
                clone.close();
                throw e;
            }
        }
        if (d == null) {
            d = new PreparedDispatch(this, kernel, shared, false, queue, global.clone(), local.clone(), args.clone());
        }
        prepared.add(d);
        return d;
    }

    /** A private clone of a library handle, or null when the backend cannot clone handles. */
    private KernelHandle cloneOrNull(KernelHandle shared) {
        try {
            return backend.cloneKernel(shared);
        } catch (UnsupportedOperationException e) {
            return null;
        }
    }

    void forget(PreparedDispatch d) {
        prepared.remove(d);
    }

    // ================================================================== projections

    /**
     * {@code out[outOff + t*rows + r] = scale * W[r] . in[inOff + t*cols ...]} for {@code t < nTokens} with the
     * batched matvec kernels. Every projection hands its transformed input to {@link GpuMatmulTap} when a
     * collection installed one.
     */
    public void matvec(GpuWeight w, GpuBuffer in, int inOff, GpuBuffer out, int outOff, int nTokens) {
        project(w, in, inOff, out, outOff, nTokens, false, true);
    }

    /**
     * {@link #matvec}, using the fast GEMM lane from {@code FAST_GEMM_MIN_TOKENS} rows or the tiled GEMM from
     * {@code TILED_MIN_TOKENS} rows when the kernel can run its work-group on this device. With batch invariance
     * on, a fast GEMM lane serves every token count and the tiled GEMM is never taken.
     */
    public void matmul(GpuWeight w, GpuBuffer in, int inOff, GpuBuffer out, int outOff, int nTokens) {
        project(w, in, inOff, out, outOff, nTokens, true, true);
    }

    /**
     * Whether the single-token fused gate and up projection with the activation ({@code {T}_MatVecFusedGateUp})
     * can serve {@code gate} and {@code up}: the weights of one type and shape whose rows the kernels decode
     * together, a SWIGLU without parity-changing parameters, and no weight scale or input transform. The fused
     * kernel writes {@code silu(gate . x) * (up . x)} directly, replacing two matvecs and one GLU launch, as
     * llama.cpp's mul_mat + mul_mat + GLU fusion does. The fused selection mode applies: {@code off} refuses the
     * form and {@code on} requires it.
     */
    public boolean gateUpGluUsable(GpuWeight gate, GpuWeight up, GluOp op, ActParams params) {
        ensureOpen();
        return fusedGateUpKernel(gate, up, op, params) != null;
    }

    /**
     * Runs {@link #gateUpGluUsable}'s fused projection: {@code out[r] = silu(gate[r] . x) * (up[r] . x)} over the
     * single input row {@code x} (at element 0 of {@code in}), written from element 0 of {@code out}. Above the
     * two-row thread floor ({@link #TWO_ROW_MIN_THREADS} live matvec threads) the launch pairs the rows of both
     * weights in the {@code {T}_MatVec2FusedGateUp} form at {@code global = ceil(rows / 2) * L}; below it the
     * kernel runs one row per workgroup.
     *
     * @throws UnsupportedOperationException when the form cannot serve the weights, op or buffers
     */
    public void gateUpGlu(GpuWeight gate, GpuWeight up, GluOp op, ActParams params, GpuBuffer in, GpuBuffer out) {
        ensureOpen();
        String kernel = fusedGateUpKernel(gate, up, op, params);
        if (kernel == null) {
            throw new UnsupportedOperationException("no fused gate+up kernel serves " + gate + " and " + up
                    + " with " + op + " on the " + backend.backendType() + " backend");
        }
        int cols = cols(gate);
        int rows = rows(gate);
        checkView("fused gate+up input", in, 0, cols);
        checkView("fused gate+up output", out, 0, rows);
        if (overlaps(in, 0, cols, out, 0, rows)) {
            throw new IllegalArgumentException("fused gate+up: input and output views overlap");
        }
        requireMatvecLocal(gate);
        GpuMatmulTap.dense(queue, gate, in, 0, 1, cols);
        GpuMatmulTap.dense(queue, up, in, 0, 1, cols);
        if (kernel.endsWith("_Q8")) {
            GpuBuffer xq = quantizedRow(new View(in, 0), cols);
            launchInt8GateUp(kernel, gate.type(), rows, cols, gate.shard(0), up.shard(0), xq, out);
        } else if (twoRowUsable(kernel, rows)) {
            launchMatvec(twoRowName(kernel), (long) ceilDiv(rows, 2) * matvecLocal, rows, cols,
                    gate.shard(0), up.shard(0), in, out);
        } else {
            launchMatvec(kernel, (long) rows * matvecLocal, rows, cols, gate.shard(0), up.shard(0), in, out);
        }
    }

    /**
     * The fused gate+up kernel for these weights, or null. Structural objections (shapes, types, scale, transform,
     * op) are not a fusion the caller asked for, so they return null even in {@link FusedMode#ON}; a structurally
     * eligible layer on a backend without the kernel is refused there.
     */
    private String fusedGateUpKernel(GpuWeight gate, GpuWeight up, GluOp op, ActParams params) {
        if (BatchInvariance.enabled()) {
            /* A fused form runs only the single-token path; batch-invariant steps take the separate products at
             * every row count. Requiring the fusion and requiring batch invariance cannot both hold. */
            if (fusedMode.requiresFusion()) {
                throw new IllegalArgumentException("-D" + FusedMode.PROPERTY + "=on requires the fused"
                        + " gate+up kernels, which -D" + BatchInvariance.PROPERTY + " turns off");
            }
            return null;
        }
        if (gate == null || up == null || op != GluOp.SWIGLU
                || (params != null && params != ActParams.NONE)) {
            return null;
        }
        if (gate.type() != up.type() || !gate.isSingleShard() || !up.isSingleShard()
                || gate.rows() != up.rows() || gate.cols() != up.cols()) {
            return null;
        }
        if (gate.hasScale() || up.hasScale() || gate.hasInputTransform() || up.hasInputTransform()
                || gate.hadamard() != null || up.hadamard() != null) {
            return null;
        }
        if (gate.type() == GGMLType.W12 && activeW12Q != null) {
            /* A W12QView scope routes every W12 projection through the 8-bit Q view, whose row set has no
             * fused form; the separate gate and up matvecs take it. */
            return null;
        }
        if (!fusedMode.permitsFusion()) {
            return null;
        }
        WeightKernels k = family(gate);
        String fused = fusedGateUpName(k.matvec);
        String q8 = k.matvecQ8 != null ? fusedGateUpName(k.matvecQ8) : null;
        if (q8 != null && cols(gate) % gate.type().getBlockSize() == 0
                && int8LaneSelected(gate.type(), (long) gate.rows() * cols(gate))
                && optionalReady(q8, matvecLocal)) {
            return q8;
        }
        if (optionalReady(fused, matvecLocal)) {
            return fused;
        }
        if (fusedMode.requiresFusion()) {
            throw new UnsupportedKernelException(fused
                    + " is required by -D" + FusedMode.PROPERTY + "=on but the " + backend.backendType()
                    + " backend does not provide it");
        }
        return null;
    }

    /** {@code {T}_MatVec} to {@code {T}_MatVecFusedGateUp} and {@code {T}_MatVec_Q8} to its fused form. */
    private static String fusedGateUpName(String matvec) {
        return matvec.replace("MatVec", "MatVecFusedGateUp");
    }

    /** {@code accum[accOff + t*rows + r] += scale * W[r] . in[inOff + t*cols ...]}. */
    public void matvecAdd(GpuWeight w, GpuBuffer in, int inOff, GpuBuffer accum, int accOff, int nTokens) {
        ensureOpen();
        int cols = cols(w);
        int rows = rows(w);
        checkTokens(nTokens);
        long nIn = (long) nTokens * cols;
        long nOut = (long) nTokens * rows;
        checkView("input of " + w.name(), in, inOff, nIn);
        checkView("accumulator of " + w.name(), accum, accOff, nOut);
        WeightKernels k = projectionFamily(w);
        boolean invariant = BatchInvariance.enabled();
        boolean q8Add = !invariant && nTokens == 1 && int8LaneUsable(k.matvecAddQ8, w);
        if (!invariant && nTokens == 1 && w.isSingleShard() && accOff == 0 && !w.hasScale()
                && !overlaps(in, inOff, nIn, accum, accOff, nOut)
                && (q8Add || optionalReady(k.matvecAdd, matvecLocal))) {
            requireMatvecLocal(w);
            View x = offsetFree(projectionInput(w, in, inOff, nTokens, cols, false), nTokens, cols);
            GpuMatmulTap.dense(queue, w, x.buf, x.off, nTokens, cols);
            if (q8Add) {
                GpuBuffer xq = quantizedRow(x, cols);
                launchMatvec(k.matvecAddQ8, (long) rows * matvecLocal, rows, cols, w.shard(0), xq, accum);
            } else {
                launchMatvec(k.matvecAdd, (long) rows * matvecLocal, rows, cols, w.shard(0), x.buf, accum);
            }
            return;
        }
        ScratchSlot result = scratch.staging(GpuScratch.Staging.RESULT, nOut);
        project(w, in, inOff, result.buffer(), result.offset(), nTokens, true, false);
        axpby((int) nOut, w.scale(), result.buffer(), result.offset(), 1.0f, accum, accOff, accum, accOff);
    }

    private void project(GpuWeight w, GpuBuffer in, int inOff, GpuBuffer out, int outOff, int nTokens,
                         boolean allowTiled, boolean applyScale) {
        ensureOpen();
        int cols = cols(w);
        int rows = rows(w);
        checkTokens(nTokens);
        long nIn = (long) nTokens * cols;
        long nOut = (long) nTokens * rows;
        checkView("input of " + w.name(), in, inOff, nIn);
        checkView("output of " + w.name(), out, outOff, nOut);
        checkInt(outOff + nOut, "output end of " + w.name());
        if (overlaps(in, inOff, nIn, out, outOff, nOut)) {
            throw new IllegalArgumentException("projection " + w.name() + ": input and output views overlap");
        }
        if (w.type() == GGMLType.W12) {
            projectW12(w, in, inOff, out, outOff, nTokens, allowTiled, applyScale);
            return;
        }
        WeightKernels k = family(w);
        /* Batch invariance pins the projection lane independently of the row count. A weight type with a fast
         * GEMM lane takes it for one row and for a full batch alike (its tiles cover the reduction dimension in a
         * fixed order); the split-K twins, which reduce a row in several slices, are never taken there. A type
         * without one keeps the float matvec family for every row count, because the exact tiled GEMM's reduction
         * of an output element does depend on the token tile. */
        boolean invariant = BatchInvariance.enabled();
        /* The multi-column integer-dot lane serves small token groups with one weight read per row for
         * the whole group, which beats the GEMM lanes and the tiled GEMM there; its per-row integer dots
         * outgrow a fast GEMM lane whose cost is flat in the row count, so a weight with a fast lane
         * crosses over at FAST_GEMM_MIN_TOKENS and one without keeps the lane up to
         * MATVEC_N_DISPATCH_MAX (the tiled GEMM's own crossover). The lane is pinned off under batch
         * invariance exactly like the single-token lane. */
        String lane = allowTiled ? lanes.fastGemm(w, cols) : null;
        int matvecMax = lane != null ? LlmKernelConstants.FAST_GEMM_MIN_TOKENS - 1
                : LlmKernelConstants.MATVEC_N_DISPATCH_MAX;
        boolean q8n = !invariant && nTokens >= 2 && nTokens <= matvecMax
                && int8LaneUsable(k.matvecNQ8, w);
        String fast = lane != null && !q8n && (invariant || nTokens >= LlmKernelConstants.FAST_GEMM_MIN_TOKENS)
                ? lane : null;
        int fastTileTokens = fast == null ? 0 : lanes.fastTileTokens(fast);
        int fastTileRows = fast == null ? 0 : lanes.fastTileRows(fast);
        int fastWorkGroup = fast == null ? 0 : lanes.fastWorkGroup(fast);
        boolean fastUsable = fast != null
                && lanes.fastUsable(fast, fastTileRows, fastTileTokens, fastWorkGroup, w, nTokens);
        /* A staged launch of fewer tokens than the device tile runs the 16-token-tile twin of the lane
         * (smallFast) when the device has it, so the weight stream fills 16-token output tiles; the
         * split-K twins keep the plain kernels. */
        String smallFast = fastUsable ? lanes.smallTileKernel(fast, w, nTokens) : null;
        int launchTileTokens = smallFast == null ? fastTileTokens : lanes.fastTileTokens(smallFast);
        Int8ActStaging int8Staging = fastUsable && fast.endsWith(GemmKernels.MATMUL_MMQ_SUFFIX)
                ? lanes.int8Staging(w, cols) : null;
        Fp8ActStaging fp8Staging = fastUsable && FastGemmLanes.fp8Fast(w, fast)
                ? lanes.fp8Staging(w, cols) : null;
        View x = projectionInput(w, in, inOff, nTokens, cols, int8Staging != null || fp8Staging != null);
        GpuMatmulTap.dense(queue, w, x.buf, x.off, nTokens, cols);
        if (fastUsable) {
            boolean staged = FastGemmLanes.stagedFast(fast);
            View activations = x;
            if (staged) {
                ScratchSlot q8;
                if (fp8Staging != null) {
                    q8 = lanes.stageFp8(fp8Staging, x.buf, x.off, nTokens, cols);
                } else if (int8Staging != null) {
                    q8 = lanes.stageInt8(int8Staging, x.buf, x.off, nTokens, cols);
                } else {
                    q8 = lanes.stageQ8(x.buf, x.off, nTokens, cols);
                }
                activations = new View(q8.buffer(), q8.offset());
            }
            for (int s = 0; s < w.shardCount(); s++) {
                int m = (int) w.shardRows(s);
                int yOff = outOff + (int) w.shardRowStart(s);
                SplitKGemmPlan split = invariant ? null
                        : lanes.splitK(fast, w, m, cols, nTokens, yOff, activations.off, rows,
                                smallFast != null);
                if (split != null) {
                    lanes.splitKLaunch(split, w.shard(s), activations.buf, out, yOff, rows);
                } else {
                    int qStride = staged ? lanes.stagedRowRecords(cols) : cols;
                    int qOff = staged ? activations.off : x.off;
                    launch2d(smallFast == null ? fast : smallFast,
                            ceilDiv(m, fastTileRows) * fastWorkGroup, ceilDiv(nTokens, launchTileTokens),
                            fastWorkGroup, m, cols, nTokens, qStride, qOff, rows, yOff, w.shard(s),
                            activations.buf, out);
                }
            }
        } else if (!invariant && allowTiled && !q8n && nTokens >= LlmKernelConstants.TILED_MIN_TOKENS
                && tiledUsable(k, w, nTokens)) {
            for (int s = 0; s < w.shardCount(); s++) {
                int m = (int) w.shardRows(s);
                int yOff = outOff + (int) w.shardRowStart(s);
                launch2d(k.tiled, ceilDiv(m, TILED_ROWS) * TILED_WG, ceilDiv(nTokens, TILED_TOKENS), TILED_WG,
                        m, cols, nTokens, cols, x.off, rows, yOff, w.shard(s), x.buf, out);
            }
        } else {
            requireMatvecLocal(w);
            x = offsetFree(x, nTokens, cols);
            boolean direct = w.isSingleShard() && outOff == 0;
            boolean q8 = !invariant && nTokens == 1 && int8LaneUsable(k.matvecQ8, w);
            GpuBuffer xq = q8 ? quantizedRow(x, cols) : q8n ? quantizeQ8N(x.buf, nTokens, cols) : null;
            for (int s = 0; s < w.shardCount(); s++) {
                int m = (int) w.shardRows(s);
                if (direct) {
                    if (q8) {
                        launchInt8Matvec(k.matvecQ8, w.type(), m, cols, w.shard(s), xq, out);
                    } else if (q8n) {
                        launchInt8MatvecN(k.matvecNQ8, m, cols, nTokens, w.shard(s), xq, out);
                    } else {
                        matvecShard(k, w.shard(s), m, cols, x.buf, out, nTokens);
                    }
                } else {
                    ScratchSlot stage = scratch.staging(GpuScratch.Staging.OUTPUT, (long) nTokens * m);
                    if (q8) {
                        launchInt8Matvec(k.matvecQ8, w.type(), m, cols, w.shard(s), xq, stage.buffer());
                    } else if (q8n) {
                        launchInt8MatvecN(k.matvecNQ8, m, cols, nTokens, w.shard(s), xq, stage.buffer());
                    } else {
                        matvecShard(k, w.shard(s), m, cols, x.buf, stage.buffer(), nTokens);
                    }
                    copy2dUnchecked(nTokens, m, stage.buffer(), stage.offset(), m,
                            out, outOff + (int) w.shardRowStart(s), rows);
                }
            }
        }
        if (applyScale && w.hasScale()) {
            axpby((int) nOut, 0.0f, null, 0, w.scale(), out, outOff, out, outOff);
        }
    }

    /**
     * Whether a projection of {@code w} with {@code nTokens} token rows runs on the float matvec family
     * ({@code project}'s fallback lane, launched at {@link #matvecLocal()}): no fast GEMM lane, no tiled
     * GEMM lane and no int8 matvec lane takes the projection, and the family's work-group size is legal
     * for the weight. The certified head's rescore (see
     * {@link com.eonmux.neuralfree.gguf.gpu.head.GpuCertArgmax}) repeats the family's arithmetic at
     * {@link #matvecLocal()}, so a token count another lane serves keeps the exact readout.
     */
    public boolean matvecFamilyServes(GpuWeight w, int nTokens) {
        ensureOpen();
        Objects.requireNonNull(w, "weight");
        checkTokens(nTokens);
        int cols = cols(w);
        WeightKernels k = family(w);
        boolean invariant = BatchInvariance.enabled();
        String lane = lanes.fastGemm(w, cols);
        int matvecMax = lane != null ? LlmKernelConstants.FAST_GEMM_MIN_TOKENS - 1
                : LlmKernelConstants.MATVEC_N_DISPATCH_MAX;
        if (!invariant && nTokens >= 2 && nTokens <= matvecMax
                && int8LaneUsable(k.matvecNQ8, w)) {
            return false;
        }
        String fast = invariant || nTokens >= LlmKernelConstants.FAST_GEMM_MIN_TOKENS ? lane : null;
        if (fast != null && lanes.fastUsable(fast, lanes.fastTileRows(fast), lanes.fastTileTokens(fast),
                lanes.fastWorkGroup(fast), w, nTokens)) {
            return false;
        }
        if (!invariant && nTokens >= LlmKernelConstants.TILED_MIN_TOKENS && tiledUsable(k, w, nTokens)) {
            return false;
        }
        if (k.matvecQ8 != null && !invariant && nTokens == 1 && int8LaneUsable(k.matvecQ8, w)) {
            return false;
        }
        return matvecLocal % MATVEC_MAX_TPS == 0;
    }

    /**
     * The projection input: the caller's view; or the weight's PRISM rotation in staging; or the full
     * {@link InputTransform} chain in staging &mdash; the PRE_SCALE part multiplying each column into the input
     * staging, the HADAMARD part rotating it into the other staging region (a copy of the input when there is no
     * rotation), ACT_QUANT fake-quantizing in place and PERMUTE gathering out of place into whichever region the
     * preceding step did not read from.
     *
     * <p>{@code actQuantStaged} says the int8 MMQ lane stages the activation codes itself
     * ({@link FastGemmLanes#stageInt8}), so the ACT_QUANT part is left to that pass: the input then keeps its
     * float values (scaled, rotated and copied, but not fake-quantized) and the scheme is applied once, as int8
     * codes.</p>
     *
     * @throws IllegalArgumentException when the weight carries a per-expert activation pre-scale (the dense
     *                                  projection has no expert index) or lacks its device pre-scale buffer
     */
    private View projectionInput(GpuWeight w, GpuBuffer in, int inOff, int nTokens, int cols,
                                 boolean actQuantStaged) {
        InputTransform t = w.inputTransform();
        InputTransform.ActQuant a = t != null ? t.actQuant() : null;
        HadamardSpec h = w.hadamard();
        float[] pre = t != null ? t.preScale() : null;
        if (pre != null && pre.length != cols) {
            throw new IllegalArgumentException("projection " + w.name() + ": an activation pre-scale of "
                    + pre.length + " values for " + cols + " columns is per expert, which the dense projection"
                    + " does not apply");
        }
        boolean quantizeHere = a != null && !actQuantStaged;
        boolean needsStaging = pre != null || h != null || quantizeHere || (t != null && t.hasPerm());
        if (!needsStaging) {
            return new View(in, inOff);
        }
        ScratchSlot first = scratch.staging(GpuScratch.Staging.INPUT, (long) nTokens * cols);
        GpuBuffer xb = first.buffer();
        int xo = first.offset();
        if (pre != null) {
            if (w.inputPreScale() == null) {
                throw new IllegalArgumentException("projection " + w.name() + ": its activation pre-scale has no"
                        + " device buffer");
            }
            scaleCols(w.inputPreScale(), in, inOff, xb, xo, nTokens, cols);
        } else if (h != null) {
            hadamard(h, in, inOff, xb, xo, nTokens, cols);
        } else {
            copy2dUnchecked(nTokens, cols, in, inOff, cols, xb, xo, cols);
        }
        if (pre != null && h != null) {
            ScratchSlot rotated = scratch.staging(GpuScratch.Staging.PERMUTED, (long) nTokens * cols);
            hadamard(h, xb, xo, rotated.buffer(), rotated.offset(), nTokens, cols);
            xb = rotated.buffer();
            xo = rotated.offset();
        }
        if (quantizeHere) {
            actFakeQuant(a.scheme(), a.scale(0), a.zeroPoint(0), a.globalScale(0), xb, xo, xb, xo, nTokens, cols);
        }
        if (t != null && t.hasPerm()) {
            // The gather must not overlap its source: PERMUTE_COLS reads out of place. The rotated result sits in
            // the permuted region, so it gathers back into the input region the rotation consumed.
            ScratchSlot p = xb == first.buffer() && xo == first.offset()
                    ? scratch.staging(GpuScratch.Staging.PERMUTED, (long) nTokens * cols)
                    : first;
            permuteCols(w.inputPerm(), xb, xo, p.buffer(), p.offset(), nTokens, cols);
            return new View(p.buffer(), p.offset());
        }
        return new View(xb, xo);
    }

    /** The input at element 0 of its buffer, copying it to staging when it starts elsewhere. */
    private View offsetFree(View x, int nTokens, int cols) {
        if (x.off == 0) {
            return x;
        }
        ScratchSlot s = scratch.staging(GpuScratch.Staging.INPUT, (long) nTokens * cols);
        copy2dUnchecked(nTokens, cols, x.buf, x.off, cols, s.buffer(), s.offset(), cols);
        return new View(s.buffer(), s.offset());
    }

    /**
     * The float matvec lanes of one shard. The single-token path runs {@code {T}_MatVec} (or its two-rows-per-
     * workgroup form) and the multi-token path {@code {T}_BatchMatMul_2D}; both reduce every output row over the
     * weight blocks in the same order, so a row's value does not depend on the token count. Batch invariance skips
     * the two-row form: it is a single-token-only lane, and the batch kernel must serve every row count instead.
     */
    private void matvecShard(WeightKernels k, GpuBuffer shard, int m, int cols, GpuBuffer x, GpuBuffer dst,
                             int nTokens) {
        long global = (long) m * matvecLocal;
        boolean invariant = BatchInvariance.enabled();
        if (!invariant && nTokens == 1 && twoRowUsable(k.matvec, m)) {
            launchMatvec(twoRowName(k.matvec), (long) ceilDiv(m, 2) * matvecLocal, m, cols, shard, x, dst);
        } else if (nTokens == 1) {
            launchMatvec(k.matvec, global, m, cols, shard, x, dst);
        } else if (m <= maxGroups[0] && nTokens <= maxGroups[1] && optionalReady(k.batch2d, matvecLocal, 1)) {
            launch2d(k.batch2d, global, nTokens, matvecLocal, m, cols, nTokens, shard, x, dst);
        } else {
            launchMatvec(k.batch, global, m, cols, nTokens, shard, x, dst);
        }
    }

    /** {@code {T}_MatVec} to {@code {T}_MatVec2}, the two-rows-per-workgroup form. */
    private static String twoRowName(String matvec) {
        return matvec.replace("MatVec", "MatVec2");
    }

    /**
     * The W12 projection: up to {@value #W12_MATVEC_MAX_TOKENS} tokens run the {@code W12_*} row kernels, which
     * decode every row's plane codes in registers and apply its window scale and escape corrections; a wider
     * prefill decodes the weight into its BF16 mirror ({@link #w12Bf16Mirror}) and re-enters {@link #project} on
     * the mirror, so the BF16 family's fast GEMM lanes and tiled GEMM serve it. An open {@link W12QView} scope
     * routes the row kernels over the nested 8-bit view of the code bytes ({@code W12Q_*}); the fused gate/up
     * form has no Q-view twin, so it is not taken while a scope is open ({@link #fusedGateUpKernel}).
     */
    private void projectW12(GpuWeight w, GpuBuffer in, int inOff, GpuBuffer out, int outOff, int nTokens,
                            boolean allowTiled, boolean applyScale) {
        long nOut = (long) nTokens * rows(w);
        if (nTokens > W12_MATVEC_MAX_TOKENS) {
            GpuWeight mirror = w12Bf16Mirror(w);
            project(mirror, in, inOff, out, outOff, nTokens, allowTiled, false);
            if (applyScale && w.hasScale()) {
                axpby((int) nOut, 0.0f, null, 0, w.scale(), out, outOff, out, outOff);
            }
            return;
        }
        int cols = cols(w);
        requireMatvecLocal(w);
        WeightKernels k = projectionFamily(w);
        View x = projectionInput(w, in, inOff, nTokens, cols, false);
        GpuMatmulTap.dense(queue, w, x.buf, x.off, nTokens, cols);
        x = offsetFree(x, nTokens, cols);
        boolean direct = w.isSingleShard() && outOff == 0;
        for (int s = 0; s < w.shardCount(); s++) {
            int m = (int) w.shardRows(s);
            if (direct) {
                matvecShard(k, w.shard(s), m, cols, x.buf, out, nTokens);
            } else {
                ScratchSlot stage = scratch.staging(GpuScratch.Staging.OUTPUT, (long) nTokens * m);
                matvecShard(k, w.shard(s), m, cols, x.buf, stage.buffer(), nTokens);
                copy2dUnchecked(nTokens, m, stage.buffer(), stage.offset(), m,
                        out, outOff + (int) w.shardRowStart(s), rows(w));
            }
        }
        if (applyScale && w.hasScale()) {
            axpby((int) nOut, 0.0f, null, 0, w.scale(), out, outOff, out, outOff);
        }
    }

    /**
     * Decodes every row of the W12 weight into its BF16 mirror in the W12 staging region (one
     * {@value #W12_DECODE_BF16} launch per row shard, each shard placed at its row start) and returns the mirror
     * as a BF16 weight of the same shape. The mirror lives in staging, so it is valid for this projection only;
     * the in-order queue reads it before the next projection reuses the region.
     */
    private GpuWeight w12Bf16Mirror(GpuWeight w) {
        int cols = cols(w);
        long elements = Math.multiplyExact(w.rows(), cols);
        ScratchSlot mirror = scratch.staging(GpuScratch.Staging.W12_BF16, elements / 2);
        for (int s = 0; s < w.shardCount(); s++) {
            launchMatvec(W12_DECODE_BF16, w.shardRows(s) * matvecLocal, (int) w.shardRows(s), cols,
                    (int) w.shardRowStart(s), w.shard(s), mirror.buffer());
        }
        return GpuWeight.of(w.name(), GGMLType.BF16, w.ne(), mirror.buffer());
    }

    /** {@link #family} for the weight, or the W12 Q-view row set while a {@link W12QView} scope is open. */
    private WeightKernels projectionFamily(GpuWeight w) {
        WeightKernels k = family(w);
        if (w.type() == GGMLType.W12 && activeW12Q != null) {
            k = W12Q_FAMILY;
        }
        return k;
    }

    /**
     * Launches the integer-dot matvec {@code kernel} over {@code m} rows of {@code weight} against the Q8_1
     * row {@code xq}: the paired {@code {T}_MatVec2_Q8} form at {@code ceil(m / 2) * L} when
     * {@link #twoRowUsable} selects it and the type measured faster paired on this backend
     * ({@link Int8MatvecTuning#pairedIsFaster}), so the activation row is read once per two weight rows,
     * otherwise the single-row form at {@code m * L}.
     */
    private void launchInt8Matvec(String kernel, GGMLType type, int m, int cols, GpuBuffer weight, GpuBuffer q8,
                                  GpuBuffer out) {
        if (pairInt8Rows(type) && twoRowUsable(kernel, m)) {
            launchMatvec(twoRowName(kernel), (long) ceilDiv(m, 2) * matvecLocal, m, cols, weight, q8, out);
        } else {
            launchMatvec(kernel, (long) m * matvecLocal, m, cols, weight, q8, out);
        }
    }

    /**
     * Launches the multi-column integer-dot matvec {@code kernel} over {@code m} rows of {@code weight}
     * against the {@code nTokens} consecutive Q8_1 rows of {@code xq} ({@code nTokens} in
     * [1, {@link LlmKernelConstants#MATVEC_N_MAX_TOKENS}]): the one-workgroup-per-row geometry of the
     * single-row form, with the weight row read once for the whole token group.
     */
    private void launchInt8MatvecN(String kernel, int m, int cols, int nTokens, GpuBuffer weight,
                                   GpuBuffer xq, GpuBuffer out) {
        launchMatvec(kernel, (long) m * matvecLocal, m, cols, nTokens, weight, xq, out);
    }

    /**
     * Launches the fused gate/up integer-dot matvec {@code kernel} over {@code m} rows of
     * {@code gateWeight} and {@code upWeight} against the Q8_1 row {@code xq}, pairing the rows in the
     * {@code {T}_MatVec2FusedGateUp_Q8} form above the two-row floor when the type measured faster paired.
     */
    private void launchInt8GateUp(String kernel, GGMLType type, int m, int cols, GpuBuffer gateWeight,
                                  GpuBuffer upWeight, GpuBuffer q8, GpuBuffer out) {
        if (pairInt8Rows(type) && twoRowUsable(kernel, m)) {
            launchMatvec(twoRowName(kernel), (long) ceilDiv(m, 2) * matvecLocal, m, cols, gateWeight, upWeight, q8,
                    out);
        } else {
            launchMatvec(kernel, (long) m * matvecLocal, m, cols, gateWeight, upWeight, q8, out);
        }
    }

    /** Whether the paired integer-dot form measured faster than the single-row form for {@code type} here. */
    private boolean pairInt8Rows(GGMLType type) {
        return Int8MatvecTuning.pairedIsFaster(backend.backendType(), type);
    }

    /**
     * Whether the two-row form of {@code kernel} serves {@code m} rows: at least two rows, enough live
     * threads ({@link #TWO_ROW_MIN_THREADS}) that pairing the rows does not starve the device more
     * than the halved activation re-reads pay back, and the kernel prepared at the matvec work-group
     * size (a weight type without the family form, such as F16 or F32, takes the single-row kernel).
     * The integer-dot lane's {@code {T}_MatVec_Q8} and {@code {T}_MatVecFusedGateUp_Q8} pair through
     * {@code {T}_MatVec2_Q8} and {@code {T}_MatVec2FusedGateUp_Q8} under the same floor.
     */
    private boolean twoRowUsable(String kernel, int m) {
        if (m < 2 || (long) ceilDiv(m, 2) * matvecLocal < TWO_ROW_MIN_THREADS) {
            return false;
        }
        return optionalReady(twoRowName(kernel), matvecLocal);
    }

    /**
     * Whether the integer-dot (Q8_1 activation) matvec lane can serve a projection of
     * {@code w} through {@code kernel} (the single-token and multi-column lanes share the gate): the
     * weight type has the kernel, its rows are a whole number of
     * weight blocks (the int8 kernels read complete 32/256-element groups), the configured
     * {@link Int8MatvecMode} and device permit it (packed dot products reported, or the mode forces the
     * lane), {@link Int8MatvecTuning} has no decode evidence against the type on this backend, and the
     * kernel is prepared at the matvec work-group size.
     */
    private boolean int8LaneUsable(String kernel, GpuWeight w) {
        if (kernel == null || cols(w) % w.type().getBlockSize() != 0) {
            return false;
        }
        return int8LaneSelected(w.type(), (long) rows(w) * cols(w)) && optionalReady(kernel, matvecLocal);
    }

    /**
     * Whether the configured {@link Int8MatvecMode} selects the lane for a projection of {@code type}
     * with {@code elements} weight elements on this backend: the mode is not {@link Int8MatvecMode#OFF},
     * and either it forces the lane or the device reports packed dot products and the type measured
     * faster with the lane here at that projection size ({@link Int8MatvecTuning#laneIsFaster}).
     */
    private boolean int8LaneSelected(GGMLType type, long elements) {
        if (!int8Mode.permitsLane()) {
            return false;
        }
        return int8Mode.forcesLane()
                || (kernels.caps().hasFeature(DeviceFeature.INTEGER_DOT_PRODUCT)
                    && Int8MatvecTuning.laneIsFaster(kernels.backend().backendType(), type, elements));
    }

    /**
     * Whether a single-token activation row at element {@code off} of {@code cols} elements can back a
     * {@link #shareQ8Activations} scope: the quantizer reads a whole row from element 0 of its buffer, so a row
     * at a nonzero offset is staged by its projections instead.
     */
    public static boolean shareableQ8Activations(int off, int cols) {
        return off == 0 && cols > 0 && cols % Q8_1_BLOCK_SIZE == 0;
    }

    /**
     * A single-token activation row quantized to Q8_1 blocks once for several projections that read it
     * ({@link #shareQ8Activations}). Every projection of the scope whose activation view is exactly the shared
     * row at element 0 of its buffer and whose weight takes the integer-dot lane reuses the shared blocks; a
     * projection of a transformed or staged input, or of a weight outside the lane, quantizes its own row as
     * it would without the share. The caller keeps the row unchanged until the scope closes. Closing enqueues
     * nothing and is idempotent.
     */
    public final class SharedQ8 implements AutoCloseable {
        private final GpuBuffer source;
        private final int cols;
        private GpuBuffer blocks;
        private boolean closed;

        private SharedQ8(GpuBuffer source, int cols) {
            this.source = source;
            this.cols = cols;
        }

        /** The Q8_1 blocks of the shared row, quantizing it on first use. */
        private GpuBuffer blocks() {
            if (closed) {
                throw new IllegalStateException("the Q8_1 activation share is closed");
            }
            if (blocks == null) {
                int n = cols / Q8_1_BLOCK_SIZE;
                ScratchSlot slot = scratch.staging(GpuScratch.Staging.SHARED_Q8_1, (long) n * Q8_1_FLOATS);
                launchMatvec(QUANTIZE_Q8_1, n, cols, source, slot.buffer());
                blocks = slot.buffer();
            }
            return blocks;
        }

        private boolean matches(View x, int width) {
            return !closed && width == cols && x.buf == source && x.off == 0;
        }

        @Override
        public void close() {
            if (closed) {
                return;
            }
            closed = true;
            if (activeShare == this) {
                activeShare = null;
            }
        }
    }

    /**
     * Opens a {@link SharedQ8} scope over the activation row at {@code x[off ...]} of {@code cols} elements, so
     * the single-token projections made before it closes quantize the row once. The row must start at element
     * 0 of its buffer ({@link #shareableQ8Activations}) and the caller must not write it inside the scope. One
     * scope is open at a time.
     *
     * @throws IllegalArgumentException when the view is out of range or cannot be shared
     * @throws IllegalStateException    when a share is already open
     */
    public SharedQ8 shareQ8Activations(GpuBuffer x, int off, int cols) {
        ensureOpen();
        ensureNoShare();
        checkPositive(cols, "Q8_1 activation share width");
        checkView("Q8_1 activation share", x, off, cols);
        if (!shareableQ8Activations(off, cols)) {
            throw new IllegalArgumentException("Q8_1 activation share needs a row at element 0 of its buffer of a"
                    + " multiple of " + Q8_1_BLOCK_SIZE + " elements, got offset " + off + " and width " + cols);
        }
        SharedQ8 share = new SharedQ8(x, cols);
        activeShare = share;
        return share;
    }

    private void ensureNoShare() {
        if (activeShare != null) {
            throw new IllegalStateException("a Q8_1 activation share is already open");
        }
    }

    /**
     * A scope that routes the projections of W12 weights through the nested 8-bit view of their code bytes (the
     * {@code W12Q_*} row set): every in-band element decodes as its window binade's sign/exponent/top-3-mantissa
     * code with the B nibble read as the fixed midpoint, at most 1/16 relative error, while escaped elements keep
     * their exact bits — the fast first-level verification pass of speculative decoding reads the Q bytes only
     * (the host twin of the view is {@code W12Codec.qViewElement}). The scope affects no other weight type, and
     * the fused gate/up form is not taken while it is open. Closing enqueues nothing and is idempotent.
     */
    public final class W12QView implements AutoCloseable {
        private boolean closed;

        private W12QView() {
        }

        @Override
        public void close() {
            if (closed) {
                return;
            }
            closed = true;
            if (activeW12Q == this) {
                activeW12Q = null;
            }
        }
    }

    /**
     * Opens a {@link W12QView} scope: until it closes, every projection of a W12 weight reads the nested 8-bit
     * view of its code bytes. One scope is open at a time.
     *
     * @throws IllegalStateException when a scope is already open
     */
    public W12QView openW12QView() {
        ensureOpen();
        if (activeW12Q != null) {
            throw new IllegalStateException("a W12 Q-view scope is already open");
        }
        W12QView scope = new W12QView();
        activeW12Q = scope;
        return scope;
    }

    /** {@link #quantizeQ8} through the open {@link SharedQ8} when it covers this view. */
    private GpuBuffer quantizedRow(View x, int cols) {
        SharedQ8 share = activeShare;
        if (share != null && share.matches(x, cols)) {
            return share.blocks();
        }
        return quantizeQ8(x.buf, cols);
    }

    /**
     * Quantizes one F32 activation row of {@code cols} elements (a multiple of {@link #Q8_1_BLOCK_SIZE})
     * to Q8_1 blocks in the Q8_1 staging slot ({@value #QUANTIZE_Q8_1}) for the integer-dot matvec lane.
     * The row must start at element 0 of {@code x}.
     */
    private GpuBuffer quantizeQ8(GpuBuffer x, int cols) {
        int blocks = cols / Q8_1_BLOCK_SIZE;
        ScratchSlot slot = scratch.staging(GpuScratch.Staging.Q8_1, (long) blocks * Q8_1_FLOATS);
        launchMatvec(QUANTIZE_Q8_1, blocks, cols, x, slot.buffer());
        return slot.buffer();
    }

    /**
     * Quantizes {@code nTokens} consecutive F32 activation rows of {@code cols} elements (each a multiple
     * of {@link #Q8_1_BLOCK_SIZE}) to Q8_1 blocks in the Q8_1 staging slot ({@value #QUANTIZE_Q8_1}) for
     * the multi-column integer-dot matvec lane. The rows are contiguous in the input and in the slot, so
     * the quantizer's linear block walk serves the whole group in one launch; the group must start at
     * element 0 of {@code x}.
     */
    private GpuBuffer quantizeQ8N(GpuBuffer x, int nTokens, int cols) {
        int blocks = cols / Q8_1_BLOCK_SIZE;
        int size = Math.multiplyExact(nTokens, cols);
        ScratchSlot slot = scratch.staging(GpuScratch.Staging.Q8_1, (long) blocks * nTokens * Q8_1_FLOATS);
        launchMatvec(QUANTIZE_Q8_1, (long) blocks * nTokens, size, x, slot.buffer());
        return slot.buffer();
    }

    private boolean tiledUsable(WeightKernels k, GpuWeight w, int nTokens) {
        if (maxWorkGroupSize < TILED_WG || ceilDiv(nTokens, TILED_TOKENS) > maxGroups[1]) {
            return false;
        }
        for (int s = 0; s < w.shardCount(); s++) {
            if (ceilDiv(w.shardRows(s), TILED_ROWS) > maxGroups[0]) {
                return false;
            }
        }
        return optionalReady(k.tiled, TILED_WG, 1);
    }

    /**
     * Whether an optional kernel (the projection kernels of the class notes, or a block's faster form of a kernel it
     * can replace, such as the tensor-core grouped expert GEMM) can be launched at work-group size
     * {@code localSize}: prepared on the first query ({@link #prewarm(String, long[])}) and remembered, so a kernel
     * the backend does not provide or cannot run at that size is never launched and its caller takes the fallback;
     * a provided kernel that fails is logged once at info. Each optional kernel is launched with one work-group size
     * only.
     */
    public boolean optionalReady(String kernel, long... localSize) {
        Boolean ready = optionalStatus.get(kernel);
        if (ready == null) {
            ready = prewarm(kernel, localSize);
            if (!ready && kernels.has(kernel)) {
                LOGGER.info("Kernel {} cannot run work-groups of {} on this device; its products use the fallback"
                        + " kernels", kernel, Arrays.toString(localSize));
            }
            optionalStatus.put(kernel, ready);
        }
        return ready;
    }

    /**
     * Prepares the projection and lookup kernels of weights stored as {@code type} for the work-group sizes this
     * class launches them with ({@link LlmKernels#prewarm(String, long[])}), so a backend that builds one pipeline
     * per work-group size (Vulkan) does it while the model binds rather than at the first dispatch. Kernels the
     * backend does not provide are skipped, as the launches skip them. The optional kernels (tiled GEMM, {@code _2D}
     * batch, in-kernel accumulate, two-row matvec) that cannot run their work-group size on this device are recorded
     * as unavailable, so the projections use their fallbacks; a required kernel that cannot run its size fails here,
     * while the model binds.
     *
     * @param type   device storage type of the weights
     * @param matmul whether weights of this type are projected: the single-token, accumulate, batched and tiled
     *               products
     * @param lookup whether weights of this type are embedding tables: the host-id and device-id row lookups
     * @throws UnsupportedOperationException when GpuOps has no kernels for {@code type}
     * @throws com.eonmux.neuralfree.acceleration.UnsupportedKernelException when a provided matvec, batch or lookup
     *                                                                       kernel cannot run its work-group size
     *                                                                       (OpenCL, Vulkan)
     * @throws IllegalArgumentException when a provided matvec, batch or lookup kernel's threads-per-block limit is
     *                                  below its work-group size (CUDA)
     */
    public void prewarmWeightKernels(GGMLType type, boolean matmul, boolean lookup) {
        ensureOpen();
        Objects.requireNonNull(type, "type");
        WeightKernels k = family(type);
        long[] matvec = {matvecLocal};
        if (matmul && matvecLocal % MATVEC_MAX_TPS == 0) {
            prewarmIfPresent(k.matvec, matvec);
            prewarmIfPresent(k.batch, matvec);
            optionalReady(k.matvecAdd, matvecLocal);
            optionalReady(k.batch2d, matvecLocal, 1);
            optionalReady(twoRowName(k.matvec), matvecLocal);
            if (k.matvecQ8 != null && int8LaneSelected(type, Long.MAX_VALUE)) {
                optionalReady(k.matvecQ8, matvecLocal);
                optionalReady(k.matvecAddQ8, matvecLocal);
                optionalReady(k.matvecNQ8, matvecLocal);
                if (pairInt8Rows(type)) {
                    optionalReady(twoRowName(k.matvecQ8), matvecLocal);
                    optionalReady(twoRowName(fusedGateUpName(k.matvecQ8)), matvecLocal);
                }
            }
            if (type == GGMLType.W12) {
                // The whole-tensor BF16 decode behind a wide prefill, and the Q-view row set of the
                // neuralfree.weights.w12 verification pass; the fused gate/up has no Q-view twin.
                prewarmIfPresent(W12_DECODE_BF16, matvec);
                optionalReady(W12Q_FAMILY.matvec, matvecLocal);
                optionalReady(W12Q_FAMILY.matvecAdd, matvecLocal);
                optionalReady(W12Q_FAMILY.batch, matvec);
                optionalReady(W12Q_FAMILY.batch2d, matvecLocal, 1);
            }
        }
        if (matmul && k.tiled != null && maxWorkGroupSize >= TILED_WG) {
            optionalReady(k.tiled, TILED_WG, 1);
            lanes.prewarm(type);
        }
        if (lookup) {
            prewarmIfPresent(k.embed, matvec);
            prewarmIfPresent(k.batchEmbed, matvec);
        }
    }

    /** Prepares a provided kernel that has no fallback; one that cannot run {@code localSize} fails the bind. */
    private void prewarmIfPresent(String kernel, long[] localSize) {
        if (kernels.has(kernel)) {
            kernels.prewarm(kernel, localSize);
        }
    }

    private void requireMatvecLocal(GpuWeight w) {
        if (matvecLocal % MATVEC_MAX_TPS != 0) {
            throw new UnsupportedOperationException("matvec of " + w.name() + " needs a work-group size that is a"
                    + " multiple of " + MATVEC_MAX_TPS + "; this device allows " + matvecLocal);
        }
    }

    // ================================================================== embeddings

    /**
     * Looks up {@code nTokens} rows of an embedding table by the token ids in {@code tokenIds} (ints, element 0
     * onward; each must be a valid row) into {@code out[outOff + t*cols ...]}, then applies the table's inverse
     * Hadamard transform when it has one (llama-graph.cpp:2683-2690).
     *
     * @throws UnsupportedOperationException when the table is split into row shards: use
     *                                       {@link #embed(GpuWeight, int[], GpuBuffer, int)}
     */
    public void embed(GpuWeight table, GpuBuffer tokenIds, int nTokens, GpuBuffer out, int outOff) {
        ensureOpen();
        int dim = cols(table);
        rows(table);
        checkTokens(nTokens);
        Objects.requireNonNull(tokenIds, "tokenIds");
        if ((long) nTokens * Integer.BYTES > tokenIds.sizeInBytes()) {
            throw new IllegalArgumentException("token id buffer of " + tokenIds.sizeInBytes() + " bytes is too small"
                    + " for " + nTokens + " ids");
        }
        if (!table.isSingleShard()) {
            throw new UnsupportedOperationException("embedding table " + table.name() + " is split into "
                    + table.shardCount() + " row shards; device token ids cannot select the shard, use the host-id"
                    + " lookup");
        }
        long n = (long) nTokens * dim;
        checkView("embedding output of " + table.name(), out, outOff, n);
        checkInt(n, "embedding output size of " + table.name());
        WeightKernels k = family(table);
        HadamardSpec inv = table.inverseHadamard();
        boolean direct = outOff == 0 && (inv == null || inv.inPlaceAllowed());
        View dst = direct ? new View(out, 0) : stagingView(GpuScratch.Staging.OUTPUT, n);
        launchMatvec(k.batchEmbed, n, dim, nTokens, tokenIds, table.shard(0), dst.buf);
        finishLookup(inv, dst, out, outOff, nTokens, dim);
    }

    /**
     * Looks up the rows of host token ids (any shard layout) into {@code out[outOff + t*cols ...]}, then applies the
     * table's inverse Hadamard transform when it has one.
     *
     * @throws IllegalArgumentException when a token id is outside the table
     */
    public void embed(GpuWeight table, int[] tokens, GpuBuffer out, int outOff) {
        embed(table, tokens, tokens == null ? 0 : tokens.length, out, outOff);
    }

    /**
     * {@link #embed(GpuWeight, int[], GpuBuffer, int)} over the first {@code count} entries of {@code tokens}.
     *
     * @throws IllegalArgumentException when {@code count} is outside {@code [0, tokens.length]} or a token id is
     *                                  outside the table
     */
    public void embed(GpuWeight table, int[] tokens, int count, GpuBuffer out, int outOff) {
        ensureOpen();
        Objects.requireNonNull(tokens, "tokens");
        if (count < 0 || count > tokens.length) {
            throw new IllegalArgumentException("embedding lookup count " + count + " is outside [0, "
                    + tokens.length + "]");
        }
        int nTokens = count;
        int dim = cols(table);
        checkTokens(nTokens);
        long rows = table.rows();
        for (int t = 0; t < nTokens; t++) {
            if (tokens[t] < 0 || tokens[t] >= rows) {
                throw new IllegalArgumentException("token id " + tokens[t] + " at " + t + " is outside the "
                        + rows + " rows of " + table.name());
            }
        }
        long n = (long) nTokens * dim;
        checkView("embedding output of " + table.name(), out, outOff, n);
        checkInt(outOff + n, "embedding output end of " + table.name());
        if (table.isSingleShard()) {
            ScratchSlot ids = scratch.staging(GpuScratch.Staging.INPUT, nTokens);
            writeInts(ids, tokens);
            embed(table, ids.buffer(), nTokens, out, outOff);
            return;
        }
        WeightKernels k = family(table);
        HadamardSpec inv = table.inverseHadamard();
        View rowsDst = inv != null && !inv.inPlaceAllowed()
                ? stagingView(GpuScratch.Staging.RESULT, n) : new View(out, outOff);
        for (int t = 0; t < nTokens; t++) {
            int s = table.shardOfRow(tokens[t]);
            int localId = (int) (tokens[t] - table.shardRowStart(s));
            int at = rowsDst.off + t * dim;
            if (at == 0) {
                launchMatvec(k.embed, dim, dim, localId, table.shard(s), rowsDst.buf);
            } else {
                ScratchSlot stage = scratch.staging(GpuScratch.Staging.OUTPUT, dim);
                launchMatvec(k.embed, dim, dim, localId, table.shard(s), stage.buffer());
                copy2dUnchecked(1, dim, stage.buffer(), stage.offset(), dim, rowsDst.buf, at, dim);
            }
        }
        if (inv != null) {
            hadamard(inv, rowsDst.buf, rowsDst.off, out, outOff, nTokens, dim);
        }
    }

    /** Moves looked-up rows from {@code dst} to the output, through the inverse transform when there is one. */
    private void finishLookup(HadamardSpec inv, View dst, GpuBuffer out, int outOff, int nTokens, int dim) {
        boolean inPlace = dst.buf == out && dst.off == outOff;
        if (inv != null) {
            hadamard(inv, dst.buf, dst.off, out, outOff, nTokens, dim);
        } else if (!inPlace) {
            copy2dUnchecked(nTokens, dim, dst.buf, dst.off, dim, out, outOff, dim);
        }
    }

    private void writeInts(ScratchSlot slot, int[] values) {
        ByteBuffer bb = ByteBuffer.allocate(values.length * Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        for (int v : values) {
            bb.putInt(v);
        }
        GpuEvent e = queue.enqueueWriteBuffer(slot.buffer(), (long) slot.offset() * Float.BYTES,
                (long) values.length * Integer.BYTES, bb.array());
        if (e != null) {
            e.close();
        }
    }

    // ================================================================== Hadamard

    /**
     * Applies a Hadamard transform to {@code nRows} rows of {@code width}: {@code y[yOff + r*width ...]} from
     * {@code x[xOff + r*width ...]} ({@value HadamardSpec#KERNEL}). In place ({@code x == y}, {@code xOff == yOff})
     * only when {@link HadamardSpec#inPlaceAllowed()}.
     *
     * @throws IllegalArgumentException when the width does not fit the transform, the block exceeds the kernel, or
     *                                  the views overlap in a way the transform does not allow
     */
    public void hadamard(HadamardSpec h, GpuBuffer x, int xOff, GpuBuffer y, int yOff, int nRows, int width) {
        ensureOpen();
        Objects.requireNonNull(h, "transform");
        h.checkWidth(width);
        if (!h.kernelSupported()) {
            throw new IllegalArgumentException("Hadamard block " + h.block() + " exceeds the GPU limit of "
                    + HadamardSpec.MAX_KERNEL_BLOCK);
        }
        checkTokens(nRows);
        long n = (long) nRows * width;
        checkView("Hadamard input", x, xOff, n);
        checkView("Hadamard output", y, yOff, n);
        boolean inPlace = x == y && xOff == yOff;
        if (inPlace ? !h.inPlaceAllowed() : overlaps(x, xOff, n, y, yOff, n)) {
            throw new IllegalArgumentException("Hadamard " + h + ": input and output views overlap"
                    + (inPlace ? " and the permutation forbids in-place use" : ""));
        }
        GpuBuffer signs = h.hasSigns() ? signBuffer(h) : placeholder();
        long groups = n / h.block();
        launch(HadamardSpec.KERNEL, groups * local, nRows, width, h.block(), h.kernelSignMode(), h.permHd(),
                h.permNk(), h.permRep(), xOff, yOff, signs, x, y);
    }

    /**
     * The device sign vector of a transform: one read-only buffer per {@link HadamardSpec#signsName()}, uploaded on
     * first use. Transforms seen before resolve by identity without touching their sign arrays.
     */
    private GpuBuffer signBuffer(HadamardSpec h) {
        GpuBuffer cached = signsBySpec.get(h);
        if (cached != null) {
            return cached;
        }
        String name = h.signsName();
        float[] values = h.signs();
        float[] known = signValues.get(name);
        if (known != null) {
            if (!Arrays.equals(known, values)) {
                throw new IllegalArgumentException("two Hadamard transforms share the sign vector name " + name
                        + " with different signs");
            }
            GpuBuffer b = signBuffers.get(name);
            signsBySpec.put(h, b);
            return b;
        }
        GpuBuffer b = backend.allocateBuffer((long) values.length * Float.BYTES, true);
        try {
            GpuEvent e = queue.enqueueWriteBuffer(b, 0, (long) values.length * Float.BYTES, values);
            if (e != null) {
                e.close();
            }
        } catch (RuntimeException ex) {
            b.close();
            throw ex;
        }
        signValues.put(name, values);
        signBuffers.put(name, b);
        signsBySpec.put(h, b);
        return b;
    }

    // ================================================================== input transforms

    /**
     * Fake-quantizes {@code rows} contiguous rows of {@code cols} elements ({@value #ACT_FAKE_QUANT}): run
     * {@code g} of {@code scheme.runLength(cols)} consecutive elements covers
     * {@code x[xOff + g*runLen ...]} and {@code y[yOff + g*runLen ...]}, and
     * {@code y = dequant(quant(x))} per {@code scheme} (gguf.quant.ActFakeQuant, the host twin). In place
     * ({@code x == y}, {@code xOff == yOff}); any other overlap is rejected.
     *
     * @param scale the {@code scale} argument ({@link ActQuantScheme#isStatic()} schemes' {@code input_scale},
     *              FP8_TOKEN's amax bound, 0 otherwise)
     * @param zp    the {@code zero_point} argument ({@link ActQuantScheme#INT8_STATIC_ASYM}'s
     *              {@code input_zero_point}, 0 otherwise)
     * @param gs    the {@code global_scale} argument ({@link ActQuantScheme#NVFP4}'s {@code input_scale},
     *              0 otherwise)
     * @throws IllegalArgumentException when the range lies outside a buffer, the row does not split into whole runs,
     *                                  or the views overlap without coinciding
     */
    public void actFakeQuant(ActQuantScheme scheme, float scale, int zp, float gs, GpuBuffer x, int xOff, GpuBuffer y,
                             int yOff, int rows, int cols) {
        ensureOpen();
        Objects.requireNonNull(scheme, "scheme");
        checkTokens(rows);
        checkPositive(cols, "activation quantization width");
        int runLen = scheme.runLength(cols);
        long runs = Math.multiplyExact((long) rows, cols / runLen);
        checkInt(runs, "activation quantization run count");
        long n = Math.multiplyExact((long) rows, cols);
        checkView("activation quantization input", x, xOff, n);
        checkView("activation quantization output", y, yOff, n);
        if (x == y && xOff != yOff && overlaps(x, xOff, n, y, yOff, n)) {
            throw new IllegalArgumentException("activation quantization: input and output views overlap without"
                    + " coinciding");
        }
        launch(ACT_FAKE_QUANT, runs * local, (int) runs, runLen, scheme.code(), scale, zp, gs, xOff, yOff, x, y);
    }

    /**
     * Gathers the columns of {@code rows} contiguous rows of {@code cols} elements ({@value #PERMUTE_COLS}):
     * {@code y[yOff + r*cols + k] = x[xOff + r*cols + perm[k]]}, where {@code perm} holds {@code cols} ints
     * ({@code perm[k]} is the input column read at position {@code k}; the GPTQ / compressed-tensors act-order
     * {@code <base>.input_perm}). Out of place: {@code y} shares no element with {@code x}.
     *
     * @throws IllegalArgumentException when a view is out of range, {@code perm} is too small, or the views overlap
     */
    public void permuteCols(GpuBuffer perm, GpuBuffer x, int xOff, GpuBuffer y, int yOff, int rows, int cols) {
        ensureOpen();
        checkTokens(rows);
        checkPositive(cols, "input permutation width");
        Objects.requireNonNull(perm, "perm");
        if ((long) cols * Integer.BYTES > perm.sizeInBytes()) {
            throw new IllegalArgumentException("input permutation needs " + cols + " ints, its buffer holds "
                    + perm.sizeInBytes() / Integer.BYTES);
        }
        long n = Math.multiplyExact((long) rows, cols);
        checkInt(n, "permuted projection width");
        checkView("permuted projection input", x, xOff, n);
        checkView("permuted projection output", y, yOff, n);
        if (overlaps(x, xOff, n, y, yOff, n)) {
            throw new IllegalArgumentException("permuted projection: input and output views overlap; the gather"
                    + " runs out of place");
        }
        launch(PERMUTE_COLS, n, rows, cols, xOff, yOff, perm, x, y);
    }

    /**
     * Multiplies every element by its column's multiplier ({@value #SCALE_COLS}):
     * {@code y[yOff + r*cols + c] = x[xOff + r*cols + c] * scale[c]} for {@code r < rows} and {@code c < cols}, where
     * {@code scale} holds at least {@code cols} floats (a weight's activation pre-scale, applied before the
     * fake-quantization). Out of place: {@code y} shares no element with {@code x}.
     *
     * @throws IllegalArgumentException when a view is out of range, {@code scale} is too small, or the views overlap
     */
    public void scaleCols(GpuBuffer scale, GpuBuffer x, int xOff, GpuBuffer y, int yOff, int rows, int cols) {
        ensureOpen();
        checkTokens(rows);
        checkPositive(cols, "activation pre-scale width");
        Objects.requireNonNull(scale, "scale");
        if ((long) cols * Float.BYTES > scale.sizeInBytes()) {
            throw new IllegalArgumentException("activation pre-scale needs " + cols + " floats, its buffer holds "
                    + scale.sizeInBytes() / Float.BYTES);
        }
        long n = Math.multiplyExact((long) rows, cols);
        checkInt(n, "pre-scaled projection width");
        checkView("pre-scaled projection input", x, xOff, n);
        checkView("pre-scaled projection output", y, yOff, n);
        if (overlaps(x, xOff, n, y, yOff, n)) {
            throw new IllegalArgumentException("pre-scaled projection: input and output views overlap; the"
                    + " multiplication runs out of place");
        }
        launch(SCALE_COLS, n, rows, cols, xOff, yOff, scale, x, y);
    }

    // ================================================================== norm and activations

    /**
     * Normalizes {@code nRows} contiguous rows of {@code width}: {@code out[outOff + r*width ...]} from
     * {@code in[inOff + r*width ...]} ({@value #ROW_NORM}, mode {@link NormKind#rowNormMode()}). {@code w} and
     * {@code b} are optional F32 weight and bias of {@code width} columns; with {@code k} rows, row {@code r} uses
     * weight row {@code r % k} (per-head norms). In place when {@code in} and {@code out} coincide.
     */
    public void norm(NormKind k, GpuWeight w, GpuWeight b, float eps, GpuBuffer in, int inOff, GpuBuffer out,
                     int outOff, int nRows, int width) {
        ensureOpen();
        Objects.requireNonNull(k, "kind");
        if (!k.isRowNorm()) {
            throw new IllegalArgumentException(k + " is not a row norm; group norms have their own kernels");
        }
        checkTokens(nRows);
        checkPositive(width, "norm width");
        long n = (long) nRows * width;
        checkView("norm input", in, inOff, n);
        checkView("norm output", out, outOff, n);
        checkSameOrDisjoint(in, inOff, out, outOff, n, "norm");
        int wRows = 1;
        GpuBuffer wb = null;
        GpuBuffer bb = null;
        if (w != null) {
            wRows = vectorRows(w, width, "norm weight");
            wb = w.shard(0);
        }
        if (b != null) {
            int bRows = vectorRows(b, width, "norm bias");
            if (w != null && bRows != wRows) {
                throw new IllegalArgumentException("norm weight " + w.name() + " has " + wRows + " rows, bias "
                        + b.name() + " " + bRows);
            }
            wRows = bRows;
            bb = b.shard(0);
        }
        Object[] args = NormKernels.rowNormArgs(k, eps, nRows, width, width, in, inOff, out, outOff, wb, bb, wRows,
                placeholder());
        launch(ROW_NORM, (long) nRows * local, args);
    }

    /**
     * Adds a residual branch to {@code nRows} stream rows of {@code width} and normalizes the sum in one
     * launch ({@value NormKernels#RESIDUAL_MIX_NORM}): {@code x[..] = mix(x, addend)} per {@code mix} and
     * {@code out[..] = norm(x[..])} with the modes, eps, weight row and bias of {@link #norm}. This is the
     * launch {@code LLM_ResidualMix} followed by {@code LLM_RowNorm} would make, fused so the layer
     * executor spends one dispatch where it spent two. {@code x} and {@code out} must be disjoint;
     * {@code addend} is read only.
     *
     * @return false when this backend provides no fused kernel at the row-reduce work-group size, in
     *         which case nothing was enqueued and the caller runs the residual mix and the norm
     */
    public boolean residualMixNorm(NormKernels.MixMode mix, float s, NormKind k, GpuWeight w, GpuWeight b,
                                   float eps, GpuBuffer x, int xOff, GpuBuffer addend, int addOff,
                                   GpuBuffer out, int outOff, int nRows, int width) {
        ensureOpen();
        Objects.requireNonNull(k, "kind");
        if (!k.isRowNorm()) {
            throw new IllegalArgumentException(k + " is not a row norm; group norms have their own kernels");
        }
        checkTokens(nRows);
        checkPositive(width, "norm width");
        int wRows = 1;
        GpuBuffer wb = null;
        GpuBuffer bb = null;
        if (w != null) {
            wRows = vectorRows(w, width, "norm weight");
            wb = w.shard(0);
        }
        if (b != null) {
            int bRows = vectorRows(b, width, "norm bias");
            if (w != null && bRows != wRows) {
                throw new IllegalArgumentException("norm weight " + w.name() + " has " + wRows + " rows, bias "
                        + b.name() + " " + bRows);
            }
            wRows = bRows;
            bb = b.shard(0);
        }
        if (!fusedMode.permitsFusion()) {
            return false;
        }
        if (!optionalReady(NormKernels.RESIDUAL_MIX_NORM, local)) {
            if (fusedMode.requiresFusion()) {
                throw new com.eonmux.neuralfree.acceleration.UnsupportedKernelException(
                        NormKernels.RESIDUAL_MIX_NORM + " is required by -D" + FusedMode.PROPERTY
                        + "=on but the " + backend.backendType() + " backend does not provide it");
            }
            return false;
        }
        Object[] args = NormKernels.residualMixNormArgs(mix, s, k, eps, nRows, width, xOff, addOff, outOff,
                x, addend, wb, bb, wRows, out, placeholder());
        launch(NormKernels.RESIDUAL_MIX_NORM, (long) nRows * local, args);
        return true;
    }

    private static int vectorRows(GpuWeight v, int width, String what) {
        if (v.type() != GGMLType.F32 || !v.isSingleShard() || v.cols() != width || v.rows() > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(what + " " + v.name() + " must be a single-buffer F32 tensor of "
                    + width + " columns, got " + v);
        }
        return (int) v.rows();
    }

    /** {@code out[outOff + i] = act(in[inOff + i])} for {@code i < n} ({@value #UNARY}). In place when they coincide. */
    public void unary(ActOp op, ActParams p, GpuBuffer in, int inOff, GpuBuffer out, int outOff, int n) {
        ensureOpen();
        Objects.requireNonNull(op, "op");
        checkPositive(n, "activation length");
        checkView("activation input", in, inOff, n);
        checkView("activation output", out, outOff, n);
        checkSameOrDisjoint(in, inOff, out, outOff, n, "activation");
        launch(UNARY, n, ActKernels.unaryArgs(op, p != null ? p : ActParams.NONE, n, in, inOff, out, outOff));
    }

    /**
     * {@code out[outOff + r*outStride + c] = glu(gate[gateOff + r*gateStride + c], up[upOff + r*upStride + c])} for
     * {@code r < nRows, c < nCols} ({@value #GLU}). The fused gate+up layout passes one buffer with
     * {@code upOff = gateOff + nCols}. {@code out} may coincide with {@code gate} or {@code up} (same offset and
     * stride); otherwise it must share no element with either (the gate half of a fused gate+up row with equal
     * strides qualifies, the up half being disjoint from it).
     */
    public void glu(GluOp op, ActParams p, GpuBuffer gate, int gateOff, int gateStride, GpuBuffer up, int upOff,
                    int upStride, GpuBuffer out, int outOff, int outStride, int nRows, int nCols) {
        ensureOpen();
        Objects.requireNonNull(op, "op");
        ActParams ap = p != null ? p : ActParams.NONE;
        checkTokens(nRows);
        checkPositive(nCols, "GLU columns");
        checkInt((long) nRows * nCols, "GLU element count");
        long gExt = extent(nRows, nCols, gateStride, "gate");
        long uExt = extent(nRows, nCols, upStride, "up");
        long oExt = extent(nRows, nCols, outStride, "out");
        checkView("GLU gate", gate, gateOff, gExt);
        checkView("GLU up", up, upOff, uExt);
        checkView("GLU output", out, outOff, oExt);
        checkStridedAlias(gate, gateOff, gateStride, out, outOff, outStride, nRows, nCols, "GLU gate");
        checkStridedAlias(up, upOff, upStride, out, outOff, outStride, nRows, nCols, "GLU up");
        launch(GLU, (long) nRows * nCols, ActKernels.gluArgs(op, ap, nRows, nCols, gate, gateOff, gateStride,
                up, upOff, upStride, out, outOff, outStride));
    }

    /**
     * Fused RMS norm and SiLU gate of a gated DeltaNet layer ({@value FusedKernels#GDN_NORM_GATE}): row
     * {@code r = t*nHeads + h} of {@code x} is normalized with weight row {@code (h % wRows)*headDim} and
     * multiplied by {@code silu(z)} of the same row, into {@code out}. {@code out} may coincide with {@code x}.
     *
     * @throws IllegalArgumentException when the geometry or the views are invalid
     */
    public void gdnNormGate(int nTokens, int nHeads, int headDim, int rowStride, GpuWeight w, float eps,
                            GpuBuffer x, int xOff, GpuBuffer z, int zOff, GpuBuffer out, int outOff) {
        ensureOpen();
        checkTokens(nTokens);
        checkPositive(rowStride, "gdn norm-gate row stride");
        long n = (long) nTokens * rowStride;
        checkView(FusedKernels.GDN_NORM_GATE + " input", x, xOff, n);
        checkView(FusedKernels.GDN_NORM_GATE + " gate", z, zOff, n);
        checkView(FusedKernels.GDN_NORM_GATE + " output", out, outOff, n);
        checkSameOrDisjoint(x, xOff, out, outOff, n, "gdn norm-gate");
        int wRows = vectorRows(w, headDim, "gdn norm-gate weight");
        Object[] args = FusedKernels.gdnNormGateArgs(nTokens, nHeads, headDim, rowStride, xOff, zOff, outOff,
                wRows, eps, x, w.shard(0), z, out);
        enqueue(FusedKernels.GDN_NORM_GATE,
                new long[]{FusedKernels.gdnNormGateGlobal(nTokens, nHeads, local)}, new long[]{local}, args);
    }

    private static long extent(int nRows, int nCols, int stride, String what) {
        if (stride < nCols) {
            throw new IllegalArgumentException("GLU " + what + " stride " + stride + " is below the " + nCols
                    + " columns");
        }
        return (long) (nRows - 1) * stride + nCols;
    }

    // ================================================================== element-wise

    /** {@code out = a + b} over {@code n} elements; {@code out} may coincide with {@code a} or {@code b}. */
    public void add(GpuBuffer a, int aOff, GpuBuffer b, int bOff, GpuBuffer out, int outOff, int n) {
        ensureOpen();
        checkPositive(n, "add length");
        axpby(n, 1.0f, a, aOff, 1.0f, b, bOff, out, outOff);
    }

    /** {@code accum += s * x} over {@code n} elements; {@code x} coincides with or is disjoint from {@code accum}. */
    public void scaleAdd(GpuBuffer x, int xOff, float s, GpuBuffer accum, int accOff, int n) {
        ensureOpen();
        checkPositive(n, "scaleAdd length");
        axpby(n, s, x, xOff, 1.0f, accum, accOff, accum, accOff);
    }

    /** {@code x *= s} over {@code n} elements in place. */
    public void scale(GpuBuffer x, int xOff, int n, float s) {
        ensureOpen();
        checkPositive(n, "scale length");
        axpby(n, 0.0f, null, 0, s, x, xOff, x, xOff);
    }

    /** Copies {@code n} floats; the views coincide (nothing to do) or are disjoint. */
    public void copy(GpuBuffer src, int srcOff, GpuBuffer dst, int dstOff, int n) {
        copy2d(src, srcOff, n, dst, dstOff, n, 1, n);
    }

    /**
     * Copies {@code rows x cols} floats between strided views ({@value #COPY2D}): row {@code r} of the source starts
     * at {@code srcOff + r*srcStride}, of the destination at {@code dstOff + r*dstStride}. The views must share no
     * element (interleaved column windows of one buffer with equal strides qualify) unless they coincide exactly
     * (nothing to do).
     */
    public void copy2d(GpuBuffer src, int srcOff, int srcStride, GpuBuffer dst, int dstOff, int dstStride, int rows,
                       int cols) {
        ensureOpen();
        checkTokens(rows);
        checkPositive(cols, "copy columns");
        long sExt = (long) (rows - 1) * srcStride + cols;
        long dExt = (long) (rows - 1) * dstStride + cols;
        if (srcStride < cols && rows > 1 || dstStride < cols && rows > 1) {
            throw new IllegalArgumentException("copy strides " + srcStride + "/" + dstStride + " are below the "
                    + cols + " columns");
        }
        checkView("copy source", src, srcOff, sExt);
        checkView("copy destination", dst, dstOff, dExt);
        checkInt((long) rows * cols, "copy element count");
        if (src == dst && srcOff == dstOff && (rows == 1 || srcStride == dstStride)) {
            return;
        }
        if (stridedOverlap(src, srcOff, srcStride, dst, dstOff, dstStride, rows, cols)) {
            throw new IllegalArgumentException("copy source and destination views overlap");
        }
        copy2dUnchecked(rows, cols, src, srcOff, srcStride, dst, dstOff, dstStride);
    }

    /**
     * Fills {@code bytes} bytes of {@code dst} from {@code byteOff} with a 32-bit pattern.
     *
     * @throws IllegalArgumentException when the range is negative, not 4-byte aligned or outside {@code dst}
     */
    public void fill(GpuBuffer dst, long byteOff, long bytes, int pattern) {
        ensureOpen();
        if (dst == null) {
            throw new IllegalArgumentException("fill destination buffer is null");
        }
        if (byteOff < 0 || bytes < 0 || (byteOff & 3) != 0 || (bytes & 3) != 0) {
            throw new IllegalArgumentException("fill offset " + byteOff + " and size " + bytes
                    + " must be non-negative multiples of 4 bytes");
        }
        if (byteOff + bytes > dst.sizeInBytes()) {
            throw new IllegalArgumentException("fill range [" + byteOff + ", " + (byteOff + bytes) + ") exceeds the "
                    + dst.sizeInBytes() + "-byte buffer");
        }
        if (bytes > 0) {
            queue.enqueueFillBuffer(dst, byteOff, bytes, pattern);
        }
    }

    private void copy2dUnchecked(int rows, int cols, GpuBuffer src, int srcOff, int srcStride, GpuBuffer dst,
                                 int dstOff, int dstStride) {
        launch(COPY2D, (long) rows * cols, rows, cols, srcStride, srcOff, dstStride, dstOff, src, dst);
    }

    /**
     * {@code out = alpha*a + beta*b} over {@code n} elements. A zero coefficient drops its operand (which may be
     * null). Operands that coincide with {@code out} are moved to the aliasable slot of {@value #AXPBY}; any other
     * overlap with {@code out} is rejected.
     */
    private void axpby(int n, float alpha, GpuBuffer a, int aOff, float beta, GpuBuffer b, int bOff, GpuBuffer out,
                       int outOff) {
        checkView("axpby output", out, outOff, n);
        boolean useA = alpha != 0.0f;
        boolean useB = beta != 0.0f;
        if (useA) {
            checkView("axpby operand", a, aOff, n);
        }
        if (useB) {
            checkView("axpby operand", b, bOff, n);
        }
        boolean aIsOut = useA && a == out && aOff == outOff;
        boolean bIsOut = useB && b == out && bOff == outOff;
        if (aIsOut && bIsOut) {
            beta = alpha + beta;
            useA = false;
        } else if (aIsOut) {
            GpuBuffer tb = a;
            int to = aOff;
            float tc = alpha;
            a = b;
            aOff = bOff;
            alpha = beta;
            b = tb;
            bOff = to;
            beta = tc;
            useA = useB;
            useB = true;
        }
        if (useA && overlaps(a, aOff, n, out, outOff, n)) {
            throw new IllegalArgumentException("axpby operand overlaps the output without coinciding with it");
        }
        if (useB && !(b == out && bOff == outOff) && overlaps(b, bOff, n, out, outOff, n)) {
            throw new IllegalArgumentException("axpby operand overlaps the output without coinciding with it");
        }
        GpuBuffer ab = useA ? a : placeholder();
        GpuBuffer bb = useB ? b : placeholder();
        launch(AXPBY, n, n, useA ? alpha : 0.0f, useB ? beta : 0.0f, useA ? aOff : 0, useB ? bOff : 0, outOff,
                ab, bb, out);
    }

    // ================================================================== launch helpers

    /** 1-D launch: {@code roundUp(n, L)} work-items of local size {@code L}. */
    void launch(String kernel, long n, Object... args) {
        long g = roundUp(n, local);
        queue.enqueueKernelNoEvent(kernels.get(kernel), new long[]{g}, new long[]{local}, args);
    }

    /** 1-D launch at the matvec work-group size: {@code roundUp(n, Lm)} work-items of local size {@code Lm}. */
    private void launchMatvec(String kernel, long n, Object... args) {
        long g = roundUp(n, matvecLocal);
        queue.enqueueKernelNoEvent(kernels.get(kernel), new long[]{g}, new long[]{matvecLocal}, args);
    }

    /** 2-D launch: {@code {roundUp(gx, lx), gy}} with local {@code {lx, 1}}. */
    private void launch2d(String kernel, long gx, long gy, int lx, Object... args) {
        queue.enqueueKernelNoEvent(kernels.get(kernel), new long[]{roundUp(gx, lx), gy}, new long[]{lx, 1}, args);
    }

    /** The read-only float dummy; every placeholder use in this class is a read-only binding. */
    private GpuBuffer placeholder() {
        return placeholders.readOnlyFloats();
    }

    private View stagingView(GpuScratch.Staging which, long floats) {
        ScratchSlot s = scratch.staging(which, floats);
        return new View(s.buffer(), s.offset());
    }

    // ================================================================== validation

    /** The weight kernels of {@code type}; throws when GpuOps has none for it. */
    private WeightKernels family(GGMLType type) {
        return weightKernels(type, backend.backendType(), kernels.caps());
    }

    private WeightKernels family(GpuWeight w) {
        WeightKernels k = FAMILIES.get(w.type());
        if (k == null) {
            throw new GpuModelUnsupportedException("tensor " + w.name() + ": no GPU kernels for weight type "
                    + w.type());
        }
        return hwFamily(w.type(), k, backend.backendType(), kernels.caps());
    }

    /**
     * The weight kernels of {@code type} on a device: the generic family, or the {@code {T}_Hw_*} fast-path set
     * when the type has hardware-FP8 decode rows and the device reports the CUDA sm_89 conversions.
     *
     * <p>The selection depends only on the weight type, the backend type and the device capabilities, so it can be
     * checked without loading a kernel library.</p>
     *
     * @throws GpuModelUnsupportedException when GpuOps has no projection kernels for {@code type}
     */
    static WeightKernels weightKernels(GGMLType type, ComputeBackend.BackendType backendType,
                                       DeviceCapabilities caps) {
        WeightKernels k = FAMILIES.get(type);
        if (k == null) {
            throw new GpuModelUnsupportedException("no GPU projection kernels for weight type " + type);
        }
        return hwFamily(type, k, backendType, caps);
    }

    /** {@code generic}, or the {@code {T}_Hw_*} fast-path set when the device selects hardware FP8 decode. */
    private static WeightKernels hwFamily(GGMLType type, WeightKernels generic,
                                          ComputeBackend.BackendType backendType, DeviceCapabilities caps) {
        return HW_FP8_TYPES.contains(type) && hardwareFp8Decode(backendType, caps)
                ? hardware(familyName(type)) : generic;
    }

    /** The kernel-family name of {@code type}: the enum name, except the types that reuse another family's rows. */
    private static String familyName(GGMLType type) {
        return type == GGMLType.F8_E4M3FN ? GGMLType.F8_E4M3.name() : type.name();
    }

    private static int cols(GpuWeight w) {
        Objects.requireNonNull(w, "weight");
        if (w.cols() > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("tensor " + w.name() + ": " + w.cols() + " columns exceed int range");
        }
        return (int) w.cols();
    }

    private static int rows(GpuWeight w) {
        if (w.rows() > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("tensor " + w.name() + ": " + w.rows() + " rows exceed int range");
        }
        return (int) w.rows();
    }

    private static void checkTokens(int n) {
        if (n < 1) {
            throw new IllegalArgumentException("row count must be at least 1, got " + n);
        }
    }

    private static void checkPositive(int v, String what) {
        if (v < 1) {
            throw new IllegalArgumentException(what + " must be positive, got " + v);
        }
    }

    private static void checkInt(long v, String what) {
        if (v > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(what + " " + v + " exceeds the kernels' int range");
        }
    }

    private static void checkView(String what, GpuBuffer b, long off, long floats) {
        if (b == null) {
            throw new IllegalArgumentException(what + ": buffer is null");
        }
        if (off < 0) {
            throw new IllegalArgumentException(what + ": negative offset " + off);
        }
        if ((off + floats) * Float.BYTES > b.sizeInBytes()) {
            throw new IllegalArgumentException(what + ": elements [" + off + ", " + (off + floats) + ") exceed the "
                    + b.sizeInBytes() + "-byte buffer");
        }
    }

    private static boolean overlaps(GpuBuffer a, long aOff, long aN, GpuBuffer b, long bOff, long bN) {
        return a == b && aOff < bOff + bN && bOff < aOff + aN;
    }

    private static void checkSameOrDisjoint(GpuBuffer a, int aOff, GpuBuffer b, int bOff, long n, String what) {
        if (!(a == b && aOff == bOff) && overlaps(a, aOff, n, b, bOff, n)) {
            throw new IllegalArgumentException(what + ": input and output views overlap without coinciding");
        }
    }

    /** Rejects an input view that shares elements with the output view without coinciding with it. */
    private static void checkStridedAlias(GpuBuffer in, int inOff, int inStride, GpuBuffer out, int outOff,
                                          int outStride, int rows, int cols, String what) {
        if (in == out && inOff == outOff && (rows == 1 || inStride == outStride)) {
            return;
        }
        if (stridedOverlap(in, inOff, inStride, out, outOff, outStride, rows, cols)) {
            throw new IllegalArgumentException(what + " and output views overlap without coinciding");
        }
    }

    /**
     * Whether two views of {@code rows x cols} floats (row {@code r} at {@code off + r*stride}, {@code stride >= cols}
     * when {@code rows > 1}) may share an element. Exact for disjoint extents and for equal strides, where every row
     * period holds the first view's columns at {@code [0, cols)} and the second's at {@code [d, d + cols)} with
     * {@code d = (bOff - aOff) mod stride}; any other intersecting extents count as overlapping.
     */
    private static boolean stridedOverlap(GpuBuffer a, long aOff, int aStride, GpuBuffer b, long bOff, int bStride,
                                          int rows, int cols) {
        long aExt = (long) (rows - 1) * aStride + cols;
        long bExt = (long) (rows - 1) * bStride + cols;
        if (!overlaps(a, aOff, aExt, b, bOff, bExt)) {
            return false;
        }
        if (rows > 1 && aStride == bStride) {
            long d = Math.floorMod(bOff - aOff, (long) aStride);
            return !(d >= cols && d + cols <= aStride);
        }
        return true;
    }

    private static void checkSizes(String kernel, long[] global, long[] local) {
        checkLocal(kernel, local);
        Objects.requireNonNull(global, "global");
        if (global.length != local.length) {
            throw new IllegalArgumentException(kernel + ": global " + Arrays.toString(global) + " and local "
                    + Arrays.toString(local) + " must have the same 1-3 dimensions");
        }
        for (long g : global) {
            if (g <= 0) {
                throw new IllegalArgumentException(kernel + ": work sizes must be positive: global "
                        + Arrays.toString(global) + ", local " + Arrays.toString(local));
            }
        }
    }

    /** Checks a kernel name and a work-group size of one to three positive dimensions. */
    private static void checkLocal(String kernel, long[] local) {
        if (kernel == null || kernel.isEmpty()) {
            throw new IllegalArgumentException("kernel name must not be empty");
        }
        Objects.requireNonNull(local, "local (LLM dispatches always pass a local size)");
        if (local.length < 1 || local.length > 3) {
            throw new IllegalArgumentException(kernel + ": local " + Arrays.toString(local)
                    + " must have 1-3 dimensions");
        }
        for (long l : local) {
            if (l <= 0) {
                throw new IllegalArgumentException(kernel + ": work sizes must be positive: local "
                        + Arrays.toString(local));
            }
        }
    }

    private static long roundUp(long n, long m) {
        return (n + m - 1) / m * m;
    }

    static long ceilDiv(long n, long d) {
        return (n + d - 1) / d;
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("GpuOps is closed");
        }
    }

    // ================================================================== lifecycle

    /**
     * Waits for the queue, so no enqueued kernel still reads them, then closes every open prepared dispatch and
     * releases the sign buffers. The placeholders belong to the caller. Idempotent.
     */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        List<AutoCloseable> toClose = new ArrayList<>(prepared);
        toClose.addAll(signBuffers.values());
        closed = true;
        RuntimeException first = null;
        try {
            queue.finish();
        } catch (RuntimeException e) {
            first = e;
        }
        for (AutoCloseable c : toClose) {
            try {
                c.close();
            } catch (Exception e) {
                RuntimeException r = e instanceof RuntimeException ? (RuntimeException) e
                        : new IllegalStateException("closing " + c + " failed", e);
                if (first == null) {
                    first = r;
                } else {
                    first.addSuppressed(r);
                }
            }
        }
        prepared.clear();
        signBuffers.clear();
        signValues.clear();
        signsBySpec.clear();
        if (first != null) {
            throw first;
        }
    }

    @Override
    public String toString() {
        return "GpuOps{" + backend.backendType() + ", L=" + local + ", prepared=" + prepared.size() + '}';
    }
}

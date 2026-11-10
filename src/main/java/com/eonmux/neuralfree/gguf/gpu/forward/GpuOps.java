package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.GpuBackendException;
import com.eonmux.neuralfree.acceleration.GpuOutOfMemoryException;
import com.eonmux.neuralfree.acceleration.UnsupportedKernelException;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
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
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuPlaceholders;
import com.eonmux.neuralfree.gguf.ops.act.ActKernels;
import com.eonmux.neuralfree.gguf.ops.fused.FusedKernels;
import com.eonmux.neuralfree.gguf.ops.gemm.GemmKernels;
import com.eonmux.neuralfree.gguf.ops.gemm.MmqConfig;
import com.eonmux.neuralfree.gguf.ops.gemm.MmqMode;
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
 * bit-identical to the generic rows. {@link GpuWeightFamilies#hardwareFp8Decode} picks the {@code _Hw} set for a device that
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
 * projections read one activation row, or one group of token rows, opens a {@link SharedQ8} scope
 * ({@link #shareQ8Activations}), so the rows are quantized once for all of them instead of once per projection.
 * {@link Int8MatvecMode}
 * (property {@code neuralfree.matvec.int8}) selects it; the default {@link Int8MatvecMode#AUTO} uses it where the
 * device reports {@link DeviceFeature#INTEGER_DOT_PRODUCT}, the weight type measured faster with the lane on that
 * device kind ({@link Int8MatvecTuning}), and the kernel is available, and falls back to the exact F32 matvec elsewhere;
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

    /** Elements per Q8_1 activation block ({@link BlockLayout#QK8_1}); the integer-dot lane's quantization unit. */
    static final int Q8_1_BLOCK_SIZE = BlockLayout.QK8_1;
    /** Floats of one Q8_1 block ({@link BlockLayout#Q8_1_BYTES}) in the Q8_1 staging slot. */
    static final int Q8_1_FLOATS = BlockLayout.Q8_1_BYTES / Float.BYTES;
    /** The kernels this class launches itself on a 1-D grid of local size {@code L}, apart from the weight families. */
    static final List<String> HELPER_KERNELS =
            Collections.unmodifiableList(Arrays.asList(ROW_NORM, UNARY, GLU, COPY2D, AXPBY, HadamardSpec.KERNEL,
                    ACT_FAKE_QUANT, PERMUTE_COLS, SCALE_COLS, QUANTIZE_Q8_1, NormKernels.RESIDUAL_MIX_NORM,
                    NormKernels.ROW_NORM_Q8, NormKernels.RESIDUAL_MIX_NORM_Q8,
                    GemmKernels.QUANT_MMQ_INT8_STATIC,
                    GemmKernels.QUANT_MMQ_INT8_ROW, GemmKernels.QUANT_MMQ_INT8_GROUP128,
                    GemmKernels.QUANT_MMQ_INT8_ASYM, GemmKernels.QUANT_MMQ_FP8_STATIC,
                    GemmKernels.QUANT_MMQ_FP8_ROW, GemmKernels.QUANT_MMQ_FP8_GROUP128,
                    GemmKernels.QUANT_MMQ_FP8_MX, GemmKernels.QUANT_MMQ_FP4_MX, GemmKernels.QUANT_MMQ_FP4_NV));

    /** A float view (buffer, element offset). */
    static final class View {
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
    final FastGemmLanes lanes;
    /** The dense projections this class's matvec, matmul, gate+up and prewarm calls run. */
    private final GpuProjections projections;
    private final Set<PreparedDispatch> prepared = Collections.newSetFromMap(new IdentityHashMap<>());
    private final HadamardSignBuffers signs;
    private final Map<String, Boolean> optionalStatus = new HashMap<>();
    private final GpuPlaceholders placeholders;
    final Int8MatvecMode int8Mode;
    final FusedMode fusedMode;
    /** The Q8_1 activation blocks of the integer-dot lane: rows a norm quantized, and the quantizer's grid. */
    final Q8Activations q8;
    SharedQ8 activeShare;
    W12QView activeW12Q;
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
        this.signs = new HadamardSignBuffers(backend, queue);
        this.q8 = new Q8Activations(scratch, kernels.caps(), int8Mode);
        this.lanes = new FastGemmLanes(this, queue, kernels, scratch);
        this.projections = new GpuProjections(this, queue, kernels, scratch);
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
        return GpuWeightFamilies.supportsWeightType(type);
    }

    /** The {@code {T}_MatVec} kernel name of a weight type ({@code Float_MatVec} for F32); null when the type has
     *  no GPU kernels. The name is the family's, so the types that share a family share the kernel rows. */
    public static String matVecKernel(GGMLType type) {
        return GpuWeightFamilies.matVecKernel(type);
    }

    /** The single-token embedding-lookup kernel name of a weight type; null when the type has no GPU kernels. */
    public static String lookupKernel(GGMLType type) {
        return GpuWeightFamilies.lookupKernel(type);
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
        q8.invalidate();
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
        projections.matvec(w, in, inOff, out, outOff, nTokens);
    }

    /**
     * {@link #matvec}, using the multi-column matvec, fast GEMM or tiled GEMM lane the measured crossover of
     * {@link SmallBatchCrossover} picks for the weight's device kind, type and projection size (the tiled GEMM
     * from {@code TILED_MIN_TOKENS} rows where no measured entry serves) when the kernel can run its
     * work-group on this device. With batch invariance on, a fast GEMM lane serves every token count and
     * the tiled GEMM and the multi-column matvec are never taken.
     */
    public void matmul(GpuWeight w, GpuBuffer in, int inOff, GpuBuffer out, int outOff, int nTokens) {
        projections.matmul(w, in, inOff, out, outOff, nTokens);
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
        return projections.gateUpGluUsable(gate, up, op, params);
    }

    /**
     * Runs {@link #gateUpGluUsable}'s fused projection: {@code out[r] = silu(gate[r] . x) * (up[r] . x)} over the
     * single input row {@code x} (at element 0 of {@code in}), written from element 0 of {@code out}. Above the
     * two-row thread floor ({@link GpuProjections#TWO_ROW_MIN_THREADS} live matvec threads) the launch pairs the rows of both
     * weights in the {@code {T}_MatVec2FusedGateUp} form at {@code global = ceil(rows / 2) * L}; below it the
     * kernel runs one row per workgroup.
     *
     * @throws UnsupportedOperationException when the form cannot serve the weights, op or buffers
     */
    public void gateUpGlu(GpuWeight gate, GpuWeight up, GluOp op, ActParams params, GpuBuffer in, GpuBuffer out) {
        projections.gateUpGlu(gate, up, op, params, in, out);
    }

    /** {@code accum[accOff + t*rows + r] += scale * W[r] . in[inOff + t*cols ...]}. */
    public void matvecAdd(GpuWeight w, GpuBuffer in, int inOff, GpuBuffer accum, int accOff, int nTokens) {
        projections.matvecAdd(w, in, inOff, accum, accOff, nTokens);
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
        return projections.matvecFamilyServes(w, nTokens);
    }

    /**
     * Whether activation rows starting at element {@code off}, {@code cols} elements each, can back a
     * {@link #shareQ8Activations} scope: the quantizer reads whole rows from element 0 of their buffer, so rows
     * at a nonzero offset are staged by their projections instead.
     */
    public static boolean shareableQ8Activations(int off, int cols) {
        return off == 0 && cols > 0 && cols % Q8_1_BLOCK_SIZE == 0;
    }

    /**
     * Consecutive activation rows quantized to Q8_1 blocks once for the projections that read them
     * ({@link #shareQ8Activations}). A projection of the scope whose view is exactly the shared rows (element 0 of
     * their buffer, the same token count) and whose weight takes an integer-dot lane, single-token or multi-column,
     * reuses the blocks, which are the per-projection quantizer's output bit for bit; any other quantizes its own
     * rows. The caller keeps the rows unchanged until the scope closes. Closing enqueues nothing and is idempotent.
     */
    public final class SharedQ8 implements AutoCloseable {
        private final GpuBuffer source;
        private final int cols;
        private final int rows;
        private GpuBuffer blocks;
        private boolean closed;

        private SharedQ8(GpuBuffer source, int cols, int rows) {
            this.source = source;
            this.cols = cols;
            this.rows = rows;
            this.blocks = q8.take(source, rows, cols);
        }

        /** The Q8_1 blocks of the shared rows, quantizing them on first use. */
        GpuBuffer blocks() {
            if (closed) {
                throw new IllegalStateException("the Q8_1 activation share is closed");
            }
            if (blocks == null) {
                long n = (long) rows * (cols / Q8_1_BLOCK_SIZE);
                ScratchSlot slot = scratch.staging(GpuScratch.Staging.SHARED_Q8_1, n * Q8_1_FLOATS);
                quantizeQ8_1(source, Math.multiplyExact(rows, cols), slot.buffer());
                blocks = slot.buffer();
            }
            return blocks;
        }

        /** Whether {@code nTokens} rows of {@code width} elements at {@code x} are the shared rows. */
        boolean matches(View x, int width, int nTokens) {
            return !closed && width == cols && nTokens == rows && x.buf == source && x.off == 0;
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

    /** {@link #shareQ8Activations(GpuBuffer, int, int, int)} over one token row. */
    public SharedQ8 shareQ8Activations(GpuBuffer x, int off, int cols) {
        return shareQ8Activations(x, off, cols, 1);
    }

    /**
     * Opens a {@link SharedQ8} scope over the {@code rows} consecutive activation rows at {@code x[off ...]} of
     * {@code cols} elements each, so the projections of those rows made before it closes quantize them once. The
     * rows must start at element 0 of their buffer ({@link #shareableQ8Activations}) and the caller must not
     * write them inside the scope. One scope is open at a time.
     *
     * @throws IllegalArgumentException when the view is out of range or cannot be shared
     * @throws IllegalStateException    when a share is already open
     */
    public SharedQ8 shareQ8Activations(GpuBuffer x, int off, int cols, int rows) {
        ensureOpen();
        ensureNoShare();
        checkPositive(cols, "Q8_1 activation share width");
        checkPositive(rows, "Q8_1 activation share rows");
        checkView("Q8_1 activation share", x, off, Math.multiplyExact((long) rows, cols));
        if (!shareableQ8Activations(off, cols)) {
            throw new IllegalArgumentException("Q8_1 activation share needs rows at element 0 of their buffer of a"
                    + " multiple of " + Q8_1_BLOCK_SIZE + " elements, got offset " + off + " and width " + cols);
        }
        SharedQ8 share = new SharedQ8(x, cols, rows);
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
        projections.prewarmWeightKernels(type, matmul, lookup);
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
        q8.invalidate();
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
        GpuBuffer signVector = h.hasSigns() ? signs.of(h) : placeholder();
        long groups = n / h.block();
        launch(HadamardSpec.KERNEL, groups * local, nRows, width, h.block(), h.kernelSignMode(), h.permHd(),
                h.permNk(), h.permRep(), xOff, yOff, signVector, x, y);
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
        if (emitsQ8(NormKernels.ROW_NORM_Q8, nRows, width, outOff, wRows)) {
            GpuBuffer blocks = q8.target(nRows, width);
            launch(NormKernels.ROW_NORM_Q8, (long) nRows * local, NormKernels.withQ8Blocks(args, nRows, width, blocks));
            q8.emitted(out, nRows, width, blocks);
            return;
        }
        launch(ROW_NORM, (long) nRows * local, args);
    }

    /**
     * Whether a row norm writes its rows' Q8_1 blocks through {@code kernel}: {@link Q8Activations#emits}, the kernel
     * is ready, fusion is permitted, batch invariance is off and no {@link SharedQ8} scope holds the slot.
     */
    private boolean emitsQ8(String kernel, int nRows, int width, int outOff, int wRows) {
        return activeShare == null && q8.emits(nRows, width, outOff, wRows) && fusedMode.permitsFusion()
                && !BatchInvariance.enabled() && optionalReady(kernel, local);
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
        if (emitsQ8(NormKernels.RESIDUAL_MIX_NORM_Q8, nRows, width, outOff, wRows)) {
            GpuBuffer blocks = q8.target(nRows, width);
            launch(NormKernels.RESIDUAL_MIX_NORM_Q8, (long) nRows * local,
                    NormKernels.withQ8Blocks(args, nRows, width, blocks));
            q8.emitted(out, nRows, width, blocks);
            return true;
        }
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
            q8.invalidate();
            queue.enqueueFillBuffer(dst, byteOff, bytes, pattern);
        }
    }

    void copy2dUnchecked(int rows, int cols, GpuBuffer src, int srcOff, int srcStride, GpuBuffer dst,
                                 int dstOff, int dstStride) {
        launch(COPY2D, (long) rows * cols, rows, cols, srcStride, srcOff, dstStride, dstOff, src, dst);
    }

    /**
     * {@code out = alpha*a + beta*b} over {@code n} elements. A zero coefficient drops its operand (which may be
     * null). Operands that coincide with {@code out} are moved to the aliasable slot of {@value #AXPBY}; any other
     * overlap with {@code out} is rejected.
     */
    void axpby(int n, float alpha, GpuBuffer a, int aOff, float beta, GpuBuffer b, int bOff, GpuBuffer out,
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
        q8.invalidate();
        long g = roundUp(n, local);
        queue.enqueueKernelNoEvent(kernels.get(kernel), new long[]{g}, new long[]{local}, args);
    }

    /** 1-D launch at the matvec work-group size: {@code roundUp(n, Lm)} work-items of local size {@code Lm}. */
    void launchMatvec(String kernel, long n, Object... args) {
        q8.invalidate();
        long g = roundUp(n, matvecLocal);
        queue.enqueueKernelNoEvent(kernels.get(kernel), new long[]{g}, new long[]{matvecLocal}, args);
    }

    /**
     * Quantizes {@code x[0 .. elements)} ({@code elements} a multiple of {@value #Q8_1_BLOCK_SIZE}) to Q8_1 blocks
     * in {@code out} ({@value #QUANTIZE_Q8_1}, grid by {@link Q8Activations#quantizeWorkItems}).
     */
    void quantizeQ8_1(GpuBuffer x, int elements, GpuBuffer out) {
        launchMatvec(QUANTIZE_Q8_1, q8.quantizeWorkItems(elements / Q8_1_BLOCK_SIZE), elements, x, out);
    }

    /** 2-D launch: {@code {roundUp(gx, lx), gy}} with local {@code {lx, 1}}. */
    void launch2d(String kernel, long gx, long gy, int lx, Object... args) {
        q8.invalidate();
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
    WeightKernels family(GGMLType type) {
        return GpuWeightFamilies.weightKernels(type, backend.backendType(), kernels.caps());
    }

    WeightKernels family(GpuWeight w) {
        return GpuWeightFamilies.family(w, backend.backendType(), kernels.caps());
    }

    static int cols(GpuWeight w) {
        Objects.requireNonNull(w, "weight");
        if (w.cols() > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("tensor " + w.name() + ": " + w.cols() + " columns exceed int range");
        }
        return (int) w.cols();
    }

    static int rows(GpuWeight w) {
        if (w.rows() > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("tensor " + w.name() + ": " + w.rows() + " rows exceed int range");
        }
        return (int) w.rows();
    }

    static void checkTokens(int n) {
        if (n < 1) {
            throw new IllegalArgumentException("row count must be at least 1, got " + n);
        }
    }

    private static void checkPositive(int v, String what) {
        if (v < 1) {
            throw new IllegalArgumentException(what + " must be positive, got " + v);
        }
    }

    static void checkInt(long v, String what) {
        if (v > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(what + " " + v + " exceeds the kernels' int range");
        }
    }

    static void checkView(String what, GpuBuffer b, long off, long floats) {
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

    static boolean overlaps(GpuBuffer a, long aOff, long aN, GpuBuffer b, long bOff, long bN) {
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

    void ensureOpen() {
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
        toClose.addAll(signs.release());
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
        if (first != null) {
            throw first;
        }
    }

    @Override
    public String toString() {
        return "GpuOps{" + backend.backendType() + ", L=" + local + ", prepared=" + prepared.size() + '}';
    }
}

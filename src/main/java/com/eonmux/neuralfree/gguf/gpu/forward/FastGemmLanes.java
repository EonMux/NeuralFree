package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.DeviceFeature;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.BatchInvariance;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.ops.gemm.CmGemmPlan;
import com.eonmux.neuralfree.gguf.ops.gemm.GemmKernels;
import com.eonmux.neuralfree.gguf.ops.gemm.MmaGemmPlan;
import com.eonmux.neuralfree.gguf.ops.gemm.MmqConfig;
import com.eonmux.neuralfree.gguf.ops.gemm.MmqGemmPlan;
import com.eonmux.neuralfree.gguf.ops.gemm.MmqMode;
import com.eonmux.neuralfree.gguf.ops.gemm.SplitKGemmPlan;
import com.eonmux.neuralfree.gguf.ops.gemm.StreamKGemmPlan;
import com.eonmux.neuralfree.gguf.quant.ActQuantScheme;
import com.eonmux.neuralfree.gguf.quant.InputTransform;

/**
 * Fast-GEMM lane dispatch of one {@link GpuOps}: the lane a weight type takes, whether the lane can run a
 * launch, the staged activation records of the int8 MMQ lane (the dynamic per-chunk codes, or the checkpoint
 * activation scheme's own int8 codes), and the split-K planning and launches. The
 * lanes are the fast quantized-weight GEMM ({@code <T>_MatMul_MMQ}, in the fp16-tile cooperative-matrix
 * form {@code <T>_MatMul_CM} on Vulkan devices with cooperative matrices) and the tensor-core MMA GEMM
 * ({@code <T>_MatMul_MMA}); every lane splits its tall-skinny launches into the {@code <T>_MatMul_*_KSplit}
 * twin plus the shared {@code LLM_SplitK_Reduce}, and a staged launch of fewer tokens than the device tile
 * runs the twin tile of its ladder row ({@code <T>_MatMul_MMQ_S} at up to 16 tokens,
 * {@code <T>_MatMul_MMQ_S64} at 17..64 — on a small-K reduction only from 49, see
 * {@link MmqConfig#twinTokens(int, int)} — and the W4A8 forms likewise, {@link #smallTileKernel})
 * when the device has it.
 *
 * <p>A lane kernel that the backend does not provide, or that cannot run the 256-work-item tile on the
 * device, is never launched ({@link GpuOps#optionalReady}) and the next slower form serves the product.
 * The split-K heuristic ({@link SplitKGemmPlan#of}) follows the lane: the staged MMQ plans carry the
 * device's {@link MmqConfig}, the CM plans the {@link CmGemmPlan} tiles and the MMA plans the
 * {@link MmaGemmPlan} tiles. Not thread-safe; owned by its GpuOps.
 */
final class FastGemmLanes {

    /** Smallest CUDA compute capability with the E4M3 tensor-core multiply ({@code sm_89}: {@code mma e4m3}). */
    static final int FP8_MIN_COMPUTE_CAPABILITY = 89;

    private final GpuOps ops;
    private final ComputeQueue queue;
    private final LlmKernels kernels;
    private final ComputeBackend backend;
    private final GpuScratch scratch;
    private final int local;
    private final long[] maxGroups;
    private final long maxWorkGroupSize;
    private final MmqMode mmqMode;
    private final MmqConfig mmqConfig;

    /** Binds the lanes to the queue, kernels and scratch of the owning GpuOps. */
    FastGemmLanes(GpuOps ops, ComputeQueue queue, LlmKernels kernels, GpuScratch scratch) {
        this.ops = ops;
        this.queue = queue;
        this.kernels = kernels;
        this.backend = kernels.backend();
        this.scratch = scratch;
        this.local = kernels.matvecLocal();
        this.maxGroups = kernels.caps().getMaxWorkGroupCount();
        this.maxWorkGroupSize = kernels.caps().getMaxWorkGroupSize();
        this.mmqMode = MmqMode.configured();
        this.mmqConfig = MmqConfig.forDevice(backend.backendType(), kernels.caps().getComputeCapability());
    }

    /**
     * The fast GEMM kernel of a weight type, or null when the type has none for this device and mode. The llama.cpp MMQ
     * types and the int8 checkpoint type I8_B128 take the fast quantized-weight lane unless {@link MmqMode#OFF}: the
     * fp16-tile CM form under {@link MmqMode#AUTO} on Vulkan devices with cooperative matrices, the int8 MMQ GEMM
     * everywhere else and whenever {@link MmqMode#ON} forces it. F16 takes the tensor-core MMA GEMM on CUDA sm_80+ and Vulkan cooperative-matrix
     * devices; BF16 only on CUDA sm_80+, whose {@code mma} has a bfloat16 form &mdash; the Vulkan module of
     * {@code BF16_MatMul_MMA} stages fp16 tiles (a native bf16 cooperative-matrix twin needs
     * {@code VK_KHR_shader_bfloat16}, whose GLSL module the build's shader compiler cannot produce yet, although the
     * devices report {@link DeviceFeature#BF16_COOPERATIVE_MATRIX}), whose range is narrower than bf16, so Vulkan keeps
     * the exact tiled GEMM for BF16.
     */
    String fastGemm(GGMLType type) {
        if (GemmKernels.isFp8(type)) {
            /* The E4M3 datapath of <T>_MatMul_MMQ exists on CUDA sm_89+ only; elsewhere, and for a scheme the
             * lane does not stage, the exact tiled GEMM serves the product. */
            return fp8Tensor() && mmqMode.fastQuantGemm() ? GemmKernels.matmulMmq(type) : null;
        }
        if (GemmKernels.hasMmq(type)) {
            if (!mmqMode.fastQuantGemm()) {
                return null;
            }
            return mmqMode.prefersCm() && cmForm() && GemmKernels.hasCm(type)
                    ? GemmKernels.matmulCm(type) : GemmKernels.matmulMmq(type);
        }
        if (GemmKernels.hasMma(type) && tensorCores()
                && (type != GGMLType.BF16 || isBf16MmaBackend())) {
            return GemmKernels.matmulMma(type);
        }
        return null;
    }

    /**
     * The fast GEMM lane of a projection that also sees the weight's activation scheme: the E4M3 lanes when the
     * scheme is staged in the E4M3 code form and the device has FP8 tensor cores (the FP8 weight types over their
     * own codes, the 4-bit weight types over codes mapped to E4M3), otherwise the lane of the type alone
     * ({@link #fastGemm(GGMLType)}). A FP8 weight type without a staged E4M3 scheme keeps the exact tiled GEMM,
     * because its codes are E4M3 and no int8 datapath may read them.
     */
    String fastGemm(GpuWeight w, int cols) {
        ActQuantScheme scheme = stageableScheme(w);
        if (GemmKernels.isFp8(w.type())) {
            return scheme != null && fp8Tensor() && mmqMode.fastQuantGemm() && fp8StagingReady(scheme)
                    ? GemmKernels.matmulMmq(w.type()) : null;
        }
        if (GemmKernels.isW4a8(w.type()) && scheme != null && fp8Tensor() && mmqMode.fastQuantGemm()
                && fp8StagingReady(scheme)) {
            return GemmKernels.matmulW4a8(w.type());
        }
        return fastGemm(w.type());
    }

    /**
     * The activation scheme of a weight whose projection can be staged from the schedule parameters of expert 0,
     * or null when the weight has no activation quantization, one entry per expert, or a column permutation (the
     * permutation runs after the quantization, and staged codes are not permutable).
     */
    private ActQuantScheme stageableScheme(GpuWeight w) {
        InputTransform t = w.inputTransform();
        if (t == null || t.hasPerm()) {
            return null;
        }
        InputTransform.ActQuant a = t.actQuant();
        return a == null || a.entries() != 1 ? null : a.scheme();
    }

    /** Whether the E4M3 staging kernel of {@code scheme} is named and runs at the row-reduce work-group size. */
    private boolean fp8StagingReady(ActQuantScheme scheme) {
        String kernel = GemmKernels.fp8ActivationStaging(scheme);
        return kernel != null && ops.optionalReady(kernel, local);
    }

    /**
     * The FP8 tensor-core support of the device: CUDA compute capability 8.9 and above
     * ({@code mma.sync.aligned.m16n8k32...e4m3.e4m3.f32}). Other backends keep the exact fallbacks.
     */
    private boolean fp8Tensor() {
        return fp8Tensor(backend.backendType(), kernels.caps().getComputeCapability());
    }

    /** Whether a device of {@code type} and {@code computeCapability} has the E4M3 tensor-core multiply. */
    static boolean fp8Tensor(ComputeBackend.BackendType type, int computeCapability) {
        return type == ComputeBackend.BackendType.CUDA && computeCapability >= FP8_MIN_COMPUTE_CAPABILITY;
    }

    /**
     * The direct E4M3 staging plan of a projection, or null when the projection keeps the fake-quantization path:
     * no activation quantization, a scheme without an E4M3 staging kernel, per-expert parameters, a column
     * permutation, or a staging kernel this backend cannot run at its work-group size.
     */
    Fp8ActStaging fp8Staging(GpuWeight w, int cols) {
        if (!GemmKernels.isFp8(w.type()) && !GemmKernels.isW4a8(w.type())) {
            return null;
        }
        InputTransform t = w.inputTransform();
        if (t == null || t.hasPerm()) {
            return null;
        }
        InputTransform.ActQuant a = t.actQuant();
        if (a == null || a.entries() != 1) {
            return null;
        }
        ActQuantScheme scheme = a.scheme();
        String kernel = GemmKernels.fp8ActivationStaging(scheme);
        if (kernel == null || !fp8StagingReady(scheme)) {
            return null;
        }
        scheme.runLength(cols);
        float argument;
        if (scheme == ActQuantScheme.FP8_TOKEN) {
            argument = a.scale(0);
        } else if (scheme.needsGlobalScale()) {
            argument = a.globalScale(0);
        } else if (scheme.isStatic()) {
            argument = a.scale(0);
        } else {
            argument = 0.0f;
        }
        boolean readsArgument = scheme.isStatic() || scheme == ActQuantScheme.FP8_TOKEN
                || scheme.needsGlobalScale();
        return new Fp8ActStaging(scheme, kernel, argument, readsArgument, fp8Kind(scheme));
    }

    private static Fp8ActStaging.Kind fp8Kind(ActQuantScheme scheme) {
        switch (scheme) {
            case FP8_TOKEN:
                return Fp8ActStaging.Kind.ROW;
            case FP8_GROUP128:
                return Fp8ActStaging.Kind.RUN128;
            case NVFP4:
                return Fp8ActStaging.Kind.CHUNK_GROUP;
            default:
                return Fp8ActStaging.Kind.CHUNK;
        }
    }

    /**
     * The staged activations of an E4M3 lane: every token row of the projection staged once into the record layout
     * of the staged GEMMs (tk/4 code ints, the chunk scale, the code sums, the zero point), shared by all row
     * tiles. The row, per-run and per-chunk work-group forms launch one workgroup per run at the row-reduce size.
     */
    ScratchSlot stageFp8(Fp8ActStaging staging, GpuBuffer x, int xOff, int nTokens, int cols) {
        int tk = mmqConfig.stageDepth();
        long records = (long) nTokens * mmqConfig.rowRecords(cols);
        ScratchSlot slot = scratch.staging(GpuScratch.Staging.Q8_1,
                Math.multiplyExact(records, mmqConfig.recordUnits()));
        long units;
        switch (staging.kind()) {
            case ROW:
                units = Math.multiplyExact((long) nTokens, local);
                break;
            case RUN128:
                units = Math.multiplyExact(Math.multiplyExact((long) nTokens, cols / 128), local);
                break;
            case CHUNK_GROUP:
                units = Math.multiplyExact(records, local);
                break;
            case CHUNK:
                units = records;
                break;
            default:
                throw new IllegalStateException(staging.scheme() + " has no E4M3 staging geometry");
        }
        ops.launch(staging.kernel(), units, stageFp8Args(staging, x, xOff, nTokens, cols, tk, slot.buffer()));
        return slot;
    }

    /** The argument list of an E4M3 staging launch: the shape, the scheme's scalar when it reads one, the input
     *  and the staged records. */
    private static Object[] stageFp8Args(Fp8ActStaging staging, GpuBuffer x, int xOff, int nTokens, int cols, int tk,
                                         GpuBuffer q8) {
        if (staging.readsArgument()) {
            return new Object[]{nTokens, cols, tk, cols, xOff, staging.argument(), x, q8};
        }
        return new Object[]{nTokens, cols, tk, cols, xOff, x, q8};
    }

    /**
     * Tokens per workgroup tile of a fast GEMM kernel: the small-tile twins' tile width
     * ({@code NF_MMQ_TN} of the {@code _S} / {@code _S64} forms), the device's {@link MmqConfig}
     * token tile for the int8 MMQ GEMM ({@code NF_MMQ_TN}), {@code NF_MMA_TN} for the MMA and CM
     * forms.
     */
    int fastTileTokens(String fastKernel) {
        int twin = twinTokensOf(fastKernel);
        if (twin > 0) {
            return twin;
        }
        return stagedFast(fastKernel) ? mmqConfig.tileTokens() : MmaGemmPlan.TILE_TOKENS;
    }

    /** Weight rows per workgroup tile of a fast GEMM kernel: the small tiles' row tile
     *  ({@link MmqConfig#smallTile}, the 256-item twin) for the twins, the
     *  {@link MmqConfig} row tile or {@code NF_MMA_TM}. */
    int fastTileRows(String fastKernel) {
        int twin = twinTokensOf(fastKernel);
        if (twin > 0) {
            return MmqConfig.smallTile(mmqConfig, twin).tileRows();
        }
        return stagedFast(fastKernel) ? mmqConfig.tileRows() : MmaGemmPlan.TILE_ROWS;
    }

    /** Work-items per workgroup of a fast GEMM kernel: the small tiles' 256-item group for the
     *  twins, the {@link MmqConfig} work-group size or {@code NF_MMA_WG}. */
    int fastWorkGroup(String fastKernel) {
        int twin = twinTokensOf(fastKernel);
        if (twin > 0) {
            return MmqConfig.smallTile(mmqConfig, twin).workGroupSize();
        }
        return stagedFast(fastKernel) ? mmqConfig.workGroupSize() : MmaGemmPlan.WORKGROUP_SIZE;
    }

    /** Whether a fast kernel reads the staged activation records of the MMQ and W4A8 lanes, the
     *  plain tiles and the small-token-tile twins alike. */
    static boolean stagedFast(String fastKernel) {
        return fastKernel.endsWith(GemmKernels.MATMUL_MMQ_SUFFIX)
                || fastKernel.endsWith(GemmKernels.MATMUL_W4A8_SUFFIX)
                || smallTileFast(fastKernel);
    }

    /** Whether a fast kernel is a small-token-tile twin ({@code <T>_MatMul_MMQ_S} / {@code _S64}
     *  or the W4A8 forms); a device tile of 16 tokens has no twin. */
    static boolean smallTileFast(String fastKernel) {
        return twinTokensOf(fastKernel) > 0;
    }

    /** Token tile width of a fast kernel's small-token-tile twin form: 16 ({@code _S}), 64
     *  ({@code _S64}), or 0 when the kernel is not a twin. */
    private static int twinTokensOf(String fastKernel) {
        if (fastKernel.endsWith(GemmKernels.MATMUL_MMQ_S64_SUFFIX)
                || fastKernel.endsWith(GemmKernels.MATMUL_W4A8_S64_SUFFIX)) {
            return MmqConfig.TWIN_MAX_TOKENS;
        }
        if (fastKernel.endsWith(GemmKernels.MATMUL_MMQ_S_SUFFIX)
                || fastKernel.endsWith(GemmKernels.MATMUL_W4A8_S_SUFFIX)) {
            return MmqConfig.TOKEN_LANES;
        }
        return 0;
    }

    /**
     * Whether a fast kernel of a weight is an E4M3 lane, whose staged records are the E4M3 codes of
     * {@link #fp8Staging}: the {@code <T>_MatMul_MMQ} of an FP8 weight type and the {@code <T>_MatMul_W4A8} of a
     * 4-bit one, with their small-token-tile twins. The int8 MMQ GEMM of a 4-bit weight reads int8 records, so an
     * E4M3 scheme on it keeps the fake-quantization path.
     */
    static boolean fp8Fast(GpuWeight w, String fastKernel) {
        if (GemmKernels.isFp8(w.type())) {
            return fastKernel.endsWith(GemmKernels.MATMUL_MMQ_SUFFIX)
                    || fastKernel.endsWith(GemmKernels.MATMUL_MMQ_S_SUFFIX)
                    || fastKernel.endsWith(GemmKernels.MATMUL_MMQ_S64_SUFFIX);
        }
        return GemmKernels.isW4a8(w.type()) && (fastKernel.endsWith(GemmKernels.MATMUL_W4A8_SUFFIX)
                || fastKernel.endsWith(GemmKernels.MATMUL_W4A8_S_SUFFIX)
                || fastKernel.endsWith(GemmKernels.MATMUL_W4A8_S64_SUFFIX));
    }

    /**
     * The small-token-tile twin of a staged fast lane for a launch of {@code nTokens}, or null when the
     * launch keeps {@code fastKernel}: only the MMQ and W4A8 lanes of a device whose tile is wider than
     * 16 tokens have twins, and the launch takes the twin of its ladder row
     * ({@link MmqConfig#twinTokens(int, int)}, the tile llama.cpp picks per launch,
     * ggml_cuda_mmq_get_J_best) &mdash;
     * the 16-token twin at up to 16 tokens and, on a small-K reduction, up to 48, the 64-token twin
     * over the rest of 17..64, and the plain tile above 64,
     * where the measured 128-token tile has no dead token columns left worth a twin (n=96 head: 3191.6us
     * plain vs 4124.5us on the 64-token twin; n=48: 2923.1 vs 2063.0, the twin wins). A device tile of
     * 64 tokens serves 17..64 with the plain tile itself, never under {@link BatchInvariance} (the tile
     * is part of the lane a batch-invariant run pins), and only when the twin can run the tile's
     * work-group size on this device ({@link GpuOps#optionalReady}; a CUDA segment without the twin's
     * function, below sm_80, records it unavailable).
     */
    String smallTileKernel(String fastKernel, GpuWeight w, int nTokens) {
        if (!stagedFast(fastKernel) || BatchInvariance.enabled()
                || mmqConfig.tileTokens() <= MmqConfig.TOKEN_LANES) {
            return null;
        }
        int tokens = MmqConfig.twinTokens(nTokens, (int) w.cols());
        if (tokens >= mmqConfig.tileTokens() || (tokens > MmqConfig.TOKEN_LANES && nTokens > tokens)) {
            return null;
        }
        String twin = fastKernel.endsWith(GemmKernels.MATMUL_W4A8_SUFFIX)
                ? GemmKernels.matmulW4a8Small(w.type(), tokens)
                : GemmKernels.matmulMmqSmall(w.type(), tokens);
        return ops.optionalReady(twin, MmqConfig.smallTile(mmqConfig, tokens).workGroupSize(), 1) ? twin : null;
    }

    /**
     * The subgroup-row form of a plain MMQ launch ({@link SubgroupMmqGemm}: {@code {T}_MatMulSg_MMQ}, same arguments),
     * or null when the launch keeps {@code fastKernel}: only the {@code {T}_MatMul_MMQ} lane of an OpenCL program built
     * with the subgroup forms, for a weight type and row length the kernel serves, over the plain 16-deep records
     * {@code LLM_Quant_MMQ_Q8_1} stages (no checkpoint activation scheme, {@code int8Staging} null), never under
     * {@link BatchInvariance} (the lane's float order is part of what a batch-invariant run pins), and only when the
     * tile's group counts fit the device and the kernel can run its work-group ({@link GpuOps#optionalReady}).
     */
    String subgroupMmqKernel(String fastKernel, GpuWeight w, int cols, int nTokens, Int8ActStaging int8Staging) {
        if (!fastKernel.endsWith(GemmKernels.MATMUL_MMQ_SUFFIX) || int8Staging != null || BatchInvariance.enabled()
                || !SubgroupMmqGemm.programBuilds(kernels) || !SubgroupMmqGemm.shapeServes(w.type(), cols)
                || mmqConfig.stageDepth() != SubgroupMmqGemm.RECORD_DEPTH) {
            return null;
        }
        String name = SubgroupMmqGemm.kernel(w.type());
        return fastUsable(name, SubgroupMmqGemm.TILE_ROWS, SubgroupMmqGemm.TILE_TOKENS, SubgroupMmqGemm.WORK_GROUP, w,
                nTokens) ? name : null;
    }

    /** Staged record units between activation rows of the int8 MMQ lane ({@code NF_MMQ_TK} chunks per row). */
    int stagedRowRecords(int cols) {
        return mmqConfig.rowRecords(cols);
    }

    /** Whether a fast GEMM kernel is an MMA-family kernel: the tensor-core or split-K form. */
    static boolean mmaFast(String fastKernel) {
        return fastKernel.endsWith(GemmKernels.MATMUL_MMA_SUFFIX)
                || fastKernel.endsWith(GemmKernels.MATMUL_MMA_KSPLIT_SUFFIX);
    }

    /**
     * Tensor-core support of the backend for the MMA GEMM: {@link DeviceFeature#COOPERATIVE_MATRIX} on CUDA
     * (sm_80+, {@code mma.m16n8k16} fragments), HIP (WMMA on RDNA3+, MFMA on CDNA), or Vulkan (fp16 matrices,
     * fp32 accumulator, full subgroups). The OpenCL kernel has no tensor-core path, so OpenCL keeps the tiled
     * GEMM.
     */
    private boolean tensorCores() {
        ComputeBackend.BackendType type = backend.backendType();
        return (type == ComputeBackend.BackendType.CUDA || type == ComputeBackend.BackendType.VULKAN
                || type == ComputeBackend.BackendType.HIP)
                && kernels.caps().hasFeature(DeviceFeature.COOPERATIVE_MATRIX);
    }

    /**
     * Whether the backend's MMA kernels run bfloat16 fragments natively: CUDA sm_80+ ({@code mma} bf16 form)
     * and HIP (MFMA bf16 on CDNA, WMMA bf16 on RDNA3+; the device only reports
     * {@link DeviceFeature#BF16_COOPERATIVE_MATRIX} there). Vulkan keeps the exact tiled GEMM for BF16.
     */
    private boolean isBf16MmaBackend() {
        ComputeBackend.BackendType type = backend.backendType();
        return (type == ComputeBackend.BackendType.CUDA || type == ComputeBackend.BackendType.HIP)
                && kernels.caps().hasFeature(DeviceFeature.BF16_COOPERATIVE_MATRIX);
    }

    /**
     * Cooperative-matrix support of the Vulkan backend: there the fast quantized-weight GEMM runs the fp16-tile CM form
     * ({@code <T>_MatMul_CM}) instead of the int8 MMQ GEMM, as llama.cpp's mul_mm_cm2 pipelines do.
     */
    private boolean cmForm() {
        return backend.backendType() == ComputeBackend.BackendType.VULKAN
                && kernels.caps().hasFeature(DeviceFeature.COOPERATIVE_MATRIX);
    }

    /**
     * Whether a fast GEMM kernel can run this launch: the work-group size and group counts of its tile
     * ({@code tiledUsable} with the fast tile) and {@link GpuOps#optionalReady}.
     */
    boolean fastUsable(String fastKernel, int tileRows, int tileTokens, int workGroup, GpuWeight w, int nTokens) {
        if (maxWorkGroupSize < workGroup || GpuOps.ceilDiv(nTokens, tileTokens) > maxGroups[1]) {
            return false;
        }
        for (int s = 0; s < w.shardCount(); s++) {
            if (GpuOps.ceilDiv(w.shardRows(s), tileRows) > maxGroups[0]) {
                return false;
            }
        }
        return ops.optionalReady(fastKernel, workGroup, 1);
    }

    /**
     * The split-K dispatch of one shard of a fast launch, or null when this launch keeps the plain kernel: the
     * heuristic ({@link SplitKGemmPlan#of}) splits only tall-skinny grids, only when both kernels of the pair can
     * run their work-group sizes on this device, and never on a device that does not report its compute units
     * ({@link SplitKGemmPlan#splitsFor} keeps one slice for
     * {@link com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities#UNKNOWN_COMPUTE_UNITS}). A launch of
     * fewer tokens than the device tile ({@code smallTile}, the {@code _S} twin's regime) splits by the wave-fill
     * heuristic ({@link SplitKGemmPlan#ofSmallTiles}) instead, whose partial grid gives every compute unit several
     * blocks. The family follows the lane: {@link SplitKGemmPlan.Family#MMA} for F16/BF16,
     * {@link SplitKGemmPlan.Family#MMQ} for the staged int8 GEMM and {@link SplitKGemmPlan.Family#CM} for the
     * quantized-weight cooperative-matrix form.
     *
     * @param xOff    offset of the activation view the launch reads, in floats or, for the staged MMQ lane, in staged
     *                record units; the plan carries it into the partial kernel's arguments
     * @param yStride element stride between output rows
     */
    SplitKGemmPlan splitK(String fastKernel, GpuWeight w, int m, int cols, int nTokens, int yOff, int xOff,
                          int yStride, boolean smallTile) {
        int computeUnits = kernels.caps().getComputeUnits();
        SplitKGemmPlan plan;
        if (fastKernel.endsWith(GemmKernels.MATMUL_MMQ_SUFFIX)) {
            MmqGemmPlan base = MmqGemmPlan.of(mmqConfig, w.type(), m, cols, nTokens,
                    mmqConfig.rowRecords(cols), xOff, yStride, yOff);
            plan = smallTile ? SplitKGemmPlan.ofSmallTiles(base, computeUnits, local)
                    : SplitKGemmPlan.of(base, computeUnits, local);
        } else if (fastKernel.endsWith(GemmKernels.MATMUL_W4A8_SUFFIX)) {
            plan = smallTile
                    ? SplitKGemmPlan.ofW4a8SmallTiles(w.type(), m, cols, nTokens, mmqConfig.rowRecords(cols), xOff,
                            mmqConfig, computeUnits, local)
                    : SplitKGemmPlan.ofW4a8(w.type(), m, cols, nTokens, mmqConfig.rowRecords(cols), xOff,
                            mmqConfig, computeUnits, local);
        } else if (fastKernel.endsWith(GemmKernels.MATMUL_CM_SUFFIX)) {
            plan = SplitKGemmPlan.of(CmGemmPlan.of(w.type(), m, cols, nTokens, cols, xOff, yStride, yOff),
                    computeUnits, local);
        } else if (mmaFast(fastKernel)) {
            plan = SplitKGemmPlan.of(MmaGemmPlan.of(w.type(), m, cols, nTokens, cols, xOff, yStride, yOff),
                    computeUnits, local);
        } else {
            return null;
        }
        if (plan == null || plan.checkDevice(kernels.caps()) != null
                || !ops.optionalReady(plan.kernelName(), plan.workGroupSize(), 1)
                || !ops.optionalReady(GemmKernels.SPLIT_K_REDUCE, local)) {
            return null;
        }
        return plan;
    }

    /**
     * One split-K launch: the shard's k-slice planes into their own staging slot (offset 0, never a view of the
     * projection's input or output), then the reduce into the output. The activation buffer is the projection input
     * (MMA and CM), or the staged records of {@link GemmKernels#QUANT_MMQ_Q8_1} (MMQ), at the offset
     * {@link SplitKGemmPlan#xOff()} carries.
     */
    void splitKLaunch(SplitKGemmPlan plan, GpuBuffer shard, GpuBuffer x, GpuBuffer out, int yOff, int yStride) {
        ScratchSlot partial = scratch.staging(GpuScratch.Staging.SPLIT_K, plan.partialElements());
        queue.enqueueKernelNoEvent(kernels.get(plan.kernelName()), plan.partialGlobalWorkSize(),
                plan.partialLocalWorkSize(), plan.partialArgs(shard, x, partial.buffer()));
        queue.enqueueKernelNoEvent(kernels.get(GemmKernels.SPLIT_K_REDUCE), plan.reduceGlobalWorkSize(),
                plan.reduceLocalWorkSize(), plan.reduceArgs(partial.buffer(), out, yStride, yOff));
    }

    /**
     * The stream-k dispatch of one shard of a plain-tile int8 MMQ launch, or null when it keeps the plain grid
     * ({@link StreamKGemmPlan#of}: the CUDA tensor tile whose waves would run part-empty), or when this device
     * lacks either kernel (the stream-k kernels exist only on the CUDA sm_80+ segments of the tensor tile).
     * The caller keeps the plain grid under batch invariance and for the small-token twins.
     *
     * @param qOff    staged record offset of the shard's activation view
     * @param yStride element stride between output rows
     */
    StreamKGemmPlan streamK(String fastKernel, GpuWeight w, int m, int cols, int nTokens, int yOff, int qOff,
                            int yStride) {
        if (!fastKernel.endsWith(GemmKernels.MATMUL_MMQ_SUFFIX) || !GemmKernels.isInt8Mmq(w.type())) {
            return null;
        }
        MmqGemmPlan base = MmqGemmPlan.of(mmqConfig, w.type(), m, cols, nTokens, mmqConfig.rowRecords(cols), qOff,
                yStride, yOff);
        StreamKGemmPlan plan = StreamKGemmPlan.of(base, kernels.caps().getComputeUnits());
        int workGroup = mmqConfig.workGroupSize();
        if (plan == null || plan.checkDevice(kernels.caps()) != null
                || !ops.optionalReady(plan.kernelName(), workGroup, 1)
                || !ops.optionalReady(GemmKernels.STREAM_K_FIXUP, workGroup, 1)) {
            return null;
        }
        return plan;
    }

    /**
     * One stream-k launch: the shard's tiles into the output, the planes of the tiles it splits into their own
     * staging slot (never a view of the projection's input or output), then the fixup that adds them.
     */
    void streamKLaunch(StreamKGemmPlan plan, GpuBuffer shard, GpuBuffer q8, GpuBuffer out) {
        ScratchSlot fixup = scratch.staging(GpuScratch.Staging.SPLIT_K, plan.fixupElements());
        queue.enqueueKernelNoEvent(kernels.get(plan.kernelName()), plan.globalWorkSize(), plan.localWorkSize(),
                plan.args(shard, q8, out, fixup.buffer()));
        if (plan.fixupNeeded()) {
            queue.enqueueKernelNoEvent(kernels.get(GemmKernels.STREAM_K_FIXUP), plan.fixupGlobalWorkSize(),
                    plan.localWorkSize(), plan.fixupArgs(fixup.buffer(), out));
        }
    }

    /**
     * The staged activations of the int8 MMQ lane: one {@link GemmKernels#QUANT_MMQ_Q8_1} launch over the whole
     * projection input, into its own staging slot (never a view of the input), shared by all row tiles of the
     * projection (llama.cpp quantize_mmq_q8_1).
     */
    ScratchSlot stageQ8(GpuBuffer x, int xOff, int nTokens, int cols) {
        int tk = mmqConfig.stageDepth();
        long records = (long) nTokens * mmqConfig.rowRecords(cols);
        ScratchSlot slot = scratch.staging(GpuScratch.Staging.Q8_1, Math.multiplyExact(records, mmqConfig.recordUnits()));
        ops.launch(GemmKernels.QUANT_MMQ_Q8_1, records, nTokens, cols, tk, cols, xOff, x, slot.buffer());
        return slot;
    }

    /**
     * The direct int8 staging plan of a projection through the int8 MMQ lane, or null when the weight keeps the
     * fake-quantization path: no activation quantization, a scheme other than the int8 ones
     * ({@link GemmKernels#int8ActivationStaging}), per-expert parameters (the projection supplies expert 0), a
     * column permutation (the permutation runs after the quantization), or a staging kernel this backend cannot
     * run at the row-reduce work-group size.
     */
    Int8ActStaging int8Staging(GpuWeight w, int cols) {
        InputTransform t = w.inputTransform();
        if (t == null || t.hasPerm()) {
            return null;
        }
        InputTransform.ActQuant a = t.actQuant();
        if (a == null || a.entries() != 1) {
            return null;
        }
        ActQuantScheme scheme = a.scheme();
        String kernel = GemmKernels.int8ActivationStaging(scheme);
        if (kernel == null || !ops.optionalReady(kernel, local)) {
            return null;
        }
        scheme.runLength(cols);
        return new Int8ActStaging(scheme, kernel, a.scale(0), a.zeroPoint(0));
    }

    /**
     * The staged activations of the int8 MMQ lane with the checkpoint's own int8 codes: the {@code scale} of the
     * static scheme, the per-token row scale, or the per-128-run scale, written into the record layout
     * {@link #stageQ8} writes. The run kernels launch one workgroup per run at the row-reduce work-group size.
     */
    ScratchSlot stageInt8(Int8ActStaging staging, GpuBuffer x, int xOff, int nTokens, int cols) {
        int tk = mmqConfig.stageDepth();
        long records = (long) nTokens * mmqConfig.rowRecords(cols);
        ScratchSlot slot = scratch.staging(GpuScratch.Staging.Q8_1,
                Math.multiplyExact(records, mmqConfig.recordUnits()));
        if (staging.scheme() == ActQuantScheme.INT8_STATIC) {
            ops.launch(staging.kernel(), records, nTokens, cols, tk, cols, xOff, staging.scale(),
                    x, slot.buffer());
        } else if (staging.scheme() == ActQuantScheme.INT8_TOKEN) {
            ops.launch(staging.kernel(), Math.multiplyExact((long) nTokens, local), nTokens, cols, tk, cols,
                    xOff, x, slot.buffer());
        } else if (staging.scheme() == ActQuantScheme.INT8_GROUP128) {
            long runs = Math.multiplyExact((long) nTokens, cols / 128);
            ops.launch(staging.kernel(), Math.multiplyExact(runs, local), nTokens, cols, tk, cols, xOff, x,
                    slot.buffer());
        } else if (staging.scheme() == ActQuantScheme.INT8_STATIC_ASYM) {
            ops.launch(staging.kernel(), records, nTokens, cols, tk, cols, xOff, staging.scale(),
                    staging.zeroPoint(), x, slot.buffer());
        } else {
            throw new IllegalStateException(staging.scheme() + " has no direct int8 staging");
        }
        return slot;
    }

    /**
     * Prepares the fast GEMM lane of weights stored as {@code type}, the kernels of its split-K pair (the twin and
     * {@code LLM_SplitK_Reduce}, and for the MMQ lane the staging pass) and its small-token-tile twin, each through
     * {@link GpuOps#optionalReady}, so a kernel the backend does not provide or cannot run at its size is recorded as
     * unavailable and its fallback serves every product. A type without a fast lane prepares nothing.
     */
    void prewarm(GGMLType type) {
        if (GemmKernels.isW4a8(type) && fp8Tensor()) {
            ops.optionalReady(GemmKernels.matmulW4a8(type), mmqConfig.workGroupSize(), 1);
            ops.optionalReady(GemmKernels.matmulW4a8KSplit(type), mmqConfig.workGroupSize(), 1);
            prewarmTwins((tokens, group) -> {
                ops.optionalReady(GemmKernels.matmulW4a8Small(type, tokens), group, 1);
                ops.optionalReady(GemmKernels.matmulW4a8SmallSplit(type, tokens), group, 1);
            });
            prewarmFp8Staging();
        }
        String fast = fastGemm(type);
        if (fast == null) {
            return;
        }
        int workGroup = fastWorkGroup(fast);
        ops.optionalReady(fast, workGroup, 1);
        if (fast.endsWith(GemmKernels.MATMUL_MMQ_SUFFIX)) {
            ops.optionalReady(GemmKernels.QUANT_MMQ_Q8_1, local);
            ops.optionalReady(GemmKernels.matmulMmqKSplit(type), workGroup, 1);
            if (SubgroupMmqGemm.programBuilds(kernels) && SubgroupMmqGemm.kernel(type) != null) {
                ops.optionalReady(SubgroupMmqGemm.kernel(type), SubgroupMmqGemm.WORK_GROUP, 1);
            }
            prewarmTwins((tokens, group) -> {
                ops.optionalReady(GemmKernels.matmulMmqSmall(type, tokens), group, 1);
                ops.optionalReady(GemmKernels.matmulMmqSmallSplit(type, tokens), group, 1);
            });
            prewarmFp8Staging();
        } else if (fast.endsWith(GemmKernels.MATMUL_CM_SUFFIX)) {
            ops.optionalReady(GemmKernels.matmulCmKSplit(type), workGroup, 1);
        } else {
            ops.optionalReady(GemmKernels.matmulMmaKSplit(type), workGroup, 1);
        }
        ops.optionalReady(GemmKernels.SPLIT_K_REDUCE, local);
    }

    /** Prepares every E4M3 activation staging kernel at the row-reduce work-group size. */
    private void prewarmFp8Staging() {
        for (ActQuantScheme s : ActQuantScheme.values()) {
            String staging = GemmKernels.fp8ActivationStaging(s);
            if (staging != null) {
                ops.optionalReady(staging, local);
            }
        }
    }

    /** Prepares the small-token-tile twins of each ladder row the device tile is wider than, both
     *  at the twins' shared work-group size ({@link MmqConfig#smallTile}: the 256-item group). */
    private void prewarmTwins(TwinPrewarm twin) {
        if (mmqConfig.tileTokens() <= MmqConfig.TOKEN_LANES) {
            return;
        }
        int twinGroup = MmqConfig.smallTile(mmqConfig).workGroupSize();
        twin.at(MmqConfig.TOKEN_LANES, twinGroup);
        if (mmqConfig.tileTokens() > MmqConfig.TWIN_MAX_TOKENS) {
            twin.at(MmqConfig.TWIN_MAX_TOKENS, twinGroup);
        }
    }

    /** One ladder row of a lane's twin prewarm: the twin and its split-K form at {@code tokens}. */
    private interface TwinPrewarm {
        void at(int tokens, int workGroup);
    }
}

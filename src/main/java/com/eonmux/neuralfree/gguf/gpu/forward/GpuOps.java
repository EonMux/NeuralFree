package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.GpuBackendException;
import com.eonmux.neuralfree.acceleration.GpuOutOfMemoryException;
import com.eonmux.neuralfree.acceleration.UnsupportedKernelException;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.acceleration.backend.KernelHandle;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.arch.GluOp;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.kernels.LlmKernelConstants;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuPlaceholders;
import com.eonmux.neuralfree.gguf.ops.act.ActKernels;
import com.eonmux.neuralfree.gguf.ops.norm.NormKernels;
import com.eonmux.neuralfree.gguf.quant.HadamardSpec;
import com.eonmux.neuralfree.gguf.spec.ActParams;
import com.eonmux.neuralfree.utils.LoggingUtil;
import org.apache.logging.log4j.Logger;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.EnumMap;
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
 * ({@code {T}_MatVec}, {@code {T}_MatVecAdd}, {@code {T}_BatchMatMul[_2D]}, {@code {T}_MatMul_Tiled}; F32, Q8_0 and
 * MXFP4 keep their legacy names). They loop over the weight's row shards, writing each shard's rows at
 * {@code outOff + t*rows + shardRowStart}. The matvec kernels have no offset arguments, so an input at a nonzero
 * offset is first copied to a staging buffer, and an output at a nonzero offset or of a sharded weight is computed into
 * staging and scattered with {@value #COPY2D}. The tiled GEMM takes offsets and strides itself. A PRISM weight
 * ({@link GpuWeight#hadamard()}) first rotates its input into staging with {@value HadamardSpec#KERNEL}
 * (llama-graph.cpp:1598-1626); a weight with an output scale multiplies the result
 * (llama-graph.cpp:1643-1645).</p>
 *
 * <p><b>Optional projection kernels.</b> The tiled GEMM, the {@code _2D} batch and the in-kernel accumulate
 * ({@code {T}_MatVecAdd}) each have a fallback: the batched matvec, the one-dimensional batch, and the product into
 * staging followed by {@value #AXPBY}. Such a kernel is prepared once for its work-group size; one the backend does
 * not provide, or whose own work-group limit (OpenCL {@code CL_KERNEL_WORK_GROUP_SIZE}, a register-limited CUDA
 * block) is below that size, is never launched and its fallback serves every product.</p>
 *
 * <p><b>Aliasing.</b> Views of one buffer either coincide (same offset, and same strides where strides apply) or are
 * disjoint; operations that allow in-place use say so, all others require disjoint views. Overlaps are rejected with
 * {@link IllegalArgumentException} before anything is enqueued.</p>
 *
 * <p><b>Launch geometry.</b> Row-reduce kernels (matvec families, {@code LLM_RowNorm}, Hadamard) run one work-group of
 * {@code L = LlmKernels.matvecLocal()} work-items per output row (per row block for Hadamard): global
 * {@code rows * L}. {@code _2D} batches add the token dimension ({@code {rows * L, nTokens}}, local {@code {L, 1}})
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
    public static final String AXPBY = "LLM_Axpby_F32";
    /**
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

    /** Work-group size of the tiled GEMM ({@code NF_GEMM_WG}). */
    static final int TILED_WG = 256;
    /** Weight rows per tiled GEMM work-group ({@code NF_GEMM_TM}). */
    static final int TILED_ROWS = 64;
    /** Tokens per tiled GEMM work-group ({@code NF_GEMM_TN}). */
    static final int TILED_TOKENS = 32;
    /** Largest cooperating-thread count of any matvec family ({@code *_TPS}); the matvec local size is a multiple. */
    static final int MATVEC_MAX_TPS = 32;
    /** The kernels this class launches itself on a 1-D grid of local size {@code L}, apart from the weight families. */
    static final List<String> HELPER_KERNELS =
            Collections.unmodifiableList(Arrays.asList(ROW_NORM, UNARY, GLU, COPY2D, AXPBY, HadamardSpec.KERNEL));

    /** Kernel names of one weight type. */
    static final class WeightKernels {
        final String matvec;
        final String matvecAdd;
        final String batch;
        final String batch2d;
        final String tiled;
        final String embed;
        final String batchEmbed;

        WeightKernels(String matvec, String matvecAdd, String batch, String batch2d, String tiled, String embed,
                      String batchEmbed) {
            this.matvec = matvec;
            this.matvecAdd = matvecAdd;
            this.batch = batch;
            this.batch2d = batch2d;
            this.tiled = tiled;
            this.embed = embed;
            this.batchEmbed = batchEmbed;
        }

        static WeightKernels standard(String t) {
            return new WeightKernels(t + "_MatVec", t + "_MatVecAdd", t + "_BatchMatMul", t + "_BatchMatMul_2D",
                    t + "_MatMul_Tiled", t + "_EmbeddingLookup", t + "_BatchEmbeddingLookup");
        }
    }

    private static final Map<GGMLType, WeightKernels> FAMILIES = families();

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
            GGMLType.TQ1_0, GGMLType.TQ2_0, GGMLType.NVFP4, GGMLType.Q1_0, GGMLType.Q2_0,
        };
        for (GGMLType t : standard) {
            m.put(t, WeightKernels.standard(t.name()));
        }
        return Collections.unmodifiableMap(m);
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
    private final long[] maxGroups;
    private final long maxWorkGroupSize;
    private final Set<PreparedDispatch> prepared = Collections.newSetFromMap(new IdentityHashMap<>());
    private final Map<String, float[]> signValues = new HashMap<>();
    private final Map<String, GpuBuffer> signBuffers = new HashMap<>();
    private final Map<HadamardSpec, GpuBuffer> signsBySpec = new IdentityHashMap<>();
    private final Map<String, Boolean> optionalStatus = new HashMap<>();
    private final GpuPlaceholders placeholders;
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
        this.queue = Objects.requireNonNull(queue, "queue");
        this.kernels = Objects.requireNonNull(kernels, "kernels");
        this.scratch = Objects.requireNonNull(scratch, "scratch");
        this.placeholders = Objects.requireNonNull(placeholders, "placeholders");
        this.backend = kernels.backend();
        this.local = kernels.matvecLocal();
        this.maxGroups = kernels.caps().getMaxWorkGroupCount();
        this.maxWorkGroupSize = kernels.caps().getMaxWorkGroupSize();
        if (local < 1 || Integer.bitCount(local) != 1) {
            throw new IllegalStateException("row-reduce work-group size " + local + " is not a power of two");
        }
        prewarmHelperKernels();
    }

    /** Prepares each of the {@link #HELPER_KERNELS} for local size {@code L} ({@link #prewarm(String, long[])}). */
    private void prewarmHelperKernels() {
        long[] size = {local};
        for (String kernel : HELPER_KERNELS) {
            prewarm(kernel, size);
        }
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

    /** True when GpuOps has projection and lookup kernels for weights of this type. */
    public static boolean supportsWeightType(GGMLType type) {
        return FAMILIES.containsKey(type);
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
     * batched matvec kernels.
     */
    public void matvec(GpuWeight w, GpuBuffer in, int inOff, GpuBuffer out, int outOff, int nTokens) {
        project(w, in, inOff, out, outOff, nTokens, false, true);
    }

    /**
     * {@link #matvec}, using the tiled GEMM when {@code nTokens >= TILED_MIN_TOKENS} and the kernel can run its
     * work-group on this device.
     */
    public void matmul(GpuWeight w, GpuBuffer in, int inOff, GpuBuffer out, int outOff, int nTokens) {
        project(w, in, inOff, out, outOff, nTokens, true, true);
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
        WeightKernels k = family(w);
        if (nTokens == 1 && w.isSingleShard() && accOff == 0 && !w.hasScale()
                && !overlaps(in, inOff, nIn, accum, accOff, nOut) && optionalReady(k.matvecAdd, local)) {
            requireMatvecLocal(w);
            View x = offsetFree(projectionInput(w, in, inOff, nTokens, cols), nTokens, cols);
            launch(k.matvecAdd, (long) rows * local, rows, cols, w.shard(0), x.buf, accum);
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
        WeightKernels k = family(w);
        View x = projectionInput(w, in, inOff, nTokens, cols);
        if (allowTiled && nTokens >= LlmKernelConstants.TILED_MIN_TOKENS && tiledUsable(k, w, nTokens)) {
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
            for (int s = 0; s < w.shardCount(); s++) {
                int m = (int) w.shardRows(s);
                if (direct) {
                    matvecShard(k, w.shard(s), m, cols, x.buf, out, nTokens);
                } else {
                    ScratchSlot stage = scratch.staging(GpuScratch.Staging.OUTPUT, (long) nTokens * m);
                    matvecShard(k, w.shard(s), m, cols, x.buf, stage.buffer(), nTokens);
                    copy2dUnchecked(nTokens, m, stage.buffer(), stage.offset(), m,
                            out, outOff + (int) w.shardRowStart(s), rows);
                }
            }
        }
        if (applyScale && w.hasScale()) {
            axpby((int) nOut, 0.0f, null, 0, w.scale(), out, outOff, out, outOff);
        }
    }

    /** The projection input: the caller's view, or its PRISM rotation in staging. */
    private View projectionInput(GpuWeight w, GpuBuffer in, int inOff, int nTokens, int cols) {
        HadamardSpec h = w.hadamard();
        if (h == null) {
            return new View(in, inOff);
        }
        ScratchSlot s = scratch.staging(GpuScratch.Staging.INPUT, (long) nTokens * cols);
        hadamard(h, in, inOff, s.buffer(), s.offset(), nTokens, cols);
        return new View(s.buffer(), s.offset());
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

    private void matvecShard(WeightKernels k, GpuBuffer shard, int m, int cols, GpuBuffer x, GpuBuffer dst,
                             int nTokens) {
        long global = (long) m * local;
        if (nTokens == 1) {
            launch(k.matvec, global, m, cols, shard, x, dst);
        } else if (m <= maxGroups[0] && nTokens <= maxGroups[1] && optionalReady(k.batch2d, local, 1)) {
            launch2d(k.batch2d, global, nTokens, local, m, cols, nTokens, shard, x, dst);
        } else {
            launch(k.batch, global, m, cols, nTokens, shard, x, dst);
        }
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
     * Whether an optional projection kernel (see the class notes) can be launched at work-group size
     * {@code localSize}: prepared on the first query ({@link #prewarm(String, long[])}) and remembered, so a kernel
     * the backend does not provide or cannot run at that size is never launched and its caller takes the fallback.
     * Each optional kernel is launched with one work-group size only.
     */
    private boolean optionalReady(String kernel, long... localSize) {
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
     * batch, in-kernel accumulate) that cannot run their work-group size on this device are recorded as unavailable,
     * so the projections use their fallbacks; a required kernel that cannot run its size fails here, while the model
     * binds.
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
        WeightKernels k = FAMILIES.get(Objects.requireNonNull(type, "type"));
        if (k == null) {
            throw new UnsupportedOperationException("no GPU projection kernels for weight type " + type);
        }
        long[] rowReduce = {local};
        if (matmul && local % MATVEC_MAX_TPS == 0) {
            prewarmIfPresent(k.matvec, rowReduce);
            prewarmIfPresent(k.batch, rowReduce);
            optionalReady(k.matvecAdd, local);
            optionalReady(k.batch2d, local, 1);
        }
        if (matmul && maxWorkGroupSize >= TILED_WG) {
            optionalReady(k.tiled, TILED_WG, 1);
        }
        if (lookup) {
            prewarmIfPresent(k.embed, rowReduce);
            prewarmIfPresent(k.batchEmbed, rowReduce);
        }
    }

    /** Prepares a provided kernel that has no fallback; one that cannot run {@code localSize} fails the bind. */
    private void prewarmIfPresent(String kernel, long[] localSize) {
        if (kernels.has(kernel)) {
            kernels.prewarm(kernel, localSize);
        }
    }

    private void requireMatvecLocal(GpuWeight w) {
        if (local % MATVEC_MAX_TPS != 0) {
            throw new UnsupportedOperationException("matvec of " + w.name() + " needs a work-group size that is a"
                    + " multiple of " + MATVEC_MAX_TPS + "; this device allows " + local);
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
        launch(k.batchEmbed, n, dim, nTokens, tokenIds, table.shard(0), dst.buf);
        finishLookup(inv, dst, out, outOff, nTokens, dim);
    }

    /**
     * Looks up the rows of host token ids (any shard layout) into {@code out[outOff + t*cols ...]}, then applies the
     * table's inverse Hadamard transform when it has one.
     *
     * @throws IllegalArgumentException when a token id is outside the table
     */
    public void embed(GpuWeight table, int[] tokens, GpuBuffer out, int outOff) {
        ensureOpen();
        Objects.requireNonNull(tokens, "tokens");
        int nTokens = tokens.length;
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
                launch(k.embed, dim, dim, localId, table.shard(s), rowsDst.buf);
            } else {
                ScratchSlot stage = scratch.staging(GpuScratch.Staging.OUTPUT, dim);
                launch(k.embed, dim, dim, localId, table.shard(s), stage.buffer());
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
    private void launch(String kernel, long n, Object... args) {
        long g = roundUp(n, local);
        queue.enqueueKernelNoEvent(kernels.get(kernel), new long[]{g}, new long[]{local}, args);
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

    private static WeightKernels family(GpuWeight w) {
        WeightKernels k = FAMILIES.get(w.type());
        if (k == null) {
            throw new UnsupportedOperationException("tensor " + w.name() + ": no GPU kernels for weight type "
                    + w.type());
        }
        return k;
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

    private static long ceilDiv(long n, long d) {
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

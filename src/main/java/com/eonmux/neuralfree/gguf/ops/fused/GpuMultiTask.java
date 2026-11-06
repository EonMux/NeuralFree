package com.eonmux.neuralfree.gguf.ops.fused;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuScratch;
import com.eonmux.neuralfree.gguf.gpu.forward.PreparedDispatch;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.ops.norm.NormKernels;
import com.eonmux.neuralfree.gguf.ops.rope.RopeKernels;
import com.eonmux.neuralfree.gguf.ops.rope.RopeMath;
import com.eonmux.neuralfree.gguf.spec.RopeSpec;

import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicLong;

/**
 * A set of independent small kernels of one decode step recorded as tasks of one persistent
 * {@value FusedKernels#MULTI_TASK} launch ({@code kernels/llm/multitask.cl}). The tasks run concurrently, one
 * workgroup block each, instead of as separate launches; the results are those of the individual kernels because
 * every task is recorded only for views it proves disjoint from the others.
 *
 * <p>Tasks and their buffers are fixed at {@link #build()}; the token count is the only value that changes per
 * step ({@link #run(int)}), so the prepared dispatch is reused unchanged. A task's row or element count is
 * {@code tokens * perToken}, which lets one descriptor serve decode and prefill steps.</p>
 *
 * <p>Each instance owns its descriptor in a scratch temporary whose name is unique for the model's scratch: two
 * launches of one step (there are several per layer) must not share a descriptor, which is read when the launch
 * runs, after later builds have written theirs.</p>
 *
 * <p>The multi-task kernel is optional: {@link #usable(GpuOps)} answers whether the backend provides it, and the
 * owning block falls back to the individual kernels otherwise. Not thread-safe.</p>
 */
public final class GpuMultiTask implements AutoCloseable {

    /** One task of the launch: a record and the buffers it addresses (resolved to launch slots at build). */
    private static final class Task {
        final int[] ints = new int[20];
        final float[] floats = new float[8];
        final GpuBuffer[] buffers = new GpuBuffer[12];
        int kind;

        Task(int kind) {
            this.kind = kind;
        }
    }

    /** Distinguishes the descriptors of the launches the model's blocks build. */
    private static final AtomicLong IDS = new AtomicLong();
    private final GpuOps ops;
    private final int ubatch;
    private final String label;
    private final int local;
    private final List<Task> tasks = new ArrayList<>();
    private final List<GpuBuffer> launchBuffers = new ArrayList<>();
    private ScratchSlot desc;
    private PreparedDispatch dispatch;
    private int blocksPerTask;
    private long workBlocks;
    private boolean built;
    private boolean closed;

    /**
     * @param ops    the dispatch helpers of the bound context
     * @param ubatch largest token count of a step (the descriptor reserves this many rows)
     * @param label  names the descriptor's scratch temporary ({@code fused.mt.<label>.<n>})
     */
    public GpuMultiTask(GpuOps ops, int ubatch, String label) {
        this.ops = Objects.requireNonNull(ops, "ops");
        if (ubatch <= 0) {
            throw new IllegalArgumentException("multi-task ubatch must be positive, got " + ubatch);
        }
        if (label == null || label.isEmpty()) {
            throw new IllegalArgumentException("multi-task label must not be empty");
        }
        this.ubatch = ubatch;
        this.label = label;
        this.local = ops.local();
    }

    /** Whether the backend provides the persistent multi-task kernel. */
    public static boolean usable(GpuOps ops) {
        return ops.fused(FusedKernels.MULTI_TASK);
    }

    /** Whether the kernel accepts the single-token row-norm local size ({@code L} must not exceed LLM_DIMX). */
    public static boolean localUsable(int local) {
        return local >= 1 && local <= 256 && Integer.bitCount(local) == 1;
    }

    /** The tasks recorded so far. */
    public int taskCount() {
        return tasks.size();
    }

    // ------------------------------------------------------------------ task recorders

    /**
     * Records a row norm ({@code LLM_RowNorm}) over {@code tokens * rowsPerToken} rows.
     *
     * @throws IllegalArgumentException when the kind has no multi-task body or the geometry is not positive
     */
    public void rowNorm(int rowsPerToken, int rowLen, int rowStride, NormKind kind, int flags, int wRows, int xOff,
                        int yOff, float eps, GpuBuffer x, GpuBuffer w, GpuBuffer b, GpuBuffer y) {
        requireEditable();
        Objects.requireNonNull(kind, "kind");
        if (kind != NormKind.RMS && kind != NormKind.L2_GDN) {
            throw new IllegalArgumentException(FusedKernels.MULTI_TASK + " has no " + kind + " row body; it runs "
                    + NormKind.RMS + " and " + NormKind.L2_GDN);
        }
        if (!(eps >= 0f) || Float.isInfinite(eps)) {
            throw new IllegalArgumentException(FusedKernels.MULTI_TASK + " row norm needs a finite non-negative "
                    + "epsilon, got " + eps);
        }
        if (rowsPerToken <= 0 || rowLen <= 0 || rowStride < rowLen || wRows < 1) {
            throw new IllegalArgumentException(FusedKernels.MULTI_TASK + " row norm needs positive rows, row length "
                    + "and weight rows and a row stride of at least the row length, got " + rowsPerToken + " rows, "
                    + rowLen + " x " + rowStride + ", " + wRows + " weight rows");
        }
        Task t = new Task(NF_MT_ROW_NORM);
        t.ints[0] = rowsPerToken;
        t.ints[1] = rowLen;
        t.ints[2] = rowStride;
        t.ints[3] = kind.rowNormMode();
        t.ints[4] = flags;
        t.ints[5] = wRows;
        t.ints[6] = xOff;
        t.ints[7] = yOff;
        t.floats[0] = eps;
        t.buffers[0] = x;
        t.buffers[1] = w;
        t.buffers[2] = b;
        t.buffers[3] = y;
        add(t, (long) ubatch * Math.max(1, rowsPerToken));
    }

    /**
     * Records an RMS norm followed by RoPE over {@code tokens * nHeads} rows (the fused norm-rope body).
     *
     * @throws IllegalArgumentException when the geometry is not positive or the rotation does not fit the row
     */
    public void normRope(RopeSpec rope, float eps, int nHeads, int headStride, int rowStride, int rowLen, int nOffs,
                         int xOff, int yOff, int flags, int wRows, boolean hasFf, GpuBuffer pos, GpuBuffer ff,
                         GpuBuffer w, GpuBuffer b, GpuBuffer x, GpuBuffer y) {
        requireEditable();
        Objects.requireNonNull(rope, "rope");
        if (!(eps >= 0f) || Float.isInfinite(eps)) {
            throw new IllegalArgumentException(FusedKernels.MULTI_TASK + " norm-rope needs a finite non-negative "
                    + "epsilon, got " + eps);
        }
        if (nHeads <= 0 || headStride <= 0 || rowStride <= 0 || rowLen <= 0 || wRows < 1) {
            throw new IllegalArgumentException(FusedKernels.MULTI_TASK + " norm-rope needs positive heads, strides, "
                    + "row length and weight rows, got " + nHeads + " heads, row " + rowLen + " of " + rowStride
                    + ", head stride " + headStride + ", " + wRows + " weight rows");
        }
        RopeKernels.validateRopeExt(rope.type(), rope.nDims(), nOffs, ubatch, nHeads, headStride, rowStride, ubatch,
                xOff);
        if ((long) nOffs + RopeMath.rotatedWidth(rope.type(), rope.nDims()) > rowLen) {
            throw new IllegalArgumentException(FusedKernels.MULTI_TASK + ": rotated dims [" + nOffs + ", "
                    + (nOffs + RopeMath.rotatedWidth(rope.type(), rope.nDims()))
                    + ") exceed the normalized row of " + rowLen);
        }
        Task t = new Task(NF_MT_NORM_ROPE);
        t.ints[0] = nHeads;
        t.ints[1] = headStride;
        t.ints[2] = rowStride;
        t.ints[3] = rowLen;
        t.ints[4] = rope.type().ggmlMode();
        t.ints[5] = rope.nDims();
        t.ints[6] = nOffs;
        t.ints[7] = hasFf ? 1 : 0;
        t.ints[8] = flags;
        t.ints[9] = wRows;
        int[] s = rope.sections();
        t.ints[10] = s[0];
        t.ints[11] = s[1];
        t.ints[12] = s[2];
        t.ints[13] = s[3];
        t.ints[14] = xOff;
        t.ints[15] = yOff;
        t.floats[0] = eps;
        t.floats[1] = rope.thetaScale();
        t.floats[2] = rope.freqScale();
        t.floats[3] = rope.extFactor();
        t.floats[4] = rope.attnFactor();
        t.floats[5] = rope.corrLo();
        t.floats[6] = rope.corrHi();
        t.buffers[0] = pos;
        t.buffers[1] = ff;
        t.buffers[2] = w;
        t.buffers[3] = b;
        t.buffers[4] = x;
        t.buffers[5] = y;
        add(t, (long) ubatch * nHeads);
    }

    /**
     * Records a gated DeltaNet gate-preparation task ({@code LLM_GDN_GatePrep}).
     *
     * @param betaScale multiplier of the sigmoid update rate (2.0 for OLMo Hybrid's negative eigenvalues, else 1.0)
     * @throws IllegalArgumentException when {@code betaScale} is not positive
     */
    public void gatePrep(int nVHeads, int nKHeads, int layout, int kdaMode, int headDim, int aStride, int aOff,
                         int bStride, int bOff, float lb, float betaScale, GpuBuffer aIn, GpuBuffer bIn,
                         GpuBuffer dtBias, GpuBuffer ssmA, GpuBuffer gOut, GpuBuffer betaOut) {
        requireEditable();
        if (!(betaScale > 0f)) {
            throw new IllegalArgumentException("beta scale must be positive, got " + betaScale);
        }
        Task t = new Task(NF_MT_GATE_PREP);
        t.ints[0] = nVHeads;
        t.ints[1] = nKHeads;
        t.ints[2] = layout;
        t.ints[3] = kdaMode;
        t.ints[4] = headDim;
        t.ints[5] = aStride;
        t.ints[6] = aOff;
        t.ints[7] = bStride;
        t.ints[8] = bOff;
        t.floats[0] = lb;
        t.floats[1] = betaScale;
        t.buffers[0] = aIn;
        t.buffers[1] = bIn;
        t.buffers[2] = dtBias;
        t.buffers[3] = ssmA;
        t.buffers[4] = gOut;
        t.buffers[5] = betaOut;
        long perHead = kdaMode < 0 ? 1L : Math.max(1, headDim);
        long elements = (long) ubatch * nVHeads * perHead;
        add(t, (elements + local - 1) / local);
    }

    /**
     * Records a dense F32 projection {@code out[t][r] = sum_k W[r][k] * x[t][k]}.
     *
     * @throws IllegalArgumentException when the weight is not one F32 buffer, or carries a transform or scale
     */
    public void matvecF32(GpuWeight w, GpuBuffer x, int xOff, GpuBuffer out, int outOff) {
        requireEditable();
        Objects.requireNonNull(w, "weight");
        if (w.type() != com.eonmux.neuralfree.gguf.GGMLType.F32 || !w.isSingleShard() || w.hasInputTransform()
                || w.hadamard() != null || w.hasScale() || w.rows() > Integer.MAX_VALUE
                || w.cols() > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(FusedKernels.MULTI_TASK + ": " + w.name()
                    + " is not a plain single-buffer F32 matrix");
        }
        int rows = (int) w.rows();
        int cols = (int) w.cols();
        Task t = new Task(NF_MT_MATVEC_F32);
        t.ints[0] = rows;
        t.ints[1] = cols;
        t.ints[2] = cols;
        t.ints[3] = xOff;
        t.ints[4] = rows;
        t.ints[5] = outOff;
        t.buffers[0] = w.shard(0);
        t.buffers[1] = x;
        t.buffers[2] = out;
        add(t, (long) ubatch * rows);
    }

    private void add(Task t, long work) {
        if (tasks.size() >= NF_MT_MAX_TASKS) {
            throw new IllegalStateException(FusedKernels.MULTI_TASK + " records at most " + NF_MT_MAX_TASKS
                    + " tasks per launch");
        }
        tasks.add(t);
        workBlocks = Math.max(workBlocks, Math.max(1L, work));
    }

    private void requireEditable() {
        if (built || closed) {
            throw new IllegalStateException("multi-task batch is already built or closed");
        }
    }

    // ------------------------------------------------------------------ build and run

    /**
     * Packs the tasks into the descriptor, allocates the descriptor buffer and prepares the launch.
     *
     * @throws IllegalStateException    when already built or closed, or no task was recorded
     * @throws IllegalArgumentException when more than {@value FusedKernels#MULTI_TASK_BUFFERS} distinct
     *                                  buffers are addressed
     * @throws com.eonmux.neuralfree.acceleration.UnsupportedKernelException when the backend lacks the kernel
     */
    public void build() {
        requireEditable();
        if (tasks.isEmpty()) {
            throw new IllegalStateException("a multi-task launch needs at least one task");
        }
        Map<GpuBuffer, Integer> slots = new IdentityHashMap<>();
        float[] records = new float[tasks.size() * FusedKernels.MULTI_TASK_WORDS];
        for (int ti = 0; ti < tasks.size(); ti++) {
            Task t = tasks.get(ti);
            int base = ti * FusedKernels.MULTI_TASK_WORDS;
            records[base] = Float.intBitsToFloat(t.kind);
            for (int s = 0; s < t.buffers.length; s++) {
                GpuBuffer buffer = t.buffers[s];
                if (buffer == null) {
                    continue;
                }
                Integer slot = slots.get(buffer);
                if (slot == null) {
                    if (slots.size() >= FusedKernels.MULTI_TASK_BUFFERS) {
                        throw new IllegalArgumentException(FusedKernels.MULTI_TASK + " addresses more than "
                                + FusedKernels.MULTI_TASK_BUFFERS + " distinct buffers");
                    }
                    slot = slots.size();
                    slots.put(buffer, slot);
                    launchBuffers.add(buffer);
                }
                records[base + 2 + s] = Float.intBitsToFloat(slot);
            }
            for (int k = 0; k < t.ints.length; k++) {
                records[base + 14 + k] = Float.intBitsToFloat(t.ints[k]);
            }
            for (int k = 0; k < t.floats.length; k++) {
                records[base + 34 + k] = t.floats[k];
            }
        }
        GpuScratch scratch = ops.scratch();
        desc = scratch.temp("fused.mt." + label + "." + IDS.incrementAndGet(), (long) records.length);
        long bytes = (long) records.length * Float.BYTES;
        GpuEvent event = ops.queue().enqueueWriteBuffer(desc.buffer(), (long) desc.offset() * Float.BYTES, bytes,
                records);
        if (event != null) {
            event.close();
        }
        // One block per row (or per local-size chunk of elements) of the widest task; shorter tasks loop fewer times.
        blocksPerTask = (int) Math.min(8192L, Math.max(1L, workBlocks));
        Object[] args = new Object[4 + FusedKernels.MULTI_TASK_BUFFERS];
        args[0] = ubatch;
        args[1] = tasks.size();
        args[2] = blocksPerTask;
        args[3] = desc.buffer();
        GpuBuffer placeholder = ops.placeholders().readOnlyFloats();
        for (int i = 0; i < FusedKernels.MULTI_TASK_BUFFERS; i++) {
            args[4 + i] = i < launchBuffers.size() ? launchBuffers.get(i) : placeholder;
        }
        long global = (long) tasks.size() * blocksPerTask * local;
        dispatch = ops.prepare(FusedKernels.MULTI_TASK, new long[]{global}, new long[]{local}, args);
        built = true;
    }

    /** True once {@link #build()} succeeded. */
    public boolean isBuilt() {
        return built;
    }

    /**
     * Launches the prepared tasks over {@code nTokens} tokens.
     *
     * @throws IllegalStateException    when not built or closed
     * @throws IllegalArgumentException when {@code nTokens} is outside {@code [1, ubatch]}
     */
    public void run(int nTokens) {
        if (closed) {
            throw new IllegalStateException("multi-task batch is closed");
        }
        if (!built) {
            throw new IllegalStateException("multi-task batch is not built");
        }
        if (nTokens < 1 || nTokens > ubatch) {
            throw new IllegalArgumentException("multi-task step of " + nTokens + " tokens is outside [1, " + ubatch
                    + "]");
        }
        dispatch.setInt(0, nTokens);
        dispatch.dispatch();
    }

    /** Releases the prepared dispatch and the descriptor. Idempotent. */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        if (dispatch != null) {
            dispatch.close();
            dispatch = null;
        }
    }

    @Override
    public String toString() {
        return "GpuMultiTask{" + tasks.size() + " tasks, blocks=" + blocksPerTask + ", local=" + local
                + (built ? "" : ", unbuilt") + (closed ? ", closed" : "") + '}';
    }

    // Task kind codes of kernels/llm/multitask.cl.
    private static final int NF_MT_ROW_NORM = 1;
    private static final int NF_MT_NORM_ROPE = 2;
    private static final int NF_MT_GATE_PREP = 3;
    private static final int NF_MT_MATVEC_F32 = 4;
    private static final int NF_MT_MAX_TASKS = FusedKernels.MULTI_TASK_MAX_TASKS;
}

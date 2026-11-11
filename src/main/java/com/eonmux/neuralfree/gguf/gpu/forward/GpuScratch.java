package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.GpuOutOfMemoryException;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.memory.MemoryPlan;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.ScratchWidths;
import com.eonmux.neuralfree.utils.Closeables;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * Device scratch memory of one model's forward pass.
 *
 * <ul>
 *   <li><b>Well-known slots</b> ({@link ScratchSlots}) are allocated once at construction, each in its own buffer,
 *       sized for {@code ubatch} rows by {@link #forModel}, and never change.</li>
 *   <li><b>Temporaries</b> ({@link #temp}) are named working buffers of the blocks, requested at bind time. A
 *       request is served by the current buffer of that name when it is large enough; a larger request allocates a
 *       new buffer that becomes the current one. Earlier holders keep their buffer, which stays valid until
 *       {@link #close()}. Callers sharing a name may overwrite each other's content between block calls, so a
 *       temporary carries data only within one block call.</li>
 *   <li><b>Flash partials</b> ({@link #partials}) follow the same rule for the split-K attention partial sums.</li>
 *   <li><b>GpuOps staging</b> buffers are temporaries reserved for {@link GpuOps}.</li>
 *   <li><b>Step token ids</b> ({@link #recordStepTokens}) are the host copy of the ids the embedding stage writes to
 *       {@link ScratchSlots#TOKEN_IDS}, so blocks that need them on the host read them without a device read. A step
 *       whose ids the device wrote ({@link #recordDeviceStepTokens}, a chained greedy decode) has no host copy, and
 *       reading it fails; {@link #stepTokenReads()} counts the reads, so a caller can tell a trunk that never reads
 *       them.</li>
 * </ul>
 *
 * <p>Every buffer is checked against the device's single-allocation limit and storage-binding range. A slot that
 * does not fit fails with {@link GpuOutOfMemoryException} naming the slot, so the context can be reduced with
 * {@code neuralfree.ubatch}. Not thread-safe: one forward pass at a time.</p>
 */
public final class GpuScratch implements AutoCloseable {

    /** {@link GpuOps} staging buffers. */
    enum Staging {
        /** Rotated or offset-free matmul input. */
        INPUT("gpuops.stage.input"),
        /** The PERMUTE gather of an input transform, out of place beside {@link #INPUT}. */
        PERMUTED("gpuops.stage.permuted"),
        /** Per-shard or offset-free matmul output. */
        OUTPUT("gpuops.stage.output"),
        /** Complete projection before an accumulate or scatter. */
        RESULT("gpuops.stage.result"),
        /** Split-K slice planes, apart from every view a projection reads or writes. */
        SPLIT_K("gpuops.stage.splitk"),
        /**
         * Staged 8-bit activation records: the int8 MMQ GEMM staging (LLM_Quant_MMQ_Q8_1's output) and the Q8_1
         * blocks of the integer-dot matvec lane (LLM_QuantizeQ8_1's output).
         */
        Q8_1("gpuops.stage.q8_1"),
        /**
         * Q8_1 blocks of an activation row shared by several projections (GpuOps.SharedQ8), apart from the
         * per-projection {@link #Q8_1} staging so an intervening projection cannot overwrite them.
         */
        SHARED_Q8_1("gpuops.stage.shared_q8_1"),
        /** The fp16 activation rows of the cooperative-matrix2 GEMM lane (LLM_Stage_F16_Rows' output). */
        F16("gpuops.stage.f16"),
        /**
         * The cooperative-matrix2 GEMM's fp16 rows of token rows shared by several projections
         * (GpuOps.SharedQ8#halfRows), apart from the per-projection {@link #F16} staging.
         */
        SHARED_F16("gpuops.stage.shared_f16"),
        /**
         * The int8 MMQ GEMM's staged records of token rows shared by several projections
         * (GpuOps.SharedQ8#records), apart from the per-projection {@link #Q8_1} staging and the shared matvec
         * blocks.
         */
        SHARED_MMQ("gpuops.stage.shared_mmq"),
        /**
         * The BF16 mirror a W12 projection decodes its weight into (GpuOps.w12Bf16Mirror), apart from every
         * staging view the projection on the mirror touches.
         */
        W12_BF16("gpuops.stage.w12_bf16");

        private final String name;

        Staging(String name) {
            this.name = name;
        }
    }

    private static final String RESERVED_PREFIX = "gpuops.";
    private static final String PARTIALS = "flash.partials";

    private final ComputeBackend backend;
    private final int ubatch;
    private final Map<String, ScratchSlot> slots;
    private final Map<String, ScratchSlot> temps = new HashMap<>();
    private final List<GpuBuffer> owned = new ArrayList<>();
    private final int[] stepTokenIds;
    private int stepTokenCount;
    /** Whether the current step's ids exist only on the device ({@link #recordDeviceStepTokens}). */
    private boolean stepTokensOnDevice;
    private long stepTokenReads;
    private long allocatedBytes;
    private boolean closed;

    private GpuScratch(ComputeBackend backend, int ubatch, Map<String, Long> slotFloats) {
        this.backend = backend;
        this.ubatch = ubatch;
        this.stepTokenIds = new int[ubatch];
        Map<String, ScratchSlot> s = new LinkedHashMap<>();
        try {
            for (Map.Entry<String, Long> e : slotFloats.entrySet()) {
                s.put(e.getKey(), allocate(e.getKey(), e.getValue()));
            }
        } catch (RuntimeException e) {
            close();
            throw e;
        }
        this.slots = Collections.unmodifiableMap(s);
    }

    /**
     * Scratch for a model, sized for the memory plan's {@code ubatch}.
     *
     * @throws GpuOutOfMemoryException when a slot exceeds the device's allocation or binding limit
     */
    public static GpuScratch forModel(ComputeBackend backend, ModelSpec spec, MemoryPlan plan) {
        Objects.requireNonNull(plan, "plan");
        return forModel(backend, spec, plan.ubatch());
    }

    /**
     * Scratch for a model run with at most {@code ubatch} tokens per step (also for models without memory).
     *
     * <p>The float slots take the per-token widths of {@link ScratchWidths}, the sizing shared with
     * {@code CpuScratch}; {@code TOKEN_IDS} holds 2 and {@code POS} 4 ints per token.</p>
     *
     * @throws IllegalArgumentException when {@code ubatch < 1}
     * @throws GpuOutOfMemoryException  when a slot exceeds the device's allocation or binding limit
     */
    public static GpuScratch forModel(ComputeBackend backend, ModelSpec spec, int ubatch) {
        Objects.requireNonNull(backend, "backend");
        Objects.requireNonNull(spec, "spec");
        checkUbatch(ubatch);
        return new GpuScratch(backend, ubatch, modelSlotFloats(spec, ubatch));
    }

    /** Scratch with explicitly sized slots (well-known or custom names). */
    public static Builder builder(ComputeBackend backend, int ubatch) {
        Objects.requireNonNull(backend, "backend");
        checkUbatch(ubatch);
        return new Builder(backend, ubatch);
    }

    private static void checkUbatch(int ubatch) {
        if (ubatch < 1) {
            throw new IllegalArgumentException("ubatch must be at least 1, got " + ubatch);
        }
    }

    /** Builder of a scratch with explicit slot sizes. */
    public static final class Builder {
        private final ComputeBackend backend;
        private final int ubatch;
        private final Map<String, Long> slots = new LinkedHashMap<>();

        private Builder(ComputeBackend backend, int ubatch) {
            this.backend = backend;
            this.ubatch = ubatch;
        }

        /**
         * Adds a slot of {@code floats} elements.
         *
         * @throws IllegalArgumentException for an empty or reserved name, a repeated name or a non-positive size
         */
        public Builder slot(String name, long floats) {
            checkName(name);
            if (floats <= 0) {
                throw new IllegalArgumentException("scratch slot " + name + " needs a positive size, got " + floats);
            }
            if (slots.put(name, floats) != null) {
                throw new IllegalArgumentException("scratch slot " + name + " is declared twice");
            }
            return this;
        }

        /** Allocates every slot. */
        public GpuScratch build() {
            return new GpuScratch(backend, ubatch, slots);
        }
    }

    // ------------------------------------------------------------------ sizing

    public static Map<String, Long> modelSlotFloats(ModelSpec spec, int ubatch) {
        ScratchWidths w = ScratchWidths.of(spec);
        Map<String, Long> m = new LinkedHashMap<>();
        m.put(ScratchSlots.X, ScratchWidths.rows(ubatch, w.stream()));
        m.put(ScratchSlots.BRANCH, ScratchWidths.rows(ubatch, w.stream()));
        m.put(ScratchSlots.NORMED, ScratchWidths.rows(ubatch, w.stream()));
        m.put(ScratchSlots.Q, ScratchWidths.rows(ubatch, w.query()));
        m.put(ScratchSlots.K, ScratchWidths.rows(ubatch, w.projection()));
        m.put(ScratchSlots.V, ScratchWidths.rows(ubatch, w.projection()));
        m.put(ScratchSlots.ATTN_OUT, ScratchWidths.rows(ubatch, w.projection()));
        m.put(ScratchSlots.FFN_GATE_UP, ScratchWidths.rows(ubatch, w.gateUp()));
        m.put(ScratchSlots.FFN_OUT, ScratchWidths.rows(ubatch, w.ffnOut()));
        m.put(ScratchSlots.LOGITS, ScratchWidths.rows(ubatch, w.logits()));
        m.put(ScratchSlots.TOKEN_IDS, ScratchWidths.rows(ubatch, ScratchWidths.TOKEN_ID_INTS));
        m.put(ScratchSlots.INPUT_EMBD, ScratchWidths.rows(ubatch, w.input()));
        m.put(ScratchSlots.POS, ScratchWidths.rows(ubatch, ScratchWidths.POS_INTS));
        return m;
    }

    // ------------------------------------------------------------------ access

    /** Maximum tokens per step. */
    public int ubatch() {
        return ubatch;
    }

    /**
     * A slot declared at construction.
     *
     * @throws IllegalArgumentException when no slot has this name
     */
    public ScratchSlot slot(String name) {
        ensureOpen();
        ScratchSlot s = slots.get(name);
        if (s == null) {
            throw new IllegalArgumentException("no scratch slot '" + name + "' (slots: " + slots.keySet() + ")");
        }
        return s;
    }

    /** True when a slot of this name was declared at construction. */
    public boolean has(String name) {
        return slots.containsKey(name);
    }

    /** Names of the slots declared at construction, in declaration order. */
    public Set<String> slotNames() {
        return slots.keySet();
    }

    /**
     * A temporary of at least {@code floats} elements under {@code name} (see the class contract).
     *
     * @throws IllegalArgumentException for an empty name, a slot name, a reserved name or a non-positive size
     * @throws GpuOutOfMemoryException  when the buffer exceeds the device's allocation or binding limit
     */
    public ScratchSlot temp(String name, long floats) {
        checkName(name);
        if (slots.containsKey(name)) {
            throw new IllegalArgumentException("scratch name '" + name + "' is a fixed slot, not a temporary");
        }
        return reserve(name, floats);
    }

    /**
     * A buffer of at least {@code floats} floats for flash-attention split partials (see the class contract).
     *
     * @throws IllegalArgumentException for a non-positive size
     * @throws GpuOutOfMemoryException  when the buffer exceeds the device's allocation or binding limit
     */
    public GpuBuffer partials(int floats) {
        return reserve(PARTIALS, floats).buffer();
    }

    /** {@link GpuOps} staging buffer of at least {@code floats} floats. */
    ScratchSlot staging(Staging which, long floats) {
        return reserve(which.name, floats);
    }

    /**
     * Records {@code ids[0 .. n)} as the current step's token ids, the values the embedding stage writes to
     * {@link ScratchSlots#TOKEN_IDS} for that step.
     *
     * @throws IllegalArgumentException when {@code n} is outside {@code [1, ubatch]} or {@code ids} is shorter
     */
    public void recordStepTokens(int[] ids, int n) {
        Objects.requireNonNull(ids, "ids");
        if (n < 1 || n > ubatch || ids.length < n) {
            throw new IllegalArgumentException("step of " + n + " token ids is outside [1, " + ubatch + "] or the "
                    + ids.length + "-element id array");
        }
        System.arraycopy(ids, 0, stepTokenIds, 0, n);
        stepTokenCount = n;
        stepTokensOnDevice = false;
    }

    /**
     * Records a step of {@code n} token ids that the device wrote to {@link ScratchSlots#TOKEN_IDS} without a host
     * copy (a chained greedy decode reads its id from the previous step's selection): {@link #readStepTokens} fails
     * until the next {@link #recordStepTokens}.
     *
     * @throws IllegalArgumentException when {@code n} is outside {@code [1, ubatch]}
     */
    public void recordDeviceStepTokens(int n) {
        if (n < 1 || n > ubatch) {
            throw new IllegalArgumentException("step of " + n + " token ids is outside [1, " + ubatch + "]");
        }
        stepTokenCount = n;
        stepTokensOnDevice = true;
    }

    /** How many times {@link #readStepTokens} ran on this scratch. */
    public long stepTokenReads() {
        return stepTokenReads;
    }

    /**
     * Copies the current step's {@code n} token ids ({@link #recordStepTokens}) to {@code dst[dstOff ..]}.
     *
     * @throws IllegalStateException    when the recorded step does not hold {@code n} tokens, or its ids exist only
     *                                  on the device ({@link #recordDeviceStepTokens})
     * @throws IllegalArgumentException when {@code dst} is too short
     */
    public void readStepTokens(int[] dst, int dstOff, int n) {
        Objects.requireNonNull(dst, "dst");
        stepTokenReads++;
        if (stepTokensOnDevice) {
            throw new IllegalStateException("the step's " + stepTokenCount + " token ids exist only on the device"
                    + " (a chained greedy decode); a block that reads them on the host cannot run in the chain");
        }
        if (n != stepTokenCount) {
            throw new IllegalStateException("the step's token ids were recorded for " + stepTokenCount
                    + " tokens, " + n + " are read");
        }
        if (dstOff < 0 || dstOff + n > dst.length) {
            throw new IllegalArgumentException("step token read of " + n + " ids at " + dstOff + " exceeds the "
                    + dst.length + "-element destination");
        }
        System.arraycopy(stepTokenIds, 0, dst, dstOff, n);
    }

    /** Bytes held by every buffer this scratch allocated, including superseded temporaries. */
    public long allocatedBytes() {
        return allocatedBytes;
    }

    private ScratchSlot reserve(String name, long floats) {
        ensureOpen();
        if (floats <= 0) {
            throw new IllegalArgumentException("scratch " + name + " needs a positive size, got " + floats);
        }
        ScratchSlot current = temps.get(name);
        if (current != null && current.floats() >= floats) {
            return current.floats() == floats ? current : current.slice(0, floats);
        }
        long grown = current == null ? floats : Math.max(floats, current.floats() + current.floats() / 2);
        ScratchSlot fresh;
        try {
            fresh = allocate(name, grown);
        } catch (GpuOutOfMemoryException e) {
            if (grown == floats) {
                throw e;
            }
            fresh = allocate(name, floats);
        }
        temps.put(name, fresh);
        return fresh.floats() == floats ? fresh : fresh.slice(0, floats);
    }

    private ScratchSlot allocate(String name, long floats) {
        long bytes = Math.multiplyExact(floats, (long) Float.BYTES);
        DeviceCapabilities caps = backend.capabilities();
        long limit = Long.MAX_VALUE;
        if (caps.getMaxAllocSizeBytes() > 0) {
            limit = caps.getMaxAllocSizeBytes();
        }
        if (caps.getMaxStorageBufferRange() > 0) {
            limit = Math.min(limit, caps.getMaxStorageBufferRange());
        }
        if (bytes > limit) {
            throw new GpuOutOfMemoryException("scratch slot " + name + " needs " + bytes + " bytes (" + floats
                    + " floats for ubatch " + ubatch + "), above the device limit of " + limit
                    + " bytes per buffer; lower neuralfree.ubatch", bytes, limit, String.valueOf(backend.backendType()));
        }
        GpuBuffer b = backend.allocateBuffer(bytes);
        owned.add(b);
        allocatedBytes += bytes;
        return ScratchSlot.of(name, b, 0, floats);
    }

    private static void checkName(String name) {
        if (name == null || name.isEmpty()) {
            throw new IllegalArgumentException("scratch name must not be empty");
        }
        if (name.startsWith(RESERVED_PREFIX) || PARTIALS.equals(name)) {
            throw new IllegalArgumentException("scratch name '" + name + "' is reserved");
        }
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("GPU scratch is closed");
        }
    }

    /**
     * Releases every buffer. The caller first waits for every queue that used them ({@code finish()}), so no
     * enqueued kernel still accesses a released buffer. Idempotent.
     */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        RuntimeException first = Closeables.closeAll(owned, null);
        owned.clear();
        temps.clear();
        if (first != null) {
            throw first;
        }
    }

    @Override
    public String toString() {
        return "GpuScratch{ubatch=" + ubatch + ", slots=" + slots.keySet() + ", temps=" + temps.keySet()
                + ", bytes=" + allocatedBytes + '}';
    }
}

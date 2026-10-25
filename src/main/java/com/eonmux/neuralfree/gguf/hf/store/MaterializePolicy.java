package com.eonmux.neuralfree.gguf.hf.store;

import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.ModelSourceOptions.Materialize;
import com.eonmux.neuralfree.gguf.StoreUse;
import com.eonmux.neuralfree.gguf.hf.tensor.PlannedTensor;
import com.sun.management.HotSpotDiagnosticMXBean;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.lang.management.BufferPoolMXBean;
import java.lang.management.ManagementFactory;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Where one reader of a Hugging Face model source keeps each derived tensor: produced on every read
 * ({@link Mode#NONE}), in direct memory ({@link Mode#MEMORY}) or in a sidecar file ({@link Mode#DISK}), either for the
 * life of the reader ({@link Decision#keep()}) or as the reader's single whole-tensor window, which is replaced when
 * another tensor is materialized. With {@link Materialize#AUTO}:
 * <table>
 *   <caption>AUTO decisions</caption>
 *   <tr><th>Reader</th><th>Row-local producer</th><th>Non-row-local producer</th></tr>
 *   <tr><td>{@link Reader#RESIDENT_UPLOAD} (one pass)</td><td>NONE</td><td>window: MEMORY when the tensor fits the
 *       budget, else DISK</td></tr>
 *   <tr><td>{@link Reader#STREAMING} (re-read per token)</td><td>DISK, kept</td><td>DISK, kept</td></tr>
 *   <tr><td>{@link Reader#CPU_REGIONS} (the reader keeps the buffers)</td><td colspan="2">MEMORY when every derived
 *       tensor together fits the budget, else DISK (kept); MEMORY regions of a row-local producer are produced per
 *       region, those of a non-row-local producer are slices of a whole-tensor window</td></tr>
 * </table>
 * Explicit MEMORY and DISK apply to every derived tensor (kept by {@link Reader#STREAMING}, a window for
 * {@link Reader#RESIDENT_UPLOAD}, as above for {@link Reader#CPU_REGIONS}). Explicit NONE produces row-local tensors on
 * every read and keeps the whole-tensor window for non-row-local ones; it is invalid for {@link Reader#CPU_REGIONS}.
 * Every tensor is therefore produced at most once per pass of a reader, so the bytes read from its sources stay within
 * 1.1 times their size.
 *
 * <p>The budget is {@code min(memoryBudget, directHeadroom)}, the headroom being {@code MaxDirectMemorySize} (0 meaning
 * {@link Runtime#maxMemory()}) minus the direct memory in use when the policy is made. Each distinct decision is logged
 * once at INFO with the reader, the mode, the tensor count, the produced bytes and the budget. Immutable.</p>
 */
public final class MaterializePolicy {

    private static final Logger LOG = LogManager.getLogger(MaterializePolicy.class);

    /** Who reads the derived tensors. */
    public enum Reader {
        /** {@code createMappedStore(RESIDENT_UPLOAD)}: one pass, the GPU upload. */
        RESIDENT_UPLOAD,
        /** {@code createMappedStore(STREAMING)}: the same tensors re-read for every token. */
        STREAMING,
        /** {@code openRegions}: the CPU weights, which keep every region they receive. */
        CPU_REGIONS
    }

    /** Where a derived tensor's bytes are kept. */
    public enum Mode {
        /** Produced on every read. */
        NONE,
        /** Produced into direct memory. */
        MEMORY,
        /** Produced into a sidecar file and mapped read-only. */
        DISK
    }

    /** The mode of one tensor and whether it is kept for the reader's life or held as its window. Immutable. */
    public static final class Decision {
        private final Mode mode;
        private final boolean keep;

        private Decision(Mode mode, boolean keep) {
            this.mode = mode;
            this.keep = keep;
        }

        public Mode mode() {
            return mode;
        }

        /** True when the materialized bytes stay for the reader's life; false for the whole-tensor window. */
        public boolean keep() {
            return keep;
        }

        @Override
        public boolean equals(Object o) {
            return o instanceof Decision && ((Decision) o).mode == mode && ((Decision) o).keep == keep;
        }

        @Override
        public int hashCode() {
            return mode.hashCode() * 2 + (keep ? 1 : 0);
        }

        @Override
        public String toString() {
            return mode == Mode.NONE ? "NONE" : mode + (keep ? " (kept)" : " (window)");
        }
    }

    private static final Decision NONE = new Decision(Mode.NONE, false);
    private static final Decision MEMORY_WINDOW = new Decision(Mode.MEMORY, false);
    private static final Decision MEMORY_KEPT = new Decision(Mode.MEMORY, true);
    private static final Decision DISK_WINDOW = new Decision(Mode.DISK, false);
    private static final Decision DISK_KEPT = new Decision(Mode.DISK, true);

    private final Reader reader;
    private final List<Decision> decisions;
    private final long budget;

    private MaterializePolicy(Reader reader, List<Decision> decisions, long budget) {
        this.reader = reader;
        this.decisions = decisions;
        this.budget = budget;
    }

    /** The reader of a weight store of {@code use}. */
    public static Reader readerOf(StoreUse use) {
        switch (Objects.requireNonNull(use, "use")) {
            case RESIDENT_UPLOAD:
                return Reader.RESIDENT_UPLOAD;
            case STREAMING:
                return Reader.STREAMING;
            default:
                throw new IllegalArgumentException("unhandled store use " + use);
        }
    }

    /**
     * The decisions of {@code reader} for the derived tensors {@code produced} (in the order of
     * {@link ProducedSegments#tensors()}) under the materialization setting and memory budget of {@code o}, with the
     * direct-memory headroom of this JVM now.
     *
     * @throws IllegalArgumentException for {@link Materialize#NONE} with {@link Reader#CPU_REGIONS}
     */
    public static MaterializePolicy of(Reader reader, ModelSourceOptions o, List<PlannedTensor> produced) {
        Objects.requireNonNull(o, "o");
        return of(reader, o.materialize(), o.memoryBudget(), produced, directHeadroom());
    }

    /**
     * As {@link #of(Reader, ModelSourceOptions, List)} with an explicit direct-memory headroom.
     *
     * @throws IllegalArgumentException for {@link Materialize#NONE} with {@link Reader#CPU_REGIONS}, or a negative
     *                                  budget or headroom
     */
    public static MaterializePolicy of(Reader reader, Materialize setting, long memoryBudget,
                                       List<PlannedTensor> produced, long headroom) {
        Objects.requireNonNull(reader, "reader");
        Objects.requireNonNull(setting, "setting");
        Objects.requireNonNull(produced, "produced");
        if (memoryBudget < 0 || headroom < 0) {
            throw new IllegalArgumentException("memory budget " + memoryBudget + " and headroom " + headroom
                    + " must not be negative");
        }
        if (reader == Reader.CPU_REGIONS && setting == Materialize.NONE) {
            throw new IllegalArgumentException("materialization NONE (" + ModelSourceOptions.PROP_MATERIALIZE
                    + "=none) is not valid for CPU weights, which keep the regions they read; use auto, memory or"
                    + " disk");
        }
        long budget = Math.min(memoryBudget, headroom);
        long total = 0;
        for (PlannedTensor t : produced) {
            total = Math.addExact(total, t.byteSize());
        }
        List<Decision> out = new ArrayList<>(produced.size());
        for (PlannedTensor t : produced) {
            out.add(decide(reader, setting, budget, total, t));
        }
        MaterializePolicy p = new MaterializePolicy(reader, Collections.unmodifiableList(out), budget);
        p.log(produced);
        return p;
    }

    private static Decision decide(Reader reader, Materialize setting, long budget, long total, PlannedTensor t) {
        boolean rowLocal = t.source().rowLocal();
        Decision window = t.byteSize() <= budget ? MEMORY_WINDOW : DISK_WINDOW;
        switch (reader) {
            case RESIDENT_UPLOAD:
                switch (setting) {
                    case MEMORY: return MEMORY_WINDOW;
                    case DISK: return DISK_WINDOW;
                    default: return rowLocal ? NONE : window;
                }
            case STREAMING:
                switch (setting) {
                    case MEMORY: return MEMORY_KEPT;
                    case DISK: return DISK_KEPT;
                    case NONE: return rowLocal ? NONE : window;
                    default: return DISK_KEPT;
                }
            case CPU_REGIONS:
                switch (setting) {
                    case MEMORY: return MEMORY_WINDOW;
                    case DISK: return DISK_KEPT;
                    default: return total <= budget ? MEMORY_WINDOW : DISK_KEPT;
                }
            default:
                throw new IllegalStateException("unhandled reader " + reader);
        }
    }

    private void log(List<PlannedTensor> produced) {
        Map<Decision, long[]> groups = new LinkedHashMap<>();
        for (int i = 0; i < produced.size(); i++) {
            long[] g = groups.computeIfAbsent(decisions.get(i), d -> new long[2]);
            g[0]++;
            g[1] += produced.get(i).byteSize();
        }
        for (Map.Entry<Decision, long[]> e : groups.entrySet()) {
            LOG.info("Derived tensors for {}: {} for {} tensor(s), {} bytes produced; memory budget {} bytes",
                    reader, e.getKey(), e.getValue()[0], e.getValue()[1], budget);
        }
    }

    /** The reader the decisions are for. */
    public Reader reader() {
        return reader;
    }

    /** The number of derived tensors. */
    public int size() {
        return decisions.size();
    }

    /** The decision for derived tensor {@code index}. */
    public Decision decision(int index) {
        return decisions.get(index);
    }

    /** The effective memory budget: {@code min(memoryBudget, directHeadroom)}. */
    public long budget() {
        return budget;
    }

    /**
     * The direct memory this JVM can still allocate: {@code MaxDirectMemorySize} ({@code 0} meaning
     * {@link Runtime#maxMemory()}) minus the memory of the {@code direct} buffer pool in use; never negative.
     */
    public static long directHeadroom() {
        long used = 0;
        for (BufferPoolMXBean pool : ManagementFactory.getPlatformMXBeans(BufferPoolMXBean.class)) {
            if ("direct".equals(pool.getName())) {
                used = Math.max(0, pool.getMemoryUsed());
            }
        }
        return Math.max(0, maxDirectMemory() - used);
    }

    private static long maxDirectMemory() {
        HotSpotDiagnosticMXBean hotspot = ManagementFactory.getPlatformMXBean(HotSpotDiagnosticMXBean.class);
        if (hotspot == null) {
            LOG.debug("No HotSpot diagnostic bean; the direct memory limit is the maximum heap size");
            return Runtime.getRuntime().maxMemory();
        }
        String value;
        try {
            value = hotspot.getVMOption("MaxDirectMemorySize").getValue();
        } catch (IllegalArgumentException noSuchOption) {
            LOG.debug("The JVM has no MaxDirectMemorySize option ({}); the direct memory limit is the maximum heap"
                    + " size", noSuchOption.getMessage());
            return Runtime.getRuntime().maxMemory();
        }
        long n;
        try {
            n = Long.parseLong(value.trim());
        } catch (NumberFormatException e) {
            LOG.warn("MaxDirectMemorySize '{}' is not a number; the direct memory limit is the maximum heap size",
                    value);
            return Runtime.getRuntime().maxMemory();
        }
        return n > 0 ? n : Runtime.getRuntime().maxMemory();
    }

    @Override
    public String toString() {
        return "MaterializePolicy{" + reader + ", " + decisions.size() + " tensors, budget " + budget + "}";
    }
}

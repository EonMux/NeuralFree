package com.eonmux.neuralfree.gguf.cpu.repack;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuSimd;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuVecDot;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec.MatvecPlan;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Collections;
import java.util.EnumSet;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Properties;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Weight repacking for the CPU matvec, the twin of llama.cpp {@code ggml_backend_cpu_repack}
 * (ggml/src/ggml-cpu/repack.cpp): matmul weights whose layout the Vector API kernels exploit are copied once
 * into an interleaved layout and their products run over the copy through a {@link MatvecPlan}. The copy is made
 * when the weight's first product asks for it, not when the tensor is mapped, so only weights that are really
 * used as matmuls are copied and no lookup table (which keeps reading its mapped rows) is duplicated.
 * <p>
 * The covered layouts are the ones {@code ggml-cpu/repack.cpp} defines for the types the CPU kernels support:
 * {@link Q4_0x8} for Q4_0, {@link Q8_0x4} for Q8_0 and {@link Q4_Kx8} for Q4_K. A weight qualifies when its rows
 * split into whole groups of the layout, the Vector API engine is available ({@link CpuSimd#available()}) and the
 * type is selected by {@value #PROP_REPACK}: {@code auto} (the default) selects the types whose repacked kernels
 * measured faster than the plain Vector API kernels, {@code on} selects every type and {@code off} none. Only
 * {@code auto} is the default because no covered type wins its measurements outright: on this CPU the Q4_0x8
 * kernel dots a single token 1.1-1.5x faster than {@link com.eonmux.neuralfree.gguf.cpu.quant.SimdQ4_0} but its
 * batched product is about half as fast, and Q8_0x4 and Q4_Kx8 are no faster than the plain kernels at all,
 * while every copy costs the weight's own bytes of heap.
 * <p>
 * <b>Memory.</b> A copy is an on-heap {@code byte[]} of the weight's own size; the mapped original is not read
 * again once the copy exists, so its file-backed pages can be reclaimed under memory pressure. Copies are
 * budgeted against the heap headroom measured when the first one is asked for: free heap minus a reserve of
 * {@value #RESERVE_FRACTION} of it, at least {@value #MIN_RESERVE_BYTES} bytes, for the activations and
 * temporary arrays the forward pass allocates after the KV cache (whose allocation, with the model's
 * preallocated scratch, precedes the first product because weights are mapped lazily). A weight whose copy does
 * not fit stays on the plain kernels, so repacking never fails or endangers a load; a copy whose allocation
 * nevertheless fails is dropped the same way. {@value #PROP_BUDGET} caps the copies explicitly, in bytes.
 * <p>
 * One summary line, logged when the owning weights close, reports the weights and bytes repacked versus kept in
 * place and why. Thread-safe.
 */
public final class CpuRepack {

    private static final Logger LOGGER = LogManager.getLogger(CpuRepack.class);

    /** {@code auto} (default) picks the measured types, {@code on} every covered type, {@code off} none. */
    public static final String PROP_REPACK = "neuralfree.cpu.repack";

    /** Bytes the copies may occupy; absent or negative means the heap headroom is measured automatically. */
    public static final String PROP_BUDGET = "neuralfree.cpu.repack.budget";

    /** Share of the free heap kept out of the repack budget for the forward pass's later allocations. */
    static final double RESERVE_FRACTION = 0.25;

    /** Smallest reserve kept out of the repack budget. */
    static final long MIN_RESERVE_BYTES = 128L << 20;

    private static final long MIB = 1024L * 1024L;

    /**
     * Types {@code auto} repacks: none. Repacking only pays when the interleaved layout beats the plain one for
     * both the one-token and the batched product, and no covered type does: Q4_0x8 is 1.1-1.5x faster per token
     * but its batched product is about half the plain kernel's, while Q8_0x4 and Q4_Kx8 measure no faster than
     * the plain kernels. {@code on} still repacks every type, for a CPU whose kernels do win.
     */
    private static final Set<GGMLType> AUTO_TYPES =
            Collections.unmodifiableSet(EnumSet.noneOf(GGMLType.class));

    /** How {@value #PROP_REPACK} picks the types to repack. */
    public enum Mode {
        /** Never repack. */
        OFF,
        /** Repack only the types whose repacked kernels measured faster than the plain ones. */
        AUTO,
        /** Repack every covered type, budget permitting. */
        ON;

        /**
         * The mode named by {@value #PROP_REPACK}: {@code auto} when absent.
         *
         * @throws IllegalArgumentException when the value is not {@code auto}, {@code on} or {@code off}
         */
        public static Mode fromProperties(Properties p) {
            Objects.requireNonNull(p, "properties");
            String v = p.getProperty(PROP_REPACK);
            if (v == null || v.trim().isEmpty()) {
                return AUTO;
            }
            String s = v.trim().toLowerCase(Locale.ROOT);
            if (s.equals("auto")) {
                return AUTO;
            }
            if (s.equals("on")) {
                return ON;
            }
            if (s.equals("off")) {
                return OFF;
            }
            throw new IllegalArgumentException(PROP_REPACK + "=" + v + " is invalid; use auto, on or off");
        }
    }

    /** Per-weight plans, one repacked view each; entries live as long as the owning weights. */
    private final ConcurrentHashMap<CpuTensorRef, MatvecPlan> plans = new ConcurrentHashMap<>();

    /** Weights a decision already kept in place, so a repeated product does not re-decide or recount them. */
    private final Set<CpuTensorRef> declined = ConcurrentHashMap.newKeySet();

    private final Mode mode;
    private final boolean fixedBudget;
    private final AtomicBoolean summaryLogged = new AtomicBoolean();
    private final Object budgetLock = new Object();

    /** Why weights were kept in place, and how many of each. Guarded by {@link #budgetLock}. */
    private final Map<String, Integer> reasons = new LinkedHashMap<>();

    /** Fixed cap from {@value #PROP_BUDGET}; sampled heap headroom when {@link #fixedBudget} is false. */
    private long cap = -1;
    private boolean capSampled;
    private boolean gcRetried;
    private long repackedBytes;
    private long keptBytes;
    private int repackedWeights;
    private int keptWeights;

    /** Repacker configured by the system properties, with the automatic heap budget. */
    public CpuRepack() {
        this(Mode.fromProperties(System.getProperties()), budgetFrom(System.getProperties()));
    }

    /** Repacker of {@code mode} with the automatic heap budget. */
    public CpuRepack(Mode mode) {
        this(mode, -1L);
    }

    /**
     * Repacker of {@code mode} with an explicit allowance for the copies.
     *
     * @param budgetBytes bytes the copies may occupy, or a negative value to measure the heap headroom
     */
    public CpuRepack(Mode mode, long budgetBytes) {
        this.mode = Objects.requireNonNull(mode, "mode");
        this.fixedBudget = budgetBytes >= 0;
        this.cap = budgetBytes;
        this.capSampled = fixedBudget;
    }

    /** The configured mode. */
    public Mode mode() {
        return mode;
    }

    /** The bytes the copies may occupy, or -1 when the automatic headroom has not been measured yet. */
    public long budgetBytes() {
        synchronized (budgetLock) {
            return capSampled ? cap : -1L;
        }
    }

    /** Bytes of the weights repacked so far. */
    public long repackedBytes() {
        synchronized (budgetLock) {
            return repackedBytes;
        }
    }

    /** Bytes of the qualifying weights kept in place so far. */
    public long keptBytes() {
        synchronized (budgetLock) {
            return keptBytes;
        }
    }

    /**
     * The plan of {@code w}: its repacked copy and the Vector API kernel of the layout, or null when {@code w} is
     * not repacked (type without a layout, shape that does not split into groups, engine unavailable, type not
     * selected, or no budget left). The first call for a weight performs the copy; the decision and its reason
     * are counted for the summary.
     */
    public MatvecPlan planFor(CpuTensorRef w) {
        if (w == null) {
            return null;
        }
        MatvecPlan cached = plans.get(w);
        if (cached != null || declined.contains(w)) {
            return cached;
        }
        GGMLType type = w.type();
        CpuVecDot kernel = groupElements(type) == 0 ? null : CpuSimd.repacked(type);
        if (kernel == null) {
            return null;
        }
        synchronized (budgetLock) {
            cached = plans.get(w);
            if (cached != null) {
                return cached;
            }
            if (declined.contains(w)) {
                return null;
            }
            if (!selected(type)) {
                decline(w, mode == Mode.OFF ? "repacking off" : "type not selected");
                return null;
            }
            if (!fits(w) || w.rows() * w.rowBytes() > Integer.MAX_VALUE - 8L) {
                decline(w, "shape");
                return null;
            }
            long bytes = sizeOf(w);
            if (bytes > availableBytes()) {
                decline(w, "over budget");
                return null;
            }
            MatvecPlan plan;
            try {
                plan = repack(w, kernel);
            } catch (OutOfMemoryError exhausted) {
                // a copy that cannot be allocated is dropped; the weight keeps the plain kernel and the load goes on
                try {
                    decline(w, "heap exhausted");
                } catch (OutOfMemoryError ignored) {
                    // the bookkeeping is best effort too: the weight still falls back to the plain kernel
                }
                return null;
            }
            plans.put(w, plan);
            repackedWeights++;
            repackedBytes += bytes;
            return plan;
        }
    }

    /**
     * Keeps {@code w} in place for {@code reason} and drops a plan it already had: a tensor read row by row (a
     * token, position or type lookup table) must keep its mapped rows, so copying it would duplicate it without
     * replacing a single read.
     */
    public void keepInPlace(CpuTensorRef w, String reason) {
        if (w == null || declined.contains(w)) {
            return;
        }
        synchronized (budgetLock) {
            if (!declined.add(w)) {
                return;
            }
            MatvecPlan existing = plans.remove(w);
            if (existing != null) {
                repackedWeights--;
                repackedBytes -= sizeOf(w);
            }
            if (selected(w.type()) && groupElements(w.type()) != 0) {
                keptWeights++;
                keptBytes += sizeOf(w);
                reasons.merge(reason, 1, Integer::sum);
            }
        }
    }

    /**
     * Drops the plan and the decline decision of {@code w}: a live weight update replaced the tensor's contents, so
     * its repacked copy (and its reason to stay plain) belong to bytes the model no longer serves. The replacement
     * decides its own plan when its first product asks for it.
     */
    public void discard(CpuTensorRef w) {
        if (w == null) {
            return;
        }
        synchronized (budgetLock) {
            MatvecPlan removed = plans.remove(w);
            declined.remove(w);
            if (removed != null) {
                repackedWeights--;
                repackedBytes -= sizeOf(w);
            }
        }
    }

    /** Whether {@code mode} repacks {@code type}. */
    private boolean selected(GGMLType type) {
        if (mode == Mode.OFF) {
            return false;
        }
        if (mode == Mode.ON) {
            return true;
        }
        return AUTO_TYPES.contains(type);
    }

    /** Elements in one repack group of {@code type}, or 0 when the type has no layout. */
    static int groupElements(GGMLType type) {
        if (type == GGMLType.Q4_0) {
            return Q4_0x8.ELEMENTS;
        }
        if (type == GGMLType.Q8_0) {
            return Q8_0x4.ELEMENTS;
        }
        if (type == GGMLType.Q4_K) {
            return Q4_Kx8.ELEMENTS;
        }
        return 0;
    }

    /** Bytes one repack group of {@code type} occupies. */
    private static int groupBytes(GGMLType type) {
        if (type == GGMLType.Q4_0) {
            return Q4_0x8.BYTES;
        }
        if (type == GGMLType.Q8_0) {
            return Q8_0x4.BYTES;
        }
        return Q4_Kx8.BYTES;
    }

    /** True when a row of {@code w} splits into whole groups of its type's layout. */
    private static boolean fits(CpuTensorRef w) {
        if (w.type() == GGMLType.Q4_0) {
            return Q4_0x8.fits(w.cols());
        }
        if (w.type() == GGMLType.Q8_0) {
            return Q8_0x4.fits(w.cols());
        }
        return Q4_Kx8.fits(w.cols());
    }

    /** Bytes a repacked copy of {@code w} occupies. */
    private static long sizeOf(CpuTensorRef w) {
        return w.rows() * w.rowBytes();
    }

    /** Bytes the next copy may use, under the lock. */
    private long availableBytes() {
        if (fixedBudget) {
            return cap == Long.MAX_VALUE ? Long.MAX_VALUE : Math.max(0L, cap - repackedBytes);
        }
        if (!capSampled) {
            cap = heapHeadroom();
            capSampled = true;
        }
        if (cap <= 0 && !gcRetried) {
            // the load's own temporaries may still be unreachable: collect once before giving up on the copies
            gcRetried = true;
            System.gc();
            cap = heapHeadroom();
        }
        return cap == Long.MAX_VALUE ? Long.MAX_VALUE : Math.max(0L, cap - repackedBytes);
    }

    /** Free heap minus the reserve the forward pass keeps, or {@link Long#MAX_VALUE} for an unbounded heap. */
    private static long heapHeadroom() {
        Runtime rt = Runtime.getRuntime();
        long max = rt.maxMemory();
        if (max == Long.MAX_VALUE) {
            return Long.MAX_VALUE;
        }
        long free = Math.max(0L, max - (rt.totalMemory() - rt.freeMemory()));
        long reserve = Math.max(MIN_RESERVE_BYTES, (long) (free * RESERVE_FRACTION));
        return Math.max(0L, free - reserve);
    }

    /** Records that {@code w} was kept in place; called under the lock. */
    private void decline(CpuTensorRef w, String reason) {
        if (declined.add(w)) {
            keptWeights++;
            keptBytes += sizeOf(w);
            reasons.merge(reason, 1, Integer::sum);
        }
    }

    /** Copies {@code w} into its repacked on-heap view. */
    private static MatvecPlan repack(CpuTensorRef w, CpuVecDot kernel) {
        long rowBytes = w.rowBytes();
        long rows = w.rows();
        int groups = (int) (w.cols() / groupElements(w.type()));
        int packedRowBytes = groups * groupBytes(w.type());
        byte[] packed = new byte[(int) (rows * rowBytes)];
        byte[] scratch = new byte[(int) rowBytes];
        for (int r = 0; r < rows; r++) {
            int dstOff = (int) (r * rowBytes);
            ByteBuffer row = w.rowBuffer(r).duplicate();
            row.position(w.rowOffset(r));
            row.get(scratch, 0, packedRowBytes);
            if (w.type() == GGMLType.Q4_0) {
                Q4_0x8.repackRow(scratch, 0, packed, dstOff, groups);
            } else if (w.type() == GGMLType.Q8_0) {
                Q8_0x4.repackRow(scratch, 0, packed, dstOff, groups);
            } else {
                Q4_Kx8.repackRow(scratch, 0, packed, dstOff, groups);
            }
        }
        // the copy keeps the original row stride, so a strided weight repacks to the same row geometry
        CpuTensorRef view = new CpuTensorRef(w.type(),
                new ByteBuffer[]{ByteBuffer.wrap(packed).order(ByteOrder.LITTLE_ENDIAN)},
                new long[]{0L}, rows, w.cols(), rowBytes);
        return new MatvecPlan(view, kernel);
    }

    /** The number of weights currently repacked. */
    public int repackedCount() {
        return plans.size();
    }

    /**
     * The one summary line: the weights and bytes repacked versus kept in place, the budget and the reasons.
     * Also logged once by {@link #clear()}.
     */
    public String summary() {
        synchronized (budgetLock) {
            return "CPU repack (" + mode.name().toLowerCase(Locale.ROOT) + "): repacked " + repackedWeights
                    + " weight(s), " + bytes(repackedBytes) + "; kept in place " + keptWeights + " weight(s), "
                    + bytes(keptBytes) + " (" + budgetText() + "; " + reasonText() + ")";
        }
    }

    /** Logs {@link #summary()} once; later calls do nothing. */
    public void logSummary() {
        if (summaryLogged.compareAndSet(false, true)) {
            LOGGER.info(summary());
        }
    }

    private String budgetText() {
        if (fixedBudget) {
            return cap == Long.MAX_VALUE ? "unbounded budget" : "budget " + bytes(cap);
        }
        return capSampled ? "heap headroom " + bytes(cap) : "heap headroom automatic";
    }

    private String reasonText() {
        if (reasons.isEmpty()) {
            return "no reason recorded";
        }
        StringBuilder sb = new StringBuilder();
        for (Map.Entry<String, Integer> e : reasons.entrySet()) {
            if (sb.length() > 0) {
                sb.append(", ");
            }
            sb.append(e.getKey()).append(' ').append(e.getValue());
        }
        return sb.toString();
    }

    private static String bytes(long n) {
        if (n >= MIB) {
            return String.format(Locale.ROOT, "%.1f MiB", n / (double) MIB);
        }
        return n + " B";
    }

    /** Forgets every plan and the summary counts; logs the summary first. */
    public void clear() {
        logSummary();
        synchronized (budgetLock) {
            plans.clear();
            declined.clear();
            reasons.clear();
            repackedBytes = 0;
            keptBytes = 0;
            repackedWeights = 0;
            keptWeights = 0;
            cap = fixedBudget ? cap : -1;
            capSampled = fixedBudget;
            gcRetried = false;
        }
    }

    /** The fixed budget of {@value #PROP_BUDGET}: the byte count, or -1 when the property is absent. */
    private static long budgetFrom(Properties p) {
        String v = p.getProperty(PROP_BUDGET);
        if (v == null || v.trim().isEmpty()) {
            return -1L;
        }
        try {
            long b = Long.parseLong(v.trim());
            if (b < 0) {
                throw new NumberFormatException("negative");
            }
            return b;
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(PROP_BUDGET + "=" + v
                    + " is not a non-negative byte count; use a number of bytes or leave it unset");
        }
    }

    @Override
    public String toString() {
        return "CpuRepack{" + mode.name().toLowerCase(Locale.ROOT) + ", " + repackedCount() + " weight(s)}";
    }
}

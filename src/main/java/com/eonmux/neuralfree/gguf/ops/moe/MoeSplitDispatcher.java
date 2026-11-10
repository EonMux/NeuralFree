package com.eonmux.neuralfree.gguf.ops.moe;

import java.util.Arrays;
import java.util.Objects;

/**
 * The split dispatcher of a hybrid CPU+GPU MoE layer: per step, whether the routed experts that are not in the
 * GPU expert cache move there when their PCIe copies end the MoE window sooner than computing their rows on the
 * host. Avalon's dispatcher (ggml/src/ggml-cuda/moe-cache.cu {@code moe_hosted_split}, :6348-6520), with its
 * fitted cost model (:6349-6368):
 * <ul>
 *   <li>host time of the rows that stay on the CPU: a fixed term plus one term per distinct expert and one per
 *       row, divided by the host team's thread count, plus a penalty per tensor copied meanwhile (the copy reads
 *       the same RAM);</li>
 *   <li>arrival of a moved expert's tensors: a per-tensor start and per-tensor term plus bytes over the link's
 *       copy rate;</li>
 *   <li>GPU work around the wait for the moved experts: a fixed term plus per-row terms before and after.</li>
 * </ul>
 * The window ends at the later of the host rows and the GPU work. Moving one expert, then one pair of experts, is
 * evaluated against the do-nothing window, and the best move with a positive gain is taken (a pair only when it
 * beats every single expert). Past a best pair, further experts are added one at a time, each the one that ends the
 * window soonest, while it ends the window sooner than the best move so far, never more than the cache's free
 * slots. Steps with fewer than two tokens never split (the reference's
 * rule), and everything is decided before any work is enqueued, so the model never sees partial moves. Immutable
 * and stateless; the cache keeps the residency.
 */
public final class MoeSplitDispatcher {

    /** Link copy rate of the fitted model: PCIe 3.0 x16 at 86% (moe-cache.cu:6363, 13.6 GB/s). */
    public static final double PCIE_BYTES_PER_US = 13600.0;

    /** Host fixed term of the fitted model (microseconds, :6350). */
    public static final double CPU_FIXED_US = 2.0;
    /** Host term per distinct expert (:6350). */
    public static final double CPU_EXPERT_US = 55.8;
    /** Host term per routed row (:6350). */
    public static final double CPU_ROW_US = 14.5;
    /** Host team size the per-expert and per-row terms were fitted with (:6354). */
    public static final double CPU_FIT_THREADS = 8.0;
    /** Host penalty per tensor copied meanwhile (:6351). */
    public static final double CPU_COPY_US = 5.2;
    /** Per-tensor start of a copy (:6352). */
    public static final double COPY_TENSOR_US = 11.4;
    /** GPU term before the wait for the moved experts (:6353). */
    public static final double GPU_BEFORE_US = 23.3;
    /** GPU term per resident row (:6353). */
    public static final double GPU_ROW_US = 2.97;
    /** GPU term after the wait (:6353). */
    public static final double GPU_AFTER_US = 66.4;
    /** GPU term per moved row (:6353). */
    public static final double GPU_MOVED_ROW_US = 1.5;

    /** Rates and calibration of the cost model. Immutable. */
    public static final class Model {
        private final double pcieBytesPerUs;
        private final double cpuScale;
        private final int cpuThreads;

        private Model(double pcieBytesPerUs, double cpuScale, int cpuThreads) {
            if (!(pcieBytesPerUs > 0) || !Double.isFinite(pcieBytesPerUs)) {
                throw new IllegalArgumentException("PCIe copy rate must be positive and finite, got "
                        + pcieBytesPerUs);
            }
            if (!(cpuScale > 0) || !Double.isFinite(cpuScale)) {
                throw new IllegalArgumentException("CPU calibration must be positive and finite, got " + cpuScale);
            }
            if (cpuThreads < 1) {
                throw new IllegalArgumentException("CPU thread count must be positive, got " + cpuThreads);
            }
            this.pcieBytesPerUs = pcieBytesPerUs;
            this.cpuScale = cpuScale;
            this.cpuThreads = cpuThreads;
        }

        /** The fitted model at the reference rates, calibrated to the given host team. */
        public static Model of(int cpuThreads) {
            return new Model(PCIE_BYTES_PER_US, 1.0, cpuThreads);
        }

        /** A model with explicit rates and calibration. */
        public static Model of(double pcieBytesPerUs, double cpuScale, int cpuThreads) {
            return new Model(pcieBytesPerUs, cpuScale, cpuThreads);
        }

        /** Host time of {@code rows} rows over {@code distinct} experts while {@code tensors} copy, in us. */
        double cpuUs(int distinct, int rows, int tensors) {
            if (distinct <= 0) {
                return 0.0;
            }
            double share = CPU_FIT_THREADS / cpuThreads;
            return cpuScale * (CPU_FIXED_US + (CPU_EXPERT_US * distinct + CPU_ROW_US * rows) * share
                    + CPU_COPY_US * tensors);
        }

        /** Arrival of {@code tensors} copies of {@code bytes} bytes in total, in us. */
        double copiesUs(long bytes, int tensors) {
            if (tensors <= 0) {
                return 0.0;
            }
            return COPY_TENSOR_US + tensors * COPY_TENSOR_US + bytes / pcieBytesPerUs;
        }

        /**
         * End of the window with {@code distinct} experts and {@code rows} rows left on the host,
         * {@code movedRows} moved rows, and {@code copiesUs} of copy time, in us.
         */
        double windowUs(int distinct, int rows, int movedRows, int tensors, double copiesUs, int residentRows) {
            double gpuBefore = GPU_BEFORE_US + GPU_ROW_US * residentRows;
            double gpuAfter = GPU_AFTER_US + GPU_MOVED_ROW_US * movedRows;
            double arrival = tensors == 0 ? 0.0 : copiesUs;
            return Math.max(cpuUs(distinct, rows, tensors), Math.max(gpuBefore, arrival) + gpuAfter);
        }

        @Override
        public String toString() {
            return "Model{pcie=" + pcieBytesPerUs + " B/us, cpuScale=" + cpuScale + ", threads=" + cpuThreads + "}";
        }
    }

    /** One step's move: the experts to copy into the cache now, in the order they gain. Immutable. */
    public static final class Plan {
        /** Experts to migrate (copy into the cache this step). */
        private final int[] migrate;

        private Plan(int[] migrate) {
            this.migrate = migrate;
        }

        /** Experts to migrate, in the order they gain (unmodifiable copy). */
        public int[] migrate() {
            return migrate.clone();
        }

        /** Whether any expert moves. */
        public boolean isEmpty() {
            return migrate.length == 0;
        }

        @Override
        public String toString() {
            return "Plan{migrate=" + Arrays.toString(migrate) + "}";
        }
    }

    private final Model model;

    /** A dispatcher over {@code model}. */
    public MoeSplitDispatcher(Model model) {
        this.model = Objects.requireNonNull(model, "model");
    }

    /**
     * Decides the move of one step of a hosted layer.
     *
     * @param nTokens           tokens of the step; fewer than two never split
     * @param rowsPerExpert     routed pair rows per expert, index = expert id
     * @param cached            which experts hold a cache slot already
     * @param freeSlots         cache slots the move may take (bounded by the cache's capacity)
     * @param expertCopyBytes   bytes of one expert's tensors on the link
     * @param tensorsPerExpert  device tensors copied per expert (three for separate gate/up/down, two fused)
     */
    public Plan decide(int nTokens, int[] rowsPerExpert, boolean[] cached, int freeSlots, long expertCopyBytes,
                       int tensorsPerExpert) {
        Objects.requireNonNull(rowsPerExpert, "rowsPerExpert");
        Objects.requireNonNull(cached, "cached");
        if (rowsPerExpert.length != cached.length) {
            throw new IllegalArgumentException("row counts cover " + rowsPerExpert.length + " experts but residency "
                    + cached.length);
        }
        if (freeSlots < 0 || tensorsPerExpert < 0) {
            throw new IllegalArgumentException("free slots " + freeSlots + " and tensor count " + tensorsPerExpert
                    + " must be non-negative");
        }
        if (nTokens < 2 || freeSlots == 0 || tensorsPerExpert == 0) {
            return new Plan(new int[0]);
        }
        int residentRows = 0;
        int d = 0;
        for (int e = 0; e < rowsPerExpert.length; e++) {
            if (rowsPerExpert[e] < 0) {
                throw new IllegalArgumentException("expert " + e + " has " + rowsPerExpert[e] + " rows");
            }
            if (cached[e]) {
                residentRows += rowsPerExpert[e];
            } else if (rowsPerExpert[e] > 0) {
                d++;
            }
        }
        int cpuRows = totalRows(rowsPerExpert) - residentRows;
        double all = model.windowUs(d, cpuRows, 0, 0, 0.0, residentRows);
        // Candidates: the uncached experts with rows, as (expert, rows); an expert is moved once.
        int[] cand = new int[d];
        int[] candRows = new int[d];
        int n = 0;
        for (int e = 0; e < rowsPerExpert.length; e++) {
            if (!cached[e] && rowsPerExpert[e] > 0) {
                cand[n] = e;
                candRows[n] = rowsPerExpert[e];
                n++;
            }
        }
        double bestGain = 0.0;
        int[] best = new int[0];
        // The best single expert, then the best pair when two slots are free, each taken only when it ends the
        // window strictly sooner: a pair that merely ties a single expert copies twice for the same window. best
        // holds positions into cand.
        for (int i = 0; i < n; i++) {
            double gain = all - window(1, candRows[i], d, cpuRows, expertCopyBytes, tensorsPerExpert, residentRows);
            if (gain > bestGain) {
                bestGain = gain;
                best = new int[]{i};
            }
        }
        for (int i = 0; i < n && freeSlots >= 2; i++) {
            for (int j = i + 1; j < n; j++) {
                double pairGain = all - window(2, candRows[i] + candRows[j], d, cpuRows, expertCopyBytes,
                        tensorsPerExpert, residentRows);
                if (pairGain > bestGain) {
                    bestGain = pairGain;
                    best = new int[]{i, j};
                }
            }
        }
        if (best.length == 0) {
            return new Plan(new int[0]);
        }
        if (best.length == 1) {
            return new Plan(new int[]{cand[best[0]]});
        }
        // Past the best pair, further experts one at a time: the one that ends the window soonest, while that is
        // sooner than the best move so far and slots remain. Past a best single expert no addition can gain, since
        // every pair it would form already scored no better.
        int cap = Math.min(freeSlots, n);
        int[] pick = Arrays.copyOf(best, cap);
        boolean[] taken = new boolean[n];
        int m = best.length;
        int movedRows = 0;
        for (int k = 0; k < m; k++) {
            taken[pick[k]] = true;
            movedRows += candRows[pick[k]];
        }
        while (m < cap) {
            int add = -1;
            for (int i = 0; i < n; i++) {
                if (taken[i]) {
                    continue;
                }
                double gain = all - window(m + 1, movedRows + candRows[i], d, cpuRows, expertCopyBytes,
                        tensorsPerExpert, residentRows);
                if (gain > bestGain) {
                    bestGain = gain;
                    add = i;
                }
            }
            if (add < 0) {
                break;
            }
            pick[m++] = add;
            taken[add] = true;
            movedRows += candRows[add];
        }
        int[] out = new int[m];
        for (int k = 0; k < m; k++) {
            out[k] = cand[pick[k]];
        }
        return new Plan(out);
    }

    /** The window with {@code m} candidates holding {@code movedRows} rows moved off the host. */
    private double window(int m, int movedRows, int d, int cpuRows, long expertCopyBytes, int tensorsPerExpert,
                          int residentRows) {
        return model.windowUs(d - m, cpuRows - movedRows, movedRows, tensorsPerExpert * m,
                model.copiesUs(expertCopyBytes * m, tensorsPerExpert * m), residentRows);
    }

    private static int totalRows(int[] rowsPerExpert) {
        long total = 0;
        for (int r : rowsPerExpert) {
            total += r;
            if (total > Integer.MAX_VALUE) {
                throw new IllegalArgumentException("routed rows exceed " + Integer.MAX_VALUE);
            }
        }
        return (int) total;
    }

    @Override
    public String toString() {
        return "MoeSplitDispatcher{" + model + "}";
    }
}

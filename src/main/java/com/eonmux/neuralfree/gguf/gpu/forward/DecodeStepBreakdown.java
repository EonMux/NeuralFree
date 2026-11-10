package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.KernelTiming;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Locale;

/**
 * Where the host time of single-sequence decode calls goes, under {@code -Dneuralfree.gpu.stepBreakdown=true}
 * ({@link #PROPERTY}): the GPU driver laps a monotonic clock at the boundaries of each one-token call against
 * memory, so every nanosecond between a call's entry and its return lands in one {@link Phase}, and the time from
 * one call's return to the next contiguous call's entry lands in {@link Phase#BETWEEN}. Each step is counted under
 * the {@link TrunkPath} its trunk took, so the steady state of a trusted decode replay reads apart from the warmup,
 * capture and verify steps.
 *
 * <p>Each lap costs one clock read at a phase boundary, and a step two reads of the thread's allocation counter;
 * nothing is allocated on their behalf, so the run keeps its pace. Each phase reports its mean and its median per
 * step, the medians over the first {@link #MAX_SAMPLES} steps of a path, and the report's columns describe the
 * {@link Report#dominantPath() path most steps took}. The phases average over every step of a path; the gap before a
 * call, the step period and the device's idle time average over the steps that continued the previous call, the only
 * ones with a period. When the queue times its commands ({@code -Dneuralfree.gpu.profile}, {@link KernelTiming}),
 * each step also carries the device time its commands took, read from the queue's report after the call's finish
 * drained them: the period less that device time, over the contiguous steps ({@link Report#meanIdleNanos}), is the
 * device's idle time, what a pipelined or position-independent decode loop could win back. A queue that leaves its
 * graph launches untimed (an OpenCL command buffer, {@link ComputeQueue#timesGraphLaunches()}) reports no device time
 * for a trunk that ran as a graph, so the steps of every path but {@link TrunkPath#DIRECT} then carry no device time
 * and are counted apart ({@link Report#untimedSteps}) instead of reporting their trunk as idle. When the JVM
 * counts allocations per thread, each step also carries the bytes the decode thread allocated
 * between the call's entry and its return ({@link Report#meanAllocatedBytes}). A disabled breakdown
 * ({@link #DISABLED}) records nothing and its calls return at once. Not thread-safe: one decode thread drives it,
 * and a reader takes {@link #report()} between calls.</p>
 */
public final class DecodeStepBreakdown {

    /** The system property that turns the breakdown on, {@code false} by default. */
    public static final String PROPERTY = "neuralfree.gpu.stepBreakdown";

    /**
     * Most steps whose per-phase values a series keeps for the medians; later steps still count toward the means.
     */
    static final int MAX_SAMPLES = 1 << 14;

    /** The intervals of one decode call and the gap before it, in call order. */
    public enum Phase {
        /** The driver's own work around the step: argument checks, the read setup, the event release. */
        CALL("call overhead"),
        /** The step-input uploads: key cells, the key-position mirror, query and rope positions, the position slot. */
        INPUTS("step inputs"),
        /** The token's embedding lookup and its per-layer inputs, enqueued. */
        EMBED("embedding"),
        /** A trusted replay's patch of the captured graph and its one submission. */
        REPLAY_SUBMIT("replay patch + launch"),
        /**
         * A trusted replay's preparation of the next position's replay while the step runs: on CUDA the patch of the
         * idle graph instantiation, on Vulkan the recording of the next ring slot's command buffer.
         */
        REPLAY_PREPARE("replay follow-up prep"),
        /** The trunk outside a trusted replay: its dispatches enqueued, verified against or recorded into a graph. */
        TRUNK("trunk enqueue"),
        /** The output norm, the LM head and its readout kernels, and the readback, enqueued. */
        HEAD("head + readback enqueue"),
        /** The queue's finish: the device's remaining work and the readback's copy into the result. */
        WAIT("wait for the device"),
        /** The result's assembly from the read rows. */
        RESULT("result assembly"),
        /** From the previous contiguous call's return to this call's entry: the caller's sampling and stops. */
        BETWEEN("between calls");

        private final String label;

        Phase(String label) {
            this.label = label;
        }

        /** The phase's report label. */
        public String label() {
            return label;
        }
    }

    /** How a step's trunk reached the device. */
    public enum TrunkPath {
        /** A trusted graph replay: patched and submitted without running the trunk body. */
        REPLAYED("replayed"),
        /** The body ran as a check of the captured graph's patch plan, then the graph was submitted. */
        VERIFIED("verified"),
        /** The body ran without a patch plan and was matched against the captured graph. */
        BODY_REPLAYED("body replay"),
        /** The body was recorded into a new graph and the graph submitted. */
        CAPTURED("captured"),
        /** The body's dispatches were enqueued directly: replay off, warming up, refused or declined. */
        DIRECT("direct");

        private final String label;

        TrunkPath(String label) {
            this.label = label;
        }

        /** The path's report label. */
        public String label() {
            return label;
        }
    }

    private static final int PHASES = Phase.values().length;
    private static final int PATHS = TrunkPath.values().length;
    /**
     * The median series of a path: the phases, then the period, the device time, the idle rest and the bytes the
     * decode thread allocated.
     */
    private static final int PERIOD_SERIES = PHASES;
    private static final int DEVICE_SERIES = PHASES + 1;
    private static final int IDLE_SERIES = PHASES + 2;
    private static final int ALLOC_SERIES = PHASES + 3;
    private static final int MEDIAN_SERIES = PHASES + 4;
    /** The per-thread allocation counter, or null when the JVM does not count allocations per thread. */
    private static final com.sun.management.ThreadMXBean ALLOCATIONS = allocationCounter();

    /** The breakdown that records nothing. */
    public static final DecodeStepBreakdown DISABLED = new DecodeStepBreakdown(false, null, false);

    private final boolean enabled;
    private final KernelTiming timing;
    /** Whether {@link #timing} holds the graph launches, so a step whose trunk ran as a graph is fully timed. */
    private final boolean graphLaunchesTimed;
    private final long[][] phaseNanos = new long[PATHS][PHASES];
    private final long[] steps = new long[PATHS];
    private final long[] betweenSteps = new long[PATHS];
    private final long[] periodNanos = new long[PATHS];
    private final long[] deviceNanos = new long[PATHS];
    private final long[] graphNanos = new long[PATHS];
    private final long[] deviceSteps = new long[PATHS];
    /** Per path, the period less the device time summed over the contiguous timed steps, and their count. */
    private final long[] idleNanos = new long[PATHS];
    private final long[] idleSteps = new long[PATHS];
    /** Per path, the steps under device timing whose graph launch the queue left untimed. */
    private final long[] untimedSteps = new long[PATHS];
    private final long[] allocatedBytes = new long[PATHS];
    /** Per path, the per-step values of every phase. */
    private final Samples[][] phaseSamples = new Samples[PATHS][PHASES];
    /** Per path, the step periods, device times and idle rests of the contiguous or timed steps. */
    private final Samples[] periodSamples = new Samples[PATHS];
    private final Samples[] deviceSamples = new Samples[PATHS];
    private final Samples[] idleSamples = new Samples[PATHS];
    private final Samples[] allocSamples = new Samples[PATHS];
    private final long[] stepNanos = new long[PHASES];
    private boolean open;
    private long mark;
    private int pos;
    private TrunkPath path;
    private long deviceAtBegin;
    private long graphAtBegin;
    private long allocAtBegin;
    private long lastEnd;
    private int lastPos = Integer.MIN_VALUE;

    private DecodeStepBreakdown(boolean enabled, KernelTiming timing, boolean graphLaunchesTimed) {
        this.enabled = enabled;
        this.timing = timing;
        this.graphLaunchesTimed = graphLaunchesTimed;
        for (int p = 0; p < PATHS; p++) {
            for (int i = 0; i < PHASES; i++) {
                phaseSamples[p][i] = new Samples();
            }
            periodSamples[p] = new Samples();
            deviceSamples[p] = new Samples();
            idleSamples[p] = new Samples();
            allocSamples[p] = new Samples();
        }
    }

    private static com.sun.management.ThreadMXBean allocationCounter() {
        java.lang.management.ThreadMXBean bean = java.lang.management.ManagementFactory.getThreadMXBean();
        if (!(bean instanceof com.sun.management.ThreadMXBean)) {
            return null;
        }
        com.sun.management.ThreadMXBean counter = (com.sun.management.ThreadMXBean) bean;
        return counter.isThreadAllocatedMemorySupported() && counter.isThreadAllocatedMemoryEnabled() ? counter : null;
    }

    /** The bytes the current thread allocated so far, or 0 without a counter. */
    private static long allocatedSoFar() {
        return ALLOCATIONS == null ? 0L : ALLOCATIONS.getThreadAllocatedBytes(Thread.currentThread().getId());
    }

    /**
     * The breakdown {@code -Dneuralfree.gpu.stepBreakdown} asks for: a recording one reading device time from
     * {@code timing} (null when the queue times nothing), or {@link #DISABLED}.
     *
     * @param graphLaunchesTimed whether {@code timing} holds every graph launch
     *                           ({@link ComputeQueue#timesGraphLaunches()})
     * @throws IllegalArgumentException when the property is set to something else than a boolean
     */
    public static DecodeStepBreakdown configured(KernelTiming timing, boolean graphLaunchesTimed) {
        String value = System.getProperty(PROPERTY);
        if (value == null || value.isEmpty() || value.equalsIgnoreCase("false")) {
            return DISABLED;
        }
        if (!value.equalsIgnoreCase("true")) {
            throw new IllegalArgumentException("-D" + PROPERTY + " must be a boolean, got '" + value + "'");
        }
        return new DecodeStepBreakdown(true, timing, graphLaunchesTimed);
    }

    /**
     * A recording breakdown, reading device time from {@code timing} when it is non-null, which holds every graph
     * launch.
     */
    public static DecodeStepBreakdown recording(KernelTiming timing) {
        return recording(timing, true);
    }

    /** A recording breakdown over {@code timing}, which holds the graph launches when {@code graphLaunchesTimed}. */
    public static DecodeStepBreakdown recording(KernelTiming timing, boolean graphLaunchesTimed) {
        return new DecodeStepBreakdown(true, timing, graphLaunchesTimed);
    }

    /** Whether this breakdown records. */
    public boolean enabled() {
        return enabled;
    }

    /**
     * A one-token decode call at {@code pos} enters the driver. The gap since the previous call's return counts as
     * {@link Phase#BETWEEN} when that call decoded {@code pos - 1}, the generation's previous token.
     */
    void begin(int pos) {
        if (!enabled) {
            return;
        }
        long now = System.nanoTime();
        Arrays.fill(stepNanos, 0L);
        stepNanos[Phase.BETWEEN.ordinal()] = lastPos != Integer.MIN_VALUE && pos == lastPos + 1 ? now - lastEnd : -1L;
        this.pos = pos;
        this.path = TrunkPath.DIRECT;
        this.mark = now;
        this.open = true;
        if (timing != null) {
            deviceAtBegin = timing.totalNanos();
            graphAtBegin = timing.graphNanos();
        }
        allocAtBegin = allocatedSoFar();
    }

    /** Adds the time since the last lap of the open call to {@code phase}. */
    void lap(Phase phase) {
        if (!open) {
            return;
        }
        long now = System.nanoTime();
        stepNanos[phase.ordinal()] += now - mark;
        mark = now;
    }

    /** Records how the open call's trunk reached the device. */
    void trunkPath(TrunkPath taken) {
        if (open) {
            path = taken;
        }
    }

    /** The open call returns: its remaining time counts as {@link Phase#CALL} and the step joins the totals. */
    void end() {
        if (!open) {
            return;
        }
        lap(Phase.CALL);
        long allocated = allocatedSoFar() - allocAtBegin;
        open = false;
        int p = path.ordinal();
        steps[p]++;
        allocatedBytes[p] += allocated;
        allocSamples[p].add(allocated);
        int betweenIdx = Phase.BETWEEN.ordinal();
        long between = stepNanos[betweenIdx];
        long period = 0;
        for (int i = 0; i < PHASES; i++) {
            if (i == betweenIdx) {
                continue;
            }
            phaseNanos[p][i] += stepNanos[i];
            period += stepNanos[i];
            phaseSamples[p][i].add(stepNanos[i]);
        }
        boolean contiguous = between >= 0;
        if (contiguous) {
            period += between;
            phaseNanos[p][betweenIdx] += between;
            betweenSteps[p]++;
            periodNanos[p] += period;
            phaseSamples[p][betweenIdx].add(between);
            periodSamples[p].add(period);
        }
        if (timing != null && !graphLaunchesTimed && path != TrunkPath.DIRECT) {
            untimedSteps[p]++;
        } else if (timing != null) {
            long device = Math.max(0L, timing.totalNanos() - deviceAtBegin);
            deviceNanos[p] += device;
            graphNanos[p] += Math.max(0L, timing.graphNanos() - graphAtBegin);
            deviceSteps[p]++;
            deviceSamples[p].add(device);
            if (contiguous) {
                idleNanos[p] += period - device;
                idleSteps[p]++;
                idleSamples[p].add(period - device);
            }
        }
        lastPos = pos;
        lastEnd = mark;
    }

    /** The open call failed or another kind of call ran: nothing of it is counted and the next gap is not. */
    void interrupt() {
        if (!enabled) {
            return;
        }
        open = false;
        lastPos = Integer.MIN_VALUE;
    }

    /** Clears every total, so the next report covers the calls from here on. */
    public void reset() {
        if (!enabled) {
            return;
        }
        for (long[] row : phaseNanos) {
            Arrays.fill(row, 0L);
        }
        for (long[] totals : new long[][]{steps, betweenSteps, periodNanos, deviceNanos, graphNanos, deviceSteps,
            idleNanos, idleSteps, untimedSteps, allocatedBytes}) {
            Arrays.fill(totals, 0L);
        }
        for (int p = 0; p < PATHS; p++) {
            for (Samples series : phaseSamples[p]) {
                series.clear();
            }
            periodSamples[p].clear();
            deviceSamples[p].clear();
            idleSamples[p].clear();
            allocSamples[p].clear();
        }
        open = false;
        lastPos = Integer.MIN_VALUE;
    }

    /** An immutable copy of the totals recorded so far. */
    public Report report() {
        long[][] phases = new long[PATHS][];
        for (int i = 0; i < PATHS; i++) {
            phases[i] = phaseNanos[i].clone();
        }
        double[][] medians = new double[PATHS][MEDIAN_SERIES];
        for (int p = 0; p < PATHS; p++) {
            for (int i = 0; i < PHASES; i++) {
                medians[p][i] = phaseSamples[p][i].median();
            }
            medians[p][PERIOD_SERIES] = periodSamples[p].median();
            medians[p][DEVICE_SERIES] = deviceSamples[p].median();
            medians[p][IDLE_SERIES] = idleSamples[p].median();
            medians[p][ALLOC_SERIES] = allocSamples[p].median();
        }
        return new Report(phases, steps.clone(), betweenSteps.clone(), periodNanos.clone(), deviceNanos.clone(),
                graphNanos.clone(), deviceSteps.clone(),
                new long[][]{idleNanos.clone(), idleSteps.clone(), untimedSteps.clone()},
                allocatedBytes.clone(), medians, timing != null, ALLOCATIONS != null);
    }

    /** A bounded series of per-step nanosecond values, for a median. */
    private static final class Samples {
        private long[] values = new long[256];
        private int count;

        void add(long v) {
            if (count == MAX_SAMPLES) {
                return;
            }
            if (count == values.length) {
                values = Arrays.copyOf(values, Math.min(MAX_SAMPLES, values.length * 2));
            }
            values[count++] = v;
        }

        void clear() {
            count = 0;
        }

        double median() {
            if (count == 0) {
                return Double.NaN;
            }
            long[] sorted = Arrays.copyOf(values, count);
            Arrays.sort(sorted);
            return count % 2 == 1 ? sorted[count / 2] : (sorted[count / 2 - 1] + sorted[count / 2]) / 2.0;
        }
    }

    /** The recorded totals and medians per trunk path. Immutable. */
    public static final class Report {
        private final long[][] phaseNanos;
        private final long[] steps;
        private final long[] betweenSteps;
        private final long[] periodNanos;
        private final long[] deviceNanos;
        private final long[] graphNanos;
        private final long[] deviceSteps;
        private final long[] idleNanos;
        private final long[] idleSteps;
        private final long[] untimedSteps;
        private final long[] allocatedBytes;
        private final double[][] medians;
        private final boolean timed;
        private final boolean allocationsCounted;

        /** @param idle per path, the idle nanoseconds, the contiguous timed steps and the untimed graph steps */
        Report(long[][] phaseNanos, long[] steps, long[] betweenSteps, long[] periodNanos, long[] deviceNanos,
               long[] graphNanos, long[] deviceSteps, long[][] idle, long[] allocatedBytes, double[][] medians,
               boolean timed, boolean allocationsCounted) {
            this.phaseNanos = phaseNanos;
            this.steps = steps;
            this.betweenSteps = betweenSteps;
            this.periodNanos = periodNanos;
            this.deviceNanos = deviceNanos;
            this.graphNanos = graphNanos;
            this.deviceSteps = deviceSteps;
            this.idleNanos = idle[0];
            this.idleSteps = idle[1];
            this.untimedSteps = idle[2];
            this.allocatedBytes = allocatedBytes;
            this.medians = medians;
            this.timed = timed;
            this.allocationsCounted = allocationsCounted;
        }

        /** Whether the JVM counted the decode thread's allocations, so the allocation figures are meaningful. */
        public boolean allocationsCounted() {
            return allocationsCounted;
        }

        /**
         * Mean bytes the decode thread allocated per step of {@code path} (every path when null), from the call's
         * entry to its return; NaN when no step ran or the JVM counts no allocations.
         */
        public double meanAllocatedBytes(TrunkPath path) {
            if (!allocationsCounted) {
                return Double.NaN;
            }
            long total = 0;
            long count = 0;
            for (TrunkPath t : paths(path)) {
                total += allocatedBytes[t.ordinal()];
                count += steps[t.ordinal()];
            }
            return count == 0 ? Double.NaN : (double) total / count;
        }

        /** The median bytes allocated per step of {@code path}, NaN when none ran or nothing was counted. */
        public double medianAllocatedBytes(TrunkPath path) {
            return allocationsCounted ? medians[path.ordinal()][ALLOC_SERIES] : Double.NaN;
        }

        /** Steps whose trunk took {@code path}. */
        public long steps(TrunkPath path) {
            return steps[path.ordinal()];
        }

        /** Steps over every path. */
        public long steps() {
            long sum = 0;
            for (long s : steps) {
                sum += s;
            }
            return sum;
        }

        /** Steps of {@code path} that continued the previous call's position, the ones carrying a gap. */
        public long contiguousSteps(TrunkPath path) {
            return betweenSteps[path.ordinal()];
        }

        /** Total nanoseconds of {@code phase} over the steps of {@code path}. */
        public long phaseNanos(TrunkPath path, Phase phase) {
            return phaseNanos[path.ordinal()][phase.ordinal()];
        }

        /**
         * Mean nanoseconds of {@code phase} per step of {@code path}, or over every path when {@code path} is null:
         * {@link Phase#BETWEEN} averages over the steps that followed a contiguous call, the others over all.
         */
        public double meanNanos(TrunkPath path, Phase phase) {
            long total = 0;
            long count = 0;
            for (TrunkPath t : paths(path)) {
                total += phaseNanos[t.ordinal()][phase.ordinal()];
                count += phase == Phase.BETWEEN ? betweenSteps[t.ordinal()] : steps[t.ordinal()];
            }
            return count == 0 ? Double.NaN : (double) total / count;
        }

        /** The median nanoseconds of {@code phase} per step of {@code path}, NaN when none ran. */
        public double medianNanos(TrunkPath path, Phase phase) {
            return medians[path.ordinal()][phase.ordinal()];
        }

        /**
         * Mean step period of {@code path} (every path when null): the call's phases plus the gap before it, over
         * the steps that followed a contiguous call.
         */
        public double meanPeriodNanos(TrunkPath path) {
            long total = 0;
            long count = 0;
            for (TrunkPath t : paths(path)) {
                total += periodNanos[t.ordinal()];
                count += betweenSteps[t.ordinal()];
            }
            return count == 0 ? Double.NaN : (double) total / count;
        }

        /** The median step period of the contiguous steps of {@code path}, NaN when none ran. */
        public double medianPeriodNanos(TrunkPath path) {
            return medians[path.ordinal()][PERIOD_SERIES];
        }

        /** Whether the steps carry device time ({@code -Dneuralfree.gpu.profile} was on). */
        public boolean timed() {
            return timed;
        }

        /** Mean device nanoseconds of the commands per step of {@code path} (every path when null). */
        public double deviceNanos(TrunkPath path) {
            return meanOf(deviceNanos, path);
        }

        /** Mean device nanoseconds of the graph launches per step of {@code path} (every path when null). */
        public double graphNanos(TrunkPath path) {
            return meanOf(graphNanos, path);
        }

        /** The median device nanoseconds per step of {@code path}, NaN without device timing. */
        public double medianDeviceNanos(TrunkPath path) {
            return medians[path.ordinal()][DEVICE_SERIES];
        }

        /**
         * Mean of the step period less its device time per contiguous step of {@code path} (every path when null),
         * both taken over the same steps; NaN without device timing or a contiguous step.
         */
        public double meanIdleNanos(TrunkPath path) {
            long total = 0;
            long count = 0;
            for (TrunkPath t : paths(path)) {
                total += idleNanos[t.ordinal()];
                count += idleSteps[t.ordinal()];
            }
            return count == 0 ? Double.NaN : (double) total / count;
        }

        /** The median of the step period less its device time per contiguous step of {@code path}. */
        public double medianIdleNanos(TrunkPath path) {
            return medians[path.ordinal()][IDLE_SERIES];
        }

        /**
         * Steps of {@code path} (every path when null) that ran under device timing but carry no device time: their
         * trunk ran as a graph whose launch the queue does not time.
         */
        public long untimedSteps(TrunkPath path) {
            long sum = 0;
            for (TrunkPath t : paths(path)) {
                sum += untimedSteps[t.ordinal()];
            }
            return sum;
        }

        /** The path most steps took, the steady state the report's first columns describe. */
        public TrunkPath dominantPath() {
            TrunkPath best = TrunkPath.REPLAYED;
            for (TrunkPath t : TrunkPath.values()) {
                if (steps(t) > steps(best)) {
                    best = t;
                }
            }
            return best;
        }

        private double meanOf(long[] totals, TrunkPath path) {
            long total = 0;
            long count = 0;
            for (TrunkPath t : paths(path)) {
                total += totals[t.ordinal()];
                count += deviceSteps[t.ordinal()];
            }
            return count == 0 ? Double.NaN : (double) total / count;
        }

        private static List<TrunkPath> paths(TrunkPath path) {
            return path == null ? Arrays.asList(TrunkPath.values()) : Collections.singletonList(path);
        }

        /**
         * The report's lines: the step counts per path, then one row per phase with its mean and median
         * microseconds per step of the {@link #dominantPath()} and its mean over every step, the step period,
         * with device timing the device time per step and the idle rest of the period of the contiguous steps, and
         * the bytes the decode thread allocated per step when the JVM counts them.
         */
        public List<String> format() {
            TrunkPath main = dominantPath();
            List<String> lines = new ArrayList<>();
            StringBuilder counts = new StringBuilder();
            for (TrunkPath t : TrunkPath.values()) {
                if (counts.length() > 0) {
                    counts.append(", ");
                }
                counts.append(t.label()).append(' ').append(steps(t));
            }
            lines.add(String.format(Locale.ROOT, "%d decode steps (%s); us per step", steps(), counts));
            lines.add(String.format(Locale.ROOT, "%-30s %14s %14s %10s", "phase", main.label() + " mean",
                    main.label() + " p50", "all mean"));
            for (Phase phase : Phase.values()) {
                lines.add(row(phase.label(), meanNanos(main, phase), medianNanos(main, phase), meanNanos(null, phase)));
            }
            lines.add(row("step period", meanPeriodNanos(main), medianPeriodNanos(main), meanPeriodNanos(null)));
            if (timed) {
                lines.add(row("device busy (timed commands)", deviceNanos(main), medianDeviceNanos(main),
                        deviceNanos(null)));
                lines.add(row("  of which graph launches", graphNanos(main), Double.NaN, graphNanos(null)));
                lines.add(row("device idle (period - busy)", meanIdleNanos(main), medianIdleNanos(main),
                        meanIdleNanos(null)));
                if (untimedSteps(null) > 0) {
                    lines.add(String.format(Locale.ROOT, "(the device rows leave out %d steps whose trunk ran as"
                            + " a graph the queue does not time)", untimedSteps(null)));
                }
            }
            if (allocationsCounted) {
                lines.add(String.format(Locale.ROOT, "%-30s %14s %14s %10s", "allocated bytes per step",
                        bytes(meanAllocatedBytes(main)), bytes(medianAllocatedBytes(main)),
                        bytes(meanAllocatedBytes(null))));
            }
            return lines;
        }

        private static String row(String label, double pathMean, double pathMedian, double allMean) {
            return String.format(Locale.ROOT, "%-30s %14s %14s %10s", label, micros(pathMean), micros(pathMedian),
                    micros(allMean));
        }

        private static String bytes(double b) {
            return Double.isNaN(b) ? "-" : String.format(Locale.ROOT, "%.0f", b);
        }

        private static String micros(double nanos) {
            return Double.isNaN(nanos) ? "-" : String.format(Locale.ROOT, "%.1f", nanos / 1.0e3);
        }
    }
}

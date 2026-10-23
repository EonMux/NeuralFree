package com.eonmux.neuralfree.gguf.gpu.diag;

import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.gguf.gpu.forward.ExecutionObserver;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

/**
 * Wall-clock GPU timing with host synchronization at every boundary, so each interval holds exactly the device work
 * enqueued inside it.
 *
 * <p>As an {@link ExecutionObserver} it times every stage of every observed step: stage {@code S} accumulates under
 * the phase {@code S.name()}, whole steps under {@link #STEP}, and {@link ExecutionObserver.Stage#LAYER} stages also
 * per layer ({@link Report#layerNanos()}). Stages nest ({@code LAYER} contains the layer's other stages), so phase
 * totals overlap. Hand-instrumented code times consecutive phases with {@link #mark()} and {@link #lap(String)}.</p>
 *
 * <p>The synchronization removes the overlap between dispatches, so totals are higher than an unobserved step;
 * per-phase times measure the device work of the phase. Not thread-safe.</p>
 */
public final class GpuProfiler implements ExecutionObserver {

    /** Phase name of whole observed steps. */
    public static final String STEP = "STEP";

    private final ComputeQueue queue;
    private final Map<String, long[]> phases = new LinkedHashMap<>();
    private final long[] stageStart = new long[Stage.values().length];
    private final List<Long> layerNanos = new ArrayList<>();
    private long stepStart;
    private long lapStart;
    private int steps;

    /** A profiler that synchronizes {@code queue}, the queue the timed work runs on. */
    public GpuProfiler(ComputeQueue queue) {
        this.queue = Objects.requireNonNull(queue, "queue");
    }

    // ------------------------------------------------------------------ ExecutionObserver

    @Override
    public void onStepStart(int pos0, int nTokens) {
        stepStart = syncedNow();
    }

    @Override
    public void onStageStart(int layer, Stage stage) {
        stageStart[stage.ordinal()] = syncedNow();
    }

    @Override
    public void onStageEnd(int layer, Stage stage, ScratchSlot result, int nRows, int width) {
        long elapsed = syncedNow() - stageStart[stage.ordinal()];
        record(stage.name(), elapsed);
        if (stage == Stage.LAYER && layer >= 0) {
            while (layerNanos.size() <= layer) {
                layerNanos.add(0L);
            }
            layerNanos.set(layer, layerNanos.get(layer) + elapsed);
        }
    }

    @Override
    public void onStepEnd(int pos0, int nTokens) {
        record(STEP, syncedNow() - stepStart);
        steps++;
    }

    // ------------------------------------------------------------------ phase timing

    /** Waits for the queued work, then starts the next lap. */
    public void mark() {
        lapStart = syncedNow();
    }

    /** Waits for the queued work and records the time since the last {@link #mark()} or lap under {@code phase}. */
    public void lap(String phase) {
        long now = syncedNow();
        record(phase, now - lapStart);
        lapStart = now;
    }

    /** Adds an externally measured interval to {@code phase}. */
    public void record(String phase, long nanos) {
        Objects.requireNonNull(phase, "phase");
        if (nanos < 0) {
            throw new IllegalArgumentException("negative interval " + nanos + " ns for phase " + phase);
        }
        long[] acc = phases.computeIfAbsent(phase, k -> new long[2]);
        acc[0] += nanos;
        acc[1]++;
    }

    /** Clears every recorded phase, layer and step. */
    public void reset() {
        phases.clear();
        layerNanos.clear();
        steps = 0;
    }

    /** An immutable snapshot of what has been recorded. */
    public Report report() {
        Map<String, Phase> snap = new LinkedHashMap<>();
        for (Map.Entry<String, long[]> e : phases.entrySet()) {
            snap.put(e.getKey(), new Phase(e.getKey(), e.getValue()[0], e.getValue()[1]));
        }
        long[] layers = new long[layerNanos.size()];
        for (int i = 0; i < layers.length; i++) {
            layers[i] = layerNanos.get(i);
        }
        return new Report(snap, layers, steps);
    }

    private long syncedNow() {
        queue.finish();
        return System.nanoTime();
    }

    /** Accumulated time of one phase. Immutable. */
    public static final class Phase {
        private final String name;
        private final long nanos;
        private final long count;

        Phase(String name, long nanos, long count) {
            this.name = name;
            this.nanos = nanos;
            this.count = count;
        }

        public String name() {
            return name;
        }

        public long nanos() {
            return nanos;
        }

        /** Number of recorded intervals. */
        public long count() {
            return count;
        }

        public double millis() {
            return nanos / 1e6;
        }
    }

    /** Snapshot of a profiler. Immutable. */
    public static final class Report {
        private final Map<String, Phase> phases;
        private final long[] layerNanos;
        private final int steps;

        Report(Map<String, Phase> phases, long[] layerNanos, int steps) {
            this.phases = Collections.unmodifiableMap(phases);
            this.layerNanos = layerNanos;
            this.steps = steps;
        }

        /** Phases in first-recorded order. Unmodifiable. */
        public Map<String, Phase> phases() {
            return phases;
        }

        /** Total nanoseconds of a phase, 0 when it was never recorded. */
        public long nanos(String phase) {
            Phase p = phases.get(phase);
            return p == null ? 0 : p.nanos();
        }

        /** Sum of the given phases. */
        public long nanos(String... phaseNames) {
            long sum = 0;
            for (String n : phaseNames) {
                sum += nanos(n);
            }
            return sum;
        }

        /** Sum over all phases; the elapsed time when the phases are consecutive laps. */
        public long sumNanos() {
            long sum = 0;
            for (Phase p : phases.values()) {
                sum += p.nanos();
            }
            return sum;
        }

        /** Time of each layer's {@code LAYER} stage, indexed by layer (copy). */
        public long[] layerNanos() {
            return layerNanos.clone();
        }

        /** Observed steps. */
        public int steps() {
            return steps;
        }

        /**
         * One line per phase with milliseconds, share of {@code referenceNanos} and interval count.
         *
         * @param referenceNanos the 100% figure; non-positive omits the share
         */
        public String format(String title, long referenceNanos) {
            StringBuilder sb = new StringBuilder();
            sb.append("=== ").append(title).append(" ===\n");
            for (Phase p : phases.values()) {
                sb.append(String.format(Locale.ROOT, "  %-24s %10.3f ms", p.name() + ":", p.millis()));
                if (referenceNanos > 0) {
                    sb.append(String.format(Locale.ROOT, "  %5.1f%%", 100.0 * p.nanos() / referenceNanos));
                }
                sb.append(String.format(Locale.ROOT, "  x%d%n", p.count()));
            }
            return sb.toString();
        }
    }
}

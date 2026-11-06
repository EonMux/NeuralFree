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
 * <p>Stages are attributed to devices through {@link #device(String, ComputeQueue)}: each device's forward context
 * observes through its own view, every boundary synchronizes every registered queue, and the report gives each
 * device's phases and the layers it ran ({@link Report#devicePhases()}, {@link Report#layerDevices()}). A layer
 * split binds one view per stage; a step hook that every view repeats is counted once.</p>
 *
 * <p>The synchronization removes the overlap between dispatches, so totals are higher than an unobserved step;
 * per-phase times measure the device work of the phase. Not thread-safe.</p>
 */
public final class GpuProfiler implements ExecutionObserver {

    /** Phase name of whole observed steps. */
    public static final String STEP = "STEP";

    /** The queues every boundary synchronizes: the profiler's own, then those of the device views. */
    private final List<ComputeQueue> queues = new ArrayList<>();
    private final Map<String, long[]> phases = new LinkedHashMap<>();
    private final Map<String, Map<String, long[]>> devicePhases = new LinkedHashMap<>();
    private final long[] stageStart = new long[Stage.values().length];
    private final List<Long> layerNanos = new ArrayList<>();
    private final List<String> layerDevices = new ArrayList<>();
    private boolean inStep;
    private long stepStart;
    private long lapStart;
    private int steps;

    /** A profiler that synchronizes {@code queue}, the queue the timed work runs on. */
    public GpuProfiler(ComputeQueue queue) {
        queues.add(Objects.requireNonNull(queue, "queue"));
    }

    /**
     * An observer that times the stages of device {@code name} into this profiler and attributes them to it;
     * {@code queue}, the device's queue, joins the queues every boundary synchronizes.
     *
     * @throws IllegalArgumentException when {@code name} is blank
     */
    public ExecutionObserver device(String name, ComputeQueue queue) {
        Objects.requireNonNull(name, "name");
        Objects.requireNonNull(queue, "queue");
        if (name.trim().isEmpty()) {
            throw new IllegalArgumentException("device name is blank");
        }
        boolean known = false;
        for (ComputeQueue q : queues) {
            known |= q == queue;
        }
        if (!known) {
            queues.add(queue);
        }
        return new DeviceView(name.trim());
    }

    // ------------------------------------------------------------------ ExecutionObserver

    @Override
    public void onStepStart(int pos0, int nTokens) {
        if (!inStep) {
            inStep = true;
            stepStart = syncedNow();
        }
    }

    @Override
    public void onStageStart(int layer, Stage stage) {
        stageStart[stage.ordinal()] = syncedNow();
    }

    @Override
    public void onStageEnd(int layer, Stage stage, ScratchSlot result, int nRows, int width) {
        stageEnded(layer, stage, null);
    }

    @Override
    public void onStepEnd(int pos0, int nTokens) {
        if (inStep) {
            inStep = false;
            record(STEP, syncedNow() - stepStart);
            steps++;
        }
    }

    /** Records the stage that ends now under its phase, its layer and, when given, its device. */
    private void stageEnded(int layer, Stage stage, String device) {
        long elapsed = syncedNow() - stageStart[stage.ordinal()];
        record(stage.name(), elapsed);
        if (device != null) {
            long[] acc = devicePhases.computeIfAbsent(device, k -> new LinkedHashMap<>())
                    .computeIfAbsent(stage.name(), k -> new long[2]);
            acc[0] += elapsed;
            acc[1]++;
        }
        if (stage == Stage.LAYER && layer >= 0) {
            while (layerNanos.size() <= layer) {
                layerNanos.add(0L);
                layerDevices.add(null);
            }
            layerNanos.set(layer, layerNanos.get(layer) + elapsed);
            if (device != null) {
                layerDevices.set(layer, device);
            }
        }
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
        devicePhases.clear();
        layerNanos.clear();
        layerDevices.clear();
        inStep = false;
        steps = 0;
    }

    /** An immutable snapshot of what has been recorded. */
    public Report report() {
        Map<String, Phase> snap = new LinkedHashMap<>();
        for (Map.Entry<String, long[]> e : phases.entrySet()) {
            snap.put(e.getKey(), new Phase(e.getKey(), e.getValue()[0], e.getValue()[1]));
        }
        Map<String, Map<String, Phase>> byDevice = new LinkedHashMap<>();
        for (Map.Entry<String, Map<String, long[]>> d : devicePhases.entrySet()) {
            Map<String, Phase> dp = new LinkedHashMap<>();
            for (Map.Entry<String, long[]> e : d.getValue().entrySet()) {
                dp.put(e.getKey(), new Phase(e.getKey(), e.getValue()[0], e.getValue()[1]));
            }
            byDevice.put(d.getKey(), Collections.unmodifiableMap(dp));
        }
        long[] layers = new long[layerNanos.size()];
        for (int i = 0; i < layers.length; i++) {
            layers[i] = layerNanos.get(i);
        }
        return new Report(snap, byDevice, layers, layerDevices.toArray(new String[0]), steps);
    }

    private long syncedNow() {
        for (ComputeQueue q : queues) {
            q.finish();
        }
        return System.nanoTime();
    }

    /** The observer of one device: its stages time into the profiler under the device's name. */
    private final class DeviceView implements ExecutionObserver {
        private final String name;

        DeviceView(String name) {
            this.name = name;
        }

        @Override
        public void onStepStart(int pos0, int nTokens) {
            GpuProfiler.this.onStepStart(pos0, nTokens);
        }

        @Override
        public void onStageStart(int layer, Stage stage) {
            GpuProfiler.this.onStageStart(layer, stage);
        }

        @Override
        public void onStageEnd(int layer, Stage stage, ScratchSlot result, int nRows, int width) {
            stageEnded(layer, stage, name);
        }

        @Override
        public void onStepEnd(int pos0, int nTokens) {
            GpuProfiler.this.onStepEnd(pos0, nTokens);
        }

        @Override
        public String toString() {
            return "GpuProfiler.device(" + name + ")";
        }
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
        private final Map<String, Map<String, Phase>> devicePhases;
        private final long[] layerNanos;
        private final String[] layerDevices;
        private final int steps;

        Report(Map<String, Phase> phases, Map<String, Map<String, Phase>> devicePhases, long[] layerNanos,
               String[] layerDevices, int steps) {
            this.phases = Collections.unmodifiableMap(phases);
            this.devicePhases = Collections.unmodifiableMap(devicePhases);
            this.layerNanos = layerNanos;
            this.layerDevices = layerDevices;
            this.steps = steps;
        }

        /**
         * The phases of every device a {@link GpuProfiler#device} view observed, by device name in first-recorded
         * order; empty when no view observed a stage. Unmodifiable.
         */
        public Map<String, Map<String, Phase>> devicePhases() {
            return devicePhases;
        }

        /** The device each layer's {@code LAYER} stage ran on, indexed by layer, null where none was named (copy). */
        public String[] layerDevices() {
            return layerDevices.clone();
        }

        /** The layers whose {@code LAYER} stage ran on {@code device}, as ranges such as {@code 0-3, 6}. */
        public String layersOf(String device) {
            StringBuilder sb = new StringBuilder();
            int i = 0;
            while (i < layerDevices.length) {
                if (!device.equals(layerDevices[i])) {
                    i++;
                    continue;
                }
                int end = i;
                while (end + 1 < layerDevices.length && device.equals(layerDevices[end + 1])) {
                    end++;
                }
                sb.append(sb.length() == 0 ? "" : ", ").append(i).append(end > i ? "-" + end : "");
                i = end + 1;
            }
            return sb.toString();
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
            for (Map.Entry<String, Map<String, Phase>> d : devicePhases.entrySet()) {
                String layers = layersOf(d.getKey());
                sb.append("  on ").append(d.getKey()).append(layers.isEmpty() ? "" : " (layers " + layers + ")")
                        .append(":\n");
                for (Phase p : d.getValue().values()) {
                    sb.append(String.format(Locale.ROOT, "    %-22s %10.3f ms  x%d%n", p.name() + ":", p.millis(),
                            p.count()));
                }
            }
            return sb.toString();
        }
    }
}

package com.eonmux.neuralfree.gguf.gpu.diag;

import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.gguf.gpu.forward.ExecutionObserver;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot;

import java.io.IOException;
import java.io.PrintStream;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.Locale;
import java.util.Objects;

/**
 * Host-side numeric diagnostics of the GPU forward path: per-stage statistics (including non-finite counts) and
 * value dumps for comparison against a reference implementation.
 *
 * <p>As an {@link ExecutionObserver} it reads every stage result of the first {@link Config#steps()} steps and
 * prints its {@link Stats} to the sink; with {@link Config#dumpEnabled()} it also appends the full values to
 * {@link Config#dumpFile()}. Every read waits for the queue, so observed steps run serialized.</p>
 *
 * <p>Not thread-safe. Messages go to the sink given at construction (standard error by default), one line per
 * report, prefixed with a tag naming the check.</p>
 */
public final class GpuDiagnostics implements ExecutionObserver {

    /** {@code -Dneuralfree.diag=true} enables stage diagnostics. */
    public static final String PROP_ENABLED = "neuralfree.diag";
    /** {@code -Dneuralfree.diag.dump=true} appends full-precision values to the dump file. */
    public static final String PROP_DUMP = "neuralfree.diag.dump";
    /** Dump file path. */
    public static final String PROP_DUMP_FILE = "neuralfree.diag.dumpfile";
    /** Number of steps the observer reports (default 1). */
    public static final String PROP_STEPS = "neuralfree.diag.steps";

    private static final int HEAD = 8;

    private final ComputeQueue queue;
    private final Config config;
    private final PrintStream out;
    private int step;

    /** Diagnostics on {@code queue}, reporting to standard error. */
    public GpuDiagnostics(ComputeQueue queue, Config config) {
        this(queue, config, System.err);
    }

    /** Diagnostics on {@code queue}, reporting to {@code out}. */
    public GpuDiagnostics(ComputeQueue queue, Config config, PrintStream out) {
        this.queue = Objects.requireNonNull(queue, "queue");
        this.config = Objects.requireNonNull(config, "config");
        this.out = Objects.requireNonNull(out, "out");
    }

    public Config config() {
        return config;
    }

    // ------------------------------------------------------------------ ExecutionObserver

    @Override
    public boolean enabled() {
        return config.enabled();
    }

    @Override
    public void onStepStart(int pos0, int nTokens) {
        step++;
    }

    @Override
    public void onStageEnd(int layer, Stage stage, ScratchSlot result, int nRows, int width) {
        if (!config.enabled() || step > config.steps() || result == null) {
            return;
        }
        long n = (long) nRows * width;
        if (n <= 0 || n > Integer.MAX_VALUE - 8) {
            out.println("[DIAG] " + stageLabel(stage, layer) + ": " + nRows + " x " + width + " values not read");
            return;
        }
        float[] v = read(result.buffer(), result.offset(), (int) n);
        Stats s = Stats.of(v, 0, (int) n);
        out.println("[DIAG] step=" + step + " " + stageLabel(stage, layer) + " [" + nRows + " x " + width + "] " + s
                + " head=" + head(v, 0, (int) n));
        if (config.dumpEnabled()) {
            for (int r = 0; r < nRows; r++) {
                appendText(stageLabel(stage, layer) + (nRows > 1 ? "#" + r : ""), v, r * width, width);
            }
        }
    }

    private static String stageLabel(Stage stage, int layer) {
        return layer >= 0 ? stage.name().toLowerCase(Locale.ROOT) + "-" + layer : stage.name().toLowerCase(Locale.ROOT);
    }

    // ------------------------------------------------------------------ read-backs and dumps

    /**
     * Waits for the queued work and reads {@code count} floats starting at float {@code floatOffset}.
     *
     * @return a new array of the values
     */
    private float[] read(GpuBuffer buf, long floatOffset, int count) {
        Objects.requireNonNull(buf, "buf");
        if (count < 0 || floatOffset < 0) {
            throw new IllegalArgumentException("read of " + count + " floats at " + floatOffset);
        }
        float[] v = new float[count];
        if (count == 0) {
            return v;
        }
        queue.finish();
        closeEvent(queue.enqueueReadBuffer(buf, floatOffset * 4L, count * 4L, v));
        queue.finish();
        return v;
    }

    private void appendText(String label, float[] v, int off, int count) {
        StringBuilder sb = new StringBuilder();
        sb.append("=== ").append(label).append(" count=").append(count).append(" ===\n");
        for (int i = 0; i < count; i++) {
            sb.append(i).append(' ').append(Float.toString(v[off + i])).append('\n');
        }
        try (Writer w = Files.newBufferedWriter(config.dumpFile(), StandardCharsets.UTF_8,
                StandardOpenOption.CREATE, StandardOpenOption.APPEND)) {
            w.write(sb.toString());
        } catch (IOException e) {
            out.println("[FDUMP] write to " + config.dumpFile() + " failed: " + e.getMessage());
        }
    }

    private static String head(float[] v, int off, int n) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < Math.min(HEAD, n); i++) {
            if (i > 0) {
                sb.append(", ");
            }
            sb.append(String.format(Locale.ROOT, "%.5f", v[off + i]));
        }
        return sb.append(n > HEAD ? ", ...]" : "]").toString();
    }

    private static void closeEvent(GpuEvent e) {
        if (e != null) {
            e.close();
        }
    }

    // ------------------------------------------------------------------ nested types

    /** Statistics of a float range. Immutable. */
    public static final class Stats {
        private final int count;
        private final int nonFinite;
        private final int firstNonFinite;
        private final float min;
        private final float max;
        private final double mean;
        private final double l2;

        private Stats(int count, int nonFinite, int firstNonFinite, float min, float max, double mean, double l2) {
            this.count = count;
            this.nonFinite = nonFinite;
            this.firstNonFinite = firstNonFinite;
            this.min = min;
            this.max = max;
            this.mean = mean;
            this.l2 = l2;
        }

        /** Statistics of {@code v[off, off + n)}; min, max, mean and L2 cover the finite values. */
        public static Stats of(float[] v, int off, int n) {
            int bad = 0;
            int first = -1;
            float min = Float.POSITIVE_INFINITY;
            float max = Float.NEGATIVE_INFINITY;
            double sum = 0;
            double sumSq = 0;
            for (int i = 0; i < n; i++) {
                float x = v[off + i];
                if (!Float.isFinite(x)) {
                    bad++;
                    if (first < 0) {
                        first = i;
                    }
                    continue;
                }
                min = Math.min(min, x);
                max = Math.max(max, x);
                sum += x;
                sumSq += (double) x * x;
            }
            int finite = n - bad;
            return new Stats(n, bad, first, finite > 0 ? min : Float.NaN, finite > 0 ? max : Float.NaN,
                    finite > 0 ? sum / finite : Double.NaN, Math.sqrt(sumSq));
        }

        public int count() {
            return count;
        }

        public int nonFinite() {
            return nonFinite;
        }

        /** Index of the first non-finite value, or -1. */
        public int firstNonFinite() {
            return firstNonFinite;
        }

        public float min() {
            return min;
        }

        public float max() {
            return max;
        }

        public double mean() {
            return mean;
        }

        public double l2() {
            return l2;
        }

        @Override
        public String toString() {
            return String.format(Locale.ROOT, "min=%.5g max=%.5g mean=%.5g l2=%.5g", min, max, mean, l2)
                    + (nonFinite > 0 ? " NaN/Inf=" + nonFinite + " firstAt=" + firstNonFinite : "");
        }
    }

    /** Immutable diagnostics configuration. */
    public static final class Config {
        private final boolean enabled;
        private final boolean dumpEnabled;
        private final Path dumpFile;
        private final int steps;

        public Config(boolean enabled, boolean dumpEnabled, Path dumpFile, int steps) {
            if (steps < 0) {
                throw new IllegalArgumentException("steps must be non-negative, got " + steps);
            }
            this.enabled = enabled;
            this.dumpEnabled = dumpEnabled;
            this.dumpFile = Objects.requireNonNull(dumpFile, "dumpFile");
            this.steps = steps;
        }

        /**
         * From {@value #PROP_ENABLED}, {@value #PROP_DUMP}, {@value #PROP_DUMP_FILE} (default
         * {@code <java.io.tmpdir>/neuralfree_diag_dump.txt}) and {@value #PROP_STEPS} (default 1).
         *
         * @throws IllegalArgumentException when {@value #PROP_STEPS} is not a non-negative integer
         */
        public static Config fromSystemProperties() {
            String file = System.getProperty(PROP_DUMP_FILE);
            Path path = file != null && !file.isEmpty() ? Paths.get(file)
                    : Paths.get(System.getProperty("java.io.tmpdir"), "neuralfree_diag_dump.txt");
            String stepsProp = System.getProperty(PROP_STEPS, "1");
            int steps;
            try {
                steps = Integer.parseInt(stepsProp.trim());
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException(PROP_STEPS + " must be a non-negative integer, got '"
                        + stepsProp + "'", e);
            }
            return new Config(Boolean.getBoolean(PROP_ENABLED), Boolean.getBoolean(PROP_DUMP), path, steps);
        }

        public boolean enabled() {
            return enabled;
        }

        public boolean dumpEnabled() {
            return dumpEnabled;
        }

        public Path dumpFile() {
            return dumpFile;
        }

        /** Steps the observer reports, counted from the first. */
        public int steps() {
            return steps;
        }
    }
}

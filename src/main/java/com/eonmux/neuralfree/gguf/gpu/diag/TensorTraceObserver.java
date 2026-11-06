package com.eonmux.neuralfree.gguf.gpu.diag;

import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.gguf.gpu.forward.ExecutionObserver;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot;

import java.io.PrintStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.regex.Pattern;
import java.util.regex.PatternSyntaxException;

/**
 * Executes llama.cpp's {@code common_debug_cb_eval} over the GPU forward stages: every observed stage's result is
 * printed as a nested, truncated table of values, optionally filtered by stage name and aborted on the first NaN.
 *
 * <p>The stage label is {@code stage} for a model-level stage and {@code stage-<layer>} for a layer stage
 * ({@code embedding}, {@code mixer-0}, {@code logits}, ...), the labels {@link GpuDiagnostics} reports. A filter
 * pattern is anchored at the start of the label, as llama.cpp anchors its {@code --tensor-filter} patterns, so
 * {@code mixer} selects every mixer stage and {@code mixer-1} one layer's. No pattern selects every stage.</p>
 *
 * <p>Every read waits for the queue, so an observed step runs serialized, like the diagnostics. Each dimension wider
 * than twice {@link #HEAD} prints its first and last {@link #HEAD} entries with an ellipsis between, and the stage's
 * values sum, both as llama.cpp's {@code common_debug_print_tensor} does. {@code abortOnNan} throws
 * {@link NonFiniteException} when a stage's sum is NaN. Not thread-safe.</p>
 */
public final class TensorTraceObserver implements ExecutionObserver {

    /** Entries printed from each end of a truncated dimension. */
    public static final int HEAD = 3;

    private static final String INDENT = "    ";

    private final ComputeQueue queue;
    private final PrintStream out;
    private final List<Pattern> filters;
    private final boolean abortOnNan;
    private final int steps;
    private final String device;
    private int step;

    /**
     * A trace of at most {@code steps} steps ({@code 0} for every step).
     *
     * @param filters   name patterns, each anchored at the start of a stage label; empty for every stage
     * @param device    the device name reports carry, or null for none
     * @throws IllegalArgumentException when a pattern does not compile
     */
    public TensorTraceObserver(ComputeQueue queue, PrintStream out, List<String> filters, boolean abortOnNan,
                               int steps, String device) {
        this.queue = Objects.requireNonNull(queue, "queue");
        this.out = Objects.requireNonNull(out, "out");
        this.filters = compile(filters);
        this.abortOnNan = abortOnNan;
        this.steps = steps;
        this.device = device == null || device.trim().isEmpty() ? null : device.trim();
        if (steps < 0) {
            throw new IllegalArgumentException("trace steps must be non-negative, got " + steps);
        }
    }

    private static List<Pattern> compile(List<String> filters) {
        List<Pattern> patterns = new ArrayList<>(filters.size());
        for (String filter : filters) {
            try {
                patterns.add(Pattern.compile("^" + filter));
            } catch (PatternSyntaxException e) {
                throw new IllegalArgumentException("Invalid regex pattern '" + filter + "': " + e.getDescription());
            }
        }
        return patterns;
    }

    @Override
    public boolean enabled() {
        return steps == 0 || step <= steps;
    }

    @Override
    public void onStepStart(int pos0, int nTokens) {
        step++;
    }

    @Override
    public void onStageEnd(int layer, Stage stage, ScratchSlot result, int nRows, int width) {
        if (!enabled() || result == null) {
            return;
        }
        String label = label(stage, layer);
        if (!matches(label)) {
            return;
        }
        long n = (long) nRows * width;
        if (n <= 0 || n > Integer.MAX_VALUE - 8) {
            out.println(header(label, nRows, width) + " values not read");
            return;
        }
        float[] values = read(result.buffer(), result.offset(), (int) n);
        double sum = print(label, values, nRows, width);
        if (abortOnNan && Double.isNaN(sum)) {
            throw new NonFiniteException(label);
        }
    }

    private boolean matches(String label) {
        if (filters.isEmpty()) {
            return true;
        }
        for (Pattern p : filters) {
            if (p.matcher(label).find()) {
                return true;
            }
        }
        return false;
    }

    private String header(String label, int nRows, int width) {
        return "[TRACE] step=" + step + (device != null ? " device=" + device : "") + " " + label + " ["
                + nRows + " x " + width + "]";
    }

    /** Prints the table and returns the sum of its values, accumulated in float as the reference does. */
    private double print(String label, float[] values, int nRows, int width) {
        out.println(header(label, nRows, width));
        float sum = 0.0f;
        for (int i = 0; i < (int) ((long) nRows * width); i++) {
            sum += values[i];
        }
        StringBuilder sb = new StringBuilder();
        sb.append(INDENT).append("[\n");
        for (int row = 0; row < nRows; row++) {
            if (row == HEAD && nRows > 2 * HEAD) {
                sb.append(INDENT).append(INDENT).append("...,\n");
                row = nRows - HEAD;
            }
            sb.append(INDENT).append(INDENT).append("[");
            appendRow(sb, values, (long) row * width, width);
            sb.append("  ],\n");
        }
        sb.append(INDENT).append("]\n");
        sb.append(INDENT).append("sum = ").append(String.format(Locale.ROOT, "%f", (float) sum)).append('\n');
        out.print(sb);
        return sum;
    }

    /** One row of {@code width} values, its first and last {@link #HEAD} entries between ellipses when truncated. */
    private static void appendRow(StringBuilder sb, float[] values, long offset, int width) {
        for (int i = 0; i < width; i++) {
            if (i == HEAD && width > 2 * HEAD) {
                sb.append("   ..., ");
                i = width - HEAD;
            }
            sb.append(String.format(Locale.ROOT, "%12.4f", values[(int) (offset + i)]));
            if (i < width - 1) {
                sb.append(", ");
            }
        }
    }

    /** Waits for the queued work and reads {@code count} floats starting at float {@code floatOffset}. */
    private float[] read(GpuBuffer buf, long floatOffset, int count) {
        Objects.requireNonNull(buf, "buf");
        float[] v = new float[count];
        queue.finish();
        closeEvent(queue.enqueueReadBuffer(buf, floatOffset * 4L, count * 4L, v));
        queue.finish();
        return v;
    }

    private static void closeEvent(GpuEvent e) {
        if (e != null) {
            e.close();
        }
    }

    private static String label(Stage stage, int layer) {
        String name = stage.name().toLowerCase(Locale.ROOT);
        return layer >= 0 ? name + "-" + layer : name;
    }

    /** Raised by {@link #onStageEnd} for a NaN stage when {@code abortOnNan} is on. */
    public static final class NonFiniteException extends RuntimeException {

        private static final long serialVersionUID = 1L;

        NonFiniteException(String label) {
            super("encountered NaN in " + label + " - aborting");
        }
    }
}

package com.eonmux.neuralfree.acceleration.backend;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Device-side time accumulated per kernel or transfer name while
 * {@code -Dneuralfree.gpu.profile=true} is set: a queue that measures its dispatches adds the
 * elapsed device nanoseconds of each completed command under its kernel name here.
 *
 * <p>The sums are diagnostic: a dispatch records device timestamps only outside a graph capture
 * or replay, and the wall time of a command includes nothing but its own execution on the stream.
 * A report groups by name, because a queue sees kernel names, not call sites.</p>
 */
public final class KernelTiming {

    /** The system property that turns per-command device timing on. */
    public static final String PROFILE_PROPERTY = "neuralfree.gpu.profile";

    private final Map<String, long[]> totals = new HashMap<>();
    private boolean any;

    /** True once at least one measured command has been added. */
    public synchronized boolean isEmpty() {
        return !any;
    }

    /** Clears every name so the next report covers the commands added from here on. */
    public synchronized void reset() {
        totals.clear();
        any = false;
    }

    /** Adds one completed command's device time under {@code name}. */
    public synchronized void add(String name, long nanos) {
        Objects.requireNonNull(name, "name");
        if (nanos < 0) {
            nanos = 0;
        }
        long[] total = totals.computeIfAbsent(name, k -> new long[2]);
        total[0]++;
        total[1] += nanos;
        any = true;
    }

    /** Total device nanoseconds over every name, for a device-busy versus wall-time comparison. */
    public synchronized long totalNanos() {
        long sum = 0;
        for (long[] total : totals.values()) {
            sum += total[1];
        }
        return sum;
    }

    /** Total dispatch count over every name. */
    public synchronized long totalCount() {
        long sum = 0;
        for (long[] total : totals.values()) {
            sum += total[0];
        }
        return sum;
    }

    /**
     * A name's aggregated device time: {@link #count} dispatches taking {@link #totalNanos}
     * device nanoseconds in total.
     */
    public static final class Entry {
        private final String name;
        private final long count;
        private final long totalNanos;

        Entry(String name, long count, long totalNanos) {
            this.name = name;
            this.count = count;
            this.totalNanos = totalNanos;
        }

        /** The kernel or transfer name the queue reported. */
        public String name() {
            return name;
        }

        /** Dispatches of this name. */
        public long count() {
            return count;
        }

        /** Device nanoseconds the dispatches took in total. */
        public long totalNanos() {
            return totalNanos;
        }

        /** Mean device nanoseconds per dispatch. */
        public long meanNanos() {
            return count == 0 ? 0 : totalNanos / count;
        }
    }

    /** Entries sorted by total device time, largest first. */
    public synchronized List<Entry> snapshot() {
        List<Entry> entries = new ArrayList<>(totals.size());
        for (Map.Entry<String, long[]> e : totals.entrySet()) {
            entries.add(new Entry(e.getKey(), e.getValue()[0], e.getValue()[1]));
        }
        entries.sort((a, b) -> Long.compare(b.totalNanos(), a.totalNanos()));
        return entries;
    }

    /**
     * The report lines of {@link #snapshot()}: one row per name with count, total milliseconds,
     * mean microseconds and the share of the total device time.
     */
    public synchronized List<String> format() {
        List<String> lines = new ArrayList<>();
        List<Entry> entries = snapshot();
        long total = totalNanos();
        lines.add(String.format("%-40s %9s %12s %10s %6s", "kernel", "count", "total ms", "mean us", "share"));
        for (Entry e : entries) {
            double share = total == 0 ? 0 : 100.0 * e.totalNanos() / total;
            lines.add(String.format("%-40s %9d %12.3f %10.2f %5.1f%%",
                    e.name(), e.count(), e.totalNanos() / 1.0e6, e.meanNanos() / 1.0e3, share));
        }
        lines.add(String.format("%-40s %9d %12.3f", "total", totalCount(), total / 1.0e6));
        return lines;
    }
}

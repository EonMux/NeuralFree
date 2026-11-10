package com.eonmux.neuralfree.demo.tools.compare;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * The comparison of two sides of benchmark rows, as llama.cpp's {@code compare-llama-bench.py} pairs two commits'
 * tests: rows are keyed by their identifying fields, the rows of one key on one side pool (several files of
 * alternated rounds, or the runs of a sweep), and each key present on both sides yields an {@link Entry} with the
 * pooled means, their ratio, Welch's p-value over the pooled samples and a {@link Verdict} against the threshold. The
 * keys present on one side only are kept apart. A key whose rows on one side differ in a configuration field
 * ({@link RowKind#configFields()}) the key leaves out pools several configurations into one mean; the comparison
 * lists each such field with its values ({@link #pooled()}), so a key given with {@code --key} that merges them is
 * never silent. Immutable.
 */
public final class BenchComparison {

    /** How a key's new mean stands against its base mean. */
    public enum Verdict {
        /** Worse by more than the threshold, and significant (or without samples to test). */
        REGRESSION("regression"),
        /** Better by more than the threshold, and significant (or without samples to test). */
        IMPROVEMENT("improvement"),
        /** Past the threshold, but the p-value does not reach the significance level. */
        NOT_SIGNIFICANT("not significant"),
        /** Within the threshold. */
        UNCHANGED("unchanged");

        private final String label;

        Verdict(String label) {
            this.label = label;
        }

        /** The report label. */
        public String label() {
            return label;
        }
    }

    /** The settings of a comparison. Immutable. */
    public static final class Settings {
        private final String metric;
        private final List<String> key;
        private final double thresholdPct;
        private final double alpha;

        /**
         * @param metric       the metric to compare, or null for each kind's default
         * @param key          the identifying fields, or null for each kind's default
         * @param thresholdPct the change in percent past which a key regresses or improves, at least 0
         * @param alpha        the significance level, in (0, 1)
         * @throws IllegalArgumentException for a negative threshold, an alpha outside (0, 1) or an empty key
         */
        public Settings(String metric, List<String> key, double thresholdPct, double alpha) {
            if (!(thresholdPct >= 0) || Double.isInfinite(thresholdPct)) {
                throw new IllegalArgumentException("threshold " + thresholdPct + "% must be a finite value >= 0");
            }
            if (!(alpha > 0 && alpha < 1)) {
                throw new IllegalArgumentException("alpha " + alpha + " must be between 0 and 1");
            }
            if (key != null && key.isEmpty()) {
                throw new IllegalArgumentException("the key names no field");
            }
            this.metric = metric;
            this.key = key == null ? null : Collections.unmodifiableList(new ArrayList<>(key));
            this.thresholdPct = thresholdPct;
            this.alpha = alpha;
        }

        /** The metric of rows of {@code kind}. */
        String metric(RowKind kind) {
            return metric != null ? metric : kind.defaultMetric();
        }

        /** The identifying fields of rows of {@code kind}. */
        List<String> key(RowKind kind) {
            return key != null ? key : kind.defaultKey();
        }
    }

    /** One key present on both sides. Immutable. */
    public static final class Entry {
        private final RowKind kind;
        private final String label;
        private final Map<String, String> key;
        private final String metric;
        private final boolean higherIsBetter;
        private final double base;
        private final double next;
        private final int baseSamples;
        private final int newSamples;
        private final double p;
        private final Verdict verdict;

        Entry(RowKind kind, String label, Map<String, String> key, String metric, boolean higherIsBetter,
              double base, double next, int baseSamples, int newSamples, double p, Verdict verdict) {
            this.kind = kind;
            this.label = label;
            this.key = Collections.unmodifiableMap(new LinkedHashMap<>(key));
            this.metric = metric;
            this.higherIsBetter = higherIsBetter;
            this.base = base;
            this.next = next;
            this.baseSamples = baseSamples;
            this.newSamples = newSamples;
            this.p = p;
            this.verdict = verdict;
        }

        public RowKind kind() {
            return kind;
        }

        /** The test label of the key's rows. */
        public String label() {
            return label;
        }

        /** The identifying fields and their normalized values. */
        public Map<String, String> key() {
            return key;
        }

        public String metric() {
            return metric;
        }

        public boolean higherIsBetter() {
            return higherIsBetter;
        }

        /** The base side's pooled mean. */
        public double base() {
            return base;
        }

        /** The new side's pooled mean. */
        public double next() {
            return next;
        }

        public int baseSamples() {
            return baseSamples;
        }

        public int newSamples() {
            return newSamples;
        }

        /** {@code new / base}. */
        public double ratio() {
            return next / base;
        }

        /** The change in percent, positive when the new side is better whatever the metric's direction. */
        public double gainPct() {
            double change = 100.0 * (next - base) / base;
            return higherIsBetter ? change : -change;
        }

        /** Welch's two-sided p-value, NaN when a side has fewer than two samples. */
        public double p() {
            return p;
        }

        public Verdict verdict() {
            return verdict;
        }
    }

    private final List<Entry> entries;
    private final List<String> onlyBase;
    private final List<String> onlyNew;
    private final List<String> pooled;

    private BenchComparison(List<Entry> entries, List<String> onlyBase, List<String> onlyNew, List<String> pooled) {
        this.entries = Collections.unmodifiableList(entries);
        this.onlyBase = Collections.unmodifiableList(onlyBase);
        this.onlyNew = Collections.unmodifiableList(onlyNew);
        this.pooled = Collections.unmodifiableList(pooled);
    }

    /** The keys on both sides, in the base side's order. */
    public List<Entry> entries() {
        return entries;
    }

    /** The keys only the base side holds, as labels with their key values. */
    public List<String> onlyBase() {
        return onlyBase;
    }

    /** The keys only the new side holds. */
    public List<String> onlyNew() {
        return onlyNew;
    }

    /**
     * The keys whose rows on one side differ in a configuration field the key leaves out, one line per side, key
     * and field: the side, the key's label and values, the field and each value it holds with the first row holding
     * it.
     */
    public List<String> pooled() {
        return pooled;
    }

    /** Whether any entry regressed. */
    public boolean regressed() {
        for (Entry e : entries) {
            if (e.verdict() == Verdict.REGRESSION) {
                return true;
            }
        }
        return false;
    }

    /**
     * Compares {@code next} against {@code base}.
     *
     * @throws IllegalArgumentException when a row lacks the compared metric, holds a non-positive base mean, or a
     *                                  sample is not a finite number
     */
    public static BenchComparison of(List<BenchRow> base, List<BenchRow> next, Settings settings) {
        Objects.requireNonNull(settings, "settings");
        Map<String, List<BenchRow>> baseGroups = group(base, settings);
        Map<String, List<BenchRow>> newGroups = group(next, settings);
        List<String> pooled = new ArrayList<>();
        pooledConfigurations("base", baseGroups, settings, pooled);
        pooledConfigurations("new", newGroups, settings, pooled);
        List<Entry> entries = new ArrayList<>();
        List<String> onlyBase = new ArrayList<>();
        List<String> onlyNew = new ArrayList<>();
        for (Map.Entry<String, List<BenchRow>> g : baseGroups.entrySet()) {
            List<BenchRow> other = newGroups.get(g.getKey());
            if (other == null) {
                onlyBase.add(describe(g.getValue().get(0), settings));
            } else {
                entries.add(compare(g.getValue(), other, settings));
            }
        }
        for (Map.Entry<String, List<BenchRow>> g : newGroups.entrySet()) {
            if (!baseGroups.containsKey(g.getKey())) {
                onlyNew.add(describe(g.getValue().get(0), settings));
            }
        }
        return new BenchComparison(entries, onlyBase, onlyNew, pooled);
    }

    private static Map<String, List<BenchRow>> group(List<BenchRow> rows, Settings settings) {
        Map<String, List<BenchRow>> groups = new LinkedHashMap<>();
        for (BenchRow r : rows) {
            StringBuilder k = new StringBuilder(r.kind().name());
            for (String field : settings.key(r.kind())) {
                k.append('\u0000').append(r.keyValue(field));
            }
            groups.computeIfAbsent(k.toString(), x -> new ArrayList<>()).add(r);
        }
        return groups;
    }

    /**
     * Adds to {@code out} a line for each group of {@code side} and each configuration field outside the key that its
     * rows hold more than one value of.
     */
    private static void pooledConfigurations(String side, Map<String, List<BenchRow>> groups, Settings settings,
                                             List<String> out) {
        for (List<BenchRow> rows : groups.values()) {
            BenchRow first = rows.get(0);
            List<String> key = settings.key(first.kind());
            for (String field : first.kind().configFields()) {
                if (key.contains(field)) {
                    continue;
                }
                Map<String, String> values = new LinkedHashMap<>();
                for (BenchRow r : rows) {
                    values.putIfAbsent(r.keyValue(field), r.source());
                }
                if (values.size() > 1) {
                    StringBuilder line = new StringBuilder(side).append(' ').append(describe(first, settings))
                            .append(" pools rows that differ in ").append(field).append(':');
                    for (Map.Entry<String, String> v : values.entrySet()) {
                        line.append(" '").append(v.getKey()).append("' (").append(v.getValue()).append(')');
                    }
                    out.add(line.toString());
                }
            }
        }
    }

    private static Map<String, String> keyOf(BenchRow r, Settings settings) {
        Map<String, String> key = new LinkedHashMap<>();
        for (String field : settings.key(r.kind())) {
            key.put(field, r.keyValue(field));
        }
        return key;
    }

    private static String describe(BenchRow r, Settings settings) {
        return r.kind().label() + " " + r.label() + " " + keyOf(r, settings) + " (" + r.source() + ")";
    }

    private static Entry compare(List<BenchRow> base, List<BenchRow> next, Settings settings) {
        BenchRow first = base.get(0);
        RowKind kind = first.kind();
        String metric = settings.metric(kind);
        double[] a = pooled(base, metric);
        double[] b = pooled(next, metric);
        double baseMean = WelchTest.mean(a);
        double newMean = WelchTest.mean(b);
        if (!(baseMean > 0)) {
            throw new IllegalArgumentException(first.source() + ": the base mean of " + metric + " is " + baseMean
                    + "; a ratio needs a positive base");
        }
        WelchTest.Result test = WelchTest.test(a, b);
        double p = test == null ? Double.NaN : test.p();
        boolean higher = kind.higherIsBetter(metric);
        double change = 100.0 * (newMean - baseMean) / baseMean;
        double gain = higher ? change : -change;
        boolean significant = Double.isNaN(p) || p < settings.alpha;
        Verdict verdict;
        if (Math.abs(gain) <= settings.thresholdPct) {
            verdict = Verdict.UNCHANGED;
        } else if (!significant) {
            verdict = Verdict.NOT_SIGNIFICANT;
        } else {
            verdict = gain < 0 ? Verdict.REGRESSION : Verdict.IMPROVEMENT;
        }
        return new Entry(kind, first.label(), keyOf(first, settings), metric, higher, baseMean, newMean, a.length,
                b.length, p, verdict);
    }

    /**
     * The pooled observations of a key's rows: every row's raw samples when each row carries them, else one value
     * per row.
     */
    private static double[] pooled(List<BenchRow> rows, String metric) {
        List<double[]> parts = new ArrayList<>();
        boolean allSampled = true;
        for (BenchRow r : rows) {
            double[] s = r.samples(metric);
            if (s == null || s.length == 0) {
                allSampled = false;
                break;
            }
            parts.add(s);
        }
        if (!allSampled) {
            double[] values = new double[rows.size()];
            for (int i = 0; i < values.length; i++) {
                values[i] = rows.get(i).metric(metric);
            }
            return values;
        }
        int n = 0;
        for (double[] s : parts) {
            n += s.length;
        }
        double[] out = new double[n];
        int at = 0;
        for (double[] s : parts) {
            System.arraycopy(s, 0, out, at, s.length);
            at += s.length;
        }
        return out;
    }
}

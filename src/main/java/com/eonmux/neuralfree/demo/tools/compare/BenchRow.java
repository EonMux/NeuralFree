package com.eonmux.neuralfree.demo.tools.compare;

import java.math.BigInteger;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * One result object of a benchmark file: its {@link RowKind}, where it was read, and its fields. Key values are
 * normalized so runs of the same test from different machines or tools match: a model path keeps its file name, a
 * backend list its case-folded names, a boolean its 0/1 spelling (llama.cpp's batched-bench writes flags as integers)
 * and an integral number its integer spelling. Immutable.
 */
public final class BenchRow {

    private final RowKind kind;
    private final String source;
    private final Map<String, Object> fields;

    BenchRow(RowKind kind, String source, Map<String, Object> fields) {
        this.kind = kind;
        this.source = source;
        this.fields = Collections.unmodifiableMap(new LinkedHashMap<>(fields));
    }

    /** The row's kind. */
    public RowKind kind() {
        return kind;
    }

    /** The file and line or index the row was read from. */
    public String source() {
        return source;
    }

    /** Whether the row carries {@code field}. */
    public boolean has(String field) {
        return fields.containsKey(field);
    }

    /**
     * The normalized value of {@code field} for keying; a field the row lacks or holds null keys as its kind's
     * {@link RowKind#absentKeyValue}.
     */
    public String keyValue(String field) {
        Object v = fields.get(field);
        if (v == null) {
            return kind.absentKeyValue(field);
        }
        if (v instanceof String) {
            String s = ((String) v).trim();
            if (field.equals("model_filename") || field.equals("model_id")) {
                return fileName(s);
            }
            return field.equals("backends") ? s.toLowerCase(Locale.ROOT) : s;
        }
        if (v instanceof Boolean) {
            return (Boolean) v ? "1" : "0";
        }
        if (v instanceof Double) {
            double d = (Double) v;
            if (d == Math.rint(d) && !Double.isInfinite(d) && Math.abs(d) < 1.0e15) {
                return Long.toString((long) d);
            }
            return Double.isInfinite(d) ? "inf" : Double.toString(d);
        }
        return String.valueOf(v);
    }

    /**
     * The numeric value of {@code metric}.
     *
     * @throws IllegalArgumentException when the row lacks the field or it is not a number
     */
    public double metric(String metric) {
        Object v = fields.get(metric);
        if (!(v instanceof Number)) {
            throw new IllegalArgumentException(source + ": the " + kind.label() + " row has no numeric '" + metric
                    + "'" + (v == null ? "" : " (it holds " + v + ")"));
        }
        return ((Number) v).doubleValue();
    }

    /**
     * The raw samples behind {@code metric}, in the metric's unit, or null when the row records none: a bench test's
     * {@code samples_ts} or {@code samples_ns}, a serve result's per-request time to first token, inter-token
     * latencies or end-to-end latencies (saved in seconds, reported in milliseconds).
     *
     * @throws IllegalArgumentException when a sample list holds something else than numbers
     */
    public double[] samples(String metric) {
        switch (kind) {
            case BENCH:
                if (metric.equals("avg_ts")) {
                    return numbers("samples_ts", 1.0);
                }
                return metric.equals("avg_ns") ? numbers("samples_ns", 1.0) : null;
            case SERVE:
                if (metric.equals("mean_ttft_ms")) {
                    return numbers("ttfts", 1.0e3);
                }
                if (metric.equals("mean_itl_ms")) {
                    return numbers("itls", 1.0e3);
                }
                return metric.equals("mean_e2el_ms") ? numbers("latencies", 1.0e3) : null;
            default:
                return null;
        }
    }

    /**
     * The row's test as a short label: {@code pp512}, {@code tg128}, {@code pp512+tg128} with {@code @ d<depth>}
     * for a bench test, the cell's shape for batched-bench, the arrival pattern for a serve result.
     */
    public String label() {
        switch (kind) {
            case BENCH: {
                String p = keyValue("n_prompt");
                String g = keyValue("n_gen");
                String d = keyValue("n_depth");
                String test = !p.equals("0") && !g.equals("0") ? "pp" + p + "+tg" + g
                        : !p.equals("0") ? "pp" + p : "tg" + g;
                return d.isEmpty() || d.equals("0") ? test : test + " @ d" + d;
            }
            case BATCHED:
                return "pp" + keyValue("pp") + " tg" + keyValue("tg") + " pl" + keyValue("pl");
            default:
                return "rate " + keyValue("request_rate") + " conc " + keyValue("max_concurrency");
        }
    }

    /** The numbers of a list field (one level of nested lists flattened) times {@code scale}, or null if absent. */
    private double[] numbers(String field, double scale) {
        Object v = fields.get(field);
        if (!(v instanceof List)) {
            return null;
        }
        List<Double> out = new ArrayList<>();
        for (Object item : (List<?>) v) {
            if (item instanceof List) {
                for (Object inner : (List<?>) item) {
                    out.add(number(field, inner) * scale);
                }
            } else if (item != null) {
                out.add(number(field, item) * scale);
            }
        }
        double[] values = new double[out.size()];
        for (int i = 0; i < values.length; i++) {
            values[i] = out.get(i);
        }
        return values;
    }

    private double number(String field, Object item) {
        if (item instanceof Long || item instanceof Double || item instanceof BigInteger) {
            return ((Number) item).doubleValue();
        }
        throw new IllegalArgumentException(source + ": '" + field + "' holds " + item + ", not a number");
    }

    private static String fileName(String path) {
        int slash = Math.max(path.lastIndexOf('/'), path.lastIndexOf('\\'));
        return slash >= 0 ? path.substring(slash + 1) : path;
    }
}

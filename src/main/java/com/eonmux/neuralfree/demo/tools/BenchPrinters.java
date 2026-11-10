package com.eonmux.neuralfree.demo.tools;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.StringJoiner;

/**
 * The output writers of {@code nftool bench}, llama.cpp llama-bench's printers
 * ({@code tools/llama-bench/llama-bench.cpp}): one header per stream, one block per
 * {@link BenchTest} and a footer. The five formats keep the reference's shapes — CSV quotes every value,
 * JSON gathers the runs in one array with the raw samples, JSONL writes one object per run, the Markdown
 * table prints only the parameters that vary and the SQL printer emits a
 * {@code CREATE TABLE IF NOT EXISTS llama_bench} plus one INSERT per run.
 */
final class BenchPrinters {

    /** One benchmark stream: a header, one block per test, a footer. */
    interface BenchPrinter {

        /** The stream's preamble, before the first test. The default prints nothing. */
        default void printHeader(BenchGrid grid) throws IOException {
        }

        /** The block of one completed test. */
        void printTest(BenchTest test) throws IOException;

        /** The stream's closing, after the last test. The default prints nothing. */
        default void printFooter() throws IOException {
        }
    }

    private BenchPrinters() {
    }

    /**
     * The printer of a format writing to {@code out}; {@code none} returns null like the reference's
     * {@code create_printer}.
     */
    static BenchPrinter create(BenchOutput format, Appendable out, BenchGrid grid) {
        switch (format) {
            case NONE:
                return null;
            case CSV:
                return new CsvPrinter(out);
            case JSON:
                return new JsonPrinter(out);
            case JSONL:
                return new JsonlPrinter(out);
            case MARKDOWN:
                return new MarkdownPrinter(out, grid);
            case SQL:
                return new SqlPrinter(out);
            default:
                throw new IllegalStateException("no printer for " + format);
        }
    }

    /** Escapes a CSV field: wrapped in quotes, interior quotes doubled. */
    static String escapeCsv(String field) {
        return "\"" + field.replace("\"", "\"\"") + "\"";
    }

    /** Escapes a JSON string value: quotes, backslashes and the control characters. */
    static String escapeJson(String value) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < value.length(); i++) {
            char c = value.charAt(i);
            if (c == '"') {
                sb.append("\\\"");
            } else if (c == '\\') {
                sb.append("\\\\");
            } else if (c <= 0x1f) {
                sb.append(String.format("\\u%04x", (int) c));
            } else {
                sb.append(c);
            }
        }
        return sb.toString();
    }

    /** A field value as JSON carries it: strings quoted and escaped, bools spelled out, numbers bare. */
    static String jsonValue(String field, String value) {
        if (BenchTest.fieldType(field) == BenchTest.FieldType.STRING) {
            return "\"" + escapeJson(value) + "\"";
        }
        return value;
    }

    /** The joined {@code samples_ns} of a test. */
    static String samplesNs(BenchTest test) {
        StringJoiner j = new StringJoiner(", ");
        for (long s : test.samplesNs) {
            j.add(Long.toString(s));
        }
        return j.toString();
    }

    /** The joined per-sample throughputs of a test, six decimal places like the reference's doubles. */
    static String samplesTs(BenchTest test) {
        StringJoiner j = new StringJoiner(", ");
        for (double t : test.getTs()) {
            j.add(BenchTest.doubleValue(t));
        }
        return j.toString();
    }

    /** The CSV printer: a header row of field names, then one all-quoted row per test. */
    private static final class CsvPrinter implements BenchPrinter {

        private final Appendable out;

        CsvPrinter(Appendable out) {
            this.out = out;
        }

        @Override
        public void printHeader(BenchGrid grid) throws IOException {
            out.append(String.join(",", BenchTest.fields())).append('\n');
        }

        @Override
        public void printTest(BenchTest test) throws IOException {
            StringJoiner row = new StringJoiner(",");
            for (String v : test.values()) {
                row.add(escapeCsv(v));
            }
            out.append(row.toString()).append('\n');
        }
    }

    /** The JSON printer: one array around one indented object per test, with the raw samples. */
    private static final class JsonPrinter implements BenchPrinter {

        private final Appendable out;
        private boolean first = true;

        JsonPrinter(Appendable out) {
            this.out = out;
        }

        @Override
        public void printHeader(BenchGrid grid) throws IOException {
            out.append("[\n");
        }

        @Override
        public void printTest(BenchTest test) throws IOException {
            if (first) {
                first = false;
            } else {
                out.append(",\n");
            }
            out.append("  {\n");
            Map<String, String> values = test.map();
            for (Map.Entry<String, String> e : values.entrySet()) {
                out.append("    \"").append(e.getKey()).append("\": ")
                        .append(jsonValue(e.getKey(), e.getValue())).append(",\n");
            }
            out.append("    \"samples_ns\": [ ").append(samplesNs(test)).append(" ],\n");
            out.append("    \"samples_ts\": [ ").append(samplesTs(test)).append(" ]\n");
            out.append("  }");
        }

        @Override
        public void printFooter() throws IOException {
            out.append("\n]\n");
        }
    }

    /** The JSONL printer: one object per line, fields inline, with the raw samples. */
    private static final class JsonlPrinter implements BenchPrinter {

        private final Appendable out;

        JsonlPrinter(Appendable out) {
            this.out = out;
        }

        @Override
        public void printTest(BenchTest test) throws IOException {
            out.append("{");
            for (Map.Entry<String, String> e : test.map().entrySet()) {
                out.append("\"").append(e.getKey()).append("\": ")
                        .append(jsonValue(e.getKey(), e.getValue())).append(", ");
            }
            out.append("\"samples_ns\": [ ").append(samplesNs(test)).append(" ],");
            out.append("\"samples_ts\": [ ").append(samplesTs(test)).append(" ]");
            out.append("}\n");
        }
    }

    /** The Markdown printer: llama-bench's table, its columns the parameters that vary. */
    private static final class MarkdownPrinter implements BenchPrinter {

        private final Appendable out;
        private final BenchGrid grid;
        private final List<String> columns;

        MarkdownPrinter(Appendable out, BenchGrid grid) {
            this.out = out;
            this.grid = grid;
            this.columns = columns();
        }

        /** A column's print width; negative widths align left. */
        private static int width(String field) {
            switch (field) {
                case "model":
                    return -30;
                case "t/s":
                    return 20;
                case "size":
                case "params":
                    return 10;
                case "n_gpu_layers":
                    return 3;
                case "n_threads":
                    return 7;
                case "n_batch":
                    return 7;
                case "n_ubatch":
                    return 8;
                case "type_k":
                case "type_v":
                    return 6;
                case "split_mode":
                    return 6;
                case "load_mode":
                    return 10;
                case "devices":
                    return -12;
                case "test":
                    return 15;
                default:
                    int w = Math.max(field.length(), 10);
                    return BenchTest.fieldType(field) == BenchTest.FieldType.STRING ? -w : w;
            }
        }

        /** A column's display name, llama-bench's abbreviations. */
        private static String displayName(String field) {
            switch (field) {
                case "n_gpu_layers":
                    return "ngl";
                case "n_threads":
                    return "threads";
                case "type_k":
                case "type_v":
                    return field;
                case "split_mode":
                    return "sm";
                case "no_kv_offload":
                    return "nkvo";
                case "devices":
                    return "dev";
                case "tensor_split":
                    return "ts";
                case "tensor_buft_overrides":
                    return "ot";
                case "load_mode":
                    return "lm";
                case "embeddings":
                    return "embd";
                case "tg_readout":
                    return "readout";
                default:
                    return field;
            }
        }

        /** The columns of this sweep: the fixed facts, the varying parameters, the test and its t/s. */
        private List<String> columns() {
            List<String> cols = new ArrayList<>(List.of("model", "size", "params", "backends"));
            if (grid.ngl.isEmpty() ? BenchGrid.DEFAULT_NGL != 0 : hasNonZero(grid.ngl)) {
                cols.add("n_gpu_layers");
            }
            if (varies(grid.nCpuMoe, grid.base.nCpuMoe())) {
                cols.add("n_cpu_moe");
            }
            if (grid.threadsGiven) {
                cols.add("n_threads");
            }
            if (varies(grid.nBatch, BenchGrid.DEFAULT_N_BATCH)) {
                cols.add("n_batch");
            }
            if (varies(grid.nUbatch, BenchGrid.DEFAULT_N_UBATCH)) {
                cols.add("n_ubatch");
            }
            if (grid.typeK.size() > 1 || (!grid.typeK.isEmpty() && grid.typeK.get(0) != BenchGrid.DEFAULT_KV_TYPE)) {
                cols.add("type_k");
            }
            if (grid.typeV.size() > 1 || (!grid.typeV.isEmpty() && grid.typeV.get(0) != BenchGrid.DEFAULT_KV_TYPE)) {
                cols.add("type_v");
            }
            if (variesMainGpu()) {
                cols.add("main_gpu");
            }
            if (variesSplitMode()) {
                cols.add("split_mode");
            }
            if (variesNoKvOffload()) {
                cols.add("no_kv_offload");
            }
            if (variesDevices()) {
                cols.add("devices");
            }
            if (variesTensorSplit()) {
                cols.add("tensor_split");
            }
            if (variesTensorOverrides()) {
                cols.add("tensor_buft_overrides");
            }
            if (variesLoadMode()) {
                cols.add("load_mode");
            }
            if (variesEmbeddings()) {
                cols.add("embeddings");
            }
            if (grid.readout.kind() != BenchReadout.Kind.LOGITS) {
                cols.add("tg_readout");
            }
            cols.add("test");
            cols.add("t/s");
            return cols;
        }

        /** Whether {@code -mg} leaves the base configuration's main GPU. */
        private boolean variesMainGpu() {
            return grid.mainGpus.size() > 1
                    || (!grid.mainGpus.isEmpty() && grid.mainGpus.get(0) != grid.base.mainGpu());
        }

        /** Whether {@code -sm} leaves the base configuration's split mode. */
        private boolean variesSplitMode() {
            return grid.splitModes.size() > 1
                    || (!grid.splitModes.isEmpty() && grid.splitModes.get(0) != grid.base.splitMode());
        }

        /** Whether {@code -nkvo} leaves the base configuration's KV offload. */
        private boolean variesNoKvOffload() {
            return grid.noKvOffload.size() > 1
                    || (!grid.noKvOffload.isEmpty() && grid.noKvOffload.get(0) != !grid.base.kvOffload());
        }

        /** Whether {@code -dev} leaves llama-bench's {@code auto} default. */
        private boolean variesDevices() {
            return grid.devicesGiven
                    && (grid.devices.size() > 1 || !grid.devices.get(0).isEmpty());
        }

        /** Whether {@code -ot} overrides anything, or selects between members. */
        private boolean variesTensorOverrides() {
            return grid.overridesGiven
                    && (grid.tensorOverrides.size() > 1 || !grid.tensorOverrides.get(0).isEmpty());
        }

        /** Whether {@code -lm} leaves llama-bench's {@code auto} default. */
        private boolean variesLoadMode() {
            return grid.loadModeGiven
                    && (grid.loadModes.size() > 1 || grid.loadModes.get(0) != BenchLoadMode.AUTO);
        }

        /** Whether {@code -embd} asks for the embedding mode. */
        private boolean variesEmbeddings() {
            return grid.embeddings.size() > 1
                    || (!grid.embeddings.isEmpty() && grid.embeddings.get(0).booleanValue());
        }

        /** Whether {@code -ts} leaves the base configuration's tensor shares. */
        private boolean variesTensorSplit() {
            return grid.tensorSplits.size() > 1
                    || (!grid.tensorSplits.isEmpty()
                        && !Arrays.equals(grid.tensorSplits.get(0), grid.base.tensorSplit()));
        }

        private static boolean hasNonZero(List<Integer> v) {
            for (int x : v) {
                if (x != 0) {
                    return true;
                }
            }
            return false;
        }

        /** Whether a parameter leaves its single default: size beyond one, or one non-default value. */
        private static boolean varies(List<Integer> v, int dflt) {
            return v.size() > 1 || (!v.isEmpty() && v.get(0) != dflt);
        }

        /** One cell's value of a test, as the Markdown table shows it. */
        private String cell(BenchTest test, String field) {
            switch (field) {
                case "model":
                    return test.modelType;
                case "size":
                    return size(test.modelSize);
                case "params":
                    return params(test.modelNParams);
                case "backends":
                    return test.backends;
                case "test":
                    return test.inst.testName();
                case "t/s":
                    return String.format(java.util.Locale.ROOT, "%.2f ± %.2f", test.avgTs(), test.stdevTs());
                default:
                    return test.map().get(field);
            }
        }

        /** A byte size as llama-bench prints one: MiB under a GiB, GiB from there. */
        private static String size(long bytes) {
            if (bytes < 1024L * 1024 * 1024) {
                return String.format(java.util.Locale.ROOT, "%.2f MiB", bytes / 1024.0 / 1024.0);
            }
            return String.format(java.util.Locale.ROOT, "%.2f GiB", bytes / 1024.0 / 1024.0 / 1024.0);
        }

        /** A parameter count as llama-bench prints one: M under a billion, B from there. */
        private static String params(long n) {
            if (n < 1000L * 1000 * 1000) {
                return String.format(java.util.Locale.ROOT, "%.2f M", n / 1e6);
            }
            return String.format(java.util.Locale.ROOT, "%.2f B", n / 1e9);
        }

        @Override
        public void printHeader(BenchGrid g) throws IOException {
            out.append("|");
            for (String field : columns) {
                out.append(" ").append(pad(displayName(field), width(field))).append(" |");
            }
            out.append("\n|");
            for (String field : columns) {
                int w = Math.abs(width(field));
                out.append(" ").append("-".repeat(w - 1)).append(width(field) > 0 ? ":" : "-").append(" |");
            }
            out.append('\n');
        }

        @Override
        public void printTest(BenchTest test) throws IOException {
            out.append("|");
            for (String field : columns) {
                out.append(" ").append(pad(cell(test, field), width(field))).append(" |");
            }
            out.append('\n');
        }

        /** Pads to the field width, right or left aligned by its sign (C printf's {@code %*s}). */
        private static String pad(String v, int width) {
            StringBuilder sb = new StringBuilder();
            int pad = Math.abs(width) - displayLength(v);
            if (pad < 0) {
                pad = 0;
            }
            if (width > 0) {
                sb.append(" ".repeat(pad));
            }
            sb.append(v);
            if (width < 0) {
                sb.append(" ".repeat(pad));
            }
            return sb.toString();
        }

        /** The display width of a cell: characters, not UTF-8 bytes. */
        private static int displayLength(String v) {
            return v.codePointCount(0, v.length());
        }
    }

    /** The SQL printer: the table DDL, then one INSERT per test. */
    private static final class SqlPrinter implements BenchPrinter {

        private final Appendable out;

        SqlPrinter(Appendable out) {
            this.out = out;
        }

        /** The SQL column type of a field: TEXT, INTEGER or REAL. */
        private static String sqlType(String field) {
            switch (BenchTest.fieldType(field)) {
                case STRING:
                    return "TEXT";
                case FLOAT:
                    return "REAL";
                case BOOL:
                case INT:
                default:
                    return "INTEGER";
            }
        }

        @Override
        public void printHeader(BenchGrid grid) throws IOException {
            String[] fields = BenchTest.fields();
            out.append("CREATE TABLE IF NOT EXISTS llama_bench (\n");
            for (int i = 0; i < fields.length; i++) {
                out.append("  ").append(fields[i]).append(' ').append(sqlType(fields[i]))
                        .append(i < fields.length - 1 ? "," : "").append('\n');
            }
            out.append(");\n\n");
        }

        @Override
        public void printTest(BenchTest test) throws IOException {
            out.append("INSERT INTO llama_bench (").append(String.join(", ", BenchTest.fields())).append(") ");
            out.append("VALUES (");
            String[] values = test.values();
            for (int i = 0; i < values.length; i++) {
                out.append('\'').append(values[i]).append('\'').append(i < values.length - 1 ? ", " : "");
            }
            out.append(");\n");
        }
    }
}

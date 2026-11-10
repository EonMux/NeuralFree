package com.eonmux.neuralfree.demo.tools;

import com.eonmux.neuralfree.demo.tools.compare.BenchComparison;
import com.eonmux.neuralfree.demo.tools.compare.BenchResultFiles;
import com.eonmux.neuralfree.demo.tools.compare.BenchRow;
import com.eonmux.neuralfree.utils.json.Json;

import java.io.IOException;
import java.io.PrintStream;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

/**
 * {@code nftool bench-compare}: the two-sided comparison of llama.cpp's {@code compare-llama-bench.py} over the
 * result files of {@code nftool bench}, {@code nftool batched-bench} and {@code nftool bench-serve} (and of the
 * references they mirror). Each side is one or more files whose rows pool by key, so alternated base/new rounds of
 * one session compare as two populations; every key on both sides prints its base and new means, their ratio, the
 * change in the better direction, Welch's p-value over the pooled raw samples, and a verdict against
 * {@code --threshold}. With {@code --fail-on-regress} a significant regression past the threshold exits with
 * {@value #REGRESSION_EXIT}, so a script can gate on it.
 */
public final class BenchCompareCommand {

    /** The exit status of {@code --fail-on-regress} when a key regressed. */
    static final int REGRESSION_EXIT = 3;

    private static final Json.Style REPORT_STYLE = new Json.Style(false, "  ", ",", ": ", false);

    /** The command surface: {@code --help} and every usage error print it. */
    static final String USAGE =
            "nftool bench-compare [options] <base>[,<base>...] <new>[,<new>...]\n"
            + "  each side is one or more result files (json, jsonl) of bench, batched-bench or bench-serve;\n"
            + "  the rows of one key pool across a side's files\n"
            + "options:\n"
            + "  -h, --help                 this text\n"
            + "  --metric <field>           the field compared (default avg_ts, speed, output_throughput)\n"
            + "  --key <field,...>          the fields identifying a row (default per result kind)\n"
            + "  --threshold <pct>          the change counted as a regression or improvement (default 2)\n"
            + "  --alpha <p>                the significance level of Welch's t-test (default 0.05)\n"
            + "  --fail-on-regress          exit " + REGRESSION_EXIT + " when a key regressed\n"
            + "  -o, --output <md|json>     the report format (default md)\n";

    /** The parsed command line. Immutable. */
    static final class Args {
        final List<Path> base;
        final List<Path> next;
        final BenchComparison.Settings settings;
        final boolean failOnRegress;
        final boolean json;

        Args(List<Path> base, List<Path> next, BenchComparison.Settings settings, boolean failOnRegress,
             boolean json) {
            this.base = base;
            this.next = next;
            this.settings = settings;
            this.failOnRegress = failOnRegress;
            this.json = json;
        }
    }

    /** A command line the command refuses. */
    static final class UsageException extends Exception {
        UsageException(String message) {
            super(message);
        }
    }

    private BenchCompareCommand() {
    }

    public static void run(String[] args) {
        try {
            Args parsed = parse(args);
            if (parsed == null) {
                System.out.print(USAGE);
                return;
            }
            int status = compare(parsed, System.out);
            if (status != 0) {
                System.exit(status);
            }
        } catch (UsageException e) {
            System.err.println("Error: " + e.getMessage());
            System.err.print(USAGE);
            System.exit(2);
        } catch (IOException | IllegalArgumentException e) {
            System.err.println("Error: " + e.getMessage());
            System.exit(1);
        }
    }

    /** Parses the command line; null when it asks for {@code --help}. */
    static Args parse(String[] args) throws UsageException {
        String metric = null;
        List<String> key = null;
        double threshold = 2.0;
        double alpha = 0.05;
        boolean failOnRegress = false;
        boolean json = false;
        List<String> positional = new ArrayList<>();
        for (int i = 0; i < args.length; i++) {
            String a = args[i];
            switch (a) {
                case "-h":
                case "--help":
                    return null;
                case "--metric":
                    metric = value(args, ++i, a);
                    break;
                case "--key":
                    key = fields(value(args, ++i, a), a);
                    break;
                case "--threshold":
                    threshold = number(value(args, ++i, a), a);
                    break;
                case "--alpha":
                    alpha = number(value(args, ++i, a), a);
                    break;
                case "--fail-on-regress":
                    failOnRegress = true;
                    break;
                case "-o":
                case "--output": {
                    String v = value(args, ++i, a).toLowerCase(Locale.ROOT);
                    if (!v.equals("md") && !v.equals("json")) {
                        throw new UsageException(a + " " + v + " is invalid; use md or json");
                    }
                    json = v.equals("json");
                    break;
                }
                default:
                    if (a.startsWith("-") && a.length() > 1) {
                        throw new UsageException("unknown option " + a);
                    }
                    positional.add(a);
            }
        }
        if (positional.size() != 2) {
            throw new UsageException("bench-compare needs a base and a new side, got " + positional.size()
                    + " argument(s)");
        }
        BenchComparison.Settings settings;
        try {
            settings = new BenchComparison.Settings(metric, key, threshold, alpha);
        } catch (IllegalArgumentException e) {
            throw new UsageException(e.getMessage());
        }
        return new Args(paths(positional.get(0)), paths(positional.get(1)), settings, failOnRegress, json);
    }

    /**
     * Reads both sides, prints the report to {@code out}, and returns the exit status: {@value #REGRESSION_EXIT}
     * when {@code --fail-on-regress} is set and a key regressed, else 0.
     *
     * @throws IOException              when a file cannot be read or holds no benchmark result
     * @throws IllegalArgumentException when the sides share no key, or a row lacks the compared metric
     */
    static int compare(Args args, PrintStream out) throws IOException {
        List<BenchRow> base = read(args.base);
        List<BenchRow> next = read(args.next);
        BenchComparison comparison = BenchComparison.of(base, next, args.settings);
        if (comparison.entries().isEmpty()) {
            throw new IllegalArgumentException("the two sides share no test; base has "
                    + comparison.onlyBase() + ", new has " + comparison.onlyNew());
        }
        if (args.json) {
            out.println(Json.toString(report(comparison), REPORT_STYLE));
        } else {
            printMarkdown(comparison, out);
        }
        return args.failOnRegress && comparison.regressed() ? REGRESSION_EXIT : 0;
    }

    private static List<BenchRow> read(List<Path> files) throws IOException {
        List<BenchRow> rows = new ArrayList<>();
        for (Path f : files) {
            rows.addAll(BenchResultFiles.read(f));
        }
        return rows;
    }

    /**
     * The report as a JSON value: one object per compared key, the one-sided keys, and the keys whose rows on one
     * side mix configurations the key leaves out ({@code pooled}).
     */
    static Map<String, Object> report(BenchComparison comparison) {
        List<Object> entries = new ArrayList<>();
        for (BenchComparison.Entry e : comparison.entries()) {
            Map<String, Object> row = new LinkedHashMap<>();
            row.put("kind", e.kind().label());
            row.put("test", e.label());
            row.put("key", new LinkedHashMap<>(e.key()));
            row.put("metric", e.metric());
            row.put("higher_is_better", e.higherIsBetter());
            row.put("base", e.base());
            row.put("new", e.next());
            row.put("ratio", e.ratio());
            row.put("gain_pct", e.gainPct());
            row.put("base_samples", e.baseSamples());
            row.put("new_samples", e.newSamples());
            row.put("p", Double.isNaN(e.p()) ? null : e.p());
            row.put("verdict", e.verdict().label());
            entries.add(row);
        }
        Map<String, Object> report = new LinkedHashMap<>();
        report.put("entries", entries);
        report.put("only_base", new ArrayList<>(comparison.onlyBase()));
        report.put("only_new", new ArrayList<>(comparison.onlyNew()));
        report.put("pooled", new ArrayList<>(comparison.pooled()));
        report.put("regressed", comparison.regressed());
        return report;
    }

    /**
     * The report as a markdown table, the one-sided keys listed after it, then the keys whose rows on one side mix
     * configurations the key leaves out.
     */
    static void printMarkdown(BenchComparison comparison, PrintStream out) {
        List<BenchComparison.Entry> entries = comparison.entries();
        Set<String> varying = varyingFields(entries);
        out.println("| Kind | Test | Config | Metric | Base | New | Ratio | Gain | p | n | Verdict |");
        out.println("|---|---|---|---|---:|---:|---:|---:|---:|---|---|");
        for (BenchComparison.Entry e : entries) {
            StringBuilder config = new StringBuilder();
            for (Map.Entry<String, String> k : e.key().entrySet()) {
                if (varying.contains(e.kind().name() + '\u0000' + k.getKey())) {
                    config.append(config.length() == 0 ? "" : " ").append(k.getKey()).append('=')
                            .append(k.getValue());
                }
            }
            out.printf(Locale.ROOT, "| %s | %s | %s | %s | %s | %s | %.3f | %+.2f%% | %s | %d/%d | %s |%n",
                    e.kind().label(), e.label(), config.length() == 0 ? "-" : config, e.metric(),
                    magnitude(e.base()), magnitude(e.next()), e.ratio(), e.gainPct(),
                    Double.isNaN(e.p()) ? "-" : String.format(Locale.ROOT, "%.3g", e.p()),
                    e.baseSamples(), e.newSamples(), e.verdict().label());
        }
        for (String only : comparison.onlyBase()) {
            out.println("only in base: " + only);
        }
        for (String only : comparison.onlyNew()) {
            out.println("only in new: " + only);
        }
        for (String pooled : comparison.pooled()) {
            out.println("pooled: " + pooled);
        }
    }

    /** The key fields, per kind, whose value differs between the compared keys of that kind. */
    private static Set<String> varyingFields(List<BenchComparison.Entry> entries) {
        Map<String, Set<String>> values = new LinkedHashMap<>();
        for (BenchComparison.Entry e : entries) {
            for (Map.Entry<String, String> k : e.key().entrySet()) {
                values.computeIfAbsent(e.kind().name() + '\u0000' + k.getKey(), x -> new HashSet<>())
                        .add(k.getValue());
            }
        }
        Set<String> varying = new HashSet<>();
        for (Map.Entry<String, Set<String>> v : values.entrySet()) {
            if (v.getValue().size() > 1) {
                varying.add(v.getKey());
            }
        }
        return varying;
    }

    private static String magnitude(double v) {
        return Math.abs(v) >= 100 ? String.format(Locale.ROOT, "%.2f", v) : String.format(Locale.ROOT, "%.4g", v);
    }

    private static String value(String[] args, int i, String flag) throws UsageException {
        if (i >= args.length) {
            throw new UsageException(flag + " needs a value");
        }
        return args[i];
    }

    private static double number(String v, String flag) throws UsageException {
        try {
            double d = Double.parseDouble(v);
            if (!Double.isFinite(d)) {
                throw new UsageException(flag + " " + v + " must be finite");
            }
            return d;
        } catch (NumberFormatException e) {
            throw new UsageException(flag + " " + v + " is not a number");
        }
    }

    private static List<String> fields(String v, String flag) throws UsageException {
        List<String> out = new ArrayList<>();
        for (String f : v.split(",", -1)) {
            String t = f.trim();
            if (t.isEmpty()) {
                throw new UsageException(flag + " " + v + " names an empty field");
            }
            out.add(t);
        }
        return out;
    }

    private static List<Path> paths(String side) throws UsageException {
        List<Path> out = new ArrayList<>();
        for (String f : side.split(",", -1)) {
            if (f.trim().isEmpty()) {
                throw new UsageException("the side '" + side + "' names an empty file");
            }
            out.add(Paths.get(f.trim()));
        }
        return out;
    }
}

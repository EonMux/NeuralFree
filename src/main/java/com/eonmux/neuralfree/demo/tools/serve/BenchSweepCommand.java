package com.eonmux.neuralfree.demo.tools.serve;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Duration;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.StringJoiner;

import com.eonmux.neuralfree.utils.json.Json;
import com.eonmux.neuralfree.utils.json.JsonException;

/**
 * {@code nftool bench-sweep}: the parameter sweep of vLLM's {@code vllm bench sweep serve} run over
 * NeuralFree's own harness. A serve parameter file and a bench parameter file (the records of
 * {@link ParameterSweep}) name the combinations; for every serve combination the command starts the
 * server over {@code --serve-cmd}, waits for its {@code /health}, and runs every valid bench
 * combination {@code --num-runs} times — each run writes {@code run=N.json} of the benchmark's own
 * result plus the record's parameters, and the combination a {@code summary.json}; the experiment
 * ends with a {@code summary.csv} over every run. {@code --warmup-num-prompts} runs an unmeasured
 * warmup first, {@code --link-vars} skips bench combinations whose linked parameters disagree with
 * the serve combination's, {@code --continue-on-error} records failures as {@code *.failure.json},
 * and {@code --dry-run} prints the plans only. Benchmarks run in this process, over the server the
 * sweep started; the server's health URL derives from its {@code --host}/{@code --port} (default
 * {@code 127.0.0.1:8080}, NeuralFree's own default port).
 */
public final class BenchSweepCommand {

    /** The command surface: {@code --help} and every error print it. */
    static final String USAGE =
            "nftool bench-sweep [options]\n"
            + "options:\n"
            + "  -h, --help                      this text\n"
            + "  --serve-cmd <command>           the server command line, quoted (required)\n"
            + "  --bench-cmd <command>           the bench-serve command line, quoted (required)\n"
            + "  --serve-params <file>           JSON: list of records, or name-to-record object\n"
            + "  --bench-params <file>           JSON: list of records, or name-to-record object\n"
            + "  --link-vars serve=bench,...     linked variables both records must carry equally\n"
            + "  -o, --output-dir <dir>          the experiment's parent, default results\n"
            + "  -e, --experiment-name <n>       the experiment's directory, default the timestamp\n"
            + "  --num-runs <n>                  runs per combination, default 3\n"
            + "  --warmup-num-prompts <n>        prompts of the unmeasured warmup run, default 0\n"
            + "  --continue-on-error             record a failed run and keep going\n"
            + "  --dry-run                       print the commands, run nothing\n"
            + "  --resume                        skip runs whose output files exist\n"
            + "  --show-stdout                   the server's output on this console, not a log file\n"
            + "  --server-ready-timeout <sec>    how long the server's /health may take, default 300\n"
            + "A record's flags replace the command line's (booleans as --flag/--no-flag); every run saves\n"
            + "its benchmark result with --percentile-metrics ttft,tpot,itl,e2el and --save-result set.\n";

    /** An invalid command line; the message names the flag. */
    static final class UsageException extends Exception {

        private static final long serialVersionUID = 1L;

        UsageException(String message) {
            super(message);
        }
    }

    /** The parsed sweep invocation. */
    static final class Args {

        final List<String> serveCmd;
        final List<String> benchCmd;
        final ParameterSweep serveParams;
        final ParameterSweep benchParams;
        final List<String[]> linkVars;
        final Path outputDir;
        final String experimentName;
        final int numRuns;
        final int warmupNumPrompts;
        final boolean continueOnError;
        final boolean dryRun;
        final boolean resume;
        final boolean showStdout;
        final int serverReadyTimeout;

        Args(List<String> serveCmd, List<String> benchCmd, ParameterSweep serveParams,
             ParameterSweep benchParams, List<String[]> linkVars, Path outputDir, String experimentName,
             int numRuns, int warmupNumPrompts, boolean continueOnError, boolean dryRun, boolean resume,
             boolean showStdout, int serverReadyTimeout) {
            this.serveCmd = serveCmd;
            this.benchCmd = benchCmd;
            this.serveParams = serveParams;
            this.benchParams = benchParams;
            this.linkVars = linkVars;
            this.outputDir = outputDir;
            this.experimentName = experimentName;
            this.numRuns = numRuns;
            this.warmupNumPrompts = warmupNumPrompts;
            this.continueOnError = continueOnError;
            this.dryRun = dryRun;
            this.resume = resume;
            this.showStdout = showStdout;
            this.serverReadyTimeout = serverReadyTimeout;
        }

        /** The experiment's directory, refused to overwrite and refused to resume when absent. */
        Path experimentDir() throws UsageException {
            Path dir = outputDir.resolve(experimentName);
            if (resume) {
                if (!Files.isDirectory(dir)) {
                    throw new UsageException("cannot resume from non-existent " + dir);
                }
                return dir;
            }
            if (Files.exists(dir)) {
                throw new UsageException("cannot overwrite existing " + dir);
            }
            return dir;
        }
    }

    private BenchSweepCommand() {
    }

    /** Runs the tool with the launcher's arguments after {@code bench-sweep}. */
    public static void run(String[] args) {
        try {
            Args parsed = parse(args);
            if (parsed == null) {
                return;
            }
            sweep(parsed, System.out, System.err);
        } catch (UsageException e) {
            System.err.println("Error: " + e.getMessage());
            System.err.print(USAGE);
            System.exit(2);
        } catch (IOException | IllegalStateException | IllegalArgumentException e) {
            System.err.println("Error: " + e.getMessage());
            System.exit(1);
        }
    }

    /** Parses the flags; null when {@code --help} printed the usage. */
    static Args parse(String[] args) throws UsageException, IOException {
        String serveCmd = null;
        String benchCmd = null;
        Path serveParams = null;
        Path benchParams = null;
        List<String[]> linkVars = new ArrayList<>();
        Path outputDir = Paths.get("results");
        String experimentName = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"));
        int numRuns = 3;
        int warmupNumPrompts = 0;
        boolean continueOnError = false;
        boolean dryRun = false;
        boolean resume = false;
        boolean showStdout = false;
        int serverReadyTimeout = 300;
        for (int i = 0; i < args.length; i++) {
            String a = args[i];
            if (a.equals("--help") || a.equals("-h")) {
                System.out.print(USAGE);
                return null;
            }
            try {
                switch (a) {
                    case "--serve-cmd":
                        serveCmd = next(args, i++);
                        break;
                    case "--bench-cmd":
                        benchCmd = next(args, i++);
                        break;
                    case "--serve-params":
                        serveParams = Paths.get(next(args, i++));
                        break;
                    case "--bench-params":
                        benchParams = Paths.get(next(args, i++));
                        break;
                    case "--link-vars":
                        linkVars = parseLinkVars(next(args, i++));
                        break;
                    case "-o":
                    case "--output-dir":
                        outputDir = Paths.get(next(args, i++));
                        break;
                    case "-e":
                    case "--experiment-name":
                        experimentName = next(args, i++);
                        break;
                    case "--num-runs":
                        numRuns = whole(next(args, i++), a, 1);
                        break;
                    case "--warmup-num-prompts":
                        warmupNumPrompts = whole(next(args, i++), a, 0);
                        break;
                    case "--continue-on-error":
                        continueOnError = true;
                        break;
                    case "--dry-run":
                        dryRun = true;
                        break;
                    case "--resume":
                        resume = true;
                        break;
                    case "--show-stdout":
                        showStdout = true;
                        break;
                    case "--server-ready-timeout":
                        serverReadyTimeout = whole(next(args, i++), a, 1);
                        break;
                    default:
                        throw new UsageException(a.startsWith("-")
                                ? "unknown option " + a + " (run with --help for the options)"
                                : "bench-sweep takes no positional argument, got '" + a + "'");
                }
            } catch (UsageException e) {
                throw new UsageException(e.getMessage());
            }
        }
        if (serveCmd == null) {
            throw new UsageException("--serve-cmd is required");
        }
        if (benchCmd == null) {
            throw new UsageException("--bench-cmd is required");
        }
        return new Args(ShellSplit.split(serveCmd), ShellSplit.split(benchCmd),
                serveParams != null ? ParameterSweep.readJson(serveParams) : ParameterSweep.none(),
                benchParams != null ? ParameterSweep.readJson(benchParams) : ParameterSweep.none(),
                linkVars, outputDir, experimentName, numRuns, warmupNumPrompts, continueOnError, dryRun,
                resume, showStdout, serverReadyTimeout);
    }

    /** The next argument, or the flag's own error. */
    private static String next(String[] args, int index) throws UsageException {
        if (index + 1 >= args.length) {
            throw new UsageException(args[index] + " needs a value");
        }
        return args[index + 1];
    }

    /** {@code --link-vars serve=bench,...} into its pairs. */
    private static List<String[]> parseLinkVars(String value) throws UsageException {
        List<String[]> pairs = new ArrayList<>();
        for (String item : value.split(",")) {
            String trimmed = item.trim();
            if (trimmed.isEmpty()) {
                continue;
            }
            int eq = trimmed.indexOf('=');
            if (eq <= 0 || eq == trimmed.length() - 1) {
                throw new UsageException("--link-vars needs serve=bench pairs, not '" + trimmed + "'");
            }
            pairs.add(new String[]{trimmed.substring(0, eq).trim(), trimmed.substring(eq + 1).trim()});
        }
        return pairs;
    }

    private static int whole(String v, String flag, int floor) throws UsageException {
        try {
            int n = Integer.parseInt(v.trim());
            if (n < floor) {
                throw new NumberFormatException();
            }
            return n;
        } catch (NumberFormatException e) {
            throw new UsageException(flag + " needs an integer of at least " + floor + ", not '" + v + "'");
        }
    }

    /**
     * The sweep: every serve combination against every bench combination, the runs and the summaries
     * of one combination under one directory.
     */
    static void sweep(Args args, Appendable out, Appendable err) throws IOException, UsageException {
        Path experimentDir = args.experimentDir();
        out.append("Experiment directory: ").append(experimentDir.toString()).append('\n');
        List<Map<String, Object>> allData = new ArrayList<>();
        for (ParameterSweep.Item serveComb : args.serveParams.items()) {
            try {
                runServeCombination(args, serveComb, experimentDir, allData, out, err);
            } catch (IOException | RuntimeException e) {
                if (!args.continueOnError) {
                    throw e instanceof IOException ? (IOException) e : new IOException(e.getMessage(), e);
                }
                err.append("[FAILED SERVER COMBINATION] ").append(String.valueOf(serveComb.name()))
                        .append(": ").append(String.valueOf(e.getMessage())).append('\n');
            }
        }
        if (args.dryRun) {
            out.append("The experiment will be saved at ").append(experimentDir.toString()).append('\n');
            return;
        }
        out.append("The experiment has been saved at ").append(experimentDir.toString()).append('\n');
        writeSummaryCsv(experimentDir.resolve("summary.csv"), allData);
        out.append("Summary over ").append(String.valueOf(allData.size()))
                .append(" runs: ").append(experimentDir.resolve("summary.csv").toString()).append('\n');
    }

    /** One serve combination: the server's lifecycle and every bench combination over it. */
    private static void runServeCombination(Args args, ParameterSweep.Item serveComb, Path experimentDir,
                                            List<Map<String, Object>> allData, Appendable out, Appendable err)
            throws IOException {
        if (!needsServer(args, serveComb, experimentDir)) {
            return;
        }
        List<String> serverCmd = serveComb.applyToCmd(args.serveCmd);
        if (args.dryRun) {
            out.append("[DRY RUN] server command: ").append(String.join(" ", serverCmd)).append('\n');
            runCombinations(args, serveComb, null, experimentDir, allData, out, err);
            return;
        }
        Process server = startServer(serverCmd, experimentDir.resolve("server.log"), args.showStdout, out);
        try {
            waitReady(server, healthUrl(serverCmd), args.serverReadyTimeout);
            out.append("[SERVER READY] ").append(healthUrl(serverCmd)).append('\n');
            runCombinations(args, serveComb, server, experimentDir, allData, out, err);
        } finally {
            server.destroy();
            try {
                server.waitFor();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new IOException("the sweep was interrupted while stopping the server", e);
            }
        }
    }

    /** Whether any bench combination of this serve combination still lacks its summary. */
    private static boolean needsServer(Args args, ParameterSweep.Item serveComb, Path experimentDir) {
        if (!args.resume && args.dryRun) {
            return true;
        }
        for (ParameterSweep.Item benchComb : args.benchParams.items()) {
            if (!isValidCombination(serveComb, benchComb, args.linkVars)) {
                continue;
            }
            if (!Files.exists(combinationDir(experimentDir, serveComb, benchComb).resolve("summary.json"))) {
                return true;
            }
        }
        return false;
    }

    /** Every bench combination of one live server. */
    private static void runCombinations(Args args, ParameterSweep.Item serveComb, Process server,
                                        Path experimentDir, List<Map<String, Object>> allData,
                                        Appendable out, Appendable err) throws IOException {
        for (ParameterSweep.Item benchComb : args.benchParams.items()) {
            if (!isValidCombination(serveComb, benchComb, args.linkVars)) {
                continue;
            }
            Path dir = combinationDir(experimentDir, serveComb, benchComb);
            runCombination(args, serveComb, benchComb, server, dir, allData, out, err);
        }
    }

    /** Whether every linked variable is equal in both records. */
    private static boolean isValidCombination(ParameterSweep.Item serveComb, ParameterSweep.Item benchComb,
                                              List<String[]> linkVars) {
        for (String[] link : linkVars) {
            Object serveValue = serveComb.valueOf(link[0]);
            Object benchValue = benchComb.valueOf(link[1]);
            if (serveValue == null || benchValue == null || !serveValue.equals(benchValue)) {
                return false;
            }
        }
        return true;
    }

    /** One combination's runs: the optional warmup, then {@code --num-runs} measured runs. */
    private static void runCombination(Args args, ParameterSweep.Item serveComb, ParameterSweep.Item benchComb,
                                       Process server, Path dir, List<Map<String, Object>> allData,
                                       Appendable out, Appendable err) throws IOException {
        if (args.warmupNumPrompts > 0) {
            Map<String, Object> warmupParams = new LinkedHashMap<>(benchComb.paramsView());
            warmupParams.remove("_benchmark_name");
            warmupParams.put("num-prompts", args.warmupNumPrompts);
            ParameterSweep.Item warmupComb = singleRecord(warmupParams);
            runOne(args, serveComb, warmupComb, server, dir.resolve("warmup.json"), -1, out, err);
        }
        List<Map<String, Object>> combData = new ArrayList<>();
        for (int runNumber = 0; runNumber < args.numRuns; runNumber++) {
            Path file = dir.resolve("run=" + runNumber + ".json");
            Map<String, Object> runData = runOne(args, serveComb, benchComb, server, file, runNumber, out, err);
            if (runData != null) {
                combData.add(runData);
                allData.add(runData);
            }
        }
        if (args.dryRun || combData.isEmpty()) {
            return;
        }
        Files.write(dir.resolve("summary.json"), jsonLines(combData).getBytes(StandardCharsets.UTF_8));
    }

    /** One benchmark run, or its read-back when {@code --resume} finds the file; null in a dry run. */
    private static Map<String, Object> runOne(Args args, ParameterSweep.Item serveComb,
                                              ParameterSweep.Item benchComb, Process server, Path file,
                                              int runNumber, Appendable out, Appendable err) throws IOException {
        List<String> benchArgs = benchComb.applyToCmd(args.benchCmd);
        benchArgs.addAll(List.of("--percentile-metrics", "ttft,tpot,itl,e2el", "--save-result",
                "--result-dir", file.getParent().toString(), "--result-filename", file.getFileName().toString()));
        out.append("[BEGIN BENCHMARK] run=").append(String.valueOf(runNumber))
                .append(" bench parameters: ").append(benchComb.name()).append('\n');
        out.append("[BENCHMARK COMMAND] ").append(String.join(" ", benchArgs)).append('\n');
        out.append("[OUTPUT FILE] ").append(file.toString()).append('\n');
        if (args.dryRun) {
            return null;
        }
        if (Files.exists(file)) {
            out.append("Found existing results.\n");
            return annotate(readJson(file), runNumber, serveComb, benchComb, file);
        }
        if (server == null) {
            throw new IOException("cannot find results at " + file);
        }
        Files.createDirectories(file.getParent());
        try {
            ServeBenchOptions opts = ServeBenchCommand.parse(benchArgs.toArray(new String[0]));
            if (opts == null) {
                throw new IOException("--bench-cmd asked for --help inside a sweep");
            }
            Map<String, Object> result = ServeBenchCommand.execute(opts, out, err);
            Map<String, Object> annotated = annotate(result, runNumber, serveComb, benchComb, file);
            Files.write(file, indentJson(annotated).getBytes(StandardCharsets.UTF_8));
            return annotated;
        } catch (ServeBenchCommand.UsageException | IOException | IllegalStateException
                 | IllegalArgumentException e) {
            if (!args.continueOnError) {
                throw e instanceof IOException ? (IOException) e : new IOException(e.getMessage(), e);
            }
            Path failure = failureFile(file);
            Map<String, Object> record = new LinkedHashMap<>();
            record.put("run_number", runNumber);
            record.put("error_type", e.getClass().getSimpleName());
            record.put("error", String.valueOf(e.getMessage()));
            record.put("serve_overrides", recordParams(serveComb));
            record.put("bench_overrides", recordParams(benchComb));
            Files.write(failure, indentJson(record).getBytes(StandardCharsets.UTF_8));
            err.append("[FAILED BENCHMARK] run=").append(String.valueOf(runNumber)).append(": ")
                    .append(e.getMessage()).append('\n');
            return null;
        }
    }

    /** The run's annotations: the number and both records' parameters, the reference's merge. */
    private static Map<String, Object> annotate(Map<String, Object> runData, int runNumber,
                                                ParameterSweep.Item serveComb, ParameterSweep.Item benchComb,
                                                Path file) throws IOException {
        Map<String, Object> out;
        try {
            out = new LinkedHashMap<>(runData);
        } catch (NullPointerException e) {
            throw new IOException("the result file " + file + " carries no JSON object");
        }
        out.put("run_number", runNumber);
        out.putAll(recordParams(serveComb));
        out.putAll(recordParams(benchComb));
        return out;
    }

    /** The record's parameters without the name key. */
    private static Map<String, Object> recordParams(ParameterSweep.Item comb) {
        Map<String, Object> out = new LinkedHashMap<>();
        for (Map.Entry<String, Object> e : comb.paramsView().entrySet()) {
            if (!"_benchmark_name".equals(e.getKey())) {
                out.put(e.getKey(), e.getValue());
            }
        }
        return out;
    }

    /** A record carrying exactly {@code params}, for the warmup's derived combination. */
    private static ParameterSweep.Item singleRecord(Map<String, Object> params) {
        return new ParameterSweep.Item(params);
    }

    /** The failure file beside a failed run. */
    private static Path failureFile(Path file) {
        String name = file.getFileName().toString();
        int dot = name.lastIndexOf('.');
        String stem = dot > 0 ? name.substring(0, dot) : name;
        return file.getParent().resolve(stem + ".failure.json");
    }

    /** The combination's directory: the {@code SERVE-} and {@code BENCH-} names that are set. */
    static Path combinationDir(Path experimentDir, ParameterSweep.Item serveComb,
                               ParameterSweep.Item benchComb) {
        StringJoiner name = new StringJoiner("-");
        String serveName = nonEmptyName(serveComb);
        String benchName = nonEmptyName(benchComb);
        if (serveName != null) {
            name.add("SERVE-" + sanitize(serveName));
        }
        if (benchName != null) {
            name.add("BENCH-" + sanitize(benchName));
        }
        return experimentDir.resolve(name.length() > 0 ? name.toString() : "default");
    }

    /** The record's name, null when the record sets no parameter. */
    private static String nonEmptyName(ParameterSweep.Item comb) {
        String name = comb.name();
        return name.isEmpty() ? null : name;
    }

    /** The file-name-safe form of a name: path separators and control runs to dashes. */
    private static String sanitize(String name) {
        return name.replaceAll("[/\\\\\0]", "-");
    }

    /** The server process: its output inherited, or on {@code logFile}. */
    static Process startServer(List<String> serverCmd, Path logFile, boolean showStdout, Appendable out)
            throws IOException {
        out.append("[BEGIN SERVER] command: ").append(String.join(" ", serverCmd)).append('\n');
        ProcessBuilder builder = new ProcessBuilder(serverCmd);
        if (showStdout) {
            builder.inheritIO();
        } else {
            Path parent = logFile.getParent();
            if (parent != null) {
                Files.createDirectories(parent);
            }
            builder.redirectOutput(ProcessBuilder.Redirect.appendTo(logFile.toFile()))
                    .redirectErrorStream(true);
        }
        return builder.start();
    }

    /** The health URL of a server command: its {@code --host}/{@code --port}, or the defaults. */
    static String healthUrl(List<String> serverCmd) {
        String host = "127.0.0.1";
        int port = 8080;
        for (int i = 0; i < serverCmd.size() - 1; i++) {
            if (serverCmd.get(i).equals("--host")) {
                host = serverCmd.get(i + 1);
            } else if (serverCmd.get(i).equals("--port")) {
                try {
                    port = Integer.parseInt(serverCmd.get(i + 1).trim());
                } catch (NumberFormatException ignored) {
                    // the server itself refuses the value; the poll fails and names it
                }
            }
        }
        return "http://" + host + ":" + port + "/health";
    }

    /** The poll of {@code /health} until it answers 200 or the timeout runs out. */
    static void waitReady(Process server, String healthUrl, int timeoutSec) throws IOException {
        HttpClient client = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(5)).build();
        HttpRequest request = HttpRequest.newBuilder(URI.create(healthUrl))
                .timeout(Duration.ofSeconds(10)).GET().build();
        long deadline = System.currentTimeMillis() + timeoutSec * 1000L;
        while (System.currentTimeMillis() < deadline) {
            if (!server.isAlive()) {
                throw new IOException("the server exited (code " + server.exitValue() + ") before "
                        + healthUrl + " answered");
            }
            try {
                HttpResponse<Void> response = client.send(request, HttpResponse.BodyHandlers.discarding());
                if (response.statusCode() == 200) {
                    return;
                }
            } catch (IOException | InterruptedException e) {
                if (e instanceof InterruptedException) {
                    Thread.currentThread().interrupt();
                    throw new IOException("the server ready poll was interrupted", e);
                }
                // not listening yet: the next poll decides
            }
            try {
                Thread.sleep(100L);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new IOException("the server ready poll was interrupted", e);
            }
        }
        throw new IOException("the server did not answer " + healthUrl + " within " + timeoutSec + " s");
    }

    /** The run file's JSON object. */
    private static Map<String, Object> readJson(Path file) throws IOException {
        try {
            Object parsed = Json.parse(Files.readString(file), Json.Limits.UNTRUSTED);
            if (!(parsed instanceof Map)) {
                throw new IOException("the result file " + file + " carries no JSON object");
            }
            return (Map<String, Object>) parsed;
        } catch (JsonException e) {
            throw new IOException("the result file " + file + " is not JSON: " + e.getMessage());
        }
    }

    /** The run list as one JSON array, Python's indent of four. */
    private static String jsonLines(List<Map<String, Object>> runs) {
        StringBuilder out = new StringBuilder();
        try {
            Json.write(runs, INDENTED, out);
        } catch (IOException e) {
            throw new IllegalStateException("a sweep summary does not write as JSON", e);
        }
        return out.toString();
    }

    /** One record as JSON, Python's indent of four. */
    private static String indentJson(Map<String, Object> record) {
        StringBuilder out = new StringBuilder();
        try {
            Json.write(record, INDENTED, out);
        } catch (IOException e) {
            throw new IllegalStateException("a sweep run record does not write as JSON", e);
        }
        return out.toString();
    }

    /** Python's {@code json.dumps(indent=4)} layout. */
    private static final Json.Style INDENTED = new Json.Style(false, "    ", ", ", ": ", false);

    /**
     * The summary CSV: every run a row, every key a column, values quoted per RFC 4180 and
     * non-scalars as JSON — pandas' {@code to_csv} for these records.
     */
    static void writeSummaryCsv(Path file, List<Map<String, Object>> runs) throws IOException {
        List<String> columns = new ArrayList<>();
        for (Map<String, Object> run : runs) {
            for (String key : run.keySet()) {
                if (!columns.contains(key)) {
                    columns.add(key);
                }
            }
        }
        StringBuilder csv = new StringBuilder();
        Path parent = file.getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
        for (int c = 0; c < columns.size(); c++) {
            if (c > 0) {
                csv.append(',');
            }
            csv.append(quote(columns.get(c)));
        }
        csv.append('\n');
        for (Map<String, Object> run : runs) {
            for (int c = 0; c < columns.size(); c++) {
                if (c > 0) {
                    csv.append(',');
                }
                csv.append(quote(cell(run.get(columns.get(c)))));
            }
            csv.append('\n');
        }
        Files.write(file, csv.toString().getBytes(StandardCharsets.UTF_8));
    }

    /** One CSV cell: empty for null, JSON for composites, the string of anything else. */
    private static String cell(Object value) {
        if (value == null) {
            return "";
        }
        if (value instanceof Map || value instanceof List) {
            return ServeBenchRunner.jsonText(value);
        }
        return String.valueOf(value);
    }

    /** One RFC 4180 field: quoted when it carries a comma, a quote or a newline. */
    private static String quote(String field) {
        if (field.contains(",") || field.contains("\"") || field.contains("\n")) {
            return "\"" + field.replace("\"", "\"\"") + "\"";
        }
        return field;
    }
}

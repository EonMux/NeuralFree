package com.eonmux.neuralfree.demo.tools.serve;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

import com.eonmux.neuralfree.utils.json.Json;

/**
 * {@code nftool bench-startup}: the startup benchmark of vLLM's {@code vllm bench startup}, adapted
 * to the server process: every iteration launches {@code --serve-cmd} and times the wait until its
 * {@code /health} answers 200, then stops the process. The command runs {@code --num-iters-cold}
 * iterations first, then {@code --num-iters-warmup} unmeasured ones, then {@code --num-iters-warm}
 * measured ones. The engine's compilation metrics have no server-process equivalent: this harness
 * reports the startup time of each phase and nothing it cannot measure. Cold and warm differ in the
 * operating system's file cache (the first launches read the model from disk), the only cache a
 * process launch exposes.
 */
public final class BenchStartupCommand {

    /** The command surface: {@code --help} and every error print it. */
    static final String USAGE =
            "nftool bench-startup [options]\n"
            + "options:\n"
            + "  -h, --help                     this text\n"
            + "  --serve-cmd <command>          the server command line, quoted (required)\n"
            + "  --num-iters-cold <n>           cold startup iterations, default 3\n"
            + "  --num-iters-warmup <n>         unmeasured iterations before the warm phase, default 1\n"
            + "  --num-iters-warm <n>           warm startup iterations, default 3\n"
            + "  --server-ready-timeout <sec>   how long /health may take per iteration, default 300\n"
            + "  --health-url <url>             the ready URL instead of the command's --host/--port\n"
            + "  --log-file <path>              the server's log; default a temporary file\n"
            + "  --show-stdout                  the server's output on this console, not the log file\n"
            + "  --output-json <file>           save the startup times and percentiles\n"
            + "Every iteration starts the command afresh and stops it once /health answered; the\n"
            + "compilation metrics of the reference need in-process engine timings and are not reported.\n";

    /** The percentiles of the report, the reference's. */
    static final List<Integer> PERCENTAGES = List.of(10, 25, 50, 75, 90, 99);

    /** An invalid command line; the message names the flag. */
    static final class UsageException extends Exception {

        private static final long serialVersionUID = 1L;

        UsageException(String message) {
            super(message);
        }
    }

    /** The parsed invocation. */
    static final class Args {

        final List<String> serveCmd;
        final int numItersCold;
        final int numItersWarmup;
        final int numItersWarm;
        final int serverReadyTimeout;
        final String healthUrl;
        final Path logFile;
        final boolean showStdout;
        final String outputJson;

        Args(List<String> serveCmd, int numItersCold, int numItersWarmup, int numItersWarm,
             int serverReadyTimeout, String healthUrl, Path logFile, boolean showStdout, String outputJson) {
            this.serveCmd = serveCmd;
            this.numItersCold = numItersCold;
            this.numItersWarmup = numItersWarmup;
            this.numItersWarm = numItersWarm;
            this.serverReadyTimeout = serverReadyTimeout;
            this.healthUrl = healthUrl;
            this.logFile = logFile;
            this.showStdout = showStdout;
            this.outputJson = outputJson;
        }
    }

    private BenchStartupCommand() {
    }

    /** Runs the tool with the launcher's arguments after {@code bench-startup}. */
    public static void run(String[] args) {
        Path temporaryLog = null;
        try {
            Args parsed = parse(args);
            if (parsed == null) {
                return;
            }
            temporaryLog = parsed.logFile == null ? Files.createTempFile("nf-bench-startup-", ".log") : null;
            if (temporaryLog != null) {
                temporaryLog.toFile().deleteOnExit();
            }
            bench(parsed, temporaryLog, System.out);
        } catch (UsageException e) {
            System.err.println("Error: " + e.getMessage());
            System.err.print(USAGE);
            System.exit(2);
        } catch (IOException | IllegalStateException | IllegalArgumentException e) {
            System.err.println("Error: " + e.getMessage());
            System.exit(1);
        } finally {
            if (temporaryLog != null) {
                try {
                    Files.deleteIfExists(temporaryLog);
                } catch (IOException ignored) {
                    // the temporary log is left behind when the platform refuses its deletion
                }
            }
        }
    }

    /** Parses the flags; null when {@code --help} printed the usage. */
    static Args parse(String[] args) throws UsageException {
        String serveCmd = null;
        int numItersCold = 3;
        int numItersWarmup = 1;
        int numItersWarm = 3;
        int serverReadyTimeout = 300;
        String healthUrl = null;
        Path logFile = null;
        boolean showStdout = false;
        String outputJson = null;
        for (int i = 0; i < args.length; i++) {
            String a = args[i];
            if (a.equals("--help") || a.equals("-h")) {
                System.out.print(USAGE);
                return null;
            }
            switch (a) {
                case "--serve-cmd":
                    serveCmd = next(a, args, i++);
                    break;
                case "--num-iters-cold":
                    numItersCold = whole(next(a, args, i++), a, 0);
                    break;
                case "--num-iters-warmup":
                    numItersWarmup = whole(next(a, args, i++), a, 0);
                    break;
                case "--num-iters-warm":
                    numItersWarm = whole(next(a, args, i++), a, 0);
                    break;
                case "--server-ready-timeout":
                    serverReadyTimeout = whole(next(a, args, i++), a, 1);
                    break;
                case "--health-url":
                    healthUrl = next(a, args, i++);
                    break;
                case "--log-file":
                    logFile = Paths.get(next(a, args, i++));
                    break;
                case "--show-stdout":
                    showStdout = true;
                    break;
                case "--output-json":
                    outputJson = next(a, args, i++);
                    break;
                default:
                    throw new UsageException(a.startsWith("-")
                            ? "unknown option " + a + " (run with --help for the options)"
                            : "bench-startup takes no positional argument, got '" + a + "'");
            }
        }
        if (serveCmd == null) {
            throw new UsageException("--serve-cmd is required");
        }
        List<String> cmd = ShellSplit.split(serveCmd);
        if (numItersCold + numItersWarm == 0) {
            throw new UsageException("--num-iters-cold and --num-iters-warm cannot both be 0");
        }
        String url = healthUrl != null ? healthUrl : BenchSweepCommand.healthUrl(cmd);
        return new Args(cmd, numItersCold, numItersWarmup, numItersWarm, serverReadyTimeout, url,
                logFile, showStdout, outputJson);
    }

    private static String next(String flag, String[] args, int index) throws UsageException {
        if (index + 1 >= args.length) {
            throw new UsageException(flag + " needs a value");
        }
        return args[index + 1];
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
     * The benchmark: the cold phase, the warmup, the warm phase, the report and the JSON.
     *
     * @param temporaryLog the log file when the invocation named none; null otherwise
     * @return the result JSON (the times and percentiles of both phases)
     */
    static Map<String, Object> bench(Args args, Path temporaryLog, Appendable out) throws IOException {
        Path logFile = args.logFile != null ? args.logFile : temporaryLog;
        if (logFile == null && !args.showStdout) {
            throw new IllegalArgumentException("a startup benchmark needs a log file or --show-stdout");
        }
        List<Double> cold = new ArrayList<>();
        out.append("Measuring cold startup time...\n");
        for (int i = 0; i < args.numItersCold; i++) {
            cold.add(startupSeconds(args, logFile));
        }
        if (args.numItersWarmup > 0) {
            out.append("Warming up for warm startup measurement...\n");
            for (int i = 0; i < args.numItersWarmup; i++) {
                startupSeconds(args, logFile);
            }
        }
        List<Double> warm = new ArrayList<>();
        out.append("Measuring warm startup time...\n");
        for (int i = 0; i < args.numItersWarm; i++) {
            warm.add(startupSeconds(args, logFile));
        }
        Map<String, Object> result = new LinkedHashMap<>();
        out.append("\n").append("=".repeat(60)).append("\n");
        out.append("STARTUP TIME BENCHMARK RESULTS\n");
        out.append("=".repeat(60)).append("\n");
        if (!cold.isEmpty()) {
            phase("COLD STARTUP", "cold", cold, result, out);
        }
        if (!warm.isEmpty()) {
            phase("WARM STARTUP", "warm", warm, result, out);
        }
        out.append("=".repeat(60)).append("\n");
        if (args.outputJson != null) {
            Path file = Paths.get(args.outputJson);
            Path parent = file.getParent();
            if (parent != null) {
                Files.createDirectories(parent);
            }
            Files.write(file, Json.toString(result, INDENTED).getBytes(StandardCharsets.UTF_8));
            out.append("Startup result saved to ").append(file.toString()).append('\n');
        }
        return result;
    }

    /** One iteration: the process up, the time until {@code /health} answers, the process down. */
    private static double startupSeconds(Args args, Path logFile) throws IOException {
        long begin = System.nanoTime();
        Process server = BenchSweepCommand.startServer(args.serveCmd, logFile, args.showStdout, new StringBuilder());
        try {
            BenchSweepCommand.waitReady(server, args.healthUrl, args.serverReadyTimeout);
            return (System.nanoTime() - begin) / 1e9;
        } finally {
            server.destroy();
            try {
                server.waitFor();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new IOException("the startup benchmark was interrupted while stopping the server", e);
            }
        }
    }

    /** One phase's report and JSON keys, the reference's layout and names. */
    private static void phase(String title, String key, List<Double> times, Map<String, Object> result,
                             Appendable out) throws IOException {
        out.append(title).append(":\n");
        double avg = BenchStats.mean(times);
        out.append(String.format(Locale.ROOT, "Avg startup time: %.2f seconds%n", avg));
        out.append("Startup time percentiles:\n");
        Map<String, Object> percentiles = new LinkedHashMap<>();
        for (int p : PERCENTAGES) {
            double value = BenchStats.percentile(times, p);
            percentiles.put(String.valueOf(p), value);
            out.append(String.format(Locale.ROOT, "  %d%%: %.2f seconds%n", p, value));
        }
        result.put("avg_" + key + "_startup_time", avg);
        result.put(key + "_startup_times", new ArrayList<>(times));
        result.put(key + "_startup_percentiles", percentiles);
    }

    /** Python's {@code json.dumps(indent=4)} layout. */
    private static final Json.Style INDENTED = new Json.Style(false, "    ", ", ", ": ", false);
}

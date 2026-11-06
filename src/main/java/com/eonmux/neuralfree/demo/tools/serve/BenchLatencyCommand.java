package com.eonmux.neuralfree.demo.tools.serve;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;

import com.eonmux.neuralfree.utils.json.Json;
import com.eonmux.neuralfree.utils.json.JsonException;

/**
 * {@code nftool bench-latency}: the single-batch latency benchmark of vLLM's {@code vllm bench
 * latency}, run against a serving endpoint instead of the in-process engine the reference
 * benchmarks: every iteration sends one batch of {@code --batch-size} synthetic token-id prompts of
 * {@code --input-len}, asking {@code --output-len} tokens each (temperature 1.0, top-p 1.0,
 * ignore-eos, as the reference fixes), and the iteration's latency is the wall time until the whole
 * batch closed. {@code --num-iters-warmup} iterations run first, the report gives the average and
 * the 10th to 99th percentiles, and {@code --output-json} saves the latencies and percentiles. The
 * reference's {@code --use-beam-search} (no beam search) and {@code --disable-detokenize} (the
 * server always detokenizes) refuse by name.
 */
public final class BenchLatencyCommand {

    /** The command surface: {@code --help} and every error print it. */
    static final String USAGE =
            "nftool bench-latency [options]\n"
            + "options:\n"
            + "  -h, --help                    this text\n"
            + "  --host <h> / --port <p> / --base-url <url> / --endpoint <path>\n"
            + "  --model <id> / --header KEY=VALUE / --insecure\n"
            + "  --input-len <n>               prompt tokens per request, default 32\n"
            + "  --output-len <n>              generated tokens per request, default 128\n"
            + "  --batch-size <n>              requests of one iteration, default 8\n"
            + "  --n <n>                       generated sequences per request, default 1\n"
            + "  --num-iters-warmup <n>        warmup iterations, default 10\n"
            + "  --num-iters <n>               measured iterations, default 30\n"
            + "  --output-json <file>          save the latencies and percentiles\n"
            + "Not supported (refused by name): --use-beam-search, --disable-detokenize. The requests are\n"
            + "random token-id prompts (uniform over the served vocabulary, seeded 0) at temperature 1.0,\n"
            + "top-p 1.0, ignore_eos, as the reference fixes.\n";

    /** The percentiles of the report, the reference's. */
    static final List<Integer> PERCENTAGES = List.of(10, 25, 50, 75, 90, 99);

    /** An invalid command line or an unsupported option; the message names the flag. */
    static final class UsageException extends Exception {

        private static final long serialVersionUID = 1L;

        UsageException(String message) {
            super(message);
        }
    }

    private BenchLatencyCommand() {
    }

    /** Runs the tool with the launcher's arguments after {@code bench-latency}. */
    public static void run(String[] args) {
        try {
            Args parsed = parse(args);
            if (parsed == null) {
                return;
            }
            bench(parsed, System.out);
        } catch (UsageException e) {
            System.err.println("Error: " + e.getMessage());
            System.err.print(USAGE);
            System.exit(2);
        } catch (IOException | IllegalStateException | IllegalArgumentException e) {
            System.err.println("Error: " + e.getMessage());
            System.exit(1);
        }
    }

    /** The parsed invocation: the server address, the shape and the iteration counts. */
    static final class Args {

        final String baseUrl;
        final String endpoint;
        final String model;
        final Map<String, String> headers;
        final boolean insecure;
        final int inputLen;
        final int outputLen;
        final int batchSize;
        final int n;
        final int numItersWarmup;
        final int numIters;
        final String outputJson;

        Args(String baseUrl, String endpoint, String model, Map<String, String> headers, boolean insecure,
             int inputLen, int outputLen, int batchSize, int n, int numItersWarmup, int numIters,
             String outputJson) {
            this.baseUrl = baseUrl;
            this.endpoint = endpoint;
            this.model = model;
            this.headers = headers;
            this.insecure = insecure;
            this.inputLen = inputLen;
            this.outputLen = outputLen;
            this.batchSize = batchSize;
            this.n = n;
            this.numItersWarmup = numItersWarmup;
            this.numIters = numIters;
            this.outputJson = outputJson;
        }
    }

    /** Parses the flags; null when {@code --help} printed the usage. */
    static Args parse(String[] args) throws UsageException {
        String host = "127.0.0.1";
        int port = 8000;
        String baseUrl = null;
        String endpoint = "/v1/completions";
        String model = null;
        Map<String, String> headers = new LinkedHashMap<>();
        boolean insecure = false;
        int inputLen = 32;
        int outputLen = 128;
        int batchSize = 8;
        int n = 1;
        int numItersWarmup = 10;
        int numIters = 30;
        String outputJson = null;
        for (int i = 0; i < args.length; i++) {
            String a = args[i];
            if (a.equals("--help") || a.equals("-h")) {
                System.out.print(USAGE);
                return null;
            }
            if (a.startsWith("--")) {
                a = a.replace('_', '-');
            }
            switch (a) {
                case "--host":
                    host = next(a, args, i++);
                    break;
                case "--port":
                    port = whole(next(a, args, i++), a, 1, 65535);
                    break;
                case "--base-url":
                    baseUrl = next(a, args, i++);
                    break;
                case "--endpoint":
                    endpoint = next(a, args, i++);
                    break;
                case "--model":
                    model = next(a, args, i++);
                    break;
                case "--header":
                    int eq = next(a, args, i).indexOf('=');
                    if (eq <= 0) {
                        throw new UsageException("--header needs the KEY=VALUE format, not '"
                                + next(a, args, i) + "'");
                    }
                    String pair = next(a, args, i);
                    headers.put(pair.substring(0, eq).trim(), pair.substring(eq + 1).trim());
                    i++;
                    break;
                case "--insecure":
                    insecure = true;
                    break;
                case "--input-len":
                    inputLen = whole(next(a, args, i++), a, 1, Integer.MAX_VALUE);
                    break;
                case "--output-len":
                    outputLen = whole(next(a, args, i++), a, 1, Integer.MAX_VALUE);
                    break;
                case "--batch-size":
                    batchSize = whole(next(a, args, i++), a, 1, Integer.MAX_VALUE);
                    break;
                case "--n":
                    n = whole(next(a, args, i++), a, 1, Integer.MAX_VALUE);
                    break;
                case "--num-iters-warmup":
                    numItersWarmup = whole(next(a, args, i++), a, 0, Integer.MAX_VALUE);
                    break;
                case "--num-iters":
                    numIters = whole(next(a, args, i++), a, 1, Integer.MAX_VALUE);
                    break;
                case "--output-json":
                    outputJson = next(a, args, i++);
                    break;
                case "--use-beam-search":
                    throw new UsageException(a + " is not supported: the engine has no beam search");
                case "--disable-detokenize":
                    throw new UsageException(a + " is not supported: the server always detokenizes the"
                            + " stream, so detokenization rides in the latency either way");
                default:
                    throw new UsageException(a.startsWith("-")
                            ? "unknown option " + a + " (run with --help for the options)"
                            : "bench-latency takes no positional argument, got '" + a + "'");
            }
        }
        return new Args(baseUrl != null ? baseUrl : "http://" + host + ":" + port, endpoint, model, headers,
                insecure, inputLen, outputLen, batchSize, n, numItersWarmup, numIters, outputJson);
    }

    private static String next(String flag, String[] args, int index) throws UsageException {
        if (index + 1 >= args.length) {
            throw new UsageException(flag + " needs a value");
        }
        return args[index + 1];
    }

    private static int whole(String v, String flag, int floor, int ceiling) throws UsageException {
        try {
            int n = Integer.parseInt(v.trim());
            if (n < floor || n > ceiling) {
                throw new NumberFormatException();
            }
            return n;
        } catch (NumberFormatException e) {
            throw new UsageException(flag + " needs an integer in [" + floor + ", " + ceiling + "], not '"
                    + v + "'");
        }
    }

    /**
     * The benchmark: the discovery (the model, the vocabulary), the warmup iterations, the measured
     * ones, the report.
     *
     * @return the result JSON (the model, the average, the latencies, the percentiles)
     */
    static Map<String, Object> bench(Args args, Appendable out) throws IOException {
        String modelId = args.model;
        if (modelId == null) {
            modelId = BenchEndpoints.firstModel(args.baseUrl, args.headers, args.insecure).id;
            out.append("First model id: ").append(modelId).append('\n');
        }
        int vocabSize = BenchEndpoints.tokenizerInfo(args.baseUrl, args.headers, args.insecure).vocabSize;
        ServeBenchClient client = new ServeBenchClient(args.baseUrl, args.endpoint, args.headers, modelId,
                null, true, samplingParams(args.n), null, args.insecure);
        Random random = new Random(0);
        out.append("Warming up...\n");
        for (int i = 0; i < args.numItersWarmup; i++) {
            runBatch(args, client, modelId, vocabSize, random);
        }
        out.append("Benchmarking ").append(String.valueOf(args.numIters)).append(" iterations...\n");
        List<Double> latencies = new ArrayList<>(args.numIters);
        for (int i = 0; i < args.numIters; i++) {
            latencies.add(runBatch(args, client, modelId, vocabSize, random));
        }
        double avg = BenchStats.mean(latencies);
        out.append(String.format(Locale.ROOT, "Avg latency: %.4f seconds%n", avg));
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("model_id", modelId);
        result.put("avg_latency", avg);
        result.put("latencies", latencies);
        Map<String, Object> percentiles = new LinkedHashMap<>();
        for (int p : PERCENTAGES) {
            double value = BenchStats.percentile(latencies, p);
            percentiles.put(String.valueOf(p), value);
            out.append(String.format(Locale.ROOT, "%d%% percentile latency: %.4f seconds%n", p, value));
        }
        result.put("percentiles", percentiles);
        if (args.outputJson != null) {
            StringBuilder json = new StringBuilder();
            Json.write(result, INDENTED, json);
            java.nio.file.Path file = java.nio.file.Paths.get(args.outputJson);
            java.nio.file.Path parent = file.getParent();
            if (parent != null && !java.nio.file.Files.exists(parent)) {
                java.nio.file.Files.createDirectories(parent);
            }
            java.nio.file.Files.write(file, json.toString().getBytes(java.nio.charset.StandardCharsets.UTF_8));
            out.append("Latency result saved to ").append(file.toString()).append('\n');
        }
        return result;
    }

    /** One iteration: a batch of prompts out together, the wall time until all closed. */
    private static double runBatch(Args args, ServeBenchClient client, String modelId, int vocabSize,
                                   Random random) throws IOException {
        List<BenchSample> batch = new ArrayList<>(args.batchSize);
        for (int b = 0; b < args.batchSize; b++) {
            int[] tokens = new int[args.inputLen];
            for (int t = 0; t < tokens.length; t++) {
                tokens[t] = random.nextInt(vocabSize);
            }
            batch.add(BenchSample.ofTokens(tokens, args.outputLen, "bench-latency-" + b));
        }
        ArrivalSchedule allAtOnce = ArrivalSchedule.of(batch.size(), Double.POSITIVE_INFINITY, 1.0,
                null, null, null, random);
        long begin = System.nanoTime();
        List<BenchRequestOutput> outputs = client.runAll(batch, allAtOnce, null, null);
        double seconds = (System.nanoTime() - begin) / 1e9;
        int failed = 0;
        String firstError = null;
        for (BenchRequestOutput output : outputs) {
            if (!output.success) {
                failed++;
                if (firstError == null) {
                    firstError = output.error;
                }
            }
        }
        if (failed == outputs.size()) {
            throw new IOException("every request of the iteration failed: " + firstError);
        }
        if (failed > 0) {
            throw new IOException(failed + " of " + outputs.size() + " requests of the iteration failed: "
                    + firstError);
        }
        return seconds;
    }

    /** The reference's fixed sampling: temperature 1.0, top-p 1.0, ignore_eos; {@code --n} sequences. */
    private static Map<String, Object> samplingParams(int n) {
        Map<String, Object> params = new LinkedHashMap<>();
        if (n > 1) {
            params.put("n", n);
        }
        params.put("temperature", 1.0);
        params.put("top_p", 1.0);
        return params;
    }

    /** Python's {@code json.dumps(indent=4)} layout. */
    private static final Json.Style INDENTED = new Json.Style(false, "    ", ", ", ": ", false);
}

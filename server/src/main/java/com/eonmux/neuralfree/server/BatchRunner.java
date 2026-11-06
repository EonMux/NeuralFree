package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.SecretProperties;
import com.eonmux.neuralfree.gguf.chat.tools.ToolCallStream;
import com.eonmux.neuralfree.gguf.chat.tools.ToolOutput;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.utils.json.Json;
import com.eonmux.neuralfree.utils.json.JsonException;
import com.eonmux.neuralfree.utils.json.JsonView;
import jakarta.ws.rs.WebApplicationException;
import jakarta.ws.rs.core.Response;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;
import java.util.function.Supplier;

/**
 * The offline batch runner (vLLM's {@code run_batch}, OpenAI's Batch API): it reads an input JSONL file whose
 * records are {@code {"custom_id", "method", "url", "body"}}, runs every body through the same mapping the HTTP
 * endpoints use ({@link ChatPlan}, {@link CompletionPlan}, {@link PoolingResource}), and writes an output JSONL
 * file whose records are {@code {"id", "custom_id", "response", "error"}} in input order.
 *
 * <p>The chat and completions records of one file run together: their jobs join shared {@link GenerationBatch}
 * waves of at most {@code maxInflight} jobs, so the engine decodes the batch continuously. Embeddings, rerank
 * and score records run one by one between the waves (a pooling forward needs the idle engine). A malformed
 * envelope (a record that is no JSON object, or without {@code custom_id}, {@code method}, {@code url} or a
 * {@code body} object) aborts the run before the model loads, naming the line. A record that fails while running
 * — an unsupported URL, a stream request, a refused body — writes an error record and the batch goes on.</p>
 *
 * <p>Supported URLs: {@code /v1/chat/completions}, {@code /v1/completions} and {@code /v1/embeddings} match
 * exactly; {@code /score} and {@code /rerank} match by suffix (vLLM's unversioned endpoints, so their
 * {@code /v1/} aliases route too). Audio transcriptions and translations accept {@code file_url} and run through
 * the audio resource. Unsupported endpoints refuse by name. {@code method} must be POST.</p>
 *
 * <p>The command ({@code neuralfree run-batch}) reads its input from a local path or an http(s) URL and writes its
 * output to a local path or uploads it to an http(s) URL ({@link BatchFiles}), and with {@code --enable-metrics}
 * serves the engine's Prometheus metrics while the batch runs ({@link BatchMetrics}).</p>
 */
public final class BatchRunner {

    /** The most jobs one wave queues at the engine (vLLM's {@code --max-inflight} floor). */
    public static final int DEFAULT_MAX_INFLIGHT = 1024;

    /** One validated input record. */
    record Line(int number, String customId, String method, String url, String bodyJson, JsonView body) { }

    /** The run's counts: the records read, how many answered with a body and how many with an error. */
    record Summary(long requests, long succeeded, long failed) { }

    /** The metrics server's default port (vLLM's {@code --port}). */
    public static final int DEFAULT_METRICS_PORT = 8000;

    /** The metrics server's default address: loopback, as the HTTP server binds by default. */
    public static final String DEFAULT_METRICS_HOST = "127.0.0.1";

    /** The runner's options: the files, the model, the wave size, the staging directory and the metrics server. */
    record Args(BatchFiles.Location input, BatchFiles.Location output, String model, int maxInflight,
                Path outputTmpDir, boolean enableMetrics, String metricsHost, int metricsPort) { }

    /**
     * One generation record readied for a wave: its plan and its job count, or the planning refusal. Never
     * planned onto the engine twice.
     */
    private static final class Planned {
        final Line line;
        final ChatPlan chat;
        final CompletionPlan completion;
        final int n;
        final int jobs;
        final WebApplicationException failure;

        Planned(Line line, ChatPlan chat, CompletionPlan completion, int n, WebApplicationException failure) {
            this.line = line;
            this.chat = chat;
            this.completion = completion;
            this.n = n;
            this.jobs = chat != null ? chat.jobs().size() : completion == null ? 0 : completion.jobs().size();
            this.failure = failure;
        }

        /** The chat plan of the record, or null. */
        boolean isChat() {
            return chat != null;
        }
    }

    private BatchRunner() {
    }

    // ------------------------------------------------------------------ the run

    /** The mutable counts of one run: the records read, how many answered with a body and how many with an error. */
    private static final class Counts {
        long requests;
        long succeeded;
        long failed;

        /** Adds one flush's outcome. */
        void flushed(long[] counts) {
            succeeded += counts[0];
            failed += counts[1];
        }
    }

    /**
     * Runs {@code input} into {@code output}. The file must have passed {@link #validate}; a record the engine or
     * the mapping refuses still writes its error record. A wave that outlives the request budget or fails on the
     * engine aborts the run with the records before it written.
     *
     * @return the counts of the records that produced output
     */
    public static Summary run(EngineService service, Path input, Path output, int maxInflight) throws IOException {
        if (maxInflight < 1) {
            throw new IllegalArgumentException("maxInflight must be at least 1, got " + maxInflight);
        }
        PoolingResource pooling = new PoolingResource(service);
        AudioResource audio = new AudioResource(service);
        Counts counts = new Counts();
        try (BufferedReader in = Files.newBufferedReader(input, StandardCharsets.UTF_8);
             BufferedWriter out = Files.newBufferedWriter(output, StandardCharsets.UTF_8)) {
            runRecords(service, pooling, audio, in, out, maxInflight, counts);
        } catch (WebApplicationException e) {
            // a wave outlived its budget or the engine failed; the records before it are on disk
            throw new IOException("the batch stopped: " + OpenAiResource.errorEnvelope(e).get("message"), e);
        }
        return new Summary(counts.requests, counts.succeeded, counts.failed);
    }

    /** The record loop: generation records join shared waves, pooling records run inline, every output lands in
     *  input order. */
    private static void runRecords(EngineService service, PoolingResource pooling, AudioResource audio,
                                   BufferedReader in, BufferedWriter out, int maxInflight, Counts counts) throws IOException {
        List<Planned> wave = new ArrayList<>();
        int jobs = 0;
        int number = 0;
        String raw;
        while ((raw = in.readLine()) != null) {
            number++;
            if (raw.isBlank()) {
                continue;
            }
            counts.requests++;
            Line line = parseLine(number, raw);
            if (!line.method().equals("POST")) {
                counts.failed++;
                write(out, errorRecord(line, 400, "invalid_request_error",
                        "method must be POST, got " + line.method()));
                continue;
            }
            String endpoint = route(line.url());
            Planned planned = isGeneration(endpoint) ? plan(service, line, endpoint) : null;
            boolean deferred = planned != null && jobs + planned.jobs <= maxInflight;
            if (!deferred && !wave.isEmpty()) {
                counts.flushed(flush(service, wave, out));
                wave.clear();
                jobs = 0;
            }
            if (planned == null) {
                Map<String, Object> record = inline(pooling, audio, line, endpoint);
                counts.succeeded += record.get("error") == null ? 1 : 0;
                counts.failed += record.get("error") == null ? 0 : 1;
                write(out, record);
            } else if (deferred) {
                jobs += planned.jobs;
                wave.add(planned);
            } else {
                // the record alone asks for more than one wave holds: it runs as its own wave
                counts.flushed(flush(service, List.of(planned), out));
            }
        }
        counts.flushed(flush(service, wave, out));
    }

    /**
     * The chat choice sink of one wave slot. Completion slots never reach a client: their choices hold a
     * pass-through output whose deltas nobody reads, but the slots keep the events of their jobs routed.
     */
    private static ChatChoice sink(Planned p) {
        return p.isChat() ? p.chat.newChoice()
                : new ChatChoice(new ToolOutput(null, false, 1, ToolCallStream::randomId), true);
    }

    /**
     * Runs one wave: submits every planned job, waits for them together, then writes each record's output in
     * order. A planning refusal or an engine refusal of one job writes that record's error record and the wave
     * goes on.
     */
    private static long[] flush(EngineService service, List<Planned> wave, BufferedWriter out) throws IOException {
        long[] counts = new long[2];
        if (wave.isEmpty()) {
            return counts;
        }
        List<GenerationBatch.Job> jobs = new ArrayList<>();
        List<ChatChoice> choices = new ArrayList<>();
        for (Planned p : wave) {
            if (p.failure != null) {
                continue;
            }
            if (p.isChat()) {
                jobs.addAll(p.chat.jobs());
            } else {
                jobs.addAll(p.completion.jobs());
            }
            for (int i = 0; i < p.jobs; i++) {
                choices.add(sink(p));
            }
        }
        GenerationBatch batch = null;
        if (!jobs.isEmpty()) {
            batch = GenerationBatch.start(service, jobs, null);
            batch.awaitAll();
            OpenAiResource.drainInto(batch, choices);
        }
        int slot = 0;
        for (Planned p : wave) {
            Map<String, Object> record;
            if (p.failure != null) {
                record = errorRecord(p.line, p.failure);
                counts[1]++;
            } else {
                int size = p.jobs;
                Map<String, Object> body = batch == null ? null
                        : generationBody(service, p, batch, choices, slot);
                record = body != null ? outputRecord(p.line.customId(), successResponse(ServerJson.write(body)))
                        : errorRecord(p.line, 400, "invalid_request_error", firstFailure(batch, slot, size));
                counts[record.get("error") == null ? 0 : 1]++;
                slot += size;
            }
            write(out, record);
        }
        return counts;
    }

    /** The first job failure of a record's slot range, or a message when none is reported. */
    private static String firstFailure(GenerationBatch batch, int slot, int size) {
        for (int i = 0; i < size; i++) {
            String failure = batch.failure(slot + i);
            if (failure != null) {
                return failure;
            }
        }
        return "the request produced no result";
    }

    /** The response body of one planned record over its finished slots, or null when the engine refused a job.
     *  A {@code best_of} record reports its best {@code n} candidates. */
    private static Map<String, Object> generationBody(EngineService service, Planned p, GenerationBatch batch,
                                                      List<ChatChoice> waveChoices, int slot) {
        int size = p.jobs;
        for (int i = 0; i < size; i++) {
            if (batch.failure(slot + i) != null) {
                return null;
            }
        }
        List<GenerationResult> candidates = new ArrayList<>();
        for (int i = 0; i < size; i++) {
            candidates.add(batch.result(slot + i));
        }
        String id = (p.isChat() ? "chatcmpl-" : "cmpl-") + System.nanoTime();
        long created = System.currentTimeMillis() / 1000;
        if (p.isChat()) {
            int[] picked = OpenAiResource.bestOfSelection(candidates, p.n, size);
            List<GenerationResult> results = OpenAiResource.selected(candidates, picked);
            List<ChatChoice> choices = new ArrayList<>();
            for (int i : picked) {
                choices.add(waveChoices.get(slot + i));
            }
            return OpenAiResource.chatBody(p.chat.bodies(service), p.chat, id, created, choices, results,
                    OpenAiResource.cachedTokensOf(batch, offset(slot, picked)));
        }
        int[] picked = OpenAiResource.bestOfSelection(candidates, p.completion.n(), size);
        return OpenAiResource.completionBody(p.completion.bodies(service), p.completion, id, created,
                OpenAiResource.selected(candidates, picked),
                OpenAiResource.cachedTokensOf(batch, offset(slot, picked)));
    }

    /** The absolute job indices of a record's picked slots. */
    private static int[] offset(int slot, int[] picked) {
        int[] out = new int[picked.length];
        for (int i = 0; i < out.length; i++) {
            out[i] = slot + picked[i];
        }
        return out;
    }

    /** Plans a generation record; its refusal rides the plan as the record's error. */
    private static Planned plan(EngineService service, Line line, String endpoint) {
        try {
            if (line.body().optBool("stream", false)) {
                throw ServerJson.badRequest("stream must be false or omitted: a batch request must not stream");
            }
            if (OpenAiRequests.beamSearch(line.body())) {
                throw ServerJson.badRequest("use_beam_search is not supported by batch requests");
            }
            if (endpoint.equals("chat")) {
                return new Planned(line, ChatPlan.of(service, line.body()), null,
                        OpenAiRequests.choices(line.body()), null);
            }
            return new Planned(line, null, CompletionPlan.of(service, line.body()), 0, null);
        } catch (WebApplicationException e) {
            return new Planned(line, null, null, 0, e);
        }
    }

    /** Runs one pooling or audio record, or returns the refusal of an unusable record. */
    private static Map<String, Object> inline(PoolingResource pooling, AudioResource audio, Line line, String endpoint) {
        if (endpoint == null) {
            return errorRecord(line, 400, "invalid_request_error", "URL " + line.url()
                    + " is not a supported endpoint. Supported endpoints: /v1/chat/completions, /v1/completions,"
                    + " /v1/embeddings, /score, /rerank, /v1/audio/transcriptions, /v1/audio/translations.");
        }
        try {
            Response response = switch (endpoint) {
                case "embeddings" -> pooling.embeddings(line.bodyJson());
                case "rerank" -> pooling.rerank(line.bodyJson());
                case "score" -> pooling.score(line.bodyJson());
                case "transcriptions" -> audio.batch(line.body(), false);
                case "translations" -> audio.batch(line.body(), true);
                default -> throw new IllegalStateException("no handler for " + endpoint);
            };
            return outputRecord(line.customId(), successResponse(String.valueOf(response.getEntity())));
        } catch (WebApplicationException e) {
            return errorRecord(line, e);
        }
    }

    // ------------------------------------------------------------------ records

    /** Routes a record's URL: OpenAI endpoints match exactly, score and rerank by suffix, as vLLM matches. */
    static String route(String url) {
        if (url.equals("/v1/chat/completions")) {
            return "chat";
        }
        if (url.equals("/v1/completions")) {
            return "completions";
        }
        if (url.equals("/v1/embeddings")) {
            return "embeddings";
        }
        if (url.equals("/v1/audio/transcriptions")) {
            return "transcriptions";
        }
        if (url.equals("/v1/audio/translations")) {
            return "translations";
        }
        if (url.endsWith("/score")) {
            return "score";
        }
        if (url.endsWith("/rerank")) {
            return "rerank";
        }
        return null;
    }

    /** Whether the endpoint plans onto the shared generation waves. */
    static boolean isGeneration(String endpoint) {
        return "chat".equals(endpoint) || "completions".equals(endpoint);
    }

    /**
     * Checks every record's envelope and returns the count; a record that is no JSON object, or without
     * {@code custom_id}, {@code method}, {@code url} or a {@code body} object, aborts the batch naming its line.
     * Blank lines are skipped. Run before the model loads, so bad input fails fast.
     */
    public static int validate(Path input) throws IOException {
        int count = 0;
        try (BufferedReader in = Files.newBufferedReader(input, StandardCharsets.UTF_8)) {
            int number = 0;
            String raw;
            while ((raw = in.readLine()) != null) {
                number++;
                if (raw.isBlank()) {
                    continue;
                }
                count++;
                parseLine(number, raw);
            }
        }
        return count;
    }

    /** Parses one record's envelope; the body stays JSON text for the endpoint handlers. */
    private static Line parseLine(int number, String raw) {
        try {
            JsonView line = JsonView.of(Json.parse(raw, Json.Limits.UNTRUSTED), "line " + number);
            String customId = line.str("custom_id");
            String method = line.str("method");
            String url = line.str("url");
            if (customId.isBlank()) {
                throw new IllegalArgumentException("custom_id must be a non-empty string");
            }
            if (method.isBlank()) {
                throw new IllegalArgumentException("method must be a non-empty string");
            }
            if (url.isBlank()) {
                throw new IllegalArgumentException("url must be a non-empty string");
            }
            Object body = line.raw("body");
            if (!(body instanceof Map)) {
                throw new IllegalArgumentException("body must be a JSON object");
            }
            JsonView bodyView = JsonView.of(body, "line " + number);
            return new Line(number, customId, method, url, ServerJson.write(body), bodyView);
        } catch (JsonException | IllegalArgumentException e) {
            throw new IllegalArgumentException("Invalid request on line " + number + ": " + e.getMessage(), e);
        }
    }

    /** The output record of a finished request: the body under {@code response} and no error. */
    private static Map<String, Object> outputRecord(String customId, Map<String, Object> response) {
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("id", "batch_req-" + UUID.randomUUID());
        out.put("custom_id", customId);
        out.put("response", response);
        out.put("error", null);
        return out;
    }

    /** The {@code response} of a finished request: the status, a request id and the endpoint's body. */
    private static Map<String, Object> successResponse(String bodyJson) {
        Map<String, Object> response = new LinkedHashMap<>();
        response.put("status_code", 200);
        response.put("request_id", "req-" + UUID.randomUUID());
        response.put("body", Json.parse(bodyJson, Json.Limits.UNTRUSTED));
        return response;
    }

    /** The output record of a refused request: the status under {@code response}, the envelope under {@code error}. */
    private static Map<String, Object> errorRecord(Line line, WebApplicationException e) {
        int status = e.getResponse().getStatus();
        Map<String, Object> envelope = OpenAiResource.errorEnvelope(e);
        return errorRecord(line, status, String.valueOf(envelope.get("type")),
                String.valueOf(envelope.get("message")));
    }

    /** {@link #errorRecord(Line, WebApplicationException)} over an explicit status, type and message. */
    private static Map<String, Object> errorRecord(Line line, int status, String type, String message) {
        Map<String, Object> response = new LinkedHashMap<>();
        response.put("status_code", status);
        response.put("request_id", "req-" + UUID.randomUUID());
        response.put("body", null);
        Map<String, Object> error = new LinkedHashMap<>();
        error.put("message", message);
        error.put("type", type);
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("id", "batch_req-" + UUID.randomUUID());
        out.put("custom_id", line.customId());
        out.put("response", response);
        out.put("error", error);
        return out;
    }

    /** Writes one output record as one JSONL line. */
    private static void write(BufferedWriter out, Map<String, Object> record) throws IOException {
        out.write(ServerJson.write(record));
        out.write("\n");
    }

    // ------------------------------------------------------------------ the command line

    /** Opens the batch's engine over a model once the input is validated. */
    @FunctionalInterface
    interface Engines {
        EngineService open(String model) throws IOException;
    }

    /**
     * The command: {@code neuralfree run-batch -i <input> -o <output> -m <model.gguf> [options]}, the input and the
     * output each a local path or an http(s) URL. The engine and server settings read their {@code neuralfree.*}
     * system properties as the server does.
     */
    public static void main(String[] args) {
        try {
            SecretProperties.loadFromEnvironment();
        } catch (RuntimeException e) {
            System.err.println("Error: " + e.getMessage());
            System.exit(1);
            return;
        }
        Args parsed;
        try {
            parsed = parseArgs(args);
        } catch (IllegalArgumentException e) {
            System.err.println("Error: " + e.getMessage());
            usage(System.err);
            System.exit(2);
            return;
        }
        if (parsed == null) {
            return;
        }
        try {
            Summary summary = execute(parsed, BatchFiles.standard(), model -> new EngineService(model,
                    EngineConfig.fromSystemProperties(), ServerOptions.fromSystemProperties(),
                    MediaOptions.fromSystemProperties(), ServerLoras.fromSystemProperties()));
            System.out.println(summary.succeeded() + " of " + summary.requests() + " requests succeeded, "
                    + summary.failed() + " failed; output " + (parsed.output().isUrl() ? "uploaded" : "written")
                    + " to " + parsed.output().display());
        } catch (IOException | RuntimeException e) {
            System.err.println("Error: " + e.getMessage());
            System.exit(1);
        }
    }

    /**
     * Runs a parsed command: checks the arguments, starts the metrics server when asked, stages the input, validates
     * it before the model loads, runs it and publishes the output. The engine {@code engines} opens is closed here.
     */
    static Summary execute(Args args, BatchFiles files, Engines engines) throws IOException {
        checkArgs(args);
        AtomicReference<EngineService> current = new AtomicReference<>();
        try (BatchMetrics metrics = args.enableMetrics() ? startMetrics(args, current) : null) {
            return runFiles(args, files, engines, service -> {
                current.set(service);
                if (metrics != null) {
                    metrics.opened(service);
                }
            });
        }
    }

    /** Starts the Prometheus metrics server over the service {@code current} will hold. */
    private static BatchMetrics startMetrics(Args args, AtomicReference<EngineService> current) throws IOException {
        BatchMetrics metrics = BatchMetrics.start(args.metricsHost(), args.metricsPort(), current::get);
        System.err.println("Prometheus metrics enabled on http://" + args.metricsHost() + ":" + metrics.port()
                + "/metrics");
        return metrics;
    }

    /**
     * Stages the input, validates it, opens the engine (handing it to {@code opened}), runs the batch and publishes
     * the output.
     */
    private static Summary runFiles(Args args, BatchFiles files, Engines engines, Consumer<EngineService> opened)
            throws IOException {
        try (BatchFiles.Input input = files.openInput(args.input(), args.outputTmpDir())) {
            int count = validate(input.path());
            System.err.println("Running " + count + " batch requests from " + args.input().display());
            EngineService service = engines.open(args.model());
            opened.accept(service);
            try (BatchFiles.Output output = files.openOutput(args.output(), args.outputTmpDir())) {
                Summary summary = run(service, input.path(), output.path(), args.maxInflight());
                output.finish();
                return summary;
            } finally {
                service.close();
            }
        }
    }

    /** The options as they are read; {@code args()} checks the combination and freezes them. */
    private static final class Options {
        BatchFiles.Location input;
        BatchFiles.Location output;
        String model = System.getProperty("neuralfree.server.model");
        int maxInflight = DEFAULT_MAX_INFLIGHT;
        Path outputTmpDir;
        boolean enableMetrics;
        String host;
        String url;
        Integer port;

        /** Reads one option and its value; false for an option the command does not have. */
        boolean read(String option, Supplier<String> value) {
            switch (option) {
                case "-i", "--input-file" -> input = BatchFiles.location("--input-file", value.get());
                case "-o", "--output-file" -> output = BatchFiles.location("--output-file", value.get());
                case "-m", "--model" -> model = value.get();
                case "--max-inflight" -> maxInflight = whole("--max-inflight", value.get());
                case "--output-tmp-dir" -> outputTmpDir = Path.of(value.get());
                case "--enable-metrics" -> enableMetrics = true;
                case "--host" -> host = value.get();
                case "--url" -> url = value.get();
                case "--port" -> port = metricsPort(value.get());
                default -> {
                    return false;
                }
            }
            return true;
        }

        /** The frozen options; the metrics server's options need {@code --enable-metrics}. */
        Args args() {
            if (!enableMetrics && (host != null || url != null || port != null)) {
                throw new IllegalArgumentException("--host, --url and --port configure the metrics server and"
                        + " need --enable-metrics");
            }
            String metricsHost = host;
            if (metricsHost == null && url != null) {
                System.err.println("Using --url for metrics is deprecated. Please use --host instead.");
                metricsHost = url;
            }
            return new Args(input, output, model, maxInflight, outputTmpDir, enableMetrics,
                    metricsHost == null ? DEFAULT_METRICS_HOST : metricsHost,
                    port == null ? DEFAULT_METRICS_PORT : port);
        }
    }

    /**
     * The options of the arguments; null for {@code --help}. A {@code --name=value} spelling is accepted. Missing
     * {@code --model} falls back to the {@code neuralfree.server.model} property.
     *
     * @throws IllegalArgumentException for an unknown option, a missing or malformed value, a URL that is not
     *                                  http(s), or a metrics option without {@code --enable-metrics}
     */
    static Args parseArgs(String[] args) {
        Options options = new Options();
        for (int i = 0; i < args.length; i++) {
            String arg = args[i];
            if (arg.equals("--help") || arg.equals("-h")) {
                usage(System.out);
                return null;
            }
            int eq = arg.indexOf('=');
            boolean inline = arg.startsWith("--") && eq > 2;
            String option = inline ? arg.substring(0, eq) : arg;
            int next = i + 1;
            boolean[] consumed = new boolean[1];
            Supplier<String> value = () -> {
                consumed[0] = true;
                return inline ? arg.substring(eq + 1) : value(args, next, option);
            };
            if (!options.read(option, value)) {
                throw new IllegalArgumentException("run-batch has no option '" + arg + "'");
            }
            if (inline && !consumed[0]) {
                throw new IllegalArgumentException(option + " takes no value");
            }
            i += consumed[0] && !inline ? 1 : 0;
        }
        return options.args();
    }

    /** The value of an option, refusing a missing one. */
    private static String value(String[] args, int index, String option) {
        if (index >= args.length) {
            throw new IllegalArgumentException(option + " needs a value");
        }
        return args[index];
    }

    /** A whole-number option value. */
    private static int whole(String option, String raw) {
        try {
            return Integer.parseInt(raw.trim());
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(option + " must be a whole number, got '" + raw + "'", e);
        }
    }

    /** The metrics port, 0 (a free port) to 65535. */
    private static int metricsPort(String raw) {
        int port = whole("--port", raw);
        if (port < 0 || port > 65535) {
            throw new IllegalArgumentException("--port must be between 0 and 65535, got " + port);
        }
        return port;
    }

    /**
     * The checks of vLLM's {@code validate_run_batch_args}: the required options, a local input readable, a local
     * output writable in an existing directory, never both the same file, and the staging directory a writable
     * directory.
     */
    static void checkArgs(Args args) throws IOException {
        if (args.input() == null) {
            throw new IllegalArgumentException("-i/--input-file is required");
        }
        if (args.output() == null) {
            throw new IllegalArgumentException("-o/--output-file is required");
        }
        if (args.model() == null || args.model().isBlank()) {
            throw new IllegalArgumentException("-m/--model (or -Dneuralfree.server.model) must name a gguf model");
        }
        if (args.maxInflight() < 1) {
            throw new IllegalArgumentException("--max-inflight must be at least 1, got " + args.maxInflight());
        }
        if (!args.input().isUrl() && !Files.isReadable(args.input().path())) {
            throw new IOException("the input file " + args.input().path() + " does not exist or is not readable");
        }
        if (!args.output().isUrl()) {
            checkLocalOutput(args.output().path());
            if (!args.input().isUrl() && Files.exists(args.output().path())
                    && Files.isSameFile(args.input().path(), args.output().path())) {
                throw new IOException("--input-file and --output-file are the same file (" + args.output().path()
                        + "). Records are written while the batch runs, so the input would be truncated before"
                        + " it has been read.");
            }
        }
        Path tmp = args.outputTmpDir();
        if (tmp != null && (!Files.isDirectory(tmp) || !Files.isWritable(tmp))) {
            throw new IOException("--output-tmp-dir " + tmp + " is not a writable directory.");
        }
    }

    /** A local output must be a writable file, or a new file in an existing writable directory. */
    private static void checkLocalOutput(Path output) throws IOException {
        if (Files.isDirectory(output)) {
            throw new IOException("--output-file " + output + " is a directory.");
        }
        Path parent = output.toAbsolutePath().getParent();
        if (!Files.exists(output) && parent != null && !Files.isDirectory(parent)) {
            throw new IOException("--output-file directory " + parent + " does not exist.");
        }
        Path target = Files.exists(output) ? output : parent;
        if (target != null && !Files.isWritable(target)) {
            throw new IOException("--output-file " + output + " is not writable.");
        }
    }

    /** The command surface as {@code --help} and every error print it. */
    static void usage(Appendable out) {
        try {
            out.append("neuralfree run-batch -i <input.jsonl|URL> -o <output.jsonl|URL> -m <model.gguf> [options]\n");
            out.append("  -i, --input-file     the OpenAI batch input, one {\"custom_id\", \"method\", \"url\","
                    + " \"body\"} record per line; a local path or an http(s) URL read with GET\n");
            out.append("  -o, --output-file    the OpenAI batch output, one {\"id\", \"custom_id\","
                    + " \"response\", \"error\"} record per line; a local path or an http(s) URL the finished file"
                    + " is uploaded to with PUT\n");
            out.append("  -m, --model          the gguf model (fallback: -Dneuralfree.server.model)\n");
            out.append("  --max-inflight N     the most jobs one wave queues at the engine (default "
                    + DEFAULT_MAX_INFLIGHT + ")\n");
            out.append("  --output-tmp-dir DIR where a URL input is downloaded and a URL output is staged"
                    + " (default: the system temporary directory)\n");
            out.append("  --enable-metrics     serve Prometheus metrics while the batch runs\n");
            out.append("  --host HOST          the metrics server's address (default " + DEFAULT_METRICS_HOST
                    + "; --url is its deprecated spelling)\n");
            out.append("  --port PORT          the metrics server's port (default " + DEFAULT_METRICS_PORT + ")\n");
            out.append("  -h, --help           this help\n");
        } catch (IOException e) {
            throw new IllegalStateException("cannot print the run-batch usage", e);
        }
    }
}

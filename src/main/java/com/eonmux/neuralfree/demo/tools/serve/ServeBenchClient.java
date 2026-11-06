package com.eonmux.neuralfree.demo.tools.serve;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.security.SecureRandom;
import java.security.cert.X509Certificate;
import java.time.Duration;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Semaphore;

import javax.net.ssl.SSLContext;
import javax.net.ssl.TrustManager;
import javax.net.ssl.X509TrustManager;

import com.eonmux.neuralfree.utils.json.Json;
import com.eonmux.neuralfree.utils.json.JsonException;

/**
 * The load generator of the serving benchmarks: an HTTP client over the server's OpenAI-compatible
 * endpoints, as vLLM's request functions are over aiohttp. The generation endpoints stream
 * ({@code stream: true} with {@code stream_options.include_usage}), and the client stamps each
 * content event's arrival: the first one gives the TTFT, the gaps between consecutive ones the
 * ITLs, the last one closes the latency, and the usage chunk carries the token counts. The
 * {@code /v1/chat/completions} endpoint wraps a sample as one user message and, like the reference,
 * skips the role-only opening chunk; {@code /rerank} sends the query with its documents and a
 * non-streamed reply; {@code /v1/audio/transcriptions} uploads the audio as multipart form data and
 * streams the transcription. One request runs per connection up to the concurrency cap
 * ({@code --max-concurrency}); a run reports per-request queue waits past the scheduled arrival.
 */
public final class ServeBenchClient {

    /** The generation endpoint of text prompts. */
    public static final String COMPLETIONS = "/v1/completions";
    /** The generation endpoint of chat messages. */
    public static final String CHAT = "/v1/chat/completions";
    /** The rerank endpoint. */
    public static final String RERANK = "/rerank";
    /** The transcription endpoint. */
    public static final String AUDIO = "/v1/audio/transcriptions";

    private final HttpClient client;
    private final String apiBaseUrl;
    private final String endpoint;
    private final Map<String, String> headers;
    private final String modelId;
    private final String modelName;
    private final boolean ignoreEos;
    private final Map<String, Object> samplingParams;
    private final Integer logprobs;
    private final boolean chat;
    private final boolean rerank;
    private final boolean audio;
    private final boolean useBeamSearch;

    /**
     * A client over {@code baseUrl + endpoint} of an OpenAI-compatible server.
     *
     * @param modelId   the payload's {@code model} (the served id, or an adapter's name)
     * @param modelName the payload's model name when the reference distinguishes the two, else null
     * @param endpoint  the URL path, see the endpoint constants
     * @param insecure  whether to trust any certificate (self-signed servers)
     * @throws IllegalArgumentException for an endpoint outside the four the harness drives
     */
    public ServeBenchClient(String baseUrl, String endpoint, Map<String, String> headers,
                            String modelId, String modelName, boolean ignoreEos,
                            Map<String, Object> samplingParams, Integer logprobs, boolean insecure) {
        this(baseUrl, endpoint, headers, modelId, modelName, ignoreEos, samplingParams, logprobs,
                insecure, false);
    }

    /** {@link #ServeBenchClient(String, String, Map, String, String, boolean, Map, Integer, boolean)}
     * with the beam-search flag of the generation requests. */
    public ServeBenchClient(String baseUrl, String endpoint, Map<String, String> headers,
                            String modelId, String modelName, boolean ignoreEos,
                            Map<String, Object> samplingParams, Integer logprobs, boolean insecure,
                            boolean useBeamSearch) {
        String path = canonicalEndpoint(endpoint);
        this.client = buildClient(insecure);
        this.apiBaseUrl = baseUrl.endsWith("/") ? baseUrl.substring(0, baseUrl.length() - 1) : baseUrl;
        this.endpoint = path;
        this.headers = headers == null ? new LinkedHashMap<>() : new LinkedHashMap<>(headers);
        this.modelId = modelId;
        this.modelName = modelName;
        this.ignoreEos = ignoreEos;
        this.samplingParams = samplingParams == null ? new LinkedHashMap<>() : samplingParams;
        this.logprobs = logprobs;
        this.chat = path.equals(CHAT);
        this.rerank = path.equals(RERANK);
        this.audio = path.equals(AUDIO);
        this.useBeamSearch = useBeamSearch;
    }

    /** The endpoint spelling the client drives, aliases normalized. */
    static String canonicalEndpoint(String endpoint) {
        String path = endpoint == null ? COMPLETIONS : endpoint;
        if (path.equals(COMPLETIONS) || path.equals(CHAT) || path.equals(RERANK) || path.equals(AUDIO)) {
            return path;
        }
        if (path.equals("/v1/rerank") || path.equals("/v1/reranking") || path.equals("/reranking")
                || path.equals("/v2/rerank")) {
            return RERANK;
        }
        if (path.equals("/audio/transcriptions")) {
            return AUDIO;
        }
        throw new IllegalArgumentException("the benchmark endpoint is one of " + COMPLETIONS + ", "
                + CHAT + ", " + RERANK + ", " + AUDIO + ", not '" + endpoint + "'");
    }

    /** One pooled client; {@code --insecure} swaps the certificate check for a trust-all manager. */
    private static HttpClient buildClient(boolean insecure) {
        HttpClient.Builder builder = HttpClient.newBuilder()
                .executor(Executors.newCachedThreadPool(r -> {
                    Thread t = new Thread(r, "bench-serve-client");
                    t.setDaemon(true);
                    return t;
                }))
                .connectTimeout(Duration.ofSeconds(30));
        if (insecure) {
            builder.sslContext(trustAll());
        }
        return builder.build();
    }

    /** The SSL context of {@code --insecure}: every certificate trusted, no host name checked. */
    private static SSLContext trustAll() {
        TrustManager trustAll = new X509TrustManager() {
            @Override
            public void checkClientTrusted(X509Certificate[] chain, String authType) {
            }

            @Override
            public void checkServerTrusted(X509Certificate[] chain, String authType) {
            }

            @Override
            public X509Certificate[] getAcceptedIssuers() {
                return new X509Certificate[0];
            }
        };
        try {
            SSLContext context = SSLContext.getInstance("TLS");
            context.init(null, new TrustManager[]{trustAll}, new SecureRandom());
            return context;
        } catch (Exception e) {
            throw new IllegalStateException("cannot build the trust-all SSL context", e);
        }
    }

    /** The endpoint this client drives. */
    public String endpoint() {
        return endpoint;
    }

    /** One streamed request, synchronous: the caller of a ready check or a warmup. */
    public BenchRequestOutput send(BenchSample sample) {
        try {
            return exchange(sample, sample.promptLen(), 0.0, System.nanoTime(), null);
        } catch (IOException e) {
            BenchRequestOutput out = new BenchRequestOutput();
            out.success = false;
            out.error = e.getMessage() == null ? e.getClass().getSimpleName() : e.getMessage();
            return out;
        }
    }

    /**
     * The whole workload: sends request {@code i} at the schedule's {@code delayAt(i)}, at most
     * {@code maxConcurrency} in flight (unbounded when null), and returns the outputs in the
     * workload's order once the last one closed. {@code requestModels} names the payload's model per
     * request (a LoRA run's adapters), null for the client's fixed model. Every output's
     * {@code startTime} is its send time in seconds from this run's begin, and its
     * {@code clientQueueTime} the wait from the scheduled arrival to the actual send (the concurrency
     * cap's queue).
     */
    public List<BenchRequestOutput> runAll(List<BenchSample> samples, ArrivalSchedule schedule,
                                           Integer maxConcurrency, String[] requestModels) {
        return runAll(samples, schedule, maxConcurrency, requestModels, null);
    }

    /**
     * The whole workload with a progress sink, see {@link #runAll(List, ArrivalSchedule, Integer,
     * String[], Progress)}.
     */
    public List<BenchRequestOutput> runAll(List<BenchSample> samples, ArrivalSchedule schedule,
                                           Integer maxConcurrency, String[] requestModels, Progress progress) {
        Semaphore permits = maxConcurrency == null ? null : new Semaphore(maxConcurrency);
        ExecutorService scope = Executors.newCachedThreadPool(r -> {
            Thread t = new Thread(r, "bench-serve-send");
            t.setDaemon(true);
            return t;
        });
        try {
            long begin = System.nanoTime();
            List<CompletableFuture<BenchRequestOutput>> futures = new ArrayList<>(samples.size());
            for (int i = 0; i < samples.size(); i++) {
                long arrival = begin + (long) (schedule.delayAt(i) * 1_000_000_000.0);
                if (arrival > System.nanoTime()) {
                    sleepUntil(arrival);
                }
                BenchSample sample = samples.get(i);
                int promptLen = sample.promptLen();
                double scheduledSeconds = schedule.delayAt(i);
                String model = requestModels != null && i < requestModels.length ? requestModels[i] : null;
                Semaphore gate = permits;
                CompletableFuture<BenchRequestOutput> future = CompletableFuture.supplyAsync(() -> {
                    if (gate != null) {
                        gate.acquireUninterruptibly();
                    }
                    try {
                        return exchange(sample, promptLen, scheduledSeconds, begin, model);
                    } catch (IOException e) {
                        BenchRequestOutput out = new BenchRequestOutput();
                        out.success = false;
                        out.error = e.getMessage() == null ? e.getClass().getSimpleName() : e.getMessage();
                        return out;
                    } finally {
                        if (gate != null) {
                            gate.release();
                        }
                    }
                }, scope);
                futures.add(future);
                if (progress != null && !progress.done()) {
                    progress.started(i + 1, samples.size());
                }
            }
            List<BenchRequestOutput> outputs = new ArrayList<>(futures.size());
            for (CompletableFuture<BenchRequestOutput> future : futures) {
                outputs.add(future.join());
            }
            return outputs;
        } finally {
            scope.shutdown();
        }
    }

    /** Sleeps until the monotonic deadline, interrupted threads unwound. */
    private static void sleepUntil(long deadline) {
        while (true) {
            long rest = deadline - System.nanoTime();
            if (rest <= 0) {
                return;
            }
            long cap = Math.min(rest, 10_000_000L);
            try {
                Thread.sleep(cap / 1_000_000L, (int) (cap % 1_000_000L));
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new IllegalStateException("the benchmark's send schedule was interrupted", e);
            }
        }
    }

    /** The JSON payload of one generation request. */
    private Map<String, Object> payload(BenchSample sample, String modelOverride) {
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("model", modelOverride != null ? modelOverride
                : modelName != null ? modelName : modelId);
        if (sample.hasTokens() && chat) {
            throw new IllegalArgumentException("a token-id prompt needs the /v1/completions endpoint:"
                    + " /v1/chat/completions takes text only");
        }
        if (chat) {
            Map<String, Object> message = new LinkedHashMap<>();
            message.put("role", "user");
            if (sample.chatContent() != null) {
                message.put("content", sample.chatContent());
            } else {
                message.put("content", sample.text());
            }
            List<Map<String, Object>> messages = new ArrayList<>();
            messages.add(message);
            body.put("messages", messages);
        } else if (sample.hasTokens()) {
            List<Long> ids = new ArrayList<>(sample.tokens().length);
            for (int id : sample.tokens()) {
                ids.add((long) id);
            }
            body.put("prompt", ids);
        } else {
            body.put("prompt", sample.text());
        }
        body.put(chat ? "max_completion_tokens" : "max_tokens", sample.outputLen());
        body.put("repetition_penalty", 1.0);
        body.put("stream", true);
        Map<String, Object> streamOptions = new LinkedHashMap<>();
        streamOptions.put("include_usage", true);
        body.put("stream_options", streamOptions);
        body.put("ignore_eos", ignoreEos);
        if (logprobs != null && !chat) {
            body.put("logprobs", logprobs);
        }
        if (useBeamSearch) {
            body.put("use_beam_search", true);
        }
        body.putAll(samplingParams);
        body.putAll(sample.requestOverrides());
        return body;
    }

    /** The JSON payload of one rerank request. */
    private Map<String, Object> rerankPayload(BenchSample sample, String modelOverride) {
        List<String> texts = sample.rerankTexts();
        if (texts == null || texts.size() < 2) {
            throw new IllegalArgumentException("a rerank request carries a query and at least one"
                    + " document");
        }
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("model", modelOverride != null ? modelOverride
                : modelName != null ? modelName : modelId);
        body.put("query", texts.get(0));
        body.put("documents", new ArrayList<>(texts.subList(1, texts.size())));
        body.put("truncate_prompt_tokens", -1);
        body.putAll(sample.requestOverrides());
        return body;
    }

    /**
     * One exchange: the timings of every streamed event, the usage, the failures.
     * {@code scheduledSeconds} is the request's scheduled arrival from the run's begin,
     * {@code runBegin} the run's monotonic clock zero, whose seconds every output's
     * {@code startTime} reports, and {@code modelOverride} the payload's model (a LoRA run's
     * adapter), null for the client's.
     */
    private BenchRequestOutput exchange(BenchSample sample, int fallbackPromptLen, double scheduledSeconds,
                                        long runBegin, String modelOverride) throws IOException {
        BenchRequestOutput out = new BenchRequestOutput();
        out.promptLen = fallbackPromptLen;
        out.inputAudioDuration = sample.audioDurationSec();
        out.numInputSequences = sample.kind() == BenchSample.Kind.RERANK
                ? Math.max(1, sample.rerankTexts().size() - 1) : 1;
        HttpRequest request;
        long st = System.nanoTime();
        out.startTime = (st - runBegin) / 1e9;
        out.clientQueueTime = Math.max(0.0, out.startTime - scheduledSeconds);
        if (rerank) {
            request = jsonRequest(apiBaseUrl + endpoint, rerankPayload(sample, modelOverride), sample);
        } else if (audio) {
            request = audioRequest(apiBaseUrl + endpoint, sample, modelOverride);
        } else {
            request = jsonRequest(apiBaseUrl + endpoint, payload(sample, modelOverride), sample);
        }
        HttpResponse<java.io.InputStream> response;
        try {
            response = client.send(request, HttpResponse.BodyHandlers.ofInputStream());
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("the benchmark request was interrupted", e);
        }
        if (response.statusCode() != 200) {
            out.error = "HTTP " + response.statusCode() + ": " + shortBody(response.body());
            return out;
        }
        if (rerank) {
            readPooling(response.body(), out, st);
            return out;
        }
        try (java.io.InputStream stream = response.body()) {
            streamEvents(stream, out, st);
        } catch (IOException e) {
            if (out.error == null && !out.success) {
                out.error = e.getMessage() == null ? e.getClass().getSimpleName() : e.getMessage();
            }
        }
        return out;
    }

    /** A JSON POST carrying the request id header and the configured headers. */
    private HttpRequest jsonRequest(String url, Map<String, Object> body, BenchSample sample) {
        HttpRequest.Builder request = HttpRequest.newBuilder()
                .uri(URI.create(url))
                .timeout(Duration.ofHours(6))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(Json.toString(body, Json.Style.API),
                        StandardCharsets.UTF_8));
        for (Map.Entry<String, String> header : headers.entrySet()) {
            request.header(header.getKey(), header.getValue());
        }
        if (sample.requestId() != null && !sample.requestId().isEmpty()) {
            request.setHeader("X-Request-Id", sample.requestId());
        }
        return request.build();
    }

    /** The multipart transcription upload the reference builds from the audio file and options. */
    private HttpRequest audioRequest(String url, BenchSample sample, String modelOverride) {
        BenchDatasets.AudioFile file = BenchDatasets.audio(sample.audioPath(),
                java.nio.file.Paths.get(sample.audioPath()).toAbsolutePath().getParent());
        String boundary = "----neuralfree-bench-" + UUID.randomUUID();
        ByteArrayOutputStream buffer = new ByteArrayOutputStream();
        try {
            writeField(buffer, boundary, "model", modelOverride != null ? modelOverride
                    : modelName != null ? modelName : modelId);
            writeField(buffer, boundary, "max_completion_tokens", String.valueOf(sample.outputLen()));
            writeField(buffer, boundary, "stream", "true");
            writeField(buffer, boundary, "language", "en");
            for (Map.Entry<String, Object> param : samplingParams.entrySet()) {
                writeField(buffer, boundary, param.getKey(), String.valueOf(param.getValue()));
            }
            buffer.write(("--" + boundary + "\r\nContent-Disposition: form-data; name=\"file\";"
                    + " filename=\"" + file.filename + "\"\r\nContent-Type: " + file.contentType
                    + "\r\n\r\n").getBytes(StandardCharsets.UTF_8));
            buffer.write(file.bytes);
            buffer.write(("\r\n--" + boundary + "--\r\n").getBytes(StandardCharsets.UTF_8));
        } catch (IOException e) {
            throw new IllegalStateException("cannot build the transcription upload", e);
        }
        HttpRequest.Builder request = HttpRequest.newBuilder()
                .uri(URI.create(url))
                .timeout(Duration.ofHours(6))
                .header("Content-Type", "multipart/form-data; boundary=" + boundary)
                .POST(HttpRequest.BodyPublishers.ofByteArray(buffer.toByteArray()));
        for (Map.Entry<String, String> header : headers.entrySet()) {
            request.header(header.getKey(), header.getValue());
        }
        if (sample.requestId() != null && !sample.requestId().isEmpty()) {
            request.setHeader("X-Request-Id", sample.requestId());
        }
        return request.build();
    }

    private static void writeField(OutputStream out, String boundary, String name, String value)
            throws IOException {
        out.write(("--" + boundary + "\r\nContent-Disposition: form-data; name=\"" + name + "\"\r\n\r\n"
                + value + "\r\n").getBytes(StandardCharsets.UTF_8));
    }

    /** A non-streamed pooling reply: success on 200, the latency as the first-token time. */
    private static void readPooling(java.io.InputStream body, BenchRequestOutput out, long st) {
        try (java.io.InputStream in = body) {
            String text = new String(in.readAllBytes(), StandardCharsets.UTF_8);
            Object parsed = Json.parse(text, Json.Limits.UNTRUSTED);
            if (parsed instanceof Map) {
                Object usage = ((Map<?, ?>) parsed).get("usage");
                if (usage instanceof Map) {
                    readUsage((Map<?, ?>) usage, out);
                }
            }
            double elapsed = (System.nanoTime() - st) / 1e9;
            out.ttft = elapsed;
            out.latency = elapsed;
            out.success = true;
        } catch (IOException | JsonException e) {
            out.success = false;
            out.error = "the pooling reply is not JSON: " + e.getMessage();
        }
    }

    /** The first 200 bytes of an error body, single-lined. */
    private static String shortBody(java.io.InputStream body) {
        try (java.io.InputStream in = body) {
            byte[] bytes = in.readNBytes(200);
            String text = new String(bytes, StandardCharsets.UTF_8).replace('\n', ' ');
            return text.isEmpty() ? "(no body)" : text;
        } catch (IOException e) {
            return "(unreadable body)";
        }
    }

    /**
     * Reads the SSE stream: every {@code data:} line is a chunk; a chunk with choices carries a
     * token (completions: any {@code choices[0].text} event; chat: the deltas the reference counts,
     * the opening role chunk excluded; audio: any choice), the first one stamps the TTFT and the
     * gaps the ITLs; the usage chunk carries the token counts. Success needs at least one content
     * event.
     */
    private void streamEvents(java.io.InputStream body, BenchRequestOutput out, long st) throws IOException {
        boolean firstChunk = true;
        long mostRecent = st;
        BufferedReader reader = new BufferedReader(new InputStreamReader(body, StandardCharsets.UTF_8));
        String line;
        while ((line = reader.readLine()) != null) {
            if (!line.startsWith("data:")) {
                continue;
            }
            String data = line.substring(5).trim();
            if (data.equals("[DONE]")) {
                break;
            }
            long now = System.nanoTime();
            Map<?, ?> chunk;
            try {
                chunk = (Map<?, ?>) Json.parse(data, Json.Limits.UNTRUSTED);
            } catch (JsonException e) {
                out.success = false;
                out.error = "a stream chunk is not JSON: " + e.getMessage();
                return;
            }
            Object error = chunk.get("error");
            if (error != null) {
                out.success = false;
                out.error = String.valueOf(error);
                return;
            }
            boolean carriesToken = carriesToken(chunk);
            Object usage = chunk.get("usage");
            if (usage instanceof Map) {
                readUsage((Map<?, ?>) usage, out);
            }
            if (carriesToken) {
                if (firstChunk) {
                    firstChunk = false;
                    out.ttft = (now - st) / 1e9;
                } else {
                    out.itls.add((now - mostRecent) / 1e9);
                }
                mostRecent = now;
                appendText(chunk, out.generatedText);
            }
        }
        if (firstChunk) {
            out.success = false;
            if (out.error == null) {
                out.error = "never received a valid chunk to calculate TTFT";
            }
            return;
        }
        out.success = true;
        out.latency = (mostRecent - st) / 1e9;
    }

    /** Whether a chunk carries one generated token, per endpoint and as the reference counts them. */
    private boolean carriesToken(Map<?, ?> chunk) {
        Object choices = chunk.get("choices");
        if (!(choices instanceof List) || ((List<?>) choices).isEmpty()) {
            return false;
        }
        Object first = ((List<?>) choices).get(0);
        if (!(first instanceof Map)) {
            return false;
        }
        Map<?, ?> choice = (Map<?, ?>) first;
        if (audio) {
            return true;
        }
        if (!chat) {
            return true;
        }
        return chatDeltaCarriesToken(choice);
    }

    /** The chat delta rule of the reference's {@code _choice_has_streamed_output}. */
    private static boolean chatDeltaCarriesToken(Map<?, ?> choice) {
        Object delta = choice.get("delta");
        if (!(delta instanceof Map)) {
            return false;
        }
        Map<?, ?> d = (Map<?, ?>) delta;
        if (truthy(d.get("content")) || truthy(d.get("reasoning"))
                || truthy(d.get("tool_calls")) || truthy(d.get("function_call"))) {
            return true;
        }
        return "".equals(d.get("content")) && !d.containsKey("role");
    }

    /** Python truthiness of a parsed JSON value: null, false, 0, "" and empty containers are false. */
    private static boolean truthy(Object value) {
        if (value == null || Boolean.FALSE.equals(value)) {
            return false;
        }
        if (value instanceof String) {
            return !((String) value).isEmpty();
        }
        if (value instanceof Number) {
            return ((Number) value).doubleValue() != 0.0;
        }
        if (value instanceof java.util.Collection) {
            return !((java.util.Collection<?>) value).isEmpty();
        }
        if (value instanceof Map) {
            return !((Map<?, ?>) value).isEmpty();
        }
        return true;
    }

    /** The generated text of a content chunk, appended when the chunk carries a text piece. */
    private static void appendText(Map<?, ?> chunk, StringBuilder text) {
        Object choices = chunk.get("choices");
        if (!(choices instanceof List) || ((List<?>) choices).isEmpty()) {
            return;
        }
        Object first = ((List<?>) choices).get(0);
        if (!(first instanceof Map)) {
            return;
        }
        Map<?, ?> choice = (Map<?, ?>) first;
        Object piece = choice.get("text");
        if (piece == null) {
            Object delta = choice.get("delta");
            if (delta instanceof Map) {
                piece = ((Map<?, ?>) delta).get("content");
            }
        }
        if (piece instanceof String) {
            text.append((String) piece);
        }
    }

    /** The usage chunk's counts: the completion tokens and, when present, the server's prompt count. */
    private static void readUsage(Map<?, ?> usage, BenchRequestOutput out) {
        Object completion = usage.get("completion_tokens");
        if (completion instanceof Number) {
            out.outputTokens = ((Number) completion).intValue();
        }
        Object prompt = usage.get("prompt_tokens");
        if (prompt instanceof Number) {
            out.promptLen = ((Number) prompt).intValue();
        }
    }

    /** The send progress of a run, reported by the caller's sink. */
    public interface Progress {

        /** Whether the sink wants no more reports. */
        boolean done();

        /** The sends started so far, of {@code total}. */
        void started(int count, int total);
    }
}

package com.eonmux.neuralfree.demo;

import com.eonmux.neuralfree.utils.json.Json;
import com.eonmux.neuralfree.utils.json.JsonException;
import com.eonmux.neuralfree.utils.json.JsonView;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.URI;
import java.net.URLEncoder;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.function.BooleanSupplier;

/**
 * The HTTP client of the console's client mode against a running NeuralFree, llama.cpp or OpenAI-compatible
 * server (llama.cpp tools/cli/cli-client.cpp), and of the OpenAI client subcommands {@code chat} and
 * {@code complete}: the {@code /health} wait, the {@code /v1/models} list, the {@code /props} properties and the
 * streaming {@code /v1/chat/completions} and {@code /v1/completions} calls, which a cancellation closes
 * mid-stream. A server-base client addresses a server by its root URL and appends the versioned paths
 * ({@link #ChatClient(String, String)}); an API-base client takes the OpenAI client's {@code base_url} and appends
 * the paths to it ({@link #forApiBase(String, String)}). Request failures surface as {@link IOException}s or as the
 * error text of a stream.
 */
final class ChatClient {

    /** The chat endpoint appended to a server base URL. */
    private static final String SERVER_CHAT = "/v1/chat/completions";

    /** The completions endpoint appended to a server base URL. */
    private static final String SERVER_COMPLETIONS = "/v1/completions";

    /** The models endpoint appended to a server base URL. */
    private static final String SERVER_MODELS = "/v1/models";

    /** The chat endpoint appended to an OpenAI API base URL. */
    private static final String API_CHAT = "/chat/completions";

    /** The completions endpoint appended to an OpenAI API base URL. */
    private static final String API_COMPLETIONS = "/completions";

    /** The models endpoint appended to an OpenAI API base URL. */
    private static final String API_MODELS = "/models";

    /** The delta handler of one streamed answer. */
    interface Deltas {

        /** Called once the server answered with a stream, before its first chunk arrives. */
        default void started() {
        }

        void reasoning(String text);

        void content(String text);

        void timings(double promptPerSecond, double predictedPerSecond);

        /** The completion tokens of the stream's {@code usage}, when the server reports them. */
        default void usage(int completionTokens) {
        }
    }

    /** A model of the server's {@code /v1/models} list: its id and alias names. Immutable. */
    static final class ModelInfo {
        final String id;
        final List<String> aliases;

        ModelInfo(String id, List<String> aliases) {
            this.id = id;
            this.aliases = Collections.unmodifiableList(new ArrayList<>(aliases));
        }

        /** The id, with the aliases in parentheses when there are any. */
        String display() {
            return aliases.isEmpty() ? id : id + " (" + String.join(", ", aliases) + ")";
        }
    }

    /** The server properties of {@code /props}; every modality is false unless the server reports it. Immutable. */
    static final class Props {
        static final Props NONE = new Props("", "", false, false, false);

        final String modelAlias;
        final String modelPath;
        final boolean vision;
        final boolean audio;
        final boolean video;

        Props(String modelAlias, String modelPath, boolean vision, boolean audio, boolean video) {
            this.modelAlias = modelAlias;
            this.modelPath = modelPath;
            this.vision = vision;
            this.audio = audio;
            this.video = video;
        }
    }

    /** Failed connections after which the health wait gives up (cli-client.cpp wait_health). */
    private static final int MAX_CONNECT_FAILURES = 10;
    private static final long HEALTH_POLL_MILLIS = 300;
    /** The read timeout of a request: prompt processing can stall a stream for long (CLI_HTTP_READ_TIMEOUT_SEC). */
    private static final Duration READ_TIMEOUT = Duration.ofHours(1);
    private static final Duration CONNECT_TIMEOUT = Duration.ofSeconds(10);
    /** The longest error body kept for the message (CLI_HTTP_MAX_ERROR_BODY). */
    private static final int MAX_ERROR_BODY = 1024 * 1024;
    private static final long CANCEL_POLL_MILLIS = 50;

    private final HttpClient http;
    private final String base;
    private final String apiKey;
    private final String chatPath;
    private final String completionsPath;
    private final String modelsPath;

    /**
     * A client for the server at {@code baseUrl}, trailing slashes stripped as llama.cpp's client strips them. A
     * path in the URL is the server's route prefix ({@code --api-prefix}); the versioned endpoints are appended to
     * it.
     *
     * @param apiKey the bearer token of every request, or null
     * @throws IllegalArgumentException when the URL is not an absolute http or https URL
     */
    ChatClient(String baseUrl, String apiKey) {
        this(baseUrl, apiKey, "--server-base", SERVER_CHAT, SERVER_COMPLETIONS, SERVER_MODELS);
    }

    /**
     * A client of an OpenAI API base URL such as {@code http://localhost:8000/v1}: every endpoint is appended to
     * the base, as the OpenAI clients append it, so a base with a path prefix routes correctly.
     *
     * @param apiKey the bearer token of every request, or null
     * @throws IllegalArgumentException when the URL is not an absolute http or https URL
     */
    static ChatClient forApiBase(String baseUrl, String apiKey) {
        return new ChatClient(baseUrl, apiKey, "--url", API_CHAT, API_COMPLETIONS, API_MODELS);
    }

    /** The validation and endpoint paths both modes share; {@code flag} names the URL option in refusals. */
    private ChatClient(String baseUrl, String apiKey, String flag, String chatPath, String completionsPath,
                       String modelsPath) {
        if (baseUrl == null) {
            throw new IllegalArgumentException(flag + " needs a URL");
        }
        String trimmed = baseUrl.trim();
        while (trimmed.endsWith("/")) {
            trimmed = trimmed.substring(0, trimmed.length() - 1);
        }
        URI uri;
        try {
            uri = URI.create(trimmed);
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException(flag + " '" + baseUrl + "' is not a URL: " + e.getMessage());
        }
        if (!"http".equals(uri.getScheme()) && !"https".equals(uri.getScheme()) || uri.getHost() == null) {
            throw new IllegalArgumentException(flag + " '" + baseUrl + "' must be an http:// or https:// URL"
                    + " with a host");
        }
        if (uri.getRawQuery() != null || uri.getRawFragment() != null) {
            throw new IllegalArgumentException(flag + " '" + baseUrl + "' must not carry a query or a fragment");
        }
        this.base = trimmed;
        this.apiKey = apiKey == null || apiKey.isEmpty() ? null : apiKey;
        this.chatPath = chatPath;
        this.completionsPath = completionsPath;
        this.modelsPath = modelsPath;
        this.http = HttpClient.newBuilder().connectTimeout(CONNECT_TIMEOUT).build();
    }

    /** The normalized base URL. */
    String base() {
        return base;
    }

    /**
     * Waits until {@code /health} answers HTTP 200 with no {@code "status": "error"} body. Any other answer keeps
     * waiting (llama.cpp answers 503 while a model loads); {@value #MAX_CONNECT_FAILURES} failed connections in
     * a row, or {@code aborted}, end the wait. Returns null once the server is ready, else the error text.
     */
    String waitHealth(BooleanSupplier aborted) throws InterruptedException {
        int connectFailures = 0;
        while (!aborted.getAsBoolean()) {
            try {
                HttpResponse<String> response = http.send(builder("/health", null).GET().build(),
                        HttpResponse.BodyHandlers.ofString());
                connectFailures = 0;
                if (response.statusCode() == 200 && !reportsError(response.body())) {
                    return null;
                }
            } catch (IOException e) {
                if (++connectFailures >= MAX_CONNECT_FAILURES) {
                    return "failed to connect to " + base + ": " + describe(e);
                }
            }
            Thread.sleep(HEALTH_POLL_MILLIS);
        }
        return "aborted while waiting for the server to become ready";
    }

    /** The server's models ({@code /v1/models}), in list order; entries without a string id are skipped. */
    List<ModelInfo> models() throws IOException, InterruptedException {
        JsonView body = parse(get(modelsPath, null), modelsPath);
        Object raw = body.raw("data");
        if (!(raw instanceof List)) {
            throw new IOException("invalid response from /v1/models: no 'data' array");
        }
        List<ModelInfo> out = new ArrayList<>();
        for (Object o : (List<?>) raw) {
            if (!(o instanceof Map) || !(((Map<?, ?>) o).get("id") instanceof String)) {
                continue;
            }
            Map<?, ?> m = (Map<?, ?>) o;
            List<String> aliases = new ArrayList<>();
            if (m.get("aliases") instanceof List) {
                for (Object a : (List<?>) m.get("aliases")) {
                    if (a instanceof String) {
                        aliases.add((String) a);
                    }
                }
            }
            out.add(new ModelInfo((String) m.get("id"), aliases));
        }
        return out;
    }

    /** The server properties ({@code /props}, routed with {@code ?model=} when a model is named). */
    Props props(String model) throws IOException, InterruptedException {
        JsonView body = parse(get("/props", model), "/props");
        boolean vision = false;
        boolean audio = false;
        boolean video = false;
        Object raw = body.raw("modalities");
        if (raw instanceof Map) {
            Map<?, ?> modalities = (Map<?, ?>) raw;
            vision = Boolean.TRUE.equals(modalities.get("vision"));
            audio = Boolean.TRUE.equals(modalities.get("audio"));
            video = Boolean.TRUE.equals(modalities.get("video"));
        }
        return new Props(text(body.raw("model_alias")), text(body.raw("model_path")), vision, audio, video);
    }

    /**
     * Streams one chat completion of {@code body} (which must ask for {@code stream: true}) and routes the deltas.
     * Returns the empty string on success or cancellation, otherwise the error text of the failed request or of
     * an {@code error} chunk.
     */
    String streamChat(Map<String, Object> body, Deltas deltas, BooleanSupplier cancelled)
            throws InterruptedException {
        return streamPost(chatPath, body, deltas, payload -> chatChunk(payload, deltas), cancelled);
    }

    /**
     * Streams one text completion of {@code body} (which must ask for {@code stream: true}); the first choice's
     * text pieces arrive as {@link Deltas#content(String)}. Returns the empty string on success or cancellation,
     * otherwise the error text of the failed request or of an {@code error} chunk.
     */
    String streamCompletion(Map<String, Object> body, Deltas deltas, BooleanSupplier cancelled)
            throws InterruptedException {
        return streamPost(completionsPath, body, deltas, payload -> completionChunk(payload, deltas), cancelled);
    }

    /**
     * Posts {@code body} to {@code path} asking for a stream, and feeds every payload to {@code chunks}. The
     * handler is told the stream started before its first payload, so it can time the answer from the response
     * headers as the OpenAI clients do.
     */
    private String streamPost(String path, Map<String, Object> body, Deltas deltas, Chunks chunks,
                              BooleanSupplier cancelled) throws InterruptedException {
        String json = Json.toString(body, Json.Style.API);
        HttpRequest request = builder(path, null)
                .header("Content-Type", "application/json")
                .header("Accept", "text/event-stream")
                .POST(HttpRequest.BodyPublishers.ofString(json, StandardCharsets.UTF_8))
                .build();
        HttpResponse<InputStream> response;
        try {
            response = http.send(request, HttpResponse.BodyHandlers.ofInputStream());
        } catch (IOException e) {
            return "failed to connect to " + base + ": " + describe(e);
        }
        try {
            if (response.statusCode() < 200 || response.statusCode() >= 300) {
                String errorBody = readBody(response.body());
                return errorText(errorBody, "request failed with status " + response.statusCode());
            }
            deltas.started();
            return stream(response.body(), chunks, cancelled);
        } catch (IOException e) {
            return cancelled.getAsBoolean() ? "" : "the stream from " + base + " failed: " + describe(e);
        }
    }

    /** The handler of one SSE payload: the error text of an {@code error} event, or null. */
    private interface Chunks {
        String handle(String payload);
    }

    /** Reads one SSE body, feeding every {@code data:} payload to {@code chunks} until the stream or a cancel ends it. */
    private String stream(InputStream body, Chunks chunks, BooleanSupplier cancelled) throws IOException {
        Thread closer = closeOnCancel(body, cancelled);
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(body, StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null && !cancelled.getAsBoolean()) {
                if (!line.startsWith("data: ")) {
                    continue;
                }
                String payload = line.substring(6);
                if (payload.equals("[DONE]")) {
                    continue;
                }
                String error = chunks.handle(payload);
                if (error != null) {
                    return error;
                }
            }
            return "";
        } finally {
            closer.interrupt();
        }
    }

    /** A daemon that closes {@code body} once {@code cancelled} turns true, unblocking the reader. */
    private static Thread closeOnCancel(InputStream body, BooleanSupplier cancelled) {
        Thread closer = new Thread(() -> {
            try {
                while (!cancelled.getAsBoolean()) {
                    Thread.sleep(CANCEL_POLL_MILLIS);
                }
                body.close();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            } catch (IOException e) {
                // the reader sees the closed stream and reports the cancellation
            }
        }, "chat-client-cancel");
        closer.setDaemon(true);
        closer.start();
        return closer;
    }

    /**
     * Routes one chat SSE payload: an {@code error} object ends the stream with its message; {@code timings},
     * {@code usage} and the first choice's reasoning (vLLM's {@code reasoning}, llama.cpp's
     * {@code reasoning_content}) and {@code content} deltas feed the handler. Payloads that are not JSON objects
     * are skipped, as the reference skips them.
     */
    private static String chatChunk(String payload, Deltas deltas) {
        Map<?, ?> chunk = chunk(payload);
        if (chunk == null) {
            return null;
        }
        String failure = errorOf(chunk);
        if (failure != null) {
            return failure;
        }
        if (chunk.get("timings") instanceof Map) {
            Map<?, ?> t = (Map<?, ?>) chunk.get("timings");
            deltas.timings(number(t.get("prompt_per_second")), number(t.get("predicted_per_second")));
        }
        reportUsage(chunk, deltas);
        Object choice = firstChoice(chunk);
        if (!(choice instanceof Map) || !(((Map<?, ?>) choice).get("delta") instanceof Map)) {
            return null;
        }
        Map<?, ?> d = (Map<?, ?>) ((Map<?, ?>) choice).get("delta");
        String reasoning = reasoningText(d);
        if (reasoning != null) {
            deltas.reasoning(reasoning);
        }
        Object content = d.get("content");
        if (content instanceof String && !((String) content).isEmpty()) {
            deltas.content((String) content);
        }
        return null;
    }

    /**
     * Routes one completion SSE payload: an {@code error} object ends the stream with its message; {@code usage}
     * and the first choice's non-empty {@code text} feed the handler.
     */
    private static String completionChunk(String payload, Deltas deltas) {
        Map<?, ?> chunk = chunk(payload);
        if (chunk == null) {
            return null;
        }
        String failure = errorOf(chunk);
        if (failure != null) {
            return failure;
        }
        reportUsage(chunk, deltas);
        Object choice = firstChoice(chunk);
        if (choice instanceof Map) {
            Object text = ((Map<?, ?>) choice).get("text");
            if (text instanceof String && !((String) text).isEmpty()) {
                deltas.content((String) text);
            }
        }
        return null;
    }

    /** One parsed payload as a JSON object, or null when it is not one. */
    private static Map<?, ?> chunk(String payload) {
        try {
            Object parsed = Json.parse(payload, Json.Limits.UNTRUSTED);
            return parsed instanceof Map ? (Map<?, ?>) parsed : null;
        } catch (JsonException e) {
            return null;
        }
    }

    /** The error text of a chunk carrying an {@code error}, or null for a chunk that carries none. */
    private static String errorOf(Map<?, ?> chunk) {
        if (!chunk.containsKey("error")) {
            return null;
        }
        String message = errorMessage(chunk.get("error"));
        return message == null ? "the server reported an error" : message;
    }

    /** The chunk's first choice, or null when it lists none. */
    private static Object firstChoice(Map<?, ?> chunk) {
        if (!(chunk.get("choices") instanceof List) || ((List<?>) chunk.get("choices")).isEmpty()) {
            return null;
        }
        return ((List<?>) chunk.get("choices")).get(0);
    }

    /** Reports the chunk's {@code usage.completion_tokens} to the handler, when it carries one. */
    private static void reportUsage(Map<?, ?> chunk, Deltas deltas) {
        if (!(chunk.get("usage") instanceof Map)) {
            return;
        }
        Object completion = ((Map<?, ?>) chunk.get("usage")).get("completion_tokens");
        if (completion instanceof Number) {
            deltas.usage(((Number) completion).intValue());
        }
    }

    /** The reasoning text of a delta: vLLM's {@code reasoning}, else llama.cpp's {@code reasoning_content}. */
    private static String reasoningText(Map<?, ?> delta) {
        Object reasoning = delta.get("reasoning");
        if (!(reasoning instanceof String) || ((String) reasoning).isEmpty()) {
            reasoning = delta.get("reasoning_content");
        }
        return reasoning instanceof String && !((String) reasoning).isEmpty() ? (String) reasoning : null;
    }

    /** The body of a GET, or an {@code IOException} naming the path, the status and the body. */
    private String get(String path, String model) throws IOException, InterruptedException {
        HttpResponse<String> response = http.send(builder(path, model).GET().build(),
                HttpResponse.BodyHandlers.ofString());
        if (response.statusCode() < 200 || response.statusCode() >= 300) {
            throw new IOException("GET " + path + " failed with status " + response.statusCode() + ": "
                    + response.body());
        }
        return response.body();
    }

    private HttpRequest.Builder builder(String path, String model) {
        String query = model == null || model.isEmpty() ? ""
                : "?model=" + URLEncoder.encode(model, StandardCharsets.UTF_8);
        HttpRequest.Builder b = HttpRequest.newBuilder(URI.create(base + path + query)).timeout(READ_TIMEOUT);
        if (apiKey != null) {
            b.header("Authorization", "Bearer " + apiKey);
        }
        return b;
    }

    /** Whether a health body is a JSON object whose {@code status} is {@code error}. */
    private static boolean reportsError(String body) {
        try {
            Object parsed = Json.parse(body, Json.Limits.UNTRUSTED);
            return parsed instanceof Map && "error".equals(((Map<?, ?>) parsed).get("status"));
        } catch (JsonException e) {
            return false;
        }
    }

    private JsonView parse(String text, String source) throws IOException {
        try {
            return JsonView.of(Json.parse(text, Json.Limits.UNTRUSTED), source);
        } catch (JsonException e) {
            throw new IOException(source + " is not valid JSON (is " + base
                    + " an OpenAI-compatible server?): " + e.getMessage(), e);
        }
    }

    /** The message of a JSON error body ({@code {"error": {"message": ...}}}), or the body itself. */
    private static String errorText(String body, String fallback) {
        if (body == null || body.isEmpty()) {
            return fallback;
        }
        try {
            Object parsed = Json.parse(body, Json.Limits.UNTRUSTED);
            if (parsed instanceof Map) {
                String message = errorMessage(((Map<?, ?>) parsed).get("error"));
                if (message != null) {
                    return message;
                }
            }
        } catch (JsonException e) {
            return body;
        }
        return body;
    }

    /** The {@code message} of an error object, the error itself when it is a string, or null. */
    private static String errorMessage(Object error) {
        if (error instanceof Map && ((Map<?, ?>) error).get("message") instanceof String) {
            return (String) ((Map<?, ?>) error).get("message");
        }
        return error instanceof String ? (String) error : null;
    }

    private static String readBody(InputStream in) throws IOException {
        try (InputStream body = in) {
            byte[] data = body.readNBytes(MAX_ERROR_BODY);
            return new String(data, StandardCharsets.UTF_8);
        }
    }

    private static double number(Object value) {
        return value instanceof Number ? ((Number) value).doubleValue() : 0;
    }

    private static String text(Object value) {
        return value instanceof String ? (String) value : "";
    }

    private static String describe(IOException e) {
        return e.getMessage() == null ? e.getClass().getSimpleName() : e.getMessage();
    }
}

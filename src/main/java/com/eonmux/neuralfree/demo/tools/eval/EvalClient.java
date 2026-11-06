package com.eonmux.neuralfree.demo.tools.eval;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Executors;
import java.util.function.Consumer;

import com.eonmux.neuralfree.utils.json.Json;
import com.eonmux.neuralfree.utils.json.JsonException;

/**
 * The chat-completions client of an evaluation run: one non-streamed {@code /v1/chat/completions}
 * request per task, resent on transport errors, HTTP 5xx and the retryable 4xx statuses, invalid
 * JSON, and replies with no usable content. The wait before each resend doubles from
 * {@code --retry-delay}. {@code /v1/models} is the reachability check; both requests carry the
 * server's bearer token.
 */
public final class EvalClient {

    /** The statuses the reference resends even though they are below 500. */
    private static final java.util.Set<Integer> RETRYABLE_HTTP =
            new java.util.HashSet<>(java.util.Arrays.asList(408, 409, 425, 429));

    private final EvalServer server;
    private final String model;
    private final int retries;
    private final double retryDelay;
    private final Consumer<String> log;
    private final HttpClient http;

    /**
     * @param model      the payload's {@code model}; {@code llama} when empty, as the reference sends
     * @param retries    resends after the first attempt, at least 0
     * @param retryDelay seconds before the first resend, doubled on each further one
     * @param log        the retry lines' sink, null for none
     */
    public EvalClient(EvalServer server, String model, int retries, double retryDelay,
                      Consumer<String> log) {
        if (retries < 0) {
            throw new IllegalArgumentException("--retries must be at least 0, got " + retries);
        }
        if (retryDelay < 0) {
            throw new IllegalArgumentException("--retry-delay must be at least 0, got " + retryDelay);
        }
        this.server = server;
        this.model = model;
        this.retries = retries;
        this.retryDelay = retryDelay;
        this.log = log;
        this.http = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(30))
                .executor(Executors.newCachedThreadPool(r -> {
                    Thread t = new Thread(r, "eval-client");
                    t.setDaemon(true);
                    return t;
                }))
                .build();
    }

    /** The server this client drives. */
    public EvalServer server() {
        return server;
    }

    /**
     * The served models' ids from {@code /v1/models}.
     *
     * @throws EvalServerException when the endpoint refuses or answers no model
     */
    public List<String> models() throws EvalServerException {
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(server.modelsUrl()))
                .GET()
                .headers(headerArray(null))
                .build();
        HttpResponse<String> response;
        try {
            response = http.send(request, HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8));
        } catch (IOException e) {
            throw new EvalServerException(typeName(e) + ": " + e.getMessage());
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new EvalServerException("the model listing was interrupted");
        }
        if (response.statusCode() < 200 || response.statusCode() >= 300) {
            throw new EvalServerException("HTTP " + response.statusCode() + ": "
                    + shortBody(response.body()));
        }
        Object parsed;
        try {
            parsed = Json.parse(response.body(), Json.Limits.UNTRUSTED);
        } catch (JsonException e) {
            throw new EvalServerException("the model listing is not JSON: " + e.getMessage());
        }
        List<String> models = new ArrayList<>();
        if (parsed instanceof Map && ((Map<?, ?>) parsed).get("data") instanceof List) {
            for (Object entry : (List<?>) ((Map<?, ?>) parsed).get("data")) {
                if (entry instanceof Map) {
                    Object id = ((Map<?, ?>) entry).get("id");
                    if (id != null) {
                        models.add(String.valueOf(id));
                    }
                }
            }
        }
        return models;
    }

    /**
     * One task's completion, resent until it answers or its retries run out.
     *
     * @param sessionId the opencode session id bound to the task, null for other servers
     * @throws EvalServerException naming the last failure after every attempt
     */
    public EvalChat generate(String prompt, EvalSampling sampling, String sessionId)
            throws EvalServerException {
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("model", model == null || model.isEmpty() ? "llama" : model);
        Map<String, Object> message = new LinkedHashMap<>();
        message.put("role", "user");
        message.put("content", prompt);
        List<Map<String, Object>> messages = new ArrayList<>();
        messages.add(message);
        body.put("messages", messages);
        sampling.applyTo(body);

        Failure last = null;
        int attempts = 0;
        for (int attempt = 0; attempt <= retries; attempt++) {
            if (attempt > 0) {
                long waitMillis = retryWaitMillis(attempt);
                report(server.url() + " " + last.reason() + "; retry " + attempt + "/" + retries
                        + " in " + (waitMillis / 1000.0) + "s");
                sleep(waitMillis);
            }
            attempts++;
            try {
                return chat(body, sessionId);
            } catch (Failure failure) {
                last = failure;
                if (!failure.retryable) {
                    break;
                }
            }
        }
        throw new EvalServerException((last == null ? "the request failed" : last.reason())
                + " (after " + attempts + " attempt" + (attempts == 1 ? "" : "s") + ")");
    }

    /**
     * One grading request: a single POST, no retries, for the LLM grader's answer extraction.
     *
     * @throws EvalServerException when the request fails or the reply is unusable
     */
    public Map<String, Object> chatOnce(Map<String, Object> body, String sessionId)
            throws EvalServerException {
        try {
            return send(body, sessionId);
        } catch (Failure failure) {
            throw new EvalServerException(failure.reason());
        }
    }

    /** One chat completion after {@code attempt} failures, read into its answer. */
    private EvalChat chat(Map<String, Object> body, String sessionId) throws Failure {
        Map<String, Object> reply = send(body, sessionId);
        Object choices = reply.get("choices");
        Map<?, ?> choice = first(choices);
        if (choice == null) {
            throw new Failure("reply has no choices", true);
        }
        Object messageValue = choice.get("message");
        Map<?, ?> message = messageValue instanceof Map ? (Map<?, ?>) messageValue : null;
        String content = message == null ? null : stringValue(message.get("content"));
        String reasoning = message == null ? null : stringValue(message.get("reasoning_content"));
        Object finishReason = choice.get("finish_reason");
        Long tokens = 0L;
        Object usageValue = reply.get("usage");
        if (usageValue instanceof Map) {
            Object completion = ((Map<?, ?>) usageValue).get("completion_tokens");
            if (completion instanceof Number) {
                tokens = ((Number) completion).longValue();
            }
        }
        Double tpsGen = null;
        Double tGenMs = null;
        Object timingsValue = reply.get("timings");
        if (timingsValue instanceof Map) {
            Object tps = ((Map<?, ?>) timingsValue).get("predicted_per_second");
            Object ms = ((Map<?, ?>) timingsValue).get("predicted_ms");
            if (tps instanceof Number) {
                tpsGen = ((Number) tps).doubleValue();
            }
            if (ms instanceof Number) {
                tGenMs = ((Number) ms).doubleValue();
            }
        }
        return new EvalChat(content, reasoning,
                finishReason == null ? "stop" : String.valueOf(finishReason), tokens, tpsGen, tGenMs);
    }

    /** One POST of a chat body, its reply parsed; every failure carries whether a resend helps. */
    private Map<String, Object> send(Map<String, Object> body, String sessionId) throws Failure {
        HttpRequest request;
        try {
            request = HttpRequest.newBuilder()
                    .uri(URI.create(server.chatUrl()))
                    .headers(headerArray(sessionId))
                    .POST(HttpRequest.BodyPublishers.ofString(
                            Json.toString(body, Json.Style.API), StandardCharsets.UTF_8))
                    .build();
        } catch (IllegalArgumentException e) {
            throw new Failure("the request cannot be built: " + e.getMessage(), false);
        }
        HttpResponse<String> response;
        try {
            response = http.send(request, HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8));
        } catch (IOException e) {
            throw new Failure(typeName(e) + ": " + e.getMessage(), true);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new Failure("the request was interrupted", false);
        }
        if (response.statusCode() >= 400) {
            boolean retryable = response.statusCode() >= 500
                    || RETRYABLE_HTTP.contains(response.statusCode());
            throw new Failure("HTTP " + response.statusCode() + ": " + shortBody(response.body()),
                    retryable);
        }
        Object parsed;
        try {
            parsed = Json.parse(response.body(), Json.Limits.UNTRUSTED);
        } catch (JsonException e) {
            throw new Failure("invalid JSON: " + shortBody(response.body()), true);
        }
        if (!(parsed instanceof Map)) {
            throw new Failure("reply is not a JSON object", true);
        }
        @SuppressWarnings("unchecked")
        Map<String, Object> reply = (Map<String, Object>) parsed;
        String nullReason = nullReplyReason(reply);
        if (nullReason != null) {
            throw new Failure(nullReason, true);
        }
        return reply;
    }

    /** Why a reply carries no usable answer, or null when it does. */
    private static String nullReplyReason(Map<String, Object> reply) {
        Object choicesValue = reply.get("choices");
        if (!(choicesValue instanceof List) || ((List<?>) choicesValue).isEmpty()) {
            return "reply has no choices";
        }
        Object first = ((List<?>) choicesValue).get(0);
        if (!(first instanceof Map)) {
            return "reply has no message";
        }
        Object messageValue = ((Map<?, ?>) first).get("message");
        if (!(messageValue instanceof Map)) {
            return "reply has no message";
        }
        Map<?, ?> message = (Map<?, ?>) messageValue;
        if (truthy(message.get("content")) || truthy(message.get("tool_calls"))
                || truthy(message.get("reasoning_content"))) {
            return null;
        }
        return "reply is empty";
    }

    /** The first choice of a reply, or null. */
    private static Map<?, ?> first(Object choices) {
        if (!(choices instanceof List) || ((List<?>) choices).isEmpty()) {
            return null;
        }
        Object first = ((List<?>) choices).get(0);
        return first instanceof Map ? (Map<?, ?>) first : null;
    }

    private static String stringValue(Object value) {
        return value instanceof String ? (String) value : null;
    }

    /** Python truthiness of a parsed JSON value. */
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

    /** The headers as the array {@link HttpRequest.Builder#headers} takes. */
    private String[] headerArray(String sessionId) {
        Map<String, String> headers = server.headers(sessionId);
        String[] out = new String[headers.size() * 2];
        int i = 0;
        for (Map.Entry<String, String> header : headers.entrySet()) {
            out[i++] = header.getKey();
            out[i++] = header.getValue();
        }
        return out;
    }

    private long retryWaitMillis(int attempt) {
        double seconds = retryDelay * Math.pow(2, attempt - 1);
        return Math.round(seconds * 1000.0);
    }

    private static void sleep(long millis) {
        if (millis <= 0) {
            return;
        }
        try {
            Thread.sleep(millis);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("the retry wait was interrupted", e);
        }
    }

    private void report(String line) {
        if (log != null) {
            log.accept(line);
        }
    }

    private static String typeName(Exception e) {
        return e.getClass().getSimpleName();
    }

    /** The first 300 characters of an error body, single-lined. */
    private static String shortBody(String body) {
        String text = body == null ? "" : body.replace('\n', ' ');
        return text.isEmpty() ? "(no body)" : text.substring(0, Math.min(300, text.length()));
    }

    /** A failed attempt and whether the reference would resend it. */
    private static final class Failure extends Exception {

        private static final long serialVersionUID = 1L;

        private final boolean retryable;

        Failure(String reason, boolean retryable) {
            super(reason);
            this.retryable = retryable;
        }

        private String reason() {
            return getMessage();
        }
    }
}

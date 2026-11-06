package com.eonmux.neuralfree.demo.tools.eval;

import com.eonmux.neuralfree.utils.json.Json;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;

import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

/**
 * A minimal non-streamed OpenAI-compatible server for the evaluation tests: the model listing and
 * the chat completion route, with scripted failures, empty replies, finish reasons and a grader
 * reply distinguished by the request's system message. It records every body, bearer token and
 * session header.
 */
final class FakeEvalServer implements AutoCloseable {

    private final HttpServer server;
    private final List<Map<String, Object>> bodies = Collections.synchronizedList(new ArrayList<>());
    private final List<String> authorizations =
            Collections.synchronizedList(new ArrayList<>());
    private final List<String> sessions = Collections.synchronizedList(new ArrayList<>());

    volatile String content = "\\boxed{42}";
    volatile String graderContent = "42";
    volatile String finishReason = "stop";
    volatile int failuresBeforeSuccess;
    volatile int emptyBeforeSuccess;
    volatile int status = 200;
    volatile long delayMillis;
    volatile double tps = 12.5;
    volatile long genMs = 560;
    volatile boolean sendUsage = true;
    volatile CountDownLatch lobby;

    FakeEvalServer() throws IOException {
        server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext("/v1/models", this::models);
        server.createContext("/v1/chat/completions", this::chat);
        server.start();
    }

    String baseUrl() {
        return "http://127.0.0.1:" + server.getAddress().getPort();
    }

    int chatCalls() {
        return bodies.size();
    }

    List<Map<String, Object>> bodies() {
        return new ArrayList<>(bodies);
    }

    List<String> authorizations() {
        return new ArrayList<>(authorizations);
    }

    List<String> sessions() {
        return new ArrayList<>(sessions);
    }

    @Override
    public void close() {
        server.stop(0);
    }

    private void models(HttpExchange exchange) throws IOException {
        Map<String, Object> model = new LinkedHashMap<>();
        model.put("id", "test-model");
        List<Object> data = new ArrayList<>();
        data.add(model);
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("data", data);
        reply(exchange, 200, Json.toString(body, Json.Style.API));
    }

    private void chat(HttpExchange exchange) throws IOException {
        String raw = new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8);
        Map<?, ?> body = (Map<?, ?>) Json.parse(raw, Json.Limits.UNTRUSTED);
        @SuppressWarnings("unchecked")
        Map<String, Object> recorded = (Map<String, Object>) body;
        bodies.add(recorded);
        String authorization = exchange.getRequestHeaders().getFirst("Authorization");
        if (authorization != null) {
            authorizations.add(authorization);
        }
        String session = exchange.getRequestHeaders().getFirst("x-opencode-session");
        if (session != null) {
            sessions.add(session);
        }
        CountDownLatch gate = lobby;
        if (gate != null) {
            gate.countDown();
            try {
                gate.await(5, TimeUnit.SECONDS);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
        if (delayMillis > 0) {
            try {
                Thread.sleep(delayMillis);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
        if (failuresBeforeSuccess > 0) {
            failuresBeforeSuccess--;
            reply(exchange, 500, "{\"error\":{\"message\":\"scripted failure\"}}");
            return;
        }
        if (status != 200) {
            reply(exchange, status, "{\"error\":{\"message\":\"refused\"}}");
            return;
        }
        boolean grading = isGrading(body);
        Map<String, Object> message = new LinkedHashMap<>();
        if (emptyBeforeSuccess > 0) {
            emptyBeforeSuccess--;
        } else {
            message.put("content", grading ? graderContent : content);
        }
        Map<String, Object> choice = new LinkedHashMap<>();
        choice.put("message", message);
        choice.put("finish_reason", finishReason);
        List<Object> choices = new ArrayList<>();
        choices.add(choice);
        Map<String, Object> usage = new LinkedHashMap<>();
        usage.put("completion_tokens", 7);
        Map<String, Object> timings = new LinkedHashMap<>();
        timings.put("predicted_per_second", tps);
        timings.put("predicted_ms", genMs);
        Map<String, Object> reply = new LinkedHashMap<>();
        reply.put("choices", choices);
        if (sendUsage) {
            reply.put("usage", usage);
            reply.put("timings", timings);
        }
        reply(exchange, 200, Json.toString(reply, Json.Style.API));
    }

    /** Whether a request carries a system message, which only the LLM grader's request does. */
    private static boolean isGrading(Map<?, ?> body) {
        Object messages = body.get("messages");
        if (!(messages instanceof List)) {
            return false;
        }
        for (Object entry : (List<?>) messages) {
            if (entry instanceof Map && "system".equals(((Map<?, ?>) entry).get("role"))) {
                return true;
            }
        }
        return false;
    }

    private static void reply(HttpExchange exchange, int code, String body) throws IOException {
        byte[] bytes = body.getBytes(StandardCharsets.UTF_8);
        exchange.getResponseHeaders().add("Content-Type", "application/json");
        exchange.sendResponseHeaders(code, bytes.length);
        try (OutputStream out = exchange.getResponseBody()) {
            out.write(bytes);
        }
    }
}

package com.eonmux.neuralfree.demo.tools.serve;

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

/**
 * A minimal OpenAI-compatible server for the harness tests: the discovery routes ({@code /v1/models},
 * {@code /tokenizer_info}, {@code /tokenize}), the two streamed generation routes with a usage chunk,
 * and {@code /health}. It records every request body and the headers the benchmark sent.
 */
final class FakeOpenAiServer implements AutoCloseable {

    private final HttpServer server;
    private final List<Map<String, Object>> bodies = Collections.synchronizedList(new ArrayList<>());
    private final List<String> requestIds = Collections.synchronizedList(new ArrayList<>());

    volatile String modelId = "served-model";
    volatile String modelRoot = "/models/base.gguf";
    volatile int vocabSize = 1000;
    volatile int tokensPerResponse = 2;
    volatile long tokenDelayMillis = 0;
    volatile boolean sendUsage = true;
    volatile boolean openingRoleChunk = true;
    volatile int generationStatus = 200;
    volatile int generationCalls;
    volatile int tokenizeCalls;
    volatile String metricsText;
    volatile long metricsDraftsPerScrape;
    volatile long metricsDraftTokensPerScrape;
    volatile long metricsAcceptedTokensPerScrape;
    volatile long metricsPerPositionAcceptedPerScrape;
    private final java.util.concurrent.atomic.AtomicInteger metricsScrapes =
            new java.util.concurrent.atomic.AtomicInteger();
    volatile int rerankPromptTokens = 7;
    volatile int profileStatus = 200;
    volatile int profileStarts;
    volatile int profileStops;

    FakeOpenAiServer() throws IOException {
        server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext("/health", ex -> reply(ex, 200, "{\"status\":\"ok\"}"));
        server.createContext("/v1/models", ex -> reply(ex, 200,
                "{\"data\":[{\"id\":\"" + modelId + "\",\"root\":\"" + modelRoot + "\"}]}"));
        server.createContext("/tokenizer_info", ex -> reply(ex, 200,
                "{\"vocab_size\":" + vocabSize + ",\"bos_token_id\":1,\"eos_token_id\":2,"
                        + "\"pad_token_id\":-1,\"unk_token_id\":0,\"stop_token_ids\":[2,3]}"));
        server.createContext("/tokenize", this::tokenize);
        server.createContext("/detokenize", this::detokenize);
        server.createContext("/metrics", this::metrics);
        server.createContext("/v1/rerank", this::rerank);
        server.createContext("/rerank", this::rerank);
        server.createContext("/v1/audio/transcriptions", this::transcribe);
        server.createContext("/v1/completions", ex -> generate(ex, false));
        server.createContext("/v1/chat/completions", ex -> generate(ex, true));
        server.createContext("/start_profile", ex -> {
            profileStarts++;
            reply(ex, profileStatus, profileStatus == 200 ? "" : "{\"error\":\"not configured\"}");
        });
        server.createContext("/stop_profile", ex -> {
            profileStops++;
            reply(ex, profileStatus, profileStatus == 200 ? "" : "{\"error\":\"not running\"}");
        });
        server.start();
    }

    private void detokenize(HttpExchange ex) throws IOException {
        Map<?, ?> body = (Map<?, ?>) Json.parse(read(ex), Json.Limits.UNTRUSTED);
        Object tokens = body.get("tokens");
        int count = tokens instanceof List ? ((List<?>) tokens).size() : 0;
        StringBuilder text = new StringBuilder();
        for (int i = 0; i < count; i++) {
            text.append("<t").append(i).append('>');
        }
        reply(ex, 200, "{\"content\":\"" + text + "\",\"prompt\":\"" + text + "\"}");
    }

    private void metrics(HttpExchange ex) throws IOException {
        String scripted = metricsText;
        if (scripted != null) {
            ex.getResponseHeaders().add("Content-Type", "text/plain");
            byte[] bytes = scripted.getBytes(StandardCharsets.UTF_8);
            ex.sendResponseHeaders(200, bytes.length);
            try (OutputStream out = ex.getResponseBody()) {
                out.write(bytes);
            }
            return;
        }
        int scrape = metricsScrapes.getAndIncrement();
        long drafts = metricsDraftsPerScrape * scrape;
        long draftTokens = metricsDraftTokensPerScrape * scrape;
        long accepted = metricsAcceptedTokensPerScrape * scrape;
        long perPosition = metricsPerPositionAcceptedPerScrape * scrape;
        String text = "vllm:spec_decode_num_drafts_total " + drafts + "\n"
                + "vllm:spec_decode_num_draft_tokens_total " + draftTokens + "\n"
                + "vllm:spec_decode_num_accepted_tokens_total " + accepted + "\n"
                + "vllm:spec_decode_num_accepted_tokens_per_pos_total{position=\"0\"} " + perPosition + "\n"
                + "vllm:spec_decode_num_accepted_tokens_per_pos_total{position=\"1\"} " + perPosition + "\n";
        ex.getResponseHeaders().add("Content-Type", "text/plain");
        byte[] bytes = text.getBytes(StandardCharsets.UTF_8);
        ex.sendResponseHeaders(200, bytes.length);
        try (OutputStream out = ex.getResponseBody()) {
            out.write(bytes);
        }
    }

    private void rerank(HttpExchange ex) throws IOException {
        String raw = read(ex);
        Map<?, ?> body = (Map<?, ?>) Json.parse(raw, Json.Limits.UNTRUSTED);
        bodies.add(cast(body));
        String requestId = ex.getRequestHeaders().getFirst("X-Request-Id");
        if (requestId != null) {
            requestIds.add(requestId);
        }
        reply(ex, 200, "{\"results\":[],\"usage\":{\"prompt_tokens\":" + rerankPromptTokens + "}}");
    }

    private void transcribe(HttpExchange ex) throws IOException {
        generationCalls++;
        read(ex);
        ex.getResponseHeaders().add("Content-Type", "text/event-stream");
        ex.sendResponseHeaders(200, 0);
        try (OutputStream out = ex.getResponseBody()) {
            for (int i = 0; i < tokensPerResponse; i++) {
                event(out, "{\"choices\":[{\"index\":0,\"delta\":{\"content\":\"t" + i + "\"}}]}");
            }
            event(out, "{\"choices\":[],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":"
                    + tokensPerResponse + "}}");
            event(out, "[DONE]");
        }
    }

    String baseUrl() {
        return "http://127.0.0.1:" + server.getAddress().getPort();
    }

    List<Map<String, Object>> bodies() {
        return new ArrayList<>(bodies);
    }

    List<String> requestIds() {
        return new ArrayList<>(requestIds);
    }

    @Override
    public void close() {
        server.stop(0);
    }

    private void tokenize(HttpExchange ex) throws IOException {
        tokenizeCalls++;
        Map<?, ?> body = (Map<?, ?>) Json.parse(read(ex), Json.Limits.UNTRUSTED);
        Object prompt = body.get("prompt");
        int count = prompt instanceof String ? ((String) prompt).length() : 0;
        StringBuilder tokens = new StringBuilder("[");
        for (int i = 0; i < count; i++) {
            if (i > 0) {
                tokens.append(',');
            }
            tokens.append(100 + i);
        }
        tokens.append(']');
        reply(ex, 200, "{\"tokens\":" + tokens + ",\"count\":" + count + "}");
    }

    private void generate(HttpExchange ex, boolean chat) throws IOException {
        generationCalls++;
        String raw = read(ex);
        Map<?, ?> body = (Map<?, ?>) Json.parse(raw, Json.Limits.UNTRUSTED);
        bodies.add(cast(body));
        String requestId = ex.getRequestHeaders().getFirst("X-Request-Id");
        if (requestId != null) {
            requestIds.add(requestId);
        }
        if (generationStatus != 200) {
            reply(ex, generationStatus, "{\"error\":{\"message\":\"refused\"}}");
            return;
        }
        ex.getResponseHeaders().add("Content-Type", "text/event-stream");
        ex.sendResponseHeaders(200, 0);
        try (OutputStream out = ex.getResponseBody()) {
            if (chat && openingRoleChunk) {
                event(out, "{\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"}}]}");
            }
            for (int i = 0; i < tokensPerResponse; i++) {
                if (tokenDelayMillis > 0) {
                    try {
                        Thread.sleep(tokenDelayMillis);
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                        return;
                    }
                }
                if (chat) {
                    event(out, "{\"choices\":[{\"index\":0,\"delta\":{\"content\":\"t" + i + "\"}}]}");
                } else {
                    event(out, "{\"choices\":[{\"index\":0,\"text\":\"t" + i + "\"}]}");
                }
            }
            if (sendUsage) {
                event(out, "{\"choices\":[],\"usage\":{\"prompt_tokens\":" + promptLen(body, chat)
                        + ",\"completion_tokens\":" + tokensPerResponse + "}}");
            }
            event(out, "[DONE]");
        }
    }

    private static int promptLen(Map<?, ?> body, boolean chat) {
        if (chat) {
            Object messages = body.get("messages");
            if (messages instanceof List && !((List<?>) messages).isEmpty()) {
                Object content = ((Map<?, ?>) ((List<?>) messages).get(0)).get("content");
                return content instanceof String ? ((String) content).length() : 0;
            }
            return 0;
        }
        Object prompt = body.get("prompt");
        if (prompt instanceof List) {
            return ((List<?>) prompt).size();
        }
        return prompt instanceof String ? ((String) prompt).length() : 0;
    }

    private static void event(OutputStream out, String data) throws IOException {
        out.write(("data: " + data + "\n\n").getBytes(StandardCharsets.UTF_8));
        out.flush();
    }

    private static String read(HttpExchange ex) throws IOException {
        return new String(ex.getRequestBody().readAllBytes(), StandardCharsets.UTF_8);
    }

    private static void reply(HttpExchange ex, int status, String body) throws IOException {
        byte[] bytes = body.getBytes(StandardCharsets.UTF_8);
        ex.getResponseHeaders().add("Content-Type", "application/json");
        ex.sendResponseHeaders(status, bytes.length);
        try (OutputStream out = ex.getResponseBody()) {
            out.write(bytes);
        }
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> cast(Map<?, ?> body) {
        return (Map<String, Object>) body;
    }

    /** A body as a fresh map, for assertions. */
    static Map<String, Object> bodyOf(Map<String, Object> body) {
        return new LinkedHashMap<>(body);
    }
}

package com.eonmux.neuralfree.demo.tools.serve;

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

import javax.net.ssl.SSLContext;
import javax.net.ssl.TrustManager;
import javax.net.ssl.X509TrustManager;
import java.security.cert.X509Certificate;

import com.eonmux.neuralfree.gguf.chat.ChatFormatter;
import com.eonmux.neuralfree.gguf.chat.ChatRequest;
import com.eonmux.neuralfree.gguf.chat.ChatTemplateSource;
import com.eonmux.neuralfree.utils.json.Json;
import com.eonmux.neuralfree.utils.json.JsonException;

/**
 * The server facts the benchmarks discover over HTTP before they load: the first served model
 * ({@code GET /v1/models}), the tokenizer's vocabulary and special ids ({@code GET
 * /tokenizer_info}) and the token count of a text prompt ({@code POST /tokenize}, the
 * {@code prompt} form whose count vLLM's local tokenizer would give). Every call names its URL
 * when it cannot reach the server or the answer's shape is wrong, so a benchmark against a
 * not-yet-started server fails with its cause.
 */
public final class BenchEndpoints {

    /** The model card the server reports first. */
    public static final class ServedModel {

        public final String id;
        public final String root;

        ServedModel(String id, String root) {
            this.id = id;
            this.root = root;
        }
    }

    /** The tokenizer facts a synthetic dataset draws from. */
    public static final class TokenizerInfo {

        public final int vocabSize;
        public final int[] skipIds;

        TokenizerInfo(int vocabSize, int[] skipIds) {
            this.vocabSize = vocabSize;
            this.skipIds = skipIds;
        }
    }

    private BenchEndpoints() {
    }

    /** The first model of {@code GET base/v1/models}, the id the requests name. */
    public static ServedModel firstModel(String baseUrl, Map<String, String> headers, boolean insecure)
            throws IOException {
        Map<?, ?> body = getJson(baseUrl + "/v1/models", headers, insecure);
        Object data = body.get("data");
        if (!(data instanceof List) || ((List<?>) data).isEmpty()) {
            throw new IOException("no models found on the server at " + baseUrl + ": make sure the server"
                    + " is running and has a model loaded");
        }
        Object first = ((List<?>) data).get(0);
        if (!(first instanceof Map)) {
            throw new IOException("the first entry of /v1/models is not an object");
        }
        Map<?, ?> card = (Map<?, ?>) first;
        Object id = card.get("id");
        if (!(id instanceof String)) {
            throw new IOException("the first entry of /v1/models carries no 'id' string");
        }
        Object root = card.get("root");
        return new ServedModel((String) id, root instanceof String ? (String) root : (String) id);
    }

    /**
     * The tokenizer facts of {@code GET base/tokenizer_info}: the vocabulary size and the ids a
     * synthetic prompt skips (the BOS, EOS, PAD, UNK and stop ids the server reports, negative ones
     * dropped).
     */
    public static TokenizerInfo tokenizerInfo(String baseUrl, Map<String, String> headers, boolean insecure)
            throws IOException {
        Map<?, ?> body = getJson(baseUrl + "/tokenizer_info", headers, insecure);
        Object vocab = body.get("vocab_size");
        if (!(vocab instanceof Number)) {
            throw new IOException("/tokenizer_info carries no 'vocab_size' number");
        }
        int vocabSize = ((Number) vocab).intValue();
        List<Integer> skip = new ArrayList<>();
        for (String key : List.of("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id")) {
            addSkip(skip, body.get(key));
        }
        Object stops = body.get("stop_token_ids");
        if (stops instanceof List) {
            for (Object stop : (List<?>) stops) {
                addSkip(skip, stop);
            }
        }
        int[] skipIds = new int[skip.size()];
        for (int i = 0; i < skipIds.length; i++) {
            skipIds[i] = skip.get(i);
        }
        return new TokenizerInfo(vocabSize, skipIds);
    }

    /** One skipped id: a non-negative integer only, added once. */
    private static void addSkip(List<Integer> skip, Object id) {
        if (id instanceof Number && ((Number) id).intValue() >= 0) {
            int value = ((Number) id).intValue();
            if (!skip.contains(value)) {
                skip.add(value);
            }
        }
    }

    /** The tokenizer over the server's {@code POST /tokenize} and {@code POST /detokenize}. */
    public static BenchTokenizer tokenizer(String baseUrl, Map<String, String> headers, boolean insecure) {
        return tokenizer(baseUrl, headers, insecure, null);
    }

    /** The tokenizer whose requests name {@code modelId}, when the server routes by model. */
    public static BenchTokenizer tokenizer(String baseUrl, Map<String, String> headers, boolean insecure,
                                           String modelId) {
        return new BenchTokenizer() {

            private BenchChatTemplate chatTemplate;
            private boolean resolved;

            @Override
            public int[] tokenize(String text) {
                Map<String, Object> body = new LinkedHashMap<>();
                if (modelId != null) {
                    body.put("model", modelId);
                }
                body.put("prompt", text);
                Map<?, ?> out;
                try {
                    out = postJson(baseUrl + "/tokenize", body, headers, insecure);
                } catch (IOException e) {
                    throw new IllegalStateException("cannot count the prompt tokens: " + e.getMessage(), e);
                }
                Object tokens = out.get("tokens");
                if (!(tokens instanceof List)) {
                    throw new IllegalStateException("/tokenize carries no 'tokens' array");
                }
                int[] ids = new int[((List<?>) tokens).size()];
                int i = 0;
                for (Object token : (List<?>) tokens) {
                    if (!(token instanceof Number)) {
                        throw new IllegalStateException("/tokenize carries a non-integer token id");
                    }
                    ids[i++] = ((Number) token).intValue();
                }
                return ids;
            }

            @Override
            public String detokenize(int[] tokens) {
                Map<String, Object> body = new LinkedHashMap<>();
                if (modelId != null) {
                    body.put("model", modelId);
                }
                List<Long> ids = new ArrayList<>(tokens.length);
                for (int token : tokens) {
                    ids.add((long) token);
                }
                body.put("tokens", ids);
                Map<?, ?> out;
                try {
                    out = postJson(baseUrl + "/detokenize", body, headers, insecure);
                } catch (IOException e) {
                    throw new IllegalStateException("cannot decode the token ids: " + e.getMessage(), e);
                }
                Object content = out.get("content");
                if (content instanceof String) {
                    return (String) content;
                }
                Object prompt = out.get("prompt");
                if (prompt instanceof String) {
                    return (String) prompt;
                }
                throw new IllegalStateException("/detokenize carries no 'content' text");
            }

            @Override
            public BenchChatTemplate chatTemplate() {
                if (!resolved) {
                    resolved = true;
                    chatTemplate = serverChatTemplate(baseUrl, headers, insecure, modelId);
                }
                return chatTemplate;
            }
        };
    }

    /**
     * The server's chat template over {@code GET /tokenizer_info}: its {@code chat_template}, the
     * BOS/EOS token texts and the add-BOS/EOS flags, rendered through NeuralFree's own formatter.
     * Null when the server reports no template (a model without one) or cannot answer the route.
     */
    private static BenchChatTemplate serverChatTemplate(String baseUrl, Map<String, String> headers,
                                                        boolean insecure, String modelId) {
        String url = baseUrl + "/tokenizer_info";
        if (modelId != null && !modelId.isEmpty()) {
            url += "?model=" + java.net.URLEncoder.encode(modelId, StandardCharsets.UTF_8);
        }
        Map<?, ?> body;
        try {
            body = getJson(url, headers, insecure);
        } catch (IOException e) {
            return null;
        }
        Object template = body.get("chat_template");
        if (!(template instanceof String) || ((String) template).trim().isEmpty()) {
            return null;
        }
        ChatFormatter formatter = new ChatFormatter(ChatTemplateSource.resolve((String) template),
                string(body.get("bos_token")), string(body.get("eos_token")),
                Boolean.TRUE.equals(body.get("add_bos_token")),
                Boolean.TRUE.equals(body.get("add_eos_token")), null);
        return (messages, kwargs) -> formatter.format(ChatRequest.builder()
                .messages(messages)
                .templateKwargs(kwargs)
                .build());
    }

    /** One JSON value as text, the empty string when it is not a string. */
    private static String string(Object value) {
        return value instanceof String ? (String) value : "";
    }

    /**
     * The server's Prometheus exposition ({@code GET base/metrics}), as text; an unreachable or
     * non-200 endpoint is the reference's "metrics not available" and yields an empty optional.
     */
    public static java.util.Optional<String> metricsText(String baseUrl, Map<String, String> headers,
                                                         boolean insecure) {
        String url = baseUrl.endsWith("/") ? baseUrl + "metrics" : baseUrl + "/metrics";
        HttpRequest.Builder request = HttpRequest.newBuilder(URI.create(url))
                .timeout(Duration.ofSeconds(60))
                .GET();
        headers.forEach(request::header);
        try {
            HttpResponse<String> response = send(request.build(), insecure);
            return response.statusCode() == 200
                    ? java.util.Optional.of(response.body()) : java.util.Optional.empty();
        } catch (IOException e) {
            return java.util.Optional.empty();
        }
    }

    /**
     * One profile-window call ({@code /start_profile} or {@code /stop_profile}) without a body:
     * null on a 200, else the HTTP status with the answer's first bytes. A failed call is the
     * reference's "profiler did not start/stop" and never aborts the benchmark.
     */
    public static String profile(String baseUrl, String action, Map<String, String> headers,
                                 boolean insecure) {
        HttpRequest.Builder request = HttpRequest.newBuilder(URI.create(baseUrl + action))
                .timeout(Duration.ofSeconds(60))
                .POST(HttpRequest.BodyPublishers.noBody());
        headers.forEach(request::header);
        try {
            HttpResponse<String> response = send(request.build(), insecure);
            if (response.statusCode() == 200) {
                return null;
            }
            String body = response.body() == null ? "" : response.body().trim().replace('\n', ' ');
            if (body.length() > 200) {
                body = body.substring(0, 200);
            }
            return "HTTP " + response.statusCode() + (body.isEmpty() ? "" : ": " + body);
        } catch (IOException e) {
            return e.getMessage();
        }
    }

    /** One GET of a JSON object; a non-200 or a non-object body names the URL and the status. */
    private static Map<?, ?> getJson(String url, Map<String, String> headers, boolean insecure)
            throws IOException {
        HttpRequest.Builder request = HttpRequest.newBuilder(URI.create(url))
                .timeout(Duration.ofSeconds(60))
                .GET();
        headers.forEach(request::header);
        HttpResponse<String> response = send(request.build(), insecure);
        return parseBody(url, response);
    }

    /** One POST of a JSON object body. */
    private static Map<?, ?> postJson(String url, Map<String, Object> body, Map<String, String> headers,
                                      boolean insecure) throws IOException {
        HttpRequest.Builder request = HttpRequest.newBuilder(URI.create(url))
                .timeout(Duration.ofSeconds(60))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(Json.toString(body, Json.Style.API),
                        StandardCharsets.UTF_8));
        headers.forEach(request::header);
        HttpResponse<String> response = send(request.build(), insecure);
        return parseBody(url, response);
    }

    /** One send over the process's pooled client for the requested certificate mode. */
    private static HttpResponse<String> send(HttpRequest request, boolean insecure) throws IOException {
        HttpClient client = insecure ? TRUST_ALL_CLIENT : PLAIN_CLIENT;
        try {
            return client.send(request, HttpResponse.BodyHandlers.ofString());
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("the request to " + request.uri() + " was interrupted", e);
        } catch (IOException e) {
            String cause = e.getMessage() != null ? e.getMessage() : e.getClass().getSimpleName();
            throw new IOException("cannot reach " + request.uri() + ": " + cause, e);
        }
    }

    /** The response body as a JSON object, a non-200 naming the URL and the status line. */
    private static Map<?, ?> parseBody(String url, HttpResponse<String> response) throws IOException {
        if (response.statusCode() != 200) {
            throw new IOException("cannot reach " + url + ": HTTP " + response.statusCode());
        }
        try {
            Object parsed = Json.parse(response.body(), Json.Limits.UNTRUSTED);
            if (!(parsed instanceof Map)) {
                throw new IOException(url + " answered a non-object JSON body");
            }
            return (Map<?, ?>) parsed;
        } catch (JsonException e) {
            throw new IOException(url + " answered a non-JSON body: " + e.getMessage());
        }
    }

    /** The discovery client of the plain and of the trust-all certificate mode, one each per process. */
    private static final HttpClient PLAIN_CLIENT = newClient(false);
    private static final HttpClient TRUST_ALL_CLIENT = newClient(true);

    private static HttpClient newClient(boolean trustAll) {
        HttpClient.Builder builder = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(30));
        if (trustAll) {
            builder.sslContext(TrustAllContext.INSTANCE);
        }
        return builder.build();
    }

    /** The trust-all SSL context, built once. */
    private static final class TrustAllContext {

        static final SSLContext INSTANCE = build();

        static SSLContext build() {
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
                context.init(null, new TrustManager[]{trustAll}, new java.security.SecureRandom());
                return context;
            } catch (Exception e) {
                throw new IllegalStateException("cannot build the trust-all SSL context", e);
            }
        }
    }
}

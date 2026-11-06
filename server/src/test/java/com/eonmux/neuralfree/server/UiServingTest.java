package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.utils.json.JsonView;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import jakarta.ws.rs.WebApplicationException;
import jakarta.ws.rs.core.Response;
import jakarta.ws.rs.core.StreamingOutput;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.InetSocketAddress;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.function.Supplier;

import static com.eonmux.neuralfree.server.ServerTestSupport.at;
import static com.eonmux.neuralfree.server.ServerTestSupport.dataFrames;
import static com.eonmux.neuralfree.server.ServerTestSupport.parsed;
import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The bundled UI over the wire: the document and asset routes answer with their headers and content types, the
 * ETag revalidates a conditional request, an unknown asset answers the named 404, and a chat completion in the
 * exact body the UI's script builds (stream, Bearer header, OpenAI messages) streams the server's answer as the
 * {@code data:} frames the script parses.
 */
public class UiServingTest {

    private static EngineService service;
    private static UiResource ui;
    private static OpenAiResource openAi;
    private static HttpServer bridge;

    @BeforeAll
    static void open() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig config = EngineConfig.builder()
                .memory(MemoryConfig.builder()
                        .nCtx(128).ubatch(8).parallel(2).typeK(KvDType.F32).typeV(KvDType.F32).build())
                .build();
        service = new EngineService(model.toString(), config);
        ui = new UiResource();
        openAi = new OpenAiResource(service);
        bridge = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        bridge.createContext("/", ex -> {
            String path = ex.getRequestURI().getPath();
            if (path.equals("/") || path.equals("/index.html") || path.equals("/ui")
                    || path.equals("/ui/")) {
                serve(ex, () -> ui.root(ex.getRequestHeaders().getFirst("If-None-Match")));
            } else if (path.startsWith("/ui/")) {
                String asset = path.substring("/ui/".length());
                String query = ex.getRequestURI().getQuery();
                String version = query != null && query.startsWith("v=") ? query.substring(2) : null;
                serve(ex, () -> ui.asset(asset, version, ex.getRequestHeaders().getFirst("If-None-Match")));
            } else if (path.equals("/v1/models")) {
                serve(ex, () -> openAi.models(null));
            } else if (path.equals("/v1/chat/completions")) {
                String body = new String(ex.getRequestBody().readAllBytes(), StandardCharsets.UTF_8);
                serve(ex, () -> openAi.chatCompletions(body));
            } else {
                ex.sendResponseHeaders(HttpURLConnection.HTTP_NOT_FOUND, -1);
            }
        });
        bridge.start();
    }

    @AfterAll
    static void close() {
        bridge.stop(0);
        service.close();
    }

    /** Writes a resource's response with its own headers, or the response of the refusal it raised. */
    private static void serve(HttpExchange ex, Supplier<Response> call) throws IOException {
        Response response;
        try {
            response = call.get();
        } catch (WebApplicationException e) {
            response = e.getResponse();
        }
        for (Map.Entry<String, List<String>> header : response.getStringHeaders().entrySet()) {
            ex.getResponseHeaders().put(header.getKey(), header.getValue());
        }
        Object entity = response.getEntity();
        if (entity instanceof StreamingOutput stream) {
            ex.sendResponseHeaders(response.getStatus(), 0);
            try (OutputStream out = ex.getResponseBody()) {
                stream.write(out);
            }
            return;
        }
        byte[] bytes = entity instanceof byte[] raw ? raw
                : String.valueOf(entity).getBytes(StandardCharsets.UTF_8);
        ex.getResponseHeaders().add("Content-Type",
                response.getMediaType() == null ? "application/json" : response.getMediaType().toString());
        ex.sendResponseHeaders(response.getStatus(), bytes.length);
        try (OutputStream out = ex.getResponseBody()) {
            out.write(bytes);
        }
    }

    private static String base() {
        return "http://127.0.0.1:" + bridge.getAddress().getPort();
    }

    /** One GET with optional headers; the status, the case-insensitive headers and the body of the answer. */
    private record Fetched(int status, Map<String, String> headers, String body) { }

    private static Fetched get(String path, String... headers) throws IOException {
        HttpURLConnection connection = (HttpURLConnection) URI.create(base() + path).toURL().openConnection();
        for (int i = 0; i + 1 < headers.length; i += 2) {
            connection.setRequestProperty(headers[i], headers[i + 1]);
        }
        int status = connection.getResponseCode();
        InputStream stream = status < 400 ? connection.getInputStream() : connection.getErrorStream();
        String body = stream == null ? "" : new String(stream.readAllBytes(), StandardCharsets.UTF_8);
        Map<String, String> responseHeaders = new TreeMap<>(String.CASE_INSENSITIVE_ORDER);
        for (Map.Entry<String, List<String>> field : connection.getHeaderFields().entrySet()) {
            if (field.getKey() != null) {
                responseHeaders.put(field.getKey(), field.getValue().get(0));
            }
        }
        return new Fetched(status, responseHeaders, body);
    }

    @Test
    void theDocumentServesWithItsHeadersAndTypes() throws IOException {
        Fetched root = get("/");
        assertEquals(200, root.status());
        assertTrue(root.body().toLowerCase().startsWith("<!doctype html"));
        assertTrue(root.headers().get("Content-Type").startsWith("text/html"));
        assertEquals("no-cache", root.headers().get("Cache-Control"));
        assertEquals("require-corp", root.headers().get("Cross-Origin-Embedder-Policy"));
        assertEquals("same-origin", root.headers().get("Cross-Origin-Opener-Policy"));

        Fetched script = get("/ui/app.js");
        assertEquals(200, script.status());
        assertTrue(script.headers().get("Content-Type").startsWith("text/javascript"));
        assertEquals("no-cache", script.headers().get("Cache-Control"));
        java.util.regex.Matcher reference = java.util.regex.Pattern.compile("/ui/app\\.js\\?v=[0-9a-f]{12}")
                .matcher(root.body());
        assertTrue(reference.find(), "the document names the current script version");
        Fetched versioned = get(reference.group());
        assertEquals("public, max-age=31536000, immutable", versioned.headers().get("Cache-Control"));
        assertEquals(script.body(), versioned.body());
        assertTrue(script.body().contains("use strict"));

        Fetched styles = get("/ui/app.css");
        assertEquals(200, styles.status());
        assertTrue(styles.headers().get("Content-Type").startsWith("text/css"));
    }

    @Test
    void aConditionalRequestRevalidatesOverTheWire() throws IOException {
        String etag = get("/").headers().get("ETag");
        assertNotNull(etag);
        Fetched fresh = get("/", "If-None-Match", etag);
        assertEquals(304, fresh.status());
        assertEquals(etag, fresh.headers().get("ETag"));
        assertTrue(fresh.body().isEmpty(), "a 304 carries no body");
    }

    @Test
    void anUnknownAssetAnswersTheNamed404OverTheWire() throws IOException {
        Fetched missing = get("/ui/absent.js");
        assertEquals(404, missing.status());
        assertTrue(missing.body().contains("absent.js"), missing.body());
        assertTrue(missing.headers().get("Content-Type").startsWith("application/json"));
    }

    @Test
    void theUisChatRequestStreamsTheServersAnswer() throws IOException {
        Map<?, ?> models = (Map<?, ?>) parsed(get("/v1/models").body());
        Map<?, ?> card = (Map<?, ?>) ((List<?>) models.get("data")).get(0);
        String modelId = String.valueOf(card.get("id"));
        String body = "{\"model\": \"" + modelId + "\", \"messages\": ["
                + "{\"role\": \"user\", \"content\": \"hello\"}], \"stream\": true, "
                + "\"temperature\": 0, \"max_tokens\": 4}";
        HttpURLConnection connection = (HttpURLConnection) URI.create(base() + "/v1/chat/completions")
                .toURL().openConnection();
        connection.setRequestMethod("POST");
        connection.setDoOutput(true);
        connection.setRequestProperty("Content-Type", "application/json");
        connection.setRequestProperty("Authorization", "Bearer unused-here");
        connection.getOutputStream().write(body.getBytes(StandardCharsets.UTF_8));
        assertEquals(200, connection.getResponseCode());
        assertTrue(connection.getContentType().startsWith("text/event-stream"),
                connection.getContentType());

        String stream;
        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(connection.getInputStream(), StandardCharsets.UTF_8))) {
            StringBuilder text = new StringBuilder();
            for (String line; (line = reader.readLine()) != null; ) {
                text.append(line).append('\n');
            }
            stream = text.toString();
        }
        assertFalse(stream.isBlank(), "the stream is not empty");
        assertTrue(stream.contains("data: [DONE]"), stream);

        StringBuilder answer = new StringBuilder();
        for (JsonView frame : dataFrames(stream)) {
            JsonView delta = at(frame, "choices", 0).obj("delta");
            if (delta.raw("content") != null) {
                answer.append(delta.str("content"));
            }
        }
        // the streamed frames spell the server's own greedy answer
        JsonView direct = view(openAi.chatCompletions(
                "{\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],\"max_tokens\":4,\"temperature\":0}")
                .getEntity());
        JsonView message = at(direct, "choices", 0).obj("message");
        assertFalse(message.str("content").isEmpty(), "the synthetic model answers with text");
        assertEquals(message.str("content"), answer.toString(),
                "the streamed frames spell the server's greedy answer");
    }
}

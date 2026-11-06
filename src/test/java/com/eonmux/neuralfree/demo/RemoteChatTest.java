package com.eonmux.neuralfree.demo;

import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.utils.json.Json;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import org.junit.After;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintStream;
import java.io.StringReader;
import java.net.InetSocketAddress;
import java.net.ServerSocket;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

/**
 * Client mode ({@link RemoteChat} over {@link ChatClient}) against an HTTP server that answers in llama.cpp
 * server's wire format: the health wait, the model choice, the {@code /props} modalities, the request body the
 * reference sends (plus the given sampling flags and the bearer key), the streamed reasoning and content deltas
 * with their timings, error chunks and error statuses, and a cancel that closes the stream.
 */
public class RemoteChatTest {

    @Rule
    public TemporaryFolder tmp = new TemporaryFolder();

    private HttpServer server;
    private final List<String> chatBodies = Collections.synchronizedList(new ArrayList<>());
    private final List<String> authHeaders = Collections.synchronizedList(new ArrayList<>());
    private final List<String> propsQueries = Collections.synchronizedList(new ArrayList<>());
    private final AtomicInteger healthCalls = new AtomicInteger();
    private volatile int healthyAfter;
    private volatile String models = "{\"data\":[{\"id\":\"only\"}]}";
    private volatile String props = "{\"model_alias\":\"alias-name\",\"modalities\":{\"vision\":true}}";
    private volatile int chatStatus = 200;
    private volatile String chatAnswer;
    private volatile CountDownLatch stall;
    private final ByteArrayOutputStream console = new ByteArrayOutputStream();

    @Before
    public void start() throws IOException {
        chatAnswer = sse(reasoning("let me think"), content("Hel"), content("lo"), timings(), "[DONE]");
        server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext("/health", ex -> {
            int n = healthCalls.incrementAndGet();
            reply(ex, n > healthyAfter ? 200 : 503, n > healthyAfter ? "{\"status\":\"ok\"}"
                    : "{\"error\":{\"message\":\"Loading model\"}}");
        });
        server.createContext("/v1/models", ex -> reply(ex, 200, models));
        server.createContext("/props", ex -> {
            propsQueries.add(String.valueOf(ex.getRequestURI().getQuery()));
            reply(ex, props == null ? 404 : 200, props == null ? "{}" : props);
        });
        server.createContext("/v1/chat/completions", this::chat);
        server.start();
    }

    @After
    public void stop() {
        if (stall != null) {
            stall.countDown();
        }
        server.stop(0);
    }

    private void chat(HttpExchange ex) throws IOException {
        chatBodies.add(new String(ex.getRequestBody().readAllBytes(), StandardCharsets.UTF_8));
        authHeaders.add(String.valueOf(ex.getRequestHeaders().getFirst("Authorization")));
        if (chatStatus != 200) {
            reply(ex, chatStatus, "{\"error\":{\"code\":400,\"message\":\"bad media\",\"type\":\"invalid\"}}");
            return;
        }
        ex.getResponseHeaders().add("Content-Type", "text/event-stream");
        ex.sendResponseHeaders(200, 0);
        try (OutputStream out = ex.getResponseBody()) {
            out.write(chatAnswer.getBytes(StandardCharsets.UTF_8));
            out.flush();
            if (stall != null) {
                stall.await(20, TimeUnit.SECONDS);
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        } catch (IOException e) {
            // the client closed the stream
        }
    }

    private static void reply(HttpExchange ex, int status, String body) throws IOException {
        byte[] bytes = body.getBytes(StandardCharsets.UTF_8);
        ex.getResponseHeaders().add("Content-Type", "application/json");
        ex.sendResponseHeaders(status, bytes.length);
        try (OutputStream out = ex.getResponseBody()) {
            out.write(bytes);
        }
    }

    private static String sse(String... payloads) {
        StringBuilder sb = new StringBuilder();
        for (String p : payloads) {
            sb.append("data: ").append(p).append("\r\n\r\n");
        }
        return sb.toString();
    }

    private static String content(String text) {
        return "{\"choices\":[{\"index\":0,\"delta\":{\"content\":\"" + text + "\"}}]}";
    }

    private static String reasoning(String text) {
        return "{\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"" + text + "\"}}]}";
    }

    private static String timings() {
        return "{\"choices\":[],\"timings\":{\"prompt_per_second\":123.45,\"predicted_per_second\":67.8}}";
    }

    private String base() {
        return "http://127.0.0.1:" + server.getAddress().getPort();
    }

    private int run(String input, String... extra) throws Exception {
        List<String> args = new ArrayList<>(List.of("--server-base", base() + "/", "--system", ""));
        args.addAll(List.of(extra));
        PrintStream out = new PrintStream(console, true);
        Interrupts interrupts = new Interrupts(code -> {
            throw new AssertionError("unexpected exit " + code);
        }, out);
        return RemoteChat.run(LLMDemoOptions.parse(args.toArray(new String[0])),
                new BufferedReader(new StringReader(input)), out, interrupts);
    }

    private String output() {
        return new String(console.toByteArray(), StandardCharsets.UTF_8);
    }

    @SuppressWarnings("unchecked")
    private Map<String, Object> body(int i) {
        return (Map<String, Object>) Json.parse(chatBodies.get(i), Json.Limits.UNTRUSTED);
    }

    @Test
    @SuppressWarnings("unchecked")
    public void aTurnStreamsReasoningAndContentAndSendsTheReferenceBody() throws Exception {
        assertEquals(0, run("hi\nagain\n", "--temp", "0.25", "-n", "7", "-r", "STOP", "--api-key", "secret"));
        String out = output();
        assertTrue(out, out.contains("model      : alias-name"));
        assertTrue(out, out.contains("modalities : text, vision"));
        assertTrue(out, out.contains("[reasoning]\nlet me think\n[content]\nHello"));
        assertTrue(out, out.contains("[ Prompt: 123.5 t/s | Generation: 67.8 t/s ]"));
        assertEquals("the props are routed to the model", "model=only", propsQueries.get(0));
        Map<String, Object> first = body(0);
        assertEquals(Boolean.TRUE, first.get("stream"));
        assertEquals(Boolean.TRUE, first.get("timings_per_token"));
        assertEquals("only", first.get("model"));
        assertEquals(0.25, ((Number) first.get("temperature")).doubleValue(), 1e-6);
        assertEquals(7L, ((Number) first.get("max_tokens")).longValue());
        assertEquals(List.of("STOP"), first.get("stop"));
        assertFalse("settings the user did not give stay the server's", first.containsKey("top_k"));
        assertFalse(first.containsKey("seed"));
        assertEquals("Bearer secret", authHeaders.get(0));
        List<Map<String, Object>> messages = (List<Map<String, Object>>) body(1).get("messages");
        assertEquals(3, messages.size());
        assertEquals("Hello", messages.get(1).get("content"));
        assertEquals("let me think", messages.get(1).get("reasoning_content"));
        assertEquals("again", messages.get(2).get("content"));
    }

    @Test
    public void theTemplateVariableFlagsTravelAsChatTemplateKwargsOverTheThinkingSwitch() throws Exception {
        run("hi\n", "--thinking", "--reasoning-effort", "high", "--chat-template-kwargs",
                "{\"enable_thinking\": false, \"persona\": \"terse\"}");
        Map<String, Object> kwargs = new LinkedHashMap<>();
        kwargs.put("enable_thinking", false);
        kwargs.put("reasoning_effort", "high");
        kwargs.put("persona", "terse");
        assertEquals(kwargs, body(0).get("chat_template_kwargs"));
        assertFalse("the effort travels as a template variable", body(0).containsKey("reasoning_effort"));
    }

    @Test
    public void theThinkingSwitchAloneSendsEnableThinking() throws Exception {
        run("hi\n", "--thinking");
        assertEquals(Collections.singletonMap("enable_thinking", true), body(0).get("chat_template_kwargs"));
    }

    @Test
    public void withoutSamplingFlagsTheBodyCarriesOnlyTheReferenceFields() throws Exception {
        run("hi\n");
        assertEquals(List.of("messages", "stream", "timings_per_token", "model"),
                new ArrayList<>(body(0).keySet()));
    }

    @Test
    @SuppressWarnings("unchecked")
    public void theExtendedSamplerBiasAndGrammarFlagsReachTheServer() throws Exception {
        run("hi\n", "--mirostat", "2", "--mirostat-ent", "4.5", "--xtc-probability", "0.3",
                "--top-n-sigma", "1.5", "--dry-multiplier", "0.8", "--dry-sequence-breaker", "##",
                "--dynatemp-range", "0.2", "--min-keep", "3", "--samplers", "top_k;temperature",
                "-l", "15+2.5", "--grammar", "root ::= \"yes\"");
        Map<String, Object> b = body(0);
        assertEquals(2L, ((Number) b.get("mirostat")).longValue());
        assertEquals(4.5, ((Number) b.get("mirostat_tau")).doubleValue(), 1e-6);
        assertFalse("mirostat_eta was not given", b.containsKey("mirostat_eta"));
        assertEquals(0.3, ((Number) b.get("xtc_probability")).doubleValue(), 1e-6);
        assertEquals(1.5, ((Number) b.get("top_n_sigma")).doubleValue(), 1e-6);
        assertEquals(0.8, ((Number) b.get("dry_multiplier")).doubleValue(), 1e-6);
        assertEquals(List.of("##"), b.get("dry_sequence_breakers"));
        assertFalse("dry_base was not given", b.containsKey("dry_base"));
        assertEquals(0.2, ((Number) b.get("dynatemp_range")).doubleValue(), 1e-6);
        assertEquals(3L, ((Number) b.get("min_keep")).longValue());
        assertEquals(List.of("top_k", "temperature"), b.get("samplers"));
        assertEquals(2.5, ((Number) ((Map<String, Object>) b.get("logit_bias")).get("15")).doubleValue(), 1e-6);
        assertEquals("root ::= \"yes\"", b.get("grammar"));
        assertFalse(b.containsKey("json_schema"));
    }

    @Test
    public void theReasoningBudgetAndAdaptivePFlagsReachTheServer() throws Exception {
        run("hi\n", "--reasoning-budget", "16", "--reasoning-budget-message", "Done thinking.",
                "--adaptive-target", "0.3", "--sampler-seq", "kt");
        Map<String, Object> b = body(0);
        assertEquals(16L, ((Number) b.get("reasoning_budget_tokens")).longValue());
        assertEquals("Done thinking.", b.get("reasoning_budget_message"));
        assertEquals(0.3, ((Number) b.get("adaptive_target")).doubleValue(), 1e-6);
        assertEquals(List.of("top_k", "temperature"), b.get("samplers"));
    }

    @Test
    @SuppressWarnings("unchecked")
    public void aJsonSchemaTravelsAsAJsonValue() throws Exception {
        run("hi\n", "-j", "{\"type\":\"object\",\"required\":[\"a\"]}");
        Map<String, Object> schema = (Map<String, Object>) body(0).get("json_schema");
        assertEquals("object", schema.get("type"));
        assertEquals(List.of("a"), schema.get("required"));
        assertFalse(body(0).containsKey("grammar"));
    }

    @Test
    @SuppressWarnings("unchecked")
    public void aReportedModalityStagesMediaAsContentParts() throws Exception {
        Path image = tmp.getRoot().toPath().resolve("pic.jpg");
        Files.write(image, new byte[]{(byte) 0xFF, (byte) 0xD8, (byte) 0xFF, 1});
        Path audio = tmp.getRoot().toPath().resolve("a.wav");
        Files.write(audio, new byte[]{1});
        run("/image " + image + "\n/audio " + audio + "\ndescribe\n");
        assertTrue(output().contains("Loaded media from '" + image + "'"));
        assertTrue(output(), output().contains("Error: /audio is refused: the server does not report audio input"));
        List<Map<String, Object>> messages = (List<Map<String, Object>>) body(0).get("messages");
        List<Object> parts = (List<Object>) messages.get(0).get("content");
        assertEquals(2, parts.size());
        Map<String, Object> imagePart = (Map<String, Object>) parts.get(0);
        assertTrue(((String) ((Map<String, Object>) imagePart.get("image_url")).get("url"))
                .startsWith("data:image/jpeg;base64,"));
        assertEquals("describe", ((Map<String, Object>) parts.get(1)).get("text"));
    }

    @Test
    public void withoutPropsNoMediaIsTaken() throws Exception {
        props = null;
        Path image = tmp.newFile("x.png").toPath();
        run("/image " + image + "\n");
        assertTrue(output(), output().contains("Error: /image is refused: the server does not report vision input"));
        assertTrue("the model id names the session", output().contains("model      : only"));
    }

    @Test
    public void theArgumentMediaJoinsThePromptTurn() throws Exception {
        Path image = tmp.newFile("arg.png").toPath();
        run("", "-p", "what is it", "--image", image.toString());
        assertEquals(1, chatBodies.size());
        assertTrue(chatBodies.get(0).contains("\"image_url\""));
        assertTrue(chatBodies.get(0).contains("what is it"));
    }

    @Test
    public void theUserPicksAmongSeveralModels() throws Exception {
        models = "{\"data\":[{\"id\":\"a\"},{\"id\":\"b\",\"aliases\":[\"bee\",\"b2\"]},{\"nope\":1}]}";
        run("7\nx\n2\nhi\n");
        String out = output();
        assertTrue(out, out.contains("  2. b (bee, b2)"));
        assertEquals(2, out.split("Error: Invalid selection. Please enter a valid number.", -1).length - 1);
        assertTrue(out.contains("Selected model: b"));
        assertEquals("b", body(0).get("model"));
    }

    @Test
    public void anEmptyModelListEndsWithAnError() throws Exception {
        models = "{\"data\":[]}";
        assertEquals(1, run("hi\n"));
        assertTrue(output().contains("Error: the server lists no models on /v1/models"));
    }

    @Test
    public void errorChunksAndStatusesSurfaceAndTheTurnIsDropped() throws Exception {
        chatAnswer = sse(content("par"), "{\"error\":{\"message\":\"context overflow\"}}");
        run("hi\n/regen\n");
        assertTrue(output(), output().contains("Error: context overflow"));
        assertTrue(output().contains("Error: No message to regenerate."));
        chatStatus = 400;
        console.reset();
        run("hi\n");
        assertTrue(output(), output().contains("Error: bad media"));
    }

    @Test
    public void theHealthWaitRidesOutA503() throws Exception {
        healthyAfter = 2;
        assertEquals(0, run("hi\n"));
        assertEquals(3, healthCalls.get());
        assertEquals(1, chatBodies.size());
    }

    @Test
    public void anUnreachableServerFailsAfterTheConnectAttempts() throws Exception {
        int port;
        try (ServerSocket socket = new ServerSocket(0)) {
            port = socket.getLocalPort();
        }
        PrintStream out = new PrintStream(console, true);
        Interrupts interrupts = new Interrupts(code -> {
        }, out);
        int code = RemoteChat.run(LLMDemoOptions.parse(new String[]{"--server-base", "http://127.0.0.1:" + port}),
                new BufferedReader(new StringReader("")), out, interrupts);
        assertEquals(1, code);
        assertTrue(output(), output().contains("Error: failed to connect to http://127.0.0.1:" + port));
    }

    @Test
    public void aCancelClosesTheStreamAndKeepsThePartialAnswer() throws Exception {
        stall = new CountDownLatch(1);
        chatAnswer = sse(content("Hel"));
        ChatClient client = new ChatClient(base(), null);
        PrintStream out = new PrintStream(console, true);
        RemoteChat backend = new RemoteChat(client, "only", ChatClient.Props.NONE,
                LLMDemoOptions.parse(new String[]{"--server-base", base()}), out);
        List<Map<String, Object>> messages = new ArrayList<>();
        messages.add(Map.of("role", "user", "content", "hi"));
        long start = System.nanoTime();
        ChatAnswer answer = backend.answer(messages, () -> output().contains("Hel"));
        long millis = (System.nanoTime() - start) / 1_000_000;
        assertNull(answer.error());
        assertEquals("Hel", answer.content());
        assertTrue("the cancel did not wait for the stalled stream: " + millis + " ms", millis < 10_000);
        assertTrue(output().contains("[interrupted]"));
    }

    @Test
    public void theBaseUrlMustBeHttp() {
        IllegalArgumentException e = org.junit.Assert.assertThrows(IllegalArgumentException.class,
                () -> new ChatClient("localhost:8080", null));
        assertNotNull(e.getMessage());
        assertTrue(e.getMessage(), e.getMessage().contains("http:// or https://"));
    }

    @Test
    public void chatTemplateKwargsFollowTheThinkingSwitchInTheRequest() {
        GenerationRequest generation = GenerationRequest.builder().build();
        assertFalse(RemoteChat.settingsFields(generation, false, Collections.emptyMap())
                .containsKey("chat_template_kwargs"));
        assertEquals(Collections.singletonMap("enable_thinking", true),
                RemoteChat.settingsFields(generation, true, Collections.emptyMap())
                        .get("chat_template_kwargs"));
        Map<String, Object> kwargs = new LinkedHashMap<>();
        kwargs.put("enable_thinking", false);
        kwargs.put("style", "terse");
        assertEquals(kwargs, RemoteChat.settingsFields(generation, true, kwargs).get("chat_template_kwargs"));
    }

    @Test
    public void theReasoningFlagsTravelAsChatTemplateKwargs() throws Exception {
        LLMDemoOptions off = LLMDemoOptions.parse(new String[]{"--server-base", "http://h:1", "--reasoning", "off",
                "--reasoning-effort", "high", "--no-reasoning-preserve"});
        Map<String, Object> expected = new LinkedHashMap<>();
        expected.put("enable_thinking", false);
        expected.put("reasoning_effort", "high");
        expected.put("preserve_reasoning", false);
        assertEquals(expected, RemoteChat.settingsFields(off.generation, off.thinking, off.templateKwargs)
                .get("chat_template_kwargs"));
        LLMDemoOptions auto = LLMDemoOptions.parse(new String[]{"--server-base", "http://h:1", "-rea", "auto"});
        assertFalse("auto leaves the server's default", RemoteChat.settingsFields(auto.generation, auto.thinking,
                auto.templateKwargs).containsKey("chat_template_kwargs"));
    }
}

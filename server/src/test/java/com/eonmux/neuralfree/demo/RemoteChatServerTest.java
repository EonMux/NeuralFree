package com.eonmux.neuralfree.demo;

import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.server.EngineService;
import com.eonmux.neuralfree.server.LlamaCppResource;
import com.eonmux.neuralfree.server.OpenAiResource;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import jakarta.ws.rs.WebApplicationException;
import jakarta.ws.rs.core.Response;
import jakarta.ws.rs.core.StreamingOutput;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintStream;
import java.io.StringReader;
import java.io.StringWriter;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.function.Supplier;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The console's client mode against the real NeuralFree server resources (the {@link OpenAiResource} chat stream,
 * the {@link LlamaCppResource} health and properties) over a synthetic model, served by an HTTP bridge: the chat
 * connects, takes the served model, streams greedy answers that match the server's own non-streamed answer,
 * regenerates the same history, writes the transcript, and refuses media because the server reports none.
 */
class RemoteChatServerTest {

    private static EngineService service;
    private static OpenAiResource openAi;
    private static LlamaCppResource llama;
    private static HttpServer bridge;

    @BeforeAll
    static void open() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig config = EngineConfig.builder()
                .memory(MemoryConfig.builder()
                        .nCtx(128).ubatch(8).parallel(2).typeK(KvDType.F32).typeV(KvDType.F32).build())
                .build();
        service = new EngineService(model.toString(), config);
        openAi = new OpenAiResource(service);
        llama = new LlamaCppResource(service);
        bridge = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        bridge.createContext("/health", ex -> serve(ex, llama::health));
        bridge.createContext("/props", ex -> serve(ex, () -> llama.props(null, null)));
        bridge.createContext("/v1/models", ex -> serve(ex, () -> openAi.models(null)));
        bridge.createContext("/v1/chat/completions", ex -> {
            String body = new String(ex.getRequestBody().readAllBytes(), StandardCharsets.UTF_8);
            serve(ex, () -> openAi.chatCompletions(body));
        });
        bridge.start();
    }

    @AfterAll
    static void close() {
        bridge.stop(0);
        service.close();
    }

    /** Writes a resource's response (or the response of the refusal it raised) to the exchange. */
    private static void serve(HttpExchange ex, Supplier<Response> call) throws IOException {
        Response response;
        try {
            response = call.get();
        } catch (WebApplicationException e) {
            response = e.getResponse();
        }
        Object entity = response.getEntity();
        String type = response.getMediaType() == null ? "application/json" : response.getMediaType().toString();
        ex.getResponseHeaders().add("Content-Type", type);
        if (entity instanceof StreamingOutput stream) {
            ex.sendResponseHeaders(response.getStatus(), 0);
            try (OutputStream out = ex.getResponseBody()) {
                stream.write(out);
            }
            return;
        }
        byte[] bytes = String.valueOf(entity).getBytes(StandardCharsets.UTF_8);
        ex.sendResponseHeaders(response.getStatus(), bytes.length);
        try (OutputStream out = ex.getResponseBody()) {
            out.write(bytes);
        }
    }

    private static String base() {
        return "http://127.0.0.1:" + bridge.getAddress().getPort();
    }

    @Test
    void theChatStreamsTheServersAnswerAndRegeneratesIt() throws Exception {
        String direct = ServerAnswer.of(openAi.chatCompletions(
                "{\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],\"max_tokens\":4,\"temperature\":0}"));
        assertFalse(direct.isEmpty(), "the synthetic model answers with text, so the comparison is meaningful");
        ByteArrayOutputStream console = new ByteArrayOutputStream();
        PrintStream out = new PrintStream(console, true, StandardCharsets.UTF_8);
        StringWriter log = new StringWriter();
        LLMDemoOptions opts = LLMDemoOptions.parse(new String[]{"--server-base", base(), "--system", "", "-n", "4",
                "--temp", "0"});
        Interrupts interrupts = new Interrupts(code -> {
            throw new AssertionError("unexpected exit " + code);
        }, out);
        String model = RemoteChat.selectModel(new ChatClient(base(), null).models(),
                new BufferedReader(new StringReader("")), out);
        ChatClient client = new ChatClient(base(), null);
        assertNull(client.waitHealth(() -> false));
        RemoteChat backend = new RemoteChat(client, model, client.props(model), opts, out);
        ChatSession chat = new ChatSession(backend, opts, new BufferedReader(new StringReader("hello\n/regen\n")),
                out, new Transcript(log), interrupts);
        chat.run();
        List<java.util.Map<String, Object>> messages = chat.messages();
        assertEquals(2, messages.size(), console.toString(StandardCharsets.UTF_8));
        assertEquals(direct, messages.get(1).get("content"), "the streamed answer is the server's greedy answer");
        assertTrue(log.toString().startsWith("User:\nhello\n\nAssistant:\n" + direct), log.toString());
        assertEquals(2, log.toString().split("Assistant:\n", -1).length - 1, "the regeneration is logged too");
    }

    @Test
    void theRunConnectsAndRefusesMediaTheServerDoesNotReport() throws Exception {
        Path image = Files.createTempFile("nf-chat", ".png");
        ByteArrayOutputStream console = new ByteArrayOutputStream();
        PrintStream out = new PrintStream(console, true, StandardCharsets.UTF_8);
        Interrupts interrupts = new Interrupts(code -> {
            throw new AssertionError("unexpected exit " + code);
        }, out);
        int code = RemoteChat.run(LLMDemoOptions.parse(new String[]{"--server-base", base() + "/", "-n", "2"}),
                new BufferedReader(new StringReader("/image " + image + "\nhi\n")), out, interrupts);
        String text = console.toString(StandardCharsets.UTF_8);
        assertEquals(0, code, text);
        assertTrue(text.contains("Connecting to server at " + base()), text);
        assertTrue(text.contains("modalities : text"), text);
        assertTrue(text.contains("Error: /image is refused: the server does not report vision input"), text);
        assertTrue(text.contains("Exiting..."), text);
        Files.delete(image);
    }

    @Test
    void theForwardedGrammarAndSamplerFlagsConstrainTheServersAnswer() throws Exception {
        ByteArrayOutputStream console = new ByteArrayOutputStream();
        PrintStream out = new PrintStream(console, true, StandardCharsets.UTF_8);
        Interrupts interrupts = new Interrupts(code -> {
            throw new AssertionError("unexpected exit " + code);
        }, out);
        LLMDemoOptions opts = LLMDemoOptions.parse(new String[]{"--server-base", base(), "--system", "", "-n", "8",
                "--temp", "0", "--grammar", "root ::= \"yes\" | \"no\"", "--dry-multiplier", "0.5",
                "--top-n-sigma", "2", "--min-keep", "1", "-l", "5+1", "--samplers", "top_k;temperature"});
        ChatClient client = new ChatClient(base(), null);
        String model = RemoteChat.selectModel(client.models(), new BufferedReader(new StringReader("")), out);
        RemoteChat backend = new RemoteChat(client, model, client.props(model), opts, out);
        ChatSession chat = new ChatSession(backend, opts, new BufferedReader(new StringReader("pick one\n")),
                out, null, interrupts);
        chat.run();
        String text = console.toString(StandardCharsets.UTF_8);
        List<java.util.Map<String, Object>> messages = chat.messages();
        assertEquals(2, messages.size(), "the server accepts every forwarded field: " + text);
        Object answer = messages.get(1).get("content");
        assertTrue("yes".equals(answer) || "no".equals(answer), "the grammar forces one choice, got " + answer);
    }

    /** The content of a non-streamed chat completion response. */
    private static final class ServerAnswer {
        static String of(Response response) {
            Object parsed = com.eonmux.neuralfree.utils.json.Json.parse(String.valueOf(response.getEntity()),
                    com.eonmux.neuralfree.utils.json.Json.Limits.UNTRUSTED);
            var choices = (List<?>) ((java.util.Map<?, ?>) parsed).get("choices");
            var message = (java.util.Map<?, ?>) ((java.util.Map<?, ?>) choices.get(0)).get("message");
            return (String) message.get("content");
        }
    }
}

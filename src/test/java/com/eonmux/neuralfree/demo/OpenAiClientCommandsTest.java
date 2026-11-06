package com.eonmux.neuralfree.demo;

import com.eonmux.neuralfree.utils.json.Json;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintStream;
import java.io.StringReader;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

/**
 * The OpenAI client subcommands against an HTTP server that answers in the OpenAI wire format (vLLM's
 * {@code chat} and {@code complete}): the API base of {@code --url}, the model choice from {@code /models}, the
 * streamed chat and completion bodies, the {@code <think>} reasoning markers, the usage-based {@code --stats}
 * metrics, the bearer key, the model list, and the refusals of a bad option, a bad URL and a failed request.
 */
public class OpenAiClientCommandsTest {

    private HttpServer server;
    private final List<String> chatBodies = Collections.synchronizedList(new ArrayList<>());
    private final List<String> completionBodies = Collections.synchronizedList(new ArrayList<>());
    private final List<String> paths = Collections.synchronizedList(new ArrayList<>());
    private final List<String> authHeaders = Collections.synchronizedList(new ArrayList<>());
    private final ByteArrayOutputStream console = new ByteArrayOutputStream();
    private final ByteArrayOutputStream errors = new ByteArrayOutputStream();

    private volatile String models = "{\"object\":\"list\",\"data\":[{\"id\":\"m0\"},{\"id\":\"m1\"}]}";
    private volatile int chatStatus = 200;
    private volatile String chatAnswer;
    private volatile String completionAnswer;
    private volatile Integer completionsStatus;

    @Before
    public void start() throws IOException {
        chatAnswer = sse("{\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"}}]}",
                "{\"choices\":[{\"index\":0,\"delta\":{\"reasoning\":\"let me think\"}}]}",
                "{\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hel\"}}]}",
                "{\"choices\":[{\"index\":0,\"delta\":{\"content\":\"lo\"}}]}",
                usage(),
                "[DONE]");
        completionAnswer = sse("{\"choices\":[{\"index\":0,\"text\":\"do\"}]}",
                "{\"choices\":[{\"index\":0,\"text\":\"ne\"}]}",
                usage(),
                "[DONE]");
        server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext("/v1/models", this::models);
        server.createContext("/v1/chat/completions", this::chat);
        server.createContext("/v1/completions", this::completions);
        server.start();
    }

    @After
    public void stop() {
        server.stop(0);
    }

    private void models(HttpExchange ex) throws IOException {
        paths.add(ex.getRequestURI().getPath());
        authHeaders.add(String.valueOf(ex.getRequestHeaders().getFirst("Authorization")));
        reply(ex, 200, models);
    }

    private void chat(HttpExchange ex) throws IOException {
        paths.add(ex.getRequestURI().getPath());
        chatBodies.add(new String(ex.getRequestBody().readAllBytes(), StandardCharsets.UTF_8));
        authHeaders.add(String.valueOf(ex.getRequestHeaders().getFirst("Authorization")));
        if (chatStatus != 200) {
            reply(ex, chatStatus, "{\"error\":{\"message\":\"boom\",\"type\":\"invalid_request_error\"}}");
            return;
        }
        stream(ex, chatAnswer);
    }

    private void completions(HttpExchange ex) throws IOException {
        paths.add(ex.getRequestURI().getPath());
        completionBodies.add(new String(ex.getRequestBody().readAllBytes(), StandardCharsets.UTF_8));
        authHeaders.add(String.valueOf(ex.getRequestHeaders().getFirst("Authorization")));
        if (completionsStatus != null && completionsStatus != 200) {
            reply(ex, completionsStatus, "{\"error\":{\"message\":\"no completions\",\"type\":\"invalid\"}}");
            return;
        }
        stream(ex, completionAnswer);
    }

    private static void stream(HttpExchange ex, String body) throws IOException {
        ex.getResponseHeaders().add("Content-Type", "text/event-stream");
        ex.sendResponseHeaders(200, 0);
        try (OutputStream out = ex.getResponseBody()) {
            out.write(body.getBytes(StandardCharsets.UTF_8));
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
        for (String payload : payloads) {
            sb.append("data: ").append(payload).append("\r\n\r\n");
        }
        return sb.toString();
    }

    private static String usage() {
        return "{\"choices\":[],\"usage\":{\"prompt_tokens\":4,\"completion_tokens\":3,\"total_tokens\":7}}";
    }

    private String apiBase() {
        return "http://127.0.0.1:" + server.getAddress().getPort() + "/v1";
    }

    private int chat(String input, String... extra) throws IOException {
        List<String> args = new ArrayList<>();
        args.add("--url");
        args.add(apiBase());
        args.addAll(List.of(extra));
        return OpenAiChatCommand.execute(args.toArray(new String[0]),
                new BufferedReader(new StringReader(input)), out(), error());
    }

    private int complete(String input, String... extra) throws IOException {
        List<String> args = new ArrayList<>();
        args.add("--url");
        args.add(apiBase());
        args.addAll(List.of(extra));
        return OpenAiCompleteCommand.execute(args.toArray(new String[0]),
                new BufferedReader(new StringReader(input)), out(), error());
    }

    private PrintStream out() {
        return new PrintStream(console, true);
    }

    private PrintStream error() {
        return new PrintStream(errors, true);
    }

    private String output() {
        return new String(console.toByteArray(), StandardCharsets.UTF_8);
    }

    private String errorOutput() {
        return new String(errors.toByteArray(), StandardCharsets.UTF_8);
    }

    @SuppressWarnings("unchecked")
    private Map<String, Object> body(List<String> bodies, int i) {
        return (Map<String, Object>) Json.parse(bodies.get(i), Json.Limits.UNTRUSTED);
    }

    // ------------------------------------------------------------------ chat

    @Test
    public void aQuickChatPrintsTheReasoningAndContentAndSendsTheReferenceBody() throws Exception {
        assertEquals(0, chat("", "--model-name", "test-model", "-q", "hello"));
        String out = output();
        assertTrue(out, out.startsWith("Using model: test-model\n"));
        assertTrue(out, out.contains("<think>let me think\n</think>Hello\n"));
        assertFalse("no stats without --stats", out.contains("TTFT"));
        assertEquals("the API base is the one the client appends to", "/v1/chat/completions", paths.get(0));
        Map<String, Object> body = body(chatBodies, 0);
        assertEquals("test-model", body.get("model"));
        assertEquals(Boolean.TRUE, body.get("stream"));
        assertFalse("the batch fields of other clients do not travel", body.containsKey("timings_per_token"));
        assertFalse(body.containsKey("stream_options"));
        List<Map<String, Object>> messages = (List<Map<String, Object>>) body.get("messages");
        assertEquals(1, messages.size());
        assertEquals("user", messages.get(0).get("role"));
        assertEquals("hello", messages.get(0).get("content"));
    }

    @Test
    public void theFirstListedModelIsUsedWhenNoModelIsNamed() throws Exception {
        assertEquals(0, chat("", "-q", "hi"));
        assertTrue(output(), output().contains("Using model: m0\n"));
        assertEquals("m0", body(chatBodies, 0).get("model"));
        assertEquals("/v1/models", paths.get(0));
    }

    @Test
    public void aLlamaCppReasoningContentStreamsAsReasoningToo() throws Exception {
        chatAnswer = sse("{\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"pondering\"}}]}",
                "{\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"}}]}", "[DONE]");
        assertEquals(0, chat("", "--model-name", "m", "-q", "hi"));
        assertTrue(output(), output().contains("<think>pondering\n</think>ok\n"));
    }

    @Test
    @SuppressWarnings("unchecked")
    public void theInteractiveChatKeepsTheConversation() throws Exception {
        assertEquals(0, chat("hi\nagain\n", "--model-name", "m", "--system-prompt", "be terse"));
        String out = output();
        assertTrue(out, out.contains("Please enter a message for the chat model:\n"));
        assertTrue("the reference's prompt precedes every read", out.contains("> "));
        assertEquals(2, chatBodies.size());
        List<Map<String, Object>> first = (List<Map<String, Object>>) body(chatBodies, 0).get("messages");
        assertEquals(2, first.size());
        assertEquals("system", first.get(0).get("role"));
        assertEquals("be terse", first.get(0).get("content"));
        assertEquals("hi", first.get(1).get("content"));
        List<Map<String, Object>> second = (List<Map<String, Object>>) body(chatBodies, 1).get("messages");
        assertEquals(4, second.size());
        assertEquals("assistant", second.get(2).get("role"));
        assertEquals("Hello", second.get(2).get("content"));
        assertEquals("again", second.get(3).get("content"));
    }

    @Test
    public void theEndOfInputEndsTheChat() throws Exception {
        assertEquals(0, chat(""));
        assertEquals("no request is sent without a message", 0, chatBodies.size());
    }

    @Test
    public void theStatsRequestUsageAndPrintTTFTAndTPS() throws Exception {
        assertEquals(0, chat("", "--model-name", "m", "-q", "hi", "--stats"));
        Map<String, Object> body = body(chatBodies, 0);
        assertEquals(Collections.singletonMap("include_usage", true), body.get("stream_options"));
        String out = output();
        assertTrue(out, out.contains("TTFT: "));
        assertTrue(out, out.contains(" ms\n"));
        assertTrue(out, out.contains("TPS:  "));
        assertTrue(out, out.contains("(3 tokens in "));
        assertTrue(out, out.contains("tokens/s"));
    }

    @Test
    public void aFailedChatPrintsTheServerMessageAndExitsOne() throws Exception {
        chatStatus = 400;
        assertEquals(1, chat("", "--model-name", "m", "-q", "hi"));
        assertTrue(errorOutput(), errorOutput().contains("Error: boom"));
    }

    @Test
    public void anErrorChunkEndsTheStreamWithItsMessage() throws Exception {
        chatAnswer = sse("{\"choices\":[{\"index\":0,\"delta\":{\"content\":\"par\"}}]}",
                "{\"error\":{\"message\":\"context overflow\"}}");
        assertEquals(1, chat("", "--model-name", "m", "-q", "hi"));
        assertTrue(errorOutput(), errorOutput().contains("Error: context overflow"));
        assertTrue("what arrived before the error is printed", output().contains("par"));
    }

    @Test
    public void anEmptyModelListIsAnError() throws Exception {
        models = "{\"object\":\"list\",\"data\":[]}";
        assertEquals(1, chat("", "-q", "hi"));
        assertTrue(errorOutput(), errorOutput().contains("the server lists no models"));
    }

    // ------------------------------------------------------------------ complete

    @Test
    public void aQuickCompletionPrintsTheStreamedTextAndSendsThePrompt() throws Exception {
        assertEquals(0, complete("", "--model-name", "m", "--max-tokens", "7", "-q", "abc"));
        assertTrue(output(), output().startsWith("Using model: m\n"));
        assertTrue(output(), output().contains("done\n"));
        assertEquals("/v1/completions", paths.get(0));
        Map<String, Object> body = body(completionBodies, 0);
        assertEquals("m", body.get("model"));
        assertEquals("abc", body.get("prompt"));
        assertEquals(Boolean.TRUE, body.get("stream"));
        assertEquals(7L, ((Number) body.get("max_tokens")).longValue());
        assertFalse(body.containsKey("messages"));
        assertFalse(body.containsKey("stream_options"));
    }

    @Test
    public void withoutMaxTokensTheServerDefaultApplies() throws Exception {
        assertEquals(0, complete("", "--model-name", "m", "-q", "abc"));
        assertFalse(body(completionBodies, 0).containsKey("max_tokens"));
    }

    @Test
    public void theInteractiveCompletionsKeepNoConversation() throws Exception {
        assertEquals(0, complete("first\nsecond\n", "--model-name", "m"));
        String out = output();
        assertTrue(out, out.contains("Please enter prompt to complete:\n"));
        assertEquals(2, completionBodies.size());
        assertEquals("first", body(completionBodies, 0).get("prompt"));
        assertEquals("second", body(completionBodies, 1).get("prompt"));
        assertFalse("a prompt starts a new completion", body(completionBodies, 1).containsKey("messages"));
    }

    @Test
    public void theCompletionsStatsUseTheUsageChunk() throws Exception {
        assertEquals(0, complete("", "--model-name", "m", "-q", "abc", "--stats"));
        assertEquals(Collections.singletonMap("include_usage", true),
                body(completionBodies, 0).get("stream_options"));
        String out = output();
        assertTrue(out, out.contains("TTFT: "));
        assertTrue(out, out.contains("(3 tokens in "));
    }

    @Test
    public void aFailedCompletionExitsOne() throws Exception {
        completionsStatus = 500;
        assertEquals(1, complete("", "--model-name", "m", "-q", "abc"));
        assertTrue(errorOutput(), errorOutput().contains("Error: no completions"));
    }

    // ------------------------------------------------------------------ the command line

    @Test
    public void theApiKeyTravelsAsABearerToken() throws Exception {
        assertEquals(0, chat("", "--model-name", "m", "-q", "hi", "--api-key", "secret"));
        assertEquals("Bearer secret", authHeaders.get(0));
    }

    @Test
    public void aFlagOfTheOtherCommandIsRefusedByName() throws Exception {
        assertEquals(2, chat("", "--max-tokens", "4"));
        assertTrue(errorOutput(), errorOutput().contains("chat has no option '--max-tokens'"));
        assertEquals(2, complete("", "--system-prompt", "x"));
        assertTrue(errorOutput(), errorOutput().contains("complete has no option '--system-prompt'"));
    }

    @Test
    public void anUnknownOptionAndAPositionalArgumentAreRefused() throws Exception {
        assertEquals(2, chat("", "--bogus"));
        assertTrue(errorOutput(), errorOutput().contains("chat has no option '--bogus'"));
        assertEquals(2, complete("", "prompt"));
        assertTrue(errorOutput(), errorOutput().contains("complete takes no positional argument 'prompt'"));
        assertEquals(2, chat("", "--stats=yes"));
        assertTrue(errorOutput(), errorOutput().contains("--stats takes no value"));
        assertEquals(2, complete("", "--max-tokens", "many"));
        assertTrue(errorOutput(), errorOutput().contains("--max-tokens must be a whole number"));
        assertEquals("no request reaches the server", 0, paths.size());
    }

    @Test
    public void aBadUrlIsRefusedBeforeAnyRequest() throws Exception {
        assertEquals(1, chat("", "--url", "ftp://host/v1", "-q", "hi"));
        assertTrue(errorOutput(), errorOutput().contains("must be an http:// or https:// URL"));
        assertEquals(1, complete("", "--url", "http://host/v1?x=1", "-q", "hi"));
        assertTrue(errorOutput(), errorOutput().contains("must not carry a query"));
        assertEquals(0, paths.size());
    }

    @Test
    public void theHelpPrintsTheCommandSurfaceAndSucceeds() throws Exception {
        assertEquals(0, chat("", "--help"));
        assertTrue(output(), output().contains("neuralfree chat [options]"));
        assertTrue(output(), output().contains("--system-prompt"));
        assertEquals(0, complete("", "--help"));
        assertTrue(output(), output().contains("neuralfree complete [options]"));
        assertTrue(output(), output().contains("--max-tokens"));
    }

    @Test
    public void theOptionsCarryThevLLMDefaults() throws Exception {
        OpenAiClientOptions chat = OpenAiClientOptions.parse(new String[]{"--quick", "x"}, true, out());
        assertNotNull(chat);
        assertEquals(OpenAiClientOptions.DEFAULT_URL, chat.url());
        assertNull(chat.modelName());
        assertNull(chat.maxTokens());
        assertNull(chat.systemPrompt());
        assertFalse(chat.stats());
        assertEquals(OpenAiClientOptions.DEFAULT_API_KEY,
                OpenAiClientOptions.parse(new String[]{"--api-key", OpenAiClientOptions.DEFAULT_API_KEY}, true,
                        out()).apiKey());
        OpenAiClientOptions complete = OpenAiClientOptions.parse(
                new String[]{"--max-tokens=12", "--stats"}, false, out());
        assertNotNull(complete);
        assertEquals(Integer.valueOf(12), complete.maxTokens());
        assertTrue(complete.stats());
        assertNull(complete.systemPrompt());
    }
}

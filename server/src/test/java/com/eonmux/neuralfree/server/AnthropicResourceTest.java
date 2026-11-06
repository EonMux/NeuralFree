package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.utils.json.Json;
import com.eonmux.neuralfree.utils.json.JsonView;
import jakarta.ws.rs.WebApplicationException;
import jakarta.ws.rs.core.Response;
import jakarta.ws.rs.core.StreamingOutput;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The Anthropic Messages surface over the real service: the message body's shape (content blocks, stop reasons,
 * the usage counts), the system prompt and multipart text blocks reaching the template, {@code max_tokens}
 * required as Anthropic documents, the named refusals (a non-text content block, a bad role, a body without
 * {@code max_tokens}), the token-count route, and the stream's event sequence from {@code message_start} to
 * {@code message_stop}.
 */
public class AnthropicResourceTest {

    private static EngineService service;
    private static AnthropicResource resource;

    @BeforeAll
    static void open() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig config = EngineConfig.builder()
                .memory(MemoryConfig.builder()
                        .nCtx(256).ubatch(8).parallel(2).typeK(KvDType.F32).typeV(KvDType.F32).build())
                .build();
        service = new EngineService(model.toString(), config);
        resource = new AnthropicResource(service);
    }

    @AfterAll
    static void close() {
        service.close();
    }

    @Test
    void aMessageAnswersInTheAnthropicShape() {
        Response response = resource.messages("""
                {"model": "test", "max_tokens": 5, "messages": [{"role": "user", "content": "Say hello"}]}""");
        assertEquals(200, response.getStatus());
        JsonView body = view(response.getEntity());
        assertEquals("message", body.str("type"));
        assertEquals("assistant", body.str("role"));
        List<?> content = (List<?>) body.raw("content");
        assertTrue(content.size() >= 1, "the answer carries a content block");
        JsonView block = JsonView.of(content.get(0), "content[0]");
        assertEquals("text", block.str("type"));
        assertNotNull(block.str("text"));
        assertTrue(body.str("stop_reason").equals("end_turn") || body.str("stop_reason").equals("max_tokens"),
                "the stop reason is named: " + body.str("stop_reason"));
        assertNull(body.raw("stop_sequence"));
        JsonView usage = JsonView.of(body.raw("usage"), "usage");
        assertTrue(usage.i64("input_tokens") >= 1, "input tokens count the rendered prompt");
        assertTrue(usage.i64("output_tokens") >= 1, "output tokens count the generation");
        assertEquals(0L, usage.i64("cache_read_input_tokens"));
        assertNull(body.raw("timings"), "the Anthropic body carries no timings");
    }

    @Test
    void systemAndMultipartTextBlocksReachTheTemplate() {
        long with = systemInputTokens();
        long without = plainInputTokens();
        assertTrue(with > without,
                "the system prompt and both blocks reached the prompt (" + with + " vs " + without + ")");
        Response withSystem;
        try {
            withSystem = resource.messages("""
                    {"model": "test", "max_tokens": 4,
                     "system": "Be brief.",
                     "messages": [{"role": "user", "content": [
                         {"type": "text", "text": "What is"},
                         {"type": "text", "text": " two plus two?"}]}]}""");
        } catch (WebApplicationException e) {
            throw new AssertionError("generation refused with " + with + " prompt tokens (without " + without
                    + "): " + body(e.getResponse()), e);
        }
        assertEquals(200, withSystem.getStatus());
        JsonView body = view(withSystem.getEntity());
        JsonView usage = JsonView.of(body.raw("usage"), "usage");
        assertTrue(usage.i64("input_tokens") >= 1 && usage.i64("input_tokens") <= with,
                "the generation counted the rendered prompt (cached prefixes count less): "
                        + usage.i64("input_tokens") + " of " + with);
    }

    private static String body(Response response) {
        return String.valueOf(response.getEntity());
    }

    @Test
    void stopSequencesNameTheirStop() {
        Response response = resource.messages("""
                {"model": "test", "max_tokens": 8, "temperature": 0,
                 "stop_sequences": ["a"],
                 "messages": [{"role": "user", "content": "no"}]}""");
        assertEquals(200, response.getStatus());
        JsonView body = view(response.getEntity());
        String stopReason = body.str("stop_reason");
        assertTrue(stopReason.equals("stop_sequence") || stopReason.equals("end_turn")
                        || stopReason.equals("max_tokens"),
                "a named stop reason: " + stopReason);
    }

    @Test
    void theRefusalsCarryTheGapByName() {
        assertEquals(400, statusOf(() -> resource.messages("""
                {"model": "test", "messages": [{"role": "user", "content": "hi"}]}""")),
                "max_tokens is required");
        assertEquals(400, statusOf(() -> resource.messages("""
                {"model": "test", "max_tokens": 4, "messages": [{"role": "user", "content": [
                    {"type": "image", "source": {}}]}]}""")));
        assertEquals(400, statusOf(() -> resource.messages("""
                {"model": "test", "max_tokens": 4, "messages": [{"role": "tool", "content": "x"}]}""")));
        assertEquals(400, statusOf(() -> resource.messages("""
                {"model": "test", "max_tokens": 4, "messages": []}""")));
    }

    @Test
    void countTokensCountsTheRenderedPrompt() {
        Response response = resource.countTokens("""
                {"model": "test", "max_tokens": 4,
                 "messages": [{"role": "user", "content": "hello world"}]}""");
        assertEquals(200, response.getStatus());
        JsonView body = view(response.getEntity());
        assertTrue(body.i64("input_tokens") >= 1, "the rendered prompt tokenizes to at least one token");
    }

    @Test
    void aStreamCarriesTheAnthropicEventSequence() throws Exception {
        Response response = resource.messages("""
                {"model": "test", "max_tokens": 3, "stream": true,
                 "messages": [{"role": "user", "content": "Say hello"}]}""");
        assertEquals(200, response.getStatus());
        assertEquals("text/event-stream", response.getMediaType().toString());
        StreamingOutput stream = (StreamingOutput) response.getEntity();
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        stream.write(out);
        String events = out.toString(StandardCharsets.UTF_8);
        assertTrue(events.contains("content_block_delta"), "the stream carries text deltas: " + events);
        assertTrue(events.startsWith("event: message_start\n"), "the stream opens with message_start");
        assertTrue(!events.contains("\"input_tokens\":0"), "message_start reports the rendered prompt's tokens");
        assertTrue(events.contains("event: content_block_start\n"), "a content block opens");
        assertTrue(events.contains("\"type\":\"text_delta\""), "the delta type is text_delta");
        assertTrue(events.contains("event: content_block_stop\n"), "the content block closes");
        assertTrue(events.contains("event: message_delta\n"), "message_delta names the stop reason");
        assertTrue(events.contains("\"stop_reason\":\"end_turn\"") || events.contains("\"stop_reason\":\"max_tokens\""),
                "the stop reason is named");
        assertTrue(events.contains("event: message_stop\n"), "the stream closes with message_stop");
    }

    private long plainInputTokens() {
        Response response = resource.countTokens("""
                {"model": "test", "max_tokens": 4,
                 "messages": [{"role": "user", "content": "What is two plus two?"}]}""");
        return view(response.getEntity()).i64("input_tokens");
    }

    private long systemInputTokens() {
        Response response = resource.countTokens("""
                {"model": "test", "max_tokens": 4,
                 "system": "Be brief.",
                 "messages": [{"role": "user", "content": [
                     {"type": "text", "text": "What is"},
                     {"type": "text", "text": " two plus two?"}]}]}""");
        return view(response.getEntity()).i64("input_tokens");
    }

    private static JsonView view(Object entity) {
        return JsonView.of(Json.parse(String.valueOf(entity), Json.Limits.UNTRUSTED), "body");
    }

    private static int statusOf(Runnable call) {
        try {
            call.run();
            return 200;
        } catch (WebApplicationException e) {
            return e.getResponse().getStatus();
        }
    }
}

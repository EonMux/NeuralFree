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

import static com.eonmux.neuralfree.server.ServerTestSupport.errorOf;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The Anthropic Messages surface over the real service: the message body's shape (content blocks, stop reasons,
 * the usage counts), the system prompt and multipart text blocks reaching the template, {@code max_tokens}
 * required as Anthropic documents, the named refusals (a non-text content block, a bad role, a body without
 * {@code max_tokens}), the token-count route, the stream's event sequence from {@code message_start} to
 * {@code message_stop}, the usage's prompt split ({@code cache_read_input_tokens} the prefix cache served, in the
 * body and both ends of a stream), the {@code cache_control} breakpoints accepted or refused by name, and the
 * {@code kv_transfer_params} shapes refused by name.
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
        assertEquals(0L, usage.i64("cache_creation_input_tokens"), "there is no explicit cache write");
        assertEquals(renderedTokens("Say hello"), usage.i64("input_tokens") + usage.i64("cache_read_input_tokens"),
                "the uncached and the cached tokens add up to the rendered prompt");
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
    void kvTransferParamsRefuseTheirBadShapesByName() {
        assertTrue(errorOf(() -> resource.messages("""
                {"model": "test", "max_tokens": 4, "messages": [{"role": "user", "content": "hi"}],
                 "kv_transfer_params": 1}""")).contains("kv_transfer_params"));
        assertTrue(errorOf(() -> resource.messages("""
                {"model": "test", "max_tokens": 4, "messages": [{"role": "user", "content": "hi"}],
                 "kv_transfer_params": {"rank": 0}}""")).contains("kv_transfer_params.rank"));
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

    @Test
    void aRepeatedMessageReadsItsPromptFromThePrefixCache() {
        String body = """
                {"model": "test", "max_tokens": 3, "temperature": 0,
                 "messages": [{"role": "user", "content": "a message the cache keeps for its repeat"}]}""";
        long total = renderedTokens("a message the cache keeps for its repeat");
        JsonView first = JsonView.of(view(resource.messages(body).getEntity()).raw("usage"), "usage");
        assertEquals(total, first.i64("input_tokens") + first.i64("cache_read_input_tokens"));
        JsonView again = JsonView.of(view(resource.messages(body).getEntity()).raw("usage"), "usage");
        assertEquals(total - 1, again.i64("cache_read_input_tokens"),
                "the repeat reads every prompt token but the last from the cache");
        assertEquals(1L, again.i64("input_tokens"), "only the last prompt token is computed");
        assertEquals(0L, again.i64("cache_creation_input_tokens"));
    }

    @Test
    void aStreamReportsTheCachedTokensInItsStartAndFinalUsage() throws Exception {
        String body = """
                {"model": "test", "max_tokens": 3, "temperature": 0, "stream": %s,
                 "messages": [{"role": "user", "content": "a streamed message read back from the cache"}]}""";
        long total = renderedTokens("a streamed message read back from the cache");
        resource.messages(body.formatted("false"));
        StreamingOutput stream = (StreamingOutput) resource.messages(body.formatted("true")).getEntity();
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        stream.write(out);
        JsonView start = null;
        JsonView delta = null;
        for (String line : out.toString(StandardCharsets.UTF_8).split("\n")) {
            if (!line.startsWith("data: ")) {
                continue;
            }
            JsonView event = view(line.substring("data: ".length()));
            if ("message_start".equals(event.str("type"))) {
                start = JsonView.of(event.obj("message").raw("usage"), "usage");
            } else if ("message_delta".equals(event.str("type"))) {
                delta = JsonView.of(event.raw("usage"), "usage");
            }
        }
        assertNotNull(start, "the stream opens with message_start");
        assertNotNull(delta, "the stream ends with message_delta");
        assertEquals(total - 1, start.i64("cache_read_input_tokens"), "message_start reads the cached prompt");
        assertEquals(1L, start.i64("input_tokens"));
        assertEquals(0L, start.i64("cache_creation_input_tokens"));
        assertEquals(0L, start.i64("output_tokens"));
        assertEquals(total - 1, delta.i64("cache_read_input_tokens"), "the final usage repeats the prompt split");
        assertEquals(1L, delta.i64("input_tokens"));
        assertTrue(delta.i64("output_tokens") >= 1, "the final usage counts the generation");
    }

    @Test
    void cacheControlBreakpointsAreAcceptedAndMalformedOnesRefusedByName() {
        Response marked = resource.messages("""
                {"model": "test", "max_tokens": 2,
                 "system": [{"type": "text", "text": "Be brief.", "cache_control": {"type": "ephemeral"}}],
                 "tools": [{"name": "lookup", "input_schema": {"type": "object"},
                            "cache_control": {"type": "ephemeral", "ttl": "1h"}}],
                 "messages": [{"role": "user", "content": [
                     {"type": "text", "text": "hi", "cache_control": {"type": "ephemeral", "ttl": "5m"}}]}]}""");
        assertEquals(200, marked.getStatus());
        String persistent = """
                {"model": "test", "max_tokens": 2, "messages": [{"role": "user", "content": [
                    {"type": "text", "text": "hi", "cache_control": {"type": "persistent"}}]}]}""";
        assertEquals(400, statusOf(() -> resource.messages(persistent)));
        assertTrue(errorOf(() -> resource.messages(persistent)).contains("cache_control"));
        String ttl = """
                {"model": "test", "max_tokens": 2, "system": [{"type": "text", "text": "x",
                    "cache_control": {"type": "ephemeral", "ttl": "2h"}}],
                 "messages": [{"role": "user", "content": "hi"}]}""";
        assertEquals(400, statusOf(() -> resource.messages(ttl)));
        assertTrue(errorOf(() -> resource.messages(ttl)).contains("cache_control.ttl"));
        String tool = """
                {"model": "test", "max_tokens": 2, "tools": [{"name": "t", "cache_control": "yes"}],
                 "messages": [{"role": "user", "content": "hi"}]}""";
        assertTrue(errorOf(() -> resource.messages(tool)).contains("cache_control of tools[0]"));
    }

    /** The rendered prompt's token count for a single user message ({@code count_tokens}). */
    private long renderedTokens(String userText) {
        Response response = resource.countTokens("""
                {"model": "test", "max_tokens": 4, "messages": [{"role": "user", "content": "%s"}]}"""
                .formatted(userText));
        return view(response.getEntity()).i64("input_tokens");
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

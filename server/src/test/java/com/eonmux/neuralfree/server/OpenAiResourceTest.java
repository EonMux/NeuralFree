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
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The OpenAI-compatible HTTP surface over the real service (PARITY P2): the synthetic two-layer llama opens on a
 * paged {@code parallel = 2} config and the resources answer every protocol form — the model list, non-streamed
 * completions and chat completions with usage and finish reasons, SSE streams ending in {@code data: [DONE]},
 * guided_choice through the grammar route, the named refusals (out-of-range {@code n}, {@code logit_bias} and
 * token ids, two constraints at once, tool choices without tools, media parts) and the llama.cpp-native
 * {@code /health} and {@code /props}. An unloaded service refuses with 503.
 */
public class OpenAiResourceTest {

    private static EngineService service;
    private static EngineService unloaded;
    private static OpenAiResource resource;
    private static LlamaCppResource llama;

    @BeforeAll
    static void open() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig config = EngineConfig.builder()
                .memory(MemoryConfig.builder()
                        .nCtx(64).ubatch(8).parallel(2).typeK(KvDType.F32).typeV(KvDType.F32).build())
                .build();
        service = new EngineService(model.toString(), config);
        unloaded = new EngineService(null, config);
        resource = new OpenAiResource(service);
        llama = new LlamaCppResource(service);
    }

    @AfterAll
    static void close() {
        service.close();
        unloaded.close();
    }

    @Test
    void theModelListNamesTheServedModel() {
        Response response = resource.models();
        assertEquals(200, response.getStatus());
        JsonView body = view(response.getEntity());
        assertEquals("list", body.str("object"));
        JsonView data = JsonView.of(((java.util.List<?>) body.raw("data")).get(0), "data");
        assertEquals("model", data.str("object"));
        assertTrue(data.str("id").endsWith(".gguf"));
    }

    @Test
    void aCompletionAnswersWithTextUsageAndFinishReason() {
        Response response = resource.completions("""
                {"prompt": "hello", "max_tokens": 4, "temperature": 0}""");
        assertEquals(200, response.getStatus());
        JsonView body = view(response.getEntity());
        assertEquals("text_completion", body.str("object"));
        JsonView choice = firstChoice(body);
        assertNotNull(choice.str("text"));
        assertTrue(choice.str("finish_reason").equals("stop") || choice.str("finish_reason").equals("length"));
        JsonView usage = JsonView.of(body.raw("usage"), "usage");
        assertTrue(usage.i64("prompt_tokens") >= 1, "prompt tokens count the encoded prompt");
        assertEquals(4L, usage.i64("completion_tokens"), "the budget bounds the completion");
        assertEquals(usage.i64("prompt_tokens") + usage.i64("completion_tokens"), usage.i64("total_tokens"));
    }

    @Test
    void chatLogprobsListTheTopTokensWithTheSampledOneLeadingUnderGreedy() {
        Response response = resource.chatCompletions("""
                {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 3, "temperature": 0,
                 "logprobs": true, "top_logprobs": 2}""");
        JsonView logprobs = JsonView.of(firstChoice(view(response.getEntity())).raw("logprobs"), "logprobs");
        List<?> content = (List<?>) logprobs.raw("content");
        assertEquals(3, content.size(), "one entry per generated token");
        for (Object o : content) {
            JsonView entry = JsonView.of(o, "content[i]");
            assertTrue(entry.f64("logprob") <= 0, "the sampled token's logprob is non-positive");
            assertTrue(entry.raw("bytes") instanceof List<?>, "the token's UTF-8 bytes");
            List<?> top = (List<?>) entry.raw("top_logprobs");
            assertEquals(2, top.size(), "top_logprobs tokens");
            JsonView best = JsonView.of(top.get(0), "top[0]");
            JsonView next = JsonView.of(top.get(1), "top[1]");
            assertEquals(entry.str("token"), best.str("token"), "greedy samples the most likely token");
            assertEquals(entry.f64("logprob"), best.f64("logprob"), 1e-9);
            assertTrue(best.f64("logprob") >= next.f64("logprob"), "most likely first");
        }
        assertEquals(400, statusOf(() -> resource.chatCompletions("""
                {"messages": [{"role": "user", "content": "hi"}], "top_logprobs": 2}""")),
                "top_logprobs needs logprobs: true");
        assertEquals(400, statusOf(() -> resource.chatCompletions("""
                {"messages": [{"role": "user", "content": "hi"}], "logprobs": true, "top_logprobs": 21}""")));
    }

    @Test
    void completionLogprobsUseTheLegacyArrays() {
        Response response = resource.completions("""
                {"prompt": "hello", "max_tokens": 3, "temperature": 0, "logprobs": 2, "echo": true}""");
        JsonView choice = firstChoice(view(response.getEntity()));
        JsonView logprobs = JsonView.of(choice.raw("logprobs"), "logprobs");
        List<?> tokens = (List<?>) logprobs.raw("tokens");
        List<?> tokenLogprobs = (List<?>) logprobs.raw("token_logprobs");
        List<?> tops = (List<?>) logprobs.raw("top_logprobs");
        List<?> offsets = (List<?>) logprobs.raw("text_offset");
        assertEquals(3, tokens.size(), "one entry per generated token");
        assertEquals(3, tokenLogprobs.size());
        assertEquals(3, tops.size());
        assertEquals("hello".length(), ((Number) offsets.get(0)).longValue(), "offsets count the echoed prompt");
        for (int i = 0; i < 3; i++) {
            Map<?, ?> top = (Map<?, ?>) tops.get(i);
            assertTrue(top.size() >= 1 && top.size() <= 2, "at most the requested alternatives");
            assertTrue(top.containsKey(tokens.get(i)), "greedy's token is among the most likely");
            if (i > 0) {
                long step = ((Number) offsets.get(i)).longValue() - ((Number) offsets.get(i - 1)).longValue();
                assertEquals(((String) tokens.get(i - 1)).length(), step, "offsets advance by the token text");
            }
        }
    }

    @Test
    void minTokensAndBadWordsRunThroughTheServerMapping() {
        Response response = resource.completions("""
                {"prompt": "hello", "max_tokens": 4, "temperature": 0,
                 "min_tokens": 2, "bad_words": ["surely absent phrase"]}""");
        assertEquals(200, response.getStatus(), "the parity stages accept the request");
        String text = firstChoice(view(response.getEntity())).str("text");
        assertNotNull(text);
    }

    @Test
    void aChatCompletionAnswersWithAnAssistantMessage() {
        Response response;
        try {
            response = resource.chatCompletions("""
                {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 4, "temperature": 0}""");
        } catch (WebApplicationException e) {
            throw new AssertionError("sequence failed: " + e.getResponse().getEntity(), e);
        }
        assertEquals(200, response.getStatus());
        JsonView body = view(response.getEntity());
        assertEquals("chat.completion", body.str("object"));
        JsonView message = JsonView.of(firstChoice(body).raw("message"), "message");
        assertEquals("assistant", message.str("role"));
        assertNotNull(message.optStr("content", null));
        JsonView usage = JsonView.of(body.raw("usage"), "usage");
        assertTrue(usage.i64("completion_tokens") >= 1);
    }

    @Test
    void aStreamedCompletionEndsInDone() throws Exception {
        Response response = resource.completions("""
                {"prompt": "hello", "max_tokens": 3, "temperature": 0, "stream": true}""");
        assertEquals(200, response.getStatus());
        assertEquals("text/event-stream", response.getMediaType().toString());
        StreamingOutput stream = (StreamingOutput) response.getEntity();
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        stream.write(out);
        String events = out.toString(StandardCharsets.UTF_8);
        assertTrue(events.startsWith("data: "), "every frame starts with data:");
        assertTrue(events.endsWith("data: [DONE]\n\n"), "the terminal frame is [DONE]");
        assertTrue(events.contains("\"text\""), "a delta carries text");
        assertTrue(events.contains("\"finish_reason\":\"stop\"") || events.contains("\"finish_reason\":\"length\""),
                "the final chunk names the finish reason");
    }

    @Test
    void aStreamedCompletionCarriesTheLegacyLogprobsPerToken() throws Exception {
        Response response = resource.completions("""
                {"prompt": "hello", "max_tokens": 3, "temperature": 0, "stream": true, "logprobs": 1}""");
        assertEquals(200, response.getStatus());
        StreamingOutput stream = (StreamingOutput) response.getEntity();
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        stream.write(out);
        int entries = 0;
        long expectedOffset = 0;
        for (String frame : out.toString(StandardCharsets.UTF_8).split("\n\n")) {
            if (!frame.startsWith("data: ") || frame.equals("data: [DONE]")) {
                continue;
            }
            JsonView chunk = JsonView.of(Json.parse(frame.substring("data: ".length()), Json.Limits.UNTRUSTED),
                    "chunk");
            JsonView choice = JsonView.of(((List<?>) chunk.raw("choices")).get(0), "choice");
            if (choice.raw("logprobs") != null) {
                JsonView logprobs = JsonView.of(choice.raw("logprobs"), "logprobs");
                assertEquals(1, ((List<?>) logprobs.raw("tokens")).size(), "one token per logprobs chunk");
                double logprob = ((Number) ((List<?>) logprobs.raw("token_logprobs")).get(0)).doubleValue();
                assertTrue(logprob <= 0, "the sampled token's logprob is non-positive");
                assertEquals(expectedOffset, ((Number) ((List<?>) logprobs.raw("text_offset")).get(0)).longValue(),
                        "the offset is the choice text streamed so far");
                entries++;
            }
            expectedOffset += choice.str("text").length();
        }
        assertTrue(entries >= 1, "the stream carries the per-token entries");
    }

    @Test
    void guidedChoiceConstrainsTheAnswerThroughTheGrammarRoute() {
        Response response;
        try {
            response = resource.completions("""
                {"prompt": "pick one", "max_tokens": 8, "temperature": 0,
                 "guided_choice": ["yes", "no"]}""");
        } catch (WebApplicationException e) {
            System.out.println("guided response body: " + e.getResponse().getEntity());
            throw e;
        }
        assertEquals(200, response.getStatus());
        String text = firstChoice(view(response.getEntity())).str("text");
        assertTrue(text.equals("yes") || text.equals("no"), "the grammar forces one choice, got " + text);
    }

    @Test
    void refusalsCarryTheGapByName() {
        assertEquals(400, statusOf(() -> resource.completions(
                "{\"prompt\": \"x\", \"n\": 0}")));
        assertEquals(400, statusOf(() -> resource.completions(
                "{\"prompt\": \"x\", \"n\": 129}")));
        assertEquals(400, statusOf(() -> resource.completions(
                "{\"prompt\": \"x\", \"logit_bias\": {\"6\": -101}}")));
        assertEquals(400, statusOf(() -> resource.completions(
                "{\"prompt\": \"x\", \"logit_bias\": {\"cat\": -100}}")));
        assertEquals(400, statusOf(() -> resource.completions(
                "{\"prompt\": [1, 99999]}")), "a token id outside the vocabulary");
        assertEquals(400, statusOf(() -> resource.completions(
                "{\"prompt\": [\"a\", 7]}")), "a mixed prompt array");
        assertEquals(400, statusOf(() -> resource.completions(
                "{\"prompt\": \"x\", \"grammar\": \"root ::= \\\"a\\\"\", \"guided_choice\": [\"a\"]}")),
                "two output constraints");
        assertEquals(400, statusOf(() -> resource.completions(
                "{\"prompt\": \"x\", \"response_format\": {\"type\": \"yaml\"}}")));
        assertEquals(400, statusOf(() -> resource.completions(
                "{\"prompt\": [\"a\", \"b\", \"c\", \"d\", \"e\", \"f\", \"g\", \"h\", \"i\"], \"n\": 128}")),
                "more generations than one request may queue");
        assertEquals(400, statusOf(() -> resource.completions(
                "{\"prompt\": \"x\", \"stream\": \"yes\"}")), "a malformed flag refuses before anything runs");
        assertEquals(400, statusOf(() -> resource.chatCompletions(
                "{\"messages\": \"nope\"}")));
        assertEquals(400, statusOf(() -> resource.chatCompletions(
                "{\"messages\": [{\"role\": \"user\", \"content\": \"x\"}], \"tool_choice\": \"required\"}")),
                "tool_choice required needs tools");
        assertEquals(400, statusOf(() -> resource.chatCompletions(
                "{\"messages\": [{\"role\": \"user\", \"content\": [{\"type\": \"image_url\","
                        + " \"image_url\": {\"url\": \"data:,\"}}]}]}")), "media parts on the text path");
        WebApplicationException unloadedModels = waeOf(() -> new OpenAiResource(unloaded).models());
        assertEquals(503, unloadedModels.getResponse().getStatus());
    }

    @Test
    void logitBiasRunsThroughTheSamplerStage() {
        Response response = resource.completions("""
                {"prompt": "hello", "max_tokens": 4, "temperature": 0.9, "seed": 7,
                 "logit_bias": {"6": -100, "7": 2.5}}""");
        assertEquals(200, response.getStatus());
        assertNotNull(firstChoice(view(response.getEntity())).str("text"));
    }

    @Test
    void healthAndPropsReportTheServingSetup() {
        assertEquals("ok", view(llama.health().getEntity()).str("status"));
        JsonView props = view(llama.props().getEntity());
        assertTrue(props.str("model_path").endsWith(".gguf"));
        assertEquals(2L, props.i64("total_slots"));
        assertEquals(64L, JsonView.of(props.raw("default_generation_settings"), "d").i64("n_ctx"));
    }

    // ------------------------------------------------------------------ helpers

    private static JsonView firstChoice(JsonView body) {
        return JsonView.of(((java.util.List<?>) body.raw("choices")).get(0), "choices[0]");
    }

    private static JsonView view(Object entity) {
        return JsonView.of(Json.parse((String) entity, Json.Limits.UNTRUSTED), "body");
    }

    private static int statusOf(Runnable call) {
        try {
            call.run();
            return 200;
        } catch (WebApplicationException e) {
            return e.getResponse().getStatus();
        }
    }

    private static WebApplicationException waeOf(java.util.function.Supplier<?> call) {
        try {
            call.get();
            throw new AssertionError("expected a WebApplicationException");
        } catch (WebApplicationException e) {
            return e;
        }
    }
}

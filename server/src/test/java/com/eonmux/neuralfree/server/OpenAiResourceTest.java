package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.EngineConfig;
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

import static com.eonmux.neuralfree.server.ServerTestSupport.errorOf;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The OpenAI-compatible HTTP surface over the real service: the synthetic two-layer llama opens on a
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
        EngineConfig config = ServerTestSupport.pagedConfig(64, 2);
        service = new EngineService(model.toString(), config,
                ServerOptions.defaults().withTrustRequestChatTemplate(true));
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
        Response response = resource.models(null);
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
        JsonView body = view(response.getEntity());
        JsonView choice = firstChoice(body);
        JsonView logprobs = JsonView.of(choice.raw("logprobs"), "logprobs");
        List<?> tokens = (List<?>) logprobs.raw("tokens");
        List<?> tokenLogprobs = (List<?>) logprobs.raw("token_logprobs");
        List<?> tops = (List<?>) logprobs.raw("top_logprobs");
        List<?> offsets = (List<?>) logprobs.raw("text_offset");
        long promptTokens = body.obj("usage").i64("prompt_tokens");
        assertEquals(promptTokens + 3, tokens.size(),
                "the echoed prompt's entries come before the generated ones (vLLM's merge)");
        assertEquals(promptTokens + 3, tokenLogprobs.size());
        assertEquals(promptTokens + 3, tops.size());
        assertNull(tokenLogprobs.get(0), "no row predicts the first prompt token");
        assertNull(tops.get(0), "no alternatives for the first prompt token");
        assertEquals(0L, ((Number) offsets.get(0)).longValue(), "the merged arrays open at offset 0");
        for (int i = 1; i < tokens.size(); i++) {
            Map<?, ?> top = (Map<?, ?>) tops.get(i);
            assertTrue(top.size() >= 1 && top.size() <= 3, "the token and at most the requested alternatives");
            assertEquals(tokens.get(i), top.keySet().iterator().next(), "every entry lists its own token first");
            assertNotNull(tokenLogprobs.get(i), "every scored position carries its logprob");
            long step = ((Number) offsets.get(i)).longValue() - ((Number) offsets.get(i - 1)).longValue();
            assertEquals(((String) tokens.get(i - 1)).length(), step, "offsets advance by the token text");
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
    void guidedRegexWithShorthandClassesConstrainsTheAnswer() {
        for (String constraint : List.of("\"guided_regex\": \"\\\\d{3}-\\\\d{4}\"",
                "\"structured_outputs\": {\"regex\": \"\\\\d{3}-\\\\d{4}\"}")) {
            Response response = resource.completions("{\"prompt\": \"call\", \"max_tokens\": 16,"
                    + " \"temperature\": 0, " + constraint + "}");
            assertEquals(200, response.getStatus());
            JsonView choice = firstChoice(view(response.getEntity()));
            String text = choice.str("text");
            assertTrue(text.matches("[0-9]{3}-[0-9]{4}"), constraint + " forces the pattern, got " + text);
            assertEquals("stop", choice.str("finish_reason"), "the full match ends the output");
        }
    }

    @Test
    void aLarkGuidedGrammarConstrainsTheAnswerAsVllmDetectsIt() {
        String lark = "start: \\\"id-\\\" DIGIT DIGIT\\nDIGIT: /[0-9]/";
        for (String constraint : List.of("\"guided_grammar\": \"" + lark + "\"",
                "\"structured_outputs\": {\"grammar\": \"" + lark + "\"}")) {
            Response response = resource.completions("{\"prompt\": \"name\", \"max_tokens\": 16,"
                    + " \"temperature\": 0, " + constraint + "}");
            assertEquals(200, response.getStatus());
            JsonView choice = firstChoice(view(response.getEntity()));
            String text = choice.str("text");
            assertTrue(text.matches("id-[0-9]{2}"), constraint + " forces the grammar, got " + text);
            assertEquals("stop", choice.str("finish_reason"), "the full match ends the output");
        }
        String broken = "{\"prompt\": \"x\", \"max_tokens\": 4, \"guided_grammar\": \"start: missing\"}";
        assertEquals(400, statusOf(() -> resource.completions(broken)));
        String message = errorOf(() -> resource.completions(broken));
        assertTrue(message.contains("unknown name 'missing' (line 1, column 8)"), message);
    }

    @Test
    void anUnsupportedGuidedRegexIsRefusedNamingTheConstruct() {
        String request = "{\"prompt\": \"x\", \"max_tokens\": 4, \"guided_regex\": \"\\\\bfoo\"}";
        assertEquals(400, statusOf(() -> resource.completions(request)));
        String message = errorOf(() -> resource.completions(request));
        assertTrue(message.contains("regex constraint") && message.contains("word boundary assertion: \\b"),
                message);
    }

    @Test
    void aStructuralTagResponseFormatInXgrammarsFormConstrainsTheAnswer() {
        Response response = resource.completions("""
                {"prompt": "pick one", "max_tokens": 8, "temperature": 0,
                 "response_format": {"type": "structural_tag", "format": {"type": "or", "elements": [
                   {"type": "const_string", "value": "yes"}, {"type": "const_string", "value": "no"}]}}}""");
        assertEquals(200, response.getStatus());
        String text = firstChoice(view(response.getEntity())).str("text");
        assertTrue(text.equals("yes") || text.equals("no"), "the tag forces one choice, got " + text);
        // the deprecated qwen_xml_parameter format is the qwen_xml style, so it compiles and serves
        Response qwen = resource.completions("""
                {"prompt": "x", "max_tokens": 8, "temperature": 0, "response_format":
                 {"type": "structural_tag", "format": {"type": "qwen_xml_parameter", "json_schema": {}}}}""");
        assertEquals(200, qwen.getStatus());
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
        WebApplicationException unloadedModels = waeOf(() -> new OpenAiResource(unloaded).models(null));
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
        JsonView props = view(llama.props(null, null).getEntity());
        assertTrue(props.str("model_path").endsWith(".gguf"));
        assertEquals(2L, props.i64("total_slots"));
        assertEquals(64L, JsonView.of(props.raw("default_generation_settings"), "d").i64("n_ctx"));
        // the built-in ChatML template: string content only, no tools and no tool calls
        JsonView caps = JsonView.of(props.raw("chat_template_caps"), "chat_template_caps");
        assertEquals(Boolean.TRUE, caps.raw("supports_string_content"));
        assertEquals(Boolean.FALSE, caps.raw("supports_tool_calls"));
        assertEquals(Boolean.FALSE, caps.raw("supports_tools"));
        assertFalse(service.defaultParallelToolCalls(), "a template that renders no tool calls fails llama.cpp's"
                + " parallel probe, so parallel_tool_calls defaults to one call");
    }

    // ------------------------------------------------------------------ per-request controls

    @Test
    void anUnknownReasoningFormatIsRefusedByName() {
        assertEquals(400, statusOf(() -> resource.chatCompletions(
                "{\"messages\": [{\"role\": \"user\", \"content\": \"x\"}], \"reasoning_format\": \"sql\"}")));
        assertTrue(errorOf(() -> resource.chatCompletions(
                "{\"messages\": [{\"role\": \"user\", \"content\": \"x\"}], \"reasoning_format\": \"sql\"}"))
                .contains("deepseek-legacy"));
        assertEquals(400, statusOf(() -> resource.completions(
                "{\"prompt\": \"x\", \"max_tokens\": 1, \"reasoning_format\": \"sql\"}")),
                "the native completion takes the same field");
        assertEquals(200, statusOf(() -> resource.chatCompletions(
                "{\"messages\": [{\"role\": \"user\", \"content\": \"x\"}], \"max_tokens\": 1,"
                        + " \"reasoning_format\": \"none\"}")));
    }

    @Test
    void malformedDocumentsAndOverridesAreRefusedByName() {
        assertEquals(400, statusOf(() -> resource.chatCompletions(
                "{\"messages\": [{\"role\": \"user\", \"content\": \"x\"}], \"documents\": \"doc\"}")));
        assertTrue(errorOf(() -> resource.chatCompletions(
                "{\"messages\": [{\"role\": \"user\", \"content\": \"x\"}], \"documents\": [\"doc\"]}"))
                .contains("documents[0]"));
        assertEquals(400, statusOf(() -> resource.chatCompletions(
                "{\"messages\": [{\"role\": \"user\", \"content\": \"x\"}], \"chat_template\": 3}")));
        assertEquals(400, statusOf(() -> resource.chatCompletions(
                "{\"messages\": [{\"role\": \"user\", \"content\": \"x\"}],"
                        + " \"chat_template\": \"{{ messages[ }}\"}")),
                "an override that does not parse");
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

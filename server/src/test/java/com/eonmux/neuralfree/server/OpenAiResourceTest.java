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

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The OpenAI-compatible HTTP surface over the real service (PARITY P2): the synthetic two-layer llama opens on a
 * paged {@code parallel = 2} config and the resources answer every protocol form — the model list, non-streamed
 * completions and chat completions with usage and finish reasons, SSE streams ending in {@code data: [DONE]},
 * guided_choice through the grammar route, the named refusals ({@code n} above 1, {@code logit_bias}, prompt
 * arrays) and the llama.cpp-native {@code /health} and {@code /props}. An unloaded service refuses with 503.
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
    void aChatCompletionAnswersWithAnAssistantMessage() {
        Response response = resource.chatCompletions("""
                {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 4, "temperature": 0}""");
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
                "{\"prompt\": \"x\", \"n\": 2}")));
        assertEquals(400, statusOf(() -> resource.completions(
                "{\"prompt\": \"x\", \"logit_bias\": {\"6\": -100}}")));
        assertEquals(400, statusOf(() -> resource.completions(
                "{\"prompt\": [\"a\", \"b\"]}")));
        assertEquals(400, statusOf(() -> resource.chatCompletions(
                "{\"messages\": \"nope\"}")));
        WebApplicationException unloadedModels = waeOf(() -> new OpenAiResource(unloaded).models());
        assertEquals(503, unloadedModels.getResponse().getStatus());
    }

    @Test
    void healthAndPropsReportTheServingSetup() {
        assertEquals("ok", view(llama.health().getEntity()).str("status"));
        JsonView props = view(llama.props().getEntity());
        assertTrue(props.str("model_path").endsWith(".gguf"));
        assertEquals(2L, props.i64("total_slots"));
        assertEquals(64L, JsonView.of(props.raw("default_generation_settings"), "d").i64("n_ctx"));
    }

    @Test
    void embeddingsServeFromAnEmbeddingModelAndRefuseAGenerativeOneByName() throws Exception {
        Path bert = SyntheticModels.bert().writeTempFile();
        EngineConfig config = EngineConfig.builder().build();
        try (EngineService embedService = new EngineService(bert.toString(), config)) {
            Response response;
            try {
                response = new OpenAiResource(embedService).embeddings("""
                        {"input": ["hello world", "second"]}""");
            } catch (WebApplicationException e) {
                throw new AssertionError("embeddings refused: " + body(e.getResponse()), e);
            }
            assertEquals(200, response.getStatus());
            JsonView body = view(response.getEntity());
            assertEquals("list", body.str("object"));
            List<?> data = (List<?>) body.raw("data");
            assertEquals(2, data.size());
            JsonView first = JsonView.of(data.get(0), "data[0]");
            assertEquals("embedding", first.str("object"));
            assertEquals(0L, first.i64("index"));
            assertTrue(((List<?>) first.raw("embedding")).size() > 8,
                    "the pooled vector is non-trivial");
        }
        // the generative llama refuses the embedding endpoint with its named pooling refusal
        WebApplicationException refused = assertThrows(WebApplicationException.class,
                () -> resource.embeddings("{\"input\": \"hello\"}"));
        assertEquals(400, refused.getResponse().getStatus());
        assertTrue(view(refused.getResponse().getEntity()).obj("error").str("message").contains("pool"),
                "the pooling refusal names the model");
    }

    @Test
    void rerankRefusesAGenerativeModelByNameAndValidatesItsBody() {
        // the synthetic llama pools by NONE: the rerank forward refuses with the model's own refusal
        WebApplicationException refused = assertThrows(WebApplicationException.class,
                () -> resource.rerank("{\"query\": \"q\", \"documents\": [\"a\", \"b\"]}"));
        assertEquals(400, refused.getResponse().getStatus());
        assertTrue(view(refused.getResponse().getEntity()).obj("error").str("message").contains("pool"),
                "the pooling refusal names the model");
        assertEquals(400, rerankStatus("{\"query\": \"q\"}"));
        assertEquals(400, rerankStatus("{\"query\": \"q\", \"documents\": [42]}"));
    }

    /** The response entity as text (JSON bodies are strings; streams render their class name). */
    private static String body(Response response) {
        Object entity = response.getEntity();
        return entity instanceof String s ? s : String.valueOf(entity);
    }

    /** The status of a rerank call whose engine refusal surfaces as HTTP 400. */
    private int rerankStatus(String body) {
        try {
            return resource.rerank(body).getStatus();
        } catch (WebApplicationException e) {
            return e.getResponse().getStatus();
        } catch (IOException e) {
            throw new AssertionError(e);
        }
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

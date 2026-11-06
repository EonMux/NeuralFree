package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.utils.json.Json;
import com.eonmux.neuralfree.utils.json.JsonView;
import jakarta.ws.rs.core.Response;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static com.eonmux.neuralfree.server.ServerTestSupport.at;
import static com.eonmux.neuralfree.server.ServerTestSupport.dataFrames;
import static com.eonmux.neuralfree.server.ServerTestSupport.errorOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.size;
import static com.eonmux.neuralfree.server.ServerTestSupport.statusOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.streamText;
import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Cohere's {@code /cohere/v2/chat} over the real service, against vLLM's protocol: the non-streamed answer's
 * finish reason, thinking/text blocks and usage, the stream's {@code message-start ... message-end [DONE]} sequence
 * with the same text, the mapped tool calls, the per-token logprobs, and the named refusals. The plain synthetic
 * llama renders the same prompt for the Cohere body and the equivalent OpenAI chat body, so greedily sampled text
 * is comparable between them.
 */
class CohereChatResourceTest {

    private static final String TEMPLATE = "{%- if tools %}<|im_start|>system\nTools:"
            + "{% for t in tools %} {{ t.function.name }}{% endfor %}. Call one inside <tool_call></tool_call>."
            + "<|im_end|>\n{% endif %}"
            + "{%- for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
            + "{%- if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";

    private static final String TOOLS = """
            [{"type": "function", "function": {"name": "get_weather", "description": "Weather of a city",
              "parameters": {"type": "object", "properties": {"city": {"type": "string", "enum": ["Paris", "Rome"]}},
                             "required": ["city"], "additionalProperties": false}}}]""";

    private static EngineService service;
    private static EngineService tools;
    private static EngineService unloaded;
    private static CohereChatResource cohere;
    private static CohereChatResource toolCohere;
    private static CohereChatResource unloadedCohere;
    private static OpenAiResource openAi;

    @BeforeAll
    static void open() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        service = new EngineService(model.toString(), ServerTestSupport.pagedConfig(1024, 2));
        SyntheticModels.Config config = SyntheticModels.Config.llama().toBuilder().nCtxTrain(2048).build();
        Path toolModel = SyntheticModels.llama(config).str("tokenizer.chat_template", TEMPLATE).writeTempFile();
        tools = new EngineService(toolModel.toString(), ServerTestSupport.pagedConfig(1024, 2));
        unloaded = new EngineService(null, ServerTestSupport.pagedConfig(64, 2));
        cohere = new CohereChatResource(service);
        toolCohere = new CohereChatResource(tools);
        unloadedCohere = new CohereChatResource(unloaded);
        openAi = new OpenAiResource(service);
    }

    @AfterAll
    static void close() {
        service.close();
        tools.close();
        unloaded.close();
    }

    private static JsonView ok(Response response) {
        assertEquals(200, response.getStatus(), () -> String.valueOf(response.getEntity()));
        return view(response.getEntity());
    }

    private static String chat(String extra) {
        return "{\"model\": \"m\", \"messages\": [{\"role\": \"user\", \"content\": \"hello\"}],"
                + " \"max_tokens\": 5, \"temperature\": 0" + extra + "}";
    }

    /** The text of the message's {@code text} content blocks, joined. */
    private static String text(JsonView response) {
        Object raw = response.obj("message").raw("content");
        if (raw == null) {
            return "";
        }
        StringBuilder out = new StringBuilder();
        for (Object block : (List<?>) raw) {
            if (block instanceof java.util.Map<?, ?> map && "text".equals(map.get("type"))) {
                out.append(map.get("text"));
            }
        }
        return out.toString();
    }

    @Test
    @Timeout(120)
    void aChatAnswersLikeTheEquivalentOpenAiChatAndCarriesCohereUsage() {
        JsonView body = ok(cohere.chat(chat("")));
        assertEquals("assistant", body.obj("message").str("role"));
        assertTrue(body.str("id").startsWith("chatcmpl-"));
        String expected = at(view(openAi.chatCompletions("{\"messages\": [{\"role\": \"user\", \"content\":"
                + " \"hello\"}], \"max_tokens\": 5, \"temperature\": 0}").getEntity()), "choices", 0)
                .obj("message").str("content");
        assertEquals(expected, text(body), "the same prompt and sampler as the OpenAI chat route");
        assertTrue(List.of("COMPLETE", "MAX_TOKENS").contains(body.str("finish_reason")));
        JsonView usage = body.obj("usage");
        assertEquals(usage.obj("tokens").i64("input_tokens"), usage.obj("billed_units").i64("input_tokens"));
        assertEquals(usage.obj("tokens").i64("output_tokens"), usage.obj("billed_units").i64("output_tokens"));
        assertTrue(usage.i64("cached_tokens") >= 0);
        assertTrue(usage.obj("tokens").i64("input_tokens") >= 1);
        assertTrue(usage.obj("tokens").i64("output_tokens") >= 0);
    }

    @Test
    @Timeout(120)
    void aStreamCarriesTheBlockSequenceAndTheSameTextThenDone() throws IOException {
        String stream = streamText(cohere.chat(chat(", \"stream\": true")));
        assertTrue(stream.endsWith("data: [DONE]\n\n"), "Cohere clients break on the terminator");
        assertFalse(stream.contains("event:"), "Cohere frames are bare data lines");
        List<JsonView> frames = dataFrames(stream);
        assertEquals("message-start", frames.get(0).str("type"));
        assertEquals("assistant", frames.get(0).obj("delta").obj("message").str("role"));
        StringBuilder assembled = new StringBuilder();
        boolean started = false;
        boolean ended = false;
        for (JsonView frame : frames) {
            if ("content-start".equals(frame.str("type"))) {
                started = true;
            } else if ("content-delta".equals(frame.str("type"))) {
                assembled.append(frame.obj("delta").obj("message").obj("content").str("text"));
            } else if ("content-end".equals(frame.str("type"))) {
                assertTrue(started);
            } else if ("message-end".equals(frame.str("type"))) {
                ended = true;
                assertTrue(List.of("COMPLETE", "MAX_TOKENS").contains(
                        frame.obj("delta").str("finish_reason")));
                assertTrue(frame.obj("delta").obj("usage").obj("tokens").i64("input_tokens") >= 1);
            }
        }
        assertTrue(started && ended);
        assertEquals(text(ok(cohere.chat(chat("")))), assembled.toString(),
                "the streamed blocks assemble the non-streamed message");
    }

    @Test
    @Timeout(180)
    void toolChoiceRequiredAnswersACallWithCohereShapeAndToolCallFinish() {
        String body = "{\"model\": \"m\", \"messages\": [{\"role\": \"user\", \"content\": \"Weather in Paris?\"}],"
                + " \"max_tokens\": 600, \"temperature\": 0, \"tools\": " + TOOLS + ","
                + " \"tool_choice\": \"REQUIRED\"}";
        JsonView response = ok(toolCohere.chat(body));
        assertEquals("TOOL_CALL", response.str("finish_reason"));
        assertNull(response.obj("message").raw("content"), "the whole output was the call");
        assertEquals(1, size(response.obj("message"), "tool_calls"));
        JsonView call = at(response.obj("message"), "tool_calls", 0);
        assertEquals("function", call.str("type"));
        assertTrue(call.str("id").startsWith("call_"));
        assertEquals("get_weather", call.obj("function").str("name"));
        JsonView arguments = JsonView.of(Json.parse(call.obj("function").str("arguments"),
                Json.Limits.UNTRUSTED), "arguments");
        assertTrue(List.of("Paris", "Rome").contains(arguments.str("city")));
    }

    @Test
    @Timeout(180)
    void aStreamedToolCallOpensAndClosesItsBlock() throws IOException {
        String stream = streamText(toolCohere.chat("{\"model\": \"m\", \"messages\": [{\"role\": \"user\","
                + " \"content\": \"Weather in Paris?\"}], \"max_tokens\": 600, \"temperature\": 0,"
                + " \"stream\": true, \"tools\": " + TOOLS + ", \"tool_choice\": \"REQUIRED\"}"));
        List<JsonView> frames = dataFrames(stream);
        boolean start = false;
        boolean end = false;
        boolean arguments = false;
        for (JsonView frame : frames) {
            String type = frame.str("type");
            if ("tool-call-start".equals(type)) {
                start = true;
                assertEquals("get_weather",
                        frame.obj("delta").obj("message").obj("tool_calls").obj("function").str("name"));
            } else if ("tool-call-delta".equals(type)) {
                arguments |= !frame.obj("delta").obj("message").obj("tool_calls").obj("function")
                        .str("arguments").isEmpty();
            } else if ("tool-call-end".equals(type)) {
                end = true;
            } else if ("message-end".equals(type)) {
                assertEquals("TOOL_CALL", frame.obj("delta").str("finish_reason"));
            }
        }
        assertTrue(start && end && arguments);
    }

    @Test
    @Timeout(120)
    void logprobsListEachGeneratedTokenWithItsIdAndProbability() {
        JsonView body = ok(cohere.chat(chat(", \"logprobs\": true")));
        List<?> items = (List<?>) body.raw("logprobs");
        assertTrue(items.size() >= 1);
        for (Object item : items) {
            JsonView entry = JsonView.of(item, "logprobs");
            assertTrue(entry.str("text") != null);
            assertEquals(1, ((List<?>) entry.raw("token_ids")).size());
            assertEquals(1, ((List<?>) entry.raw("logprobs")).size());
            assertTrue(((Number) ((List<?>) entry.raw("logprobs")).get(0)).doubleValue() <= 0.0);
        }
        assertNull(ok(cohere.chat(chat(""))).raw("logprobs"), "no capture without the field");
    }

    @Test
    @Timeout(120)
    void refusalsNameTheFieldAndTheUnloadedEngineIsUnavailable() {
        assertEquals(400, statusOf(() -> cohere.chat("{\"messages\": []}")));
        assertTrue(errorOf(() -> cohere.chat("{\"model\": \"m\", \"messages\": []}")).contains("messages"));
        assertTrue(errorOf(() -> cohere.chat("{\"model\": \"m\", \"messages\": [{\"role\": \"user\","
                + " \"content\": \"hi\"}], \"safety_mode\": \"STRICT\"}")).contains("safety_mode"));
        assertTrue(errorOf(() -> cohere.chat("{\"model\": \"m\", \"messages\": [{\"role\": \"user\","
                + " \"content\": \"hi\"}], \"tool_choice\": \"REQUIRED\"}")).contains("REQUIRED"));
        assertTrue(errorOf(() -> toolCohere.chat("{\"model\": \"m\", \"messages\": [{\"role\": \"user\","
                + " \"content\": \"hi\"}], \"tools\": [{\"type\": \"function\","
                + " \"function\": {\"name\": \"f\"}}], \"tool_choice\": \"AUTO\"}")).contains("tool_choice"));
        assertEquals(503, statusOf(() -> unloadedCohere.chat(chat(""))));
    }
}

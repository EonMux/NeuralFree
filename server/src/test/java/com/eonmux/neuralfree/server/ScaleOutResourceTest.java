package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.chat.ChatRequest;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.utils.json.Json;
import com.eonmux.neuralfree.utils.json.JsonView;
import jakarta.ws.rs.core.Response;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static com.eonmux.neuralfree.server.ServerTestSupport.errorOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.size;
import static com.eonmux.neuralfree.server.ServerTestSupport.statusOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The vLLM render endpoints and the prompt token-count routes: the rendered token ids and sampling parameters of a
 * chat, completions, Anthropic messages and Responses request, the parse context a derender step reads, and the
 * refusals that name what a render cannot represent.
 */
public class ScaleOutResourceTest {

    private static EngineService service;
    private static ScaleOutResource resource;

    @BeforeAll
    static void open() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        service = new EngineService(model.toString(), ServerTestSupport.pagedConfig(64, 2));
        resource = new ScaleOutResource(service);
    }

    @AfterAll
    static void close() {
        service.close();
    }

    @Test
    void chatRenderAnswersTheRenderedTokensAndSamplingParameters() {
        String body = "{\"messages\": [{\"role\": \"user\", \"content\": \"hello there\"}],"
                + " \"max_tokens\": 8, \"temperature\": 0.5, \"n\": 2}";
        JsonView answer = view(resource.renderChat(body).getEntity());
        assertTrue(answer.raw("token_ids") instanceof List<?>);
        assertEquals(renderedChatTokens(body).length, size(answer, "token_ids"));
        for (int i = 0; i < size(answer, "token_ids"); i++) {
            assertEquals(renderedChatTokens(body)[i], ((Number) ((List<?>) answer.raw("token_ids")).get(i)).intValue());
        }
        JsonView sampling = answer.obj("sampling_params");
        assertEquals(8L, sampling.i64("max_tokens"));
        assertEquals(2L, sampling.i64("n"));
        assertEquals(0.5, sampling.f64("temperature"), 1e-6);
        assertTrue(sampling.bool("detokenize"));
        assertEquals("final_only", sampling.str("output_kind"));
        JsonView context = answer.obj("parse_context");
        assertEquals(0, size(context, "tools"));
        assertEquals("none", context.str("tool_choice"));
        assertTrue(context.bool("parallel_tool_calls"));
        assertTrue(context.bool("include_reasoning"));
        assertEquals("chatcmpl-", answer.str("request_id").substring(0, "chatcmpl-".length()));
    }

    @Test
    void aRenderRoundTripsThroughTheGenerateEndpoint() {
        String chat = "{\"messages\": [{\"role\": \"user\", \"content\": \"hello\"}], \"max_tokens\": 3,"
                + " \"temperature\": 0}";
        JsonView rendered = view(resource.renderChat(chat).getEntity());
        StringBuilder generate = new StringBuilder("{\"request_id\": \"round-trip\", \"token_ids\": ");
        generate.append(ServerJson.write(rendered.raw("token_ids")));
        generate.append(", \"sampling_params\": ").append(ServerJson.write(rendered.raw("sampling_params")));
        generate.append(", \"output_mode\": \"tokens\"}");
        JsonView response = view(resource.generate(generate.toString()).getEntity());
        assertEquals("tokens", response.str("output_mode"));
        assertTrue(response.raw("prompt_token_ids") == null,
                "return_token_ids is off, so the generate answer carries no prompt ids");
        assertTrue(response.obj("usage").i64("prompt_tokens") > 0);
        assertEquals(1, size(response, "choices"));
    }

    /**
     * The per-request {@code watermarking} switch travels in the render answer's {@code sampling_params}, and the
     * generate endpoint accepts it back, so an opt-out survives the split between the two servers.
     */
    @Test
    void aRenderCarriesTheWatermarkingSwitchToTheGenerateEndpoint() {
        String chat = "{\"messages\": [{\"role\": \"user\", \"content\": \"hello\"}], \"max_tokens\": 3,"
                + " \"temperature\": 0, \"watermarking\": false}";
        JsonView rendered = view(resource.renderChat(chat).getEntity());
        assertFalse(rendered.obj("sampling_params").bool("watermarking"));
        StringBuilder generate = new StringBuilder("{\"request_id\": \"watermark-round-trip\", \"token_ids\": ");
        generate.append(ServerJson.write(rendered.raw("token_ids")));
        generate.append(", \"sampling_params\": ").append(ServerJson.write(rendered.raw("sampling_params")));
        generate.append(", \"output_mode\": \"tokens\"}");
        JsonView response = view(resource.generate(generate.toString()).getEntity());
        assertEquals("tokens", response.str("output_mode"));
        assertTrue(response.obj("usage").i64("prompt_tokens") > 0);
    }

    @Test
    void completionsRenderAnswersOneRequestPerPrompt() {
        Response response = resource.renderCompletions(
                "{\"prompt\": [\"one\", \"two\"], \"max_tokens\": 4, \"logprobs\": 2}");
        List<?> requests = (List<?>) Json.parse(String.valueOf(response.getEntity()), Json.Limits.UNTRUSTED);
        assertEquals(2, requests.size());
        JsonView first = JsonView.of(requests.get(0), "answers[0]");
        assertEquals(service.encode("one", true, true).length, size(first, "token_ids"));
        assertTrue(first.str("request_id").endsWith("-0"));
        JsonView second = JsonView.of(requests.get(1), "answers[1]");
        assertTrue(second.str("request_id").endsWith("-1"));
        assertEquals(4L, first.obj("sampling_params").i64("max_tokens"));
        assertTrue(first.obj("sampling_params").i64("logprobs") >= 0);
    }

    @Test
    void oneCompletionPromptKeepsTheBareRequestId() {
        List<?> requests = (List<?>) Json.parse(String.valueOf(resource.renderCompletions(
                "{\"prompt\": \"one\", \"max_tokens\": 4}").getEntity()), Json.Limits.UNTRUSTED);
        assertEquals(1, requests.size());
        assertFalse(JsonView.of(requests.get(0), "answers[0]").str("request_id").endsWith("-0"));
    }

    @Test
    void messagesRenderUsesTheAnthropicConversion() {
        String body = "{\"model\": \"test\", \"max_tokens\": 5, \"temperature\": 0.2,"
                + " \"messages\": [{\"role\": \"user\", \"content\": \"hi\"}]}";
        JsonView answer = view(resource.renderMessages(body).getEntity());
        ChatRequest chat = AnthropicRequests.chat(JsonView.of(Json.parse(body, Json.Limits.UNTRUSTED), "body"),
                service.options());
        int[] expected = service.encode(service.formatChat(chat, false), false, true);
        assertEquals(expected.length, size(answer, "token_ids"));
        assertEquals(5L, answer.obj("sampling_params").i64("max_tokens"));
        assertEquals(0.2, answer.obj("sampling_params").f64("temperature"), 1e-6);
    }

    @Test
    void responsesRenderConvertsToAChatPrompt() {
        JsonView answer = view(resource.renderResponses("{\"input\": \"hello\", \"max_output_tokens\": 6}")
                .getEntity());
        Map<String, Object> chatBody = ResponsesRequests.chatBody(
                JsonView.of(Json.parse("{\"input\": \"hello\", \"max_output_tokens\": 6}", Json.Limits.UNTRUSTED),
                        "body"), List.of());
        ChatRequest chat = OpenAiRequests.chat(JsonView.of(chatBody, "body"), service.options());
        assertEquals(service.encode(service.formatChat(chat, false), false, true).length, size(answer, "token_ids"));
        assertEquals(6L, answer.obj("sampling_params").i64("max_tokens"));
        assertEquals(400, statusOf(() -> resource.renderResponses(
                "{\"input\": \"hello\", \"previous_response_id\": \"resp_1\"}")));
        assertTrue(errorOf(() -> resource.renderResponses(
                "{\"input\": \"hello\", \"previous_response_id\": \"resp_1\"}")).contains("previous_response_id"));
    }

    @Test
    void aRenderRefusesBestOfAndUnrepresentableInputsByTheFieldTheyComeFrom() {
        String bestOf = "{\"messages\": [{\"role\": \"user\", \"content\": \"hi\"}], \"n\": 1, \"best_of\": 3}";
        assertEquals(400, statusOf(() -> resource.renderChat(bestOf)));
        assertTrue(errorOf(() -> resource.renderChat(bestOf)).contains("best_of"));
        // an embedding part is a media prompt and renders into serialized rows without a projector
        String embeds = "{\"messages\": [{\"role\": \"user\", \"content\": [{\"type\": \"image_embeds\","
                + " \"image_embeds\": " + embedRow() + "}]}]}";
        JsonView embedded = view(resource.renderChat(embeds).getEntity());
        assertEquals(1, size(embedded.obj("features").obj("mm_hashes"), "embeddings"));
        List<?> ranges = (List<?>) embedded.obj("features").obj("mm_placeholders").raw("embeddings");
        assertEquals(1, ranges.size());
        assertEquals(1L, ((Number) ((Map<?, ?>) ranges.get(0)).get("length")).longValue(),
                "the one embedding row is one placeholder row");
        assertTrue(embedded.obj("features").obj("kwargs_data").raw("embeddings") instanceof List<?>);
        assertEquals(400, statusOf(() -> resource.renderCompletions(
                "{\"prompt\": \"hi\", \"prompt_embeds\": " + embedRow() + "}")));
        assertTrue(errorOf(() -> resource.renderCompletions(
                "{\"prompt\": \"hi\", \"prompt_embeds\": " + embedRow() + "}")).contains("prompt_embeds"));
        // an image part needs a projector, refused with llama.cpp's hint
        String image = "{\"model\": \"test\", \"max_tokens\": 5, \"messages\": [{\"role\": \"user\", \"content\":"
                + " [{\"type\": \"image\", \"source\": {\"type\": \"base64\", \"media_type\": \"image/png\","
                + " \"data\": \"AAAA\"}}]}]}";
        assertEquals(400, statusOf(() -> resource.renderMessages(image)));
        assertTrue(errorOf(() -> resource.renderMessages(image)).contains("image input is not supported"));
    }

    @Test
    void theChatAndResponsesTokenCountsAnswerTheRenderedRows() {
        OpenAiResource openAi = new OpenAiResource(service);
        String body = "{\"messages\": [{\"role\": \"user\", \"content\": \"hello\"}]}";
        JsonView chat = view(openAi.chatInputTokens(body).getEntity());
        assertEquals("response.input_tokens", chat.str("object"));
        assertEquals(renderedChatTokens(body).length, chat.i64("input_tokens"));
        JsonView responses = view(new ResponsesResource(service).inputTokens("{\"input\": \"hello\"}")
                .getEntity());
        assertEquals("response.input_tokens", responses.str("object"));
        assertTrue(responses.i64("input_tokens") >= 1);
    }

    /** The tokens of {@code body}'s chat prompt, rendered exactly as the chat endpoint renders them. */
    private static int[] renderedChatTokens(String body) {
        JsonView view = JsonView.of(Json.parse(body, Json.Limits.UNTRUSTED), "body");
        ChatRequest chat = OpenAiRequests.chat(view, service.options());
        return service.encode(service.formatChat(chat, false), false, true);
    }

    /** One embedding row of the model's width, as JSON. */
    private static String embedRow() {
        StringBuilder row = new StringBuilder("[");
        for (int i = 0; i < service.embeddingLength(); i++) {
            row.append(i == 0 ? "0" : ", 0");
        }
        return row.append(']').toString();
    }
}

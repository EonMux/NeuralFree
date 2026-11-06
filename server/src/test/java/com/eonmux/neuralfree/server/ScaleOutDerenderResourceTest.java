package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.utils.json.JsonView;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static com.eonmux.neuralfree.server.ServerTestSupport.at;
import static com.eonmux.neuralfree.server.ServerTestSupport.errorOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.size;
import static com.eonmux.neuralfree.server.ServerTestSupport.statusOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The derender endpoints over the real service: a generate response's tokens turned back into a chat completion
 * (optionally parsed against the original chat request) and into completions, the stateless streaming form that
 * carries one chunk plus its client-held state per call, the usage, the seeded detokenization, the logprobs
 * translation and the named refusals.
 */
public class ScaleOutDerenderResourceTest {

    private static EngineService service;
    private static ScaleOutResource resource;

    @BeforeAll
    static void open() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        service = new EngineService(model.toString(), ServerTestSupport.pagedConfig(128, 2));
        resource = new ScaleOutResource(service);
    }

    @AfterAll
    static void close() {
        service.close();
    }

    @Test
    void chatDerenderAnswersTheDetokenizedMessageAndTheUsage() {
        int[] tokens = service.encode("hello world", false, true);
        JsonView body = view(resource.derenderChat("{\"generate_response\": " + generateResponse(tokens, null)
                + ", \"prompt_tokens\": 7}").getEntity());
        assertEquals("chat.completion", body.str("object"));
        assertEquals(1, size(body, "choices"));
        JsonView choice = at(body, "choices", 0);
        assertEquals("assistant", choice.obj("message").str("role"));
        assertEquals(service.decode(tokens), choice.obj("message").str("content"));
        assertNull(choice.raw("logprobs"));
        JsonView usage = body.obj("usage");
        assertEquals(7L, usage.i64("prompt_tokens"));
        assertEquals(tokens.length, usage.i64("completion_tokens"));
        assertEquals(7L + tokens.length, usage.i64("total_tokens"));
        assertTrue(body.str("id").startsWith("chatcmpl-"));
    }

    @Test
    void promptTokenIdsSeedTheDetokenization() {
        int[] prompt = service.encode("a prompt", true, true);
        int[] tokens = service.encode("there", false, true);
        JsonView body = view(resource.derenderChat("{\"generate_response\": " + generateResponse(tokens, prompt)
                + ", \"prompt_token_ids\": " + json(prompt) + "}").getEntity());
        String expected = ScaleOutDerender.detokenize(service, tokens, prompt);
        assertEquals(expected, at(body, "choices", 0).obj("message").str("content"));
    }

    @Test
    void aChatRequestParsesTheAnswerThroughTheModelParsers() {
        int[] tokens = service.encode("hello there", false, true);
        String chatRequest = "{\"messages\": [{\"role\": \"user\", \"content\": \"hi\"}], \"stream\": false}";
        JsonView body = view(resource.derenderChat("{\"generate_response\": " + generateResponse(tokens, null)
                + ", \"chat_request\": " + chatRequest + "}").getEntity());
        JsonView message = at(body, "choices", 0).obj("message");
        assertEquals("assistant", message.str("role"));
        assertEquals(service.decode(tokens), message.str("content"));
    }

    @Test
    void completionsDerenderAnswersOneChoicePerGenerateResponse() {
        int[] first = service.encode("one", false, true);
        int[] second = service.encode("two", false, true);
        String body = "{\"generate_responses\": [" + generateResponse(first, null) + ", "
                + generateResponse(second, null) + "], \"prompt_tokens\": [3, 4]}";
        JsonView answer = view(resource.derenderCompletions(body).getEntity());
        assertEquals("text_completion", answer.str("object"));
        assertEquals(2, size(answer, "choices"));
        assertEquals(0L, at(answer, "choices", 0).i64("index"));
        assertEquals(1L, at(answer, "choices", 1).i64("index"));
        assertEquals(service.decode(first), at(answer, "choices", 0).str("text"));
        assertEquals(service.decode(second), at(answer, "choices", 1).str("text"));
        JsonView usage = answer.obj("usage");
        assertEquals(7L, usage.i64("prompt_tokens"));
        assertEquals(first.length + second.length, usage.i64("completion_tokens"));
    }

    @Test
    void integerLogprobsBecomeTheOpenAiShapes() {
        int[] tokens = service.encode("hi", false, true);
        String generate = generateResponseWithLogprobs(tokens, tokens[0]);
        JsonView body = view(resource.derenderChat("{\"generate_response\": " + generate + "}").getEntity());
        JsonView logprobs = at(body, "choices", 0).obj("logprobs");
        JsonView entry = at(logprobs, "content", 0);
        assertEquals(service.tokenText(tokens[0]), entry.str("token"));
        assertEquals(tokens.length, size(logprobs, "content"));
        assertTrue(entry.raw("top_logprobs") instanceof List<?>);
    }

    @Test
    void aGenerateResponseRoundTripsThroughGenerateAndDerender() {
        int[] prompt = service.encode("hello", true, true);
        String generated = String.valueOf(resource.generate("{\"token_ids\": " + json(prompt)
                + ", \"sampling_params\": {\"max_tokens\": 3, \"temperature\": 0, \"ignore_eos\": true},"
                + " \"return_token_ids\": true}").getEntity());
        String derender = "{\"generate_response\": " + generated
                + ", \"prompt_token_ids\": " + json(prompt) + ", \"prompt_tokens\": " + prompt.length
                + ", \"chat_request\": {\"messages\": [{\"role\": \"user\", \"content\": \"hello\"}]}}";
        JsonView answer = view(resource.derenderChat(derender).getEntity());
        JsonView choice = at(answer, "choices", 0);
        assertNotNull(choice.obj("message").str("content"));
        assertEquals(prompt.length, answer.obj("usage").i64("prompt_tokens"));
    }

    @Test
    void theDerenderRefusesMalformedBodiesByName() {
        int[] tokens = service.encode("hi", false, true);
        int[] prompt = service.encode("a prompt", true, true);
        assertEquals(400, statusOf(() -> resource.derenderChat("{\"generate_response\": "
                + generateResponse(tokens, prompt) + ", \"stream\": true}")));
        assertTrue(errorOf(() -> resource.derenderChat("{\"generate_response\": "
                + generateResponse(tokens, prompt) + ", \"stream\": true}")).contains("generate_chunk"),
                "a streamed call takes generate_chunk, not generate_response");
        assertEquals(400, statusOf(() -> resource.derenderChat(
                "{\"generate_response\": {\"choices\": [{\"index\": 0}]}}")));
        assertTrue(errorOf(() -> resource.derenderChat(
                "{\"generate_response\": {\"choices\": [{\"index\": 0}]}}")).contains("token_ids"));
        assertEquals(400, statusOf(() -> resource.derenderChat("{\"generate_response\": {"
                + "\"output_mode\": \"text\", " + choices(tokens) + "}}")));
        assertEquals(400, statusOf(() -> resource.derenderCompletions(
                "{\"generate_responses\": [" + generateResponse(tokens, null) + "], \"prompt_tokens\": [1, 2]}")));
    }

    @Test
    void streamingChatDerenderEmitsTheDeltaAndCarriesTheState() {
        int[] tokens = service.encode("hello world", false, true);
        int half = Math.max(1, tokens.length / 2);
        int[] first = java.util.Arrays.copyOfRange(tokens, 0, half);
        int[] second = java.util.Arrays.copyOfRange(tokens, half, tokens.length);
        JsonView answer = view(resource.derenderChat("{\"stream\": true, \"generate_chunk\": "
                + generateChunk(first, null) + "}").getEntity());
        JsonView chunk = answer.obj("chunk");
        assertEquals("chat.completion.chunk", chunk.str("object"));
        assertEquals("assistant", at(chunk, "choices", 0).obj("delta").str("role"));
        assertEquals(service.decode(first), at(chunk, "choices", 0).obj("delta").str("content"));
        JsonView state = answer.obj("stream_state");
        assertEquals(first.length, size(state, "output_token_ids"));
        assertEquals(1, size(state, "output_chunk_lens"));
        assertTrue(state.bool("role_sent"));

        JsonView second_ = view(resource.derenderChat("{\"stream\": true, \"generate_chunk\": "
                + generateChunk(second, "stop") + ", \"stream_state\": "
                + ServerJson.write(answer.raw("stream_state")) + "}").getEntity());
        JsonView delta = at(second_.obj("chunk"), "choices", 0).obj("delta");
        String all = service.decode(tokens);
        assertEquals(all.substring(service.decode(first).length()), delta.str("content"),
                "only the second chunk's text streams");
        assertNull(delta.raw("role"), "the role is sent once");
        assertEquals("stop", at(second_.obj("chunk"), "choices", 0).str("finish_reason"));
        assertEquals(tokens.length, size(second_.obj("stream_state"), "output_token_ids"));
        assertEquals(2, size(second_.obj("stream_state"), "output_chunk_lens"));
    }

    @Test
    void streamingCompletionsDerenderEmitsTheTextDelta() {
        int[] tokens = service.encode("hello world", false, true);
        int half = Math.max(1, tokens.length / 2);
        int[] first = java.util.Arrays.copyOfRange(tokens, 0, half);
        int[] second = java.util.Arrays.copyOfRange(tokens, half, tokens.length);
        JsonView answer = view(resource.derenderCompletions("{\"stream\": true, \"generate_chunk\": "
                + generateChunk(first, null) + "}").getEntity());
        JsonView chunk = answer.obj("chunk");
        assertEquals("text_completion", chunk.str("object"));
        assertEquals(service.decode(first), at(chunk, "choices", 0).str("text"));
        JsonView state = answer.obj("stream_state");
        JsonView next = view(resource.derenderCompletions("{\"stream\": true, \"generate_chunk\": "
                + generateChunk(second, "length") + ", \"stream_state\": "
                + ServerJson.write(answer.raw("stream_state")) + "}").getEntity());
        assertEquals(service.decode(tokens).substring(service.decode(first).length()),
                at(next.obj("chunk"), "choices", 0).str("text"));
        assertEquals("length", at(next.obj("chunk"), "choices", 0).str("finish_reason"));
        assertEquals(tokens.length, size(next.obj("stream_state"), "output_token_ids"));
    }

    @Test
    void aStreamingDerenderRefusesMalformedStateAndChunksByName() {
        int[] tokens = service.encode("hi", false, true);
        assertEquals(400, statusOf(() -> resource.derenderChat("{\"stream\": true, \"generate_chunk\": "
                + generateChunk(tokens, null) + ", \"stream_state\": {\"output_token_ids\": [1, 2],"
                + " \"output_chunk_lens\": [1]}}")));
        assertTrue(errorOf(() -> resource.derenderChat("{\"stream\": true, \"generate_chunk\": "
                + generateChunk(tokens, null) + ", \"stream_state\": {\"output_token_ids\": [1, 2],"
                + " \"output_chunk_lens\": [1]}}")).contains("output_chunk_lens"));
        assertEquals(400, statusOf(() -> resource.derenderChat("{\"stream\": true, \"generate_chunk\": "
                + "{\"choices\": [{\"token_ids\": []}, {\"token_ids\": []}]}}")));
        assertTrue(errorOf(() -> resource.derenderChat("{\"stream\": true, \"generate_chunk\": "
                + generateChunk(tokens, null) + ", \"stream_state\": {\"bogus\": 1}}")).contains("bogus"));
    }

    // ------------------------------------------------------------------ fixtures

    /** One generate tokens-mode chunk over {@code tokens} with its finish reason (or null). */
    private static String generateChunk(int[] tokens, String finish) {
        return "{\"request_id\": \"gen-1\", \"output_mode\": \"tokens\", \"choices\": [{\"index\": 0,"
                + " \"finish_reason\": " + (finish == null ? "null" : "\"" + finish + "\"")
                + ", \"token_ids\": " + json(tokens) + "}]}";
    }

    /** A generate tokens-mode response over {@code tokens}, with the prompt ids when given. */
    private static String generateResponse(int[] tokens, int[] promptIds) {
        StringBuilder out = new StringBuilder("{\"request_id\": \"gen-1\", \"output_mode\": \"tokens\", ");
        out.append(choices(tokens));
        if (promptIds != null) {
            out.append(", \"prompt_token_ids\": ").append(json(promptIds));
        }
        return out.append('}').toString();
    }

    /** One choice carrying the token sequence. */
    private static String choices(int[] tokens) {
        return "\"choices\": [{\"index\": 0, \"finish_reason\": \"stop\", \"stop_reason\": null,"
                + " \"token_ids\": " + json(tokens) + "}]";
    }

    /** A response whose first choice carries a tokens-shape logprobs capture. */
    private static String generateResponseWithLogprobs(int[] tokens, int tokenId) {
        StringBuilder content = new StringBuilder("[");
        for (int i = 0; i < tokens.length; i++) {
            content.append(i == 0 ? "" : ", ").append("{\"token_id\": ").append(tokens[i])
                    .append(", \"logprob\": -0.5, \"rank\": 1, \"top_logprobs\": [{\"token_id\": ")
                    .append(tokenId).append(", \"logprob\": -0.5, \"rank\": 1}]}");
        }
        content.append(']');
        return "{\"request_id\": \"gen-logprobs\", \"output_mode\": \"tokens\", \"choices\": [{\"index\": 0,"
                + " \"finish_reason\": \"stop\", \"token_ids\": " + json(tokens) + ", \"logprobs\": {\"content\": "
                + content + "}}]}";
    }

    /** A JSON value as parsed, for embedding in another body. */
    private static String json(int[] values) {
        StringBuilder out = new StringBuilder("[");
        for (int i = 0; i < values.length; i++) {
            out.append(i == 0 ? "" : ", ").append(values[i]);
        }
        return out.append(']').toString();
    }
}

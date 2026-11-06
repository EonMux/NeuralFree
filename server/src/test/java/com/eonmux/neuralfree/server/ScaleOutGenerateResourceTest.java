package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.utils.json.Json;
import com.eonmux.neuralfree.utils.json.JsonView;
import jakarta.ws.rs.core.Response;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static com.eonmux.neuralfree.server.ServerTestSupport.at;
import static com.eonmux.neuralfree.server.ServerTestSupport.dataFrames;
import static com.eonmux.neuralfree.server.ServerTestSupport.errorOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.size;
import static com.eonmux.neuralfree.server.ServerTestSupport.statusOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.streamText;
import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The token-in, token-out generate endpoint over the real service: token and text answers, parallel choices,
 * integer-id logprobs, streamed deltas with their usage chunks, the request-id abort route, and the fields the
 * endpoint refuses by name instead of ignoring.
 */
public class ScaleOutGenerateResourceTest {

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
    void tokensModeAnswersTheGeneratedIdsAndTheUsage() {
        JsonView body = view(resource.generate(generateBody(tokens("hello"), "\"max_tokens\": 3,"
                + " \"temperature\": 0, \"ignore_eos\": true", false)).getEntity());
        assertEquals("tokens", body.str("output_mode"));
        assertEquals(1, size(body, "choices"));
        JsonView choice = at(body, "choices", 0);
        int generated = size(choice, "token_ids");
        assertTrue(generated > 0, "ignore_eos generates the full budget");
        assertTrue(choice.str("finish_reason").equals("length") || choice.str("finish_reason").equals("stop"));
        JsonView usage = body.obj("usage");
        assertEquals(tokens("hello").length, usage.i64("prompt_tokens"));
        assertEquals(generated, usage.i64("completion_tokens"));
        assertEquals(usage.i64("prompt_tokens") + generated, usage.i64("total_tokens"));
        assertNull(body.raw("prompt_token_ids"), "return_token_ids was not asked");
    }

    @Test
    void returnTokenIdsAddsThePromptIds() {
        JsonView body = view(resource.generate(generateBody(tokens("hello"), "\"max_tokens\": 2,"
                + " \"temperature\": 0, \"ignore_eos\": true", true)).getEntity());
        int[] expected = tokens("hello");
        assertEquals(expected.length, size(body, "prompt_token_ids"));
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], ((Number) ((List<?>) body.raw("prompt_token_ids")).get(i)).intValue());
        }
    }

    @Test
    void textModeAnswersTheDetokenizedOutputBesideTheIds() {
        JsonView body = view(resource.generate(generateBody(tokens("hello"), "\"max_tokens\": 3,"
                + " \"temperature\": 0, \"ignore_eos\": true", false).replace("\"output_mode\": \"tokens\"",
                "\"output_mode\": \"text\"")).getEntity());
        assertEquals("text", body.str("output_mode"));
        JsonView choice = at(body, "choices", 0);
        assertNotNull(choice.str("text"));
        assertEquals(size(choice, "token_ids"), body.obj("usage").i64("completion_tokens"));
    }

    @Test
    void promptLogprobsScoreThePromptRows() {
        int[] prompt = tokens("hello");
        JsonView body = view(resource.generate(generateBody(prompt, "\"max_tokens\": 1,"
                + " \"temperature\": 0, \"prompt_logprobs\": 1", false)).getEntity());
        assertTrue(body.raw("prompt_logprobs") instanceof List<?>);
        assertEquals(prompt.length, size(body, "prompt_logprobs"));
        assertNull(((List<?>) body.raw("prompt_logprobs")).get(0), "no row predicts the first prompt token");
        JsonView row = at(body, "prompt_logprobs", 1);
        assertFalse(row.keys().isEmpty(), "the row scores its predicted token");
        Object first = row.raw(row.keys().iterator().next());
        assertTrue(first instanceof Map<?, ?>);
        assertTrue(((Map<?, ?>) first).get("logprob") instanceof Number);
    }

    @Test
    void parallelChoicesAnswerSideBySide() {
        JsonView body = view(resource.generate(generateBody(tokens("hello"), "\"max_tokens\": 2,"
                + " \"temperature\": 0, \"ignore_eos\": true, \"n\": 2", false)).getEntity());
        assertEquals(2, size(body, "choices"));
        assertEquals(0L, at(body, "choices", 0).i64("index"));
        assertEquals(1L, at(body, "choices", 1).i64("index"));
        assertEquals(size(at(body, "choices", 0), "token_ids") + size(at(body, "choices", 1), "token_ids"),
                body.obj("usage").i64("completion_tokens"));
    }

    @Test
    void logprobsCarryIntegerIdsInTokensMode() {
        JsonView body = view(resource.generate(generateBody(tokens("hello"), "\"max_tokens\": 2,"
                + " \"temperature\": 0, \"ignore_eos\": true, \"logprobs\": 2", false)).getEntity());
        JsonView choice = at(body, "choices", 0);
        JsonView logprobs = choice.obj("logprobs");
        assertEquals(size(choice, "token_ids"), size(logprobs, "content"));
        JsonView entry = at(logprobs, "content", 0);
        assertTrue(entry.i64("token_id") >= 0);
        assertTrue(entry.raw("logprob") instanceof Number);
        assertTrue(size(entry, "top_logprobs") >= 1);
        assertTrue(at(entry, "top_logprobs", 0).i64("token_id") >= 0);
    }

    @Test
    void aTokensStreamDeliversDeltasThenTheFinishAndTheUsage() throws IOException {
        String body = generateBody(tokens("hello"), "\"max_tokens\": 3, \"temperature\": 0,"
                + " \"ignore_eos\": true", true)
                .replace("\"stream\": false", "\"stream\": true, \"stream_options\": {\"include_usage\": true}");
        String stream = streamText(resource.generate(body));
        assertTrue(stream.trim().endsWith("[DONE]"));
        List<JsonView> frames = dataFrames(stream);
        List<Integer> ids = new ArrayList<>();
        JsonView finish = null;
        JsonView usage = null;
        boolean promptIds = false;
        for (JsonView frame : frames) {
            if (frame.raw("choices") instanceof List<?> choices && choices.isEmpty()) {
                usage = frame;
                continue;
            }
            if (frame.raw("prompt_token_ids") instanceof List<?> promptIdsOnFrame) {
                promptIds = promptIdsOnFrame.size() == tokens("hello").length;
            }
            JsonView choice = at(frame, "choices", 0);
            if (choice.raw("token_ids") instanceof List<?> chunk) {
                for (Object id : chunk) {
                    ids.add(((Number) id).intValue());
                }
            }
            if (choice.raw("finish_reason") != null) {
                finish = choice;
            }
        }
        assertEquals(3, ids.size(), "every generated token reaches the stream");
        assertTrue(promptIds, "return_token_ids stamps the prompt ids on the first chunk");
        assertNotNull(finish, "the stream names the stop decision");
        assertNotNull(usage, "include_usage adds the final usage chunk");
        assertEquals(3L, usage.obj("usage").i64("completion_tokens"));
    }

    @Test
    void aTextStreamSendsTheSameTextAsTheFullAnswer() throws IOException {
        String sampling = "\"max_tokens\": 4, \"temperature\": 0, \"ignore_eos\": true";
        JsonView full = view(resource.generate(generateBody(tokens("hello"), sampling, false).replace(
                "\"output_mode\": \"tokens\"", "\"output_mode\": \"text\"")).getEntity());
        String stream = streamText(resource.generate(generateBody(tokens("hello"), sampling, false)
                .replace("\"output_mode\": \"tokens\"", "\"output_mode\": \"text\"")
                .replace("\"stream\": false", "\"stream\": true")));
        StringBuilder text = new StringBuilder();
        for (JsonView frame : dataFrames(stream)) {
            if (frame.raw("choices") instanceof List<?> choices && !choices.isEmpty()) {
                JsonView choice = at(frame, "choices", 0);
                if (choice.raw("text") instanceof String piece) {
                    text.append(piece);
                }
            }
        }
        assertEquals(at(full, "choices", 0).str("text"), text.toString());
    }

    @Test
    void abortRequestsAnswersAnEmptyOkAndRefusesAMalformedBody() {
        Response response = resource.abortRequests("{\"request_ids\": [\"no-such-request\"]}");
        assertEquals(200, response.getStatus());
        assertNull(response.getEntity());
        assertEquals(400, statusOf(() -> resource.abortRequests("{}")));
        assertEquals(400, statusOf(() -> resource.abortRequests("{\"request_ids\": [1]}")));
    }

    @Test
    void unknownAndUnsupportedFieldsAreRefusedByName() {
        String valid = generateBody(tokens("hi"), "\"max_tokens\": 1", false);
        assertTrue(errorOf(() -> resource.generate(valid.replace("\"sampling_params\": {",
                "\"sampling_params\": {\"bogus\": 1, "))).contains("bogus"));
        assertTrue(errorOf(() -> resource.generate(valid.replace("{\"token_ids\"",
                "{\"features\": {}, \"token_ids\""))).contains("features.mm_hashes"));
        assertTrue(errorOf(() -> resource.generate(valid.replace("{\"token_ids\"",
                "{\"token_offsets\": [], \"token_ids\""))).contains("token_offsets"));
        // derender without a parse context is served as plain text on a server with no parser
        JsonView derived = view(resource.generate(valid.replace("\"output_mode\": \"tokens\"",
                "\"output_mode\": \"derender\"").replace("\"max_tokens\": 1",
                "\"max_tokens\": 1, \"temperature\": 0, \"ignore_eos\": true")).getEntity());
        assertEquals("derender", derived.str("output_mode"));
        assertNotNull(at(derived, "choices", 0).obj("message").str("content"));
        assertTrue(errorOf(() -> resource.generate("{\"token_ids\": [1]}")).contains("sampling_params"));
        assertEquals(400, statusOf(() -> resource.generate(
                "{\"token_ids\": [-1], \"sampling_params\": {\"max_tokens\": 1}}")));
        assertEquals(400, statusOf(() -> resource.generate(
                "{\"token_ids\": [], \"sampling_params\": {\"max_tokens\": 1}}")));
        String textMode = valid.replace("\"output_mode\": \"tokens\"", "\"output_mode\": \"text\"");
        assertEquals(400, statusOf(() -> resource.generate(textMode.replace("\"max_tokens\": 1",
                "\"max_tokens\": 1, \"detokenize\": false"))));
        assertEquals(400, statusOf(() -> resource.generate(textMode.replace("\"output_mode\": \"text\"",
                "\"output_mode\": \"derender\"").replace("\"stream\": false", "\"stream\": true"))));
        String streamed = valid.replace("\"stream\": false,", "\"stream\": true,");
        assertEquals(400, statusOf(() -> resource.generate(streamed.replace("\"max_tokens\": 1",
                "\"max_tokens\": 1, \"output_kind\": \"final_only\""))));
        assertEquals(400, statusOf(() -> resource.generate(streamed.replace("\"max_tokens\": 1",
                "\"max_tokens\": 1, \"prompt_logprobs\": 1\""))));
    }

    /** A generate body over {@code prompt}, with {@code sampling} as the sampling parameters. */
    private static String generateBody(int[] prompt, String sampling, boolean returnTokenIds) {
        return "{\"token_ids\": " + json(prompt) + ", \"sampling_params\": {" + sampling + "},"
                + " \"output_mode\": \"tokens\", \"stream\": false, \"return_token_ids\": " + returnTokenIds + "}";
    }

    /** {@code text} encoded with the BOS the rendered prompt carries. */
    private static int[] tokens(String text) {
        return service.encode(text, true, true);
    }

    private static String json(int[] values) {
        StringBuilder out = new StringBuilder("[");
        for (int i = 0; i < values.length; i++) {
            out.append(i == 0 ? "" : ", ").append(values[i]);
        }
        return out.append(']').toString();
    }
}

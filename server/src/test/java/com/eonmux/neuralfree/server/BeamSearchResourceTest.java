package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.utils.json.JsonView;
import jakarta.ws.rs.core.Response;
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
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * {@code use_beam_search} over the serving API (vLLM's {@code use_beam_search} on the chat and completion
 * protocols, {@code beam_width = n}) on the real service: the chat answers one final body of ranked message
 * choices whose beams share the prompt's cached rows, the completions answer prompt-major ranked choices, the
 * greedy request is deterministic, and every combination the search cannot honor — a stream, audio, {@code
 * best_of}, a grammar, stop strings, prompt logprobs, {@code echo} beside {@code logprobs}, the batch chat
 * endpoint — refuses by name.
 */
public class BeamSearchResourceTest {

    private static EngineService service;
    private static OpenAiResource resource;

    @BeforeAll
    static void open() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        service = new EngineService(model.toString(), ServerTestSupport.pagedConfig(256, 2));
        resource = new OpenAiResource(service);
    }

    @AfterAll
    static void close() {
        service.close();
    }

    /** A greedy chat-completions body with {@code use_beam_search} and the extra members. */
    private static String chat(String extra) {
        return "{\"messages\": [{\"role\": \"user\", \"content\": \"hi\"}], \"temperature\": 0,"
                + " \"max_tokens\": 4, \"use_beam_search\": true" + (extra.isEmpty() ? "" : ", " + extra) + "}";
    }

    /** A greedy completions body with {@code use_beam_search} and the extra members. */
    private static String completion(String extra) {
        return "{\"prompt\": \"hi\", \"temperature\": 0, \"max_tokens\": 4, \"use_beam_search\": true"
                + (extra.isEmpty() ? "" : ", " + extra) + "}";
    }

    private static JsonView usage(JsonView body) {
        return body.obj("usage");
    }

    @Test
    void aBeamChatAnswersRankedMessageChoices() {
        JsonView body = view(resource.chatCompletions(chat("\"n\": 3")).getEntity());
        assertEquals("chat.completion", body.str("object"));
        assertEquals(3, size(body, "choices"));
        for (int i = 0; i < 3; i++) {
            JsonView choice = at(body, "choices", i);
            assertEquals(i, choice.i64("index"));
            assertNotNull(choice.obj("message").str("content"), "choice " + i + " has content");
            assertNotNull(choice.str("finish_reason"));
        }
        assertTrue(usage(body).i64("prompt_tokens") > 0);
        assertTrue(usage(body).i64("completion_tokens") > 0);
        // the beams shared the prompt through the prefix cache: the steps' submissions reused the prompt's rows
        assertTrue(usage(body).obj("prompt_tokens_details").i64("cached_tokens") > 0,
                "the beam steps reused prompt rows");
    }

    @Test
    void aBeamChatIsDeterministicAndRanksTheSameBeamsOnEveryRun() {
        JsonView first = view(resource.chatCompletions(chat("\"n\": 3")).getEntity());
        JsonView second = view(resource.chatCompletions(chat("\"n\": 3")).getEntity());
        for (int i = 0; i < 3; i++) {
            assertEquals(at(first, "choices", i).obj("message").str("content"),
                    at(second, "choices", i).obj("message").str("content"));
        }
    }

    @Test
    void aBeamChatRendersLogprobsOfTheTopEntries() {
        JsonView body = view(resource.chatCompletions(
                chat("\"n\": 1, \"logprobs\": true, \"top_logprobs\": 1")).getEntity());
        JsonView logprobs = at(body, "choices", 0).obj("logprobs");
        List<?> content = (List<?>) logprobs.raw("content");
        assertFalse(content.isEmpty(), "the captures ride the choice");
        List<?> tops = (List<?>) ((Map<?, ?>) content.get(0)).get("top_logprobs");
        assertFalse(tops.isEmpty(), "the step's top entries are rendered");
    }

    @Test
    void aBeamCompletionAnswersRankedChoicesPromptMajor() {
        JsonView body = view(resource.completions(completion("\"n\": 2")).getEntity());
        assertEquals("text_completion", body.str("object"));
        assertEquals(2, size(body, "choices"));
        for (int i = 0; i < 2; i++) {
            JsonView choice = at(body, "choices", i);
            assertEquals(i, choice.i64("index"));
            assertNotNull(choice.str("text"));
            assertNull(choice.raw("logprobs"), "no logprobs were asked for");
            assertNotNull(choice.str("finish_reason"));
        }
        assertTrue(usage(body).i64("completion_tokens") > 0);
    }

    @Test
    void aBeamCompletionRendersTheTopLogprobsItAskedFor() {
        JsonView body = view(resource.completions(completion("\"n\": 1, \"logprobs\": 1")).getEntity());
        JsonView logprobs = at(body, "choices", 0).obj("logprobs");
        assertNotNull(logprobs);
        assertTrue(size(logprobs, "tokens") > 0, "the generated tokens' entries are there");
        assertTrue(size(logprobs, "top_logprobs") > 0);
    }

    @Test
    void aZeroLengthPenaltyRanksByTheBareCumulativeLogProbability() {
        JsonView body = view(resource.completions(completion("\"n\": 2, \"length_penalty\": 0")).getEntity());
        assertEquals(2, size(body, "choices"));
        for (int i = 0; i < 2; i++) {
            assertNotNull(at(body, "choices", i).str("text"));
        }
    }

    @Test
    void beamChatRefusalsNameTheirField() {
        assertEquals(400, statusOf(() -> resource.chatCompletions(chat("\"stream\": true"))));
        assertEquals(400, statusOf(() -> resource.chatCompletions(chat("\"use_beam_search\": 1"))));
        assertTrue(errorOf(() -> resource.chatCompletions(chat("\"stream\": true")))
                .contains("stream must be false or omitted"));
        assertTrue(errorOf(() -> resource.chatCompletions(chat("\"modalities\": [\"audio\"]")))
                .contains("use_beam_search"));
        assertTrue(errorOf(() -> resource.chatCompletions(chat("\"best_of\": 6, \"n\": 3")))
                .contains("best_of"));
        assertTrue(errorOf(() -> resource.chatCompletions(
                chat("\"response_format\": {\"type\": \"json_object\"}"))).contains("constraint"));
        assertTrue(errorOf(() -> resource.chatCompletions(chat("\"stop\": [\"halt\"]"))).contains("stop"));
        assertTrue(errorOf(() -> resource.chatCompletions(chat("\"prompt_logprobs\": 1")))
                .contains("prompt_logprobs"));
        assertTrue(errorOf(() -> resource.chatCompletions(chat("\"min_tokens\": 2"))).contains("min_tokens"));
        assertTrue(errorOf(() -> resource.chatCompletions(chat("\"bad_words\": [\"nope\"]")))
                .contains("bad_words"));
        assertTrue(errorOf(() -> resource.chatCompletions(chat("\"logit_bias\": {\"4\": 1.0}")))
                .contains("logit_bias"));
        assertTrue(errorOf(() -> resource.chatCompletions(chat("\"vllm_xargs\": {\"target_token\": 4}")))
                .contains("vllm_xargs"));
        assertTrue(errorOf(() -> resource.chatCompletions(chat("\"n_keep\": 4"))).contains("n_keep"));
        assertTrue(errorOf(() -> resource.chatCompletions("{\"messages\": [{\"role\": \"user\","
                + " \"content\": \"hi\"}], \"use_beam_search\": \"yes\"}"))
                .contains("use_beam_search must be a boolean"));
        assertTrue(errorOf(() -> resource.chatCompletions(chat("\"length_penalty\": \"steep\"")))
                .contains("length_penalty must be a number"));
    }

    @Test
    void beamCompletionRefusalsNameTheirField() {
        assertEquals(400, statusOf(() -> resource.completions(completion("\"stream\": true"))));
        assertTrue(errorOf(() -> resource.completions(completion("\"stream\": true")))
                .contains("use_beam_search cannot serve a stream"));
        assertTrue(errorOf(() -> resource.completions(completion("\"echo\": true, \"logprobs\": 1")))
                .contains("prompt logprobs"));
        assertTrue(errorOf(() -> resource.completions(completion("\"stop\": [\"halt\"]"))).contains("stop"));
    }

    @Test
    void theBatchChatEndpointRefusesBeamSearchByName() {
        String batchBody = "{\"messages\": [{\"role\": \"user\", \"content\": \"hi\"}],"
                + " \"use_beam_search\": true}";
        assertTrue(errorOf(() -> resource.batchChatCompletions(batchBody)).contains("use_beam_search"));
    }

    @Test
    void withoutTheFieldTheRequestsStayOnTheSamplingPath() {
        JsonView chat = view(resource.chatCompletions(
                "{\"messages\": [{\"role\": \"user\", \"content\": \"hi\"}], \"temperature\": 0,"
                        + " \"max_tokens\": 2, \"n\": 2}").getEntity());
        assertEquals(2, size(chat, "choices"));
        JsonView completion = view(resource.completions(
                "{\"prompt\": \"hi\", \"temperature\": 0, \"max_tokens\": 2, \"n\": 2}").getEntity());
        assertEquals(2, size(completion, "choices"));
    }
}

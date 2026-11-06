package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.StopReason;
import com.eonmux.neuralfree.gguf.engine.TokenScore;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.utils.json.JsonView;
import io.vertx.core.MultiMap;
import jakarta.ws.rs.core.Response;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.lang.reflect.Proxy;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static com.eonmux.neuralfree.server.ServerTestSupport.dataFrames;
import static com.eonmux.neuralfree.server.ServerTestSupport.errorOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.pagedConfig;
import static com.eonmux.neuralfree.server.ServerTestSupport.statusOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.streamText;
import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The vLLM request/response extras over the real service: {@code stop_token_ids} and
 * {@code include_stop_str_in_output} on the text, {@code truncate_prompt_tokens} on the prompt, a single
 * {@code allowed_token_ids} id repeated in the output (which {@code repetition_detection} ends with
 * {@code finish_reason: "repetition"}), {@code prompt_logprobs} with its null first entry,
 * {@code return_token_ids} and {@code return_tokens_as_token_ids} on the response, the echoed
 * {@code X-Request-Id}, {@code usage.prompt_tokens_details.cached_tokens} and the per-chunk usage under
 * {@code stream_options.continuous_usage_stats}. {@code cache_salt} accepts and answers, and so does
 * {@code priority} 0, while the default {@code fcfs} scheduling policy refuses any other {@code priority} by name.
 * The response side: {@code stop_reason} on choices and finish chunks, the echoed prompt's entries merged into the
 * legacy logprobs arrays, {@code best_of}'s selection and {@code logprob_token_ids}' alternatives; the vLLM
 * pass-through fields refuse by name.
 */
public class OpenAiExtrasResourceTest {

    /** One allowed id whose byte piece is deterministic under greedy decoding. */
    private static final int ALLOWED = 20;

    private static EngineService service;
    private static OpenAiResource resource;

    @BeforeAll
    static void open() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig config = pagedConfig(64, 2);
        service = new EngineService(model.toString(), config);
        resource = new OpenAiResource(service);
    }

    @AfterAll
    static void close() {
        service.close();
    }

    private static JsonView firstChoice(JsonView body) {
        return JsonView.of(((List<?>) body.raw("choices")).get(0), "choices[0]");
    }

    /** The id of the first token a greedy run generates, via {@code return_token_ids}. */
    private static long firstGeneratedToken(String prompt) {
        Response probe = resource.completions("{\"prompt\": \"" + prompt + "\", \"max_tokens\": 1, "
                + "\"temperature\": 0, \"return_token_ids\": true}");
        return (Long) ((List<?>) firstChoice(view(probe.getEntity())).raw("token_ids")).get(0);
    }

    /** The piece text of one token id. */
    private static String tokenText(long id) {
        return service.decode(new int[]{(int) id});
    }

    /** An HTTP request stub whose {@code headers()} answers the given {@code X-Request-Id}. */
    private static void requestHeader(String value) {
        MultiMap headers = MultiMap.caseInsensitiveMultiMap();
        if (value != null) {
            headers.add("X-Request-Id", value);
        }
        Object response = Proxy.newProxyInstance(OpenAiExtrasResourceTest.class.getClassLoader(),
                new Class<?>[]{io.vertx.core.http.HttpServerResponse.class},
                (proxy, method, args) -> null);
        resource.http = (io.vertx.core.http.HttpServerRequest) Proxy.newProxyInstance(
                OpenAiExtrasResourceTest.class.getClassLoader(),
                new Class<?>[]{io.vertx.core.http.HttpServerRequest.class},
                (proxy, method, args) -> {
                    if (method.getName().equals("headers")) {
                        return headers;
                    }
                    if (method.getName().equals("response")) {
                        return response;
                    }
                    Class<?> type = method.getReturnType();
                    if (type == boolean.class) {
                        return false;
                    }
                    if (type == int.class) {
                        return 0;
                    }
                    if (type == long.class) {
                        return 0L;
                    }
                    return null;
                });
    }

    @Test
    void usageReportsTheCachedPromptTokens() {
        Response first = resource.completions("""
                {"prompt": "usage details", "max_tokens": 2, "temperature": 0}""");
        JsonView usage = view(first.getEntity()).obj("usage");
        long plainCached = usage.obj("prompt_tokens_details").i64("cached_tokens");
        Response second = resource.completions("""
                {"prompt": "usage details", "max_tokens": 2, "temperature": 0}""");
        JsonView reused = view(second.getEntity()).obj("usage");
        assertEquals(reused.i64("prompt_tokens") - 1, reused.obj("prompt_tokens_details").i64("cached_tokens"),
                "the repeat reuses every cached page but the evaluated last token");
        assertTrue(reused.obj("prompt_tokens_details").i64("cached_tokens") > plainCached,
                "the first run cannot reuse the identical prompt's pages");
    }

    @Test
    void stopTokenIdsEndTheCompletionWithStop() {
        long first = firstGeneratedToken("stop me");
        Response stopped = resource.completions("{\"prompt\": \"stop me\", \"max_tokens\": 4, \"temperature\": 0,"
                + " \"stop_token_ids\": [" + first + "]}");
        assertEquals(200, stopped.getStatus());
        JsonView choice = firstChoice(view(stopped.getEntity()));
        assertEquals("stop", choice.str("finish_reason"));
        assertEquals("", choice.str("text"), "the stop token stays out of the output");
        assertEquals(1L, view(stopped.getEntity()).obj("usage").i64("completion_tokens"));
    }

    @Test
    void includeStopStrInOutputKeepsTheMatch() {
        long first = firstGeneratedToken("keep the stop");
        String piece = tokenText(first);
        Response cut = resource.completions("{\"prompt\": \"keep the stop\", \"max_tokens\": 4, \"temperature\": 0,"
                + " \"stop\": [\"" + piece + "\"]}");
        assertEquals("", firstChoice(view(cut.getEntity())).str("text"), "the stop string is cut by default");
        Response kept = resource.completions("{\"prompt\": \"keep the stop\", \"max_tokens\": 4, \"temperature\": 0,"
                + " \"stop\": [\"" + piece + "\"], \"include_stop_str_in_output\": true}");
        assertEquals(piece, firstChoice(view(kept.getEntity())).str("text"), "the stop string stays in");
    }

    @Test
    void truncatePromptTokensBoundsThePrompt() {
        Response response = resource.completions("""
                {"prompt": "one two three four five six seven", "truncate_prompt_tokens": 3,
                 "max_tokens": 1, "temperature": 0}""");
        JsonView usage = view(response.getEntity()).obj("usage");
        assertEquals(3L, usage.i64("prompt_tokens"), "only the kept tokens prefill");
    }

    @Test
    void aSingleAllowedTokenIdFillsTheOutput() {
        String token = tokenText(ALLOWED);
        Response response = resource.completions("{\"prompt\": \"allowed\", \"max_tokens\": 3, \"temperature\": 0,"
                + " \"allowed_token_ids\": [" + ALLOWED + "]}");
        assertEquals(200, response.getStatus());
        JsonView choice = firstChoice(view(response.getEntity()));
        assertEquals(token.repeat(3), choice.str("text"), "greedy can only draw the allowed id");
    }

    @Test
    void promptLogprobsAnswerPerPromptTokenWithANullFirstEntry() {
        Response response = resource.completions("""
                {"prompt": "logprob me", "prompt_logprobs": 0, "max_tokens": 1, "temperature": 0}""");
        JsonView body = view(response.getEntity());
        JsonView choice = firstChoice(body);
        List<?> promptLogprobs = (List<?>) choice.raw("prompt_logprobs");
        assertEquals(body.obj("usage").i64("prompt_tokens"), promptLogprobs.size());
        assertNull(promptLogprobs.get(0), "no row predicts the first prompt token");
        for (int i = 1; i < promptLogprobs.size(); i++) {
            JsonView entry = JsonView.of(promptLogprobs.get(i), "prompt_logprobs[" + i + "]");
            assertEquals(1, entry.keys().size(), "prompt_logprobs 0 reports the prompt token alone");            JsonView logprob = JsonView.of(entry.raw(entry.keys().iterator().next()), "entry");
            assertTrue(logprob.f64("logprob") <= 0);
            assertTrue(logprob.i64("rank") >= 1);
            assertNotNull(logprob.str("decoded_token"));
        }
        assertEquals(400, statusOf(() -> resource.completions("""
                {"prompt": "s", "prompt_logprobs": 2, "max_tokens": 1, "stream": true}""")),
                "prompt_logprobs with alternatives refuse on a stream");
    }

    @Test
    void returnTokenIdsAndTokensAsTokenIdsShapeTheResponse() {
        Response response = resource.completions("""
                {"prompt": "token ids", "max_tokens": 2, "temperature": 0, "return_token_ids": true,
                 "return_tokens_as_token_ids": true, "logprobs": 1}""");
        JsonView body = view(response.getEntity());
        JsonView choice = firstChoice(body);
        assertEquals(body.obj("usage").i64("completion_tokens"), sizeOf(choice.raw("token_ids")));
        assertEquals(body.obj("usage").i64("prompt_tokens"), sizeOf(choice.raw("prompt_token_ids")));
        for (Object t : (List<?>) choice.obj("logprobs").raw("tokens")) {
            assertTrue(String.valueOf(t).startsWith("token_id:"), "the legacy tokens render as ids");
        }
    }

    @Test
    void everyChoiceNamesThePromptItAnswers() {
        Response response = resource.completions("""
                {"prompt": ["first prompt", "the second one"], "n": 2, "max_tokens": 1, "temperature": 0,
                 "return_token_ids": true}""");
        JsonView body = view(response.getEntity());
        List<?> choices = (List<?>) body.raw("choices");
        assertEquals(4, choices.size());
        String[] prompts = {"first prompt", "the second one"};
        for (int i = 0; i < choices.size(); i++) {
            JsonView choice = JsonView.of(choices.get(i), "choices[i]");
            int[] expected = service.encode(prompts[i / 2], true, true);
            List<?> ids = (List<?>) choice.raw("prompt_token_ids");
            assertEquals(expected.length, ids.size(), "choice " + i + " answers prompt " + i / 2);
            for (int t = 0; t < expected.length; t++) {
                assertEquals((long) expected[t], ids.get(t));
            }
        }
    }

    @Test
    void chatCarriesTokenIdsPromptIdsAndIdRenderedLogprobs() {
        Response response = resource.chatCompletions("""
                {"messages": [{"role": "user", "content": "chat ids"}], "max_tokens": 2, "temperature": 0,
                 "return_token_ids": true, "logprobs": true, "top_logprobs": 1,
                 "return_tokens_as_token_ids": true}""");
        JsonView body = view(response.getEntity());
        assertEquals(body.obj("usage").i64("prompt_tokens"), sizeOf(body.raw("prompt_token_ids")),
                "the response names the prompt ids");
        assertEquals(body.obj("usage").i64("completion_tokens"),
                sizeOf(firstChoice(body).raw("token_ids")), "the choice names its generated ids");
        JsonView entry = JsonView.of(((List<?>) firstChoice(body).obj("logprobs").raw("content")).get(0), "entry");
        assertTrue(entry.str("token").startsWith("token_id:"), "the sampled token renders as its id");
    }

    @Test
    void theRequestIdHeaderNamesTheResponse() {
        requestHeader("trace-42");
        try {
            Response completion = resource.completions("""
                    {"prompt": "hi", "max_tokens": 1, "temperature": 0}""");
            assertEquals("cmpl-trace-42", view(completion.getEntity()).str("id"));
            Response chat = resource.chatCompletions("""
                    {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1, "temperature": 0}""");
            assertEquals("chatcmpl-trace-42", view(chat.getEntity()).str("id"));
        } finally {
            requestHeader(null);
        }
        Response plain = resource.completions("""
                {"prompt": "hi", "max_tokens": 1, "temperature": 0}""");
        assertFalse(view(plain.getEntity()).str("id").equals("cmpl-trace-42"),
                "without the header the id is fresh");
    }

    @Test
    void cacheSaltAndPriorityZeroAcceptAndAnswer() {
        assertEquals(200, statusOf(() -> resource.completions("""
                {"prompt": "hi", "max_tokens": 1, "temperature": 0, "priority": 0, "cache_salt": "tenant-a"}""")));
        assertEquals(200, statusOf(() -> resource.chatCompletions("""
                {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1, "temperature": 0,
                 "cache_salt": "tenant-b"}""")));
    }

    @Test
    void fcfsRefusesANonzeroPriorityByNameBeforeAnyStreamOpens() {
        String refusal = errorOf(() -> resource.completions("""
                {"prompt": "hi", "max_tokens": 1, "temperature": 0, "priority": -5}"""));
        assertTrue(refusal.startsWith("priority -5"), refusal);
        assertTrue(refusal.contains("neuralfree.scheduling.policy=priority"), refusal);
        assertEquals(400, statusOf(() -> resource.chatCompletions("""
                {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1, "priority": 5}""")));
        assertEquals(400, statusOf(() -> resource.completions("""
                {"prompt": "hi", "max_tokens": 1, "priority": 2, "stream": true}""")),
                "a stream answers the refusal as a 400, not an error frame");
        assertEquals(400, statusOf(() -> resource.chatCompletions("""
                {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1, "priority": 1,
                 "stream": true}""")));
    }

    @Test
    void continuousUsageStatsCarryUsageOnEveryChunk() throws IOException {
        Response response = resource.completions("""
                {"prompt": "hi", "max_tokens": 2, "temperature": 0, "stream": true,
                 "stream_options": {"include_usage": true, "continuous_usage_stats": true}}""");
        List<JsonView> frames = dataFrames(streamText(response));
        assertFalse(frames.isEmpty());
        for (JsonView frame : frames) {
            assertNotNull(frame.raw("usage"), "every chunk carries the running usage");
            assertTrue(frame.obj("usage").i64("prompt_tokens") >= 1);
        }
    }

    @Test
    void returnTokenIdsRideTheStreamChunks() throws IOException {
        Response response = resource.completions("{\"prompt\": \"hi\", \"max_tokens\": 2, \"temperature\": 0,"
                + " \"stream\": true, \"return_token_ids\": true, \"allowed_token_ids\": [" + ALLOWED + "]}");
        List<JsonView> frames = dataFrames(streamText(response));
        assertFalse(frames.isEmpty());
        JsonView first = frames.get(0);
        JsonView choice = JsonView.of(((List<?>) first.raw("choices")).get(0), "choices[0]");
        assertTrue(choice.raw("prompt_token_ids") instanceof List<?>, "the first chunk carries the prompt ids");
        assertFalse(choice.str("text").isEmpty(), "the first chunk carries the first token's text");
        assertEquals(1, sizeOf(choice.raw("token_ids")), "one delta token per chunk");
    }

    @Test
    void anEchoedStreamNamesThePromptIdsOnItsEchoChunk() throws IOException {
        Response response = resource.completions("""
                {"prompt": "hi", "max_tokens": 2, "temperature": 0, "stream": true, "echo": true,
                 "return_token_ids": true}""");
        List<JsonView> frames = dataFrames(streamText(response));
        JsonView echo = JsonView.of(((List<?>) frames.get(0).raw("choices")).get(0), "choices[0]");
        assertEquals("hi", echo.str("text"), "the echo is the first chunk");
        assertEquals(service.encode("hi", true, true).length, sizeOf(echo.raw("prompt_token_ids")),
                "the echo chunk carries the prompt ids");
        JsonView next = JsonView.of(((List<?>) frames.get(1).raw("choices")).get(0), "choices[0]");
        assertNull(next.raw("prompt_token_ids"), "only the first chunk names the prompt ids");
    }

    /** A greedy body whose output repeats {@link #ALLOWED}, with {@code extra} members appended. */
    private static String repeating(String head, String extra) {
        return "{" + head + ", \"max_tokens\": 8, \"temperature\": 0, \"allowed_token_ids\": [" + ALLOWED + "]"
                + extra + "}";
    }

    private static final String DETECT_THREE =
            ", \"repetition_detection\": {\"max_pattern_size\": 2, \"min_pattern_size\": 1, \"min_count\": 3}";

    @Test
    void repetitionDetectionEndsACompletionWithRepetition() {
        Response stopped = resource.completions(repeating("\"prompt\": \"loop\"", DETECT_THREE));
        assertEquals(200, stopped.getStatus());
        JsonView choice = firstChoice(view(stopped.getEntity()));
        assertEquals("repetition", choice.str("finish_reason"), "vLLM's FinishReason.REPETITION");
        assertEquals(tokenText(ALLOWED).repeat(3), choice.str("text"), "the repeats stay in the text");
        assertEquals(3L, view(stopped.getEntity()).obj("usage").i64("completion_tokens"));

        Response plain = resource.completions(repeating("\"prompt\": \"loop\"", ""));
        assertEquals("length", firstChoice(view(plain.getEntity())).str("finish_reason"),
                "without detection the budget ends it");
    }

    @Test
    void minTokensDefersTheRepetitionStop() {
        Response stopped = resource.completions(repeating("\"prompt\": \"loop\"", DETECT_THREE
                + ", \"min_tokens\": 5"));
        assertEquals("repetition", firstChoice(view(stopped.getEntity())).str("finish_reason"));
        assertEquals(5L, view(stopped.getEntity()).obj("usage").i64("completion_tokens"));
    }

    @Test
    void repetitionDetectionEndsAChatChoiceWithRepetition() {
        Response stopped = resource.chatCompletions(repeating(
                "\"messages\": [{\"role\": \"user\", \"content\": \"loop\"}]", DETECT_THREE));
        assertEquals(200, stopped.getStatus());
        JsonView choice = firstChoice(view(stopped.getEntity()));
        assertEquals("repetition", choice.str("finish_reason"));
        assertEquals(3L, view(stopped.getEntity()).obj("usage").i64("completion_tokens"));
    }

    @Test
    void aStreamedRepetitionStopNamesItsFinishReason() throws IOException {
        Response response = resource.completions(repeating("\"prompt\": \"loop\", \"stream\": true", DETECT_THREE));
        List<JsonView> frames = dataFrames(streamText(response));
        String finish = null;
        for (JsonView frame : frames) {
            for (Object c : (List<?>) frame.raw("choices")) {
                Object reason = JsonView.of(c, "choice").raw("finish_reason");
                if (reason != null) {
                    finish = (String) reason;
                }
            }
        }
        assertEquals("repetition", finish);
    }

    @Test
    void malformedRepetitionDetectionRefusesByName() {
        assertTrue(errorOf(() -> resource.completions(repeating("\"prompt\": \"loop\"",
                ", \"repetition_detection\": {\"max_pattern_size\": 2, \"min_count\": 1}")))
                .contains("min_count"));
        assertEquals(400, statusOf(() -> resource.chatCompletions(repeating(
                "\"messages\": [{\"role\": \"user\", \"content\": \"loop\"}]",
                ", \"repetition_detection\": [2, 3]"))));
    }

    @Test
    void outOfVocabularyIdsRefuseByName() {
        assertTrue(errorOf(() -> resource.completions("{\"prompt\": \"hi\", \"max_tokens\": 1,"
                + " \"stop_token_ids\": [" + service.vocabSize() + "]}")).contains("stop_token_ids"));
        assertTrue(errorOf(() -> resource.completions("{\"prompt\": \"hi\", \"max_tokens\": 1,"
                + " \"allowed_token_ids\": [" + service.vocabSize() + "]}")).contains("allowed_token_ids"));
    }

    @Test
    void stopReasonNamesTheMatchedStopTokenIdOrString() {
        long first = firstGeneratedToken("reason me");
        Response byToken = resource.completions("{\"prompt\": \"reason me\", \"max_tokens\": 4, \"temperature\": 0,"
                + " \"stop_token_ids\": [" + first + "]}");
        JsonView tokenChoice = firstChoice(view(byToken.getEntity()));
        assertEquals("stop", tokenChoice.str("finish_reason"));
        assertEquals(first, (long) (Long) tokenChoice.raw("stop_reason"), "the matched stop token id");
        String piece = tokenText(first);
        Response byString = resource.completions("{\"prompt\": \"reason me\", \"max_tokens\": 4, \"temperature\": 0,"
                + " \"stop\": [\"" + piece + "\"]}");
        assertEquals(piece, firstChoice(view(byString.getEntity())).raw("stop_reason"), "the matched stop string");
        Response byLength = resource.completions(
                "{\"prompt\": \"reason me\", \"max_tokens\": 1, \"temperature\": 0}");
        assertNull(firstChoice(view(byLength.getEntity())).raw("stop_reason"), "a length end reports none");
    }

    @Test
    void chatAndTheFinishChunkCarryTheStopReason() throws IOException {
        Response probe = resource.chatCompletions("{\"messages\": [{\"role\": \"user\", \"content\":"
                + " \"chat reason\"}], \"max_tokens\": 1, \"temperature\": 0, \"return_token_ids\": true}");
        long first = (Long) ((List<?>) firstChoice(view(probe.getEntity())).raw("token_ids")).get(0);
        Response stopped = resource.chatCompletions("{\"messages\": [{\"role\": \"user\", \"content\":"
                + " \"chat reason\"}], \"max_tokens\": 4, \"temperature\": 0, \"stop_token_ids\": [" + first + "]}");
        assertEquals(first, (long) (Long) firstChoice(view(stopped.getEntity())).raw("stop_reason"));

        long completionFirst = firstGeneratedToken("chat reason");
        Response stream = resource.completions("{\"prompt\": \"chat reason\", \"max_tokens\": 4, \"temperature\": 0,"
                + " \"stop_token_ids\": [" + completionFirst + "], \"stream\": true}");
        JsonView finish = null;
        for (JsonView frame : dataFrames(streamText(stream))) {
            JsonView choice = JsonView.of(((List<?>) frame.raw("choices")).get(0), "choices[0]");
            if (choice.raw("finish_reason") != null) {
                finish = choice;
            }
        }
        assertEquals(completionFirst, (long) (Long) finish.raw("stop_reason"),
                "the finish chunk names the stop token id");

        Response chatStream = resource.chatCompletions("{\"messages\": [{\"role\": \"user\", \"content\":"
                + " \"chat reason\"}], \"max_tokens\": 4, \"temperature\": 0, \"stop_token_ids\": [" + first + "],"
                + " \"stream\": true}");
        JsonView chatFinish = null;
        for (JsonView frame : dataFrames(streamText(chatStream))) {
            List<?> choices = (List<?>) frame.raw("choices");
            if (choices.isEmpty()) {
                continue;
            }
            JsonView choice = JsonView.of(choices.get(0), "choices[0]");
            if (choice.raw("finish_reason") != null) {
                chatFinish = choice;
            }
        }
        assertEquals(first, (long) (Long) chatFinish.raw("stop_reason"),
                "the chat finish chunk names the stop token id");
    }

    @Test
    void echoLogprobsMergeThePromptEntriesIntoTheLegacyArrays() {
        Response response = resource.completions("{\"prompt\": \"echo me\", \"echo\": true, \"logprobs\": 1,"
                + " \"max_tokens\": 2, \"temperature\": 0}");
        JsonView body = view(response.getEntity());
        JsonView choice = firstChoice(body);
        JsonView logprobs = choice.obj("logprobs");
        long promptTokens = body.obj("usage").i64("prompt_tokens");
        long completionTokens = body.obj("usage").i64("completion_tokens");
        assertEquals(promptTokens + completionTokens, sizeOf(logprobs.raw("tokens")),
                "the prompt's entries come before the generated ones");
        assertEquals(0L, (Long) ((List<?>) logprobs.raw("text_offset")).get(0), "the first prompt token opens at 0");
        assertNull(((List<?>) logprobs.raw("token_logprobs")).get(0), "no row predicts the first prompt token");
        assertNull(((List<?>) logprobs.raw("top_logprobs")).get(0), "no alternatives for the first prompt token");
        for (int i = 1; i < sizeOf(logprobs.raw("tokens")); i++) {
            assertNotNull(((List<?>) logprobs.raw("token_logprobs")).get(i));
        }
        int[] prompt = service.encode("echo me", true, true);
        assertEquals(service.tokenText(prompt[0]), ((List<?>) logprobs.raw("tokens")).get(0),
                "the first entry is the prompt token's text");
        assertTrue(choice.str("text").startsWith("echo me"), "the echoed prompt opens the text");
    }

    @Test
    void anEchoedStreamCarriesTheMergedArraysOnItsEchoChunk() throws IOException {
        Response response = resource.completions("{\"prompt\": \"echo me\", \"echo\": true, \"logprobs\": 1,"
                + " \"max_tokens\": 2, \"temperature\": 0, \"stream\": true}");
        List<JsonView> frames = dataFrames(streamText(response));
        JsonView echo = JsonView.of(((List<?>) frames.get(0).raw("choices")).get(0), "choices[0]");
        assertFalse(echo.str("text").isEmpty());
        JsonView logprobs = echo.obj("logprobs");
        assertNotNull(logprobs, "the echo chunk carries the merged prompt arrays");
        assertNull(((List<?>) logprobs.raw("token_logprobs")).get(0), "the first prompt token has no row");
        assertEquals(0L, (Long) ((List<?>) logprobs.raw("text_offset")).get(0));
        assertTrue(sizeOf(logprobs.raw("tokens")) > 1, "every prompt token has an entry");
    }

    @Test
    void bestOfReturnsTheBestNChoicesAndRefusesAStream() {
        Response single = resource.completions(
                "{\"prompt\": \"pick the best\", \"n\": 1, \"best_of\": 2, \"max_tokens\": 2, \"temperature\": 0}");
        JsonView body = view(single.getEntity());
        assertEquals(1, sizeOf(body.raw("choices")), "only the best candidate answers");
        assertEquals(2L, body.obj("usage").i64("completion_tokens"),
                "the usage counts the returned choice alone");
        assertEquals(400, statusOf(() -> resource.completions(
                "{\"prompt\": \"pick the best\", \"n\": 1, \"best_of\": 2, \"max_tokens\": 1, \"stream\": true}")));
        assertTrue(errorOf(() -> resource.completions(
                "{\"prompt\": \"p\", \"n\": 1, \"best_of\": 2, \"max_tokens\": 1, \"stream\": true}"))
                .contains("best_of"));
        assertEquals(400, statusOf(() -> resource.completions(
                "{\"prompt\": \"pick the best\", \"n\": 2, \"best_of\": 1, \"max_tokens\": 1}")),
                "best_of below n refuses");
    }

    @Test
    void logprobTokenIdsShapeTheLegacyTopMaps() {
        Response named = resource.completions("{\"prompt\": \"name the ids\", \"logprobs\": 1,"
                + " \"logprob_token_ids\": [4, 5], \"max_tokens\": 2, \"temperature\": 0}");
        JsonView namedLogprobs = firstChoice(view(named.getEntity())).obj("logprobs");
        List<?> sampled = (List<?>) namedLogprobs.raw("tokens");
        List<?> namedTops = (List<?>) namedLogprobs.raw("top_logprobs");
        for (int i = 0; i < namedTops.size(); i++) {
            Map<?, ?> map = (Map<?, ?>) namedTops.get(i);
            assertTrue(map.containsKey(service.tokenText(4)) && map.containsKey(service.tokenText(5)),
                    "both named ids answer beyond the logprobs count");
            assertEquals(sampled.get(i), map.keySet().iterator().next(), "the sampled token leads the map");
        }
        Response plain = resource.completions(
                "{\"prompt\": \"name the ids\", \"logprobs\": 1, \"max_tokens\": 2, \"temperature\": 0}");
        JsonView plainLogprobs = firstChoice(view(plain.getEntity())).obj("logprobs");
        for (Object t : (List<?>) plainLogprobs.raw("top_logprobs")) {
            assertEquals(1, ((Map<?, ?>) t).size(), "without the named ids the map holds the sampled token alone");
        }
        assertEquals(400, statusOf(() -> resource.completions(
                "{\"prompt\": \"p\", \"logprob_token_ids\": [4], \"max_tokens\": 1}")),
                "logprob_token_ids needs logprobs");
    }

    @Test
    void chatTopLogprobsListTheNamedIdsAfterTheSampledToken() {
        Response response = resource.chatCompletions("{\"messages\": [{\"role\": \"user\", \"content\":"
                + " \"label it\"}], \"logprobs\": true, \"top_logprobs\": 0, \"logprob_token_ids\": [6, 9],"
                + " \"max_tokens\": 2, \"temperature\": 0}");
        List<?> content = (List<?>) firstChoice(view(response.getEntity())).obj("logprobs").raw("content");
        assertEquals(2, content.size());
        for (Object o : content) {
            JsonView entry = JsonView.of(o, "content[]");
            List<?> tops = (List<?>) entry.raw("top_logprobs");
            JsonView lead = JsonView.of(tops.get(0), "top_logprobs[0]");
            assertEquals(entry.str("token"), lead.str("token"), "the sampled token leads (vLLM's return_all)");
            List<Object> texts = new ArrayList<>();
            for (Object t : tops) {
                texts.add(JsonView.of(t, "top").str("token"));
            }
            assertTrue(texts.contains(service.tokenText(6)) && texts.contains(service.tokenText(9)),
                    "top_logprobs 0 still lists the named ids: " + texts);
        }
    }

    @Test
    void bestOfHidesTheRankingCaptureUnlessLogprobsAreAsked() {
        Response completion = resource.completions("{\"prompt\": \"rank quietly\", \"n\": 1, \"best_of\": 3,"
                + " \"max_tokens\": 2, \"temperature\": 0.9, \"seed\": 5}");
        assertNull(firstChoice(view(completion.getEntity())).raw("logprobs"),
                "the candidates' ranking capture is not the client's logprobs");
        Response chat = resource.chatCompletions("{\"messages\": [{\"role\": \"user\", \"content\":"
                + " \"rank quietly\"}], \"n\": 2, \"best_of\": 3, \"max_tokens\": 2, \"temperature\": 0.9,"
                + " \"seed\": 5}");
        JsonView chatBody = view(chat.getEntity());
        assertEquals(2, sizeOf(chatBody.raw("choices")), "the best n of the candidates answer");
        for (Object c : (List<?>) chatBody.raw("choices")) {
            assertNull(JsonView.of(c, "choice").raw("logprobs"));
        }
        Response asked = resource.completions("{\"prompt\": \"rank quietly\", \"n\": 1, \"best_of\": 3,"
                + " \"logprobs\": 0, \"max_tokens\": 2, \"temperature\": 0.9, \"seed\": 5}");
        assertNotNull(firstChoice(view(asked.getEntity())).raw("logprobs"), "asked logprobs answer");
    }

    @Test
    void bestOfKeepsTheHighestCumulativeLogprobsInSubmissionOrderOnTies() {
        List<GenerationResult> results = List.of(
                scored(-3.0f), scored(-0.5f), scored(-2.0f),   // prompt 0
                scored(-1.0f), scored(-1.0f), scored(-4.0f));  // prompt 1
        assertArrayEquals(new int[]{1, 2, 3, 4}, OpenAiResource.bestOfSelection(results, 2, 3),
                "per prompt the best two by cumulative logprob, prompt-major; ties keep submission order");
        assertArrayEquals(new int[]{0, 1}, OpenAiResource.bestOfSelection(results.subList(0, 2), 2, 2),
                "best_of = n keeps every candidate in order");
    }

    /** A one-token result whose sampled token has log probability {@code logprob}. */
    private static GenerationResult scored(float logprob) {
        return new GenerationResult(new int[]{1}, "a",
                StopReason.MAX_TOKENS, null,
                GenerationResult.Timings.of(1, 1, 1, 1))
                .withLogprobs(List.<TokenScore[]>of(
                        new TokenScore[]{
                                new TokenScore(1, logprob)}));
    }

    @Test
    void unsupportedVllmFieldsRefuseByName() {
        assertTrue(errorOf(() -> resource.completions("{\"prompt\": \"hi\", \"max_tokens\": 1,"
                + " \"vllm_xargs\": {\"flag\": {}}}")).contains("vllm_xargs.flag"));
        assertTrue(errorOf(() -> resource.chatCompletions("{\"messages\": [{\"role\": \"user\", \"content\": \"hi\"}],"
                + " \"max_tokens\": 1, \"kv_transfer_params\": {}}")).contains("do_remote_decode"));
        assertTrue(errorOf(() -> resource.chatCompletions("{\"messages\": [{\"role\": \"user\","
                + " \"content\": \"hi\"}], \"max_tokens\": 1, \"kv_transfer_params\": 1}"))
                .contains("kv_transfer_params"));
    }

    private static int sizeOf(Object list) {
        return ((List<?>) list).size();
    }
}

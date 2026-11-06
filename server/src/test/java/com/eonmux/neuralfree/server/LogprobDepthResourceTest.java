package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.LogprobsMode;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.utils.json.JsonView;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.StringJoiner;

import static com.eonmux.neuralfree.server.ServerTestSupport.at;
import static com.eonmux.neuralfree.server.ServerTestSupport.dataFrames;
import static com.eonmux.neuralfree.server.ServerTestSupport.errorOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.pagedConfig;
import static com.eonmux.neuralfree.server.ServerTestSupport.size;
import static com.eonmux.neuralfree.server.ServerTestSupport.statusOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.streamText;
import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Logprob depth over the real service: vLLM's {@code logprob_token_ids} on chat and completions and their
 * refusals, the server's {@code --logprobs-mode} choosing raw or processed scores, an echoed completion's legacy
 * arrays leading with the prompt tokens (non-streamed and in the first streamed chunk), the streaming refusal of
 * an explicit {@code prompt_logprobs}, vLLM's chat echo default for {@code prompt_logprobs}, and llama.cpp's
 * {@code post_sampling_probs} and {@code return_tokens} on the native completion.
 */
class LogprobDepthResourceTest {

    /** A token the logit bias forces under greedy decoding. */
    private static final int FORCED = 100;

    private static EngineService service;
    private static EngineService processed;
    private static OpenAiResource resource;
    private static OpenAiResource processedResource;
    private static LlamaCompletionResource llama;

    @BeforeAll
    static void open() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig config = pagedConfig(64, 2);
        service = new EngineService(model.toString(), config);
        processed = new EngineService(model.toString(), config,
                ServerOptions.defaults().withLogprobsMode(LogprobsMode.PROCESSED_LOGPROBS));
        resource = new OpenAiResource(service);
        processedResource = new OpenAiResource(processed);
        llama = new LlamaCompletionResource(service);
    }

    @AfterAll
    static void close() {
        service.close();
        processed.close();
    }

    private static JsonView firstChoice(JsonView body) {
        return at(body, "choices", 0);
    }

    private static JsonView completionLogprobs(OpenAiResource api, String body) {
        return firstChoice(view(api.completions(body).getEntity())).obj("logprobs");
    }

    @Test
    void chatLogprobTokenIdsScoreTheSampledTokenAndEveryRequestedId() {
        JsonView body = view(resource.chatCompletions("""
                {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 3, "temperature": 0,
                 "logprobs": true, "top_logprobs": 5, "logprob_token_ids": [5, 7, 5],
                 "return_tokens_as_token_ids": true}""").getEntity());
        List<?> content = (List<?>) firstChoice(body).obj("logprobs").raw("content");
        assertFalse(content.isEmpty());
        for (Object o : content) {
            Map<?, ?> entry = (Map<?, ?>) o;
            List<String> tokens = tokensOf((List<?>) entry.get("top_logprobs"));
            assertEquals(entry.get("token"), tokens.get(0), "the sampled token leads");
            assertTrue(tokens.contains("token_id:5") && tokens.contains("token_id:7"), tokens.toString());
            assertTrue(tokens.size() <= 3, "the requested ids replace top_logprobs: " + tokens);
            assertEquals(tokens.size(), tokens.stream().distinct().count(), "each id is listed once");
        }
    }

    @Test
    void completionLogprobTokenIdsFillTheTopLogprobsDict() {
        JsonView logprobs = completionLogprobs(resource, """
                {"prompt": "hello", "max_tokens": 2, "temperature": 0, "logprobs": 0,
                 "logprob_token_ids": [9], "return_tokens_as_token_ids": true}""");
        List<?> tops = (List<?>) logprobs.raw("top_logprobs");
        List<?> tokens = (List<?>) logprobs.raw("tokens");
        assertEquals(2, tops.size());
        for (int i = 0; i < tops.size(); i++) {
            Map<?, ?> top = (Map<?, ?>) tops.get(i);
            assertEquals(tokens.get(i), top.keySet().iterator().next());
            assertTrue(top.containsKey("token_id:9"), top.toString());
        }
    }

    @Test
    void logprobTokenIdsRefuseByName() {
        String chat = "{\"messages\": [{\"role\": \"user\", \"content\": \"hi\"}], \"max_tokens\": 1, ";
        assertTrue(errorOf(() -> resource.chatCompletions(chat + "\"logprob_token_ids\": [5]}"))
                .contains("logprob_token_ids"), "the ids need logprobs");
        assertEquals(400, statusOf(() -> resource.completions(
                "{\"prompt\": \"hi\", \"max_tokens\": 1, \"logprobs\": 0, \"logprob_token_ids\": [99999999]}")));
        StringJoiner many = new StringJoiner(", ", "[", "]");
        for (int i = 0; i < 129; i++) {
            many.add(String.valueOf(i));
        }
        assertTrue(errorOf(() -> resource.completions("{\"prompt\": \"hi\", \"max_tokens\": 1, \"logprobs\": 0,"
                + " \"logprob_token_ids\": " + many + "}")).contains("logprob_token_ids"));
        assertEquals(200, resource.chatCompletions(chat + "\"logprob_token_ids\": []}").getStatus(),
                "an empty list asks for nothing");
    }

    /**
     * A strong bias makes the chain all but certain of the forced token: the processed mode reports that near-0
     * logprob, the default raw mode the model's own lower one.
     */
    @Test
    void theLogprobsModeChoosesRawOrProcessedScores() {
        String body = "{\"prompt\": \"hello\", \"max_tokens\": 2, \"temperature\": 0, \"logprobs\": 0,"
                + " \"logit_bias\": {\"" + FORCED + "\": 30}, \"return_tokens_as_token_ids\": true}";
        JsonView raw = completionLogprobs(resource, body);
        JsonView chain = completionLogprobs(processedResource, body);
        assertEquals(raw.raw("tokens"), chain.raw("tokens"), "the mode changes the scores, not the tokens");
        assertEquals("token_id:" + FORCED, ((List<?>) raw.raw("tokens")).get(0));
        for (int i = 0; i < 2; i++) {
            double r = ((Number) ((List<?>) raw.raw("token_logprobs")).get(i)).doubleValue();
            double p = ((Number) ((List<?>) chain.raw("token_logprobs")).get(i)).doubleValue();
            assertTrue(p > -1e-3, "the biased chain is near-certain: " + p);
            assertTrue(r < p - 0.1, "the model alone is not: " + r);
        }
    }

    @Test
    void aStreamedEchoCarriesThePromptLogprobsOfTheNonStreamedBody() throws IOException {
        String body = """
                {"prompt": "hello", "max_tokens": 2, "temperature": 0, "logprobs": 1, "echo": true""";
        JsonView whole = completionLogprobs(resource, body + "}");
        List<JsonView> frames = dataFrames(streamText(resource.completions(body + ", \"stream\": true}")));
        JsonView echo = firstChoice(frames.get(0));
        assertEquals("hello", echo.str("text"));
        JsonView scored = echo.obj("logprobs");
        int prompt = service.encode("hello", true, true).length;
        List<?> tokens = (List<?>) scored.raw("tokens");
        List<?> logprobs = (List<?>) scored.raw("token_logprobs");
        assertEquals(prompt, tokens.size(), "the echo chunk scores the prompt tokens");
        assertNull(logprobs.get(0));
        assertEquals(0L, ((Number) ((List<?>) scored.raw("text_offset")).get(0)).longValue());
        assertEquals(((List<?>) whole.raw("tokens")).subList(0, prompt), tokens);
        assertEquals(((List<?>) whole.raw("token_logprobs")).subList(0, prompt), logprobs);
        JsonView next = firstChoice(frames.get(1));
        long offset = ((Number) ((List<?>) next.obj("logprobs").raw("text_offset")).get(0)).longValue();
        assertEquals(((Number) ((List<?>) whole.raw("text_offset")).get(prompt)).longValue(), offset,
                "the generated tokens continue after the echoed prompt tokens, as in the non-streamed body");
    }

    @Test
    void onlyAnExplicitPromptLogprobsRefusesToStream() {
        String head = "{\"prompt\": \"hi\", \"max_tokens\": 1, \"stream\": true, ";
        assertTrue(errorOf(() -> resource.completions(head + "\"prompt_logprobs\": 1}"))
                .contains("prompt_logprobs"));
        assertTrue(errorOf(() -> resource.completions(head + "\"prompt_logprobs\": -1}"))
                .contains("prompt_logprobs"));
        assertEquals(200, resource.completions(head + "\"prompt_logprobs\": 0}").getStatus());
        assertEquals(200, resource.completions(head + "\"logprobs\": 3, \"echo\": true}").getStatus(),
                "the echo default streams");
    }

    @Test
    void aChatEchoFillsPromptLogprobsWithTopLogprobs() {
        JsonView body = view(resource.chatCompletions("""
                {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1, "echo": true}""").getEntity());
        List<?> prompt = (List<?>) body.raw("prompt_logprobs");
        assertNotNull(prompt, "vLLM's chat echo defaults prompt_logprobs to top_logprobs (0)");
        assertNull(prompt.get(0));
        assertEquals(1, ((Map<?, ?>) prompt.get(1)).size(), "the prompt token alone");
    }

    @Test
    void postSamplingProbsAreTheChainsProbabilities() {
        JsonView chain = view(llama.completion("""
                {"prompt": "hello", "n_predict": 2, "ignore_eos": true, "n_probs": 4, "post_sampling_probs": true,
                 "temperature": 1, "top_k": 2, "seed": 7}""").getEntity());
        assertTrue(chain.obj("generation_settings").bool("post_sampling_probs"));
        JsonView first = at(chain, "completion_probabilities", 0);
        assertFalse(first.has("logprob") || first.has("top_logprobs"));
        double prob = first.f64("prob");
        assertTrue(prob > 0 && prob <= 1, "the sampled token's chain probability: " + prob);
        List<?> top = (List<?>) first.raw("top_probs");
        assertEquals(2, top.size(), "top_k left two candidates; the removed ones are left out");
        double sum = 0;
        for (Object o : top) {
            sum += ((Number) ((Map<?, ?>) o).get("prob")).doubleValue();
        }
        assertEquals(1.0, sum, 1e-4, "the chain's distribution over its candidates");
        JsonView model = view(llama.completion("""
                {"prompt": "hello", "n_predict": 2, "ignore_eos": true, "n_probs": 4,
                 "temperature": 1, "top_k": 2, "seed": 7}""").getEntity());
        assertFalse(model.obj("generation_settings").bool("post_sampling_probs"));
        assertEquals(4, size(at(model, "completion_probabilities", 0), "top_logprobs"),
                "the model's probabilities cover the vocabulary");
    }

    @Test
    void returnTokensListsTheGeneratedIds() {
        JsonView with = view(llama.completion("""
                {"prompt": "hello", "n_predict": 3, "ignore_eos": true, "return_tokens": true}""").getEntity());
        JsonView without = view(llama.completion("""
                {"prompt": "hello", "n_predict": 3, "ignore_eos": true}""").getEntity());
        assertEquals(3, size(with, "tokens"));
        assertEquals(0, size(without, "tokens"));
    }

    @Test
    void theLaunchFlagSetsTheModeAndRefusesAnUnknownOne() {
        assertEquals("processed_logits", ServerLaunchArgs.parse("--logprobs-mode", "processed_logits")
                .properties().get(ServerOptions.LOGPROBS_MODE_PROPERTY));
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> ServerLaunchArgs.parse("--logprobs-mode", "softmax"));
        assertTrue(e.getMessage().contains("--logprobs-mode"), e.getMessage());
        String previous = System.getProperty(ServerOptions.LOGPROBS_MODE_PROPERTY);
        try {
            System.setProperty(ServerOptions.LOGPROBS_MODE_PROPERTY, "raw_logits");
            assertEquals(LogprobsMode.RAW_LOGITS, ServerOptions.fromSystemProperties().logprobsMode());
            System.setProperty(ServerOptions.LOGPROBS_MODE_PROPERTY, "nope");
            IllegalArgumentException p = assertThrows(IllegalArgumentException.class,
                    ServerOptions::fromSystemProperties);
            assertTrue(p.getMessage().contains(ServerOptions.LOGPROBS_MODE_PROPERTY), p.getMessage());
        } finally {
            if (previous == null) {
                System.clearProperty(ServerOptions.LOGPROBS_MODE_PROPERTY);
            } else {
                System.setProperty(ServerOptions.LOGPROBS_MODE_PROPERTY, previous);
            }
        }
        assertEquals(LogprobsMode.RAW_LOGPROBS, ServerOptions.defaults().logprobsMode());
    }

    private static List<String> tokensOf(List<?> entries) {
        return entries.stream().map(o -> (String) ((Map<?, ?>) o).get("token")).toList();
    }
}

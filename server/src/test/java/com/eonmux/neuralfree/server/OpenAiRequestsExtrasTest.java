package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.RepetitionDetection;
import com.eonmux.neuralfree.gguf.grammar.ConstraintSpec;
import org.junit.jupiter.api.Test;

import static com.eonmux.neuralfree.server.ServerTestSupport.errorOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.statusOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The vLLM request extras of a body: their mapping onto the engine request, the fields that refuse by name, vLLM's
 * {@code repetition_detection} object, the null and empty defaults of the token controls, and vLLM's
 * {@code cache_salt} validation on the OpenAI and Anthropic bodies.
 */
class OpenAiRequestsExtrasTest {

    private static final int VOCAB = 16;

    private static GenerationRequest sampling(String json) {
        return OpenAiRequests.sampling(view(json), -1, VOCAB, false);
    }

    @Test
    void theExtrasMapOntoTheEngineRequest() {
        GenerationRequest r = sampling("""
                {"stop_token_ids": [4, 9, 4], "allowed_token_ids": [2, 7], "prompt_logprobs": 2,
                 "include_stop_str_in_output": true, "skip_special_tokens": false,
                 "spaces_between_special_tokens": false, "priority": -3, "cache_salt": "tenant"}""");
        assertArrayEquals(new int[]{4, 9}, r.stopTokenIds());
        assertArrayEquals(new int[]{2, 7}, r.allowedTokenIds());
        assertEquals(3, r.promptLogprobs(), "vLLM's n alternatives become n + 1 entries: the prompt token leads");
        assertTrue(r.includeStopString());
        assertTrue(!r.skipSpecialTokens());
        assertTrue(!r.spacesBetweenSpecialTokens());
        assertEquals(-3, r.priority());
        assertEquals("tenant", r.cacheSalt());
    }

    @Test
    void promptLogprobsMapsEveryVocabularyTokenAndDefaultsUnderEcho() {
        assertEquals(0, sampling("{}").promptLogprobs());
        assertEquals(1, sampling("{\"prompt_logprobs\": 0}").promptLogprobs(), "0 means the prompt token alone");
        assertEquals(GenerationRequest.PROMPT_LOGPROBS_ALL, sampling("{\"prompt_logprobs\": -1}").promptLogprobs());
        assertEquals(4, OpenAiRequests.echoPromptLogprobs(0, true, 3), "the echo default takes the logprobs count");
        assertEquals(5, OpenAiRequests.echoPromptLogprobs(5, true, 3), "an explicit value beats the echo default");
        assertEquals(0, OpenAiRequests.echoPromptLogprobs(0, false, 3));
    }

    @Test
    void truncateCutsFirstOrLast() {
        int[] tokens = {0, 1, 2, 3, 4, 5};
        assertArrayEquals(tokens, OpenAiRequests.truncate(view("{}"), tokens));
        assertArrayEquals(new int[]{0, 1},
                OpenAiRequests.truncate(view("{\"truncate_prompt_tokens\": 2}"), tokens), "right keeps the first n");
        assertArrayEquals(new int[]{4, 5},
                OpenAiRequests.truncate(view("{\"truncate_prompt_tokens\": 2, \"truncation_side\": \"left\"}"),
                        tokens), "left keeps the last n");
        assertArrayEquals(tokens,
                OpenAiRequests.truncate(view("{\"truncate_prompt_tokens\": 99}"), tokens),
                "a bound at or above the length keeps everything");
    }

    @Test
    void theRefusalsNameTheField() {
        assertEquals(400, statusOf(() -> sampling("{\"stop_token_ids\": [16]}")));
        assertEquals(400, statusOf(() -> sampling("{\"allowed_token_ids\": [4, 99]}")));
        assertEquals(400, statusOf(() -> sampling("{\"allowed_token_ids\": []}")));
        assertEquals(400, statusOf(() -> sampling("{\"prompt_logprobs\": -2}")));
        assertEquals(400, statusOf(() -> sampling("{\"cache_salt\": \"\"}")));
        int[] tokens = {0, 1, 2};
        assertEquals(400, statusOf(() -> OpenAiRequests.truncate(view("{\"truncate_prompt_tokens\": 0}"), tokens)));
        assertEquals(400, statusOf(() -> OpenAiRequests.truncate(view(
                "{\"truncate_prompt_tokens\": 2, \"truncation_side\": \"middle\"}"), tokens)));
        assertTrue(errorOf(() -> sampling("{\"stop_token_ids\": [16]}")).contains("stop_token_ids"));
        assertTrue(errorOf(() -> OpenAiRequests.truncate(view("{\"truncate_prompt_tokens\": 0}"), tokens))
                .contains("truncate_prompt_tokens"));
    }

    @Test
    void repetitionDetectionMapsVllmsObject() {
        assertEquals(RepetitionDetection.of(8, 2, 3), sampling("""
                {"repetition_detection": {"max_pattern_size": 8, "min_pattern_size": 2, "min_count": 3}}""")
                .repetitionDetection());
        assertEquals(RepetitionDetection.of(4, 0, 2), sampling("""
                {"repetition_detection": {"max_pattern_size": 4, "min_count": 2}}""").repetitionDetection(),
                "an absent min_pattern_size is 0, patterns from one token up");
        assertEquals(RepetitionDetection.disabled(), sampling("{}").repetitionDetection());
        assertEquals(RepetitionDetection.disabled(), sampling("{\"repetition_detection\": null}")
                .repetitionDetection());
        assertEquals(RepetitionDetection.disabled(), sampling("{\"repetition_detection\": {}}")
                .repetitionDetection(), "vLLM's all-zero defaults disable detection");
        assertEquals(RepetitionDetection.of(Integer.MAX_VALUE, 1, 2), sampling("""
                {"repetition_detection": {"max_pattern_size": 99999999999, "min_pattern_size": 1,
                 "min_count": 2}}""").repetitionDetection(), "sizes beyond int saturate");
    }

    @Test
    void repetitionDetectionRefusesByName() {
        String[] refused = {
                "[1, 2]",
                "3",
                "{\"max_pattern_size\": 4, \"min_count\": 1}",
                "{\"max_pattern_size\": 4}",
                "{\"max_pattern_size\": -1, \"min_count\": 2}",
                "{\"max_pattern_size\": 2, \"min_pattern_size\": 3, \"min_count\": 2}",
                "{\"max_pattern_size\": 0, \"min_pattern_size\": 1}",
                "{\"max_pattern_size\": 2.5, \"min_count\": 2}",
                "{\"max_pattern_size\": \"4\", \"min_count\": 2}",
                "{\"max_pattern_size\": 4, \"min_count\": 2, \"window\": 9}"};
        for (String value : refused) {
            String body = "{\"repetition_detection\": " + value + "}";
            assertEquals(400, statusOf(() -> sampling(body)), body);
            assertTrue(errorOf(() -> sampling(body)).contains("repetition_detection"), body);
        }
        assertTrue(errorOf(() -> sampling("{\"repetition_detection\": {\"window\": 9}}")).contains("window"),
                "an unknown member is named");
    }

    @Test
    void nullAndEmptyTokenControlsTakeVllmsDefaults() {
        GenerationRequest r = sampling("""
                {"stop_token_ids": null, "allowed_token_ids": null, "include_stop_str_in_output": null,
                 "skip_special_tokens": null, "spaces_between_special_tokens": null}""");
        assertEquals(0, r.stopTokenIds().length);
        assertEquals(0, r.allowedTokenIds().length);
        assertTrue(!r.includeStopString());
        assertTrue(r.skipSpecialTokens());
        assertTrue(r.spacesBetweenSpecialTokens());
        assertEquals(0, sampling("{\"stop_token_ids\": []}").stopTokenIds().length,
                "vLLM's stop_token_ids default is the empty list");
        for (String body : new String[]{"{\"skip_special_tokens\": \"no\"}",
                "{\"spaces_between_special_tokens\": 0}", "{\"include_stop_str_in_output\": \"yes\"}",
                "{\"stop_token_ids\": 4}", "{\"stop_token_ids\": [1.5]}"}) {
            String field = body.substring(2, body.indexOf('"', 2));
            assertEquals(400, statusOf(() -> sampling(body)), body);
            assertTrue(errorOf(() -> sampling(body)).contains(field), body);
        }
    }

    @Test
    void theCacheSaltFollowsVllmsValidation() {
        assertEquals("x".repeat(OpenAiRequests.MAX_CACHE_SALT),
                sampling("{\"cache_salt\": \"" + "x".repeat(OpenAiRequests.MAX_CACHE_SALT) + "\"}").cacheSalt());
        assertNull(sampling("{\"cache_salt\": null}").cacheSalt(), "null means no salt, as omitted");
        assertNull(sampling("{}").cacheSalt());
        String[] refused = {"\"\"", "7", "[\"a\"]", "\"" + "x".repeat(OpenAiRequests.MAX_CACHE_SALT + 1) + "\"",
                "\"tenant@a\"", "\"tenant/a\"", "\"tenant\\\\a\"", "\"tenant\\u0000a\""};
        for (String value : refused) {
            String body = "{\"cache_salt\": " + value + "}";
            assertEquals(400, statusOf(() -> sampling(body)), body);
            assertTrue(errorOf(() -> sampling(body)).contains("cache_salt"), body);
        }
    }

    @Test
    void anAnthropicBodyCarriesItsCacheSalt() {
        assertEquals("tenant", AnthropicRequests.request(view(
                "{\"max_tokens\": 4, \"cache_salt\": \"tenant\"}")).cacheSalt());
        assertEquals(400, statusOf(() -> AnthropicRequests.request(view(
                "{\"max_tokens\": 4, \"cache_salt\": \"a/b\"}"))));
    }

    @Test
    void continuousUsageStatsNeedsBothStreamOptions() {
        assertTrue(!OpenAiRequests.continuousUsageStats(view("{}")));
        assertTrue(!OpenAiRequests.continuousUsageStats(view(
                "{\"stream\": true, \"stream_options\": {\"continuous_usage_stats\": true}}")),
                "continuous usage needs include_usage");
        assertTrue(OpenAiRequests.continuousUsageStats(view(
                "{\"stream\": true, \"stream_options\": {\"include_usage\": true, \"continuous_usage_stats\": true}}")));
    }

    @Test
    void bestOfDefaultsToNAndRefusesByName() {
        assertEquals(2, OpenAiRequests.bestOf(view("{}"), 2, false), "absent best_of means n");
        assertEquals(4, OpenAiRequests.bestOf(view("{\"best_of\": 4}"), 2, false));
        assertEquals(400, statusOf(() -> OpenAiRequests.bestOf(view("{\"best_of\": 1}"), 2, false)),
                "best_of below n refuses");
        assertEquals(400, statusOf(() -> OpenAiRequests.bestOf(view(
                "{\"best_of\": " + (OpenAiRequests.MAX_CHOICES + 1) + "}"), 1, false)),
                "best_of above the choices bound refuses");
        assertEquals(400, statusOf(() -> OpenAiRequests.bestOf(view("{\"best_of\": \"two\"}"), 1, false)));
        assertEquals(400, statusOf(() -> OpenAiRequests.bestOf(view("{\"best_of\": 3}"), 1, true)),
                "best_of over n refuses on a stream");
        assertTrue(errorOf(() -> OpenAiRequests.bestOf(view("{\"best_of\": 1}"), 2, false)).contains("best_of"));
        assertTrue(errorOf(() -> OpenAiRequests.bestOf(view("{\"best_of\": 3}"), 1, true)).contains("best_of"));
    }

    @Test
    void logprobTokenIdsMapOntoTheEngineRequestAndRefuseByName() {
        GenerationRequest r = OpenAiRequests.sampling(view("{\"logprobs\": 2, \"logprob_token_ids\": [7, 3, 7]}"),
                2, VOCAB, false);
        assertArrayEquals(new int[]{7, 3}, r.logprobTokenIds(),
                "the request keeps the ids in request order, duplicates drop");
        assertEquals(0, OpenAiRequests.sampling(view("{\"logprobs\": 2}"), 2, VOCAB, false).logprobTokenIds()
                .length, "absent logprob_token_ids leaves the top-k capture");
        assertEquals(400, statusOf(() -> OpenAiRequests.sampling(
                view("{\"logprob_token_ids\": [7]}"), -1, VOCAB, false)), "logprob_token_ids needs logprobs");
        assertEquals(0, OpenAiRequests.sampling(view("{\"logprobs\": 1, \"logprob_token_ids\": []}"), 1, VOCAB,
                false).logprobTokenIds().length, "an empty list stands for absent, as vLLM reads it");
        assertEquals(0, OpenAiRequests.sampling(view("{\"logprob_token_ids\": null}"), -1, VOCAB, false)
                .logprobTokenIds().length, "null stands for absent, even without logprobs");
        assertEquals(400, statusOf(() -> OpenAiRequests.sampling(
                view("{\"logprobs\": 1, \"logprob_token_ids\": \"4\"}"), 1, VOCAB, false)), "a non-array refuses");
        assertEquals(400, statusOf(() -> OpenAiRequests.sampling(
                view("{\"logprobs\": 1, \"logprob_token_ids\": [" + VOCAB + "]}"), 1, VOCAB, false)),
                "an out-of-vocabulary id refuses");
        assertTrue(errorOf(() -> OpenAiRequests.sampling(view("{\"logprob_token_ids\": [7]}"), -1, VOCAB, false))
                .contains("logprob_token_ids"));
    }

    @Test
    void logprobTokenIdsFollowVllmsBounds() {
        StringBuilder ids = new StringBuilder();
        for (int i = 0; i <= OpenAiRequests.MAX_LOGPROB_TOKEN_IDS; i++) {
            ids.append(i == 0 ? "" : ", ").append(i % VOCAB);
        }
        String tooMany = "{\"logprobs\": 1, \"logprob_token_ids\": [" + ids + "]}";
        assertTrue(errorOf(() -> OpenAiRequests.sampling(view(tooMany), 1, VOCAB, false))
                .contains("at most " + OpenAiRequests.MAX_LOGPROB_TOKEN_IDS), "vLLM's MAX_LOGPROB_TOKEN_IDS");
    }

    @Test
    void bestOfCandidatesCaptureTheirScoresOnlyWhenRankingNeedsThem() {
        GenerationRequest plain = sampling("{}");
        assertEquals(1, OpenAiRequests.scoredCandidates(plain, 1, 3).logprobs(),
                "candidates of best_of > n capture the sampled token's logprob for the ranking");
        assertEquals(0, OpenAiRequests.scoredCandidates(plain, 2, 2).logprobs(), "best_of = n ranks nothing");
        GenerationRequest asked = OpenAiRequests.sampling(view("{\"logprobs\": 3}"), 3, VOCAB, false);
        assertEquals(4, OpenAiRequests.scoredCandidates(asked, 1, 3).logprobs(), "a requested capture stands");
    }

    @Test
    void vllmOnlyPassThroughFieldsRefuseByName() {
        assertEquals(400, statusOf(() -> sampling("{\"vllm_xargs\": \"flag\"}")),
                "vllm_xargs must be an object");
        assertTrue(errorOf(() -> sampling("{\"vllm_xargs\": {\"my_flag\": {}}}")).contains("vllm_xargs.my_flag"));
        assertEquals(400, statusOf(() -> sampling("{\"kv_transfer_params\": {\"rank\": 0}}")));
        assertTrue(errorOf(() -> sampling("{\"kv_transfer_params\": {\"rank\": 0}}"))
                .contains("kv_transfer_params.rank"));
        assertEquals(400, statusOf(() -> sampling("{\"kv_transfer_params\": {}}")));
        assertTrue(errorOf(() -> sampling("{\"kv_transfer_params\": {}}")).contains("do_remote_decode"));
        assertEquals(400, statusOf(() -> sampling("{\"kv_transfer_params\": \"rank\"}")),
                "a value no connector could read refuses by name");
        assertTrue(errorOf(() -> sampling("{\"kv_transfer_params\": []}")).contains("kv_transfer_params"));
        assertEquals(200, statusOf(() -> sampling("{\"vllm_xargs\": null}")),
                "an explicit null stands for absent");
        assertEquals(200, statusOf(() -> sampling("{\"vllm_xargs\": {}}")),
                "an empty object carries no argument");
    }

    @Test
    void structuredOutputsJsonOptionsMapOntoTheConstraint() {
        ConstraintSpec spec = OpenAiRequests.constraint(view("{\"structured_outputs\": {\"json\": {\"type\":"
                + " \"object\"}, \"disable_any_whitespace\": true, \"whitespace_pattern\": \"[ ]?\","
                + " \"disable_additional_properties\": true}}"));
        assertEquals(ConstraintSpec.Kind.JSON_SCHEMA, spec.kind());
        assertEquals(com.eonmux.neuralfree.gguf.grammar.JsonSchemaToGbnf.WhitespaceMode.COMPACT, spec.whitespace());
        assertEquals("[ ]?", spec.whitespacePattern());
        assertTrue(spec.disableAdditionalProperties());

        ConstraintSpec plain = OpenAiRequests.constraint(view(
                "{\"structured_outputs\": {\"json\": {\"type\": \"object\"}}}"));
        assertNull(plain.whitespacePattern(), "an absent whitespace_pattern is the mode's rule");
        assertTrue(!plain.disableAdditionalProperties());
        assertEquals(com.eonmux.neuralfree.gguf.grammar.JsonSchemaToGbnf.WhitespaceMode.FLEXIBLE,
                plain.whitespace(), "an absent disable_any_whitespace keeps the flexible mode");

        ConstraintSpec jsonObject = OpenAiRequests.constraint(view(
                "{\"structured_outputs\": {\"json_object\": true, \"disable_any_whitespace\": true}}"));
        assertEquals(ConstraintSpec.Kind.JSON_OBJECT, jsonObject.kind());
        assertEquals(com.eonmux.neuralfree.gguf.grammar.JsonSchemaToGbnf.WhitespaceMode.COMPACT,
                jsonObject.whitespace());
    }

    @Test
    void structuredOutputsJsonOptionsRefuseNonJsonKinds() {
        try {
            OpenAiRequests.constraint(view("{\"structured_outputs\": {\"regex\": \"[0-9]+\","
                    + " \"disable_additional_properties\": true}}"));
            org.junit.jupiter.api.Assertions.fail("the options must not attach to a regex");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("disable_additional_properties"), e.getMessage());
            assertTrue(e.getMessage().contains("apply to json and json_object constraints only"), e.getMessage());
        }
        assertEquals(400, statusOf(() -> OpenAiRequests.constraint(view(
                "{\"structured_outputs\": {\"whitespace_pattern\": \"[ ]*\"}}"))));
    }
}

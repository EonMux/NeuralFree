package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.GGUFTokenizer;
import com.eonmux.neuralfree.gguf.tokenizer.TokenizerVocab;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The vLLM request extras on the engine's stop and render side: {@code stop_token_ids} end a generation like an
 * end-of-generation token, {@code include_stop_str_in_output} keeps a matched stop string in the text,
 * {@code skip_special_tokens} false reveals control-token texts (consecutive ones spaced under
 * {@code spaces_between_special_tokens}), {@code repetition_detection} ends a generation whose output repeats a
 * token pattern (vLLM's {@code check_stop} order: end of generation, the budget, {@code min_tokens}, then
 * repetition; a stop string completed by the same token wins), and the builder validates and carries the new
 * settings.
 */
public class RequestExtrasTest {

    /** Vocab {@code a .. h} with the token {@code c} (id 2) a control token; no EOS token configured. */
    private static GGUFTokenizer tokenizer() {
        return new GGUFTokenizer(TokenizerVocab.fromMetadata(metadata()));
    }

    /** {@link #tokenizer()} with {@code h} (id 7) the vocabulary's EOS token. */
    private static GGUFTokenizer tokenizerWithEos() {
        Map<String, Object> md = metadata();
        md.put("tokenizer.ggml.eos_token_id", EOS_ID);
        return new GGUFTokenizer(TokenizerVocab.fromMetadata(md));
    }

    private static final int EOS_ID = 7;

    private static Map<String, Object> metadata() {
        List<Object> tokens = new ArrayList<>(Arrays.<Object>asList("a", "b", "c", "d", "e", "f", "g", "h"));
        List<Object> types = new ArrayList<>(Arrays.<Object>asList(1, 1, 3, 1, 1, 1, 1, 1));
        Map<String, Object> md = new HashMap<>();
        md.put(TokenizerVocab.KEY_MODEL, "gpt2");
        md.put(TokenizerVocab.KEY_PRE, "gpt-2");
        md.put(TokenizerVocab.KEY_TOKENS, tokens);
        md.put(TokenizerVocab.KEY_TOKEN_TYPE, types);
        md.put(TokenizerVocab.KEY_MERGES, new ArrayList<Object>());
        return md;
    }

    private static StopCriteria criteria(GGUFTokenizer tok, GenerationRequest request) {
        return StopCriteria.of(request, tok, ReasoningFilter.none());
    }

    private static StopCriteria.Session session(GGUFTokenizer tok, GenerationRequest request) {
        return criteria(tok, request).start((channel, tokenId, text) -> { }, null);
    }

    private static GenerationResult.Timings timings() {
        return GenerationResult.Timings.of(0, 0, 1, 1);
    }

    @Test
    public void stopTokenIdsEndGenerationLikeEosAndStayOutOfTheOutput() {
        GGUFTokenizer tok = tokenizer();
        StopCriteria.Session s = session(tok, GenerationRequest.builder()
                .sampler(SamplerConfig.greedyDecoding()).stopTokenIds(3).build());
        assertNull(s.offer(0));
        assertNull(s.offer(1));
        assertEquals(StopReason.EOS, s.offer(3));
        GenerationResult result = s.finish(StopReason.EOS, timings());
        assertArrayEquals("the stop token itself stays out", new int[]{0, 1}, result.tokens());
        assertEquals("the accepted tokens' text stands", "ab", result.text());
        assertEquals(StopReason.EOS, result.stopReason());
        assertEquals("vLLM's stop_reason names the matched stop token id", 3, result.stopTokenId());
    }

    @Test
    public void otherEndingsLeaveTheStopTokenIdUnset() {
        GGUFTokenizer tok = tokenizer();
        StopCriteria.Session budget = session(tok, GenerationRequest.builder()
                .sampler(SamplerConfig.greedyDecoding()).maxNewTokens(1).build());
        assertEquals(StopReason.MAX_TOKENS, budget.offer(4));
        assertEquals("a length end reports no stop token", GenerationResult.NO_STOP_TOKEN,
                budget.finish(StopReason.MAX_TOKENS, timings()).stopTokenId());

        StopCriteria.Session string = session(tok, GenerationRequest.builder()
                .sampler(SamplerConfig.greedyDecoding()).stopStrings("ee").build());
        assertNull("the first e may begin the stop string and holds back", string.offer(4));
        assertEquals(StopReason.STOP_STRING, string.offer(4));
        GenerationResult ended = string.finish(StopReason.STOP_STRING, timings());
        assertEquals("a stop-string end names the string, not a token id", GenerationResult.NO_STOP_TOKEN,
                ended.stopTokenId());
        assertEquals("ee", ended.stopString());
    }

    /** vLLM's {@code check_stop}: the EOS token ends without a stop reason even when it is also a stop token id,
     *  but under ignore-eos only the stop token ids can end on it, and the id is reported. */
    @Test
    public void theEosTokenReportsAStopTokenIdOnlyUnderIgnoreEos() {
        GGUFTokenizer tok = tokenizerWithEos();
        StopCriteria.Session plain = session(tok, GenerationRequest.builder()
                .sampler(SamplerConfig.greedyDecoding()).stopTokenIds(EOS_ID).build());
        assertEquals(StopReason.EOS, plain.offer(EOS_ID));
        assertEquals("the EOS end carries no stop reason", GenerationResult.NO_STOP_TOKEN,
                plain.finish(StopReason.EOS, timings()).stopTokenId());

        StopCriteria.Session ignoring = session(tok, GenerationRequest.builder()
                .sampler(SamplerConfig.greedyDecoding()).ignoreEos(true).stopTokenIds(EOS_ID).build());
        assertEquals(StopReason.EOS, ignoring.offer(EOS_ID));
        assertEquals("under ignore-eos the stop token id ended it", EOS_ID,
                ignoring.finish(StopReason.EOS, timings()).stopTokenId());
    }

    @Test
    public void stopTokenIdsStopEvenWhenIgnoreEosDropsAndTheVocabularyHasNoEos() {
        GGUFTokenizer tok = tokenizer();
        StopCriteria.Session s = session(tok, GenerationRequest.builder()
                .sampler(SamplerConfig.greedyDecoding()).ignoreEos(true).stopTokenIds(5).build());
        assertNull(s.offer(4));
        assertEquals(StopReason.EOS, s.offer(5));
    }

    /** Stop string over three tokens, the middle one a control token rendered under {@code skip_special_tokens} false. */
    @Test
    public void includeStopStringKeepsTheMatchInTheReleasedText() {
        GGUFTokenizer tok = tokenizer();
        GenerationRequest kept = GenerationRequest.builder()
                .sampler(SamplerConfig.greedyDecoding()).stopStrings("cde").skipSpecialTokens(false)
                .includeStopString(true).build();
        StopCriteria.Session s = session(tok, kept);
        assertNull(s.offer(0));
        assertNull(s.offer(1));
        assertNull(s.offer(2));
        assertNull(s.offer(3));
        assertEquals(StopReason.STOP_STRING, s.offer(4));
        assertEquals("the match itself releases with the rest", "abcde",
                s.finish(StopReason.STOP_STRING, timings()).text());

        GenerationRequest cut = GenerationRequest.builder()
                .sampler(SamplerConfig.greedyDecoding()).stopStrings("cde").skipSpecialTokens(false).build();
        StopCriteria.Session cutSession = session(tok, cut);
        assertNull(cutSession.offer(0));
        assertNull(cutSession.offer(1));
        assertNull(cutSession.offer(2));
        assertNull(cutSession.offer(3));
        assertEquals(StopReason.STOP_STRING, cutSession.offer(4));
        assertEquals("the text stops before the match", "ab",
                cutSession.finish(StopReason.STOP_STRING, timings()).text());
    }

    @Test
    public void specialTokensStayHiddenByDefaultAndRevealUnderSkipSpecialTokensFalse() {
        GGUFTokenizer tok = tokenizer();
        StopCriteria.Session hidden = session(tok, GenerationRequest.builder()
                .sampler(SamplerConfig.greedyDecoding()).build());
        assertNull(hidden.offer(0));
        assertNull(hidden.offer(2));
        assertEquals("a", hidden.finish(StopReason.MAX_TOKENS, timings()).text());

        StopCriteria.Session shown = session(tok, GenerationRequest.builder()
                .sampler(SamplerConfig.greedyDecoding()).skipSpecialTokens(false).build());
        shown.offer(2);
        assertEquals("c", shown.finish(StopReason.MAX_TOKENS, timings()).text());
    }

    @Test
    public void consecutiveSpecialTokenTextsSpaceUnderSpacesBetweenSpecialTokens() {
        GGUFTokenizer tok = tokenizer();
        StopCriteria.Session spaced = session(tok, GenerationRequest.builder()
                .sampler(SamplerConfig.greedyDecoding()).skipSpecialTokens(false).build());
        spaced.offer(2);
        spaced.offer(2);
        assertEquals("c c", spaced.finish(StopReason.MAX_TOKENS, timings()).text());

        StopCriteria.Session joined = session(tok, GenerationRequest.builder()
                .sampler(SamplerConfig.greedyDecoding()).skipSpecialTokens(false)
                .spacesBetweenSpecialTokens(false).build());
        joined.offer(2);
        joined.offer(2);
        assertEquals("cc", joined.finish(StopReason.MAX_TOKENS, timings()).text());
    }

    @Test
    public void renderPromptFollowsTheSameSpecialRendering() {
        GGUFTokenizer tok = tokenizer();
        StopCriteria hidden = criteria(tok, GenerationRequest.builder().build());
        assertEquals("", hidden.renderPrompt(new int[]{2, 2}));
        StopCriteria shownSpaced = criteria(tok, GenerationRequest.builder().skipSpecialTokens(false).build());
        assertEquals("c c", shownSpaced.renderPrompt(new int[]{2, 2}));
        StopCriteria shownJoined = criteria(tok, GenerationRequest.builder().skipSpecialTokens(false)
                .spacesBetweenSpecialTokens(false).build());
        assertEquals("cc", shownJoined.renderPrompt(new int[]{2, 2}));
    }

    /** Feeds {@code tokens} until a stop, returning the stop reason (null when none) and the offered count. */
    private static StopReason feed(StopCriteria.Session s, int... tokens) {
        for (int token : tokens) {
            StopReason reason = s.offer(token);
            if (reason != null) {
                return reason;
            }
        }
        return null;
    }

    private static GenerationRequest.Builder repeating() {
        return GenerationRequest.builder().sampler(SamplerConfig.greedyDecoding()).maxNewTokens(100)
                .repetitionDetection(RepetitionDetection.of(5, 2, 3));
    }

    @Test
    public void aRepeatedPatternEndsGenerationWithRepetitionAndKeepsItsTokens() {
        GGUFTokenizer tok = tokenizer();
        StopCriteria.Session s = session(tok, repeating().build());
        assertNull(feed(s, 0, 1, 0, 1, 0));
        assertEquals(StopReason.REPETITION, s.offer(1));
        GenerationResult result = s.finish(StopReason.REPETITION, timings());
        assertArrayEquals(new int[]{0, 1, 0, 1, 0, 1}, result.tokens());
        assertEquals("ababab", result.text());
        assertEquals(StopReason.REPETITION, result.stopReason());
        assertNull(result.stopString());
    }

    @Test
    public void withoutDetectionTheSameOutputContinues() {
        GGUFTokenizer tok = tokenizer();
        StopCriteria.Session s = session(tok, repeating().repetitionDetection(RepetitionDetection.disabled()).build());
        assertNull(feed(s, 0, 1, 0, 1, 0, 1, 0, 1));
        StopCriteria.Session varied = session(tok, repeating().build());
        assertNull("no pattern, no stop", feed(varied, 0, 1, 3, 4, 5, 6));
    }

    @Test
    public void repetitionWaitsForMinTokens() {
        GGUFTokenizer tok = tokenizer();
        StopCriteria.Session s = session(tok, repeating().minTokens(8).build());
        assertNull("six tokens are below min_tokens", feed(s, 0, 1, 0, 1, 0, 1));
        assertNull(s.offer(0));
        assertEquals("the eighth token reaches min_tokens", StopReason.REPETITION, s.offer(1));
    }

    @Test
    public void stopTokensAndTheBudgetComeBeforeRepetition() {
        GGUFTokenizer tok = tokenizer();
        StopCriteria.Session eos = session(tok, repeating().stopTokenIds(7).build());
        assertEquals(StopReason.EOS, feed(eos, 0, 1, 0, 1, 7));
        StopCriteria.Session budget = session(tok, repeating().maxNewTokens(6).build());
        assertEquals("the budget ends the same token first", StopReason.MAX_TOKENS, feed(budget, 0, 1, 0, 1, 0, 1));
    }

    @Test
    public void aStopStringCompletedByTheSameTokenWins() {
        GGUFTokenizer tok = tokenizer();
        StopCriteria.Session s = session(tok, repeating().repetitionDetection(RepetitionDetection.of(2, 1, 2))
                .stopStrings("bab").build());
        assertNull(feed(s, 0, 1, 0));
        assertEquals(StopReason.STOP_STRING, s.offer(1));
        assertEquals("a", s.finish(StopReason.STOP_STRING, timings()).text());
    }

    @Test
    public void theRepetitionStopReleasesTextHeldForAStopString() {
        GGUFTokenizer tok = tokenizer();
        StopCriteria.Session s = session(tok, repeating().repetitionDetection(RepetitionDetection.of(1, 1, 3))
                .stopStrings("ddx").build());
        assertEquals(StopReason.REPETITION, feed(s, 0, 3, 3, 3));
        GenerationResult result = s.finish(StopReason.REPETITION, timings());
        assertEquals("the held-back tail releases at the stop", "addd", result.text());
        assertEquals(StopReason.REPETITION, result.stopReason());
    }

    @Test
    public void theBuilderValidatesTheNewIdsAndCounts() {
        GenerationRequest.Builder b = GenerationRequest.builder();
        assertThrows(IllegalArgumentException.class, () -> b.stopTokenIds(2, -1));
        assertThrows(IllegalArgumentException.class, () -> b.allowedTokenIds(-4));
        assertThrows(IllegalArgumentException.class, () -> b.promptLogprobs(-2));
        assertEquals("duplicates collapse sorted", 1, GenerationRequest.builder().stopTokenIds(5, 5)
                .build().stopTokenIds().length);
        assertEquals("prompt logprobs accept -1 for every token", GenerationRequest.PROMPT_LOGPROBS_ALL,
                GenerationRequest.builder().promptLogprobs(-1).build().promptLogprobs());
    }

    @Test
    public void toBuilderCarriesTheExtras() {
        GenerationRequest original = GenerationRequest.builder()
                .sampler(SamplerConfig.greedyDecoding())
                .stopTokenIds(3, 5)
                .allowedTokenIds(1, 2, 3)
                .logprobTokenIds(6, 2, 6)
                .promptLogprobs(2)
                .includeStopString(true)
                .skipSpecialTokens(false)
                .spacesBetweenSpecialTokens(false)
                .repetitionDetection(RepetitionDetection.of(4, 2, 3))
                .priority(-3)
                .cacheSalt("tenant")
                .build();
        GenerationRequest copy = original.toBuilder().build();
        assertArrayEquals(new int[]{3, 5}, copy.stopTokenIds());
        assertArrayEquals(new int[]{1, 2, 3}, copy.allowedTokenIds());
        assertArrayEquals("the logprob token ids keep their request order, duplicates drop",
                new int[]{6, 2}, copy.logprobTokenIds());
        assertTrue(copy.namesLogprobTokenIds());
        assertEquals(2, copy.promptLogprobs());
        assertTrue(copy.includeStopString());
        assertTrue(!copy.skipSpecialTokens());
        assertTrue(!copy.spacesBetweenSpecialTokens());
        assertEquals(RepetitionDetection.of(4, 2, 3), copy.repetitionDetection());
        assertEquals("model defaults keep the detection", RepetitionDetection.of(4, 2, 3), original
                .withModelDefaults(SamplingDefaults.builder().topK(7).build()).repetitionDetection());
        assertEquals(-3, copy.priority());
        assertEquals("tenant", copy.cacheSalt());
        assertTrue(GenerationRequest.defaults().cacheSalt() == null);
        assertEquals(0, GenerationRequest.defaults().priority());
        assertTrue(GenerationRequest.defaults().skipSpecialTokens());
        assertTrue(GenerationRequest.defaults().spacesBetweenSpecialTokens());
        assertEquals(RepetitionDetection.disabled(), GenerationRequest.defaults().repetitionDetection());
        assertThrows(NullPointerException.class, () -> GenerationRequest.builder().repetitionDetection(null));
        assertArrayEquals("the logprob token ids default to none", new int[0],
                GenerationRequest.defaults().logprobTokenIds());
    }

    @Test
    public void theStopTokenIdCarriesThroughTheCopyMethods() {
        GenerationResult base = new GenerationResult(new int[]{1}, "a", StopReason.EOS, null, timings());
        assertEquals(GenerationResult.NO_STOP_TOKEN, base.stopTokenId());
        assertEquals(3, base.withStopTokenId(3).stopTokenId());
        assertEquals("a", base.withStopTokenId(3).text());
        assertThrows(IllegalArgumentException.class, () -> base.withStopTokenId(-2));
        GenerationResult stopped = base.withStopTokenId(3);
        TokenScore[] step = {new TokenScore(1, -0.5f)};
        assertEquals("the logprobs copy keeps it", 3,
                stopped.withLogprobs(Arrays.<TokenScore[]>asList(step)).stopTokenId());
        assertEquals("the prompt logprobs copy keeps it", 3,
                stopped.withPromptLogprobs(Arrays.<TokenScore[]>asList(null, step)).stopTokenId());
        assertEquals("the reasoning copy keeps it", 3, stopped.withReasoningTokens(1).stopTokenId());
    }
}

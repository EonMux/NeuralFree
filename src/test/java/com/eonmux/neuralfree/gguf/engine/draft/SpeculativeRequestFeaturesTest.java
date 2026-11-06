package com.eonmux.neuralfree.gguf.engine.draft;

import com.eonmux.neuralfree.gguf.GGUFTokenizer;
import com.eonmux.neuralfree.gguf.TokenCallback;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.LogprobsMode;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.engine.SpeculativeDecoder;
import com.eonmux.neuralfree.gguf.engine.StopReason;
import com.eonmux.neuralfree.gguf.engine.TokenScore;
import com.eonmux.neuralfree.gguf.engine.draft.SpeculativeDecoderFixtures.FakeTarget;
import com.eonmux.neuralfree.gguf.engine.draft.SpeculativeDecoderFixtures.PlainDrafter;
import com.eonmux.neuralfree.gguf.grammar.ConstraintSpec;
import org.junit.Test;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.Set;

import static com.eonmux.neuralfree.gguf.engine.draft.SpeculativeDecoderFixtures.N_CTX;
import static com.eonmux.neuralfree.gguf.engine.draft.SpeculativeDecoderFixtures.tokenizer;
import static com.eonmux.neuralfree.gguf.engine.draft.SpeculativeDecoderFixtures.tokenizerWithEos;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The request's sampler chain under every verify mode of {@link SpeculativeDecoder} (llama.cpp
 * {@code common_sampler_sample_and_accept_n} runs the full chain over the verify rows; vLLM advances the grammar over
 * the speculative tokens and rolls it back): the grammar, {@code min_tokens} and {@code bad_words} see the draft
 * tokens before a verify row even when block verification scores every row before deciding, {@code logit_bias}
 * holds in every mode, processed logprobs ({@link LogprobsMode#PROCESSED_LOGPROBS}, the chain's
 * distribution of each row) cover every emitted token in the verifier modes, synthetic acceptance never confirms a
 * token the chain excludes, adaptive-p is refused by name outside MATCH while mirostat runs in every mode, and a full
 * context shifts when context shift is on. Tokens {@code a b c d} are ids 0..3.
 */
public class SpeculativeRequestFeaturesTest {

    private static final int A = 0;
    private static final int B = 1;
    private static final int C = 2;
    private static final int D = 3;
    private static final int EOS = 4;
    private static final float[] UNIFORM = {0f, 0f, 0f, 0f};
    private static final double LN_HALF = Math.log(0.5);

    private static SamplerConfig stochastic(long seed) {
        return SamplerConfig.builder().temperature(1f).topK(0).topP(1f).minP(0f).seed(seed).build();
    }

    private static SpeculativeDecoder decoder(FakeTarget target, PlainDrafter drafter, GGUFTokenizer tokenizer,
                                              VerifyMode mode) {
        return SpeculativeDecoder.builder(target, drafter, tokenizer).draftMax(2).verifyMode(mode).build();
    }

    /** The target for {@code a (b | c) d}: uniform rows for the prefill, one verify round and one decode. */
    private static FakeTarget uniformTarget() {
        return new FakeTarget(false).add(UNIFORM).add(UNIFORM, UNIFORM, UNIFORM).add(UNIFORM);
    }

    private static GenerationRequest grammarRequest(long seed) {
        return GenerationRequest.builder().maxNewTokens(3).sampler(stochastic(seed)).logprobs(4)
                .logprobsMode(LogprobsMode.PROCESSED_LOGPROBS)
                .constraint(ConstraintSpec.gbnf("root ::= \"a\" (\"b\" | \"c\") \"d\"")).build();
    }

    /**
     * {@code root ::= "a" ("b" | "c") "d"} over uniform rows with the proposal {@code [b, d]} after {@code a}: the
     * verify row of {@code d} must be scored with {@code b} accepted into the grammar, or block verification sees
     * {@code d} excluded there and rejects into an {@code a} the grammar cannot take. Every lossless mode emits only
     * grammar sentences, both of them across seeds, and every token's logprobs are its grammar-masked row: {@code a}
     * and {@code d} with probability 1, {@code b} and {@code c} at one half each.
     */
    @Test
    public void theGrammarSeesTheDraftTokensBeforeEveryVerifyRow() {
        for (VerifyMode mode : new VerifyMode[]{VerifyMode.MATCH, VerifyMode.REJECTION, VerifyMode.BLOCK,
                VerifyMode.FLY}) {
            Set<Integer> middles = new HashSet<>();
            for (long seed = 0; seed < 40; seed++) {
                FakeTarget target = uniformTarget();
                GenerationResult r = decoder(target, new PlainDrafter(new int[]{B, D}), tokenizer(), mode)
                        .generate(new int[]{D}, grammarRequest(seed), (TokenCallback) null);
                String what = mode + " seed " + seed + " " + Arrays.toString(r.tokens());
                assertEquals(what, 1, target.verifyCalls.size());
                assertEquals(what, 3, r.tokens().length);
                assertEquals(what, A, r.tokens()[0]);
                assertTrue(what, r.tokens()[1] == B || r.tokens()[1] == C);
                assertEquals(what, D, r.tokens()[2]);
                middles.add(r.tokens()[1]);
                assertGrammarLogprobs(what, r);
            }
            if (mode != VerifyMode.FLY) {
                assertEquals(mode + " draws both grammar sentences", 2, middles.size());
            }
        }
    }

    private static void assertGrammarLogprobs(String what, GenerationResult r) {
        assertEquals(what + ": one capture per token", 3, r.logprobs().size());
        TokenScore[] first = r.logprobs().get(0);
        assertEquals(what, 1, first.length);
        assertEquals(what, A, first[0].tokenId());
        assertEquals(what, 0.0, first[0].logprob(), 1e-6);
        TokenScore[] middle = r.logprobs().get(1);
        assertEquals(what + ": the grammar leaves b and c", 2, middle.length);
        assertEquals(what, r.tokens()[1], middle[0].tokenId());
        assertEquals(what, r.tokens()[1] == B ? C : B, middle[1].tokenId());
        assertEquals(what, LN_HALF, middle[0].logprob(), 1e-5);
        assertEquals(what, LN_HALF, middle[1].logprob(), 1e-5);
        TokenScore[] last = r.logprobs().get(2);
        assertEquals(what, 1, last.length);
        assertEquals(what, D, last[0].tokenId());
        assertEquals(what, 0.0, last[0].logprob(), 1e-6);
    }

    /**
     * {@code min_tokens(2)} on rows that favour end-of-sequence: the first two tokens cannot end, the third can. The
     * proposal {@code [b, </s>]} puts the end at the third token, so its verify row must count the drafted {@code b}
     * ahead: block verification then confirms both and the generation ends inside the round.
     */
    @Test
    public void minTokensCountsTheDraftTokensBeforeEveryVerifyRow() {
        float[] eosFirst = {0f, 1f, 0f, 0f, 5f};
        for (VerifyMode mode : new VerifyMode[]{VerifyMode.MATCH, VerifyMode.REJECTION, VerifyMode.BLOCK,
                VerifyMode.FLY}) {
            FakeTarget target = new FakeTarget(false).add(eosFirst).add(eosFirst, eosFirst, eosFirst);
            GenerationRequest request = GenerationRequest.builder().maxNewTokens(8)
                    .sampler(SamplerConfig.greedyDecoding()).minTokens(2).build();
            GenerationResult r = decoder(target, new PlainDrafter(new int[]{B, EOS}), tokenizerWithEos(), mode)
                    .generate(new int[]{A}, request, (TokenCallback) null);
            String what = mode + " " + Arrays.toString(r.tokens());
            assertEquals(what, StopReason.EOS, r.stopReason());
            assertEquals(what, B, r.tokens()[0]);
            assertEquals(what, B, r.tokens()[1]);
            assertEquals(what + ": the end came from the verify round", 1, target.verifyCalls.size());
            assertTrue(what + ": no decode after the round", target.decodes.isEmpty());
        }
    }

    /**
     * {@code bad_words ["bc"]} with the proposal {@code [b, c]} after {@code a} on rows whose argmax is the
     * proposal: the verify row of {@code c} follows {@code b}, so the phrase bans it there and the round emits the
     * row's next choice {@code d} instead.
     */
    @Test
    public void badWordsSeeTheDraftTokensBeforeEveryVerifyRow() {
        float[] wantA = {3f, 0f, 0f, 0f};
        float[] wantB = {0f, 3f, 0f, 0f};
        float[] wantC = {0f, 0f, 3f, 2f};
        for (VerifyMode mode : new VerifyMode[]{VerifyMode.MATCH, VerifyMode.REJECTION, VerifyMode.BLOCK,
                VerifyMode.FLY}) {
            FakeTarget target = new FakeTarget(false).add(wantA).add(wantB, wantC, wantA);
            GenerationRequest request = GenerationRequest.builder().maxNewTokens(3)
                    .sampler(SamplerConfig.greedyDecoding()).badWords(Collections.singletonList("bc")).build();
            GenerationResult r = decoder(target, new PlainDrafter(new int[]{B, C}), tokenizer(), mode)
                    .generate(new int[]{D}, request, (TokenCallback) null);
            assertArrayEquals(mode.toString(), new int[]{A, B, D}, r.tokens());
        }
    }

    /** {@code logit_bias} outweighs every row and every proposal in every mode. */
    @Test
    public void logitBiasHoldsInEveryMode() {
        for (VerifyMode mode : new VerifyMode[]{VerifyMode.MATCH, VerifyMode.REJECTION, VerifyMode.BLOCK,
                VerifyMode.FLY}) {
            FakeTarget target = new FakeTarget(false).add(UNIFORM).add(UNIFORM, UNIFORM, UNIFORM)
                    .add(UNIFORM, UNIFORM, UNIFORM);
            GenerationRequest request = GenerationRequest.builder().maxNewTokens(3)
                    .sampler(stochastic(3)).logitBias(C, 50f).logprobs(1)
                    .logprobsMode(LogprobsMode.PROCESSED_LOGPROBS).build();
            GenerationResult r = decoder(target, new PlainDrafter(new int[]{A, B}, new int[]{A, B}), tokenizer(),
                    mode).generate(new int[]{D}, request, (TokenCallback) null);
            assertArrayEquals(mode.toString(), new int[]{C, C, C}, r.tokens());
            assertEquals(mode.toString(), 3, r.logprobs().size());
            for (TokenScore[] scores : r.logprobs()) {
                assertEquals(mode.toString(), C, scores[0].tokenId());
                assertEquals(mode.toString(), 0.0, scores[0].logprob(), 1e-5);
            }
        }
    }

    /**
     * {@code logprob_token_ids [d, a]} under {@code bad_words ["d"]} over uniform rows: every emitted token's capture
     * is the token, then {@code d} and {@code a} in request order (vLLM's {@code gather_specific_token_logprobs}), in
     * every lossless mode. The banned {@code d} scores {@code -inf} and {@code a} one third, the row the chain left.
     */
    @Test
    public void namedLogprobTokenIdsCoverEveryEmittedTokenInEveryMode() {
        double lnThird = Math.log(1.0 / 3.0);
        for (VerifyMode mode : new VerifyMode[]{VerifyMode.MATCH, VerifyMode.REJECTION, VerifyMode.BLOCK,
                VerifyMode.FLY}) {
            FakeTarget target = new FakeTarget(false).add(UNIFORM).add(UNIFORM, UNIFORM, UNIFORM)
                    .add(UNIFORM, UNIFORM, UNIFORM);
            GenerationRequest request = GenerationRequest.builder().maxNewTokens(3).sampler(stochastic(5))
                    .badWords(Collections.singletonList("d")).logprobTokenIds(D, A)
                    .logprobsMode(LogprobsMode.PROCESSED_LOGPROBS).build();
            GenerationResult r = decoder(target, new PlainDrafter(new int[]{A, B}, new int[]{A, B}), tokenizer(),
                    mode).generate(new int[]{C}, request, (TokenCallback) null);
            assertEquals(mode.toString(), 3, r.logprobs().size());
            for (int i = 0; i < 3; i++) {
                TokenScore[] scores = r.logprobs().get(i);
                assertEquals(mode + ": the token, then the named ids", 3, scores.length);
                assertEquals(mode.toString(), r.tokens()[i], scores[0].tokenId());
                assertEquals(mode.toString(), lnThird, scores[0].logprob(), 1e-5);
                assertEquals(mode.toString(), D, scores[1].tokenId());
                assertEquals(mode + ": the banned id", Float.NEGATIVE_INFINITY, scores[1].logprob(), 0f);
                assertEquals(mode.toString(), A, scores[2].tokenId());
                assertEquals(mode.toString(), lnThird, scores[2].logprob(), 1e-5);
            }
        }
    }

    /**
     * Synthetic acceptance at rate 1 with the proposal {@code [d, d]} under {@code bad_words ["d"]}: the proposal
     * passes the grammar cut (there is no grammar), but the chain bans {@code d} at the first verify row, so the
     * round rejects it and emits the chain's own draw. No emitted token is {@code d}, and every token's logprobs are
     * its banned row: the three other tokens at one third each.
     */
    @Test
    public void syntheticAcceptanceNeverConfirmsATokenTheChainExcludes() {
        String previous = System.getProperty(SyntheticVerifier.PROP_ENABLED);
        System.setProperty(SyntheticVerifier.PROP_ENABLED, "true");
        try {
            for (long seed = 0; seed < 8; seed++) {
                FakeTarget target = uniformTarget();
                SpeculativeDecoder dec = SpeculativeDecoder.builder(target, new PlainDrafter(new int[]{D, D}),
                        tokenizer()).draftMax(2).verifyMode(VerifyMode.SYNTHETIC)
                        .syntheticAcceptanceRates(1.0, 1.0).build();
                GenerationRequest request = GenerationRequest.builder().maxNewTokens(3).sampler(stochastic(seed))
                        .badWords(Collections.singletonList("d")).logprobs(4)
                        .logprobsMode(LogprobsMode.PROCESSED_LOGPROBS).build();
                GenerationResult r = dec.generate(new int[]{D}, request, (TokenCallback) null);
                String what = "seed " + seed + " " + Arrays.toString(r.tokens());
                assertEquals(what, 1, target.verifyCalls.size());
                assertEquals(what, 3, r.tokens().length);
                assertEquals(what, 3, r.logprobs().size());
                for (int i = 0; i < 3; i++) {
                    assertTrue(what, r.tokens()[i] != D);
                    TokenScore[] scores = r.logprobs().get(i);
                    assertEquals(what, 3, scores.length);
                    assertEquals(what, r.tokens()[i], scores[0].tokenId());
                    for (TokenScore score : scores) {
                        assertTrue(what, score.tokenId() != D);
                        assertEquals(what, Math.log(1.0 / 3.0), score.logprob(), 1e-5);
                    }
                }
            }
        } finally {
            if (previous == null) {
                System.clearProperty(SyntheticVerifier.PROP_ENABLED);
            } else {
                System.setProperty(SyntheticVerifier.PROP_ENABLED, previous);
            }
        }
    }

    /**
     * adaptive-p folds only its own draw into its average: every verifier mode refuses it by name, MATCH runs it.
     * Mirostat adapts on the accepted token, so every mode runs it, and so does a mirostat chain whose order names
     * adaptive_p, which builds no adaptive-p stage.
     */
    @Test
    public void adaptivePIsRefusedOutsideMatchAndMirostatRunsInEveryMode() {
        SamplerConfig adaptive = SamplerConfig.builder().temperature(1f).samplers("temperature;adaptive_p")
                .seed(1).build();
        SamplerConfig mirostat = SamplerConfig.builder().temperature(1f).mirostat(2).seed(1).build();
        SamplerConfig mirostatOverAdaptive = SamplerConfig.builder().temperature(1f).mirostat(2)
                .samplers("temperature;adaptive_p").seed(1).build();
        for (VerifyMode mode : new VerifyMode[]{VerifyMode.REJECTION, VerifyMode.BLOCK, VerifyMode.FLY}) {
            FakeTarget refused = uniformTarget();
            GenerationRequest request = GenerationRequest.builder().maxNewTokens(2).sampler(adaptive).build();
            IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                    () -> decoder(refused, new PlainDrafter(new int[]{B, D}), tokenizer(), mode)
                            .generate(new int[]{D}, request, (TokenCallback) null));
            assertTrue(e.getMessage(), e.getMessage().contains("adaptive_p"));
            assertTrue("refused before any forward call", refused.prefills.isEmpty());
            for (SamplerConfig config : new SamplerConfig[]{mirostat, mirostatOverAdaptive}) {
                GenerationResult r = decoder(uniformTarget(), new PlainDrafter(new int[]{B, D}), tokenizer(), mode)
                        .generate(new int[]{D}, GenerationRequest.builder().maxNewTokens(2).sampler(config).build(),
                                (TokenCallback) null);
                assertEquals(mode + " runs mirostat", 2, r.tokens().length);
            }
        }
        for (SamplerConfig config : new SamplerConfig[]{mirostat, adaptive}) {
            FakeTarget target = new FakeTarget(false).add(UNIFORM);
            GenerationResult r = decoder(target, new PlainDrafter(), tokenizer(), VerifyMode.MATCH).generate(
                    new int[]{D}, GenerationRequest.builder().maxNewTokens(1).sampler(config).build(),
                    (TokenCallback) null);
            assertEquals(1, r.tokens().length);
        }
    }

    /** A target whose prompt fills the context to {@code N_CTX - 2}, with greedy rows that always want {@code b}. */
    private static FakeTarget fullingTarget(boolean shiftable, int decodes) {
        float[] wantB = {0f, 3f, 0f, 0f};
        FakeTarget target = new FakeTarget(false).add(wantB);
        for (int i = 0; i < decodes; i++) {
            target.add(wantB);
        }
        target.shiftable = shiftable;
        return target.otherwise(wantB);
    }

    private static int[] longPrompt() {
        int[] prompt = new int[N_CTX - 2];
        Arrays.fill(prompt, A);
        return prompt;
    }

    /**
     * With context shift on, a full context discards half of the stored positions (no BOS to keep) and generation
     * continues; a drafter that reads only the target's tokens resets and re-aligns at its next draft, so the
     * shifted target rows stay as the shift moved them. Off, or on a memory that cannot shift, the generation ends
     * with {@code CONTEXT_FULL}.
     */
    @Test
    public void aFullContextShiftsWhenContextShiftIsOn() {
        GenerationRequest request = GenerationRequest.builder().maxNewTokens(5)
                .sampler(SamplerConfig.greedyDecoding()).build();
        FakeTarget target = fullingTarget(true, 4);
        PlainDrafter drafter = new PlainDrafter();
        SpeculativeDecoder shifting = SpeculativeDecoder.builder(target, drafter, tokenizer()).draftMax(2)
                .contextShift(true).build();
        GenerationResult r = shifting.generate(longPrompt(), request, (TokenCallback) null);
        assertEquals(StopReason.MAX_TOKENS, r.stopReason());
        assertArrayEquals(new int[]{B, B, B, B, B}, r.tokens());
        assertEquals(1, target.shifts.size());
        assertArrayEquals(new int[]{0, N_CTX / 2}, target.shifts.get(0));
        assertEquals("one reset per generation and one after the shift", 2, drafter.resets);
        assertEquals("a token-stream drafter re-evaluates nothing", 1, target.prefills.size());
        assertEquals(N_CTX / 2 + 2, target.memory.sequence.nPast());

        for (boolean shiftOn : new boolean[]{false, true}) {
            FakeTarget full = fullingTarget(false, 4);
            GenerationResult stopped = SpeculativeDecoder.builder(full, new PlainDrafter(), tokenizer())
                    .draftMax(2).contextShift(shiftOn).build().generate(longPrompt(), request, (TokenCallback) null);
            assertEquals("context shift " + shiftOn, StopReason.CONTEXT_FULL, stopped.stopReason());
            assertEquals(3, stopped.tokens().length);
        }
    }

    /**
     * A drafter that reads the target's hidden rows rebuilds them after a shift: it starts the re-alignment
     * through {@code beginGeneration} (which resets it here) and the target re-evaluates the post-shift context
     * from position 0, so the drafter's rows match the moved tokens.
     */
    @Test
    public void aHiddenDrafterHasTheTargetReplayTheShiftedContext() {
        GenerationRequest request = GenerationRequest.builder().maxNewTokens(5)
                .sampler(SamplerConfig.greedyDecoding()).build();
        FakeTarget target = fullingTarget(true, 4);
        PlainDrafter drafter = new PlainDrafter();
        drafter.readsHidden = true;
        GenerationResult r = SpeculativeDecoder.builder(target, drafter, tokenizer()).draftMax(2)
                .contextShift(true).build().generate(longPrompt(), request, (TokenCallback) null);

        assertEquals(StopReason.MAX_TOKENS, r.stopReason());
        assertArrayEquals(new int[]{B, B, B, B, B}, r.tokens());
        assertEquals("one reset per generation and one after the shift", 2, drafter.resets);
        assertEquals("the initial prefill and the post-shift re-evaluation", 2, target.prefills.size());
        assertEquals("the re-evaluation starts at the shifted context's first position", 0,
                (int) target.prefillPositions.get(1));
        assertEquals("the whole post-shift context is re-evaluated", N_CTX / 2, target.prefills.get(1).length);
    }
}

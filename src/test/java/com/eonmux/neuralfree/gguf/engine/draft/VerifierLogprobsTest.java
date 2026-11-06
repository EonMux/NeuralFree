package com.eonmux.neuralfree.gguf.engine.draft;

import com.eonmux.neuralfree.gguf.TokenCallback;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.LogprobsMode;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.engine.SpeculativeDecoder;
import com.eonmux.neuralfree.gguf.engine.TokenScore;
import com.eonmux.neuralfree.gguf.engine.draft.SpeculativeDecoderFixtures.FakeTarget;
import com.eonmux.neuralfree.gguf.engine.draft.SpeculativeDecoderFixtures.PlainDrafter;
import org.junit.Test;

import static com.eonmux.neuralfree.gguf.engine.draft.SpeculativeDecoderFixtures.logits;
import static com.eonmux.neuralfree.gguf.engine.draft.SpeculativeDecoderFixtures.tokenizer;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Scores of the verifier modes' emissions: every token a {@link Verifier} confirms or emits carries the capture of
 * the target's row at its position — the raw row under a raw mode, the distribution the verifier decided over under
 * a processed mode, the chain's distribution of that row — as vLLM scores accepted and bonus tokens from the target
 * logits.
 *
 * <p>The trace: the prefill row samples {@code 1}, the drafter proposes {@code [2, 3]}, verify row 0 confirms
 * {@code 2} and verify row 1 rejects {@code 3} for {@code 0}; a greedy target makes every lossless mode decide this
 * way. Each row holds the logs of probabilities summing to 1, so its raw log-softmax gives them back.
 */
public class VerifierLogprobsTest {

    private static final double LN2 = Math.log(2);

    private static GenerationResult run(VerifyMode mode, GenerationRequest request) {
        FakeTarget target = new FakeTarget(false)
                .add(logits(0.1, 0.6, 0.2, 0.1))
                .add(logits(0.1, 0.1, 0.7, 0.1), logits(0.5, 0.2, 0.2, 0.1), logits(0.25, 0.25, 0.25, 0.25));
        return SpeculativeDecoder.builder(target, new PlainDrafter(new int[]{2, 3}), tokenizer()).draftMax(2)
                .verifyMode(mode).build().generate(new int[]{0}, request, (TokenCallback) null);
    }

    private static GenerationRequest.Builder greedy() {
        return GenerationRequest.builder().maxNewTokens(3).sampler(SamplerConfig.greedyDecoding());
    }

    @Test
    public void everyLosslessModeScoresEachEmissionOverItsVerifyRow() {
        for (VerifyMode mode : new VerifyMode[]{VerifyMode.MATCH, VerifyMode.REJECTION, VerifyMode.BLOCK}) {
            GenerationResult r = run(mode, greedy().logprobs(2).build());
            assertArrayEquals(mode.toString(), new int[]{1, 2, 0}, r.tokens());
            assertEquals(mode + ": one capture per emitted token", 3, r.logprobs().size());
            double[] expected = {Math.log(0.6), Math.log(0.7), Math.log(0.5)};
            for (int i = 0; i < 3; i++) {
                TokenScore[] s = r.logprobs().get(i);
                assertEquals(mode + " step " + i, r.tokens()[i], s[0].tokenId());
                assertEquals(mode + " step " + i, expected[i], s[0].logprob(), 1e-6);
                assertEquals(mode + " step " + i, 1, s[0].rank());
                assertEquals(2, s.length);
            }
        }
    }

    @Test
    public void processedModesScoreTheDistributionTheVerifierDecidedOver() {
        // a +ln 2 bias on token 3 changes no argmax but renormalizes every processed row over 1.1
        GenerationRequest processed = greedy().logprobs(1).logitBias(3, (float) LN2)
                .logprobsMode(LogprobsMode.PROCESSED_LOGPROBS).build();
        GenerationRequest raw = processed.toBuilder().logprobsMode(LogprobsMode.RAW_LOGPROBS).build();
        GenerationResult p = run(VerifyMode.REJECTION, processed);
        GenerationResult q = run(VerifyMode.REJECTION, raw);
        assertArrayEquals(new int[]{1, 2, 0}, p.tokens());
        double[] probs = {0.6, 0.7, 0.5};
        for (int i = 0; i < 3; i++) {
            assertEquals("processed step " + i, Math.log(probs[i] / 1.1), p.logprobs().get(i)[0].logprob(), 1e-6);
            assertEquals("raw step " + i, Math.log(probs[i]), q.logprobs().get(i)[0].logprob(), 1e-6);
        }
    }

    @Test
    public void explicitTokenIdsAreScoredOnEveryEmission() {
        GenerationResult r = run(VerifyMode.REJECTION, greedy().logprobTokenIds(3, 0).build());
        assertEquals(3, r.logprobs().size());
        TokenScore[] confirmed = r.logprobs().get(1);
        assertEquals(3, confirmed.length);
        assertEquals(2, confirmed[0].tokenId());
        assertEquals(3, confirmed[1].tokenId());
        assertEquals(Math.log(0.1), confirmed[1].logprob(), 1e-6);
        assertEquals(0, confirmed[2].tokenId());
        assertEquals(Math.log(0.1), confirmed[2].logprob(), 1e-6);
    }

    /**
     * Synthetic acceptance confirms both draft tokens ({@code [1, 2, 3]}); every confirmed row still runs through the
     * chain, so the processed modes score it over the chain's distribution as the lossless modes do.
     */
    @Test
    public void syntheticVerificationScoresTheProcessedModes() {
        String previous = System.getProperty(SyntheticVerifier.PROP_ENABLED);
        System.setProperty(SyntheticVerifier.PROP_ENABLED, "true");
        try {
            GenerationRequest processed = greedy().logprobs(1).logitBias(3, (float) LN2)
                    .logprobsMode(LogprobsMode.PROCESSED_LOGPROBS).build();
            GenerationResult p = synthetic(processed);
            GenerationResult q = synthetic(processed.toBuilder().logprobsMode(LogprobsMode.RAW_LOGPROBS).build());
            assertArrayEquals(new int[]{1, 2, 3}, p.tokens());
            assertArrayEquals(p.tokens(), q.tokens());
            double[] probs = {0.6, 0.7, 0.1};
            double[] biased = {0.6, 0.7, 0.2};
            for (int i = 0; i < 3; i++) {
                assertEquals("processed step " + i, Math.log(biased[i] / 1.1), p.logprobs().get(i)[0].logprob(),
                        1e-6);
                assertEquals("raw step " + i, Math.log(probs[i]), q.logprobs().get(i)[0].logprob(), 1e-6);
            }
        } finally {
            if (previous == null) {
                System.clearProperty(SyntheticVerifier.PROP_ENABLED);
            } else {
                System.setProperty(SyntheticVerifier.PROP_ENABLED, previous);
            }
        }
    }

    private static GenerationResult synthetic(GenerationRequest request) {
        FakeTarget target = new FakeTarget(false)
                .add(logits(0.1, 0.6, 0.2, 0.1))
                .add(logits(0.1, 0.1, 0.7, 0.1), logits(0.5, 0.2, 0.2, 0.1), logits(0.25, 0.25, 0.25, 0.25));
        return SpeculativeDecoder.builder(target, new PlainDrafter(new int[]{2, 3}), tokenizer()).draftMax(2)
                .verifyMode(VerifyMode.SYNTHETIC).syntheticAcceptanceRates(1.0, 1.0).build()
                .generate(new int[]{0}, request, (TokenCallback) null);
    }

    @Test
    public void promptRowScoringRefusesOnTheSingleSequenceDecoder() {
        SpeculativeDecoder dec = SpeculativeDecoder.builder(new FakeTarget(false), new PlainDrafter(new int[]{2}),
                tokenizer()).draftMax(2).build();
        UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class, () -> dec.generate(
                new int[]{0, 1}, greedy().promptLogprobs(1).build(), (TokenCallback) null));
        assertTrue(e.getMessage().contains("prompt_logprobs"));
    }
}

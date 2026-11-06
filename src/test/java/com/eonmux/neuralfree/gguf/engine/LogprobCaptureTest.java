package com.eonmux.neuralfree.gguf.engine;

import org.junit.Test;

import java.util.Collections;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The per-token capture of every {@link LogprobsMode} (vLLM's {@code --logprobs-mode}) over one known row: the raw
 * modes score the model's row before any sampling stage, the processed modes the chain's final candidates (a
 * greedy chain's biased logits; a sampling chain's logits after temperature and top-k, removed tokens at -inf),
 * and explicit token ids ({@code logprob_token_ids}) replace the best alternatives.
 */
public class LogprobCaptureTest {

    /** ln of probabilities 0.5, 0.25, 0.125, 0.125: the raw log-softmax gives them back exactly. */
    private static float[] row() {
        return new float[]{(float) Math.log(0.5), (float) Math.log(0.25), (float) Math.log(0.125),
                (float) Math.log(0.125)};
    }

    private static TokenScore[] capture(GenerationRequest request, SamplerConfig config, float bias3) {
        Sampler sampler = Sampler.create(config, new int[0],
                bias3 == 0f ? Collections.<Integer, Float>emptyMap() : Collections.singletonMap(3, bias3));
        float[] row = row();
        Sampler.Candidates cur = sampler.apply(row);
        return LogprobCapture.of(request).capture(row, cur, cur.selectedId());
    }

    private static GenerationRequest request(LogprobsMode mode, int n) {
        return GenerationRequest.builder().logprobs(n).logprobsMode(mode).build();
    }

    @Test
    public void rawLogprobsAreTheModelsDistributionWhateverTheChainDoes() {
        // a +ln(8) bias on token 3 makes it the greedy pick; the raw capture still scores the unbiased row
        TokenScore[] s = capture(request(LogprobsMode.RAW_LOGPROBS, 3), SamplerConfig.greedyDecoding(),
                (float) Math.log(8));
        assertEquals("the chain picked the biased token", 3, s[0].tokenId());
        assertEquals(Math.log(0.125), s[0].logprob(), 1e-6);
        assertEquals("tokens 0 and 1 score higher in the raw row", 3, s[0].rank());
        assertEquals(3, s.length);
        assertEquals(0, s[1].tokenId());
        assertEquals(Math.log(0.5), s[1].logprob(), 1e-6);
        assertEquals(1, s[1].rank());
        assertEquals(1, s[2].tokenId());
        assertEquals(Math.log(0.25), s[2].logprob(), 1e-6);
        assertEquals(2, s[2].rank());
    }

    @Test
    public void rawLogitsReportTheRowItself() {
        TokenScore[] s = capture(request(LogprobsMode.RAW_LOGITS, 2), SamplerConfig.greedyDecoding(), 0f);
        assertEquals(0, s[0].tokenId());
        assertEquals((float) Math.log(0.5), s[0].logprob(), 0f);
        assertEquals(1, s[1].tokenId());
        assertEquals((float) Math.log(0.25), s[1].logprob(), 0f);
    }

    @Test
    public void processedLogprobsFollowTheGreedyChainsBiasedLogits() {
        // biased probabilities 0.5, 0.25, 0.125, 1.0 renormalized over 1.875
        TokenScore[] s = capture(request(LogprobsMode.PROCESSED_LOGPROBS, 2), SamplerConfig.greedyDecoding(),
                (float) Math.log(8));
        assertEquals(3, s[0].tokenId());
        assertEquals(Math.log(1.0 / 1.875), s[0].logprob(), 1e-6);
        assertEquals(1, s[0].rank());
        assertEquals(0, s[1].tokenId());
        assertEquals(Math.log(0.5 / 1.875), s[1].logprob(), 1e-6);
    }

    @Test
    public void processedModesSeeTemperatureAndTruncation() {
        SamplerConfig topTwoHot = SamplerConfig.builder().temperature(0.5f).topK(2).topP(1f).minP(0f).seed(7)
                .build();
        TokenScore[] lp = capture(request(LogprobsMode.PROCESSED_LOGPROBS, 4), topTwoHot, 0f);
        // temperature 0.5 squares the kept probabilities 0.5 and 0.25: 0.25 / 0.3125 and 0.0625 / 0.3125
        assertEquals("top-k 2 leaves two candidates", 2, lp.length);
        double first = lp[0].tokenId() == 0 ? lp[0].logprob() : lp[1].logprob();
        assertEquals(Math.log(0.25 / 0.3125), first, 1e-5);
        TokenScore[] logits = capture(request(LogprobsMode.PROCESSED_LOGITS, 4), topTwoHot, 0f);
        for (TokenScore t : logits) {
            assertEquals("the logit after temperature", 2 * row()[t.tokenId()], t.logprob(), 1e-5);
        }
        TokenScore[] raw = capture(request(LogprobsMode.RAW_LOGPROBS, 4), topTwoHot, 0f);
        assertEquals("the raw mode keeps every token", 4, raw.length);
    }

    @Test
    public void explicitTokenIdsReplaceTheBestAlternativesInTheirOrder() {
        GenerationRequest r = GenerationRequest.builder().logprobs(3).logprobTokenIds(3, 2, 3).build();
        TokenScore[] s = capture(r, SamplerConfig.greedyDecoding(), 0f);
        assertEquals("the sampled token, then each distinct requested id", 3, s.length);
        assertEquals(0, s[0].tokenId());
        assertEquals(1, s[0].rank());
        assertEquals(3, s[1].tokenId());
        assertEquals(Math.log(0.125), s[1].logprob(), 1e-6);
        assertEquals(2, s[2].tokenId());
        TokenScore[] processed = capture(GenerationRequest.builder().logprobTokenIds(1)
                        .logprobsMode(LogprobsMode.PROCESSED_LOGPROBS).build(),
                SamplerConfig.builder().temperature(1f).topK(1).seed(3).build(), 0f);
        assertEquals("a token the chain removed scores -inf", Float.NEGATIVE_INFINITY, processed[1].logprob(), 0f);
    }

    @Test
    public void anInactiveCaptureReportsNothing() {
        assertEquals(0, capture(GenerationRequest.defaults(), SamplerConfig.greedyDecoding(), 0f).length);
        assertTrue(LogprobCapture.of(GenerationRequest.builder().logprobTokenIds(1).build()).active());
    }

    @Test
    public void theModeNamesParseAndUnknownNamesRefuse() {
        assertEquals(LogprobsMode.RAW_LOGPROBS, LogprobsMode.parse("raw_logprobs"));
        assertEquals(LogprobsMode.PROCESSED_LOGITS, LogprobsMode.parse(" Processed_Logits "));
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> LogprobsMode.parse("logits"));
        assertTrue(e.getMessage().contains("logprobs_mode"));
        assertEquals("the default mode is vLLM's", LogprobsMode.RAW_LOGPROBS,
                GenerationRequest.defaults().logprobsMode());
    }

    @Test
    public void theRequestValidatesTheLogprobFields() {
        assertTrue(assertThrows(IllegalArgumentException.class,
                () -> GenerationRequest.builder().logprobTokenIds(-2)).getMessage().contains("logprob_token_ids"));
        int[] many = new int[GenerationRequest.MAX_LOGPROB_TOKEN_IDS + 1];
        for (int i = 0; i < many.length; i++) {
            many[i] = i;
        }
        assertThrows(IllegalArgumentException.class, () -> GenerationRequest.builder().logprobTokenIds(many));
        assertTrue(assertThrows(IllegalArgumentException.class,
                () -> GenerationRequest.builder().promptLogprobStart(2).build()).getMessage()
                .contains("prompt_logprob_start"));
        assertThrows(IllegalArgumentException.class,
                () -> GenerationRequest.builder().promptLogprobTokenIds(new int[][]{{}}));
        assertThrows(IllegalArgumentException.class,
                () -> GenerationRequest.builder().promptLogprobTokenIds(new int[][]{{-3}}));
        GenerationRequest padded = GenerationRequest.builder()
                .promptLogprobTokenIds(new int[][]{{1, 2}, {3}}).promptLogprobStart(1).build();
        assertEquals("short rows pad with -1", GenerationRequest.PROMPT_TOKEN_ID_PAD,
                padded.promptLogprobTokenIds()[1][1]);
        GenerationRequest copy = padded.toBuilder().build();
        assertEquals(1, copy.promptLogprobStart());
        assertEquals(2, copy.promptLogprobTokenIds().length);
        assertTrue(copy.readsPromptRows());
        assertTrue(assertThrows(UnsupportedOperationException.class,
                () -> copy.requireNoPromptRows("here")).getMessage().contains("prompt_logprob_token_ids"));
    }
}

package com.eonmux.neuralfree.gguf.engine.watermark;

import org.junit.Test;

import java.util.SplittableRandom;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

/**
 * The draft leg of speculative watermarking: the key-A draw and its context, the temperature-scaled proposal it
 * reports, the ordinary draw of a repeated context, and the target-only configuration without a draft key.
 */
public class DraftWatermarkerTest {

    private static final float[] LOGITS = {1.5f, 0.25f, 2.0f, -0.5f, 0.75f};
    private static final long KEY = 1234L;

    private static WatermarkConfig config(WatermarkContextScope scope) {
        return WatermarkConfig.builder(KEY)
                .algorithm(WatermarkAlgorithm.DUAL_KEY_GUMBEL)
                .contextWidth(2)
                .deduplicateContexts(scope)
                .build();
    }

    private static float[] probabilities(float[] logits, float temperature) {
        float max = Float.NEGATIVE_INFINITY;
        for (float v : logits) {
            max = Math.max(max, v / temperature);
        }
        double sum = 0.0;
        float[] q = new float[logits.length];
        for (int i = 0; i < logits.length; i++) {
            q[i] = (float) Math.exp(logits[i] / temperature - max);
            sum += q[i];
        }
        for (int i = 0; i < q.length; i++) {
            q[i] = (float) (q[i] / sum);
        }
        return q;
    }

    private static int categorical(float[] q, double u) {
        double total = 0.0;
        for (float v : q) {
            total += v;
        }
        double run = 0.0;
        int last = -1;
        for (int i = 0; i < q.length; i++) {
            if (q[i] > 0f) {
                last = i;
                run += q[i];
                if (run > u * total) {
                    return i;
                }
            }
        }
        return last;
    }

    @Test
    public void theDraftDrawUsesKeyAOverTheTrailingContext() {
        WatermarkConfig config = config(WatermarkContextScope.NONE);
        DraftWatermarker watermark = DraftWatermarker.of(config);
        assertNotNull(watermark);
        watermark.reset(0);
        watermark.sync(new int[]{10, 11, 12}, 13);
        watermark.beginRound();

        GumbelWatermarker keyA = new GumbelWatermarker(WatermarkConfig.keyA(KEY), 2);
        int expected = keyA.sample(LOGITS, new int[]{12, 13}, () -> 0.5);
        DraftWatermarker.Draw first = watermark.draw(LOGITS, LOGITS.length, 1f, () -> 0.5);
        assertEquals(expected, first.token());

        // the earlier draft is part of the next draw's context
        int secondExpected = keyA.sample(LOGITS, new int[]{13, first.token()}, () -> 0.5);
        DraftWatermarker.Draw second = watermark.draw(LOGITS, LOGITS.length, 1f, () -> 0.5);
        assertEquals(secondExpected, second.token());
    }

    @Test
    public void theProposalIsTheTemperatureScaledDistributionItWasDrawnFrom() {
        DraftWatermarker watermark = DraftWatermarker.of(config(WatermarkContextScope.NONE));
        watermark.reset(0);
        watermark.sync(new int[]{1, 2}, 3);
        watermark.beginRound();
        DraftWatermarker.Draw draw = watermark.draw(LOGITS, LOGITS.length, 0.5f, () -> 0.5);
        float[] expected = probabilities(LOGITS, 0.5f);
        float[] actual = draw.proposal();
        assertEquals(expected.length, actual.length);
        for (int i = 0; i < actual.length; i++) {
            assertEquals("proposal " + i, expected[i], actual[i], 1e-6f);
        }
        assertTrue("the proposal is the row's distribution, not one-hot", actual[draw.token()] > 0f);
        assertTrue("a temperature-scaled row has a top probability",
                draw.topProbability() > 0f && draw.topProbability() <= 1f);
    }

    @Test
    public void aRepeatedContextUsesTheOrdinaryDraw() {
        // committed [7, 8, 7], pending 8: the context {7, 8} already occurred in the generated history
        WatermarkConfig config = config(WatermarkContextScope.ALL);
        DraftWatermarker watermark = DraftWatermarker.of(config);
        watermark.reset(0);
        watermark.sync(new int[]{7, 8, 7}, 8);
        watermark.beginRound();
        float temperature = 1f;
        float[] q = probabilities(LOGITS, temperature);
        double u = 0.42;
        DraftWatermarker.Draw draw = watermark.draw(LOGITS, LOGITS.length, temperature, () -> u);
        assertEquals(categorical(q, u), draw.token());
    }

    @Test
    public void aContextRepeatedWithinTheRoundUsesTheOrdinaryDraw() {
        // width 1 with the committed tail {1}: the first draw takes the row's only candidate, token 1, so the
        // second draw's context {1} repeats the round's first context.
        WatermarkConfig config = WatermarkConfig.builder(KEY)
                .algorithm(WatermarkAlgorithm.DUAL_KEY_GUMBEL)
                .contextWidth(1)
                .deduplicateContexts(WatermarkContextScope.SINGLE_TURN)
                .build();
        DraftWatermarker watermark = DraftWatermarker.of(config);
        watermark.reset(0);
        watermark.sync(new int[]{5, 1}, 9);
        watermark.beginRound();
        float[] single = {Float.NEGATIVE_INFINITY, 0f, Float.NEGATIVE_INFINITY};
        assertEquals(1, watermark.draw(single, single.length, 1f, () -> 0.5).token());

        float[] row = {(float) Math.log(0.6), (float) Math.log(0.4)};
        double u = 0.9;
        assertEquals("the repeated context falls back to the ordinary draw",
                categorical(probabilities(row, 1f), u),
                watermark.draw(row, row.length, 1f, () -> u).token());
    }

    @Test
    public void aSingleKeyConfigurationWithoutTargetOnlyHasNoDraftRole() {
        WatermarkConfig single = WatermarkConfig.builder(KEY).build();
        try {
            DraftWatermarker.of(single);
            throw new AssertionError("the single-key configuration has no draft role");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains("allow_target_only_watermarking"));
        }
        assertNull(DraftWatermarker.of(WatermarkConfig.builder(KEY).allowTargetOnlyWatermarking(true).build()));
        assertNull(DraftWatermarker.of(null));
    }

    @Test
    public void aPromptContextCountsOnlyWhenTheScopeSearchesThePrompt() {
        // prompt [7, 8, 9], then generated [7] and the pending 8: the draw's context {7, 8} occurs in the prompt
        float temperature = 1f;
        int[] committed = {7, 8, 9, 7};
        int pending = 8;
        double u = 0.9;
        GumbelWatermarker keyA = new GumbelWatermarker(WatermarkConfig.keyA(KEY), 2);

        DraftWatermarker singleTurn = DraftWatermarker.of(config(WatermarkContextScope.SINGLE_TURN));
        singleTurn.reset(3);
        singleTurn.sync(committed, pending);
        singleTurn.beginRound();
        assertEquals("a single-turn scope leaves the prompt out of its search, so the keyed draw runs",
                keyA.sample(LOGITS, new int[]{7, 8}, () -> 0.5),
                singleTurn.draw(LOGITS, LOGITS.length, temperature, () -> u).token());

        DraftWatermarker all = DraftWatermarker.of(config(WatermarkContextScope.ALL));
        all.reset(3);
        all.sync(committed, pending);
        all.beginRound();
        assertEquals("an all scope searches the prompt, so the context repeats",
                categorical(probabilities(LOGITS, temperature), u),
                all.draw(LOGITS, LOGITS.length, temperature, () -> u).token());
    }

    @Test
    public void anAllScopeDrawsTheFirstContextAfterThePromptUnwatermarked() {
        // the prompt alone is committed: fewer than context_width tokens were generated, so the context is partial
        DraftWatermarker watermark = DraftWatermarker.of(config(WatermarkContextScope.ALL));
        watermark.reset(2);
        watermark.sync(new int[]{3, 4});
        watermark.beginRound();
        double u = 0.42;
        assertEquals("a partial context is drawn without the watermark",
                categorical(probabilities(LOGITS, 1f), u),
                watermark.draw(LOGITS, LOGITS.length, 1f, () -> u).token());
    }

    @Test
    public void syncExtendsTheHistoryIncrementallyAndRebuildsAfterARewrite() {
        WatermarkConfig config = config(WatermarkContextScope.NONE);
        DraftWatermarker watermark = DraftWatermarker.of(config);
        watermark.reset(2);
        watermark.sync(new int[]{1, 2}, 3);
        watermark.beginRound();
        GumbelWatermarker keyA = new GumbelWatermarker(WatermarkConfig.keyA(KEY), 2);
        assertEquals(keyA.sample(LOGITS, new int[]{2, 3}, () -> 0.5),
                watermark.draw(LOGITS, LOGITS.length, 1f, () -> 0.5).token());

        // the target commits the draw and samples the next token: the incremental sync keeps the history
        watermark.sync(new int[]{1, 2, 3}, 4);
        watermark.beginRound();
        assertEquals(keyA.sample(LOGITS, new int[]{3, 4}, () -> 0.5),
                watermark.draw(LOGITS, LOGITS.length, 1f, () -> 0.5).token());

        // a rewrite (context shift) restarts the history from the stored record
        watermark.sync(new int[]{9, 9}, 9);
        watermark.beginRound();
        assertEquals(keyA.sample(LOGITS, new int[]{9, 9}, () -> 0.5),
                watermark.draw(LOGITS, LOGITS.length, 1f, () -> 0.5).token());
    }
}

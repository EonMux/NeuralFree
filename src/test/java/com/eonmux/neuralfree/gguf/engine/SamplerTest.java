package com.eonmux.neuralfree.gguf.engine;

import org.junit.Test;

import java.util.Arrays;
import java.util.SplittableRandom;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Traces of {@link Sampler} against llama.cpp's sampler chain (llama.cpp src/llama-sampler.cpp) on the 5-logit row
 * {@code [2, 1, 0.5, 3, -1]} (token ids 0..4). Probabilities are hand-computed softmaxes.
 */
public class SamplerTest {

    private static final float[] LOGITS = {2.0f, 1.0f, 0.5f, 3.0f, -1.0f};
    private static final float EPS = 1e-4f;
    private static final long SEED = 42L;

    private static SamplerConfig.Builder off() {
        return SamplerConfig.builder().topK(0).topP(1f).minP(0f).typicalP(1f).temperature(1f).seed(SEED);
    }

    private static int[] ids(Sampler.Candidates c) {
        int[] out = new int[c.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = c.id(i);
        }
        return out;
    }

    /**
     * Full chain: repeat penalty 1.5 on token 3 (3 -> 2), top-k 3 sorts to [0, 3, 1] (equal logits keep vocabulary
     * order), top-p 0.9 keeps all three (cumulative 0.4223, 0.8446, 1.0), min-p 0.1 keeps logits >= 2 + ln 0.1,
     * temperature 1, then the draw over p = [0.4223, 0.4223, 0.1554] with the seeded uniform.
     */
    @Test
    public void fullChainTraceOfFiveLogits() {
        SamplerConfig cfg = off().topK(3).topP(0.9f).minP(0.1f).repeatPenalty(1.5f).penaltyLastN(64).build();
        Sampler s = Sampler.create(cfg);
        s.accept(3);
        Sampler.Candidates c = s.apply(LOGITS);

        assertArrayEquals(new int[]{0, 3, 1}, ids(c));
        assertTrue(c.sorted());
        assertEquals(2.0f, c.logit(0), 0f);
        assertEquals(2.0f, c.logit(1), 0f);
        assertEquals(1.0f, c.logit(2), 0f);
        assertEquals(0.42232f, c.prob(0), EPS);
        assertEquals(0.42232f, c.prob(1), EPS);
        assertEquals(0.15536f, c.prob(2), EPS);

        double u = new SplittableRandom(SEED).nextDouble();
        int expected = u * 1.0 <= 0.42232 ? 0 : u <= 0.84464 ? 3 : 1;
        assertEquals(expected, c.selectedId());
    }

    @Test
    public void nucleusKeepsTheSmallestHeadReachingTopP() {
        // softmax: id3 0.6236, id0 0.2294, id1 0.0844, id2 0.0512, id4 0.0114; 0.6236 >= 0.5 keeps one token.
        Sampler s = Sampler.create(off().topP(0.5f).build());
        Sampler.Candidates c = s.apply(LOGITS);
        assertArrayEquals(new int[]{3}, ids(c));
        assertEquals(3, c.selectedId());
        assertEquals(1f, c.prob(0), 0f);

        // 0.8 needs id3 + id0 (0.8530).
        Sampler.Candidates c2 = Sampler.create(off().topP(0.8f).build()).apply(LOGITS);
        assertArrayEquals(new int[]{3, 0}, ids(c2));
    }

    @Test
    public void minPKeepsLogitsWithinLogOfMaxInVocabularyOrder() {
        // unsorted path: min logit 3 + ln 0.3 = 1.796 keeps ids 0 and 3 in their order.
        Sampler.Candidates c = Sampler.create(off().minP(0.3f).build()).apply(LOGITS);
        assertArrayEquals(new int[]{0, 3}, ids(c));
        assertFalse(c.sorted());
        assertEquals(0.26894f, c.prob(0), EPS);
        assertEquals(0.73106f, c.prob(1), EPS);
    }

    @Test
    public void typicalKeepsTheTokensClosestToTheEntropy() {
        // H = 1.0441; |-ln p - H|: id0 0.428, id3 0.572, id1 1.428, id2 1.928, id4 3.428.
        // Cumulative 0.2294, 0.8530 > 0.5 keeps [0, 3], unsorted.
        Sampler.Candidates c = Sampler.create(off().typicalP(0.5f).build()).apply(LOGITS);
        assertArrayEquals(new int[]{0, 3}, ids(c));
        assertFalse(c.sorted());
    }

    @Test
    public void topKAloneSortsAndTruncates() {
        Sampler.Candidates c = Sampler.create(off().topK(2).build()).apply(LOGITS);
        assertArrayEquals(new int[]{3, 0}, ids(c));
        assertTrue(c.sorted());
    }

    @Test
    public void temperatureScalesTheSurvivors() {
        Sampler.Candidates c = Sampler.create(off().topK(2).temperature(0.5f).build()).apply(LOGITS);
        assertEquals(6.0f, c.logit(0), 0f);
        assertEquals(4.0f, c.logit(1), 0f);
        assertEquals(0.88080f, c.prob(0), EPS);
    }

    @Test
    public void greedyTakesTheFirstMaximumAfterPenalties() {
        assertEquals(3, Sampler.create(off().greedy(true).build()).sample(LOGITS));
        assertEquals(3, Sampler.create(off().temperature(0f).build()).sample(LOGITS));

        Sampler penalized = Sampler.create(off().greedy(true).repeatPenalty(1.5f).penaltyLastN(64).build());
        penalized.accept(3);
        assertEquals(0, penalized.sample(LOGITS));
    }

    @Test
    public void argmaxOnlyNeedsNoPenaltiesBansOrTypical() {
        assertTrue(Sampler.create(off().greedy(true).build()).argmaxOnly());
        assertFalse(Sampler.create(off().build()).argmaxOnly());
        assertFalse(Sampler.create(off().greedy(true).repeatPenalty(1.1f).build()).argmaxOnly());
        assertFalse(Sampler.create(off().greedy(true).typicalP(0.9f).build()).argmaxOnly());
        assertFalse(Sampler.create(off().greedy(true).build(), new int[]{4}).argmaxOnly());
        assertTrue(Sampler.create(off().topK(1).build()).argmaxOnly());
        assertFalse(Sampler.create(off().topK(1).build(), new int[]{4}).argmaxOnly());
    }

    @Test
    public void signedZeroLogitsTieByTokenId() {
        Sampler.Candidates c = Sampler.create(off().topK(2).build()).apply(new float[]{-1f, 0.0f, -0.0f});
        assertArrayEquals(new int[]{1, 2}, ids(c));
        Sampler.Candidates d = Sampler.create(off().topK(2).build()).apply(new float[]{-1f, -0.0f, 0.0f});
        assertArrayEquals(new int[]{1, 2}, ids(d));
    }

    @Test
    public void infiniteLogitsTakeAllTheMass() {
        float inf = Float.POSITIVE_INFINITY;
        Sampler.Candidates c = Sampler.create(off().build()).apply(new float[]{1f, inf, 2f});
        assertEquals(1, c.selectedId());
        assertEquals(1f, probOf(c, 1), 0f);
        assertEquals(0f, probOf(c, 0), 0f);
    }

    @Test
    public void bannedTokensAreNeverSelected() {
        Sampler s = Sampler.create(off().greedy(true).build(), new int[]{3});
        assertEquals(0, s.sample(LOGITS));
        Sampler.Candidates c = Sampler.create(off().build(), new int[]{3, 0}).apply(LOGITS);
        assertEquals(0f, probOf(c, 3), 0f);
        assertEquals(0f, probOf(c, 0), 0f);
        assertThrows(IllegalStateException.class,
                () -> Sampler.create(off().greedy(true).build(), new int[]{0, 1, 2, 3, 4}).sample(LOGITS));
    }

    private static float probOf(Sampler.Candidates c, int id) {
        for (int i = 0; i < c.size(); i++) {
            if (c.id(i) == id) {
                return c.prob(i);
            }
        }
        return 0f;
    }

    /**
     * Window of 2: accepting 1, 1, 4 evicts the first 1, leaving counts {1: 1, 4: 1}. Repeat 2, frequency 0.5,
     * presence 0.25: id1 1/2 - 0.5 - 0.25 = -0.25; id4 -1*2 - 0.5 - 0.25 = -2.75.
     */
    @Test
    public void penaltiesUseTheLastNWindow() {
        Sampler s = Sampler.create(off().greedy(true).repeatPenalty(2f).frequencyPenalty(0.5f)
                .presencePenalty(0.25f).penaltyLastN(2).build());
        s.accept(new int[]{1, 1, 4}, 0, 3);
        Sampler.Candidates c = s.apply(LOGITS);
        assertEquals(-0.25f, c.logit(1), 0f);
        assertEquals(-2.75f, c.logit(4), 0f);
        assertEquals(2.0f, c.logit(0), 0f);
        assertEquals(3.0f, c.logit(3), 0f);
    }

    @Test
    public void wholeContextPenaltyCountsEveryAcceptedToken() {
        Sampler s = Sampler.create(off().greedy(true).frequencyPenalty(1f).penaltyLastN(-1).build());
        s.accept(new int[]{1, 1, 1}, 0, 3);
        assertEquals(-2.0f, s.apply(LOGITS).logit(1), 0f);
    }

    @Test
    public void zeroWindowDisablesPenalties() {
        Sampler s = Sampler.create(off().greedy(true).repeatPenalty(2f).penaltyLastN(0).build());
        s.accept(3);
        assertEquals(3, s.sample(LOGITS));
    }

    @Test
    public void seededSamplersRepeatAndResetRestartsTheStream() {
        SamplerConfig cfg = off().build();
        Sampler a = Sampler.create(cfg);
        Sampler b = Sampler.create(cfg);
        int[] first = new int[16];
        for (int i = 0; i < first.length; i++) {
            first[i] = a.sample(LOGITS);
            assertEquals(first[i], b.sample(LOGITS));
        }
        a.reset();
        for (int expected : first) {
            assertEquals(expected, a.sample(LOGITS));
        }
        assertEquals(SEED, a.seed());
    }

    @Test
    public void nanLogitsNeverWin() {
        float[] row = {Float.NaN, 1f, Float.NaN};
        assertEquals(1, Sampler.create(off().greedy(true).build()).sample(row));
        assertEquals(1, Sampler.create(off().build()).sample(row));
    }

    @Test
    public void inputLogitsAreNotModified() {
        float[] row = LOGITS.clone();
        Sampler s = Sampler.create(off().topK(2).temperature(0.5f).repeatPenalty(1.5f).penaltyLastN(8).build());
        s.accept(3);
        s.sample(row);
        assertArrayEquals(LOGITS, row, 0f);
    }

    @Test
    public void diffusionChainDrawsEvenWithoutTemperature() {
        Sampler s = Sampler.diffusionChain(0, 1f, 0f, SEED);
        assertFalse(s.greedy());
        Sampler.Candidates c = s.apply(LOGITS);
        assertEquals(5, c.size());
        float sum = 0f;
        for (int i = 0; i < c.size(); i++) {
            sum += c.prob(i);
        }
        assertEquals(1f, sum, EPS);
        assertEquals(0.62359f, probOf(c, 3), EPS);
    }

    @Test
    public void partialSortMatchesFullSortOnALargeRow() {
        float[] row = new float[1000];
        SplittableRandom r = new SplittableRandom(3);
        for (int i = 0; i < row.length; i++) {
            row[i] = (float) (r.nextDouble() * 8.0 - 4.0);
        }
        row[17] = row[400];
        float[] sortedCopy = row.clone();
        Arrays.sort(sortedCopy);
        float threshold = sortedCopy[row.length - 40];

        Sampler.Candidates c = Sampler.create(off().topK(40).build()).apply(row);
        int[] top = ids(c);
        assertEquals(40, top.length);
        assertTrue(c.sorted());
        float prev = Float.POSITIVE_INFINITY;
        for (int id : top) {
            assertTrue(row[id] <= prev);
            assertTrue(row[id] >= threshold);
            prev = row[id];
        }
    }

    @Test
    public void invalidArgumentsAreRejected() {
        assertThrows(IllegalArgumentException.class, () -> Sampler.create(off().build(), new int[]{-1}));
        assertThrows(IllegalArgumentException.class, () -> Sampler.create(off().build()).sample(new float[0]));
        assertThrows(IllegalArgumentException.class, () -> Sampler.diffusionChain(0, 1.5f, 1f, SEED));
        assertThrows(IllegalArgumentException.class, () -> Sampler.create(off().penaltyLastN(4).repeatPenalty(1.2f)
                .build()).accept(-3));
    }
}

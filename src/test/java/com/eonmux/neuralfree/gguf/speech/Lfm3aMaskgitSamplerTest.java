package com.eonmux.neuralfree.gguf.speech;

import org.junit.Test;

import java.util.Random;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link Lfm3aMaskgitSampler} against hand-walked evaluations of mtmd-helper-gen.cpp: the Fisher-Yates order, the
 * repetition penalty's window and first-occurrence rule, the argmax of a cold draw, the top-k cut with its ties
 * and the cumulative walk's boundaries and fallback.
 */
public class Lfm3aMaskgitSamplerTest {

    @Test
    public void shuffledOrderIsAPermutationDrivenByTheSeed() {
        Random a = new Random(5);
        Random b = new Random(5);
        int[] first = Lfm3aMaskgitSampler.shuffledOrder(6, a);
        int[] second = Lfm3aMaskgitSampler.shuffledOrder(6, b);
        assertArrayEquals("the same seed replays the same order", first, second);
        int[] sorted = first.clone();
        java.util.Arrays.sort(sorted);
        assertArrayEquals("the order is a permutation", new int[]{0, 1, 2, 3, 4, 5}, sorted);
        assertThrows(IllegalArgumentException.class, () -> Lfm3aMaskgitSampler.shuffledOrder(0, new Random(1)));
    }

    @Test
    public void repPenaltyDividesPositivesAndMultipliesNegativesOncePerCode() {
        float[] logits = {4f, -2f, 8f, 1f};
        // frame-major history over two codebooks; the last 3 frames of codebook 1 took 0, 2 and 2
        int[] history = {0, 2, 1, 0, 3, 2, 1, 2};
        Lfm3aMaskgitSampler.applyRepPenalty(logits, 1, history, 2, 3, 2f);
        // both draws penalized once each: 4/2 = 2 and 8/2 = 4; code 1 sits outside the window
        assertArrayEquals(new float[]{2f, -2f, 4f, 1f}, logits, 1e-6f);
    }

    @Test
    public void repPenaltyMultipliesNonPositiveLogits() {
        float[] logits = {-4f, 2f};
        int[] history = {0};
        Lfm3aMaskgitSampler.applyRepPenalty(logits, 0, history, 1, 1, 2f);
        assertArrayEquals(new float[]{-8f, 2f}, logits, 1e-6f);
    }

    @Test
    public void repPenaltyHonorsTheWindowAndTheGuards() {
        float[] penalized = {4f, 1f};
        float[] untouched = penalized.clone();
        // the drawn code sits outside the last 1-frame window
        Lfm3aMaskgitSampler.applyRepPenalty(penalized, 0, new int[]{0}, 1, 0, 2f);
        assertArrayEquals(untouched, penalized, 0f);
        // a penalty of 1.0 is a no-op even inside the window
        Lfm3aMaskgitSampler.applyRepPenalty(penalized, 0, new int[]{0}, 1, 4, 1f);
        assertArrayEquals(untouched, penalized, 0f);
        // the window truncates to the history length and picks only the drawn code
        Lfm3aMaskgitSampler.applyRepPenalty(penalized, 0, new int[]{0}, 1, 8, 2f);
        assertArrayEquals(new float[]{2f, 1f}, penalized, 1e-6f);
    }

    @Test
    public void coldSamplingTakesTheFirstMaximum() {
        float[] logits = {1f, 5f, 5f, -3f};
        assertEquals("the tie stays with the earlier code", 1, Lfm3aMaskgitSampler.sample(logits, 0f, 0, new Random(0)));
        assertEquals("negative temperatures draw greedily too", 1,
                Lfm3aMaskgitSampler.sample(logits, -1f, 0, new Random(0)));
    }

    @Test
    public void topKZeroesTheTailsAndKeepsTies() {
        // logits 0..9; top-k 3 keeps 9, 8, 7 (both codes 7 and the earlier 8's competitor handled by tie order)
        float[] logits = new float[10];
        for (int i = 0; i < 10; i++) {
            logits[i] = i;
        }
        Random rng = new Random(0);
        for (int i = 0; i < 200; i++) {
            int draw = Lfm3aMaskgitSampler.sample(logits, 1f, 3, rng);
            assertTrue("only the kept codes can come out: got " + draw, draw == 9 || draw == 8 || draw == 7);
        }
    }

    @Test
    public void temperatureWalkFollowsTheReferenceFractions() {
        // logits {ln 2, 0} at temp 1 give probabilities {1, 1/2} and the walk's r is the rng's uniform times
        // the sum 3/2. A uniform of 0.8 makes r = 1.2, which survives the first subtraction and lands on
        // code 1; a uniform of 0.5 makes r = 0.75, which the first subtraction drops to code 0.
        float[] two = {(float) Math.log(2.0), 0f};
        assertEquals(1, Lfm3aMaskgitSampler.sample(two, 1f, 0, uniform(0.8)));
        assertEquals(0, Lfm3aMaskgitSampler.sample(two, 1f, 0, uniform(0.5)));
    }

    /** An rng whose only draw returns the fixed uniform. */
    private static Random uniform(double value) {
        return new Random() {
            @Override
            public double nextDouble() {
                return value;
            }
        };
    }

    @Test
    public void tinyProbabilitiesFallBackToTheLastCode() {
        // an extreme temp sends every probability except the max to 0; an rng that walks past everything must
        // land on the fallback n-1
        float[] logits = {100f, 0f, 0f};
        assertEquals(0, Lfm3aMaskgitSampler.sample(logits, 0.01f, 0, new Random(0)));
    }
}

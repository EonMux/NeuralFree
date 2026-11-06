package com.eonmux.neuralfree.gguf.engine.beam;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * The beam ranking score (vLLM's {@code get_beam_search_score}, utils.py:42): the cumulative log probability
 * divided by the generated-token count raised to the length penalty, a sub-one count clamped to one, and the
 * comparison that ranks the higher score first.
 */
public final class BeamSearchScoringTest {

    @Test
    public void theScoreDividesByTheTokenCountToThePenalty() {
        assertEquals(-4.0 / 2.0, BeamSearchScoring.score(2, -4.0, 1.0f), 1e-12);
        assertEquals(-4.0 / 8.0, BeamSearchScoring.score(2, -4.0, 3.0f), 1e-12);
        assertEquals(-4.0 / Math.sqrt(2), BeamSearchScoring.score(2, -4.0, 0.5f), 1e-12);
    }

    @Test
    public void aSubOneCountCountsAsOneToken() {
        // vLLM's max(num_generated_tokens, 1): an EOS on the very first step does not divide
        assertEquals(-1.5, BeamSearchScoring.score(0, -1.5, 1.0f), 1e-12);
        assertEquals(-1.5, BeamSearchScoring.score(1, -1.5, 1.0f), 1e-12);
        assertEquals(-1.5, BeamSearchScoring.score(-3, -1.5, 1.0f), 1e-12);
    }

    @Test
    public void aZeroPenaltyRanksByTheBareCumulativeLogProbability() {
        assertEquals(-2.5, BeamSearchScoring.score(9, -2.5, 0.0f), 1e-12);
        assertEquals(-2.5, BeamSearchScoring.score(1, -2.5, 0.0f), 1e-12);
    }

    @Test
    public void theComparisonRanksTheHigherScoreFirst() {
        assertTrue(BeamSearchScoring.compare(-0.25, -0.4) < 0);
        assertTrue(BeamSearchScoring.compare(-0.4, -0.25) > 0);
        assertEquals(0, BeamSearchScoring.compare(-0.3, -0.3));
    }

    @Test
    public void aLongerBeamRanksBetterThanAShorterOneAtTheSameAverage() {
        // 4 tokens at cum -1.0 average -0.25 per token; 1 token at cum -0.25: the same score at penalty 1
        assertEquals(BeamSearchScoring.score(4, -1.0, 1.0f), BeamSearchScoring.score(1, -0.25, 1.0f), 1e-12);
        // the same cum over different lengths: the longer beam's higher average wins at penalty 1,
        // the raw cum ties at penalty 0
        assertTrue(BeamSearchScoring.score(4, -1.0, 1.0f) > BeamSearchScoring.score(1, -1.0, 1.0f));
        assertEquals(BeamSearchScoring.score(1, -1.0, 0.0f), BeamSearchScoring.score(4, -1.0, 0.0f), 0.0);
    }
}

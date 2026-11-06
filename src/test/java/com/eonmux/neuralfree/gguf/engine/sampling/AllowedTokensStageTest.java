package com.eonmux.neuralfree.gguf.engine.sampling;

import com.eonmux.neuralfree.gguf.engine.Sampler;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import org.junit.Test;

import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The {@link AllowedTokensStage} of a request (vLLM {@code allowed_token_ids}): every candidate outside the set
 * takes a {@code -inf} logit, so the chain draws only among the allowed ids — a greedy argmax outside the set
 * falls to the best allowed one.
 */
public class AllowedTokensStageTest {

    /** A flat logit row with three moderate peaks; id 5 peaks highest. */
    private static float[] row() {
        float[] row = new float[32];
        row[5] = 2.0f;
        row[9] = 1.8f;
        row[14] = 1.5f;
        return row;
    }

    private static SamplerConfig seeded() {
        return SamplerConfig.builder().temperature(1f).topK(0).topP(1f).minP(0f).typicalP(1f)
                .penaltyLastN(0).seed(11L).build();
    }

    @Test
    public void theStageMasksEveryIdOutsideTheAllowedSet() {
        Sampler.Candidates c = Sampler.create(seeded(), new int[0],
                List.of(new AllowedTokensStage(new int[]{2, 5}))).apply(row());
        assertEquals(Float.NEGATIVE_INFINITY, c.logit(0), 0f);
        assertEquals(Float.NEGATIVE_INFINITY, c.logit(9), 0f);
        assertEquals(Float.NEGATIVE_INFINITY, c.logit(14), 0f);
        assertEquals("allowed ids keep their logit", row()[2], c.logit(2), 0f);
        assertEquals(row()[5], c.logit(5), 0f);
    }

    @Test
    public void aWholeVocabularyAllowedSetMasksNothing() {
        int[] all = new int[32];
        for (int i = 0; i < all.length; i++) {
            all[i] = i;
        }
        Sampler.Candidates c = Sampler.create(seeded(), new int[0],
                List.of(new AllowedTokensStage(all))).apply(row());
        assertEquals(row()[5], c.logit(5), 0f);
        assertEquals(row()[9], c.logit(9), 0f);
    }

    @Test
    public void greedyFallsToTheBestAllowedIdWhenItsArgmaxIsMasked() {
        Sampler allowed = Sampler.create(SamplerConfig.greedyDecoding(), new int[0],
                List.of(new AllowedTokensStage(new int[]{14, 9, 2})));
        assertEquals("the peak at 5 is masked; the next best allowed wins", 9, allowed.sample(row()));
        Sampler single = Sampler.create(SamplerConfig.greedyDecoding(), new int[0],
                List.of(new AllowedTokensStage(new int[]{2})));
        assertEquals(2, single.sample(row()));
    }

    @Test
    public void theRequestStageOnlyBuildsForANonEmptyNonNegativeSet() {
        assertThrows(IllegalArgumentException.class, () -> new AllowedTokensStage(new int[0]));
        assertThrows(IllegalArgumentException.class, () -> new AllowedTokensStage(new int[]{1, -2}));
        assertTrue(Arrays.equals(new int[]{3, 7}, new AllowedTokensStage(new int[]{7, 3}).allowed()));
    }
}

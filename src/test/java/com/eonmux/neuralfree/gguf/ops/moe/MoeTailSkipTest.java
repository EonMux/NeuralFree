package com.eonmux.neuralfree.gguf.ops.moe;

import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link MoeTailSkip}: Avalon's hosted tail-skip rule over routed ids ({@code STRATA_ROUTE_TAIL_SKIP},
 * ggml/src/ggml-cuda/moe-cache.cu:6805-6860) — an uncached expert is dropped only when every token of the step
 * routes it at the given 0-based top-k position or lower, a cached expert is never dropped, and rank 0 keeps every
 * expert.
 */
public class MoeTailSkipTest {

    private static final int K = 2;
    private static final int N_EXPERT = 8;

    private static boolean[] noneCached() {
        return new boolean[N_EXPERT];
    }

    @Test
    public void dropsOnlyTheExpertsAtTheTailOfEveryToken() {
        // two tokens: (0 rank0, 1 rank1), (2 rank0, 3 rank1)
        int[] ids = {0, 1, 2, 3};
        boolean[] skip = MoeTailSkip.skippedExperts(ids, ids.length, K, 1, noneCached(), N_EXPERT);
        assertFalse("expert 0 is a token's first pick", skip[0]);
        assertTrue("expert 1 is only ever a second pick", skip[1]);
        assertFalse("expert 2 is a token's first pick", skip[2]);
        assertTrue("expert 3 is only ever a second pick", skip[3]);
    }

    @Test
    public void keepsAnExpertSomeTokenRoutesAtAHigherRank() {
        // token 0 routes 1 second; token 1 routes 1 first: expert 1 is not a tail pick of the step
        int[] ids = {0, 1, 1, 2};
        boolean[] skip = MoeTailSkip.skippedExperts(ids, ids.length, K, 1, noneCached(), N_EXPERT);
        assertFalse(skip[1]);
        assertTrue("expert 2 is only ever token 1's second pick", skip[2]);
        assertFalse(skip[0]);
    }

    @Test
    public void neverDropsACachedExpertOrAnythingAtARankAboveTheTail() {
        int[] ids = {0, 1, 2, 3};
        boolean[] cached = noneCached();
        cached[1] = true;
        cached[3] = true;
        boolean[] skip = MoeTailSkip.skippedExperts(ids, ids.length, K, 1, cached, N_EXPERT);
        assertFalse(skip[1]);
        assertFalse(skip[3]);

        // rank 2 is below no position of a 2-of-8 router
        boolean[] none = MoeTailSkip.skippedExperts(ids, ids.length, K, 2, noneCached(), N_EXPERT);
        for (boolean b : none) {
            assertFalse(b);
        }
    }

    @Test
    public void rankZeroKeepsEveryExpert() {
        int[] ids = {0, 1, 2, 3};
        boolean[] skip = MoeTailSkip.skippedExperts(ids, ids.length, K, MoeTailSkip.OFF, noneCached(), N_EXPERT);
        assertArrayEquals(new boolean[N_EXPERT], skip);
    }

    @Test
    public void invalidInputsAreRefused() {
        int[] ids = {0, 1, 2};
        assertThrows(IllegalArgumentException.class,
                () -> MoeTailSkip.skippedExperts(ids, ids.length, K, 1, noneCached(), N_EXPERT));
        assertThrows(IllegalArgumentException.class,
                () -> MoeTailSkip.skippedExperts(new int[]{0, 8}, 2, K, 1, noneCached(), N_EXPERT));
        assertThrows(IllegalArgumentException.class,
                () -> MoeTailSkip.skippedExperts(ids, 2, K, 10, noneCached(), N_EXPERT));
        assertThrows(IllegalArgumentException.class,
                () -> MoeTailSkip.skippedExperts(ids, 2, K, -1, noneCached(), N_EXPERT));
    }
}

package com.eonmux.neuralfree.gguf.engine;

import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The equal step split of {@link StepSizes}: as few steps as the cap allows, of equal size up to one token, and the
 * rollback tail llama.cpp keeps in the last step ({@code n_keep_tail}, src/llama-batch.cpp:714-765).
 */
public final class StepSizesTest {

    /** The steps a run of {@code total} tokens takes, calling {@link StepSizes#next} as the backends' loops do. */
    private static int[] steps(int total, int maxStep) {
        return steps(total, maxStep, 0);
    }

    private static int[] steps(int total, int maxStep, int keepTail) {
        List<Integer> sizes = new ArrayList<>();
        for (int left = total; left > 0; ) {
            int n = StepSizes.next(left, maxStep, keepTail);
            sizes.add(n);
            left -= n;
        }
        int[] out = new int[sizes.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = sizes.get(i);
        }
        return out;
    }

    private static void assertEqualSplit(int total, int maxStep) {
        int[] sizes = steps(total, maxStep);
        int sum = 0;
        int min = Integer.MAX_VALUE;
        int max = 0;
        for (int s : sizes) {
            assertTrue(total + "/" + maxStep + ": step of " + s + " exceeds the cap", s <= maxStep);
            sum += s;
            min = Math.min(min, s);
            max = Math.max(max, s);
        }
        assertEquals(total + "/" + maxStep + ": the steps hold every token", total, sum);
        assertEquals(total + "/" + maxStep + ": as few steps as the cap allows", (total + maxStep - 1) / maxStep,
                sizes.length);
        assertTrue(total + "/" + maxStep + ": steps differ by at most one token", max - min <= 1);
    }

    @Test
    public void aPromptSplitsIntoAsFewEqualStepsAsTheCapAllows() {
        int[] sizes = steps(19014, 8192);
        assertEquals(3, sizes.length);
        assertEquals("19014 at a cap of 8192 is 3 steps of 6338, not 8192 / 8192 / 2630", 6338, sizes[0]);
        assertEquals(6338, sizes[1]);
        assertEquals(6338, sizes[2]);
        assertArrayEqualsInts(new int[]{1667, 1667, 1666}, steps(5000, 2048));
        assertArrayEqualsInts(new int[]{5, 5}, steps(10, 8));
        assertArrayEqualsInts(new int[]{4, 3, 3}, steps(10, 4));
        assertArrayEqualsInts(new int[]{1}, steps(1, 8));
    }

    @Test
    public void everyRunSplitsEquallyUpToItsCap() {
        for (int maxStep = 1; maxStep <= 40; maxStep++) {
            for (int total = 1; total <= 300; total++) {
                assertEqualSplit(total, maxStep);
            }
        }
    }

    @Test
    public void theRollbackTailStaysInTheLastStep() {
        for (int maxStep = 2; maxStep <= 24; maxStep++) {
            for (int keepTail = 1; keepTail < maxStep; keepTail++) {
                for (int total = keepTail; total <= 300; total++) {
                    int[] sizes = steps(total, maxStep, keepTail);
                    int sum = 0;
                    for (int s : sizes) {
                        assertTrue("step of " + s + " exceeds " + maxStep, s <= maxStep);
                        sum += s;
                    }
                    assertEquals(total + "/" + maxStep + "/" + keepTail, total, sum);
                    assertTrue("the last step of " + total + "/" + maxStep + "/" + keepTail + " holds "
                            + sizes[sizes.length - 1] + " tokens, fewer than the tail",
                            sizes[sizes.length - 1] >= keepTail);
                }
            }
        }
        assertArrayEqualsInts(new int[]{4, 3, 4}, steps(11, 5, 4));
        assertArrayEqualsInts(new int[]{5, 5}, steps(10, 8, 4));
        assertArrayEqualsInts(new int[]{4, 4}, steps(8, 5, 4));
    }

    @Test
    public void aStepNoLargerThanTheTailSplitsPlainly() {
        assertArrayEqualsInts(new int[]{4, 2}, steps(6, 4, 4));
        assertArrayEqualsInts(new int[]{4, 4, 2}, steps(10, 4, 6));
    }

    @Test
    public void refusesBadRequests() {
        assertThrows(IllegalArgumentException.class, () -> StepSizes.next(0, 8));
        assertThrows(IllegalArgumentException.class, () -> StepSizes.next(8, 0));
        assertThrows(IllegalArgumentException.class, () -> StepSizes.next(8, 4, -1));
    }

    private static void assertArrayEqualsInts(int[] expected, int[] actual) {
        assertEquals(expected.length, actual.length);
        for (int i = 0; i < expected.length; i++) {
            assertEquals("step " + i, expected[i], actual[i]);
        }
    }
}

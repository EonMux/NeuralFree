package com.eonmux.neuralfree.gguf.engine;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link RepetitionDetection} against vLLM's {@code check_sequence_repetition} cases
 * (tests/v1/core/test_repetition_detection.py) and {@code RepetitionDetectionParams} validation.
 */
public class RepetitionDetectionTest {

    private static boolean detects(RepetitionDetection d, int... tokens) {
        return d.detects(tokens, tokens.length);
    }

    private static int[] repeat(int[] pattern, int times) {
        int[] out = new int[pattern.length * times];
        for (int i = 0; i < out.length; i++) {
            out[i] = pattern[i % pattern.length];
        }
        return out;
    }

    @Test
    public void aPatternRepeatedMinCountTimesIsDetected() {
        assertTrue(detects(RepetitionDetection.of(3, 2, 3), 1, 2, 3, 1, 2, 3, 1, 2, 3));
        assertTrue(detects(RepetitionDetection.of(5, 2, 4), 1, 2, 1, 2, 1, 2, 1, 2));
    }

    @Test
    public void fewerRepeatsThanMinCountAreNot() {
        assertFalse(detects(RepetitionDetection.of(3, 2, 3), 1, 2, 3, 1, 2, 3));
        assertFalse(detects(RepetitionDetection.of(5, 2, 5), 10, 20, 10, 20, 10, 20));
    }

    @Test
    public void variedOrIncompleteRepeatsAreNot() {
        assertFalse(detects(RepetitionDetection.of(5, 2, 2), 1, 2, 3, 4, 5, 6, 7, 8, 9));
        assertFalse("the tail breaks the pattern", detects(RepetitionDetection.of(3, 2, 3), 1, 2, 3, 1, 2, 3, 1, 2, 4));
    }

    @Test
    public void onlyTheTailCounts() {
        assertTrue(detects(RepetitionDetection.of(3, 2, 3), 1, 2, 3, 4, 5, 6, 5, 6, 5, 6));
        assertFalse("a repeat earlier in the output is not a stop",
                detects(RepetitionDetection.of(3, 2, 3), 5, 6, 5, 6, 5, 6, 1, 2, 3));
    }

    @Test
    public void largePatternsAndSizeBoundariesAreChecked() {
        assertTrue(detects(RepetitionDetection.of(10, 2, 3), repeat(new int[]{1, 2, 3, 4, 5, 6, 7, 8}, 5)));
        assertTrue("exactly max = min = pattern length",
                detects(RepetitionDetection.of(3, 3, 2), 10, 20, 30, 10, 20, 30));
        assertTrue(detects(RepetitionDetection.of(5, 2, 3), repeat(new int[]{7, 8, 9, 10}, 3)));
    }

    @Test
    public void minPatternSizeFiltersShorterPatterns() {
        assertFalse(detects(RepetitionDetection.of(5, 3, 3), 10, 20, 10, 20, 10, 20));
        assertTrue("min 0 means single-token patterns", detects(RepetitionDetection.of(2, 0, 3), 4, 7, 7, 7));
        assertFalse("min 2 skips the single-token run", detects(RepetitionDetection.of(2, 2, 3), 4, 7, 7, 7));
    }

    @Test
    public void emptyAndDisabledNeverDetect() {
        assertFalse(detects(RepetitionDetection.of(3, 2, 2)));
        assertFalse(detects(RepetitionDetection.disabled(), 1, 2, 1, 2, 1, 2));
        assertFalse(RepetitionDetection.disabled().enabled());
    }

    @Test
    public void onlyTheFirstCountTokensAreRead() {
        int[] buffer = {1, 1, 1, 9, 9, 9};
        assertTrue(RepetitionDetection.of(1, 1, 3).detects(buffer, 3));
        assertFalse(RepetitionDetection.of(1, 1, 3).detects(buffer, 2));
        assertThrows(IllegalArgumentException.class, () -> RepetitionDetection.of(1, 1, 3).detects(buffer, 7));
    }

    @Test
    public void hugeSizesStopSearchingOnceRepeatsCannotFit() {
        assertFalse(detects(RepetitionDetection.of(Integer.MAX_VALUE, 1, Integer.MAX_VALUE / 2 + 1), 3, 3));
        assertTrue(detects(RepetitionDetection.of(Integer.MAX_VALUE, 0, 2), 3, 4, 3, 4));
    }

    @Test
    public void validationFollowsVllm() {
        assertThrows(IllegalArgumentException.class, () -> RepetitionDetection.of(-1, 0, 2));
        assertThrows(IllegalArgumentException.class, () -> RepetitionDetection.of(3, -1, 2));
        assertThrows(IllegalArgumentException.class, () -> RepetitionDetection.of(2, 3, 2));
        assertThrows(IllegalArgumentException.class, () -> RepetitionDetection.of(0, 1, 0));
        IllegalArgumentException count = assertThrows(IllegalArgumentException.class,
                () -> RepetitionDetection.of(3, 1, 1));
        assertTrue(count.getMessage().contains("min_count"));
        assertSame("max 0 disables whatever the count", RepetitionDetection.disabled(),
                RepetitionDetection.of(0, 0, 0));
        assertSame(RepetitionDetection.disabled(), RepetitionDetection.of(0, 0, 1));
    }

    @Test
    public void valuesCompareByTheirSettings() {
        assertEquals(RepetitionDetection.of(4, 1, 3), RepetitionDetection.of(4, 1, 3));
        assertEquals(RepetitionDetection.of(4, 1, 3).hashCode(), RepetitionDetection.of(4, 1, 3).hashCode());
        assertNotEquals(RepetitionDetection.of(4, 1, 3), RepetitionDetection.of(4, 2, 3));
        assertEquals(4, RepetitionDetection.of(4, 1, 3).maxPatternSize());
        assertEquals(1, RepetitionDetection.of(4, 1, 3).minPatternSize());
        assertEquals(3, RepetitionDetection.of(4, 1, 3).minCount());
    }
}

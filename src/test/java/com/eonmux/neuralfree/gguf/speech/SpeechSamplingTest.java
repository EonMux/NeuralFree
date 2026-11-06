package com.eonmux.neuralfree.gguf.speech;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link SpeechSampling}: the top-k cut, the top-p prefix with its crossing element, the cumulative draw and the
 * lower-index tie break of llama.cpp's {@code do_sampling}.
 */
public class SpeechSamplingTest {

    @Test
    public void topKKeepsTheLargestLogits() {
        float[] logits = {-1f, 2f, -1f, 0f};
        // order: 1 (2.0), 3 (0.0), then 0 and 2
        assertEquals(1, SpeechSampling.draw(logits, 2, 1f, 0f));
        assertEquals(1, SpeechSampling.draw(logits, 2, 1f, 0.87f));
        assertEquals(3, SpeechSampling.draw(logits, 2, 1f, 0.95f));
    }

    @Test
    public void topPKeepsTheCrossingElement() {
        float[] logits = {2f, 1f, 0f};
        // probabilities are [0.665, 0.245, 0.090]; the first element already crosses 0.5
        for (float rand : new float[]{0f, 0.5f, 0.999f}) {
            assertEquals(0, SpeechSampling.draw(logits, 0, 0.5f, rand));
        }
        // p = 1 keeps every element
        assertEquals(2, SpeechSampling.draw(logits, 0, 1f, 0.99f));
    }

    @Test
    public void drawWalksTheCumulativeDistribution() {
        float[] logits = {0f, 0f, 0f, 0f};
        assertEquals(0, SpeechSampling.draw(logits, 1, 1f, 0.0f));
        assertEquals(1, SpeechSampling.draw(logits, 0, 1f, 0.30f));
        assertEquals(2, SpeechSampling.draw(logits, 0, 1f, 0.55f));
        assertEquals(3, SpeechSampling.draw(logits, 0, 1f, 0.99f));
    }

    @Test
    public void tiesBreakByVocabularyIndex() {
        float[] logits = {1f, 1f, 1f};
        assertEquals(0, SpeechSampling.argmax(logits));
        assertEquals(0, SpeechSampling.draw(logits, 0, 1f, 0f));
    }

    @Test
    public void argmaxPicksTheHighestLogit() {
        assertEquals(2, SpeechSampling.argmax(new float[]{-3f, 0f, 4f, 1f}));
    }

    @Test
    public void refusesBadArguments() {
        assertThrows(IllegalArgumentException.class, () -> SpeechSampling.draw(new float[0], 0, 1f, 0f));
        assertThrows(IllegalArgumentException.class, () -> SpeechSampling.draw(new float[]{1f}, -1, 1f, 0f));
        assertThrows(IllegalArgumentException.class, () -> SpeechSampling.draw(new float[]{1f}, 0, 0f, 0f));
        assertThrows(IllegalArgumentException.class, () -> SpeechSampling.draw(new float[]{1f}, 0, 1.5f, 0f));
        assertThrows(IllegalArgumentException.class, () -> SpeechSampling.draw(new float[]{1f}, 0, 1f, 1f));
        assertThrows(IllegalArgumentException.class, () -> SpeechSampling.draw(new float[]{1f}, 0, 1f, -0.1f));
    }

    @Test
    public void topKAboveTheRowLengthDisablesTheCut() {
        float[] logits = {0f, 0f};
        assertTrue(SpeechSampling.draw(logits, 10, 1f, 0.4f) == 0
                || SpeechSampling.draw(logits, 10, 1f, 0.4f) == 1);
        assertEquals(1, SpeechSampling.draw(logits, 10, 1f, 0.75f));
    }
}

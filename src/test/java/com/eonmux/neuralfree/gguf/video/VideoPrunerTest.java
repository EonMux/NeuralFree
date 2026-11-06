package com.eonmux.neuralfree.gguf.video;

import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The video token pruning: the retained counts of vLLM's {@code compute_retained_tokens_count}, the EVS mask that
 * always keeps the first frame and scores tokens by their dissimilarity to the frame before them, the VidCom2
 * mask that keeps an exact count deterministically, and the named refusals.
 */
public class VideoPrunerTest {

    /** {@code frames} x {@code perFrame} rows of {@code dim} values, every row of a frame the same vector. */
    private static float[][] video(int frames, int perFrame, float[][] frameVectors) {
        float[][] rows = new float[frames * perFrame][];
        for (int f = 0; f < frames; f++) {
            for (int t = 0; t < perFrame; t++) {
                rows[f * perFrame + t] = frameVectors[f];
            }
        }
        return rows;
    }

    @Test
    public void theRetainedCountIsOneMinusTheRateAtOneFrameLeast() {
        assertEquals(40, VideoPruner.retainedCount(4, 10, 0));
        assertEquals(20, VideoPruner.retainedCount(4, 10, 0.5));
        assertEquals(4, VideoPruner.retainedCount(4, 10, 0.9));
        assertEquals(4, VideoPruner.retainedCount(4, 10, 0.99));
        assertThrows(IllegalArgumentException.class, () -> VideoPruner.retainedCount(4, 10, 1));
        assertThrows(IllegalArgumentException.class, () -> VideoPruner.retainedCount(0, 10, 0.5));
    }

    @Test
    public void vidcom2RetainsAtLeastOneRowPerFrame() {
        // two rows per frame over ten frames: EVS keeps two rows, VidCom2 one per frame
        assertEquals(2, VideoPruner.retainedCount(2, 10, 0.9));
        assertEquals(20, VideoPruner.retainedCount(2, 10, 0));
        assertEquals(10, VideoPruner.vidcom2RetainedCount(2, 10, 0.9));
        assertEquals(20, VideoPruner.vidcom2RetainedCount(2, 10, 0));
        // never more than the video's rows
        assertEquals(6, VideoPruner.vidcom2RetainedCount(3, 2, 0));
        assertThrows(IllegalArgumentException.class, () -> VideoPruner.vidcom2RetainedCount(2, 10, 1));
    }

    @Test
    public void reconciliationDropsTheHighestSimilarityAndAddsTheLowest() {
        // four retained rows, three wanted: the most similar row (index 1) goes
        boolean[] drop = {true, true, true, true};
        double[][] similarity = {{0.1}, {0.9}, {0.5}, {0.2}};
        VideoPruner.reconcile(drop, similarity, 4, 3);
        assertArrayEquals(new boolean[] {true, false, true, true}, drop);
        // one retained row, three wanted: the two least similar dropped rows (3 and 2) come back
        boolean[] add = {true, false, false, false};
        VideoPruner.reconcile(add, similarity, 1, 3);
        assertArrayEquals(new boolean[] {true, false, true, true}, add);
    }

    @Test
    public void evsAlwaysKeepsTheFirstFrameAndTheFramesThatDiffer() {
        // three frames of four rows: frame 2 repeats frame 1, frame 3 is its opposite
        float[][] vectors = {
                {1, 0, 0, 0, 0, 0, 0, 0},
                {1, 0, 0, 0, 0, 0, 0, 0},
                {-1, 0, 0, 0, 0, 0, 0, 0},
        };
        float[][] rows = video(3, 4, vectors);
        boolean[] mask = VideoPruner.evsMask(rows, 3, 4, 0.5);
        assertEquals(6, VideoPruner.retainedCount(4, 3, 0.5));
        int kept = 0;
        for (boolean b : mask) {
            if (b) {
                kept++;
            }
        }
        assertEquals(6, kept);
        // frame 1's tokens score above everything, so every one of them stays
        for (int t = 0; t < 4; t++) {
            assertTrue("first frame row " + t, mask[t]);
        }
        // frame 2 equals frame 1, so none of its tokens survive the top 6 against frame 3's
        for (int t = 4; t < 8; t++) {
            assertFalse("repeated frame row " + t, mask[t]);
        }
        // frame 3's four dissimilar tokens outrank frame 2's, and the stable tie keeps its first two
        for (int t = 8; t < 10; t++) {
            assertTrue("changed frame row " + t, mask[t]);
        }
        assertFalse(mask[10]);
        assertFalse(mask[11]);
    }

    @Test
    public void vidcom2KeepsItsExactCountDeterministically() {
        float[][] rows = new float[4 * 4][8];
        for (int f = 0; f < 4; f++) {
            for (int t = 0; t < 4; t++) {
                for (int c = 0; c < 8; c++) {
                    rows[f * 4 + t][c] = (f * 4 + t) * (c % 2 == 0 ? 0.5f : -0.25f) + c * 0.125f;
                }
            }
        }
        boolean[] mask = VideoPruner.vidcom2Mask(rows, 4, 4, 0.5);
        int kept = 0;
        for (boolean b : mask) {
            if (b) {
                kept++;
            }
        }
        assertEquals(8, kept);
        boolean[] again = VideoPruner.vidcom2Mask(rows, 4, 4, 0.5);
        assertArrayEquals(mask, again);
        // q = 0 keeps everything
        boolean[] full = VideoPruner.vidcom2Mask(rows, 4, 4, 0);
        for (boolean b : full) {
            assertTrue(b);
        }
    }

    @Test
    public void prunersRefuseRowsThatAreNotFramesOfARowCount() {
        float[][] rows = new float[10][4];
        assertThrows(IllegalArgumentException.class, () -> VideoPruner.evsMask(rows, 3, 4, 0.5));
        assertThrows(IllegalArgumentException.class, () -> VideoPruner.vidcom2Mask(rows, 3, 4, 0.5));
    }
}

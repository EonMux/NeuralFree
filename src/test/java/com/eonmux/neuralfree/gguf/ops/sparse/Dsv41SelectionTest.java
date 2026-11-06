package com.eonmux.neuralfree.gguf.ops.sparse;

import org.junit.Test;

import java.util.Arrays;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;

/**
 * The DeepSeek-V4.1 two-level candidate selection ({@link Dsv41Selection}) against the hand-computed rule of
 * models/deepseek41.cpp:588-611.
 */
public class Dsv41SelectionTest {

    private static int[] sorted(int[] v) {
        int[] c = v.clone();
        Arrays.sort(c);
        return c;
    }

    @Test
    public void bestBlocksAreKept() {
        float[] scores = {-5f, 1f, 3f, -2f, 4f, 0f, 2f};
        float[] bmax = new float[Dsv41Selection.blockCount(7, 2)];
        int[] blocks = new int[2];
        int k = Dsv41Selection.blockTopK(scores, 0, 7, bmax, 2, 2, 7, blocks, 0);
        assertEquals(2, k);
        // block maxima are 1, 3, 4, 2 and the newest row (6) sits in block 3, pinned: blocks 3 and 2 win
        assertArrayEquals(new int[]{2, 3}, sorted(blocks));
    }

    @Test
    public void theNewestVisibleBlockIsAlwaysKept() {
        // block 0 is far better, but the newest visible compressed row sits in block 2 and one block is kept
        float[] scores = {9f, 9f, 1f, 1f, 2f, 2f};
        float[] bmax = new float[Dsv41Selection.blockCount(6, 2)];
        int[] blocks = new int[1];
        int k = Dsv41Selection.blockTopK(scores, 0, 6, bmax, 2, 1, 5, blocks, 0);
        assertEquals(1, k);
        assertEquals("the pinned block wins", 2, blocks[0]);
    }

    @Test
    public void withoutVisibleRowsThereIsNoPin() {
        float[] scores = {9f, 9f, 1f, 1f, 2f, 2f};
        float[] bmax = new float[Dsv41Selection.blockCount(6, 2)];
        int[] blocks = new int[1];
        Dsv41Selection.blockTopK(scores, 0, 6, bmax, 2, 1, 0, blocks, 0);
        assertEquals(0, blocks[0]);
    }

    @Test
    public void partialBlocksCountAsBlocks() {
        // 5 rows of block size 2: 3 blocks, the last one half full
        assertEquals(3, Dsv41Selection.blockCount(5, 2));
        float[] scores = {0f, 0f, 0f, 0f, 7f};
        float[] bmax = new float[3];
        int[] blocks = new int[3];
        int k = Dsv41Selection.blockTopK(scores, 0, 5, bmax, 2, 3, 5, blocks, 0);
        assertEquals(3, k);
        assertArrayEquals(new int[]{0, 1, 2}, sorted(Arrays.copyOf(blocks, k)));
    }

    @Test
    public void rowsOutsideTheCandidateBlocksAreMasked() {
        float[] scores = {1f, 2f, 3f, 4f, 5f, 6f};
        int[] mask = new int[Dsv41Selection.blockCount(6, 2)];
        Dsv41Selection.restrictToBlocks(scores, 0, 6, 2, new int[]{1}, 0, 1, mask);
        assertArrayEquals(new float[]{Float.NEGATIVE_INFINITY, Float.NEGATIVE_INFINITY, 3f, 4f,
                Float.NEGATIVE_INFINITY, Float.NEGATIVE_INFINITY}, scores, 0f);
    }

    @Test
    public void invalidBlocksAreIgnored() {
        float[] scores = {1f, 2f, 3f, 4f};
        int[] mask = new int[2];
        Dsv41Selection.restrictToBlocks(scores, 0, 4, 2, new int[]{-1, 5, 0}, 0, 3, mask);
        assertArrayEquals(new float[]{1f, 2f, Float.NEGATIVE_INFINITY, Float.NEGATIVE_INFINITY}, scores, 0f);
    }

    @Test
    public void malformedShapesAreRefused() {
        assertThrows(IllegalArgumentException.class, () -> Dsv41Selection.blockCount(4, 0));
        float[] scores = new float[4];
        assertThrows(IllegalArgumentException.class,
                () -> Dsv41Selection.blockTopK(scores, 0, 4, new float[2], 2, 1, 5, new int[1], 0));
        assertThrows(IllegalArgumentException.class,
                () -> Dsv41Selection.blockTopK(scores, 0, 4, new float[1], 2, 1, 4, new int[1], 0));
        assertThrows(IllegalArgumentException.class,
                () -> Dsv41Selection.restrictToBlocks(scores, 0, 4, 2, new int[1], 0, 2, new int[2]));
    }
}

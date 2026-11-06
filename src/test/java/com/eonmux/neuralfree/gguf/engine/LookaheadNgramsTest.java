package com.eonmux.neuralfree.gguf.engine;

import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/** The n-gram table of lookahead decoding: llama.cpp's per-first-token ring, its dedupe and its overwrite. */
public class LookaheadNgramsTest {

    @Test
    public void storesAGramPerFirstToken() {
        LookaheadNgrams table = new LookaheadNgrams(3, 3);
        assertTrue(table.add(7, new int[] {11, 12}));
        assertTrue(table.add(7, new int[] {13, 14}));
        assertEquals(2, table.count(7));
        assertArrayEquals(new int[] {11, 12}, table.gram(7, 0));
        assertArrayEquals(new int[] {13, 14}, table.gram(7, 1));
        assertEquals(0, table.count(8));
        assertEquals(1, table.distinctFirstTokens());
        assertEquals(2, table.total());
    }

    @Test
    public void aDuplicateIsNotStoredAgain() {
        LookaheadNgrams table = new LookaheadNgrams(4, 3);
        int[] gram = {1, 2, 3};
        assertTrue(table.add(9, gram));
        assertFalse(table.add(9, new int[] {1, 2, 3}));
        assertEquals(1, table.count(9));
        assertEquals(1, table.total());
        // a duplicate of an older entry is skipped as well, whatever the ring position
        table.add(9, new int[] {4, 5, 6});
        assertFalse(table.add(9, new int[] {1, 2, 3}));
        assertEquals(2, table.count(9));
    }

    @Test
    public void aFullRingOverwritesItsOldestEntry() {
        LookaheadNgrams table = new LookaheadNgrams(2, 2);
        table.add(5, new int[] {1});
        table.add(5, new int[] {2});
        assertTrue(table.add(5, new int[] {3}));
        assertEquals(2, table.count(5));
        // the ring advanced: entry 0 now holds the newest gram, entry 1 the second
        assertArrayEquals(new int[] {3}, table.gram(5, 0));
        assertArrayEquals(new int[] {2}, table.gram(5, 1));
        assertEquals(3, table.total());
    }

    @Test
    public void gramCopiesItsEntry() {
        LookaheadNgrams table = new LookaheadNgrams(3, 2);
        int[] gram = {1, 2};
        table.add(4, gram);
        gram[0] = 99;
        assertArrayEquals(new int[] {1, 2}, table.gram(4, 0));
        int[] read = table.gram(4, 0);
        read[1] = 99;
        assertArrayEquals(new int[] {1, 2}, table.gram(4, 0));
    }

    @Test
    public void invalidKeysAndContinuationsRefused() {
        LookaheadNgrams table = new LookaheadNgrams(3, 2);
        assertThrows(IllegalArgumentException.class, () -> table.add(-1, new int[] {1, 2}));
        assertThrows(IllegalArgumentException.class, () -> table.add(1, new int[] {1}));
        assertThrows(IllegalArgumentException.class, () -> table.add(1, new int[] {1, -2}));
        assertThrows(IllegalArgumentException.class, () -> new LookaheadNgrams(1, 2));
        assertThrows(IllegalArgumentException.class, () -> new LookaheadNgrams(3, 0));
    }

    @Test
    public void gramOutsideTheRingRefused() {
        LookaheadNgrams table = new LookaheadNgrams(3, 2);
        table.add(1, new int[] {2, 3});
        assertThrows(IndexOutOfBoundsException.class, () -> table.gram(1, 1));
        assertThrows(IndexOutOfBoundsException.class, () -> table.gram(2, 0));
    }
}

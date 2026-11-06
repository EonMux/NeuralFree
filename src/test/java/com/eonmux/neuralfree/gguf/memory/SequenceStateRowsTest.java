package com.eonmux.neuralfree.gguf.memory;

import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The per-row record of a paged sequence ({@link SequenceState#appendRows}): one entry per stored row with its
 * position, media rows sharing one position, the llama.cpp {@code seq_pos_tok_le} lookup
 * ({@link SequenceState#tokenAtOrBefore}), and the removal, shift and copy that follow the rows' cells.
 */
public class SequenceStateRowsTest {

    private static int[] positions(SequenceState s) {
        int[] out = new int[s.nPast()];
        for (int i = 0; i < out.length; i++) {
            out[i] = s.position(i);
        }
        return out;
    }

    private static final int STAND_IN = 7;
    private static final int NONE = -1;

    /** Text at 0..2, four media rows recording stand-in 7 at 3, text at 5..6. */
    private static SequenceState mediaRecord() {
        SequenceState s = new SequenceState(16, 32);
        s.appendRows(new int[] {10, 11, 12, -1, -1, -1, -1, 20, 21}, new int[] {0, 1, 2, 3, 3, 3, 3, 5, 6}, 0, 9,
                STAND_IN);
        return s;
    }

    @Test
    public void rowsKeepTheirPositionsAndMayShareOne() {
        SequenceState s = mediaRecord();
        assertEquals(9, s.nPast());
        assertArrayEquals(new int[] {10, 11, 12, 7, 7, 7, 7, 20, 21}, s.tokens());
        assertArrayEquals(new int[] {0, 1, 2, 3, 3, 3, 3, 5, 6}, positions(s));
        assertEquals("a paged append raises no contiguous reuse frontier", 0, s.reusable());
    }

    @Test
    public void embeddingRowsAreMarkedAndLeftOutOfTheTextView() {
        SequenceState s = mediaRecord();
        for (int i = 0; i < s.nPast(); i++) {
            assertEquals("entry " + i, i >= 3 && i < 7, s.embeddingRow(i));
        }
        assertArrayEquals(new int[] {10, 11, 12, 20, 21}, s.textTokens());
        SequenceState none = new SequenceState(16, 32);
        none.appendRows(new int[] {4, -1, 5}, new int[] {0, 1, 2}, 0, 3, NONE);
        assertArrayEquals("without a stand-in an embedding row records no id", new int[] {4, -1, 5}, none.tokens());
        assertArrayEquals(new int[] {4, 5}, none.textTokens());
        SequenceState contiguous = new SequenceState(16);
        contiguous.append(new int[] {1, 2}, 0, 2);
        contiguous.appendEmbeddings(2, STAND_IN);
        contiguous.append(new int[] {3}, 0, 1);
        assertArrayEquals(new int[] {1, 2, 7, 7, 3}, contiguous.tokens());
        assertArrayEquals(new int[] {1, 2, 3}, contiguous.textTokens());
    }

    @Test
    public void theLookupTakesTheNearestEntryAtOrBeforeThePosition() {
        SequenceState s = new SequenceState(16, 32);
        s.appendRows(new int[] {10, 11, 12, 40, 41, 20}, new int[] {0, 1, 2, 3, 3, 5}, 0, 6, NONE);
        assertEquals("nothing before position 0", -1, s.tokenAtOrBefore(-1));
        assertEquals(10, s.tokenAtOrBefore(0));
        assertEquals("the last row written at a shared position", 41, s.tokenAtOrBefore(3));
        assertEquals("a gap reads the entry before it", 41, s.tokenAtOrBefore(4));
        assertEquals(20, s.tokenAtOrBefore(5));
        assertEquals(20, s.tokenAtOrBefore(100));
        assertEquals(-1, new SequenceState(4).tokenAtOrBefore(3));
    }

    @Test
    public void descendingPositionsAndBadRangesRefuse() {
        SequenceState s = new SequenceState(16, 32);
        s.appendRows(new int[] {1, 2}, new int[] {4, 5}, 0, 2, NONE);
        assertThrows(IllegalArgumentException.class, () -> s.appendRows(new int[] {3}, new int[] {4}, 0, 1, NONE));
        assertThrows(IllegalArgumentException.class,
                () -> s.appendRows(new int[] {3}, new int[] {6, 7}, 0, 2, NONE));
        assertThrows(IllegalArgumentException.class, () -> s.appendRows(new int[] {3}, new int[] {6}, -1, 1, NONE));
        assertEquals(2, s.nPast());
    }

    @Test
    public void theRowCapacityBoundsTheRecordAndCannotFallBelowTheContext() {
        SequenceState s = new SequenceState(4, 6);
        assertEquals(4, s.nCtx());
        assertEquals(6, s.capacity());
        s.appendRows(new int[] {-1, -1, -1, -1, -1, -1}, new int[] {0, 0, 0, 0, 0, 0}, 0, 6, STAND_IN);
        assertThrows(IllegalStateException.class, () -> s.appendRows(new int[] {1}, new int[] {1}, 0, 1, NONE));
        assertThrows(IllegalArgumentException.class, () -> new SequenceState(8, 4));
    }

    @Test
    public void aTextAppendContinuesAfterTheLastStoredPosition() {
        SequenceState s = mediaRecord();
        s.append(new int[] {30, 31}, 0, 2);
        assertArrayEquals(new int[] {0, 1, 2, 3, 3, 3, 3, 5, 6, 7, 8}, positions(s));
        s.appendEmbeddings(1, STAND_IN);
        assertEquals(9, s.position(s.nPast() - 1));
        assertEquals(STAND_IN, s.token(s.nPast() - 1));
        assertTrue(s.embeddingRow(s.nPast() - 1));
    }

    @Test
    public void aRemovalDropsTheRowsOfItsPositions() {
        SequenceState tail = mediaRecord();
        tail.removePositions(3, Integer.MAX_VALUE);
        assertArrayEquals("a tail removal drops every media row at the shared position", new int[] {10, 11, 12},
                tail.tokens());
        SequenceState interior = mediaRecord();
        interior.removePositions(1, 4);
        assertArrayEquals(new int[] {10, 20, 21}, interior.tokens());
        assertArrayEquals(new int[] {0, 5, 6}, positions(interior));
        assertThrows(IllegalArgumentException.class, () -> interior.removePositions(-1, 2));
    }

    @Test
    public void aRemovalLowersTheReuseFrontierToTheFirstDroppedEntry() {
        SequenceState s = new SequenceState(16);
        s.append(new int[] {1, 2, 3, 4, 5}, 0, 5);
        assertEquals(5, s.reusable());
        s.removePositions(2, 3);
        assertArrayEquals(new int[] {1, 2, 4, 5}, s.tokens());
        assertEquals(2, s.reusable());
        s.removePositions(10, 12);
        assertEquals("a removal of nothing keeps the frontier", 2, s.reusable());
    }

    @Test
    public void aShiftMovesTheRowsOfItsRangeAndDropsThoseBelowZero() {
        SequenceState s = mediaRecord();
        s.removePositions(1, 3);
        s.shiftPositions(3, Integer.MAX_VALUE, -2);
        assertArrayEquals(new int[] {10, 7, 7, 7, 7, 20, 21}, s.tokens());
        assertArrayEquals(new int[] {0, 1, 1, 1, 1, 3, 4}, positions(s));
        assertArrayEquals("the moved entries stay embedding rows", new int[] {10, 20, 21}, s.textTokens());
        assertEquals(20, s.tokenAtOrBefore(3));
        s.shiftPositions(0, 2, -1);
        assertArrayEquals("the row moved to -1 leaves, as its cell does", new int[] {7, 7, 7, 7, 20, 21},
                s.tokens());
        assertArrayEquals(new int[] {0, 0, 0, 0, 3, 4}, positions(s));
    }

    @Test
    public void aShiftReordersByPositionKeepingTheWrittenOrderOnTies() {
        SequenceState past = new SequenceState(16, 32);
        past.appendRows(new int[] {1, -1, 3}, new int[] {0, 4, 5}, 0, 3, STAND_IN);
        past.shiftPositions(4, 5, 3);
        assertArrayEquals("a row moved past another follows it", new int[] {1, 3, STAND_IN}, past.tokens());
        assertArrayEquals(new int[] {0, 5, 7}, positions(past));
        assertTrue("the embedding mark moves with its entry", past.embeddingRow(2));
        assertEquals(3, past.tokenAtOrBefore(6));
        SequenceState s = new SequenceState(16, 32);
        s.appendRows(new int[] {1, 2, 3, 4}, new int[] {0, 1, 5, 6}, 0, 4, NONE);
        s.shiftPositions(5, 7, -4);
        assertArrayEquals(new int[] {1, 2, 3, 4}, s.tokens());
        assertArrayEquals(new int[] {0, 1, 1, 2}, positions(s));
        assertEquals("the later-written row wins the shared position", 3, s.tokenAtOrBefore(1));
    }

    @Test
    public void aCopyTakesTheEntriesOfTheRange() {
        SequenceState src = mediaRecord();
        SequenceState dst = new SequenceState(16, 32);
        dst.appendRows(new int[] {99}, new int[] {0}, 0, 1, NONE);
        dst.copyFrom(src, 0, 4);
        assertArrayEquals(new int[] {10, 11, 12, 7, 7, 7, 7}, dst.tokens());
        assertArrayEquals(new int[] {0, 1, 2, 3, 3, 3, 3}, positions(dst));
        dst.copyFrom(src, 2, 6);
        assertArrayEquals(new int[] {12, 7, 7, 7, 7, 20}, dst.tokens());
        assertThrows(IllegalArgumentException.class, () -> dst.copyFrom(dst, 0, 4));
    }

    @Test
    public void aCopyFromZeroKeepsTheCopiedReusePrefix() {
        SequenceState src = new SequenceState(16);
        src.append(new int[] {1, 2, 3, 4}, 0, 4);
        SequenceState dst = new SequenceState(16);
        dst.copyFrom(src, 0, 3);
        assertEquals(3, dst.reusable());
        assertEquals(3, dst.commonPrefix(new int[] {1, 2, 3, 9}));
        dst.copyFrom(src, 1, 3);
        assertEquals("a copy that skips position 0 reuses nothing", 0, dst.reusable());
    }

    /**
     * The compaction commit's record fix ({@link SequenceState#retokenize}): the entries stay at their
     * positions, only their ids follow the moved rows, a rewritten embedding row becomes a text row and the
     * reuse frontier falls to the first rewritten entry, whose row a plain decode would have computed with
     * another token.
     */
    @Test
    public void aRetokenizeRewritesIdsAtTheirPositionsAndCapsTheFrontier() {
        SequenceState s = new SequenceState(16);
        s.append(new int[] {10, 11, 12, 20, 21}, 0, 5);
        assertEquals(5, s.reusable());
        s.retokenize(new int[] {3, 4}, new int[] {25, 26});
        assertArrayEquals("the entries keep their positions", new int[] {0, 1, 2, 3, 4}, positions(s));
        assertArrayEquals(new int[] {10, 11, 12, 25, 26}, s.tokens());
        assertEquals("the frontier falls to the first rewritten entry", 3, s.reusable());
        assertEquals(25, s.tokenAtOrBefore(3));
        assertEquals(26, s.tokenAtOrBefore(4));

        SequenceState media = mediaRecord();
        media.retokenize(new int[] {5}, new int[] {25});
        assertEquals("the first entry at the position takes the id", 25, media.token(7));
        assertTrue("the rewritten media row reads as text", !media.embeddingRow(7));
        assertTrue("its shared-position siblings keep theirs", media.embeddingRow(4));
        assertArrayEquals(new int[] {10, 11, 12, 25, 21}, media.textTokens());

        assertThrows(IllegalArgumentException.class, () -> s.retokenize(new int[] {5}, new int[] {1, 2}));
        assertArrayEquals("a refused retokenize leaves the record alone", new int[] {10, 11, 12, 25, 26},
                s.tokens());
        assertThrows(IllegalArgumentException.class, () -> s.retokenize(new int[] {3, 2}, new int[] {1, 2}));
        assertThrows(IllegalArgumentException.class, () -> s.retokenize(new int[] {2, 2}, new int[] {1, 2}));
        assertThrows(IllegalArgumentException.class, () -> s.retokenize(new int[] {50}, new int[] {1}));
    }
}

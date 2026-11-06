package com.eonmux.neuralfree.gguf.memory;

import org.junit.Test;

import java.util.Arrays;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The embedding-row record ({@link SequenceState#appendEmbeddings}): the positions count into {@code nPast},
 * the record holds the {@code -1} embedding marker at them, the reuse frontier stays, and an overflow refuses.
 */
public class SequenceStateEmbedsTest {

    @Test
    public void embeddingRowsRecordMinusOneAndRaiseNoReuse() {
        SequenceState seq = new SequenceState(16);
        seq.append(new int[] {3, 4, 5}, 0, 3);
        assertEquals(3, seq.reusable());
        seq.appendEmbeddings(2);
        assertEquals(5, seq.nPast());
        assertEquals("the embedding rows raise no reuse frontier", 3, seq.reusable());
        assertArrayEquals(new int[] {-1, -1}, Arrays.copyOfRange(seq.tokens(), 3, 5));
        assertEquals("the reuse frontier ends before the embedding rows", 3,
                seq.commonPrefix(new int[] {3, 4, 5, -1, -1}));
        seq.append(new int[] {6}, 0, 1);
        assertEquals("a token append after embedding rows extends no reusable prefix", 3, seq.reusable());
        assertEquals(6, seq.nPast());
    }

    @Test
    public void overflowAndNegativeCountsRefuse() {
        SequenceState seq = new SequenceState(4);
        assertThrows(IllegalArgumentException.class, () -> seq.appendEmbeddings(-1));
        assertThrows(IllegalStateException.class, () -> seq.appendEmbeddings(5));
        seq.appendEmbeddings(4);
        assertEquals(4, seq.nPast());
    }

    @Test
    public void aTruncateDropsTheRecordedRows() {
        SequenceState seq = new SequenceState(16);
        seq.append(new int[] {1}, 0, 1);
        seq.appendEmbeddings(2);
        seq.truncate(2);
        assertEquals(1, seq.nPast());
        assertArrayEquals(new int[] {1}, seq.tokens());
        assertTrue(seq.reusable() <= seq.nPast());
    }
}

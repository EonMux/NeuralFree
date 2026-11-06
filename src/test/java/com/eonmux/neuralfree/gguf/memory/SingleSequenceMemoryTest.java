package com.eonmux.neuralfree.gguf.memory;

import org.junit.Test;

import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The single-sequence adapter over a {@link ModelMemory}: the default {@code sequences()} wiring and the mappings
 * between the sequence API and rollback / reset.
 */
public final class SingleSequenceMemoryTest {

    /** Minimal contiguous memory: records the rollback / reset calls and answers canRollback by policy. */
    private static final class Record implements ModelMemory {

        final SequenceState state = new SequenceState(16);
        final AtomicInteger rollbacks = new AtomicInteger();
        final AtomicInteger resets = new AtomicInteger();
        boolean rollbackable = true;

        @Override
        public MemoryPlan plan() {
            return null; // the adapter uses the sequence record only
        }

        @Override
        public SequenceState sequence() {
            return state;
        }

        @Override
        public void reset() {
            resets.incrementAndGet();
            state.clear();
        }

        @Override
        public boolean canRollback(int nTokens) {
            return rollbackable && nTokens >= 0 && state.nPast() >= nTokens;
        }

        @Override
        public void rollback(int nTokens) {
            rollbacks.addAndGet(nTokens);
            state.truncate(nTokens);
        }

        @Override
        public void snapshot() {
            throw new UnsupportedOperationException("no recurrent state in the record");
        }

        @Override
        public void restore() {
            throw new IllegalStateException("no snapshot");
        }

        @Override
        public void close() {
        }
    }

    @Test
    public void defaultSequencesWrapsThisMemory() {
        Record memory = new Record();
        SequenceMemory sequences = memory.sequences(); // the ModelMemory default builds the adapter
        assertTrue(sequences instanceof SingleSequenceMemory);
        assertEquals(1, sequences.maxSequences());
        assertSame(memory.sequence(), sequences.sequence(0));
        assertThrows(IndexOutOfBoundsException.class, () -> sequences.sequence(1));
        assertThrows(IndexOutOfBoundsException.class, () -> sequences.keep(1));
    }

    @Test
    public void posRangeOfHeldTokens() {
        Record memory = new Record();
        SequenceMemory sequences = memory.sequences();
        assertEquals(-1, sequences.posMin(0));
        assertEquals(-1, sequences.posMax(0));
        memory.sequence().append(new int[]{7, 8, 9}, 0, 3);
        assertEquals(0, sequences.posMin(0));
        assertEquals(2, sequences.posMax(0));
        assertEquals(16, sequences.totalCells());
        assertEquals(13, sequences.freeCells());
        assertEquals(3, sequences.sequence(0).nPast());
    }

    @Test
    public void removeSuffixRollsBack() {
        Record memory = new Record();
        SequenceMemory sequences = memory.sequences();
        memory.sequence().append(new int[]{1, 2, 3, 4, 5}, 0, 5);
        // drop from position p to the end
        assertNull(sequences.removeRefusal(0, 3, -1));
        sequences.remove(0, 3, -1);
        // positions [3, posMax] dropped: 5 - 3 = 2 rollbacks, positions [0, 3) kept
        assertEquals(3, sequences.sequence(0).nPast());
        assertEquals(2, sequences.posMax(0));
        assertEquals(2, memory.rollbacks.get());
        // dropping the whole range is a reset
        assertNull(sequences.removeRefusal(0, 0, -1));
        sequences.remove(0, 0, -1);
        assertEquals(1, memory.resets.get());
        assertEquals(0, sequences.sequence(0).nPast());
        // removing an interior range is refused
        assertEquals("contiguous single-sequence memory can only remove a suffix of positions",
                sequences.removeRefusal(0, 1, 3));
        assertThrows(UnsupportedOperationException.class, () -> sequences.remove(0, 1, 3));
        // dropping more than the memory can roll back is refused
        Record hard = new Record();
        SequenceMemory hardSeq = hard.sequences();
        hard.sequence().append(new int[]{1, 2, 3}, 0, 3);
        hard.rollbackable = false;
        String refusal = hardSeq.removeRefusal(0, 0, -1);
        assertNull(refusal); // p0 == 0 maps to reset, which always works
        refusal = hardSeq.removeRefusal(0, 2, -1);
        assertTrue(refusal, refusal != null && refusal.contains("canRollback(1)"));
        assertThrows(UnsupportedOperationException.class, () -> hardSeq.remove(0, 2, -1));
    }

    @Test
    public void copyAndShiftRefused() {
        Record memory = new Record();
        SequenceMemory sequences = memory.sequences();
        assertEquals("contiguous single-sequence memory cannot copy between sequences",
                sequences.copyRefusal());
        assertThrows(UnsupportedOperationException.class, () -> sequences.copy(0, 0, 0, -1));
        assertTrue(sequences.shiftRefusal(), sequences.shiftRefusal().contains("shiftContext"));
        assertThrows(UnsupportedOperationException.class, () -> sequences.add(0, 2, -1, -3));
        assertTrue(sequences.divRefusal(), sequences.divRefusal().contains("paged memory"));
        assertThrows(UnsupportedOperationException.class, () -> sequences.div(0, 0, -1, 2));
    }

    @Test
    public void keepBeyondTheOneSequenceRefused() {
        Record memory = new Record();
        SequenceMemory sequences = memory.sequences();
        memory.sequence().append(new int[]{1}, 0, 1);
        sequences.keep(0); // nothing outside sequence 0 to empty
        assertEquals(1, sequences.sequence(0).nPast());
    }

    @Test
    public void clearEmptiesTheMemory() {
        Record memory = new Record();
        SequenceMemory sequences = memory.sequences();
        memory.sequence().append(new int[]{1, 2}, 0, 2);
        sequences.clear();
        assertEquals(1, memory.resets.get());
        assertEquals(0, sequences.sequence(0).nPast());
        assertEquals(-1, sequences.posMax(0));
        assertFalse(memory.sequence().wrapped());
    }
}

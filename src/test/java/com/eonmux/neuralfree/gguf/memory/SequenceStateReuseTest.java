package com.eonmux.neuralfree.gguf.memory;

import com.eonmux.neuralfree.gguf.memory.gpu.DeviceStepInputs;
import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The contiguous-reuse frontier {@code SequenceState.reusable} behind the engine's prefix reuse: it
 * is the length of the stored prefix whose rows were computed with exactly that prefix, so it rises only by
 * contiguous appends, falls on every rewrite ({@link SequenceState#truncate}, {@link SequenceState#capReusable},
 * the context shift), resets on {@link SequenceState#clear}, and {@link SequenceState#commonPrefix} never reports
 * past it — the fix of the shift defect where the rewritten token list let a following prompt reuse rows computed
 * with the discarded context.
 */
public class SequenceStateReuseTest {

    private static SequenceState state() {
        SequenceState s = new SequenceState(8);
        s.append(new int[] {0, 1, 2, 3, 4, 5, 6, 7}, 0, 8);
        return s;
    }

    @Test
    public void contiguousAppendsRaiseTheFrontierAndCommonPrefixReportsIt() {
        SequenceState s = new SequenceState(8);

        assertEquals(0, s.reusable());
        s.append(new int[] {0, 1, 2}, 0, 3);
        assertEquals(3, s.reusable());
        s.append(new int[] {3, 4}, 0, 2);
        assertEquals(5, s.nPast());
        assertEquals(5, s.reusable());
        assertEquals(5, s.commonPrefix(new int[] {0, 1, 2, 3, 4, 9}));
        assertEquals(2, s.commonPrefix(new int[] {0, 1, 9, 3, 4}));
        assertEquals(0, s.commonPrefix(new int[] {9, 9}));
    }

    @Test
    public void truncateLowersAndTheNextContiguousAppendReextends() {
        SequenceState s = state();
        s.truncate(3);                                    // rows after the truncate point are gone

        assertEquals(5, s.nPast());
        assertEquals(5, s.reusable());
        s.append(new int[] {5, 6}, 0, 2);                 // extends the exact same prefix again
        assertEquals(7, s.nPast());
        assertEquals(7, s.reusable());
        assertEquals(7, s.commonPrefix(new int[] {0, 1, 2, 3, 4, 5, 6, 9}));

        SequenceState partial = state();
        partial.truncate(6);
        assertEquals(2, partial.nPast());
        assertEquals(2, partial.reusable());
        assertEquals(2, partial.commonPrefix(new int[] {0, 1, 2}));
    }

    @Test
    public void capReusableLowersNeverRaisesAndBlocksTheNonContiguousRaise() {
        SequenceState s = state();                        // everything contiguous
        assertEquals(8, s.reusable());

        s.capReusable(3);                                 // a rewrite at position 3 invalidates the rows after it
        assertEquals(3, s.reusable());
        s.capReusable(6);                                 // a cap never raises
        assertEquals(3, s.reusable());
        s.capReusable(-1);
        assertEquals(0, s.reusable());
        s.capReusable(8);
        assertEquals(0, s.reusable());                    // an unknown frontier stays lowered

        SequenceState blocked = state();
        blocked.truncate(8);                              // empty, frontier 0
        blocked.append(new int[] {0, 1}, 0, 2);           // contiguous from 0 again
        blocked.capReusable(1);
        blocked.append(new int[] {2, 3, 4}, 0, 3);        // non-contiguous: nPast rises, the frontier stays
        assertEquals(5, blocked.nPast());
        assertEquals(1, blocked.reusable());
        assertEquals(1, blocked.commonPrefix(new int[] {0, 1, 2, 3, 4}));

        blocked.clear();
        assertEquals(0, blocked.nPast());
        assertEquals(0, blocked.reusable());
        assertEquals(0, blocked.commonPrefix(new int[] {0, 1, 2, 3, 4}));
    }

    /** The context-shift defect fix: the rewrite caps the frontier at the kept count. */
    @Test
    public void shiftRewritesTheTokenListAndCapsTheFrontierAtTheKeptCount() {
        SequenceState s = state();
        s.markWrapped();

        s.shift(2, 3);                                    // keep [0, 1], drop [2, 3, 4], tail moves down

        assertEquals(5, s.nPast());
        assertEquals(2, s.reusable());
        assertTrue(s.wrapped());
        assertEquals(2, s.commonPrefix(s.tokens()));      // never past the frontier, even on its own tokens
        assertEquals(2, s.commonPrefix(new int[] {0, 1, 1, 1, 1, 1, 1, 1})); // 0 and 1 match, capped at 2 anyway
        assertArrayEquals(new int[] {0, 1, 5, 6, 7}, s.tokens());

        SequenceState tail = state();
        tail.shift(0, 7);
        assertEquals(1, tail.nPast());
        assertEquals(0, tail.reusable());                 // the frontier can go to 0: kept nothing
        assertArrayEquals(new int[] {7}, tail.tokens());
    }

    @Test
    public void shiftValidatesItsRange() {
        SequenceState s = state();

        assertThrows(IllegalArgumentException.class, () -> s.shift(-1, 1));
        assertThrows(IllegalArgumentException.class, () -> s.shift(2, 0));   // nothing discarded
        assertThrows(IllegalArgumentException.class, () -> s.shift(2, -1));
        assertThrows(IllegalArgumentException.class, () -> s.shift(5, 4));   // keep + discard over nPast
        assertEquals(8, s.nPast());
        assertEquals(8, s.reusable());
    }

    /** {@code SequenceRules.shiftSequence} is the record's shift: the same rewrite, wrap flag and frontier. */
    @Test
    public void theSequenceRulesShiftIsTheStateShift() {
        SequenceState s = state();
        s.markWrapped();

        DeviceStepInputs.SequenceRules.shiftSequence(s, 2, 3);

        assertEquals(5, s.nPast());
        assertEquals(2, s.reusable());
        assertTrue(s.wrapped());
        assertArrayEquals(new int[] {0, 1, 5, 6, 7}, s.tokens());

        DeviceStepInputs.SequenceRules.shiftSequence(s, 1, 2);
        assertEquals(3, s.nPast());
        assertEquals(1, s.reusable());
        assertArrayEquals(new int[] {0, 6, 7}, s.tokens());
        assertThrows(IllegalArgumentException.class,
                () -> DeviceStepInputs.SequenceRules.shiftSequence(s, 0, 0));
    }

    /** The frontier never exceeds nPast on any append. */
    @Test
    public void appendsNeverPushTheFrontierPastNpast() {
        SequenceState s = state();
        s.truncate(4);
        assertEquals(4, s.reusable());
        s.append(new int[] {0, 1, 2}, 0, 3);              // overflows the context: rejected whole
        assertThrows(IllegalStateException.class, () -> s.append(new int[] {0, 1, 2, 3, 4, 0, 0, 0}, 0, 8));
        assertEquals(7, s.nPast());
        assertEquals(7, s.reusable());
    }
}

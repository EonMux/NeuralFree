package com.eonmux.neuralfree.acceleration.backend.vulkan;

import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * Barrier decisions of {@link VkHazardTracker}: read-after-write, write-after-read and
 * write-after-write conflicts on whole buffers and on byte ranges, half-open ranges, merging of
 * adjacent accesses, the {@link VkHazardTracker#MAX_PENDING} saturation and {@link VkHazardTracker#reset()}.
 * No device is needed.
 */
public class VkHazardTrackerTest {

    private static final long[] NONE = {};

    private VkHazardTracker tracker;

    @Before
    public void createTracker() {
        tracker = new VkHazardTracker();
    }

    // ------------------------------------------------------------------ whole buffers (dispatches)

    @Test
    public void readAfterWriteOfAWholeBufferNeedsABarrier() {
        tracker.record(NONE, ids(1));

        assertTrue(tracker.needsBarrier(ids(1), NONE));
    }

    @Test
    public void writeAfterReadOfAWholeBufferNeedsABarrier() {
        tracker.record(ids(1), NONE);

        assertTrue(tracker.needsBarrier(NONE, ids(1)));
    }

    @Test
    public void writeAfterWriteOfAWholeBufferNeedsABarrier() {
        tracker.record(NONE, ids(1));

        assertTrue(tracker.needsBarrier(NONE, ids(1)));
    }

    @Test
    public void readAfterReadNeverNeedsABarrier() {
        tracker.record(ids(1, 2), NONE);

        assertFalse(tracker.needsBarrier(ids(1, 2), NONE));
    }

    @Test
    public void accessesOfOtherBuffersDoNotConflict() {
        tracker.record(ids(1), ids(2));

        assertFalse(tracker.needsBarrier(ids(3), ids(4)));
        assertFalse(tracker.needsBarrier(ids(1), ids(3)));
    }

    @Test
    public void aBufferBoundTwiceIsOneAccessPerKind() {
        tracker.record(ids(5, 5), ids(6, 6));

        assertEquals(1, tracker.pendingReads());
        assertEquals(1, tracker.pendingWrites());
    }

    @Test
    public void nullAccessListsAreEmpty() {
        tracker.record(null, null);

        assertEquals(0, tracker.pendingReads());
        assertEquals(0, tracker.pendingWrites());
        assertFalse(tracker.needsBarrier(null, null));

        tracker.record(null, ids(1));
        assertTrue(tracker.needsBarrier(ids(1), null));
        assertFalse(tracker.needsBarrier(null, ids(2)));
    }

    // ------------------------------------------------------------------ byte ranges (transfers)

    @Test
    public void overlappingRangesConflict() {
        tracker.record(0, 0, 0, 5, 0, 64);

        assertTrue("read after write", tracker.needsBarrier(5, 32, 16, 0, 0, 0));
        assertTrue("write after write", tracker.needsBarrier(0, 0, 0, 5, 63, 1));

        tracker.reset();
        tracker.record(5, 100, 50, 0, 0, 0);
        assertTrue("write after read", tracker.needsBarrier(0, 0, 0, 5, 140, 20));
    }

    @Test
    public void disjointRangesOfOneBufferDoNotConflict() {
        tracker.record(5, 0, 64, 5, 128, 64);

        assertFalse(tracker.needsBarrier(5, 256, 64, 5, 64, 64));
    }

    @Test
    public void touchingHalfOpenRangesDoNotConflict() {
        tracker.record(0, 0, 0, 5, 0, 16);

        assertFalse(tracker.needsBarrier(5, 16, 16, 0, 0, 0));
        assertFalse(tracker.needsBarrier(0, 0, 0, 5, 16, 16));
        assertTrue(tracker.needsBarrier(5, 15, 16, 0, 0, 0));
    }

    @Test
    public void readAfterReadOfOneRangeDoesNotConflict() {
        tracker.record(5, 0, 64, 0, 0, 0);

        assertFalse(tracker.needsBarrier(5, 0, 64, 6, 0, 64));
    }

    @Test
    public void adjacentAccessesMergeIntoOneEntry() {
        tracker.record(0, 0, 0, 7, 0, 16);
        tracker.record(0, 0, 0, 7, 16, 16);
        assertEquals(1, tracker.pendingWrites());

        tracker.record(0, 0, 0, 7, 8, 16);
        assertEquals("a contained range adds no entry", 1, tracker.pendingWrites());

        tracker.record(0, 0, 0, 7, 32, 16);
        assertEquals(1, tracker.pendingWrites());
        assertTrue("the merged range covers [0, 48)", tracker.needsBarrier(7, 40, 8, 0, 0, 0));

        tracker.record(0, 0, 0, 8, 0, 16);
        tracker.record(0, 0, 0, 7, 100, 16);
        assertEquals("another buffer and a gap add entries", 3, tracker.pendingWrites());
        assertFalse("the gap [48, 100) stays free", tracker.needsBarrier(7, 48, 52, 0, 0, 0));
        assertTrue(tracker.needsBarrier(7, 110, 4, 0, 0, 0));
    }

    @Test
    public void zeroIdOrZeroSizeIsNoAccess() {
        tracker.record(0, 0, 16, 0, 0, 16);
        tracker.record(5, 0, 0, 5, 0, 0);

        assertEquals(0, tracker.pendingReads());
        assertEquals(0, tracker.pendingWrites());

        tracker.record(0, 0, 0, 5, 0, 64);
        assertFalse(tracker.needsBarrier(5, 0, 0, 0, 0, 0));
        assertFalse(tracker.needsBarrier(0, 0, 64, 0, 0, 0));
        assertFalse(tracker.needsBarrier(0, 0, 0, 5, 0, 0));
    }

    // ------------------------------------------------------------------ whole buffers against ranges

    @Test
    public void wholeBufferWriteConflictsWithAnyLaterRangeOfThatBuffer() {
        tracker.record(NONE, ids(9));

        assertTrue(tracker.needsBarrier(9, 1L << 40, 4, 0, 0, 0));
        assertTrue(tracker.needsBarrier(0, 0, 0, 9, 4096, 4));
        assertFalse(tracker.needsBarrier(10, 0, 4, 0, 0, 0));
    }

    @Test
    public void rangeAccessConflictsWithALaterWholeBufferAccess() {
        tracker.record(0, 0, 0, 9, 4096, 4);
        assertTrue("read after a ranged write", tracker.needsBarrier(ids(9), NONE));

        tracker.reset();
        tracker.record(9, 4096, 4, 0, 0, 0);
        assertTrue("write after a ranged read", tracker.needsBarrier(NONE, ids(9)));
        assertFalse("read after a ranged read", tracker.needsBarrier(ids(9), NONE));
    }

    @Test
    public void wholeRangeEndsAtTheLargestOffset() {
        tracker.record(NONE, ids(3));

        assertTrue(tracker.needsBarrier(3, VkHazardTracker.WHOLE - 1, 1, 0, 0, 0));
    }

    // ------------------------------------------------------------------ saturation and reset

    @Test
    public void rangedAccessOfASaturatedKindAlwaysNeedsABarrier() {
        recordDistinctReads(VkHazardTracker.MAX_PENDING);
        assertEquals(VkHazardTracker.MAX_PENDING, tracker.pendingReads());

        assertTrue("any read once the reads are saturated",
                tracker.needsBarrier(1_000_000, 0, 4, 0, 0, 0));
        assertFalse("writes are not saturated and the buffer is untouched",
                tracker.needsBarrier(0, 0, 0, 1_000_001, 0, 4));
    }

    @Test
    public void wholeBufferAccessSaturatesWhenItWouldExceedTheLimit() {
        recordDistinctReads(VkHazardTracker.MAX_PENDING - 1);

        assertFalse("exactly the limit is still tracked", tracker.needsBarrier(ids(1_000_000), NONE));
        assertTrue("one past the limit", tracker.needsBarrier(ids(1_000_000, 1_000_001), NONE));
        assertFalse("an empty command never needs a barrier", tracker.needsBarrier(NONE, NONE));
        assertFalse(tracker.needsBarrier(null, null));
    }

    @Test
    public void resetClearsThePendingAccesses() {
        tracker.record(ids(1), ids(2));
        tracker.record(3, 0, 16, 4, 0, 16);
        assertTrue(tracker.needsBarrier(ids(2), NONE));

        tracker.reset();

        assertEquals(0, tracker.pendingReads());
        assertEquals(0, tracker.pendingWrites());
        assertFalse(tracker.needsBarrier(ids(1, 2, 3, 4), ids(1, 2, 3, 4)));
        assertFalse(tracker.needsBarrier(3, 0, 16, 4, 0, 16));
    }

    @Test
    public void resetAfterSaturationTracksAgain() {
        recordDistinctReads(VkHazardTracker.MAX_PENDING);
        tracker.reset();

        assertFalse(tracker.needsBarrier(1_000_000, 0, 4, 0, 0, 0));
        tracker.record(1_000_000, 0, 4, 0, 0, 0);
        assertEquals(1, tracker.pendingReads());
    }

    private void recordDistinctReads(int n) {
        long[] reads = new long[n];
        for (int i = 0; i < n; i++) {
            reads[i] = i + 1;
        }
        tracker.record(reads, NONE);
    }

    private static long[] ids(long... ids) {
        return ids;
    }
}

package com.eonmux.neuralfree.gguf.engine.batch;

import org.junit.Test;

import java.util.LinkedHashSet;
import java.util.Set;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;

/**
 * The batch engine's encoder output cache ({@link EncoderCache}, vLLM's {@code EncoderCacheManager}): an
 * identified chunk's rows store once and every later intern of the same identity returns them, a live
 * submission's reference keeps its entry while the least recently used unreferenced one evicts for the space,
 * eviction is all-or-nothing, an entry larger than the whole budget passes through unshared, and a disabled cache
 * never stores.
 */
public class EncoderCacheTest {

    private static float[][] rows(int n) {
        float[][] r = new float[n][8];
        for (int i = 0; i < n; i++) {
            for (int d = 0; d < 8; d++) {
                r[i][d] = 0.1f * i + 0.01f * d;
            }
        }
        return r;
    }

    @Test
    public void anInternedIdentityStoresOnceAndLaterInternsReturnItsRows() {
        EncoderCache cache = new EncoderCache(16);
        float[][] first = rows(8);
        assertSame("a miss stores the caller's rows as the canonical ones", first, cache.intern("img", first));
        assertSame("a hit returns the stored rows", first, cache.intern("img", rows(8)));
        assertEquals(8, cache.usedRows());
        assertEquals(1, cache.hits());
        assertEquals(0, cache.evictions());
    }

    @Test
    public void aSubmissionReferencesEachIdentityOnceAndReleasesWhatItHolds() {
        EncoderCache cache = new EncoderCache(16);
        Set<String> held = new LinkedHashSet<>();
        float[][] a = rows(8);
        assertSame(a, cache.reference("a", a, held));
        assertSame("the same identity twice in one prompt shares the first rows", a,
                cache.reference("a", rows(8), held));
        assertEquals(Set.of("a"), held);
        cache.release(held);
        float[][] b = rows(16);
        assertSame("one release dropped the single reference: a evicts for b", b, cache.intern("b", b));
        assertEquals(1, cache.evictions());
    }

    @Test
    public void aLiveReferenceKeepsItsEntryWhileAnUnreferencedOneEvicts() {
        EncoderCache cache = new EncoderCache(8);
        Set<String> heldA = new LinkedHashSet<>();
        float[][] a = rows(8);
        cache.reference("a", a, heldA);
        Set<String> heldB = new LinkedHashSet<>();
        float[][] b = rows(8);
        assertSame("b does not fit beside the referenced a: it stays unshared", b, cache.reference("b", b, heldB));
        assertTrue("b holds no reference to an entry the cache never stored", heldB.isEmpty());
        assertEquals(8, cache.usedRows());
        assertEquals(0, cache.evictions());
        cache.release(heldA);
        assertSame("the released a evicted for b", b, cache.intern("b", b));
        assertEquals(8, cache.usedRows());
        assertEquals(1, cache.evictions());
        assertSame("a's entry is gone: its rows store again", a, cache.intern("a", a));
    }

    @Test
    public void aPromptsLaterChunkNeverEvictsItsEarlierOne() {
        EncoderCache cache = new EncoderCache(8);
        Set<String> held = new LinkedHashSet<>();
        float[][] first = rows(8);
        cache.reference("img#0", first, held);
        float[][] second = rows(8);
        assertSame("the second chunk passes through unshared", second, cache.reference("img#1", second, held));
        assertEquals(0, cache.evictions());
        assertSame("the first chunk's entry survived", first, cache.intern("img#0", rows(8)));
    }

    @Test
    public void evictionIsAllOrNothing() {
        EncoderCache cache = new EncoderCache(16);
        Set<String> held = new LinkedHashSet<>();
        float[][] pinned = rows(8);
        cache.reference("pinned", pinned, held);
        float[][] loose = rows(8);
        cache.intern("loose", loose);
        float[][] big = rows(16);
        assertSame("the referenced entry leaves too little room: big passes through", big, cache.intern("big", big));
        assertEquals("the unreferenced entry was not evicted for nothing", 0, cache.evictions());
        assertSame(loose, cache.intern("loose", rows(8)));
        assertEquals(16, cache.usedRows());
    }

    @Test
    public void theLeastRecentlyUsedUnreferencedEntryEvictsFirst() {
        EncoderCache cache = new EncoderCache(16);
        float[][] old = rows(8);
        cache.intern("old", old);
        cache.intern("new", rows(8));
        cache.intern("old", rows(8)); // a hit refreshes old's recency
        float[][] fresh = rows(8);
        assertSame(fresh, cache.intern("fresh", fresh));
        assertEquals(1, cache.evictions());
        assertSame("the refreshed old stays", old, cache.intern("old", rows(8)));
        assertSame(fresh, cache.intern("fresh", rows(8)));
        assertEquals(16, cache.usedRows());
    }

    @Test
    public void anEntryLargerThanTheWholeBudgetPassesThroughUnshared() {
        EncoderCache cache = new EncoderCache(16);
        float[][] big = rows(24);
        Set<String> held = new LinkedHashSet<>();
        assertSame(big, cache.reference("big", big, held));
        assertTrue(held.isEmpty());
        assertEquals(0, cache.usedRows());
        assertEquals(0, cache.hits());
    }

    @Test
    public void aDisabledCacheNeverStores() {
        EncoderCache cache = new EncoderCache(0);
        float[][] rows = rows(8);
        Set<String> held = new LinkedHashSet<>();
        assertSame(rows, cache.reference("img", rows, held));
        assertTrue(held.isEmpty());
        cache.release(held);
        assertFalse(cache.enabled());
        assertEquals(0, cache.usedRows());
    }

    @Test
    public void aCachedEntryOfAnotherShapeNeverStandsInForTheRows() {
        EncoderCache cache = new EncoderCache(16);
        cache.intern("img", rows(8));
        float[][] other = rows(4);
        assertSame("a row-count mismatch keeps the caller's rows", other, cache.intern("img", other));
        assertEquals(0, cache.hits());
    }

    @Test
    public void anUnidentifiedChunkIsANoOp() {
        EncoderCache cache = new EncoderCache(16);
        float[][] rows = rows(8);
        Set<String> held = new LinkedHashSet<>();
        assertSame(rows, cache.reference(null, rows, held));
        assertTrue(held.isEmpty());
        assertEquals(0, cache.usedRows());
    }

    @Test
    public void everyReferencingSubmissionReleasesBeforeTheEntryEvicts() {
        EncoderCache cache = new EncoderCache(8);
        float[][] a = rows(8);
        Set<String> first = new LinkedHashSet<>();
        Set<String> second = new LinkedHashSet<>();
        cache.reference("a", a, first);
        assertSame(a, cache.reference("a", rows(8), second)); // two submissions reference the same entry
        cache.release(first);
        float[][] b = rows(8);
        assertSame("one reference still holds a: b stays unshared", b, cache.intern("b", b));
        assertEquals(0, cache.evictions());
        cache.release(second);
        assertSame("the last release made a evictable", b, cache.intern("b", b));
        assertEquals(1, cache.evictions());
    }

    @Test
    public void aClearDropsEveryEntryAndALaterReleaseKeepsTheNewReferences() {
        EncoderCache cache = new EncoderCache(8);
        float[][] a = rows(8);
        Set<String> old = new LinkedHashSet<>();
        cache.reference("a", a, old);
        cache.clear();
        assertEquals(0, cache.usedRows());
        float[][] fresh = rows(8);
        assertSame("the cleared identity stores its new rows", fresh, cache.intern("a", fresh));

        Set<String> current = new LinkedHashSet<>();
        cache.reference("a", rows(8), current);
        cache.release(old);
        float[][] b = rows(8);
        assertSame("the pre-clear release left the new reference: a is not evicted", b, cache.intern("b", b));
        assertEquals(0, cache.evictions());
        cache.release(current);
        assertSame("the last reference released: a evicts for b", b, cache.intern("b", b));
        assertEquals(1, cache.evictions());
    }

    @Test
    public void aReleasedEntryKeepsItsRowsUntilItsSpaceIsClaimed() {
        EncoderCache cache = new EncoderCache(16);
        float[][] a = rows(8);
        Set<String> held = new LinkedHashSet<>();
        cache.reference("a", a, held);
        cache.release(held);
        assertEquals("the unreferenced entry stays cached", 8, cache.usedRows());
        assertSame("a later request hits it instead of holding a second copy", a,
                cache.reference("a", rows(8), new LinkedHashSet<>()));
    }
}

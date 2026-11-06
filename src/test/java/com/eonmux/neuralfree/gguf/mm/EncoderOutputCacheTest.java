package com.eonmux.neuralfree.gguf.mm;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link EncoderOutputCache}: a least-recently-used cache bounded by bytes, as vLLM's LRU processor cache — the
 * least recently used entries leave first, a lookup refreshes its entry, an entry larger than the budget is not
 * stored, a zero budget stores nothing, and replacing a key keeps the byte count exact.
 */
public class EncoderOutputCacheTest {

    /** A cache of strings sized by their length. */
    private static EncoderOutputCache<String, String> cache(long maxBytes) {
        return new EncoderOutputCache<>(maxBytes, String::length);
    }

    @Test
    public void theLeastRecentlyUsedEntryLeavesFirst() {
        EncoderOutputCache<String, String> c = cache(10);
        c.put("a", "aaaa");
        c.put("b", "bbbb");
        assertEquals("aaaa", c.get("a"));
        c.put("c", "cccc");
        assertTrue(c.contains("a"));
        assertFalse(c.contains("b"));
        assertTrue(c.contains("c"));
        assertEquals(8, c.bytes());
        assertEquals(2, c.size());
    }

    @Test
    public void containsNeitherCountsNorRefreshes() {
        EncoderOutputCache<String, String> c = cache(8);
        c.put("a", "aaaa");
        c.put("b", "bbbb");
        assertTrue(c.contains("a"));
        c.put("c", "cccc");
        assertFalse("contains must not refresh the entry", c.contains("a"));
        assertEquals(0, c.queries());
    }

    @Test
    public void lookupsCountQueriesAndHits() {
        EncoderOutputCache<String, String> c = cache(8);
        c.put("a", "aa");
        assertEquals("aa", c.get("a"));
        assertNull(c.get("b"));
        assertEquals(2, c.queries());
        assertEquals(1, c.hits());
    }

    @Test
    public void anEntryLargerThanTheBudgetIsNotStored() {
        EncoderOutputCache<String, String> c = cache(4);
        c.put("a", "aa");
        c.put("big", "bbbbb");
        assertFalse(c.contains("big"));
        assertTrue("an oversize entry evicts nothing", c.contains("a"));
        assertEquals(2, c.bytes());
    }

    @Test
    public void aZeroBudgetStoresNothing() {
        EncoderOutputCache<String, String> c = cache(0);
        c.put("a", "a");
        assertFalse(c.contains("a"));
        assertEquals(0, c.size());
    }

    @Test
    public void replacingAKeyKeepsTheByteCountExact() {
        EncoderOutputCache<String, String> c = cache(10);
        c.put("a", "aaaa");
        c.put("a", "aaaaaa");
        assertEquals(6, c.bytes());
        assertEquals(1, c.size());
        c.put("b", "bbbb");
        assertEquals(10, c.bytes());
        c.put("b", "bbbbbbbbbb");
        assertFalse("the replaced entry evicts the others, never itself", c.contains("a"));
        assertTrue(c.contains("b"));
        assertEquals(10, c.bytes());
    }

    @Test
    public void clearDropsEveryEntry() {
        EncoderOutputCache<String, String> c = cache(10);
        c.put("a", "aa");
        c.put("b", "bb");
        c.clear();
        assertEquals(0, c.size());
        assertEquals(0, c.bytes());
        assertNull(c.get("a"));
    }

    @Test
    public void aNegativeBudgetIsRefused() {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> cache(-1));
        assertTrue(e.getMessage(), e.getMessage().contains("-1"));
    }
}

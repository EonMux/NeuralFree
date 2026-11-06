package com.eonmux.neuralfree.gguf.engine.batch;

import org.junit.Test;

import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The host-RAM tier's bookkeeping ({@link HostBlockCache}): a lookup finds a block only behind the very chain of
 * tokens it was evicted from and under its own extra keys, the budget drops the least recently used blocks first and
 * never keeps a block larger than itself, a re-kept chain replaces its old copy, and {@code within} answers the
 * blocks a history left.
 */
public class HostBlockCacheTest {

    private static final int P = 4;

    /** A block image of {@code tokens} with one cache layer of {@code bytesPerRow} K bytes per row and no V. */
    private static BlockImage image(int[] tokens, boolean full, int bytesPerRow) {
        int[] positions = new int[tokens.length];
        return new BlockImage(tokens, positions, full, new byte[][] {new byte[tokens.length * bytesPerRow]},
                new byte[][] {null});
    }

    /** A stand-in digest naming {@code marker}, the way the pool's hashes name blocks. */
    private static byte[] digest(long marker) {
        byte[] out = new byte[8];
        for (int i = 0; i < 8; i++) {
            out[i] = (byte) (marker >>> (56 - 8 * i));
        }
        return out;
    }

    private static HostBlockCache.Entry entry(long hash, long parent, PrefixKey key, int[] chain, int rows,
                                              int bytesPerRow) {
        int[] own = java.util.Arrays.copyOfRange(chain, chain.length - rows, chain.length);
        return new HostBlockCache.Entry(digest(hash), digest(parent), key, chain, image(own, rows == P, bytesPerRow));
    }

    @Test
    public void aLookupNeedsTheWholeChainAndTheKey() {
        HostBlockCache host = new HostBlockCache(-1);
        int[] chain = {1, 2, 3, 4, 5, 6, 7, 8};
        HostBlockCache.Entry second = entry(22, 11, PrefixKey.NONE, chain, P, 8);
        host.put(second);
        int[] want = {1, 2, 3, 4, 5, 6, 7, 8, 9};
        assertSame(second, host.whole(digest(11), PrefixKey.NONE, want, P, P));
        int[] otherPrefix = {1, 2, 3, 0, 5, 6, 7, 8};
        assertNull("another prefix under the same parent hash misses", host.whole(digest(11), PrefixKey.NONE,
                otherPrefix, P, P));
        assertNull("another key misses", host.whole(digest(11), PrefixKey.restored(null, null, "salt"), want, P, P));
        HostBlockCache.Hit hit = host.best(digest(11), PrefixKey.NONE, new int[] {1, 2, 3, 4, 5, 6, 0}, P, 7);
        assertEquals("the boundary shares two rows", 2, hit.rows);
        assertEquals(1, host.within(want, PrefixKey.NONE).size());
        assertEquals("a shorter history does not hold the block", 0,
                host.within(new int[] {1, 2, 3, 4, 5}, PrefixKey.NONE).size());
    }

    @Test
    public void theBudgetDropsTheOldestAndRefusesOversizedBlocks() {
        long perEntry = entry(1, 0, PrefixKey.NONE, new int[] {1, 2, 3, 4}, P, 100).bytes();
        HostBlockCache host = new HostBlockCache(2 * perEntry);
        HostBlockCache.Entry a = entry(1, 0, PrefixKey.NONE, new int[] {1, 2, 3, 4}, P, 100);
        HostBlockCache.Entry b = entry(2, 0, PrefixKey.NONE, new int[] {5, 6, 7, 8}, P, 100);
        HostBlockCache.Entry c = entry(3, 0, PrefixKey.NONE, new int[] {9, 10, 11, 12}, P, 100);
        assertTrue(host.put(a).isEmpty());
        assertTrue(host.put(b).isEmpty());
        host.touch(a);
        List<HostBlockCache.Entry> dropped = host.put(c);
        assertEquals("the least recently used goes first", List.of(b), dropped);
        assertEquals(2, host.size());
        assertEquals(2 * perEntry, host.bytes());
        HostBlockCache.Entry huge = entry(4, 0, PrefixKey.NONE, new int[] {1, 2, 3, 4}, P, 1000);
        assertEquals("a block larger than the budget is not kept", List.of(huge), host.put(huge));
        HostBlockCache.Entry aAgain = entry(1, 0, PrefixKey.NONE, new int[] {1, 2, 3, 4}, P, 100);
        assertTrue(host.put(aAgain).isEmpty());
        assertEquals("a re-kept chain replaces its copy", 2, host.size());
        assertSame(aAgain, host.whole(digest(0), PrefixKey.NONE, new int[] {1, 2, 3, 4}, 0, P));
        assertEquals(2, host.clear().size());
        assertEquals(0, host.bytes());
        assertThrows(IllegalArgumentException.class, () -> new HostBlockCache(0));
    }
}

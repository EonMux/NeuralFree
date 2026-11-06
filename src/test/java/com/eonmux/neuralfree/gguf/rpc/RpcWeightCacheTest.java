package com.eonmux.neuralfree.gguf.rpc;

import org.junit.After;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/** The server-side weight cache: store and load by content hash, stale entries, and the directory accounting. */
public class RpcWeightCacheTest {

    @Rule
    public TemporaryFolder folder = new TemporaryFolder();

    private RpcWeightCache cache;

    @Before
    public void open() throws IOException {
        cache = new RpcWeightCache(folder.newFolder("cache").toPath());
    }

    @After
    public void wipe() {
        cache.clear();
    }

    @Test
    public void storesAndLoadsByContentHash() {
        byte[] payload = "the-shard-bytes".getBytes();
        long hash = RpcWire.fnv1a(payload);
        cache.store(hash, payload);
        assertArrayEquals(payload, cache.load(hash));
        assertEquals(1, cache.entries());
        assertEquals(payload.length, cache.bytes());
        assertTrue(Files.exists(cache.directory().resolve(RpcWire.hashName(hash))));
    }

    @Test
    public void missesAnUnknownHash() {
        assertNull(cache.load(0xdeadbeefcafeL));
        assertEquals(0, cache.entries());
    }

    @Test
    public void replacesAnEntryOnASecondStore() {
        byte[] first = "one".getBytes();
        byte[] second = "two-bytes".getBytes();
        cache.store(RpcWire.fnv1a(first), first);
        cache.store(RpcWire.fnv1a(first), first);
        assertEquals(1, cache.entries());
        cache.store(RpcWire.fnv1a(second), second);
        assertEquals(2, cache.entries());
        assertArrayEquals(first, cache.load(RpcWire.fnv1a(first)));
    }

    @Test
    public void dropsAnEntryWhoseSizeIsStale() throws IOException {
        byte[] payload = "payload".getBytes();
        long hash = RpcWire.fnv1a(payload);
        cache.store(hash, payload);
        assertFalse(cache.dropIfStale(hash, payload.length));
        assertTrue(cache.dropIfStale(hash, payload.length + 1));
        assertNull("the stale entry is gone", cache.load(hash));
        assertFalse("an absent entry is not stale", cache.dropIfStale(hash, 1));
    }

    @Test
    public void clearRemovesEveryEntryAndKeepsTheDirectory() {
        cache.store(1L, new byte[]{1});
        cache.store(2L, new byte[]{2, 3});
        assertEquals(2, cache.clear());
        assertEquals(0, cache.entries());
        assertTrue(Files.isDirectory(cache.directory()));
    }

    @Test
    public void refusesADirectoryItCannotCreate() {
        Path file = folder.getRoot().toPath().resolve("a-file");
        try {
            Files.write(file, new byte[]{1});
            new RpcWeightCache(file);
            fail("opened a file as the cache directory");
        } catch (IOException | IllegalArgumentException e) {
            // the write or the constructor refuses
        }
    }

    @Test
    public void theDefaultDirectoryIsUnderTheUserCache() {
        String xdg = System.getenv("XDG_CACHE_HOME");
        Path expected = RpcWeightCache.defaultCacheDir();
        if (xdg != null && !xdg.trim().isEmpty()) {
            assertEquals(java.nio.file.Paths.get(xdg.trim(), "neuralfree", "rpc"), expected);
        } else {
            assertEquals(java.nio.file.Paths.get(System.getProperty("user.home"), ".cache", "neuralfree", "rpc"),
                    expected);
        }
    }

    @Test
    public void hashNamesAreSixteenHexDigits() {
        String name = RpcWire.hashName(0xabcL);
        assertEquals(16, name.length());
        assertEquals("0000000000000abc", name);
        assertEquals(name, RpcWire.hashName(0xabcL));
        try {
            cache.store(1L, null);
            fail("stored null");
        } catch (NullPointerException expected) {
            // the payload must be given
        }
    }
}

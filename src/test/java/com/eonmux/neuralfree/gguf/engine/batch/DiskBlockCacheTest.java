package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.arch.MemoryKind;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.KvLayout;
import com.eonmux.neuralfree.gguf.memory.KvWindow;
import com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.memory.MemoryPlan;
import com.eonmux.neuralfree.gguf.memory.StateTransfer;
import com.eonmux.neuralfree.gguf.memory.paged.PagedSequences;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.stream.Stream;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

/**
 * The disk tier's bookkeeping ({@link DiskBlockCache}): a lookup answers a block only behind the very chain of
 * tokens it was spilled from and under its own extra keys, the budget drops the least recently used files first and
 * never keeps a block larger than itself, a block left behind by an earlier run is found again under its digest,
 * another layout's or another hash seed's files are foreign — they miss and are left intact — and a file that fails
 * its own verification is dropped, while {@code clear} deletes the tier's own files.
 */
public class DiskBlockCacheTest {

    private static final int P = 4;
    private static final int N_CTX = 32;
    private static final int BPR = 32;
    /** The hasher the tests spill and read with: vLLM's default algorithm and seed. */
    private static final BlockHasher HASHER = BlockHasher.of(PrefixHashAlgo.SHA256, BlockHasher.DEFAULT_NONE_HASH_SEED);
    /** A chain's first block's parent digest (the hasher's {@code NONE_HASH}). */
    private static final byte[] ROOT = HASHER.noneHash();

    /** One full-attention KV group of {@link #N_CTX} rows in pages of 4, one F32 cache layer of width 8. */
    private static BlockLayout layout() {
        MemoryConfig cfg = MemoryConfig.builder()
                .typeK(KvDType.F32).typeV(KvDType.F32)
                .nCtx(N_CTX).ubatch(8).kvPaged(true).kvPageSize(P).build();
        KvWindow w = KvWindow.FULL_CAUSAL;
        MemoryPlan.KvGroup g = MemoryPlan.KvGroup.of(0, N_CTX, false, w, P, N_CTX / P);
        KvLayout kv = KvLayout.of(N_CTX, false, 8, 8, KvDType.F32, KvDType.F32, w);
        MemoryPlan plan = MemoryPlan.builder(MemoryKind.KV, N_CTX, cfg.ubatch())
                .group(g)
                .layer(LayerMemoryPlan.builder(0, LayerMemoryPlan.Kind.KV).kv(kv, 0).build())
                .config(cfg)
                .build();
        byte[][] rows = new byte[][] {new byte[N_CTX * BPR]};
        StateTransfer.Cache cache = new StateTransfer.Cache() {
            @Override public int layer() { return 0; }
            @Override public int group() { return 0; }
            @Override public KvLayout layout() { return kv; }
            @Override public void read(boolean v, int row0, int n, byte[] dst) {
                byte[] src = rows[v ? 1 : 0];
                System.arraycopy(src, row0 * BPR, dst, 0, Math.min(n * BPR, dst.length));
            }
            @Override public void write(boolean v, int row0, int n, byte[] src) {
                System.arraycopy(src, 0, rows[v ? 1 : 0], row0 * BPR, Math.min(n * BPR, src.length));
            }
        };
        return new BlockLayout(new PagedSequences(plan, 1), List.of(cache));
    }

    /** A parent digest of {@code HASHER}'s length, every byte {@code v}. */
    private static byte[] parent(byte v) {
        byte[] p = new byte[HASHER.algo().digestSize()];
        Arrays.fill(p, v);
        return p;
    }

    /** The chain hash of a block holding {@code own} after the digest {@code parent}, the key mixed into a root. */
    private static byte[] hash(BlockHasher hasher, byte[] parent, int[] own, PrefixKey key) {
        boolean root = Arrays.equals(parent, hasher.noneHash());
        return PrefixCache.chainHash(hasher, root ? null : parent, CacheRows.ofTokens(own), 0, own.length,
                root ? key : null);
    }

    private static HostBlockCache.Entry entry(BlockHasher hasher, byte[] parent, PrefixKey key, int[] chain,
                                              int rows, int bytesPerRow) {
        int[] own = Arrays.copyOfRange(chain, chain.length - rows, chain.length);
        byte[] hash = hash(hasher, parent, own, key);
        BlockImage image = new BlockImage(own, new int[rows], rows == P,
                new byte[][] {new byte[rows * bytesPerRow]}, new byte[][] {new byte[rows * bytesPerRow]});
        return new HostBlockCache.Entry(hash, parent, key, chain, image);
    }

    private static long countFiles(Path dir) throws IOException {
        try (Stream<Path> files = Files.walk(dir)) {
            return files.filter(f -> f.getFileName().toString().endsWith(".bin")).count();
        }
    }

    /** Waits for the tier's background writer and applies its results, answering the events. */
    private static List<DiskBlockCache.Event> flush(DiskBlockCache disk) {
        assertTrue("the background writer drains (" + disk + ")", disk.awaitWrites(10_000));
        return disk.pump();
    }

    @Test
    public void aLookupNeedsTheWholeChainAndTheKey() throws IOException {
        Path dir = Files.createTempDirectory("nf-disk-");
        DiskBlockCache disk = new DiskBlockCache(dir, -1, layout(), HASHER);
        byte[] parent = parent((byte) 11);
        int[] chain = {1, 2, 3, 4, 5, 6, 7, 8};
        HostBlockCache.Entry second = entry(HASHER, parent, PrefixKey.NONE, chain, P, BPR);
        assertTrue(disk.put(second).stored);
        int[] want = {1, 2, 3, 4, 5, 6, 7, 8, 9};
        HostBlockCache.Entry hit = disk.whole(parent, PrefixKey.NONE, want, P, P);
        assertTrue(Arrays.equals(second.hash, hit.hash));
        assertTrue(Arrays.equals(second.chain(), hit.chain()));
        int[] otherPrefix = {1, 2, 3, 0, 5, 6, 7, 8};
        assertNull("another prefix under the same parent digest misses", disk.whole(parent, PrefixKey.NONE,
                otherPrefix, P, P));
        assertNull("another key misses", disk.whole(parent, PrefixKey.restored(null, null, "salt"), want, P, P));
        HostBlockCache.Hit best = disk.best(parent, PrefixKey.NONE, new int[] {1, 2, 3, 4, 5, 6, 0}, P, 7);
        assertEquals("the boundary shares two rows", 2, best.rows);
        assertEquals(1, disk.within(want, PrefixKey.NONE).size());
        assertEquals("a shorter history does not hold the block", 0,
                disk.within(new int[] {1, 2, 3, 4, 5}, PrefixKey.NONE).size());
    }

    @Test
    public void theBudgetDropsTheOldestAndRefusesOversizedBlocks() throws IOException {
        PrefixKey key = PrefixKey.NONE;
        DiskBlockCache wide = new DiskBlockCache(Files.createTempDirectory("nf-disk-"), -1, layout(), HASHER);
        wide.put(entry(HASHER, ROOT, key, new int[] {1, 2, 3, 4}, P, BPR));
        wide.put(entry(HASHER, ROOT, key, new int[] {5, 6, 7, 8}, P, BPR));
        flush(wide);
        long perFile = wide.bytes() / 2;
        Path dir = Files.createTempDirectory("nf-disk-");
        DiskBlockCache disk = new DiskBlockCache(dir, 2 * perFile, layout(), HASHER);
        HostBlockCache.Entry a = entry(HASHER, ROOT, key, new int[] {1, 2, 3, 4}, P, BPR);
        HostBlockCache.Entry b = entry(HASHER, ROOT, key, new int[] {5, 6, 7, 8}, P, BPR);
        HostBlockCache.Entry c = entry(HASHER, ROOT, key, new int[] {9, 10, 11, 12}, P, BPR);
        assertTrue(disk.put(a).stored);
        assertTrue(disk.put(b).stored);
        flush(disk);
        assertTrue("the least recently used read refreshes it", disk.whole(a.parentHash, key,
                new int[] {1, 2, 3, 4}, 0, P) != null);
        assertTrue(disk.put(c).stored);
        List<byte[]> evicted = new ArrayList<>();
        for (DiskBlockCache.Event event : flush(disk)) {
            if (event.removed != null) {
                evicted.add(event.removed);
            }
        }
        assertEquals("the least recently used file goes first", 1, evicted.size());
        assertTrue(Arrays.equals(b.hash, evicted.get(0)));
        assertFalse(disk.holds(b.hash));
        assertEquals(2 * perFile, disk.bytes());
        HostBlockCache.Entry huge = entry(HASHER, ROOT, key, new int[] {13, 14, 15, 16}, P, 20 * BPR);
        DiskBlockCache.Spill refused = disk.put(huge);
        assertFalse("a block larger than the budget is not written", refused.stored);
        assertFalse(disk.holds(huge.hash));
        assertTrue(disk.put(a).stored);
        flush(disk);
        assertEquals("a re-written chain replaces its copy", 2, disk.size());
        assertEquals(2, disk.clear().size());
        assertEquals(0, disk.bytes());
        assertEquals("clear deletes every file", 0, countFiles(dir));
    }

    @Test
    public void aRunFindsTheBlocksAnEarlierRunLeftBehind() throws IOException {
        Path dir = Files.createTempDirectory("nf-disk-");
        PrefixKey key = PrefixKey.NONE;
        DiskBlockCache first = new DiskBlockCache(dir, -1, layout(), HASHER);
        HostBlockCache.Entry a = entry(HASHER, ROOT, key, new int[] {1, 2, 3, 4}, P, BPR);
        first.put(a);
        flush(first);
        DiskBlockCache second = new DiskBlockCache(dir, -1, layout(), HASHER);
        assertTrue("the scan indexed the block left behind", second.holds(a.hash));
        HostBlockCache.Entry hit = second.whole(ROOT, key, new int[] {1, 2, 3, 4}, 0, P);
        assertTrue(Arrays.equals(a.hash, hit.hash));
        assertTrue(Arrays.equals(a.chain(), hit.chain()));
        assertTrue(second.holds(a.hash));
        second.remove(a.hash);
        assertFalse(second.holds(a.hash));
        assertEquals(0, countFiles(dir));
    }

    @Test
    public void anotherLayoutsFileIsKeptButNeverServed() throws IOException {
        Path dir = Files.createTempDirectory("nf-disk-");
        BlockLayout wide = layout();
        MemoryConfig cfg = MemoryConfig.builder()
                .typeK(KvDType.F32).typeV(KvDType.F32)
                .nCtx(N_CTX).ubatch(8).kvPaged(true).kvPageSize(P).build();
        KvWindow w = KvWindow.FULL_CAUSAL;
        MemoryPlan.KvGroup g = MemoryPlan.KvGroup.of(0, N_CTX, false, w, P, N_CTX / P);
        KvLayout kv = KvLayout.of(N_CTX, false, 16, 16, KvDType.F32, KvDType.F32, w);
        MemoryPlan plan = MemoryPlan.builder(MemoryKind.KV, N_CTX, cfg.ubatch())
                .group(g)
                .layer(LayerMemoryPlan.builder(0, LayerMemoryPlan.Kind.KV).kv(kv, 0).build())
                .config(cfg)
                .build();
        StateTransfer.Cache cache = new StateTransfer.Cache() {
            @Override public int layer() { return 0; }
            @Override public int group() { return 0; }
            @Override public KvLayout layout() { return kv; }
            @Override public void read(boolean v, int row0, int n, byte[] dst) { }
            @Override public void write(boolean v, int row0, int n, byte[] src) { }
        };
        BlockLayout narrow = new BlockLayout(new PagedSequences(plan, 1), List.of(cache));
        PrefixKey key = PrefixKey.NONE;
        DiskBlockCache writer = new DiskBlockCache(dir, -1, wide, HASHER);
        writer.put(entry(HASHER, ROOT, key, new int[] {1, 2, 3, 4}, P, BPR));
        flush(writer);
        DiskBlockCache other = new DiskBlockCache(dir, -1, narrow, HASHER);
        assertEquals("another layout's files are another namespace's, not this tier's", 0, other.size());
        assertNull("another layout's file is never served", other.whole(ROOT, key, new int[] {1, 2, 3, 4}, 0, P));
        assertEquals("it is left for the layout it belongs to", 1, countFiles(dir));
    }

    @Test
    public void anotherSeedsFilesAreForeignAndLeftIntact() throws IOException {
        Path dir = Files.createTempDirectory("nf-disk-");
        PrefixKey key = PrefixKey.NONE;
        // two xxhash engines over one directory: the default seed is random per process, so their NONE_HASH
        // digests differ and each cache's namespace excludes the other's files
        BlockHasher one = BlockHasher.of(PrefixHashAlgo.XXHASH, "seed-one");
        BlockHasher two = BlockHasher.of(PrefixHashAlgo.XXHASH, "seed-two");
        DiskBlockCache first = new DiskBlockCache(dir, -1, layout(), one);
        assertTrue(first.put(entry(one, one.noneHash(), key, new int[] {1, 2, 3, 4}, P, BPR)).stored);
        flush(first);
        long files = countFiles(dir);
        DiskBlockCache second = new DiskBlockCache(dir, -1, layout(), two);
        assertEquals("another seed's files are not this tier's", 0, second.size());
        assertNull("a block spilled under another seed's NONE_HASH misses",
                second.whole(one.noneHash(), key, new int[] {1, 2, 3, 4}, 0, P));
        assertEquals("a foreign file is left intact, never deleted as corruption", files, countFiles(dir));
        assertTrue(second.put(entry(two, two.noneHash(), key, new int[] {5, 6, 7, 8}, P, BPR)).stored);
        flush(second);
        assertEquals("the other namespace keeps sitting beside the new file", files + 1, countFiles(dir));
        assertEquals("the first seed's block still reads under its own hasher",
                1, first.size());
    }

    @Test
    public void aSpillReturnsBeforeTheDiskWrite() throws Exception {
        Path dir = Files.createTempDirectory("nf-disk-");
        CountDownLatch entered = new CountDownLatch(1);
        CountDownLatch release = new CountDownLatch(1);
        DiskBlockCache.Io slow = new RealIo() {
            @Override
            public void write(byte[] blob, Path file) throws IOException {
                entered.countDown();
                await(release);
                super.write(blob, file);
            }
        };
        DiskBlockCache disk = new DiskBlockCache(dir, -1, layout(), HASHER, slow, 4, 1);
        try {
            HostBlockCache.Entry a = entry(HASHER, ROOT, PrefixKey.NONE, new int[] {1, 2, 3, 4}, P, BPR);
            long start = System.nanoTime();
            assertTrue(disk.put(a).stored);
            long elapsed = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - start);
            assertTrue("put hands the block to the writer, took " + elapsed + " ms", elapsed < 1000);
            assertTrue("the writer started", entered.await(5, TimeUnit.SECONDS));
            assertTrue("the block is promotable from its host image until its file is durable",
                    disk.wholeLoaded(ROOT, PrefixKey.NONE, new int[] {1, 2, 3, 4}, 0, P) != null);
            assertEquals("no file before the write completes", 0, countFiles(dir));
            assertEquals("no storage event before the file is durable", 0, stored(disk.pump()));
            release.countDown();
            assertTrue(disk.awaitWrites(5000));
            List<DiskBlockCache.Event> events = disk.pump();
            assertEquals("the durable block publishes its storage store", 1, stored(events));
            assertTrue("the file is durable", disk.durable(a.hash));
            assertEquals(1, countFiles(dir));
        } finally {
            release.countDown();
            disk.close();
        }
    }

    /**
     * The drain counts every accepted write from its admission to its outcome: a write in flight and one queued
     * behind it keep the drain waiting, and once it answers true both writes' store events answer through
     * {@link #pump}. The old drain read the queue and the in-flight count separately, so it could answer true
     * while the writer had taken a job it had not started (and false while a write it would finish stood between
     * its two reads) — the flake {@code anotherSeedsFilesAreForeignAndLeftIntact} hit once.
     */
    @Test
    public void awaitWritesAnswersTrueOnlyAfterEveryAcceptedWriteFinished() throws Exception {
        Path dir = Files.createTempDirectory("nf-disk-");
        CountDownLatch entered = new CountDownLatch(1);
        CountDownLatch release = new CountDownLatch(1);
        DiskBlockCache.Io slow = new RealIo() {
            @Override
            public void write(byte[] blob, Path file) throws IOException {
                entered.countDown();
                await(release);
                super.write(blob, file);
            }
        };
        DiskBlockCache disk = new DiskBlockCache(dir, -1, layout(), HASHER, slow, 4, 1);
        try {
            HostBlockCache.Entry a = entry(HASHER, ROOT, PrefixKey.NONE, new int[] {1, 2, 3, 4}, P, BPR);
            assertTrue(disk.put(a).stored);
            assertTrue("the writer started", entered.await(5, TimeUnit.SECONDS));
            HostBlockCache.Entry b = entry(HASHER, ROOT, PrefixKey.NONE, new int[] {5, 6, 7, 8}, P, BPR);
            assertTrue(disk.put(b).stored);
            assertEquals("one write in flight, one queued: neither finished", 2, disk.unfinishedWriteCount());
            assertFalse("an unfinished write keeps the drain waiting", disk.awaitWrites(50));
            release.countDown();
            assertTrue(disk.awaitWrites(5000));
            assertEquals("true means the writer finished every accepted write", 0, disk.unfinishedWriteCount());
            List<DiskBlockCache.Event> events = disk.pump();
            assertEquals("a drained tier answers both writes' stores", 2, stored(events));
            assertTrue(Arrays.equals(a.hash, events.get(0).stored.hash));
            assertTrue(Arrays.equals(b.hash, events.get(1).stored.hash));
            assertTrue(disk.durable(a.hash));
            assertTrue(disk.durable(b.hash));
        } finally {
            release.countDown();
            disk.close();
        }
    }

    @Test
    public void closeDrainsTheAcceptedWritesThenRefusesMore() throws Exception {
        Path dir = Files.createTempDirectory("nf-disk-");
        DiskBlockCache disk = new DiskBlockCache(dir, -1, layout(), HASHER, DiskBlockCache.FILES, 4, 1);
        HostBlockCache.Entry a = entry(HASHER, ROOT, PrefixKey.NONE, new int[] {1, 2, 3, 4}, P, BPR);
        assertTrue(disk.put(a).stored);
        disk.close(30_000);
        assertEquals("shutdown drains the accepted write", 1, stored(disk.pump()));
        assertTrue(disk.durable(a.hash));
        assertFalse("a closed tier accepts no further spill",
                disk.put(entry(HASHER, ROOT, PrefixKey.NONE, new int[] {5, 6, 7, 8}, P, BPR)).stored);
        disk.close();
        assertEquals(1, countFiles(dir));
    }

    @Test
    public void closeAbandonsAWriteThatOutlivesItsBudget() throws Exception {
        Path dir = Files.createTempDirectory("nf-disk-");
        CountDownLatch entered = new CountDownLatch(1);
        CountDownLatch release = new CountDownLatch(1);
        DiskBlockCache.Io slow = new RealIo() {
            @Override
            public void write(byte[] blob, Path file) throws IOException {
                entered.countDown();
                await(release);
                super.write(blob, file);
            }
        };
        DiskBlockCache disk = new DiskBlockCache(dir, -1, layout(), HASHER, slow, 4, 1);
        HostBlockCache.Entry a = entry(HASHER, ROOT, PrefixKey.NONE, new int[] {1, 2, 3, 4}, P, BPR);
        assertTrue(disk.put(a).stored);
        assertTrue("the writer started", entered.await(5, TimeUnit.SECONDS));
        long start = System.nanoTime();
        disk.close(50);
        long elapsed = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - start);
        assertTrue("shutdown abandons a slow write, took " + elapsed + " ms", elapsed < 3000);
        assertFalse("a closed tier accepts no spill",
                disk.put(entry(HASHER, ROOT, PrefixKey.NONE, new int[] {5, 6, 7, 8}, P, BPR)).stored);
        release.countDown();
    }

    /** An {@link DiskBlockCache.Io} over the real filesystem whose tests slow one operation. */
    private static class RealIo implements DiskBlockCache.Io {
        @Override
        public byte[] read(Path file) throws IOException {
            return Files.readAllBytes(file);
        }

        @Override
        public void write(byte[] blob, Path file) throws IOException {
            DiskBlockCache.FILES.write(blob, file);
        }

        @Override
        public void delete(Path file) throws IOException {
            Files.deleteIfExists(file);
        }
    }

    private static void await(CountDownLatch latch) {
        try {
            latch.await(10, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    /** How many storage stores the events hold. */
    private static int stored(List<DiskBlockCache.Event> events) {
        int n = 0;
        for (DiskBlockCache.Event e : events) {
            if (e.stored != null) {
                n++;
            }
        }
        return n;
    }

    @Test
    public void aCorruptedFileIsDeletedAndMisses() throws IOException {
        Path dir = Files.createTempDirectory("nf-disk-");
        PrefixKey key = PrefixKey.NONE;
        DiskBlockCache disk = new DiskBlockCache(dir, -1, layout(), HASHER);
        disk.put(entry(HASHER, ROOT, key, new int[] {1, 2, 3, 4}, P, BPR));
        flush(disk);
        Path file;
        try (Stream<Path> files = Files.walk(dir)) {
            file = files.filter(f -> f.getFileName().toString().endsWith(".bin")).findFirst().orElseThrow();
        }
        byte[] blob = Files.readAllBytes(file);
        Files.write(file, Arrays.copyOf(blob, blob.length - 8));
        DiskBlockCache reopened = new DiskBlockCache(dir, -1, layout(), HASHER);
        assertNull(reopened.whole(ROOT, key, new int[] {1, 2, 3, 4}, 0, P));
        assertFalse(reopened.holds(PrefixCache.chainHash(HASHER, null, CacheRows.ofTokens(new int[] {1, 2, 3, 4}),
                0, 4, key)));
        assertEquals("the truncated file is gone", 0, countFiles(dir));
    }

    @Test
    public void aFileWhoseRowsDoNotHashToItsNameIsDeleted() throws IOException {
        Path dir = Files.createTempDirectory("nf-disk-");
        PrefixKey key = PrefixKey.NONE;
        DiskBlockCache disk = new DiskBlockCache(dir, -1, layout(), HASHER);
        HostBlockCache.Entry a = entry(HASHER, ROOT, key, new int[] {1, 2, 3, 4}, P, BPR);
        disk.put(a);
        flush(disk);
        Path file;
        try (Stream<Path> files = Files.walk(dir)) {
            file = files.filter(f -> f.getFileName().toString().endsWith(".bin")).findFirst().orElseThrow();
        }
        // the same rows and parent under a different name: the name's digest no longer matches the rows
        byte[] renamed = HASHER.hash(ROOT, new int[] {1, 2, 3, 5}, null);
        Path copy = file.resolveSibling(DiskBlockCacheTest.nameOf(renamed, ROOT));
        Files.copy(file, copy);
        Files.delete(file);
        DiskBlockCache reopened = new DiskBlockCache(dir, -1, layout(), HASHER);
        assertTrue("the scan indexed the renamed file under its own digest", reopened.holds(renamed));
        assertNull("its rows do not hash to that name, so it never serves", reopened.whole(ROOT, key,
                new int[] {1, 2, 3, 4}, 0, P));
        assertFalse(reopened.holds(renamed));
        assertEquals("the lying file is gone", 0, countFiles(dir));
    }

    /** The file name {@link DiskBlockCache} gives a block: its digest and its parent's, hex. */
    private static String nameOf(byte[] hash, byte[] parent) {
        return DiskBlockCacheTest.hex(hash) + "_" + DiskBlockCacheTest.hex(parent) + ".bin";
    }

    private static String hex(byte[] bytes) {
        StringBuilder sb = new StringBuilder(bytes.length * 2);
        for (byte b : bytes) {
            sb.append(Character.forDigit((b >> 4) & 0xf, 16)).append(Character.forDigit(b & 0xf, 16));
        }
        return sb.toString();
    }
}

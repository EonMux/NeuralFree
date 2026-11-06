package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.TokenCallback;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.engine.kvevents.KvCacheEvent;
import com.eonmux.neuralfree.gguf.engine.kvevents.KvEventBatch;
import com.eonmux.neuralfree.gguf.engine.kvevents.KvEventPublishers;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Test;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;
import java.util.stream.Stream;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The disk tier of the prefix cache ({@code neuralfree.cache.disk.mib} and {@code neuralfree.cache.disk.dir},
 * vLLM's tiering filesystem tier): the blocks a one-MiB host tier cannot keep move on to the disk as files, and the
 * prompt that owns them comes back whole from there — in the same engine and, through the scan the next engine
 * makes of the directory, across engines. A slot whose blocks sit on the disk still saves them, and a spill
 * publishes its storage-medium events. The properties parse and refuse a size below -1, a tier without its
 * directory, a directory beside an off tier and a disk tier beside an off host tier, all by name.
 */
public class BatchDiskCacheTest {

    private static final int PAGE = 256;
    /** Three pages per group: one prompt's block and partial, and a two-block prompt that must evict them. */
    private static final int N_CTX = 3 * PAGE;
    private static final int N_NEW = 5;

    /**
     * A paged CPU memory of three pages of 256 rows with a one-MiB host tier: one whole block of this model's two
     * F32 cache layers of width 512 holds two MiB, so the host tier keeps none of it and every evicted whole block
     * moves on to the disk.
     */
    private static EngineConfig config(Path diskDir, int parallel) {
        return EngineConfig.builder().backend(EngineConfig.BackendMode.CPU)
                .memory(MemoryConfig.builder().nCtx(N_CTX).ubatch(PAGE).kvPaged(true).parallel(parallel)
                        .kvPageSize(PAGE).typeK(KvDType.F32).typeV(KvDType.F32)
                        .cacheRamMib(1).cacheDiskMib(-1).cacheDiskDir(diskDir.toString()).build())
                .build();
    }

    /** The model whose whole KV block of one page of rows exceeds the host tier's one MiB. */
    private static Path model() throws IOException {
        return SyntheticModels.llama(SyntheticModels.Config.llama().toBuilder()
                .nEmbd(512).nHead(8).nHeadKv(8).headDim(64).nRot(64).build()).writeTempFile();
    }

    private static GenerationRequest request() {
        return GenerationRequest.builder().maxNewTokens(N_NEW).sampler(SamplerConfig.greedyDecoding()).build();
    }

    /** {@code n} distinct prompt tokens of whole page blocks. */
    private static int[] prompt(int seed, int n) {
        int[] t = new int[n];
        for (int i = 0; i < n; i++) {
            t[i] = 4 + (seed * 31 + i * 37) % 250;
        }
        return t;
    }

    private static BatchSequence run(BatchEngine engine, int[] prompt) throws InterruptedException {
        BatchSequence seq = engine.submit(prompt, request(), (TokenCallback) null);
        engine.drain();
        assertEquals(BatchSequence.State.DONE, seq.state());
        return seq;
    }

    private static long countFiles(Path dir) throws IOException {
        try (Stream<Path> files = Files.walk(dir)) {
            return files.filter(f -> f.getFileName().toString().endsWith(".bin")).count();
        }
    }

    @Test
    public void evictedBlocksComeBackFromDisk() throws Exception {
        Path model = model();
        Path dir = Files.createTempDirectory("nf-disk-engine-");
        EngineConfig config = config(dir, 1);
        int[] a = prompt(1, PAGE);
        int[] b = prompt(2, 2 * PAGE);
        BatchSequence first;
        BatchSequence again;
        try (GGUFModel m = new GGUFModel(model.toString(), config);
             BatchEngine engine = BatchEngine.forModel(m, config)) {
            first = run(engine, a);
            run(engine, b);
            assertTrue("the blocks the one-MiB host tier cannot keep sit on the disk", countFiles(dir) > 0);
            again = run(engine, a);
        }
        assertEquals("the disk tier serves the whole prompt again (the last token always prefills)", PAGE - 1,
                again.reusedPrefix());
        assertArrayEquals("the promoted rows decode exactly like the first run", first.result().tokens(),
                again.result().tokens());
    }

    @Test
    public void aLaterEngineFindsTheBlocksTheEarlierOneLeft() throws Exception {
        Path model = model();
        Path dir = Files.createTempDirectory("nf-disk-engine-");
        EngineConfig config = config(dir, 1);
        int[] a = prompt(1, PAGE);
        int[] expected;
        try (GGUFModel m = new GGUFModel(model.toString(), config);
             BatchEngine engine = BatchEngine.forModel(m, config)) {
            expected = run(engine, a).result().tokens();
            run(engine, prompt(2, 2 * PAGE));
        }
        assertTrue(countFiles(dir) > 0);
        try (GGUFModel m = new GGUFModel(model.toString(), config);
             BatchEngine engine = BatchEngine.forModel(m, config)) {
            BatchSequence again = run(engine, a);
            assertEquals("the scan of the directory serves the earlier engine's blocks", PAGE - 1,
                    again.reusedPrefix());
            assertArrayEquals(expected, again.result().tokens());
        }
    }

    @Test
    public void aSlotWhoseBlocksSitOnTheDiskStillSavesThem() throws Exception {
        Path model = model();
        Path dir = Files.createTempDirectory("nf-disk-engine-");
        Path slots = Files.createTempDirectory("nf-disk-slots-");
        Path saved = slots.resolve("a.slot");
        Path fromDisk = slots.resolve("a-from-disk.slot");
        int[] a = prompt(1, PAGE);
        EngineConfig one = config(dir, 1);
        try (GGUFModel m = new GGUFModel(model.toString(), one);
             BatchEngine engine = BatchEngine.forModel(m, one)) {
            run(engine, a);
            engine.saveSlot(0, saved);
        }
        EngineConfig two = config(dir, 2);
        try (GGUFModel m = new GGUFModel(model.toString(), two);
             BatchEngine engine = BatchEngine.forModel(m, two)) {
            engine.restoreSlot(1, saved);
            // B runs on slot 0 and evicts slot 1's restored blocks into the host tier and onto the disk
            run(engine, prompt(2, 2 * PAGE));
            assertTrue(countFiles(dir) > 0);
            SlotResult again = engine.saveSlot(1, fromDisk);
            assertEquals(PAGE + N_NEW - 1, again.tokens());
            assertArrayEquals("the slot saves the same rows, whether in the pools, in host RAM or on the disk",
                    Files.readAllBytes(saved), Files.readAllBytes(fromDisk));
        }
    }

    @Test
    public void spillsPublishStorageMediumEvents() throws Exception {
        Path model = model();
        Path dir = Files.createTempDirectory("nf-disk-engine-");
        EngineConfig config = config(dir, 1);
        List<KvCacheEvent> events = new ArrayList<>();
        Consumer<KvEventBatch> listener = batch -> events.addAll(batch.events());
        try (GGUFModel m = new GGUFModel(model.toString(), config);
             BatchEngine engine = BatchEngine.forModel(m, config, KvEventPublishers.listener(listener))) {
            run(engine, prompt(1, PAGE));
            run(engine, prompt(2, 2 * PAGE));
        }
        long storedOnDisk = events.stream()
                .filter(e -> e instanceof KvCacheEvent.BlockStored
                        && KvCacheEvent.MEDIUM_STORAGE.equals(((KvCacheEvent.BlockStored) e).medium()))
                .count();
        assertTrue("the spill publishes a block stored on the storage medium", storedOnDisk > 0);
    }

    @Test
    public void aBlockPromotedFromDiskKeepsItsVllmDigest() throws Exception {
        Path model = model();
        Path dir = Files.createTempDirectory("nf-disk-engine-");
        EngineConfig config = config(dir, 1);
        int[] a = prompt(1, PAGE);
        List<KvCacheEvent> events = new ArrayList<>();
        Consumer<KvEventBatch> listener = batch -> events.addAll(batch.events());
        try (GGUFModel m = new GGUFModel(model.toString(), config);
             BatchEngine engine = BatchEngine.forModel(m, config, KvEventPublishers.listener(listener))) {
            run(engine, a);
            run(engine, prompt(2, 2 * PAGE));
            int stored = countStores(events);
            assertTrue("a's whole block left through the host tier onto the disk", countFiles(dir) > 0);
            run(engine, a);
            assertTrue("the promotion stores the block again on the engine's medium",
                    countStores(events) > stored);
        }
        // vLLM's hash_block_tokens digest of a's first whole block, wherever it was stored
        byte[] digest = null;
        int media = 0;
        for (KvCacheEvent e : events) {
            if (!(e instanceof KvCacheEvent.BlockStored)) {
                continue;
            }
            KvCacheEvent.BlockStored s = (KvCacheEvent.BlockStored) e;
            if (!startsWith(s.tokenIds(), a)) {
                continue;
            }
            assertTrue(KvCacheEvent.MEDIUM_GPU.equals(s.medium()) || KvCacheEvent.MEDIUM_CPU.equals(s.medium())
                    || KvCacheEvent.MEDIUM_STORAGE.equals(s.medium()));
            if (KvCacheEvent.MEDIUM_STORAGE.equals(s.medium())) {
                continue;
            }
            media++;
            if (digest == null) {
                digest = s.blockHashes()[0];
            }
            assertArrayEquals("every store of the block keeps the digest a router recomputes from its tokens",
                    digest, s.blockHashes()[0]);
            assertNull("a chain's first whole block has no parent", s.parentBlockHash());
        }
        assertTrue("the block was stored on a compute medium before and after the disk round trip", media >= 2);
    }

    /** Whether {@code tokens} begins with every element of {@code want}. */
    private static boolean startsWith(int[] tokens, int[] want) {
        if (tokens.length < want.length) {
            return false;
        }
        for (int i = 0; i < want.length; i++) {
            if (tokens[i] != want[i]) {
                return false;
            }
        }
        return true;
    }

    private static int countStores(List<KvCacheEvent> events) {
        int n = 0;
        for (KvCacheEvent e : events) {
            if (e instanceof KvCacheEvent.BlockStored) {
                n++;
            }
        }
        return n;
    }

    @Test
    public void theEngineStepDoesNotWaitForDiskWrites() throws Exception {
        Path model = model();
        Path dir = Files.createTempDirectory("nf-disk-engine-");
        EngineConfig config = config(dir, 1);
        CountDownLatch entered = new CountDownLatch(1);
        CountDownLatch release = new CountDownLatch(1);
        Set<String> writeThreads = ConcurrentHashMap.newKeySet();
        DiskBlockCache.Io slow = new RealIo() {
            @Override
            public void write(byte[] blob, Path file) throws IOException {
                writeThreads.add(Thread.currentThread().getName());
                entered.countDown();
                awaitLatch(release);
                super.write(blob, file);
            }
        };
        try (GGUFModel m = new GGUFModel(model.toString(), config);
             BatchEngine engine = BatchEngine.forModel(m, config, KvEventPublishers.listener(batch -> { }), slow)) {
            BatchSequence a = engine.submit(prompt(1, PAGE), request(), (TokenCallback) null);
            BatchSequence b = engine.submit(prompt(2, 2 * PAGE), request(), (TokenCallback) null);
            long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(60);
            while ((a.state() != BatchSequence.State.DONE || b.state() != BatchSequence.State.DONE)
                    && System.nanoTime() < deadline) {
                engine.pump();
            }
            assertEquals("the first prompt finished while the disk write was held",
                    BatchSequence.State.DONE, a.state());
            assertEquals("the second prompt finished while the disk write was held",
                    BatchSequence.State.DONE, b.state());
            assertTrue("the eviction reached the writer", entered.await(5, TimeUnit.SECONDS));
            assertEquals("the held write left no file behind", 0, countFiles(dir));
            assertFalse("the writer runs on its own thread",
                    writeThreads.contains(Thread.currentThread().getName()));
            release.countDown();
        } finally {
            release.countDown();
        }
    }

    @Test
    public void aQueuedPromptWaitsForItsDiskBlocksOnAWorkerThread() throws Exception {
        Path model = model();
        Path dir = Files.createTempDirectory("nf-disk-engine-");
        EngineConfig config = config(dir, 1);
        int[] a = prompt(1, PAGE);
        int[] expected;
        try (GGUFModel m = new GGUFModel(model.toString(), config);
             BatchEngine engine = BatchEngine.forModel(m, config)) {
            expected = run(engine, a).result().tokens();
            run(engine, prompt(2, 2 * PAGE));
        }
        assertTrue(countFiles(dir) > 0);
        Set<String> readThreads = ConcurrentHashMap.newKeySet();
        DiskBlockCache.Io slow = new RealIo() {
            @Override
            public byte[] read(Path file) throws IOException {
                readThreads.add(Thread.currentThread().getName());
                sleep(100);
                return super.read(file);
            }
        };
        try (GGUFModel m = new GGUFModel(model.toString(), config);
             BatchEngine engine = BatchEngine.forModel(m, config, KvEventPublishers.listener(batch -> { }), slow)) {
            BatchSequence again = engine.submit(a, request(), (TokenCallback) null);
            engine.drain();
            assertEquals("the admission waits for the read and then reuses the blocks", PAGE - 1,
                    again.reusedPrefix());
            assertArrayEquals("the promoted rows decode exactly like the first run", expected,
                    again.result().tokens());
        }
        assertFalse("the disk tier read the block", readThreads.isEmpty());
        assertFalse("the read runs off the engine's thread",
                readThreads.contains(Thread.currentThread().getName()));
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

    private static void awaitLatch(CountDownLatch latch) {
        try {
            latch.await(10, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    private static void sleep(long millis) {
        try {
            Thread.sleep(millis);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    @Test
    public void thePropertiesParseAndRefuseByName() throws Exception {
        Properties p = new Properties();
        p.setProperty(MemoryConfig.PROP_CACHE_RAM, "64");
        p.setProperty(MemoryConfig.PROP_CACHE_DISK_MIB, "512");
        p.setProperty(MemoryConfig.PROP_CACHE_DISK_DIR, "/tmp/nf-cache");
        MemoryConfig parsed = MemoryConfig.fromProperties(p);
        assertEquals(512, parsed.cacheDiskMib());
        assertEquals("/tmp/nf-cache", parsed.cacheDiskDir());
        assertEquals(64, parsed.cacheRamMib());

        p.setProperty(MemoryConfig.PROP_CACHE_DISK_MIB, "-2");
        IllegalArgumentException badSize = assertThrows(IllegalArgumentException.class,
                () -> MemoryConfig.fromProperties(p));
        assertTrue(badSize.getMessage(), badSize.getMessage().contains(MemoryConfig.PROP_CACHE_DISK_MIB));

        p.setProperty(MemoryConfig.PROP_CACHE_DISK_MIB, "512");
        p.remove(MemoryConfig.PROP_CACHE_DISK_DIR);
        IllegalArgumentException noDir = assertThrows(IllegalArgumentException.class,
                () -> MemoryConfig.fromProperties(p));
        assertTrue(noDir.getMessage(), noDir.getMessage().contains(MemoryConfig.PROP_CACHE_DISK_DIR));

        p.setProperty(MemoryConfig.PROP_CACHE_DISK_DIR, "/tmp/nf-cache");
        p.setProperty(MemoryConfig.PROP_CACHE_DISK_MIB, "0");
        IllegalArgumentException noTier = assertThrows(IllegalArgumentException.class,
                () -> MemoryConfig.fromProperties(p));
        assertTrue(noTier.getMessage(), noTier.getMessage().contains(MemoryConfig.PROP_CACHE_DISK_MIB));

        p.setProperty(MemoryConfig.PROP_CACHE_DISK_MIB, "512");
        p.remove(MemoryConfig.PROP_CACHE_RAM);
        IllegalArgumentException noRam = assertThrows(IllegalArgumentException.class,
                () -> MemoryConfig.fromProperties(p));
        assertTrue(noRam.getMessage(), noRam.getMessage().contains(MemoryConfig.PROP_CACHE_DISK_MIB));
        assertTrue(noRam.getMessage(), noRam.getMessage().contains(MemoryConfig.PROP_CACHE_RAM));

        Path model = model();
        EngineConfig noCache = EngineConfig.builder().backend(EngineConfig.BackendMode.CPU)
                .memory(MemoryConfig.builder().nCtx(N_CTX).ubatch(PAGE).kvPaged(true)
                        .typeK(KvDType.F32).typeV(KvDType.F32).prefixReuse(false)
                        .cacheRamMib(1).cacheDiskMib(-1).cacheDiskDir(
                                Files.createTempDirectory("nf-disk-engine-").toString()).build())
                .build();
        try (GGUFModel m = new GGUFModel(model.toString(), noCache)) {
            UnsupportedOperationException off = assertThrows(UnsupportedOperationException.class,
                    () -> BatchEngine.forModel(m, noCache));
            assertTrue(off.getMessage(), off.getMessage().contains(MemoryConfig.PROP_CACHE_RAM));
            assertTrue(off.getMessage(), off.getMessage().contains(MemoryConfig.PROP_PREFIX_REUSE));
        }

        EngineConfig unusableDir = EngineConfig.builder().backend(EngineConfig.BackendMode.CPU)
                .memory(MemoryConfig.builder().nCtx(N_CTX).ubatch(PAGE).kvPaged(true)
                        .typeK(KvDType.F32).typeV(KvDType.F32)
                        .cacheRamMib(1).cacheDiskMib(-1).cacheDiskDir(
                                model.resolve("no-such-parent").resolve("cache").toString()).build())
                .build();
        try (GGUFModel m = new GGUFModel(model.toString(), unusableDir)) {
            assertThrows(UncheckedIOException.class, () -> BatchEngine.forModel(m, unusableDir));
        }
    }
}

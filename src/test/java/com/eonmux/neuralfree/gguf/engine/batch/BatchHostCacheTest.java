package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.TokenCallback;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Properties;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The host-RAM tier of the prefix cache ({@code neuralfree.cache.ram}, llama.cpp's {@code --cache-ram}) over a
 * paged CPU memory too small for two prompts: the blocks a second prompt evicts move to the host, and the first
 * prompt's rerun brings them back and reuses its whole prefix with the same tokens, where the engine without the
 * tier reuses only what stayed in the pools. A slot whose blocks moved to the host still saves them. The property
 * parses and refuses a size below -1, an engine without a prefix cache refuses it by name, and a memory that
 * restores a cached prefix only at a context checkpoint refuses it and its slot actions by name.
 */
public class BatchHostCacheTest {

    private static final int N_NEW = 5;

    private static EngineConfig config(int cacheRamMib) {
        return config(cacheRamMib, 1);
    }

    /** A paged CPU memory of 32 rows in pages of 4, shared by {@code parallel} slots. */
    private static EngineConfig config(int cacheRamMib, int parallel) {
        return EngineConfig.builder().backend(EngineConfig.BackendMode.CPU)
                .memory(MemoryConfig.builder().nCtx(32).ubatch(8).kvPaged(true).parallel(parallel).kvPageSize(4)
                        .typeK(KvDType.F32).typeV(KvDType.F32).cacheRamMib(cacheRamMib).build())
                .build();
    }

    private static GenerationRequest request() {
        return GenerationRequest.builder().maxNewTokens(N_NEW).sampler(SamplerConfig.greedyDecoding()).build();
    }

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

    /** Runs A, then B (which evicts most of A), then A again; answers A's two runs. */
    private static BatchSequence[] evictAndRerun(Path model, int cacheRamMib) throws Exception {
        EngineConfig config = config(cacheRamMib);
        try (GGUFModel m = new GGUFModel(model.toString(), config);
             BatchEngine engine = BatchEngine.forModel(m, config)) {
            BatchSequence first = run(engine, prompt(1, 14));
            run(engine, prompt(2, 20));
            BatchSequence again = run(engine, prompt(1, 14));
            return new BatchSequence[] {first, again};
        }
    }

    @Test
    public void evictedBlocksComeBackFromHostRam() throws Exception {
        Path model = SyntheticModels.llama().writeTempFile();
        BatchSequence[] withTier = evictAndRerun(model, -1);
        BatchSequence[] withoutTier = evictAndRerun(model, 0);
        assertEquals("the host tier serves the whole prefix again (the last prompt token always prefills)", 13,
                withTier[1].reusedPrefix());
        assertTrue("without the tier the evicted blocks are lost, reused " + withoutTier[1].reusedPrefix(),
                withoutTier[1].reusedPrefix() < 13);
        assertArrayEquals("the promoted rows decode exactly like the first run", withTier[0].result().tokens(),
                withTier[1].result().tokens());
        assertArrayEquals(withoutTier[0].result().tokens(), withTier[1].result().tokens());
    }

    @Test
    public void aSlotWhoseBlocksMovedToTheHostStillSavesThem() throws Exception {
        Path model = SyntheticModels.llama().writeTempFile();
        Path dir = Files.createTempDirectory("nf-host-slot-");
        Path saved = dir.resolve("a.slot");
        Path fromHost = dir.resolve("a-from-host.slot");
        int[] a = prompt(1, 14);
        int[] expected;
        try (GGUFModel m = new GGUFModel(model.toString(), config(-1, 1));
             BatchEngine engine = BatchEngine.forModel(m, config(-1, 1))) {
            expected = run(engine, a).result().tokens();
            engine.saveSlot(0, saved);
        }
        EngineConfig twoSlots = config(-1, 2);
        try (GGUFModel m = new GGUFModel(model.toString(), twoSlots);
             BatchEngine engine = BatchEngine.forModel(m, twoSlots)) {
            engine.restoreSlot(1, saved);
            // B runs on slot 0 and evicts most of slot 1's restored blocks into the host tier
            run(engine, prompt(2, 20));
            SlotResult again = engine.saveSlot(1, fromHost);
            assertEquals(a.length + N_NEW - 1, again.tokens());
            assertArrayEquals("slot 1 saves the same rows, whether in the pools or in host RAM",
                    Files.readAllBytes(saved), Files.readAllBytes(fromHost));
            BatchSequence adopted = run(engine, a);
            assertEquals("the host tier serves the restored prefix", a.length - 1, adopted.reusedPrefix());
            assertArrayEquals(expected, adopted.result().tokens());
        }
    }

    @Test
    public void theSizeParsesAndRefusesByName() throws Exception {
        Properties p = new Properties();
        p.setProperty(MemoryConfig.PROP_CACHE_RAM, "512");
        assertEquals(512, MemoryConfig.fromProperties(p).cacheRamMib());
        p.setProperty(MemoryConfig.PROP_CACHE_RAM, "-2");
        IllegalArgumentException bad = assertThrows(IllegalArgumentException.class,
                () -> MemoryConfig.fromProperties(p));
        assertTrue(bad.getMessage(), bad.getMessage().contains(MemoryConfig.PROP_CACHE_RAM));
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig noCache = EngineConfig.builder().backend(EngineConfig.BackendMode.CPU)
                .memory(MemoryConfig.builder().nCtx(32).ubatch(8).kvPaged(true).prefixReuse(false)
                        .cacheRamMib(64).build())
                .build();
        try (GGUFModel m = new GGUFModel(model.toString(), noCache)) {
            UnsupportedOperationException off = assertThrows(UnsupportedOperationException.class,
                    () -> BatchEngine.forModel(m, noCache));
            assertTrue(off.getMessage(), off.getMessage().contains(MemoryConfig.PROP_CACHE_RAM));
            assertTrue(off.getMessage(), off.getMessage().contains(MemoryConfig.PROP_PREFIX_REUSE));
        }
    }

    /** gemma2 with an 8-row window: a memory whose cached prefixes restore only at a context checkpoint. */
    private static Path gemma2() throws IOException {
        Path path = Files.createTempFile("gemma2-", ".gguf");
        path.toFile().deleteOnExit();
        GgufTestWriter w = SyntheticModels.gemma2(
                SyntheticModels.Config.gemma2().toBuilder().slidingWindow(8).build());
        w.writeTo(path);
        return path;
    }

    /** A paged CPU memory of 128 rows in pages of 4 with the host tier on. */
    private static EngineConfig checkpointedConfig() {
        return EngineConfig.builder().backend(EngineConfig.BackendMode.CPU)
                .memory(MemoryConfig.builder().nCtx(128).ubatch(8).kvPaged(true).kvPageSize(4).parallel(1)
                        .typeK(KvDType.F32).typeV(KvDType.F32).cacheRamMib(64).build())
                .build();
    }

    @Test
    public void refusesOverACheckpointedMemoryByName() throws Exception {
        Path model = gemma2();
        EngineConfig config = checkpointedConfig();
        try (GGUFModel m = new GGUFModel(model.toString(), config)) {
            UnsupportedOperationException built = assertThrows(UnsupportedOperationException.class,
                    () -> BatchEngine.forModel(m, config));
            assertTrue(built.getMessage(), built.getMessage().contains(MemoryConfig.PROP_CACHE_RAM));
            assertTrue(built.getMessage(), built.getMessage().contains(MemoryConfig.PROP_CTX_CHECKPOINTS));
        }
        EngineConfig noTier = EngineConfig.builder().backend(EngineConfig.BackendMode.CPU)
                .memory(MemoryConfig.builder().nCtx(128).ubatch(8).kvPaged(true).kvPageSize(4).parallel(1)
                        .typeK(KvDType.F32).typeV(KvDType.F32).build())
                .build();
        try (GGUFModel m = new GGUFModel(model.toString(), noTier);
             BatchEngine engine = BatchEngine.forModel(m, noTier)) {
            UnsupportedOperationException save = assertThrows(UnsupportedOperationException.class,
                    () -> engine.saveSlot(0, Files.createTempFile("gemma2-slot-", ".state")));
            assertTrue(save.getMessage(), save.getMessage().contains(MemoryConfig.PROP_CTX_CHECKPOINTS));
            assertThrows(UnsupportedOperationException.class, () -> engine.restoreSlot(0, Path.of("x")));
            assertThrows(UnsupportedOperationException.class, () -> engine.eraseSlot(0));
        }
    }
}

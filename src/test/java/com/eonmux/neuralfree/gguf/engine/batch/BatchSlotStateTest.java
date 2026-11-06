package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.TokenCallback;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The slot actions of the batch engine (llama.cpp's {@code /slots/{id}?action=save|restore|erase}) over a paged CPU
 * memory: a saved slot carries the history its last sequence left in the prefix cache, restores into a fresh engine
 * whose next request with that prompt adopts the restored rows and generates exactly what the original engine did,
 * and saves back byte for byte — for an F32 and a BF16 cache alike. Erase forgets the slot and frees its
 * blocks; a running slot, an id outside the slots, an engine without a prefix cache and a file of another layout
 * refuse by name.
 */
public class BatchSlotStateTest {

    private static final int PAGE = 4;
    private static final int N_NEW = 5;

    private static EngineConfig config(KvDType type, boolean reuse, int pageSize) {
        return EngineConfig.builder().backend(EngineConfig.BackendMode.CPU)
                .memory(MemoryConfig.builder().nCtx(64).ubatch(8).parallel(2).kvPageSize(pageSize)
                        .typeK(type).typeV(type).prefixReuse(reuse).build())
                .build();
    }

    private static GenerationRequest request() {
        return GenerationRequest.builder().maxNewTokens(N_NEW).sampler(SamplerConfig.greedyDecoding()).build();
    }

    /** Fourteen prompt tokens: three whole blocks of four, then a partial block. */
    private static int[] prompt() {
        int[] t = new int[14];
        for (int i = 0; i < t.length; i++) {
            t[i] = 4 + (i * 37) % 250;
        }
        return t;
    }

    private static BatchSequence run(BatchEngine engine, int[] prompt) throws InterruptedException {
        BatchSequence seq = engine.submit(prompt, request(), (TokenCallback) null);
        engine.drain();
        assertEquals(BatchSequence.State.DONE, seq.state());
        return seq;
    }

    @Test
    public void aSavedSlotRestoresIntoAFreshEngineWhoseNextRequestAdoptsIt() throws Exception {
        assertRoundTrip(KvDType.F32);
    }

    @Test
    public void aHalfPrecisionCacheRoundTripsItsStoredBits() throws Exception {
        assertRoundTrip(KvDType.BF16);
    }

    private static void assertRoundTrip(KvDType type) throws Exception {
        Path model = SyntheticModels.llama().writeTempFile();
        Path dir = Files.createTempDirectory("nf-slot-");
        Path saved = dir.resolve("slot0.bin");
        Path again = dir.resolve("slot0-again.bin");
        EngineConfig config = config(type, true, PAGE);
        int[] prompt = prompt();
        int[] expected;
        SlotResult save;
        try (GGUFModel m = new GGUFModel(model.toString(), config);
             BatchEngine engine = BatchEngine.forModel(m, config)) {
            expected = run(engine, prompt).result().tokens();
            save = engine.saveSlot(0, saved);
        }
        // the history is the prompt plus every fed decode (the last sampled token never fed)
        assertEquals(prompt.length + N_NEW - 1, save.tokens());
        assertEquals(Files.size(saved), save.bytes());
        try (GGUFModel m = new GGUFModel(model.toString(), config);
             BatchEngine engine = BatchEngine.forModel(m, config)) {
            SlotResult restore = engine.restoreSlot(1, saved);
            assertEquals(save.tokens(), restore.tokens());
            assertEquals(save.bytes(), restore.bytes());
            engine.saveSlot(1, again);
            assertArrayEquals("the restored rows save back bit for bit", Files.readAllBytes(saved),
                    Files.readAllBytes(again));
            BatchSequence adopted = run(engine, prompt);
            assertEquals("the restored blocks serve the prompt (its last token always prefills)",
                    prompt.length - 1, adopted.reusedPrefix());
            assertArrayEquals("the restored rows decode exactly like the original run", expected,
                    adopted.result().tokens());
        }
    }

    @Test
    public void eraseForgetsTheSlotAndFreesItsBlocks() throws Exception {
        Path model = SyntheticModels.llama().writeTempFile();
        Path file = Files.createTempDirectory("nf-slot-").resolve("erased.bin");
        EngineConfig config = config(KvDType.F32, true, PAGE);
        try (GGUFModel m = new GGUFModel(model.toString(), config);
             BatchEngine engine = BatchEngine.forModel(m, config)) {
            run(engine, prompt());
            assertTrue("the retired sequence's blocks stay cached", engine.stats().kvUsage() > 0);
            assertEquals(prompt().length + N_NEW - 1, engine.eraseSlot(0));
            assertEquals("the erased slot's blocks returned to the pools", 0.0, engine.stats().kvUsage(), 0.0);
            assertEquals("an erased slot holds nothing", 0, engine.eraseSlot(0));
            assertEquals(0, engine.saveSlot(0, file).tokens());
            assertEquals("nothing is left to adopt", 0, run(engine, prompt()).reusedPrefix());
        }
    }

    @Test
    public void slotActionsRefuseByName() throws Exception {
        Path model = SyntheticModels.llama().writeTempFile();
        Path dir = Files.createTempDirectory("nf-slot-");
        Path file = dir.resolve("slot.bin");
        EngineConfig config = config(KvDType.F32, true, PAGE);
        try (GGUFModel m = new GGUFModel(model.toString(), config);
             BatchEngine engine = BatchEngine.forModel(m, config)) {
            IllegalArgumentException outside = assertThrows(IllegalArgumentException.class,
                    () -> engine.saveSlot(2, file));
            assertTrue(outside.getMessage(), outside.getMessage().contains("slot 2"));
            engine.submit(prompt(), request(), (TokenCallback) null);
            engine.pump();
            IllegalStateException busy = assertThrows(IllegalStateException.class, () -> engine.saveSlot(0, file));
            assertTrue(busy.getMessage(), busy.getMessage().contains("processing"));
            engine.drain();
            engine.saveSlot(0, file);
        }
        EngineConfig noCache = config(KvDType.F32, false, PAGE);
        try (GGUFModel m = new GGUFModel(model.toString(), noCache);
             BatchEngine engine = BatchEngine.forModel(m, noCache)) {
            UnsupportedOperationException off = assertThrows(UnsupportedOperationException.class,
                    () -> engine.saveSlot(0, file));
            assertTrue(off.getMessage(), off.getMessage().contains(MemoryConfig.PROP_PREFIX_REUSE));
        }
        EngineConfig otherPages = config(KvDType.F32, true, 8);
        try (GGUFModel m = new GGUFModel(model.toString(), otherPages);
             BatchEngine engine = BatchEngine.forModel(m, otherPages)) {
            IOException mismatch = assertThrows(IOException.class, () -> engine.restoreSlot(0, file));
            assertTrue(mismatch.getMessage(), mismatch.getMessage().contains("page size"));
            Path foreign = Files.write(dir.resolve("foreign.bin"), new byte[] {1, 2, 3, 4, 5, 6, 7, 8});
            IOException notSlot = assertThrows(IOException.class, () -> engine.restoreSlot(0, foreign));
            assertTrue(notSlot.getMessage(), notSlot.getMessage().contains("not a NeuralFree slot file"));
        }
        EngineConfig half = config(KvDType.F16, true, PAGE);
        try (GGUFModel m = new GGUFModel(model.toString(), half);
             BatchEngine engine = BatchEngine.forModel(m, half)) {
            IOException dtype = assertThrows(IOException.class, () -> engine.restoreSlot(0, file));
            assertTrue(dtype.getMessage(), dtype.getMessage().contains("K dtype"));
        }
    }
}

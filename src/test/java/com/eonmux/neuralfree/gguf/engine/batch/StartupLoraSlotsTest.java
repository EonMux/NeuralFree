package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.TokenCallback;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.GenerationLoop;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.engine.lora.LoraConfig;
import com.eonmux.neuralfree.gguf.engine.lora.LoraHandle;
import com.eonmux.neuralfree.gguf.engine.lora.LoraRegistry;
import com.eonmux.neuralfree.gguf.engine.lora.LoraSelection;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.LoraAdapterFiles;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Startup adapters ({@code --lora}) in the per-request registry (llama.cpp server's {@code lora_adapters}): they are
 * resident first under names from their paths and run slot 0 at their startup scale for a request that selects
 * none; a request's selection replaces the startup set (unlisted adapters at 0), so it rescales or disables them per
 * request, through the batch engine and the single-sequence loop alike, each matching a model loaded with that set
 * alone. They are neither unloaded nor rescaled globally.
 */
public class StartupLoraSlotsTest {

    private static final int RANK = 4;
    private static final float ALPHA = 8.0f;
    private static final float SCALE_A = 0.75f;
    private static final float SCALE_B = 0.5f;
    private static final int N_NEW = 6;
    private static final TokenCallback NONE = null;

    private static EngineConfig.Builder config(int parallel) {
        MemoryConfig.Builder mc = MemoryConfig.builder().nCtx(64).ubatch(8).typeK(KvDType.F32).typeV(KvDType.F32)
                .prefixReuse(true);
        if (parallel > 1) {
            mc = mc.parallel(parallel);
        }
        return EngineConfig.builder().memory(mc.build())
                .loraConfig(LoraConfig.builder().maxAdapters(2).maxRank(64).cacheEntries(2).build());
    }

    private static GenerationRequest request(LoraSelection lora) {
        return GenerationRequest.builder().maxNewTokens(N_NEW).sampler(SamplerConfig.greedyDecoding())
                .lora(lora).build();
    }

    private static int[] tokens(int seed) {
        int[] t = new int[10];
        for (int i = 0; i < t.length; i++) {
            t[i] = 4 + (seed * 7 + i * 13) % 250;
        }
        return t;
    }

    /** The prompt through the single-sequence loop of a model loaded with {@code adapter} alone (null: none). */
    private static GenerationResult reference(Path model, Path adapter, float scale, int[] prompt)
            throws IOException {
        EngineConfig.Builder b = config(1);
        if (adapter != null) {
            b.addLora(adapter.toString(), scale);
        }
        EngineConfig cfg = b.build();
        try (GGUFModel one = new GGUFModel(model.toString(), cfg)) {
            return GenerationLoop.forModel(one, cfg).generate(prompt, request(null), NONE);
        }
    }

    private static LoraSelection only(String name, float scale) {
        return LoraSelection.builder().add(name, scale).build();
    }

    @Test
    public void theStartupAdapterIsResidentFirstAndRunsSlotZero() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        Path a = LoraAdapterFiles.llama(1000, RANK, ALPHA);
        EngineConfig cfg = config(2).addLora(a.toString(), SCALE_A).build();
        try (GGUFModel m = new GGUFModel(model.toString(), cfg)) {
            LoraRegistry registry = m.loraRegistry();
            List<LoraHandle> handles = registry.list();
            assertEquals(1, handles.size());
            LoraHandle startup = handles.get(0);
            assertTrue(startup.startup());
            assertEquals(SCALE_A, startup.scale(), 0f);
            String name = startup.name();
            assertFalse("the name drops the .gguf suffix", name.endsWith(".gguf"));
            assertEquals("the startup set is slot 0's key", name, registry.defaultKey().label());
            assertNull("a selection equal to the startup set runs slot 0",
                    registry.resolve(LoraSelection.of(name)));
            assertTrue("a selection disabling it runs the bare key",
                    registry.resolve(only(name, 0f)).isBare());
            assertEquals("the bare key runs past the slot table", registry.slotCount() + 1, registry.bareSlot());

            assertTrue("a selection at another scale of the startup set is a rescale key",
                    registry.isRescale(registry.resolve(only(name, 0.25f))));
            registry.load("b", LoraAdapterFiles.llama(5000, RANK, ALPHA).toString(), SCALE_B);
            assertFalse("a key of another adapter is no rescale", registry.isRescale(registry.resolve(
                    LoraSelection.of("b"))));
            assertFalse("the bare key takes no slot to rescale",
                    registry.isRescale(registry.resolve(only(name, 0f))));
            assertEquals("startup adapters stay first", name, registry.list().get(0).name());
            IllegalStateException unload = assertThrows(IllegalStateException.class, () -> registry.unload(name));
            assertTrue(unload.getMessage(), unload.getMessage().contains(name));
            assertThrows(IllegalStateException.class, () -> registry.setScale(name, 0.1f));
        }
    }

    @Test
    public void requestsRescaleDisableOrReplaceTheStartupAdapterInSharedSteps() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        Path a = LoraAdapterFiles.llama(1000, RANK, ALPHA);
        Path b = LoraAdapterFiles.llama(5000, RANK, ALPHA);
        int[] pDefault = tokens(1);
        int[] pBare = tokens(2);
        int[] pQuarter = tokens(3);
        int[] pB = tokens(4);
        GenerationResult refDefault = reference(model, a, SCALE_A, pDefault);
        GenerationResult refBare = reference(model, null, 0f, pBare);
        GenerationResult refQuarter = reference(model, a, 0.25f, pQuarter);
        GenerationResult refB = reference(model, b, SCALE_B, pB);
        assertFalse("the adapter must change the text",
                refDefault.text().equals(reference(model, null, 0f, pDefault).text()));

        EngineConfig cfg = config(4).addLora(a.toString(), SCALE_A).build();
        try (GGUFModel m = new GGUFModel(model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(m, cfg)) {
            LoraRegistry registry = m.loraRegistry();
            String name = registry.list().get(0).name();
            registry.load("b", b.toString(), SCALE_B);
            BatchSequence sDefault = engine.submit(pDefault, request(null), NONE);
            BatchSequence sBare = engine.submit(pBare, request(only(name, 0f)), NONE);
            BatchSequence sQuarter = engine.submit(pQuarter, request(only(name, 0.25f)), NONE);
            BatchSequence sB = engine.submit(pB, request(LoraSelection.of("b")), NONE);
            engine.pump();
            assertEquals("the bare key takes no slot of the table: all four run", 4, engine.running());
            engine.drain();
            assertMatches(refDefault, sDefault);
            assertMatches(refBare, sBare);
            assertMatches(refQuarter, sQuarter);
            assertMatches(refB, sB);
            for (int slot = 1; slot <= registry.slotCount(); slot++) {
                assertEquals("slot " + slot + " is released", 0, registry.slotRequests(slot));
            }
        }
    }

    @Test
    public void theSingleSequenceLoopRunsTheSameSelectionsAndKeepsPrefixesApart() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        Path a = LoraAdapterFiles.llama(1000, RANK, ALPHA);
        int[] prompt = tokens(5);
        GenerationResult refDefault = reference(model, a, SCALE_A, prompt);
        GenerationResult refBare = reference(model, null, 0f, prompt);
        GenerationResult refQuarter = reference(model, a, 0.25f, prompt);

        EngineConfig cfg = config(1).addLora(a.toString(), SCALE_A).build();
        try (GGUFModel m = new GGUFModel(model.toString(), cfg)) {
            String name = m.loraRegistry().list().get(0).name();
            GenerationLoop loop = GenerationLoop.forModel(m, cfg);
            // the same prompt in turn: the loop reuses no row computed under other adapters
            assertMatches(refBare, loop.generate(prompt, request(only(name, 0f)), NONE));
            assertMatches(refDefault, loop.generate(prompt, request(null), NONE));
            assertMatches(refQuarter, loop.generate(prompt, request(only(name, 0.25f)), NONE));
            assertMatches(refQuarter, loop.generate(prompt, request(only(name, 0.25f)), NONE));
            assertMatches(refDefault, loop.generate(prompt, request(LoraSelection.of(name)), NONE));
            for (int slot = 1; slot <= m.loraRegistry().slotCount(); slot++) {
                assertEquals("slot " + slot + " is released", 0, m.loraRegistry().slotRequests(slot));
            }
        }
    }

    private static void assertMatches(GenerationResult reference, BatchSequence sequence) {
        assertEquals("expected DONE but was " + sequence.state()
                + (sequence.error() == null ? "" : ": " + sequence.error()), BatchSequence.State.DONE,
                sequence.state());
        assertMatches(reference, sequence.result());
    }

    private static void assertMatches(GenerationResult reference, GenerationResult result) {
        assertArrayEquals("token ids", reference.tokens(), result.tokens());
        assertEquals("text", reference.text(), result.text());
    }
}

package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.TokenCallback;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.GenerationLoop;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.MediaPrompt;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.engine.kvevents.BlockExtraKey;
import com.eonmux.neuralfree.gguf.engine.batch.BlockHasher;
import com.eonmux.neuralfree.gguf.engine.batch.PrefixHashAlgo;
import com.eonmux.neuralfree.gguf.engine.kvevents.KvCacheEvent;
import com.eonmux.neuralfree.gguf.engine.kvevents.KvEventBatch;
import com.eonmux.neuralfree.gguf.engine.kvevents.KvEventPublishers;
import com.eonmux.neuralfree.gguf.engine.lora.LoraConfig;
import com.eonmux.neuralfree.gguf.engine.lora.LoraRegistry;
import com.eonmux.neuralfree.gguf.engine.lora.LoraSelection;
import com.eonmux.neuralfree.gguf.engine.lora.LoraSlotKey;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.ops.rope.MediaRopePositions;
import com.eonmux.neuralfree.gguf.testutil.LoraAdapterFiles;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Per-request LoRA adapters through the {@link BatchEngine} (vLLM's {@code lora_request} per request, llama.cpp
 * server's per-request {@code lora} list): sequences of different adapters and of the base model run in shared
 * steps and each produces the text a model with its adapter attached to every row ({@code --lora}) produces alone;
 * a request whose adapter key finds every slot busy waits while others admit, and runs once a slot frees; a running
 * request keeps its adapter loaded; the prefix cache's block pool keys a chain's first block by the adapters'
 * content identity beside the cache salt, so it never serves rows computed under other adapters, and its
 * {@code BlockStored} events carry that identity and the adapters' name (vLLM's {@code lora_name}); a media
 * chain under adapters keys its blocks by both the adapters and the item's key; unknown adapters refuse
 * by name on the batch and single-sequence paths.
 */
public class BatchEngineLoraTest {

    private static final int RANK = 4;
    private static final float ALPHA = 8.0f;
    private static final float SCALE_A = 0.75f;
    private static final float SCALE_B = 0.5f;
    private static final int N_NEW = 6;
    /** Rows per page of {@link #pagedConfig()}. */
    private static final int PAGE_ROWS = 4;
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

    /** The prompt through the single-sequence loop of a model with {@code adapter} on every row (null: none). */
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

    @Test
    public void sequencesOfDifferentAdaptersShareStepsAndMatchTheirSingleAdapterRuns() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        Path a = LoraAdapterFiles.llama(1000, RANK, ALPHA);
        Path b = LoraAdapterFiles.llama(5000, RANK, ALPHA);
        GenerationResult refA = reference(model, a, SCALE_A, tokens(1));
        GenerationResult refBase = reference(model, null, 0f, tokens(2));
        GenerationResult refB = reference(model, b, SCALE_B, tokens(3));
        GenerationResult refAQuarter = reference(model, a, 0.25f, tokens(4));
        assertFalse("the adapter must change the text", refA.text().equals(reference(model, null, 0f,
                tokens(1)).text()));

        EngineConfig cfg = config(4).build();
        try (GGUFModel m = new GGUFModel(model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(m, cfg)) {
            LoraRegistry registry = m.loraRegistry();
            registry.load("a", a.toString(), SCALE_A);
            registry.load("b", b.toString(), SCALE_B);
            BatchSequence sa = engine.submit(tokens(1), request(LoraSelection.of("a")), NONE);
            BatchSequence sBase = engine.submit(tokens(2), request(null), NONE);
            BatchSequence sb = engine.submit(tokens(3), request(LoraSelection.of("b")), NONE);
            // a third key: both slots serve running requests, so it waits while the others run
            BatchSequence sQuarter = engine.submit(tokens(4),
                    request(LoraSelection.builder().add("a", 0.25f).build()), NONE);

            engine.pump();
            assertEquals(3, engine.running());
            assertEquals(BatchSequence.State.QUEUED, sQuarter.state());
            IllegalStateException busy = assertThrows(IllegalStateException.class, () -> registry.unload("a"));
            assertTrue(busy.getMessage(), busy.getMessage().contains("running request"));

            engine.drain();
            assertMatches(refA, sa);
            assertMatches(refBase, sBase);
            assertMatches(refB, sb);
            assertMatches(refAQuarter, sQuarter);
            for (int slot = 1; slot <= registry.slotCount(); slot++) {
                assertEquals("slot " + slot + " is released", 0, registry.slotRequests(slot));
            }
            registry.unload("a");
        }
    }

    @Test
    public void thePrefixCacheServesOnlyRowsOfTheSameAdapters() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        Path a = LoraAdapterFiles.llama(1000, RANK, ALPHA);
        int[] prompt = tokens(5);
        GenerationResult refA = reference(model, a, SCALE_A, prompt);

        EngineConfig cfg = config(2).build();
        try (GGUFModel m = new GGUFModel(model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(m, cfg)) {
            m.loraRegistry().load("a", a.toString(), SCALE_A);
            BatchSequence base = engine.submit(prompt, request(null), NONE);
            engine.drain();
            assertEquals(BatchSequence.State.DONE, base.state());

            BatchSequence adapted = engine.submit(prompt, request(LoraSelection.of("a")), NONE);
            engine.drain();
            assertEquals("rows of the base model are not reused under an adapter", 0, adapted.reusedPrefix());
            assertMatches(refA, adapted);

            BatchSequence again = engine.submit(prompt, request(LoraSelection.of("a")), NONE);
            engine.drain();
            assertTrue("rows of the same adapter are reused", again.reusedPrefix() > 0);
            assertMatches(refA, again);
        }
    }

    @Test
    public void theBlockPoolKeysChainsByTheAdaptersBesideTheSaltAndTheEventsCarryThem() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        Path a = LoraAdapterFiles.llama(1000, RANK, ALPHA);
        int[] prompt = tokens(6);
        EngineConfig cfg = pagedConfig().build();
        List<KvEventBatch> batches = new ArrayList<>();
        try (GGUFModel m = new GGUFModel(model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(m, cfg, KvEventPublishers.listener(batches::add))) {
            LoraRegistry registry = m.loraRegistry();
            registry.load("a", a.toString(), SCALE_A);
            registry.load("a-copy", a.toString(), SCALE_A);
            LoraSlotKey key = registry.resolve(LoraSelection.of("a"));
            String identity = key.identity();

            BatchSequence salted = run(engine, prompt, salted(LoraSelection.of("a"), "tenant"));
            List<KvCacheEvent.BlockStored> stored = stores(batches);
            KvCacheEvent.BlockStored head = stored.get(0);
            assertNull("the chain's first block has no parent", head.parentBlockHashInt());
            assertEquals("the adapters' identity, then the salt",
                    Arrays.asList(BlockExtraKey.lora(identity), BlockExtraKey.salt("tenant")),
                    head.extraKeys().get(0));
            assertEquals(KvCacheEvent.hashInt(BlockHasher.forEnvironment(PrefixHashAlgo.parse("sha256", "test"))
                            .hash(null, Arrays.copyOfRange(prompt, 0, PAGE_ROWS),
                                    Arrays.asList(BlockExtraKey.lora(identity), BlockExtraKey.salt("tenant")))),
                    head.blockHashInts()[0]);
            assertNotNull("the adapters carry their event id (vLLM's lora_id)", head.loraId());
            for (KvCacheEvent.BlockStored e : stored) {
                assertEquals("every store of the chain names its adapters", "a", e.loraName());
                for (int k = e == head ? 1 : 0; k < e.extraKeys().size(); k++) {
                    assertNull("later blocks inherit the keys through the chain", e.extraKeys().get(k));
                }
            }

            assertEquals("the base model under the same salt shares no block", 0,
                    run(engine, prompt, salted(null, "tenant")).reusedPrefix());
            BatchSequence unsalted = run(engine, prompt, request(LoraSelection.of("a")));
            assertEquals("the adapters without the salt share no block", 0, unsalted.reusedPrefix());
            assertEquals(Arrays.asList(BlockExtraKey.lora(identity)), lastChainHead(batches).extraKeys().get(0));
            assertEquals("a salt spelling the adapters' identity shares no block with them", 0,
                    run(engine, prompt, salted(null, identity)).reusedPrefix());

            BatchSequence copy = run(engine, prompt, salted(LoraSelection.of("a-copy"), "tenant"));
            assertEquals("the same content under another name shares the chain", prompt.length - 1,
                    copy.reusedPrefix());
            assertArrayEquals(salted.result().tokens(), copy.result().tokens());
            assertArrayEquals(salted.result().tokens(), unsalted.result().tokens());
        }
    }

    @Test
    public void aMediaChainUnderAdaptersKeysItsBlocksByTheAdaptersAndTheItem() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        Path a = LoraAdapterFiles.llama(1000, RANK, ALPHA);
        int width = SyntheticModels.Config.llama().nEmbd();
        int[] head = Arrays.copyOf(tokens(8), 3);
        int[] tail = Arrays.copyOf(tokens(9), 3);
        MediaPrompt image = withItem(head, mediaRows(4, width, 0.2f), tail);
        MediaPrompt otherImage = withItem(head, mediaRows(4, width, 0.6f), tail);
        String hash = image.chunks().get(1).item().id();
        EngineConfig cfg = pagedConfig().build();
        List<KvEventBatch> batches = new ArrayList<>();
        try (GGUFModel m = new GGUFModel(model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(m, cfg, KvEventPublishers.listener(batches::add))) {
            LoraRegistry registry = m.loraRegistry();
            registry.load("a", a.toString(), SCALE_A);
            registry.load("a-copy", a.toString(), SCALE_A);
            String identity = registry.resolve(LoraSelection.of("a")).identity();

            BatchSequence first = run(engine, image, request(LoraSelection.of("a")));
            List<KvCacheEvent.BlockStored> stored = stores(batches);
            KvCacheEvent.BlockStored head0 = stored.get(0);
            assertEquals("the chain's first block: the adapters, then the image it begins",
                    Arrays.asList(BlockExtraKey.lora(identity), BlockExtraKey.media(hash, head.length)),
                    head0.extraKeys().get(0));
            assertEquals("the next block continues the image and inherits the adapters through the chain",
                    Arrays.asList(BlockExtraKey.media(hash, head.length - PAGE_ROWS)), secondBlockKeys(stored));
            for (KvCacheEvent.BlockStored e : stored) {
                assertEquals("every store of the media chain names its adapters", "a", e.loraName());
            }

            BatchSequence again = run(engine, image, request(LoraSelection.of("a")));
            assertEquals("the same image under the same adapters reuses every row but the last",
                    image.nTokens() - 1, again.reusedPrefix());
            assertArrayEquals(first.result().tokens(), again.result().tokens());
            assertEquals("the same image under the base model shares no block", 0,
                    run(engine, image, request(null)).reusedPrefix());
            assertEquals("another image under the same adapters shares only the text before it", head.length,
                    run(engine, otherImage, request(LoraSelection.of("a"))).reusedPrefix());
            BatchSequence copy = run(engine, image, request(LoraSelection.of("a-copy")));
            assertEquals("the same adapter content under another name shares the media chain",
                    image.nTokens() - 1, copy.reusedPrefix());
            assertArrayEquals(first.result().tokens(), copy.result().tokens());
        }
    }

    /** {@code head}, a non-causal media item of {@code rows} at the cursor, then {@code tail}. */
    private static MediaPrompt withItem(int[] head, float[][] rows, int[] tail) {
        return MediaPrompt.builder().addText(head)
                .addMedia(MediaRopePositions.normal(head.length, rows.length), rows, false)
                .addText(tail).build();
    }

    private static float[][] mediaRows(int n, int width, float v) {
        float[][] r = new float[n][width];
        for (int i = 0; i < n; i++) {
            for (int d = 0; d < width; d++) {
                r[i][d] = v * (1 + 0.05f * ((i * 3 + d) % 5)) - 0.02f * (d % 4);
            }
        }
        return r;
    }

    /** The extra keys of the chain's second block among {@code stored}, the first chain's stores in order. */
    private static List<BlockExtraKey> secondBlockKeys(List<KvCacheEvent.BlockStored> stored) {
        int seen = 0;
        for (KvCacheEvent.BlockStored e : stored) {
            for (List<BlockExtraKey> keys : e.extraKeys()) {
                if (seen++ == 1) {
                    return keys;
                }
            }
        }
        throw new AssertionError("the chain stored fewer than two blocks");
    }

    @Test
    public void unknownAdaptersRefuseByNameOnTheBatchAndSingleSequencePaths() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig cfg = config(2).build();
        try (GGUFModel m = new GGUFModel(model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(m, cfg)) {
            IllegalArgumentException unknown = assertThrows(IllegalArgumentException.class,
                    () -> engine.submit(tokens(1), request(LoraSelection.of("ghost")), NONE));
            assertTrue(unknown.getMessage(), unknown.getMessage().contains("ghost"));
            assertEquals(0, engine.queued());
        }
        EngineConfig single = config(1).build();
        try (GGUFModel one = new GGUFModel(model.toString(), single)) {
            IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                    () -> GenerationLoop.forModel(one, single).generate(tokens(1), request(LoraSelection.of("a")),
                            NONE));
            assertTrue(e.getMessage(), e.getMessage().contains("LoRA adapter a is not loaded"));
        }
    }

    /** Two sequences over paged memory of {@link #PAGE_ROWS}-row blocks, so prompts fill whole cached blocks. */
    private static EngineConfig.Builder pagedConfig() {
        return EngineConfig.builder().memory(MemoryConfig.builder().nCtx(64).ubatch(8).kvPaged(true)
                        .kvPageSize(PAGE_ROWS).parallel(2).typeK(KvDType.F32).typeV(KvDType.F32).prefixReuse(true)
                        .build())
                .loraConfig(LoraConfig.builder().maxAdapters(2).maxRank(64).cacheEntries(2).build());
    }

    private static GenerationRequest salted(LoraSelection lora, String salt) {
        return request(lora).toBuilder().cacheSalt(salt).build();
    }

    private static BatchSequence run(BatchEngine engine, int[] prompt, GenerationRequest request) {
        BatchSequence seq = engine.submit(prompt, request, NONE);
        engine.drain();
        assertEquals("the request completed: " + seq.error(), BatchSequence.State.DONE, seq.state());
        return seq;
    }

    private static BatchSequence run(BatchEngine engine, MediaPrompt prompt, GenerationRequest request) {
        BatchSequence seq = engine.submit(prompt, request, NONE);
        engine.drain();
        assertEquals("the request completed: " + seq.error(), BatchSequence.State.DONE, seq.state());
        return seq;
    }

    /** The {@code BlockStored} events of every batch so far, in order. */
    private static List<KvCacheEvent.BlockStored> stores(List<KvEventBatch> batches) {
        List<KvCacheEvent.BlockStored> out = new ArrayList<>();
        for (KvEventBatch b : batches) {
            for (KvCacheEvent e : b.events()) {
                if (e instanceof KvCacheEvent.BlockStored) {
                    out.add((KvCacheEvent.BlockStored) e);
                }
            }
        }
        return out;
    }

    /** The last store that began a chain (its first block has no parent). */
    private static KvCacheEvent.BlockStored lastChainHead(List<KvEventBatch> batches) {
        KvCacheEvent.BlockStored last = null;
        for (KvCacheEvent.BlockStored e : stores(batches)) {
            if (e.parentBlockHashInt() == null) {
                last = e;
            }
        }
        assertNotNull("a chain was stored", last);
        return last;
    }

    private static void assertMatches(GenerationResult reference, BatchSequence sequence) {
        assertEquals("expected DONE but was " + sequence.state()
                + (sequence.error() == null ? "" : ": " + sequence.error()), BatchSequence.State.DONE,
                sequence.state());
        GenerationResult r = sequence.result();
        assertNotNull(r);
        assertArrayEquals("token ids", reference.tokens(), r.tokens());
        assertEquals("text", reference.text(), r.text());
    }
}

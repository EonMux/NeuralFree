package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.TokenCallback;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.engine.kvevents.BlockExtraKey;
import com.eonmux.neuralfree.gguf.engine.kvevents.KvCacheEvent;
import com.eonmux.neuralfree.gguf.engine.kvevents.KvEventBatch;
import com.eonmux.neuralfree.gguf.engine.kvevents.KvEventPublishers;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;

/**
 * The prefix cache's hash-keyed block pool over pages of four rows: prompts reuse the whole blocks of any earlier
 * history along the block hash chain — interleaved histories, a conversation's next turn over the previous
 * answer's decoded blocks, a preempted sequence's own history at its recompute — with the tokens of the uncached
 * engine; free blocks give way least recently used and tail first; and every store, eviction and clear publishes
 * the matching KV cache event (vLLM's {@code BlockStored}, {@code BlockRemoved}, {@code AllBlocksCleared}).
 */
public class BatchPrefixBlockPoolTest {

    private static final int PAGE = 4;

    private static EngineConfig config(int nCtx, int parallel, boolean reuse) {
        return EngineConfig.builder().memory(MemoryConfig.builder().nCtx(nCtx).ubatch(8).kvPaged(true)
                .kvPageSize(PAGE).parallel(parallel).typeK(KvDType.F32).typeV(KvDType.F32).prefixReuse(reuse)
                .build()).build();
    }

    private static GenerationRequest request(int n) {
        return GenerationRequest.builder().maxNewTokens(n).ignoreEos(true)
                .sampler(SamplerConfig.greedyDecoding()).build();
    }

    private static int[] tokens(int seed, int n) {
        int[] t = new int[n];
        for (int i = 0; i < n; i++) {
            t[i] = 4 + (seed * 11 + i * 7) % 250;
        }
        return t;
    }

    private static int[] concat(int[]... parts) {
        int n = 0;
        for (int[] p : parts) {
            n += p.length;
        }
        int[] out = new int[n];
        int at = 0;
        for (int[] p : parts) {
            System.arraycopy(p, 0, out, at, p.length);
            at += p.length;
        }
        return out;
    }

    private static BatchSequence run(BatchEngine engine, int[] prompt, GenerationRequest request) {
        BatchSequence seq = engine.submit(prompt, request, (TokenCallback) null);
        for (int step = 0; step < 2_000 && !seq.finished(); step++) {
            engine.pump();
        }
        assertEquals("the request completed: " + seq.error(), BatchSequence.State.DONE, seq.state());
        return seq;
    }

    /** The tokens each prompt decodes on an engine without a prefix cache, one after the other. */
    private static int[][] uncached(Path model, int nCtx, int[][] prompts, int n) throws IOException {
        EngineConfig cfg = config(nCtx, 1, false);
        int[][] out = new int[prompts.length][];
        try (GGUFModel m = new GGUFModel(model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(m, cfg)) {
            for (int i = 0; i < prompts.length; i++) {
                out[i] = run(engine, prompts[i], request(n)).result().tokens();
            }
        }
        return out;
    }

    @Test
    public void interleavedHistoriesReuseEachOthersWholeBlocksWithTheUncachedTokens() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        int[] shared = tokens(1, 8);
        int[] a = concat(shared, tokens(2, 9));
        int[] b = concat(shared, tokens(3, 7));
        int[] a2 = concat(Arrays.copyOf(a, 16), tokens(4, 5));
        int[] b2 = concat(Arrays.copyOf(b, 12), tokens(5, 6));
        int[] c = concat(Arrays.copyOf(a, 10), tokens(6, 4));
        int[][] prompts = {a, b, a2, b2, c};
        int[] expectedReuse = {0, 8, 16, 12, 10};
        int[][] expected = uncached(model, 128, prompts, 4);
        EngineConfig cfg = config(128, 2, true);
        try (GGUFModel m = new GGUFModel(model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(m, cfg)) {
            for (int i = 0; i < prompts.length; i++) {
                BatchSequence seq = run(engine, prompts[i], request(4));
                assertEquals("prompt " + i + " reuses the blocks of the histories it shares", expectedReuse[i],
                        seq.reusedPrefix());
                assertArrayEquals("prompt " + i + " decodes as uncached", expected[i], seq.result().tokens());
            }
        }
    }

    @Test
    public void theNextTurnReusesThePreviousAnswersDecodedBlocks() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig cfg = config(128, 1, true);
        try (GGUFModel m = new GGUFModel(model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(m, cfg)) {
            int[] first = tokens(7, 13);
            BatchSequence turn1 = run(engine, first, request(10));
            int[] second = concat(first, turn1.result().tokens(), tokens(8, 5));
            BatchSequence turn2 = run(engine, second, request(4));
            assertEquals("every row the first turn committed (all but its last token) is adopted",
                    first.length + 10 - 1, turn2.reusedPrefix());
            int[][] expected = uncached(model, 128, new int[][] {second}, 4);
            assertArrayEquals(expected[0], turn2.result().tokens());
        }
    }

    @Test
    public void freeBlocksEvictLeastRecentlyUsedAndTailFirst() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        // eight pages: each history of 21 rows needs six, so the second evicts four of the first's
        EngineConfig cfg = config(32, 1, true);
        int[] x = tokens(9, 18);
        int[] y = tokens(10, 18);
        int[][] expected = uncached(model, 32, new int[][] {x, y}, 4);
        try (GGUFModel m = new GGUFModel(model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(m, cfg)) {
            run(engine, x, request(4));
            BatchSequence other = run(engine, y, request(4));
            assertEquals(0, other.reusedPrefix());
            BatchSequence again = run(engine, x, request(4));
            assertTrue("the head blocks outlive the evicted tail, got " + again.reusedPrefix(),
                    again.reusedPrefix() >= PAGE && again.reusedPrefix() % PAGE == 0
                            && again.reusedPrefix() < x.length - 1);
            assertArrayEquals(expected[0], again.result().tokens());
            assertArrayEquals(expected[1], other.result().tokens());
        }
    }

    @Test
    public void aPreemptedSequenceRecomputesOverItsCachedBlocks() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        int[] a = tokens(11, 12);
        int[] b = tokens(12, 12);
        long[] recomputed = new long[2];
        int[][][] results = new int[2][][];
        for (int pass = 0; pass < 2; pass++) {
            EngineConfig cfg = config(64, 2, pass == 1);
            try (GGUFModel m = new GGUFModel(model.toString(), cfg);
                 BatchEngine engine = BatchEngine.forModel(m, cfg)) {
                BatchSequence first = engine.submit(a, request(40), (TokenCallback) null);
                BatchSequence second = engine.submit(b, request(40), (TokenCallback) null);
                for (int step = 0; step < 4_000 && !(first.finished() && second.finished()); step++) {
                    engine.pump();
                }
                assertEquals(BatchSequence.State.DONE, first.state());
                assertEquals(BatchSequence.State.DONE, second.state());
                assertTrue("the two outgrow the 64 shared cells", engine.stats().preemptions() > 0);
                recomputed[pass] = engine.stats().recomputedTokens();
                results[pass] = new int[][] {first.result().tokens(), second.result().tokens()};
            }
        }
        assertTrue("the cached blocks spare recompute rows: " + recomputed[1] + " of " + recomputed[0],
                recomputed[1] < recomputed[0]);
        assertArrayEquals(results[0][0], results[1][0]);
        assertArrayEquals(results[0][1], results[1][1]);
    }

    @Test
    public void storesEvictionsAndClearsPublishTheirEvents() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig cfg = config(32, 1, true);
        List<KvEventBatch> batches = new ArrayList<>();
        int[] x = tokens(13, 9);
        try (GGUFModel m = new GGUFModel(model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(m, cfg, KvEventPublishers.listener(batches::add))) {
            BatchSequence first = run(engine, x, request(3));
            List<KvCacheEvent> stored = events(batches);
            // 9 prompt rows and 2 fed answer tokens: blocks 0 and 1 as they commit, the partial block at retirement
            int[] history = concat(x, Arrays.copyOf(first.result().tokens(), 2));
            assertChain(stored, history, null);
            assertEquals(3, ((KvCacheEvent.BlockStored) stored.get(stored.size() - 1)).blockSize());
            KvCacheEvent.BlockStored head = (KvCacheEvent.BlockStored) stored.get(0);
            assertNull("the base model carries no adapter id (vLLM's lora_id)", head.loraId());
            assertNull("the base model names no adapters", head.loraName());
            assertEquals("the events name the group the blocks live in (vLLM's kv_cache_spec_kind)",
                    "full_attention", head.kvCacheSpecKind());
            assertNull("a full-attention group publishes no sliding window",
                    head.kvCacheSpecSlidingWindow());

            batches.clear();
            run(engine, x, request(3));
            assertTrue("a repeat stores no block anew", events(batches).isEmpty());

            run(engine, tokens(14, 18), request(4));
            List<KvCacheEvent> pressure = events(batches);
            List<Long> removed = new ArrayList<>();
            for (KvCacheEvent e : pressure) {
                if (e instanceof KvCacheEvent.BlockRemoved) {
                    for (long h : ((KvCacheEvent.BlockRemoved) e).blockHashInts()) {
                        removed.add(h);
                    }
                }
            }
            long[] chain = chainHashes(history, null);
            assertTrue("the new prompt evicts the cached blocks, got " + pressure, !removed.isEmpty());
            for (int i = 0; i < removed.size(); i++) {
                assertEquals("eviction " + i + " takes the tail first", chain[chain.length - 1 - i],
                        (long) removed.get(i));
            }

            batches.clear();
            engine.clearPrefixCache();
            assertEquals(1, batches.size());
            assertEquals(Arrays.asList(KvCacheEvent.AllBlocksCleared.INSTANCE), batches.get(0).events());
            assertNull("a single engine publishes no rank", batches.get(0).dataParallelRank());
        }
    }

    @Test
    public void aSaltedChainCarriesItsSaltInTheFirstBlocksExtraKeys() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig cfg = config(64, 1, true);
        List<KvEventBatch> batches = new ArrayList<>();
        int[] x = tokens(15, 12);
        try (GGUFModel m = new GGUFModel(model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(m, cfg, KvEventPublishers.listener(batches::add))) {
            BatchSequence salted = run(engine, x, request(1).toBuilder().cacheSalt("tenant-a").build());
            List<KvCacheEvent> stored = events(batches);
            assertChain(stored, x, "tenant-a");
            KvCacheEvent.BlockStored head = (KvCacheEvent.BlockStored) stored.get(0);
            assertEquals(Collections.singletonList(BlockExtraKey.salt("tenant-a")), head.extraKeys().get(0));
            assertNull("a base-model chain names no adapters", head.loraName());
            assertNull("a base-model chain carries no adapter id", head.loraId());
            if (head.extraKeys().size() > 1) {
                assertNull("later blocks inherit the salt through the chain", head.extraKeys().get(1));
            }
            assertEquals(KvCacheEvent.MEDIUM_CPU, head.medium());

            BatchSequence other = run(engine, x, request(1).toBuilder().cacheSalt("tenant-b").build());
            assertEquals("another salt shares no block", 0, other.reusedPrefix());
            BatchSequence same = run(engine, x, request(1).toBuilder().cacheSalt("tenant-a").build());
            assertEquals(x.length - 1, same.reusedPrefix());
            assertArrayEquals(salted.result().tokens(), other.result().tokens());
            assertArrayEquals(salted.result().tokens(), same.result().tokens());
        }
    }

    /**
     * The selected algorithm reaches the chain: over canonical CBOR ({@code sha256_cbor}, vLLM's cross-language
     * form) the same prompt stores the chain the CBOR hash function computes, different from the pickle form's.
     */
    @Test
    public void theConfiguredHashAlgoHashesTheChain() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig cfg = EngineConfig.builder().memory(MemoryConfig.builder().nCtx(64).ubatch(8).kvPaged(true)
                .kvPageSize(PAGE).parallel(1).typeK(KvDType.F32).typeV(KvDType.F32).prefixReuse(true)
                .prefixHashAlgo(PrefixHashAlgo.SHA256_CBOR).build()).build();
        List<KvEventBatch> batches = new ArrayList<>();
        int[] x = tokens(4, 8);
        try (GGUFModel m = new GGUFModel(model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(m, cfg, KvEventPublishers.listener(batches::add))) {
            run(engine, x, request(1));
            BlockHasher cbor = BlockHasher.of(PrefixHashAlgo.SHA256_CBOR, BlockHasher.DEFAULT_NONE_HASH_SEED);
            byte[] expected = cbor.hash(null, Arrays.copyOfRange(x, 0, PAGE), null);
            KvCacheEvent.BlockStored head = (KvCacheEvent.BlockStored) events(batches).get(0);
            assertArrayEquals("the chain hashes per the configured algorithm", expected,
                    head.blockHashes()[0]);
            assertEquals("the integers read the digest's leading 64 bits",
                    KvCacheEvent.hashInt(expected), head.blockHashInts()[0]);

            BlockHasher pickle = BlockHasher.of(PrefixHashAlgo.SHA256, BlockHasher.DEFAULT_NONE_HASH_SEED);
            assertFalse("the CBOR chain differs from the pickle chain of the same tokens",
                    Arrays.equals(pickle.hash(null, x, null), head.blockHashes()[0]));
        }
    }

    /**
     * A block the host-RAM tier demoted and promoted back carries the same digest through the round trip: the
     * device store, the demotion store and the promotion store all publish the digest vLLM's hash function computes
     * for the block's tokens, the removal names it, and the demotion event carries the block's tokens.
     */
    @Test
    public void aBlockRoundTrippingThroughTheHostTierKeepsItsVllmDigest() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig cfg = EngineConfig.builder().memory(MemoryConfig.builder().nCtx(32).ubatch(8).kvPaged(true)
                .kvPageSize(PAGE).parallel(1).typeK(KvDType.F32).typeV(KvDType.F32).cacheRamMib(-1).build()).build();
        int[] a = tokens(1, 14);
        List<KvEventBatch> batches = new ArrayList<>();
        try (GGUFModel m = new GGUFModel(model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(m, cfg, KvEventPublishers.listener(batches::add))) {
            run(engine, a, request(2));
            // 30 prompt tokens fill all eight pages of the 32-row memory, so every cached block of a leaves
            // the device for the host tier
            run(engine, tokens(2, 30), request(2));
            run(engine, a, request(2));
        }
        long[] chain = chainHashes(a, null);
        for (int k = 0; k < 3; k++) {
            long digest = chain[k];
            List<KvCacheEvent.BlockStored> stores = stored(batches, digest);
            assertEquals("block " + k + " is stored on the device, demoted to the host and promoted back", 3,
                    stores.size());
            assertEquals("block " + k + " left the device once", 1, removed(batches, digest));
            KvCacheEvent.BlockStored demoted = stores.get(1);
            assertEquals(PAGE, demoted.blockSize());
            assertArrayEquals("the host copy keeps the block's tokens",
                    Arrays.copyOfRange(a, k * PAGE, k * PAGE + PAGE), demoted.tokenIds());
            assertTrue("the host copy names the host medium",
                    KvCacheEvent.MEDIUM_CPU.equals(demoted.medium()));
        }
    }

    /** The events of every batch so far, in order. */
    private static List<KvCacheEvent> events(List<KvEventBatch> batches) {
        List<KvCacheEvent> out = new ArrayList<>();
        for (KvEventBatch b : batches) {
            out.addAll(b.events());
        }
        return out;
    }

    /** Every {@code BlockStored} naming {@code digest}, in order. */
    private static List<KvCacheEvent.BlockStored> stored(List<KvEventBatch> batches, long digest) {
        List<KvCacheEvent.BlockStored> out = new ArrayList<>();
        for (KvCacheEvent e : events(batches)) {
            if (!(e instanceof KvCacheEvent.BlockStored)) {
                continue;
            }
            KvCacheEvent.BlockStored s = (KvCacheEvent.BlockStored) e;
            for (long h : s.blockHashInts()) {
                if (h == digest) {
                    out.add(s);
                    break;
                }
            }
        }
        return out;
    }

    /** How many {@code BlockRemoved} name {@code digest}. */
    private static int removed(List<KvEventBatch> batches, long digest) {
        int n = 0;
        for (KvCacheEvent e : events(batches)) {
            if (e instanceof KvCacheEvent.BlockRemoved) {
                for (long h : ((KvCacheEvent.BlockRemoved) e).blockHashInts()) {
                    if (h == digest) {
                        n++;
                    }
                }
            }
        }
        return n;
    }

    /**
     * The chain hashes of {@code history}'s blocks of {@link #PAGE} tokens, a trailing partial block last, computed
     * the way the engine computes them: vLLM's {@code hash_block_tokens} over sha256 with its default seed, so the
     * published values equal what vLLM's own hash function produces for the same tokens.
     */
    private static long[] chainHashes(int[] history, String salt) {
        BlockHasher hasher = BlockHasher.forEnvironment(PrefixHashAlgo.parse("sha256", "test"));
        int n = (history.length + PAGE - 1) / PAGE;
        long[] out = new long[n];
        byte[] parent = null;
        for (int k = 0; k < n; k++) {
            int len = Math.min(PAGE, history.length - k * PAGE);
            List<BlockExtraKey> keys = k == 0 && salt != null
                    ? Collections.singletonList(BlockExtraKey.salt(salt)) : null;
            parent = hasher.hash(parent, Arrays.copyOfRange(history, k * PAGE, k * PAGE + len), keys);
            out[k] = KvCacheEvent.hashInt(parent);
        }
        return out;
    }

    /**
     * Asserts that {@code stored} holds exactly the {@code BlockStored} events of {@code history}'s blocks in
     * chain order: each event's parent is the previous event's last block, the first has none, and the tokens and
     * hashes are the history's.
     */
    private static void assertChain(List<KvCacheEvent> stored, int[] history, String salt) {
        long[] chain = chainHashes(history, salt);
        List<Long> hashes = new ArrayList<>();
        List<Integer> tokens = new ArrayList<>();
        Long parent = null;
        for (KvCacheEvent e : stored) {
            assertTrue("only stores, got " + e, e instanceof KvCacheEvent.BlockStored);
            KvCacheEvent.BlockStored s = (KvCacheEvent.BlockStored) e;
            assertEquals("each run continues the chain", parent, s.parentBlockHashInt());
            for (long h : s.blockHashInts()) {
                hashes.add(h);
            }
            for (int t : s.tokenIds()) {
                tokens.add(t);
            }
            parent = hashes.get(hashes.size() - 1);
        }
        assertEquals(chain.length, hashes.size());
        for (int k = 0; k < chain.length; k++) {
            assertEquals("block " + k + " hash", chain[k], (long) hashes.get(k));
        }
        assertEquals(history.length, tokens.size());
        for (int i = 0; i < history.length; i++) {
            assertEquals((int) tokens.get(i), history[i]);
        }
        assertSame(KvCacheEvent.BlockStored.class, stored.get(0).getClass());
    }
}

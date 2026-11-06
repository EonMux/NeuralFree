package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.GenerationLoop;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Path;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * The longest-prefix cache over the paged memory (PARITY P3): a second request whose prompt extends a cached
 * prefix reuses the shared pages (its {@code reusedPrefix} counts the copied tokens, at least the last prompt
 * token prefills), computes the same tokens and text as the uncached run, keeps working across LRU evictions on
 * a one-slot memory, and the {@code neuralfree.prefix.reuse=false} switch disables the whole route. The cache
 * never changes a result: same tokens, same logits, same sampling.
 */
public class BatchPrefixCacheTest {

    private static final int N_TOKENS = 12;
    private static final int N_NEW = 6;

    private static EngineConfig.Builder config(boolean paged, boolean reuse) {
        MemoryConfig.Builder mc = MemoryConfig.builder().nCtx(64).ubatch(8)
                .typeK(KvDType.F32).typeV(KvDType.F32).prefixReuse(reuse);
        if (paged) {
            mc = mc.kvPaged(true);
        }
        return EngineConfig.builder().memory(mc.build());
    }

    private static GenerationRequest request(int maxNewTokens) {
        return GenerationRequest.builder()
                .maxNewTokens(maxNewTokens)
                .sampler(SamplerConfig.greedyDecoding())
                .build();
    }

    /** A deterministic prompt; {@code suffix} varies the tail so two prompts share a prefix. */
    private static int[] prompt(int seed, int suffix) {
        int[] t = new int[N_TOKENS];
        for (int i = 0; i < t.length; i++) {
            t[i] = 4 + (seed * 11 + i * 7) % 250;
        }
        t[N_TOKENS - 1] = 4 + suffix;
        return t;
    }

    private static GenerationResult reference(Path model, int[] prompt) throws IOException {
        try (GGUFModel one = new GGUFModel(model.toString(), config(false, true).build())) {
            return GenerationLoop.forModel(one, config(false, true).build())
                    .generate(prompt, request(N_NEW), (com.eonmux.neuralfree.gguf.TokenCallback) null);
        }
    }

    @Test
    public void aRepeatedPromptReusesItsPagesAndKeepsTheResultIdentical() throws IOException, InterruptedException {
        Path model = SyntheticModels.llama().writeTempFile();
        try (GGUFModel m = new GGUFModel(model.toString(), config(true, true).build());
             BatchEngine engine = BatchEngine.forModel(m, config(true, true).build())) {
            BatchSequence first = engine.submit(prompt(1, 0), request(N_NEW),
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.drain();
            assertEquals("the first run prefills everything", 0, first.reusedPrefix());
            BatchSequence second = engine.submit(prompt(1, 0), request(N_NEW),
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.drain();
            assertTrue("the second run reuses the shared pages (all but the evaluated last token), got "
                            + second.reusedPrefix(),
                    second.reusedPrefix() == N_TOKENS - 1);
            assertArrayEquals("same tokens as the same engine's uncached run", first.result().tokens(),
                    second.result().tokens());
            assertEquals(first.result().text(), second.result().text());
        }
    }

    @Test
    public void aSharedPrefixReusesOnlyTheCommonTokens() throws IOException, InterruptedException {
        Path model = SyntheticModels.llama().writeTempFile();
        try (GGUFModel m = new GGUFModel(model.toString(), config(true, true).build());
             BatchEngine engine = BatchEngine.forModel(m, config(true, true).build())) {
            engine.submit(prompt(2, 0), request(N_NEW), (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.drain();
            BatchSequence shared = engine.submit(prompt(2, 5), request(N_NEW),
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.drain();
            assertEquals("only the common prefix copies (the last prompt token always prefills)",
                    N_TOKENS - 1, shared.reusedPrefix());
            BatchSequence different = engine.submit(prompt(9, 0), request(N_NEW),
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.drain();
            assertEquals("an unrelated prompt matches no cached prefix", 0, different.reusedPrefix());
        }
    }

    /**
     * The regression of the adopt-cut page walk: over a context wide enough that an adopt's cut removes whole
     * pages above a partially cut page, the walk ran ascending, so every whole page above the partial one sat
     * behind an interior unref — its freed id was handed back by the pool and re-entered the same list as a live
     * page, and the next eviction unref'd the live page twice ({@code page N is not held}). The walk runs from
     * the back now: a run of sequential adopts and evictions keeps every page held exactly once and every
     * generation finishes.
     */
    @Test
    public void sequentialAdoptCutsAndEvictionsKeepEveryPageHeldOnce() throws IOException, InterruptedException {
        MemoryConfig.Builder mc = MemoryConfig.builder().nCtx(256).ubatch(8).parallel(2)
                .typeK(KvDType.F32).typeV(KvDType.F32).prefixReuse(true).kvPaged(true);
        EngineConfig config = EngineConfig.builder().memory(mc.build()).build();
        Path model = SyntheticModels.llama().writeTempFile();
        try (GGUFModel m = new GGUFModel(model.toString(), config);
             BatchEngine engine = BatchEngine.forModel(m, config)) {
            int[] longPrompt = new int[24];
            for (int i = 0; i < longPrompt.length; i++) {
                longPrompt[i] = 4 + (i * 13) % 250;
            }
            for (int round = 0; round < 6; round++) {
                int[] prompt = new int[longPrompt.length];
                System.arraycopy(longPrompt, 0, prompt, 0, longPrompt.length);
                prompt[round * 3] = 4 + round * 17; // a divergent tail the adopt must cut
                BatchSequence seq = engine.submit(prompt, request(8),
                        (com.eonmux.neuralfree.gguf.TokenCallback) null);
                engine.drain();
                assertEquals("round " + round + " finishes", BatchSequence.State.DONE, seq.state());
                if (round > 0) {
                    assertTrue("round " + round + " adopts the shared prefix, got " + seq.reusedPrefix(),
                            seq.reusedPrefix() >= 1);
                }
            }
        }
    }

    @Test
    public void aOneSlotMemoryEvictsTheOldestCachedPrefixForANewGeneration() throws IOException, InterruptedException {
        Path model = SyntheticModels.llama().writeTempFile();
        try (GGUFModel m = new GGUFModel(model.toString(), config(true, true).build());
             BatchEngine engine = BatchEngine.forModel(m, config(true, true).build())) {
            engine.submit(prompt(3, 0), request(N_NEW), (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.drain();
            // the second run needs the one slot: the first run's cached prefix evicts (LRU)
            engine.submit(prompt(4, 0), request(N_NEW), (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.drain();
            // the evicted prefix no longer matches
            BatchSequence again = engine.submit(prompt(3, 0), request(N_NEW),
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.drain();
            assertEquals("the evicted prefix is gone", 0, again.reusedPrefix());
            // and it is cached again for the next round
            BatchSequence onceMore = engine.submit(prompt(3, 0), request(N_NEW),
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.drain();
            assertTrue(onceMore.reusedPrefix() > 0);
        }
    }

    @Test
    public void theReuseSwitchDisablesTheCache() throws IOException, InterruptedException {
        Path model = SyntheticModels.llama().writeTempFile();
        try (GGUFModel m = new GGUFModel(model.toString(), config(true, false).build());
             BatchEngine engine = BatchEngine.forModel(m, config(true, false).build())) {
            engine.submit(prompt(5, 0), request(N_NEW), (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.drain();
            BatchSequence second = engine.submit(prompt(5, 0), request(N_NEW),
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.drain();
            assertEquals("neuralfree.prefix.reuse=false caches nothing", 0, second.reusedPrefix());
        }
    }

    @Test
    public void repeatedRunsReachASteadyStateWithNoCellLeak() throws IOException, InterruptedException {
        Path model = SyntheticModels.llama().writeTempFile();
        try (GGUFModel m = new GGUFModel(model.toString(), config(true, true).build());
             BatchEngine engine = BatchEngine.forModel(m, config(true, true).build())) {
            com.eonmux.neuralfree.gguf.memory.paged.PagedSequences paged =
                    ((com.eonmux.neuralfree.gguf.memory.cpu.HeapModelMemory) m.forwardBackend().memory())
                            .pagedSequences();
            BatchSequence first = engine.submit(prompt(6, 0), request(N_NEW),
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.drain();
            assertTrue("the retired sequence stays cached", paged.freeCells() < paged.totalCells());
            long prior = Long.MIN_VALUE;
            for (int round = 0; round < 4; round++) {
                BatchSequence run = engine.submit(prompt(6, 0), request(N_NEW),
                        (com.eonmux.neuralfree.gguf.TokenCallback) null);
                engine.drain();
                assertTrue(run.reusedPrefix() > 0);
                long free = paged.freeCells();
                if (round > 0) {
                    assertEquals("every adopted round lands on the same steady state: no leak",
                            prior, free);
                }
                prior = free;
            }
        }
    }

    /** Long enough that two copies never fit the 64 shared cells together. */
    private static final int LONG = 50;

    private static int[] longPrompt(int seed) {
        int[] t = new int[LONG];
        for (int i = 0; i < t.length; i++) {
            t[i] = 4 + (seed * 13 + i * 7) % 250;
        }
        return t;
    }

    private static EngineConfig twoSlots() {
        return EngineConfig.builder().memory(MemoryConfig.builder().nCtx(64).ubatch(8).kvPaged(true).parallel(2)
                .typeK(KvDType.F32).typeV(KvDType.F32).build()).build();
    }

    /** Pumps until every handle finished, failing instead of spinning when the engine stops making progress. */
    private static void pumpUntilFinished(BatchEngine engine, BatchSequence... handles) {
        for (int step = 0; step < 2_000; step++) {
            boolean all = true;
            for (BatchSequence h : handles) {
                all &= h.finished();
            }
            if (all) {
                for (BatchSequence h : handles) {
                    assertEquals("the request completed: " + h.error(), BatchSequence.State.DONE, h.state());
                }
                return;
            }
            engine.pump();
        }
        throw new AssertionError("the engine stopped admitting: a queued request never ran");
    }

    @Test
    public void aQueuedCopyOfARunningPromptAdmitsOverItsCachedPrefix() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        try (GGUFModel m = new GGUFModel(model.toString(), twoSlots());
             BatchEngine engine = BatchEngine.forModel(m, twoSlots())) {
            BatchSequence first = engine.submit(longPrompt(1), request(3),
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            BatchSequence second = engine.submit(longPrompt(1), request(3),
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            pumpUntilFinished(engine, first, second);
            assertEquals("the copy adopts the cached prompt, all but its last token", LONG - 1,
                    second.reusedPrefix());
            assertArrayEquals(first.result().tokens(), second.result().tokens());
            assertTrue(engine.idle());
        }
    }

    @Test
    public void aNewPromptEvictsCachedPrefixesForTheCellsItNeeds() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        try (GGUFModel m = new GGUFModel(model.toString(), twoSlots());
             BatchEngine engine = BatchEngine.forModel(m, twoSlots())) {
            BatchSequence first = engine.submit(longPrompt(1), request(3),
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            pumpUntilFinished(engine, first);
            BatchSequence other = engine.submit(longPrompt(2), request(3),
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            pumpUntilFinished(engine, other);
            assertEquals("an unrelated prompt prefills whole", 0, other.reusedPrefix());
            assertEquals(3, other.result().tokens().length);
        }
    }

    @Test
    public void sequencesThatOutgrowTheSharedPoolPreemptAndKeepTheirResults() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        int[] a = prompt(3, 1);
        int[] b = prompt(4, 2);
        int[] soloA;
        int[] soloB;
        try (GGUFModel m = new GGUFModel(model.toString(), twoSlots());
             BatchEngine engine = BatchEngine.forModel(m, twoSlots())) {
            BatchSequence one = engine.submit(a, unbounded(40), (com.eonmux.neuralfree.gguf.TokenCallback) null);
            pumpUntilFinished(engine, one);
            soloA = one.result().tokens();
        }
        try (GGUFModel m = new GGUFModel(model.toString(), twoSlots());
             BatchEngine engine = BatchEngine.forModel(m, twoSlots())) {
            BatchSequence one = engine.submit(b, unbounded(40), (com.eonmux.neuralfree.gguf.TokenCallback) null);
            pumpUntilFinished(engine, one);
            soloB = one.result().tokens();
        }
        // 12 + 40 positions each: together they need 104 of the 64 shared cells, so one is preempted mid-decode
        try (GGUFModel m = new GGUFModel(model.toString(), twoSlots());
             BatchEngine engine = BatchEngine.forModel(m, twoSlots())) {
            BatchSequence first = engine.submit(a, unbounded(40), (com.eonmux.neuralfree.gguf.TokenCallback) null);
            BatchSequence second = engine.submit(b, unbounded(40), (com.eonmux.neuralfree.gguf.TokenCallback) null);
            pumpUntilFinished(engine, first, second);
            assertArrayEquals("the older sequence decodes as it does alone", soloA, first.result().tokens());
            assertArrayEquals("the preempted sequence recomputes and continues unchanged", soloB,
                    second.result().tokens());
            assertTrue(engine.idle());
        }
    }

    /** Greedy decoding of exactly {@code n} tokens: the end-of-generation token does not stop it. */
    private static GenerationRequest unbounded(int n) {
        return GenerationRequest.builder().maxNewTokens(n).ignoreEos(true)
                .sampler(SamplerConfig.greedyDecoding()).build();
    }
}

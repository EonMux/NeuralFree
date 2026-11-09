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
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * The prefix cache's block pool over the paged memory: a second request whose prompt extends a
 * cached prefix reuses the shared pages (its {@code reusedPrefix} counts the adopted tokens, at least the last
 * prompt token prefills), computes the same tokens and text as the uncached run, keeps working across LRU
 * evictions on a one-slot memory, and the {@code neuralfree.prefix.reuse=false} switch disables the whole route.
 * A model larger than the physical RAM turns an unset cache off (Avalon server-context.cpp:1550-1566), an
 * explicit setting stands. The cache never changes a result: same tokens, same logits, same sampling.
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

    /**
     * Cached blocks outlive their sequence's slot and give way only to pool pressure: on a one-slot memory of
     * four pages, a long unrelated prompt needs every page, so the free blocks of the first prompt evict (least
     * recent first) and the first prompt prefills whole again, then is cached again for the round after.
     */
    @Test
    public void aOneSlotMemoryEvictsTheCachedBlocksANewGenerationNeeds() throws IOException {
        EngineConfig oneSlot = EngineConfig.builder().memory(MemoryConfig.builder().nCtx(64).ubatch(8)
                .kvPaged(true).parallel(1).typeK(KvDType.F32).typeV(KvDType.F32).prefixReuse(true).build()).build();
        Path model = SyntheticModels.llama().writeTempFile();
        try (GGUFModel m = new GGUFModel(model.toString(), oneSlot);
             BatchEngine engine = BatchEngine.forModel(m, oneSlot)) {
            BatchSequence first = engine.submit(longPrompt(3), request(3),
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            pumpUntilFinished(engine, first);
            BatchSequence repeat = engine.submit(longPrompt(3), request(3),
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            pumpUntilFinished(engine, repeat);
            assertEquals("the blocks outlive the retired slot", LONG - 1, repeat.reusedPrefix());
            BatchSequence other = engine.submit(longPrompt(4), request(3),
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            pumpUntilFinished(engine, other);
            assertEquals(0, other.reusedPrefix());
            BatchSequence again = engine.submit(longPrompt(3), request(3),
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            pumpUntilFinished(engine, again);
            assertEquals("the evicted blocks are gone", 0, again.reusedPrefix());
            assertArrayEquals(first.result().tokens(), again.result().tokens());
            BatchSequence onceMore = engine.submit(longPrompt(3), request(3),
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            pumpUntilFinished(engine, onceMore);
            assertEquals("and cached again for the next round", LONG - 1, onceMore.reusedPrefix());
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

    /** The paged memory of {@link #config}, without the prefix-reuse switch. */
    private static MemoryConfig.Builder pagedMemory() {
        return MemoryConfig.builder().nCtx(64).ubatch(8).typeK(KvDType.F32).typeV(KvDType.F32).kvPaged(true);
    }

    @Test
    public void thePrefixCachePolicyMirrorsAvalon() {
        MemoryConfig unset = MemoryConfig.defaults();
        assertFalse(BatchEngineFactory.prefixReuseForHostRam(unset, 2, 1));
        assertTrue(BatchEngineFactory.prefixReuseForHostRam(unset, 1, 2));
        assertTrue("an unknown RAM keeps the default", BatchEngineFactory.prefixReuseForHostRam(unset, 2, -1));
        assertTrue("an explicit neuralfree.prefix.reuse=true stands beside any host",
                BatchEngineFactory.prefixReuseForHostRam(MemoryConfig.builder().prefixReuse(true).build(), 2, 1));
        assertFalse(BatchEngineFactory.prefixReuseForHostRam(MemoryConfig.builder().prefixReuse(false).build(), 1, 2));
    }

    @Test
    public void aModelLargerThanRamDisablesTheCacheUnlessSetExplicitly() throws IOException, InterruptedException {
        // the 1-byte host makes the synthetic model "larger than RAM"
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig unset = EngineConfig.builder().memory(pagedMemory().build()).build();
        try (GGUFModel m = new GGUFModel(model.toString(), unset);
             BatchEngine engine = BatchEngine.forModel(m, unset, 1)) {
            engine.submit(prompt(5, 0), request(N_NEW), (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.drain();
            BatchSequence second = engine.submit(prompt(5, 0), request(N_NEW),
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.drain();
            assertEquals("a larger-than-RAM model caches no prefix", 0, second.reusedPrefix());
        }
        EngineConfig forced = EngineConfig.builder().memory(pagedMemory().prefixReuse(true).build()).build();
        try (GGUFModel m = new GGUFModel(model.toString(), forced);
             BatchEngine engine = BatchEngine.forModel(m, forced, 1)) {
            engine.submit(prompt(5, 0), request(N_NEW), (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.drain();
            BatchSequence second = engine.submit(prompt(5, 0), request(N_NEW),
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.drain();
            assertEquals("an explicit neuralfree.prefix.reuse=true keeps the cache", N_TOKENS - 1,
                    second.reusedPrefix());
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
            // the two copies outgrow the 64 shared cells: the copy is preempted or requeued, and its retry adopts
            // the whole cached blocks of the prompt that survive the evictions making room for it
            assertTrue("the copy adopts the cached prompt's whole blocks, got " + second.reusedPrefix(),
                    second.reusedPrefix() >= 3 * MemoryConfig.DEFAULT_PAGE_SIZE
                            && second.reusedPrefix() <= LONG - 1);
            assertArrayEquals(first.result().tokens(), second.result().tokens());
            assertTrue(engine.idle());
        }
    }

    /**
     * A request admitted while an identical twin still decodes forks off the twin's committed pages
     * ({@code PagedSequences.copy}, whole pages shared) instead of prefilling the prompt again: its
     * {@code reusedPrefix} covers all but the last prompt token and both decode the same tokens.
     */
    @Test
    public void aPromptAdmittedWhileItsTwinRunsForksOffItsCommittedPages() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        try (GGUFModel m = new GGUFModel(model.toString(), twoSlots());
             BatchEngine engine = BatchEngine.forModel(m, twoSlots())) {
            BatchSequence first = engine.submit(prompt(1, 0), request(N_NEW),
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.pump(); // admits and prefills the first request
            BatchSequence second = engine.submit(prompt(1, 0), request(N_NEW),
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            pumpUntilFinished(engine, first, second);
            assertEquals(0, first.reusedPrefix());
            assertEquals("the fork copies the twin's committed prefix pages", N_TOKENS - 1,
                    second.reusedPrefix());
            assertArrayEquals(first.result().tokens(), second.result().tokens());
            assertTrue(engine.idle());
        }
    }

    /** The server's n-choices pattern: identical prompts decode side by side over one shared prefill. */
    @Test
    public void threeChoicesShareOnePrefillOfTheirCommonPrompt() throws IOException {
        MemoryConfig mc = MemoryConfig.builder().nCtx(128).ubatch(8).kvPaged(true).parallel(3)
                .typeK(KvDType.F32).typeV(KvDType.F32).build();
        EngineConfig config = EngineConfig.builder().memory(mc).build();
        Path model = SyntheticModels.llama().writeTempFile();
        try (GGUFModel m = new GGUFModel(model.toString(), config);
             BatchEngine engine = BatchEngine.forModel(m, config)) {
            BatchSequence first = engine.submit(prompt(1, 0), request(N_NEW),
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.pump();
            BatchSequence second = engine.submit(prompt(1, 0), request(N_NEW),
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            BatchSequence third = engine.submit(prompt(1, 0), request(N_NEW),
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            pumpUntilFinished(engine, first, second, third);
            assertEquals(0, first.reusedPrefix());
            assertEquals(N_TOKENS - 1, second.reusedPrefix());
            assertEquals(N_TOKENS - 1, third.reusedPrefix());
            assertArrayEquals(first.result().tokens(), second.result().tokens());
            assertArrayEquals(first.result().tokens(), third.result().tokens());
            assertTrue(engine.idle());
        }
    }

    /**
     * The n choices of one HTTP request arrive together, before the first pump admits anything: the first choice
     * prefills, the queued copies wait as forks on it and prefill only their tails once its prompt committed —
     * the same one shared prefill, with the donors' prefill still pending at admission time.
     */
    @Test
    public void choicesSubmittedTogetherWaitForTheFirstPrefillAndForkOffIt() throws IOException {
        MemoryConfig mc = MemoryConfig.builder().nCtx(128).ubatch(8).kvPaged(true).parallel(3)
                .typeK(KvDType.F32).typeV(KvDType.F32).build();
        EngineConfig config = EngineConfig.builder().memory(mc).build();
        Path model = SyntheticModels.llama().writeTempFile();
        try (GGUFModel m = new GGUFModel(model.toString(), config);
             BatchEngine engine = BatchEngine.forModel(m, config)) {
            BatchSequence first = engine.submit(prompt(1, 0), request(N_NEW),
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            BatchSequence second = engine.submit(prompt(1, 0), request(N_NEW),
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            BatchSequence third = engine.submit(prompt(1, 0), request(N_NEW),
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            pumpUntilFinished(engine, first, second, third);
            assertEquals(0, first.reusedPrefix());
            assertEquals(N_TOKENS - 1, second.reusedPrefix());
            assertEquals(N_TOKENS - 1, third.reusedPrefix());
            assertArrayEquals(first.result().tokens(), second.result().tokens());
            assertArrayEquals(first.result().tokens(), third.result().tokens());
            assertTrue(engine.idle());
        }
    }

    /**
     * The cache's token budget evicts the least recently used entries once their combined tokens pass it: a
     * two-slot engine whose second cached entry pushes the first over the budget leaves the first prompt without
     * a match (its pages freed) while the newest entry still serves its prefix. Every offer trims, so a request
     * served later in submit order keeps its prefix only while it is the cache's newest.
     */
    /**
     * More identical prompts arrive together than the engine has sequence slots (the server's {@code best_of}
     * candidates): the overflow waits in the queue and admits as slots retire, forking off or adopting the cached
     * prompt, and every one of them finishes with the same greedy tokens. The longest prompt fills the shared
     * cells with its two new tokens, so a fork and its donor cannot both write the page they share: the youngest
     * — the fork — yields to its donor instead of the donor being preempted for it over and over.
     */
    @Test
    public void moreIdenticalConcurrentPromptsThanSlotsAllFinish() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        for (int length : new int[]{12, 40, 62}) {
            int[] prompt = new int[length];
            for (int i = 0; i < length; i++) {
                prompt[i] = 4 + (i * 13) % 250;
            }
            for (int copies = 3; copies <= 4; copies++) {
                try (GGUFModel m = new GGUFModel(model.toString(), twoSlots());
                     BatchEngine engine = BatchEngine.forModel(m, twoSlots())) {
                    BatchSequence[] handles = new BatchSequence[copies];
                    for (int c = 0; c < copies; c++) {
                        handles[c] = engine.submit(prompt, unbounded(2),
                                (com.eonmux.neuralfree.gguf.TokenCallback) null);
                    }
                    pumpUntilFinished(engine, handles);
                    for (BatchSequence h : handles) {
                        assertArrayEquals("length " + length + ", " + copies + " copies: same greedy tokens",
                                handles[0].result().tokens(), h.result().tokens());
                    }
                    assertTrue("length " + length + ", " + copies + " copies: the engine drains", engine.idle());
                }
            }
        }
    }

    @Test
    public void theBudgetEvictsTheLeastRecentlyUsedCachedPrefixes() throws IOException {
        MemoryConfig mc = MemoryConfig.builder().nCtx(64).ubatch(8).kvPaged(true).parallel(2)
                .typeK(KvDType.F32).typeV(KvDType.F32).prefixCacheBudget(20).build();
        EngineConfig config = EngineConfig.builder().memory(mc).build();
        Path model = SyntheticModels.llama().writeTempFile();
        try (GGUFModel m = new GGUFModel(model.toString(), config);
             BatchEngine engine = BatchEngine.forModel(m, config)) {
            BatchSequence one = engine.submit(prompt(1, 0), request(N_NEW),
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.pump();
            BatchSequence two = engine.submit(prompt(9, 0), request(N_NEW),
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            pumpUntilFinished(engine, one, two);
            BatchSequence keeper = engine.submit(prompt(9, 0), request(N_NEW),
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.drain();
            assertEquals("the newest entry stays", N_TOKENS - 1, keeper.reusedPrefix());
            BatchSequence again = engine.submit(prompt(1, 0), request(N_NEW),
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.drain();
            assertEquals("the budget evicted the oldest entry", 0, again.reusedPrefix());
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

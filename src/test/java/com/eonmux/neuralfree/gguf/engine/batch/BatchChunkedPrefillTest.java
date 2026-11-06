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
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Chunked prefill with a per-step token budget (vLLM V1's scheduler shape, llama.cpp's decode tokens and prompt
 * chunks sharing one n_batch batch): a long prompt prefills in budget-bounded chunks beside the decode rows of
 * the sequences around it, every step's rows stay inside the budget, the long-prefill cap caps one chunk's share
 * while other requests hold work (and does not apply to a lone request), and with chunked prefill off the oldest
 * prefill still runs alone while the decodes wait. Every finished sequence carries the same tokens and text as
 * the same prompt and request run one at a time through {@link GenerationLoop}.
 */
public class BatchChunkedPrefillTest {

    private static final int LONG_TOKENS = 40;
    private static final int SHORT_TOKENS = 6;
    private static final int N_NEW = 8;

    private static MemoryConfig.Builder memory() {
        return MemoryConfig.builder().nCtx(128).ubatch(8).typeK(KvDType.F32).typeV(KvDType.F32).parallel(3);
    }

    private static GenerationRequest request(int maxNewTokens) {
        return GenerationRequest.builder()
                .maxNewTokens(maxNewTokens)
                .sampler(SamplerConfig.greedyDecoding())
                .build();
    }

    /** Deterministic token ids inside the synthetic vocab, distinct per seed (as {@code BatchEngineTest}). */
    private static int[] tokens(int seed, int n) {
        int[] t = new int[n];
        for (int i = 0; i < n; i++) {
            t[i] = 4 + (seed * 7 + i * 13) % 250;
        }
        return t;
    }

    /** The reference: the same prompt and request through the single-sequence GenerationLoop on contiguous memory. */
    private static GenerationResult reference(Path model, MemoryConfig.Builder memory, int[] prompt,
                                              GenerationRequest request) throws IOException {
        EngineConfig cfg = EngineConfig.builder().memory(memory.parallel(1).kvPaged(false).build()).build();
        try (GGUFModel one = new GGUFModel(model.toString(), cfg)) {
            return GenerationLoop.forModel(one, cfg).generate(prompt, request,
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
        }
    }

    /** The prompt-token counters at every step boundary: one entry per pump. */
    private static final class StepTrace {
        final List<Long> promptTokens = new ArrayList<>();
        final List<Long> decodeSteps = new ArrayList<>();

        void record(BatchStats stats) {
            promptTokens.add(stats.promptTokens());
            decodeSteps.add(stats.decodeSteps());
        }
    }

    /** DONE with the reference result, or fails naming the engine error. */
    private static void assertDone(BatchSequence sequence) {
        assertTrue("expected DONE but was " + sequence.state()
                        + (sequence.error() == null ? "" : ": " + sequence.error()),
                sequence.state() == BatchSequence.State.DONE);
    }

    private static void assertResultsMatch(GenerationResult reference, GenerationResult batched) {
        assertEquals("stop reason", reference.stopReason(), batched.stopReason());
        assertEquals("token count", reference.tokenCount(), batched.tokenCount());
        org.junit.Assert.assertArrayEquals("token ids", reference.tokens(), batched.tokens());
        assertEquals("text", reference.text(), batched.text());
    }

    @Test
    public void aLongPromptPrefillsInBudgetBoundedChunksBesideTheDecodesOfAnEarlierSequence() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        MemoryConfig.Builder mem = memory().maxBatchedTokens(6);
        EngineConfig cfg = EngineConfig.builder().memory(mem.build()).build();
        GenerationRequest request = request(N_NEW);
        GenerationResult refA = reference(model, memory(), tokens(1, LONG_TOKENS), request);
        GenerationResult refB = reference(model, memory(), tokens(2, SHORT_TOKENS), request);
        assertTrue(refA.tokenCount() > 0 && refB.tokenCount() > 0);

        try (GGUFModel m = new GGUFModel(model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(m, cfg)) {
            int[] bTokens = new int[1];
            BatchSequence b = engine.submit(tokens(2, SHORT_TOKENS), request,
                    (tokenId, text) -> bTokens[0]++);
            BatchSequence a = engine.submit(tokens(1, LONG_TOKENS), request,
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            int[] bTokensAtAEnd = new int[] {-1};
            a.onFinish(() -> bTokensAtAEnd[0] = bTokens[0]);
            StepTrace trace = new StepTrace();
            while (!a.finished()) {
                engine.pump();
                trace.record(engine.stats());
            }
            assertDone(a);
            assertDone(b);
            // the earlier sequence decoded while the long prompt was still prefilling: the step mixed its
            // decode row with the prompt chunk (with prefill steps alone, no decode runs before the prompt ends)
            assertTrue("the decode beside the chunking prompt made progress, got " + bTokensAtAEnd[0],
                    bTokensAtAEnd[0] >= 1);
            boolean mixedStep = false;
            for (int i = 0; i < trace.decodeSteps.size(); i++) {
                assertTrue("step " + i + " scheduled " + (trace.promptTokens.get(i) - before(trace, i))
                                + " prompt rows over the budget",
                        trace.promptTokens.get(i) - before(trace, i) <= 6);
                if (trace.decodeSteps.get(i) > 0 && trace.promptTokens.get(i) < LONG_TOKENS + SHORT_TOKENS) {
                    mixedStep = true;
                }
            }
            assertTrue("a step mixed a decode row with a still-incomplete prefill", mixedStep);
            assertEquals("both prompts prefilled exactly once", LONG_TOKENS + SHORT_TOKENS,
                    trace.promptTokens.get(trace.promptTokens.size() - 1).intValue());
            assertResultsMatch(refA, a.result());
            assertResultsMatch(refB, b.result());
        }
    }

    /** The cumulative prompt tokens before step {@code i}. */
    private static long before(StepTrace trace, int i) {
        return i == 0 ? 0 : trace.promptTokens.get(i - 1);
    }

    @Test
    public void everyChunkOfASinglePromptStaysInsideTheBudgetAndTheReferenceIsExact() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        MemoryConfig.Builder mem = memory().maxBatchedTokens(5);
        EngineConfig cfg = EngineConfig.builder().memory(mem.build()).build();
        GenerationRequest request = request(N_NEW);
        GenerationResult ref = reference(model, memory(), tokens(3, LONG_TOKENS), request);

        try (GGUFModel m = new GGUFModel(model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(m, cfg)) {
            int[] firstTokenAtPump = new int[] {-1};
            int[] pumps = new int[1];
            BatchSequence a = engine.submit(tokens(3, LONG_TOKENS), request, (tokenId, text) -> {
                if (firstTokenAtPump[0] < 0) {
                    firstTokenAtPump[0] = pumps[0];
                }
            });
            StepTrace trace = new StepTrace();
            while (!a.finished()) {
                pumps[0]++;
                engine.pump();
                trace.record(engine.stats());
            }
            assertDone(a);
            assertEquals("the 40-row prompt prefilled in 5-row chunks", 40,
                    trace.promptTokens.get(trace.promptTokens.size() - 1).intValue());
            for (int i = 0; i < trace.promptTokens.size(); i++) {
                assertTrue("step " + i + " prefilled " + (trace.promptTokens.get(i) - before(trace, i))
                        + " rows over the budget of 5", trace.promptTokens.get(i) - before(trace, i) <= 5);
            }
            assertTrue("the first token waited for the whole prompt: chunked prefill needs at least "
                            + (LONG_TOKENS / 5) + " prefill steps, got " + firstTokenAtPump[0],
                    firstTokenAtPump[0] >= LONG_TOKENS / 5);
            assertEquals("the budget ends the sequence", N_NEW, a.result().tokenCount());
            assertResultsMatch(ref, a.result());
        }
    }

    @Test
    public void theLongPrefillCapCapsEachChunkWhileAnotherRequestHoldsWorkButNotForALoneRequest()
            throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        MemoryConfig.Builder mem = memory().maxBatchedTokens(16).longPrefillTokens(4);
        EngineConfig cfg = EngineConfig.builder().memory(mem.build()).build();
        GenerationRequest request = request(N_NEW);
        GenerationResult refA = reference(model, memory(), tokens(1, LONG_TOKENS), request);
        GenerationResult refB = reference(model, memory(), tokens(4, LONG_TOKENS), request);

        try (GGUFModel m = new GGUFModel(model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(m, cfg)) {
            BatchSequence lone = engine.submit(tokens(5, LONG_TOKENS), request,
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.pump();
            long loneFirstStep = engine.stats().promptTokens();
            assertTrue("the cap does not apply to the only request: it takes the whole budget, got "
                    + loneFirstStep, loneFirstStep == 16);
            engine.drain();
            assertDone(lone);

            BatchSequence a = engine.submit(tokens(1, LONG_TOKENS), request,
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            BatchSequence b = engine.submit(tokens(4, LONG_TOKENS), request,
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            long promptBefore = engine.stats().promptTokens();
            long decodesBefore = engine.stats().decodeSteps();
            engine.pump();
            BatchStats afterFirstStep = engine.stats();
            assertEquals("each chunk takes the cap, not the budget", 8,
                    afterFirstStep.promptTokens() - promptBefore);
            assertEquals("no decode runs while both prompts prefill", 0,
                    afterFirstStep.decodeSteps() - decodesBefore);
            engine.drain();
            assertDone(a);
            assertDone(b);
            assertResultsMatch(refA, a.result());
            assertResultsMatch(refB, b.result());
        }
    }

    @Test
    public void withChunkedPrefillOffTheWholeTailPrefillsInOneStepAndTheDecodesWait() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        MemoryConfig.Builder mem = memory().maxBatchedTokens(6).chunkedPrefill(false);
        EngineConfig cfg = EngineConfig.builder().memory(mem.build()).build();
        GenerationRequest request = request(N_NEW);
        GenerationResult refA = reference(model, memory(), tokens(1, LONG_TOKENS), request);
        GenerationResult refB = reference(model, memory(), tokens(2, SHORT_TOKENS), request);

        try (GGUFModel m = new GGUFModel(model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(m, cfg)) {
            int[] bTokens = new int[1];
            BatchSequence b = engine.submit(tokens(2, SHORT_TOKENS), request, (tokenId, text) -> bTokens[0]++);
            BatchSequence a = engine.submit(tokens(1, LONG_TOKENS), request,
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.pump();
            assertEquals("the first prefill runs whole", SHORT_TOKENS, engine.stats().promptTokens());
            assertEquals(0, engine.stats().decodeSteps());
            engine.pump();
            BatchStats afterSecond = engine.stats();
            assertEquals("the long tail ignores the budget and prefills in one step",
                    LONG_TOKENS + SHORT_TOKENS, afterSecond.promptTokens());
            // the stall the feature removes: no decode step ran while either prompt prefilled, so the earlier
            // sequence made no decode progress (its one routed token is its own prefill's sample)
            assertEquals("the decodes wait through every prefill", 0, afterSecond.decodeSteps());
            assertEquals("the earlier sequence decoded nothing while the prompt prefilled", 1, bTokens[0]);
            engine.drain();
            assertDone(a);
            assertDone(b);
            assertResultsMatch(refA, a.result());
            assertResultsMatch(refB, b.result());
        }
    }

    @Test
    public void aChunkedTailPrefillsOnlyItsRowsAfterAnAdoptedPrefix() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        MemoryConfig.Builder mem = memory().maxBatchedTokens(5);
        EngineConfig cfg = EngineConfig.builder().memory(mem.build()).build();
        GenerationRequest request = request(N_NEW);
        int[] prompt = tokens(6, 24);
        GenerationResult ref = reference(model, memory(), prompt, request);

        try (GGUFModel m = new GGUFModel(model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(m, cfg)) {
            BatchSequence first = engine.submit(prompt, request, (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.drain();
            assertDone(first);
            assertEquals("the first run chunked the whole prompt", 0, first.reusedPrefix());
            assertEquals(24, engine.stats().promptTokens());
            BatchSequence second = engine.submit(prompt, request, (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.drain();
            assertDone(second);
            assertEquals("the second run adopts the cached prefix and prefills the 1-row tail", 23,
                    second.reusedPrefix());
            assertEquals("only the tail prefilled again", 1,
                    engine.stats().promptTokens() - 24);
            assertResultsMatch(ref, second.result());
        }
    }

    /**
     * A shared prefix that ends mid-prompt re-prefills only its tail, in budget-bounded chunks, at the rows
     * past the reused prefix: every chunk writes at the position cursor the reused prefix left, so a wrong
     * cursor (a tail chunk writing at position 0, over the adopted rows) corrupts the run and the tokens leave
     * the reference.
     */
    @Test
    public void aChunkedTailAfterAPartialSharedPrefixWritesPastTheReusedRows() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        MemoryConfig.Builder mem = memory().maxBatchedTokens(5);
        EngineConfig cfg = EngineConfig.builder().memory(mem.build()).build();
        GenerationRequest request = request(N_NEW);
        int[] warm = tokens(6, 16);
        int[] prompt = new int[36];
        System.arraycopy(warm, 0, prompt, 0, 12);
        System.arraycopy(tokens(9, 24), 0, prompt, 12, 24);
        GenerationResult ref = reference(model, memory(), prompt, request);

        try (GGUFModel m = new GGUFModel(model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(m, cfg)) {
            BatchSequence first = engine.submit(warm, request, (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.drain();
            assertDone(first);
            long afterWarm = engine.stats().promptTokens();
            BatchSequence second = engine.submit(prompt, request, (com.eonmux.neuralfree.gguf.TokenCallback) null);
            long seen = afterWarm;
            List<Long> chunks = new ArrayList<>();
            while (!second.finished()) {
                engine.pump();
                long now = engine.stats().promptTokens();
                if (now != seen) {
                    chunks.add(now - seen);
                    seen = now;
                }
            }
            assertDone(second);
            assertEquals("the 12 shared rows are adopted, the differing row 13 ends the match", 12,
                    second.reusedPrefix());
            assertEquals("only the 24-row tail prefilled", 24, seen - afterWarm);
            for (long chunk : chunks) {
                assertTrue("a chunk of " + chunk + " rows over the budget of 5", chunk <= 5);
            }
            assertResultsMatch(ref, second.result());
        }
    }

    /**
     * A prompt prefills in chunks of the size its ubatches split into up to its end: 45 rows at a ubatch (and
     * budget) of 8 take 8 / 8 / 8 / 7 / 7 / 7 rows rather than the greedy 8 / 8 / 8 / 8 / 8 / 5, so no chunk is a
     * short trailing ubatch (avalon tools/server/server-context.cpp:4126).
     */
    @Test
    public void aPromptPrefillsInEqualUbatchSizedChunks() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        MemoryConfig.Builder mem = memory().maxBatchedTokens(8); // the ubatch, whose 8 rows the chunks fill
        EngineConfig cfg = EngineConfig.builder().memory(mem.build()).build();
        GenerationRequest request = request(N_NEW);
        int[] prompt = tokens(7, 45);
        GenerationResult ref = reference(model, memory(), prompt, request);

        try (GGUFModel m = new GGUFModel(model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(m, cfg)) {
            BatchSequence a = engine.submit(prompt, request, (com.eonmux.neuralfree.gguf.TokenCallback) null);
            List<Long> chunks = new ArrayList<>();
            long seen = 0;
            while (!a.finished()) {
                engine.pump();
                long now = engine.stats().promptTokens();
                if (now != seen) {
                    chunks.add(now - seen);
                    seen = now;
                }
            }
            assertDone(a);
            assertEquals("45 rows split into as few 8-row ubatches as the cap allows, equally",
                    Arrays.asList(8L, 8L, 8L, 7L, 7L, 7L), chunks);
            assertResultsMatch(ref, a.result());
        }
    }

    /**
     * Two prompts that fit the admission hint fill the shared pool with their decodes: the mixed decode step
     * runs out of pages, rolls both rows back, and the relieve preempts the younger sequence, which re-admits
     * after the first retires, recomputes its history in chunks, and finishes with its exact reference tokens.
     */
    @Test
    public void aMixedStepThatRunsOutOfPagesRollsBackAndThePreemptedSequenceRecoversByChunkedRecompute()
            throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        MemoryConfig.Builder mem = MemoryConfig.builder().nCtx(64).ubatch(8)
                .typeK(KvDType.F32).typeV(KvDType.F32).parallel(2).prefixReuse(false).maxBatchedTokens(7);
        EngineConfig cfg = EngineConfig.builder().memory(mem.build()).build();
        // ignore_eos keeps both sequences decoding to the budget, so their rows fill the shared pool together
        GenerationRequest request = request(N_NEW).toBuilder().ignoreEos(true).build();
        GenerationResult refA = reference(model, MemoryConfig.builder().nCtx(64).ubatch(8)
                .typeK(KvDType.F32).typeV(KvDType.F32), tokens(1, 30), request);
        GenerationResult refB = reference(model, MemoryConfig.builder().nCtx(64).ubatch(8)
                .typeK(KvDType.F32).typeV(KvDType.F32), tokens(2, 30), request);

        try (GGUFModel m = new GGUFModel(model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(m, cfg)) {
            com.eonmux.neuralfree.gguf.memory.paged.PagedSequences paged =
                    ((com.eonmux.neuralfree.gguf.memory.cpu.HeapModelMemory) m.forwardBackend().memory())
                            .pagedSequences();
            int freeBefore = paged.freeCells();
            BatchSequence a = engine.submit(tokens(1, 30), request, (com.eonmux.neuralfree.gguf.TokenCallback) null);
            BatchSequence b = engine.submit(tokens(2, 30), request, (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.drain();
            assertDone(a);
            assertDone(b);
            assertEquals("the same tokens as the sequential reference, preemption and recompute included",
                    refA.tokenCount(), a.result().tokenCount());
            assertEquals(refB.tokenCount(), b.result().tokenCount());
            assertTrue("the pool filled beside both prefills and the relieve preempted the younger sequence",
                    engine.stats().preemptions() >= 1);
            assertEquals("no rows leak through the rolled-back steps", freeBefore, paged.freeCells());
            assertResultsMatch(refA, a.result());
            assertResultsMatch(refB, b.result());
        }
    }
}

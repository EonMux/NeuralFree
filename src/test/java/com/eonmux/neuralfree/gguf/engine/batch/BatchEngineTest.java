package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.arch.SupportClass;
import com.eonmux.neuralfree.gguf.engine.ContextFullException;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.GenerationLoop;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.ReasoningFilter;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.engine.StopReason;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.memory.cpu.HeapModelMemory;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Continuous batching over the paged memory (PARITY P2): a {@link BatchEngine} over the synthetic two-layer llama
 * admits, prefills and decodes several sequences at once, and every finished sequence carries the same tokens and
 * text as the same prompt and request run one at a time through {@link GenerationLoop} on the contiguous path.
 * Oversize prompts refuse at submit, the budget ends a sequence and frees its slot for the next queued one, a
 * cancellation stops with {@code StopReason.CANCELLED}, an unpaged memory refuses the engine by name and a
 * non-generating model refuses with its support class.
 */
public class BatchEngineTest {

    private static final int N_TOKENS = 12;
    private static final int N_NEW = 8;

    private static EngineConfig.Builder config(boolean paged) {
        MemoryConfig.Builder mc = MemoryConfig.builder().nCtx(64).ubatch(8).typeK(KvDType.F32).typeV(KvDType.F32);
        if (paged) {
            mc = mc.parallel(2);
        }
        return EngineConfig.builder().memory(mc.build());
    }

    /** Greedy decoding over a bounded budget: the same prompt yields the same tokens on every path. */
    private static GenerationRequest request(int maxNewTokens) {
        return GenerationRequest.builder()
                .maxNewTokens(maxNewTokens)
                .sampler(SamplerConfig.greedyDecoding())
                .build();
    }

    /** The tokens a sequence runs (deterministic ids inside the synthetic vocab, as CpuForwardBatchTest). */
    private static int[] tokens(int seed) {
        int[] t = new int[N_TOKENS];
        for (int i = 0; i < t.length; i++) {
            t[i] = 4 + (seed * 7 + i * 13) % 250;
        }
        return t;
    }

    /** The reference: the same prompt and request through the single-sequence GenerationLoop. */
    private static GenerationResult reference(Path model, EngineConfig cfg, int[] prompt,
                                              GenerationRequest request) throws IOException {
        try (GGUFModel one = new GGUFModel(model.toString(), cfg)) {
            return GenerationLoop.forModel(one, cfg).generate(prompt, request,
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
        }
    }

    @Test
    public void twoConcurrentSequencesMatchTheSequentialGenerationLoopRuns() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig paged = config(true).build();
        GenerationRequest request = request(N_NEW);
        GenerationResult refA = reference(model, config(false).build(), tokens(1), request);
        GenerationResult refB = reference(model, config(false).build(), tokens(2), request);
        assertTrue(refA.tokenCount() > 0 && refB.tokenCount() > 0);

        List<String> pieces = new ArrayList<>();
        try (GGUFModel two = new GGUFModel(model.toString(), paged);
             BatchEngine engine = BatchEngine.forModel(two, paged)) {
            assertEquals(SupportClass.GEN, two.spec().supportClass());
            assertEquals(2, engine.maxSequences());
            StringBuilder textA = new StringBuilder();
            BatchSequence a = engine.submit(tokens(1), request, (channel, tokenId, piece) -> {
                if (channel == ReasoningFilter.Channel.CONTENT) {
                    textA.append(piece);
                }
            }, null);
            BatchSequence b = engine.submit(tokens(2), request,
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            assertEquals(BatchSequence.State.QUEUED, a.state());
            assertEquals(2, engine.queued());
            engine.drain();
            assertDone(a);
            assertDone(b);
            assertEquals(0, engine.running());
            assertTrue(engine.idle());
            assertResultsMatch(refA, a.result());
            assertResultsMatch(refB, b.result());
            assertEquals("routed content pieces build the result text", refA.text(), textA.toString());
        }
    }

    @Test
    public void aQueuedRequestAdmitsWhenASlotFreesAndMatchesItsOwnReference() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig paged = config(true).build();
        GenerationRequest request = request(N_NEW);
        GenerationResult refC = reference(model, config(false).build(), tokens(3), request);

        try (GGUFModel two = new GGUFModel(model.toString(), paged);
             BatchEngine engine = BatchEngine.forModel(two, paged)) {
            BatchSequence a = engine.submit(tokens(1), request, (com.eonmux.neuralfree.gguf.TokenCallback) null);
            BatchSequence b = engine.submit(tokens(2), request, (com.eonmux.neuralfree.gguf.TokenCallback) null);
            BatchSequence c = engine.submit(tokens(3), request, (com.eonmux.neuralfree.gguf.TokenCallback) null);
            assertEquals(BatchSequence.State.QUEUED, c.state());
            assertEquals(-1, c.seq());
            engine.drain();
            assertDone(a);
            assertDone(b);
            assertDone(c);
            assertTrue("the queued request ran through a freed slot", c.seq() == a.seq() || c.seq() == b.seq());
            assertResultsMatch(refC, c.result());
        }
    }

    @Test
    public void anOversizePromptRefusesAtSubmitAndTheEngineKeepsWorking() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig paged = config(true).build();
        try (GGUFModel two = new GGUFModel(model.toString(), paged);
             BatchEngine engine = BatchEngine.forModel(two, paged)) {
            // a 65-token prompt does not fit the 64-position context: submit refuses before any slot is taken
            int[] oversize = new int[65];
            for (int i = 0; i < oversize.length; i++) {
                oversize[i] = 4 + (i * 3) % 250;
            }
            assertThrows(ContextFullException.class, () -> engine.submit(oversize, request(N_NEW),
                    (com.eonmux.neuralfree.gguf.TokenCallback) null));
            assertTrue(engine.idle());
            BatchSequence ok = engine.submit(tokens(1), request(N_NEW),
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.drain();
            assertDone(ok);
            assertEquals(N_NEW, ok.result().tokenCount());
        }
    }

    @Test
    public void theBudgetEndsASequenceAndFreesItsSlotAndCells() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig paged = config(true).build();
        try (GGUFModel two = new GGUFModel(model.toString(), paged);
             BatchEngine engine = BatchEngine.forModel(two, paged)) {
            HeapModelMemory mem = (HeapModelMemory) two.forwardBackend().memory();
            assertNotNull(mem.pagedSequences());
            int freeBefore = mem.pagedSequences().freeCells();
            BatchSequence a = engine.submit(tokens(1), request(N_NEW),
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.drain();
            assertDone(a);
            assertEquals(N_NEW, a.result().tokenCount());
            assertEquals(StopReason.MAX_TOKENS, a.result().stopReason());
            assertEquals("the retired sequence freed its pages", freeBefore, mem.pagedSequences().freeCells());
            assertEquals("nothing of the retired sequence remains", -1, mem.pagedSequences().posMax(a.seq()));
            BatchSequence next = engine.submit(tokens(2), request(N_NEW),
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.drain();
            assertEquals("the freed slot is reused", a.seq(), next.seq());
            assertDone(next);
        }
    }

    @Test
    public void aCancellationStopsTheSequenceWithItsResult() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig paged = config(true).build();
        try (GGUFModel two = new GGUFModel(model.toString(), paged);
             BatchEngine engine = BatchEngine.forModel(two, paged)) {
            GenerationRequest cancelled = GenerationRequest.builder()
                    .maxNewTokens(N_NEW)
                    .sampler(SamplerConfig.greedyDecoding())
                    .cancellation(() -> true)
                    .build();
            BatchSequence a = engine.submit(tokens(1), cancelled,
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.drain();
            assertEquals(BatchSequence.State.CANCELLED, a.state());
            assertNotNull(a.result());
            assertEquals(StopReason.CANCELLED, a.result().stopReason());
        }
    }

    @Test
    public void aSequenceThatOutgrowsTheContextStopsContextFullAndTheEngineSurvives() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig paged = config(true).build();
        try (GGUFModel two = new GGUFModel(model.toString(), paged);
             BatchEngine engine = BatchEngine.forModel(two, paged)) {
            // a 60-token prompt into the 64-position context leaves 4 decode positions before the context ends
            int[] longPrompt = new int[60];
            for (int i = 0; i < longPrompt.length; i++) {
                longPrompt[i] = 4 + (i * 3) % 250;
            }
            BatchSequence a = engine.submit(longPrompt, request(64),
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.drain();
            assertDone(a);
            assertEquals(StopReason.CONTEXT_FULL, a.result().stopReason());
            BatchSequence ok = engine.submit(tokens(1), request(N_NEW),
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.drain();
            assertDone(ok);
        }
    }

    @Test
    public void anUnpagedMemoryRefusesTheEngineByName() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        try (GGUFModel one = new GGUFModel(model.toString(), config(false).build())) {
            UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class,
                    () -> BatchEngine.forModel(one, config(false).build()));
            assertTrue(e.getMessage(), e.getMessage().contains("paged"));
        }
    }

    @Test
    public void aNonGeneratingModelRefusesWithItsSupportClass() throws IOException {
        Path model = SyntheticModels.bert().writeTempFile();
        EngineConfig cfg = EngineConfig.builder().build();
        try (GGUFModel bert = new GGUFModel(model.toString(), cfg)) {
            assertEquals(SupportClass.EMBED, bert.spec().supportClass());
            UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class,
                    () -> BatchEngine.forModel(bert, cfg));
            assertTrue(e.getMessage(), e.getMessage().contains("EMBED"));
        }
    }

    /** Tokens, text and stop reason of a batch result match the sequential reference exactly. */
    private static void assertResultsMatch(GenerationResult reference, GenerationResult batched) {
        assertNotNull(batched);
        assertEquals("stop reason", reference.stopReason(), batched.stopReason());
        assertEquals("token count", reference.tokenCount(), batched.tokenCount());
        org.junit.Assert.assertArrayEquals("token ids", reference.tokens(), batched.tokens());
        assertEquals("text", reference.text(), batched.text());
    }

    /** DONE with the reference result, or fails naming the engine error. */
    private static void assertDone(BatchSequence sequence) {
        assertTrue("expected DONE but was " + sequence.state()
                        + (sequence.error() == null ? "" : ": " + sequence.error()),
                sequence.state() == BatchSequence.State.DONE);
    }
}

package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.arch.SupportClass;
import com.eonmux.neuralfree.gguf.engine.ContextFullException;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.engine.ForwardBatch;
import com.eonmux.neuralfree.gguf.engine.GenerationLoop;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.ReasoningFilter;
import com.eonmux.neuralfree.gguf.engine.RowTopK;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.engine.StopReason;
import com.eonmux.neuralfree.gguf.engine.watermark.WatermarkConfig;
import com.eonmux.neuralfree.gguf.grammar.ConstraintSpec;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.memory.cpu.HeapModelMemory;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Continuous batching over the paged memory: a {@link BatchEngine} over the synthetic two-layer llama
 * admits, prefills and decodes several sequences at once, and every finished sequence carries the same tokens and
 * text as the same prompt and request run one at a time through {@link GenerationLoop} on the contiguous path.
 * Oversize prompts refuse at submit, the budget ends a sequence and frees its slot for the next queued one, a
 * cancellation stops with {@code StopReason.CANCELLED}, an unpaged memory refuses the engine by name and a
 * memoryless encoder refuses with its memory form.
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

    /**
     * The default {@link ForwardBackend#forwardBatchTopK} (a backend without the device readout) selects the same
     * candidates the batch's rows hold, on the host, in batch order; a width below one refuses.
     */
    @Test
    public void theHostBatchTopKMatchesTheRowsItSelectsFrom() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig paged = config(true).build();
        int[] a = tokens(1);
        int[] b = tokens(2);
        ForwardBatch.Builder builder = ForwardBatch.builder(2 * N_TOKENS);
        for (int i = 0; i < N_TOKENS; i++) {
            builder.add(a[i], i, 0, i == N_TOKENS - 1);
            builder.add(b[i], i, 1, i == N_TOKENS - 1);
        }
        ForwardBatch batch = builder.build();
        int k = 5;
        try (GGUFModel two = new GGUFModel(model.toString(), paged)) {
            ForwardBackend f = two.forwardBackend();
            RowTopK expected = RowTopK.of(f.forwardBatch(batch), k);
            f.reset();
            RowTopK actual = f.forwardBatchTopK(batch, k);
            assertEquals(expected.rows(), actual.rows());
            assertEquals(expected.k(), actual.k());
            for (int r = 0; r < expected.rows(); r++) {
                assertArrayEquals("row " + r + " ids", expected.ids(r), actual.ids(r));
                assertArrayEquals("row " + r + " logits", expected.logits(r), actual.logits(r), 0f);
            }
            f.reset();
            assertThrows(IllegalArgumentException.class, () -> f.forwardBatchTopK(batch, 0));
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

    /**
     * A watermarked sequence decodes on the continuous-batching path exactly as it does one at a time: the prompt
     * marks where generation starts, so a context the prompt repeated is not a repeated generated context. The
     * prompt ends with the four tokens it began with, which a scope that searched the prompt too would treat as a
     * repeated context and sample with the ordinary draw instead of the watermark.
     */
    @Test
    public void aWatermarkedSequenceMatchesTheSequentialGenerationLoop() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig paged = config(true).build();
        int[] prompt = {4, 5, 6, 7, 4, 5, 6, 7};
        GenerationRequest request = GenerationRequest.builder()
                .maxNewTokens(N_NEW)
                .sampler(SamplerConfig.builder().topK(0).topP(1f).minP(0f).typicalP(1f).temperature(1f)
                        .seed(7L)
                        .watermark(WatermarkConfig.builder(1234L).contextWidth(4).build())
                        .build())
                .build();
        GenerationResult ref = reference(model, config(false).build(), prompt, request);
        try (GGUFModel two = new GGUFModel(model.toString(), paged);
             BatchEngine engine = BatchEngine.forModel(two, paged)) {
            BatchSequence a = engine.submit(prompt, request,
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.drain();
            assertDone(a);
            assertResultsMatch(ref, a.result());
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
        // reuse off: the test asserts the retirement frees the pages, which the prefix cache keeps taken
        MemoryConfig.Builder mc = MemoryConfig.builder().nCtx(64).ubatch(8)
                .typeK(KvDType.F32).typeV(KvDType.F32).prefixReuse(false).parallel(2);
        EngineConfig paged = EngineConfig.builder().memory(mc.build()).build();
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

    /**
     * A pooling-only model over a memory that holds no paged sequences (the BERT family plans no memory) refuses
     * the engine by name; with a paged memory the engine serves its pooling requests, and its generation submits
     * refuse by name (see BatchEnginePoolingTest's refusals for the pooling-only refusals).
     */
    @Test
    public void aNonGeneratingModelRefusesWithItsMemoryForm() throws IOException {
        Path model = SyntheticModels.bert().writeTempFile();
        EngineConfig cfg = EngineConfig.builder().build();
        try (GGUFModel bert = new GGUFModel(model.toString(), cfg)) {
            assertEquals(SupportClass.EMBED, bert.spec().supportClass());
            UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class,
                    () -> BatchEngine.forModel(bert, cfg));
            assertTrue(e.getMessage(), e.getMessage().contains("paged memory"));
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

    @Test
    public void aRequestCarriedConstraintCompilesAtSubmitAndShapesTheOutput() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig paged = config(true).build();
        GenerationRequest request = GenerationRequest.builder()
                .maxNewTokens(N_NEW)
                .sampler(SamplerConfig.greedyDecoding())
                .constraint(ConstraintSpec.choice(List.of("yes", "no")))
                .build();
        try (GGUFModel two = new GGUFModel(model.toString(), paged);
             BatchEngine engine = BatchEngine.forModel(two, paged)) {
            BatchSequence a = engine.submit(tokens(1), request, (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.drain();
            assertDone(a);
            String text = a.result().text();
            assertTrue("constrained text was '" + text + "'", text.matches("(yes|no)+"));

            GenerationRequest unusable = GenerationRequest.builder()
                    .maxNewTokens(2)
                    .sampler(SamplerConfig.greedyDecoding())
                    .constraint(ConstraintSpec.gbnf("root ::= (unclosed"))
                    .build();
            assertThrows(IllegalArgumentException.class,
                    () -> engine.submit(tokens(1), unusable, (com.eonmux.neuralfree.gguf.TokenCallback) null));
        }
    }
}

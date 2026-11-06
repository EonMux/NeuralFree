package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.engine.ContextFullException;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.GenerationLoop;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.LookupDrafterFactory;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.engine.SpeculativeOptions;
import com.eonmux.neuralfree.gguf.engine.StopReason;
import com.eonmux.neuralfree.gguf.engine.draft.NgramDrafter;
import com.eonmux.neuralfree.gguf.engine.lora.LoraConfig;
import com.eonmux.neuralfree.gguf.engine.lora.LoraSelection;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.memory.SequenceMemory;
import com.eonmux.neuralfree.gguf.memory.SequenceState;
import com.eonmux.neuralfree.gguf.testutil.LoraAdapterFiles;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Speculative decoding inside the batch engine (vLLM {@code scheduled_spec_decode_tokens}, llama.cpp server's
 * per-slot speculative loop): a {@link BatchEngine} over a {@link BatchDrafter} verifies every sequence's proposal
 * through the sequence's own sampling chain and keeps the lossless result — every finished sequence carries the
 * same tokens as the same prompt run plain, one at a time, through {@link GenerationLoop}. The prompt-lookup
 * drafter proposes from the paged token record, and scripted drafters pin the acceptance and the rejection walks.
 */
public class BatchEngineSpeculativeTest {

    private static final int N_TOKENS = 24;
    private static final int N_NEW = 8;

    private static EngineConfig.Builder config(int nCtx) {
        MemoryConfig mc = MemoryConfig.builder().nCtx(nCtx).ubatch(8)
                .typeK(KvDType.F32).typeV(KvDType.F32).parallel(2).build();
        return EngineConfig.builder().memory(mc);
    }

    /** Greedy decoding over a bounded budget: the same prompt yields the same tokens on every path. */
    private static GenerationRequest request(int maxNewTokens) {
        return GenerationRequest.builder()
                .maxNewTokens(maxNewTokens)
                .sampler(SamplerConfig.greedyDecoding())
                .build();
    }

    /** A prompt whose token n-grams repeat, so prompt lookup finds matches from the first decode steps. */
    private static int[] tokens() {
        int[] t = new int[N_TOKENS];
        for (int i = 0; i < t.length; i++) {
            t[i] = 4 + (i % 5) * 7;
        }
        return t;
    }

    /**
     * The prompt the prompt-lookup test runs: {@code (4, 5, v)} triples over v = 6..105 ending on the pair
     * (4, 5). The model's greedy continuation of it is a run of token 86 (pinned once against the fixture's
     * fixed weights), which the prompt stores after the (4, 5) pair — the first decode step's lookup pattern
     * (4, 5, 86) matches, so the drafter proposes from the first round on.
     */
    private static int[] lookupPrompt() {
        int[] prompt = new int[3 * 100 + 2];
        int w = 0;
        for (int v = 6; v <= 105; v++) {
            prompt[w++] = 4;
            prompt[w++] = 5;
            prompt[w++] = v;
        }
        prompt[w++] = 4;
        prompt[w++] = 5;
        return prompt;
    }

    /** The reference: the same prompt and request through the single-sequence GenerationLoop. */
    private static GenerationResult reference(Path model, int nCtx, int[] prompt, GenerationRequest request)
            throws IOException {
        MemoryConfig mc = MemoryConfig.builder().nCtx(nCtx).ubatch(8)
                .typeK(KvDType.F32).typeV(KvDType.F32).build();
        EngineConfig cfg = EngineConfig.builder().memory(mc).build();
        try (GGUFModel one = new GGUFModel(model.toString(), cfg)) {
            return GenerationLoop.forModel(one, cfg).generate(prompt, request,
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
        }
    }

    /** DONE with the reference result, or fails naming the engine error. */
    private static void assertDone(BatchSequence sequence) {
        assertTrue("expected DONE but was " + sequence.state()
                        + (sequence.error() == null ? "" : ": " + sequence.error()),
                sequence.state() == BatchSequence.State.DONE);
    }

    /** Tokens, text and stop reason of a batch result match the sequential reference exactly. */
    private static void assertResultsMatch(GenerationResult reference, GenerationResult batched) {
        assertEquals("stop reason", reference.stopReason(), batched.stopReason());
        assertEquals("token count", reference.tokenCount(), batched.tokenCount());
        assertArrayEquals("token ids", reference.tokens(), batched.tokens());
        assertEquals("text", reference.text(), batched.text());
    }

    @Test
    public void promptLookupRoundsKeepTheReferenceResultAndShowInTheStats() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        // sizes the short repeats of the lookup prompt match (llama.cpp's default 12 / 48 needs longer ones)
        EngineConfig cfg = config(700).draftType(EngineConfig.DraftType.NGRAM).speculativeOptions(SpeculativeOptions
                .builder().ngram(NgramDrafter.Options.builder(NgramDrafter.Kind.SIMPLE).sizeN(3).sizeM(8).build())
                .build()).build();
        GenerationRequest request = request(N_NEW);
        int[] prompt = lookupPrompt();
        GenerationResult ref = reference(model, 700, prompt, request);

        try (GGUFModel two = new GGUFModel(model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(two, cfg)) {
            BatchSequence a = engine.submit(prompt, request,
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.drain();
            assertDone(a);
            assertResultsMatch(ref, a.result());
            BatchStats stats = engine.stats();
            assertTrue("the drafter proposed", stats.specDrafted() > 0);
            assertTrue("the proposals were verified in rounds", stats.specRounds() > 0);
            assertTrue("accepted cannot exceed drafted", stats.specAccepted() <= stats.specDrafted());
        }
    }

    @Test
    public void aSingleKeyWatermarkedRequestRefusesSpeculativeDecodingByName() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig cfg = config(64).build();
        GenerationRequest request = GenerationRequest.builder().maxNewTokens(N_NEW)
                .sampler(SamplerConfig.greedyDecoding().toBuilder()
                        .watermark(com.eonmux.neuralfree.gguf.engine.watermark.WatermarkConfig.builder(7L)
                                .build())
                        .build())
                .build();
        try (GGUFModel two = new GGUFModel(model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(two, cfg, new ScriptedDrafter(new int[]{1}, false),
                     new BatchSpec(3, 0))) {
            IllegalArgumentException refused = assertThrows(IllegalArgumentException.class,
                    () -> engine.submit(tokens(), request,
                            (com.eonmux.neuralfree.gguf.TokenCallback) null));
            assertTrue(refused.getMessage(), refused.getMessage().contains("allow_target_only_watermarking"));
        }
    }

    @Test
    public void aDrafterProposingTheGreedyContinuationLosesAtMostTheStoppingPosition() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig cfg = config(64).build();
        // a budget past the free context: the round lengths follow the context, not the budget
        GenerationRequest request = request(64);
        int[] prompt = tokens();
        GenerationResult ref = reference(model, 64, prompt, request);

        try (GGUFModel two = new GGUFModel(model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(two, cfg,
                     new ScriptedDrafter(ref.tokens(), false), new BatchSpec(3, 0))) {
            BatchSequence a = engine.submit(prompt, request,
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.drain();
            assertDone(a);
            assertResultsMatch(ref, a.result());
            ScriptedDrafter drafter = lastDrafter;
            BatchStats stats = engine.stats();
            assertTrue(drafter.drafted > 0);
            assertEquals("the engine reported the drafter's own counters",
                    drafter.drafted, stats.specDrafted());
            assertEquals(drafter.accepted, stats.specAccepted());
            // every proposal matched the chain's sample except at most the one position the round stopped on
            assertTrue("accepted " + drafter.accepted + " of drafted " + drafter.drafted + " over "
                            + stats.specRounds() + " rounds",
                    drafter.accepted >= drafter.drafted - stats.specRounds());
            assertTrue(drafter.accepted > 0);
        }
    }

    @Test
    public void aDrafterProposingWrongTokensIsRejectedWithoutChangingTheResult() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig cfg = config(64).build();
        GenerationRequest request = request(N_NEW);
        int[] prompt = tokens();
        GenerationResult ref = reference(model, 64, prompt, request);

        try (GGUFModel two = new GGUFModel(model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(two, cfg,
                     new ScriptedDrafter(ref.tokens(), true), new BatchSpec(3, 0))) {
            BatchSequence a = engine.submit(prompt, request,
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.drain();
            assertDone(a);
            // every round rejected at its first draft position: the paged rollback left the plain result
            assertResultsMatch(ref, a.result());
            ScriptedDrafter drafter = lastDrafter;
            assertTrue(drafter.drafted > 0);
            assertEquals(0, drafter.accepted);
            BatchStats stats = engine.stats();
            assertEquals(drafter.drafted, stats.specDrafted());
            assertEquals(0, stats.specAccepted());
        }
    }

    @Test
    public void aCallerDrafterIsNotClosedByTheEngineAndADraftMinOfThreeVerifiesFullProposals() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig cfg = config(64).build();
        GenerationRequest request = request(N_NEW);
        int[] prompt = tokens();
        GenerationResult ref = reference(model, 64, prompt, request);
        ScriptedDrafter drafter = new ScriptedDrafter(ref.tokens(), false);

        try (GGUFModel two = new GGUFModel(model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(two, cfg, drafter, new BatchSpec(3, 3))) {
            BatchSequence a = engine.submit(prompt, request,
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.drain();
            assertDone(a);
            // proposals of three tokens verify; shorter ones decode plain, and the result is unchanged either way
            assertResultsMatch(ref, a.result());
            BatchStats stats = engine.stats();
            assertEquals(drafter.drafted, stats.specDrafted());
            assertEquals(drafter.accepted, stats.specAccepted());
            assertFalse("the caller's drafter stays open", drafter.closed);
        }
    }

    @Test
    public void aRequestAdapterRunsOnTheVerifyRowsBesideABareSequence() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        Path adapter = LoraAdapterFiles.llama(1000, 4, 8.0f);
        EngineConfig cfg = config(700).draftType(EngineConfig.DraftType.NGRAM).speculativeOptions(SpeculativeOptions
                .builder().ngram(NgramDrafter.Options.builder(NgramDrafter.Kind.SIMPLE).sizeN(3).sizeM(8).build())
                .build()).loraConfig(LoraConfig.builder().maxAdapters(2).maxRank(64).build()).build();
        int[] prompt = lookupPrompt();
        GenerationRequest adaptedReq = GenerationRequest.builder().maxNewTokens(N_NEW)
                .sampler(SamplerConfig.greedyDecoding()).lora(LoraSelection.of("p")).build();
        GenerationRequest plainReq = request(N_NEW);
        GenerationResult refAdapted = adapterReference(model, 700, prompt, adaptedReq, adapter.toString(), 3.0f, false);
        GenerationResult refPlain = reference(model, 700, prompt, plainReq);
        assertFalse("the adapter must change the text",
                Arrays.equals(refAdapted.tokens(), refPlain.tokens()));

        try (GGUFModel two = new GGUFModel(model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(two, cfg)) {
            two.loraRegistry().load("p", adapter.toString(), 3.0f);
            BatchSequence a = engine.submit(prompt, adaptedReq, (com.eonmux.neuralfree.gguf.TokenCallback) null);
            BatchSequence b = engine.submit(prompt, plainReq, (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.drain();
            assertDone(a);
            assertDone(b);
            assertResultsMatch(refAdapted, a.result());
            assertResultsMatch(refPlain, b.result());
            BatchStats stats = engine.stats();
            assertTrue("the drafter proposed", stats.specDrafted() > 0);
            assertTrue("the proposals were verified in rounds", stats.specRounds() > 0);
        }
    }

    @Test
    public void aStartupAloraRunsOnTheVerifyRowsOfASpeculatingSequence() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        Path alora = LoraAdapterFiles.activated(3000, 4, 8.0f, 4, 5);
        EngineConfig cfg = config(700).draftType(EngineConfig.DraftType.NGRAM).speculativeOptions(SpeculativeOptions
                .builder().ngram(NgramDrafter.Options.builder(NgramDrafter.Kind.SIMPLE).sizeN(3).sizeM(8).build())
                .build()).addLora(alora.toString(), 3.0f).build();
        int[] prompt = lookupPrompt(); // ends on the (4, 5) invocation pair
        GenerationRequest plainReq = request(N_NEW);
        GenerationResult ref = adapterReference(model, 700, prompt, plainReq, alora.toString(), 3.0f, true);

        try (GGUFModel two = new GGUFModel(model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(two, cfg)) {
            BatchSequence a = engine.submit(prompt, plainReq, (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.drain();
            assertDone(a);
            assertResultsMatch(ref, a.result());
            BatchStats stats = engine.stats();
            assertTrue("the drafter proposed", stats.specDrafted() > 0);
            assertTrue("the proposals were verified in rounds", stats.specRounds() > 0);
        }
    }

    /**
     * The reference: the same prompt and request through a single-sequence loop over a model that carries
     * {@code adapter} — per request under {@code p} when {@code startup} is false, at startup otherwise.
     */
    private static GenerationResult adapterReference(Path model, int nCtx, int[] prompt, GenerationRequest request,
                                                     String adapter, float scale, boolean startup) throws IOException {
        MemoryConfig mc = MemoryConfig.builder().nCtx(nCtx).ubatch(8)
                .typeK(KvDType.F32).typeV(KvDType.F32).build();
        EngineConfig cfg = startup
                ? EngineConfig.builder().memory(mc).addLora(adapter, scale).build()
                : EngineConfig.builder().memory(mc)
                        .loraConfig(LoraConfig.builder().maxAdapters(2).maxRank(64).build()).build();
        try (GGUFModel one = new GGUFModel(model.toString(), cfg)) {
            if (!startup) {
                one.loraRegistry().load("p", adapter, scale);
            }
            return GenerationLoop.forModel(one, cfg).generate(prompt, request,
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
        }
    }

    @Test
    public void aCancellationAndTheContextEndStillStopTheSequence() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig cfg = config(64).draftType(EngineConfig.DraftType.NGRAM).build();
        try (GGUFModel two = new GGUFModel(model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(two, cfg)) {
            GenerationRequest cancelled = GenerationRequest.builder()
                    .maxNewTokens(N_NEW)
                    .sampler(SamplerConfig.greedyDecoding())
                    .cancellation(() -> true)
                    .build();
            BatchSequence a = engine.submit(tokens(), cancelled,
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.drain();
            assertEquals(BatchSequence.State.CANCELLED, a.state());
            assertEquals(StopReason.CANCELLED, a.result().stopReason());

            // a 60-token prompt leaves 4 positions: the rounds bound to the free context and stop context-full
            int[] longPrompt = new int[60];
            for (int i = 0; i < longPrompt.length; i++) {
                longPrompt[i] = 4 + (i % 5) * 7;
            }
            BatchSequence b = engine.submit(longPrompt, request(64),
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.drain();
            assertDone(b);
            assertEquals(StopReason.CONTEXT_FULL, b.result().stopReason());
        }
    }

    @Test
    public void anOversizePromptStillRefusesAtSubmitWithASpeculativeEngine() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig cfg = config(64).draftType(EngineConfig.DraftType.NGRAM).build();
        try (GGUFModel two = new GGUFModel(model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(two, cfg)) {
            int[] oversize = new int[65];
            Arrays.fill(oversize, 4);
            assertThrows(ContextFullException.class, () -> engine.submit(oversize, request(N_NEW),
                    (com.eonmux.neuralfree.gguf.TokenCallback) null));
            assertTrue(engine.idle());
        }
    }

    @Test
    public void theConfiguredNgramSimpleSizesReachTheBatchDrafter() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        // a 64-token lookup n-gram never recurs in the prompt's distinct triples, so nothing is drafted
        EngineConfig cfg = config(700).draftType(EngineConfig.DraftType.NGRAM)
                .speculativeOptions(SpeculativeOptions.builder().ngram(NgramDrafter.Options
                        .builder(NgramDrafter.Kind.SIMPLE).sizeN(64).sizeM(64).build()).build())
                .build();
        GenerationRequest request = request(N_NEW);
        int[] prompt = lookupPrompt();
        GenerationResult ref = reference(model, 700, prompt, request);
        try (GGUFModel two = new GGUFModel(model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(two, cfg)) {
            BatchSequence a = engine.submit(prompt, request, (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.drain();
            assertDone(a);
            assertResultsMatch(ref, a.result());
            assertEquals("the configured lookup n-gram finds no match", 0, engine.stats().specDrafted());
        }
    }

    @Test
    public void theSpecBoundsValidateAtConstruction() {
        assertThrows(IllegalArgumentException.class, () -> new BatchSpec(0, 0));
        assertThrows(IllegalArgumentException.class, () -> new BatchSpec(3, -1));
        assertThrows(IllegalArgumentException.class, () -> new BatchSpec(3, 4));
        assertEquals(3, new BatchSpec(3, 0).draftMax());
        assertEquals(0, new BatchSpec(3, 0).draftMin());
    }

    @Test
    public void thePerSequenceLookupDrafterKeepsTheSimpleRule() throws IOException {
        RecordingMemory memory = new RecordingMemory(2);
        memory.history = new int[]{1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4};
        EngineConfig cfg = EngineConfig.builder().draftType(EngineConfig.DraftType.NGRAM)
                .speculativeOptions(SpeculativeOptions.builder().ngram(NgramDrafter.Options
                        .builder(NgramDrafter.Kind.SIMPLE).sizeN(3).sizeM(8).build()).build())
                .build();
        LookupDrafterFactory factory = LookupDrafterFactory.open(cfg);
        BatchLookupDrafter drafter = new BatchLookupDrafter(factory, memory);
        // the pattern (3, 4, sampled 5) matches at position 7: the tokens that followed are 1, 2, 3, 4
        assertArrayEquals(new int[]{1, 2, 3}, drafter.draft(0, 5, 3));
        assertArrayEquals(new int[]{1, 2, 3, 4}, drafter.draft(0, 5, 8));
        // no match: the pattern (3, 4, 9) never occurred
        assertEquals(0, drafter.draft(0, 9, 3).length);
        // a history too short for the lookup drafts nothing
        memory.history = new int[]{1, 2};
        assertEquals(0, drafter.draft(0, 3, 3).length);
        assertEquals(0, drafter.draft(0, -1, 3).length);
        // sequence bounds and construction validation
        assertThrows(IndexOutOfBoundsException.class, () -> drafter.draft(5, 1, 3));
        assertThrows(IndexOutOfBoundsException.class, () -> drafter.accept(5, 0));
        assertThrows(IllegalArgumentException.class, () -> drafter.accept(0, -1));
        assertThrows(IllegalArgumentException.class, () -> new BatchLookupDrafter(
                LookupDrafterFactory.open(EngineConfig.builder().build()), memory));
        assertThrows(NullPointerException.class, () -> new BatchLookupDrafter(factory, null));
        drafter.close();
        assertThrows(IllegalStateException.class, () -> drafter.draft(0, 5, 3));
    }

    /** The last drafter a test built, for the assertions on its counters. */
    private static ScriptedDrafter lastDrafter;

    /**
     * Drafts the greedy continuation the reference run produced: with {@code wrong} the proposals differ from the
     * target's own next token at every position, so every round rejects at its first draft position. The cursor
     * follows the sequence's stream through {@code accept}.
     */
    private static final class ScriptedDrafter implements BatchDrafter {

        private final int[] continuation;
        private final boolean wrong;
        private final int[] cursor = new int[8];
        private int drafted;
        private int accepted;
        private boolean closed;

        ScriptedDrafter(int[] continuation, boolean wrong) {
            this.continuation = continuation;
            this.wrong = wrong;
            lastDrafter = this;
        }

        @Override
        public int[] draft(int seq, int lastToken, int n) {
            // re-anchor the cursor on the token the target just sampled (the stream is the reference's)
            while (cursor[seq] < continuation.length && continuation[cursor[seq]] != lastToken) {
                cursor[seq]++;
            }
            int[] out = new int[Math.min(n, continuation.length - cursor[seq] - 1)];
            for (int i = 0; i < out.length; i++) {
                int truth = continuation[cursor[seq] + 1 + i];
                out[i] = wrong ? (truth == 4 ? 5 : 4) : truth;
            }
            drafted += out.length;
            return out;
        }

        @Override
        public void accept(int seq, int nAccepted) {
            accepted += nAccepted;
            cursor[seq] += 1 + nAccepted;
        }

        @Override
        public void close() {
            closed = true;
        }
    }

    /** A {@link SequenceMemory} stub holding one scripted history: the drafter reads only its token record. */
    private static final class RecordingMemory implements SequenceMemory {

        final int maxSequences;
        int[] history = new int[0];

        RecordingMemory(int maxSequences) {
            this.maxSequences = maxSequences;
        }

        @Override public int maxSequences() {
            return maxSequences;
        }

        @Override public SequenceState sequence(int seq) {
            SequenceState state = new SequenceState(4096);
            state.append(history, 0, history.length);
            return state;
        }

        @Override public int posMin(int seq) {
            return 0;
        }

        @Override public int posMax(int seq) {
            return history.length - 1;
        }

        @Override public String removeRefusal(int seq, int p0, int p1) {
            return null;
        }

        @Override public void remove(int seq, int p0, int p1) {
            throw new UnsupportedOperationException();
        }

        @Override public String removeKeepRecurrentRefusal(int seq, int p0) {
            return null;
        }

        @Override public void removeKeepRecurrent(int seq, int p0) {
            throw new UnsupportedOperationException();
        }

        @Override public String copyRefusal() {
            return null;
        }

        @Override public void copy(int src, int dst, int p0, int p1) {
            throw new UnsupportedOperationException();
        }

        @Override public void keep(int seq) {
            throw new UnsupportedOperationException();
        }

        @Override public void clear() {
            history = new int[0];
        }

        @Override public String shiftRefusal() {
            return "no shift";
        }

        @Override public void add(int seq, int p0, int p1, int delta) {
            throw new UnsupportedOperationException();
        }

        @Override public String divRefusal() {
            return "no division";
        }

        @Override public void div(int seq, int p0, int p1, int d) {
            throw new UnsupportedOperationException();
        }

        @Override public String moveRowsRefusal(int seq, int[] srcPos, int[] dstPos) {
            return "no compaction";
        }

        @Override public void moveRows(int seq, int[] srcPos, int[] dstPos) {
            throw new UnsupportedOperationException();
        }

        @Override public int freeCells() {
            return 0;
        }

        @Override public int totalCells() {
            return 0;
        }
    }
}

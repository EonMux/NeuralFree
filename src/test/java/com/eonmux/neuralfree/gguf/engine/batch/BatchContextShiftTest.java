package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.engine.EmbedsPrompt;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.GenerationLoop;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.engine.StopReason;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The batch engine's slot context shift ({@link ContextShift}, llama.cpp server-context.cpp:3783-3824): a
 * decoding sequence whose next position would leave the context drops the positions the request's
 * {@code n_keep}/{@code n_discard} spare and keeps decoding in its slot, computing what the single-sequence
 * {@code GenerationLoop} shift computes for the same prompt and request. The keep/discard arithmetic follows
 * llama.cpp's server, the shifted sequence survives a preemption's recompute over its compacted history, the
 * shift off still stops with {@code CONTEXT_FULL}, and a request naming {@code n_keep}/{@code n_discard} on a
 * sequence the shift cannot serve refuses at submit.
 */
public class BatchContextShiftTest {

    private static final int N_CTX = 24;
    private static final int N_TOKENS = 12;
    /** The synthetic model's BOS id (SyntheticModels.llama). */
    private static final int BOS = 1;

    private static EngineConfig config(boolean contextShift, boolean prefixReuse) {
        MemoryConfig.Builder mc = MemoryConfig.builder().nCtx(N_CTX).ubatch(8).parallel(2)
                .typeK(KvDType.F32).typeV(KvDType.F32).contextShift(contextShift).kvPageSize(8);
        if (!prefixReuse) {
            mc = mc.prefixReuse(false);
        }
        return EngineConfig.builder().memory(mc.build()).build();
    }

    /** The single-sequence reference runs the contiguous memory: no {@code parallel}, hence no paging. */
    private static EngineConfig loopConfig(boolean contextShift) {
        MemoryConfig mc = MemoryConfig.builder().nCtx(N_CTX).ubatch(8)
                .typeK(KvDType.F32).typeV(KvDType.F32).contextShift(contextShift).build();
        return EngineConfig.builder().memory(mc).build();
    }

    /** Greedy decoding over a bounded budget. */
    private static GenerationRequest request(int maxNewTokens) {
        return GenerationRequest.builder()
                .maxNewTokens(maxNewTokens)
                .sampler(SamplerConfig.greedyDecoding())
                .build();
    }

    /** The tokens a sequence runs (deterministic ids inside the synthetic vocab, none of them the BOS). */
    private static int[] tokens(int seed) {
        int[] t = new int[N_TOKENS];
        for (int i = 0; i < t.length; i++) {
            t[i] = 4 + (seed * 7 + i * 13) % 250;
        }
        return t;
    }

    /** The reference: the same prompt and request through the single-sequence GenerationLoop with its shift. */
    private static GenerationResult reference(Path model, int[] prompt, GenerationRequest request) throws IOException {
        EngineConfig cfg = loopConfig(true);
        try (GGUFModel one = new GGUFModel(model.toString(), cfg)) {
            return GenerationLoop.forModel(one, cfg).generate(prompt, request,
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
        }
    }

    // ------------------------------------------------------------------ the llama.cpp arithmetic

    @Test
    public void keepAndDiscardCountsFollowTheLlamaCppServerRules() {
        GenerationRequest plain = request(1); // n_keep 0, n_discard 0

        // no BOS in the prompt: nothing is kept, half of the stored positions leave
        assertEquals(0, ContextShift.keepCount(plain, N_TOKENS, tokens(1)[0], BOS, N_CTX, N_CTX));
        assertEquals(N_CTX / 2, ContextShift.discardCount(plain, 0, N_CTX));

        // a leading BOS is kept on top of the request's count (llama.cpp's add_bos_token)
        assertEquals(1, ContextShift.keepCount(plain, N_TOKENS, BOS, BOS, N_CTX, N_CTX));
        assertEquals((N_CTX - 1) / 2, ContextShift.discardCount(plain, 1, N_CTX));

        // the request's n_keep holds, -1 keeping the whole prompt
        assertEquals(4, ContextShift.keepCount(request(1).toBuilder().nKeep(4).build(), N_TOKENS,
                tokens(1)[0], BOS, N_CTX, N_CTX));
        assertEquals(N_TOKENS, ContextShift.keepCount(request(1).toBuilder().nKeep(-1).build(), N_TOKENS,
                tokens(1)[0], BOS, N_CTX, N_CTX));

        // n_keep never passes the n_ctx - 4 margin nor the stored positions
        assertEquals(N_CTX - 4, ContextShift.keepCount(request(1).toBuilder().nKeep(100).build(), N_TOKENS,
                tokens(1)[0], BOS, N_CTX, N_CTX));
        assertEquals(N_CTX - 6, ContextShift.keepCount(request(1).toBuilder().nKeep(100).build(), N_TOKENS,
                tokens(1)[0], BOS, N_CTX, N_CTX - 6));

        // the request's n_discard holds, clamped to leave the last row in place; 0 takes half
        assertEquals(3, ContextShift.discardCount(request(1).toBuilder().nDiscard(3).build(), 4, N_CTX));
        assertEquals(N_CTX - 1, ContextShift.discardCount(request(1).toBuilder().nDiscard(1000).build(), 0, N_CTX));
        assertEquals(0, ContextShift.discardCount(plain, N_CTX - 1, N_CTX));
    }

    // ------------------------------------------------------------------ parity with the single-sequence shift

    @Test
    public void aFullContextShiftsAndMatchesTheSingleSequenceShift() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        int[] prompt = tokens(1);
        int budget = 30; // fills the 24-position context twice on the default keep/discard counts
        GenerationRequest shifted = request(budget);
        GenerationResult ref = reference(model, prompt, shifted);
        assertTrue("the reference generation crossed the context boundary (shifted): " + ref.tokenCount(),
                ref.tokenCount() > N_CTX - prompt.length);

        GenerationResult result;
        try (GGUFModel two = new GGUFModel(model.toString(), config(true, false));
             BatchEngine engine = BatchEngine.forModel(two, config(true, false))) {
            BatchSequence a = engine.submit(prompt, shifted,
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.drain();
            assertEquals(BatchSequence.State.DONE, a.state());
            assertTrue("the shifted sequence reports truncated", a.truncated());
            assertTrue("two fills, two shifts, got " + engine.stats().contextShifts(),
                    engine.stats().contextShifts() >= 2);
            result = a.result();
        }
        assertEquals(ref.stopReason(), result.stopReason());
        assertArrayEquals(ref.tokens(), result.tokens());
        assertEquals(ref.text(), result.text());
    }

    /**
     * The request's {@code n_keep} and {@code n_discard} steer the shift: a kept whole prompt discards fewer
     * positions than the default, so the generations diverge, and a small {@code n_discard} shifts at every
     * second token.
     */
    @Test
    public void perRequestNKeepAndNDiscardSteerTheShift() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        int[] prompt = tokens(2);
        int budget = 20;
        GenerationResult[] results = new GenerationResult[3];
        long[] shifts = new long[3];
        GenerationRequest[] requests = {
                request(budget),
                request(budget).toBuilder().nKeep(N_TOKENS).build(),
                request(budget).toBuilder().nKeep(0).nDiscard(2).build(),
        };
        for (int i = 0; i < requests.length; i++) {
            try (GGUFModel two = new GGUFModel(model.toString(), config(true, false));
                 BatchEngine engine = BatchEngine.forModel(two, config(true, false))) {
                BatchSequence a = engine.submit(prompt, requests[i],
                        (com.eonmux.neuralfree.gguf.TokenCallback) null);
                engine.drain();
                assertTrue(String.valueOf(a.error()), a.state() == BatchSequence.State.DONE);
                assertTrue("run " + i + " shifted", a.truncated());
                assertTrue("run " + i + " crossed the boundary", a.result().tokenCount() == budget);
                results[i] = a.result();
                shifts[i] = engine.stats().contextShifts();
            }
        }
        assertNotEquals(results[0].text(), results[1].text(),
                "keeping the whole prompt discards other positions than the default");
        assertTrue("n_discard 2 shifts far more often than the default half: "
                + shifts[2] + " vs " + shifts[0], shifts[2] > shifts[0]);
    }

    // ------------------------------------------------------------------ shift off, refusals, preemption

    @Test
    public void withTheShiftOffAFullContextStillStopsWithContextFull() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        try (GGUFModel two = new GGUFModel(model.toString(), config(false, false));
             BatchEngine engine = BatchEngine.forModel(two, config(false, false))) {
            BatchSequence a = engine.submit(tokens(1), request(30),
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.drain();
            assertEquals(BatchSequence.State.DONE, a.state());
            assertFalse(a.truncated());
            assertEquals(StopReason.CONTEXT_FULL, a.result().stopReason());
            assertEquals(N_CTX - N_TOKENS + 1, a.result().tokenCount());
            assertEquals(0, engine.stats().contextShifts());
        }
    }

    /**
     * A preemption before any shift recomputes the sequence exactly: the replayed prompt and fed decodes are the
     * rows the live run held, so the tokens carry on identical.
     */
    @Test
    public void aPreemptedSequenceRecomputesExactly() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        int[] prompt = tokens(3);
        GenerationRequest request = request(20);
        GenerationResult plain;
        try (GGUFModel two = new GGUFModel(model.toString(), config(true, false));
             BatchEngine engine = BatchEngine.forModel(two, config(true, false))) {
            BatchSequence a = engine.submit(prompt, request, (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.drain();
            plain = a.result();
        }
        GenerationResult afterPreemption;
        try (GGUFModel two = new GGUFModel(model.toString(), config(true, false));
             BatchEngine engine = BatchEngine.forModel(two, config(true, false))) {
            BatchSequence b = engine.submit(prompt, request, (com.eonmux.neuralfree.gguf.TokenCallback) null);
            // pump into the decodes, before the context fills: the replay covers the plain prompt and fed tokens
            for (int i = 0; i < 10; i++) {
                engine.pump();
            }
            assertTrue("the preemption runs before the first shift", engine.stats().contextShifts() == 0);
            engine.preemptAll();
            engine.drain();
            assertTrue(String.valueOf(b.error()), b.state() == BatchSequence.State.DONE);
            afterPreemption = b.result();
        }
        assertArrayEquals(plain.tokens(), afterPreemption.tokens());
        assertEquals(plain.text(), afterPreemption.text());
    }

    /**
     * A preemption after a shift survives: the recompute replays the compacted history, the shifted sequence
     * carries its budget out and reports its shift.
     */
    @Test
    public void aShiftedSequenceSurvivesAPreemption() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        try (GGUFModel two = new GGUFModel(model.toString(), config(true, false));
             BatchEngine engine = BatchEngine.forModel(two, config(true, false))) {
            BatchSequence b = engine.submit(tokens(3), request(30),
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            for (int i = 0; i < 200 && engine.stats().contextShifts() == 0; i++) {
                engine.pump();
            }
            assertTrue("the run shifted before the preemption", engine.stats().contextShifts() >= 1);
            engine.preemptAll();
            engine.drain();
            assertTrue(String.valueOf(b.error()), b.state() == BatchSequence.State.DONE);
            assertTrue("the shifted sequence reports truncated", b.truncated());
            assertEquals(30, b.result().tokenCount());
            assertTrue("generation continued to the next shift: " + engine.stats().contextShifts(),
                    engine.stats().contextShifts() >= 2);
        }
    }

    @Test
    public void aShiftRunsBesideThePrefixCacheAndItsSharedPages() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        int[] prompt = tokens(4);
        GenerationRequest keepsPrompt = request(24).toBuilder().nKeep(N_TOKENS).build();
        try (GGUFModel two = new GGUFModel(model.toString(), config(true, true));
             BatchEngine engine = BatchEngine.forModel(two, config(true, true))) {
            BatchSequence a = engine.submit(prompt, keepsPrompt, (com.eonmux.neuralfree.gguf.TokenCallback) null);
            // a second request adopts the first one's cached prompt prefix and decodes beside the shifting one
            BatchSequence b = engine.submit(Arrays.copyOf(prompt, prompt.length), request(4),
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.drain();
            assertTrue(String.valueOf(a.error()), a.state() == BatchSequence.State.DONE);
            assertTrue(String.valueOf(b.error()), b.state() == BatchSequence.State.DONE);
            assertTrue("the shifting sequence reports truncated", a.truncated());
            assertTrue("the twin adopted the cached prefix", b.reusedPrefix() > 0);
        }
    }

    @Test
    public void aShiftedNameOnAnUnservableSequenceRefusesAtSubmit() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig paged = config(true, false);
        GenerationRequest keeps = request(4).toBuilder().nKeep(2).build();
        try (GGUFModel m = new GGUFModel(model.toString(), paged)) {
            float[][] rows;
            try (GGUFFileParser parser = new GGUFFileParser(model.toString())) {
                GGUFTensorInfo table = parser.getTensor("token_embd.weight");
                float[] flat = parser.readTensorDataAsFloat(table);
                int nEmbd = m.embeddingLength();
                rows = new float[1][];
                rows[0] = Arrays.copyOfRange(flat, 5 * nEmbd, 6 * nEmbd);
            }
            EmbedsPrompt embeds = EmbedsPrompt.ofRows(rows);
            try (BatchEngine engine = BatchEngine.forModel(m, paged)) {
                UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class,
                        () -> engine.submit(embeds, keeps, (com.eonmux.neuralfree.gguf.TokenCallback) null));
                assertTrue(e.getMessage(), e.getMessage().contains("n_keep"));
            }
        }
    }

    @Test
    public void theRequestBuilderRefusesOutOfRangeShiftCounts() {
        assertThrows(IllegalArgumentException.class, () -> request(1).toBuilder().nKeep(-2).build());
        assertThrows(IllegalArgumentException.class, () -> request(1).toBuilder().nDiscard(-1).build());
        assertEquals(-1, request(1).toBuilder().nKeep(-1).build().nKeep());
    }
}

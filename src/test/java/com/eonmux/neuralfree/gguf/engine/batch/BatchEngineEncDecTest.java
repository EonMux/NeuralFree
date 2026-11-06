package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.TokenCallback;
import com.eonmux.neuralfree.gguf.arch.SupportClass;
import com.eonmux.neuralfree.gguf.engine.ContextFullException;
import com.eonmux.neuralfree.gguf.engine.EmbedsPrompt;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.GenerationLoop;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.engine.SequenceEncoder;
import com.eonmux.neuralfree.gguf.engine.SpeculativeOptions;
import com.eonmux.neuralfree.gguf.engine.draft.NgramDrafter;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Path;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Batched generation of an encoder-decoder model: a {@link BatchEngine} over the synthetic T5 on paged CPU memory
 * encodes every admitted request's prompt into its sequence's own cross caches and decodes the sequences together,
 * and every finished sequence carries the same tokens, text and prompt count as the same prompt run alone through
 * {@link GenerationLoop} (which encodes the prompt and decodes from the decoder start token). A request admitted
 * into a freed slot encodes its own source there, a preempted sequence re-encodes its source when it re-admits, the
 * retired sequences leave no cached prefix behind, and requests the encoder cannot take refuse at submit.
 */
public class BatchEngineEncDecTest {

    private static final int N_NEW = 6;

    private static EngineConfig config(boolean paged) {
        MemoryConfig.Builder mc = MemoryConfig.builder().nCtx(64).ubatch(16).typeK(KvDType.F32).typeV(KvDType.F32);
        if (paged) {
            mc = mc.parallel(2);
        }
        return EngineConfig.builder().memory(mc.build()).build();
    }

    private static GenerationRequest request() {
        return request(N_NEW);
    }

    private static GenerationRequest request(int maxNewTokens) {
        return GenerationRequest.builder().maxNewTokens(maxNewTokens).sampler(SamplerConfig.greedyDecoding()).build();
    }

    /** An encoder source inside the synthetic vocabulary, clear of the special ids. */
    private static int[] source(int seed, int n) {
        int[] t = new int[n];
        for (int i = 0; i < n; i++) {
            t[i] = 4 + (seed * 11 + i * 17) % 250;
        }
        return t;
    }

    private static GenerationResult reference(Path model, int[] source) throws IOException {
        return reference(model, source, request());
    }

    private static GenerationResult reference(Path model, int[] source, GenerationRequest request)
            throws IOException {
        EngineConfig cfg = config(false);
        try (GGUFModel one = new GGUFModel(model.toString(), cfg)) {
            return GenerationLoop.forModel(one, cfg).generate(source, request, (TokenCallback) null);
        }
    }

    private static void assertMatches(String what, GenerationResult reference, BatchSequence batched) {
        assertTrue(what + ": expected DONE but was " + batched.state()
                        + (batched.error() == null ? "" : ": " + batched.error()),
                batched.state() == BatchSequence.State.DONE);
        GenerationResult r = batched.result();
        assertNotNull(r);
        assertEquals(what + ": stop reason", reference.stopReason(), r.stopReason());
        assertArrayEquals(what + ": token ids", reference.tokens(), r.tokens());
        assertEquals(what + ": text", reference.text(), r.text());
        assertEquals(what + ": the prompt phase counts the encoder source", reference.timings().promptTokens(),
                r.timings().promptTokens());
    }

    @Test
    public void concurrentSequencesOverTheirOwnSourcesMatchTheGenerationLoopRuns() throws IOException {
        Path model = SyntheticModels.t5().writeTempFile();
        int[] srcA = source(1, 7);
        int[] srcB = source(2, 12);
        int[] srcC = source(3, 4);
        GenerationResult refA = reference(model, srcA);
        GenerationResult refB = reference(model, srcB);
        GenerationResult refC = reference(model, srcC);
        assertTrue(refA.tokenCount() > 0 && refB.tokenCount() > 0 && refC.tokenCount() > 0);
        assertEquals(srcB.length, refB.timings().promptTokens());

        EngineConfig paged = config(true);
        try (GGUFModel two = new GGUFModel(model.toString(), paged);
             BatchEngine engine = BatchEngine.forModel(two, paged)) {
            assertEquals(SupportClass.ENC_DEC, two.spec().supportClass());
            assertEquals(2, engine.maxSequences());
            BatchSequence a = engine.submit(srcA, request(), (TokenCallback) null);
            BatchSequence b = engine.submit(srcB, request(), (TokenCallback) null);
            BatchSequence c = engine.submit(srcC, request(), (TokenCallback) null);
            engine.drain();
            assertMatches("sequence A", refA, a);
            assertMatches("sequence B", refB, b);
            assertMatches("the request admitted into a freed slot encodes its own source", refC, c);
            assertTrue(c.seq() == a.seq() || c.seq() == b.seq());
            assertEquals("the engine counts the encoder sources as prompt tokens",
                    srcA.length + srcB.length + srcC.length, engine.stats().promptTokens());
            for (int seq = 0; seq < engine.maxSequences(); seq++) {
                assertEquals("a retired encoder-decoder sequence leaves no cached prefix", -1,
                        two.forwardBackend().memory().sequences().posMax(seq));
            }
        }
    }

    /**
     * Preempted sequences re-encode their sources when they re-admit: the cross caches are overwritten while they
     * wait (another user of the slots), and the recomputed sequences still finish with their reference results.
     */
    @Test
    public void aPreemptedSequenceReEncodesItsSourceWhenItReAdmits() throws IOException {
        Path model = SyntheticModels.t5().writeTempFile();
        int[] srcA = source(4, 9);
        int[] srcB = source(5, 6);
        GenerationResult refA = reference(model, srcA);
        GenerationResult refB = reference(model, srcB);

        EngineConfig paged = config(true);
        try (GGUFModel two = new GGUFModel(model.toString(), paged);
             BatchEngine engine = BatchEngine.forModel(two, paged)) {
            BatchSequence a = engine.submit(srcA, request(), (TokenCallback) null);
            BatchSequence b = engine.submit(srcB, request(), (TokenCallback) null);
            while (engine.stats().decodeSteps() < 2) {
                engine.pump();
            }
            engine.preemptAll();
            assertEquals(0, engine.running());
            SequenceEncoder encoder = two.forwardBackend().sequenceEncoder();
            encoder.encodeSequence(0, source(6, 5));
            encoder.encodeSequence(1, source(7, 5));
            engine.drain();
            assertTrue(engine.stats().preemptions() >= 2);
            assertTrue("the re-encoded sources count as recomputed tokens",
                    engine.stats().recomputedTokens() >= srcA.length + srcB.length);
            assertMatches("preempted sequence A", refA, a);
            assertMatches("preempted sequence B", refB, b);
        }
    }

    /** Prompt-lookup rounds over the decoder histories verify losslessly: the results equal the plain runs. */
    @Test
    public void speculativeRoundsOverTheDecoderHistoriesStayLossless() throws IOException {
        Path model = SyntheticModels.t5().writeTempFile();
        int longRun = 24;
        int[] srcA = source(11, 8);
        int[] srcB = source(12, 5);
        GenerationResult refA = reference(model, srcA, request(longRun));
        GenerationResult refB = reference(model, srcB, request(longRun));

        MemoryConfig mc = MemoryConfig.builder().nCtx(64).ubatch(16).typeK(KvDType.F32).typeV(KvDType.F32)
                .parallel(2).build();
        // sizes the short decoder histories match (llama.cpp's default 12 / 48 needs longer repeats)
        EngineConfig speculative = EngineConfig.builder().memory(mc).draftType(EngineConfig.DraftType.NGRAM)
                .speculativeOptions(SpeculativeOptions.builder().ngram(NgramDrafter.Options
                        .builder(NgramDrafter.Kind.SIMPLE).sizeN(3).sizeM(8).build()).build()).build();
        try (GGUFModel two = new GGUFModel(model.toString(), speculative);
             BatchEngine engine = BatchEngine.forModel(two, speculative)) {
            BatchSequence a = engine.submit(srcA, request(longRun), (TokenCallback) null);
            BatchSequence b = engine.submit(srcB, request(longRun), (TokenCallback) null);
            engine.drain();
            assertTrue("the decoder histories repeat, so the drafter proposes tokens",
                    engine.stats().specDrafted() > 0);
            assertMatches("speculative sequence A", refA, a);
            assertMatches("speculative sequence B", refB, b);
        }
    }

    @Test
    public void requestsTheEncoderCannotTakeRefuseAtSubmit() throws IOException {
        Path model = SyntheticModels.t5().writeTempFile();
        EngineConfig paged = config(true);
        try (GGUFModel two = new GGUFModel(model.toString(), paged);
             BatchEngine engine = BatchEngine.forModel(two, paged)) {
            int limit = two.forwardBackend().sequenceEncoder().maxSourceTokens();
            assertThrows(ContextFullException.class,
                    () -> engine.submit(source(8, limit + 1), request(), (TokenCallback) null));
            GenerationRequest logprobs = GenerationRequest.builder().maxNewTokens(N_NEW)
                    .sampler(SamplerConfig.greedyDecoding()).promptLogprobs(1).build();
            UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class,
                    () -> engine.submit(source(9, 4), logprobs, (TokenCallback) null));
            assertTrue(e.getMessage(), e.getMessage().contains("prompt_logprobs"));
            EmbedsPrompt embeds = EmbedsPrompt.ofRows(new float[][]{new float[two.spec().hparams().nEmbd()]});
            e = assertThrows(UnsupportedOperationException.class,
                    () -> engine.submit(embeds, request(), (TokenCallback) null));
            assertTrue(e.getMessage(), e.getMessage().contains("token prompts"));
            assertEquals("nothing refused was queued", 0, engine.queued());
        }
    }

    @Test
    public void anUnpagedMemoryStillRefusesTheEngineByName() throws IOException {
        Path model = SyntheticModels.t5().writeTempFile();
        EngineConfig contiguous = config(false);
        try (GGUFModel one = new GGUFModel(model.toString(), contiguous)) {
            UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class,
                    () -> BatchEngine.forModel(one, contiguous));
            assertTrue(e.getMessage(), e.getMessage().contains("paged memory"));
        }
    }
}

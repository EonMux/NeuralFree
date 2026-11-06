package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.TokenCallback;
import com.eonmux.neuralfree.gguf.arch.SupportClass;
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

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * The batch engine over the LFM2 short-convolution hybrid: two concurrent
 * sequences decode on the paged memory and carry the tokens of the same prompts run one at a time through
 * {@link GenerationLoop}, a cancellation stops them with {@code StopReason.CANCELLED}, and the next submission in a
 * cancelled slot starts from the zeroed conv state and matches its own reference.
 */
public class BatchEngineShortConvTest {

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
        return GenerationRequest.builder().maxNewTokens(maxNewTokens)
                .sampler(SamplerConfig.greedyDecoding()).build();
    }

    /** The tokens a sequence runs (deterministic ids inside the synthetic vocab, as BatchEngineTest). */
    private static int[] tokens(int seed) {
        int[] t = new int[N_TOKENS];
        for (int i = 0; i < t.length; i++) {
            t[i] = 4 + (seed * 7 + i * 13) % 250;
        }
        return t;
    }

    /** The reference: the same prompt and request through the single-sequence GenerationLoop. */
    private static GenerationResult reference(Path model, int[] prompt, GenerationRequest request) throws IOException {
        EngineConfig cfg = config(false).build();
        try (GGUFModel one = new GGUFModel(model.toString(), cfg)) {
            return GenerationLoop.forModel(one, cfg).generate(prompt, request, (TokenCallback) null);
        }
    }

    /** Pumps until nothing is queued or running, then once more over the idle engine (cancelAll's flag clears). */
    private static void pumpOut(BatchEngine engine) {
        while (engine.running() > 0 || engine.queued() > 0) {
            engine.pump();
        }
        engine.pump();
    }

    @Test
    public void twoConcurrentSequencesMatchTheSequentialGenerationLoopRuns() throws IOException {
        Path model = SyntheticModels.lfm2().writeTempFile();
        EngineConfig paged = config(true).build();
        GenerationRequest request = request(N_NEW);
        GenerationResult refA = reference(model, tokens(1), request);
        GenerationResult refB = reference(model, tokens(2), request);
        assertTrue(refA.tokenCount() > 0 && refB.tokenCount() > 0);

        try (GGUFModel two = new GGUFModel(model.toString(), paged);
             BatchEngine engine = BatchEngine.forModel(two, paged)) {
            assertEquals(SupportClass.GEN, two.spec().supportClass());
            assertEquals(2, engine.maxSequences());
            BatchSequence a = engine.submit(tokens(1), request, (TokenCallback) null);
            BatchSequence b = engine.submit(tokens(2), request, (TokenCallback) null);
            engine.drain();
            assertEquals("sequence a finished: " + a.error(), BatchSequence.State.DONE, a.state());
            assertEquals("sequence b finished: " + b.error(), BatchSequence.State.DONE, b.state());
            assertArrayEquals("sequence a matches the contiguous loop", refA.tokens(), a.result().tokens());
            assertArrayEquals("sequence b matches the contiguous loop", refB.tokens(), b.result().tokens());
        }
    }

    @Test
    public void aCancelledSlotServesTheNextSequenceFromAFreshConvState() throws IOException {
        Path model = SyntheticModels.lfm2().writeTempFile();
        EngineConfig paged = config(true).build();
        GenerationRequest request = request(N_NEW);
        GenerationResult ref = reference(model, tokens(4), request);

        try (GGUFModel two = new GGUFModel(model.toString(), paged);
             BatchEngine engine = BatchEngine.forModel(two, paged)) {
            BatchSequence a = engine.submit(tokens(1), request(64), (TokenCallback) null);
            BatchSequence b = engine.submit(tokens(2), request(64), (TokenCallback) null);
            while (engine.running() < 2) {
                engine.pump();
            }
            engine.cancelAll();
            pumpOut(engine);
            assertEquals(BatchSequence.State.CANCELLED, a.state());
            assertEquals(BatchSequence.State.CANCELLED, b.state());
            assertEquals(StopReason.CANCELLED, a.result().stopReason());
            assertTrue(engine.idle());

            BatchSequence c = engine.submit(tokens(4), request, (TokenCallback) null);
            engine.drain();
            assertEquals("the sequence after the cancel finished: " + c.error(), BatchSequence.State.DONE, c.state());
            assertArrayEquals("the reused slot started from a zeroed conv state", ref.tokens(), c.result().tokens());
        }
    }
}

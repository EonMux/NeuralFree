package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
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
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

/**
 * The engine sleep operations ({@link BatchEngine#cancelAll} and {@link BatchEngine#preemptAll}): a cancel stops
 * every running and queued submission with {@code StopReason.CANCELLED} and leaves the engine working for later
 * submissions, and a preempt of running sequences resumes them with exactly the results an uninterrupted run
 * produces, since the sleep of a server wakes its frozen requests by recomputing their histories.
 */
public class BatchEngineSleepOpsTest {

    private static final int N_TOKENS = 12;
    private static final int N_NEW = 8;

    private static EngineConfig pagedConfig() {
        return EngineConfig.builder().memory(MemoryConfig.builder().nCtx(64).ubatch(8)
                .typeK(KvDType.F32).typeV(KvDType.F32).parallel(2).build()).build();
    }

    private static GenerationRequest request(int maxNewTokens) {
        return GenerationRequest.builder().maxNewTokens(maxNewTokens)
                .sampler(SamplerConfig.greedyDecoding()).build();
    }

    private static int[] tokens(int seed) {
        int[] t = new int[N_TOKENS];
        for (int i = 0; i < t.length; i++) {
            t[i] = 4 + (seed * 7 + i * 13) % 250;
        }
        return t;
    }

    /** Pumps until nothing is queued or running, then once more over the idle engine (cancelAll's flag clears). */
    private static void pumpOut(BatchEngine engine) {
        while (engine.running() > 0 || engine.queued() > 0) {
            engine.pump();
        }
        engine.pump();
    }

    @Test
    public void cancelAllStopsRunningAndQueuedSubmissionsAndTheEngineKeepsWorking() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig paged = pagedConfig();
        try (GGUFModel two = new GGUFModel(model.toString(), paged);
             BatchEngine engine = BatchEngine.forModel(two, paged)) {
            BatchSequence a = engine.submit(tokens(1), request(64),
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            BatchSequence b = engine.submit(tokens(2), request(64),
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.pump();
            engine.pump();
            engine.pump();
            assertTrue("both sequences decode", engine.running() == 2);
            BatchSequence c = engine.submit(tokens(3), request(N_NEW),
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            assertEquals(BatchSequence.State.QUEUED, c.state());

            engine.cancelAll();
            pumpOut(engine);

            assertEquals(BatchSequence.State.CANCELLED, a.state());
            assertEquals(BatchSequence.State.CANCELLED, b.state());
            assertEquals(BatchSequence.State.CANCELLED, c.state());
            assertEquals(StopReason.CANCELLED, a.result().stopReason());
            assertEquals(StopReason.CANCELLED, c.result().stopReason());
            assertTrue(engine.idle());

            BatchSequence d = engine.submit(tokens(4), request(N_NEW),
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.drain();
            assertEquals("a submission after the cancel runs to its budget", BatchSequence.State.DONE, d.state());
            assertEquals(N_NEW, d.result().tokenCount());
        }
    }

    @Test
    public void cancelAllOnAnIdleEngineChangesNothing() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig paged = pagedConfig();
        try (GGUFModel two = new GGUFModel(model.toString(), paged);
             BatchEngine engine = BatchEngine.forModel(two, paged)) {
            assertTrue(engine.idle());
            engine.cancelAll();
            BatchSequence a = engine.submit(tokens(1), request(N_NEW),
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.drain();
            assertEquals(BatchSequence.State.DONE, a.state());
            assertEquals(N_NEW, a.result().tokenCount());
        }
    }

    @Test
    public void preemptAllResumesTheFrozenSequencesWithTheUninterruptedResults() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig paged = pagedConfig();
        GenerationRequest request = request(N_NEW);
        GenerationResult refA;
        GenerationResult refB;
        try (GGUFModel one = new GGUFModel(model.toString(), paged);
             BatchEngine reference = BatchEngine.forModel(one, paged)) {
            BatchSequence a = reference.submit(tokens(1), request,
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            BatchSequence b = reference.submit(tokens(2), request,
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            reference.drain();
            refA = a.result();
            refB = b.result();
        }
        assertNotNull(refA);
        assertNotNull(refB);

        try (GGUFModel two = new GGUFModel(model.toString(), paged);
             BatchEngine engine = BatchEngine.forModel(two, paged)) {
            BatchSequence a = engine.submit(tokens(1), request,
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            BatchSequence b = engine.submit(tokens(2), request,
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            while (engine.stats().decodeSteps() < 1) {
                engine.pump();
            }
            engine.preemptAll();
            assertEquals("the freeze frees every running sequence", 0, engine.running());
            assertTrue("the frozen sequences wait in the preempted set", engine.stats().preempted() >= 2);

            engine.drain();
            assertEquals(BatchSequence.State.DONE, a.state());
            assertEquals(BatchSequence.State.DONE, b.state());
            assertArrayEquals("the preempted sequence recomputes to the same tokens", refA.tokens(), a.result().tokens());
            assertArrayEquals(refB.tokens(), b.result().tokens());
            assertEquals(N_NEW, a.result().tokenCount());
        }
    }
}

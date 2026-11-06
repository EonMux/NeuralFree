package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.arch.PoolingKind;
import com.eonmux.neuralfree.gguf.engine.ContextFullException;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.GenerationLoop;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.PoolingOutput;
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
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Pooling requests on the batch engine (gap batched-pooling-scheduling): a pooling request queues like a
 * generation prompt, prefills beside the running sequences under the same token budget, and completes with the
 * pooled output the model's own {@code encode} answers — without draining the engine or wiping the prefix cache,
 * as vLLM's scheduler batches its {@code pooling_params} requests with generation and llama.cpp runs its embedding
 * tasks in slots of the shared batch. A chunked pooling prefill accumulates the whole prompt's hidden rows, a
 * last-row pooling request reuses the cached prefix while a pooled-every-row one does not, a queued request that
 * runs out of pages starts its prefill over and still pools its own rows, and the unsupported inputs refuse by
 * name.
 */
public class BatchEnginePoolingTest {

    private static final int N_TOKENS = 12;
    private static final int N_NEW = 40;

    /** A config over the F32 KV cache with the given pooling override, slot count and step budget. */
    private static EngineConfig config(PoolingKind pooling, int parallel, int nCtx, Integer maxBatched) {
        MemoryConfig.Builder mc = MemoryConfig.builder().nCtx(nCtx).ubatch(8)
                .typeK(KvDType.F32).typeV(KvDType.F32);
        if (parallel > 1) {
            mc = mc.parallel(parallel);
        }
        if (maxBatched != null) {
            mc = mc.maxBatchedTokens(maxBatched);
        }
        return EngineConfig.builder().pooling(pooling).memory(mc.build()).build();
    }

    /** The tokens a request runs (deterministic ids inside the synthetic vocab, as BatchEngineTest). */
    private static int[] tokens(int seed) {
        return tokens(seed, N_TOKENS);
    }

    private static int[] tokens(int seed, int n) {
        int[] t = new int[n];
        for (int i = 0; i < t.length; i++) {
            t[i] = 4 + (seed * 7 + i * 13) % 250;
        }
        return t;
    }

    /** The reference: the same tokens pooled by the model's own one-shot encode on a contiguous memory. */
    private static float[] referenceVector(Path model, PoolingKind pooling, int[] tokens) throws IOException {
        try (GGUFModel m = new GGUFModel(model.toString(), config(pooling, 1, 64, null))) {
            return m.encode(tokens);
        }
    }

    private static GenerationRequest generation(int maxNewTokens) {
        return GenerationRequest.builder().maxNewTokens(maxNewTokens)
                .sampler(SamplerConfig.greedyDecoding()).build();
    }

    /** The reference: the same prompt and request through the single-sequence GenerationLoop. */
    private static GenerationResult reference(Path model, PoolingKind pooling, int[] prompt,
                                              GenerationRequest request) throws IOException {
        try (GGUFModel one = new GGUFModel(model.toString(), config(pooling, 1, 64, null))) {
            return GenerationLoop.forModel(one, config(pooling, 1, 64, null)).generate(prompt, request,
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
        }
    }

    private static float[] pooled(PoolingSequence pooling) {
        assertEquals("the pooling request completed: " + pooling.handle().error(),
                BatchSequence.State.DONE, pooling.handle().state());
        assertNotNull(pooling.pooled());
        return (float[]) pooling.pooled();
    }

    /** Pumps until the pooling request reaches a terminal state, failing after 2,000 idle steps. */
    private static void pumpUntilDone(BatchEngine engine, PoolingSequence pooling) {
        for (int step = 0; step < 2_000; step++) {
            if (pooling.handle().finished()) {
                return;
            }
            engine.pump();
        }
        throw new AssertionError("the pooling request never completed: " + pooling.handle());
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
    public void aPooledRequestAnswersTheModelEncodeVector() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        float[] reference = referenceVector(model, PoolingKind.MEAN, tokens(1));
        try (GGUFModel m = new GGUFModel(model.toString(), config(PoolingKind.MEAN, 2, 64, null));
             BatchEngine engine = BatchEngine.forModel(m, config(PoolingKind.MEAN, 2, 64, null))) {
            PoolingSequence pooling = engine.submitPooling(tokens(1), PoolingOutput.POOLED);
            pumpUntilDone(engine, pooling);
            assertArrayEquals("the batched pooling matches the one-shot encode", reference, pooled(pooling), 1e-6f);
            assertTrue(engine.idle());
        }
    }

    @Test
    public void aChunkedPoolingPrefillAccumulatesTheWholePrompt() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        // a 4-token step budget splits the 12-token pooling prompt over three steps; the pooled vector must be
        // the whole prompt's, so the run has to carry every chunk's hidden rows
        float[] reference = referenceVector(model, PoolingKind.MEAN, tokens(2));
        try (GGUFModel m = new GGUFModel(model.toString(), config(PoolingKind.MEAN, 2, 64, 4));
             BatchEngine engine = BatchEngine.forModel(m, config(PoolingKind.MEAN, 2, 64, 4))) {
            PoolingSequence pooling = engine.submitPooling(tokens(2), PoolingOutput.POOLED);
            pumpUntilDone(engine, pooling);
            assertArrayEquals(reference, pooled(pooling), 1e-6f);
        }
    }

    @Test
    public void aPoolingRequestRunsBesideAGenerationWithoutDrainingIt() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        GenerationRequest longRun = generation(N_NEW);
        GenerationResult twin = reference(model, PoolingKind.MEAN, tokens(1), longRun);
        float[] vector = referenceVector(model, PoolingKind.MEAN, tokens(2));
        try (GGUFModel m = new GGUFModel(model.toString(), config(PoolingKind.MEAN, 2, 128, null));
             BatchEngine engine = BatchEngine.forModel(m, config(PoolingKind.MEAN, 2, 128, null))) {
            int[] routed = new int[1];
            BatchSequence gen = engine.submit(tokens(1), longRun,
                    (tokenId, text) -> routed[0]++);
            assertEquals("nothing routed before the first pump", 0, routed[0]);
            for (int i = 0; i < 4; i++) {
                engine.pump();
            }
            assertTrue("the generation decoded before the pooling request arrived", routed[0] > 0);
            assertFalse(gen.finished());
            PoolingSequence pooling = engine.submitPooling(tokens(2), PoolingOutput.POOLED);
            pumpUntilDone(engine, pooling);
            assertArrayEquals("the pooling vector matches its one-shot encode", vector, pooled(pooling), 1e-6f);
            assertFalse("the pooling request completed without waiting for the generation", gen.finished());
            pumpUntilFinished(engine, gen);
            assertArrayEquals("the interleaved generation decodes as it does alone", twin.tokens(),
                    gen.result().tokens());
        }
    }

    @Test
    public void thePrefixCacheSurvivesAPoolingRequest() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        try (GGUFModel m = new GGUFModel(model.toString(), config(PoolingKind.MEAN, 2, 64, null));
             BatchEngine engine = BatchEngine.forModel(m, config(PoolingKind.MEAN, 2, 64, null))) {
            BatchSequence warmup = engine.submit(tokens(1), generation(2),
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            pumpUntilFinished(engine, warmup);
            BatchSequence again = engine.submit(tokens(1), generation(2),
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            pumpUntilFinished(engine, again);
            assertTrue("the warm prompt is cached", again.reusedPrefix() > 0);
            PoolingSequence pooling = engine.submitPooling(tokens(1), PoolingOutput.POOLED);
            pumpUntilDone(engine, pooling);
            assertEquals(BatchSequence.State.DONE, pooling.handle().state());
            BatchSequence after = engine.submit(tokens(1), generation(2),
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            pumpUntilFinished(engine, after);
            assertTrue("the pooling request left the prefix cache in place", after.reusedPrefix() > 0);
        }
    }

    @Test
    public void aLastRowPoolingRequestReusesTheCachedPrefix() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        try (GGUFModel m = new GGUFModel(model.toString(), config(PoolingKind.LAST, 2, 64, null));
             BatchEngine engine = BatchEngine.forModel(m, config(PoolingKind.LAST, 2, 64, null))) {
            PoolingSequence first = engine.submitPooling(tokens(1), PoolingOutput.POOLED);
            pumpUntilDone(engine, first);
            PoolingSequence second = engine.submitPooling(tokens(1), PoolingOutput.POOLED);
            pumpUntilDone(engine, second);
            assertTrue("the LAST-pooling request reuses the prefix its twin cached", second.handle()
                    .reusedPrefix() > 0);
            assertArrayEquals(pooled(first), pooled(second), 1e-6f);
        }
    }

    @Test
    public void aPooledEveryRowRequestReusesNoCachedPrefix() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        try (GGUFModel m = new GGUFModel(model.toString(), config(PoolingKind.MEAN, 2, 64, null));
             BatchEngine engine = BatchEngine.forModel(m, config(PoolingKind.MEAN, 2, 64, null))) {
            PoolingSequence first = engine.submitPooling(tokens(1), PoolingOutput.POOLED);
            pumpUntilDone(engine, first);
            PoolingSequence second = engine.submitPooling(tokens(1), PoolingOutput.POOLED);
            pumpUntilDone(engine, second);
            assertEquals("the MEAN pooling pools every row, so it prefills every row too", 0,
                    second.handle().reusedPrefix());
            assertArrayEquals(pooled(first), pooled(second), 1e-6f);
        }
    }

    @Test
    public void aPoolingPrefillThatRunsOutOfPagesStartsOverAndPoolsItsOwnRows() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        GenerationRequest longRun = generation(N_NEW).toBuilder().ignoreEos(true).build();
        int[] poolingPrompt = tokens(2, 30);
        float[] vector = referenceVector(model, PoolingKind.MEAN, poolingPrompt);
        GenerationResult twinA = reference(model, PoolingKind.MEAN, tokens(1), longRun);
        GenerationResult twinC = reference(model, PoolingKind.MEAN, tokens(3), longRun);
        // two long generations fill the 64 shared cells while the 30-row pooling prompt waits; the prompt admits
        // into the shrinking memory, runs out of pages, queues again and pools from its own completed prefill
        try (GGUFModel m = new GGUFModel(model.toString(), config(PoolingKind.MEAN, 3, 64, null));
             BatchEngine engine = BatchEngine.forModel(m, config(PoolingKind.MEAN, 3, 64, null))) {
            BatchSequence a = engine.submit(tokens(1), longRun, (com.eonmux.neuralfree.gguf.TokenCallback) null);
            PoolingSequence pooling = engine.submitPooling(poolingPrompt, PoolingOutput.POOLED);
            BatchSequence c = engine.submit(tokens(3), longRun, (com.eonmux.neuralfree.gguf.TokenCallback) null);
            pumpUntilDone(engine, pooling);
            pumpUntilFinished(engine, a, c);
            assertArrayEquals("the restarted pooling pools the same vector", vector, pooled(pooling), 1e-6f);
            assertArrayEquals(twinA.tokens(), a.result().tokens());
            assertArrayEquals(twinC.tokens(), c.result().tokens());
            assertTrue(engine.idle());
        }
    }

    @Test
    public void unsupportedPoolingInputsRefuseByName() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        try (GGUFModel bare = new GGUFModel(model.toString(), config(null, 2, 64, null));
             BatchEngine bareEngine = BatchEngine.forModel(bare, config(null, 2, 64, null))) {
            // a model without pooling heads refuses the pooled vector by name
            UnsupportedOperationException noHeads = assertThrows(UnsupportedOperationException.class,
                    () -> bareEngine.submitPooling(tokens(1), PoolingOutput.POOLED));
            assertTrue(noHeads.getMessage(), noHeads.getMessage().contains("pooling"));
            // and its generation submit still runs (the pooling override is what carries the heads)
            BatchSequence gen = bareEngine.submit(tokens(1), generation(1),
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            pumpUntilFinished(bareEngine, gen);
        }
        try (GGUFModel m = new GGUFModel(model.toString(), config(PoolingKind.MEAN, 2, 64, null));
             BatchEngine engine = BatchEngine.forModel(m, config(PoolingKind.MEAN, 2, 64, null))) {
            // per-token embeddings refuse on a pooled model
            UnsupportedOperationException wrongTask = assertThrows(UnsupportedOperationException.class,
                    () -> engine.submitPooling(tokens(1), PoolingOutput.TOKEN_EMBEDDINGS));
            assertTrue(wrongTask.getMessage(), wrongTask.getMessage().contains("per-token"));
            // empty and negative prompts refuse before anything queues
            assertThrows(IllegalArgumentException.class,
                    () -> engine.submitPooling(new int[0], PoolingOutput.POOLED));
            assertThrows(IllegalArgumentException.class,
                    () -> engine.submitPooling(new int[] {5, -1, 7}, PoolingOutput.POOLED));
            // an oversize prompt refuses like a generation prompt
            ContextFullException oversize = assertThrows(ContextFullException.class,
                    () -> engine.submitPooling(tokens(1, 65), PoolingOutput.POOLED));
            assertEquals(65, oversize.requiredPositions());
        }
    }
}

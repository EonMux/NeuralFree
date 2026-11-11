package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.arch.PoolingKind;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

import static com.eonmux.neuralfree.server.ServerTestSupport.backend;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Pooling requests scheduled and batched with generation over the real service (gap batched-pooling-scheduling):
 * an embedding request on the pooling-capable paged model queues into the replica's batch engine and completes
 * while a generation still decodes (vLLM schedules its {@code pooling_params} requests in the same token-budget
 * scheduler; llama.cpp runs its embedding tasks in slots of the shared batch), the pooling request leaves the
 * prefix cache in place (a later identical prompt reuses it), and the batched request answers the vector the
 * model's own one-shot encode answers.
 */
class PoolingBatchSchedulingTest {

    private static final int N_TOKENS = 12;

    private static Path model;
    private static EngineService pooled;
    private static EngineService solo;

    private static EngineConfig config() {
        // The context leaves room for the unbounded generation of the overlap test, so the pooling request
        // can never lose a race with a token budget the test did not intend to set.
        MemoryConfig.Builder memory = MemoryConfig.builder().nCtx(8192).ubatch(8).parallel(2)
                .typeK(KvDType.F32).typeV(KvDType.F32);
        return EngineConfig.builder().backend(backend()).pooling(PoolingKind.LAST)
                .memory(memory.build()).build();
    }

    private static int[] tokens(int seed) {
        int[] t = new int[N_TOKENS];
        for (int i = 0; i < t.length; i++) {
            t[i] = 4 + (seed * 7 + i * 13) % 250;
        }
        return t;
    }

    private static GenerationRequest longRun() {
        return GenerationRequest.builder().maxNewTokens(40).ignoreEos(true)
                .sampler(SamplerConfig.greedyDecoding()).build();
    }

    @BeforeAll
    static void open() throws IOException {
        // The fixture's context matches the engine's, so the unbounded generation of the overlap test has
        // room to decode far past the tokens the test waits for before cancelling it.
        model = SyntheticModels.llama(SyntheticModels.Config.llama().toBuilder().nCtxTrain(8192).build())
                .writeTempFile();
        pooled = new EngineService(model.toString(), config());
        solo = new EngineService(model.toString(), config());
    }

    @AfterAll
    static void close() {
        pooled.close();
        solo.close();
    }

    private static GenerationResult generate(EngineService service, int[] prompt) throws Exception {
        EngineService.Submission submission = service.submit(prompt, longRun(), (token, text) -> { });
        assertTrue(submission.await(60, TimeUnit.SECONDS));
        return submission.sequence().result();
    }

    @Test
    void anEmbeddingRequestCompletesWhileTheGenerationStillDecodes() throws Exception {
        int[] prompt = tokens(1);
        int[] embeddingInput = tokens(2);
        // the reference answers, produced alone: the generation text and the pooled vector
        GenerationResult twin = generate(solo, prompt);
        float[] vector;
        try (EngineService alone = new EngineService(model.toString(), config())) {
            vector = alone.pooling().embed(null, embeddingInput);
        }

        // No token budget: the generation decodes until the test cancels it, so under load the embedding
        // cannot lose a race with a budget and answer after the generation has already finished.
        AtomicBoolean cancel = new AtomicBoolean();
        CountDownLatch firstToken = new CountDownLatch(1);
        CountDownLatch twinLength = new CountDownLatch(twin.tokens().length);
        GenerationRequest unbounded = GenerationRequest.builder()
                .maxNewTokens(GenerationRequest.UNLIMITED).ignoreEos(true)
                .sampler(SamplerConfig.greedyDecoding())
                .cancellation(cancel::get)
                .build();
        EngineService.Submission generation = pooled.submit(prompt, unbounded, (token, text) -> {
            firstToken.countDown();
            twinLength.countDown();
        });
        try {
            assertTrue(firstToken.await(60, TimeUnit.SECONDS), "the generation started decoding");
            // the pooling request queues into the same engine and answers while the generation runs on
            float[] batched = pooled.pooling().embed(null, embeddingInput);
            assertArrayEquals(vector, batched, 1e-6f);
            assertFalse(generation.sequence().finished(),
                    "the embedding answered without waiting for the generation");
            assertTrue(twinLength.await(60, TimeUnit.SECONDS),
                    "the pool's generation kept decoding beside the pooling request");
        } finally {
            cancel.set(true);
            assertTrue(generation.await(60, TimeUnit.SECONDS), "the generation stopped after its cancellation");
        }
        GenerationResult interleaved = generation.sequence().result();
        assertNotNull(interleaved, "the interleaved generation failed: " + generation.sequence());
        assertTrue(interleaved.tokens().length >= twin.tokens().length,
                "the interleaved generation produced " + interleaved.tokens().length + " tokens, fewer than the "
                        + twin.tokens().length + " of its twin");
        assertArrayEquals(twin.tokens(), Arrays.copyOf(interleaved.tokens(), twin.tokens().length),
                "the interleaved generation decodes as it does alone");
    }

    @Test
    void thePrefixCacheSurvivesThePoolingRequests() throws Exception {
        int[] prompt = tokens(3);
        long before = pooled.stats().promptTokensCached();
        pooled.pooling().embed(null, prompt);
        pooled.pooling().embed(null, prompt);
        long afterPooling = cachedAbove(before, "the second identical pooling prompt reused the cached prefix"
                + " (LAST pooling reads the last row)");
        generate(pooled, prompt);
        cachedAbove(afterPooling, "a generation prompt reuses what the pooling requests left cached");
    }

    /**
     * The pool's cached prompt tokens once they exceed {@code floor}: a replica publishes its counters after the step
     * that answered a request, so a reading right after the answer can precede that step's publication.
     */
    private static long cachedAbove(long floor, String what) throws InterruptedException {
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(10);
        long cached = pooled.stats().promptTokensCached();
        while (cached <= floor && System.nanoTime() < deadline) {
            Thread.sleep(5);
            cached = pooled.stats().promptTokensCached();
        }
        assertTrue(cached > floor, what + ": " + cached + " cached prompt tokens, " + floor + " before");
        return cached;
    }

    @Test
    void arrayInputsRunAsOneBatchAndMatchTheOneShotVectors() throws Exception {
        int[][] inputs = {tokens(4), tokens(5), tokens(6)};
        float[][] expected;
        try (EngineService alone = new EngineService(model.toString(), config())) {
            expected = alone.pooling().embedAll(null, inputs);
        }
        // the batched request answers the one-shot vectors, within the forward's float noise (the GPU matmul's
        // reduction order differs across engine instances)
        float[][] batched = pooled.pooling().embedAll(null, inputs);
        for (int i = 0; i < expected.length; i++) {
            assertArrayEquals(expected[i], batched[i], 1e-5f, "input " + i);
        }
    }
}

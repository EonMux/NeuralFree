package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.engine.TokenScore;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

/**
 * The vLLM request extras on the batch engine: {@code prompt_logprobs} read every prompt row's logits back (and a
 * prompt-logprobs request reuses no cached prefix, so every entry is present), {@code priority} admits the lowest
 * value first among queued requests under the {@code priority} scheduling policy, and a {@code cache_salt} lookup matches only entries of the same salt.
 */
public class BatchEngineExtrasTest {

    private static final int N_TOKENS = 12;
    private static final int N_NEW = 6;

    private static EngineConfig config(int parallel) {
        return config(parallel, SchedulingPolicy.FCFS);
    }

    private static EngineConfig config(int parallel, SchedulingPolicy policy) {
        return EngineConfig.builder()
                .memory(MemoryConfig.builder().nCtx(64).ubatch(8).parallel(parallel)
                        .typeK(KvDType.F32).typeV(KvDType.F32).schedulingPolicy(policy).build())
                .build();
    }

    private static GenerationRequest request(int maxNewTokens) {
        return GenerationRequest.builder().maxNewTokens(maxNewTokens).sampler(SamplerConfig.greedyDecoding()).build();
    }

    private static int[] prompt(int seed) {
        int[] t = new int[N_TOKENS];
        for (int i = 0; i < t.length; i++) {
            t[i] = 4 + (seed * 11 + i * 7) % 250;
        }
        return t;
    }

    @Test
    public void promptLogprobsReadEveryPromptRowBackWithTheTokenLeading() throws IOException, InterruptedException {
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig cfg = config(2);
        int[] prompt = prompt(1);
        try (GGUFModel m = new GGUFModel(model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(m, cfg)) {
            BatchSequence run = engine.submit(prompt,
                    request(N_NEW).toBuilder().promptLogprobs(2).build(),
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.drain();
            List<TokenScore[]> captures = run.result().promptLogprobs();
            assertEquals("one capture per prompt token", N_TOKENS, captures.size());
            assertNull("no row predicts the first prompt token", captures.get(0));
            for (int k = 1; k < N_TOKENS; k++) {
                TokenScore[] capture = captures.get(k);
                assertNotNull(capture);
                assertEquals("the prompt token leads wherever it ranks", prompt[k], capture[0].tokenId());
                assertTrue("the rank is 1-based", capture[0].rank() >= 1);
                assertTrue("a log probability is non-positive", capture[0].logprob() <= 0f);
                assertEquals("the prompt token merged with the best token (vLLM's dict)",
                        capture[0].rank() == 1 ? 1 : 2, capture.length);
                if (capture.length == 2) {
                    assertEquals("the alternative is the best token", 1, capture[1].rank());
                }
            }
        }
    }

    @Test
    public void aPromptLogprobsRequestReusesNoCachedPrefix() throws IOException, InterruptedException {
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig cfg = config(2);
        GenerationRequest request = request(N_NEW).toBuilder().promptLogprobs(1).build();
        try (GGUFModel m = new GGUFModel(model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(m, cfg)) {
            BatchSequence first = engine.submit(prompt(3), request,
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.drain();
            BatchSequence second = engine.submit(prompt(3), request,
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.drain();
            assertEquals("the first run prefills everything", 0, first.reusedPrefix());
            assertEquals("the second run reads the rows itself, reusing nothing", 0, second.reusedPrefix());
            assertEquals(N_TOKENS, second.result().promptLogprobs().size());
            assertArrayEquals("the cache changes no result", first.result().tokens(), second.result().tokens());
        }
    }

    @Test
    public void priorityAdmitsTheLowestValueFirstAmongQueuedRequests()
            throws IOException, InterruptedException {
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig cfg = config(2, SchedulingPolicy.PRIORITY);
        try (GGUFModel m = new GGUFModel(model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(m, cfg)) {
            BatchSequence running = engine.submit(prompt(5), request(30),
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            // queued in this order: priority 5 first, priority 1 second — FIFO would admit the 5
            BatchSequence lowValueSubmittedLast = engine.submit(prompt(6),
                    request(N_NEW).toBuilder().priority(1).build(),
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            BatchSequence highValueSubmittedFirst = engine.submit(prompt(7),
                    request(N_NEW).toBuilder().priority(5).build(),
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.pump();
            assertEquals("the slot went to the lower priority", BatchSequence.State.RUNNING,
                    lowValueSubmittedLast.state());
            assertEquals("the higher priority still waits", BatchSequence.State.QUEUED,
                    highValueSubmittedFirst.state());
            assertEquals("the first submission prefills", BatchSequence.State.RUNNING, running.state());
            engine.drain();
            assertEquals(BatchSequence.State.DONE, lowValueSubmittedLast.state());
            assertEquals(BatchSequence.State.DONE, highValueSubmittedFirst.state());
        }
    }

    @Test
    public void aCacheSaltLookupMatchesOnlyEntriesOfTheSameSalt() throws IOException, InterruptedException {
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig cfg = config(2);
        int[] prompt = prompt(8);
        try (GGUFModel m = new GGUFModel(model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(m, cfg)) {
            BatchSequence saltedB = engine.submit(prompt, request(N_NEW).toBuilder().cacheSalt("b").build(),
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.drain();
            assertEquals("the first run prefills everything", 0, saltedB.reusedPrefix());

            BatchSequence saltedA = engine.submit(prompt, request(N_NEW).toBuilder().cacheSalt("a").build(),
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.drain();
            assertEquals("the cached entry is another salt's and never matches", 0, saltedA.reusedPrefix());
            assertArrayEquals("the salt changes no result", saltedB.result().tokens(), saltedA.result().tokens());

            BatchSequence sameSalt = engine.submit(prompt, request(N_NEW).toBuilder().cacheSalt("b").build(),
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.drain();
            assertEquals("the matching salt reuses its own blocks beside the other salt's", N_TOKENS - 1,
                    sameSalt.reusedPrefix());
            BatchSequence unsalted = engine.submit(prompt, request(N_NEW),
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.drain();
            assertEquals("no salt matches no salted block", 0, unsalted.reusedPrefix());
            assertArrayEquals(saltedB.result().tokens(), unsalted.result().tokens());
        }
    }
}

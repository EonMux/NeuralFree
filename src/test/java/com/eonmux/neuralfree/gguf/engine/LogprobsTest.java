package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.engine.batch.BatchEngine;
import com.eonmux.neuralfree.gguf.engine.batch.BatchSequence;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Path;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * The logprobs capture (PARITY P4): a request's {@code logprobs(n)} makes every generated step report the
 * sampled token plus its top alternatives of the post-chain distribution as log probabilities — through the
 * single-sequence {@link GenerationLoop} (whose greedy on-device path must stand down so the logits reach the
 * host) and through the batch engine — while a request without the setting reports nothing. The sampled token
 * is always the first entry, alternatives follow in descending probability, and a greedy chain's probabilities
 * softmax over the surviving candidates exactly once.
 */
public class LogprobsTest {

    private static final int N_TOKENS = 8;
    private static final int N_NEW = 5;

    private static EngineConfig config(boolean paged) {
        MemoryConfig.Builder mc = MemoryConfig.builder().nCtx(64).ubatch(8)
                .typeK(KvDType.F32).typeV(KvDType.F32);
        if (paged) {
            mc = mc.parallel(2);
        }
        return EngineConfig.builder().memory(mc.build()).build();
    }

    private static int[] tokens(int seed) {
        int[] t = new int[N_TOKENS];
        for (int i = 0; i < t.length; i++) {
            t[i] = 4 + (seed * 13 + i * 5) % 250;
        }
        return t;
    }

    private static GenerationRequest request(int logprobs) {
        return GenerationRequest.builder()
                .maxNewTokens(N_NEW)
                .sampler(SamplerConfig.greedyDecoding())
                .logprobs(logprobs)
                .build();
    }

    @Test
    public void theLoopReportsTopAlternativesWithTheSampledTokenFirst() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        try (GGUFModel m = new GGUFModel(model.toString(), config(false))) {
            GenerationResult r = GenerationLoop.forModel(m, config(false))
                    .generate(tokens(1), request(3), (com.eonmux.neuralfree.gguf.TokenCallback) null);
            assertEquals("one entry per generated token", N_NEW, r.logprobs().size());
            for (int step = 0; step < r.logprobs().size(); step++) {
                TokenScore[] scores = r.logprobs().get(step);
                assertEquals("exactly the requested entries, the sampled token first", 3, scores.length);
                assertEquals("the sampled token leads", r.tokens()[step], scores[0].tokenId());
                assertTrue("the sampled token's logprob is finite", !Float.isInfinite(scores[0].logprob()));
                for (int k = 1; k < scores.length; k++) {
                    assertTrue("alternatives follow in descending probability",
                            scores[k].logprob() <= scores[k - 1].logprob());
                    assertTrue("log probabilities are at most 0", scores[k].logprob() <= 0f);
                }
                double sum = 0;
                for (TokenScore score : scores) {
                    sum += Math.exp(score.logprob());
                }
                assertTrue("the reported probabilities stay within the distribution: " + sum, sum <= 1.0 + 1e-6);
            }
        }
    }

    @Test
    public void theBatchEngineCarriesTheSameCapture() throws IOException, InterruptedException {
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig paged = config(true);
        try (GGUFModel m = new GGUFModel(model.toString(), paged);
             BatchEngine engine = BatchEngine.forModel(m, paged)) {
            BatchSequence seq = engine.submit(tokens(2), request(2),
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.drain();
            assertEquals(BatchSequence.State.DONE, seq.state());
            assertEquals("one entry per generated token", N_NEW, seq.result().logprobs().size());
            for (int step = 0; step < seq.result().logprobs().size(); step++) {
                TokenScore[] scores = seq.result().logprobs().get(step);
                assertEquals("exactly the requested entries, the sampled token first", 2, scores.length);
                assertEquals(seq.result().tokens()[step], scores[0].tokenId());
            }
        }
    }

    @Test
    public void aRequestWithoutLogprobsReportsNothing() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        try (GGUFModel m = new GGUFModel(model.toString(), config(false))) {
            GenerationResult r = GenerationLoop.forModel(m, config(false))
                    .generate(tokens(3), request(0), (com.eonmux.neuralfree.gguf.TokenCallback) null);
            assertTrue(r.logprobs().isEmpty());
            assertEquals(N_NEW, r.tokenCount());
        }
    }
}

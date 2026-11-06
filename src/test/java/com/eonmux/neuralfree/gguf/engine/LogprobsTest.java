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
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

/**
 * The logprobs capture: a request's {@code logprobs(n)} makes every generated step report the
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

    @Test
    public void theLoopReadsThePromptRowsBackForPromptLogprobs() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        int[] prompt = tokens(4);
        GenerationRequest r = GenerationRequest.builder().maxNewTokens(N_NEW)
                .sampler(SamplerConfig.greedyDecoding()).promptLogprobs(2).build();
        try (GGUFModel m = new GGUFModel(model.toString(), config(false))) {
            GenerationLoop loop = GenerationLoop.forModel(m, config(false));
            GenerationResult first = loop.generate(prompt, r, (com.eonmux.neuralfree.gguf.TokenCallback) null);
            assertPromptCapture(first, prompt);
            GenerationResult second = loop.generate(prompt, r, (com.eonmux.neuralfree.gguf.TokenCallback) null);
            assertPromptCapture(second, prompt);
            assertArrayEquals("the capture changes no generated token", first.tokens(), second.tokens());
        }
    }

    /** The single-sequence capture answers the batch engine's: same size, null first entry, prompt token leading,
     *  and the same log probabilities (the raw distributions agree over the same rows). */
    @Test
    public void theLoopAndTheBatchEngineReportTheSamePromptLogprobs() throws IOException, InterruptedException {
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig paged = config(true);
        int[] prompt = tokens(5);
        GenerationRequest r = GenerationRequest.builder().maxNewTokens(N_NEW)
                .sampler(SamplerConfig.greedyDecoding()).promptLogprobs(2).build();
        try (GGUFModel loopModel = new GGUFModel(model.toString(), config(false));
             GGUFModel batchModel = new GGUFModel(model.toString(), paged);
             BatchEngine engine = BatchEngine.forModel(batchModel, paged)) {
            GenerationResult fromLoop = GenerationLoop.forModel(loopModel, config(false))
                    .generate(prompt, r, (com.eonmux.neuralfree.gguf.TokenCallback) null);
            BatchSequence seq = engine.submit(prompt, r, (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.drain();
            List<TokenScore[]> fromBatch = seq.result().promptLogprobs();
            assertPromptCapture(fromLoop, prompt);
            assertPromptCapture(seq.result(), prompt);
            assertEquals(fromBatch.size(), fromLoop.promptLogprobs().size());
            for (int k = 1; k < fromLoop.promptLogprobs().size(); k++) {
                assertEquals("the prompt token leads on both paths", fromBatch.get(k)[0].tokenId(),
                        fromLoop.promptLogprobs().get(k)[0].tokenId());
                assertEquals("the raw distributions agree", fromBatch.get(k)[0].logprob(),
                        fromLoop.promptLogprobs().get(k)[0].logprob(), 1e-4f);
            }
        }
    }

    /**
     * {@code prompt_logprob_start} and {@code prompt_logprob_token_ids} through the single-sequence loop: the rows
     * before the start report nothing, each scored row scores its ids (a pad at {@code -inf}), and both captures
     * answer the batch engine's over the same rows.
     */
    @Test
    public void theLoopScoresPromptTokenIdRowsLikeTheBatchEngine() throws IOException, InterruptedException {
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig paged = config(true);
        int[] prompt = tokens(7);
        int start = 3;
        int[][] ids = new int[prompt.length - 1 - start][];
        for (int i = 0; i < ids.length; i++) {
            ids[i] = i == 0 ? new int[]{prompt[start + 1]} : new int[]{prompt[start + i + 1], 9};
        }
        GenerationRequest r = GenerationRequest.builder().maxNewTokens(N_NEW)
                .sampler(SamplerConfig.greedyDecoding()).promptLogprobs(1).promptLogprobStart(start)
                .promptLogprobTokenIds(ids).build();
        try (GGUFModel loopModel = new GGUFModel(model.toString(), config(false));
             GGUFModel batchModel = new GGUFModel(model.toString(), paged);
             BatchEngine engine = BatchEngine.forModel(batchModel, paged)) {
            GenerationLoop loop = GenerationLoop.forModel(loopModel, config(false));
            GenerationResult fromLoop = loop.generate(prompt, r, (com.eonmux.neuralfree.gguf.TokenCallback) null);
            BatchSequence seq = engine.submit(prompt, r, (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.drain();
            float[][] loopScores = fromLoop.promptTokenIdLogprobs();
            float[][] batchScores = seq.result().promptTokenIdLogprobs();
            assertEquals("one scored row per prompt row from the start", ids.length, loopScores.length);
            assertEquals(batchScores.length, loopScores.length);
            assertEquals("a ragged row's pad scores -inf", Float.NEGATIVE_INFINITY, loopScores[0][1], 0f);
            for (int i = 0; i < ids.length; i++) {
                TokenScore[] entry = fromLoop.promptLogprobs().get(start + i + 1);
                assertEquals("the row scores the prompt token like its prompt_logprobs entry",
                        entry[0].logprob(), loopScores[i][0], 1e-6f);
                for (int j = 0; j < loopScores[i].length; j++) {
                    assertEquals("the loop answers the batch engine", batchScores[i][j], loopScores[i][j], 1e-4f);
                }
            }
            for (int k = 0; k <= start; k++) {
                assertNull("no entry up to and including the start", fromLoop.promptLogprobs().get(k));
            }
            assertArrayEquals("the capture changes no generated token", seq.result().tokens(), fromLoop.tokens());
            GenerationRequest misshapen = r.toBuilder().promptLogprobTokenIds(new int[][]{{1}}).build();
            IllegalArgumentException e = org.junit.Assert.assertThrows(IllegalArgumentException.class,
                    () -> loop.generate(prompt, misshapen, (com.eonmux.neuralfree.gguf.TokenCallback) null));
            assertTrue(e.getMessage(), e.getMessage().contains("prompt_logprob_token_ids"));
        }
    }

    private static void assertPromptCapture(GenerationResult result, int[] prompt) {
        assertEquals("one capture per prompt token", prompt.length, result.promptLogprobs().size());
        assertNull("no row predicts the first prompt token", result.promptLogprobs().get(0));
        for (int k = 1; k < prompt.length; k++) {
            TokenScore[] capture = result.promptLogprobs().get(k);
            assertEquals("the request asked for two entries", 2, capture.length);
            assertEquals("the prompt token leads wherever it ranks", prompt[k], capture[0].tokenId());
            assertTrue(capture[0].logprob() <= 0f);
            assertTrue("the alternative is not ranked below the token", capture[1].rank() <= capture[0].rank());
        }
    }

    @Test
    public void namedLogprobTokenIdsReplaceTheTopAlternatives() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        GenerationRequest r = GenerationRequest.builder().maxNewTokens(N_NEW)
                .sampler(SamplerConfig.greedyDecoding()).logprobTokenIds(4, 9, 4).build();
        try (GGUFModel m = new GGUFModel(model.toString(), config(false))) {
            GenerationResult result = GenerationLoop.forModel(m, config(false))
                    .generate(tokens(6), r, (com.eonmux.neuralfree.gguf.TokenCallback) null);
            assertEquals("one capture per generated token", N_NEW, result.logprobs().size());
            for (int step = 0; step < N_NEW; step++) {
                TokenScore[] capture = result.logprobs().get(step);
                assertEquals("the sampled token leads, the named ids follow in request order", 3, capture.length);
                assertEquals(result.tokens()[step], capture[0].tokenId());
                assertEquals(4, capture[1].tokenId());
                assertEquals(9, capture[2].tokenId());
                for (TokenScore t : capture) {
                    assertTrue("a vocabulary token's logprob is finite: " + t, !Float.isInfinite(t.logprob()));
                    assertTrue(t.logprob() <= 0f);
                }
            }
        }
    }
}

package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.TokenCallback;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.LogprobsMode;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.engine.TokenScore;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Logprob depth on the batch engine over a synthetic llama: a generated token's raw score equals the prompt-row
 * score of the same token at the same position (both are the model's own distribution, vLLM's
 * {@code raw_logprobs}), a processed mode follows the sampler chain instead, {@code logprob_token_ids} score
 * exactly the requested ids, and {@code prompt_logprob_token_ids} from {@code prompt_logprob_start} score the
 * prompt rows the {@code prompt_logprobs: -1} capture scores.
 */
public class BatchLogprobDepthTest {

    private static final int N_TOKENS = 10;
    private static final int N_NEW = 4;
    private static final double TOLERANCE = 2e-3;

    private static EngineConfig config() {
        return EngineConfig.builder()
                .memory(MemoryConfig.builder().nCtx(64).ubatch(8).parallel(2)
                        .typeK(KvDType.F32).typeV(KvDType.F32).build())
                .build();
    }

    private static int[] prompt() {
        int[] t = new int[N_TOKENS];
        for (int i = 0; i < t.length; i++) {
            t[i] = 4 + (5 * 17 + i * 13) % 250;
        }
        return t;
    }

    private static GenerationRequest.Builder greedy() {
        return GenerationRequest.builder().maxNewTokens(N_NEW).sampler(SamplerConfig.greedyDecoding());
    }

    private static GenerationResult run(BatchEngine engine, int[] prompt, GenerationRequest request) {
        BatchSequence seq = engine.submit(prompt, request, (TokenCallback) null);
        engine.drain();
        assertEquals(BatchSequence.State.DONE, seq.state());
        return seq.result();
    }

    private static int[] concat(int[] a, int[] b) {
        int[] out = java.util.Arrays.copyOf(a, a.length + b.length);
        System.arraycopy(b, 0, out, a.length, b.length);
        return out;
    }

    /**
     * Rescoring prompt plus output as a prompt reproduces the raw generated-token scores; a strong logit bias on
     * a token makes the chain pick it at ~0 processed logprob while its raw score stays the model's.
     */
    @Test
    public void rawScoresAreTheModelsDistributionAndProcessedScoresTheChains() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig cfg = config();
        int[] prompt = prompt();
        try (GGUFModel m = new GGUFModel(model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(m, cfg)) {
            int forced = 100;
            GenerationRequest biased = greedy().logprobs(1).logitBias(forced, 30f).build();
            GenerationResult raw = run(engine, prompt, biased);
            GenerationResult processed = run(engine, prompt,
                    biased.toBuilder().logprobsMode(LogprobsMode.PROCESSED_LOGPROBS).build());
            int[] generated = raw.tokens();
            assertEquals(N_NEW, generated.length);
            GenerationResult rescored = run(engine, concat(prompt, generated),
                    GenerationRequest.builder().maxNewTokens(1).sampler(SamplerConfig.greedyDecoding())
                            .promptLogprobs(1).build());
            List<TokenScore[]> rows = rescored.promptLogprobs();
            for (int i = 0; i < generated.length; i++) {
                assertEquals("the bias forces the token", forced, generated[i]);
                TokenScore r = raw.logprobs().get(i)[0];
                TokenScore promptRow = rows.get(prompt.length + i)[0];
                assertEquals(forced, promptRow.tokenId());
                assertEquals("step " + i + ": the raw score is the prompt-row score", promptRow.logprob(),
                        r.logprob(), TOLERANCE);
                assertEquals("step " + i + ": the raw rank is the prompt-row rank", promptRow.rank(), r.rank());
                float p = processed.logprobs().get(i)[0].logprob();
                assertTrue("step " + i + ": the biased chain is near-certain: " + p, p > -1e-3f);
                assertTrue("step " + i + ": the model alone is not: " + r.logprob(), r.logprob() < p - 0.1f);
            }
        }
    }

    @Test
    public void logprobTokenIdsScoreExactlyTheRequestedIds() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig cfg = config();
        try (GGUFModel m = new GGUFModel(model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(m, cfg)) {
            GenerationResult ids = run(engine, prompt(), greedy().logprobTokenIds(9, 3).build());
            GenerationResult all = run(engine, prompt(), greedy().logprobs(1 << 20).build());
            assertEquals(N_NEW, ids.logprobs().size());
            for (int i = 0; i < N_NEW; i++) {
                TokenScore[] s = ids.logprobs().get(i);
                assertEquals(3, s.length);
                assertEquals(ids.tokens()[i], s[0].tokenId());
                assertEquals(9, s[1].tokenId());
                assertEquals(3, s[2].tokenId());
                assertEquals(scoreOf(all.logprobs().get(i), 9), s[1].logprob(), 1e-6);
                assertEquals(scoreOf(all.logprobs().get(i), 3), s[2].logprob(), 1e-6);
            }
            IllegalArgumentException outside = assertThrows(IllegalArgumentException.class,
                    () -> engine.submit(prompt(), greedy().logprobTokenIds(1 << 20).build(), (TokenCallback) null));
            assertTrue(outside.getMessage().contains("logprob_token_ids"));
        }
    }

    @Test
    public void promptTokenIdRowsFromTheStartScoreThePromptRows() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig cfg = config();
        int[] prompt = prompt();
        int start = 4;
        int[][] ids = new int[N_TOKENS - 1 - start][];
        for (int i = 0; i < ids.length; i++) {
            ids[i] = i % 2 == 0 ? new int[]{prompt[start + i + 1], 5} : new int[]{6};
        }
        try (GGUFModel m = new GGUFModel(model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(m, cfg)) {
            GenerationResult all = run(engine, prompt, greedy().promptLogprobs(GenerationRequest.PROMPT_LOGPROBS_ALL)
                    .build());
            GenerationResult scored = run(engine, prompt, greedy().promptLogprobs(1).promptLogprobStart(start)
                    .promptLogprobTokenIds(ids).build());
            for (int k = 0; k <= start; k++) {
                assertNull("position " + k + " is at or before the start", scored.promptLogprobs().get(k));
            }
            for (int k = start + 1; k < N_TOKENS; k++) {
                assertNotNull(scored.promptLogprobs().get(k));
            }
            float[][] scores = scored.promptTokenIdLogprobs();
            assertEquals(ids.length, scores.length);
            for (int i = 0; i < ids.length; i++) {
                TokenScore[] row = all.promptLogprobs().get(start + i + 1);
                assertEquals(2, scores[i].length);
                assertEquals(scoreOf(row, ids[i][0]), scores[i][0], 1e-6);
                if (i % 2 == 0) {
                    assertEquals(scoreOf(row, 5), scores[i][1], 1e-6);
                } else {
                    assertEquals("the padding scores -inf", Float.NEGATIVE_INFINITY, scores[i][1], 0f);
                }
            }
            GenerationResult logits = run(engine, prompt, greedy().promptLogprobTokenIds(new int[][]{{5}})
                    .promptLogprobStart(N_TOKENS - 2).logprobsMode(LogprobsMode.RAW_LOGITS).build());
            assertTrue("the token-id rows alone report no prompt_logprobs", logits.promptLogprobs().isEmpty());
            assertTrue("a logits mode reports a logit, not a log probability",
                    Math.abs(logits.promptTokenIdLogprobs()[0][0]
                            - scoreOf(all.promptLogprobs().get(N_TOKENS - 1), 5)) > 1e-3);
            IllegalArgumentException rows = assertThrows(IllegalArgumentException.class, () -> engine.submit(prompt,
                    greedy().promptLogprobTokenIds(new int[][]{{1}}).build(), (TokenCallback) null));
            assertTrue(rows.getMessage().contains("prompt_logprob_token_ids"));
        }
    }

    private static float scoreOf(TokenScore[] capture, int id) {
        for (TokenScore t : capture) {
            if (t.tokenId() == id) {
                return t.logprob();
            }
        }
        throw new AssertionError("token " + id + " is not in the capture");
    }
}

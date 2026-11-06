package com.eonmux.neuralfree.gguf.engine.beam;

import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.arch.SupportClass;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.GenerationLoop;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.RepetitionDetection;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.engine.sampling.LogitsProcessorArgs;
import com.eonmux.neuralfree.gguf.grammar.ConstraintSpec;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link BeamSearchEngine} over the synthetic llama: a width-one search is the greedy single-sequence run of
 * {@link GenerationLoop} (same tokens, text and end), a wider search ranks its beams by the length-penalized score
 * with a cumulative log probability the width-one run cannot fall below, the beams of a step share the prompt
 * through the batch engine's prefix cache, and the features the search cannot honor refuse by field name — as does
 * an encoder-decoder model, whose decoder sequences the full-resubmission scheme cannot serve.
 */
public final class BeamSearchEngineTest {

    private static final int N_NEW = 8;

    private static EngineConfig config(int parallel) {
        MemoryConfig.Builder mc = MemoryConfig.builder().nCtx(128).ubatch(8)
                .typeK(KvDType.F32).typeV(KvDType.F32);
        if (parallel > 1) {
            mc = mc.parallel(parallel);
        }
        return EngineConfig.builder().memory(mc.build()).build();
    }

    /** Deterministic token ids inside the synthetic vocabulary, clear of the special ids. */
    private static int[] prompt(int seed, int n) {
        int[] t = new int[n];
        for (int i = 0; i < n; i++) {
            t[i] = 4 + (seed * 11 + i * 17) % 250;
        }
        return t;
    }

    private static GenerationRequest beamRequest(int beamWidth, float lengthPenalty) {
        return GenerationRequest.builder()
                .maxNewTokens(N_NEW)
                .sampler(SamplerConfig.greedyDecoding())
                .beamWidth(beamWidth)
                .lengthPenalty(lengthPenalty)
                .build();
    }

    private static GenerationResult greedyReference(Path model, int[] prompt) throws IOException {
        EngineConfig cfg = config(1);
        try (GGUFModel one = new GGUFModel(model.toString(), cfg)) {
            return GenerationLoop.forModel(one, cfg).generate(prompt,
                    GenerationRequest.builder().maxNewTokens(N_NEW).sampler(SamplerConfig.greedyDecoding()).build(),
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
        }
    }

    @Test
    public void widthOneIsTheGreedySingleSequenceRun() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        int[] prompt = prompt(3, 6);
        GenerationResult reference = greedyReference(model, prompt);
        assertTrue("the reference generated", reference.tokenCount() > 0);
        try (GGUFModel m = new GGUFModel(model.toString(), config(2));
             BeamSearchEngine engine = BeamSearchEngine.forModel(m, config(2))) {
            BeamSearchOutput out = engine.search(prompt, beamRequest(1, 1.0f));
            assertEquals(1, out.sequences().size());
            BeamSearchSequence best = out.sequences().get(0);
            assertArrayEquals("token ids", reference.tokens(), best.tokens());
            assertEquals("text", reference.text(), best.text());
            assertEquals("end reason", reference.stopReason(), best.stopReason());
            assertTrue("the search ran no more steps than the reference generated tokens",
                    out.steps() <= reference.tokenCount() + 1);
        }
    }

    @Test
    public void widthTwoRanksItsBeamsAndNeverFallsBelowTheWidthOneScore() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        int[] prompt = prompt(5, 5);
        EngineConfig cfg = config(2);
        try (GGUFModel m = new GGUFModel(model.toString(), cfg);
             BeamSearchEngine engine = BeamSearchEngine.forModel(m, cfg)) {
            BeamSearchOutput one = engine.search(prompt, beamRequest(1, 1.0f));
            BeamSearchOutput two = engine.search(prompt, beamRequest(2, 1.0f));
            assertEquals(2, two.sequences().size());
            double first = two.sequences().get(0).score(1.0f);
            double second = two.sequences().get(1).score(1.0f);
            assertTrue("the ranking ordered the beams", first >= second);
            assertTrue("the wider search matched or beat the greedy run's score",
                    first >= one.sequences().get(0).score(1.0f) - 1e-9);
            // the two beams grew apart: the runner-up is not a copy of the winner
            assertNotEquals(two.sequences().get(0).cumulativeLogprob(),
                    two.sequences().get(1).cumulativeLogprob(), 0.0);
        }
    }

    @Test
    public void anEncoderDecoderModelRefusesTheSearch() throws IOException {
        Path model = SyntheticModels.t5().writeTempFile();
        try (GGUFModel m = new GGUFModel(model.toString(), config(1))) {
            UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class,
                    () -> BeamSearchEngine.forModel(m, config(1)));
            assertTrue(e.getMessage(), e.getMessage().contains("encoder-decoder"));
            assertEquals(SupportClass.ENC_DEC, m.supportClass());
        }
    }

    @Test
    public void theRefusedFeaturesNameTheirRequestField() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        try (GGUFModel m = new GGUFModel(model.toString(), config(2));
             BeamSearchEngine engine = BeamSearchEngine.forModel(m, config(2))) {
            int[] prompt = prompt(7, 4);
            List<String> refused = List.of(
                    "stop", "bad_words", "constraint", "logit_bias", "logprob_token_ids",
                    "prompt_logprobs", "min_tokens", "n_keep", "n_discard", "repetition_detection",
                    "reasoning_budget", "grp_attn_n", "vllm_xargs");
            List<Function<GenerationRequest.Builder, GenerationRequest.Builder>> setters = List.of(
                    b -> b.stopStrings("halt"),
                    b -> b.badWords(List.of("banned")),
                    b -> b.constraint(ConstraintSpec.choice(List.of("yes", "no"))),
                    b -> b.logitBias(4, 1.0f),
                    b -> b.logprobs(1).logprobTokenIds(4),
                    b -> b.promptLogprobs(1),
                    b -> b.minTokens(2),
                    b -> b.nKeep(-1),
                    b -> b.nDiscard(2),
                    b -> b.repetitionDetection(RepetitionDetection.of(3, 1, 4)),
                    b -> b.reasoningBudget(64),
                    b -> b.grpAttnN(4),
                    b -> b.logitsProcessorArgs(LogitsProcessorArgs.of(Map.of("target_token", 5L))));
            for (int i = 0; i < setters.size(); i++) {
                GenerationRequest request = setters.get(i)
                        .apply(GenerationRequest.builder().maxNewTokens(N_NEW)
                                .sampler(SamplerConfig.greedyDecoding()).beamWidth(2))
                        .build();
                IllegalArgumentException e = assertThrows(refused.get(i),
                        IllegalArgumentException.class, () -> engine.search(prompt, request));
                assertTrue(refused.get(i) + " was not named in: " + e.getMessage(),
                        e.getMessage().contains(refused.get(i)));
            }
        }
    }

    @Test
    public void anEmptyPromptRefusesAndTheStepRequestKeepsEndTokensSampleable() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        try (GGUFModel m = new GGUFModel(model.toString(), config(2));
             BeamSearchEngine engine = BeamSearchEngine.forModel(m, config(2))) {
            assertThrows(IllegalArgumentException.class, () -> engine.search(new int[0], beamRequest(1, 1.0f)));
            // vLLM's base_sampling_params: one token and the wide raw row, no stop handling of its own — the
            // end decision belongs to the search over the captured rows
            GenerationRequest step = BeamSearchEngine.stepRequest(beamRequest(2, 1.0f));
            assertEquals(1, step.maxNewTokens());
            assertEquals(4, step.logprobs());
            assertTrue("the step samples end tokens; the host decides over the rows",
                    step.stopTokenIds().length == 0 && step.stopStrings().isEmpty() && !step.ignoreEos());
        }
    }
}

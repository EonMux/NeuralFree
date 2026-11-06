package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticEagle;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.file.Path;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertTrue;

/**
 * Speculative decoding with an EAGLE-3 drafter in vLLM's Qwen3 layout ({@code qwen3_eagle3.Eagle3Qwen3ForCausalLM},
 * the Qwen3 EAGLE-3 extensions of {@link SyntheticEagle#eagle3Qwen3}): the Q/K projection norms, the projection
 * biases, the whole-input {@code norm_before_fc} norm and the per-tap {@code fc_norm} encoder norms all load through
 * the EAGLE3 row, and the drafter keeps the target's plain greedy tokens (the verification is lossless). A
 * {@code logit_scale} only reweights the draft pick; it cannot change the emitted tokens.
 */
public class Eagle3Qwen3SpeculationTest {

    private static final int[] PROMPT = {1, 17, 42, 5, 99, 26, 3, 64, 8, 51};
    private static final int GREEDY_TOKENS = 12;
    private static final int TARGET_LAYERS = 4;
    private static final int[] TAPS = {1, 2, 3};

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    /** A four-layer llama target over the shared synthetic dimensions. */
    private Path target() throws IOException {
        return SyntheticModels.llama(SyntheticModels.Config.llama().toBuilder().nLayer(TARGET_LAYERS).build())
                .writeTempFile();
    }

    private static SyntheticModels.Config drafterDims() {
        return SyntheticModels.Config.llama().toBuilder().nLayer(1).seed(44L).build();
    }

    private static EngineConfig.Builder base() {
        MemoryConfig memory = MemoryConfig.builder().nCtx(64).ubatch(8).typeK(KvDType.F32).typeV(KvDType.F32)
                .build();
        return EngineConfig.builder().backend(EngineConfig.BackendMode.CPU).memory(memory);
    }

    private static GenerationRequest greedy(int maxNewTokens) {
        return GenerationRequest.builder().maxNewTokens(maxNewTokens).sampler(SamplerConfig.greedyDecoding())
                .build();
    }

    @Test
    public void theQwen3AttentionLayoutKeepsThePlainGreedyTokens() throws IOException {
        keepsThePlainGreedyTokens(false, 1f);
    }

    @Test
    public void thePerTapEncoderNormsKeepThePlainGreedyTokens() throws IOException {
        keepsThePlainGreedyTokens(true, 1f);
    }

    @Test
    public void aLogitScaleOnlyReweightsTheDraftPick() throws IOException {
        keepsThePlainGreedyTokens(true, 2.5f, false);
    }

    @Test
    public void theWholeInputAndPerTapNormsTogetherKeepThePlainGreedyTokens() throws IOException {
        keepsThePlainGreedyTokens(true, 1f, true);
    }

    private void keepsThePlainGreedyTokens(boolean fcNorm, float logitScale) throws IOException {
        keepsThePlainGreedyTokens(fcNorm, logitScale, false);
    }

    private void keepsThePlainGreedyTokens(boolean fcNorm, float logitScale, boolean normBeforeFc)
            throws IOException {
        Path target = target();
        Path draft = SyntheticEagle.eagle3Qwen3(drafterDims(), TAPS, fcNorm, logitScale, normBeforeFc)
                .writeTempFile();
        try (EngineModel m = LLMModelFactory.open(target.toString(), base().build())) {
            int[] plain = m.generate(PROMPT, greedy(GREEDY_TOKENS), null).tokens();
            m.resetKVCache();

            try (SpeculativeSession s = LLMModelFactory.openSpeculative(target.toString(), draft.toString(),
                    base().draftMax(3).build())) {
                GenerationResult r = s.generate(PROMPT, greedy(GREEDY_TOKENS), null);
                assertArrayEquals("qwen3 eagle3 greedy over the target", plain, r.tokens());
                SpeculativeDecoder.Stats stats = s.lastStats();
                assertTrue("the drafter made verify rounds: " + stats, stats.rounds() > 0);
                assertTrue("the drafter proposed tokens: " + stats, stats.drafted() > 0);
            }
        }
    }
}

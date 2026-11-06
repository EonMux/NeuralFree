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
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Speculative decoding with an EAGLE-1 drafter file ({@code LLM_ARCH_EAGLE}, the fused cnets drafter) on the CPU
 * engine: the drafter bound through {@link LLMModelFactory#openSpeculative} keeps the plain greedy tokens of its
 * target (the verification is lossless) with rounds that verify drafted tokens, in both checkpoint shapes — the
 * mistral one with the input and final norms and the llama one with only the layer's {@code ffn_norm}. A drafter of
 * another width refuses its target by name, and the draft file refuses to open standalone.
 */
public class Eagle1SpeculationTest {

    private static final int[] PROMPT = {1, 17, 42, 5, 99, 26, 3, 64, 8, 51};
    private static final int GREEDY_TOKENS = 12;
    private static final int TARGET_LAYERS = 4;
    private static final int TARGET_LAST = TARGET_LAYERS - 1;

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    /** A four-layer llama target over the shared synthetic dimensions. */
    private Path target() throws IOException {
        return SyntheticModels.llama(SyntheticModels.Config.llama().toBuilder().nLayer(TARGET_LAYERS).build())
                .writeTempFile();
    }

    /** A one-layer EAGLE drafter of the target's dimensions, tapping the target's last layer. */
    private Path drafter(boolean withNorms) throws IOException {
        return SyntheticEagle.eagle(drafterDims(), withNorms, TARGET_LAST).writeTempFile();
    }

    private static SyntheticModels.Config drafterDims() {
        return SyntheticModels.Config.llama().toBuilder().nLayer(1).seed(43L).build();
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
    public void theNormedDrafterKeepsThePlainGreedyTokens() throws IOException {
        keepsThePlainGreedyTokens(target(), drafter(true));
    }

    @Test
    public void theBareDrafterWithoutNormsKeepsThePlainGreedyTokens() throws IOException {
        keepsThePlainGreedyTokens(target(), drafter(false));
    }

    private void keepsThePlainGreedyTokens(Path target, Path draft) throws IOException {
        try (EngineModel m = LLMModelFactory.open(target.toString(), base().build())) {
            int[] plain = m.generate(PROMPT, greedy(GREEDY_TOKENS), null).tokens();
            m.resetKVCache();

            try (SpeculativeSession s = LLMModelFactory.openSpeculative(target.toString(), draft.toString(),
                    base().draftMax(3).build())) {
                GenerationResult r = s.generate(PROMPT, greedy(GREEDY_TOKENS), null);
                assertArrayEquals("eagle greedy over the fused drafter", plain, r.tokens());
                SpeculativeDecoder.Stats stats = s.lastStats();
                assertTrue("the drafter made verify rounds: " + stats, stats.rounds() > 0);
                assertTrue("the drafter proposed tokens: " + stats, stats.drafted() > 0);
                assertTrue("every round emits at most drafted + 1 tokens: " + stats,
                        stats.accepted() <= stats.drafted());
            }
        }
    }

    /** A drafter whose fusion width is not the target's hidden size refuses the pair by name. */
    @Test
    public void aDrafterOfAnotherWidthRefusesItsTarget() throws IOException {
        Path target = target();
        Path draft = SyntheticEagle.eagle(SyntheticModels.Config.llama().toBuilder().nLayer(1).nEmbd(32)
                        .nHead(2).nHeadKv(1).nFf(64).seed(43L).build(), TARGET_LAST)
                .writeTempFile();
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> LLMModelFactory.openSpeculative(target.toString(), draft.toString(),
                        base().draftMax(3).build()));
        assertTrue(e.getMessage(), e.getMessage().contains("target_hidden_size"));
    }

    /** The draft file alone is refused: draft-only architectures run beside their target. */
    @Test
    public void anEagleFileRefusesToOpenStandalone() throws IOException {
        Path draft = SyntheticEagle.eagleWithTokenizer(drafterDims(), true, TARGET_LAST).writeTempFile();
        UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class,
                () -> LLMModelFactory.open(draft.toString(), base().build()));
        assertTrue(e.getMessage(), e.getMessage().contains("draft model"));
    }
}

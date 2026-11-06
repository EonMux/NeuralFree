package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Speculative decoding over a model's own NextN / MTP blocks on the CPU engine ({@code neuralfree.mtp}): the
 * drafter bound through {@link LLMModelFactory#speculate} keeps the plain greedy tokens of the same model (the
 * verification is lossless) while its rounds verify drafted tokens, and the drafter catches up over every target
 * call — a prompt batch's cache writes only, a verification's pairs with the next draft's first pass.
 */
public class NextnMtpSpeculationTest {

    private static final int[] PROMPT = {1, 17, 42, 5, 99, 26, 3, 64, 8, 51};
    private static final int GREEDY_TOKENS = 12;

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    /** A glm4moe model of two trunk layers and one NextN block. */
    private Path mtpModel() throws IOException {
        return SyntheticModels.glm4moe(SyntheticModels.Config.qwen3moe().toBuilder().nLayer(2).build(), 1)
                .writeTempFile();
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
    public void theNextnDrafterKeepsThePlainGreedyTokens() throws IOException {
        Path model = mtpModel();
        try (EngineModel m = LLMModelFactory.open(model.toString(), base().build())) {
            int[] plain = m.generate(PROMPT, greedy(GREEDY_TOKENS), null).tokens();
            m.resetKVCache();

            EngineConfig cfg = base().mtp(true).draftMax(3).build();
            try (SpeculativeSession s = LLMModelFactory.speculate(m, cfg)) {
                GenerationResult r = s.generate(PROMPT, greedy(GREEDY_TOKENS), null);
                assertArrayEquals("MTP greedy over the target's own blocks", plain, r.tokens());
                SpeculativeDecoder.Stats stats = s.lastStats();
                assertTrue("the drafter made verify rounds: " + stats, stats.rounds() > 0);
                assertTrue("the drafter proposed tokens: " + stats, stats.drafted() > 0);
                assertEquals("every round emits at most drafted + 1 tokens", true,
                        stats.accepted() <= stats.drafted());
            }
        }
    }

    /** A qwen2 model of two trunk layers and one MiMo-style NextN block. */
    private Path qwen2MtpModel() throws IOException {
        return SyntheticModels.qwen2Mtp(SyntheticModels.Config.qwen2().toBuilder().nLayer(2).build(), 1)
                .writeTempFile();
    }

    @Test
    public void theQwen2MimoNextnDrafterKeepsThePlainGreedyTokens() throws IOException {
        Path model = qwen2MtpModel();
        try (EngineModel m = LLMModelFactory.open(model.toString(), base().build())) {
            int[] plain = m.generate(PROMPT, greedy(GREEDY_TOKENS), null).tokens();
            m.resetKVCache();

            EngineConfig cfg = base().mtp(true).draftMax(3).build();
            try (SpeculativeSession s = LLMModelFactory.speculate(m, cfg)) {
                GenerationResult r = s.generate(PROMPT, greedy(GREEDY_TOKENS), null);
                assertArrayEquals("Qwen2/MiMo MTP greedy over the target's own blocks", plain, r.tokens());
                assertTrue("the drafter proposed tokens: " + s.lastStats(), s.lastStats().drafted() > 0);
                assertTrue(s.lastStats().accepted() <= s.lastStats().drafted());
            }
        } finally {
            Files.deleteIfExists(model);
        }
    }

    /**
     * A separate draft file carrying the target model's MTP block, the draft-file form vLLM's {@code MiMoMTPModel}
     * takes (it loads the MTP layer, the shared embedding and the head of the checkpoint the target is served from):
     * the drafter reads the file's NextN block, never its trunk, and keeps the target's greedy tokens.
     */
    @Test
    public void aStandaloneMtpFileDraftsForItsTarget() throws IOException {
        SyntheticModels.Config c = SyntheticModels.Config.qwen2().toBuilder().nLayer(2).build();
        Path target = SyntheticModels.qwen2Mtp(c, 0).writeTempFile();
        Path draft = SyntheticModels.qwen2Mtp(c, 1).writeTempFile();
        try (EngineModel m = LLMModelFactory.open(target.toString(), base().build())) {
            int[] plain = m.generate(PROMPT, greedy(GREEDY_TOKENS), null).tokens();
            m.resetKVCache();

            try (SpeculativeSession s = LLMModelFactory.openSpeculative(target.toString(), draft.toString(),
                    base().draftMax(3).build())) {
                GenerationResult r = s.generate(PROMPT, greedy(GREEDY_TOKENS), null);
                assertArrayEquals("a standalone MTP draft keeps the target's greedy tokens", plain, r.tokens());
                SpeculativeDecoder.Stats stats = s.lastStats();
                assertTrue("the standalone MTP file drafted: " + stats, stats.drafted() > 0);
            }
        } finally {
            Files.deleteIfExists(target);
            Files.deleteIfExists(draft);
        }
    }

    /** A NextN-less model refuses the MTP head by name. */
    @Test
    public void aModelWithoutNextnBlocksRefusesTheMtpHead() throws IOException {
        Path model = SyntheticModels.llama(SyntheticModels.Config.llama().toBuilder().nLayer(2).build())
                .writeTempFile();
        try (EngineModel m = LLMModelFactory.open(model.toString(), base().build())) {
            RuntimeException e = assertThrows(RuntimeException.class,
                    () -> LLMModelFactory.speculate(m, base().mtp(true).build()).close());
            assertTrue(e.getMessage(), e.getMessage().contains("NextN"));
        } finally {
            Files.deleteIfExists(model);
        }
    }
}

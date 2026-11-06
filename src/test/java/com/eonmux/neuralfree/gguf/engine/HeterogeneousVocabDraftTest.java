package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.TokenCallback;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertTrue;

/**
 * Single-sequence drafting with a draft model whose vocabulary differs from the target's: the tokens both
 * tokenizers carry (the same texts at other ids) drive the draft, and greedy verification keeps the target's
 * tokens, so the speculative run produces exactly the plain run's output.
 */
public class HeterogeneousVocabDraftTest {

    private static final int N_CTX = 256;
    private static final int NEW_TOKENS = 12;
    private static final int[] PROMPT = {1, 4, 5, 6, 7, 8};

    private static Path targetFile;
    private static Path draftFile;

    @BeforeClass
    public static void writeModels() throws IOException {
        targetFile = SyntheticModels.llama().writeTempFile();
        draftFile = SyntheticModels.llamaRotatedVocab(SyntheticModels.Config.llama(), 7).writeTempFile();
    }

    @AfterClass
    public static void deleteModels() throws IOException {
        Files.deleteIfExists(targetFile);
        Files.deleteIfExists(draftFile);
    }

    private static EngineConfig.Builder config(int draftMax) {
        MemoryConfig memory = MemoryConfig.builder().nCtx(N_CTX).ubatch(8).typeK(KvDType.F32).typeV(KvDType.F32)
                .build();
        return EngineConfig.builder().backend(EngineConfig.BackendMode.CPU).memory(memory).draftMax(draftMax);
    }

    private static GenerationRequest greedy() {
        return GenerationRequest.builder().maxNewTokens(NEW_TOKENS).sampler(SamplerConfig.greedyDecoding()).build();
    }

    @Test
    public void aDraftOfAnotherVocabularyDraftsThroughTheTokenIntersection() throws IOException {
        int[] expected;
        EngineConfig plain = config(1).build();
        try (GGUFModel model = new GGUFModel(targetFile.toString(), plain)) {
            expected = GenerationLoop.forModel(model, plain).generate(PROMPT, greedy(), (TokenCallback) null).tokens();
        }

        EngineConfig cfg = config(4).draftModelPath(draftFile.toString())
                .draftType(EngineConfig.DraftType.MODEL)
                .speculativeOptions(SpeculativeOptions.builder().heterogeneousVocab(true).build())
                .build();
        try (SpeculativeSession session = LLMModelFactory.openSpeculative(targetFile.toString(), null, cfg)) {
            assertTrue("the draft file runs as a separate model", session.drafter() instanceof ModelDrafter);
            GenerationResult result = session.generate(PROMPT, greedy(), (TokenCallback) null);
            assertArrayEquals("greedy speculation keeps the target's tokens", expected, result.tokens());
            SpeculativeDecoder.Stats stats = session.lastStats();
            assertTrue("the draft proposed tokens: " + stats, stats.drafted() > 0);
        }
    }

    @Test
    public void theHeterogeneousFlagReachesTheDrafter() throws IOException {
        EngineConfig cfg = config(2).draftModelPath(draftFile.toString())
                .draftType(EngineConfig.DraftType.MODEL)
                .speculativeOptions(SpeculativeOptions.builder().heterogeneousVocab(true).build())
                .build();
        assertTrue(cfg.speculativeOptions().heterogeneousVocab());
        try (SpeculativeSession session = LLMModelFactory.openSpeculative(targetFile.toString(), null, cfg)) {
            assertTrue(session.drafter().toString(), session.drafter().toString().contains("heterogeneous vocab"));
        }
    }

    @Test
    public void withoutTheFlagTheIncompatibleVocabularyIsRefusedByName() throws IOException {
        EngineConfig cfg = config(2).draftModelPath(draftFile.toString())
                .draftType(EngineConfig.DraftType.MODEL)
                .build();
        try {
            LLMModelFactory.openSpeculative(targetFile.toString(), null, cfg).close();
            throw new AssertionError("a token-incompatible draft model without heterogeneous drafting is refused");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("cannot draft for"));
            assertTrue(e.getMessage(), e.getMessage().contains("differs"));
        }
    }
}

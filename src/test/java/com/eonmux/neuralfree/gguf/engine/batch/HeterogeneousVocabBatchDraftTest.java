package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.TokenCallback;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.GenerationLoop;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.engine.SpeculativeOptions;
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
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The batch engine's draft model with a vocabulary of its own: the target's tokens are translated into the draft
 * vocabulary, the draft logits are masked to the token-level intersection, and the proposals come back in the
 * target's ids, so greedy sequences finish with the tokens of a plain batch run. Heterogeneous drafting beside
 * probabilistic drafting is refused by name.
 */
public class HeterogeneousVocabBatchDraftTest {

    private static final int N_CTX = 512;
    private static final int NEW_TOKENS = 12;
    private static final int[] FIRST = {1, 4, 5, 6, 7, 8};
    private static final int[] SECOND = {1, 9, 10, 11, 12, 13};

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

    private static EngineConfig.Builder config(int parallel) {
        MemoryConfig memory = MemoryConfig.builder().nCtx(N_CTX).ubatch(8).typeK(KvDType.F32).typeV(KvDType.F32)
                .parallel(parallel).build();
        return EngineConfig.builder().backend(EngineConfig.BackendMode.CPU).memory(memory).draftMax(4);
    }

    private static GenerationRequest greedy() {
        return GenerationRequest.builder().maxNewTokens(NEW_TOKENS).sampler(SamplerConfig.greedyDecoding()).build();
    }

    private static EngineConfig heterogeneous() {
        return config(2).draftModelPath(draftFile.toString())
                .draftType(EngineConfig.DraftType.MODEL)
                .speculativeOptions(SpeculativeOptions.builder().heterogeneousVocab(true).build())
                .build();
    }

    /** The same prompt and request through the single-sequence loop over the target alone. */
    private static GenerationResult reference(int[] prompt) throws IOException {
        EngineConfig cfg = config(1).build();
        try (GGUFModel model = new GGUFModel(targetFile.toString(), cfg)) {
            return GenerationLoop.forModel(model, cfg).generate(prompt, greedy(), (TokenCallback) null);
        }
    }

    @Test
    public void bothRunningSequencesDraftAcrossTheirVocabularyIntersection() throws IOException {
        GenerationResult first = reference(FIRST);
        GenerationResult second = reference(SECOND);
        EngineConfig cfg = heterogeneous();
        try (GGUFModel target = new GGUFModel(targetFile.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(target, cfg)) {
            BatchSequence a = engine.submit(FIRST, greedy(), (TokenCallback) null);
            BatchSequence b = engine.submit(SECOND, greedy(), (TokenCallback) null);
            engine.drain();
            assertArrayEquals("first sequence", first.tokens(), a.result().tokens());
            assertArrayEquals("second sequence", second.tokens(), b.result().tokens());
            BatchStats stats = engine.stats();
            assertTrue("the draft proposed tokens: " + stats, stats.specDrafted() > 0);
            assertTrue("rounds verified: " + stats, stats.specRounds() > 0);
        }
    }

    @Test
    public void heterogeneousDraftingRefusesProbabilisticDraftingByName() {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> config(1)
                .draftModelPath(draftFile.toString())
                .draftType(EngineConfig.DraftType.MODEL)
                .speculativeOptions(SpeculativeOptions.builder().heterogeneousVocab(true)
                        .probabilisticDraft(true).build())
                .build());
        assertTrue(e.getMessage(), e.getMessage().contains(SpeculativeOptions.PROP_HETEROGENEOUS_VOCAB));
        assertTrue(e.getMessage(), e.getMessage().contains(SpeculativeOptions.PROP_DRAFT_SAMPLE_METHOD));
    }
}

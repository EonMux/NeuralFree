package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.gguf.TokenCallback;
import com.eonmux.neuralfree.gguf.engine.draft.VerifyMode;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Single-pass draft-tree verification with a dflash drafter ({@code neuralfree.draft.tree}): the drafter proposes a
 * best-first tree over the block's per-slot candidates and the target verifies it in one forward over page-shared
 * sequences. Every committed token equals the target-only loop's, greedy and sampled, for the plain DFlash and the
 * DFlash2 drafts, on the CPU and on every available GPU backend; unsupported shapes are refused by name. The random
 * fixture pair's acceptance is a seed-level coin flip (the drafter's weights are uncorrelated with the target's), so
 * the accepted-commit path itself runs under {@link TreeVerifySinglePassTest}'s deterministic drafter.
 */
public class DflashTreeSpeculativeTest {

    private static final int N_CTX = 256;
    private static final int N_NEW = 16;
    private static final int DRAFT_MAX = 4;
    private static final int PARALLEL = 3;
    private static final int BLOCK = 8;
    private static final int[] TARGET_LAYERS = {0, 2};

    private static Path targetFile;
    private static Path dflashFile;
    private static Path dflash2File;

    @BeforeClass
    public static void writeModels() throws IOException {
        SyntheticModels.Config c = SyntheticModels.Config.llama();
        targetFile = SyntheticModels.llama(c.toBuilder().nLayer(4).build()).writeTempFile();
        dflashFile = SyntheticModels.dflashDrafter(c.toBuilder().nLayer(2).build(), TARGET_LAYERS, BLOCK)
                .writeTempFile();
        dflash2File = SyntheticModels.dflashDrafter(c.toBuilder().nLayer(2).build(), TARGET_LAYERS, BLOCK, false, 0,
                null, new SyntheticModels.Dflash2(2, 4, 8, 4)).writeTempFile();
    }

    @AfterClass
    public static void deleteModels() throws IOException {
        for (Path p : Arrays.asList(targetFile, dflashFile, dflash2File)) {
            Files.deleteIfExists(p);
        }
    }

    private static GenerationRequest greedy(int maxNewTokens) {
        return GenerationRequest.builder().maxNewTokens(maxNewTokens).sampler(SamplerConfig.greedyDecoding())
                .build();
    }

    private static GenerationRequest sampled(int maxNewTokens) {
        return GenerationRequest.builder().maxNewTokens(maxNewTokens)
                .sampler(SamplerConfig.builder().temperature(0.8f).seed(42L).build()).build();
    }

    /** The tree configuration over the paged unified memory, with the tree verification on. */
    private static EngineConfig treeConfig(Path draft, EngineConfig.BackendMode backend) {
        MemoryConfig memory = MemoryConfig.builder().nCtx(N_CTX).ubatch(8)
                .typeK(KvDType.F32).typeV(KvDType.F32).parallel(PARALLEL).kvUnified(true).build();
        return EngineConfig.builder().backend(backend).memory(memory)
                .draftModelPath(draft.toString()).draftMax(DRAFT_MAX)
                .speculativeOptions(SpeculativeOptions.builder().treeVerify(true).build()).build();
    }

    /** The target-only loop over the same model and backend. */
    private static GenerationResult plain(int[] prompt, GenerationRequest request,
                                          EngineConfig.BackendMode backend) throws IOException {
        MemoryConfig memory = MemoryConfig.builder().nCtx(N_CTX).ubatch(8)
                .typeK(KvDType.F32).typeV(KvDType.F32).build();
        EngineConfig cfg = EngineConfig.builder().backend(backend).memory(memory).build();
        try (EngineModel model = LLMModelFactory.open(targetFile.toString(), cfg)) {
            return GenerationLoop.forModel(model, cfg).generate(prompt, request, (TokenCallback) null);
        }
    }

    private static int[] prompt() {
        return new int[] {4, 5, 6, 4, 5, 7, 4, 5, 8, 4, 5};
    }

    private static void assertTreeGeneration(Path draft, EngineConfig.BackendMode backend) throws IOException {
        int[] prompt = prompt();
        GenerationRequest request = greedy(N_NEW);
        GenerationResult expected = plain(prompt, request, backend);
        EngineConfig cfg = treeConfig(draft, backend);
        try (EngineModel model = LLMModelFactory.open(targetFile.toString(), cfg);
             SpeculativeSession session = LLMModelFactory.speculate(model, cfg)) {
            GenerationResult actual = session.generate(prompt, request, (TokenCallback) null);
            assertArrayEquals("single-pass tree drafting keeps the target-only greedy tokens", expected.tokens(),
                    actual.tokens());
            assertEquals("stop reason", expected.stopReason(), actual.stopReason());
            SpeculativeDecoder.Stats s = session.lastStats();
            assertTrue("tree rounds ran: " + s, s.rounds() > 0);
            assertTrue("the drafter proposed trees: " + s, s.drafted() > 0);
        }
    }

    @Test
    public void aPlainDflashTreeKeepsTheGreedyTokens() throws IOException {
        assertTreeGeneration(dflashFile, EngineConfig.BackendMode.CPU);
    }

    @Test
    public void aDflash2TreeKeepsTheGreedyTokens() throws IOException {
        assertTreeGeneration(dflash2File, EngineConfig.BackendMode.CPU);
    }

    @Test
    public void thePlainDflashTreeKeepsTheGreedyTokensOnCuda() throws IOException {
        GpuTestUtil.assumeCudaAvailable();
        assertTreeGeneration(dflashFile, EngineConfig.BackendMode.CUDA);
    }

    @Test
    public void thePlainDflashTreeKeepsTheGreedyTokensOnOpenCL() throws IOException {
        GpuTestUtil.assumeOpenCLAvailable();
        assertTreeGeneration(dflashFile, EngineConfig.BackendMode.OPENCL);
    }

    @Test
    public void aSampledTreeKeepsThePlainSampledTokens() throws IOException {
        int[] prompt = prompt();
        GenerationRequest request = sampled(9);
        GenerationResult expected = plain(prompt, request, EngineConfig.BackendMode.CPU);
        EngineConfig cfg = treeConfig(dflashFile, EngineConfig.BackendMode.CPU);
        int[] first;
        int[] second;
        SpeculativeDecoder.Stats stats;
        try (EngineModel model = LLMModelFactory.open(targetFile.toString(), cfg);
             SpeculativeSession session = LLMModelFactory.speculate(model, cfg)) {
            first = session.generate(prompt, request, (TokenCallback) null).tokens();
            stats = session.lastStats();
            second = session.generate(prompt, request, (TokenCallback) null).tokens();
        }
        assertArrayEquals("a sampled generation repeats for a fixed seed", first, second);
        assertEquals("the budget is spent", 9, first.length);
        assertArrayEquals("single-pass tree verification reproduces the plain sampled tokens", expected.tokens(),
                first);
        assertTrue("rounds ran: " + stats, stats.rounds() > 0);
        assertTrue("tokens were drafted: " + stats, stats.drafted() > 0);
    }

    @Test
    public void repeatedGenerationsRestartFromThePrompt() throws IOException {
        EngineConfig cfg = treeConfig(dflashFile, EngineConfig.BackendMode.CPU);
        try (EngineModel model = LLMModelFactory.open(targetFile.toString(), cfg);
             SpeculativeSession session = LLMModelFactory.speculate(model, cfg)) {
            int[] first = session.generate(prompt(), greedy(N_NEW), (TokenCallback) null).tokens();
            int[] second = session.generate(prompt(), greedy(N_NEW), (TokenCallback) null).tokens();
            assertArrayEquals("each generation resets the memory and the drafter's rows", first, second);
        }
    }

    @Test
    public void theChainFallbackWithoutTheTreeOptionStaysLossless() throws IOException {
        // the same drafter without the tree option runs its ordinary chain rounds; the chain path is a
        // contiguous-memory route (its prefill maps KV positions through the plan, not PagedSequences), so the
        // fallback test runs over the plain memory
        int[] prompt = prompt();
        GenerationRequest request = greedy(N_NEW);
        GenerationResult expected = plain(prompt, request, EngineConfig.BackendMode.CPU);
        MemoryConfig memory = MemoryConfig.builder().nCtx(N_CTX).ubatch(8)
                .typeK(KvDType.F32).typeV(KvDType.F32).build();
        EngineConfig chained = EngineConfig.builder().backend(EngineConfig.BackendMode.CPU).memory(memory)
                .draftModelPath(dflashFile.toString()).draftMax(DRAFT_MAX)
                .speculativeOptions(SpeculativeOptions.builder().treeVerify(false).build()).build();
        try (EngineModel model = LLMModelFactory.open(targetFile.toString(), chained);
             SpeculativeSession session = LLMModelFactory.speculate(model, chained)) {
            GenerationResult actual = session.generate(prompt, request, (TokenCallback) null);
            assertArrayEquals("the chain fallback keeps the greedy tokens too", expected.tokens(), actual.tokens());
            assertTrue(session.lastStats().rounds() > 0);
        }
    }

    @Test
    public void aNonMatchVerifyModeRefusesTheTree() throws IOException {
        EngineConfig cfg = treeConfig(dflashFile, EngineConfig.BackendMode.CPU).toBuilder()
                .speculativeOptions(SpeculativeOptions.builder().treeVerify(true)
                        .verifyMode(VerifyMode.REJECTION).build())
                .build();
        try (EngineModel model = LLMModelFactory.open(targetFile.toString(), cfg);
             SpeculativeSession session = LLMModelFactory.speculate(model, cfg)) {
            IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                    () -> session.generate(prompt(), greedy(N_NEW), (TokenCallback) null));
            assertTrue(e.getMessage(), e.getMessage().contains("neuralfree.draft.verify"));
        }
    }

    /**
     * A non-unified paged memory (each sequence holds a fixed {@code nCtx / parallel} slice) runs the same round:
     * the verify is one target sequence of the fixed-width rows, so nothing needs the sequences to share one
     * context, and the tokens stay the plain ones.
     */
    @Test
    public void aNonUnifiedPagedMemoryRunsTheTreeLosslessly() throws IOException {
        MemoryConfig memory = MemoryConfig.builder().nCtx(N_CTX).ubatch(8)
                .typeK(KvDType.F32).typeV(KvDType.F32).parallel(PARALLEL).kvUnified(false).build();
        EngineConfig cfg = EngineConfig.builder().backend(EngineConfig.BackendMode.CPU).memory(memory)
                .draftModelPath(dflashFile.toString()).draftMax(DRAFT_MAX)
                .speculativeOptions(SpeculativeOptions.builder().treeVerify(true).build()).build();
        int[] prompt = prompt();
        GenerationRequest request = greedy(N_NEW);
        GenerationResult expected = plain(prompt, request, EngineConfig.BackendMode.CPU);
        try (EngineModel model = LLMModelFactory.open(targetFile.toString(), cfg);
             SpeculativeSession session = LLMModelFactory.speculate(model, cfg)) {
            GenerationResult actual = session.generate(prompt, request, (TokenCallback) null);
            assertArrayEquals("the tree over a non-unified paged memory keeps the target-only greedy tokens",
                    expected.tokens(), actual.tokens());
            assertTrue(session.lastStats().rounds() > 0);
        }
    }
}

package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
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
import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Multi-branch (tree) speculative drafting over the synthetic llama on the CPU engine: with a target and a draft
 * model over the same paged sequences, a greedy round keeps the target-only greedy tokens exactly, the proposal
 * splits into branches (llama.cpp's {@code p_split}), the whole tree verifies in one fixed-width forward of the one
 * target sequence under the tree mask, and the round reports proposals and acceptances.
 */
public class TreeSpeculativeDecoderTest {

    private static final int N_CTX = 256;
    private static final int N_NEW = 12;
    private static final int DRAFT_MAX = 4;
    private static final int PARALLEL = 3;

    private static Path modelFile;

    @BeforeClass
    public static void writeModel() throws IOException {
        modelFile = SyntheticModels.llama().writeTempFile();
    }

    @AfterClass
    public static void deleteModel() throws IOException {
        if (modelFile != null) {
            Files.deleteIfExists(modelFile);
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

    private static EngineConfig config(double pSplit) {
        return config(pSplit, EngineConfig.BackendMode.CPU);
    }

    private static EngineConfig config(double pSplit, EngineConfig.BackendMode backend) {
        MemoryConfig memory = MemoryConfig.builder().nCtx(N_CTX).ubatch(8)
                .typeK(KvDType.F32).typeV(KvDType.F32).parallel(PARALLEL).kvUnified(true).build();
        return EngineConfig.builder().backend(backend).memory(memory)
                .draftModelPath(modelFile.toString()).draftMax(DRAFT_MAX)
                .speculativeOptions(SpeculativeOptions.builder().pSplit(pSplit).build()).build();
    }

    /** The plain single-sequence loop over the same model. */
    private static GenerationResult plain(int[] prompt, GenerationRequest request) throws IOException {
        return plain(prompt, request, EngineConfig.BackendMode.CPU);
    }

    private static GenerationResult plain(int[] prompt, GenerationRequest request,
                                          EngineConfig.BackendMode backend) throws IOException {
        MemoryConfig memory = MemoryConfig.builder().nCtx(N_CTX).ubatch(8)
                .typeK(KvDType.F32).typeV(KvDType.F32).build();
        EngineConfig cfg = EngineConfig.builder().backend(backend).memory(memory).build();
        try (EngineModel model = LLMModelFactory.open(modelFile.toString(), cfg)) {
            return GenerationLoop.forModel(model, cfg).generate(prompt, request, (TokenCallback) null);
        }
    }

    private static int[] prompt() {
        return new int[] {4, 5, 6, 4, 5, 7, 4, 5, 8, 4, 5};
    }

    @Test
    public void greedyTreeKeepsThePlainOutputAndAcceptsDrafts() throws IOException {
        int[] prompt = prompt();
        GenerationRequest request = greedy(N_NEW);
        GenerationResult expected = plain(prompt, request);
        EngineConfig cfg = config(0.1);
        List<SpeculativeDecoder.Stats> stats = new ArrayList<>();
        try (EngineModel model = LLMModelFactory.open(modelFile.toString(), cfg);
             SpeculativeSession session = LLMModelFactory.speculate(model, cfg)) {
            GenerationResult actual = session.generate(prompt, request, (TokenCallback) null);
            assertArrayEquals("tree drafting is lossless for a greedy target", expected.tokens(), actual.tokens());
            assertEquals("stop reason", expected.stopReason(), actual.stopReason());
            assertEquals("text", expected.text(), actual.text());
            assertTrue("the last generation reported stats", session.lastStats() != null);
            stats.add(session.lastStats());
        }
        SpeculativeDecoder.Stats s = stats.get(0);
        assertTrue("rounds ran: " + s, s.rounds() > 0);
        assertTrue("tokens were drafted: " + s, s.drafted() > 0);
        assertTrue("the target confirmed the chain of every round: " + s, s.accepted() > 0);
        assertTrue("the proposal branched past a linear chain: " + s, s.drafted() > s.rounds() * DRAFT_MAX);
    }

    @Test
    public void aSplitProbabilityOfOneProposesOneChain() throws IOException {
        int[] prompt = prompt();
        // the first sampled token plus whole rounds of DRAFT_MAX drafts and their bonus token: no round is cut
        // short by the token budget, so every drafted token is verified
        GenerationRequest request = greedy(1 + 2 * (DRAFT_MAX + 1));
        GenerationResult expected = plain(prompt, request);
        EngineConfig cfg = config(1.0);
        try (EngineModel model = LLMModelFactory.open(modelFile.toString(), cfg);
             SpeculativeSession session = LLMModelFactory.speculate(model, cfg)) {
            GenerationResult actual = session.generate(prompt, request, (TokenCallback) null);
            assertArrayEquals("a linear tree keeps the greedy tokens", expected.tokens(), actual.tokens());
            assertEquals("the drafter proposed one chain per round: " + session.lastStats(),
                    2 * DRAFT_MAX, session.lastStats().drafted());
            assertEquals("a linear proposal mirrors the target: " + session.lastStats(),
                    session.lastStats().drafted(), session.lastStats().accepted());
        }
    }

    @Test
    public void aSampledTreeKeepsThePlainSampledTokens() throws IOException {
        int[] prompt = prompt();
        GenerationRequest request = sampled(9);
        GenerationResult expected = plain(prompt, request);
        // p_split 1 keeps every round linear, 0.1 lets rounds branch: llama.cpp's token-match acceptance samples
        // the target once per emitted token either way, so the sampler's stream reproduces the plain loop
        for (double pSplit : new double[] {1.0, 0.1}) {
            EngineConfig cfg = config(pSplit);
            int[] first;
            int[] second;
            SpeculativeDecoder.Stats stats;
            try (EngineModel model = LLMModelFactory.open(modelFile.toString(), cfg);
                 SpeculativeSession session = LLMModelFactory.speculate(model, cfg)) {
                first = session.generate(prompt, request, (TokenCallback) null).tokens();
                stats = session.lastStats();
                second = session.generate(prompt, request, (TokenCallback) null).tokens();
            }
            assertArrayEquals("a sampled generation repeats for a fixed seed (p_split " + pSplit + ")",
                    first, second);
            assertEquals("the budget is spent (p_split " + pSplit + ")", 9, first.length);
            assertArrayEquals("tree verification reproduces the plain sampled tokens (p_split " + pSplit + ")",
                    expected.tokens(), first);
            assertTrue("rounds ran (p_split " + pSplit + "): " + stats, stats.rounds() > 0);
            assertTrue("tokens were drafted (p_split " + pSplit + "): " + stats, stats.drafted() > 0);
            assertTrue("confirmations stay within the proposals (p_split " + pSplit + "): " + stats,
                    stats.accepted() >= 0 && stats.accepted() <= stats.drafted());
        }
    }

    @Test
    public void repeatedGenerationsRestartFromThePrompt() throws IOException {
        EngineConfig cfg = config(0.1);
        try (EngineModel model = LLMModelFactory.open(modelFile.toString(), cfg);
             SpeculativeSession session = LLMModelFactory.speculate(model, cfg)) {
            int[] first = session.generate(prompt(), greedy(N_NEW), (TokenCallback) null).tokens();
            int[] second = session.generate(prompt(), greedy(N_NEW), (TokenCallback) null).tokens();
            assertArrayEquals("each generation resets the memory", first, second);
        }
    }

    /** The same tree round on a CUDA device: the fixed-width verify runs over the GPU's paged sequences. */
    @Test
    public void treeDraftingKeepsTheGreedyTokensOnCuda() throws IOException {
        GpuTestUtil.assumeCudaAvailable();
        treeDraftingKeepsTheGreedyTokensOnGpu(EngineConfig.BackendMode.CUDA);
    }

    /** The same tree round on an OpenCL device: the tree-masked step runs the GLSL twin's flash kernels. */
    @Test
    public void treeDraftingKeepsTheGreedyTokensOnOpenCL() throws IOException {
        GpuTestUtil.assumeOpenCLAvailable();
        treeDraftingKeepsTheGreedyTokensOnGpu(EngineConfig.BackendMode.OPENCL);
    }

    /** The tree round of {@code backend}: lossless greedy tokens, real rounds, drafts and acceptances. */
    private void treeDraftingKeepsTheGreedyTokensOnGpu(EngineConfig.BackendMode backend) throws IOException {
        int[] prompt = prompt();
        GenerationRequest request = greedy(N_NEW);
        GenerationResult expected = plain(prompt, request, backend);
        EngineConfig cfg = config(0.1, backend);
        try (EngineModel model = LLMModelFactory.open(modelFile.toString(), cfg);
             SpeculativeSession session = LLMModelFactory.speculate(model, cfg)) {
            GenerationResult actual = session.generate(prompt, request, (TokenCallback) null);
            assertArrayEquals("tree drafting on the GPU keeps the target's greedy tokens", expected.tokens(),
                    actual.tokens());
            assertEquals("stop reason", expected.stopReason(), actual.stopReason());
            assertTrue("rounds ran: " + session.lastStats(), session.lastStats().rounds() > 0);
            assertTrue("tokens were drafted: " + session.lastStats(), session.lastStats().drafted() > 0);
            assertTrue("the target confirmed the chain of every round: " + session.lastStats(),
                    session.lastStats().accepted() > 0);
        }
    }

    /**
     * The verify runs as one sequence's fixed-width forward, so the sequences' KV spaces need not share pages: a
     * non-unified paged memory keeps the target-only greedy tokens exactly.
     */
    @Test
    public void aNonUnifiedPagedMemoryKeepsThePlainOutput() throws IOException {
        int[] prompt = prompt();
        GenerationRequest request = greedy(N_NEW);
        GenerationResult expected = plain(prompt, request);
        MemoryConfig memory = MemoryConfig.builder().nCtx(N_CTX).ubatch(8)
                .typeK(KvDType.F32).typeV(KvDType.F32).parallel(PARALLEL).kvUnified(false).build();
        EngineConfig cfg = EngineConfig.builder().backend(EngineConfig.BackendMode.CPU).memory(memory)
                .draftModelPath(modelFile.toString()).draftMax(DRAFT_MAX)
                .speculativeOptions(SpeculativeOptions.builder().pSplit(0.1).build()).build();
        try (EngineModel model = LLMModelFactory.open(modelFile.toString(), cfg);
             SpeculativeSession session = LLMModelFactory.speculate(model, cfg)) {
            GenerationResult actual = session.generate(prompt, request, (TokenCallback) null);
            assertArrayEquals("tree drafting on a non-unified paged memory is lossless", expected.tokens(),
                    actual.tokens());
            assertTrue("rounds ran: " + session.lastStats(), session.lastStats().rounds() > 0);
        }
    }

    @Test
    public void aNonMatchVerifyModeRefusesTreeDrafting() throws IOException {
        EngineConfig cfg = config(0.1).toBuilder()
                .speculativeOptions(SpeculativeOptions.builder().pSplit(0.1)
                        .verifyMode(com.eonmux.neuralfree.gguf.engine.draft.VerifyMode.REJECTION).build())
                .build();
        try (EngineModel model = LLMModelFactory.open(modelFile.toString(), cfg);
             SpeculativeSession session = LLMModelFactory.speculate(model, cfg)) {
            IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                    () -> session.generate(prompt(), greedy(N_NEW), (TokenCallback) null));
            assertTrue(e.getMessage(), e.getMessage().contains("neuralfree.draft.verify"));
        }
    }
}

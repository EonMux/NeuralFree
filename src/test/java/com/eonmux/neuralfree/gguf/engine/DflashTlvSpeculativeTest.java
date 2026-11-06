package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.gguf.GGMLType;
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
import java.util.Arrays;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Two-level verification with a dflash drafter ({@code neuralfree.draft.tlv}): the F32 target's quantized view
 * verifies every proposal greedily first — committing nothing — the exact target verifies the composed proposal
 * in one pass and commits, so every token equals the target-only greedy loop's whatever the Q copy proposes.
 * Runs on the CPU, on a contiguous memory and on every available GPU backend; the configurations the mechanism
 * cannot serve are refused by name.
 */
public class DflashTlvSpeculativeTest {

    private static final int N_CTX = 256;
    private static final int N_NEW = 16;
    private static final int DRAFT_MAX = 4;
    private static final int PARALLEL = 3;
    private static final int BLOCK = 8;
    private static final int[] TARGET_LAYERS = {0, 2};

    private static Path targetFile;
    private static Path quantizedTargetFile;
    private static Path dflashFile;

    @BeforeClass
    public static void writeModels() throws IOException {
        SyntheticModels.Config c = SyntheticModels.Config.llama();
        targetFile = SyntheticModels.llama(c.toBuilder().nLayer(4).build()).writeTempFile();
        // an 8-bit target streams no more than the bits=8 copy would: refused by name
        quantizedTargetFile = SyntheticModels.llama(c.toBuilder().nLayer(4).weightType(GGMLType.Q8_0).build())
                .writeTempFile();
        dflashFile = SyntheticModels.dflashDrafter(c.toBuilder().nLayer(2).build(), TARGET_LAYERS, BLOCK)
                .writeTempFile();
    }

    @AfterClass
    public static void deleteModels() throws IOException {
        for (Path p : Arrays.asList(targetFile, quantizedTargetFile, dflashFile)) {
            Files.deleteIfExists(p);
        }
    }

    private static GenerationRequest greedy(int maxNewTokens) {
        return GenerationRequest.builder().maxNewTokens(maxNewTokens).sampler(SamplerConfig.greedyDecoding())
                .build();
    }

    /** The TLV configuration over the paged unified memory, with the tree verification on. */
    private static EngineConfig tlvConfig(Path draft, EngineConfig.BackendMode backend, int bits) {
        MemoryConfig memory = MemoryConfig.builder().nCtx(N_CTX).ubatch(8)
                .typeK(KvDType.F32).typeV(KvDType.F32).parallel(PARALLEL).kvUnified(true).build();
        return EngineConfig.builder().backend(backend).memory(memory)
                .draftModelPath(draft.toString()).draftMax(DRAFT_MAX)
                .speculativeOptions(SpeculativeOptions.builder().treeVerify(true).twoLevelVerify(true)
                        .tlvBits(bits).build()).build();
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

    private static void assertTlvGeneration(Path draft, EngineConfig.BackendMode backend) throws IOException {
        int[] prompt = prompt();
        GenerationRequest request = greedy(N_NEW);
        GenerationResult expected = plain(prompt, request, backend);
        EngineConfig cfg = tlvConfig(draft, backend, 8);
        try (EngineModel model = LLMModelFactory.open(targetFile.toString(), cfg);
             SpeculativeSession session = LLMModelFactory.speculate(model, cfg)) {
            GenerationResult actual = session.generate(prompt, request, (TokenCallback) null);
            assertArrayEquals("two-level verification keeps the target-only greedy tokens", expected.tokens(),
                    actual.tokens());
            assertEquals("stop reason", expected.stopReason(), actual.stopReason());
            SpeculativeDecoder.Stats s = session.lastStats();
            assertTrue("rounds ran: " + s, s.rounds() > 0);
            assertTrue("the drafter proposed trees: " + s, s.drafted() > 0);
        }
    }

    @Test
    public void tlvKeepsTheGreedyTokensOnTheCpu() throws IOException {
        assertTlvGeneration(dflashFile, EngineConfig.BackendMode.CPU);
    }

    @Test
    public void tlvKeepsTheGreedyTokensOnANonUnifiedPagedMemory() throws IOException {
        // the dflash drafter needs paged sequences to branch; a non-unified paged memory (each sequence holds a
        // fixed slice) runs the same round: the verify is one target sequence of the fixed-width rows
        int[] prompt = prompt();
        GenerationRequest request = greedy(N_NEW);
        GenerationResult expected = plain(prompt, request, EngineConfig.BackendMode.CPU);
        MemoryConfig memory = MemoryConfig.builder().nCtx(N_CTX).ubatch(8)
                .typeK(KvDType.F32).typeV(KvDType.F32).parallel(PARALLEL).kvUnified(false).build();
        EngineConfig cfg = EngineConfig.builder().backend(EngineConfig.BackendMode.CPU).memory(memory)
                .draftModelPath(dflashFile.toString()).draftMax(DRAFT_MAX)
                .speculativeOptions(SpeculativeOptions.builder().treeVerify(true).twoLevelVerify(true)
                        .tlvBits(8).build()).build();
        try (EngineModel model = LLMModelFactory.open(targetFile.toString(), cfg);
             SpeculativeSession session = LLMModelFactory.speculate(model, cfg)) {
            GenerationResult actual = session.generate(prompt, request, (TokenCallback) null);
            assertArrayEquals("two-level verification over a non-unified paged memory keeps the greedy tokens",
                    expected.tokens(), actual.tokens());
            assertTrue(session.lastStats().rounds() > 0);
        }
    }

    @Test
    public void tlvKeepsTheGreedyTokensOnCuda() throws IOException {
        GpuTestUtil.assumeCudaAvailable();
        assertTlvGeneration(dflashFile, EngineConfig.BackendMode.CUDA);
    }

    @Test
    public void tlvKeepsTheGreedyTokensOnOpenCL() throws IOException {
        GpuTestUtil.assumeOpenCLAvailable();
        assertTlvGeneration(dflashFile, EngineConfig.BackendMode.OPENCL);
    }

    @Test
    public void aSampledTlvRequestIsRefusedByName() throws IOException {
        EngineConfig cfg = tlvConfig(dflashFile, EngineConfig.BackendMode.CPU, 8);
        try (EngineModel model = LLMModelFactory.open(targetFile.toString(), cfg);
             SpeculativeSession session = LLMModelFactory.speculate(model, cfg)) {
            GenerationRequest sampled = GenerationRequest.builder().maxNewTokens(4)
                    .sampler(SamplerConfig.builder().temperature(0.8f).seed(42L).build()).build();
            IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                    () -> session.generate(prompt(), sampled, (TokenCallback) null));
            assertTrue(e.getMessage(), e.getMessage().contains("neuralfree.draft.tlv"));
        }
    }

    @Test
    public void aSixBitCopyOfANarrowModelIsRefusedByName() throws IOException {
        // the synthetic model's 64-wide matrices take no whole Q6_K blocks: no matrix is eligible, the attach
        // refuses with the property named
        EngineConfig cfg = tlvConfig(dflashFile, EngineConfig.BackendMode.CPU, 6);
        try (EngineModel model = LLMModelFactory.open(targetFile.toString(), cfg)) {
            UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class,
                    () -> LLMModelFactory.speculate(model, cfg));
            assertTrue(e.getMessage(), e.getMessage().contains("neuralfree.draft.tlv"));
        }
    }

    @Test
    public void aQuantizedTargetIsRefusedByName() throws IOException {
        // an already-8-bit target streams no fewer bytes than the bits=8 copy would: the mechanism has nothing
        // to offer, and says so
        EngineConfig cfg = tlvConfig(dflashFile, EngineConfig.BackendMode.CPU, 8);
        try (EngineModel model = LLMModelFactory.open(quantizedTargetFile.toString(), cfg)) {
            UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class,
                    () -> LLMModelFactory.speculate(model, cfg));
            assertTrue(e.getMessage(), e.getMessage().contains("neuralfree.draft.tlv"));
        }
    }

    @Test
    public void aNonTreeDrafterIsRefusedByName() throws IOException {
        // the same drafter without the tree option verifies depth by depth, which the Q round cannot mirror
        EngineConfig cfg = tlvConfig(dflashFile, EngineConfig.BackendMode.CPU, 8).toBuilder()
                .speculativeOptions(SpeculativeOptions.builder().treeVerify(false).twoLevelVerify(true)
                        .tlvBits(8).build())
                .build();
        try (EngineModel model = LLMModelFactory.open(targetFile.toString(), cfg)) {
            UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class,
                    () -> LLMModelFactory.speculate(model, cfg));
            assertTrue(e.getMessage(), e.getMessage().contains("neuralfree.draft.tlv"));
        }
    }
}

package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.ModelSources;
import com.eonmux.neuralfree.gguf.TokenCallback;
import com.eonmux.neuralfree.gguf.memory.ContextFitter;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotSame;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Two-level verification on a W12 target's nested 8-bit Q view ({@code neuralfree.weights.w12} with {@code
 * neuralfree.draft.tlv}): the Q round opens the target's own backend under its Q-view scope and reads the
 * nested 8-bit view of the same W12 code bytes, so no copy is built, no second backend exists and the context
 * fit counts no Q-copy bytes — and every committed token still equals the target-only greedy loop's. Runs on
 * the CUDA and OpenCL backends; the 6-bit width the copy serves is refused by name on the W12 path. The
 * synthetic model's weights are written as BF16, which W12 stores without loss.
 */
@RunWith(Parameterized.class)
public class W12TwoLevelVerifyTest {

    private static final int N_CTX = 256;
    private static final int N_NEW = 16;
    private static final int DRAFT_MAX = 4;
    private static final int PARALLEL = 3;
    private static final int BLOCK = 8;
    private static final int[] TARGET_LAYERS = {0, 2};
    private static final int[] PROMPT = {4, 5, 6, 4, 5, 7, 4, 5, 8, 4, 5};

    @Parameterized.Parameters(name = "{0}")
    public static List<Object[]> backends() {
        return Arrays.asList(new Object[]{EngineConfig.BackendMode.OPENCL},
                new Object[]{EngineConfig.BackendMode.CUDA});
    }

    private static Path targetFile;
    private static Path dflashFile;

    private final EngineConfig.BackendMode backend;

    public W12TwoLevelVerifyTest(EngineConfig.BackendMode backend) {
        this.backend = backend;
    }

    @BeforeClass
    public static void writeModels() throws IOException {
        SyntheticModels.Config c = SyntheticModels.Config.llama();
        targetFile = SyntheticModels.llama(c.toBuilder().nLayer(4).weightType(GGMLType.BF16).build())
                .writeTempFile();
        dflashFile = SyntheticModels.dflashDrafter(c.toBuilder().nLayer(2).weightType(GGMLType.BF16).build(),
                TARGET_LAYERS, BLOCK).writeTempFile();
    }

    @AfterClass
    public static void deleteModels() throws IOException {
        Files.deleteIfExists(targetFile);
        Files.deleteIfExists(dflashFile);
    }

    @Before
    public void assumeDevice() {
        if (backend == EngineConfig.BackendMode.OPENCL) {
            GpuTestUtil.assumeOpenCLAvailable();
        } else {
            GpuTestUtil.assumeCudaAvailable();
        }
    }

    private static GenerationRequest greedy(int maxNewTokens) {
        return GenerationRequest.builder().maxNewTokens(maxNewTokens).sampler(SamplerConfig.greedyDecoding())
                .build();
    }

    /** The TLV configuration over the paged unified memory, with the W12 streaming on. */
    private EngineConfig tlvConfig(Path draft) {
        MemoryConfig memory = MemoryConfig.builder().nCtx(N_CTX).ubatch(8)
                .typeK(KvDType.F32).typeV(KvDType.F32).parallel(PARALLEL).kvUnified(true).build();
        return EngineConfig.builder().backend(backend).memory(memory)
                .draftModelPath(draft.toString()).draftMax(DRAFT_MAX)
                .speculativeOptions(SpeculativeOptions.builder().treeVerify(true).twoLevelVerify(true)
                        .tlvBits(8).w12(true).build()).build();
    }

    /** The target-only loop over the same model and backend. */
    private GenerationResult plain(int[] prompt, GenerationRequest request) throws IOException {
        MemoryConfig memory = MemoryConfig.builder().nCtx(N_CTX).ubatch(8)
                .typeK(KvDType.F32).typeV(KvDType.F32).build();
        EngineConfig cfg = EngineConfig.builder().backend(backend).memory(memory).build();
        try (EngineModel model = LLMModelFactory.open(targetFile.toString(), cfg)) {
            return GenerationLoop.forModel(model, cfg).generate(prompt, request, (TokenCallback) null);
        }
    }

    @Test
    public void theNestedQViewKeepsTheGreedyTokens() throws IOException {
        GenerationRequest request = greedy(N_NEW);
        GenerationResult expected = plain(PROMPT, request);
        EngineConfig cfg = tlvConfig(dflashFile);
        try (EngineModel model = LLMModelFactory.open(targetFile.toString(), cfg);
             SpeculativeSession session = LLMModelFactory.speculate(model, cfg)) {
            GenerationResult actual = session.generate(PROMPT, request, (TokenCallback) null);
            assertArrayEquals("two-level verification on the nested W12 Q view keeps the target-only greedy"
                    + " tokens", expected.tokens(), actual.tokens());
            assertEquals("stop reason", expected.stopReason(), actual.stopReason());
            SpeculativeDecoder.Stats s = session.lastStats();
            assertTrue("rounds ran: " + s, s.rounds() > 0);
            assertTrue("the drafter proposed trees: " + s, s.drafted() > 0);
            assertTrue("the Q round accepted and the exact verify confirmed: " + s, s.accepted() > 0);
        }
    }

    @Test
    public void theNestedQViewRunsTheTargetBackendAndAddsNoBytes() throws IOException {
        EngineConfig cfg = tlvConfig(dflashFile);
        try (EngineModel model = LLMModelFactory.open(targetFile.toString(), cfg);
             QuantizedView view = model.openQuantizedView(8)) {
            assertEquals("the nested W12 Q view adds no weight bytes", 0L, view.weightBytes());
            assertEquals("the nested view is fixed 8-bit", 8, view.bits());
            assertSame("the nested view runs the target's own backend, not a second one",
                    model.forwardBackend(), view.backend());
        }
    }

    @Test
    public void theContextFitCountsNoQCopyBytesForAW12Target() throws IOException {
        EngineConfig cfg = tlvConfig(dflashFile);
        try (ModelSource source = ModelSources.open(targetFile.toString())) {
            assertSame("the W12 target's nested Q view adds no copy bytes to the context fit",
                    ContextFitter.Companion.NONE, LLMModelFactory.twoLevelCompanion(cfg, source, true));
            assertNotSame("the same target without W12 still reserves the copy",
                    ContextFitter.Companion.NONE, LLMModelFactory.twoLevelCompanion(cfg, source, false));
        }
    }

    @Test
    public void aSixBitViewOfAW12TargetIsRefusedByName() throws IOException {
        EngineConfig cfg = tlvConfig(dflashFile);
        try (EngineModel model = LLMModelFactory.open(targetFile.toString(), cfg)) {
            UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class,
                    () -> model.openQuantizedView(6));
            assertTrue(e.getMessage(), e.getMessage().contains("neuralfree.draft.tlv.bits"));
        }
    }
}

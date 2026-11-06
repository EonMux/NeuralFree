package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.LLMModel;
import com.eonmux.neuralfree.gguf.engine.LLMModelFactory;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.engine.SpeculativeDecoder;
import com.eonmux.neuralfree.gguf.engine.SpeculativeOptions;
import com.eonmux.neuralfree.gguf.engine.SpeculativeSession;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertTrue;

/**
 * The lossless W12 streaming of the target weights ({@code neuralfree.weights.w12}) against the plain greedy loop:
 * every committed token of the W12 target — alone, with the output head, under the dflash drafter and over a prefill
 * wider than the W12 matvec budget — equals the target-only greedy tokens, on the CPU and on every available GPU
 * backend. The synthetic model's weights are written as BF16, which W12 stores without loss.
 */
@RunWith(Parameterized.class)
public class W12SpeculativeParityTest {

    private static final int[] PROMPT = {1, 17, 42, 5, 99, 26};
    private static final int GREEDY_TOKENS = 12;
    private static final int N_LAYER = 4;
    private static final int N_VOCAB = 320;
    private static final int LONG_PROMPT_TOKENS = 96;

    @Parameterized.Parameters(name = "{0}")
    public static List<Object[]> backends() {
        return Arrays.asList(new Object[]{"cpu"}, new Object[]{"opencl"}, new Object[]{"cuda"},
                new Object[]{"vulkan"});
    }

    private final String api;
    private final List<Path> files = new ArrayList<>();
    private Path target;
    private Path draft;

    public W12SpeculativeParityTest(String api) {
        this.api = api;
    }

    @Before
    public void writeModels() throws IOException {
        SyntheticModels.Config c = SyntheticModels.Config.llama().toBuilder().nLayer(N_LAYER)
                .weightType(GGMLType.BF16).build();
        target = written(SyntheticModels.llama(c));
        draft = written(SyntheticModels.dflashDrafter(c.toBuilder().nLayer(2).build(), new int[] {0, 2}, 8));
    }

    @After
    public void cleanUp() throws IOException {
        for (Path p : files) {
            Files.deleteIfExists(p);
        }
    }

    private Path written(GgufTestWriter w) throws IOException {
        Path p = w.writeTempFile();
        files.add(p);
        return p;
    }

    // ------------------------------------------------------------------ configs and drivers

    private void assumeBackend() {
        if (api.equals("opencl")) {
            GpuTestUtil.assumeOpenCLAvailable();
        } else if (api.equals("cuda")) {
            GpuTestUtil.assumeCudaAvailable();
        } else if (api.equals("vulkan")) {
            GpuTestUtil.assumeVulkanAvailable();
        }
    }

    private EngineConfig.Builder base() {
        MemoryConfig memory = MemoryConfig.defaults().toBuilder().nCtx(128).ubatch(128).build();
        EngineConfig.BackendMode mode = api.equals("opencl") ? EngineConfig.BackendMode.OPENCL
                : api.equals("cuda") ? EngineConfig.BackendMode.CUDA : EngineConfig.BackendMode.VULKAN;
        if (api.equals("cpu")) {
            return EngineConfig.builder().backend(EngineConfig.BackendMode.CPU).memory(memory);
        }
        return EngineConfig.builder().backend(mode).addGpuDevice("0").memory(memory);
    }

    private static SpeculativeOptions w12(boolean head) {
        return SpeculativeOptions.defaults().toBuilder().w12(true).w12Head(head).build();
    }

    private static int[] greedy(LLMModel m, EngineConfig cfg, int[] prompt) throws IOException {
        GenerationRequest request = GenerationRequest.builder().maxNewTokens(GREEDY_TOKENS)
                .sampler(cfg.legacySampler(0f, 1f, 1f, 0, 0f, 0)).build();
        return m.generate(prompt, request, null).tokens();
    }

    /** The target-only greedy tokens on this backend (the W12 flag rides in {@code cfg}). */
    private int[] baseline(EngineConfig cfg, int[] prompt) throws IOException {
        // GGUFModelGPU's load throws when the W12 flag has no target on this model; no CPU fallback
        if (api.equals("cpu")) {
            try (GGUFModel m = new GGUFModel(target.toString(), cfg)) {
                return greedy(m, cfg, prompt);
            }
        }
        try (GGUFModelGPU m = new GGUFModelGPU(target.toString(), null, cfg)) {
            int[] tokens = greedy(m, cfg, prompt);
            m.forwardBackend().reset();
            return tokens;
        }
    }

    // ------------------------------------------------------------------ the parity

    @Test
    public void theW12TargetKeepsTheGreedyTokens() throws IOException {
        assumeBackend();
        int[] reference = baseline(base().build(), PROMPT);
        assertArrayEquals(reference, baseline(base().speculativeOptions(w12(false)).build(), PROMPT));
    }

    @Test
    public void theW12TargetWithTheHeadKeepsTheGreedyTokens() throws IOException {
        assumeBackend();
        int[] reference = baseline(base().build(), PROMPT);
        assertArrayEquals(reference, baseline(base().speculativeOptions(w12(true)).build(), PROMPT));
    }

    @Test
    public void theW12TargetUnderTheDrafterKeepsTheCommittedTokens() throws IOException {
        assumeBackend();
        int[] reference = baseline(base().build(), PROMPT);
        EngineConfig cfg = base().speculativeOptions(w12(false)).draftModelPath(draft.toString()).draftMax(4).build();
        try (SpeculativeSession s = LLMModelFactory.openSpeculative(target.toString(), null, cfg)) {
            GenerationRequest request = GenerationRequest.builder().maxNewTokens(GREEDY_TOKENS)
                    .sampler(cfg.legacySampler(0f, 1f, 1f, 0, 0f, 0)).build();
            int[] tokens = s.generate(PROMPT, request, null).tokens();
            assertArrayEquals("speculative greedy over the W12 target", reference, tokens);
            SpeculativeDecoder.Stats stats = s.lastStats();
            assertTrue(String.valueOf(stats), stats.drafted() > 0);
            assertTrue("the drafter proposes tokens the W12 target verifies: " + stats, stats.accepted() > 0);
        }
    }

    @Test
    public void aWidePrefillRunsTheDecodedMirrorAndKeepsTheGreedyTokens() throws IOException {
        assumeBackend();
        int[] prompt = new int[LONG_PROMPT_TOKENS];
        for (int i = 0; i < prompt.length; i++) {
            prompt[i] = (i * 7 + 3) % N_VOCAB;
        }
        int[] reference = baseline(base().build(), prompt);
        // the head too: its prefill row runs over the mirror with the projections
        assertArrayEquals(reference, baseline(base().speculativeOptions(w12(true)).build(), prompt));
    }
}

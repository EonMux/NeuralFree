package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.gguf.GGUFModelGPU;
import com.eonmux.neuralfree.gguf.TokenCallback;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.engine.SpeculativeOptions;
import com.eonmux.neuralfree.gguf.engine.draft.VerifyMode;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.AfterClass;
import org.junit.Assume;
import org.junit.BeforeClass;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * The batch engine's speculative MATCH rounds on a GPU backend read each verify row's sampling candidates through
 * the device top-k readout, prefiltered over the row's own history, and emit the tokens of the same requests
 * decoded without speculation: greedy, and greedy under repetition penalties that move the argmax.
 */
public class GpuBatchConedVerifyTest {

    private static final int N_NEW = 12;

    private static Path model;

    @BeforeClass
    public static void writeModel() throws IOException {
        model = SyntheticModels.llama().writeTempFile();
    }

    @AfterClass
    public static void deleteModel() throws IOException {
        if (model != null) {
            Files.deleteIfExists(model);
        }
    }

    @Test
    public void cudaMatchRoundsEmitThePlainTokens() throws IOException {
        Assume.assumeTrue("CUDA not available", GpuTestUtil.isCudaAvailable());
        assertMatchRoundsEmitThePlainTokens(EngineConfig.BackendMode.CUDA);
    }

    @Test
    public void openclMatchRoundsEmitThePlainTokens() throws IOException {
        Assume.assumeTrue("OpenCL not available", GpuTestUtil.isOpenCLAvailable());
        assertMatchRoundsEmitThePlainTokens(EngineConfig.BackendMode.OPENCL);
    }

    @Test
    public void vulkanMatchRoundsEmitThePlainTokens() throws IOException {
        Assume.assumeTrue("Vulkan not available", GpuTestUtil.isVulkanAvailable());
        assertMatchRoundsEmitThePlainTokens(EngineConfig.BackendMode.VULKAN);
    }

    private static void assertMatchRoundsEmitThePlainTokens(EngineConfig.BackendMode mode) throws IOException {
        for (SamplerConfig sampler : new SamplerConfig[]{SamplerConfig.greedyDecoding(),
                SamplerConfig.builder().greedy(true).repeatPenalty(1.6f).presencePenalty(0.8f).penaltyLastN(64)
                        .build()}) {
            List<GenerationRequest> requests = new ArrayList<>();
            for (int i = 0; i < 3; i++) {
                requests.add(GenerationRequest.builder().maxNewTokens(N_NEW).sampler(sampler).build());
            }
            Run plain = run(mode, requests, false);
            Run speculative = run(mode, requests, true);
            for (int i = 0; i < requests.size(); i++) {
                assertArrayEquals(mode + " request " + i, plain.results.get(i).tokens(),
                        speculative.results.get(i).tokens());
            }
            assertTrue("the sequences speculated: " + speculative.stats, speculative.stats.specRounds() > 0);
            assertTrue("every emitted token came from the device readout: " + speculative.stats,
                    speculative.stats.deviceSampledRows() >= speculative.stats.generatedTokens());
        }
    }

    private static final class Run {
        final List<GenerationResult> results = new ArrayList<>();
        BatchStats stats;
    }

    private static Run run(EngineConfig.BackendMode mode, List<GenerationRequest> requests, boolean speculate)
            throws IOException {
        MemoryConfig mc = MemoryConfig.builder().nCtx(1024).ubatch(16).typeK(KvDType.F32).typeV(KvDType.F32)
                .parallel(3).build();
        EngineConfig cfg = EngineConfig.builder().backend(mode).gpuLayers(EngineConfig.GPU_LAYERS_ALL).memory(mc)
                .build();
        Run out = new Run();
        try (GGUFModelGPU m = new GGUFModelGPU(model.toString(), null, cfg);
             BatchEngine engine = speculate
                     ? BatchEngine.forModel(m, cfg, new RepeatDrafter(),
                             BatchSpec.of(3, 0, SpeculativeOptions.builder().verifyMode(VerifyMode.MATCH).build()))
                     : BatchEngine.forModel(m, cfg)) {
            List<BatchSequence> running = new ArrayList<>();
            for (int i = 0; i < requests.size(); i++) {
                running.add(engine.submit(prompt(i), requests.get(i), (TokenCallback) null));
            }
            engine.drain();
            for (BatchSequence s : running) {
                assertEquals(String.valueOf(s.error()), BatchSequence.State.DONE, s.state());
                out.results.add(s.result());
            }
            out.stats = engine.stats();
        }
        return out;
    }

    private static int[] prompt(int seed) {
        int[] t = new int[10 + seed];
        for (int i = 0; i < t.length; i++) {
            t[i] = 4 + (seed * 7 + i * 13) % 250;
        }
        return t;
    }

    /** Proposes the sequence's last token and the two after it in the vocabulary, every round. */
    private static final class RepeatDrafter implements BatchDrafter {
        @Override
        public int[] draft(int seq, int lastToken, int n) {
            int[] out = new int[Math.min(n, 3)];
            for (int i = 0; i < out.length; i++) {
                out[i] = 4 + (lastToken + i) % 250;
            }
            return out;
        }

        @Override
        public void accept(int seq, int nAccepted) {
        }

        @Override
        public void close() {
        }
    }
}

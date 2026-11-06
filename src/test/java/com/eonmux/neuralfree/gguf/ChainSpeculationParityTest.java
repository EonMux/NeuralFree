package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.LLMModelFactory;
import com.eonmux.neuralfree.gguf.engine.RowTopK;
import com.eonmux.neuralfree.gguf.engine.SpeculativeDecoder;
import com.eonmux.neuralfree.gguf.engine.SpeculativeOptions;
import com.eonmux.neuralfree.gguf.engine.SpeculativeSession;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardBackend;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.ops.flow.DraftModelLoader;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static com.eonmux.neuralfree.gguf.SplitFixtures.GREEDY_TOKENS;
import static com.eonmux.neuralfree.gguf.SplitFixtures.PROMPT;
import static com.eonmux.neuralfree.gguf.SplitFixtures.greedy;
import static com.eonmux.neuralfree.gguf.SplitFixtures.oneDevice;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * A chain round keeps the target-only greedy tokens while the decode replay is on: a self-drafting session on a
 * GPU target (device greedy verification, the drafter's picks taken off its device top-k readback) emits the plain
 * greedy generation's tokens on OpenCL and on CUDA, with the replay disabled and with the speculative round's
 * replay ({@code neuralfree.gpu.chainReplay}) on as well. Skipped when the backend's GPU is unavailable.
 */
public class ChainSpeculationParityTest {

    private Path model;

    @Before
    public void writeModel() throws IOException {
        model = SyntheticModels.llama(SyntheticModels.Config.llama().toBuilder().nLayer(2).build()).writeTempFile();
    }

    @After
    public void deleteModel() throws IOException {
        if (model != null) {
            Files.deleteIfExists(model);
        }
    }

    /** OpenCL: the chain with device and host verification keeps the plain greedy tokens, replay on (the default). */
    @Test
    public void chainKeepsThePlainGreedyTokensOnOpenCl() throws IOException {
        GpuTestUtil.assumeOpenCLAvailable();
        int[] reference;
        try (GGUFModelGPU m = new GGUFModelGPU(model.toString(), null, oneDevice().build())) {
            reference = greedy(m);
        }
        assertArrayEquals(reference, chainGreedy(openClChain(true)));
        assertArrayEquals(reference, chainGreedy(openClChain(false)));
    }

    /** CUDA: the same parity with the device verification, the speculative round's replay on and off. */
    @Test
    public void chainKeepsThePlainGreedyTokensOnCuda() throws IOException {
        GpuTestUtil.assumeCudaAvailable();
        EngineConfig.Builder base = EngineConfig.builder()
                .backend(EngineConfig.BackendMode.CUDA)
                .gpuLayers(EngineConfig.GPU_LAYERS_ALL)
                .memory(MemoryConfig.defaults().toBuilder().nCtx(64).ubatch(4).build());
        int[] reference;
        try (GGUFModelGPU m = new GGUFModelGPU(model.toString(), null, base.build())) {
            reference = greedy(m);
        }
        EngineConfig.Builder chain = base.draftModelPath(model.toString()).draftMax(4)
                .speculativeOptions(SpeculativeOptions.builder().pSplit(0).treeVerify(false)
                        .verifyOnDevice(true).build());
        assertArrayEquals(reference, chainGreedy(chain));
        assertArrayEquals(reference, chainGreedy(chain, true));
    }

    /** OpenCL, the replay off: the chain still keeps the plain greedy tokens. */
    @Test
    public void chainKeepsThePlainGreedyTokensWithTheReplayOff() throws IOException {
        GpuTestUtil.assumeOpenCLAvailable();
        int[] reference;
        try (GGUFModelGPU m = new GGUFModelGPU(model.toString(), null, oneDevice().build())) {
            reference = greedy(m);
        }
        System.setProperty(GpuForwardBackend.DECODE_REPLAY_PROPERTY, "false");
        try {
            assertArrayEquals(reference, chainGreedy(openClChain(true)));
        } finally {
            System.clearProperty(GpuForwardBackend.DECODE_REPLAY_PROPERTY);
        }
    }

    /** OpenCL, the speculative round's replay on: the chain still keeps the plain greedy tokens. */
    @Test
    public void chainKeepsThePlainGreedyTokensWithTheChainReplayOn() throws IOException {
        GpuTestUtil.assumeOpenCLAvailable();
        int[] reference;
        try (GGUFModelGPU m = new GGUFModelGPU(model.toString(), null, oneDevice().build())) {
            reference = greedy(m);
        }
        assertArrayEquals(reference, chainGreedy(openClChain(true), true));
    }

    /**
     * The draft pick off a device top-k readback is the pick off the whole row: the greedy id and the top-k
     * probability agree on ties and when the row's non-finite logits stay out of the candidates.
     */
    @Test
    public void theDeviceTopKPickEqualsTheFullRowPick() {
        float[] row = {0.5f, 3.25f, 3.25f, -1f, 2f, 1f, 0.75f, 1.5f, -2f, 2.75f, Float.NEGATIVE_INFINITY,
                Float.NaN, 0.25f};
        DraftModelLoader.Pick host = DraftModelLoader.pick(row, row.length);
        DraftModelLoader.Pick device = DraftModelLoader.pick(RowTopK.of(new float[][]{row},
                DraftModelLoader.DRAFT_TOP_K), 0);
        assertEquals(host.id, device.id);
        assertEquals(host.p, device.p, 1e-6f);
    }

    /** The chain config of a self-drafting session on the OpenCL device 0. */
    private EngineConfig.Builder openClChain(boolean verifyOnDevice) {
        return oneDevice().draftModelPath(model.toString()).draftMax(4)
                .speculativeOptions(SpeculativeOptions.builder().pSplit(0).treeVerify(false)
                        .verifyOnDevice(verifyOnDevice).build());
    }

    /** The greedy tokens of one chain generation, with its round stats checked. */
    private int[] chainGreedy(EngineConfig.Builder builder) throws IOException {
        return chainGreedy(builder, false);
    }

    /** The greedy tokens of one chain generation, with the speculative round's replay forced to {@code chain}. */
    private int[] chainGreedy(EngineConfig.Builder builder, boolean chain) throws IOException {
        System.setProperty(GpuForwardBackend.CHAIN_REPLAY_PROPERTY, String.valueOf(chain));
        EngineConfig cfg = builder.build();
        try (GGUFModelGPU target = new GGUFModelGPU(model.toString(), null, cfg);
             SpeculativeSession s = LLMModelFactory.speculate(target, cfg)) {
            GenerationRequest request = GenerationRequest.builder().maxNewTokens(GREEDY_TOKENS)
                    .sampler(cfg.legacySampler(0f, 1f, 1f, 0, 0f, 0)).build();
            int[] tokens = s.generate(PROMPT, request, null).tokens();
            SpeculativeDecoder.Stats stats = s.lastStats();
            assertTrue(String.valueOf(stats), stats.rounds() > 0);
            assertTrue("a self-draft is confirmed: " + stats, stats.accepted() > 0);
            target.forwardBackend().reset();
            return tokens;
        } finally {
            System.clearProperty(GpuForwardBackend.CHAIN_REPLAY_PROPERTY);
        }
    }
}

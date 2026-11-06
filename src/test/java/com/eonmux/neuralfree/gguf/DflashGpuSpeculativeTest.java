package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.LLMModelFactory;
import com.eonmux.neuralfree.gguf.engine.SpeculativeDecoder;
import com.eonmux.neuralfree.gguf.engine.SpeculativeSession;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertTrue;

/**
 * Speculative decoding with a dflash drafter on a GPU target: the drafter's backbone runs on the target's device
 * ({@link com.eonmux.neuralfree.gguf.ops.flow.dflash.DflashDevice}), and the session's greedy tokens are the target's
 * plain greedy tokens, on the device and on the host alike.
 */
public class DflashGpuSpeculativeTest {

    private static final int[] PROMPT = {1, 17, 42, 5, 99, 26};
    private static final int GREEDY_TOKENS = 12;

    private final List<Path> files = new ArrayList<>();
    private Path target;
    private Path draft;

    @Before
    public void writeModels() throws IOException {
        SyntheticModels.Config c = SyntheticModels.Config.llama();
        target = written(SyntheticModels.llama(c.toBuilder().nLayer(4).build()));
        draft = written(SyntheticModels.dflashDrafter(c.toBuilder().nLayer(2).build(), new int[] {0, 2}, 8));
    }

    @After
    public void cleanUp() throws IOException {
        for (Path p : files) {
            Files.deleteIfExists(p);
        }
    }

    private Path written(com.eonmux.neuralfree.gguf.testutil.GgufTestWriter w) throws IOException {
        Path p = w.writeTempFile();
        files.add(p);
        return p;
    }

    private static EngineConfig.Builder gpu() {
        return EngineConfig.builder().backend(EngineConfig.BackendMode.OPENCL).addGpuDevice("0")
                .memory(MemoryConfig.defaults().toBuilder().nCtx(128).ubatch(32).build());
    }

    private static EngineConfig.Builder cpu() {
        return EngineConfig.builder().backend(EngineConfig.BackendMode.CPU)
                .memory(MemoryConfig.defaults().toBuilder().nCtx(128).ubatch(32).build());
    }

    /** The greedy tokens of a speculative session over {@code cfg}, with the session's drafting stats. */
    private static int[] speculativeGreedy(Path target, EngineConfig cfg, SpeculativeDecoder.Stats[] stats)
            throws IOException {
        try (SpeculativeSession s = LLMModelFactory.openSpeculative(target.toString(), null, cfg)) {
            GenerationRequest request = GenerationRequest.builder().maxNewTokens(GREEDY_TOKENS)
                    .sampler(cfg.legacySampler(0f, 1f, 1f, 0, 0f, 0)).build();
            int[] tokens = s.generate(PROMPT, request, null).tokens();
            stats[0] = s.lastStats();
            return tokens;
        }
    }

    @Test
    public void aDflashDrafterOnTheDeviceKeepsTheGreedyTokensAndDrafts() throws IOException {
        GpuTestUtil.assumeOpenCLAvailable();
        EngineConfig plain = gpu().build();
        int[] reference;
        try (GGUFModelGPU m = new GGUFModelGPU(target.toString(), null, plain)) {
            GenerationRequest request = GenerationRequest.builder().maxNewTokens(GREEDY_TOKENS)
                    .sampler(plain.legacySampler(0f, 1f, 1f, 0, 0f, 0)).build();
            reference = m.generate(PROMPT, request, null).tokens();
            m.forwardBackend().reset();
        }
        EngineConfig cfg = gpu().draftModelPath(draft.toString()).draftMax(4).build();
        SpeculativeDecoder.Stats[] stats = new SpeculativeDecoder.Stats[1];
        int[] tokens = speculativeGreedy(target, cfg, stats);
        assertArrayEquals("speculative greedy over the device drafter", reference, tokens);
        assertTrue(String.valueOf(stats[0]), stats[0].drafted() > 0);
        assertTrue("the device drafter proposes tokens the target verifies: " + stats[0], stats[0].accepted() > 0);
    }

    @Test
    public void theDeviceAndHostDraftersAgreeOnTheGreedyTokens() throws IOException {
        GpuTestUtil.assumeOpenCLAvailable();
        SpeculativeDecoder.Stats[] deviceStats = new SpeculativeDecoder.Stats[1];
        int[] fromDevice = speculativeGreedy(target, gpu().draftModelPath(draft.toString()).draftMax(4).build(),
                deviceStats);
        assertTrue(deviceStats[0].drafted() > 0);
        SpeculativeDecoder.Stats[] hostStats = new SpeculativeDecoder.Stats[1];
        int[] fromHost = speculativeGreedy(target, cpu().draftModelPath(draft.toString()).draftMax(4).build(),
                hostStats);
        assertTrue(hostStats[0].drafted() > 0);
        assertArrayEquals("the device and host drafters produce the same greedy tokens", fromHost, fromDevice);
    }
}

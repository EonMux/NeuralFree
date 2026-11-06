package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.LLMModelFactory;
import com.eonmux.neuralfree.gguf.engine.SpeculativeDecoder;
import com.eonmux.neuralfree.gguf.engine.SpeculativeSession;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static com.eonmux.neuralfree.gguf.SplitFixtures.GREEDY_TOKENS;
import static com.eonmux.neuralfree.gguf.SplitFixtures.PROMPT;
import static com.eonmux.neuralfree.gguf.SplitFixtures.oneDevice;
import static com.eonmux.neuralfree.gguf.SplitFixtures.split;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertNotSame;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;

/**
 * Speculative decoding over a two-device layer split: a separate draft model ({@code neuralfree.draft}) runs on the
 * main device, the model's own NextN / MTP blocks on the device of the last trunk layer; either way the greedy
 * tokens are those of the one-device load without a drafter, and the drafter proposed tokens.
 */
public class GpuSplitSpeculativeTest {

    private final List<Path> files = new ArrayList<>();

    @Before
    public void requireTwoDevices() {
        SplitFixtures.assumeTwoDevices();
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

    /** A glm4moe model of four trunk layers and one NextN block. */
    private Path mtpModel() throws IOException {
        return written(SyntheticModels.glm4moe(SyntheticModels.Config.qwen3moe().toBuilder().nLayer(4).build(), 1));
    }

    /** The greedy tokens of the one-device load without a drafter. */
    private static int[] plainGreedy(Path model) throws IOException {
        try (GGUFModelGPU m = new GGUFModelGPU(model.toString(), null, oneDevice().build())) {
            return SplitFixtures.greedy(m);
        }
    }

    /** The greedy tokens of a speculative session over {@code target}, with the session's drafting stats. */
    private static int[] speculativeGreedy(GGUFModelGPU target, EngineConfig cfg, SpeculativeDecoder.Stats[] stats)
            throws IOException {
        try (SpeculativeSession s = LLMModelFactory.speculate(target, cfg)) {
            GenerationRequest request = GenerationRequest.builder().maxNewTokens(GREEDY_TOKENS)
                    .sampler(cfg.legacySampler(0f, 1f, 1f, 0, 0f, 0)).build();
            int[] tokens = s.generate(PROMPT, request, null).tokens();
            stats[0] = s.lastStats();
            return tokens;
        }
    }

    @Test
    public void aDraftModelOnTheMainDeviceKeepsTheGreedyTokens() throws IOException {
        Path model = written(SyntheticModels.llama(SyntheticModels.Config.llama().toBuilder().nLayer(4).build()));
        int[] reference = plainGreedy(model);
        EngineConfig cfg = split().draftModelPath(model.toString()).draftMax(3).build();
        try (GGUFModelGPU target = new GGUFModelGPU(model.toString(), null, cfg)) {
            SpeculativeDecoder.Stats[] stats = new SpeculativeDecoder.Stats[1];
            int[] tokens = speculativeGreedy(target, cfg, stats);
            assertArrayEquals("speculative greedy over the split", reference, tokens);
            assertTrue(String.valueOf(stats[0]), stats[0].drafted() > 0);
            assertTrue("the drafter drafts its own tokens back: " + stats[0], stats[0].accepted() > 0);
        }
    }

    @Test
    public void theModelsOwnMtpBlocksRunOnTheDeviceOfTheLastLayer() throws IOException {
        Path model = mtpModel();
        int[] reference = plainGreedy(model);
        EngineConfig cfg = split().mtp(true).draftMax(1).build();
        try (GGUFModelGPU target = new GGUFModelGPU(model.toString(), null, cfg)) {
            assertNotSame("the NextN blocks sit on the device of the last layer, not the main one",
                    target.forwardContext().backend(), target.nextnContext().backend());
            SpeculativeDecoder.Stats[] stats = new SpeculativeDecoder.Stats[1];
            int[] tokens = speculativeGreedy(target, cfg, stats);
            assertArrayEquals("MTP greedy over the split", reference, tokens);
            assertTrue(String.valueOf(stats[0]), stats[0].drafted() > 0);
        }
    }

    @Test
    public void mtpBlocksLeftOffAtLoadArePlacedWhenTheDrafterAttaches() throws IOException {
        Path model = mtpModel();
        int[] reference = plainGreedy(model);
        EngineConfig loadCfg = split().build();
        EngineConfig draftCfg = split().mtp(true).draftMax(1).build();
        try (GGUFModelGPU target = new GGUFModelGPU(model.toString(), null, loadCfg)) {
            SpeculativeDecoder.Stats[] stats = new SpeculativeDecoder.Stats[1];
            int[] tokens = speculativeGreedy(target, draftCfg, stats);
            assertArrayEquals("MTP greedy after a late NextN placement", reference, tokens);
            assertTrue(String.valueOf(stats[0]), stats[0].drafted() > 0);
        }
    }

    @Test
    public void withTheMainDeviceLastTheMtpBlocksShareItsContext() throws IOException {
        Path model = mtpModel();
        int[] reference = plainGreedy(model);
        EngineConfig cfg = split().mainGpu(1).mtp(true).draftMax(1).build();
        try (GGUFModelGPU target = new GGUFModelGPU(model.toString(), null, cfg)) {
            assertSame(target.forwardContext(), target.nextnContext());
            SpeculativeDecoder.Stats[] stats = new SpeculativeDecoder.Stats[1];
            assertArrayEquals(reference, speculativeGreedy(target, cfg, stats));
        }
    }
}

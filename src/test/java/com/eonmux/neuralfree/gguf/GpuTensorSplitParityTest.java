package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static com.eonmux.neuralfree.gguf.SplitFixtures.assertParity;
import static com.eonmux.neuralfree.gguf.SplitFixtures.run;

/**
 * Tensor-parallel parity on the OpenCL devices of the machine: a synthetic llama loaded with
 * {@code neuralfree.split.mode=tensor} (and no device list, so every enumerated device participates) produces the
 * same logits and greedy tokens as the one-device load of the same settings. On a single-GPU machine the mode keeps
 * the model on one device, as it does when the backend serves only one rank; on a multi-GPU machine every rank runs
 * its attention heads and dense feed-forward shard with its own KV cache, and the all-reduce sums the partials.
 */
public class GpuTensorSplitParityTest {

    private Path model;

    @Before
    public void writeModel() throws IOException {
        GpuTestUtil.assumeOpenCLAvailable();
        model = SyntheticModels.llama(SyntheticModels.Config.llama().toBuilder().nLayer(2).build()).writeTempFile();
    }

    @After
    public void deleteModel() throws IOException {
        if (model != null) {
            Files.deleteIfExists(model);
        }
    }

    /** The load settings of both runs, 64 positions in ubatches of four. */
    private static EngineConfig.Builder base() {
        return EngineConfig.builder()
                .backend(EngineConfig.BackendMode.OPENCL)
                .memory(MemoryConfig.defaults().toBuilder().nCtx(64).ubatch(4).build());
    }

    @Test
    public void theTensorSplitMatchesTheOneDeviceLoad() throws IOException {
        SplitFixtures.Run reference = run(model, base().build());
        SplitFixtures.Run split = run(model, base().splitMode(EngineConfig.SplitMode.TENSOR).build());
        assertParity("split-mode tensor", reference, split);
    }
}

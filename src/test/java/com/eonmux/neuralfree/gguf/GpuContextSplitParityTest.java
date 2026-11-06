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
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Context-parallel parity on the OpenCL devices of the machine: a synthetic llama loaded with
 * {@code neuralfree.split.mode=context} produces the same logits and greedy tokens as the one-device load of the
 * same settings. On a single-GPU machine the mode keeps the model on one device, as it does when the backend serves
 * only one rank; on a multi-GPU machine the ranks divide the context positions, every rank attends over its own
 * block of the cache and the log-sum-exp merge reproduces the one-device attention.
 */
public class GpuContextSplitParityTest {

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
    public void theContextSplitMatchesTheOneDeviceLoad() throws IOException {
        SplitFixtures.Run reference = run(model, base().build());
        SplitFixtures.Run split = run(model, base().splitMode(EngineConfig.SplitMode.CONTEXT).build());
        assertParity("split-mode context", reference, split);
    }

    @Test
    public void thePrefillQuerySplitMatchesTheOneDeviceLoad() throws IOException {
        SplitFixtures.assumeTwoDevices();
        SplitFixtures.Run reference = run(model, base().build());
        SplitFixtures.Run split = run(model, SplitFixtures.oneDevice("0").addGpuDevice("1")
                .splitMode(EngineConfig.SplitMode.CONTEXT)
                .prefillContextParallelSize(2)
                .build());
        assertParity("split-mode context with the prefill query split", reference, split);
    }

    @Test
    public void aPrefillQuerySplitWithoutTheRanksIsRefusedByName() throws IOException {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> {
                    try (GGUFModelGPU ignored = new GGUFModelGPU(model.toString(), null,
                            base().splitMode(EngineConfig.SplitMode.CONTEXT)
                                    .addGpuDevice("0").prefillContextParallelSize(2).build())) {
                        throw new AssertionError("the load should have refused the prefill query split");
                    }
                });
        assertTrue(e.getMessage(), e.getMessage().contains(EngineConfig.PROP_PREFILL_CONTEXT_PARALLEL));
    }
}

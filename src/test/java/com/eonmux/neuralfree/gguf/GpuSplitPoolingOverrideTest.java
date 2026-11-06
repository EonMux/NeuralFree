package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.gguf.arch.PoolingKind;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static com.eonmux.neuralfree.gguf.SplitFixtures.PROMPT;
import static com.eonmux.neuralfree.gguf.SplitFixtures.TOL;
import static com.eonmux.neuralfree.gguf.SplitFixtures.oneDevice;
import static com.eonmux.neuralfree.gguf.SplitFixtures.split;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * The pooling type override ({@link EngineConfig#pooling()}) over a two-device layer split on the OpenCL devices of
 * the machine: when the split fits a context below the training context, it rebuilds the model's specification at
 * that context, and the rebuilt specification keeps the override, so the split pools as the one-device load of the
 * same context does.
 */
public class GpuSplitPoolingOverrideTest {

    private static final int TRAIN_CONTEXT = 8192;
    /** A per-device cap under which every layer on the GPUs leaves room for less than the training context. */
    private static final long CAP_MB = 12;

    private Path model;

    @Before
    public void setUp() throws IOException {
        SplitFixtures.assumeTwoDevices();
        model = SyntheticModels.llama(SyntheticModels.Config.llama().toBuilder().nLayer(8).nEmbd(256).nHead(4)
                .nHeadKv(2).headDim(64).nRot(64).nFf(1024).nCtxTrain(TRAIN_CONTEXT).build()).writeTempFile();
    }

    @After
    public void cleanUp() throws IOException {
        if (model != null) {
            Files.deleteIfExists(model);
        }
    }

    /** The automatic context under the cap, every layer on the GPUs: the split fits a shorter context. */
    private static EngineConfig.Builder cappedSplit() {
        return split().memory(MemoryConfig.defaults().toBuilder().nCtx(0).ubatch(4).build())
                .gpuVramLimitMb(CAP_MB).gpuLayers(EngineConfig.GPU_LAYERS_ALL);
    }

    @Test
    public void theSpecTheSplitRebuildsAtItsFittedContextKeepsTheOverride() throws IOException {
        try (GGUFModelGPU plain = new GGUFModelGPU(model.toString(), null, cappedSplit().build())) {
            assertEquals("the GGUF's pooling without an override", PoolingKind.NONE, plain.spec().head().pooling());
        }
        int nCtx;
        float[] splitRow;
        try (GGUFModelGPU m = new GGUFModelGPU(model.toString(), null,
                cappedSplit().pooling(PoolingKind.MEAN).build())) {
            assertEquals(2, m.deviceCount());
            nCtx = m.memoryPlan().nCtx();
            assertTrue("the split fits a context below the training context, so it rebuilds its spec: " + nCtx,
                    nCtx < TRAIN_CONTEXT);
            assertEquals(PoolingKind.MEAN, m.spec().head().pooling());
            assertEquals(LLMModel.PoolingType.MEAN, m.poolingType());
            splitRow = m.encode(PROMPT);
        }
        EngineConfig reference = oneDevice().pooling(PoolingKind.MEAN)
                .memory(MemoryConfig.defaults().toBuilder().nCtx(nCtx).ubatch(4).build()).build();
        try (GGUFModelGPU one = new GGUFModelGPU(model.toString(), null, reference)) {
            assertArrayEquals("the split pools as the one-device load", one.encode(PROMPT), splitRow, TOL);
        }
    }
}

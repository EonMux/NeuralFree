package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ocl.OclComputeBackend;
import com.eonmux.neuralfree.gguf.arch.PoolingKind;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Path;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/**
 * The pooling override and the batched encode on the GPU orchestrator: a causal model loaded with the override
 * pools by it, and its paged memory runs the sequences as multi-sequence forwards (two at a time) answering the
 * same rows the one-sequence encodes answer.
 */
public class GpuPoolingOverrideBatchTest {

    private static final float TOL = 2e-3f;

    private HWAcceleration acceleration;
    private ComputeBackend backend;

    @Before
    public void setUp() {
        Assume.assumeTrue("OpenCL GPU not available", GpuTestUtil.isOpenCLAvailable());
        acceleration = new HWAcceleration();
        backend = new OclComputeBackend(acceleration.getCLDevices().get(0));
    }

    @After
    public void tearDown() {
        if (acceleration != null) {
            acceleration.shutdown();
        }
    }

    private static int[] tokens(int seed, int n) {
        int[] t = new int[n];
        for (int i = 0; i < n; i++) {
            t[i] = 8 + (seed * 5 + i * 3) % 180;
        }
        return t;
    }

    @Test
    public void theOverridePoolsAndBatchesOnTheGpuModel() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig cfg = EngineConfig.builder()
                .pooling(PoolingKind.MEAN)
                .memory(MemoryConfig.builder().nCtx(64).ubatch(8).parallel(2)
                        .typeK(KvDType.F32).typeV(KvDType.F32).build())
                .build();
        try (GGUFModelGPU m = new GGUFModelGPU(model.toString(), backend, cfg)) {
            assertEquals(LLMModel.PoolingType.MEAN, m.poolingType());
            int[][] prompts = {tokens(1, 6), tokens(2, 4), tokens(3, 5)};
            float[][] rows = m.encodeBatch(prompts);
            assertEquals("three sequences over a two-sequence memory run as two forwards", 3, rows.length);
            for (int i = 0; i < prompts.length; i++) {
                assertArrayEquals("sequence " + i + " pools as its own forward",
                        m.encode(prompts[i]), rows[i], TOL);
            }
        }
    }
}

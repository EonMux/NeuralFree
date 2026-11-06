package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ocl.OclComputeBackend;
import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.GGUFModelGPU;
import com.eonmux.neuralfree.gguf.TokenCallback;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.GenerationLoop;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.engine.batch.BatchEngine;
import com.eonmux.neuralfree.gguf.engine.batch.BatchSequence;
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
 * Batched short-convolution generation on the GPU paged memory: two
 * concurrent sequences over the synthetic lfm2 decode against their per-sequence device conv-state rows, and both
 * carry the tokens of the same prompts run through the contiguous CPU loop.
 */
public class GpuBatchShortConvTest {

    private static final int N_TOKENS = 24;
    private static final int N_NEW = 6;

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
        if (backend != null) {
            backend.close();
        }
        if (acceleration != null) {
            acceleration.shutdown();
        }
    }

    private static EngineConfig paged() {
        return EngineConfig.builder().memory(MemoryConfig.builder().nCtx(128).ubatch(8).kvPaged(true)
                .kvPageSize(4).parallel(2).typeK(KvDType.F32).typeV(KvDType.F32).build()).build();
    }

    private static GenerationRequest request() {
        return GenerationRequest.builder().maxNewTokens(N_NEW).ignoreEos(true)
                .sampler(SamplerConfig.greedyDecoding()).build();
    }

    private static int[] prompt(int seed, int n) {
        int[] t = new int[n];
        for (int i = 0; i < n; i++) {
            t[i] = 4 + (seed * 37 + i * 11 + (i * i) % 13) % 300;
        }
        return t;
    }

    private static GenerationResult contiguousCpu(Path model, int[] prompt) throws IOException {
        EngineConfig cfg = EngineConfig.builder().memory(MemoryConfig.builder().nCtx(128).ubatch(8)
                .typeK(KvDType.F32).typeV(KvDType.F32).build()).build();
        try (GGUFModel m = new GGUFModel(model.toString(), cfg)) {
            return GenerationLoop.forModel(m, cfg).generate(prompt, request(), (TokenCallback) null);
        }
    }

    @Test
    public void aBatchedShortConvGpuStepComputesTheContiguousTokens() throws IOException {
        Path model = SyntheticModels.lfm2().writeTempFile();
        int[] pa = prompt(3, N_TOKENS);
        int[] pb = prompt(4, N_TOKENS);
        EngineConfig paged = paged();
        try (GGUFModelGPU m = new GGUFModelGPU(model.toString(), backend, paged);
             BatchEngine engine = BatchEngine.forModel(m, paged)) {
            BatchSequence a = engine.submit(pa, request(), (TokenCallback) null);
            BatchSequence b = engine.submit(pb, request(), (TokenCallback) null);
            engine.drain();
            assertEquals("sequence a finished: " + a.error(), BatchSequence.State.DONE, a.state());
            assertEquals("sequence b finished: " + b.error(), BatchSequence.State.DONE, b.state());
            assertArrayEquals("sequence a against the contiguous CPU loop",
                    contiguousCpu(model, pa).tokens(), a.result().tokens());
            assertArrayEquals("sequence b against the contiguous CPU loop",
                    contiguousCpu(model, pb).tokens(), b.result().tokens());
        }
    }
}

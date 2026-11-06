package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.gguf.GGUFModelGPU;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.EngineModel;
import com.eonmux.neuralfree.gguf.engine.ForwardBatch;
import com.eonmux.neuralfree.gguf.engine.LLMModelFactory;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Assume;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Path;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/**
 * Multi-token segments of a paged batch over F16 K/V caches: every row's logits match the CPU's contiguous runs of
 * each sequence. On a backend with tensor cores the prefill plan is the MMA plan, whose segments run the paged MMA
 * prefill kernel ({@code LLM_FlashAttnMmaPaged_F16}); the tiled backends run {@code LLM_FlashAttnPaged_F16}.
 */
public class GpuPagedF16PrefillTest {

    private static final int N_TOKENS = 32;
    private static final float TOL = 5e-3f;

    private static int[] tokens(int seed, int n) {
        int[] t = new int[n];
        for (int i = 0; i < n; i++) {
            t[i] = 4 + (seed * 7 + i * 13) % 250;
        }
        return t;
    }

    @Test
    public void cudaPagedF16SegmentsMatchTheContiguousRows() throws IOException {
        Assume.assumeTrue("CUDA not available", GpuTestUtil.isCudaAvailable());
        assertBatchMatches(EngineConfig.BackendMode.CUDA);
    }

    @Test
    public void openclPagedF16SegmentsMatchTheContiguousRows() throws IOException {
        Assume.assumeTrue("OpenCL not available", GpuTestUtil.isOpenCLAvailable());
        assertBatchMatches(EngineConfig.BackendMode.OPENCL);
    }

    private static void assertBatchMatches(EngineConfig.BackendMode mode) throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        int[] a = tokens(1, N_TOKENS);
        int[] b = tokens(2, N_TOKENS);
        float[][] refA;
        float[][] refB;
        MemoryConfig contiguous = MemoryConfig.builder().nCtx(2 * N_TOKENS).typeK(KvDType.F32).typeV(KvDType.F32)
                .build();
        try (EngineModel cpu = LLMModelFactory.open(model.toString(), EngineConfig.builder()
                .backend(EngineConfig.BackendMode.CPU).memory(contiguous).build())) {
            refA = cpu.forwardBackend().forwardRows(a, 0, null);
            cpu.forwardBackend().reset();
            refB = cpu.forwardBackend().forwardRows(b, 0, null);
        }
        MemoryConfig paged = MemoryConfig.builder().nCtx(2 * N_TOKENS).typeK(KvDType.F16).typeV(KvDType.F16)
                .parallel(2).build();
        try (EngineModel gpu = LLMModelFactory.open(model.toString(), EngineConfig.builder().backend(mode)
                .gpuLayers(EngineConfig.GPU_LAYERS_ALL).memory(paged).build())) {
            Assume.assumeTrue(mode + " did not open the GPU engine", gpu instanceof GGUFModelGPU);
            ForwardBatch.Builder batch = ForwardBatch.builder(2 * N_TOKENS);
            for (int i = 0; i < N_TOKENS; i++) {
                batch.add(a[i], i, 0, true);
            }
            for (int i = 0; i < N_TOKENS; i++) {
                batch.add(b[i], i, 1, true);
            }
            float[][] out = gpu.forwardBackend().forwardBatch(batch.build());
            assertEquals(2 * N_TOKENS, out.length);
            for (int i = 0; i < N_TOKENS; i++) {
                assertArrayEquals(mode + " sequence 0 row " + i, refA[i], out[i], TOL);
                assertArrayEquals(mode + " sequence 1 row " + i, refB[i], out[N_TOKENS + i], TOL);
            }
        }
    }
}

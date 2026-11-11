package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.acceleration.backend.KernelTiming;
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
import java.util.Set;
import java.util.stream.Collectors;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

/**
 * Prefill over Q8_0 K/V caches (head dim 64, so the blocks tile every head): a prompt of three chunks over a dense
 * cache and a two-sequence batch over a paged cache match the CPU engine's rows over the same Q8_0 caches. On a
 * CUDA device with tensor cores the prefill plan is the MMA plan, whose kernels ({@code LLM_FlashAttnMma_Q8_0} and
 * {@code LLM_FlashAttnMmaPaged_Q8_0}) dequantize the cache blocks to F16 as they stage the tiles, and the CUDA case
 * checks in the queue's dispatch timing that the dense run launched the one and the paged batch the other; the other
 * backends run the tiled Q8_0 kernels.
 */
public class GpuQ8KvPrefillTest {

    private static final int N_TOKENS = 40;
    private static final int UBATCH = 16;
    private static final float TOL = 5e-3f;
    /** The paged pool: each sequence's pages round its 40 rows up, so two sequences need more than 80 cells. */
    private static final int PAGED_CELLS = 4 * N_TOKENS;

    private static int[] tokens(int seed, int n) {
        int[] t = new int[n];
        for (int i = 0; i < n; i++) {
            t[i] = 4 + (seed * 7 + i * 13) % 250;
        }
        return t;
    }

    @Test
    public void cudaQ8PrefillMatchesTheCpuEngine() throws IOException {
        Assume.assumeTrue("CUDA not available", GpuTestUtil.isCudaAvailable());
        String prior = System.getProperty(KernelTiming.PROFILE_PROPERTY);
        System.setProperty(KernelTiming.PROFILE_PROPERTY, "true");
        try {
            assertRowsMatch(EngineConfig.BackendMode.CUDA);
        } finally {
            if (prior == null) {
                System.clearProperty(KernelTiming.PROFILE_PROPERTY);
            } else {
                System.setProperty(KernelTiming.PROFILE_PROPERTY, prior);
            }
        }
    }

    @Test
    public void openclQ8PrefillMatchesTheCpuEngine() throws IOException {
        Assume.assumeTrue("OpenCL not available", GpuTestUtil.isOpenCLAvailable());
        assertRowsMatch(EngineConfig.BackendMode.OPENCL);
    }

    /**
     * On a device that runs the tensor-core attention, asserts that the model's queue launched {@code kernel}; the
     * queue times its dispatches under the {@link KernelTiming#PROFILE_PROPERTY} the CUDA case sets.
     */
    private static void assertMmaLaunched(EngineModel gpu, String kernel) {
        GpuForwardContext ctx = ((GpuForwardBackend) gpu.forwardBackend()).context();
        if (!ctx.kernels().flashMmaPath() || !ctx.kernels().has(kernel)) {
            return;
        }
        KernelTiming timing = ctx.queue().kernelTiming();
        assertNotNull("the CUDA queue times its dispatches", timing);
        Set<String> launched = timing.snapshot().stream().map(e -> e.name().split("#", 2)[0])
                .collect(Collectors.toSet());
        assertTrue(kernel + " ran; launched " + launched, launched.contains(kernel));
    }

    /** Q8_0 caches of {@code nCtx} cells, prefilled in {@link #UBATCH}-token steps. */
    private static MemoryConfig.Builder q8(int nCtx) {
        return MemoryConfig.builder().nCtx(nCtx).ubatch(UBATCH).typeK(KvDType.Q8_0).typeV(KvDType.Q8_0);
    }

    private static void assertRowsMatch(EngineConfig.BackendMode mode) throws IOException {
        Path model = SyntheticModels.llama(SyntheticModels.Config.llama().toBuilder().headDim(64).nRot(64).build())
                .writeTempFile();
        int[] a = tokens(1, N_TOKENS);
        int[] b = tokens(2, N_TOKENS);
        float[][] refA;
        float[][] refB;
        try (EngineModel cpu = LLMModelFactory.open(model.toString(), EngineConfig.builder()
                .backend(EngineConfig.BackendMode.CPU).memory(q8(2 * N_TOKENS).build()).build())) {
            refA = cpu.forwardBackend().forwardRows(a, 0, null);
            cpu.forwardBackend().reset();
            refB = cpu.forwardBackend().forwardRows(b, 0, null);
        }
        try (EngineModel gpu = LLMModelFactory.open(model.toString(), EngineConfig.builder().backend(mode)
                .gpuLayers(EngineConfig.GPU_LAYERS_ALL).memory(q8(2 * N_TOKENS).build()).build())) {
            Assume.assumeTrue(mode + " did not open the GPU engine", gpu instanceof GGUFModelGPU);
            float[][] rows = gpu.forwardBackend().forwardRows(a, 0, null);
            for (int i = 0; i < N_TOKENS; i++) {
                assertArrayEquals(mode + " dense row " + i, refA[i], rows[i], TOL);
            }
            assertMmaLaunched(gpu, "LLM_FlashAttnMma_Q8_0");
        }
        try (EngineModel gpu = LLMModelFactory.open(model.toString(), EngineConfig.builder().backend(mode)
                .gpuLayers(EngineConfig.GPU_LAYERS_ALL).memory(q8(PAGED_CELLS).parallel(2).build()).build())) {
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
                assertArrayEquals(mode + " paged sequence 0 row " + i, refA[i], out[i], TOL);
                assertArrayEquals(mode + " paged sequence 1 row " + i, refB[i], out[N_TOKENS + i], TOL);
            }
            assertMmaLaunched(gpu, "LLM_FlashAttnMmaPaged_Q8_0");
        }
    }
}

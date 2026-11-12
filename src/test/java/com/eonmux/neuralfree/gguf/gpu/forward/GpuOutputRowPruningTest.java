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
import java.util.function.UnaryOperator;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Output-row pruning of the last layer on a device ({@link LastLayerRows}): a prompt of three chunks runs the last
 * layer's cache writes alone in its first two chunks and the last layer's FFN over the read row in the third, and the
 * read rows of a call that reads a few rows per chunk (one, a non-contiguous pair) move to the front of the stream.
 * Every read row matches the CPU engine's row of the whole prompt, and a decode step after the prompt, which reads
 * every layer's caches the pruned chunks wrote, matches the CPU engine's. The Mixture-of-Experts cases prune a
 * qwen3moe last layer whose experts sit on the device, and in RAM behind the device expert cache
 * ({@code moeCacheMb}), so the routing and the cached expert pass, with its host-computed misses, run at the pruned
 * row counts. Over a paged memory the batched steps of continuous batching prune the same way: two prompts
 * interleaved across ubatches read only their last rows, a ubatch reading none skips the last FFN, and the
 * sampled rows and a decode step of both sequences after them match the CPU engine's.
 */
public class GpuOutputRowPruningTest {

    private static final int N_TOKENS = 40;
    private static final int UBATCH = 16;
    private static final int NEXT = 17;
    private static final float TOL = 5e-3f;
    /** One row in each of the first two chunks and a non-contiguous pair in the last. */
    private static final int[] READ = {5, 20, 35, 39};

    private static int[] tokens() {
        int[] t = new int[N_TOKENS];
        for (int i = 0; i < N_TOKENS; i++) {
            t[i] = 4 + (i * 13 + 7) % 250;
        }
        return t;
    }

    @Test
    public void cudaPrunedRowsMatchTheCpuEngine() throws IOException {
        Assume.assumeTrue("CUDA not available", GpuTestUtil.isCudaAvailable());
        assertPrunedRowsMatch(EngineConfig.BackendMode.CUDA);
    }

    @Test
    public void openclPrunedRowsMatchTheCpuEngine() throws IOException {
        Assume.assumeTrue("OpenCL not available", GpuTestUtil.isOpenCLAvailable());
        assertPrunedRowsMatch(EngineConfig.BackendMode.OPENCL);
    }

    @Test
    public void vulkanPrunedRowsMatchTheCpuEngine() throws IOException {
        Assume.assumeTrue("Vulkan not available", GpuTestUtil.isVulkanAvailable());
        assertPrunedRowsMatch(EngineConfig.BackendMode.VULKAN);
    }

    @Test
    public void cudaPrunedMoeRowsMatchTheCpuEngine() throws IOException {
        Assume.assumeTrue("CUDA not available", GpuTestUtil.isCudaAvailable());
        assertPrunedMoeRowsMatch(EngineConfig.BackendMode.CUDA);
    }

    @Test
    public void openclPrunedMoeRowsMatchTheCpuEngine() throws IOException {
        Assume.assumeTrue("OpenCL not available", GpuTestUtil.isOpenCLAvailable());
        assertPrunedMoeRowsMatch(EngineConfig.BackendMode.OPENCL);
    }

    @Test
    public void cudaBatchedPrunedRowsMatchTheCpuEngine() throws IOException {
        Assume.assumeTrue("CUDA not available", GpuTestUtil.isCudaAvailable());
        assertBatchedPrunedRowsMatch(EngineConfig.BackendMode.CUDA);
    }

    @Test
    public void openclBatchedPrunedRowsMatchTheCpuEngine() throws IOException {
        Assume.assumeTrue("OpenCL not available", GpuTestUtil.isOpenCLAvailable());
        assertBatchedPrunedRowsMatch(EngineConfig.BackendMode.OPENCL);
    }

    @Test
    public void vulkanBatchedPrunedRowsMatchTheCpuEngine() throws IOException {
        Assume.assumeTrue("Vulkan not available", GpuTestUtil.isVulkanAvailable());
        assertBatchedPrunedRowsMatch(EngineConfig.BackendMode.VULKAN);
    }

    /**
     * Two prompts in one paged batch, sequence 0's 29 rows before sequence 1's 11 (ubatches of 16: the first reads
     * no row, the second sequence 0's last, the third sequence 1's last), then one decode row each.
     */
    private static void assertBatchedPrunedRowsMatch(EngineConfig.BackendMode mode) throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        int[] tokens = tokens();
        int split = 29;
        ForwardBatch.Builder prompts = ForwardBatch.builder(N_TOKENS);
        for (int i = 0; i < split; i++) {
            prompts.add(tokens[i], i, 0, i == split - 1);
        }
        for (int i = split; i < N_TOKENS; i++) {
            prompts.add(tokens[i], i - split, 1, i == N_TOKENS - 1);
        }
        ForwardBatch batch = prompts.build();
        ForwardBatch decode = ForwardBatch.builder(2).add(NEXT, split, 0, true)
                .add(NEXT + 1, N_TOKENS - split, 1, true).build();
        MemoryConfig memory = MemoryConfig.builder().nCtx(4 * N_TOKENS).ubatch(UBATCH).parallel(2)
                .typeK(KvDType.F32).typeV(KvDType.F32).build();
        float[][] ref;
        float[][] refNext;
        try (EngineModel cpu = LLMModelFactory.open(model.toString(), EngineConfig.builder()
                .backend(EngineConfig.BackendMode.CPU).memory(memory).build())) {
            ref = cpu.forwardBackend().forwardBatch(batch);
            refNext = cpu.forwardBackend().forwardBatch(decode);
        }
        try (EngineModel gpu = LLMModelFactory.open(model.toString(), EngineConfig.builder().backend(mode)
                .gpuLayers(EngineConfig.GPU_LAYERS_ALL).memory(memory).build())) {
            Assume.assumeTrue(mode + " did not open the GPU engine", gpu instanceof GGUFModelGPU);
            float[][] rows = gpu.forwardBackend().forwardBatch(batch);
            assertEquals(2, rows.length);
            for (int i = 0; i < rows.length; i++) {
                assertArrayEquals(mode + " batched prompt row " + i, ref[i], rows[i], TOL);
            }
            float[][] next = gpu.forwardBackend().forwardBatch(decode);
            for (int i = 0; i < next.length; i++) {
                assertArrayEquals(mode + " decode after the pruned batch, sequence " + i, refNext[i], next[i], TOL);
            }
        }
    }

    private static void assertPrunedRowsMatch(EngineConfig.BackendMode mode) throws IOException {
        assertPrunedRowsMatch(mode, SyntheticModels.llama().writeTempFile(), "", b -> b);
    }

    /** The qwen3moe model with its experts on the device, and in RAM behind the device expert cache. */
    private static void assertPrunedMoeRowsMatch(EngineConfig.BackendMode mode) throws IOException {
        Path model = SyntheticModels.qwen3moe().writeTempFile();
        assertPrunedRowsMatch(mode, model, "moe device experts: ", b -> b);
        assertPrunedRowsMatch(mode, model, "moe cached experts: ", b -> b.moeCacheMb(4));
    }

    private static void assertPrunedRowsMatch(EngineConfig.BackendMode mode, Path model, String what,
                                              UnaryOperator<EngineConfig.Builder> gpuConfig) throws IOException {
        int[] tokens = tokens();
        MemoryConfig memory = MemoryConfig.builder().nCtx(2 * N_TOKENS).ubatch(UBATCH).typeK(KvDType.F32)
                .typeV(KvDType.F32).build();
        float[][] ref;
        float[] refNext;
        try (EngineModel cpu = LLMModelFactory.open(model.toString(), EngineConfig.builder()
                .backend(EngineConfig.BackendMode.CPU).memory(memory).build())) {
            ref = cpu.forwardBackend().forwardRows(tokens, 0, null);
            refNext = cpu.forwardBackend().decode(NEXT, N_TOKENS);
        }
        try (EngineModel gpu = LLMModelFactory.open(model.toString(), gpuConfig.apply(EngineConfig.builder()
                .backend(mode).gpuLayers(EngineConfig.GPU_LAYERS_ALL).memory(memory)).build())) {
            Assume.assumeTrue(mode + " did not open the GPU engine", gpu instanceof GGUFModelGPU);
            String at = what + mode;
            GpuLayerExecutor executor = ((GpuForwardBackend) gpu.forwardBackend()).layerExecutor();
            assertTrue(at + " prunes the last layer of a chunk read at its last row", executor.prunesOutputRows(
                    UBATCH, GpuLayerExecutor.StepMode.PREFILL, GpuLayerExecutor.TapSink.NONE,
                    new int[]{UBATCH - 1}));
            float[] last = gpu.forwardBackend().prefill(tokens, 0);
            assertArrayEquals(at + " prompt's last row", ref[N_TOKENS - 1], last, TOL);
            assertArrayEquals(at + " decode after the pruned prompt", refNext,
                    gpu.forwardBackend().decode(NEXT, N_TOKENS), TOL);

            gpu.forwardBackend().reset();
            float[][] rows = gpu.forwardBackend().forwardRows(tokens, 0, READ);
            assertEquals(READ.length, rows.length);
            for (int i = 0; i < READ.length; i++) {
                assertArrayEquals(at + " row " + READ[i], ref[READ[i]], rows[i], TOL);
            }
            assertArrayEquals(at + " decode after the pruned rows", refNext,
                    gpu.forwardBackend().decode(NEXT, N_TOKENS), TOL);
        }
    }
}

package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.EngineModel;
import com.eonmux.neuralfree.gguf.engine.LLMModelFactory;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Assume;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;

/**
 * The per-token-head KV caches over a real synthetic llama model on the GPU against the CPU: both engines store the
 * same codes and per-(token, KV head) scales, so the flash attention's row-scale lookups must reproduce the CPU
 * attention's. The INT4 cache exercises the nibble loads, the INT8 cache the signed bytes and the FP8 cache the E4M3
 * codes.
 */
public class GpuPerTokenHeadKvTest {

    private static final List<KvDType> TYPES = Arrays.asList(KvDType.FP8_PER_TOKEN_HEAD,
            KvDType.INT8_PER_TOKEN_HEAD, KvDType.INT4_PER_TOKEN_HEAD);
    private static final int[] TOKENS = {1, 7, 23, 42, 99, 128, 201, 256};
    private static final float TOL = 2e-2f;

    @Test
    public void cudaMatchesTheCpuForEveryPerTokenHeadType() throws IOException {
        Assume.assumeTrue("CUDA not available", GpuTestUtil.isCudaAvailable());
        assertMatchesCpu(EngineConfig.BackendMode.CUDA);
    }

    @Test
    public void openclMatchesTheCpuForEveryPerTokenHeadType() throws IOException {
        Assume.assumeTrue("OpenCL not available", GpuTestUtil.isOpenCLAvailable());
        assertMatchesCpu(EngineConfig.BackendMode.OPENCL);
    }

    @Test
    public void vulkanMatchesTheCpuForEveryPerTokenHeadType() throws IOException {
        Assume.assumeTrue("Vulkan not available", GpuTestUtil.isVulkanAvailable());
        assertMatchesCpu(EngineConfig.BackendMode.VULKAN);
    }

    private static void assertMatchesCpu(EngineConfig.BackendMode mode) throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        try {
            for (KvDType t : TYPES) {
                float[][][] cpu = steps(model, EngineConfig.BackendMode.CPU, t);
                float[][][] gpu = steps(model, mode, t);
                for (int s = 0; s < cpu.length; s++) {
                    for (int r = 0; r < cpu[s].length; r++) {
                        assertArrayEquals(mode + " " + t + " step " + s + " row " + r, cpu[s][r], gpu[s][r], TOL);
                    }
                }
            }
        } finally {
            java.nio.file.Files.deleteIfExists(model);
        }
    }

    @Test
    public void cudaSessionsCarryTheRowScales() throws IOException {
        Assume.assumeTrue("CUDA not available", GpuTestUtil.isCudaAvailable());
        assertSessions(EngineConfig.BackendMode.CUDA);
    }

    @Test
    public void openclSessionsCarryTheRowScales() throws IOException {
        Assume.assumeTrue("OpenCL not available", GpuTestUtil.isOpenCLAvailable());
        assertSessions(EngineConfig.BackendMode.OPENCL);
    }

    /** A CPU session of each type loads on {@code mode}, saves back bit for bit and continues the CPU's sequence. */
    private static void assertSessions(EngineConfig.BackendMode mode) throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        Path dir = java.nio.file.Files.createTempDirectory("nf-pt-session-");
        int next = 101;
        try {
            for (KvDType t : TYPES) {
                Path saved = dir.resolve(t + "-cpu.nfst");
                Path copy = dir.resolve(t + "-gpu.nfst");
                float[] cpuNext;
                try (EngineModel cpu = openEngine(model, EngineConfig.BackendMode.CPU, t)) {
                    cpu.forwardBackend().prefill(TOKENS, 0);
                    cpu.forwardBackend().memory().saveSession(saved);
                    cpuNext = cpu.forwardBackend().decode(next, TOKENS.length);
                }
                try (EngineModel gpu = openEngine(model, mode, t)) {
                    com.eonmux.neuralfree.gguf.memory.ModelMemory memory = gpu.forwardBackend().memory();
                    memory.loadSession(saved);
                    memory.saveSession(copy);
                    assertArrayEquals(t + ": the device holds the CPU's rows and row scales bit for bit",
                            java.nio.file.Files.readAllBytes(saved), java.nio.file.Files.readAllBytes(copy));
                    assertArrayEquals(t + ": the device continues the CPU's sequence", cpuNext,
                            gpu.forwardBackend().decode(next, TOKENS.length), TOL);
                }
            }
        } finally {
            try (java.util.stream.Stream<Path> files = java.nio.file.Files.list(dir)) {
                files.forEach(p -> p.toFile().delete());
            }
            java.nio.file.Files.deleteIfExists(dir);
            java.nio.file.Files.deleteIfExists(model);
        }
    }

    /** Opens {@code mode} on a per-token-head {@code type}, skipping when a GPU fallback did not build its kernels. */
    private static EngineModel openEngine(Path model, EngineConfig.BackendMode mode, KvDType type) throws IOException {
        MemoryConfig memory = MemoryConfig.builder().nCtx(64).ubatch(TOKENS.length)
                .typeK(type).typeV(type).build();
        EngineConfig.Builder config = EngineConfig.builder().backend(mode).memory(memory);
        if (mode != EngineConfig.BackendMode.CPU) {
            config.gpuLayers(EngineConfig.GPU_LAYERS_ALL);
        }
        EngineModel engine = LLMModelFactory.open(model.toString(), config.build());
        if (mode != EngineConfig.BackendMode.CPU && !(engine instanceof com.eonmux.neuralfree.gguf.GGUFModelGPU)) {
            engine.close();
            Assume.assumeTrue(mode + " kernels were not built: the engine fell back to the CPU", false);
        }
        return engine;
    }

    /** A prefill of every {@link #TOKENS} row, then one decode row through the per-token-head cache. */
    private static float[][][] steps(Path model, EngineConfig.BackendMode mode, KvDType type) throws IOException {
        MemoryConfig memory = MemoryConfig.builder().nCtx(64).ubatch(TOKENS.length)
                .typeK(type).typeV(type).build();
        EngineConfig.Builder config = EngineConfig.builder().backend(mode).memory(memory);
        if (mode != EngineConfig.BackendMode.CPU) {
            config.gpuLayers(EngineConfig.GPU_LAYERS_ALL);
        }
        try (EngineModel engine = LLMModelFactory.open(model.toString(), config.build())) {
            if (mode != EngineConfig.BackendMode.CPU) {
                Assume.assumeTrue(mode + " kernels were not built: the engine fell back to the CPU",
                        engine instanceof com.eonmux.neuralfree.gguf.GGUFModelGPU);
            }
            com.eonmux.neuralfree.gguf.engine.ForwardBackend backend = engine.forwardBackend();
            float[][] prefill = backend.forwardRows(TOKENS, 0, null);
            float[][] decode = backend.forwardRows(new int[]{TOKENS[0]}, TOKENS.length, new int[]{0});
            return new float[][][]{prefill, decode};
        }
    }
}

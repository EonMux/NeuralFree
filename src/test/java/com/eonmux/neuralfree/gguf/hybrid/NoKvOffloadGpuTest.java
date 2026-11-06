package com.eonmux.neuralfree.gguf.hybrid;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.GGUFModelGPU;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.After;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * A GPU load with {@code neuralfree.kv.offload=false} (llama.cpp {@code -nkvo}) of a synthetic llama against the CPU
 * engine: the weights, projections and FFN compute on the device, every layer's attention computes on the host over
 * a host-resident KV cache, and both engines agree on the prefill logits, the greedy tokens and the hidden rows.
 */
public class NoKvOffloadGpuTest {

    private static final int N_LAYER = 4;
    private static final float LOGITS_TOL = 1e-3f;
    private static final int[] PROMPT = {3, 17, 5, 9, 200, 42};

    private Path model;

    @BeforeClass
    public static void requireGpu() {
        GpuTestUtil.assumeOpenCLAvailable();
    }

    @Before
    public void openModel() throws IOException {
        model = SyntheticModels.llama(SyntheticModels.Config.llama().toBuilder().nLayer(N_LAYER).build())
                .writeTempFile();
    }

    @After
    public void closeModel() throws IOException {
        if (model != null) {
            Files.deleteIfExists(model);
        }
    }

    @Test
    public void prefillLogitsAndGreedyTokensMatchTheCpuEngine() throws IOException {
        float[] cpuLogits = cpuPrefillLogits();
        int[] reference = cpuGreedy();
        try (GGUFModelGPU gpu = open()) {
            assertTrue("the load runs the hybrid backend", gpu.forwardBackend() instanceof HybridForwardBackend);
            assertTrue("the KV caches are in host RAM", gpu.kvOnHost());
            ForwardBackend f = gpu.forwardBackend();
            float[] logits = f.forwardRows(PROMPT, 0, new int[]{PROMPT.length - 1})[0];
            assertClose("prefill logits", cpuLogits, logits);
            assertArrayEquals("greedy tokens", reference, greedy(f, PROMPT, 8));
        }
    }

    @Test
    public void decodeStepsMatchTheCpuEngine() throws IOException {
        int[] reference;
        try (GGUFModel cpu = new GGUFModel(model.toString(), cpuConfig())) {
            reference = greedy(cpu.forwardBackend(), PROMPT, 6);
        }
        try (GGUFModelGPU gpu = open()) {
            ForwardBackend f = gpu.forwardBackend();
            f.reset();
            float[] logits = f.forwardRows(PROMPT, 0, new int[]{PROMPT.length - 1})[0];
            for (int i = 0; i < 6; i++) {
                int next = ForwardBackend.argmax(logits);
                assertEquals("token " + i + " of the continuation", reference[PROMPT.length + i], next);
                logits = f.decode(next, PROMPT.length + i);
            }
            assertEquals(PROMPT.length + 6, f.memory().sequence().nPast());
        }
    }

    @Test
    public void theHostKvMemoryRollsBackAndResetsLikeOneMemory() throws IOException {
        int[] reference = cpuGreedy();
        try (GGUFModelGPU gpu = open()) {
            ForwardBackend f = gpu.forwardBackend();
            f.decode(PROMPT[0], 0);
            f.decode(PROMPT[1], 1);
            f.decode(PROMPT[2], 2);
            assertEquals(3, f.memory().sequence().nPast());
            assertTrue(f.canRollback(2));
            f.rollback(2);
            assertEquals(1, f.memory().sequence().nPast());
            f.reset();
            assertEquals(0, f.memory().sequence().nPast());
            assertArrayEquals("greedy tokens after a rollback and reset", reference, greedy(f, PROMPT, 8));
        }
    }

    @Test
    public void hiddenRowsMatchTheCpuEngine() throws IOException {
        float[][] reference;
        try (GGUFModel cpu = new GGUFModel(model.toString(), cpuConfig())) {
            reference = cpu.forwardBackend().hiddenRows(PROMPT, null);
        }
        try (GGUFModelGPU gpu = open()) {
            float[][] rows = gpu.forwardBackend().hiddenRows(PROMPT, null);
            assertEquals(reference.length, rows.length);
            for (int i = 0; i < reference.length; i++) {
                assertClose("hidden row " + i, reference[i], rows[i]);
            }
        }
    }

    /** A Q8_0 cache is rotated and quantized on store; the host stage applies the same CPU rotation. */
    @Test
    public void aRotatedQuantizedKvCacheMatchesTheCpuEngine() throws IOException {
        Path wide = SyntheticModels.llama(SyntheticModels.Config.llama().toBuilder().nLayer(N_LAYER)
                .headDim(64).nRot(64).build()).writeTempFile();
        try {
            MemoryConfig memory = MemoryConfig.defaults().toBuilder()
                    .typeK(KvDType.Q8_0).typeV(KvDType.Q8_0).build();
            int[] reference;
            try (GGUFModel cpu = new GGUFModel(wide.toString(), EngineConfig.builder()
                    .backend(EngineConfig.BackendMode.CPU).memory(memory).build())) {
                reference = greedy(cpu.forwardBackend(), PROMPT, 3);
            }
            try (GGUFModelGPU gpu = new GGUFModelGPU(wide.toString(), null, EngineConfig.builder()
                    .backend(EngineConfig.BackendMode.OPENCL)
                    .gpuLayers(EngineConfig.GPU_LAYERS_ALL)
                    .kvOffload(false)
                    .memory(memory)
                    .build())) {
                assertTrue(gpu.kvOnHost());
                assertArrayEquals("greedy tokens over a rotated Q8_0 host cache", reference,
                        greedy(gpu.forwardBackend(), PROMPT, 3));
            }
        } finally {
            Files.deleteIfExists(wide);
        }
    }

    @Test
    public void anFp8KvCacheMatchesTheCpuEngine() throws IOException {
        MemoryConfig memory = MemoryConfig.defaults().toBuilder()
                .typeK(KvDType.FP8_E4M3).typeV(KvDType.FP8_E5M2).build();
        int[] reference;
        try (GGUFModel cpu = new GGUFModel(model.toString(), EngineConfig.builder()
                .backend(EngineConfig.BackendMode.CPU).memory(memory).build())) {
            reference = greedy(cpu.forwardBackend(), PROMPT, 3);
        }
        try (GGUFModelGPU gpu = new GGUFModelGPU(model.toString(), null, EngineConfig.builder()
                .backend(EngineConfig.BackendMode.OPENCL)
                .gpuLayers(EngineConfig.GPU_LAYERS_ALL)
                .kvOffload(false)
                .memory(memory)
                .build())) {
            assertTrue(gpu.kvOnHost());
            assertArrayEquals("greedy tokens over an FP8 host cache", reference,
                    greedy(gpu.forwardBackend(), PROMPT, 3));
        }
    }

    /** A partial layer split keeps the host prefix whole and the device layers' KV in host RAM. */
    @Test
    public void aPartialSplitKeepsTheDeviceLayersKvOnTheHost() throws IOException {
        int[] reference = cpuGreedy();
        try (GGUFModelGPU gpu = new GGUFModelGPU(model.toString(), null, EngineConfig.builder()
                .backend(EngineConfig.BackendMode.OPENCL)
                .gpuLayers(3)
                .kvOffload(false)
                .build())) {
            assertTrue(gpu.kvOnHost());
            assertArrayEquals("greedy tokens of a partial split with host KV", reference,
                    greedy(gpu.forwardBackend(), PROMPT, 8));
        }
    }

    /** A sliding-window, soft-capped model attends over the host cache with the same window rules as the CPU. */
    @Test
    public void aSlidingWindowModelMatchesTheCpuEngine() throws IOException {
        Path gemma = SyntheticModels.gemma2().writeTempFile();
        try {
            int[] reference;
            try (GGUFModel cpu = new GGUFModel(gemma.toString(), cpuConfig())) {
                reference = greedy(cpu.forwardBackend(), PROMPT, 4);
            }
            try (GGUFModelGPU gpu = new GGUFModelGPU(gemma.toString(), null, EngineConfig.builder()
                    .backend(EngineConfig.BackendMode.OPENCL)
                    .gpuLayers(EngineConfig.GPU_LAYERS_ALL)
                    .kvOffload(false)
                    .build())) {
                assertTrue(gpu.kvOnHost());
                assertArrayEquals("greedy tokens over a sliding-window host cache", reference,
                        greedy(gpu.forwardBackend(), PROMPT, 4));
            }
        } finally {
            Files.deleteIfExists(gemma);
        }
    }

    /** Only the attention layer's KV moves to the host; the recurrent layers keep their state on the device. */
    @Test
    public void aHybridRecurrentModelMatchesTheCpuEngine() throws IOException {
        Path hybrid = SyntheticModels.qwen3next().writeTempFile();
        try {
            int[] reference;
            try (GGUFModel cpu = new GGUFModel(hybrid.toString(), cpuConfig())) {
                reference = greedy(cpu.forwardBackend(), PROMPT, 4);
            }
            try (GGUFModelGPU gpu = new GGUFModelGPU(hybrid.toString(), null, EngineConfig.builder()
                    .backend(EngineConfig.BackendMode.OPENCL)
                    .gpuLayers(EngineConfig.GPU_LAYERS_ALL)
                    .kvOffload(false)
                    .build())) {
                assertTrue(gpu.kvOnHost());
                assertArrayEquals("greedy tokens with only the attention KV on the host", reference,
                        greedy(gpu.forwardBackend(), PROMPT, 4));
            }
        } finally {
            Files.deleteIfExists(hybrid);
        }
    }

    private GGUFModelGPU open() throws IOException {
        return new GGUFModelGPU(model.toString(), null, EngineConfig.builder()
                .backend(EngineConfig.BackendMode.OPENCL)
                .gpuLayers(EngineConfig.GPU_LAYERS_ALL)
                .kvOffload(false)
                .build());
    }

    private EngineConfig cpuConfig() {
        return EngineConfig.builder().backend(EngineConfig.BackendMode.CPU).build();
    }

    private int[] cpuGreedy() throws IOException {
        try (GGUFModel cpu = new GGUFModel(model.toString(), cpuConfig())) {
            return greedy(cpu.forwardBackend(), PROMPT, 8);
        }
    }

    private float[] cpuPrefillLogits() throws IOException {
        try (GGUFModel cpu = new GGUFModel(model.toString(), cpuConfig())) {
            ForwardBackend f = cpu.forwardBackend();
            f.reset();
            return f.forwardRows(PROMPT, 0, new int[]{PROMPT.length - 1})[0];
        }
    }

    private static int[] greedy(ForwardBackend f, int[] prompt, int n) {
        int[] out = Arrays.copyOf(prompt, prompt.length + n);
        f.reset();
        float[] logits = f.forwardRows(prompt, 0, new int[]{prompt.length - 1})[0];
        for (int i = 0; i < n; i++) {
            int next = ForwardBackend.argmax(logits);
            out[prompt.length + i] = next;
            logits = f.decode(next, prompt.length + i);
        }
        return out;
    }

    private static void assertClose(String what, float[] expected, float[] actual) {
        assertEquals(what + ": length", expected.length, actual.length);
        float max = 0;
        for (int i = 0; i < expected.length; i++) {
            max = Math.max(max, Math.abs(expected[i] - actual[i]));
        }
        assertTrue(what + ": max abs diff " + max, max <= LOGITS_TOL);
    }
}

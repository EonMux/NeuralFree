package com.eonmux.neuralfree.gguf.hybrid;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.GGUFModelGPU;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
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
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * A hybrid CPU+GPU load of a synthetic llama against the CPU engine over the real OpenCL backend: the split layers
 * compute on the host with their KV there, the rest on the device with its KV there, and both engines agree on the
 * logits, the greedy tokens and the hidden rows. The {@code --cpu-moe} form runs the same agreement with the
 * experts of a MoE model computed on the host inside device layers.
 */
public class HybridLayerOffloadGpuTest {

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
    public void halfOffloadedLayerSplitMatchesTheCpuEngine() throws IOException {
        int[] reference = cpuGreedy();
        float[] cpuLogits = cpuPrefillLogits();
        try (GGUFModelGPU hybrid = open(EngineConfig.builder().gpuLayers(3))) {
            ForwardBackend f = hybrid.forwardBackend();
            assertTrue(f instanceof HybridForwardBackend);
            float[] logits = f.forwardRows(PROMPT, 0, new int[]{PROMPT.length - 1})[0];
            assertClose("prefill logits", cpuLogits, logits);
            int[] greedy = greedy(f, PROMPT, 8);
            assertArrayEquals("greedy tokens of the split load", reference, greedy);
        }
    }

    /**
     * A verify-shaped call (the prompt followed by the CPU engine's greedy continuation) selected on the device: the
     * greedy id of every row from the last prompt token on is the continuation's next token.
     */
    @Test
    public void greedyRowsSelectedByTheSplitHeadFollowTheCpuContinuation() throws IOException {
        int[] reference = cpuGreedy();
        try (GGUFModelGPU hybrid = open(EngineConfig.builder().gpuLayers(3))) {
            ForwardBackend f = hybrid.forwardBackend();
            f.reset();
            int[] batch = Arrays.copyOf(reference, reference.length - 1);
            int[] rows = new int[reference.length - PROMPT.length];
            for (int i = 0; i < rows.length; i++) {
                rows[i] = PROMPT.length - 1 + i;
            }
            int[] ids = f.forwardRowsGreedy(batch, 0, rows);
            assertArrayEquals(Arrays.copyOfRange(reference, PROMPT.length, reference.length), ids);
            assertEquals(batch.length, f.memory().sequence().nPast());
        }
    }

    @Test
    public void oneDeviceLayerSplitMatchesTheCpuEngine() throws IOException {
        int[] reference = cpuGreedy();
        try (GGUFModelGPU hybrid = open(EngineConfig.builder().gpuLayers(1))) {
            assertTrue(hybrid.forwardBackend() instanceof HybridForwardBackend);
            int[] greedy = greedy(hybrid.forwardBackend(), PROMPT, 8);
            assertArrayEquals("greedy tokens with every trunk layer on the host", reference, greedy);
        }
    }

    @Test
    public void theSplitMemoryRollsBackAndResetsLikeOneMemory() throws IOException {
        int[] reference = cpuGreedy();
        try (GGUFModelGPU hybrid = open(EngineConfig.builder().gpuLayers(3))) {
            ForwardBackend f = hybrid.forwardBackend();
            int[] prefix = Arrays.copyOf(PROMPT, 3);
            f.decode(prefix[0], 0);
            f.decode(prefix[1], 1);
            f.decode(prefix[2], 2);
            assertEquals(3, f.memory().sequence().nPast());
            assertTrue(f.canRollback(2));
            f.rollback(2);
            assertEquals(1, f.memory().sequence().nPast());
            f.reset();
            assertEquals(0, f.memory().sequence().nPast());
            int[] greedy = greedy(f, PROMPT, 8);
            assertArrayEquals("greedy tokens after a rollback and reset", reference, greedy);
        }
    }

    @Test
    public void hiddenRowsOfTheSplitMatchTheCpuEngine() throws IOException {
        float[][] reference;
        try (GGUFModel cpu = new GGUFModel(model.toString(), EngineConfig.builder()
                .backend(EngineConfig.BackendMode.CPU).build())) {
            reference = cpu.forwardBackend().hiddenRows(PROMPT, null);
        }
        try (GGUFModelGPU hybrid = open(EngineConfig.builder().gpuLayers(3))) {
            float[][] rows = hybrid.forwardBackend().hiddenRows(PROMPT, null);
            assertEquals(reference.length, rows.length);
            for (int i = 0; i < rows.length; i++) {
                assertClose("hidden row " + i, reference[i], rows[i]);
            }
        }
    }

    @Test
    public void theGenerationLoopOfTheSplitMatchesTheCpuEngine() throws IOException {
        int[] reference;
        try (GGUFModel cpu = new GGUFModel(model.toString(), EngineConfig.builder()
                .backend(EngineConfig.BackendMode.CPU).build())) {
            reference = cpu.generate(PROMPT, 8, 0f, 1f);
        }
        try (GGUFModelGPU hybrid = open(EngineConfig.builder().gpuLayers(3))) {
            assertArrayEquals("generated tokens of the split load through the generation loop",
                    reference, hybrid.generate(PROMPT, 8, 0f, 1f));
        }
    }

    @Test
    public void anAutomaticRequestWithRoomForEverythingKeepsThePureGpuLoad() throws IOException {
        try (GGUFModelGPU gpu = open(EngineConfig.builder().gpuLayers(EngineConfig.GPU_LAYERS_AUTO))) {
            assertFalse(gpu.forwardBackend() instanceof HybridForwardBackend);
            int[] reference = cpuGreedy();
            assertArrayEquals("greedy tokens of the pure GPU load", reference,
                    greedy(gpu.forwardBackend(), PROMPT, 8));
        }
    }

    @Test
    public void cpuMoeComputesTheExpertsOnTheHostOfADeviceLayer() throws IOException {
        Path moe = SyntheticModels.qwen3next().writeTempFile();
        try {
            int[] reference;
            try (GGUFModel cpu = new GGUFModel(moe.toString(), EngineConfig.builder()
                    .backend(EngineConfig.BackendMode.CPU).build())) {
                reference = greedy(cpu.forwardBackend(), PROMPT, 4);
            }
            try (GGUFModelGPU hybrid = openMoe(moe, EngineConfig.builder()
                    .gpuLayers(EngineConfig.GPU_LAYERS_ALL).cpuMoe(true))) {
                ForwardBackend f = hybrid.forwardBackend();
                assertTrue(f instanceof HybridForwardBackend);
                assertArrayEquals("greedy tokens with the experts on the host", reference,
                        greedy(f, PROMPT, 4));
            }
        } finally {
            Files.deleteIfExists(moe);
        }
    }

    private GGUFModelGPU open(EngineConfig.Builder b) throws IOException {
        return new GGUFModelGPU(model.toString(), null, b.backend(EngineConfig.BackendMode.OPENCL).build());
    }

    private GGUFModelGPU openMoe(Path path, EngineConfig.Builder b) throws IOException {
        return new GGUFModelGPU(path.toString(), null, b.backend(EngineConfig.BackendMode.OPENCL).build());
    }

    private int[] cpuGreedy() throws IOException {
        try (GGUFModel cpu = new GGUFModel(model.toString(), EngineConfig.builder()
                .backend(EngineConfig.BackendMode.CPU).build())) {
            return greedy(cpu.forwardBackend(), PROMPT, 8);
        }
    }

    private float[] cpuPrefillLogits() throws IOException {
        try (GGUFModel cpu = new GGUFModel(model.toString(), EngineConfig.builder()
                .backend(EngineConfig.BackendMode.CPU).build())) {
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

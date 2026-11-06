package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ocl.OclComputeBackend;
import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.GGUFModelGPU;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.engine.ForwardBatch;
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
import static org.junit.Assert.assertTrue;

/**
 * The paged multi-sequence steps of the GPU MLA mixer over the synthetic deepseek2 (q_lora MLA, absorbed latent
 * cache): an interleaved two-sequence prompt batch (one paged prefill per segment) and the decode round after it (one
 * paged decode launch) equal the CPU backend on the same batches, causal and non-causal. A non-causal step lifts the
 * causal bit of the paged launches ({@code GpuStepInputs.stepNonCausal}), so a sequence's first row sees its later
 * rows, while the two sequences still never see each other.
 */
public class GpuMlaPagedBatchTest {

    private static final int SEQ_A = 0;
    private static final int SEQ_B = 1;
    private static final int N_TOKENS = 4;
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
        if (backend != null) {
            backend.close();
        }
        if (acceleration != null) {
            acceleration.shutdown();
        }
    }

    private static EngineConfig config() {
        return EngineConfig.builder().memory(MemoryConfig.builder().nCtx(64).ubatch(8).parallel(2)
                .typeK(KvDType.F32).typeV(KvDType.F32).build()).build();
    }

    private static int[] tokens(int seed, int n) {
        int[] t = new int[n];
        for (int i = 0; i < n; i++) {
            t[i] = 4 + (seed * 7 + i * 13) % 250;
        }
        return t;
    }

    /** Two interleaved prompts, every row an output row, then one decode row per sequence. */
    private static float[][] run(ForwardBackend f, int[] a, int[] b, boolean nonCausal) {
        ForwardBatch.Builder batch = ForwardBatch.builder(a.length + b.length);
        for (int i = 0; i < a.length; i++) {
            batch.add(a[i], i, SEQ_A, true);
            batch.add(b[i], i, SEQ_B, true);
        }
        if (nonCausal) {
            batch.nonCausal();
        }
        float[][] prompt = f.forwardBatch(batch.build());
        ForwardBatch.Builder round = ForwardBatch.builder(2);
        round.add(a[0], a.length, SEQ_A, true);
        round.add(b[0], b.length, SEQ_B, true);
        float[][] decoded = f.forwardBatch(round.build());
        float[][] all = java.util.Arrays.copyOf(prompt, prompt.length + decoded.length);
        System.arraycopy(decoded, 0, all, prompt.length, decoded.length);
        return all;
    }

    private float[][][] both(Path model, int[] a, int[] b, boolean nonCausal) throws IOException {
        float[][][] out = new float[2][][];
        try (GGUFModel cpu = new GGUFModel(model.toString(), config())) {
            out[0] = run(cpu.forwardBackend(), a, b, nonCausal);
        }
        try (GGUFModelGPU gpu = new GGUFModelGPU(model.toString(), backend, config())) {
            out[1] = run(gpu.forwardBackend(), a, b, nonCausal);
        }
        return out;
    }

    private static void assertRowsEqual(String what, float[][] expected, float[][] actual) {
        assertEquals(what + ": row count", expected.length, actual.length);
        for (int i = 0; i < expected.length; i++) {
            assertArrayEquals(what + ": row " + i, expected[i], actual[i], TOL);
        }
    }

    @Test
    public void aCausalTwoSequenceBatchMatchesTheCpuBatch() throws IOException {
        Path model = SyntheticModels.deepseek2().writeTempFile();
        float[][][] out = both(model, tokens(1, N_TOKENS), tokens(2, N_TOKENS), false);
        assertRowsEqual("GPU paged MLA batch vs CPU", out[0], out[1]);
    }

    @Test
    public void aNonCausalTwoSequenceBatchMatchesTheCpuBatchAndSeesLaterRows() throws IOException {
        Path model = SyntheticModels.deepseek2().writeTempFile();
        int[] a = tokens(3, N_TOKENS);
        int[] b = tokens(4, N_TOKENS);
        float[][][] nonCausal = both(model, a, b, true);
        assertRowsEqual("GPU non-causal paged MLA batch vs CPU", nonCausal[0], nonCausal[1]);
        float[][] causal;
        try (GGUFModelGPU gpu = new GGUFModelGPU(model.toString(), backend, config())) {
            causal = run(gpu.forwardBackend(), a, b, false);
        }
        float diff = 0f;
        for (int v = 0; v < causal[0].length; v++) {
            diff = Math.max(diff, Math.abs(causal[0][v] - nonCausal[1][0][v]));
        }
        assertTrue("sequence A's first row sees its later rows only without the causal mask: " + diff, diff > 1e-2f);
    }
}

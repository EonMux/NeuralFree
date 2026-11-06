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
import com.eonmux.neuralfree.gguf.memory.SequenceMemory;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

/**
 * The paged multi-sequence steps of the GPU DSA sparse-MLA mixer over the synthetic deepseek32 (two DSA layers with
 * a full lightning indexer, top-k 8): two interleaved 12-token prompts (three ubatches of four rows per sequence, so
 * the later steps select among more cells than the top-k holds) and the decode round after them equal the CPU paged
 * batch, causal and non-causal. Every segment scores all cells against its own sequence's {@code kv_pos}, so the two
 * sequences never select each other's rows. A sequence forked from part of another shares its pages until it writes
 * into one; the copy-on-write page copy then moves the indexer keys with the latent rows, so the fork selects as the
 * CPU fork does.
 */
public class GpuDsaPagedBatchTest {

    private static final int SEQ_A = 0;
    private static final int SEQ_B = 1;
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

    static int[] tokens(int seed, int n) {
        int[] t = new int[n];
        for (int i = 0; i < n; i++) {
            t[i] = 4 + (seed * 7 + i * 13) % 250;
        }
        return t;
    }

    /** Two interleaved prompts, every row an output row, then one decode row per sequence. */
    static float[][] run(ForwardBackend f, int[] a, int[] b, boolean nonCausal) {
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
        round.add(b[1], b.length, SEQ_B, true);
        float[][] decoded = f.forwardBatch(round.build());
        float[][] all = Arrays.copyOf(prompt, prompt.length + decoded.length);
        System.arraycopy(decoded, 0, all, prompt.length, decoded.length);
        return all;
    }

    static void assertRowsEqual(String what, float[][] expected, float[][] actual, float tol) {
        assertEquals(what + ": row count", expected.length, actual.length);
        for (int i = 0; i < expected.length; i++) {
            assertArrayEquals(what + ": row " + i, expected[i], actual[i], tol);
        }
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

    @Test
    public void aCausalTwoSequenceBatchMatchesTheCpuBatch() throws IOException {
        Path model = SyntheticModels.deepseek32().writeTempFile();
        float[][][] out = both(model, tokens(1, 12), tokens(2, 12), false);
        assertRowsEqual("GPU paged DSA batch vs CPU", out[0], out[1], TOL);
    }

    /** A 14-token prompt of A, B forked from A's first 12 positions, then one decode row on each. */
    private static float[][] fork(ForwardBackend f) {
        int[] a = tokens(5, 14);
        ForwardBatch.Builder prompt = ForwardBatch.builder(a.length);
        for (int i = 0; i < a.length; i++) {
            prompt.add(a[i], i, SEQ_A, i == a.length - 1);
        }
        f.forwardBatch(prompt.build());
        SequenceMemory seqs = f.memory().sequences();
        assertNull(seqs.copyRefusal());
        seqs.copy(SEQ_A, SEQ_B, 0, 12);
        assertTrue("the fork shares A's pages until it writes", seqs.sharesPages(SEQ_B));
        ForwardBatch.Builder round = ForwardBatch.builder(2);
        round.add(a[3], a.length, SEQ_A, true);
        round.add(a[7], 12, SEQ_B, true);
        return f.forwardBatch(round.build());
    }

    @Test
    public void aForkCopiesTheIndexerKeysOfItsSharedPages() throws IOException {
        Path model = SyntheticModels.deepseek32().writeTempFile();
        float[][] cpu;
        try (GGUFModel m = new GGUFModel(model.toString(), config())) {
            cpu = fork(m.forwardBackend());
        }
        float[][] gpu;
        try (GGUFModelGPU m = new GGUFModelGPU(model.toString(), backend, config())) {
            gpu = fork(m.forwardBackend());
        }
        assertRowsEqual("GPU fork vs CPU fork", cpu, gpu, TOL);
    }

    @Test
    public void aNonCausalTwoSequenceBatchMatchesTheCpuBatch() throws IOException {
        Path model = SyntheticModels.deepseek32().writeTempFile();
        int[] a = tokens(3, 4);
        int[] b = tokens(4, 4);
        float[][][] nonCausal = both(model, a, b, true);
        assertRowsEqual("GPU non-causal paged DSA batch vs CPU", nonCausal[0], nonCausal[1], TOL);
        float[][][] causal = both(model, a, b, false);
        float diff = 0f;
        for (int v = 0; v < causal[1][0].length; v++) {
            diff = Math.max(diff, Math.abs(causal[1][0][v] - nonCausal[1][0][v]));
        }
        assertTrue("sequence A's first row selects its later rows only without the causal mask: " + diff,
                diff > 1e-3f);
    }
}

package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ocl.OclComputeBackend;
import com.eonmux.neuralfree.acceleration.ocl.CLDevice;
import com.eonmux.neuralfree.gguf.BatchInvariance;
import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.GGUFModelGPU;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Properties;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Batch invariance: with {@link BatchInvariance} on, the logits of a sequence are bit-identical whether it runs
 * alone or beside other sequences in a batch of 2, 4 or 8. The runs go through {@code ForwardBackend.forwardBatch}
 * over a paged memory, so the projections, the row norms, the per-row attention and the head all see different
 * row counts while each sequence's own rows must not change. The CPU engine is already per-row deterministic and
 * carries the mode for the attention key-split rule; the GPU loses its tiled GEMM, Q8_1 matvec and batch-wide
 * flash splits, which is what the last test shows the mode turns off. The Mixture-of-Experts and the MLA
 * (deepseek2, absorbed latent cache) tests pin their routed-expert and per-head paths; the MLA sequences hold
 * prompts of different lengths, so the batch's longest sequence cannot coincide with a sequence's own key count
 * at the shared decode step.
 */
public class BatchInvarianceTest {

    /** Prompt rows of every sequence. */
    private static final int PROMPT = 5;
    /** Largest batch the tests build. */
    private static final int MAX_SEQUENCES = 8;

    private HWAcceleration acceleration;
    private ComputeBackend backend;
    private Path model;

    @BeforeClass
    public static void enable() {
        BatchInvariance.setEnabled(true);
    }

    @AfterClass
    public static void disable() {
        BatchInvariance.clearOverride();
    }

    @Before
    public void open() throws IOException {
        model = SyntheticModels.llama().writeTempFile();
    }

    @After
    public void close() {
        if (backend != null) {
            backend.close();
            backend = null;
        }
        if (acceleration != null) {
            acceleration.shutdown();
            acceleration = null;
        }
    }

    /** A paged memory of {@code nCtx} rows and {@code parallel} slots, the tests' fixed small geometry. */
    private static EngineConfig config(int parallel) {
        return EngineConfig.builder()
                .memory(MemoryConfig.builder()
                        .nCtx(256).ubatch(64).parallel(parallel).kvPageSize(16)
                        .typeK(KvDType.F32).typeV(KvDType.F32).build())
                .build();
    }

    /** {@link #config(int)} with the MoE expert cache on, so routed experts take the cached-miss path. */
    private static EngineConfig cachedConfig(int parallel) {
        return EngineConfig.builder()
                .memory(MemoryConfig.builder()
                        .nCtx(256).ubatch(64).parallel(parallel).kvPageSize(16)
                        .typeK(KvDType.F32).typeV(KvDType.F32).build())
                .moeCache(MoeCacheConfig.parse("on"))
                .build();
    }

    /** Deterministic token ids inside the synthetic vocabulary. */
    private static int[] tokens(int seed, int n) {
        int[] t = new int[n];
        for (int i = 0; i < n; i++) {
            t[i] = 4 + (seed * 71 + i * 13) % 250;
        }
        return t;
    }

    /** The first {@code k} sequences' prompts, the sequence {@code s} prompt from {@code tokens(s + 1)}. */
    private static int[][] prompts(int k) {
        int[][] p = new int[k][];
        for (int s = 0; s < k; s++) {
            p[s] = tokens(s + 1, PROMPT);
        }
        return p;
    }

    /** The prefill rows of every sequence, ordered by position then sequence; the last row is its own output. */
    private static ForwardBatch prefillBatch(int[][] prompts) {
        int k = prompts.length;
        int n = prompts[0].length;
        ForwardBatch.Builder b = ForwardBatch.builder(k * n);
        for (int i = 0; i < n; i++) {
            for (int s = 0; s < k; s++) {
                b.add(prompts[s][i], i, s, i == n - 1);
            }
        }
        return b.build();
    }

    /** One decode row per sequence, every row an output row, ordered by sequence. */
    private static ForwardBatch decodeBatch(int[] next, int pos) {
        ForwardBatch.Builder b = ForwardBatch.builder(next.length);
        for (int s = 0; s < next.length; s++) {
            b.add(next[s], pos, s, true);
        }
        return b.build();
    }

    /**
     * The decode rows of sequences {@code prompts[0..k)} after their prefill: the prompt batch, then one batched
     * decode step. The output rows are ordered by sequence, so entry {@code s} is sequence {@code s}.
     */
    private static float[][] decodeRows(ForwardBackend f, int[][] prompts, int[] next) {
        f.reset();
        f.forwardBatch(prefillBatch(prompts));
        float[][] out = f.forwardBatch(decodeBatch(next, prompts[0].length));
        assertEquals("one output row per sequence", prompts.length, out.length);
        return out;
    }

    /** The decode rows of every sequence run alone, in a call of its own. */
    private static float[][] soloDecodeRows(ForwardBackend f, int[][] prompts, int[] next) {
        float[][] out = new float[prompts.length][];
        for (int s = 0; s < prompts.length; s++) {
            out[s] = decodeRows(f, new int[][]{prompts[s]}, new int[]{next[s]})[0];
        }
        f.reset();
        return out;
    }

    /**
     * The routed-expert path: a Mixture-of-Experts model's logits are batch-invariant too, so the mode pins the
     * sorted per-pair expert product and turns the grouped expert GEMM off.
     */
    @Test
    public void gpuMoeLogitsAreBitIdenticalInBatchesOfTwoFourAndEight() throws IOException {
        org.junit.Assume.assumeTrue("OpenCL GPU not available",
                com.eonmux.neuralfree.acceleration.GpuTestUtil.isOpenCLAvailable());
        acceleration = new HWAcceleration();
        CLDevice device = acceleration.getCLDevices().get(0);
        backend = new OclComputeBackend(device);
        Path moe = SyntheticModels.qwen3moe().writeTempFile();
        int[][] all = prompts(MAX_SEQUENCES);
        int[] next = tokens(99, MAX_SEQUENCES);
        try (GGUFModelGPU one = new GGUFModelGPU(moe.toString(), backend, config(MAX_SEQUENCES))) {
            ForwardBackend f = one.forwardBackend();
            float[][] solo = soloDecodeRows(f, all, next);
            for (int k : new int[]{2, 4, 8}) {
                float[][] batched = decodeRows(f, slice(all, k), slice(next, k));
                for (int s = 0; s < k; s++) {
                    assertArrayEquals("MoE sequence " + s + " in a batch of " + k,
                            solo[s], batched[s], 0.0f);
                }
            }
        }
    }

    /**
     * A quantized model takes the fast GEMM lane (the int8 MMQ GEMM on this device) at every row count: the lane
     * covers the reduction dimension in a fixed order, so one row and eight rows must agree bit for bit once the
     * split-K twins are off.
     */
    @Test
    public void gpuQuantizedLogitsAreBitIdenticalWithoutSplitK() throws IOException {
        org.junit.Assume.assumeTrue("OpenCL GPU not available",
                com.eonmux.neuralfree.acceleration.GpuTestUtil.isOpenCLAvailable());
        acceleration = new HWAcceleration();
        CLDevice device = acceleration.getCLDevices().get(0);
        backend = new OclComputeBackend(device);
        Path quantized = SyntheticModels.llama(SyntheticModels.Config.llama().toBuilder()
                .weightType(com.eonmux.neuralfree.gguf.GGMLType.Q8_0).build()).writeTempFile();
        int[][] all = prompts(MAX_SEQUENCES);
        int[] next = tokens(99, MAX_SEQUENCES);
        try (GGUFModelGPU one = new GGUFModelGPU(quantized.toString(), backend, config(MAX_SEQUENCES))) {
            ForwardBackend f = one.forwardBackend();
            float[][] solo = soloDecodeRows(f, all, next);
            for (int k : new int[]{2, 4, 8}) {
                float[][] batched = decodeRows(f, slice(all, k), slice(next, k));
                for (int s = 0; s < k; s++) {
                    assertArrayEquals("quantized sequence " + s + " in a batch of " + k,
                            solo[s], batched[s], 0.0f);
                }
            }
        }
    }

    /**
     * The routed experts on the other GPU path: with the expert cache serving a model that fits, the cached pass
     * pins its sorted per-pair product exactly as the direct pass does, so a hit or a miss answered beside other
     * sequences gives the same logits as alone.
     */
    @Test
    public void gpuMoeWithTheExpertCacheIsBitIdenticalInBatches() throws IOException {
        org.junit.Assume.assumeTrue("OpenCL GPU not available",
                com.eonmux.neuralfree.acceleration.GpuTestUtil.isOpenCLAvailable());
        acceleration = new HWAcceleration();
        CLDevice device = acceleration.getCLDevices().get(0);
        backend = new OclComputeBackend(device);
        Path moe = SyntheticModels.qwen3moe().writeTempFile();
        int[][] all = prompts(MAX_SEQUENCES);
        int[] next = tokens(99, MAX_SEQUENCES);
        try (GGUFModelGPU one = new GGUFModelGPU(moe.toString(), backend, cachedConfig(MAX_SEQUENCES))) {
            assertTrue("the expert cache must serve this model", one.moeExpertCache() != null);
            ForwardBackend f = one.forwardBackend();
            float[][] solo = soloDecodeRows(f, all, next);
            for (int k : new int[]{2, 4, 8}) {
                float[][] batched = decodeRows(f, slice(all, k), slice(next, k));
                for (int s = 0; s < k; s++) {
                    assertArrayEquals("cached MoE sequence " + s + " in a batch of " + k,
                            solo[s], batched[s], 0.0f);
                }
            }
        }
    }

    /** The routed-expert path on the CPU engine: its matvec serves every row count, so the logits do not move. */
    @Test
    public void cpuMoeLogitsAreBitIdenticalInBatchesOfTwoFourAndEight() throws IOException {
        Path moe = SyntheticModels.qwen3moe().writeTempFile();
        int[][] all = prompts(MAX_SEQUENCES);
        int[] next = tokens(99, MAX_SEQUENCES);
        try (GGUFModel one = new GGUFModel(moe.toString(), config(MAX_SEQUENCES))) {
            ForwardBackend f = one.forwardBackend();
            float[][] solo = soloDecodeRows(f, all, next);
            for (int k : new int[]{2, 4, 8}) {
                float[][] batched = decodeRows(f, slice(all, k), slice(next, k));
                for (int s = 0; s < k; s++) {
                    assertArrayEquals("CPU MoE sequence " + s + " in a batch of " + k,
                            solo[s], batched[s], 0.0f);
                }
            }
        }
    }

    @Test
    public void cpuLogitsAreBitIdenticalInBatchesOfTwoFourAndEight() throws IOException {
        int[][] all = prompts(MAX_SEQUENCES);
        int[] next = tokens(99, MAX_SEQUENCES);
        try (GGUFModel one = new GGUFModel(model.toString(), config(MAX_SEQUENCES))) {
            ForwardBackend f = one.forwardBackend();
            float[][] solo = soloDecodeRows(f, all, next);
            for (int k : new int[]{2, 4, 8}) {
                float[][] batched = decodeRows(f, slice(all, k), slice(next, k));
                for (int s = 0; s < k; s++) {
                    assertArrayEquals("CPU sequence " + s + " in a batch of " + k,
                            solo[s], batched[s], 0.0f);
                }
            }
        }
    }

    @Test
    public void gpuLogitsAreBitIdenticalInBatchesOfTwoFourAndEight() throws IOException {
        org.junit.Assume.assumeTrue("OpenCL GPU not available",
                com.eonmux.neuralfree.acceleration.GpuTestUtil.isOpenCLAvailable());
        acceleration = new HWAcceleration();
        CLDevice device = acceleration.getCLDevices().get(0);
        backend = new OclComputeBackend(device);
        int[][] all = prompts(MAX_SEQUENCES);
        int[] next = tokens(99, MAX_SEQUENCES);
        try (GGUFModelGPU one = new GGUFModelGPU(model.toString(), backend, config(MAX_SEQUENCES))) {
            ForwardBackend f = one.forwardBackend();
            float[][] solo = soloDecodeRows(f, all, next);
            for (int k : new int[]{2, 4, 8}) {
                float[][] batched = decodeRows(f, slice(all, k), slice(next, k));
                for (int s = 0; s < k; s++) {
                    assertArrayEquals("GPU sequence " + s + " in a batch of " + k,
                            solo[s], batched[s], 0.0f);
                }
            }
        }
    }

    /**
     * Without the mode the GPU's tiled GEMM takes over at eight rows and the same sequence's logits change: the
     * equality the tests above assert is the mode's doing, not an accident of the model.
     */
    @Test
    public void theGpuBatchChangesTheLogitsWithoutTheMode() throws IOException {
        org.junit.Assume.assumeTrue("OpenCL GPU not available",
                com.eonmux.neuralfree.acceleration.GpuTestUtil.isOpenCLAvailable());
        acceleration = new HWAcceleration();
        CLDevice device = acceleration.getCLDevices().get(0);
        backend = new OclComputeBackend(device);
        int[][] all = prompts(MAX_SEQUENCES);
        int[] next = tokens(99, MAX_SEQUENCES);
        BatchInvariance.setEnabled(false);
        try {
            try (GGUFModelGPU one = new GGUFModelGPU(model.toString(), backend, config(MAX_SEQUENCES))) {
                ForwardBackend f = one.forwardBackend();
                float[] solo = decodeRows(f, slice(all, 1), slice(next, 1))[0];
                float[] batched = decodeRows(f, all, next)[0];
                assertTrue("the tiled GEMM of an eight-row batch must move the row's logits; if it does not,"
                        + " this test no longer proves the mode is what keeps them fixed",
                        differs(solo, batched));
            }
        } finally {
            BatchInvariance.setEnabled(true);
        }
    }

    /**
     * The MLA path: a deepseek2's absorbed latent cache sorts its per-head products by head for a several-token
     * step and sizes a batched paged decode split by the batch's longest sequence; the mode keeps the per-head
     * matvecs and every sequence's own key split at every row count, so its decode logits are bit-identical in
     * batches too. The prompts hold different lengths, so a neighbour changes the batch's longest sequence.
     */
    @Test
    public void gpuMlaLogitsAreBitIdenticalInBatchesOfTwoFourAndEight() throws IOException {
        org.junit.Assume.assumeTrue("OpenCL GPU not available",
                com.eonmux.neuralfree.acceleration.GpuTestUtil.isOpenCLAvailable());
        acceleration = new HWAcceleration();
        CLDevice device = acceleration.getCLDevices().get(0);
        backend = new OclComputeBackend(device);
        Path mla = SyntheticModels.deepseek2().writeTempFile();
        int[][] all = mlaPrompts(MAX_SEQUENCES);
        int[] next = tokens(99, MAX_SEQUENCES);
        try (GGUFModelGPU one = new GGUFModelGPU(mla.toString(), backend, config(MAX_SEQUENCES))) {
            ForwardBackend f = one.forwardBackend();
            float[][] solo = soloMlaDecodeRows(f, all, next);
            for (int k : new int[]{2, 4, 8}) {
                float[][] batched = mlaDecodeRows(f, slice(all, k), slice(next, k));
                for (int s = 0; s < k; s++) {
                    assertArrayEquals("MLA sequence " + s + " in a batch of " + k,
                            solo[s], batched[s], 0.0f);
                }
            }
        }
    }

    /** Prompts of {@code 4 + s} rows for sequence {@code s}: different lengths, still inside the test geometry. */
    private static int[][] mlaPrompts(int k) {
        int[][] p = new int[k][];
        for (int s = 0; s < k; s++) {
            p[s] = tokens(s + 1, 4 + s);
        }
        return p;
    }

    /** Prefill rows of sequences of different lengths, ordered by position then sequence, each last row an output. */
    private static ForwardBatch mlaPrefillBatch(int[][] prompts) {
        int rows = 0;
        int maxLen = 0;
        for (int[] p : prompts) {
            rows += p.length;
            maxLen = Math.max(maxLen, p.length);
        }
        ForwardBatch.Builder b = ForwardBatch.builder(rows);
        for (int i = 0; i < maxLen; i++) {
            for (int s = 0; s < prompts.length; s++) {
                if (i < prompts[s].length) {
                    b.add(prompts[s][i], i, s, i == prompts[s].length - 1);
                }
            }
        }
        return b.build();
    }

    /** One decode row per sequence at its own next position, every row an output row, ordered by sequence. */
    private static ForwardBatch mlaDecodeBatch(int[][] prompts, int[] next) {
        ForwardBatch.Builder b = ForwardBatch.builder(next.length);
        for (int s = 0; s < next.length; s++) {
            b.add(next[s], prompts[s].length, s, true);
        }
        return b.build();
    }

    /** The decode rows of sequences of different lengths after their prefill; entry {@code s} is sequence {@code s}. */
    private static float[][] mlaDecodeRows(ForwardBackend f, int[][] prompts, int[] next) {
        f.reset();
        f.forwardBatch(mlaPrefillBatch(prompts));
        float[][] out = f.forwardBatch(mlaDecodeBatch(prompts, next));
        assertEquals("one output row per sequence", prompts.length, out.length);
        return out;
    }

    /** The decode rows of every sequence run alone, in a call of its own. */
    private static float[][] soloMlaDecodeRows(ForwardBackend f, int[][] prompts, int[] next) {
        float[][] out = new float[prompts.length][];
        for (int s = 0; s < prompts.length; s++) {
            out[s] = mlaDecodeRows(f, new int[][]{prompts[s]}, new int[]{next[s]})[0];
        }
        f.reset();
        return out;
    }

    /** The MLA path on the CPU engine: its attention runs through {@code CpuAttentionCore}, whose pinned key
     *  split keeps a row's reduction independent of the batch, so the decode logits do not move either. */
    @Test
    public void cpuMlaLogitsAreBitIdenticalInBatchesOfTwoFourAndEight() throws IOException {
        Path mla = SyntheticModels.deepseek2().writeTempFile();
        int[][] all = mlaPrompts(MAX_SEQUENCES);
        int[] next = tokens(99, MAX_SEQUENCES);
        try (GGUFModel one = new GGUFModel(mla.toString(), config(MAX_SEQUENCES))) {
            ForwardBackend f = one.forwardBackend();
            float[][] solo = soloMlaDecodeRows(f, all, next);
            for (int k : new int[]{2, 4, 8}) {
                float[][] batched = mlaDecodeRows(f, slice(all, k), slice(next, k));
                for (int s = 0; s < k; s++) {
                    assertArrayEquals("CPU MLA sequence " + s + " in a batch of " + k,
                            solo[s], batched[s], 0.0f);
                }
            }
        }
    }

    @Test
    public void theModeReadsThePropertyAndTheEnvironment() {
        Boolean previous = BatchInvariance.enabled();
        String old = System.getProperty(BatchInvariance.PROPERTY);
        try {
            System.setProperty(BatchInvariance.PROPERTY, "1");
            BatchInvariance.clearOverride();
            assertTrue("vLLM's 1 form", BatchInvariance.enabled());
            System.setProperty(BatchInvariance.PROPERTY, "false");
            BatchInvariance.clearOverride();
            assertEquals(false, BatchInvariance.enabled());
            System.setProperty(BatchInvariance.PROPERTY, "sometimes");
            BatchInvariance.clearOverride();
            try {
                BatchInvariance.enabled();
                throw new AssertionError("a malformed value must be refused");
            } catch (IllegalArgumentException refused) {
                assertTrue(refused.getMessage(), refused.getMessage().contains(BatchInvariance.PROPERTY));
            }
        } finally {
            if (old == null) {
                System.clearProperty(BatchInvariance.PROPERTY);
            } else {
                System.setProperty(BatchInvariance.PROPERTY, old);
            }
            BatchInvariance.setEnabled(previous);
        }
    }

    /** The engine-config boundary refuses a malformed mode value naming the property, before the first projection. */
    @Test
    public void theEngineConfigRefusesAMalformedModeValue() {
        Properties p = new Properties();
        p.setProperty(BatchInvariance.PROPERTY, "sometimes");
        try {
            EngineConfig.fromProperties(p);
            throw new AssertionError("a malformed mode value must be refused at the config boundary");
        } catch (IllegalArgumentException refused) {
            assertTrue(refused.getMessage(), refused.getMessage().contains(BatchInvariance.PROPERTY));
        }
    }

    private static int[][] slice(int[][] values, int k) {
        int[][] out = new int[k][];
        System.arraycopy(values, 0, out, 0, k);
        return out;
    }

    private static int[] slice(int[] values, int k) {
        int[] out = new int[k];
        System.arraycopy(values, 0, out, 0, k);
        return out;
    }

    private static boolean differs(float[] a, float[] b) {
        assertEquals(a.length, b.length);
        for (int i = 0; i < a.length; i++) {
            if (Float.floatToIntBits(a[i]) != Float.floatToIntBits(b[i])) {
                return true;
            }
        }
        return false;
    }
}

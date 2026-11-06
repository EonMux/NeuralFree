package com.eonmux.neuralfree.gguf.ops.attn;

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
import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * CPU/OpenCL parity of the dual-chunk attention on a synthetic Qwen2 model (Qwen2.5-1M) whose context runs past
 * several chunk lengths: the contiguous prefill and decodes and the paged multi-sequence steps (prefill rows crossing
 * a chunk boundary and one-row decode segments) produce the CPU logits on the GPU. The CPU path itself is checked
 * against vLLM's formulas in {@link CpuDualChunkAttentionTest} and {@link CpuDualChunkMixerTest}.
 */
public class GpuDualChunkParityTest {

    private static final int CHUNK = 8;
    private static final int LOCAL = 2;
    private static final int ORIGINAL_MAX = 4;
    private static final int UBATCH = 8;
    private static final float TOL = 4e-3f;

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

    private static EngineConfig config(boolean paged) {
        MemoryConfig.Builder memory = MemoryConfig.builder().nCtx(64).ubatch(UBATCH).typeK(KvDType.F32)
                .typeV(KvDType.F32);
        if (paged) {
            memory = memory.parallel(2);
        }
        return EngineConfig.builder().memory(memory.build()).build();
    }

    private static Path model() throws IOException {
        return SyntheticModels.qwen2DualChunk(SyntheticModels.Config.qwen2(), CHUNK, LOCAL, ORIGINAL_MAX)
                .writeTempFile();
    }

    /** The same model with 64-value heads, the narrowest block of a rotated quantized cache. */
    private static Path rotatedModel() throws IOException {
        SyntheticModels.Config c = SyntheticModels.Config.qwen2().toBuilder()
                .nEmbd(128).nHead(2).nHeadKv(1).headDim(64).nRot(64).nFf(256).build();
        return SyntheticModels.qwen2DualChunk(c, CHUNK, LOCAL, ORIGINAL_MAX).writeTempFile();
    }

    /** A paged Q8_0 cache with the rotation on (the default for a quantized cache). */
    private static EngineConfig rotatedConfig() {
        return EngineConfig.builder().memory(MemoryConfig.builder().nCtx(64).ubatch(UBATCH).typeK(KvDType.Q8_0)
                .typeV(KvDType.Q8_0).parallel(2).build()).build();
    }

    private static int[] tokens(int seed, int n) {
        int[] t = new int[n];
        for (int i = 0; i < n; i++) {
            t[i] = 3 + (seed * 11 + i * 17) % 250;
        }
        return t;
    }

    /** The logits of every row of a 14-token prompt and of two decodes after it. */
    private static float[][] contiguous(ForwardBackend f) {
        f.reset();
        int[] prompt = tokens(1, 14);
        int[] rows = new int[prompt.length];
        for (int i = 0; i < rows.length; i++) {
            rows[i] = i;
        }
        List<float[]> out = new ArrayList<>();
        add(out, f.forwardRows(prompt, 0, rows));
        add(out, f.forwardRows(new int[]{prompt[1]}, prompt.length, new int[]{0}));
        add(out, f.forwardRows(new int[]{prompt[2]}, prompt.length + 1, new int[]{0}));
        return out.toArray(new float[0][]);
    }

    /** The logits of a 4 + 6 + 6 token chunked prefill (the last two steps start inside a chunk) and a decode. */
    private static float[][] chunked(ForwardBackend f) {
        f.reset();
        int[] prompt = tokens(4, 17);
        List<float[]> out = new ArrayList<>();
        int pos = 0;
        for (int len : new int[]{4, 6, 6}) {
            add(out, f.forwardRows(java.util.Arrays.copyOfRange(prompt, pos, pos + len), pos, rows(len)));
            pos += len;
        }
        add(out, f.forwardRows(new int[]{prompt[0]}, pos, new int[]{0}));
        return out.toArray(new float[0][]);
    }

    private static int[] rows(int n) {
        int[] r = new int[n];
        for (int i = 0; i < n; i++) {
            r[i] = i;
        }
        return r;
    }

    /** The logits of a two-sequence paged run: {@code steps[s]} rows per sequence at the running positions, each row
     * of both sequences an output row. */
    private static float[][] paged(ForwardBackend f, int[] steps, int[] seqA, int[] seqB) {
        f.reset();
        List<float[]> out = new ArrayList<>();
        int pos = 0;
        for (int s = 0; s < steps.length; s++) {
            ForwardBatch.Builder batch = ForwardBatch.builder(2 * steps[s]);
            for (int t = 0; t < steps[s]; t++) {
                batch.add(seqA[pos + t], pos + t, 0, true);
                batch.add(seqB[pos + t], pos + t, 1, true);
            }
            add(out, f.forwardBatch(batch.build()));
            pos += steps[s];
        }
        return out.toArray(new float[0][]);
    }

    /** The logits of a one-sequence paged run, a step per entry of {@code steps}. */
    private static float[][] pagedSingle(ForwardBackend f, int[] steps, int[] seq) {
        f.reset();
        List<float[]> out = new ArrayList<>();
        int pos = 0;
        for (int len : steps) {
            ForwardBatch.Builder batch = ForwardBatch.builder(len);
            for (int t = 0; t < len; t++) {
                batch.add(seq[pos + t], pos + t, 0, true);
            }
            add(out, f.forwardBatch(batch.build()));
            pos += len;
        }
        return out.toArray(new float[0][]);
    }

    private static void add(List<float[]> to, float[][] rows) {
        for (float[] row : rows) {
            to.add(row);
        }
    }

    @Test
    public void contiguousPrefillAndDecodeMatchTheCpu() throws IOException {
        Path model = model();
        float[][] cpu;
        try (GGUFModel m = new GGUFModel(model.toString(), config(false))) {
            cpu = contiguous(m.forwardBackend());
        }
        float[][] gpu;
        try (GGUFModelGPU m = new GGUFModelGPU(model.toString(), backend, config(false))) {
            gpu = contiguous(m.forwardBackend());
        }
        assertRows("dual-chunk contiguous GPU vs CPU", cpu, gpu);
    }

    @Test
    public void chunkedPrefillResumingInsideAChunkMatchesTheCpu() throws IOException {
        Path model = model();
        float[][] cpu;
        try (GGUFModel m = new GGUFModel(model.toString(), config(false))) {
            cpu = chunked(m.forwardBackend());
        }
        float[][] gpu;
        try (GGUFModelGPU m = new GGUFModelGPU(model.toString(), backend, config(false))) {
            gpu = chunked(m.forwardBackend());
        }
        assertRows("dual-chunk chunked prefill GPU vs CPU", cpu, gpu);
    }

    @Test
    public void pagedSequencesAcrossChunksMatchTheCpu() throws IOException {
        Path model = model();
        int[] steps = {3, 3, 3, 3, 2, 1};
        int[] a = tokens(2, 15);
        int[] b = tokens(3, 15);
        float[][] cpu;
        try (GGUFModel m = new GGUFModel(model.toString(), config(true))) {
            cpu = paged(m.forwardBackend(), steps, a, b);
        }
        float[][] gpu;
        try (GGUFModelGPU m = new GGUFModelGPU(model.toString(), backend, config(true))) {
            gpu = paged(m.forwardBackend(), steps, a, b);
        }
        assertRows("dual-chunk paged GPU vs CPU", cpu, gpu);
    }

    @Test
    public void pagedPrefillSpanningTwoChunksMatchesTheCpu() throws IOException {
        Path model = model();
        int[] steps = {8, 5, 1};
        int[] seq = tokens(5, 14);
        float[][] cpu;
        try (GGUFModel m = new GGUFModel(model.toString(), config(true))) {
            cpu = pagedSingle(m.forwardBackend(), steps, seq);
        }
        float[][] gpu;
        try (GGUFModelGPU m = new GGUFModelGPU(model.toString(), backend, config(true))) {
            gpu = pagedSingle(m.forwardBackend(), steps, seq);
        }
        assertRows("dual-chunk paged multi-chunk prefill GPU vs CPU", cpu, gpu);
    }

    @Test
    public void aRotatedQuantizedCacheMatchesTheCpu() throws IOException {
        Path model = rotatedModel();
        int[] steps = {3, 3, 3, 3, 2, 1};
        int[] a = tokens(2, 15);
        int[] b = tokens(3, 15);
        float[][] cpu;
        try (GGUFModel m = new GGUFModel(model.toString(), rotatedConfig())) {
            cpu = paged(m.forwardBackend(), steps, a, b);
        }
        float[][] gpu;
        try (GGUFModelGPU m = new GGUFModelGPU(model.toString(), backend, rotatedConfig())) {
            gpu = paged(m.forwardBackend(), steps, a, b);
        }
        assertRows("dual-chunk rotated Q8_0 cache GPU vs CPU", cpu, gpu, 2e-2f);
    }

    private static void assertRows(String what, float[][] expected, float[][] actual) {
        assertRows(what, expected, actual, TOL);
    }

    private static void assertRows(String what, float[][] expected, float[][] actual, float tol) {
        assertEquals(what + ": row count", expected.length, actual.length);
        int compared = 0;
        for (int i = 0; i < expected.length; i++) {
            assertEquals(what + ": row " + i + " width", expected[i].length, actual[i].length);
            for (int v = 0; v < expected[i].length; v++) {
                float e = expected[i][v];
                float a = actual[i][v];
                assertTrue(what + ": row " + i + " value " + v + " CPU " + e + " GPU " + a,
                        Math.abs(e - a) <= tol * Math.max(1f, Math.abs(e)));
                compared++;
            }
        }
        assertTrue(compared > 0);
    }
}

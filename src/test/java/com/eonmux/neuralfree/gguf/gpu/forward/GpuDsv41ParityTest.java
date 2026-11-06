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
import com.eonmux.neuralfree.gguf.testutil.SyntheticRoutedModels;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Path;

import static com.eonmux.neuralfree.gguf.gpu.forward.GpuDsaPagedBatchTest.assertRowsEqual;
import static com.eonmux.neuralfree.gguf.gpu.forward.GpuDsaPagedBatchTest.tokens;

/**
 * CPU/GPU parity of the DeepSeek-V4.1 mixer over the synthetic fixtures: the shared compressed rows of a KV source
 * layer, the index key stored beside each latent, the two-level candidate selection (a source picking blocks, a later
 * source scoring inside them), the Engram extra, the V4.1 head collapse and the paged multi-sequence step all produce
 * the CPU logits. Quantized (Q8_0) compressed rows exercise the combined-row layout and its block rotation.
 */
public class GpuDsv41ParityTest {

    private static final int SEQ_A = 0;
    private static final int SEQ_B = 1;
    private static final float TOL = 3e-3f;

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
        return contiguousConfig(KvDType.F32);
    }

    private static EngineConfig contiguousConfig(KvDType type) {
        return EngineConfig.builder().memory(MemoryConfig.builder().nCtx(64).ubatch(8).typeK(type).typeV(type)
                .build()).build();
    }

    private static EngineConfig pagedConfig() {
        return EngineConfig.builder().memory(MemoryConfig.builder().nCtx(64).ubatch(8).parallel(2)
                .typeK(KvDType.F32).typeV(KvDType.F32).build()).build();
    }

    /** The logits of every row of {@code prompt}, then of a decode at its end, on one backend. */
    private static float[][] contiguousRun(ForwardBackend f, int[] prompt) {
        f.reset();
        int[] rows = new int[prompt.length];
        for (int i = 0; i < rows.length; i++) {
            rows[i] = i;
        }
        float[][] first = f.forwardRows(prompt, 0, rows);
        float[][] decode = f.forwardRows(new int[]{prompt[0]}, prompt.length, new int[]{0});
        float[][] out = new float[first.length + decode.length][];
        System.arraycopy(first, 0, out, 0, first.length);
        System.arraycopy(decode, 0, out, first.length, decode.length);
        return out;
    }

    /** Two interleaved prompts of equal length, every row an output row. */
    private static ForwardBatch interleaved(int[] a, int[] b) {
        ForwardBatch.Builder batch = ForwardBatch.builder(a.length + b.length);
        for (int i = 0; i < a.length; i++) {
            batch.add(a[i], i, SEQ_A, true);
            batch.add(b[i], i, SEQ_B, true);
        }
        return batch.build();
    }

    private float[][] contiguousCpu(Path model, EngineConfig cfg, int[] prompt) throws IOException {
        try (GGUFModel m = new GGUFModel(model.toString(), cfg)) {
            return contiguousRun(m.forwardBackend(), prompt);
        }
    }

    private float[][] contiguousGpu(Path model, EngineConfig cfg, int[] prompt) throws IOException {
        try (GGUFModelGPU m = new GGUFModelGPU(model.toString(), backend, cfg)) {
            return contiguousRun(m.forwardBackend(), prompt);
        }
    }

    private float[][] batchCpu(Path model, EngineConfig cfg, ForwardBatch batch) throws IOException {
        try (GGUFModel m = new GGUFModel(model.toString(), cfg)) {
            m.forwardBackend().reset();
            return m.forwardBackend().forwardBatch(batch);
        }
    }

    private float[][] batchGpu(Path model, EngineConfig cfg, ForwardBatch batch) throws IOException {
        try (GGUFModelGPU m = new GGUFModelGPU(model.toString(), backend, cfg)) {
            m.forwardBackend().reset();
            return m.forwardBackend().forwardBatch(batch);
        }
    }

    @Test
    public void contiguousStepsMatchTheCpu() throws IOException {
        Path model = SyntheticRoutedModels.deepseek41().writeTempFile();
        int[] prompt = tokens(1, 9);
        assertRowsEqual("DeepSeek-V4.1 GPU prefill and decode vs CPU", contiguousCpu(model, config(), prompt),
                contiguousGpu(model, config(), prompt), TOL);
    }

    @Test
    public void theCandidateSelectionMatchesTheCpu() throws IOException {
        Path model = SyntheticRoutedModels.deepseek41Candidate().writeTempFile();
        // 14 tokens leave 7 visible compressed rows, four blocks of two: the two-level selection keeps the top two
        // blocks (and always the newest one), so the row restriction actually drops rows
        int[] prompt = tokens(2, 14);
        assertRowsEqual("DeepSeek-V4.1 candidate selection GPU vs CPU", contiguousCpu(model, config(), prompt),
                contiguousGpu(model, config(), prompt), TOL);
    }

    @Test
    public void pagedSequencesMatchTheCpu() throws IOException {
        Path model = SyntheticRoutedModels.deepseek41().writeTempFile();
        int[] a = tokens(3, 3);
        int[] b = tokens(4, 3);
        assertRowsEqual("DeepSeek-V4.1 paged GPU vs CPU", batchCpu(model, pagedConfig(), interleaved(a, b)),
                batchGpu(model, pagedConfig(), interleaved(a, b)), TOL);
    }

    @Test
    public void pagedCandidateSequencesMatchTheCpu() throws IOException {
        Path model = SyntheticRoutedModels.deepseek41Candidate().writeTempFile();
        // two sequences of twelve tokens, over segments that start mid compression block
        int[] a = tokens(5, 12);
        int[] b = tokens(6, 12);
        assertRowsEqual("DeepSeek-V4.1 paged candidate GPU vs CPU", batchCpu(model, pagedConfig(), interleaved(a, b)),
                batchGpu(model, pagedConfig(), interleaved(a, b)), TOL);
    }

    @Test
    public void quantizedCompressedRowsMatchTheCpu() throws IOException {
        Path model = SyntheticRoutedModels.deepseek41().writeTempFile();
        int[] prompt = tokens(7, 9);
        EngineConfig q8 = contiguousConfig(KvDType.Q8_0);
        assertRowsEqual("DeepSeek-V4.1 Q8_0 compressed rows GPU vs CPU", contiguousCpu(model, q8, prompt),
                contiguousGpu(model, q8, prompt), 5e-3f);
    }
}

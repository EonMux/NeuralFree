package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.demo.tools.CvectorGenerator;
import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.GGUFModelGPU;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.ForwardBatch;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Random;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Control vectors over the paged multi-sequence forward batch on both backends: {@code GGUFModelGPU.forwardBatch}
 * splits a batch into EQUAL-mode ubatches and runs the control-vector extras' batched {@code afterBlock(StepLayout,
 * ScratchSlot)} hook, so its logits match the same batch over the CPU paged path ({@code GGUFModel.forwardBatch},
 * the {@code afterBlock(StepLayout, float[])} hook) with the same vectors — for the prefill batch and for a batched
 * decode step after it. The vectors change both backends' batch logits from the uncontrolled runs, so a batched hook
 * that adds nothing cannot pass.
 */
public class GpuControlVectorBatchTest {

    private static final int N_LAYERS = 3;
    private static final int N_EMBD = 64;
    private static final int N_TOKENS = 12;
    private static final int SEQ_A = 0;
    private static final int SEQ_B = 1;
    private static final float SCALE = 1.5f;
    private static final float TOL = 1e-3f;

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    @Test
    public void thePagedBatchMatchesTheCpuBatchedPathWithControlVectors() throws IOException {
        Path model = SyntheticModels.llama(SyntheticModels.Config.llama().toBuilder().nLayer(N_LAYERS).build())
                .writeTempFile();
        Path vector = vector();
        int[] a = tokens(1);
        int[] b = tokens(2);
        ForwardBatch prefill = interleave(a, b);
        ForwardBatch decode = decode(a[0], b[0], N_TOKENS);

        float[][] cpuBase;
        try (GGUFModel m = new GGUFModel(model.toString(), config().build())) {
            cpuBase = m.forwardBackend().forwardBatch(prefill);
        }
        float[][] gpuBase;
        try (GGUFModelGPU m = new GGUFModelGPU(model.toString(), null, config().build())) {
            gpuBase = m.forwardBackend().forwardBatch(prefill);
        }

        EngineConfig controlled = config().addControlVector(vector.toString(), SCALE).build();
        float[][] cpuPrefill;
        float[][] cpuDecode;
        try (GGUFModel m = new GGUFModel(model.toString(), controlled)) {
            ForwardBackend f = m.forwardBackend();
            cpuPrefill = f.forwardBatch(prefill);
            cpuDecode = f.forwardBatch(decode);
        }
        float[][] gpuPrefill;
        float[][] gpuDecode;
        try (GGUFModelGPU m = new GGUFModelGPU(model.toString(), null, controlled)) {
            ForwardBackend f = m.forwardBackend();
            gpuPrefill = f.forwardBatch(prefill);
            gpuDecode = f.forwardBatch(decode);
        }

        assertEquals("one output row per sequence", cpuPrefill.length, gpuPrefill.length);
        for (int i = 0; i < cpuPrefill.length; i++) {
            assertArrayEquals("prefill output row " + i + " matches the CPU batched run",
                    cpuPrefill[i], gpuPrefill[i], TOL);
            assertTrue("the vector changes the CPU prefill row " + i,
                    maxAbsDiff(cpuBase[i], cpuPrefill[i]) > 100 * TOL);
            assertTrue("the vector changes the GPU prefill row " + i,
                    maxAbsDiff(gpuBase[i], gpuPrefill[i]) > 100 * TOL);
        }
        assertEquals("one output row per decoding sequence", cpuDecode.length, gpuDecode.length);
        for (int i = 0; i < cpuDecode.length; i++) {
            assertArrayEquals("decode output row " + i + " matches the CPU batched run",
                    cpuDecode[i], gpuDecode[i], TOL);
        }
    }

    // ------------------------------------------------------------------ fixtures

    /** Paged memory on both backends: every batch row lands in a page table. */
    private static EngineConfig.Builder config() {
        return EngineConfig.builder().memory(MemoryConfig.builder().nCtx(64).ubatch(8).parallel(2)
                .typeK(KvDType.F32).typeV(KvDType.F32).build());
    }

    /** One control-vector GGUF of {@code N_LAYERS} rows for the synthetic llama. */
    private Path vector() throws IOException {
        float[][] rows = new float[N_LAYERS][];
        for (int il = 0; il < N_LAYERS; il++) {
            Random r = new Random(700 + il);
            float[] row = new float[N_EMBD];
            for (int i = 0; i < N_EMBD; i++) {
                row[i] = 0.2f * (r.nextFloat() - 0.5f);
            }
            rows[il] = row;
        }
        Path out = tmp.getRoot().toPath().resolve("cv-batch.gguf");
        return CvectorGenerator.export(Arrays.asList(rows), N_EMBD, "llama", out, true);
    }

    /** The tokens a sequence runs (deterministic ids inside the synthetic vocab). */
    private static int[] tokens(int seed) {
        int[] t = new int[N_TOKENS];
        for (int i = 0; i < t.length; i++) {
            t[i] = 4 + (seed * 7 + i * 13) % 250;
        }
        return t;
    }

    /** A batch whose rows interleave two prompts; the last row of each sequence is its output row. */
    private static ForwardBatch interleave(int[] a, int[] b) {
        ForwardBatch.Builder batch = ForwardBatch.builder(a.length + b.length);
        for (int i = 0; i < a.length; i++) {
            batch.add(a[i], i, SEQ_A, i == a.length - 1);
            batch.add(b[i], i, SEQ_B, i == b.length - 1);
        }
        return batch.build();
    }

    /** The batched decode step of both sequences at {@code pos}, the token after their prefilled prompts. */
    private static ForwardBatch decode(int tokenA, int tokenB, int pos) {
        return ForwardBatch.builder(2)
                .add(tokenA, pos, SEQ_A, true)
                .add(tokenB, pos, SEQ_B, true)
                .build();
    }

    private static double maxAbsDiff(float[] a, float[] b) {
        assertEquals(a.length, b.length);
        double m = 0;
        for (int i = 0; i < a.length; i++) {
            m = Math.max(m, Math.abs(a[i] - b[i]));
        }
        return m;
    }
}

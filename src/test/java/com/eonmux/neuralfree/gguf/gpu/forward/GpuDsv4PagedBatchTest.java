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
import com.eonmux.neuralfree.gguf.testutil.SyntheticRoutedModels;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static com.eonmux.neuralfree.gguf.gpu.forward.GpuDsaPagedBatchTest.assertRowsEqual;
import static com.eonmux.neuralfree.gguf.gpu.forward.GpuDsaPagedBatchTest.tokens;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The paged multi-sequence steps of the GPU DeepSeek-V4 mixer over the synthetic deepseek4 (block 0 a raw window,
 * block 1 a CSA layer whose compressor and lightning-indexer compressor fold every fourth position into the
 * sequence's own compressed streams) equal the CPU paged batch: two interleaved prompts of unequal lengths (their
 * segments start mid compression block in later ubatches), the decode rounds after them, a tail removal that restores
 * one sequence's rollback snapshot planes, and a whole-sequence copy whose streams continue like their source. A
 * non-causal step is refused by name.
 */
public class GpuDsv4PagedBatchTest {

    private static final int SEQ_A = 0;
    private static final int SEQ_B = 1;
    private static final float TOL = 2e-3f;
    /** Compressor rollback snapshots per sequence ({@code n_rs_seq}), so the tail removal can restore. */
    private static final int SNAPSHOTS = 3;

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
                .rollbackSnapshots(SNAPSHOTS).typeK(KvDType.F32).typeV(KvDType.F32).build()).build();
    }

    /** One batch of every sequence's rows from its next position, every row an output row. */
    private static float[][] step(ForwardBackend f, int[][] rows, int[] seqs, int[] pos0) {
        int total = 0;
        for (int[] r : rows) {
            total += r.length;
        }
        ForwardBatch.Builder batch = ForwardBatch.builder(total);
        int longest = 0;
        for (int[] r : rows) {
            longest = Math.max(longest, r.length);
        }
        for (int i = 0; i < longest; i++) {
            for (int s = 0; s < rows.length; s++) {
                if (i < rows[s].length) {
                    batch.add(rows[s][i], pos0[s] + i, seqs[s], true);
                }
            }
        }
        return f.forwardBatch(batch.build());
    }

    /**
     * The scenario on one backend: interleaved prompts of 11 and 6 tokens, a decode round, two dropped rows of A
     * removed and replaced, A copied over B, then one more round on both.
     */
    private static float[][] scenario(ForwardBackend f) {
        int[] a = tokens(1, 11);
        int[] b = tokens(2, 6);
        List<float[][]> out = new ArrayList<>();
        out.add(step(f, new int[][]{a, b}, new int[]{SEQ_A, SEQ_B}, new int[]{0, 0}));
        out.add(step(f, new int[][]{{a[3]}, {b[2]}}, new int[]{SEQ_A, SEQ_B}, new int[]{11, 6}));
        out.add(step(f, new int[][]{{a[5], a[6]}}, new int[]{SEQ_A}, new int[]{12}));
        SequenceMemory seqs = f.memory().sequences();
        assertNull("the last step's rows are within the rollback snapshots", seqs.removeRefusal(SEQ_A, 12, -1));
        seqs.remove(SEQ_A, 12, -1);
        out.add(step(f, new int[][]{{a[7], a[8]}}, new int[]{SEQ_A}, new int[]{12}));
        seqs.remove(SEQ_B, 0, -1);
        seqs.copy(SEQ_A, SEQ_B, 0, -1);
        out.add(step(f, new int[][]{{a[9]}, {a[9]}}, new int[]{SEQ_A, SEQ_B}, new int[]{14, 14}));
        List<float[]> rows = new ArrayList<>();
        for (float[][] o : out) {
            for (float[] r : o) {
                rows.add(r);
            }
        }
        return rows.toArray(new float[0][]);
    }

    @Test
    public void pagedStepsAndSequenceOperationsMatchTheCpuBatch() throws IOException {
        Path model = SyntheticRoutedModels.deepseek4().writeTempFile();
        float[][] cpu;
        try (GGUFModel m = new GGUFModel(model.toString(), config())) {
            cpu = scenario(m.forwardBackend());
        }
        float[][] gpu;
        try (GGUFModelGPU m = new GGUFModelGPU(model.toString(), backend, config())) {
            gpu = scenario(m.forwardBackend());
        }
        assertRowsEqual("GPU paged DeepSeek-V4 batch vs CPU", cpu, gpu, TOL);
        int last = gpu.length;
        assertRowsEqual("the copied sequence continues like its source", new float[][]{gpu[last - 2]},
                new float[][]{gpu[last - 1]}, TOL);
    }

    @Test
    public void aNonCausalStepIsRefusedByName() throws IOException {
        Path model = SyntheticRoutedModels.deepseek4().writeTempFile();
        try (GGUFModelGPU m = new GGUFModelGPU(model.toString(), backend, config())) {
            int[] a = tokens(3, 4);
            ForwardBatch.Builder batch = ForwardBatch.builder(a.length);
            for (int i = 0; i < a.length; i++) {
                batch.add(a[i], i, SEQ_A, i == a.length - 1);
            }
            batch.nonCausal();
            UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class,
                    () -> m.forwardBackend().forwardBatch(batch.build()));
            assertTrue(e.getMessage(), e.getMessage().contains("non-causal"));
        }
    }
}

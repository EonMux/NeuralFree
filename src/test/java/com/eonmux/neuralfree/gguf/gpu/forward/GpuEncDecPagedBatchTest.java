package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ocl.OclComputeBackend;
import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.GGUFModelGPU;
import com.eonmux.neuralfree.gguf.TokenCallback;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.engine.ForwardBatch;
import com.eonmux.neuralfree.gguf.engine.GenerationLoop;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.engine.SequenceEncoder;
import com.eonmux.neuralfree.gguf.engine.batch.BatchEngine;
import com.eonmux.neuralfree.gguf.engine.batch.BatchSequence;
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

import static com.eonmux.neuralfree.gguf.gpu.forward.GpuDsaPagedBatchTest.assertRowsEqual;
import static com.eonmux.neuralfree.gguf.gpu.forward.GpuDsaPagedBatchTest.tokens;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Multi-sequence decoder batches of the GPU encoder-decoder backend over the synthetic T5 (relative-position bias,
 * cross-attention in every decoder layer) on paged memory: every sequence encodes its own source into its own cross
 * caches ({@link SequenceEncoder#encodeSequence}); interleaved decoder rows of two sequences, the decode round after
 * them, and a sequence re-encoded and restarted beside the other equal the CPU paged batch. The batch engine runs
 * the GPU encoder-decoder, every request finishing with the tokens the GPU generation loop produces for it alone.
 */
public class GpuEncDecPagedBatchTest {

    private static final int SEQ_A = 0;
    private static final int SEQ_B = 1;
    private static final float TOL = 2e-3f;
    private static final int START = SyntheticModels.T5_DECODER_START;

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
        MemoryConfig.Builder mc = MemoryConfig.builder().nCtx(64).ubatch(16).typeK(KvDType.F32).typeV(KvDType.F32);
        if (paged) {
            mc = mc.parallel(2);
        }
        return EngineConfig.builder().memory(mc.build()).build();
    }

    /** One batch of each listed sequence's decoder rows from {@code pos0}, every row an output row. */
    private static float[][] step(ForwardBackend f, int[][] rows, int[] seqs, int[] pos0) {
        int total = 0;
        int longest = 0;
        for (int[] r : rows) {
            total += r.length;
            longest = Math.max(longest, r.length);
        }
        ForwardBatch.Builder batch = ForwardBatch.builder(total);
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
     * Two sources encoded, the start token and four decoder rows of A beside the start token and two rows of B, a
     * decode round, then B re-encoded from a new source and restarted while A decodes on.
     */
    private static float[][] scenario(ForwardBackend f) {
        SequenceEncoder enc = f.sequenceEncoder();
        assertNotNull("paged memory keeps one encoding per sequence", enc);
        enc.encodeSequence(SEQ_A, tokens(1, 7));
        enc.encodeSequence(SEQ_B, tokens(2, 12));
        int[] a = tokens(3, 6);
        int[] b = tokens(4, 4);
        List<float[][]> out = new ArrayList<>();
        out.add(step(f, new int[][]{{START, a[0], a[1], a[2], a[3]}, {START, b[0], b[1]}},
                new int[]{SEQ_A, SEQ_B}, new int[]{0, 0}));
        out.add(step(f, new int[][]{{a[4]}, {b[2]}}, new int[]{SEQ_A, SEQ_B}, new int[]{5, 3}));
        f.memory().sequences().remove(SEQ_B, 0, -1);
        enc.encodeSequence(SEQ_B, tokens(5, 3));
        out.add(step(f, new int[][]{{a[5]}, {START, b[3]}}, new int[]{SEQ_A, SEQ_B}, new int[]{6, 0}));
        List<float[]> rows = new ArrayList<>();
        for (float[][] o : out) {
            for (float[] r : o) {
                rows.add(r);
            }
        }
        return rows.toArray(new float[0][]);
    }

    @Test
    public void pagedDecoderBatchesMatchTheCpuBatch() throws IOException {
        Path model = SyntheticModels.t5().writeTempFile();
        float[][] cpu;
        try (GGUFModel m = new GGUFModel(model.toString(), config(true))) {
            cpu = scenario(m.forwardBackend());
        }
        float[][] gpu;
        try (GGUFModelGPU m = new GGUFModelGPU(model.toString(), backend, config(true))) {
            assertEquals(2, m.forwardBackend().maxSequences());
            assertNull(m.forwardBackend().batchRefusal());
            gpu = scenario(m.forwardBackend());
        }
        assertRowsEqual("GPU paged encoder-decoder batch vs CPU", cpu, gpu, TOL);
    }

    @Test
    public void aContiguousMemoryKeepsOneEncodingAndUnencodedRowsRefuse() throws IOException {
        Path model = SyntheticModels.t5().writeTempFile();
        try (GGUFModelGPU one = new GGUFModelGPU(model.toString(), backend, config(false))) {
            assertNull("a contiguous memory keeps one encoding", one.forwardBackend().sequenceEncoder());
            assertNotNull(one.forwardBackend().batchRefusal());
        }
        try (GGUFModelGPU two = new GGUFModelGPU(model.toString(), backend, config(true))) {
            ForwardBackend f = two.forwardBackend();
            f.sequenceEncoder().encodeSequence(SEQ_A, tokens(1, 4));
            IllegalStateException e = assertThrows(IllegalStateException.class,
                    () -> step(f, new int[][]{{START}}, new int[]{SEQ_B}, new int[]{0}));
            assertTrue(e.getMessage(), e.getMessage().contains("encodeSequence"));
        }
    }

    @Test
    public void theBatchEngineRunsTheGpuEncoderDecoder() throws IOException {
        Path model = SyntheticModels.t5().writeTempFile();
        int[] srcA = tokens(6, 7);
        int[] srcB = tokens(7, 12);
        GenerationRequest request = GenerationRequest.builder().maxNewTokens(5)
                .sampler(SamplerConfig.greedyDecoding()).build();
        GenerationResult refA;
        GenerationResult refB;
        EngineConfig single = config(false);
        try (GGUFModelGPU one = new GGUFModelGPU(model.toString(), backend, single)) {
            GenerationLoop loop = GenerationLoop.forModel(one, single);
            refA = loop.generate(srcA, request, (TokenCallback) null);
            refB = loop.generate(srcB, request, (TokenCallback) null);
        }
        EngineConfig paged = config(true);
        try (GGUFModelGPU two = new GGUFModelGPU(model.toString(), backend, paged);
             BatchEngine engine = BatchEngine.forModel(two, paged)) {
            BatchSequence a = engine.submit(srcA, request, (TokenCallback) null);
            BatchSequence b = engine.submit(srcB, request, (TokenCallback) null);
            engine.drain();
            assertEquals(BatchSequence.State.DONE, a.state());
            assertEquals(BatchSequence.State.DONE, b.state());
            assertArrayEquals("sequence A", refA.tokens(), a.result().tokens());
            assertArrayEquals("sequence B", refB.tokens(), b.result().tokens());
        }
    }
}

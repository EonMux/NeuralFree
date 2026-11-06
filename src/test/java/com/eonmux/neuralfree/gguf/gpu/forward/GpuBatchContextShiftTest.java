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
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.engine.StopReason;
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

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * The context shift on the GPU: the per-sequence shift over the paged device memory ({@code
 * ForwardBackend#shiftSequence}) moves the rows, re-rotates the moved keys by the move's per-row deltas
 * ({@code LLM_KV_RopeShift_*}, for the F16 and the quantized Q8_0 K with its Hadamard rotation) and keeps
 * generation identical to the CPU backend's shift over the same batches (logits within tolerance). The batch
 * engine shifts over the GPU memory like it does over the heap: a full context keeps decoding, reports
 * {@code truncated} and, with the shift off, still stops with {@code CONTEXT_FULL}. Skipped without OpenCL.
 */
public class GpuBatchContextShiftTest {

    private static final int SEQ = 0;
    private static final int N_CTX = 32;
    private static final int UBATCH = 8;
    private static final int PAGE = 8;
    private static final int PARALLEL = 2;
    /** The shift the equivalence runs take: keeps 2, drops 10 of the 22 stored positions. */
    private static final int N_KEEP = 2;
    private static final int N_DISCARD = 10;
    private static final float TOL = 2e-3f;
    private static final float QUANT_TOL = 6e-3f;

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

    private static EngineConfig config(KvDType typeK) {
        MemoryConfig mc = MemoryConfig.builder().nCtx(N_CTX).ubatch(UBATCH).parallel(PARALLEL).kvPageSize(PAGE)
                .typeK(typeK).typeV(KvDType.F32).build();
        return EngineConfig.builder().memory(mc).build();
    }

    private static EngineConfig batchConfig(KvDType typeK, boolean contextShift) {
        MemoryConfig mc = MemoryConfig.builder().nCtx(24).ubatch(UBATCH).parallel(PARALLEL).kvPageSize(PAGE)
                .typeK(typeK).typeV(KvDType.F32).contextShift(contextShift).build();
        return EngineConfig.builder().memory(mc).build();
    }

    private static int[] tokens(int seed, int n) {
        int[] t = new int[n];
        for (int i = 0; i < n; i++) {
            t[i] = 4 + (seed * 7 + i * 13) % 250;
        }
        return t;
    }

    private static ForwardBatch single(int[] t, int seq, int pos0) {
        ForwardBatch.Builder batch = ForwardBatch.builder(t.length);
        for (int i = 0; i < t.length; i++) {
            batch.add(t[i], pos0 + i, seq, i == t.length - 1);
        }
        return batch.build();
    }

    private static ForwardBatch decode(int token, int pos, int seq) {
        return ForwardBatch.builder(1).add(token, pos, seq, true).build();
    }

    /**
     * Prefills the prompt, decodes two rows, shifts (keeps 2, drops 10 of the 22 stored positions) and decodes two
     * more at the moved positions: the four logits rows, the last two computed over the shifted context.
     */
    private static float[][] shiftRun(ForwardBackend f, int[] prompt, int[] decoded) {
        f.forwardBatch(single(prompt, SEQ, 0));
        int pos = prompt.length;
        float[][] rows = new float[decoded.length][];
        for (int i = 0; i < decoded.length; i++) {
            rows[i] = f.forwardBatch(decode(decoded[i], pos, SEQ))[0];
            pos++;
            if (i == 1) {
                f.shiftSequence(SEQ, N_KEEP, N_DISCARD);
                pos = f.memory().sequences().posMax(SEQ) + 1;
            }
        }
        return rows;
    }

    /** The GPU logits of the shifted run equal the CPU backend's over the same batches and shift. */
    private void shiftedGpuRowsEqualTheCpuRows(Path model, EngineConfig cfg, float tol) throws IOException {
        int[] prompt = tokens(1, 20);
        int[] decoded = tokens(2, 4);
        float[][][] out = new float[2][][];
        try (GGUFModel cpu = new GGUFModel(model.toString(), cfg)) {
            out[0] = shiftRun(cpu.forwardBackend(), prompt, decoded);
        }
        try (GGUFModelGPU gpu = new GGUFModelGPU(model.toString(), backend, cfg)) {
            out[1] = shiftRun(gpu.forwardBackend(), prompt, decoded);
        }
        assertRowsEqual("GPU shifted decode rows vs CPU", out[0], out[1], tol);
    }

    private static void assertRowsEqual(String what, float[][] expected, float[][] actual, float tol) {
        assertEquals(what + ": row count", expected.length, actual.length);
        for (int i = 0; i < expected.length; i++) {
            assertArrayEquals(what + ": row " + i, expected[i], actual[i], tol);
        }
    }

    @Test
    public void theF16KShiftKeepsTheGpuLogitsAtTheCpuRows() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        shiftedGpuRowsEqualTheCpuRows(model, config(KvDType.F16), TOL);
    }

    /** The quantized K shifts through the kernel's dequantize, rotate and requantize path (with the K rotation). */
    @Test
    public void theQ80KShiftKeepsTheGpuLogitsAtTheCpuRows() throws IOException {
        Path model = SyntheticModels.llama(
                SyntheticModels.Config.llama().toBuilder().headDim(32).nRot(32).build()).writeTempFile();
        shiftedGpuRowsEqualTheCpuRows(model, config(KvDType.Q8_0), QUANT_TOL);
    }

    /** The shift moves the context: the same decode over the unshifted memory computes other rows. */
    @Test
    public void theShiftChangesTheContextTheNextDecodeReads() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        int[] prompt = tokens(1, 20);
        int[] decoded = tokens(2, 3);
        float[][] unshifted;
        float[][] shifted;
        try (GGUFModelGPU gpu = new GGUFModelGPU(model.toString(), backend, config(KvDType.F16))) {
            ForwardBackend f = gpu.forwardBackend();
            f.forwardBatch(single(prompt, SEQ, 0));
            f.forwardBatch(decode(decoded[0], 20, SEQ));
            f.forwardBatch(decode(decoded[1], 21, SEQ));
            f.shiftSequence(SEQ, N_KEEP, N_DISCARD);
            int next = f.memory().sequences().posMax(SEQ) + 1;
            assertEquals("the survivors moved down by the discard count", 12, next);
            shifted = f.forwardBatch(decode(decoded[2], next, SEQ));
        }
        try (GGUFModelGPU gpu = new GGUFModelGPU(model.toString(), backend, config(KvDType.F16))) {
            ForwardBackend f = gpu.forwardBackend();
            f.forwardBatch(single(prompt, SEQ, 0));
            f.forwardBatch(decode(decoded[0], 20, SEQ));
            f.forwardBatch(decode(decoded[1], 21, SEQ));
            unshifted = f.forwardBatch(decode(decoded[2], 22, SEQ));
        }
        float diff = 0f;
        for (int v = 0; v < shifted[0].length; v++) {
            diff = Math.max(diff, Math.abs(unshifted[0][v] - shifted[0][v]));
        }
        assertTrue("the shifted decode reads another context than the plain continuation: " + diff, diff > 1e-4f);
    }

    // ------------------------------------------------------------------ the batch engine over the GPU memory

    private static GenerationRequest request(int maxNewTokens) {
        return GenerationRequest.builder().maxNewTokens(maxNewTokens)
                .sampler(SamplerConfig.greedyDecoding()).build();
    }

    /** A full context shifts and keeps decoding in its slot, reporting {@code truncated}. */
    private int[] shiftedRun(Path model, EngineConfig cfg, int[] prompt, GenerationRequest request)
            throws IOException {
        try (GGUFModelGPU m = new GGUFModelGPU(model.toString(), backend, cfg);
             BatchEngine engine = BatchEngine.forModel(m, cfg)) {
            BatchSequence a = engine.submit(prompt, request, (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.drain();
            assertTrue("the run completed: " + a.error(), a.state() == BatchSequence.State.DONE);
            assertTrue("the shifted sequence reports truncated", a.truncated());
            assertTrue("the engine shifted: " + engine.stats().contextShifts(),
                    engine.stats().contextShifts() >= 1);
            assertEquals("generation ran past the context on every shift", 30, a.result().tokenCount());
            return a.result().tokens();
        }
    }

    @Test
    public void theBatchEngineShiftsOverTheGpuMemory() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig cfg = batchConfig(KvDType.F16, true);
        int[] prompt = tokens(3, 12);
        GenerationRequest request = request(30);
        int[] first = shiftedRun(model, cfg, prompt, request);
        int[] second = shiftedRun(model, cfg, prompt, request);
        assertArrayEquals("the run is deterministic", first, second);
    }

    @Test
    public void withTheShiftOffTheGpuMemoryStillStopsWithContextFull() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig cfg = batchConfig(KvDType.F16, false);
        try (GGUFModelGPU m = new GGUFModelGPU(model.toString(), backend, cfg);
             BatchEngine engine = BatchEngine.forModel(m, cfg)) {
            BatchSequence a = engine.submit(tokens(3, 12), request(30),
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.drain();
            assertEquals(BatchSequence.State.DONE, a.state());
            assertFalse(a.truncated());
            assertEquals(StopReason.CONTEXT_FULL, a.result().stopReason());
            assertEquals(13, a.result().tokenCount());
            assertEquals(0, engine.stats().contextShifts());
        }
    }

    // ------------------------------------------------------------------ the single-sequence loop over the GPU memory

    private static EngineConfig loopConfig(KvDType typeK) {
        MemoryConfig mc = MemoryConfig.builder().nCtx(24).ubatch(UBATCH).typeK(typeK).typeV(KvDType.F32)
                .contextShift(true).build();
        return EngineConfig.builder().memory(mc).build();
    }

    /**
     * The single-sequence loop shifts over the contiguous device memory with the same mechanism: the generation
     * crosses the context, keeps decoding to its budget and stops where the CPU loop stops.
     */
    @Test
    public void theSingleSequenceLoopShiftsOverTheContiguousGpuMemory() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        int[] prompt = tokens(4, 12);
        GenerationRequest request = request(30);
        int[] loopTokens;
        StopReason loopReason;
        try (GGUFModelGPU m = new GGUFModelGPU(model.toString(), backend, loopConfig(KvDType.F16))) {
            GenerationResult result = m.generate(prompt, request, null);
            assertEquals("the loop shifted and ran to its budget", 30, result.tokenCount());
            loopTokens = result.tokens();
            loopReason = result.stopReason();
            int[] again = m.generate(prompt, request, null).tokens();
            assertArrayEquals("the loop run is deterministic", loopTokens, again);
        }
        try (GGUFModel cpu = new GGUFModel(model.toString(), loopConfig(KvDType.F16))) {
            GenerationResult result = cpu.generate(prompt, request, null);
            assertEquals("the CPU loop shifts the same way", 30, result.tokenCount());
            assertEquals(loopReason, result.stopReason());
        }
    }
}

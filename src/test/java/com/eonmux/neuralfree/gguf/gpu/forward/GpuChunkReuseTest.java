package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ocl.OclComputeBackend;
import com.eonmux.neuralfree.acceleration.ocl.CLDevice;
import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.GGUFModelGPU;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.engine.ForwardBatch;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.memory.gpu.DeviceModelMemory;
import com.eonmux.neuralfree.gguf.memory.paged.PagedSequences;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

/**
 * The KV chunk reuse on the OpenCL backend ({@link ForwardBackend#reuseChunk}): the device memory copies the
 * source pages' K/V to the claimant's pages and, for a shifted run, re-rotates the copied keys' positions by the
 * run's delta on the device. An unshifted run whose donor context is the prompt's own context computes like the
 * same prompt's full prefill (the CPU reference) within the GPU arithmetic drift; a shifted run computes like the
 * CPU backend's own chunk reuse of the same geometry — the re-appended rows carry the donor's prefix in their
 * deeper-layer state (the accuracy trade llama.cpp's opt-in {@code n_cache_reuse} accepts), so the comparison
 * pins the device copy and rotation against the CPU kernel rather than a full prefill. The synthetic two-layer
 * llama runs paged with pages of 8 over two sequences.
 */
public class GpuChunkReuseTest {

    private static final int PAGE = 8;
    private static final int SEQ_A = 0;
    private static final int SEQ_B = 1;
    private static final float TOL = 1e-3f;

    private HWAcceleration acceleration;
    private ComputeBackend backend;

    @Before
    public void setUp() {
        assumeTrue("OpenCL GPU not available",
                com.eonmux.neuralfree.acceleration.GpuTestUtil.isOpenCLAvailable());
        acceleration = new HWAcceleration();
        CLDevice device = acceleration.getCLDevices().get(0);
        backend = new OclComputeBackend(device);
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

    private static EngineConfig paged() {
        return EngineConfig.builder().memory(MemoryConfig.builder()
                .nCtx(64).ubatch(PAGE).typeK(KvDType.F32).typeV(KvDType.F32)
                .parallel(2).kvPageSize(PAGE).build()).build();
    }

    private static EngineConfig cpuReference() {
        return EngineConfig.builder().memory(MemoryConfig.builder()
                .nCtx(64).ubatch(PAGE).typeK(KvDType.F32).typeV(KvDType.F32).build()).build();
    }

    /** The cached prompt: 24 distinct tokens (three pages). */
    private static int[] cached() {
        int[] t = new int[3 * PAGE];
        for (int i = 0; i < t.length; i++) {
            t[i] = 4 + (i * 13 + 5) % 250;
        }
        return t;
    }

    /** A batch over {@code prompt[from..to)} at its own positions for {@code seq}, the last row its output row. */
    private static ForwardBatch batch(int[] prompt, int from, int to, int seq) {
        ForwardBatch.Builder b = ForwardBatch.builder(to - from);
        for (int i = from; i < to; i++) {
            b.add(prompt[i], i, seq, i == to - 1);
        }
        return b.build();
    }

    private static float[] fullPrefillReference(Path model, int[] prompt) throws IOException {
        try (GGUFModel one = new GGUFModel(model.toString(), cpuReference())) {
            return one.forwardBackend().forwardRows(prompt, 0, new int[]{prompt.length - 1})[0];
        }
    }

    @Test
    public void anUnshiftedReAppendMatchesTheFullPrefill() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        int[] cache = cached();
        int[] prompt = new int[22];
        System.arraycopy(cache, 0, prompt, 0, 12);
        System.arraycopy(cache, 12, prompt, 12, 8);
        prompt[20] = 200;
        prompt[21] = 210;
        float[] reference = fullPrefillReference(model, prompt);

        try (GGUFModelGPU two = new GGUFModelGPU(model.toString(), backend, paged())) {
            ForwardBackend f = two.forwardBackend();
            PagedSequences mem = paged(f);
            f.forwardBatch(batch(cache, 0, cache.length, SEQ_A));        // the cache side: three pages
            int[] mid = mem.blockPages(SEQ_A, 1, PAGE);
            int[] top = mem.blockPages(SEQ_A, 2, PAGE);
            f.forwardBatch(batch(prompt, 0, 12, SEQ_B));                 // the claimant's own first 12 rows
            f.reuseChunk(SEQ_B, new int[][]{mid, top}, 4, 12, 8, 0,
                    Arrays.copyOfRange(prompt, 12, 20), -1);
            float[] out = f.forwardBatch(batch(prompt, 20, prompt.length, SEQ_B))[0];
            assertArrayEquals("the re-appended rows compute like a full prefill", reference, out, TOL);
            assertTrue(mem.pendingCopies().isEmpty());
        }
    }

    /**
     * A shifted re-append: the claimant prefills rows [0, 12) and re-appends the cache's rows [16, 24) at rows
     * [12, 20), a shift of -4 whose keys re-rotate on the device. The device's logits equal the CPU backend's own
     * chunk reuse of the same geometry within the GPU arithmetic drift.
     */
    @Test
    public void aShiftedReAppendMatchesTheCpuChunkReuse() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        int[] cache = cached();
        int[] prompt = new int[21];
        System.arraycopy(cache, 0, prompt, 0, 12);
        System.arraycopy(cache, 16, prompt, 12, 8);
        prompt[20] = 220;
        float[] reference;
        try (GGUFModel cpu = new GGUFModel(model.toString(), paged())) {
            ForwardBackend cf = cpu.forwardBackend();
            PagedSequences cm = cpuPaged(cf);
            cf.forwardBatch(batch(cache, 0, cache.length, SEQ_A));
            int[] ctop = cm.blockPages(SEQ_A, 2, PAGE);
            cf.forwardBatch(batch(prompt, 0, 12, SEQ_B));
            cf.reuseChunk(SEQ_B, new int[][]{ctop}, 0, 12, 8, -4,
                    Arrays.copyOfRange(prompt, 12, 20), -1);
            reference = cf.forwardBatch(batch(prompt, 20, prompt.length, SEQ_B))[0];
        }

        try (GGUFModelGPU two = new GGUFModelGPU(model.toString(), backend, paged())) {
            ForwardBackend f = two.forwardBackend();
            PagedSequences mem = paged(f);
            f.forwardBatch(batch(cache, 0, cache.length, SEQ_A));
            int[] top = mem.blockPages(SEQ_A, 2, PAGE);
            f.forwardBatch(batch(prompt, 0, 12, SEQ_B));
            // rows [12, 20) re-append source rows [16, 24): every copied key re-rotates by -4 on the device
            f.reuseChunk(SEQ_B, new int[][]{top}, 0, 12, 8, -4,
                    Arrays.copyOfRange(prompt, 12, 20), -1);
            float[] out = f.forwardBatch(batch(prompt, 20, prompt.length, SEQ_B))[0];
            assertArrayEquals("the device re-append computes like the CPU chunk reuse", reference, out, TOL);
            assertTrue(mem.pendingCopies().isEmpty());
        }
    }

    private static PagedSequences paged(ForwardBackend f) {
        PagedSequences mem = ((DeviceModelMemory) f.memory()).pagedSequences();
        assertNotNull("the test needs a paged device memory", mem);
        return mem;
    }

    private static PagedSequences cpuPaged(ForwardBackend f) {
        PagedSequences mem = ((com.eonmux.neuralfree.gguf.memory.cpu.HeapModelMemory) f.memory()).pagedSequences();
        assertNotNull("the test needs a paged heap memory", mem);
        return mem;
    }
}

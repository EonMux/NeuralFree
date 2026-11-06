package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ocl.OclComputeBackend;
import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.GGUFModelGPU;
import com.eonmux.neuralfree.gguf.TokenCallback;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.GenerationLoop;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
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
import java.util.Arrays;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/**
 * Sliding-window prefix reuse on the GPU paged memory: the batch engine over a gemma2 whose 8-row window the
 * prompts outgrow resumes a conversation's next turn at the retirement checkpoint (the window's pages adopted beside
 * the full-attention blocks, its shared last page copied on the device before the first write) and a repeated
 * prompt at the checkpoint before its last page, computing the tokens of a fresh engine and of the contiguous CPU
 * loop.
 */
public class GpuBatchContextCheckpointTest {

    private static final int PAGE = 4;
    private static final int N_NEW = 6;

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

    private static Path gemma2() throws IOException {
        return SyntheticModels.gemma2(SyntheticModels.Config.gemma2().toBuilder().slidingWindow(8).build())
                .writeTempFile();
    }

    private static EngineConfig paged(boolean reuse) {
        return EngineConfig.builder().memory(MemoryConfig.builder().nCtx(128).ubatch(8).kvPaged(true)
                .kvPageSize(PAGE).parallel(2).typeK(KvDType.F32).typeV(KvDType.F32).prefixReuse(reuse).build())
                .build();
    }

    private static GenerationRequest request() {
        return GenerationRequest.builder().maxNewTokens(N_NEW).ignoreEos(true)
                .sampler(SamplerConfig.greedyDecoding()).build();
    }

    private static int[] prompt(int seed, int n) {
        int[] t = new int[n];
        for (int i = 0; i < n; i++) {
            t[i] = 4 + (seed * 37 + i * 11 + (i * i) % 13) % 300;
        }
        return t;
    }

    private static BatchSequence submit(BatchEngine engine, int[] prompt) {
        BatchSequence seq = engine.submit(prompt, request(), (TokenCallback) null);
        engine.drain();
        assertEquals("the sequence finished: " + seq.error(), BatchSequence.State.DONE, seq.state());
        return seq;
    }

    private GenerationResult freshGpu(Path model, int[] prompt) throws IOException {
        try (GGUFModelGPU m = new GGUFModelGPU(model.toString(), backend, paged(false));
             BatchEngine engine = BatchEngine.forModel(m, paged(false))) {
            return submit(engine, prompt).result();
        }
    }

    private static GenerationResult contiguousCpu(Path model, int[] prompt) throws IOException {
        EngineConfig cfg = EngineConfig.builder().memory(MemoryConfig.builder().nCtx(128).ubatch(8)
                .typeK(KvDType.F32).typeV(KvDType.F32).build()).build();
        try (GGUFModel m = new GGUFModel(model.toString(), cfg)) {
            return GenerationLoop.forModel(m, cfg).generate(prompt, request(), (TokenCallback) null);
        }
    }

    @Test
    public void aNextTurnAndARepeatResumeAtTheirCheckpointsOnTheGpu() throws IOException {
        Path model = gemma2();
        int[] first = prompt(1, 21);
        try (GGUFModelGPU m = new GGUFModelGPU(model.toString(), backend, paged(true));
             BatchEngine engine = BatchEngine.forModel(m, paged(true))) {
            BatchSequence a = submit(engine, first);
            assertArrayEquals("the windowed GPU prefill matches the contiguous CPU loop",
                    contiguousCpu(model, first).tokens(), a.result().tokens());

            int[] history = a.result().withPrompt(first);
            int[] turn = Arrays.copyOf(history, history.length + 3);
            System.arraycopy(prompt(2, 3), 0, turn, history.length, 3);
            BatchSequence b = submit(engine, turn);
            assertEquals("the next turn resumes at the retired sequence's committed rows",
                    first.length + N_NEW - 1, b.reusedPrefix());
            assertArrayEquals(freshGpu(model, turn).tokens(), b.result().tokens());

            BatchSequence c = submit(engine, first);
            assertEquals("the repeat resumes at the last page boundary before its last token",
                    (first.length - 1) / PAGE * PAGE, c.reusedPrefix());
            assertArrayEquals(a.result().tokens(), c.result().tokens());
        }
    }

    /** mamba2 and qwen3next with the GDN layers, whose recurrent state rows live on the device. */
    private static Path recurrentModel(String name) throws IOException {
        if (name.equals("qwen3next")) {
            return SyntheticModels.qwen3next().writeTempFile();
        }
        return SyntheticModels.mamba2().writeTempFile();
    }

    /**
     * Batched recurrent generation on the GPU memory: every sequence's rows dispatch as its own contiguous step
     * against the sequence's device state row, and a batch computes the contiguous CPU loop's tokens — two
     * sequences decoding together, and a two-sequence mixed ubatch of prefill and decode rows.
     */
    @Test
    public void aBatchedRecurrentGpuStepComputesTheContiguousTokens() throws IOException {
        for (String name : new String[] {"mamba2", "qwen3next"}) {
            Path model = recurrentModel(name);
            int[] pa = prompt(3, 24);
            int[] pb = prompt(4, 24);
            EngineConfig paged = paged(false);
            try (GGUFModelGPU m = new GGUFModelGPU(model.toString(), backend, paged);
                 BatchEngine engine = BatchEngine.forModel(m, paged)) {
                BatchSequence a = engine.submit(pa, request(), (TokenCallback) null);
                BatchSequence b = engine.submit(pb, request(), (TokenCallback) null);
                engine.drain();
                assertEquals(name + " sequence a finished: " + a.error(), BatchSequence.State.DONE, a.state());
                assertEquals(name + " sequence b finished: " + b.error(), BatchSequence.State.DONE, b.state());
                assertArrayEquals(name + " sequence a against the contiguous loop",
                        contiguousCpu(model, pa).tokens(), a.result().tokens());
                assertArrayEquals(name + " sequence b against the contiguous loop",
                        contiguousCpu(model, pb).tokens(), b.result().tokens());
            }
        }
    }

    /**
     * Context checkpoints over the GPU recurrent memory: the saved state copy restores a conversation's next
     * turn at the retired sequence's committed rows and a repeated prompt at its last page boundary, both
     * computing the fresh engine's tokens (the device rows the checkpoints copy are the same rows the batched
     * steps advance).
     */
    @Test
    public void aGpuRecurrentNextTurnAndRepeatResumeAtTheirCheckpoints() throws IOException {
        Path model = recurrentModel("mamba2");
        int[] first = prompt(5, 21);
        try (GGUFModelGPU m = new GGUFModelGPU(model.toString(), backend, paged(true));
             BatchEngine engine = BatchEngine.forModel(m, paged(true))) {
            BatchSequence a = submit(engine, first);
            assertArrayEquals(contiguousCpu(model, first).tokens(), a.result().tokens());

            int[] history = a.result().withPrompt(first);
            int[] turn = Arrays.copyOf(history, history.length + 3);
            System.arraycopy(prompt(6, 3), 0, turn, history.length, 3);
            BatchSequence b = submit(engine, turn);
            assertEquals("the next turn resumes at the retired sequence's committed rows",
                    first.length + N_NEW - 1, b.reusedPrefix());
            assertArrayEquals(freshGpu(model, turn).tokens(), b.result().tokens());

            BatchSequence c = submit(engine, first);
            assertEquals("the repeat resumes at the last page boundary before its last token",
                    (first.length - 1) / PAGE * PAGE, c.reusedPrefix());
            assertArrayEquals(a.result().tokens(), c.result().tokens());
        }
    }
}

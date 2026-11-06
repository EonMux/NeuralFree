package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ocl.OclComputeBackend;
import com.eonmux.neuralfree.acceleration.ocl.CLDevice;
import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.GGUFModelGPU;
import com.eonmux.neuralfree.gguf.TokenCallback;
import com.eonmux.neuralfree.gguf.engine.EmbedsPrompt;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.GenerationLoop;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.MediaPrompt;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.engine.batch.BatchEngine;
import com.eonmux.neuralfree.gguf.engine.batch.BatchSequence;
import com.eonmux.neuralfree.gguf.engine.kvevents.KvCacheEvent;
import com.eonmux.neuralfree.gguf.engine.kvevents.KvEventBatch;
import com.eonmux.neuralfree.gguf.engine.kvevents.KvEventPublishers;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.ops.rope.MediaRopePositions;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * The prefix cache's block pool over the GPU paged memory, in pages of four rows: interleaved histories adopt
 * each other's whole blocks and the shared head of a boundary block (copied on the first write), free blocks
 * evict under pool pressure, media and prompt-embedding prompts reuse the blocks of their identical rows only, and
 * every prompt decodes with the tokens of the CPU run. Stores publish {@code BlockStored} events on the {@code GPU}
 * medium.
 */
public class GpuForwardBatchBlockPoolTest {

    private static final int PAGE = 4;
    private static final int N_NEW = 4;

    private HWAcceleration acceleration;
    private ComputeBackend backend;

    @Before
    public void setUp() {
        Assume.assumeTrue("OpenCL GPU not available",
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

    private static EngineConfig paged(int nCtx, int parallel) {
        return EngineConfig.builder().memory(MemoryConfig.builder().nCtx(nCtx).ubatch(8).kvPaged(true)
                .kvPageSize(PAGE).parallel(parallel).prefixReuse(true)
                .typeK(KvDType.F32).typeV(KvDType.F32).build()).build();
    }

    private static EngineConfig contiguous(int nCtx) {
        return EngineConfig.builder().memory(MemoryConfig.builder().nCtx(nCtx).ubatch(8)
                .typeK(KvDType.F32).typeV(KvDType.F32).build()).build();
    }

    private static GenerationRequest request() {
        return GenerationRequest.builder().maxNewTokens(N_NEW).ignoreEos(true)
                .sampler(SamplerConfig.greedyDecoding()).build();
    }

    private static int[] tokens(int seed, int n) {
        int[] t = new int[n];
        for (int i = 0; i < n; i++) {
            t[i] = 4 + (seed * 7 + i * 13) % 250;
        }
        return t;
    }

    private static int[] concat(int[] head, int[] tail) {
        int[] out = Arrays.copyOf(head, head.length + tail.length);
        System.arraycopy(tail, 0, out, head.length, tail.length);
        return out;
    }

    private static int[][] reference(Path model, int nCtx, int[][] prompts) throws IOException {
        int[][] out = new int[prompts.length][];
        try (GGUFModel one = new GGUFModel(model.toString(), contiguous(nCtx))) {
            for (int i = 0; i < prompts.length; i++) {
                out[i] = GenerationLoop.forModel(one, contiguous(nCtx))
                        .generate(prompts[i], request(), (TokenCallback) null).tokens();
            }
        }
        return out;
    }

    private static BatchSequence run(BatchEngine engine, int[] prompt) {
        BatchSequence seq = engine.submit(prompt, request(), (TokenCallback) null);
        for (int step = 0; step < 2_000 && !seq.finished(); step++) {
            engine.pump();
        }
        assertEquals("the request completed: " + seq.error(), BatchSequence.State.DONE, seq.state());
        return seq;
    }

    @Test
    public void interleavedHistoriesAdoptCachedBlocksWithTheReferenceTokens() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        int[] shared = tokens(1, 8);
        int[] a = concat(shared, tokens(2, 9));
        int[] b = concat(shared, tokens(3, 7));
        int[] a2 = concat(Arrays.copyOf(a, 16), tokens(4, 5));
        int[] c = concat(Arrays.copyOf(a, 10), tokens(6, 4));
        int[][] prompts = {a, b, a2, c};
        int[] expectedReuse = {0, 8, 16, 10};
        int[][] expected = reference(model, 128, prompts);
        List<KvEventBatch> batches = new ArrayList<>();
        try (GGUFModelGPU gpu = new GGUFModelGPU(model.toString(), backend, paged(128, 2));
             BatchEngine engine = BatchEngine.forModel(gpu, paged(128, 2), KvEventPublishers.listener(batches::add))) {
            for (int i = 0; i < prompts.length; i++) {
                BatchSequence seq = run(engine, prompts[i]);
                assertEquals("prompt " + i + " adopts the blocks it shares", expectedReuse[i], seq.reusedPrefix());
                assertArrayEquals("prompt " + i + " decodes with the reference tokens", expected[i],
                        seq.result().tokens());
            }
        }
        assertTrue(!batches.isEmpty());
        for (KvEventBatch batch : batches) {
            for (KvCacheEvent e : batch.events()) {
                if (e instanceof KvCacheEvent.BlockStored) {
                    assertEquals(KvCacheEvent.MEDIUM_GPU, ((KvCacheEvent.BlockStored) e).medium());
                }
            }
        }
    }

    private static float[][] mediaRows(int n, float v) {
        float[][] r = new float[n][64];
        for (int i = 0; i < n; i++) {
            for (int d = 0; d < 64; d++) {
                r[i][d] = v * (1 + 0.03f * ((i * 3 + d) % 7)) - 0.05f * (d % 3);
            }
        }
        return r;
    }

    private static MediaPrompt withMedia(int[] head, float[][] media, int[] tail) {
        return MediaPrompt.builder().addText(head)
                .addMedia(MediaRopePositions.normal(head.length, media.length), media, false)
                .addText(tail).build();
    }

    private static BatchSequence run(BatchEngine engine, MediaPrompt prompt) {
        BatchSequence seq = engine.submit(prompt, request(), (TokenCallback) null);
        for (int step = 0; step < 2_000 && !seq.finished(); step++) {
            engine.pump();
        }
        assertEquals("the request completed: " + seq.error(), BatchSequence.State.DONE, seq.state());
        return seq;
    }

    @Test
    public void mediaAndEmbeddingRowsReuseOnlyTheBlocksOfTheirOwnRows() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        int[] head = tokens(11, 6);
        int[] tail = tokens(12, 5);
        float[][] rows = mediaRows(7, 0.2f);
        MediaPrompt[] prompts = {
                withMedia(head, mediaRows(6, 0.3f), tail),
                withMedia(head, mediaRows(6, 0.3f), tail),
                withMedia(head, mediaRows(6, 0.6f), tail),
                MediaPrompt.ofText(concat(head, tokens(13, 4))),
                EmbedsPrompt.ofRows(rows).toMediaPrompt(),
                EmbedsPrompt.ofRows(rows).toMediaPrompt()};
        int[] expectedReuse = {0, 6 + 6 + 5 - 1, 6, 6, 0, rows.length - 1};
        int[][] expected = new int[prompts.length][];
        EngineConfig cpu = EngineConfig.builder().memory(MemoryConfig.builder().nCtx(128).ubatch(8).kvPaged(true)
                .kvPageSize(PAGE).parallel(2).prefixReuse(false).typeK(KvDType.F32).typeV(KvDType.F32).build())
                .build();
        try (GGUFModel m = new GGUFModel(model.toString(), cpu);
             BatchEngine engine = BatchEngine.forModel(m, cpu)) {
            for (int i = 0; i < prompts.length; i++) {
                expected[i] = run(engine, prompts[i]).result().tokens();
            }
        }
        try (GGUFModelGPU gpu = new GGUFModelGPU(model.toString(), backend, paged(128, 2));
             BatchEngine engine = BatchEngine.forModel(gpu, paged(128, 2))) {
            for (int i = 0; i < prompts.length; i++) {
                BatchSequence seq = run(engine, prompts[i]);
                assertEquals("prompt " + i + " reuses the rows it shares", expectedReuse[i], seq.reusedPrefix());
                assertArrayEquals("prompt " + i + " decodes with the CPU tokens", expected[i],
                        seq.result().tokens());
            }
        }
    }

    @Test
    public void blocksEvictedUnderPressureKeepTheReferenceTokens() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        int[] x = tokens(9, 18);
        int[] y = tokens(10, 18);
        int[][] prompts = {x, y, x};
        int[][] expected = reference(model, 32, prompts);
        try (GGUFModelGPU gpu = new GGUFModelGPU(model.toString(), backend, paged(32, 1));
             BatchEngine engine = BatchEngine.forModel(gpu, paged(32, 1))) {
            for (int i = 0; i < prompts.length; i++) {
                BatchSequence seq = run(engine, prompts[i]);
                assertArrayEquals("prompt " + i + " decodes with the reference tokens", expected[i],
                        seq.result().tokens());
                if (i == 2) {
                    assertTrue("the head blocks outlive the evicted tail, got " + seq.reusedPrefix(),
                            seq.reusedPrefix() >= PAGE && seq.reusedPrefix() < x.length - 1);
                }
            }
        }
    }
}

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
import com.eonmux.neuralfree.gguf.engine.lora.LoraConfig;
import com.eonmux.neuralfree.gguf.engine.lora.LoraRegistry;
import com.eonmux.neuralfree.gguf.engine.lora.LoraSelection;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.LoraAdapterFiles;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Path;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Per-request adapters on the device batched forward: one {@code forwardBatch} over the interleaved rows of a
 * sequence in adapter {@code a}'s slot, one on the base model and one in adapter {@code b}'s slot gives each the
 * logits of the host model with that adapter attached to every row ({@code --lora}), for a prefill split across
 * mixed-slot ubatches and a decode step; closing the slots releases their deltas.
 */
public class GpuMultiLoraForwardTest {

    private static final int RANK = 4;
    private static final float ALPHA = 8.0f;
    private static final float SCALE_A = 0.75f;
    private static final float SCALE_B = 0.5f;
    private static final float TOL = 2e-3f;
    private static final int[][] PROMPTS = {{1, 17, 42, 5, 99}, {3, 8, 120, 64, 7}, {250, 2, 33, 18, 91}};
    private static final int NEXT = 77;

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

    private static EngineConfig.Builder config(int parallel) {
        MemoryConfig.Builder mc = MemoryConfig.builder().nCtx(48).ubatch(4).typeK(KvDType.F32).typeV(KvDType.F32);
        if (parallel > 1) {
            mc = mc.parallel(parallel);
        }
        return EngineConfig.builder().memory(mc.build())
                .loraConfig(LoraConfig.builder().enabled(true).maxAdapters(2).maxRank(64).build());
    }

    @Test
    public void rowsOfDifferentAdaptersInOneDeviceBatchMatchTheirSingleAdapterModels() throws IOException {
        Path base = SyntheticModels.llama().writeTempFile();
        Path a = LoraAdapterFiles.llama(1000, RANK, ALPHA);
        Path b = LoraAdapterFiles.llama(5000, RANK, ALPHA);
        float[][] refA = reference(base, config(1).addLora(a.toString(), SCALE_A).build(), PROMPTS[0]);
        float[][] refBase = reference(base, config(1).build(), PROMPTS[1]);
        float[][] refB = reference(base, config(1).addLora(b.toString(), SCALE_B).build(), PROMPTS[2]);

        try (GGUFModelGPU m = new GGUFModelGPU(base.toString(), backend, config(3).build())) {
            LoraRegistry registry = m.loraRegistry();
            registry.load("a", a.toString(), SCALE_A);
            registry.load("b", b.toString(), SCALE_B);
            int sa = registry.acquire(registry.resolve(LoraSelection.of("a")));
            int sb = registry.acquire(registry.resolve(LoraSelection.of("b")));
            int[] slotOfSeq = {sa, 0, sb};
            ForwardBackend f = m.forwardBackend();

            float[][] prefill = f.forwardBatch(interleavedPrefill(slotOfSeq));
            float[][] decode = f.forwardBatch(decodeStep(slotOfSeq));
            assertArrayEquals("prefill of the adapter-a sequence", refA[0], prefill[0], TOL);
            assertArrayEquals("prefill of the base sequence", refBase[0], prefill[1], TOL);
            assertArrayEquals("prefill of the adapter-b sequence", refB[0], prefill[2], TOL);
            assertArrayEquals("decode of the adapter-a sequence", refA[1], decode[0], TOL);
            assertArrayEquals("decode of the base sequence", refBase[1], decode[1], TOL);
            assertArrayEquals("decode of the adapter-b sequence", refB[1], decode[2], TOL);

            registry.release(sa);
            registry.release(sb);
            registry.unload("a");
            registry.unload("b");
            assertEquals(0, registry.list().size());
            ForwardBatch stale = ForwardBatch.builder(1).add(NEXT, PROMPTS[0].length + 1, 0, true)
                    .adapterSlots(new int[]{sa}).build();
            RuntimeException e = org.junit.Assert.assertThrows(RuntimeException.class, () -> f.forwardBatch(stale));
            assertTrue(e.getMessage(), e.getMessage().contains("adapterSlots"));
        }
    }

    private static ForwardBatch interleavedPrefill(int[] slotOfSeq) {
        int len = PROMPTS[0].length;
        int n = PROMPTS.length * len;
        ForwardBatch.Builder fb = ForwardBatch.builder(n);
        int[] slots = new int[n];
        int row = 0;
        for (int i = 0; i < len; i++) {
            for (int s = 0; s < PROMPTS.length; s++) {
                fb.add(PROMPTS[s][i], i, s, i == len - 1);
                slots[row++] = slotOfSeq[s];
            }
        }
        return fb.adapterSlots(slots).build();
    }

    private static ForwardBatch decodeStep(int[] slotOfSeq) {
        ForwardBatch.Builder fb = ForwardBatch.builder(PROMPTS.length);
        for (int s = 0; s < PROMPTS.length; s++) {
            fb.add(NEXT, PROMPTS[s].length, s, true);
        }
        return fb.adapterSlots(slotOfSeq).build();
    }

    /** Host prefill logits of {@code prompt} and decode logits of {@link #NEXT}. */
    private static float[][] reference(Path model, EngineConfig cfg, int[] prompt) throws IOException {
        try (GGUFModel m = new GGUFModel(model.toString(), cfg)) {
            ForwardBackend f = m.forwardBackend();
            float[] prefill = f.prefill(prompt, 0).clone();
            float[] decode = f.decode(NEXT, prompt.length).clone();
            return new float[][]{prefill, decode};
        }
    }
}

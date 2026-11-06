package com.eonmux.neuralfree.gguf.cpu.forward;

import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.engine.ForwardBatch;
import com.eonmux.neuralfree.gguf.engine.lora.LoraRegistry;
import com.eonmux.neuralfree.gguf.engine.lora.LoraSelection;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.LoraAdapterFiles;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Path;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Per-request adapters on the host batched forward (vLLM multi-LoRA batching): one {@code forwardBatch} over the
 * interleaved rows of three sequences — one in adapter {@code a}'s slot, one on the base model, one in adapter
 * {@code b}'s slot — gives each sequence the logits a model with that adapter attached to every row
 * ({@code --lora}) gives it alone, for a prefill split across ubatches and a decode step; a row in a slot that
 * holds no adapter is refused by name.
 */
public class CpuMultiLoraForwardTest {

    private static final int RANK = 4;
    private static final float ALPHA = 8.0f;
    private static final float SCALE_A = 0.75f;
    private static final float SCALE_B = 0.5f;
    private static final float TOL = 1e-3f;
    private static final int[][] PROMPTS = {{1, 17, 42, 5, 99}, {3, 8, 120, 64, 7}, {250, 2, 33, 18, 91}};
    private static final int NEXT = 77;

    private static EngineConfig.Builder config(int parallel, int ubatch) {
        MemoryConfig.Builder mc = MemoryConfig.builder().nCtx(48).ubatch(ubatch).typeK(KvDType.F32)
                .typeV(KvDType.F32);
        if (parallel > 1) {
            mc = mc.parallel(parallel);
        }
        return EngineConfig.builder().memory(mc.build());
    }

    @Test
    public void rowsOfDifferentAdaptersInOneBatchMatchTheirSingleAdapterModels() throws IOException {
        Path base = SyntheticModels.llama().writeTempFile();
        Path a = LoraAdapterFiles.llama(1000, RANK, ALPHA);
        Path b = LoraAdapterFiles.llama(5000, RANK, ALPHA);
        float[][] refA = reference(base, config(1, 4).addLora(a.toString(), SCALE_A).build(), PROMPTS[0]);
        float[][] refBase = reference(base, config(1, 4).build(), PROMPTS[1]);
        float[][] refB = reference(base, config(1, 4).addLora(b.toString(), SCALE_B).build(), PROMPTS[2]);
        assertTrue(maxAbsDiff(refA[0], reference(base, config(1, 4).build(), PROMPTS[0])[0]) > 100 * TOL);

        // ubatch 4 splits the 15 prompt rows into mixed-slot ubatches; ubatch 16 runs them in one
        for (int ubatch : new int[]{4, 16}) {
            EngineConfig cfg = config(3, ubatch).build();
            try (GGUFModel m = new GGUFModel(base.toString(), cfg)) {
                LoraRegistry registry = m.loraRegistry();
                registry.load("a", a.toString(), SCALE_A);
                registry.load("b", b.toString(), SCALE_B);
                int sa = registry.acquire(registry.resolve(LoraSelection.of("a")));
                int sb = registry.acquire(registry.resolve(LoraSelection.of("b")));
                assertNotEquals(sa, sb);
                int[] slotOfSeq = {sa, 0, sb};
                ForwardBackend f = m.forwardBackend();

                float[][] prefill = f.forwardBatch(interleavedPrefill(slotOfSeq));
                float[][] decode = f.forwardBatch(decodeStep(slotOfSeq));
                String at = "ubatch " + ubatch + ": ";
                assertArrayEquals(at + "prefill of the adapter-a sequence", refA[0], prefill[0], TOL);
                assertArrayEquals(at + "prefill of the base sequence", refBase[0], prefill[1], TOL);
                assertArrayEquals(at + "prefill of the adapter-b sequence", refB[0], prefill[2], TOL);
                assertArrayEquals(at + "decode of the adapter-a sequence", refA[1], decode[0], TOL);
                assertArrayEquals(at + "decode of the base sequence", refBase[1], decode[1], TOL);
                assertArrayEquals(at + "decode of the adapter-b sequence", refB[1], decode[2], TOL);
            }
        }
    }

    @Test
    public void aRowInASlotThatHoldsNoAdapterIsRefusedByName() throws IOException {
        Path base = SyntheticModels.llama().writeTempFile();
        EngineConfig cfg = config(2, 4).build();
        try (GGUFModel m = new GGUFModel(base.toString(), cfg)) {
            ForwardBatch batch = ForwardBatch.builder(1).add(5, 0, 0, true).adapterSlots(new int[]{2}).build();
            RuntimeException e = assertThrows(RuntimeException.class, () -> m.forwardBackend().forwardBatch(batch));
            assertTrue(e.getMessage(), e.getMessage().contains("adapterSlots"));
            assertTrue(e.getMessage(), e.getMessage().contains("slot 2"));
        }
    }

    /** The prompts' rows interleaved row by row, each sequence asking the logits of its last row. */
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

    /** Prefill logits of {@code prompt} and decode logits of {@link #NEXT} on a single-sequence model. */
    private static float[][] reference(Path model, EngineConfig cfg, int[] prompt) throws IOException {
        try (GGUFModel m = new GGUFModel(model.toString(), cfg)) {
            ForwardBackend f = m.forwardBackend();
            float[] prefill = f.prefill(prompt, 0).clone();
            float[] decode = f.decode(NEXT, prompt.length).clone();
            return new float[][]{prefill, decode};
        }
    }

    private static float maxAbsDiff(float[] x, float[] y) {
        float m = 0f;
        for (int i = 0; i < x.length; i++) {
            m = Math.max(m, Math.abs(x[i] - y[i]));
        }
        return m;
    }
}

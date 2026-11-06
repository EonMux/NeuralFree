package com.eonmux.neuralfree.gguf.cpu.forward;

import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuInt8Fusion;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.ForwardBatch;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.memory.cpu.HeapModelMemory;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

/**
 * The fused int8 attention through the CPU forward backend: a synthetic llama run with an
 * {@code int8_per_token_head} KV cache returns the same logits with {@link CpuInt8Fusion#PROP_INT8} on and off, so the
 * int8 queries perturb the outputs by the quantization error only and the layer executor, the mixer and the cache's
 * raw row view are wired the same way in both runs. A paged pair of sequences reaches the same logits as the two
 * contiguous runs, so the mixer's slot-to-row map feeds the fused kernel like the generic one.
 */
public class CpuInt8AttentionForwardTest {

    /** Largest logit difference the int8 query quantization may cause on the synthetic llama. */
    private static final float TOL = 5e-3f;

    private static EngineConfig config(boolean paged) {
        MemoryConfig.Builder memory = MemoryConfig.builder().nCtx(64).ubatch(8)
                .typeK(KvDType.INT8_PER_TOKEN_HEAD).typeV(KvDType.INT8_PER_TOKEN_HEAD);
        if (paged) {
            memory = memory.parallel(2);
        }
        return EngineConfig.builder().backend(EngineConfig.BackendMode.CPU).memory(memory.build()).build();
    }

    private static EngineConfig config() {
        return config(false);
    }

    /** Sets {@link CpuInt8Fusion#PROP_INT8} and returns its previous value, null when it was unset. */
    private static String selectFused(boolean fused) {
        String previous = System.getProperty(CpuInt8Fusion.PROP_INT8);
        System.setProperty(CpuInt8Fusion.PROP_INT8, Boolean.toString(fused));
        return previous;
    }

    private static void restore(String previous) {
        if (previous == null) {
            System.clearProperty(CpuInt8Fusion.PROP_INT8);
        } else {
            System.setProperty(CpuInt8Fusion.PROP_INT8, previous);
        }
    }

    /** The logits of {@code tokens}' last row with the fused paths selected or not. */
    private static float[] logits(Path model, int[] tokens, boolean fused) throws IOException {
        String previous = selectFused(fused);
        try (GGUFModel m = new GGUFModel(model.toString(), config())) {
            return m.forwardBackend().forwardRows(tokens, 0, new int[]{tokens.length - 1})[0];
        } finally {
            restore(previous);
        }
    }

    /** The logits of {@code a} and {@code b} run one after the other on a contiguous cache. */
    private static float[][] contiguous(Path model, int[] a, int[] b) throws IOException {
        try (GGUFModel m = new GGUFModel(model.toString(), config())) {
            float[] first = m.forwardBackend().forwardRows(a, 0, new int[]{a.length - 1})[0];
            m.forwardBackend().reset();
            float[] second = m.forwardBackend().forwardRows(b, 0, new int[]{b.length - 1})[0];
            return new float[][]{first, second};
        }
    }

    /** The batch rows interleaving the sequences {@code a} and {@code b}, the last row of each an output row. */
    private static ForwardBatch interleave(int[] a, int[] b) {
        ForwardBatch.Builder batch = ForwardBatch.builder(2 * a.length);
        for (int i = 0; i < a.length; i++) {
            batch.add(a[i], i, 0, i == a.length - 1);
            batch.add(b[i], i, 1, i == b.length - 1);
        }
        return batch.build();
    }

    @Test
    public void anInt8KvModelMatchesTheDequantizingPath() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        try {
            int[] tokens = {4, 11, 17, 23, 30, 42, 7, 19};
            float[] generic = logits(model, tokens, false);
            float[] fused = logits(model, tokens, true);
            float max = 0f;
            for (int i = 0; i < generic.length; i++) {
                assertTrue("logit " + i + " is finite", Float.isFinite(fused[i]));
                max = Math.max(max, Math.abs(generic[i] - fused[i]));
            }
            assertTrue("largest logit difference " + max, max <= TOL);
        } finally {
            Files.deleteIfExists(model);
        }
    }

    @Test
    public void decodeAfterAPrefillMatchesTheDequantizingPath() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        try {
            int[] prompt = {4, 11, 17, 23};
            float[] generic;
            float[] fused;
            String previous = selectFused(false);
            try {
                try (GGUFModel m = new GGUFModel(model.toString(), config())) {
                    m.forwardBackend().forwardRows(prompt, 0, new int[]{prompt.length - 1});
                    generic = m.forwardBackend().forwardRows(new int[]{30}, prompt.length, new int[]{0})[0];
                }
                System.setProperty(CpuInt8Fusion.PROP_INT8, "true");
                try (GGUFModel m = new GGUFModel(model.toString(), config())) {
                    m.forwardBackend().forwardRows(prompt, 0, new int[]{prompt.length - 1});
                    fused = m.forwardBackend().forwardRows(new int[]{30}, prompt.length, new int[]{0})[0];
                }
            } finally {
                restore(previous);
            }
            float max = 0f;
            for (int i = 0; i < generic.length; i++) {
                max = Math.max(max, Math.abs(generic[i] - fused[i]));
            }
            assertTrue("largest decode logit difference " + max, max <= TOL);
        } finally {
            Files.deleteIfExists(model);
        }
    }

    @Test
    public void aPagedPairOfInt8SequencesMatchesTheContiguousRuns() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        try {
            int[] a = {4, 11, 17, 23, 30, 42, 7, 19};
            int[] b = {5, 12, 18, 24, 31, 43, 8, 20};
            String previous = selectFused(true);
            try {
                float[][] expected = contiguous(model, a, b);
                try (GGUFModel m = new GGUFModel(model.toString(), config(true))) {
                    HeapModelMemory memory = (HeapModelMemory) m.forwardBackend().memory();
                    assertNotNull("two sequences page the int8 cache", memory.pagedSequences());
                    float[][] out = m.forwardBackend().forwardBatch(interleave(a, b));
                    assertArrayEquals("paged sequence A logits", expected[0], out[0], TOL);
                    assertArrayEquals("paged sequence B logits", expected[1], out[1], TOL);
                }
            } finally {
                restore(previous);
            }
        } finally {
            Files.deleteIfExists(model);
        }
    }
}

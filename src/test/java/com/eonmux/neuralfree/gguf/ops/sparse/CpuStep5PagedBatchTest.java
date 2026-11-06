package com.eonmux.neuralfree.gguf.ops.sparse;

import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.GgufWriter;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.ForwardBatch;
import com.eonmux.neuralfree.gguf.hf.HfModelSource;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.memory.cpu.HeapModelMemory;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.nio.file.Path;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Paged multi-sequence steps over the Step5 sparse region attention: {@code CpuForwardBackend.forwardBatch} runs the
 * synthetic Step5 model through the paged memory, where each sequence keeps its own region-summary and open-region ring
 * state in its own state slot and every segment selects among its own sequence's positions only. The batch logits equal
 * the same tokens run one sequence at a time on the contiguous path, for an interleaved prompt batch and the decode
 * round after it; a non-causal step is refused by name because the region summaries and the sparse gather are causal.
 */
public class CpuStep5PagedBatchTest {

    private static final int N_TOKENS = 12;
    private static final int SEQ_A = 0;
    private static final int SEQ_B = 1;
    private static final float TOL = 1e-4f;

    @Rule public final TemporaryFolder tmp = new TemporaryFolder();

    private static EngineConfig config(boolean paged) {
        MemoryConfig.Builder mc = MemoryConfig.builder().nCtx(64).ubatch(8)
                .typeK(KvDType.BF16).typeV(KvDType.BF16);
        if (paged) {
            mc = mc.parallel(2);
        }
        return EngineConfig.builder().memory(mc.build()).build();
    }

    private static int[] tokens(int seed, int n) {
        int[] t = new int[n];
        for (int i = 0; i < t.length; i++) {
            t[i] = (seed * 3 + i * 5 + 1) % 7;
        }
        return t;
    }

    private static int[] append(int[] a, int token) {
        int[] b = new int[a.length + 1];
        System.arraycopy(a, 0, b, 0, a.length);
        b[a.length] = token;
        return b;
    }

    private static int argmax(float[] v) {
        int best = 0;
        for (int i = 1; i < v.length; i++) {
            if (v[i] > v[best]) {
                best = i;
            }
        }
        return best;
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

    /** The contiguous logits of the last row of {@code t}, after a reset. */
    private static float[] contiguous(GGUFModel model, int[] t) {
        model.forwardBackend().reset();
        return model.forwardBackend().forwardRows(t, 0, new int[]{t.length - 1})[0];
    }

    @Test
    public void pagedStep5BatchMatchesTheContiguousRuns() throws Exception {
        Path model = export("step5-paged");
        int[] a = tokens(1, N_TOKENS);
        int[] b = tokens(2, N_TOKENS);
        float[] refA;
        float[] refB;
        try (GGUFModel one = new GGUFModel(model.toString(), config(false))) {
            refA = contiguous(one, a);
            refB = contiguous(one, b);
        }
        try (GGUFModel two = new GGUFModel(model.toString(), config(true))) {
            HeapModelMemory mem = (HeapModelMemory) two.forwardBackend().memory();
            assertNotNull("the two-sequence memory is paged", mem.pagedSequences());
            float[][] out = two.forwardBackend().forwardBatch(interleave(a, b));
            assertEquals(2, out.length);
            assertArrayEquals("sequence A's logits match the contiguous run", refA, out[0], TOL);
            assertArrayEquals("sequence B's logits match the contiguous run", refB, out[1], TOL);

            int nextA = argmax(out[0]);
            int nextB = argmax(out[1]);
            float[] refNextA;
            float[] refNextB;
            try (GGUFModel one = new GGUFModel(model.toString(), config(false))) {
                refNextA = contiguous(one, append(a, nextA));
                refNextB = contiguous(one, append(b, nextB));
            }
            ForwardBatch.Builder round = ForwardBatch.builder(2);
            round.add(nextA, N_TOKENS, SEQ_A, true);
            round.add(nextB, N_TOKENS, SEQ_B, true);
            float[][] decoded = two.forwardBackend().forwardBatch(round.build());
            assertArrayEquals("the decode round continues sequence A's own region state", refNextA, decoded[0], TOL);
            assertArrayEquals("the decode round continues sequence B's own region state", refNextB, decoded[1], TOL);
        }
    }

    @Test
    public void nonCausalPagedStep5RefusesByName() throws Exception {
        Path model = export("step5-paged-noncausal");
        try (GGUFModel two = new GGUFModel(model.toString(), config(true))) {
            ForwardBatch.Builder batch = ForwardBatch.builder(2);
            batch.add(1, 0, SEQ_A, true);
            batch.add(2, 0, SEQ_B, true);
            batch.nonCausal();
            UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class,
                    () -> two.forwardBackend().forwardBatch(batch.build()));
            assertTrue(e.getMessage(), e.getMessage().contains("non-causal"));
        }
    }

    /** Converts the synthetic Step5 checkpoint and exports it as a GGUF. */
    private Path export(String name) throws Exception {
        Path dir = CpuStep5ParityTest.builder().write(tmp.newFolder(name + "-hf").toPath());
        Path gguf = tmp.getRoot().toPath().resolve(name + ".gguf");
        try (HfModelSource src = HfModelSource.open(dir, ModelSourceOptions.builder()
                .cacheDir(tmp.newFolder(name + "-cache").toPath()).build(), null)) {
            GgufWriter.write(src, gguf, new GgufWriter.Options(GgufWriter.Mode.NATIVE));
        }
        return gguf;
    }
}

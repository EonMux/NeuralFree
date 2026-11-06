package com.eonmux.neuralfree.gguf.cpu.forward;

import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.arch.SupportClass;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.ForwardBatch;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.memory.cpu.HeapModelMemory;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Path;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

/**
 * Multi-sequence (paged) batch steps over recurrent and hybrid mixers: {@code CpuForwardBackend.forwardBatch} runs
 * the synthetic Mamba-2 (pure
 * recurrent state) and qwen3next (GDN layers plus attention) through the paged memory, whose recurrent state holds
 * one row per sequence ({@code RecurrentStateAccessor.slots()}), and the batch logits equal the same tokens run one
 * sequence at a time on the contiguous path. A prompt batch over a recurrent memory splits
 * {@code split_equal}-style (llama-memory-recurrent.cpp:455): multi-segment ubatches, one state update per sequence
 * per ubatch; a decode-style batch (every row an output) splits by sequence.
 */
public class CpuForwardBatchRecurrentTest {

    private static final int N_TOKENS = 12;
    private static final int SEQ_A = 0;
    private static final int SEQ_B = 1;
    private static final float TOL = 1e-3f;

    private static EngineConfig.Builder config(boolean paged) {
        MemoryConfig.Builder mc = MemoryConfig.builder().nCtx(64).ubatch(8).typeK(KvDType.F32).typeV(KvDType.F32);
        if (paged) {
            mc = mc.parallel(2);
        }
        return EngineConfig.builder().memory(mc.build());
    }

    /** The tokens a sequence runs (deterministic ids inside the synthetic vocab, as CpuForwardBatchTest). */
    private static int[] tokens(int seed) {
        int[] t = new int[N_TOKENS];
        for (int i = 0; i < t.length; i++) {
            t[i] = 4 + (seed * 7 + i * 13) % 250;
        }
        return t;
    }

    /** A batch whose rows interleave two prompts; the last row of each sequence is its output row. */
    private static ForwardBatch interleave(int[] a, int[] b) {
        ForwardBatch.Builder batch = ForwardBatch.builder(2 * N_TOKENS);
        for (int i = 0; i < N_TOKENS; i++) {
            batch.add(a[i], i, SEQ_A, i == N_TOKENS - 1);
            batch.add(b[i], i, SEQ_B, i == N_TOKENS - 1);
        }
        return batch.build();
    }

    /** A decode-style batch: one positioned token per sequence, every row an output row. */
    private static ForwardBatch decodeStep(int[] promptA, int[] promptB, int tokenA, int tokenB) {
        ForwardBatch.Builder batch = ForwardBatch.builder(2);
        batch.add(tokenA, promptA.length, SEQ_A, true);
        batch.add(tokenB, promptB.length, SEQ_B, true);
        return batch.build();
    }

    private static float[][] reference(Path model, EngineConfig cfg, int[] a, int[] b) throws IOException {
        try (GGUFModel one = new GGUFModel(model.toString(), cfg)) {
            assertEquals(SupportClass.GEN, one.spec().supportClass());
            float[][] out = new float[][]{
                    one.forwardBackend().forwardRows(a, 0, new int[]{a.length - 1})[0], null};
            one.forwardBackend().reset();
            out[1] = one.forwardBackend().forwardRows(b, 0, new int[]{b.length - 1})[0];
            return out;
        }
    }

    private void check(Path model) throws IOException {
        int[] a = tokens(1);
        int[] b = tokens(2);
        float[][] ref = reference(model, config(false).build(), a, b);
        try (GGUFModel two = new GGUFModel(model.toString(), config(true).build())) {
            HeapModelMemory mem = (HeapModelMemory) two.forwardBackend().memory();
            assertNotNull(mem.pagedSequences());
            assertTrue("the synthetic model keeps recurrent state", mem.plan().hasRecurrentState());
            float[][] out = two.forwardBackend().forwardBatch(interleave(a, b));
            assertEquals(2, out.length);
            assertArrayEquals("sequence A logits match the contiguous run", ref[0], out[0], TOL);
            assertArrayEquals("sequence B logits match the contiguous run", ref[1], out[1], TOL);
            // a second decode round: the sequences continue their paged states, one row per segment
            int[] tokenA = argmax(out[0]);
            int[] tokenB = argmax(out[1]);
            float[][] refNext = referenceNext(model, config(false).build(), a, tokenA[0], b, tokenB[0]);
            float[][] decoded = two.forwardBackend().forwardBatch(
                    decodeStep(a, b, tokenA[0], tokenB[0]));
            assertArrayEquals("the decoded round continues the paged states", refNext[0], decoded[0], TOL);
            assertArrayEquals(refNext[1], decoded[1], TOL);
        }
    }

    private static int[] argmax(float[] logits) {
        int best = 0;
        for (int i = 1; i < logits.length; i++) {
            if (logits[i] > logits[best]) {
                best = i;
            }
        }
        return new int[]{best};
    }

    /** The contiguous decode reference: prefill the prompt, feed the sampled token at the next position. */
    private static float[][] referenceNext(Path model, EngineConfig cfg, int[] a, int tokenA, int[] b, int tokenB)
            throws IOException {
        try (GGUFModel one = new GGUFModel(model.toString(), cfg)) {
            float[][] out = new float[2][];
            float[][] rowsA = one.forwardBackend().forwardRows(append(a, tokenA), 0, new int[]{a.length});
            out[0] = rowsA[0];
            one.forwardBackend().reset();
            float[][] rowsB = one.forwardBackend().forwardRows(append(b, tokenB), 0, new int[]{b.length});
            out[1] = rowsB[0];
            return out;
        }
    }

    private static int[] append(int[] prompt, int token) {
        int[] t = new int[prompt.length + 1];
        System.arraycopy(prompt, 0, t, 0, prompt.length);
        t[prompt.length] = token;
        return t;
    }

    @Test
    public void mamba2PagedBatchMatchesTheContiguousRuns() throws IOException {
        check(SyntheticModels.mamba2().writeTempFile());
    }

    @Test
    public void qwen3nextPagedBatchMatchesTheContiguousRuns() throws IOException {
        check(SyntheticModels.qwen3next().writeTempFile());
    }

    @Test
    public void mamba2WithoutKvStillPlansAPagedMemory() throws IOException {
        Path model = SyntheticModels.mamba2().writeTempFile();
        try (GGUFModel two = new GGUFModel(model.toString(), config(true).build())) {
            HeapModelMemory mem = (HeapModelMemory) two.forwardBackend().memory();
            assertNotNull("a pure-recurrent model tracks its sequences over the bookkeeping group",
                    mem.pagedSequences());
            assertEquals(2, mem.pagedSequences().maxSequences());
            assertTrue(mem.pagedSequences().totalCells() > 0);
        }
    }
}

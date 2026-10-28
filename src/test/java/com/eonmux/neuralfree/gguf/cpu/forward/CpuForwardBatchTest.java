package com.eonmux.neuralfree.gguf.cpu.forward;

import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.arch.SupportClass;
import com.eonmux.neuralfree.gguf.engine.ContextFullException;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.ForwardBatch;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
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
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Multi-sequence forward batch over the paged memory (PARITY P1): {@code CpuForwardBackend.forwardBatch} runs
 * SEQ-mode ubatches through {@code PagedSequences.prepare} -> the page tables -> {@code PagedSequences.commit}, so a
 * batch over several sequences computes the same logits as the same tokens evaluated one sequence at a time on the
 * contiguous path (same tokens -> same logits), a single-sequence batch runs the paged step too, and an unpaged
 * memory refuses the batch by name. The synthetic two-layer llama opens unpaged (the reference) and paged with
 * {@code parallel = 2}.
 */
public class CpuForwardBatchTest {

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

    /** The tokens a sequence runs (deterministic ids inside the synthetic vocab). */
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

    private static ForwardBatch single(int[] a) {
        ForwardBatch.Builder batch = ForwardBatch.builder(N_TOKENS);
        for (int i = 0; i < N_TOKENS; i++) {
            batch.add(a[i], i, SEQ_A, i == N_TOKENS - 1);
        }
        return batch.build();
    }

    @Test
    public void aBatchOverTwoSequencesMatchesTheSequentialContiguousRuns() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        float[][] reference;
        try (GGUFModel one = new GGUFModel(model.toString(), config(false).build())) {
            assertEquals(SupportClass.GEN, one.spec().supportClass());
            reference = new float[2][];
            int[] a = tokens(1);
            int[] b = tokens(2);
            reference[0] = one.forwardBackend().forwardRows(a, 0, new int[]{a.length - 1})[0];
            one.forwardBackend().reset();
            reference[1] = one.forwardBackend().forwardRows(b, 0, new int[]{b.length - 1})[0];
        }
        try (GGUFModel two = new GGUFModel(model.toString(), config(true).build())) {
            HeapModelMemory mem = (HeapModelMemory) two.forwardBackend().memory();
            assertNotNull(mem.pagedSequences());
            assertEquals(2, mem.pagedSequences().maxSequences());
            float[][] out = two.forwardBackend().forwardBatch(interleave(tokens(1), tokens(2)));
            assertEquals(2, out.length);
            assertArrayEquals("sequence A logits match the contiguous run", reference[0], out[0], TOL);
            assertArrayEquals("sequence B logits match the contiguous run", reference[1], out[1], TOL);
        }
    }

    @Test
    public void thePagedPathAcceptsASingleSequenceBatchAndMatchesTheContiguousOne() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        int[] a = tokens(3);
        float[] reference;
        try (GGUFModel one = new GGUFModel(model.toString(), config(false).build())) {
            reference = one.forwardBackend().forwardRows(a, 0, new int[]{a.length - 1})[0];
        }
        try (GGUFModel two = new GGUFModel(model.toString(), config(true).build())) {
            float[][] out = two.forwardBackend().forwardBatch(single(a));
            assertArrayEquals("single-sequence paged run matches the contiguous one", reference, out[0], TOL);
        }
    }

    @Test
    public void anUnpagedMemoryRefusesTheBatchByName() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        try (GGUFModel one = new GGUFModel(model.toString(), config(false).build())) {
            UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class,
                    () -> one.forwardBackend().forwardBatch(interleave(tokens(1), tokens(2))));
            assertTrue(e.getMessage(), e.getMessage().contains("paged"));
        }
    }

    @Test
    public void aContextOverflowRefusesBeforeAnyPageIsTaken() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        try (GGUFModel two = new GGUFModel(model.toString(), config(true).build())) {
            // a sequence whose next token would exceed the 64-position context (63 stored, decode at 64)
            ForwardBatch.Builder batch = ForwardBatch.builder(65);
            for (int i = 0; i < 64; i++) {
                batch.add(4 + i, i, SEQ_A, false);
            }
            batch.add(4, 64, SEQ_A, true);
            assertThrows(ContextFullException.class, () -> two.forwardBackend().forwardBatch(batch.build()));
        }
    }
}

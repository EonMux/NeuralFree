package com.eonmux.neuralfree.gguf.cpu.forward;

import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.arch.SupportClass;
import com.eonmux.neuralfree.gguf.engine.BatchSplitter;
import com.eonmux.neuralfree.gguf.engine.ContextFullException;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.ForwardBatch;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.engine.HiddenTap;
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
 * Multi-sequence forward batch over the paged memory: {@code CpuForwardBackend.forwardBatch} splits the
 * batch into cross-sequence ubatches ({@code split_simple}, one trunk pass over many sequences' tokens) through
 * {@code PagedSequences.prepare} -> the page tables -> {@code PagedSequences.commit}, so a batch over several
 * sequences computes the same logits as the same tokens evaluated one sequence at a time on the contiguous path
 * (same tokens -> same logits), a single-sequence batch runs the paged step too, and an unpaged memory refuses the
 * batch by name. The hidden tap sees every row of a call across the ubatch splits. The synthetic two-layer llama
 * opens unpaged (the reference) and paged with {@code parallel = 2}.
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
    public void aKvOnlyMemoryMixesSequencesInOneUbatchAndARecurrentOneSplitsBySequence() {
        // llama.cpp: split_simple over a unified KV cache, split_equal / split_seq over recurrent and hybrid memory
        assertEquals(BatchSplitter.Mode.SIMPLE, CpuForwardBackend.splitMode(false, false));
        assertEquals(BatchSplitter.Mode.SIMPLE, CpuForwardBackend.splitMode(false, true));
        assertEquals(BatchSplitter.Mode.EQUAL, CpuForwardBackend.splitMode(true, false));
        assertEquals(BatchSplitter.Mode.SEQ, CpuForwardBackend.splitMode(true, true));
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

    /**
     * A prompt shared by a sequence copy ({@code llama_memory_seq_cp}) decodes as if each sequence had prefilled it:
     * the copy's partial page and the copy-on-write of the shared pages at the first decode both run on the host
     * before the trunk reads them, so both sequences' next rows match the contiguous run of the same tokens.
     */
    @Test
    public void aCopiedPromptDecodesInBothSequencesLikeTheContiguousRun() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        int[] a = tokens(4);
        int next = 9;
        int after = 17;
        float[] reference;
        float[] referenceAfter;
        try (GGUFModel one = new GGUFModel(model.toString(), config(false).build())) {
            one.forwardBackend().forwardRows(a, 0, new int[]{a.length - 1});
            reference = one.forwardBackend().forwardRows(new int[]{next}, a.length, new int[]{0})[0];
            referenceAfter = one.forwardBackend().forwardRows(new int[]{after}, a.length + 1, new int[]{0})[0];
        }
        try (GGUFModel two = new GGUFModel(model.toString(), config(true).build())) {
            HeapModelMemory mem = (HeapModelMemory) two.forwardBackend().memory();
            two.forwardBackend().forwardBatch(single(a));
            mem.pagedSequences().copy(SEQ_A, SEQ_B, 0, -1);
            float[][] rows = two.forwardBackend().forwardBatch(ForwardBatch.builder(2)
                    .add(next, N_TOKENS, SEQ_A, true).add(next, N_TOKENS, SEQ_B, true).build());
            assertArrayEquals("the source sequence continues its prompt", reference, rows[0], TOL);
            assertArrayEquals("the copy continues the shared prompt", reference, rows[1], TOL);
            float[][] later = two.forwardBackend().forwardBatch(ForwardBatch.builder(1)
                    .add(after, N_TOKENS + 1, SEQ_B, true).build());
            assertArrayEquals("the copy's own rows stay its own", referenceAfter, later[0], TOL);
            assertTrue("every page copy ran and was released", mem.pagedSequences().pendingCopies().isEmpty());
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

    @Test
    public void aDecodeStepOfTwoSequencesMatchesTheSingleSequenceDecodes() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        int[] a = tokens(1);
        int[] b = tokens(2);
        float[][] reference = new float[2][];
        try (GGUFModel m = new GGUFModel(model.toString(), config(true).build())) {
            ForwardBackend backend = m.forwardBackend();
            prefill(backend, a, SEQ_A);
            prefill(backend, b, SEQ_B);
            reference[0] = backend.forwardBatch(decodeRow(a[0], N_TOKENS, SEQ_A))[0];
            reference[1] = backend.forwardBatch(decodeRow(b[0], N_TOKENS, SEQ_B))[0];
        }
        try (GGUFModel m = new GGUFModel(model.toString(), config(true).build())) {
            ForwardBackend backend = m.forwardBackend();
            prefill(backend, a, SEQ_A);
            prefill(backend, b, SEQ_B);
            // one decode row per sequence in one batch: one ubatch holds both segments, so one trunk pass runs
            ForwardBatch.Builder batch = ForwardBatch.builder(2);
            batch.add(a[0], N_TOKENS, SEQ_A, true);
            batch.add(b[0], N_TOKENS, SEQ_B, true);
            float[][] out = backend.forwardBatch(batch.build());
            assertEquals(2, out.length);
            assertArrayEquals("sequence A decode matches its single-sequence decode", reference[0], out[0], TOL);
            assertArrayEquals("sequence B decode matches its single-sequence decode", reference[1], out[1], TOL);
        }
    }

    @Test
    public void aBatchOrderedAgainstTheSequenceOrderReturnsEachRowItsOwnLogits() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        int[] a = tokens(1);
        int[] b = tokens(2);
        float[][] reference = new float[2][];
        try (GGUFModel one = new GGUFModel(model.toString(), config(false).build())) {
            reference[0] = one.forwardBackend().forwardRows(a, 0, new int[]{a.length - 1})[0];
            one.forwardBackend().reset();
            reference[1] = one.forwardBackend().forwardRows(b, 0, new int[]{b.length - 1})[0];
        }
        try (GGUFModel two = new GGUFModel(model.toString(), config(true).build())) {
            // the batch rows hold sequence B before sequence A: the ubatches reorder them by sequence then
            // position, so an output row's batch order can be the reverse of its ubatch order
            ForwardBatch.Builder batch = ForwardBatch.builder(2 * N_TOKENS);
            for (int i = 0; i < N_TOKENS; i++) {
                batch.add(b[i], i, SEQ_B, i == N_TOKENS - 1);
            }
            for (int i = 0; i < N_TOKENS; i++) {
                batch.add(a[i], i, SEQ_A, i == N_TOKENS - 1);
            }
            float[][] out = two.forwardBackend().forwardBatch(batch.build());
            assertEquals(2, out.length);
            // batch row 11 is sequence B's last row, batch row 23 sequence A's: the output rows keep batch order
            assertArrayEquals("sequence B logits land on sequence B's row", reference[1], out[0], TOL);
            assertArrayEquals("sequence A logits land on sequence A's row", reference[0], out[1], TOL);
        }
    }

    @Test
    public void hiddenTapsCoverEveryRowAcrossTheUbatchSplits() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        int[] a = tokens(1);
        int[] b = tokens(2);
        float[][] reference;
        try (GGUFModel m = new GGUFModel(model.toString(), config(true).build())) {
            ForwardBackend backend = m.forwardBackend();
            CollectingTap tap = new CollectingTap();
            backend.setHiddenTap(tap);
            prefill(backend, a, SEQ_A);
            float[][] tapA = tap.take(0);
            prefill(backend, b, SEQ_B);
            float[][] tapB = tap.take(0);
            reference = new float[2 * N_TOKENS][];
            for (int i = 0; i < N_TOKENS; i++) {
                reference[2 * i] = tapA[i];
                reference[2 * i + 1] = tapB[i];
            }
        }
        try (GGUFModel m = new GGUFModel(model.toString(), config(true).build())) {
            ForwardBackend backend = m.forwardBackend();
            CollectingTap tap = new CollectingTap();
            backend.setHiddenTap(tap);
            backend.forwardBatch(interleave(a, b));
            float[][] rows = tap.take(0);
            assertEquals(2 * N_TOKENS, rows.length);
            assertEquals(tap.width(0), tap.width(2));
            // batch order is a0, b0, a1, b1, ...: the interleaved row of each sequence matches its own run
            for (int i = 0; i < 2 * N_TOKENS; i++) {
                assertArrayEquals("tap row " + i, reference[i], rows[i], TOL);
            }
        }
    }

    /** Runs one single-sequence prefill batch (output on the last row); its logits are discarded. */
    private static void prefill(ForwardBackend backend, int[] tokens, int seq) {
        ForwardBatch.Builder batch = ForwardBatch.builder(tokens.length);
        for (int i = 0; i < tokens.length; i++) {
            batch.add(tokens[i], i, seq, i == tokens.length - 1);
        }
        backend.forwardBatch(batch.build());
    }

    /** A one-row decode batch of {@code token} at {@code pos} (the output row). */
    private static ForwardBatch decodeRow(int token, int pos, int seq) {
        ForwardBatch.Builder batch = ForwardBatch.builder(1);
        batch.add(token, pos, seq, true);
        return batch.build();
    }

    /** Collects the rows of every wanted tap point of the calls it observes. */
    private static final class CollectingTap implements HiddenTap {
        private final java.util.Map<Integer, float[][]> rows = new java.util.HashMap<>();
        private final java.util.Map<Integer, Integer> widths = new java.util.HashMap<>();

        @Override
        public boolean wants(int layer) {
            return layer == 0 || layer == 2;
        }

        @Override
        public void onHidden(int layer, float[] data, int nRows, int width) {
            float[][] r = new float[nRows][];
            for (int i = 0; i < nRows; i++) {
                r[i] = java.util.Arrays.copyOfRange(data, i * width, (i + 1) * width);
            }
            rows.put(layer, r);
            widths.put(layer, width);
        }

        /** The rows of tap point {@code layer} since the last {@code take}. */
        float[][] take(int layer) {
            return rows.remove(layer);
        }

        int width(int layer) {
            return widths.get(layer);
        }
    }
}

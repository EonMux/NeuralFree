package com.eonmux.neuralfree.gguf.cpu.forward;

import com.eonmux.neuralfree.gguf.GGUFModel;
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
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;

/**
 * Multi-sequence (paged) batch steps over the DSA sparse mixer: {@code CpuForwardBackend.forwardBatch} runs the
 * synthetic mistral-large4 (MLA latent attention with a lightning indexer on every layer, layer 1 reading layer 0's
 * latent cache, top-k 8 of up to 13 rows) through the paged memory, where the indexer keys sit at the step's cells and
 * every segment selects among its own sequence's slots only. The batch logits equal the same tokens run one sequence
 * at a time on the contiguous path, for an interleaved prompt batch and the decode round after it; a non-causal batch
 * selects without the causal mask, and its two sequences still never see each other.
 */
public class CpuForwardBatchSparseTest {

    private static final int N_TOKENS = 12;
    private static final int SEQ_A = 0;
    private static final int SEQ_B = 1;
    private static final float TOL = 1e-3f;

    private static EngineConfig config(boolean paged) {
        MemoryConfig.Builder mc = MemoryConfig.builder().nCtx(64).ubatch(8).typeK(KvDType.F32).typeV(KvDType.F32);
        if (paged) {
            mc = mc.parallel(2);
        }
        return EngineConfig.builder().memory(mc.build()).build();
    }

    private static int[] tokens(int seed, int n) {
        int[] t = new int[n];
        for (int i = 0; i < t.length; i++) {
            t[i] = 4 + (seed * 7 + i * 13) % 250;
        }
        return t;
    }

    /** A batch whose rows interleave two prompts; the last row of each sequence is its output row. */
    private static ForwardBatch interleave(int[] a, int[] b, boolean nonCausal) {
        ForwardBatch.Builder batch = ForwardBatch.builder(a.length + b.length);
        for (int i = 0; i < a.length; i++) {
            batch.add(a[i], i, SEQ_A, i == a.length - 1);
            batch.add(b[i], i, SEQ_B, i == b.length - 1);
        }
        if (nonCausal) {
            batch.nonCausal();
        }
        return batch.build();
    }

    /** One sequence's prompt as a batch of its own; the last row is the output row. */
    private static ForwardBatch single(int[] t, int seq, boolean nonCausal) {
        ForwardBatch.Builder batch = ForwardBatch.builder(t.length);
        for (int i = 0; i < t.length; i++) {
            batch.add(t[i], i, seq, i == t.length - 1);
        }
        if (nonCausal) {
            batch.nonCausal();
        }
        return batch.build();
    }

    /** The contiguous logits of the last row of {@code t}, after a reset. */
    private static float[] contiguous(GGUFModel model, int[] t) {
        model.forwardBackend().reset();
        return model.forwardBackend().forwardRows(t, 0, new int[]{t.length - 1})[0];
    }

    @Test
    public void dsaPagedBatchMatchesTheContiguousRuns() throws IOException {
        Path model = SyntheticModels.deepseek32().writeTempFile();
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
            assertNotNull(mem.pagedSequences());
            assertNotNull("the synthetic model keeps a DSA indexer cache", mem.indexer(0));
            float[][] out = two.forwardBackend().forwardBatch(interleave(a, b, false));
            assertEquals(2, out.length);
            assertArrayEquals("sequence A logits match the contiguous run", refA, out[0], TOL);
            assertArrayEquals("sequence B logits match the contiguous run", refB, out[1], TOL);

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
            assertArrayEquals("the decode round continues sequence A's paged caches", refNextA, decoded[0], TOL);
            assertArrayEquals("the decode round continues sequence B's paged caches", refNextB, decoded[1], TOL);
        }
    }

    @Test
    public void nonCausalDsaBatchKeepsTheSequencesApart() throws IOException {
        Path model = SyntheticModels.deepseek32().writeTempFile();
        int[] a = tokens(3, 4);
        int[] b = tokens(4, 4);
        float[] aloneA;
        float[] aloneB;
        float[] causalA;
        try (GGUFModel one = new GGUFModel(model.toString(), config(true))) {
            aloneA = one.forwardBackend().forwardBatch(single(a, SEQ_A, true))[0];
            one.forwardBackend().reset();
            aloneB = one.forwardBackend().forwardBatch(single(b, SEQ_B, true))[0];
            one.forwardBackend().reset();
            causalA = one.forwardBackend().forwardBatch(single(a, SEQ_A, false))[0];
        }
        assertFalse("a non-causal step lets the earlier rows select the later ones",
                java.util.Arrays.equals(aloneA, causalA));
        try (GGUFModel two = new GGUFModel(model.toString(), config(true))) {
            float[][] out = two.forwardBackend().forwardBatch(interleave(a, b, true));
            assertArrayEquals("sequence A's non-causal rows match its own batch", aloneA, out[0], TOL);
            assertArrayEquals("sequence B's non-causal rows match its own batch", aloneB, out[1], TOL);
        }
    }

    /**
     * A forked sequence (seq_cp of a prompt that ends inside a page) decodes into its own copy of the shared partial
     * page: the copy-on-write page copy carries the indexer keys with the latent rows, so both sequences' next rows
     * equal the contiguous runs.
     */
    @Test
    public void aForkedDsaSequenceDecodesOverItsCopiedIndexerRows() throws IOException {
        Path model = SyntheticModels.deepseek32().writeTempFile();
        int[] prompt = tokens(5, 11);
        int nextA = 17;
        int nextB = 29;
        float[] refA;
        float[] refB;
        try (GGUFModel one = new GGUFModel(model.toString(), config(false))) {
            refA = contiguous(one, append(prompt, nextA));
            refB = contiguous(one, append(prompt, nextB));
        }
        try (GGUFModel two = new GGUFModel(model.toString(), config(true))) {
            two.forwardBackend().forwardBatch(single(prompt, SEQ_A, false));
            HeapModelMemory mem = (HeapModelMemory) two.forwardBackend().memory();
            assertEquals(null, mem.sequences().copyRefusal());
            mem.sequences().copy(SEQ_A, SEQ_B, 0, prompt.length);
            ForwardBatch.Builder round = ForwardBatch.builder(2);
            round.add(nextA, prompt.length, SEQ_A, true);
            round.add(nextB, prompt.length, SEQ_B, true);
            float[][] out = two.forwardBackend().forwardBatch(round.build());
            assertArrayEquals("the source sequence continues its rows", refA, out[0], TOL);
            assertArrayEquals("the fork decodes over its copied latent and indexer rows", refB, out[1], TOL);
        }
    }

    private static int argmax(float[] logits) {
        int best = 0;
        for (int i = 1; i < logits.length; i++) {
            if (logits[i] > logits[best]) {
                best = i;
            }
        }
        return best;
    }

    private static int[] append(int[] prompt, int token) {
        int[] t = java.util.Arrays.copyOf(prompt, prompt.length + 1);
        t[prompt.length] = token;
        return t;
    }
}

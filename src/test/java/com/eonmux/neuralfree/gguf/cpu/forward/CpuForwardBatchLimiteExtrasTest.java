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
 * Batched steps through the limite extras: the synthetic limite carries the
 * MUDD dense connections and the learned residual scales, whose hooks fold one step row at a time over the bank of
 * earlier layers' inputs. {@code CpuForwardBackend.forwardBatch} runs it over the paged memory, where an ubatch
 * holds both sequences' rows ({@code split_simple}, the model keeps no recurrent state), and the batch logits equal
 * the same tokens run one sequence at a time on the contiguous path.
 */
public class CpuForwardBatchLimiteExtrasTest {

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

    /** The tokens a sequence runs (deterministic ids inside the synthetic vocab, as CpuForwardBatchRecurrentTest). */
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

    private static float[][] reference(Path model, EngineConfig cfg, int[] a, int[] b) throws IOException {
        try (GGUFModel one = new GGUFModel(model.toString(), cfg)) {
            float[][] out = new float[][]{
                    one.forwardBackend().forwardRows(a, 0, new int[]{a.length - 1})[0], null};
            one.forwardBackend().reset();
            out[1] = one.forwardBackend().forwardRows(b, 0, new int[]{b.length - 1})[0];
            return out;
        }
    }

    @Test
    public void limitePagedBatchMatchesTheContiguousRuns() throws IOException {
        Path model = SyntheticModels.limite().writeTempFile();
        int[] a = tokens(1);
        int[] b = tokens(2);
        float[][] ref = reference(model, config(false).build(), a, b);
        try (GGUFModel two = new GGUFModel(model.toString(), config(true).build())) {
            HeapModelMemory mem = (HeapModelMemory) two.forwardBackend().memory();
            assertNotNull(mem.pagedSequences());
            assertFalse("limite is attention-only, the extras run beside paged KV", mem.plan().hasRecurrentState());
            float[][] out = two.forwardBackend().forwardBatch(interleave(a, b));
            assertEquals(2, out.length);
            assertArrayEquals("sequence A logits match the contiguous run", ref[0], out[0], TOL);
            assertArrayEquals("sequence B logits match the contiguous run", ref[1], out[1], TOL);
        }
    }
}

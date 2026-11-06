package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Path;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link GGUFModel#encodeBatch(int[][])}: a paged memory ({@code neuralfree.parallel} slots) runs the sequences of
 * one batch as one multi-sequence forward and answers the pooled rows the sequential {@code encode} loop answers,
 * one output row per sequence, the RANK classifier scores included. A memoryless encoder (the BERT family plans no
 * memory) embeds its sequences one prefill at a time through the same entry point, with the same rows.
 */
public class GGUFModelBatchedEncodeTest {

    private static int[] tokens(int seed, int n) {
        int[] t = new int[n];
        for (int i = 0; i < n; i++) {
            t[i] = 8 + (seed * 5 + i * 3) % 180;
        }
        return t;
    }

    private static EngineConfig paged(int parallel) {
        return EngineConfig.builder().backend(EngineConfig.BackendMode.CPU)
                .memory(MemoryConfig.builder().nCtx(64).ubatch(8).parallel(parallel)
                        .typeK(KvDType.F32).typeV(KvDType.F32).build()).build();
    }

    @Test
    public void batchedRowsEqualTheSequentialRows() throws IOException {
        Path model = SyntheticModels.bert().writeTempFile();
        int[][] prompts = {tokens(1, 6), tokens(2, 4), tokens(3, 8)};
        try (GGUFModel sequential = new GGUFModel(model.toString(), paged(1));
             GGUFModel batched = new GGUFModel(model.toString(), paged(3))) {
            float[][] expected = new float[prompts.length][];
            for (int i = 0; i < prompts.length; i++) {
                expected[i] = sequential.encode(prompts[i]);
            }
            float[][] rows = batched.encodeBatch(prompts);
            assertEquals(prompts.length, rows.length);
            for (int i = 0; i < prompts.length; i++) {
                assertArrayEquals("sequence " + i + " pools to its own row", expected[i], rows[i], 1e-6f);
            }
            float[][] single = batched.encodeBatch(new int[][]{prompts[1]});
            assertArrayEquals(expected[1], single[0], 1e-6f);
        }
    }

    private static EngineConfig pooledLlama(int parallel, int nCtx) {
        return EngineConfig.builder().backend(EngineConfig.BackendMode.CPU)
                .pooling(com.eonmux.neuralfree.gguf.arch.PoolingKind.MEAN)
                .memory(MemoryConfig.builder().nCtx(nCtx).ubatch(8).parallel(parallel)
                        .typeK(KvDType.F32).typeV(KvDType.F32).build()).build();
    }

    /**
     * The real multi-sequence path: a causal model plans a paged memory ({@code neuralfree.parallel} slots), and
     * its pooling override turns it into an embedding model whose sequences run as multi-sequence forwards, more
     * sequences than the memory holds splitting into several, every row the contiguous model's.
     */
    @Test
    public void aCausalModelBatchSequencesThroughThePagedMemory() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        int[][] prompts = {tokens(1, 6), tokens(2, 4), tokens(3, 5)};
        try (GGUFModel contiguous = new GGUFModel(model.toString(), pooledLlama(1, 64));
             GGUFModel m = new GGUFModel(model.toString(), pooledLlama(2, 64))) {
            assertEquals(LLMModel.PoolingType.MEAN, m.poolingType());
            assertEquals(2, m.forwardBackend().maxSequences());
            float[][] rows = m.encodeBatch(prompts);
            assertEquals(3, rows.length);
            for (int i = 0; i < prompts.length; i++) {
                assertArrayEquals("sequence " + i + " pools as its own forward",
                        contiguous.encode(prompts[i]), rows[i], 1e-5f);
            }
        }
    }

    /**
     * The batches pack to the memory's pages: three 10-token sequences take 16-row pages each, so a 32-row context
     * runs them as two forwards; a sequence longer than one sequence's context refuses by name before any runs.
     */
    @Test
    public void batchesPackToTheMemoryContext() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        int[][] prompts = {tokens(4, 10), tokens(5, 10), tokens(6, 10)};
        try (GGUFModel contiguous = new GGUFModel(model.toString(), pooledLlama(1, 32));
             GGUFModel m = new GGUFModel(model.toString(), pooledLlama(3, 32))) {
            float[][] rows = m.encodeBatch(prompts);
            for (int i = 0; i < prompts.length; i++) {
                assertArrayEquals("sequence " + i, contiguous.encode(prompts[i]), rows[i], 1e-5f);
            }
            IllegalArgumentException tooLong = assertThrows(IllegalArgumentException.class,
                    () -> m.encodeBatch(new int[][]{tokens(1, 4), tokens(2, 40)}));
            assertTrue(tooLong.getMessage(), tooLong.getMessage().startsWith("sequence 1 holds 40 tokens"));
            assertArrayEquals("the memory is free after the refusal", rows[0],
                    m.encodeBatch(new int[][]{prompts[0]})[0], 1e-5f);
        }
    }

    @Test
    public void rankBatchesTheClassifierScores() throws IOException {
        Path rank = SyntheticModels.rank("neg", "pos").writeTempFile();
        int[][] prompts = {tokens(6, 5), tokens(7, 7)};
        try (GGUFModel sequential = new GGUFModel(rank.toString(), paged(1));
             GGUFModel batched = new GGUFModel(rank.toString(), paged(2))) {
            float[][] expected = new float[prompts.length][];
            for (int i = 0; i < prompts.length; i++) {
                expected[i] = sequential.encode(prompts[i]);
                assertEquals(2, expected[i].length);
            }
            float[][] rows = batched.encodeBatch(prompts);
            for (int i = 0; i < prompts.length; i++) {
                assertArrayEquals(expected[i], rows[i], 1e-6f);
            }
        }
    }

    @Test
    public void nonePoolingRefusesTheBatchByName() throws IOException {
        com.eonmux.neuralfree.gguf.testutil.GgufTestWriter w = SyntheticModels.bert();
        String key = w.archKey("pooling_type");
        Path none = w.remove(key).u32(key, 0).writeTempFile();
        try (GGUFModel m = new GGUFModel(none.toString(), paged(2))) {
            UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class,
                    () -> m.encodeBatch(new int[][]{tokens(1, 3), tokens(2, 3)}));
            assertTrue(e.getMessage(), e.getMessage().contains("NONE"));
        }
    }

    @Test
    public void emptyAndNullSequencesRefuse() throws IOException {
        Path model = SyntheticModels.bert().writeTempFile();
        try (GGUFModel m = new GGUFModel(model.toString(), paged(2))) {
            assertThrows(IllegalArgumentException.class, () -> m.encodeBatch(new int[][]{tokens(1, 3), {}}));
            assertThrows(IllegalArgumentException.class, () -> m.encodeBatch(new int[0][]));
            assertThrows(NullPointerException.class, () -> m.encodeBatch(null));
        }
    }
}

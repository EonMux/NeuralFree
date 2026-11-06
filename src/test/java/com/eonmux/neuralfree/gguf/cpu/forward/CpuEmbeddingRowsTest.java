package com.eonmux.neuralfree.gguf.cpu.forward;

import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.ForwardBatch;
import com.eonmux.neuralfree.gguf.engine.PagedEvaluations;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Embedding-row inputs on the CPU backend (gap prompt-embeds-input): prefilling the token table's own rows
 * computes the same logits as prefilling the tokens themselves (llama.cpp builds the input stage so provided
 * rows replace the lookup while every other step runs unchanged), a paged batch of embedding rows matches the
 * contiguous run, a ubatch mixing token and embedding rows matches the same rows as tokens, the rows record no token
 * id at their positions, a row of another width than {@code n_embd} refuses by name, and so does a model whose blocks read token ids.
 */
public class CpuEmbeddingRowsTest {

    private static final int N_TOKENS = 6;
    private static final int SEQ_A = 0;
    private static final float TOL = 1e-4f;

    private static EngineConfig.Builder config(boolean paged) {
        MemoryConfig.Builder mc = MemoryConfig.builder().nCtx(64).ubatch(8).typeK(KvDType.F32).typeV(KvDType.F32);
        if (paged) {
            mc = mc.parallel(2);
        }
        return EngineConfig.builder().memory(mc.build());
    }

    /** Deterministic token ids inside the synthetic vocab, as CpuForwardBatchTest picks them. */
    private static int[] tokens(int seed) {
        int[] t = new int[N_TOKENS];
        for (int i = 0; i < t.length; i++) {
            t[i] = 4 + (seed * 7 + i * 13) % 250;
        }
        return t;
    }

    /** The token table's rows of {@code tokens}: one {@code n_embd} row per id, read from the GGUF file. */
    private static float[][] tableRows(Path model, int[] tokens, int nEmbd) throws IOException {
        try (GGUFFileParser parser = new GGUFFileParser(model.toString())) {
            GGUFTensorInfo table = parser.getTensor("token_embd.weight");
            float[] flat = parser.readTensorDataAsFloat(table);
            assertEquals("the table holds one row per vocabulary entry",
                    flat.length, nEmbd * table.dims()[1]);
            float[][] rows = new float[tokens.length][];
            for (int i = 0; i < rows.length; i++) {
                assertTrue("token id " + tokens[i] + " inside the table", tokens[i] < table.dims()[1]);
                rows[i] = Arrays.copyOfRange(flat, tokens[i] * nEmbd, (tokens[i] + 1) * nEmbd);
            }
            return rows;
        }
    }

    @Test
    public void theTokenTablesRowsProduceTheTokenLogits() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        try (GGUFModel m = new GGUFModel(model.toString(), config(false).build())) {
            int[] tokens = tokens(1);
            float[] reference = m.forwardBackend().forwardRows(tokens, 0, new int[]{tokens.length - 1})[0];
            m.forwardBackend().reset();
            float[] logits = m.forwardBackend()
                    .prefillEmbeddings(tableRows(model, tokens, m.embeddingLength()), 0);
            assertArrayEquals("rows of the table embed like the tokens they came from", reference, logits, TOL);
        }
    }

    @Test
    public void aPagedBatchOfEmbeddingRowsMatchesTheContiguousRun() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        int[] tokens = tokens(2);
        float[] reference;
        try (GGUFModel one = new GGUFModel(model.toString(), config(false).build())) {
            reference = one.forwardBackend().forwardRows(tokens, 0, new int[]{tokens.length - 1})[0];
        }
        try (GGUFModel two = new GGUFModel(model.toString(), config(true).build())) {
            float[][] rows = tableRows(model, tokens, two.embeddingLength());
            ForwardBatch.Builder batch = ForwardBatch.builder(rows.length);
            for (int i = 0; i < rows.length; i++) {
                batch.addEmbedding(rows[i], i, SEQ_A, i == rows.length - 1);
            }
            float[][] out = two.forwardBackend().forwardBatch(batch.build());
            assertEquals(1, out.length);
            assertArrayEquals("the paged embedding ubatch matches the contiguous run", reference, out[0], TOL);
            com.eonmux.neuralfree.gguf.memory.cpu.HeapModelMemory mem =
                    (com.eonmux.neuralfree.gguf.memory.cpu.HeapModelMemory) two.forwardBackend().memory();
            assertEquals("the rows hold their positions", rows.length - 1, mem.sequences().posMax(SEQ_A));
            com.eonmux.neuralfree.gguf.memory.SequenceState record = mem.pagedSequences().sequence(SEQ_A);
            int[] noIds = new int[rows.length];
            Arrays.fill(noIds, -1);
            assertArrayEquals("one record entry per row, an embedding row recording no id (LLAMA_TOKEN_NULL)",
                    noIds, record.tokens());
            for (int i = 0; i < rows.length; i++) {
                assertEquals("row " + i + " records its position", i, record.position(i));
            }
        }
    }

    @Test
    public void aRowOfAnotherWidthRefusesByName() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        try (GGUFModel m = new GGUFModel(model.toString(), config(false).build())) {
            float[][] wrong = new float[][] {new float[m.embeddingLength() + 1]};
            IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                    () -> m.forwardBackend().prefillEmbeddings(wrong, 0));
            assertTrue(e.getMessage(), e.getMessage().contains("n_embd"));
        }
    }

    @Test
    public void aRowsPrefillAlignsTheMemoryLikeATokenPrefill() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        try (GGUFModel m = new GGUFModel(model.toString(), config(false).build())) {
            float[][] rows = tableRows(model, Arrays.copyOf(tokens(3), 3), m.embeddingLength());
            m.forwardBackend().prefillEmbeddings(rows, 0);
            assertEquals(3, m.forwardBackend().memory().sequence().nPast());
            // a rows prefill that re-evaluates the stored positions rolls back first, as a token prefill does
            m.forwardBackend().prefillEmbeddings(rows, 0);
            assertEquals(3, m.forwardBackend().memory().sequence().nPast());
        }
    }

    @Test
    public void aUbatchMixingTokenAndEmbeddingRowsMatchesTheTokenBatch() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        int[] a = tokens(4);
        int[] b = tokens(5);
        try (GGUFModel m = new GGUFModel(model.toString(), config(true).build())) {
            float[][] rowsA = tableRows(model, a, m.embeddingLength());
            float[][] tokenOut = m.forwardBackend().forwardBatch(interleaved(a, b, null));
            m.forwardBackend().memory().sequences().remove(SEQ_A, 0, -1);
            m.forwardBackend().memory().sequences().remove(SEQ_A + 1, 0, -1);
            float[][] mixedOut = m.forwardBackend().forwardBatch(interleaved(a, b, rowsA));
            assertEquals(2, mixedOut.length);
            for (int s = 0; s < 2; s++) {
                assertArrayEquals("sequence " + s + " of the mixed ubatch", tokenOut[s], mixedOut[s], TOL);
            }
        }
    }

    /** Sequences 0 and 1 interleaved row by row; sequence 0 as {@code rowsA} when given, its tokens otherwise. */
    private static ForwardBatch interleaved(int[] a, int[] b, float[][] rowsA) {
        ForwardBatch.Builder batch = ForwardBatch.builder(a.length + b.length);
        for (int i = 0; i < a.length; i++) {
            boolean last = i == a.length - 1;
            if (rowsA != null) {
                batch.addEmbedding(rowsA[i], i, SEQ_A, last);
            } else {
                batch.add(a[i], i, SEQ_A, last);
            }
            batch.add(b[i], i, SEQ_A + 1, last);
        }
        return batch.build();
    }

    @Test
    public void aHiddenBatchMatchesTheContiguousHiddenRows() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        int[] tokens = tokens(6);
        float[][] reference;
        try (GGUFModel one = new GGUFModel(model.toString(), config(false).build())) {
            reference = one.forwardBackend().hiddenRows(tokens, null);
        }
        try (GGUFModel two = new GGUFModel(model.toString(), config(true).build())) {
            float[][] paged = PagedEvaluations.hiddenRows(two.forwardBackend(), tokens);
            assertEquals(reference.length, paged.length);
            for (int i = 0; i < paged.length; i++) {
                assertArrayEquals("hidden row " + i + " of the paged batch", reference[i], paged[i], TOL);
            }
            assertEquals("the evaluation frees its sequence", -1,
                    two.forwardBackend().memory().sequences().posMax(SEQ_A));
        }
    }

    @Test
    public void aRowOutsideAdapterSlotZeroRefusesByName() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        try (GGUFModel m = new GGUFModel(model.toString(), config(true).build())) {
            ForwardBatch batch = ForwardBatch.builder(2).add(5, 0, SEQ_A, false).add(6, 1, SEQ_A, true)
                    .adapterSlots(new int[]{0, 1}).build();
            UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class,
                    () -> m.forwardBackend().forwardBatch(batch));
            assertTrue(e.getMessage(), e.getMessage().contains("adapterSlots") && e.getMessage().contains("slot 1"));
        }
    }

    @Test
    public void aModelReadingTokenIdsRefusesEmbeddingRowsByName() throws IOException {
        Path model = SyntheticModels.limite().writeTempFile();
        try (GGUFModel m = new GGUFModel(model.toString(), config(true).build())) {
            String refusal = m.forwardBackend().embeddingRowsRefusal();
            assertNotNull("value embeddings read the token ids", refusal);
            assertTrue(refusal, refusal.contains("value embeddings"));
            float[][] rows = {new float[m.embeddingLength()]};
            UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class,
                    () -> m.forwardBackend().forwardBatch(ForwardBatch.builder(1)
                            .addEmbedding(rows[0], 0, SEQ_A, true).build()));
            assertEquals(refusal, e.getMessage());
        }
    }
}

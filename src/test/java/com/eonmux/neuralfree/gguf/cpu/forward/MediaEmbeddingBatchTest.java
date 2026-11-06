package com.eonmux.neuralfree.gguf.cpu.forward;

import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.ForwardBatch;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.engine.HiddenTap;
import com.eonmux.neuralfree.gguf.memory.ModelMemory;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.memory.cpu.HeapModelMemory;
import com.eonmux.neuralfree.gguf.ops.rope.MediaRopePositions;
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
 * The engine path for media embeddings (llama.cpp {@code mtmd_helper_decode_image_chunk}): an embedding batch
 * runs over the paged memory with its rows written by the embedding stage, an image chunk's rows share the
 * chunk's base position (so every row of the chunk attends the whole chunk, as the shared temporal position makes
 * the causal mask do in llama.cpp), its explicit M-RoPE sections reach the rope through the position slot, the
 * sequence's position cursor advances by the chunk's n_pos ({@code max(nx, ny)}), and a non-causal batch lets its
 * rows attend each other. The synthetic qwen2vl (M-RoPE), llama (single-section) and mamba2 (rope-less) models run
 * paged.
 */
public class MediaEmbeddingBatchTest {

    private static final int SEQ = 0;
    private static final int N_EMBD = 64;
    private static final int TEXT = 4;
    private static final float TOL = 1e-4f;

    private static EngineConfig config() {
        return EngineConfig.builder().memory(MemoryConfig.builder().nCtx(64).ubatch(8)
                .typeK(KvDType.F32).typeV(KvDType.F32).parallel(2).build()).build();
    }

    /** Media rows of distinct values, the model input width. */
    private static float[][] mediaRows(int n, float v) {
        float[][] r = new float[n][N_EMBD];
        for (int i = 0; i < n; i++) {
            for (int d = 0; d < N_EMBD; d++) {
                r[i][d] = v * (1 + 0.05f * i);
            }
        }
        return r;
    }

    /** One media chunk as one embedding batch, logits of its {@code outputRow}. */
    private static float[] mediaChunkLogits(GGUFModel model, MediaRopePositions p, float[][] rows,
                                            boolean nonCausal, int outputRow) {
        ForwardBatch.Builder b = ForwardBatch.builder(rows.length);
        for (int i = 0; i < rows.length; i++) {
            b.addEmbedding(rows[i], p.position(0, i), SEQ, i == outputRow);
        }
        if (p.kind() != MediaRopePositions.Kind.NORMAL) {
            b.ropePositions(p.sections());
        }
        if (nonCausal) {
            b.nonCausal();
        }
        return model.forwardBackend().forwardBatch(b.build())[0];
    }

    private static void runText(ForwardBackend backend, int from, int n, boolean outputLast) {
        ForwardBatch.Builder b = ForwardBatch.builder(n);
        for (int i = 0; i < n; i++) {
            b.add(TEXT + (i % 9), from + i, SEQ, outputLast && i == n - 1);
        }
        backend.forwardBatch(b.build());
    }

    @Test
    public void anImageChunkSharesItsBasePositionAndLeavesTheCursorAtNPos() throws IOException {
        Path model = SyntheticModels.qwen2vl().writeTempFile();
        try (GGUFModel m = new GGUFModel(model.toString(), config())) {
            assertEquals("qwen2vl", m.spec().arch().ggufName());
            ForwardBackend backend = m.forwardBackend();
            HeapModelMemory mem = (HeapModelMemory) backend.memory();
            runText(backend, 0, 5, false);
            // a 4 x 4 image at base 5: 16 rows, 4 temporal positions; the batch spans three ubatches
            mediaChunkLogits(m, MediaRopePositions.image2d(5, 4, 4), mediaRows(16, 0.25f), false, 0);
            runText(backend, 9, 3, true);
            assertEquals(0, mem.sequences().posMin(SEQ));
            assertEquals("the cursor advanced by the chunk's n_pos, not its 16 rows",
                    11, mem.sequences().posMax(SEQ));
        }
    }

    @Test
    public void aUbatchSplitMediaChunkComputesTheRowsItsUbatchRuns() throws IOException {
        Path model = SyntheticModels.qwen2vl().writeTempFile();
        float[] wholeBatch;
        float[] chunkPerUbatch;
        try (GGUFModel m = new GGUFModel(model.toString(), config())) {
            ForwardBackend backend = m.forwardBackend();
            runText(backend, 0, 5, false);
            // one 16-row media batch: the backend splits it into ubatches of 8 and each ubatch must embed its
            // own rows
            MediaRopePositions p = MediaRopePositions.image2d(5, 4, 4);
            float[][] rows = mediaRows(16, 0.25f);
            ForwardBatch.Builder b = ForwardBatch.builder(16);
            for (int i = 0; i < rows.length; i++) {
                b.addEmbedding(rows[i], p.position(0, i), SEQ, false);
            }
            b.ropePositions(p.sections());
            backend.forwardBatch(b.build());
            runText(backend, 9, 3, true);
            wholeBatch = lastTokenLogits(backend, TEXT + 2, 12);
            backend.reset();
            // the same run with the chunk submitted as one batch per ubatch run
            runText(backend, 0, 5, false);
            for (int off = 0; off < 16; off += 8) {
                ForwardBatch.Builder part = ForwardBatch.builder(8);
                for (int i = 0; i < 8; i++) {
                    part.addEmbedding(rows[off + i], p.position(0, off + i), SEQ, false);
                }
                part.ropePositions(slice(p.sections(), off, 8));
                backend.forwardBatch(part.build());
            }
            runText(backend, 9, 3, true);
            chunkPerUbatch = lastTokenLogits(backend, TEXT + 2, 12);
        }
        assertArrayEquals("a split media chunk embeds the same rows as ubatch-sized chunks",
                wholeBatch, chunkPerUbatch, 1e-4f);
    }

    /** One decode token as a one-row batch; returns its logits. */
    private static float[] lastTokenLogits(ForwardBackend backend, int token, int pos) {
        ForwardBatch.Builder b = ForwardBatch.builder(1);
        b.add(token, pos, SEQ, true);
        return backend.forwardBatch(b.build())[0];
    }

    private static int[][] slice(int[][] sections, int off, int n) {
        int[][] out = new int[sections.length][];
        for (int s = 0; s < sections.length; s++) {
            out[s] = java.util.Arrays.copyOfRange(sections[s], off, off + n);
        }
        return out;
    }

    @Test
    public void sharedBasePositionsMakeTheChunkRowsAttendEachOther() throws IOException {
        Path model = SyntheticModels.qwen2vl().writeTempFile();
        float[] shared;
        float[] distinct;
        try (GGUFModel m = new GGUFModel(model.toString(), config())) {
            // the same 4 image rows after the same 5 text tokens: once at the shared base 5 (cursor 9), once at
            // positions 5..8 (cursor 9). Image row 0 sees rows 1..3 only when they share its position.
            runText(m.forwardBackend(), 0, 5, false);
            shared = mediaChunkLogits(m, MediaRopePositions.image2d(5, 4, 1), mediaRows(4, 1), false, 0);
            m.forwardBackend().reset();
            runText(m.forwardBackend(), 0, 5, false);
            distinct = mediaChunkLogits(m, MediaRopePositions.normal(5, 4), mediaRows(4, 1), false, 0);
        }
        boolean differs = false;
        for (int i = 0; i < shared.length; i++) {
            if (Math.abs(shared[i] - distinct[i]) > 1e-6f) {
                differs = true;
                break;
            }
        }
        assertTrue("shared base positions let the first image row attend the later ones", differs);
    }

    @Test
    public void aNonCausalChunkAttendsItsWholeSelfWhileACausalOneDoesNot() throws IOException {
        Path model = SyntheticModels.qwen2vl().writeTempFile();
        float[] causal;
        float[] nonCausal;
        try (GGUFModel m = new GGUFModel(model.toString(), config())) {
            // two media rows at distinct positions with distinct values: row 0 attends row 1 only without the
            // causal mask
            causal = mediaChunkLogits(m, MediaRopePositions.normal(0, 2), mediaRows(2, 1), false, 0);
            m.forwardBackend().reset();
            nonCausal = mediaChunkLogits(m, MediaRopePositions.normal(0, 2), mediaRows(2, 1), true, 0);
        }
        boolean differs = false;
        for (int i = 0; i < causal.length; i++) {
            if (Math.abs(causal[i] - nonCausal[i]) > 1e-6f) {
                differs = true;
                break;
            }
        }
        assertTrue("the non-causal mask changes the logits of a chunk whose rows hold distinct positions", differs);
    }

    @Test
    public void explicitTextSectionsReproduceTheTextPatternLogits() throws IOException {
        Path model = SyntheticModels.qwen2vl().writeTempFile();
        int[] tokens = {4, 9, 14, 19};
        float[] pattern;
        float[] explicitSections;
        try (GGUFModel m = new GGUFModel(model.toString(), config())) {
            ForwardBatch.Builder b = ForwardBatch.builder(tokens.length);
            for (int i = 0; i < tokens.length; i++) {
                b.add(tokens[i], i, SEQ, i == tokens.length - 1);
            }
            pattern = m.forwardBackend().forwardBatch(b.build())[0];
            m.forwardBackend().reset();
            // the same run with the sections written out ([p, p, p, 0], the qwen2vl text pattern)
            int[][] sections = new int[4][tokens.length];
            for (int i = 0; i < tokens.length; i++) {
                sections[0][i] = i;
                sections[1][i] = i;
                sections[2][i] = i;
                sections[3][i] = 0;
            }
            ForwardBatch.Builder b2 = ForwardBatch.builder(tokens.length);
            for (int i = 0; i < tokens.length; i++) {
                b2.add(tokens[i], i, SEQ, i == tokens.length - 1);
            }
            b2.ropePositions(sections);
            explicitSections = m.forwardBackend().forwardBatch(b2.build())[0];
        }
        assertArrayEquals(pattern, explicitSections, TOL);
    }

    @Test
    public void aMediaChunkOnASingleSectionModelRunsNormalPositions() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        try (GGUFModel m = new GGUFModel(model.toString(), config())) {
            HeapModelMemory mem = (HeapModelMemory) m.forwardBackend().memory();
            float[] logits = mediaChunkLogits(m, MediaRopePositions.normal(0, 6), mediaRows(6, 0.5f), false, 5);
            assertEquals(m.tokenizer().vocabSize(), logits.length);
            assertEquals(5, mem.sequences().posMax(SEQ));
        }
    }

    @Test
    public void aMediaChunkOnARopelessModelRunsNormalPositions() throws IOException {
        Path model = SyntheticModels.mamba2().writeTempFile();
        try (GGUFModel m = new GGUFModel(model.toString(), config())) {
            HeapModelMemory mem = (HeapModelMemory) m.forwardBackend().memory();
            assertNotNull("the rope-less model plans a paged memory", mem.pagedSequences());
            float[] logits = mediaChunkLogits(m, MediaRopePositions.normal(0, 6), mediaRows(6, 0.5f), false, 5);
            assertEquals(m.tokenizer().vocabSize(), logits.length);
            assertEquals(5, mem.sequences().posMax(SEQ));
        }
    }

    /** The token table's rows of {@code tokens}, read from the GGUF file. */
    private static float[][] tableRows(Path model, int[] tokens) throws IOException {
        try (GGUFFileParser parser = new GGUFFileParser(model.toString())) {
            GGUFTensorInfo table = parser.getTensor("token_embd.weight");
            float[] flat = parser.readTensorDataAsFloat(table);
            int width = (int) table.dim(0);
            float[][] rows = new float[tokens.length][];
            for (int i = 0; i < tokens.length; i++) {
                rows[i] = java.util.Arrays.copyOfRange(flat, tokens[i] * width, (tokens[i] + 1) * width);
            }
            return rows;
        }
    }

    /**
     * The token table's rows of a prompt answer like their tokens through the rope-less paged model: the media
     * batch reproduces the token batch's logits (llama.cpp builds the mtmd embedding batch on
     * {@code build_inp_embd}'s output, so a table row is the token's input; no learned position or type table and
     * no input norm runs on mamba2).
     */
    @Test
    public void theTokenTableRowsAnswerLikeTheirTokensOnARopelessModel() throws IOException {
        Path model = SyntheticModels.mamba2().writeTempFile();
        int[] tokens = {4, 17, 30, 43, 56, 69};
        float[] asTokens;
        float[] asRows;
        try (GGUFModel m = new GGUFModel(model.toString(), config())) {
            ForwardBatch.Builder b = ForwardBatch.builder(tokens.length);
            for (int i = 0; i < tokens.length; i++) {
                b.add(tokens[i], i, SEQ, i == tokens.length - 1);
            }
            asTokens = m.forwardBackend().forwardBatch(b.build())[0];
            m.forwardBackend().reset();
            float[][] rows = tableRows(model, tokens);
            asRows = mediaChunkLogits(m, MediaRopePositions.normal(0, tokens.length), rows, false,
                    tokens.length - 1);
        }
        assertArrayEquals("the token table's rows answer like their tokens", asTokens, asRows, TOL);
    }

    @Test
    public void gridSectionsOnASingleSectionModelRefuseByName() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        try (GGUFModel m = new GGUFModel(model.toString(), config())) {
            MediaRopePositions p = MediaRopePositions.image2d(0, 2, 2);
            ForwardBatch.Builder b = ForwardBatch.builder(4);
            float[][] rows = mediaRows(4, 1);
            for (int i = 0; i < rows.length; i++) {
                b.addEmbedding(rows[i], p.position(0, i), SEQ, false);
            }
            b.ropePositions(p.sections());
            IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                    () -> m.forwardBackend().forwardBatch(b.build()));
            assertTrue(e.getMessage(), e.getMessage().contains("sections"));
        }
    }

    @Test
    public void gridSectionsOnARopelessModelRefuseByName() throws IOException {
        Path model = SyntheticModels.mamba2().writeTempFile();
        try (GGUFModel m = new GGUFModel(model.toString(), config())) {
            MediaRopePositions p = MediaRopePositions.image2d(0, 2, 2);
            ForwardBatch.Builder b = ForwardBatch.builder(4);
            float[][] rows = mediaRows(4, 1);
            for (int i = 0; i < rows.length; i++) {
                b.addEmbedding(rows[i], p.position(0, i), SEQ, false);
            }
            b.ropePositions(p.sections());
            IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                    () -> m.forwardBackend().forwardBatch(b.build()));
            assertTrue(e.getMessage(), e.getMessage().contains("sections"));
        }
    }

    @Test
    public void aWrongMediaRowWidthRefusesBeforeAnyPageIsTaken() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        try (GGUFModel m = new GGUFModel(model.toString(), config())) {
            ForwardBatch.Builder b = ForwardBatch.builder(1);
            b.addEmbedding(new float[7], 0, SEQ, false);
            IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                    () -> m.forwardBackend().forwardBatch(b.build()));
            assertTrue(e.getMessage(), e.getMessage().contains("model input"));
            // nothing was written: the sequence holds no positions
            HeapModelMemory mem = (HeapModelMemory) m.forwardBackend().memory();
            assertNotNull(mem.pagedSequences());
            assertEquals(-1, mem.sequences().posMax(SEQ));
        }
    }

    @Test
    public void theDefaultForwardPathRefusesMediaRequestsByName() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig unpaged = EngineConfig.builder().memory(MemoryConfig.builder().nCtx(64).ubatch(8)
                .typeK(KvDType.F32).typeV(KvDType.F32).build()).build();
        try (GGUFModel m = new GGUFModel(model.toString(), unpaged)) {
            // a backend that keeps the interface's default forwardBatch (the GPU path runs it) over the model's
            // memory
            ForwardBackend defaults = new ForwardBackend() {
                @Override public float[] prefill(int[] tokens, int pos0) { throw new UnsupportedOperationException(); }
                @Override public float[] decode(int token, int pos) { throw new UnsupportedOperationException(); }
                @Override public float[][] forwardRows(int[] tokens, int pos0, int[] outputRows) {
                    throw new UnsupportedOperationException();
                }
                @Override public float[][] forwardAll(int[] tokens, int[] outputRows) {
                    throw new UnsupportedOperationException();
                }
                @Override public float[][] hiddenRows(int[] tokens, int[] tokenTypes) {
                    throw new UnsupportedOperationException();
                }
                @Override public void reset() { }
                @Override public boolean canRollback(int nTokens) { return nTokens == 0; }
                @Override public void rollback(int nTokens) { }
                @Override public void shiftContext(int nKeep, int nDiscard) { }
                @Override public ModelMemory memory() { return m.forwardBackend().memory(); }
                @Override public void setHiddenTap(HiddenTap tap) { }
                @Override public void close() { }
            };
            ForwardBatch.Builder embd = ForwardBatch.builder(1);
            embd.addEmbedding(mediaRows(1, 1)[0], 0, SEQ, true);
            UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class,
                    () -> defaults.forwardBatch(embd.build()));
            assertTrue(e.getMessage(), e.getMessage().contains("embedding"));

            ForwardBatch.Builder nonCausal = ForwardBatch.builder(1);
            nonCausal.add(TEXT, 0, SEQ, true).nonCausal();
            UnsupportedOperationException e2 = assertThrows(UnsupportedOperationException.class,
                    () -> defaults.forwardBatch(nonCausal.build()));
            assertTrue(e2.getMessage(), e2.getMessage().contains("non-causal"));

            ForwardBatch.Builder gap = ForwardBatch.builder(2);
            gap.add(TEXT, 0, SEQ, false).add(TEXT + 1, 2, SEQ, true);
            UnsupportedOperationException e3 = assertThrows(UnsupportedOperationException.class,
                    () -> defaults.forwardBatch(gap.build()));
            assertTrue(e3.getMessage(), e3.getMessage().contains("consecutive"));
        }
    }
}

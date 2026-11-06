package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ocl.OclComputeBackend;
import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.GGUFModelGPU;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.engine.ForwardBatch;
import com.eonmux.neuralfree.gguf.engine.HiddenTap;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.memory.paged.PagedSequences;
import com.eonmux.neuralfree.gguf.ops.rope.MediaRopePositions;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.function.Function;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The request kinds the CPU batched forward serves, on the GPU batched forward over the paged memory, checked
 * against the CPU backend on the same batches: ubatches mixing token rows and precomputed embedding rows (the token
 * table's own rows answer like their tokens), an M-RoPE media chunk with explicit sections and the non-causal mask
 * split over several ubatches, the final hidden rows of a batch ({@code hiddenBatch}) and the pooling / scoring
 * evaluations over a paged memory, the hidden tap over a multi-sequence batch, adapter slots (slot 0 explicit is the
 * attached adapters; another slot refuses by name on both backends) and the contiguous {@code prefillEmbeddings}.
 */
public class GpuBatchedForwardCompletenessTest {

    private static final int SEQ_A = 0;
    private static final int SEQ_B = 1;
    private static final float TOL = 2e-3f;

    private HWAcceleration acceleration;
    private ComputeBackend backend;

    @Before
    public void setUp() {
        Assume.assumeTrue("OpenCL GPU not available", GpuTestUtil.isOpenCLAvailable());
        acceleration = new HWAcceleration();
        backend = new OclComputeBackend(acceleration.getCLDevices().get(0));
    }

    @After
    public void tearDown() {
        if (backend != null) {
            backend.close();
        }
        if (acceleration != null) {
            acceleration.shutdown();
        }
    }

    private static EngineConfig config(int parallel) {
        MemoryConfig.Builder mc = MemoryConfig.builder().nCtx(64).ubatch(8).typeK(KvDType.F32).typeV(KvDType.F32);
        return EngineConfig.builder().memory((parallel > 1 ? mc.parallel(parallel) : mc).build()).build();
    }

    private static int[] tokens(int seed, int n) {
        int[] t = new int[n];
        for (int i = 0; i < n; i++) {
            t[i] = 4 + (seed * 7 + i * 13) % 250;
        }
        return t;
    }

    /** The token table's rows of {@code tokens}, read from the GGUF file. */
    private static float[][] tableRows(Path model, int[] tokens) throws IOException {
        try (GGUFFileParser parser = new GGUFFileParser(model.toString())) {
            GGUFTensorInfo table = parser.getTensor("token_embd.weight");
            float[] flat = parser.readTensorDataAsFloat(table);
            int width = (int) table.dim(0);
            float[][] rows = new float[tokens.length][];
            for (int i = 0; i < tokens.length; i++) {
                rows[i] = Arrays.copyOfRange(flat, tokens[i] * width, (tokens[i] + 1) * width);
            }
            return rows;
        }
    }

    /** Deterministic media rows of distinct values. */
    private static float[][] mediaRows(int n, int width) {
        float[][] r = new float[n][width];
        for (int i = 0; i < n; i++) {
            for (int d = 0; d < width; d++) {
                r[i][d] = 0.3f * (float) Math.sin(0.37 * (i * width + d) + 0.11 * i);
            }
        }
        return r;
    }

    /** Runs {@code call} on the CPU and on the GPU model of {@code model}, both over a paged memory of two slots. */
    private <T> T[] both(Path model, Function<ForwardBackend, T> call, T[] out) throws IOException {
        try (GGUFModel cpu = new GGUFModel(model.toString(), config(2))) {
            out[0] = call.apply(cpu.forwardBackend());
        }
        try (GGUFModelGPU gpu = new GGUFModelGPU(model.toString(), backend, config(2))) {
            out[1] = call.apply(gpu.forwardBackend());
        }
        return out;
    }

    private static void assertRowsEqual(String what, float[][] expected, float[][] actual) {
        assertEquals(what + ": row count", expected.length, actual.length);
        for (int i = 0; i < expected.length; i++) {
            assertNotNull(what + ": row " + i, actual[i]);
            assertArrayEquals(what + ": row " + i, expected[i], actual[i], TOL);
        }
    }

    /**
     * Sequence A as token rows and sequence B as the token table's rows of its tokens, interleaved so every ubatch
     * mixes both kinds; the last row of each sequence is an output row.
     */
    private static ForwardBatch mixed(int[] a, float[][] b) {
        ForwardBatch.Builder batch = ForwardBatch.builder(a.length + b.length);
        for (int i = 0; i < Math.max(a.length, b.length); i++) {
            if (i < a.length) {
                batch.add(a[i], i, SEQ_A, i == a.length - 1);
            }
            if (i < b.length) {
                batch.addEmbedding(b[i], i, SEQ_B, i == b.length - 1);
            }
        }
        return batch.build();
    }

    private static ForwardBatch tokensOf(int[] a, int[] b) {
        ForwardBatch.Builder batch = ForwardBatch.builder(a.length + b.length);
        for (int i = 0; i < Math.max(a.length, b.length); i++) {
            if (i < a.length) {
                batch.add(a[i], i, SEQ_A, i == a.length - 1);
            }
            if (i < b.length) {
                batch.add(b[i], i, SEQ_B, i == b.length - 1);
            }
        }
        return batch.build();
    }

    @Test
    public void aBatchMixingTokenAndEmbeddingRowsMatchesTheCpuBatch() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        int[] a = tokens(1, 10);
        int[] b = tokens(2, 7);
        float[][] rowsB = tableRows(model, b);
        float[][][] out = both(model, f -> f.forwardBatch(mixed(a, rowsB)), new float[2][][]);
        assertRowsEqual("GPU mixed batch vs CPU mixed batch", out[0], out[1]);
        try (GGUFModelGPU gpu = new GGUFModelGPU(model.toString(), backend, config(2))) {
            float[][] asTokens = gpu.forwardBackend().forwardBatch(tokensOf(a, b));
            assertRowsEqual("the table's rows answer like their tokens", asTokens, out[1]);
        }
    }

    @Test
    public void anMropeMediaChunkWithItsSectionsAndNonCausalMaskMatchesTheCpuChunk() throws IOException {
        Path model = SyntheticModels.qwen2vl().writeTempFile();
        int[] prefix = tokens(3, 5);
        MediaRopePositions grid = MediaRopePositions.image2d(prefix.length, 4, 4);
        float[][] rows = mediaRows(16, tableRows(model, prefix)[0].length);
        float[][][] out = both(model, f -> mediaRun(f, prefix, grid, rows), new float[2][][]);
        assertRowsEqual("GPU media chunk (sections, non-causal, three ubatches) vs CPU", out[0], out[1]);
    }

    /**
     * A text prefix, a 16-row non-causal media chunk at the prefix's end (it spans three ubatches of 8) whose
     * every row is an output row, then a decode token after the chunk's positions.
     */
    private static float[][] mediaRun(ForwardBackend f, int[] prefix, MediaRopePositions grid, float[][] rows) {
        f.forwardBatch(single(prefix, SEQ_A, 0));
        ForwardBatch.Builder chunk = ForwardBatch.builder(rows.length);
        for (int i = 0; i < rows.length; i++) {
            chunk.addEmbedding(rows[i], grid.position(0, i), SEQ_A, true);
        }
        chunk.ropePositions(grid.sections()).nonCausal();
        float[][] chunkRows = f.forwardBatch(chunk.build());
        int next = prefix.length + grid.nPositions();
        float[] decode = f.forwardBatch(ForwardBatch.builder(1).add(tokens(4, 1)[0], next, SEQ_A, true).build())[0];
        float[][] all = Arrays.copyOf(chunkRows, chunkRows.length + 1);
        all[chunkRows.length] = decode;
        return all;
    }

    @Test
    public void aNonCausalChunkLetsItsRowsSeeLaterRowsAsOnTheCpu() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        int[] prefix = tokens(13, 3);
        float[][] rows = mediaRows(10, tableRows(model, prefix)[0].length);
        float[][][] nonCausal = both(model, f -> chunkAtConsecutivePositions(f, prefix, rows, true),
                new float[2][][]);
        assertRowsEqual("GPU non-causal chunk vs CPU", nonCausal[0], nonCausal[1]);
        float[][] causal;
        try (GGUFModelGPU gpu = new GGUFModelGPU(model.toString(), backend, config(2))) {
            causal = chunkAtConsecutivePositions(gpu.forwardBackend(), prefix, rows, false);
        }
        float diff = 0f;
        for (int v = 0; v < causal[0].length; v++) {
            diff = Math.max(diff, Math.abs(causal[0][v] - nonCausal[1][0][v]));
        }
        assertTrue("the chunk's first row sees its later rows only without the causal mask: " + diff, diff > 1e-2f);
    }

    /** A token prefix, then embedding rows at consecutive positions after it, every chunk row an output row. */
    private static float[][] chunkAtConsecutivePositions(ForwardBackend f, int[] prefix, float[][] rows,
                                                         boolean nonCausal) {
        f.forwardBatch(single(prefix, SEQ_B, 0));
        ForwardBatch.Builder chunk = ForwardBatch.builder(rows.length);
        for (int i = 0; i < rows.length; i++) {
            chunk.addEmbedding(rows[i], prefix.length + i, SEQ_B, true);
        }
        if (nonCausal) {
            chunk.nonCausal();
        }
        return f.forwardBatch(chunk.build());
    }

    @Test
    public void sectionBatchesOnASingleSectionOrRopelessModelRefuseLikeTheCpu() throws IOException {
        Path[] models = {SyntheticModels.llama().writeTempFile(), SyntheticModels.mamba2().writeTempFile()};
        for (Path model : models) {
            int width = tableRows(model, new int[]{4})[0].length;
            String[] refusals = both(model, f -> {
                PagedSequences paged = f.memory().pagedSequences();
                assertNotNull("a paged memory", paged);
                int freeCells = paged.freeCells();
                MediaRopePositions grid = MediaRopePositions.image2d(0, 2, 2);
                ForwardBatch.Builder b = ForwardBatch.builder(4);
                float[][] rows = mediaRows(4, width);
                for (int i = 0; i < rows.length; i++) {
                    b.addEmbedding(rows[i], grid.position(0, i), SEQ_A, false);
                }
                b.ropePositions(grid.sections());
                IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                        () -> f.forwardBatch(b.build()));
                assertEquals("the refusal reserved no cells", freeCells, paged.freeCells());
                assertEquals("the refusal stored no position", -1, paged.posMax(SEQ_A));
                return e.getMessage();
            }, new String[2]);
            assertEquals("the CPU and GPU refusals carry the same message", refusals[0], refusals[1]);
            assertTrue(refusals[1], refusals[1].contains("4 rope sections"));
            assertTrue(refusals[1], refusals[1].contains("rope reads 1"));
        }
    }

    private static ForwardBatch single(int[] t, int seq, int pos0) {
        ForwardBatch.Builder batch = ForwardBatch.builder(t.length);
        for (int i = 0; i < t.length; i++) {
            batch.add(t[i], pos0 + i, seq, i == t.length - 1);
        }
        return batch.build();
    }

    @Test
    public void hiddenRowsOfABatchMatchTheCpuHiddenRows() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        int[] a = tokens(5, 11);
        int[] b = tokens(6, 4);
        float[][][] out = both(model, f -> f.hiddenBatch(tokensOf(a, b)), new float[2][][]);
        assertEquals(2, out[1].length);
        assertRowsEqual("GPU hidden batch rows vs CPU", out[0], out[1]);
        float[][] contiguous;
        try (GGUFModel cpu = new GGUFModel(model.toString(), config(1))) {
            contiguous = cpu.forwardBackend().hiddenRows(a, null);
        }
        // output rows come back in batch order: B's last row (batch row 7) before A's (batch row 14)
        assertArrayEquals("the batch's last row of A is A's last hidden row", contiguous[a.length - 1], out[1][1],
                TOL);
    }

    @Test
    public void nextTokenLogitsOverAPagedGpuMemoryMatchTheContiguousCpuRun() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        int[] prompt = tokens(7, 9);
        float[] reference;
        try (GGUFModel cpu = new GGUFModel(model.toString(), config(1))) {
            reference = cpu.nextLogits(prompt);
        }
        try (GGUFModel cpu = new GGUFModel(model.toString(), config(2))) {
            assertArrayEquals("CPU paged next-token logits", reference, cpu.nextLogits(prompt), TOL);
        }
        try (GGUFModelGPU gpu = new GGUFModelGPU(model.toString(), backend, config(2))) {
            assertArrayEquals("GPU paged next-token logits", reference, gpu.nextLogits(prompt), TOL);
            assertArrayEquals("a second scoring call starts from an empty memory", reference,
                    gpu.nextLogits(prompt), TOL);
        }
    }

    @Test
    public void theHiddenTapOverAMultiSequenceBatchMatchesTheCpuTap() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        int[] a = tokens(8, 9);
        int[] b = tokens(9, 6);
        float[][][] out = both(model, f -> {
            Capture tap = new Capture(1);
            f.setHiddenTap(tap);
            f.forwardBatch(tokensOf(a, b));
            f.setHiddenTap(HiddenTap.NONE);
            assertEquals("every batch row is tapped", a.length + b.length, tap.nRows);
            return new float[][]{tap.rows};
        }, new float[2][][]);
        assertArrayEquals("GPU tap rows at their batch rows vs CPU", out[0][0], out[1][0], TOL);
    }

    /** Captures the rows of one tap point. */
    private static final class Capture implements HiddenTap {
        private final int layer;
        float[] rows;
        int nRows;

        Capture(int layer) {
            this.layer = layer;
        }

        @Override
        public void onHidden(int l, float[] r, int n, int width) {
            rows = Arrays.copyOf(r, n * width);
            nRows = n;
        }

        @Override
        public boolean wants(int l) {
            return l == layer;
        }
    }

    @Test
    public void slotZeroRunsTheAttachedAdaptersAndAnotherSlotRefusesByNameOnBothBackends() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        int[] a = tokens(10, 6);
        int[] b = tokens(11, 6);
        ForwardBatch plain = tokensOf(a, b);
        ForwardBatch slotZero = withSlots(a, b, 0, 0);
        ForwardBatch twoSlots = withSlots(a, b, 0, 1);
        String[] refusals = both(model, f -> {
            float[][] unset = f.forwardBatch(plain);
            f.reset();
            float[][] zero = f.forwardBatch(slotZero);
            assertRowsEqual("explicit slot 0 is the unset slot", unset, zero);
            f.reset();
            UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class,
                    () -> f.forwardBatch(twoSlots));
            return e.getMessage();
        }, new String[2]);
        for (String r : refusals) {
            assertTrue(r, r.contains("adapterSlots") && r.contains("slot 1"));
        }
        assertTrue(refusals[0], refusals[0].contains("CpuForwardBackend"));
        assertTrue(refusals[1], refusals[1].contains("GpuForwardBackend"));
    }

    private static ForwardBatch withSlots(int[] a, int[] b, int slotA, int slotB) {
        ForwardBatch.Builder batch = ForwardBatch.builder(a.length + b.length);
        int[] slots = new int[a.length + b.length];
        int k = 0;
        for (int i = 0; i < a.length; i++) {
            batch.add(a[i], i, SEQ_A, i == a.length - 1);
            slots[k++] = slotA;
            batch.add(b[i], i, SEQ_B, i == b.length - 1);
            slots[k++] = slotB;
        }
        return batch.adapterSlots(slots).build();
    }

    @Test
    public void embeddingRowsPrefillTheContiguousGpuMemoryLikeTheirTokens() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        int[] prompt = tokens(12, 11);
        float[][] rows = tableRows(model, prompt);
        try (GGUFModelGPU gpu = new GGUFModelGPU(model.toString(), backend, config(1))) {
            ForwardBackend f = gpu.forwardBackend();
            float[] fromTokens = f.prefill(prompt, 0).clone();
            float[] nextFromTokens = f.decode(prompt[0], prompt.length).clone();
            f.reset();
            float[] fromRows = f.prefillEmbeddings(rows, 0).clone();
            assertArrayEquals("embedding-row prefill logits", fromTokens, fromRows, TOL);
            assertEquals(prompt.length, f.memory().sequence().nPast());
            assertArrayEquals("the rows' positions are stored like the tokens'", nextFromTokens,
                    f.decode(prompt[0], prompt.length), TOL);
            assertThrows(IllegalArgumentException.class,
                    () -> f.prefillEmbeddings(new float[][]{new float[3]}, prompt.length + 1));
        }
    }
}

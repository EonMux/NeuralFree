package com.eonmux.neuralfree.gguf.ops.gemma3n;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ocl.OclComputeBackend;
import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.GGUFModelGPU;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.engine.ForwardBatch;
import com.eonmux.neuralfree.gguf.engine.HiddenTap;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.memory.SequenceState;
import com.eonmux.neuralfree.gguf.ops.gemma3n.Gemma3nBlocks.NgramLayout;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import com.eonmux.neuralfree.gguf.testutil.SyntheticQwen4ExpModels;
import org.junit.Assume;
import org.junit.Test;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.Function;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

/**
 * Embedding rows on the qwen4exp n-gram per-layer embedding, on the CPU and the GPU backend: an embedding row hashes
 * the stand-in id ({@code ple.image_token_id}, or the EOS id when that is 0) and records it as its history token
 * (llama.cpp qwen4exp.cpp:1326-1335, llama-kv-cache.cpp:1192-1200). The reference is the token prompt with the
 * stand-in id at the rows' places, the rows being the stand-in's {@code token_embd} rows, so both runs give every
 * block the same input; prefills of text, rows and text, a paged ubatch mixing one sequence's rows with another's
 * tokens, and media rows sharing one M-RoPE position (checked at the input of layer 1, which does not depend on
 * positions) are covered. The predecessor rule is also checked row by row against a transcription of llama.cpp
 * {@code get_prev_tokens} (llama-kv-cache.cpp:1859-1920).
 */
public class NgramPleEmbeddingRowsTest {

    private static final int SEQ_A = 0;
    private static final int SEQ_B = 1;
    private static final float CPU_TOL = 1e-4f;
    private static final float GPU_TOL = 2e-3f;
    /** The {@code ple.image_token_id} of the image-token fixture. */
    private static final int IMAGE_TOKEN = 7;
    private static final int[] HEAD = {71, 120, 8};
    private static final int[] TAIL = {301, 64};
    private static final int[] OTHER = {17, 40, 263, 99, 5, 150, 33, 12};
    private static final int ROWS = 3;

    private static EngineConfig config(int parallel) {
        MemoryConfig.Builder mc = MemoryConfig.builder().nCtx(64).ubatch(8).typeK(KvDType.F32).typeV(KvDType.F32);
        return EngineConfig.builder().memory((parallel > 1 ? mc.parallel(parallel) : mc).build()).build();
    }

    // ------------------------------------------------------------------ CPU, contiguous

    @Test
    public void theModelTakesEmbeddingRows() throws IOException {
        try (GGUFModel m = new GGUFModel(model(IMAGE_TOKEN).toString(), config(1))) {
            assertNull("the n-gram embedding hashes a stand-in", m.forwardBackend().embeddingRowsRefusal());
        }
    }

    @Test
    public void rowsBetweenTextHashTheImageTokenOnTheCpu() throws IOException {
        assertSplicedMatchesStandIn(IMAGE_TOKEN, IMAGE_TOKEN);
    }

    @Test
    public void rowsBetweenTextHashTheEosWithoutAnImageTokenOnTheCpu() throws IOException {
        assertSplicedMatchesStandIn(0, SyntheticQwen4ExpModels.EOS);
    }

    private static void assertSplicedMatchesStandIn(int imageToken, int standIn) throws IOException {
        int[] whole = whole(standIn);
        try (GGUFModel m = new GGUFModel(model(imageToken).toString(), config(1))) {
            float[] spliced = spliced(m.forwardBackend(), rows(standIn, ROWS));
            assertArrayEquals("the contiguous record keeps the stand-in for the rows", whole,
                    m.forwardBackend().memory().sequence().tokens());
            m.forwardBackend().reset();
            float[] reference = lastLogits(m.forwardBackend(), whole);
            assertArrayEquals("rows hash and record the stand-in " + standIn, reference, spliced, CPU_TOL);
        }
    }

    @Test
    public void rowsOfAnotherTokenStillHashTheStandIn() throws IOException {
        int token = 150;
        int[] asTokens = whole(token);
        try (GGUFModel m = new GGUFModel(model(IMAGE_TOKEN).toString(), config(1))) {
            float[] spliced = spliced(m.forwardBackend(), rows(token, ROWS));
            m.forwardBackend().reset();
            float[] tokens = lastLogits(m.forwardBackend(), asTokens);
            assertDiffers("hashing the stand-in instead of the row's own token", tokens, spliced);
        }
    }

    // ------------------------------------------------------------------ CPU, paged

    @Test
    public void aPagedUbatchMixingRowsAndTokensHashesTheStandInOnTheCpu() throws IOException {
        float[] referenceA;
        float[] referenceB;
        try (GGUFModel one = new GGUFModel(denseModel(IMAGE_TOKEN).toString(), config(1))) {
            referenceA = lastLogits(one.forwardBackend(), whole(IMAGE_TOKEN));
            one.forwardBackend().reset();
            referenceB = lastLogits(one.forwardBackend(), OTHER);
        }
        try (GGUFModel m = new GGUFModel(denseModel(IMAGE_TOKEN).toString(), config(2))) {
            ForwardBackend be = m.forwardBackend();
            float[][] rows = rows(IMAGE_TOKEN, ROWS);
            ForwardBatch.Builder head = ForwardBatch.builder(2 * HEAD.length);
            for (int i = 0; i < HEAD.length; i++) {
                head.add(HEAD[i], i, SEQ_A, i == HEAD.length - 1).add(OTHER[i], i, SEQ_B, i == HEAD.length - 1);
            }
            be.forwardBatch(head.build());
            ForwardBatch.Builder mixed = ForwardBatch.builder(2 * ROWS);
            for (int i = 0; i < ROWS; i++) {
                int pos = HEAD.length + i;
                mixed.addEmbedding(rows[i], pos, SEQ_A, i == ROWS - 1).add(OTHER[pos], pos, SEQ_B, i == ROWS - 1);
            }
            be.forwardBatch(mixed.build());
            ForwardBatch.Builder tail = ForwardBatch.builder(2 * TAIL.length);
            for (int i = 0; i < TAIL.length; i++) {
                int pos = HEAD.length + ROWS + i;
                boolean last = i == TAIL.length - 1;
                tail.add(TAIL[i], pos, SEQ_A, last).add(OTHER[pos], pos, SEQ_B, last);
            }
            float[][] out = be.forwardBatch(tail.build());
            assertArrayEquals("sequence A's rows hash the stand-in in a mixed ubatch", referenceA, out[0], CPU_TOL);
            assertArrayEquals("sequence B matches its contiguous run", referenceB, out[1], CPU_TOL);
            SequenceState record = be.memory().pagedSequences().sequence(SEQ_A);
            assertArrayEquals("the paged record keeps the stand-in for the rows", whole(IMAGE_TOKEN),
                    record.tokens());
        }
    }

    @Test
    public void mediaRowsSharingOnePositionResolvePredecessorsByOrderOnTheCpu() throws IOException {
        int media = 4;
        int[] whole = new int[HEAD.length + media + TAIL.length];
        System.arraycopy(HEAD, 0, whole, 0, HEAD.length);
        Arrays.fill(whole, HEAD.length, HEAD.length + media, IMAGE_TOKEN);
        System.arraycopy(TAIL, 0, whole, HEAD.length + media, TAIL.length);
        LayerTap reference = new LayerTap(1);
        try (GGUFModel one = new GGUFModel(denseModel(IMAGE_TOKEN).toString(), config(1))) {
            one.forwardBackend().setHiddenTap(reference);
            lastLogits(one.forwardBackend(), whole);
        }
        LayerTap paged = new LayerTap(1);
        try (GGUFModel m = new GGUFModel(denseModel(IMAGE_TOKEN).toString(), config(2))) {
            ForwardBackend be = m.forwardBackend();
            be.setHiddenTap(paged);
            float[][] rows = rows(IMAGE_TOKEN, media);
            int shared = HEAD.length;
            ForwardBatch.Builder first = ForwardBatch.builder(HEAD.length + media);
            for (int i = 0; i < HEAD.length; i++) {
                first.add(HEAD[i], i, SEQ_A, false);
            }
            for (int i = 0; i < media; i++) {
                first.addEmbedding(rows[i], shared, SEQ_A, i == media - 1);
            }
            be.forwardBatch(first.build());
            // the image took two positions: the text continues past the gap
            int next = shared + 2;
            ForwardBatch.Builder second = ForwardBatch.builder(TAIL.length);
            for (int i = 0; i < TAIL.length; i++) {
                second.add(TAIL[i], next + i, SEQ_A, i == TAIL.length - 1);
            }
            be.forwardBatch(second.build());
            SequenceState record = be.memory().pagedSequences().sequence(SEQ_A);
            assertArrayEquals(whole, record.tokens());
            int[] positions = new int[record.nPast()];
            for (int i = 0; i < positions.length; i++) {
                positions[i] = record.position(i);
            }
            assertArrayEquals("one record entry per row, media rows at their shared position",
                    new int[] {0, 1, 2, 3, 3, 3, 3, 5, 6}, positions);
        }
        float[][] expected = reference.rows();
        float[][] actual = paged.rows();
        assertEquals(expected.length, actual.length);
        for (int i = 0; i < expected.length; i++) {
            assertArrayEquals("layer 1 input of row " + i, expected[i], actual[i], CPU_TOL);
        }
    }

    // ------------------------------------------------------------------ GPU

    @Test
    public void rowsBetweenTextHashTheImageTokenOnTheGpu() throws IOException {
        float[][] spliced = onGpu(be -> new float[][] {spliced(be, rows(IMAGE_TOKEN, ROWS)),
            toFloats(be.memory().sequence().tokens())}, model(IMAGE_TOKEN));
        float[][] reference = onGpu(be -> new float[][] {lastLogits(be, whole(IMAGE_TOKEN))}, model(IMAGE_TOKEN));
        assertArrayEquals("GPU rows hash the stand-in", reference[0], spliced[0], GPU_TOL);
        assertArrayEquals("the GPU contiguous record keeps the stand-in", toFloats(whole(IMAGE_TOKEN)), spliced[1],
                0f);
    }

    @Test
    public void rowsBetweenTextHashTheEosWithoutAnImageTokenOnTheGpu() throws IOException {
        int eos = SyntheticQwen4ExpModels.EOS;
        float[][] spliced = onGpu(be -> new float[][] {spliced(be, rows(eos, ROWS))}, model(0));
        float[][] reference = onGpu(be -> new float[][] {lastLogits(be, whole(eos))}, model(0));
        assertArrayEquals("GPU rows hash the EOS stand-in", reference[0], spliced[0], GPU_TOL);
    }

    private static float[][] onGpu(Function<ForwardBackend, float[][]> call, Path model) throws IOException {
        Assume.assumeTrue("OpenCL GPU not available", GpuTestUtil.isOpenCLAvailable());
        HWAcceleration acceleration = new HWAcceleration();
        try (ComputeBackend backend = new OclComputeBackend(acceleration.getCLDevices().get(0));
             GGUFModelGPU gpu = new GGUFModelGPU(model.toString(), backend, config(1))) {
            return call.apply(gpu.forwardBackend());
        } finally {
            acceleration.shutdown();
        }
    }

    // ------------------------------------------------------------------ the predecessor rule

    /**
     * Ubatches of one sequence, each window completed as the extras complete it, against llama.cpp's
     * {@code get_prev_tokens} and {@code tok_of} over the cells stored so far: text, media rows sharing one
     * position, text after the media gap, an image split over two ubatches, and a mixed ubatch.
     */
    @Test
    public void theWindowMatchesTheReferencePredecessorLookup() throws IOException {
        NgramLayout layout = layout();
        int standIn = layout.standIn();
        assertEquals(IMAGE_TOKEN, standIn);
        SequenceState record = new SequenceState(64, 128);
        List<int[]> cells = new ArrayList<>();
        int[][][] ubatches = {
            {{10, 11, 12}, {0, 1, 2}},
            {{-1, -1, -1, -1}, {3, 3, 3, 3}},
            {{20, 21}, {5, 6}},
            {{-1, -1}, {7, 7}},
            {{-1, -1}, {7, 7}},
            {{30, -1, -1, 31, 32}, {8, 9, 9, 11, 12}},
        };
        for (int[][] u : ubatches) {
            int[] tokens = u[0];
            int[] pos = u[1];
            int n = tokens.length;
            int back = layout.lookBehind();
            int[] window = new int[back + n];
            System.arraycopy(tokens, 0, window, back, n);
            layout.completeWindow(record, false, pos[0], n, window);
            int[] recorded = new int[n];
            for (int i = 0; i < n; i++) {
                recorded[i] = tokens[i] < 0 ? standIn : tokens[i];
                cells.add(new int[] {pos[i], recorded[i]});
            }
            int[][] expected = referenceContexts(cells, tokens, pos, standIn, layout.nGram());
            for (int i = 0; i < n; i++) {
                for (int s = 0; s < layout.nGram(); s++) {
                    assertEquals("ubatch at " + pos[0] + ", row " + i + ", predecessor " + s, expected[i][s],
                            window[back + i - s]);
                }
            }
            record.appendRows(tokens, pos, 0, n, standIn);
        }
    }

    /**
     * llama.cpp's contexts {@code ctx[0] = tok_of(i)}, {@code ctx[s]} the predecessor {@code s} back before the
     * EOS cut, {@code -1} for {@code LLAMA_TOKEN_NULL}: {@code get_prev_tokens} with {@code cells} holding every
     * stored cell (this ubatch's last, as {@code apply_ubatch} stored them) in cell-index order.
     */
    private static int[][] referenceContexts(List<int[]> cells, int[] tokens, int[] pos, int standIn, int nGram) {
        int n = tokens.length;
        int nPrev = nGram - 1;
        boolean embd = false;
        for (int t : tokens) {
            embd |= t < 0;
        }
        int[][] ctx = new int[n][nGram];
        for (int i = 0; i < n; i++) {
            ctx[i][0] = tokens[i] < 0 ? standIn : tokens[i];
            for (int j = 0; j < nPrev; j++) {
                int d = nPrev - j;
                int p;
                if (embd) {
                    int k = i - d;
                    p = k >= 0 ? pos[k] : pos[0] + k;
                } else {
                    p = pos[i] - d;
                }
                // prev[i*n_prev + j] is the predecessor d back: ctx[s] reads prev[n_prev - s]
                ctx[i][d] = p < 0 ? -1 : tokenAtOrBefore(cells, p);
            }
        }
        return ctx;
    }

    /** {@code seq_pos_tok_le}: the token of the cell at the largest position at most {@code p}, the last cell on ties. */
    private static int tokenAtOrBefore(List<int[]> cells, int p) {
        int bestPos = -1;
        int token = -1;
        for (int[] c : cells) {
            if (c[0] <= p && c[0] >= bestPos) {
                bestPos = c[0];
                token = c[1];
            }
        }
        return token;
    }

    private static NgramLayout layout() throws IOException {
        Path file = model(IMAGE_TOKEN);
        try (com.eonmux.neuralfree.gguf.GGUFFileParser parser =
                     new com.eonmux.neuralfree.gguf.GGUFFileParser(file.toString())) {
            com.eonmux.neuralfree.gguf.spec.TensorCatalog catalog =
                    com.eonmux.neuralfree.gguf.spec.TensorCatalog.of(parser);
            com.eonmux.neuralfree.gguf.spec.ModelSpec spec = com.eonmux.neuralfree.gguf.spec.resolve.ModelSpecBuilder
                    .build(com.eonmux.neuralfree.gguf.hparams.GgufKv.of(parser), catalog, MemoryConfig.defaults());
            return NgramLayout.resolve(spec, spec.layer(0));
        }
    }

    // ------------------------------------------------------------------ fixtures

    private static Path model(int imageToken) throws IOException {
        return SyntheticQwen4ExpModels.qwen4exp(imageToken).writeTempFile();
    }

    /** The fixture with dense attention on block 1: the QSA mixer has no multi-sequence step. */
    private static Path denseModel(int imageToken) throws IOException {
        return SyntheticQwen4ExpModels.qwen4exp(imageToken, false).writeTempFile();
    }

    /** {@code n} copies of the {@code token_embd} row of {@code token}: the input its lookup produces. */
    private static float[][] rows(int token, int n) {
        GgufTestWriter w = SyntheticQwen4ExpModels.qwen4exp();
        int nEmbd = SyntheticQwen4ExpModels.config().nEmbd();
        ByteBuffer table = ByteBuffer.wrap(w.tensorData("token_embd.weight")).order(ByteOrder.LITTLE_ENDIAN);
        float[][] rows = new float[n][nEmbd];
        for (int i = 0; i < n; i++) {
            for (int d = 0; d < nEmbd; d++) {
                rows[i][d] = table.getFloat((token * nEmbd + d) * Float.BYTES);
            }
        }
        return rows;
    }

    /** {@link #HEAD}, {@link #ROWS} copies of {@code token}, {@link #TAIL}. */
    private static int[] whole(int token) {
        int[] all = new int[HEAD.length + ROWS + TAIL.length];
        System.arraycopy(HEAD, 0, all, 0, HEAD.length);
        Arrays.fill(all, HEAD.length, HEAD.length + ROWS, token);
        System.arraycopy(TAIL, 0, all, HEAD.length + ROWS, TAIL.length);
        return all;
    }

    /** {@link #HEAD} as text, {@code rows} as embedding rows, {@link #TAIL} as text; the last row's logits. */
    private static float[] spliced(ForwardBackend be, float[][] rows) {
        be.prefill(HEAD, 0);
        be.prefillEmbeddings(rows, HEAD.length);
        return be.forwardRows(TAIL, HEAD.length + rows.length, new int[] {TAIL.length - 1})[0];
    }

    private static float[] lastLogits(ForwardBackend be, int[] tokens) {
        return be.forwardRows(tokens, 0, new int[] {tokens.length - 1})[0];
    }

    private static float[] toFloats(int[] v) {
        float[] out = new float[v.length];
        for (int i = 0; i < v.length; i++) {
            out[i] = v[i];
        }
        return out;
    }

    private static void assertDiffers(String what, float[] a, float[] b) {
        double maxDiff = 0;
        for (int i = 0; i < a.length; i++) {
            maxDiff = Math.max(maxDiff, Math.abs(a[i] - b[i]));
        }
        assertTrue(what + " changes the logits (max diff " + maxDiff + ")", maxDiff > 1e-3);
    }

    /** Collects the rows entering one layer over every forward call. */
    private static final class LayerTap implements HiddenTap {
        private final int layer;
        private final List<float[]> rows = new ArrayList<>();

        LayerTap(int layer) {
            this.layer = layer;
        }

        @Override
        public void onHidden(int at, float[] data, int nRows, int width) {
            for (int r = 0; r < nRows; r++) {
                rows.add(Arrays.copyOfRange(data, r * width, (r + 1) * width));
            }
        }

        @Override
        public boolean wants(int at) {
            return at == layer;
        }

        float[][] rows() {
            return rows.toArray(new float[0][]);
        }
    }
}

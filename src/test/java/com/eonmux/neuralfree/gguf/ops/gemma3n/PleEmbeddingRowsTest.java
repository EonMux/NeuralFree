package com.eonmux.neuralfree.gguf.ops.gemma3n;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ocl.OclComputeBackend;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.GGUFModelGPU;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.engine.ForwardBatch;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import com.eonmux.neuralfree.gguf.testutil.SyntheticGemmaModels;
import org.junit.Assume;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.function.Function;
import java.util.function.Supplier;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

/**
 * Embedding rows on the per-layer-embedding models (gemma3n and dense gemma4), on the CPU and the GPU backend: an
 * embedding row's per-layer token embedding is the padding token's row 0 of {@code per_layer_token_embd} (llama.cpp
 * gemma3n.cpp:329-349, gemma4.cpp:490-509). The reference is the same model with the per-layer rows of the prompt
 * tokens overwritten by row 0, run on the tokens themselves; the embedding rows are those tokens' scaled
 * {@code token_embd} rows, so both runs give every block the same input. A prefill of rows alone, a span of rows
 * between text and a paged ubatch mixing one sequence's rows with another's tokens are covered (the token sequence
 * also matching its contiguous run), and the substitution is shown to change the logits.
 */
@RunWith(Parameterized.class)
public class PleEmbeddingRowsTest {

    private static final String PER_LAYER_TABLE = "per_layer_token_embd.weight";
    private static final int SEQ_A = 0;
    private static final int SEQ_B = 1;
    private static final float CPU_TOL = 1e-4f;
    private static final float GPU_TOL = 2e-3f;
    /** Prompt tokens given as embedding rows (none is the padding token 0) and as text around them. */
    private static final int[] ROW_TOKENS = {17, 40, 263, 99, 5};
    private static final int[] TEXT_TOKENS = {71, 120, 8, 301, 64};

    /** A per-layer-embedding fixture: its writer and its embedding width. */
    private static final class Fixture {
        final String name;
        final Supplier<GgufTestWriter> writer;
        final int nEmbd;

        Fixture(String name, Supplier<GgufTestWriter> writer, int nEmbd) {
            this.name = name;
            this.writer = writer;
            this.nEmbd = nEmbd;
        }

        @Override
        public String toString() {
            return name;
        }
    }

    @Parameterized.Parameters(name = "{0}")
    public static List<Object[]> fixtures() {
        return Arrays.asList(
                new Object[] {new Fixture("gemma3n", SyntheticGemmaModels::gemma3n,
                        SyntheticGemmaModels.gemma3nConfig().nEmbd())},
                new Object[] {new Fixture("gemma4", SyntheticGemmaModels::gemma4,
                        SyntheticGemmaModels.gemma4Config().nEmbd())});
    }

    private final Fixture fixture;

    public PleEmbeddingRowsTest(Fixture fixture) {
        this.fixture = fixture;
    }

    /** A contiguous memory. */
    private static EngineConfig config() {
        return config(1);
    }

    /** A contiguous memory for one sequence, a paged memory of {@code parallel} slots otherwise. */
    private static EngineConfig config(int parallel) {
        MemoryConfig.Builder mc = MemoryConfig.builder().nCtx(64).ubatch(8).typeK(KvDType.F32).typeV(KvDType.F32);
        return EngineConfig.builder().memory((parallel > 1 ? mc.parallel(parallel) : mc).build()).build();
    }

    // ------------------------------------------------------------------ CPU

    @Test
    public void theModelTakesEmbeddingRows() throws IOException {
        try (GGUFModel m = new GGUFModel(model().toString(), config())) {
            assertNull("per-layer token embeddings take the padding row", m.forwardBackend().embeddingRowsRefusal());
        }
    }

    @Test
    public void aRowsPrefillReadsThePaddingRowOnTheCpu() throws IOException {
        float[][] out = new float[3][];
        try (GGUFModel m = new GGUFModel(model().toString(), config())) {
            out[0] = m.forwardBackend().prefillEmbeddings(scaledRows(ROW_TOKENS), 0);
            m.forwardBackend().reset();
            out[2] = lastLogits(m.forwardBackend(), ROW_TOKENS);
        }
        try (GGUFModel ref = new GGUFModel(padded().toString(), config())) {
            out[1] = lastLogits(ref.forwardBackend(), ROW_TOKENS);
        }
        assertArrayEquals("embedding rows read row 0 of " + PER_LAYER_TABLE, out[1], out[0], CPU_TOL);
        assertDiffers("the per-layer row of the tokens themselves", out[2], out[0]);
    }

    @Test
    public void aSpanOfRowsBetweenTextReadsThePaddingRowOnTheCpu() throws IOException {
        float[] spliced;
        float[] reference;
        try (GGUFModel m = new GGUFModel(model().toString(), config())) {
            spliced = splicedPrompt(m.forwardBackend());
        }
        try (GGUFModel ref = new GGUFModel(padded().toString(), config())) {
            reference = lastLogits(ref.forwardBackend(), wholePrompt());
        }
        assertArrayEquals("text after the rows sees their padding-row inputs", reference, spliced, CPU_TOL);
    }

    @Test
    public void aPagedUbatchMixingRowsAndTokensReadsThePaddingRowOnTheCpu() throws IOException {
        float[][] mixed;
        float[][] reference;
        float[] contiguous;
        try (GGUFModel m = new GGUFModel(model().toString(), config(2))) {
            mixed = m.forwardBackend().forwardBatch(mixedBatch(true));
        }
        try (GGUFModel ref = new GGUFModel(padded().toString(), config(2))) {
            reference = ref.forwardBackend().forwardBatch(mixedBatch(false));
        }
        try (GGUFModel one = new GGUFModel(model().toString(), config())) {
            contiguous = lastLogits(one.forwardBackend(), TEXT_TOKENS);
        }
        assertRowsEqual("CPU mixed ubatch", reference, mixed, CPU_TOL);
        assertArrayEquals("the token sequence of the paged batch matches its contiguous run", contiguous,
                mixed[SEQ_B], CPU_TOL);
    }

    // ------------------------------------------------------------------ GPU

    @Test
    public void aRowsPrefillReadsThePaddingRowOnTheGpu() throws IOException {
        float[][] out = onGpu(be -> new float[][] {be.prefillEmbeddings(
                scaledRows(ROW_TOKENS), 0)}, model());
        float[][] ref = onGpu(be -> new float[][] {lastLogits(be, ROW_TOKENS)}, padded());
        assertArrayEquals("GPU embedding rows read row 0 of " + PER_LAYER_TABLE, ref[0], out[0], GPU_TOL);
    }

    @Test
    public void aSpanOfRowsBetweenTextReadsThePaddingRowOnTheGpu() throws IOException {
        float[][] spliced = onGpu(be -> new float[][] {splicedPrompt(be)}, model());
        float[][] ref = onGpu(be -> new float[][] {lastLogits(be, wholePrompt())}, padded());
        assertArrayEquals("GPU text after the rows sees their padding-row inputs", ref[0], spliced[0], GPU_TOL);
    }

    @Test
    public void aPagedUbatchMixingRowsAndTokensReadsThePaddingRowOnTheGpu() throws IOException {
        float[][] mixed = onGpu(be -> be.forwardBatch(mixedBatch(true)), model(), 2);
        float[][] ref = onGpu(be -> be.forwardBatch(mixedBatch(false)), padded(), 2);
        float[][] contiguous = onGpu(be -> new float[][] {lastLogits(be, TEXT_TOKENS)}, model(), 1);
        assertRowsEqual("GPU mixed ubatch", ref, mixed, GPU_TOL);
        assertArrayEquals("the token sequence of the paged GPU batch matches its contiguous run", contiguous[0],
                mixed[SEQ_B], GPU_TOL);
    }

    /** Runs {@code call} on the GPU model of {@code model} over a contiguous memory. */
    private static float[][] onGpu(Function<ForwardBackend, float[][]> call, Path model) throws IOException {
        return onGpu(call, model, 1);
    }

    /** Runs {@code call} on the GPU model of {@code model} over {@link #config(int)}. */
    private static float[][] onGpu(Function<ForwardBackend, float[][]> call, Path model, int parallel)
            throws IOException {
        Assume.assumeTrue("OpenCL GPU not available", GpuTestUtil.isOpenCLAvailable());
        HWAcceleration acceleration = new HWAcceleration();
        try (ComputeBackend backend = new OclComputeBackend(acceleration.getCLDevices().get(0));
             GGUFModelGPU gpu = new GGUFModelGPU(model.toString(), backend, config(parallel))) {
            return call.apply(gpu.forwardBackend());
        } finally {
            acceleration.shutdown();
        }
    }

    // ------------------------------------------------------------------ fixtures

    private Path model() throws IOException {
        return fixture.writer.get().writeTempFile();
    }

    /** The fixture with the {@code per_layer_token_embd} rows of {@link #ROW_TOKENS} set to row 0. */
    private Path padded() throws IOException {
        GgufTestWriter w = fixture.writer.get();
        long[] shape = w.tensorShape(PER_LAYER_TABLE);
        assertEquals(GGMLType.F32, w.tensorType(PER_LAYER_TABLE));
        int rowBytes = Math.toIntExact(shape[0] * Float.BYTES);
        byte[] data = w.tensorData(PER_LAYER_TABLE).clone();
        for (int token : ROW_TOKENS) {
            System.arraycopy(data, 0, data, token * rowBytes, rowBytes);
        }
        w.removeTensor(PER_LAYER_TABLE);
        w.tensor(PER_LAYER_TABLE, GGMLType.F32, shape, data);
        return w.writeTempFile();
    }

    /**
     * The {@code token_embd} rows of {@code tokens} times the gemma embedding scale {@code sqrt(n_embd)}: the model
     * input their lookup produces, which an embedding row carries as given.
     */
    private float[][] scaledRows(int[] tokens) {
        GgufTestWriter w = fixture.writer.get();
        int nEmbd = fixture.nEmbd;
        float scale = (float) Math.sqrt(nEmbd);
        ByteBuffer table = ByteBuffer.wrap(w.tensorData("token_embd.weight")).order(ByteOrder.LITTLE_ENDIAN);
        float[][] rows = new float[tokens.length][nEmbd];
        for (int i = 0; i < tokens.length; i++) {
            for (int d = 0; d < nEmbd; d++) {
                rows[i][d] = table.getFloat((tokens[i] * nEmbd + d) * Float.BYTES) * scale;
            }
        }
        return rows;
    }

    /**
     * A media-shaped prompt as {@code GenerationLoop} prefills it: the first half of {@link #TEXT_TOKENS}, the
     * embedding rows of {@link #ROW_TOKENS}, then the second half of the text; the logits of the last text row.
     */
    private float[] splicedPrompt(ForwardBackend be) {
        int half = TEXT_TOKENS.length / 2;
        int[] head = Arrays.copyOfRange(TEXT_TOKENS, 0, half);
        int[] tail = Arrays.copyOfRange(TEXT_TOKENS, half, TEXT_TOKENS.length);
        be.prefill(head, 0);
        be.prefillEmbeddings(scaledRows(ROW_TOKENS), head.length);
        return be.forwardRows(tail, head.length + ROW_TOKENS.length, new int[] {tail.length - 1})[0];
    }

    /** The tokens of {@link #splicedPrompt}, the rows as their own tokens. */
    private static int[] wholePrompt() {
        int half = TEXT_TOKENS.length / 2;
        int[] all = new int[TEXT_TOKENS.length + ROW_TOKENS.length];
        System.arraycopy(TEXT_TOKENS, 0, all, 0, half);
        System.arraycopy(ROW_TOKENS, 0, all, half, ROW_TOKENS.length);
        System.arraycopy(TEXT_TOKENS, half, all, half + ROW_TOKENS.length, TEXT_TOKENS.length - half);
        return all;
    }

    /**
     * {@link #ROW_TOKENS} in sequence A and {@link #TEXT_TOKENS} in sequence B, interleaved so every ubatch mixes
     * both; sequence A as embedding rows when {@code rows}, as tokens otherwise. The last row of each is an output.
     */
    private ForwardBatch mixedBatch(boolean rows) {
        float[][] embd = scaledRows(ROW_TOKENS);
        ForwardBatch.Builder batch = ForwardBatch.builder(ROW_TOKENS.length + TEXT_TOKENS.length);
        for (int i = 0; i < ROW_TOKENS.length; i++) {
            boolean last = i == ROW_TOKENS.length - 1;
            if (rows) {
                batch.addEmbedding(embd[i], i, SEQ_A, last);
            } else {
                batch.add(ROW_TOKENS[i], i, SEQ_A, last);
            }
            batch.add(TEXT_TOKENS[i], i, SEQ_B, last);
        }
        return batch.build();
    }

    private static void assertRowsEqual(String what, float[][] expected, float[][] actual, float tol) {
        assertEquals(what + ": row count", expected.length, actual.length);
        for (int i = 0; i < expected.length; i++) {
            assertArrayEquals(what + ": row " + i, expected[i], actual[i], tol);
        }
    }

    private static float[] lastLogits(ForwardBackend be, int[] tokens) {
        return be.forwardRows(tokens, 0, new int[] {tokens.length - 1})[0];
    }

    private static void assertDiffers(String what, float[] a, float[] b) {
        double maxDiff = 0;
        for (int i = 0; i < a.length; i++) {
            maxDiff = Math.max(maxDiff, Math.abs(a[i] - b[i]));
        }
        assertTrue(what + " changes the logits (max diff " + maxDiff + ")", maxDiff > 1e-2);
    }
}

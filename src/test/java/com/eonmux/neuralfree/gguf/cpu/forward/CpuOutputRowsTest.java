package com.eonmux.neuralfree.gguf.cpu.forward;

import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.engine.MoeCacheConfig;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.ops.moe.MoeCacheRanking;
import com.eonmux.neuralfree.gguf.ops.moe.MoeRankingFile;
import com.eonmux.neuralfree.gguf.ops.moe.RankedExpert;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Random;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Consumer;
import java.util.function.Supplier;
import java.util.stream.Stream;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Output-row pruning of the CPU stack's last layer ({@link CpuOutputRows}, {@link CpuLayerExecutor#forwardRows}):
 * the read rows of a step are found, checked and moved to the front of the stream; the last layer runs its FFN over
 * the read rows alone and its cache writes alone when nothing is read; and the causal calls of the backend that read
 * a few rows (the prefill's last row, a row subset, the greedy ids) return the bits of the run over every row and
 * leave the caches the next token reads, over a dense and a routed (MoE) last layer and for the prefill of embedding
 * rows as for tokens; the rows the routed layers count through the expert-ranking profile show which rows the last
 * layer ran.
 */
public class CpuOutputRowsTest {

    private static final int N_TOKENS = 12;

    @Test
    public void theReadRowsOfAStepAreItsDistinctLocalRowsAscending() {
        assertArrayEquals(new int[]{1, 3}, CpuOutputRows.read(new int[]{11, 9, 3, 9, 1, 30}, 8, 4));
        assertArrayEquals(new int[]{1, 3, 7}, CpuOutputRows.read(new int[]{3, 1, 7, 8}, 0, 8));
        assertArrayEquals(CpuOutputRows.NONE, CpuOutputRows.read(new int[]{11}, 0, 8));
    }

    @Test
    public void onlyAStrictlyAscendingProperSubsetPrunes() {
        assertTrue(CpuOutputRows.subset(8, new int[]{7}));
        assertTrue("nothing read keeps the cache writes", CpuOutputRows.subset(8, CpuOutputRows.NONE));
        assertTrue(CpuOutputRows.subset(8, new int[]{0, 2, 3}));
        assertFalse("every row read", CpuOutputRows.subset(3, new int[]{0, 1, 2}));
        assertFalse("not ascending", CpuOutputRows.subset(8, new int[]{3, 2}));
        assertFalse("a repeated row", CpuOutputRows.subset(8, new int[]{2, 2}));
        assertFalse("past the step", CpuOutputRows.subset(8, new int[]{8}));
        assertFalse("negative", CpuOutputRows.subset(8, new int[]{-1}));
        assertFalse(CpuOutputRows.subset(8, null));
    }

    @Test
    public void compactionMovesEachReadRowToItsIndexInOrder() {
        int width = 3;
        float[] x = new float[6 * width];
        for (int i = 0; i < x.length; i++) {
            x[i] = i;
        }
        CpuOutputRows.compact(x, width, new int[]{0, 2, 3, 5});
        assertArrayEquals(new float[]{0, 1, 2, 6, 7, 8, 9, 10, 11, 15, 16, 17},
                Arrays.copyOf(x, 4 * width), 0f);
        assertEquals("row of the full stream", 5, CpuOutputRows.stackRow(null, 5));
        assertEquals("index among the read rows", 2, CpuOutputRows.stackRow(new int[]{1, 4, 6}, 6));
    }

    /**
     * The layer over four rows read at rows 1 and 3: the FFN runs over two rows, and the stream's first two rows are
     * bit for bit rows 1 and 3 of the whole layer's output; with nothing read the FFN does not run.
     */
    @Test
    public void theLastLayerRunsItsFfnOverTheReadRowsAlone() throws IOException {
        CpuLayerFixture f = new CpuLayerFixture();
        try {
            int n = CpuLayerFixture.UBATCH;
            float[] prompt = stream(n, f.width);
            assertTrue("the llama layer prunes", f.executor.rowsPrunable());

            float[] whole = prompt.clone();
            f.forward(0, n, whole);

            f.memory.reset();
            f.ffnRuns.set(0);
            float[] pruned = prompt.clone();
            f.forwardRows(0, n, pruned, new int[]{1, 3});
            assertEquals(1, f.ffnRuns.get());
            assertEquals("the FFN runs over the read rows", 2, f.ffnRows.get());
            assertArrayEquals(Arrays.copyOfRange(whole, f.width, 2 * f.width),
                    Arrays.copyOfRange(pruned, 0, f.width), 0f);
            assertArrayEquals(Arrays.copyOfRange(whole, 3 * f.width, 4 * f.width),
                    Arrays.copyOfRange(pruned, f.width, 2 * f.width), 0f);

            f.memory.reset();
            f.ffnRuns.set(0);
            f.forwardRows(0, n, prompt.clone(), CpuOutputRows.NONE);
            assertEquals("nothing read runs the cache writes alone", 0, f.ffnRuns.get());

            for (int[] bad : new int[][]{{0, 1, 2, 3}, {2, 1}, {4}}) {
                f.memory.reset();
                assertThrows(Arrays.toString(bad), IllegalArgumentException.class,
                        () -> f.forwardRows(0, n, prompt.clone(), bad));
            }
        } finally {
            f.close();
        }
    }

    /**
     * Over a two-layer llama in steps of eight rows, the calls that read a few rows return the bits of the run over
     * every row: the prefill's last row, a subset spread over both steps, the greedy ids; and a token decoded after
     * the pruned prefill reads the same caches as after the full one.
     */
    @Test
    public void callsThatReadAFewRowsReturnTheBitsOfTheRunOverEveryRow() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        int[] tokens = tokens();
        try (GGUFModel m = new GGUFModel(model.toString(), config())) {
            ForwardBackend b = m.forwardBackend();
            float[][] all = b.forwardRows(tokens, 0, null);
            b.reset();
            b.forwardRows(tokens, 0, null);
            float[] decodedAfterAll = b.decode(9, N_TOKENS).clone();

            b.reset();
            float[] prefilled = b.prefill(tokens, 0).clone();
            assertArrayEquals("the prefill's last row", all[N_TOKENS - 1], prefilled, 0f);
            float[] decodedAfterPrefill = b.decode(9, N_TOKENS).clone();
            assertArrayEquals("the token after the pruned prefill reads the same caches", decodedAfterAll,
                    decodedAfterPrefill, 0f);

            b.reset();
            int[] rows = {11, 1, 5, 9};
            float[][] some = b.forwardRows(tokens, 0, rows);
            for (int i = 0; i < rows.length; i++) {
                assertArrayEquals("row " + rows[i], all[rows[i]], some[i], 0f);
            }

            b.reset();
            int[] ids = b.forwardRowsGreedy(tokens, 0, rows);
            for (int i = 0; i < rows.length; i++) {
                assertEquals("greedy id of row " + rows[i], argmax(all[rows[i]]), ids[i]);
            }
        }
    }

    /**
     * The prefill of embedding rows ({@code stepRows}) prunes its last layer as the token prefill does: over two
     * steps of eight rows its logits and the caches the next token reads are those of the same call with every row
     * kept (a collecting matmul observer installed), over a llama and a routed (MoE) model; and the routed rows the
     * MoE model's layers count show that its last layer ran the read row alone in the pruned prefill and every row
     * in the observed one.
     */
    @Test
    public void anEmbeddingRowPrefillReadsItsLastRowLikeTheRunOverEveryRow() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        try (GGUFModel m = new GGUFModel(model.toString(), config())) {
            embeddingRowPrefillAgainstEveryRow(m);
        }
        long[] routed = routedRows(CpuOutputRowsTest::embeddingRowPrefillAgainstEveryRow);
        assertEquals("the first layer routes every row of both prefills and of the token after each",
                N_TOKENS + 1 + N_TOKENS + 1, routed[0]);
        assertEquals("the last layer routes the pruned prefill's last row alone, then every row of the observed one",
                1 + 1 + N_TOKENS + 1, routed[routed.length - 1]);
    }

    /**
     * The pruned embedding-row prefill of {@code m} and the token after it, against the same calls with every row
     * kept.
     */
    private static void embeddingRowPrefillAgainstEveryRow(GGUFModel m) {
        CpuForwardBackend b = (CpuForwardBackend) m.forwardBackend();
        float[][] rows = rows(N_TOKENS, m.embeddingLength());
        assertTrue("the last step prunes", b.prunes(N_TOKENS - 8, null, new int[]{N_TOKENS - 9}));
        float[] pruned = b.prefillEmbeddings(rows, 0).clone();
        float[] decodedAfterPruned = b.decode(9, N_TOKENS).clone();

        b.reset();
        float[][] every = overEveryRow(() -> new float[][]{b.prefillEmbeddings(rows, 0).clone(),
                b.decode(9, N_TOKENS).clone()});
        assertArrayEquals("the last row's logits", every[0], pruned, 0f);
        assertArrayEquals("the token after the pruned prefill reads the same caches", every[1],
                decodedAfterPruned, 0f);
    }

    /**
     * A routed (MoE) last layer runs its experts over the read rows alone and returns the bits of the run over every
     * row: the prefill's last row, a row subset spread over both steps, the greedy ids of that subset, and the token
     * decoded after the prefill. The routed rows each layer counts show the last layer ran the read rows alone in the
     * pruned calls.
     */
    @Test
    public void aMoeLastLayerOverTheReadRowsReturnsTheBitsOfTheRunOverEveryRow() throws IOException {
        int[] rows = {11, 1, 5, 9};
        long[] routed = routedRows(m -> {
            int[] tokens = tokens();
            CpuForwardBackend b = (CpuForwardBackend) m.forwardBackend();
            assertTrue("the routed last layer prunes", b.prunes(8, null, new int[]{7}));
            float[][] all = b.forwardRows(tokens, 0, null);
            float[] decodedAfterAll = b.decode(9, N_TOKENS).clone();

            b.reset();
            float[] prefilled = b.prefill(tokens, 0).clone();
            assertArrayEquals("the prefill's last row", all[N_TOKENS - 1], prefilled, 0f);
            assertArrayEquals("the token after the pruned prefill reads the same caches", decodedAfterAll,
                    b.decode(9, N_TOKENS), 0f);

            b.reset();
            float[][] some = b.forwardRows(tokens, 0, rows);
            for (int i = 0; i < rows.length; i++) {
                assertArrayEquals("row " + rows[i], all[rows[i]], some[i], 0f);
            }

            b.reset();
            int[] ids = b.forwardRowsGreedy(tokens, 0, rows);
            for (int i = 0; i < rows.length; i++) {
                assertEquals("greedy id of row " + rows[i], argmax(all[rows[i]]), ids[i]);
            }
        });
        assertEquals("the first layer routes every row of every call",
                N_TOKENS + 1 + N_TOKENS + 1 + N_TOKENS + N_TOKENS, routed[0]);
        assertEquals("the last layer routes every row of the full run, then only the prefill's last row and the"
                        + " subset's rows", N_TOKENS + 1 + 1 + 1 + rows.length + rows.length,
                routed[routed.length - 1]);
    }

    /**
     * The rows each layer of the synthetic qwen3moe model routed over the calls of {@code calls}: the experts' heats
     * the model's expert-ranking profile counts and saves at its close, over the experts each row selects.
     */
    private static long[] routedRows(Consumer<GGUFModel> calls) throws IOException {
        SyntheticModels.Config shape = SyntheticModels.Config.qwen3moe();
        Path dir = Files.createTempDirectory("cpu-output-rows");
        try {
            Path model = dir.resolve("model.gguf");
            SyntheticModels.qwen3moe(shape).writeTo(model);
            Path profile = dir.resolve(MoeCacheRanking.RANKING_FILE_NAME);
            EngineConfig profiled = config().toBuilder().moeCache(MoeCacheConfig.off().withRanking(
                    new MoeCacheRanking.Options(MoeCacheRanking.PreloadMode.OFF, null, profile))).build();
            try (GGUFModel m = new GGUFModel(model.toString(), profiled)) {
                calls.accept(m);
            }
            long[] heat = new long[shape.nLayer()];
            for (RankedExpert e : MoeRankingFile.read(profile).entries()) {
                heat[e.layer()] += e.heat();
            }
            long[] rows = new long[heat.length];
            for (int il = 0; il < heat.length; il++) {
                assertEquals("layer " + il + " routes " + shape.nExpertUsed() + " experts per row", 0,
                        heat[il] % shape.nExpertUsed());
                rows[il] = heat[il] / shape.nExpertUsed();
            }
            return rows;
        } finally {
            try (Stream<Path> files = Files.walk(dir)) {
                files.sorted(Comparator.reverseOrder()).forEach(p -> p.toFile().delete());
            }
        }
    }

    /** {@code call} with a collecting matmul observer installed, so every step keeps every row. */
    private static <T> T overEveryRow(Supplier<T> call) {
        AtomicLong observed = new AtomicLong();
        CpuMatVec.observeGlobally((weight, rowStart, rowCount, experts, x, xOff, xStride, cols, nTokens) ->
                observed.addAndGet(nTokens));
        try {
            T result = call.get();
            assertTrue("the observer saw the products", observed.get() > 0);
            return result;
        } finally {
            CpuMatVec.observeGlobally(null);
        }
    }

    /** {@code n} embedding rows of {@code width} values in [-1, 1). */
    private static float[][] rows(int n, int width) {
        Random rnd = new Random(17L * n + width);
        float[][] rows = new float[n][width];
        for (float[] row : rows) {
            for (int i = 0; i < width; i++) {
                row[i] = rnd.nextFloat() * 2f - 1f;
            }
        }
        return rows;
    }

    /** The backend prunes a proper row subset of a causal step, and keeps every row while a matmul tap collects. */
    @Test
    public void thePruningDecisionFollowsTheReadRowsAndTheMatmulTap() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        try (GGUFModel m = new GGUFModel(model.toString(), config())) {
            CpuForwardBackend b = (CpuForwardBackend) m.forwardBackend();
            assertTrue(b.prunes(8, null, new int[]{7}));
            assertTrue(b.prunes(8, null, CpuOutputRows.NONE));
            assertFalse("every row read", b.prunes(2, null, new int[]{0, 1}));
            assertFalse("no read set", b.prunes(8, null, null));
            CpuMatVec.observeGlobally((weight, rowStart, rowCount, experts, x, xOff, xStride, cols, nTokens) -> {
            });
            try {
                assertFalse("a collecting matmul tap sees every row", b.prunes(8, null, new int[]{7}));
            } finally {
                CpuMatVec.observeGlobally(null);
            }
        }
    }

    private static EngineConfig config() {
        return EngineConfig.builder().memory(MemoryConfig.builder().nCtx(64).ubatch(8).typeK(KvDType.F32)
                .typeV(KvDType.F32).build()).build();
    }

    private static int[] tokens() {
        int[] t = new int[N_TOKENS];
        for (int i = 0; i < t.length; i++) {
            t[i] = 4 + (7 + i * 13) % 250;
        }
        return t;
    }

    private static int argmax(float[] v) {
        int best = 0;
        for (int i = 1; i < v.length; i++) {
            if (v[i] > v[best]) {
                best = i;
            }
        }
        return best;
    }

    private static float[] stream(int n, int width) {
        Random rnd = new Random(31L * n);
        float[] x = new float[n * width];
        for (int i = 0; i < x.length; i++) {
            x[i] = rnd.nextFloat() * 2f - 1f;
        }
        return x;
    }
}

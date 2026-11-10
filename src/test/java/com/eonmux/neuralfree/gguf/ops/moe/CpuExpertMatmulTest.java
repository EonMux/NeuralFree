package com.eonmux.neuralfree.gguf.ops.moe;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.cpu.quant.ImatrixCollector;
import com.eonmux.neuralfree.gguf.quant.write.ImatrixFile;
import org.junit.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.Random;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link CpuExpertMatmul} against per-pair {@link CpuMatVec#matvecExpert} products: the counting-sort grouping, and
 * bit-identical outputs for token-row and pair-row inputs, experts with more pairs than one tile block and more rows
 * than one row tile, duplicate ids, per-expert output scales and a weight split into row shards that cut through an
 * expert; and the input observer's per-pair rows.
 */
public class CpuExpertMatmulTest {

    /** F16 bit patterns of small block scales: 1/16, 1/8, 1/4, -1/8, -1/16. */
    private static final short[] HALF_SCALES = {0x2C00, 0x3000, 0x3400, (short) 0xB000, (short) 0xAC00};

    private static float[] random(int n, long seed) {
        Random rnd = new Random(seed);
        float[] v = new float[n];
        for (int i = 0; i < n; i++) {
            v[i] = rnd.nextFloat() * 2.0f - 1.0f;
        }
        return v;
    }

    private static CpuTensorRef f32(int rows, int cols, long seed) {
        float[] values = random(rows * cols, seed);
        ByteBuffer buf = ByteBuffer.allocate(values.length * 4).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < values.length; i++) {
            buf.putFloat(i * 4, values[i]);
        }
        return CpuTensorRef.of(GGMLType.F32, buf, rows, cols);
    }

    /**
     * Q8_0 rows (per 32-value block an F16 scale then 32 random int8 quants) held in one row shard per entry of
     * {@code shardRows}.
     */
    private static CpuTensorRef q8_0Sharded(int cols, long seed, int... shardRows) {
        Random rnd = new Random(seed);
        int rowBytes = cols / 32 * 34;
        int rows = 0;
        for (int r : shardRows) {
            rows += r;
        }
        byte[] b = new byte[rows * rowBytes];
        for (int k = 0; k < rows * (cols / 32); k++) {
            short half = HALF_SCALES[rnd.nextInt(HALF_SCALES.length)];
            b[k * 34] = (byte) half;
            b[k * 34 + 1] = (byte) (half >> 8);
            for (int j = 0; j < 32; j++) {
                b[k * 34 + 2 + j] = (byte) (rnd.nextInt(256) - 128);
            }
        }
        ByteBuffer[] shards = new ByteBuffer[shardRows.length];
        long[] offsets = new long[shardRows.length];
        int row = 0;
        for (int i = 0; i < shardRows.length; i++) {
            offsets[i] = (long) row * rowBytes;
            shards[i] = ByteBuffer.wrap(b, row * rowBytes, shardRows[i] * rowBytes).slice()
                    .order(ByteOrder.LITTLE_ENDIAN);
            row += shardRows[i];
        }
        return new CpuTensorRef(GGMLType.Q8_0, shards, offsets, rows, cols, rowBytes);
    }

    /**
     * K-quant rows (Q4_K or Q6_K super-blocks of random bytes) whose F16 scales are small finite values, so every
     * row dots to a finite value.
     */
    private static CpuTensorRef kQuant(GGMLType type, int rows, int cols, long seed) {
        Random rnd = new Random(seed);
        int blockBytes = type == GGMLType.Q4_K ? 144 : 210;
        int[] halves = type == GGMLType.Q4_K ? new int[]{0, 2} : new int[]{208};
        int blocks = rows * (cols / 256);
        byte[] b = new byte[blocks * blockBytes];
        rnd.nextBytes(b);
        for (int k = 0; k < blocks; k++) {
            for (int h : halves) {
                short half = HALF_SCALES[rnd.nextInt(HALF_SCALES.length)];
                b[k * blockBytes + h] = (byte) half;
                b[k * blockBytes + h + 1] = (byte) (half >> 8);
            }
        }
        return CpuTensorRef.of(type, ByteBuffer.wrap(b).order(ByteOrder.LITTLE_ENDIAN), rows, cols);
    }

    /** The reference: one {@link CpuMatVec#matvecExpert} per pair. */
    private static float[] perPair(CpuMatVec mv, CpuTensorRef w, int nExpert, int[] ids, int pairs, float[] x,
                                   int pairsPerRow) {
        int perExpert = (int) (w.rows() / nExpert);
        int cols = (int) w.cols();
        float[] y = new float[pairs * perExpert];
        for (int p = 0; p < pairs; p++) {
            mv.matvecExpert(w, ids[p], nExpert, x, (p / pairsPerRow) * cols, y, p * perExpert);
        }
        return y;
    }

    private static float[] grouped(CpuMatVec mv, CpuTensorRef w, int nExpert, int[] ids, int pairs, float[] x,
                                   int pairsPerRow) {
        int perExpert = (int) (w.rows() / nExpert);
        float[] y = new float[pairs * perExpert];
        CpuExpertMatmul.Groups g = CpuExpertMatmul.Groups.of(ids, pairs, nExpert);
        new CpuExpertMatmul(mv).run(w, nExpert, g, x, 0, pairsPerRow, y, 0);
        return y;
    }

    // ------------------------------------------------------------------ grouping

    @Test
    public void groupsAreACountingSortThatKeepsPairOrder() {
        int[] ids = {3, 1, 3, 0, 1, 3, 9};
        CpuExpertMatmul.Groups g = CpuExpertMatmul.Groups.of(ids, 6, 5);
        assertArrayEquals(new int[]{0, 1, 3}, g.active);
        assertArrayEquals(new int[]{0, 1, 3, 6}, g.start);
        assertArrayEquals(new int[]{3, 1, 4, 0, 2, 5}, g.order);
        assertEquals(3, g.count(2));
    }

    @Test
    public void groupsRejectIdsOutsideTheExperts() {
        assertThrows(IllegalArgumentException.class, () -> CpuExpertMatmul.Groups.of(new int[]{0, 5}, 2, 5));
        assertThrows(IllegalArgumentException.class, () -> CpuExpertMatmul.Groups.of(new int[]{-1}, 1, 5));
        assertThrows(IllegalArgumentException.class, () -> CpuExpertMatmul.Groups.of(new int[]{0}, 2, 5));
    }

    // ------------------------------------------------------------------ products

    /**
     * Explicit source rows: a compacted host pair list of a hybrid step reads its token rows through
     * {@code srcRows}; the grouped product matches one {@link CpuMatVec#matvecExpert} per pair over the same rows,
     * for repeated rows (pairs of one token) and a non-contiguous subset.
     */
    @Test
    public void explicitSourceRowsMatchThePerPairProducts() {
        CpuMatVec mv = CpuMatVec.PLAIN;
        int nExpert = 4;
        int cols = 64;
        CpuTensorRef w = f32(nExpert * 32, cols, 17L);
        int tokens = 6;
        float[] x = random(tokens * cols, 5L);
        // pairs (token, expert) of a step with k = 2, of which the host keeps a non-contiguous subset
        int[] srcRows = {0, 0, 3, 3, 3, 5};
        int[] ids = {1, 2, 2, 0, 3, 1};
        int pairs = ids.length;
        int perExpert = 32;
        float[] y = new float[pairs * perExpert];
        CpuExpertMatmul.Groups g = CpuExpertMatmul.Groups.of(ids, pairs, nExpert);
        new CpuExpertMatmul(mv).run(w, nExpert, g, x, 0, srcRows, y, 0);
        for (int p = 0; p < pairs; p++) {
            float[] ref = new float[perExpert];
            mv.matvecExpert(w, ids[p], nExpert, x, srcRows[p] * cols, ref, 0);
            assertArrayEquals(ref, Arrays.copyOfRange(y, p * perExpert, (p + 1) * perExpert), 1e-4f);
        }
        assertThrows(IllegalArgumentException.class,
                () -> new CpuExpertMatmul(mv).run(w, nExpert, g, x, 0, (int[]) null, new float[pairs * perExpert], 0));
    }

    /**
     * Token-row input (k = 2) on a sharded Q8_0 tensor with per-expert scales: expert 0 takes 20 pairs (two pair
     * blocks) and every expert has 70 rows (two row tiles), with shard cuts inside experts 1 and 3.
     */
    @Test
    public void tokenRowsMatchPerPairProductsOnShardedScaledWeights() {
        int nExpert = 5;
        int perExpert = 70;
        int cols = 64;
        int k = 2;
        int n = 20;
        CpuTensorRef w = q8_0Sharded(cols, 7L, 100, 150, 100);
        float[] scale = {0.5f, 2.0f, -1.25f, 1.0f, 3.0f};
        CpuMatVec mv = new CpuMatVec(t -> t == w ? CpuMatVec.Transforms.of(null, scale, nExpert)
                : CpuMatVec.Transforms.NONE);
        int[] ids = new int[n * k];
        for (int t = 0; t < n; t++) {
            ids[t * k] = 0;
            ids[t * k + 1] = 1 + (t * 7) % (nExpert - 1);
        }
        float[] x = random(n * cols, 11L);
        assertEquals(perExpert * nExpert, w.rows());
        assertArrayEquals(perPair(mv, w, nExpert, ids, n * k, x, k), grouped(mv, w, nExpert, ids, n * k, x, k), 0f);
    }

    /**
     * K-quant experts through the batched tile kernels: one expert holds more pairs than a pair block and more rows
     * than a row tile, another a single pair, and the outputs equal one product per pair bit for bit.
     */
    @Test
    public void kQuantExpertsMatchPerPairProductsThroughTheTileKernels() {
        int nExpert = 4;
        int perExpert = 70;
        int cols = 512;
        int n = 20;
        int k = 2;
        int[] ids = new int[n * k];
        for (int t = 0; t < n; t++) {
            ids[t * k] = 1;
            ids[t * k + 1] = t == 3 ? 0 : 2 + t % 2;
        }
        float[] x = random(n * cols, 31L);
        for (GGMLType type : new GGMLType[]{GGMLType.Q4_K, GGMLType.Q6_K}) {
            CpuTensorRef w = kQuant(type, nExpert * perExpert, cols, 17L + type.ordinal());
            CpuMatVec mv = CpuMatVec.PLAIN;
            assertArrayEquals(type.toString(), perPair(mv, w, nExpert, ids, n * k, x, k),
                    grouped(mv, w, nExpert, ids, n * k, x, k), 0f);
        }
    }

    /**
     * The parts of a grouped product hold at most one slab of one expert's pairs, start on slab multiples within
     * their expert, cover every pair once in pair order, and the passes hold at most one slab of pairs each.
     */
    @Test
    public void partsAndPassesBoundThePreparedLayoutsToOneSlab() {
        int nExpert = 5;
        int[] ids = new int[100];
        for (int p = 0; p < ids.length; p++) {
            ids[p] = p < 70 ? 1 : p < 75 ? 3 : p % 2 == 0 ? 0 : 4;
        }
        CpuExpertMatmul.Groups g = CpuExpertMatmul.Groups.of(ids, ids.length, nExpert);
        int slab = 16;
        CpuExpertMatmul.Parts parts = CpuExpertMatmul.Parts.of(g, slab);
        int[] seen = new int[ids.length];
        int last = -1;
        for (int i = 0; i < parts.count.length; i++) {
            int a = parts.slot[i];
            assertTrue("part " + i + " holds " + parts.count[i], parts.count[i] >= 1 && parts.count[i] <= slab);
            assertEquals("part " + i + " starts on a slab", 0, (parts.first[i] - g.start[a]) % slab);
            for (int k = 0; k < parts.count[i]; k++) {
                int q = parts.first[i] + k;
                assertTrue(q > last);
                last = q;
                seen[g.order[q]]++;
                assertEquals(g.active[a], ids[g.order[q]]);
            }
        }
        for (int p = 0; p < ids.length; p++) {
            assertEquals("pair " + p, 1, seen[p]);
        }
        assertEquals(0, parts.passStart[0]);
        assertEquals(parts.count.length, parts.passStart[parts.passes()]);
        for (int pass = 0; pass < parts.passes(); pass++) {
            int pairs = 0;
            for (int i = parts.passStart[pass]; i < parts.passStart[pass + 1]; i++) {
                pairs += parts.count[i];
            }
            assertTrue("pass " + pass + " holds " + pairs + " pairs", pairs >= 1 && pairs <= slab);
        }
    }

    /**
     * An expert routed more pairs than one token slab ({@link CpuMatVec#tokenSlab}) runs over several parts and
     * passes with the per-pair products' values.
     */
    @Test
    public void anExpertWithMorePairsThanOneSlabMatchesPerPairProducts() {
        int nExpert = 2;
        int perExpert = 2;
        int cols = 1 << 16;
        CpuTensorRef w = kQuant(GGMLType.Q4_K, nExpert * perExpert, cols, 61L);
        CpuMatVec mv = CpuMatVec.PLAIN;
        int slab = CpuMatVec.tokenSlab(cols, CpuMatVec.vecDot(GGMLType.Q4_K));
        int pairs = slab + 44 + 20;
        int[] ids = new int[pairs];
        int[] srcRows = new int[pairs];
        for (int p = 0; p < pairs; p++) {
            ids[p] = p < slab + 44 ? 0 : 1;
            srcRows[p] = p % 4;
        }
        CpuExpertMatmul.Groups g = CpuExpertMatmul.Groups.of(ids, pairs, nExpert);
        CpuExpertMatmul.Parts parts = CpuExpertMatmul.Parts.of(g, cols, CpuMatVec.vecDot(GGMLType.Q4_K));
        assertEquals("expert 0 in two parts, expert 1 in one", 3, parts.count.length);
        for (int pass = 0; pass < parts.passes(); pass++) {
            int inPass = 0;
            for (int i = parts.passStart[pass]; i < parts.passStart[pass + 1]; i++) {
                inPass += parts.count[i];
            }
            assertTrue("pass " + pass + " prepares " + inPass + " pairs", inPass <= slab);
        }
        float[] x = random(4 * cols, 63L);
        float[] y = new float[pairs * perExpert];
        new CpuExpertMatmul(mv).run(w, nExpert, g, x, 0, srcRows, y, 0);
        for (int p = 0; p < pairs; p++) {
            float[] ref = new float[perExpert];
            mv.matvecExpert(w, ids[p], nExpert, x, srcRows[p] * cols, ref, 0);
            assertArrayEquals("pair " + p, ref, Arrays.copyOfRange(y, p * perExpert, (p + 1) * perExpert), 0f);
        }
    }

    /** Pair-row input (every pair its own row) on F32 weights, with a token that picks one expert twice. */
    @Test
    public void pairRowsMatchPerPairProductsWithDuplicateIds() {
        int nExpert = 4;
        int perExpert = 33;
        int cols = 24;
        int pairs = 9;
        CpuTensorRef w = f32(nExpert * perExpert, cols, 3L);
        CpuMatVec mv = CpuMatVec.PLAIN;
        int[] ids = {2, 2, 0, 3, 1, 2, 0, 0, 3};
        float[] x = random(pairs * cols, 5L);
        assertArrayEquals(perPair(mv, w, nExpert, ids, pairs, x, 1), grouped(mv, w, nExpert, ids, pairs, x, 1), 0f);
    }

    /** One decode token (k = 3) touches three experts once each. */
    @Test
    public void singleTokenMatchesPerPairProducts() {
        int nExpert = 8;
        int cols = 64;
        CpuTensorRef w = q8_0Sharded(cols, 21L, 8 * 40);
        CpuMatVec mv = CpuMatVec.PLAIN;
        int[] ids = {6, 1, 4};
        float[] x = random(cols, 13L);
        assertArrayEquals(perPair(mv, w, nExpert, ids, 3, x, 3), grouped(mv, w, nExpert, ids, 3, x, 3), 0f);
    }

    // ------------------------------------------------------------------ input observer

    /**
     * The installed {@link CpuMatVec.InputObserver} sees the grouped product's inputs exactly as it sees the per-pair
     * {@link CpuMatVec#matvecExpert} calls: one row per (token, slot) pair with its expert, so an imatrix collection
     * gets the same per-expert statistics on either path.
     */
    @Test
    public void theObserverSeesEveryPairAsThePerPairProductsReportIt() {
        CpuMatVec mv = CpuMatVec.PLAIN;
        int nExpert = 4;
        int cols = 64;
        CpuTensorRef w = f32(nExpert * 16, cols, 29L);
        int[] ids = {2, 0, 2, 3, 1, 2};
        float[] x = random(3 * cols, 31L);
        ImatrixCollector perPair = new ImatrixCollector(t -> "w");
        CpuMatVec.observeGlobally(perPair);
        try {
            perPair(mv, w, nExpert, ids, ids.length, x, 2);
        } finally {
            CpuMatVec.observeGlobally(null);
        }
        ImatrixCollector grouped = new ImatrixCollector(t -> "w");
        CpuMatVec.observeGlobally(grouped);
        try {
            grouped(mv, w, nExpert, ids, ids.length, x, 2);
        } finally {
            CpuMatVec.observeGlobally(null);
        }
        ImatrixFile.Entry a = perPair.entries().get(0);
        ImatrixFile.Entry b = grouped.entries().get(0);
        assertEquals(nExpert, b.nMat());
        assertArrayEquals(new long[]{1, 1, 3, 1}, b.counts());
        assertArrayEquals(a.counts(), b.counts());
        assertArrayEquals(a.sumSq(), b.sumSq(), 0f);
        assertArrayEquals(a.sums(), b.sums(), 0f);
    }

    // ------------------------------------------------------------------ validation

    @Test
    public void expertCountsMustAgree() {
        CpuTensorRef w = f32(12, 8, 1L);
        CpuExpertMatmul m = new CpuExpertMatmul(CpuMatVec.PLAIN);
        float[] x = new float[8];
        float[] y = new float[12];
        CpuExpertMatmul.Groups three = CpuExpertMatmul.Groups.of(new int[]{0}, 1, 3);
        assertThrows(IllegalArgumentException.class, () -> m.run(w, 5, three, x, 0, 1, y, 0));
        assertThrows(IllegalArgumentException.class, () -> m.run(w, 4, three, x, 0, 1, y, 0));
        CpuMatVec known = new CpuMatVec(t -> CpuMatVec.Transforms.of(null, null, 4));
        CpuExpertMatmul withKnown = new CpuExpertMatmul(known);
        assertThrows(IllegalArgumentException.class, () -> withKnown.run(w, 3, three, x, 0, 1, y, 0));
    }

    @Test
    public void viewsMustFitTheirArrays() {
        CpuTensorRef w = f32(12, 8, 1L);
        CpuExpertMatmul m = new CpuExpertMatmul(CpuMatVec.PLAIN);
        CpuExpertMatmul.Groups g = CpuExpertMatmul.Groups.of(new int[]{0, 2}, 2, 3);
        assertThrows(IllegalArgumentException.class, () -> m.run(w, 3, g, new float[15], 0, 1, new float[8], 0));
        assertThrows(IllegalArgumentException.class, () -> m.run(w, 3, g, new float[16], 0, 1, new float[7], 0));
    }
}

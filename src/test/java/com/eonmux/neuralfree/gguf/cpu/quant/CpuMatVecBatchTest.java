package com.eonmux.neuralfree.gguf.cpu.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuVecDot.QuantizedActivations;
import com.eonmux.neuralfree.gguf.cpu.repack.CpuRepack;
import com.eonmux.neuralfree.gguf.quant.HadamardSpec;
import org.junit.After;
import org.junit.Assume;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The batched {@link CpuMatVec} products: the {@link CpuTaskGrid} covers every (row, token) output exactly once with
 * tile-aligned panels, batched products over several row chunks and token panels (plain and output-scaled) equal
 * one matvec per token bit for bit, and {@link CpuMatVec#matmulGroup} equals one matmul per
 * weight (output-scaled and repacked weights included), shares one quantization between weights of one activation
 * type, falls back for input transforms, shows
 * the observer each weight once and rejects mismatched weights by name.
 */
public class CpuMatVecBatchTest {

    private static final int[] TOKEN_COUNTS = {1, 7, 8, 9, 63, 64, 65};

    @After
    public void tearDown() {
        CpuMatVec.observeGlobally(null);
    }

    private static CpuTensorRef weight(GGMLType t, int rows, int cols, long seed) {
        return CpuTensorRef.of(t, SimdVecDotTest.randomRows(t, rows, cols, new Random(seed)), rows, cols);
    }

    private static float[] input(int n, long seed) {
        return SimdVecDotTest.gaussian(n, new Random(seed));
    }

    /** One matvec per token: the reference of every batched product. */
    private static float[] perToken(CpuMatVec mv, CpuTensorRef w, float[] x, int nTokens) {
        int rows = (int) w.rows();
        int cols = (int) w.cols();
        float[] y = new float[nTokens * rows];
        for (int t = 0; t < nTokens; t++) {
            mv.matvec(w, x, t * cols, y, t * rows);
        }
        return y;
    }

    /** The tiling of the int16 kernels (Q4_K, Q6_K): two rows, panels of 8-token tile pairs, two-byte lanes. */
    private static final CpuTaskGrid.Tiling INT16 = CpuTaskGrid.Tiling.of(8, 2, 16, 2);

    /** The tiling of the Q8_0 float tiles: four rows by two eight-token tiles of four-byte lanes. */
    private static final CpuTaskGrid.Tiling FLOAT = CpuTaskGrid.Tiling.of(8, 4, 16, 4);

    @Test
    public void theGridCoversEveryOutputOnceWithTileAlignedPanels() {
        int[][] shapes = {{1, 1, 32, 34}, {70, 65, 4096, 4352}, {3, 513, 256, 288}, {1000, 9, 1024, 1088},
                {151, 300, 8192, 8704}, {2, 2, 32, 34}};
        for (CpuTaskGrid.Tiling tiling : new CpuTaskGrid.Tiling[]{INT16, FLOAT, CpuTaskGrid.Tiling.UNPREPARED}) {
            for (int[] s : shapes) {
                for (int workers : new int[]{1, 8, 24}) {
                    CpuTaskGrid g = CpuTaskGrid.of(s[0], s[1], s[2], s[3], tiling, workers);
                    int[] seen = new int[s[0] * s[1]];
                    for (int task = 0; task < g.tasks(); task++) {
                        assertEquals(g + " token start", 0, g.tokenStart(task) % tiling.panelTokens);
                        assertEquals(g + " row chunks hold whole tiles", 0, g.rowStart(task) % tiling.rows);
                        assertTrue(g + " task " + task + " is empty", g.rows(task) > 0 && g.tokens(task) > 0);
                        for (int t = g.tokenStart(task); t < g.tokenStart(task) + g.tokens(task); t++) {
                            for (int r = g.rowStart(task); r < g.rowStart(task) + g.rows(task); r++) {
                                seen[t * s[0] + r]++;
                            }
                        }
                    }
                    for (int i = 0; i < seen.length; i++) {
                        assertEquals(g + " output " + i, 1, seen[i]);
                    }
                }
            }
        }
        assertThrows(IllegalArgumentException.class, () -> CpuTaskGrid.of(0, 1, 32, 34, INT16, 1));
        assertThrows(IllegalArgumentException.class, () -> CpuTaskGrid.of(1, 1, 32, 34, INT16, 0));
        assertThrows(IllegalArgumentException.class, () -> CpuTaskGrid.Tiling.of(8, 0, 8, 2));
    }

    /**
     * Panels hold about 128 Ki laid-out elements in whole multiples of the kernel's panel tokens, and slabs 32 MiB of
     * its lanes: at 9728 columns the 13 tokens that fit round up to one 16-token pair of tiles, without a layout to
     * one 8-token tile, and the int16 kernels' slabs hold twice the tokens of the Q8_0 float tiles' four-byte lanes.
     */
    @Test
    public void panelsAndSlabsFollowTheKernelsTiling() {
        assertEquals("128 Ki elements hold 32 tokens of 4096 columns", 32,
                CpuTaskGrid.of(64, 512, 4096, 4352, INT16, 8).tokens(0));
        assertEquals(32, CpuTaskGrid.of(64, 512, 4096, 4352, FLOAT, 8).tokens(0));
        assertEquals("13 tokens fit, a panel holds at least its tile pair", 16,
                CpuTaskGrid.of(64, 512, 9728, 10336, INT16, 8).tokens(0));
        assertEquals(16, CpuTaskGrid.of(64, 512, 9728, 10336, FLOAT, 8).tokens(0));
        assertEquals("without a layout, one 8-token panel", 8,
                CpuTaskGrid.of(64, 512, 9728, 10336, CpuTaskGrid.Tiling.UNPREPARED, 8).tokens(0));
        assertEquals((32 << 20) / (2 * 9728) / 16 * 16, CpuTaskGrid.slabTokens(9728, INT16));
        assertEquals((32 << 20) / (4 * 9728) / 16 * 16, CpuTaskGrid.slabTokens(9728, FLOAT));
        CpuTaskGrid.Tiling joined = INT16.join(FLOAT);
        assertEquals("a group satisfies both tiles", 16, joined.panelTokens);
        assertEquals(4, joined.rows);
        assertEquals("and bounds the wider layout", 4, joined.laneBytes);
        Assume.assumeTrue(CpuSimd.available());
        for (GGMLType type : new GGMLType[]{GGMLType.Q4_K, GGMLType.Q6_K}) {
            CpuVecDot dot = CpuSimd.vecDot(type);
            assertEquals(type + " slabs hold 32 MiB of two-byte lanes", (32 << 20) / (2 * 9728) / 16 * 16,
                    CpuMatVec.tokenSlab(9728, dot));
            assertEquals(type + " register tile rows", 2, dot.tileRows());
            assertEquals(type + " panel tokens", 16, dot.tilePanelTokens());
            assertEquals(type + " lane bytes", 2, dot.tileLaneBytes());
        }
        CpuVecDot q8 = CpuSimd.vecDot(GGMLType.Q8_0);
        assertEquals("Q8_0 float tiles", (32 << 20) / (4 * 9728) / 16 * 16, CpuMatVec.tokenSlab(9728, q8));
        assertEquals(4, q8.tileRows());
        assertEquals(16, q8.tilePanelTokens());
        assertEquals(4, q8.tileLaneBytes());
    }

    @Test
    public void batchedProductsOverSeveralChunksAndPanelsMatchOneMatvecPerToken() {
        int rows = 70;
        int cols = 4096;
        float[] scale = {0.75f};
        for (GGMLType type : new GGMLType[]{GGMLType.Q8_0, GGMLType.Q4_K, GGMLType.Q6_K, GGMLType.F16}) {
            CpuTensorRef w = weight(type, rows, cols, 41L + type.ordinal());
            CpuMatVec scaled = new CpuMatVec(t -> t == w ? CpuMatVec.Transforms.of(null, scale, 0)
                    : CpuMatVec.Transforms.NONE);
            for (int n : TOKEN_COUNTS) {
                float[] x = input(n * cols, 7L * n);
                float[] y = new float[n * rows];
                CpuMatVec.PLAIN.matmul(w, x, 0, y, 0, n);
                assertArrayEquals(type + " plain, " + n + " tokens", perToken(CpuMatVec.PLAIN, w, x, n), y,
                        0f);
                float[] ys = new float[n * rows];
                scaled.matmul(w, x, 0, ys, 0, n);
                assertArrayEquals(type + " scaled, " + n + " tokens", perToken(scaled, w, x, n), ys, 0f);
            }
        }
    }

    @Test
    public void aGroupEqualsOneMatmulPerWeightAndQuantizesOncePerActivationType() {
        int cols = 512;
        CpuTensorRef q = weight(GGMLType.Q8_0, 40, cols, 1L);
        CpuTensorRef k = weight(GGMLType.Q4_K, 9, cols, 2L);
        CpuTensorRef v = weight(GGMLType.Q6_K, 17, cols, 3L);
        CpuTensorRef[] ws = {q, k, v};
        for (int n : TOKEN_COUNTS) {
            float[] x = input(n * cols, 100L + n);
            float[][] ys = {new float[n * 40], new float[n * 9], new float[n * 17]};
            CpuMatVec.PLAIN.matmulGroup(ws, x, 0, ys, n);
            for (int i = 0; i < ws.length; i++) {
                float[] ref = new float[ys[i].length];
                CpuMatVec.PLAIN.matmul(ws[i], x, 0, ref, 0, n);
                assertArrayEquals(ws[i].type() + " in a group, " + n + " tokens", ref, ys[i], 0f);
            }
        }
    }

    @Test
    public void aGroupOfScaledAndRepackedWeightsEqualsOneMatmulPerWeight() {
        Assume.assumeTrue(CpuSimd.available());
        int cols = 2048;
        CpuTensorRef q = weight(GGMLType.Q8_0, 24, cols, 21L);
        CpuTensorRef k = weight(GGMLType.Q4_K, 16, cols, 22L);
        CpuTensorRef v = weight(GGMLType.Q6_K, 10, cols, 23L);
        CpuTensorRef[] ws = {q, k, v};
        float[] qScale = {0.75f};
        float[] vScale = {1.5f};
        CpuMatVec.TransformSource scales = t -> t == q ? CpuMatVec.Transforms.of(null, qScale, 0)
                : t == v ? CpuMatVec.Transforms.of(null, vScale, 0) : CpuMatVec.Transforms.NONE;
        CpuRepack repack = new CpuRepack(CpuRepack.Mode.ON);
        CpuMatVec mv = new CpuMatVec(scales, repack::planFor);
        for (int n : TOKEN_COUNTS) {
            float[] x = input(n * cols, 300L + n);
            float[][] ys = {new float[n * 24], new float[n * 16], new float[n * 10]};
            mv.matmulGroup(ws, x, 0, ys, n);
            for (int i = 0; i < ws.length; i++) {
                float[] ref = new float[ys[i].length];
                mv.matmul(ws[i], x, 0, ref, 0, n);
                assertArrayEquals(ws[i].type() + " scaled or repacked in a group, " + n + " tokens", ref, ys[i], 0f);
                float[] plain = perToken(mv, ws[i], x, n);
                assertArrayEquals(ws[i].type() + " equals one matvec per token, " + n + " tokens", plain, ys[i], 0f);
            }
        }
        assertTrue(repack.summary(), repack.repackedCount() >= 2);
    }

    /**
     * A batch of more tokens than one slab ({@link CpuMatVec#tokenSlab}) prepares layouts of at most one slab, for
     * one weight and for a group, with the values of one matvec per token.
     */
    @Test
    public void aBatchLargerThanOneSlabPreparesBoundedLayoutsWithTheSameValues() {
        Assume.assumeTrue(CpuSimd.available());
        int cols = 1 << 16;
        CpuVecDot simd = CpuSimd.vecDot(GGMLType.Q8_0);
        int slab = CpuMatVec.tokenSlab(cols, simd);
        assertEquals("a slab holds 32 MiB of four-byte lanes", (32 << 20) / (4 * cols), slab);
        int n = slab + 9;
        CpuTensorRef a = weight(GGMLType.Q8_0, 3, cols, 71L);
        CpuTensorRef b = weight(GGMLType.Q8_0, 2, cols, 72L);
        RecordingDot recording = new RecordingDot(simd);
        CpuMatVec mv = new CpuMatVec(w -> CpuMatVec.Transforms.NONE, w -> new CpuMatVec.MatvecPlan(w, recording));
        float[] x = input(n * cols, 73L);
        float[] y = new float[n * 3];
        mv.matmul(a, x, 0, y, 0, n);
        assertArrayEquals("one weight over " + n + " tokens", perToken(CpuMatVec.PLAIN, a, x, n), y, 0f);
        assertEquals("the layouts of one product", 2, recording.layouts.get());
        assertEquals(slab, recording.largest.get());

        recording.layouts.set(0);
        recording.largest.set(0);
        float[][] ys = {new float[n * 3], new float[n * 2]};
        mv.matmulGroup(new CpuTensorRef[]{a, b}, x, 0, ys, n);
        assertArrayEquals(perToken(CpuMatVec.PLAIN, a, x, n), ys[0], 0f);
        assertArrayEquals(perToken(CpuMatVec.PLAIN, b, x, n), ys[1], 0f);
        assertEquals("one shared layout per slab of the group", 2, recording.layouts.get());
        assertEquals(slab, recording.largest.get());
    }

    /** A kernel that delegates to another and records the token counts of the layouts it prepares. */
    private static final class RecordingDot implements CpuVecDot {
        private final CpuVecDot inner;
        final AtomicInteger layouts = new AtomicInteger();
        final AtomicInteger largest = new AtomicInteger();

        RecordingDot(CpuVecDot inner) {
            this.inner = inner;
        }

        @Override
        public float dotRow(CpuTensorRef w, long row, QuantizedActivations x) {
            return inner.dotRow(w, row, x);
        }

        @Override
        public float dotRowBytes(byte[] row, int off, QuantizedActivations x) {
            return inner.dotRowBytes(row, off, x);
        }

        @Override
        public void dotRows(CpuTensorRef w, long rowStart, int rowCount, QuantizedActivations x, float[] y,
                            int yOff) {
            inner.dotRows(w, rowStart, rowCount, x, y, yOff);
        }

        @Override
        public TileActivations prepareTiles(QuantizedActivations[] x, int nTokens) {
            layouts.incrementAndGet();
            largest.accumulateAndGet(nTokens, Math::max);
            return inner.prepareTiles(x, nTokens);
        }

        @Override
        public int tileTokens() {
            return inner.tileTokens();
        }

        @Override
        public int tileRows() {
            return inner.tileRows();
        }

        @Override
        public int tilePanelTokens() {
            return inner.tilePanelTokens();
        }

        @Override
        public int tileLaneBytes() {
            return inner.tileLaneBytes();
        }

        @Override
        public void dotTiles(CpuTensorRef w, long rowStart, int rowCount, TileActivations tiles,
                             QuantizedActivations[] x, int tokenStart, int nTokens, float[] y, int yOff,
                             int yStride) {
            inner.dotTiles(w, rowStart, rowCount, tiles, x, tokenStart, nTokens, y, yOff, yStride);
        }
    }

    @Test
    public void anInputTransformMakesEveryWeightRunItsOwnProduct() {
        int cols = 256;
        CpuTensorRef a = weight(GGMLType.Q8_0, 12, cols, 5L);
        CpuTensorRef b = weight(GGMLType.Q8_0, 6, cols, 6L);
        CpuMatVec mv = new CpuMatVec(t -> t == b ? CpuMatVec.Transforms.of(HadamardSpec.rotation(64), null, 0)
                : CpuMatVec.Transforms.NONE);
        int n = 9;
        float[] x = input(n * cols, 8L);
        float[][] ys = {new float[n * 12], new float[n * 6]};
        mv.matmulGroup(new CpuTensorRef[]{a, b}, x, 0, ys, n);
        float[] ra = new float[n * 12];
        float[] rb = new float[n * 6];
        mv.matmul(a, x, 0, ra, 0, n);
        mv.matmul(b, x, 0, rb, 0, n);
        assertArrayEquals(ra, ys[0], 0f);
        assertArrayEquals("the rotated weight", rb, ys[1], 0f);
    }

    @Test
    public void theObserverSeesEveryGroupedWeightOnceAsSeparateProductsShowIt() {
        int cols = 64;
        CpuTensorRef a = weight(GGMLType.Q8_0, 4, cols, 9L);
        CpuTensorRef b = weight(GGMLType.Q4_K, 2, 256, 10L);
        CpuTensorRef c = weight(GGMLType.Q8_0, 3, cols, 11L);
        List<String> seen = Collections.synchronizedList(new ArrayList<>());
        CpuMatVec.observeGlobally((w, rowStart, rows, experts, x, xOff, xStride, nCols, nTokens) ->
                seen.add(w.type() + ":" + rowStart + "+" + rows + "x" + nCols + "/" + nTokens + "@" + x[xOff]));
        float[] x = input(5 * cols, 12L);
        CpuMatVec.PLAIN.matmulGroup(new CpuTensorRef[]{a, c}, x, 0, new float[][]{new float[20], new float[15]}, 5);
        List<String> grouped = new ArrayList<>(seen);
        seen.clear();
        CpuMatVec.PLAIN.matmul(a, x, 0, new float[20], 0, 5);
        CpuMatVec.PLAIN.matmul(c, x, 0, new float[15], 0, 5);
        assertEquals(seen, grouped);

        IllegalArgumentException mixed = assertThrows(IllegalArgumentException.class, () -> CpuMatVec.PLAIN
                .matmulGroup(new CpuTensorRef[]{a, b}, x, 0, new float[][]{new float[20], new float[10]}, 5));
        assertTrue(mixed.getMessage(), mixed.getMessage().contains("columns"));
        IllegalArgumentException counts = assertThrows(IllegalArgumentException.class, () -> CpuMatVec.PLAIN
                .matmulGroup(new CpuTensorRef[]{a, c}, x, 0, new float[][]{new float[20]}, 5));
        assertTrue(counts.getMessage(), counts.getMessage().contains("outputs"));
        assertThrows(IllegalArgumentException.class, () -> CpuMatVec.PLAIN
                .matmulGroup(new CpuTensorRef[]{a, c}, x, 0, new float[][]{new float[20], new float[14]}, 5));
    }
}

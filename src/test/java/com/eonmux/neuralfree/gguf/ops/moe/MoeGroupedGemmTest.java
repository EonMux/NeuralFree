package com.eonmux.neuralfree.gguf.ops.moe;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.ops.gemm.GemmKernels;
import org.junit.Test;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;
import java.util.TreeSet;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The grouped MoE GEMM contracts on the host: {@code MoeKernels.groupedTiles} bounds the aligned layout of
 * {@code MoeCombine.alignBlocks} ({@code LLM_MoE_AlignBlocks}), the aligned rows keep
 * {@code MoeCombine.bucketize}'s order per expert bucket with -1 tile padding, the grouped GEMM walk over that
 * layout (the {@code {T}_GroupedGemm} guards and write positions) produces the per-pair outputs of
 * {@code {T}_BatchMatMulId}, the argument lists and launch geometries follow the kernel signatures, the kernel
 * sources instantiate exactly {@link MoeKernels#weightTypes()} with a Vulkan variants row each, and the CUDA-only
 * tensor-core source instantiates exactly the {@link MoeKernels#hasGroupedGemmMma} types. No GPU is used.
 */
public class MoeGroupedGemmTest {

    private static final int ALIGN_TILE = MoeKernels.ALIGN_TILE;

    // ------------------------------------------------------------------ aligned layout

    /** Buckets of two, one and one pairs pad to three tiles, and the bound covers them. */
    @Test
    public void alignBlocksPadsEveryBucketToWholeTiles() {
        int[] ids = {0, 2, 1, 0};
        MoeCombine.Buckets buckets = MoeCombine.bucketize(4, 1, 4, ids);
        MoeCombine.Aligned aligned = MoeCombine.alignBlocks(4, 1, 4, buckets.counts(), buckets.offsets(),
                buckets.rowToken(), buckets.rowSlot());

        assertEquals(3, aligned.tiles());
        assertEquals(3 * ALIGN_TILE, aligned.gemmToken().length);
        int[] token = aligned.gemmToken();
        int[] slot = aligned.gemmSlot();
        assertEquals(0, token[0]);
        assertEquals(0, slot[0]);
        assertEquals(3, token[1]);
        assertEquals(0, slot[1]);
        assertEquals(2, token[ALIGN_TILE]);
        assertEquals(1, slot[ALIGN_TILE]);
        assertEquals(1, token[2 * ALIGN_TILE]);
        assertEquals(2, slot[2 * ALIGN_TILE]);
        assertEquals(-1, token[2]);
        assertEquals(-1, slot[2]);
        assertEquals(-1, token[2 * ALIGN_TILE + 1]);
        assertEquals(-1, slot[2 * ALIGN_TILE + 1]);
        assertTrue(aligned.tiles() <= MoeKernels.groupedTiles(4, 4));
    }

    /** Pairs with ids outside {@code [0, nExpert)} land in the last tiles with slot -1. */
    @Test
    public void alignBlocksKeepsOutOfRangePairsInTheLastTiles() {
        int[] ids = {9, -1};
        MoeCombine.Buckets buckets = MoeCombine.bucketize(2, 1, 2, ids);
        assertEquals(-1, buckets.rowSlot()[0]);
        assertEquals(-1, buckets.rowSlot()[1]);
        MoeCombine.Aligned aligned = MoeCombine.alignBlocks(2, 1, 2, buckets.counts(), buckets.offsets(),
                buckets.rowToken(), buckets.rowSlot());

        assertEquals(1, aligned.tiles());
        assertEquals(0, aligned.gemmToken()[0]);
        assertEquals(-1, aligned.gemmSlot()[0]);
        assertEquals(1, aligned.gemmToken()[1]);
        assertEquals(-1, aligned.gemmSlot()[1]);
        for (int i = 2; i < ALIGN_TILE; i++) {
            assertEquals(-1, aligned.gemmToken()[i]);
            assertEquals(-1, aligned.gemmSlot()[i]);
        }
        assertTrue(aligned.tiles() <= MoeKernels.groupedTiles(2, 2));
    }

    /**
     * A deterministic id mix over 40 tokens: the aligned rows are exactly the bucketize rows in order, every real
     * row appears once, the bound holds, and every tile's first position is a real row.
     */
    @Test
    public void alignBlocksPreservesTheSortedRowsOverAMixedRouting() {
        int nTokens = 40, k = 3, nExpert = 7;
        int nRows = nTokens * k;
        int[] ids = new int[nRows];
        for (int p = 0; p < nRows; p++) {
            ids[p] = (p * 5) % 11 - 2;
        }
        MoeCombine.Buckets buckets = MoeCombine.bucketize(nTokens, k, nExpert, ids);
        MoeCombine.Aligned aligned = MoeCombine.alignBlocks(nTokens, k, nExpert, buckets.counts(), buckets.offsets(),
                buckets.rowToken(), buckets.rowSlot());
        int[] token = aligned.gemmToken();
        int[] slot = aligned.gemmSlot();

        assertTrue(aligned.tiles() <= MoeKernels.groupedTiles(nRows, nExpert));
        Set<Integer> pairs = new HashSet<>();
        for (int q = 0; q < aligned.tiles(); q++) {
            assertTrue("tile " + q + " starts with a filler", token[q * ALIGN_TILE] >= 0);
            for (int i = q * ALIGN_TILE; i < (q + 1) * ALIGN_TILE; i++) {
                int p = token[i];
                if (p < 0) {
                    assertEquals(-1, slot[i]);
                    continue;
                }
                assertEquals("position " + i + " carries the pair's expert",
                        ids[p] >= 0 && ids[p] < nExpert ? ids[p] : -1, slot[i]);
                assertTrue(pairs.add(p));
            }
        }
        assertEquals(nRows, pairs.size());
    }

    // ------------------------------------------------------------------ grouped GEMM walk

    /**
     * The {@code {T}_GroupedGemm} walk of the shards over the aligned layout: for each tile the expert guard of
     * {@code common/matmul_grouped_family.h}, and for each real position the dot products of the tile's expert
     * written at the pair's rows. {@code shardWeights[s]} binds the experts
     * {@code [shardBases[s], shardBases[s] + shardCounts[s])} of the tensor, as a shard dispatch does.
     */
    private static void groupedGemmWalk(float[][] shardWeights, int[] shardBases, int[] shardCounts, int outDim,
                                        int inDim, int xDiv, int inOff, boolean scaled, int[] gemmToken,
                                        int[] gemmSlot, float[] x, float[] scales, float[] out) {
        Arrays.fill(out, Float.NaN);
        for (int s = 0; s < shardBases.length; s++) {
            for (int q = 0; q < gemmToken.length / ALIGN_TILE; q++) {
                int e = gemmSlot[q * ALIGN_TILE];
                if (e < 0) {
                    if (shardBases[s] != 0) {
                        continue;
                    }
                    for (int i = q * ALIGN_TILE; i < (q + 1) * ALIGN_TILE; i++) {
                        int p = gemmToken[i];
                        if (p >= 0) {
                            for (int r = 0; r < outDim; r++) {
                                out[p * outDim + r] = 0.0f;
                            }
                        }
                    }
                    continue;
                }
                if (e < shardBases[s] || e - shardBases[s] >= shardCounts[s]) {
                    continue;
                }
                for (int i = q * ALIGN_TILE; i < (q + 1) * ALIGN_TILE; i++) {
                    int p = gemmToken[i];
                    if (p < 0) {
                        continue;
                    }
                    int row = inOff + (p / xDiv) * inDim;
                    for (int r = 0; r < outDim; r++) {
                        float acc = 0.0f;
                        for (int kk = 0; kk < inDim; kk++) {
                            acc += shardWeights[s][((e - shardBases[s]) * outDim + r) * inDim + kk] * x[row + kk];
                        }
                        out[p * outDim + r] = scaled ? acc * scales[e] : acc;
                    }
                }
            }
        }
    }

    /** The per-pair reference: expert e's rows dot the pair's input row, scaled; 0 for an out-of-range id. */
    private static float[] directReference(float[] weights, int nExpert, int outDim, int inDim, int xDiv,
                                           boolean scaled, int[] ids, float[] x, float[] scales) {
        float[] out = new float[ids.length * outDim];
        for (int p = 0; p < ids.length; p++) {
            int e = ids[p];
            if (e < 0 || e >= nExpert) {
                continue;
            }
            int row = (p / xDiv) * inDim;
            for (int r = 0; r < outDim; r++) {
                float acc = 0.0f;
                for (int kk = 0; kk < inDim; kk++) {
                    acc += weights[(e * outDim + r) * inDim + kk] * x[row + kk];
                }
                out[p * outDim + r] = scaled ? acc * scales[e] : acc;
            }
        }
        return out;
    }

    private static float[] values(int n, long seed) {
        float[] v = new float[n];
        for (int i = 0; i < n; i++) {
            v[i] = (float) (Math.sin(seed * 0.017 + i * 0.29) * 2.0);
        }
        return v;
    }

    private static void assertClose(float[] expected, float[] actual, String what) {
        assertEquals(expected.length, actual.length);
        for (int i = 0; i < expected.length; i++) {
            assertTrue(what + "[" + i + "]: " + actual[i] + " != " + expected[i],
                    Math.abs(expected[i] - actual[i]) <= 1e-4 * Math.max(1.0f, Math.abs(expected[i])));
        }
    }

    /**
     * Token-row reads ({@code xDiv = k}), shared scales and a two-way shard split produce exactly the direct
     * per-pair outputs, and out-of-range pairs are 0.
     */
    @Test
    public void groupedGemmWalkMatchesTheDirectPerPairProducts() {
        int nTokens = 9, k = 3, nExpert = 5, inDim = 16, outDim = 8;
        int[] ids = {0, 3, 1, 2, 2, 4, -1, 0, 3, 1, 1, 1, 4, 4, 4, 2, 0, 0, 3, -1, 5, 2, 1, 0, 3, 3, 3};
        assertEquals(nTokens * k, ids.length);
        float[] weights = values(nExpert * outDim * inDim, 3);
        float[] x = values(nTokens * inDim, 5);
        float[] scales = values(nExpert, 7);
        float[] expected = directReference(weights, nExpert, outDim, inDim, k, true, ids, x, scales);

        MoeCombine.Buckets buckets = MoeCombine.bucketize(nTokens, k, nExpert, ids);
        MoeCombine.Aligned aligned = MoeCombine.alignBlocks(nTokens, k, nExpert, buckets.counts(), buckets.offsets(),
                buckets.rowToken(), buckets.rowSlot());

        int rowsPerExpert = outDim * inDim;
        float[][] shardWeights = {
                Arrays.copyOfRange(weights, 0, 3 * rowsPerExpert),
                Arrays.copyOfRange(weights, 3 * rowsPerExpert, weights.length),};
        float[] walked = new float[ids.length * outDim];
        groupedGemmWalk(shardWeights, new int[]{0, 3}, new int[]{3, 2}, outDim, inDim, k, 0, true,
                aligned.gemmToken(), aligned.gemmSlot(), x, scales, walked);
        assertClose(expected, walked, "sharded grouped gemm");
    }

    /** Pair-row reads ({@code xDiv = 1}) without scales, one shard, and every out-of-range output exactly 0. */
    @Test
    public void groupedGemmWalkWritesZeroForOutOfRangePairsWithoutScales() {
        int nTokens = 4, k = 2, nExpert = 3, inDim = 12, outDim = 6;
        int[] ids = {1, -1, 7, 0, 2, 2, 1, 1};
        float[] weights = values(nExpert * outDim * inDim, 11);
        float[] x = values(nTokens * k * inDim, 13);
        float[] expected = directReference(weights, nExpert, outDim, inDim, 1, false, ids, x, null);

        MoeCombine.Buckets buckets = MoeCombine.bucketize(nTokens, k, nExpert, ids);
        MoeCombine.Aligned aligned = MoeCombine.alignBlocks(nTokens, k, nExpert, buckets.counts(), buckets.offsets(),
                buckets.rowToken(), buckets.rowSlot());

        float[] walked = new float[ids.length * outDim];
        groupedGemmWalk(new float[][]{weights}, new int[]{0}, new int[]{nExpert}, outDim, inDim, 1, 0, false,
                aligned.gemmToken(), aligned.gemmSlot(), x, null, walked);
        for (int p = 0; p < ids.length; p++) {
            if (ids[p] < 0 || ids[p] >= nExpert) {
                for (int r = 0; r < outDim; r++) {
                    assertEquals("pair " + p + " row " + r, 0.0f, walked[p * outDim + r], 0.0f);
                }
            }
        }
        assertClose(expected, walked, "ungated grouped gemm");
    }

    // ------------------------------------------------------------------ host API

    @Test
    public void groupedTileBoundHoldsAndRejectsBadShapes() {
        assertEquals(1, MoeKernels.groupedTiles(1, 1));
        assertEquals(2, MoeKernels.groupedTiles(33, 1));
        assertEquals((40 * 3 + 8 * (ALIGN_TILE - 1)) / ALIGN_TILE, MoeKernels.groupedTiles(120, 7));
        assertEquals(MoeKernels.groupedTiles(120, 7) * (long) ALIGN_TILE, MoeKernels.alignedPositions(120, 7));
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> MoeKernels.groupedTiles(10, MoeKernels.MAX_EXPERTS + 1));
        assertTrue(e.getMessage(), e.getMessage().contains("experts"));
        assertThrows(IllegalArgumentException.class, () -> MoeKernels.groupedTiles(0, 4));
        assertThrows(IllegalArgumentException.class, () -> MoeKernels.groupedTiles(10, 0));
    }

    @Test
    public void groupedGemmNamesAndGeometryFollowTheTiledFamily() {
        assertEquals("F32_GroupedGemm", MoeKernels.groupedGemm(GGMLType.F32));
        assertEquals("Q4_K_GroupedGemm", MoeKernels.groupedGemm(GGMLType.Q4_K));
        for (GGMLType t : MoeKernels.weightTypes()) {
            assertTrue("grouped GEMM needs a tiled kernel for " + t, GemmKernels.hasTiled(t));
        }
        assertArrayEquals(new long[]{256, 4}, MoeKernels.groupedGemmGlobal(64, 4));
        assertArrayEquals(new long[]{2 * 256, 1}, MoeKernels.groupedGemmGlobal(65, 1));
        assertArrayEquals(new long[]{MoeKernels.GROUPED_WORKGROUP, 1}, MoeKernels.groupedGemmLocal());
        assertThrows(UnsupportedOperationException.class, () -> MoeKernels.groupedGemm(null));
    }

    @Test
    public void groupedGemmArgumentsFollowTheDeclaredOrder() {
        com.eonmux.neuralfree.acceleration.backend.GpuBuffer b =
                new com.eonmux.neuralfree.gguf.testutil.FakeGpuBuffer(1, 64, true);
        Object[] args = MoeKernels.groupedGemmArgs(8, 16, 3, 2, 20, 4, 6, true, 1, 2, b, b, b, b, b, b);
        assertEquals(MoeKernels.GG_ARG_COUNT, args.length);
        assertEquals(8, args[0]);
        assertEquals(16, args[1]);
        assertEquals(3, args[MoeKernels.ARG_GG_N_TILES]);
        assertEquals(2, args[MoeKernels.ARG_GG_X_DIV]);
        assertEquals(20, args[MoeKernels.ARG_GG_X_STRIDE]);
        assertEquals(4, args[MoeKernels.ARG_GG_IN_OFF]);
        assertEquals(6, args[MoeKernels.ARG_GG_OUT_OFF]);
        assertEquals(MoeKernels.SCALE_SHARED, args[MoeKernels.ARG_GG_HAS_SCALE]);
        assertEquals(1, args[MoeKernels.ARG_GG_EXPERT_BASE]);
        assertEquals(2, args[MoeKernels.ARG_GG_EXPERT_COUNT]);
        assertSame(b, args[MoeKernels.ARG_GG_WEIGHT]);
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> MoeKernels.groupedGemmArgs(8, 16, 3, 2, 16, -1, 0, false, 0, 2, b, b, b, b, b, b));
        assertTrue(e.getMessage(), e.getMessage().contains("in_off"));
        e = assertThrows(IllegalArgumentException.class,
                () -> MoeKernels.groupedGemmArgs(8, 16, 3, 2, 16, 0, -1, false, 0, 2, b, b, b, b, b, b));
        assertTrue(e.getMessage(), e.getMessage().contains("out_off"));
        e = assertThrows(IllegalArgumentException.class,
                () -> MoeKernels.groupedGemmArgs(8, 16, 3, 2, 15, 0, 0, false, 0, 2, b, b, b, b, b, b));
        assertTrue(e.getMessage(), e.getMessage().contains("x_stride"));
        assertThrows(IllegalArgumentException.class,
                () -> MoeKernels.groupedGemmArgs(8, 16, 3, 2, 16, 0, 0, false, 0, 0, b, b, b, b, b, b));
    }

    @Test
    public void mappedGroupedGemmArgumentsFollowTheDeclaredOrder() {
        com.eonmux.neuralfree.acceleration.backend.GpuBuffer[] b = new com.eonmux.neuralfree.acceleration.backend
                .GpuBuffer[7];
        for (int i = 0; i < b.length; i++) {
            b[i] = new com.eonmux.neuralfree.gguf.testutil.FakeGpuBuffer(i + 1, 64, true);
        }
        Object[] args = MoeKernels.groupedGemmIndArgs(8, 16, 3, 2, 20, 4, 6, true, b[0], b[1], b[2], b[3], b[4],
                b[5], b[6]);
        assertEquals(MoeKernels.GG_IND_ARG_COUNT, args.length);
        assertArrayEquals(new Object[]{8, 16, 3, 2, 20, 4, 6, MoeKernels.SCALE_SHARED}, Arrays.copyOf(args, 8));
        assertEquals(3, args[MoeKernels.ARG_GG_N_TILES]);
        assertEquals(4, args[MoeKernels.ARG_GG_IN_OFF]);
        assertSame(b[0], args[MoeKernels.ARG_GGI_GEMM_TOKEN]);
        assertSame(b[1], args[MoeKernels.ARG_GGI_GEMM_SLOT]);
        assertSame(b[2], args[MoeKernels.ARG_GGI_MAP]);
        assertSame(b[3], args[MoeKernels.ARG_GGI_WEIGHT]);
        assertSame(b[4], args[MoeKernels.ARG_GGI_INPUT]);
        assertSame(b[5], args[MoeKernels.ARG_GGI_SCALES]);
        assertSame(b[6], args[MoeKernels.ARG_GGI_OUTPUT]);
        assertEquals("Q4_K_GroupedGemmInd", MoeKernels.groupedGemmInd(GGMLType.Q4_K));
        assertEquals("Q4_K_GroupedGemmMmaInd", MoeKernels.groupedGemmMmaInd(GGMLType.Q4_K));
        assertThrows(UnsupportedOperationException.class, () -> MoeKernels.groupedGemmMmaInd(GGMLType.F32));
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> MoeKernels.groupedGemmIndArgs(8, 16, 3, 2, 15, 0, 0, false, b[0], b[1], b[2], b[3], b[4],
                        b[5], b[6]));
        assertTrue(e.getMessage(), e.getMessage().contains("x_stride"));
        e = assertThrows(IllegalArgumentException.class,
                () -> MoeKernels.groupedGemmIndArgs(8, 16, 3, 2, 16, -1, 0, false, b[0], b[1], b[2], b[3], b[4],
                        b[5], b[6]));
        assertTrue(e.getMessage(), e.getMessage().contains("in_off"));
        NullPointerException npe = assertThrows(NullPointerException.class,
                () -> MoeKernels.groupedGemmIndArgs(8, 16, 3, 2, 16, 0, 0, false, b[0], b[1], null, b[3], b[4],
                        b[5], b[6]));
        assertTrue(npe.getMessage(), npe.getMessage().contains("map"));
    }

    @Test
    public void tensorCoreGroupedGemmCoversEveryGroupedTypeButF32() {
        assertEquals("Q4_K_GroupedGemmMma", MoeKernels.groupedGemmMma(GGMLType.Q4_K));
        assertEquals("BF16_GroupedGemmMma", MoeKernels.groupedGemmMma(GGMLType.BF16));
        assertTrue(!MoeKernels.hasGroupedGemmMma(GGMLType.F32));
        assertThrows(UnsupportedOperationException.class, () -> MoeKernels.groupedGemmMma(GGMLType.F32));
        assertThrows(UnsupportedOperationException.class, () -> MoeKernels.groupedGemmMma(null));
        for (GGMLType t : MoeKernels.weightTypes()) {
            assertEquals(t.name(), t != GGMLType.F32, MoeKernels.hasGroupedGemmMma(t));
        }
    }

    @Test
    public void multiGroupSortGeometryAndArgumentsFollowTheKernels() {
        assertEquals(1, MoeKernels.bucketGroups(1, MoeKernels.BUCKET_CHUNK));
        assertEquals(1, MoeKernels.bucketGroups(MoeKernels.BUCKET_CHUNK, MoeKernels.BUCKET_CHUNK));
        assertEquals(2, MoeKernels.bucketGroups(MoeKernels.BUCKET_CHUNK + 1, MoeKernels.BUCKET_CHUNK));
        assertEquals(3L * 9, MoeKernels.bucketHistInts(130, 8, 64));
        assertThrows(IllegalArgumentException.class, () -> MoeKernels.bucketGroups(0, 64));
        assertThrows(IllegalArgumentException.class, () -> MoeKernels.bucketGroups(10, 0));
        assertArrayEquals(new long[]{64}, MoeKernels.alignGlobal(2, 64));
        assertArrayEquals(new long[]{2 * 64}, MoeKernels.alignGlobal(3, 64));
        assertArrayEquals(new long[]{(long) MoeKernels.MAX_ALIGN_GROUPS * 32}, MoeKernels.alignGlobal(100_000, 32));

        com.eonmux.neuralfree.acceleration.backend.GpuBuffer b =
                new com.eonmux.neuralfree.gguf.testutil.FakeGpuBuffer(1, 64, true);
        Object[] count = MoeKernels.bucketCountArgs(5, 2, 8, 64, b, b);
        assertEquals(MoeKernels.BUCKET_COUNT_ARG_COUNT, count.length);
        assertArrayEquals(new Object[]{5, 2, 8, 64}, Arrays.copyOf(count, 4));
        assertEquals(MoeKernels.BUCKET_SCAN_ARG_COUNT, MoeKernels.bucketScanArgs(5, 2, 8, 64, b, b, b).length);
        assertEquals(MoeKernels.BUCKET_SCATTER_ARG_COUNT,
                MoeKernels.bucketScatterArgs(5, 2, 8, 64, b, b, b, b).length);
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> MoeKernels.bucketScanArgs(5, 2, MoeKernels.MAX_EXPERTS + 1, 64, b, b, b));
        assertTrue(e.getMessage(), e.getMessage().contains("experts"));
        e = assertThrows(IllegalArgumentException.class, () -> MoeKernels.bucketCountArgs(5, 2, 8, 0, b, b));
        assertTrue(e.getMessage(), e.getMessage().contains("chunk"));
    }

    @Test
    public void alignArgumentsFollowTheDeclaredOrder() {
        com.eonmux.neuralfree.acceleration.backend.GpuBuffer b =
                new com.eonmux.neuralfree.gguf.testutil.FakeGpuBuffer(1, 64, true);
        Object[] args = MoeKernels.alignBlocksArgs(5, 2, 8, 9, b, b, b, b, b, b);
        assertEquals(MoeKernels.ALIGN_ARG_COUNT, args.length);
        assertEquals(5, args[MoeKernels.ARG_N_TOKENS]);
        assertEquals(2, args[1]);
        assertEquals(8, args[2]);
        assertEquals(9, args[3]);
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> MoeKernels.alignBlocksArgs(5, 2, MoeKernels.MAX_EXPERTS + 1, 9, b, b, b, b, b, b));
        assertTrue(e.getMessage(), e.getMessage().contains("experts"));
    }

    // ------------------------------------------------------------------ kernel sources

    @Test
    public void openClSourceInstantiatesExactlyTheExpertTypes() throws IOException {
        String source = resource("kernels/llm/moe_gemm.cl");
        Matcher m = Pattern.compile("(?m)^NF_GEMM_GROUPED\\(\\s*(\\w+)\\s*,").matcher(source);
        Set<String> instantiated = new TreeSet<>();
        while (m.find()) {
            assertTrue("duplicate instantiation " + m.group(1), instantiated.add(m.group(1)));
        }
        Set<String> expected = new TreeSet<>();
        for (GGMLType t : MoeKernels.weightTypes()) {
            expected.add(t.name());
        }
        assertEquals(expected, instantiated);
    }

    @Test
    public void cudaTensorCoreSourceInstantiatesExactlyTheMmaTypes() throws IOException {
        String source = resource("kernels/llm/moe_gemm_mma.cl");
        Matcher m = Pattern.compile("(?m)^NF_GEMM_GROUPED_MMA\\(\\s*(\\w+)\\s*,.*,\\s*(F16|BF16)\\)\\s*$")
                .matcher(source);
        Set<String> instantiated = new TreeSet<>();
        while (m.find()) {
            assertTrue("duplicate instantiation " + m.group(1), instantiated.add(m.group(1)));
            assertEquals("operand format of " + m.group(1), m.group(1).equals("BF16") ? "BF16" : "F16", m.group(2));
        }
        Set<String> expected = new TreeSet<>();
        for (GGMLType t : MoeKernels.weightTypes()) {
            if (MoeKernels.hasGroupedGemmMma(t)) {
                expected.add(t.name());
            }
        }
        assertEquals(expected, instantiated);
        assertTrue("the tensor-core source is CUDA-only",
                resource("kernels/llm/manifest.txt").contains("\n[cuda] moe_gemm_mma.cl"));
    }

    /** The variant tables live only in the source tree ({@code processResources} excludes the shader sources). */
    private static String sourceTree(String relative) throws IOException {
        java.nio.file.Path path = java.nio.file.Path.of("src", "main", "resources")
                .resolve(relative);
        assertTrue("missing source file " + path.toAbsolutePath() + " (run from the project root)",
                java.nio.file.Files.exists(path));
        return new String(java.nio.file.Files.readAllBytes(path), StandardCharsets.UTF_8);
    }

    @Test
    public void everyGroupedKernelHasAVulkanVariantRow() throws IOException {
        Set<String> expected = new TreeSet<>();
        for (GGMLType t : MoeKernels.weightTypes()) {
            expected.add(MoeKernels.groupedGemm(t) + "\tQTYPE=QT_" + t.name() + "\t");
            expected.add(MoeKernels.groupedGemmInd(t) + "\tQTYPE=QT_" + t.name() + " IND=1\t");
        }
        assertEquals(expected, variantRows("moe_gemm.tsv", "moe_gemm.comp"));
    }

    /**
     * The cooperative-matrix shader serves every tensor-core grouped type but BF16, in both forms, from rows
     * flagged {@code sg} (full subgroups).
     */
    @Test
    public void everyTensorCoreGroupedKernelButBf16HasACooperativeMatrixRow() throws IOException {
        Set<String> expected = new TreeSet<>();
        for (GGMLType t : MoeKernels.weightTypes()) {
            if (MoeKernels.hasGroupedGemmMma(t) && t != GGMLType.BF16) {
                expected.add(MoeKernels.groupedGemmMma(t) + "\tQTYPE=QT_" + t.name() + "\tsg");
                expected.add(MoeKernels.groupedGemmMmaInd(t) + "\tQTYPE=QT_" + t.name() + " IND=1\tsg");
            }
        }
        assertEquals(expected, variantRows("moe_gemm_mma.tsv", "moe_gemm_mma.comp"));
    }

    /** {@code name<TAB>defines<TAB>flags} of every row of a variant table, each checked to build {@code source}. */
    private static Set<String> variantRows(String table, String source) throws IOException {
        Set<String> rows = new TreeSet<>();
        for (String line : sourceTree("shaders/llm/variants/" + table).split("\n")) {
            if (line.isEmpty() || line.startsWith("#")) {
                continue;
            }
            String[] parts = line.split("\t", -1);
            assertEquals("row source of " + parts[0], source, parts[1]);
            assertTrue("duplicate row " + parts[0], rows.add(parts[0] + "\t" + parts[2] + "\t"
                    + (parts.length > 3 ? parts[3] : "")));
        }
        return rows;
    }

    @Test
    public void theAlignKernelHasAVulkanVariantRowAndTheManifestListsTheSources() throws IOException {
        String[][] rows = {{MoeKernels.ALIGN_BLOCKS, "OP_MOE_ALIGN_BLOCKS"},
                {MoeKernels.BUCKET_COUNT, "OP_MOE_BUCKET_COUNT"}, {MoeKernels.BUCKET_SCAN, "OP_MOE_BUCKET_SCAN"},
                {MoeKernels.BUCKET_SCATTER, "OP_MOE_BUCKET_SCATTER"}};
        String table = sourceTree("shaders/llm/variants/moe.tsv");
        for (String[] row : rows) {
            assertTrue("no moe.tsv row for " + row[0],
                    table.contains("\n" + row[0] + "\tmoe_misc.comp\tOP=" + row[1] + "\n"));
        }
        String manifest = resource("kernels/llm/manifest.txt");
        assertTrue(manifest.contains("common/matmul_grouped_family.h"));
        assertTrue(manifest.contains("\nmoe_gemm.cl"));
        assertTrue("the grouped family header needs the tiled family's constants first",
                manifest.indexOf("common/matmul_tiled_family.h") < manifest.indexOf("common/matmul_grouped_family.h"));
    }

    private static String resource(String path) throws IOException {
        try (InputStream in = MoeGroupedGemmTest.class.getClassLoader().getResourceAsStream(path)) {
            assertNotNull("missing resource " + path, in);
            ByteArrayOutputStream out = new ByteArrayOutputStream();
            byte[] buf = new byte[8192];
            int r;
            while ((r = in.read(buf)) > 0) {
                out.write(buf, 0, r);
            }
            return new String(out.toByteArray(), StandardCharsets.UTF_8);
        }
    }
}

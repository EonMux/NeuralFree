package com.eonmux.neuralfree.gguf.ops.gemm;

import com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.testutil.FakeGpuBuffer;
import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * Pure-logic checks of the split-K plan: the llama.cpp heuristic, slice arithmetic, plane coverage,
 * argument order of the partial and reduce launches, and the device limits. No GPU is used.
 */
public class SplitKGemmPlanTest {

    @Test
    public void theHeuristicFollowsLlamaCppSplitK() {
        assertEquals("small K never splits", 1, SplitKGemmPlan.splitsFor(64, 1, 1, 2047));
        assertEquals("half-filled grid splits by the fill ratio", 4,
                SplitKGemmPlan.splitsFor(8, 2, 1, 4096));
        assertEquals("a one-tile grid over 8 units saturates the cap", SplitKGemmPlan.MAX_SPLITS,
                SplitKGemmPlan.splitsFor(8, 1, 1, 4096));
        assertEquals("8 is the cap", SplitKGemmPlan.MAX_SPLITS, SplitKGemmPlan.splitsFor(128, 1, 1, 4096));
        assertEquals("two-thirds-filled grid splits by 3", 3, SplitKGemmPlan.splitsFor(6, 4, 1, 4096));
        assertEquals("a filled grid does not split", 1, SplitKGemmPlan.splitsFor(4, 4, 1, 4096));
        assertEquals(1, SplitKGemmPlan.splitsFor(4, 3, 1, 4096));
    }

    /** The CM2 launches split only grids that keep their slices in one wave: the 16-tile grid of a 1024-row
     * projection over 512 tokens on 60 units splits by 3, the 40-tile grid of a 2560-row one stays whole. */
    @Test
    public void theCm2HeuristicKeepsOnlyTheSingleWaveBand() {
        assertEquals(3, SplitKGemmPlan.cm2SplitsFor(60, 8, 2, 3072));
        assertEquals(1, SplitKGemmPlan.cm2SplitsFor(60, 20, 2, 9728));
        assertEquals("the general heuristic splits that grid by 3", 3, SplitKGemmPlan.splitsFor(60, 20, 2, 9728));
        assertEquals("small K never splits", 1, SplitKGemmPlan.cm2SplitsFor(60, 1, 1, 2047));
        assertEquals(SplitKGemmPlan.MAX_SPLITS, SplitKGemmPlan.cm2SplitsFor(60, 1, 1, 4096));
    }

    /** The 16-token band of the small-token-tile heuristic splits a staged grid up to about 1.3
     * waves to the capped slice count and leaves bigger grids alone (the measured RTX 5080 Laptop
     * twins: the 64-tile 4096x2560 Q4_K twin runs 86-93us whole and 59-60us as 8 slices over 60
     * units, the 152-tile 9728x2560 twin 101-110us whole and 119-125us sliced). */
    @Test
    public void theSmallTileHeuristicFillsTheWaveWithTheCappedSliceCount() {
        int units = 48;
        assertEquals("a 40-tile grid over 48 units fills the device with 8 slices",
                SplitKGemmPlan.MAX_SPLITS, SplitKGemmPlan.smallTileSplitsFor(units, 40, 1, 9728,
                        MmqConfig.TOKEN_LANES));
        assertEquals("a 16-tile grid over 48 units splits by the cap too",
                SplitKGemmPlan.MAX_SPLITS, SplitKGemmPlan.smallTileSplitsFor(units, 16, 1, 2560,
                        MmqConfig.TOKEN_LANES));
        assertEquals("a full wave splits to lose the tail wave", SplitKGemmPlan.MAX_SPLITS,
                SplitKGemmPlan.smallTileSplitsFor(units, 48, 1, 9728, MmqConfig.TOKEN_LANES));
        assertEquals("a grid past 1.3 waves keeps one slice", 1,
                SplitKGemmPlan.smallTileSplitsFor(units, 64, 1, 9728, MmqConfig.TOKEN_LANES));
        assertEquals("small K keeps one slice", 1, SplitKGemmPlan.smallTileSplitsFor(units, 40, 1, 2047,
                MmqConfig.TOKEN_LANES));
        int unknown = DeviceCapabilities.UNKNOWN_COMPUTE_UNITS;
        assertEquals("an unreported core count keeps one slice", 1,
                SplitKGemmPlan.smallTileSplitsFor(unknown, 40, 1, 9728, MmqConfig.TOKEN_LANES));
        expectInvalid("non-negative", () -> SplitKGemmPlan.smallTileSplitsFor(-1, 40, 1, 9728,
                MmqConfig.TOKEN_LANES));
    }

    /** The 17..64-token band of the heuristic splits every grid of up to six waves at the cap: the
     * measured RTX 5080 Laptop 64-token twins keep gaining far past one wave (Q4_K n=24 over 60
     * units: 4096x2560 65.8us as 8 slices vs 91.6 whole, 2560x9728 146.1 vs 287.9; the 152-tile
     * 9728x2560 runs 133.2us at 8 slices vs 148.4 at 4), while the 2374-tile head past the wave
     * bound stays whole. Small K keeps one slice (launches below {@link SplitKGemmPlan#MIN_SPLIT_K}
     * run the 16-token twin by {@link MmqConfig#twinTokens(int, int)}, so the 64-token band only
     * ever sees K past the threshold). */
    @Test
    public void theTwin64HeuristicSplitsEveryGridOfSixWaves() {
        int units = 48;
        assertEquals("a 40-tile grid splits at the cap",
                SplitKGemmPlan.MAX_SPLITS, SplitKGemmPlan.smallTileSplitsFor(units, 40, 1, 9728,
                        MmqConfig.TWIN_MAX_TOKENS));
        assertEquals("a 152-tile grid (2.9 waves over 48 units) splits at the cap too",
                SplitKGemmPlan.MAX_SPLITS, SplitKGemmPlan.smallTileSplitsFor(units, 152, 1, 9728,
                        MmqConfig.TWIN_MAX_TOKENS));
        assertEquals("a grid past six waves keeps one slice", 1,
                SplitKGemmPlan.smallTileSplitsFor(units, 289, 1, 9728, MmqConfig.TWIN_MAX_TOKENS));
        assertEquals("small K keeps one slice", 1, SplitKGemmPlan.smallTileSplitsFor(units, 40, 1, 2047,
                MmqConfig.TWIN_MAX_TOKENS));
        assertEquals("an unreported core count keeps one slice", 1,
                SplitKGemmPlan.smallTileSplitsFor(DeviceCapabilities.UNKNOWN_COMPUTE_UNITS, 40, 1, 9728,
                        MmqConfig.TWIN_MAX_TOKENS));
    }

    /** The small-token-tile plan runs the twin tile's _SKSplit kernel over the twin's grid and slices
     *  K across the cap; the 17..64-token launches run the 64-token twin's _SKSplit64. */
    @Test
    public void theSmallTilePlanSlicesKAcrossTheDeviceTileKernel() {
        MmqGemmPlan base = MmqGemmPlan.of(MmqConfig.INT8_TENSOR, GGMLType.Q8_0, 2560, 9728, 9,
                MmqConfig.INT8_TENSOR.rowRecords(9728), 0, 2560, 0);
        SplitKGemmPlan p = SplitKGemmPlan.ofSmallTiles(base, 48, 256);
        MmqConfig twin = MmqConfig.smallTile(MmqConfig.INT8_TENSOR);
        assertNotNull("a 40-tile twin grid over 48 units splits", p);
        assertEquals("Q8_0_MatMul_MMQ_SKSplit", p.kernelName());
        assertEquals(twin, p.stagedConfig());
        assertEquals(twin.tileTokens(), p.stagedConfig().tileTokens());
        assertEquals("9 tokens need 1 twin tile", 1, p.tokenTiles());
        assertEquals(SplitKGemmPlan.MAX_SPLITS, p.splits());
        assertEquals("slices are whole stage depths", 0, p.kSlice() % MmqConfig.INT8_TENSOR.stageDepth());
        assertTrue("the last slice is not empty", (long) p.kSlice() * (p.splits() - 1) < p.k());
        assertArrayEquals(new long[]{40L * twin.workGroupSize() * p.splits(), 1},
                p.partialGlobalWorkSize());
        assertNotNull("a full-wave grid splits too (the tail wave quantization)",
                SplitKGemmPlan.ofSmallTiles(MmqGemmPlan.of(MmqConfig.INT8_TENSOR, GGMLType.Q8_0, 3072, 9728, 9,
                        MmqConfig.INT8_TENSOR.rowRecords(9728), 0, 3072, 0), 48, 256));
        assertNull("a grid past 1.3 waves keeps the plain kernel",
                SplitKGemmPlan.ofSmallTiles(MmqGemmPlan.of(MmqConfig.INT8_TENSOR, GGMLType.Q8_0, 8192, 9728, 9,
                        MmqConfig.INT8_TENSOR.rowRecords(9728), 0, 8192, 0), 48, 256));
        SplitKGemmPlan w4a8 = SplitKGemmPlan.ofW4a8SmallTiles(GGMLType.MXFP4, 2560, 4096, 9, 128, 0,
                MmqConfig.INT8_TENSOR, 48, 256);
        assertNotNull(w4a8);
        assertEquals("MXFP4_MatMul_W4A8_SKSplit", w4a8.kernelName());
        assertEquals(SplitKGemmPlan.MAX_SPLITS, w4a8.splits());
    }

    /** A launch of 17..64 tokens splits the 64-token twin's _SKSplit64 at the cap, and a 64-token
     *  device tile (whose plain kernel is that tile) keeps the plain launch. */
    @Test
    public void theTwin64PlanSlicesThe64TokenTwinKernel() {
        MmqGemmPlan base = MmqGemmPlan.of(MmqConfig.INT8_TENSOR, GGMLType.Q4_K, 2560, 9728, 24,
                MmqConfig.INT8_TENSOR.rowRecords(9728), 0, 2560, 0);
        SplitKGemmPlan p = SplitKGemmPlan.ofSmallTiles(base, 48, 256);
        MmqConfig twin64 = MmqConfig.smallTile(MmqConfig.INT8_TENSOR, MmqConfig.TWIN_MAX_TOKENS);
        assertNotNull("the 40-tile S64 grid over 48 units splits", p);
        assertEquals("Q4_K_MatMul_MMQ_SKSplit64", p.kernelName());
        assertEquals(twin64, p.stagedConfig());
        assertEquals("24 tokens need 1 S64 tile", 1, p.tokenTiles());
        assertEquals(SplitKGemmPlan.MAX_SPLITS, p.splits());
        assertArrayEquals(new long[]{40L * twin64.workGroupSize() * p.splits(), 1},
                p.partialGlobalWorkSize());

        SplitKGemmPlan w4a8 = SplitKGemmPlan.ofW4a8SmallTiles(GGMLType.MXFP4, 2560, 4096, 48, 128, 0,
                MmqConfig.INT8_TENSOR, 48, 256);
        assertNotNull(w4a8);
        assertEquals("MXFP4_MatMul_W4A8_SKSplit64", w4a8.kernelName());
        assertEquals(SplitKGemmPlan.MAX_SPLITS, w4a8.splits());

        /* The 64-token CDNA tile is the widest twin row: no twin serves it, the plain kernel runs. */
        MmqGemmPlan cdna = MmqGemmPlan.of(MmqConfig.CDNA, GGMLType.Q4_K, 2560, 9728, 24,
                MmqConfig.CDNA.rowRecords(9728), 0, 2560, 0);
        assertNull(SplitKGemmPlan.ofSmallTiles(cdna, 48, 256));
        assertNull(SplitKGemmPlan.ofW4a8SmallTiles(GGMLType.MXFP4, 2560, 4096, 24, 128, 0,
                MmqConfig.CDNA, 48, 256));
        assertNotNull("a 9-token launch keeps the 16-token twin on the same tile",
                SplitKGemmPlan.ofSmallTiles(MmqGemmPlan.of(MmqConfig.CDNA, GGMLType.Q4_K, 2560, 9728, 9,
                        MmqConfig.CDNA.rowRecords(9728), 0, 2560, 0), 48, 256));
    }

    @Test
    public void anUnknownComputeUnitCountNeverSplits() {
        int unknown = DeviceCapabilities.UNKNOWN_COMPUTE_UNITS;
        assertEquals("an unreported core count keeps one slice even over a one-tile grid", 1,
                SplitKGemmPlan.splitsFor(unknown, 1, 1, 4096));

        MmaGemmPlan mma = MmaGemmPlan.contiguous(GGMLType.F16, 64, 4096, 32);
        assertNull("MMA", SplitKGemmPlan.of(mma, unknown, 256));
        MmqGemmPlan mmq = MmqGemmPlan.of(MmqConfig.PACKED_DOT, GGMLType.Q4_K, 64, 4096, 32,
                MmqConfig.PACKED_DOT.rowRecords(4096), 0, 64, 0);
        assertNull("MMQ", SplitKGemmPlan.of(mmq, unknown, 256));
        CmGemmPlan cm = CmGemmPlan.of(GGMLType.Q4_K, 64, 4096, 32, 4096, 0, 64, 0);
        assertNull("CM", SplitKGemmPlan.of(cm, unknown, 256));
        assertNull("W4A8", SplitKGemmPlan.ofW4a8(GGMLType.MXFP4, 128, 4096, 32, 4096, 0, 64, 32, unknown,
                256));

        expectInvalid("non-negative", () -> SplitKGemmPlan.splitsFor(-1, 1, 1, 4096));
    }

    @Test
    public void sliceLengthRoundsToWholeStages() {
        assertEquals(512, SplitKGemmPlan.sliceLength(4096, 8));
        assertEquals(256, SplitKGemmPlan.sliceLength(2048, 8));
        assertEquals(304, SplitKGemmPlan.sliceLength(2425, 8));
        try {
            SplitKGemmPlan.sliceLength(8, 2);
            fail("expected the stage-depth rejection");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("K >= 16"));
        }
    }

    @Test
    public void planCoversEveryKExactlyOnce() {
        MmaGemmPlan base = MmaGemmPlan.contiguous(GGMLType.F16, 70, 2425, 40);
        SplitKGemmPlan p = SplitKGemmPlan.of(base, 8, 256);
        assertNotNull(p);
        assertEquals("F16_MatMul_MMA_KSplit", p.kernelName());
        assertEquals(GGMLType.F16, p.type());
        assertEquals(2425, p.k());
        assertEquals("a 2x2 tile grid over 8 compute units splits by the fill ratio", 2, p.splits());
        assertEquals("ceil(2425/2) rounds to whole stages", 1216, p.kSlice());
        assertTrue("the last slice is not empty", (long) p.kSlice() * (p.splits() - 1) < p.k());
        assertEquals("whole slices plus the kernel's K-clamped tail cover K", 2425,
                (int) Math.min((long) p.kSlice() * p.splits(), 2425));
        assertEquals(2, p.rowTiles());
        assertEquals(2, p.tokenTiles());
        assertArrayEquals(new long[]{2L * 256 * p.splits(), 2}, p.partialGlobalWorkSize());
        assertArrayEquals(new long[]{256, 1}, p.partialLocalWorkSize());
        assertEquals(70L * 40, p.planeElements());
        assertEquals(p.splits() * 70L * 40, p.partialElements());
    }

    @Test
    public void stagedW4a8PlansCountTheRecordUnitsOfTheStageDepth() {
        /* MXFP4 with the 32-deep tensor stages: 4 + 2 + 2 units per chunk record. */
        SplitKGemmPlan p = SplitKGemmPlan.ofW4a8(GGMLType.MXFP4, 128, 4096, 32, 4096, 0, 64, 32, 8, 256);
        assertNotNull("a two-tile grid over 8 compute units splits", p);
        assertEquals("MXFP4_MatMul_W4A8_KSplit", p.kernelName());
        assertEquals(GGMLType.MXFP4, p.type());
        long units = 32 / 4 + 2 + 32 / 16;
        assertEquals("the staged rows are records of " + units + " units",
                (long) (p.n() - 1) * 4096 * units + (long) (4096 / 32) * units, p.xElementsRequired());

        SplitKGemmPlan heuristic = SplitKGemmPlan.ofW4a8(GGMLType.MXFP4, 128, 4096, 32, 4096, 0, 64, 32, 8, 256);
        assertNotNull("a two-tile grid over 8 compute units splits", heuristic);
        assertEquals(4, heuristic.splits());
        assertEquals(1024, heuristic.kSlice());
        try {
            SplitKGemmPlan.ofW4a8(GGMLType.F16, 128, 4096, 32, 4096, 0, 64, 32, 8, 256);
            fail("expected the W4A8 kernel rejection");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("no split-K W4A8 GEMM kernel"));
        }
    }

    @Test
    public void theHeuristicThatDoesNotSplitReturnsNull() {
        MmaGemmPlan filled = MmaGemmPlan.contiguous(GGMLType.F16, 64 * 64, 4096, 32);
        assertNull("a full grid stays on the plain kernel", SplitKGemmPlan.of(filled, 4, 256));
        MmaGemmPlan smallK = MmaGemmPlan.contiguous(GGMLType.F16, 64, 2047, 32);
        assertNull("K below the threshold stays on the plain kernel", SplitKGemmPlan.of(smallK, 64, 256));
    }

    @Test
    public void partialArgumentsFollowTheDeclaredOrder() {
        MmaGemmPlan base = MmaGemmPlan.contiguous(GGMLType.BF16, 128, 4096, 32);
        SplitKGemmPlan p = SplitKGemmPlan.of(base, 8, 64);
        assertNotNull(p);
        FakeGpuBuffer w = new FakeGpuBuffer(1, p.weightBytes(), true);
        FakeGpuBuffer x = new FakeGpuBuffer(2, p.xElementsRequired() * Float.BYTES, true);
        FakeGpuBuffer partial = new FakeGpuBuffer(3, p.partialElements() * Float.BYTES, false);
        Object[] args = p.partialArgs(w, x, partial);
        assertEquals(GemmKernels.KSPLIT_ARG_COUNT, args.length);
        assertEquals(128, args[GemmKernels.KSPLIT_ARG_M]);
        assertEquals(4096, args[GemmKernels.KSPLIT_ARG_K]);
        assertEquals(32, args[GemmKernels.KSPLIT_ARG_N]);
        assertEquals(4096, args[GemmKernels.KSPLIT_ARG_X_STRIDE]);
        assertEquals(0, args[GemmKernels.KSPLIT_ARG_X_OFF]);
        assertEquals(p.kSlice(), args[GemmKernels.KSPLIT_ARG_K_SLICE]);
        assertTrue(w == args[GemmKernels.KSPLIT_ARG_WEIGHT]);
        assertTrue(x == args[GemmKernels.KSPLIT_ARG_X]);
        assertTrue(partial == args[GemmKernels.KSPLIT_ARG_PARTIAL]);
        assertEquals("BF16_MatMul_MMA_KSplit", p.kernelName());
    }

    @Test
    public void reduceArgumentsScatterToTheStridedOutput() {
        MmaGemmPlan base = MmaGemmPlan.contiguous(GGMLType.F16, 64, 4096, 32);
        SplitKGemmPlan p = SplitKGemmPlan.of(base, 8, 256);
        assertNotNull(p);
        FakeGpuBuffer partial = new FakeGpuBuffer(1, p.partialElements() * Float.BYTES, true);
        FakeGpuBuffer y = new FakeGpuBuffer(2, p.yElementsRequired(128, 5) * Float.BYTES, false);
        Object[] args = p.reduceArgs(partial, y, 128, 5);
        assertEquals(GemmKernels.SPLIT_K_REDUCE_ARG_COUNT, args.length);
        assertEquals((int) p.planeElements(), args[GemmKernels.REDUCE_ARG_NE]);
        assertEquals(p.splits(), args[GemmKernels.REDUCE_ARG_SPLITS]);
        assertEquals(64, args[GemmKernels.REDUCE_ARG_M]);
        assertEquals(128, args[GemmKernels.REDUCE_ARG_Y_STRIDE]);
        assertEquals(5, args[GemmKernels.REDUCE_ARG_Y_OFF]);
        assertTrue(partial == args[GemmKernels.REDUCE_ARG_PARTIAL]);
        assertTrue(y == args[GemmKernels.REDUCE_ARG_Y]);
        long items = (p.planeElements() + SplitKGemmPlan.REDUCE_ELEMENTS_PER_ITEM - 1)
                / SplitKGemmPlan.REDUCE_ELEMENTS_PER_ITEM;
        assertArrayEquals(new long[]{(items + 255) / 256 * 256}, p.reduceGlobalWorkSize());
        assertArrayEquals(new long[]{256}, p.reduceLocalWorkSize());
    }

    @Test
    public void undersizedBuffersAndBadShapesAreRejected() {
        MmaGemmPlan base = MmaGemmPlan.contiguous(GGMLType.F16, 64, 4096, 32);
        SplitKGemmPlan p = SplitKGemmPlan.of(base, 8, 256);
        assertNotNull(p);
        FakeGpuBuffer w = new FakeGpuBuffer(1, p.weightBytes(), true);
        FakeGpuBuffer x = new FakeGpuBuffer(2, p.xElementsRequired() * Float.BYTES, true);
        FakeGpuBuffer small = new FakeGpuBuffer(3, p.partialElements() * Float.BYTES - 4, false);
        try {
            p.partialArgs(w, x, small);
            fail("expected the partial-size rejection");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("partial"));
        }
        expectInvalid("splits <=", () -> SplitKGemmPlan.of(GGMLType.F16, 64, 4096, 32, 4096, 0,
                SplitKGemmPlan.MAX_SPLITS + 1, 512, 256));
        expectInvalid("multiple of", () -> SplitKGemmPlan.of(GGMLType.F16, 64, 4096, 32, 4096, 0, 2, 100, 256));
        expectInvalid("empty", () -> SplitKGemmPlan.of(GGMLType.F16, 64, 4096, 32, 4096, 0, 8, 592, 256));
        expectInvalid("no split-K MMA GEMM kernel", () -> SplitKGemmPlan.of(GGMLType.Q8_0, 64, 4096, 32,
                4096, 0, 2, 2048, 256));
    }

    @Test
    public void deviceLimitsCarryTheSplitInDimensionZero() {
        DeviceCapabilities big = new DeviceCapabilities.Builder()
                .deviceName("test-device")
                .maxWorkGroupSize(1024)
                .maxSharedMemoryBytes(MmaGemmPlan.TILE_SHARED_BYTES)
                .maxWorkGroupCount(new long[]{65535, 65535, 65535})
                .build();
        MmaGemmPlan base = MmaGemmPlan.contiguous(GGMLType.F16, 64, 4096, 32);
        SplitKGemmPlan p = SplitKGemmPlan.of(base, 8, 256);
        assertNotNull(p);
        assertNull(p.checkDevice(big));

        DeviceCapabilities fewGroups = new DeviceCapabilities.Builder()
                .deviceName("narrow-device")
                .maxWorkGroupSize(1024)
                .maxSharedMemoryBytes(MmaGemmPlan.TILE_SHARED_BYTES)
                .maxWorkGroupCount(new long[]{p.rowTiles() * p.splits() - 1, 65535, 65535})
                .build();
        String reason = p.checkDevice(fewGroups);
        assertNotNull(reason);
        assertTrue(reason, reason.contains("dimension 0"));
        assertTrue(reason, reason.contains("splits"));
    }

    /**
     * The subgroup-row MMQ split: only a grid of at most one 128 x 32 tile per eight compute units splits, into eight
     * slices of whole 64-deep stages, the subgroup twin's name,
     * its tile and work-group in the partial launch, and the staged records' stride and offset of the base.
     */
    @Test
    public void theSubgroupMmqPlanSplitsOnlyCoarseGridsOverItsOwnTile() {
        MmqGemmPlan base = MmqGemmPlan.of(MmqConfig.PACKED_DOT, GGMLType.Q4_K, 1024, 3072, 16,
                MmqConfig.PACKED_DOT.rowRecords(3072), 7, 1024, 0);
        SplitKGemmPlan p = SplitKGemmPlan.ofSubgroupMmq(base, 128, 32, 128, 64, 256);
        assertNotNull("an eight-tile grid over 64 units splits", p);
        assertEquals(SplitKGemmPlan.Family.MMQ_SG, p.family());
        assertEquals("Q4_K_MatMulSg_MMQ_KSplit", p.kernelName());
        assertEquals(SplitKGemmPlan.MAX_SPLITS, p.splits());
        assertEquals(384, p.kSlice());
        assertEquals(0, p.kSlice() % SplitKGemmPlan.SUBGROUP_MMQ_SLICE_DEPTH);
        assertEquals(8, p.rowTiles());
        assertEquals(1, p.tokenTiles());
        assertEquals(128, p.workGroupSize());
        assertArrayEquals(new long[]{8L * 128 * SplitKGemmPlan.MAX_SPLITS, 1}, p.partialGlobalWorkSize());
        assertEquals(MmqConfig.PACKED_DOT.rowRecords(3072), p.xStride());
        assertEquals(7, p.xOff());

        assertNull("16 tiles over 64 units stay whole", SplitKGemmPlan.ofSubgroupMmq(
                MmqGemmPlan.of(MmqConfig.PACKED_DOT, GGMLType.Q4_K, 2048, 3072, 16,
                        MmqConfig.PACKED_DOT.rowRecords(3072), 0, 2048, 0), 128, 32, 128, 64, 256));
        assertNull("a K below 2048 stays whole", SplitKGemmPlan.ofSubgroupMmq(MmqGemmPlan.of(MmqConfig.PACKED_DOT,
                GGMLType.Q8_0, 1024, 1024, 16, MmqConfig.PACKED_DOT.rowRecords(1024), 0, 1024, 0), 128, 32, 128, 64,
                256));
        assertNull("unknown compute units never split", SplitKGemmPlan.ofSubgroupMmq(base, 128, 32, 128,
                DeviceCapabilities.UNKNOWN_COMPUTE_UNITS, 256));
        assertEquals(SplitKGemmPlan.MAX_SPLITS, SplitKGemmPlan.subgroupSplitsFor(64, 4, 2, 2048));
        assertEquals(1, SplitKGemmPlan.subgroupSplitsFor(64, 9, 1, 2048));
        assertEquals(1, SplitKGemmPlan.subgroupSplitsFor(64, 4, 4, 2048));
        assertEquals(1, SplitKGemmPlan.subgroupSplitsFor(64, 1, 1, 2047));
        assertEquals(SplitKGemmPlan.MAX_SPLITS, SplitKGemmPlan.subgroupSplitsFor(128, 1, 1, 4096));
        expectInvalid("multiple of 64", () -> SplitKGemmPlan.ofSubgroupMmq(MmqGemmPlan.of(MmqConfig.PACKED_DOT,
                GGMLType.Q8_0, 128, 2080, 1, MmqConfig.PACKED_DOT.rowRecords(2080), 0, 128, 0), 128, 32, 128, 64,
                256));
    }

    /**
     * The subgroup split counts at most the 64 compute units of the device it was measured on: a device of 512 units
     * splits the eight-tile grid that split there, but neither the 16-tile nor the 20-32-tile grids that measured
     * 0.82-1.2x split; a device of fewer units keeps its own bound.
     */
    @Test
    public void theSubgroupSplitCountsNoMoreUnitsThanTheMeasuredDevice() {
        assertEquals(SplitKGemmPlan.MAX_SPLITS, SplitKGemmPlan.subgroupSplitsFor(512, 8, 1, 2048));
        assertEquals("16 tiles", 1, SplitKGemmPlan.subgroupSplitsFor(512, 16, 1, 2560));
        assertEquals("4096 x 2560 at 64 tokens", 1, SplitKGemmPlan.subgroupSplitsFor(512, 32, 2, 2560));
        assertEquals("2560 x 9728 at 32 tokens", 1, SplitKGemmPlan.subgroupSplitsFor(512, 20, 1, 9728));
        assertEquals(1, SplitKGemmPlan.subgroupSplitsFor(65, 9, 1, 2048));
        assertEquals(SplitKGemmPlan.MAX_SPLITS, SplitKGemmPlan.subgroupSplitsFor(32, 4, 1, 2048));
        assertEquals(1, SplitKGemmPlan.subgroupSplitsFor(32, 5, 1, 2048));
        assertNull("the 20-tile 2560 x 9728 grid over 512 units stays whole", SplitKGemmPlan.ofSubgroupMmq(
                MmqGemmPlan.of(MmqConfig.PACKED_DOT, GGMLType.Q4_K, 2560, 9728, 32,
                        MmqConfig.PACKED_DOT.rowRecords(9728), 0, 2560, 0), 128, 32, 128, 512, 256));
    }

    /**
     * The subgroup split's device check judges its own tile, not the packed-dot config whose records it reads: the
     * 128-item OpenCL tile with its 8192-byte double-buffered stage runs on a device of 128-item work-groups and
     * 12 KiB of local memory, below the config's 256 items and 15040 bytes, while the 256-item work-group of a
     * token-split tile and a device below 8192 bytes are refused.
     */
    @Test
    public void theSubgroupSplitChecksItsOwnTileOnTheDevice() {
        MmqGemmPlan base = MmqGemmPlan.of(MmqConfig.PACKED_DOT, GGMLType.Q8_0, 1024, 2048, 16,
                MmqConfig.PACKED_DOT.rowRecords(2048), 0, 1024, 0);
        SplitKGemmPlan p = SplitKGemmPlan.ofSubgroupMmq(base, 128, 32, 128, 64, 256);
        assertNotNull(p);
        assertEquals(8192, p.subgroupSharedBytes());
        assertTrue(MmqConfig.PACKED_DOT.workGroupSize() > 128);
        assertTrue(MmqConfig.PACKED_DOT.sharedBytes() > 12288);
        assertNull(p.checkDevice(subgroupDevice(128, 12288)));
        String wide = SplitKGemmPlan.ofSubgroupMmq(base, 128, 32, 256, 64, 256).checkDevice(subgroupDevice(128,
                12288));
        assertNotNull(wide);
        assertTrue(wide, wide.contains("Q8_0_MatMulSg_MMQ_KSplit needs work-groups of 256"));
        String small = p.checkDevice(subgroupDevice(1024, 8188));
        assertNotNull(small);
        assertTrue(small, small.contains("needs 8192 bytes of local memory"));
    }

    private static DeviceCapabilities subgroupDevice(int maxWorkGroup, long sharedBytes) {
        return new DeviceCapabilities.Builder()
                .deviceName("subgroup-device")
                .maxWorkGroupSize(maxWorkGroup)
                .maxSharedMemoryBytes(sharedBytes)
                .maxWorkGroupCount(new long[]{65535, 65535, 65535})
                .build();
    }

    private static void expectInvalid(String messagePart, Runnable action) {
        try {
            action.run();
            fail("expected IllegalArgumentException containing '" + messagePart + "'");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage(), e.getMessage().contains(messagePart));
        }
    }
}

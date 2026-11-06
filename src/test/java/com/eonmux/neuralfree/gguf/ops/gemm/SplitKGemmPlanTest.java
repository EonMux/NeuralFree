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

    /** The small-token-tile heuristic splits a sub-wave staged grid to the cap and leaves a full wave alone. */
    @Test
    public void theSmallTileHeuristicFillsTheWaveWithTheCappedSliceCount() {
        int units = 48;
        assertEquals("a 40-tile grid over 48 units fills the device with 8 slices",
                SplitKGemmPlan.MAX_SPLITS, SplitKGemmPlan.smallTileSplitsFor(units, 40, 1, 9728));
        assertEquals("a 16-tile grid over 48 units splits by the cap too",
                SplitKGemmPlan.MAX_SPLITS, SplitKGemmPlan.smallTileSplitsFor(units, 16, 1, 2560));
        assertEquals("a full wave keeps one slice", 1, SplitKGemmPlan.smallTileSplitsFor(units, 48, 1, 9728));
        assertEquals("a grid past the wave keeps one slice", 1, SplitKGemmPlan.smallTileSplitsFor(units, 64, 1, 9728));
        assertEquals("small K keeps one slice", 1, SplitKGemmPlan.smallTileSplitsFor(units, 40, 1, 2047));
        int unknown = DeviceCapabilities.UNKNOWN_COMPUTE_UNITS;
        assertEquals("an unreported core count keeps one slice", 1,
                SplitKGemmPlan.smallTileSplitsFor(unknown, 40, 1, 9728));
        expectInvalid("non-negative", () -> SplitKGemmPlan.smallTileSplitsFor(-1, 40, 1, 9728));
    }

    /** The small-token-tile plan keeps the device tile's KSplit kernel and slices K across the cap. */
    @Test
    public void theSmallTilePlanSlicesKAcrossTheDeviceTileKernel() {
        MmqGemmPlan base = MmqGemmPlan.of(MmqConfig.INT8_TENSOR, GGMLType.Q8_0, 2560, 9728, 9,
                MmqConfig.INT8_TENSOR.rowRecords(9728), 0, 2560, 0);
        SplitKGemmPlan p = SplitKGemmPlan.ofSmallTiles(base, 48, 256);
        assertNotNull("a 40-tile grid over 48 units splits", p);
        assertEquals("Q8_0_MatMul_MMQ_KSplit", p.kernelName());
        assertEquals(SplitKGemmPlan.MAX_SPLITS, p.splits());
        assertEquals("slices are whole stage depths", 0, p.kSlice() % MmqConfig.INT8_TENSOR.stageDepth());
        assertTrue("the last slice is not empty", (long) p.kSlice() * (p.splits() - 1) < p.k());
        assertArrayEquals(new long[]{40L * 256 * p.splits(), 1}, p.partialGlobalWorkSize());
        assertNull("a full wave keeps the plain kernel",
                SplitKGemmPlan.ofSmallTiles(MmqGemmPlan.of(MmqConfig.INT8_TENSOR, GGMLType.Q8_0, 3072, 9728, 9,
                        MmqConfig.INT8_TENSOR.rowRecords(9728), 0, 3072, 0), 48, 256));
        SplitKGemmPlan w4a8 = SplitKGemmPlan.ofW4a8SmallTiles(GGMLType.MXFP4, 2560, 4096, 9, 128, 0,
                MmqConfig.INT8_TENSOR, 48, 256);
        assertNotNull(w4a8);
        assertEquals("MXFP4_MatMul_W4A8_KSplit", w4a8.kernelName());
        assertEquals(SplitKGemmPlan.MAX_SPLITS, w4a8.splits());
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

    private static void expectInvalid(String messagePart, Runnable action) {
        try {
            action.run();
            fail("expected IllegalArgumentException containing '" + messagePart + "'");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage(), e.getMessage().contains(messagePart));
        }
    }
}

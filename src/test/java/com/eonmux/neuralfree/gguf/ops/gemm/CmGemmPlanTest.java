package com.eonmux.neuralfree.gguf.ops.gemm;

import com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.testutil.FakeGpuBuffer;
import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * Pure-logic checks of the CM GEMM plan: the MMQ type set with the MMA tile geometry, shape and view
 * validation, device limits and argument order. No GPU is used.
 */
public class CmGemmPlanTest {

    @Test
    public void geometryMatchesTheMmaTileWithMmqWeightBytes() {
        CmGemmPlan p = CmGemmPlan.contiguous(GGMLType.Q4_K, 4096 + 1, 4096, 100);
        assertEquals("Q4_K_MatMul_CM", p.kernelName());
        assertEquals(GGMLType.Q4_K, p.type());
        assertEquals(65, p.rowTiles());
        assertEquals(4, p.tokenTiles());
        assertArrayEquals(new long[]{65L * 256, 4}, p.globalWorkSize());
        assertArrayEquals(new long[]{256, 1}, p.localWorkSize());
        assertEquals(16L * 144, p.weightRowBytes());
        assertEquals(16L * 144 * 4097, p.weightBytes());
        assertEquals(99L * 4096 + 4096, p.xElementsRequired());
        assertEquals(99L * 4097 + 4097, p.yElementsRequired());
        assertEquals(MmaGemmPlan.TILE_ROWS, CmGemmPlan.TILE_ROWS);
        assertEquals(MmaGemmPlan.TILE_TOKENS, CmGemmPlan.TILE_TOKENS);
        assertEquals(MmaGemmPlan.WORKGROUP_SIZE, CmGemmPlan.WORKGROUP_SIZE);
        assertEquals("fp16 tiles of 64 x 24 and 32 x 40 halfs plus the 8 x 16 x 16 float stage",
                2L * (64 * 24 + 32 * 40) + 4L * 8 * 16 * 16, CmGemmPlan.TILE_SHARED_BYTES);
    }

    @Test
    public void viewsAddTheirOffsets() {
        CmGemmPlan p = CmGemmPlan.of(GGMLType.Q8_0, 64, 96, 32, 300, 12, 200, 7);
        assertEquals(1, p.rowTiles());
        assertEquals(1, p.tokenTiles());
        assertEquals(3L * 34, p.weightRowBytes());
        assertEquals(12L + 31L * 300 + 96, p.xElementsRequired());
        assertEquals(7L + 31L * 200 + 64, p.yElementsRequired());
    }

    @Test
    public void invalidShapesAreRejectedWithTheReason() {
        expectInvalid("multiple of the block size", () -> CmGemmPlan.contiguous(GGMLType.Q4_K, 64, 4096 + 32, 8));
        expectInvalid("M, K, N >= 1", () -> CmGemmPlan.contiguous(GGMLType.Q4_K, 64, 256, 0));
        expectInvalid("shorter than K", () -> CmGemmPlan.of(GGMLType.Q8_0, 8, 64, 2, 63, 0, 8, 0));
        expectInvalid("shorter than M", () -> CmGemmPlan.of(GGMLType.Q8_0, 8, 64, 2, 64, 0, 7, 0));
        expectInvalid("offsets must be >= 0", () -> CmGemmPlan.of(GGMLType.Q8_0, 8, 64, 2, 64, -1, 8, 0));
        expectInvalid("no CM GEMM kernel", () -> CmGemmPlan.contiguous(GGMLType.F16, 8, 64, 2));
        expectInvalid("overflow", () -> CmGemmPlan.contiguous(GGMLType.Q4_K, Integer.MAX_VALUE, 256, 1));
        CmGemmPlan valid = CmGemmPlan.of(GGMLType.Q4_K, 64, 256, 4, 256, 0, 64, 0);
        try {
            valid.args(null, null, null);
            fail("expected the null-buffer rejection");
        } catch (NullPointerException expected) {
            // args validates its buffers
        }
    }

    @Test
    public void argumentsFollowTheDeclaredOrder() {
        CmGemmPlan p = CmGemmPlan.of(GGMLType.Q4_K, 128, 256, 9, 256, 5, 128, 3);
        FakeGpuBuffer w = new FakeGpuBuffer(1, p.weightBytes(), true);
        FakeGpuBuffer x = new FakeGpuBuffer(2, p.xElementsRequired() * Float.BYTES, true);
        FakeGpuBuffer y = new FakeGpuBuffer(3, p.yElementsRequired() * Float.BYTES, false);
        Object[] args = p.args(w, x, y);
        assertEquals(GemmKernels.MATMUL_TILED_ARG_COUNT, args.length);
        assertEquals(128, args[GemmKernels.ARG_M]);
        assertEquals(256, args[GemmKernels.ARG_K]);
        assertEquals(9, args[GemmKernels.ARG_N]);
        assertEquals(256, args[GemmKernels.ARG_X_STRIDE]);
        assertEquals(5, args[GemmKernels.ARG_X_OFF]);
        assertEquals(128, args[GemmKernels.ARG_Y_STRIDE]);
        assertEquals(3, args[GemmKernels.ARG_Y_OFF]);
        assertTrue(w == args[GemmKernels.ARG_WEIGHT]);
        assertTrue(x == args[GemmKernels.ARG_X]);
        assertTrue(y == args[GemmKernels.ARG_Y]);
    }

    @Test
    public void deviceLimitsAreReported() {
        DeviceCapabilities big = caps(1024, CmGemmPlan.TILE_SHARED_BYTES, new long[]{65535, 65535, 65535});
        assertNull(CmGemmPlan.deviceSupport(GGMLType.Q4_K, big));
        assertNull(CmGemmPlan.contiguous(GGMLType.Q4_K, 4096, 4096, 512).checkDevice(big));

        DeviceCapabilities smallWg = caps(128, CmGemmPlan.TILE_SHARED_BYTES, new long[]{65535, 65535, 65535});
        assertNotNull(CmGemmPlan.deviceSupport(GGMLType.Q4_K, smallWg));

        DeviceCapabilities tinyShared = caps(1024, CmGemmPlan.TILE_SHARED_BYTES - 1,
                new long[]{65535, 65535, 65535});
        assertNotNull(CmGemmPlan.deviceSupport(GGMLType.Q4_K, tinyShared));
        assertNotNull("types without a CM kernel report their own reason",
                CmGemmPlan.deviceSupport(GGMLType.F16, big));
        assertFalse(CmGemmPlan.deviceSupport(GGMLType.F16, big).isEmpty());
    }

    private static DeviceCapabilities caps(long maxWg, long sharedBytes, long[] maxGroups) {
        return new DeviceCapabilities.Builder()
                .deviceName("test-device")
                .maxWorkGroupSize(maxWg)
                .maxSharedMemoryBytes(sharedBytes)
                .maxWorkGroupCount(maxGroups)
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

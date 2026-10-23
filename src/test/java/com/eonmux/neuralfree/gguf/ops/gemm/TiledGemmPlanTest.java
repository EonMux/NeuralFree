package com.eonmux.neuralfree.gguf.ops.gemm;

import com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.testutil.FakeGpuBuffer;
import org.junit.Test;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * Pure-logic checks of the GEMM family plans: kernel naming, the 2-D tile geometry, shape and view
 * validation, device limits, argument order, and that {@code matmul_tiled.cl} instantiates exactly the
 * {@link GemmKernels#TILED_TYPES}. No GPU is used.
 */
public class TiledGemmPlanTest {

    @Test
    public void everyWeightTypeOfTheFamilyHasOneTiledKernel() {
        assertEquals(29, GemmKernels.TILED_TYPES.size());
        assertEquals("Q4_K_MatMul_Tiled", GemmKernels.matmulTiled(GGMLType.Q4_K));
        assertEquals("BF16_MatMul_Tiled", GemmKernels.matmulTiled(GGMLType.BF16));
        assertEquals("Q2_0_MatMul_Tiled", GemmKernels.matmulTiled(GGMLType.Q2_0));
        assertFalse(GemmKernels.hasTiled(GGMLType.Q8_1));
        assertFalse(GemmKernels.hasTiled(GGMLType.I32));
        assertFalse(GemmKernels.hasTiled(null));
        List<String> names = GemmKernels.kernelNames();
        assertEquals(30, names.size());
        assertEquals("F32_MatMul_Tiled", names.get(0));
        assertEquals(GemmKernels.TOPK_PARTIAL, names.get(names.size() - 1));
        assertEquals(30, new TreeSet<>(names).size());
    }

    @Test(expected = IllegalArgumentException.class)
    public void typesWithoutTiledKernelAreRejected() {
        GemmKernels.matmulTiled(GGMLType.Q8_1);
    }

    @Test
    public void openClSourceInstantiatesExactlyTheTiledTypes() throws IOException {
        String source = resource("kernels/llm/matmul_tiled.cl");
        Matcher m = Pattern.compile("(?m)^NF_GEMM_TILED\\(\\s*(\\w+)\\s*,").matcher(source);
        Set<String> instantiated = new TreeSet<>();
        while (m.find()) {
            assertTrue("duplicate instantiation " + m.group(1), instantiated.add(m.group(1)));
        }
        Set<String> expected = new TreeSet<>();
        for (GGMLType t : GemmKernels.TILED_TYPES) {
            expected.add(t.name());
        }
        assertEquals(expected, instantiated);
    }

    @Test
    public void geometryCoversEveryRowAndTokenTile() {
        TiledGemmPlan p = TiledGemmPlan.contiguous(GGMLType.Q4_K, 4096 + 1, 4096, 100);
        assertEquals("Q4_K_MatMul_Tiled", p.kernelName());
        assertEquals(65, p.rowTiles());
        assertEquals(4, p.tokenTiles());
        assertArrayEquals(new long[]{65L * 256, 4}, p.globalWorkSize());
        assertArrayEquals(new long[]{256, 1}, p.localWorkSize());
        assertEquals(4096, p.xStride());
        assertEquals(4097, p.yStride());
        assertEquals(16L * 144, p.weightRowBytes());
        assertEquals(16L * 144 * 4097, p.weightBytes());
        assertEquals(99L * 4096 + 4096, p.xElementsRequired());
        assertEquals(99L * 4097 + 4097, p.yElementsRequired());
    }

    @Test
    public void viewsAddTheirOffsets() {
        TiledGemmPlan p = TiledGemmPlan.of(GGMLType.F16, 64, 96, 32, 300, 12, 200, 7);
        assertEquals(1, p.rowTiles());
        assertEquals(1, p.tokenTiles());
        assertEquals(96L * 2, p.weightRowBytes());
        assertEquals(12L + 31L * 300 + 96, p.xElementsRequired());
        assertEquals(7L + 31L * 200 + 64, p.yElementsRequired());
    }

    @Test
    public void invalidShapesAreRejectedWithTheReason() {
        expectInvalid("multiple of the block size", () -> TiledGemmPlan.contiguous(GGMLType.Q4_K, 64, 4096 + 32, 8));
        expectInvalid("M, K, N >= 1", () -> TiledGemmPlan.contiguous(GGMLType.F32, 64, 64, 0));
        expectInvalid("shorter than K", () -> TiledGemmPlan.of(GGMLType.F32, 8, 64, 2, 63, 0, 8, 0));
        expectInvalid("shorter than M", () -> TiledGemmPlan.of(GGMLType.F32, 8, 64, 2, 64, 0, 7, 0));
        expectInvalid("offsets must be >= 0", () -> TiledGemmPlan.of(GGMLType.F32, 8, 64, 2, 64, -1, 8, 0));
        expectInvalid("no tiled GEMM kernel", () -> TiledGemmPlan.contiguous(GGMLType.Q8_1, 8, 64, 2));
    }

    @Test
    public void argumentsFollowTheDeclaredOrder() {
        TiledGemmPlan p = TiledGemmPlan.of(GGMLType.Q8_0, 128, 64, 9, 64, 5, 128, 3);
        FakeGpuBuffer w = new FakeGpuBuffer(1, p.weightBytes(), true);
        FakeGpuBuffer x = new FakeGpuBuffer(2, p.xElementsRequired() * Float.BYTES, true);
        FakeGpuBuffer y = new FakeGpuBuffer(3, p.yElementsRequired() * Float.BYTES, false);
        Object[] args = p.args(w, x, y);
        assertEquals(GemmKernels.MATMUL_TILED_ARG_COUNT, args.length);
        assertEquals(128, args[GemmKernels.ARG_M]);
        assertEquals(64, args[GemmKernels.ARG_K]);
        assertEquals(9, args[GemmKernels.ARG_N]);
        assertEquals(64, args[GemmKernels.ARG_X_STRIDE]);
        assertEquals(5, args[GemmKernels.ARG_X_OFF]);
        assertEquals(128, args[GemmKernels.ARG_Y_STRIDE]);
        assertEquals(3, args[GemmKernels.ARG_Y_OFF]);
        assertSame(w, args[GemmKernels.ARG_WEIGHT]);
        assertSame(x, args[GemmKernels.ARG_X]);
        assertSame(y, args[GemmKernels.ARG_Y]);
    }

    @Test
    public void undersizedBuffersAreRejected() {
        TiledGemmPlan p = TiledGemmPlan.contiguous(GGMLType.F32, 16, 32, 4);
        FakeGpuBuffer w = new FakeGpuBuffer(1, p.weightBytes(), true);
        FakeGpuBuffer x = new FakeGpuBuffer(2, p.xElementsRequired() * Float.BYTES - 4, true);
        FakeGpuBuffer y = new FakeGpuBuffer(3, p.yElementsRequired() * Float.BYTES, false);
        expectInvalid("activation buffer", () -> p.args(w, x, y));
        expectInvalid("distinct buffers", () -> p.args(w, y, y));
    }

    @Test
    public void deviceLimitsAreReported() {
        DeviceCapabilities big = caps(1024, 49152, new long[]{65535, 65535, 65535});
        assertNull(TiledGemmPlan.deviceSupport(GGMLType.IQ2_S, big));
        assertNull(TiledGemmPlan.contiguous(GGMLType.Q4_0, 4096, 4096, 512).checkDevice(big));

        DeviceCapabilities smallWg = caps(128, 49152, new long[]{65535, 65535, 65535});
        assertNotNull(TiledGemmPlan.deviceSupport(GGMLType.F32, smallWg));

        assertEquals(6272L, TiledGemmPlan.TILE_SHARED_BYTES);
        assertEquals(6272L + 8192L, TiledGemmPlan.requiredSharedBytes(GGMLType.IQ2_S));
        DeviceCapabilities tinyShared = caps(1024, 8192, new long[]{65535, 65535, 65535});
        assertNull(TiledGemmPlan.deviceSupport(GGMLType.Q4_K, tinyShared));
        assertNotNull(TiledGemmPlan.deviceSupport(GGMLType.IQ2_S, tinyShared));

        DeviceCapabilities fewGroups = caps(1024, 49152, new long[]{100, 65535, 65535});
        String reason = TiledGemmPlan.contiguous(GGMLType.F16, 64 * 101, 64, 8).checkDevice(fewGroups);
        assertNotNull(reason);
        assertTrue(reason, reason.contains("dimension 0"));
    }

    @Test
    public void topKPartialSlicesTheVocabulary() {
        GemmKernels.TopKPartialPlan p = GemmKernels.TopKPartialPlan.of(152064, 40, 256);
        assertEquals(38, p.groups());
        assertEquals(4002, p.sliceLength());
        assertEquals(38 * 40, p.candidateCount());
        assertArrayEquals(new long[]{38L * 256}, p.globalWorkSize());
        assertArrayEquals(new long[]{256}, p.localWorkSize());

        GemmKernels.TopKPartialPlan small = GemmKernels.TopKPartialPlan.of(10, 64, 32);
        assertEquals(1, small.groups());
        assertEquals(64, small.candidateCount());

        GemmKernels.TopKPartialPlan huge = GemmKernels.TopKPartialPlan.of(GemmKernels.TopKPartialPlan.MAX_N, 1, 64);
        assertEquals(GemmKernels.TopKPartialPlan.MAX_GROUPS, huge.groups());

        FakeGpuBuffer logits = new FakeGpuBuffer(1, 152064L * 4, true);
        FakeGpuBuffer val = new FakeGpuBuffer(2, 38L * 40 * 4, false);
        FakeGpuBuffer idx = new FakeGpuBuffer(3, 38L * 40 * 4, false);
        Object[] args = p.args(0.5f, logits, val, idx);
        assertEquals(GemmKernels.TOPK_ARG_COUNT, args.length);
        assertEquals(152064, args[GemmKernels.TOPK_ARG_N]);
        assertEquals(40, args[GemmKernels.TOPK_ARG_K]);
        assertEquals(0.5f, (Float) args[GemmKernels.TOPK_ARG_INV_TEMP], 0.0f);
        assertSame(logits, args[GemmKernels.TOPK_ARG_LOGITS]);
        assertSame(val, args[GemmKernels.TOPK_ARG_CAND_VAL]);
        assertSame(idx, args[GemmKernels.TOPK_ARG_CAND_IDX]);
    }

    @Test
    public void topKPartialRejectsInvalidRequests() {
        expectInvalid("1 <= k <= 64", () -> GemmKernels.TopKPartialPlan.of(100, 65, 256));
        expectInvalid("1 <= k <= 64", () -> GemmKernels.TopKPartialPlan.of(100, 0, 256));
        expectInvalid("at least one logit", () -> GemmKernels.TopKPartialPlan.of(0, 4, 256));
        expectInvalid("at most", () -> GemmKernels.TopKPartialPlan.of(GemmKernels.TopKPartialPlan.MAX_N + 1, 4, 256));
        expectInvalid("power of two", () -> GemmKernels.TopKPartialPlan.of(100, 4, 96));
        expectInvalid("power of two", () -> GemmKernels.TopKPartialPlan.of(100, 4, 512));
        GemmKernels.TopKPartialPlan p = GemmKernels.TopKPartialPlan.of(100, 4, 64);
        FakeGpuBuffer b = new FakeGpuBuffer(1, 400, false);
        expectInvalid("inv_temp", () -> p.args(0.0f, b, b, b));
        expectInvalid("inv_temp", () -> p.args(Float.NaN, b, b, b));
        expectInvalid("inv_temp", () -> p.args(Float.POSITIVE_INFINITY, b, b, b));
        expectInvalid("distinct buffers", () -> p.args(1.0f, b, b, new FakeGpuBuffer(2, 16, false)));

        FakeGpuBuffer logits = new FakeGpuBuffer(3, 100L * 4, true);
        FakeGpuBuffer val = new FakeGpuBuffer(4, 4L * 4, false);
        FakeGpuBuffer shortIdx = new FakeGpuBuffer(5, 4L * 4 - 1, false);
        expectInvalid("cand_idx buffer", () -> p.args(1.0f, logits, val, shortIdx));
        FakeGpuBuffer shortLogits = new FakeGpuBuffer(6, 99L * 4, true);
        expectInvalid("logits buffer", () -> p.args(1.0f, shortLogits, val, new FakeGpuBuffer(7, 16, false)));
    }

    @Test
    public void topKPartialDeviceLimitsAreReported() {
        GemmKernels.TopKPartialPlan p = GemmKernels.TopKPartialPlan.of(152064, 40, 256);
        assertNull(p.checkDevice(caps(1024, 49152, new long[]{65535, 65535, 65535})));
        String reason = p.checkDevice(caps(128, 49152, new long[]{65535, 65535, 65535}));
        assertNotNull(reason);
        assertTrue(reason, reason.contains("work-groups of 256"));
        reason = p.checkDevice(caps(1024, 49152, new long[]{16, 65535, 65535}));
        assertNotNull(reason);
        assertTrue(reason, reason.contains("38 work-groups"));
    }

    @Test
    public void tileIndexOverflowIsRejected() {
        expectInvalid("overflow", () -> TiledGemmPlan.contiguous(GGMLType.F32, Integer.MAX_VALUE, 64, 1));
        expectInvalid("overflow", () -> TiledGemmPlan.contiguous(GGMLType.F32, 64, 64, Integer.MAX_VALUE - 8));
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

    private static String resource(String path) throws IOException {
        try (InputStream in = TiledGemmPlanTest.class.getClassLoader().getResourceAsStream(path)) {
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

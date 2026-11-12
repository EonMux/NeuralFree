package com.eonmux.neuralfree.gguf.ops.gemm;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
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
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * Pure-logic checks of the fast GEMM family plans: the MMQ type set (the llama.cpp MMQ types plus the
 * i-quants iq1_s..iq4_xs) and kernel naming, the per-architecture MMQ tiles and the 32-token MMA tile
 * geometry, shape and view validation in staged-record units, device limits, argument order, and that
 * {@code matmul_mmq.cl} / {@code matmul_cm.cl} / {@code matmul_mma.cl} instantiate exactly the planned
 * types. No GPU is used.
 */
public class MmqGemmPlanTest {

    @Test
    public void mmqTypesAreTheLlamaCppMmqSet() {
        assertEquals(19, GemmKernels.MMQ_TYPES.size());
        assertTrue(GemmKernels.hasMmq(GGMLType.Q4_K));
        assertTrue(GemmKernels.hasMmq(GGMLType.Q8_0));
        assertTrue("the i-quants llama.cpp mmq serves have MMQ kernels", GemmKernels.hasMmq(GGMLType.IQ1_S));
        assertTrue(GemmKernels.hasMmq(GGMLType.IQ2_XS));
        assertTrue(GemmKernels.hasMmq(GGMLType.IQ4_XS));
        assertTrue("I8_B128 is the W8A8 int8 weight format", GemmKernels.hasMmq(GGMLType.I8_B128));
        assertFalse("dense and extension types have no MMQ kernel", GemmKernels.hasMmq(GGMLType.F16));
        assertFalse("IQ1_M is not in llama.cpp's mmq set", GemmKernels.hasMmq(GGMLType.IQ1_M));
        assertFalse("TQ1_0 is not in llama.cpp's mmq set", GemmKernels.hasMmq(GGMLType.TQ1_0));
        assertFalse(GemmKernels.hasMmq(GGMLType.Q8_K));
        assertFalse(GemmKernels.hasMmq(null));
        assertEquals("Q4_K_MatMul_MMQ", GemmKernels.matmulMmq(GGMLType.Q4_K));
        assertEquals("Q8_0_MatMul_MMQ", GemmKernels.matmulMmq(GGMLType.Q8_0));
        assertEquals("IQ2_XS_MatMul_MMQ", GemmKernels.matmulMmq(GGMLType.IQ2_XS));
        assertEquals("F16_MatMul_MMA", GemmKernels.matmulMma(GGMLType.F16));
        assertEquals("BF16_MatMul_MMA", GemmKernels.matmulMma(GGMLType.BF16));
        assertEquals("Q4_K_MatMul_CM", GemmKernels.matmulCm(GGMLType.Q4_K));
        assertEquals("Q4_K_MatMul_MMQ_KSplit", GemmKernels.matmulMmqKSplit(GGMLType.Q4_K));
        assertEquals("Q4_K_MatMul_CM_KSplit", GemmKernels.matmulCmKSplit(GGMLType.Q4_K));
        assertEquals("F16_MatMul_MMA_KSplit", GemmKernels.matmulMmaKSplit(GGMLType.F16));
        assertEquals("BF16_MatMul_MMA_KSplit", GemmKernels.matmulMmaKSplit(GGMLType.BF16));
        assertEquals("Q4_K_MatMul_MMQ_S", GemmKernels.matmulMmqSmall(GGMLType.Q4_K));
        assertEquals("Q4_K_MatMul_MMQ_S64", GemmKernels.matmulMmqSmall(GGMLType.Q4_K, MmqConfig.TWIN_MAX_TOKENS));
        assertEquals("MXFP4_MatMul_W4A8_S64",
                GemmKernels.matmulW4a8Small(GGMLType.MXFP4, MmqConfig.TWIN_MAX_TOKENS));
        List<String> names = GemmKernels.kernelNames();
        assertTrue(names.contains("Q4_K_MatMul_MMQ"));
        assertTrue(names.contains("Q4_K_MatMul_MMQ_KSplit"));
        assertTrue(names.contains("Q4_K_MatMul_MMQ_S"));
        assertTrue(names.contains("Q4_K_MatMul_MMQ_S64"));
        assertTrue(names.contains("Q4_K_MatMul_MMQ_SKSplit64"));
        assertTrue(names.contains("MXFP4_MatMul_W4A8_SKSplit64"));
        assertTrue(names.contains("F16_MatMul_MMA"));
        assertTrue(names.contains("BF16_MatMul_MMA"));
        assertTrue(names.contains("Q4_K_MatMul_CM"));
        assertTrue(names.contains("Q4_K_MatMul_CM_KSplit"));
        assertTrue(names.contains("Q4_K_MatMul_CM2"));
        assertTrue(names.contains("Q4_K_MatMul_CM2_KSplit"));
        assertTrue(names.contains("Q4_K_MatMul_CM2V"));
        assertTrue(names.contains("Q4_K_MatMul_CM2V_KSplit"));
        assertFalse(names.contains("F16_MatMul_CM2V"));
        assertTrue(names.contains("F16_MatMul_MMA_KSplit"));
        assertTrue(names.contains("BF16_MatMul_MMA_KSplit"));
        assertTrue(names.contains("LLM_Quant_MMQ_Q8_1"));
        assertTrue(names.contains("LLM_Quant_MMQ_Int8Static"));
        assertTrue(names.contains("LLM_Quant_MMQ_Int8Row"));
        assertTrue(names.contains("LLM_Quant_MMQ_Int8Group128"));
        assertTrue(names.contains("LLM_SplitK_Reduce"));
        assertTrue(names.contains(GemmKernels.LOGITS_PREFILTER));
        assertEquals(GemmKernels.TILED_TYPES.size() + GemmKernels.MMQ_TYPES.size() * 8
                        + GemmKernels.FP8_TYPES.size() * 6 + GemmKernels.W4A8_TYPES.size() * 6
                        + GemmKernels.MMA_TYPES.size() * 2 + GemmKernels.CM2_TYPES.size() * 2
                        + (GemmKernels.CM2_TYPES.size() - 1) * 4 + 1 + 14,
                names.size());
    }

    @Test(expected = IllegalArgumentException.class)
    public void typesWithoutMmqKernelAreRejected() {
        GemmKernels.matmulMmq(GGMLType.F16);
    }

    @Test
    public void openClSourcesInstantiateExactlyTheFastTypes() throws IOException {
        Set<String> mmq = new TreeSet<>(instantiated("kernels/llm/matmul_mmq.cl",
                "(?m)^NF_MMQ_GEMM\\(\\s*(\\w+)\\s*,"));
        Set<String> mmqKSplit = new TreeSet<>(instantiated("kernels/llm/matmul_mmq.cl",
                "(?m)^NF_MMQ_GEMM_KSPLIT\\(\\s*(\\w+)\\s*,"));
        Set<String> mma = new TreeSet<>(instantiated("kernels/llm/matmul_mma.cl",
                "(?m)^NF_MMA_GEMM\\(\\s*(\\w+)\\s*,"));
        Set<String> cm = new TreeSet<>(instantiated("kernels/llm/matmul_cm.cl",
                "(?m)^NF_GEMM_TILED_AS\\(\\s*(\\w+)\\s*,\\s*MatMul_CM,"));
        Set<String> cmKSplit = new TreeSet<>(instantiated("kernels/llm/matmul_cm.cl",
                "(?m)^NF_GEMM_TILED_KSPLIT_AS\\(\\s*(\\w+)\\s*,\\s*MatMul_CM,"));
        Set<String> expectedMmq = new TreeSet<>();
        Set<String> expectedCm = new TreeSet<>();
        for (GGMLType t : GemmKernels.MMQ_TYPES) {
            expectedMmq.add(t.name());
            expectedCm.add(t.name());
        }
        assertEquals(expectedMmq, mmq);
        assertEquals(expectedMmq, mmqKSplit);
        assertEquals(expectedCm, cm);
        assertEquals(expectedCm, cmKSplit);
        Set<String> expectedMma = new TreeSet<>();
        for (GGMLType t : GemmKernels.MMA_TYPES) {
            expectedMma.add(t.name());
        }
        assertEquals(expectedMma, mma);
    }

    @Test
    public void mmqGeometryCoversEveryRowAndTokenTile() {
        MmqGemmPlan p = MmqGemmPlan.contiguous(MmqConfig.PACKED_DOT, GGMLType.Q4_K, 4096 + 1, 4096, 33);
        assertEquals("Q4_K_MatMul_MMQ", p.kernelName());
        assertEquals(65, p.rowTiles());
        assertEquals(3, p.tokenTiles());
        assertArrayEquals(new long[]{65L * 256, 3}, p.globalWorkSize());
        assertArrayEquals(new long[]{256, 1}, p.localWorkSize());
        assertEquals("contiguous staged rows hold one record per 16-deep chunk", 256, p.qStride());
        assertEquals(4097, p.yStride());
        assertEquals(16L * 144, p.weightRowBytes());
        assertEquals(16L * 144 * 4097, p.weightBytes());
        assertEquals("33 staged rows of 256 chunk records of 7 units",
                33L * 256 * 7, p.q8ElementsRequired());
        assertEquals(32L * 4097 + 4097, p.yElementsRequired());
        assertEquals(64, MmqGemmPlan.TILE_ROWS);
        assertEquals(256, MmqGemmPlan.WORKGROUP_SIZE);
    }

    @Test
    public void theTensorConfigTilesOneHundredTwentyEightTokens() {
        MmqGemmPlan p = MmqGemmPlan.contiguous(MmqConfig.INT8_TENSOR, GGMLType.Q4_K, 4096 + 1, 4096, 33);
        assertEquals("33 tokens fit one 128-token tile", 1, p.tokenTiles());
        assertArrayEquals(new long[]{33L * 256, 1}, p.globalWorkSize());
        assertEquals(MmqConfig.INT8_TENSOR, p.config());
        MmqGemmPlan wide = MmqGemmPlan.contiguous(MmqConfig.INT8_TENSOR, GGMLType.Q4_K, 4096, 4096, 129);
        assertEquals("129 tokens need two 128-token tiles", 2, wide.tokenTiles());
        assertEquals("the tensor stage is 32 deep: 129 rows of 128 records of 12 units",
                129L * 128 * 12, wide.q8ElementsRequired());
    }

    /** The small-tile twin launches the {@code _S} kernel over 16-token tiles of the 64-row twin tile. */
    @Test
    public void theSmallTileTwinPlansTheSuffixKernelOver16TokenTiles() {
        MmqGemmPlan p = MmqGemmPlan.smallOf(MmqConfig.INT8_TENSOR, GGMLType.Q4_K, 4096 + 1, 4096, 16,
                128, 0, 4097, 0);
        assertEquals("Q4_K_MatMul_MMQ_S", p.kernelName());
        assertEquals(MmqConfig.smallTile(MmqConfig.INT8_TENSOR), p.config());
        assertEquals("16 tokens need one 16-token tile", 1, p.tokenTiles());
        assertEquals("4097 rows need 65 64-row twin tiles", 65, p.rowTiles());
        assertArrayEquals(new long[]{65L * 256, 1}, p.globalWorkSize());
        assertEquals("the stage depth is the device tile's: 16 rows of 128 records of 12 units",
                16L * 128 * 12, p.q8ElementsRequired());
        expectInvalid("no 16-token twin", () -> MmqGemmPlan.smallOf(MmqConfig.PACKED_DOT, GGMLType.Q4_K,
                64, 256, 8, 16, 0, 64, 0));
    }

    /** The 17..64-token launches plan the S64 twin: the {@code _S64} kernel over 64-token tiles of
     *  the same 64-row twin tile; the twin widths are only 16 and 64 tokens. */
    @Test
    public void theTwin64PlansTheSuffixKernelOver64TokenTiles() {
        MmqGemmPlan p = MmqGemmPlan.smallOf(MmqConfig.INT8_TENSOR, GGMLType.Q4_K, MmqConfig.TWIN_MAX_TOKENS,
                4096 + 1, 4096, 48, 128, 0, 4097, 0);
        assertEquals("Q4_K_MatMul_MMQ_S64", p.kernelName());
        assertEquals(MmqConfig.smallTile(MmqConfig.INT8_TENSOR, MmqConfig.TWIN_MAX_TOKENS), p.config());
        assertEquals("48 tokens need one 64-token tile", 1, p.tokenTiles());
        assertEquals("4097 rows need 65 64-row twin tiles", 65, p.rowTiles());
        assertArrayEquals(new long[]{65L * 256, 1}, p.globalWorkSize());
        assertEquals("the stage depth is the device tile's: 48 rows of 128 records of 12 units",
                48L * 128 * 12, p.q8ElementsRequired());
        expectInvalid("twin tiles come in", () -> MmqGemmPlan.smallOf(MmqConfig.INT8_TENSOR, GGMLType.Q4_K,
                32, 4096, 4096, 32, 128, 0, 4096, 0));
        expectInvalid("no 64-token twin", () -> MmqGemmPlan.smallOf(MmqConfig.CDNA, GGMLType.Q4_K,
                MmqConfig.TWIN_MAX_TOKENS, 256, 4096, 32, 256, 0, 256, 0));
    }

    @Test
    public void configsFollowTheDevice() {
        assertEquals(MmqConfig.INT8_TENSOR, MmqConfig.forDevice(ComputeBackend.BackendType.CUDA, 80));
        assertEquals(MmqConfig.INT8_TENSOR, MmqConfig.forDevice(ComputeBackend.BackendType.CUDA, 120));
        assertEquals(MmqConfig.PACKED_DOT, MmqConfig.forDevice(ComputeBackend.BackendType.CUDA, 75));
        assertEquals(MmqConfig.PACKED_DOT, MmqConfig.forDevice(ComputeBackend.BackendType.OPENCL, 120));
        assertEquals(MmqConfig.PACKED_DOT, MmqConfig.forDevice(ComputeBackend.BackendType.VULKAN, 0));
        assertEquals(16, MmqConfig.PACKED_DOT.stageDepth());
        assertEquals(16, MmqConfig.PACKED_DOT.tileTokens());
        assertEquals(32, MmqConfig.INT8_TENSOR.stageDepth());
        assertEquals("the tensor token tile follows the per-architecture config", 128,
                MmqConfig.INT8_TENSOR.tileTokens());
        assertEquals(4 + 2 + 1, MmqConfig.PACKED_DOT.recordUnits());
        assertEquals(8 + 2 + 2, MmqConfig.INT8_TENSOR.recordUnits());
        assertEquals(4, MmqConfig.PACKED_DOT.rowRecords(64));
        assertEquals(256, MmqConfig.PACKED_DOT.rowRecords(4096));
        assertEquals(128, MmqConfig.INT8_TENSOR.rowRecords(4096));
        assertEquals(1536L, MmqConfig.INT8_TENSOR.rowRecords(4096) * MmqConfig.INT8_TENSOR.recordUnits());
        assertEquals("the pipelined stage holds three copies of the weight tiles and one of the "
                + "tokens' records", 4L * (3 * 64 * 16 + 9 * 64 + 16 * (4 + 2 + 1)),
                MmqConfig.PACKED_DOT.sharedBytes());
        assertEquals("the tensor tile's plain kernels hold two groups of two weight and two record stages",
                4L * (4 * 128 * 32 / 4 + 8 * 128 * 2 + 4 * 128 * (8 + 2 + 2)), MmqConfig.INT8_TENSOR.sharedBytes());
        assertEquals("a twin's bound is the larger of the pipelined stage and the tensor tile at its geometry",
                4L * Math.max(3 * 64 * 32 / 4 + 9 * 64 * 2 + 16 * (8 + 2 + 2),
                        4 * 64 * 32 / 4 + 8 * 64 * 2 + 4 * 16 * (8 + 2 + 2)),
                MmqConfig.smallTile(MmqConfig.INT8_TENSOR).sharedBytes());
    }

    @Test
    public void mmqViewsAddTheirOffsets() {
        MmqGemmPlan p = MmqGemmPlan.of(MmqConfig.PACKED_DOT, GGMLType.Q8_0, 64, 96, 32, 300, 12, 200, 7);
        assertEquals(1, p.rowTiles());
        assertEquals(2, p.tokenTiles());
        assertEquals(3L * 34, p.weightRowBytes());
        assertEquals("staged view arithmetic: qOff in units, the rows in records of 7 units",
                12L + (31L * 300 + 6) * 7, p.q8ElementsRequired());
        assertEquals(7L + 31L * 200 + 64, p.yElementsRequired());
    }

    @Test
    public void mmqInvalidShapesAreRejectedWithTheReason() {
        expectInvalid("multiple of the block size", () -> MmqGemmPlan.contiguous(MmqConfig.PACKED_DOT, GGMLType.Q4_K, 64, 4096 + 32, 8));
        expectInvalid("M, K, N >= 1", () -> MmqGemmPlan.contiguous(MmqConfig.PACKED_DOT, GGMLType.Q4_K, 64, 256, 0));
        expectInvalid("shorter than the", () -> MmqGemmPlan.of(MmqConfig.PACKED_DOT, GGMLType.Q8_0, 8, 64, 2, 3, 0, 8, 0));
        expectInvalid("shorter than M", () -> MmqGemmPlan.of(MmqConfig.PACKED_DOT, GGMLType.Q8_0, 8, 64, 2, 64, 0, 7, 0));
        expectInvalid("offsets must be >= 0", () -> MmqGemmPlan.of(MmqConfig.PACKED_DOT, GGMLType.Q8_0, 8, 64, 2, 64, -1, 8, 0));
        expectInvalid("no MMQ GEMM kernel", () -> MmqGemmPlan.contiguous(MmqConfig.PACKED_DOT, GGMLType.F16, 8, 64, 2));
        expectInvalid("overflow", () -> MmqGemmPlan.contiguous(MmqConfig.PACKED_DOT, GGMLType.Q4_K, Integer.MAX_VALUE, 256, 1));
    }

    @Test
    public void mmqArgumentsFollowTheDeclaredOrder() {
        MmqGemmPlan p = MmqGemmPlan.of(MmqConfig.PACKED_DOT, GGMLType.Q4_K, 128, 256, 9, 256, 5, 128, 3);
        FakeGpuBuffer w = new FakeGpuBuffer(1, p.weightBytes(), true);
        FakeGpuBuffer x = new FakeGpuBuffer(2, p.q8ElementsRequired() * Integer.BYTES, true);
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
        assertSameBuffer(w, args[GemmKernels.ARG_WEIGHT]);
        assertSameBuffer(x, args[GemmKernels.ARG_X]);
        assertSameBuffer(y, args[GemmKernels.ARG_Y]);
    }

    @Test
    public void mmqUndersizedBuffersAreRejected() {
        MmqGemmPlan p = MmqGemmPlan.contiguous(MmqConfig.PACKED_DOT, GGMLType.Q4_K, 64, 256, 4);
        FakeGpuBuffer w = new FakeGpuBuffer(1, p.weightBytes(), true);
        FakeGpuBuffer x = new FakeGpuBuffer(2, p.q8ElementsRequired() * Integer.BYTES - 4, true);
        FakeGpuBuffer y = new FakeGpuBuffer(3, p.yElementsRequired() * Float.BYTES, false);
        expectInvalid("staged activations", () -> p.args(w, x, y));
        expectInvalid("distinct buffers", () -> p.args(w, y, y));
    }

    @Test
    public void mmqDeviceLimitsAreReported() {
        DeviceCapabilities big = caps(1024, 49152, new long[]{65535, 65535, 65535});
        assertNull(MmqGemmPlan.deviceSupport(MmqConfig.PACKED_DOT, GGMLType.Q4_K, big));
        assertNull(MmqGemmPlan.contiguous(MmqConfig.PACKED_DOT, GGMLType.Q4_K, 4096, 4096, 512).checkDevice(big));

        DeviceCapabilities smallWg = caps(128, 49152, new long[]{65535, 65535, 65535});
        assertNotNull(MmqGemmPlan.deviceSupport(MmqConfig.PACKED_DOT, GGMLType.Q4_K, smallWg));

        DeviceCapabilities tinyShared = caps(1024, MmqConfig.PACKED_DOT.sharedBytes() - 1,
                new long[]{65535, 65535, 65535});
        assertNotNull(MmqGemmPlan.deviceSupport(MmqConfig.PACKED_DOT, GGMLType.Q4_K, tinyShared));
        assertNull(MmqGemmPlan.deviceSupport(MmqConfig.PACKED_DOT, GGMLType.Q4_K,
                caps(1024, MmqConfig.PACKED_DOT.sharedBytes(), new long[]{65535, 65535, 65535})));

        DeviceCapabilities fewGroups = caps(1024, 49152, new long[]{100, 65535, 65535});
        String reason = MmqGemmPlan.contiguous(MmqConfig.PACKED_DOT, GGMLType.Q4_K, 64 * 101, 256, 8).checkDevice(fewGroups);
        assertNotNull(reason);
        assertTrue(reason, reason.contains("dimension 0"));
        reason = MmqGemmPlan.contiguous(MmqConfig.PACKED_DOT, GGMLType.Q4_K, 64, 256, 33).checkDevice(
                caps(1024, 49152, new long[]{65535, 2, 65535}));
        assertNotNull(reason);
        assertTrue(reason, reason.contains("dimension 1"));
    }

    @Test
    public void mmaGeometryAndValidation() {
        MmaGemmPlan p = MmaGemmPlan.contiguous(GGMLType.F16, 4096 + 1, 4096, 100);
        assertEquals("F16_MatMul_MMA", p.kernelName());
        assertEquals(GGMLType.F16, p.type());
        assertEquals(65, p.rowTiles());
        assertEquals(4, p.tokenTiles());
        assertArrayEquals(new long[]{65L * 256, 4}, p.globalWorkSize());
        assertArrayEquals(new long[]{256, 1}, p.localWorkSize());
        assertEquals(2L * 4096, p.weightRowBytes());
        assertEquals(2L * 4096 * 4097, p.weightBytes());
        assertEquals(99L * 4096 + 4096, p.xElementsRequired());
        assertEquals(99L * 4097 + 4097, p.yElementsRequired());
        assertEquals(64, MmaGemmPlan.TILE_ROWS);
        assertEquals(32, MmaGemmPlan.TILE_TOKENS);
        assertEquals(64L * 24 + 16L * 40, MmaGemmPlan.TILE_ELEMENTS);
        assertEquals("the cooperative-matrix twin's footprint bounds the dialects",
                2L * (64 * 24 + 32 * 40) + 4L * 8 * 16 * 16, MmaGemmPlan.TILE_SHARED_BYTES);

        MmaGemmPlan b = MmaGemmPlan.contiguous(GGMLType.BF16, 64, 96, 32);
        assertEquals("BF16_MatMul_MMA", b.kernelName());
        assertEquals(GGMLType.BF16, b.type());

        MmaGemmPlan v = MmaGemmPlan.of(GGMLType.F16, 64, 96, 32, 300, 12, 200, 7);
        assertEquals(12L + 31L * 300 + 96, v.xElementsRequired());
        assertEquals(7L + 31L * 200 + 64, v.yElementsRequired());

        expectInvalid("no MMA GEMM kernel", () -> MmaGemmPlan.contiguous(GGMLType.Q8_0, 64, 64, 2));
        expectInvalid("M, K, N >= 1", () -> MmaGemmPlan.contiguous(GGMLType.F16, 64, 64, 0));
        expectInvalid("shorter than K", () -> MmaGemmPlan.of(GGMLType.F16, 8, 64, 2, 63, 0, 8, 0));
        expectInvalid("shorter than M", () -> MmaGemmPlan.of(GGMLType.F16, 8, 64, 2, 64, 0, 7, 0));
        expectInvalid("offsets must be >= 0", () -> MmaGemmPlan.of(GGMLType.F16, 8, 64, 2, 64, -1, 8, 0));
        expectInvalid("overflow", () -> MmaGemmPlan.contiguous(GGMLType.F16, Integer.MAX_VALUE, 64, 1));
    }

    @Test
    public void mmaArgumentsAndDeviceLimits() {
        DeviceCapabilities big = caps(1024, 49152, new long[]{65535, 65535, 65535});
        assertNull(MmaGemmPlan.deviceSupport(big));
        assertNull(MmaGemmPlan.contiguous(GGMLType.F16, 4096, 4096, 512).checkDevice(big));

        DeviceCapabilities smallWg = caps(128, 49152, new long[]{65535, 65535, 65535});
        assertNotNull(MmaGemmPlan.deviceSupport(smallWg));
        DeviceCapabilities tinyShared = caps(1024, MmaGemmPlan.TILE_SHARED_BYTES - 1,
                new long[]{65535, 65535, 65535});
        assertNotNull(MmaGemmPlan.deviceSupport(tinyShared));

        MmaGemmPlan p = MmaGemmPlan.of(GGMLType.F16, 128, 64, 9, 64, 5, 128, 3);
        FakeGpuBuffer w = new FakeGpuBuffer(1, p.weightBytes(), true);
        FakeGpuBuffer x = new FakeGpuBuffer(2, p.xElementsRequired() * Float.BYTES, true);
        FakeGpuBuffer y = new FakeGpuBuffer(3, p.yElementsRequired() * Float.BYTES, false);
        Object[] args = p.args(w, x, y);
        assertEquals(GemmKernels.MATMUL_TILED_ARG_COUNT, args.length);
        assertEquals(128, args[GemmKernels.ARG_M]);
        assertEquals(64, args[GemmKernels.ARG_K]);
        assertSameBuffer(w, args[GemmKernels.ARG_WEIGHT]);
        assertSameBuffer(x, args[GemmKernels.ARG_X]);
        assertSameBuffer(y, args[GemmKernels.ARG_Y]);
        expectInvalid("activation buffer", () -> p.args(w,
                new FakeGpuBuffer(2, p.xElementsRequired() * Float.BYTES - 4, true), y));
        expectInvalid("distinct buffers", () -> p.args(w, y, y));
    }

    private static void assertSameBuffer(Object expected, Object actual) {
        assertTrue("expected the same buffer instance", expected == actual);
    }

    private static Set<String> instantiated(String resource, String regex) throws IOException {
        Matcher m = Pattern.compile(regex).matcher(learnResource(resource));
        Set<String> out = new TreeSet<>();
        while (m.find()) {
            assertTrue("duplicate instantiation " + m.group(1), out.add(m.group(1)));
        }
        return out;
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

    private static String learnResource(String path) throws IOException {
        try (InputStream in = MmqGemmPlanTest.class.getClassLoader().getResourceAsStream(path)) {
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

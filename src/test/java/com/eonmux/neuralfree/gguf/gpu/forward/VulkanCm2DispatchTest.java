package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities;
import com.eonmux.neuralfree.acceleration.backend.DeviceClass;
import com.eonmux.neuralfree.acceleration.backend.DeviceFeature;
import com.eonmux.neuralfree.acceleration.backend.DeviceIdentity;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuPlaceholders;
import com.eonmux.neuralfree.gguf.ops.gemm.Cm2GemmPlan;
import com.eonmux.neuralfree.gguf.ops.gemm.GemmKernels;
import com.eonmux.neuralfree.gguf.ops.gemm.MmqMode;
import com.eonmux.neuralfree.gguf.testutil.FakeComputeBackend;
import com.eonmux.neuralfree.gguf.testutil.FakeDispatch;
import org.junit.After;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotSame;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;

/**
 * The F16 projections of {@link GpuOps#matmul} on a faked Vulkan NVIDIA device: with
 * {@link DeviceFeature#COOPERATIVE_MATRIX_2} the dense weight takes the workgroup-scope {@code F16_MatMul_CM2}
 * GEMM over the staged fp16 rows, independent of the quantized-weight switch ({@link MmqMode}); a reduction the
 * CM2 rows cannot align (K not a multiple of {@link Cm2GemmPlan#DENSE_K_MULTIPLE}) and a device without the
 * extension keep the KHR {@code F16_MatMul_MMA} GEMM. Projections of the rows of a {@link GpuOps.SharedQ8} scope
 * stage the fp16 rows once.
 */
public class VulkanCm2DispatchTest {

    private static final int ROWS = 96;
    private static final int TOKENS = 8;
    private static final int UBATCH = 16;

    private FakeComputeBackend backend;
    private GpuScratch scratch;
    private GpuPlaceholders placeholders;
    private GpuOps ops;

    @After
    public void tearDown() {
        System.clearProperty(MmqMode.PROPERTY);
        close();
    }

    private void close() {
        if (ops != null) {
            ops.close();
        }
        if (placeholders != null) {
            placeholders.close();
        }
        if (scratch != null) {
            scratch.close();
        }
        if (backend != null) {
            backend.close();
        }
        ops = null;
        placeholders = null;
        scratch = null;
        backend = null;
    }

    private void open(boolean cm2, int cols) {
        open(cm2, false, cols);
    }

    private void open(boolean cm2, boolean decodeVector, int cols) {
        open(cm2, decodeVector, cols, UBATCH);
    }

    private void open(boolean cm2, boolean decodeVector, int cols, int ubatch, String... unavailable) {
        close();
        backend = FakeComputeBackend.builder().backendType(ComputeBackend.BackendType.VULKAN).strictLlmArgs(true)
                .capabilities(vulkanNvidia(cm2, decodeVector)).unavailableKernels(unavailable).build();
        scratch = GpuScratch.builder(backend, ubatch).slot(ScratchSlots.X, (long) ubatch * cols).build();
        placeholders = GpuPlaceholders.allocate(backend, backend.computeQueue());
        ops = new GpuOps(backend.computeQueue(), LlmKernels.open(backend), scratch, placeholders);
    }

    /** An RTX-class Vulkan device: the KHR cooperative matrices, and the NV workgroup-scope ones and their
     *  four-element decode callbacks when asked. */
    private static DeviceCapabilities vulkanNvidia(boolean cm2, boolean decodeVector) {
        DeviceCapabilities.Builder b = new DeviceCapabilities.Builder()
                .deviceName("Cm2Device")
                .totalVramBytes(8L << 30)
                .maxAllocSizeBytes(2L << 30)
                .computeUnits(60)
                .maxClockFrequencyMhz(1500)
                .maxWorkGroupSize(1024)
                .driverVersion("fake-1.0")
                .supportsFloat16(true)
                .supportsSubgroups(true)
                .maxLocalMemoryBytes(48L * 1024)
                .feature(DeviceFeature.COOPERATIVE_MATRIX)
                .feature(DeviceFeature.STORAGE_8BIT).feature(DeviceFeature.STORAGE_16BIT)
                .feature(DeviceFeature.TIMELINE_SEMAPHORE)
                .deviceClass(DeviceClass.DISCRETE)
                .identity(new DeviceIdentity("Cm2Device", FakeComputeBackend.NVIDIA_VENDOR_ID, null, null));
        if (cm2) {
            b.feature(DeviceFeature.COOPERATIVE_MATRIX_2);
        }
        if (decodeVector) {
            b.feature(DeviceFeature.COOPERATIVE_MATRIX_2_DECODE_VECTOR);
        }
        return b.build();
    }

    private GpuWeight f16Weight(String name, int cols) {
        return GpuWeight.of(name, GGMLType.F16, new long[]{cols, ROWS},
                backend.allocateBuffer((long) ROWS * cols * Short.BYTES, true));
    }

    private List<String> project(int cols) {
        return project(f16Weight("blk.0.attn_q.weight", cols), cols, TOKENS);
    }

    private List<String> project(GpuWeight w, int cols, int tokens) {
        return names(projectDispatches(w, cols, tokens));
    }

    private List<FakeDispatch> projectDispatches(GpuWeight w, int cols, int tokens) {
        GpuBuffer in = backend.allocateBuffer((long) tokens * cols * Float.BYTES);
        GpuBuffer out = backend.allocateBuffer((long) tokens * GpuOps.rows(w) * Float.BYTES);
        backend.clearLog();
        ops.matmul(w, in, 0, out, 0, tokens);
        return backend.dispatches();
    }

    private GpuWeight q8Weight(String name, int rows, int cols) {
        return GpuWeight.of(name, GGMLType.Q8_0, new long[]{cols, rows},
                backend.allocateBuffer((long) rows * (cols / 32) * 34, true));
    }

    private static List<String> names(List<FakeDispatch> ds) {
        List<String> names = new ArrayList<>();
        for (FakeDispatch d : ds) {
            names.add(d.kernelName());
        }
        return names;
    }

    /**
     * The projections of a scope's rows read one staging of them, in the scope's own slot; a projection of
     * other rows inside the scope stages its own, so it cannot overwrite the shared rows.
     */
    @Test
    public void theProjectionsOfASharedScopeStageTheirRowsOnce() {
        int cols = 256;
        open(true, cols);
        GpuWeight q = f16Weight("blk.0.attn_q.weight", cols);
        GpuWeight k = f16Weight("blk.0.attn_k.weight", cols);
        GpuWeight v = f16Weight("blk.0.attn_v.weight", cols);
        GpuBuffer in = backend.allocateBuffer((long) TOKENS * cols * Float.BYTES);
        GpuBuffer other = backend.allocateBuffer((long) TOKENS * cols * Float.BYTES);
        GpuBuffer qOut = backend.allocateBuffer((long) TOKENS * ROWS * Float.BYTES);
        GpuBuffer kOut = backend.allocateBuffer((long) TOKENS * ROWS * Float.BYTES);
        GpuBuffer vOut = backend.allocateBuffer((long) TOKENS * ROWS * Float.BYTES);
        backend.clearLog();
        try (GpuOps.SharedQ8 ignored = ops.shareQ8Activations(in, 0, cols, TOKENS)) {
            ops.matmul(q, in, 0, qOut, 0, TOKENS);
            ops.matmul(k, other, 0, kOut, 0, TOKENS);
            ops.matmul(v, in, 0, vOut, 0, TOKENS);
        }
        List<FakeDispatch> ds = backend.dispatches();
        assertEquals(Arrays.asList(Cm2GemmPlan.STAGE_KERNEL, "F16_MatMul_CM2", Cm2GemmPlan.STAGE_KERNEL,
                "F16_MatMul_CM2", "F16_MatMul_CM2"), names(ds));
        Object shared = ds.get(0).args().get(ds.get(0).argCount() - 1);
        Object own = ds.get(2).args().get(ds.get(2).argCount() - 1);
        assertSame("Q reads the shared rows", shared, ds.get(1).bufferArg(GemmKernels.ARG_X));
        assertSame("the other rows read their own staging", own, ds.get(3).bufferArg(GemmKernels.ARG_X));
        assertSame("V reads the shared rows staged for Q", shared, ds.get(4).bufferArg(GemmKernels.ARG_X));
        assertNotSame("the shared rows sit apart from the per-projection slot", shared, own);
    }

    @Test
    public void f16WeightsTakeTheCm2GemmOverStagedRows() {
        open(true, 256);
        assertEquals(Arrays.asList(Cm2GemmPlan.STAGE_KERNEL, "F16_MatMul_CM2"), project(256));
    }

    @Test
    public void theQuantizedWeightSwitchLeavesTheF16Cm2GemmOn() {
        System.setProperty(MmqMode.PROPERTY, "off");
        open(true, 256);
        assertEquals(Arrays.asList(Cm2GemmPlan.STAGE_KERNEL, "F16_MatMul_CM2"), project(256));
    }

    @Test
    public void aReductionTheCm2RowsCannotAlignKeepsTheMmaGemm() {
        int cols = 252;
        assertFalse(Cm2GemmPlan.denseShapeServes(cols));
        open(true, cols);
        assertEquals(Arrays.asList("F16_MatMul_MMA"), project(cols));
    }

    /** A quantized weight takes the four-element decode form where the device reports decode vectors, and the
     *  per-element form elsewhere; the dense F16 GEMM has no decode callback and keeps its one form. */
    @Test
    public void quantizedWeightsTakeTheDecodeVectorFormWhereTheDeviceReportsIt() {
        int cols = 256;
        open(true, true, cols);
        assertEquals(Arrays.asList(Cm2GemmPlan.STAGE_KERNEL, "Q8_0_MatMul_CM2V"),
                project(q8Weight("blk.0.attn_q.weight", ROWS, cols), cols, UBATCH));
        assertEquals(Arrays.asList(Cm2GemmPlan.STAGE_KERNEL, "F16_MatMul_CM2"), project(cols));
        open(true, false, cols);
        assertEquals(Arrays.asList(Cm2GemmPlan.STAGE_KERNEL, "Q8_0_MatMul_CM2"),
                project(q8Weight("blk.0.attn_q.weight", ROWS, cols), cols, UBATCH));
    }

    /** A device that reports decode vectors but cannot build the {@code _CM2V} pipeline keeps the cooperative-matrix2
     *  lane on the per-element form, which computes the same values, for the whole and the split launch alike. */
    @Test
    public void aDecodeVectorKernelThatCannotBuildFallsBackToThePerElementForm() {
        open(true, true, 4096, UBATCH, "Q8_0_MatMul_CM2V");
        assertEquals(Arrays.asList(Cm2GemmPlan.STAGE_KERNEL, "Q8_0_MatMul_CM2"),
                project(q8Weight("blk.0.attn_q.weight", ROWS, 256), 256, UBATCH));
        assertEquals(Arrays.asList(Cm2GemmPlan.STAGE_KERNEL, "Q8_0_MatMul_CM2_KSplit", GemmKernels.SPLIT_K_REDUCE),
                project(q8Weight("blk.0.ffn_down.weight", ROWS, 4096), 4096, UBATCH));
    }

    /** A tall-skinny reduction splits through the split-K twin of the form the device runs. */
    @Test
    public void theSplitLaunchRunsTheTwinOfTheDecodeVectorForm() {
        int cols = 4096;
        open(true, true, cols);
        assertEquals(Arrays.asList(Cm2GemmPlan.STAGE_KERNEL, "Q8_0_MatMul_CM2V_KSplit", GemmKernels.SPLIT_K_REDUCE),
                project(q8Weight("blk.0.ffn_down.weight", ROWS, cols), cols, UBATCH));
        open(true, false, cols);
        assertEquals(Arrays.asList(Cm2GemmPlan.STAGE_KERNEL, "Q8_0_MatMul_CM2_KSplit", GemmKernels.SPLIT_K_REDUCE),
                project(q8Weight("blk.0.ffn_down.weight", ROWS, cols), cols, UBATCH));
    }

    /**
     * On the 60-unit device a 512-token launch over 1024 rows (16 tiles of 256 tokens, at most half the device) whose
     * 1024-element reduction split-K keeps whole runs the 128-token tile, over twice the token groups; 2048 rows
     * (32 tiles) keep the 256-token tile, and a 2048-element reduction of 1024 rows splits on the 256-token tile's
     * twin instead.
     */
    @Test
    public void aSmallGridThatDoesNotSplitRunsTheMediumTile() {
        int tokens = 512;
        open(true, true, 2048, tokens);
        List<FakeDispatch> medium = projectDispatches(q8Weight("blk.0.attn_k.weight", 1024, 1024), 1024, tokens);
        assertEquals(Arrays.asList(Cm2GemmPlan.STAGE_KERNEL, "Q8_0_MatMul_CM2V_N128"), names(medium));
        assertEquals(tokens / Cm2GemmPlan.MEDIUM_TILE_TOKENS, medium.get(1).global()[1]);
        assertEquals(Arrays.asList(Cm2GemmPlan.STAGE_KERNEL, "Q8_0_MatMul_CM2V"),
                project(q8Weight("blk.0.attn_q.weight", 2048, 1024), 1024, tokens));
        assertEquals(Arrays.asList(Cm2GemmPlan.STAGE_KERNEL, "Q8_0_MatMul_CM2V_KSplit", GemmKernels.SPLIT_K_REDUCE),
                project(q8Weight("blk.0.attn_output.weight", 1024, 2048), 2048, tokens));
        assertEquals(Arrays.asList(Cm2GemmPlan.STAGE_KERNEL, "F16_MatMul_CM2_N128"),
                project(GpuWeight.of("blk.0.attn_v.weight", GGMLType.F16, new long[]{1024, 1024},
                        backend.allocateBuffer(1024L * 1024 * Short.BYTES, true)), 1024, tokens));
    }

    @Test
    public void prewarmPreparesTheDecodeVectorPair() {
        open(true, true, 256);
        ops.prewarmWeightKernels(GGMLType.Q8_0, true, false);
        List<String> prepared = backend.preparedKernels();
        for (String kernel : new String[]{"Q8_0_MatMul_CM2V", "Q8_0_MatMul_CM2V_KSplit", "Q8_0_MatMul_CM2V_N128",
                Cm2GemmPlan.STAGE_KERNEL}) {
            assertTrue(prepared.toString(), prepared.contains(kernel));
        }
        assertFalse(prepared.toString(), prepared.contains("Q8_0_MatMul_CM2"));
    }

    @Test
    public void withoutCooperativeMatrix2TheMmaGemmServes() {
        open(false, 256);
        assertEquals(Arrays.asList("F16_MatMul_MMA"), project(256));
    }

    /** The MMA pair is prepared too: it serves the reductions the CM2 rows cannot align. */
    @Test
    public void prewarmPreparesTheCm2PairItsStagingAndTheMmaFallback() {
        open(true, 256);
        ops.prewarmWeightKernels(GGMLType.F16, true, false);
        List<String> prepared = backend.preparedKernels();
        for (String kernel : new String[]{"F16_MatMul_CM2", "F16_MatMul_CM2_KSplit", Cm2GemmPlan.STAGE_KERNEL,
                "F16_MatMul_MMA", "F16_MatMul_MMA_KSplit"}) {
            assertTrue(prepared.toString(), prepared.contains(kernel));
        }
    }
}

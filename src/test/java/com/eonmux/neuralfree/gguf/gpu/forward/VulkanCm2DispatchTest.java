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
        close();
        backend = FakeComputeBackend.builder().backendType(ComputeBackend.BackendType.VULKAN).strictLlmArgs(true)
                .capabilities(vulkanNvidia(cm2)).build();
        scratch = GpuScratch.builder(backend, UBATCH).slot(ScratchSlots.X, (long) UBATCH * cols).build();
        placeholders = GpuPlaceholders.allocate(backend, backend.computeQueue());
        ops = new GpuOps(backend.computeQueue(), LlmKernels.open(backend), scratch, placeholders);
    }

    /** An RTX-class Vulkan device: the KHR cooperative matrices, and the NV workgroup-scope ones when asked. */
    private static DeviceCapabilities vulkanNvidia(boolean cm2) {
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
        return b.build();
    }

    private GpuWeight f16Weight(String name, int cols) {
        return GpuWeight.of(name, GGMLType.F16, new long[]{cols, ROWS},
                backend.allocateBuffer((long) ROWS * cols * Short.BYTES, true));
    }

    private List<String> project(int cols) {
        GpuWeight f16 = f16Weight("blk.0.attn_q.weight", cols);
        GpuBuffer in = backend.allocateBuffer((long) TOKENS * cols * Float.BYTES);
        GpuBuffer out = backend.allocateBuffer((long) TOKENS * ROWS * Float.BYTES);
        backend.clearLog();
        ops.matmul(f16, in, 0, out, 0, TOKENS);
        return names(backend.dispatches());
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

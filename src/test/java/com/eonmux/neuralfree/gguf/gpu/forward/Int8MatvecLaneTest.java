package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities;
import com.eonmux.neuralfree.acceleration.backend.DeviceFeature;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.arch.GluOp;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.kernels.LlmKernelConstants;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuPlaceholders;
import com.eonmux.neuralfree.gguf.spec.ActParams;
import com.eonmux.neuralfree.gguf.testutil.FakeComputeBackend;
import com.eonmux.neuralfree.gguf.testutil.FakeComputeBackend.Dispatch;
import org.junit.After;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Dispatch traces of the int8-activation (Q8_1) matvec lanes of {@link GpuOps}: with
 * {@link Int8MatvecMode#ON} a single-token projection quantizes the activation row once and launches the
 * {@code {T}_MatVec_Q8} kernel, a token group of two to {@link LlmKernelConstants#MATVEC_N_DISPATCH_MAX}
 * rows quantizes the whole group and launches the {@code {T}_MatVecN_Q8} kernel &mdash; for the legacy and
 * 4-bit types as for the K-quants and i-quants &mdash;, {@link Int8MatvecMode#OFF} keeps the
 * F32-activation matvec, and {@link Int8MatvecMode#AUTO} follows the device's packed-dot-products feature
 * and the decode evidence of {@link Int8MatvecTuning} for the backend and weight type, and a missing int8
 * kernel falls back to the float form.
 */
public class Int8MatvecLaneTest {

    private static final int COLS = 256;
    private static final int ROWS = 32;
    /**
     * Columns and rows of the weights the {@link Int8MatvecMode#AUTO} tests use: {@code AUTO_COLS *
     * AUTO_ROWS} is at least {@link Int8MatvecTuning#LANE_MIN_ELEMENTS}, so the lane is selected for a
     * type that only measured faster on the larger projections, while the rows stay below the two-row
     * thread floor so the single-row form is launched.
     */
    private static final int AUTO_COLS = 262144;
    private static final int AUTO_ROWS = 16;

    private FakeComputeBackend backend;
    private GpuScratch scratch;
    private GpuPlaceholders placeholders;
    private GpuOps ops;

    private void open(Int8MatvecMode mode, String... unavailable) {
        open(mode, null, ComputeBackend.BackendType.OPENCL, unavailable);
    }

    private void open(Int8MatvecMode mode, DeviceFeature feature, String... unavailable) {
        open(mode, feature, ComputeBackend.BackendType.OPENCL, unavailable);
    }

    private void open(Int8MatvecMode mode, DeviceFeature feature, ComputeBackend.BackendType type,
                      String... unavailable) {
        FakeComputeBackend.Builder builder = FakeComputeBackend.builder().strictLlmArgs(true)
                .backendType(type)
                .unavailableKernels(unavailable);
        if (feature != null) {
            builder.capabilities(new DeviceCapabilities.Builder()
                    .deviceName("FakeDevice")
                    .totalVramBytes(8L << 30)
                    .maxAllocSizeBytes(2L << 30)
                    .computeUnits(32)
                    .maxClockFrequencyMhz(1500)
                    .maxWorkGroupSize(1024)
                    .supportsAsyncTransfer(true)
                    .driverVersion("fake-1.0")
                    .maxLocalMemoryBytes(48L * 1024)
                    .computeCapability(120)
                    .feature(feature)
                    .build());
        }
        backend = builder.build();
        scratch = GpuScratch.builder(backend, 1).slot(ScratchSlots.X, COLS).build();
        placeholders = GpuPlaceholders.allocate(backend, backend.computeQueue());
        ops = new GpuOps(backend.computeQueue(), LlmKernels.open(backend), scratch, placeholders, mode);
    }

    @After
    public void tearDown() {
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
    }

    private GpuWeight weight(GGMLType type) {
        return weight(type, ROWS);
    }

    private GpuWeight weight(GGMLType type, int rows) {
        long rowBytes = (long) (COLS / type.getBlockSize()) * type.getTypeSize();
        return GpuWeight.of("blk.0.ffn_up.weight", type, new long[]{COLS, rows},
                backend.allocateBuffer(rows * rowBytes, true));
    }

    private GpuWeight q4k() {
        return weight(GGMLType.Q4_K);
    }

    /** A projection at or above {@link Int8MatvecTuning#LANE_MIN_ELEMENTS} that stays single-row. */
    private GpuWeight autoWeight(GGMLType type) {
        long rowBytes = (long) (AUTO_COLS / type.getBlockSize()) * type.getTypeSize();
        return GpuWeight.of("blk.0.ffn_up.weight", type, new long[]{AUTO_COLS, AUTO_ROWS},
                backend.allocateBuffer(AUTO_ROWS * rowBytes, true));
    }

    private GpuBuffer autoInput() {
        return backend.allocateBuffer((long) AUTO_COLS * 4);
    }

    private GpuBuffer autoOutput() {
        return backend.allocateBuffer((long) AUTO_ROWS * 4);
    }

    private List<String> run(Runnable r) {
        backend.clearLog();
        r.run();
        List<String> names = new ArrayList<>();
        for (Dispatch d : backend.dispatches()) {
            names.add(d.kernelName());
        }
        return names;
    }

    @Test
    public void configuredModeReadsThePropertyAndRefusesUnknownValues() {
        String previous = System.getProperty(Int8MatvecMode.PROPERTY);
        try {
            System.clearProperty(Int8MatvecMode.PROPERTY);
            assertEquals(Int8MatvecMode.AUTO, Int8MatvecMode.configured());
            System.setProperty(Int8MatvecMode.PROPERTY, "on");
            assertEquals(Int8MatvecMode.ON, Int8MatvecMode.configured());
            System.setProperty(Int8MatvecMode.PROPERTY, " OFF ");
            assertEquals(Int8MatvecMode.OFF, Int8MatvecMode.configured());
            System.setProperty(Int8MatvecMode.PROPERTY, "sometimes");
            IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                    Int8MatvecMode::configured);
            assertTrue(e.getMessage(), e.getMessage().contains(Int8MatvecMode.PROPERTY));
            assertTrue(e.getMessage(), e.getMessage().contains("sometimes"));
        } finally {
            if (previous == null) {
                System.clearProperty(Int8MatvecMode.PROPERTY);
            } else {
                System.setProperty(Int8MatvecMode.PROPERTY, previous);
            }
        }
    }

    @Test
    public void modeOnQuantizesTheActivationAndUsesTheInt8Matvec() {
        open(Int8MatvecMode.ON);
        GpuWeight w = q4k();
        List<String> names = run(() -> ops.matvec(w, backend.allocateBuffer((long) COLS * 4), 0,
                backend.allocateBuffer((long) ROWS * 4), 0, 1));
        assertTrue(names.toString(), names.contains(GpuOps.QUANTIZE_Q8_1));
        assertTrue(names.toString(), names.contains("Q4_K_MatVec_Q8"));
        assertFalse(names.toString(), names.contains("Q4_K_MatVec"));
    }

    @Test
    public void modeOnUsesTheInt8AccumulatingMatvec() {
        open(Int8MatvecMode.ON);
        GpuWeight w = q4k();
        List<String> names = run(() -> ops.matvecAdd(w, backend.allocateBuffer((long) COLS * 4), 0,
                backend.allocateBuffer((long) ROWS * 4), 0, 1));
        assertTrue(names.toString(), names.contains(GpuOps.QUANTIZE_Q8_1));
        assertTrue(names.toString(), names.contains("Q4_K_MatVecAdd_Q8"));
        assertFalse(names.toString(), names.contains("Q4_K_MatVecAdd"));
    }

    @Test
    public void modeOffKeepsTheFloatMatvec() {
        open(Int8MatvecMode.OFF);
        GpuWeight w = q4k();
        List<String> names = run(() -> ops.matvec(w, backend.allocateBuffer((long) COLS * 4), 0,
                backend.allocateBuffer((long) ROWS * 4), 0, 1));
        assertTrue(names.toString(), names.contains("Q4_K_MatVec"));
        assertFalse(names.toString(), names.contains("Q4_K_MatVec_Q8"));
        assertFalse(names.toString(), names.contains(GpuOps.QUANTIZE_Q8_1));
    }

    @Test
    public void missingInt8KernelFallsBackToTheFloatMatvec() {
        open(Int8MatvecMode.ON, "Q4_K_MatVec_Q8");
        GpuWeight w = q4k();
        List<String> names = run(() -> ops.matvec(w, backend.allocateBuffer((long) COLS * 4), 0,
                backend.allocateBuffer((long) ROWS * 4), 0, 1));
        assertTrue(names.toString(), names.contains("Q4_K_MatVec"));
        assertFalse(names.toString(), names.contains("Q4_K_MatVec_Q8"));
    }

    @Test
    public void legacyAndFourBitTypesUseTheirInt8Matvec() {
        open(Int8MatvecMode.ON);
        for (GGMLType type : new GGMLType[]{GGMLType.Q4_0, GGMLType.Q4_1, GGMLType.Q5_0, GGMLType.Q5_1,
                GGMLType.Q8_0, GGMLType.MXFP4, GGMLType.NVFP4}) {
            GpuWeight w = weight(type);
            List<String> names = run(() -> ops.matvec(w, backend.allocateBuffer((long) COLS * 4), 0,
                    backend.allocateBuffer((long) ROWS * 4), 0, 1));
            assertTrue(names.toString(), names.contains(GpuOps.QUANTIZE_Q8_1));
            assertTrue(names.toString(), names.contains(type + "_MatVec_Q8"));
            assertFalse(names.toString(), names.contains(type + "_MatVec"));
        }
    }

    @Test
    public void autoUsesTheInt8MatvecWhenTheDeviceReportsPackedDotProducts() {
        open(Int8MatvecMode.AUTO, DeviceFeature.INTEGER_DOT_PRODUCT);
        GpuWeight w = autoWeight(GGMLType.Q4_1);
        List<String> names = run(() -> ops.matvec(w, autoInput(), 0, autoOutput(), 0, 1));
        assertTrue(names.toString(), names.contains("Q4_1_MatVec_Q8"));
    }

    @Test
    public void autoKeepsTheFloatMatvecWithoutPackedDotProducts() {
        open(Int8MatvecMode.AUTO);
        GpuWeight w = weight(GGMLType.Q4_0);
        List<String> names = run(() -> ops.matvec(w, backend.allocateBuffer((long) COLS * 4), 0,
                backend.allocateBuffer((long) ROWS * 4), 0, 1));
        assertTrue(names.toString(), names.contains("Q4_0_MatVec"));
        assertFalse(names.toString(), names.contains("Q4_0_MatVec_Q8"));
        assertFalse(names.toString(), names.contains(GpuOps.QUANTIZE_Q8_1));
    }

    @Test
    public void autoKeepsTheFloatMatvecForATypeThatMeasuredSlowerOnTheBackend() {
        open(Int8MatvecMode.AUTO, DeviceFeature.INTEGER_DOT_PRODUCT);
        // OpenCL MXFP4 measured at most 1.00x with the lane at every projection shape.
        GpuWeight w = autoWeight(GGMLType.MXFP4);
        List<String> names = run(() -> ops.matvec(w, autoInput(), 0, autoOutput(), 0, 1));
        assertTrue(names.toString(), names.contains("MXFP4_MatVec"));
        assertFalse(names.toString(), names.contains("MXFP4_MatVec_Q8"));
        assertFalse(names.toString(), names.contains(GpuOps.QUANTIZE_Q8_1));
    }

    @Test
    public void autoKeepsTheFloatMatvecForAProjectionBelowTheMeasuredLaneThreshold() {
        open(Int8MatvecMode.AUTO, DeviceFeature.INTEGER_DOT_PRODUCT, ComputeBackend.BackendType.CUDA);
        // Q4_K measured slower with the lane at 1024x1024 elements, faster from 4096x4096 up.
        GpuWeight w = q4k();
        List<String> names = run(() -> ops.matvec(w, backend.allocateBuffer((long) COLS * 4), 0,
                backend.allocateBuffer((long) ROWS * 4), 0, 1));
        assertTrue(names.toString(), names.contains("Q4_K_MatVec"));
        assertFalse(names.toString(), names.contains("Q4_K_MatVec_Q8"));
        assertFalse(names.toString(), names.contains(GpuOps.QUANTIZE_Q8_1));
    }

    @Test
    public void autoUsesTheInt8MatvecForATypeWithoutAMeasuredLoss() {
        open(Int8MatvecMode.AUTO, DeviceFeature.INTEGER_DOT_PRODUCT, ComputeBackend.BackendType.CUDA);
        GpuWeight w = autoWeight(GGMLType.Q4_K);
        List<String> names = run(() -> ops.matvec(w, autoInput(), 0, autoOutput(), 0, 1));
        assertTrue(names.toString(), names.contains("Q4_K_MatVec_Q8"));
        assertFalse(names.toString(), names.contains("Q4_K_MatVec"));
    }

    @Test
    public void autoTakesTheQ8_0LaneOnALargeCudaProjectionAndNotASmallOne() {
        open(Int8MatvecMode.AUTO, DeviceFeature.INTEGER_DOT_PRODUCT, ComputeBackend.BackendType.CUDA);
        // 32x256 measured slower with the lane: the F32 matvec keeps it.
        GpuWeight small = weight(GGMLType.Q8_0);
        List<String> smallNames = run(() -> ops.matvec(small, backend.allocateBuffer((long) COLS * 4), 0,
                backend.allocateBuffer((long) ROWS * 4), 0, 1));
        assertTrue(smallNames.toString(), smallNames.contains("Q8_0_MatVec"));
        assertFalse(smallNames.toString(), smallNames.contains("Q8_0_MatVec_Q8"));
        assertFalse(smallNames.toString(), smallNames.contains(GpuOps.QUANTIZE_Q8_1));
        // A projection at or above the measured lane threshold takes the lane.
        GpuWeight large = autoWeight(GGMLType.Q8_0);
        List<String> largeNames = run(() -> ops.matvec(large, autoInput(), 0, autoOutput(), 0, 1));
        assertTrue(largeNames.toString(), largeNames.contains("Q8_0_MatVec_Q8"));
        assertFalse(largeNames.toString(), largeNames.contains("Q8_0_MatVec"));
    }

    @Test
    public void offNeverRunsTheQ8_0Lane() {
        open(Int8MatvecMode.OFF, DeviceFeature.INTEGER_DOT_PRODUCT, ComputeBackend.BackendType.CUDA);
        GpuWeight large = autoWeight(GGMLType.Q8_0);
        List<String> names = run(() -> ops.matvec(large, autoInput(), 0, autoOutput(), 0, 1));
        assertTrue(names.toString(), names.contains("Q8_0_MatVec"));
        assertFalse(names.toString(), names.contains("Q8_0_MatVec_Q8"));
        assertFalse(names.toString(), names.contains("Q8_0_MatVec2_Q8"));
        assertFalse(names.toString(), names.contains(GpuOps.QUANTIZE_Q8_1));
    }

    @Test
    public void tuningKeepsTheLaneOnlyForTypesWithoutAMeasuredLoss() {
        // Measured slower with the lane at every projection shape, so never selected.
        assertFalse(Int8MatvecTuning.laneIsFaster(ComputeBackend.BackendType.OPENCL, GGMLType.MXFP4));
        assertFalse(Int8MatvecTuning.laneIsFaster(ComputeBackend.BackendType.OPENCL, GGMLType.NVFP4));
        assertFalse(Int8MatvecTuning.laneIsFaster(ComputeBackend.BackendType.OPENCL, GGMLType.IQ4_XS));
        assertFalse(Int8MatvecTuning.laneIsFaster(ComputeBackend.BackendType.CUDA, GGMLType.Q4_0));
        assertFalse(Int8MatvecTuning.laneIsFaster(ComputeBackend.BackendType.CUDA, GGMLType.Q4_1));
        assertFalse(Int8MatvecTuning.laneIsFaster(ComputeBackend.BackendType.CUDA, GGMLType.Q5_0));
        assertFalse(Int8MatvecTuning.laneIsFaster(ComputeBackend.BackendType.CUDA, GGMLType.Q5_1));
        assertFalse(Int8MatvecTuning.laneIsFaster(ComputeBackend.BackendType.CUDA, GGMLType.IQ4_NL));
        assertFalse(Int8MatvecTuning.laneIsFaster(ComputeBackend.BackendType.VULKAN, GGMLType.Q4_0));
        // Measured faster on the larger projections only.
        assertTrue(Int8MatvecTuning.laneIsFaster(ComputeBackend.BackendType.OPENCL, GGMLType.Q4_0));
        assertTrue(Int8MatvecTuning.laneIsFaster(ComputeBackend.BackendType.OPENCL, GGMLType.Q8_0));
        assertTrue(Int8MatvecTuning.laneIsFaster(ComputeBackend.BackendType.OPENCL, GGMLType.Q4_K));
        assertTrue(Int8MatvecTuning.laneIsFaster(ComputeBackend.BackendType.CUDA, GGMLType.Q4_K));
        assertTrue(Int8MatvecTuning.laneIsFaster(ComputeBackend.BackendType.CUDA, GGMLType.Q8_0));
        // No measured loss on these backends: the device feature decides.
        assertTrue(Int8MatvecTuning.laneIsFaster(ComputeBackend.BackendType.VULKAN, GGMLType.Q4_1));
        assertTrue(Int8MatvecTuning.laneIsFaster(ComputeBackend.BackendType.VULKAN, GGMLType.Q4_K));
        assertTrue(Int8MatvecTuning.laneIsFaster(ComputeBackend.BackendType.VULKAN, GGMLType.Q8_0));
        assertTrue(Int8MatvecTuning.laneIsFaster(ComputeBackend.BackendType.HIP, GGMLType.Q4_K));
    }

    @Test
    public void tuningGatesTheSmallProjectionOfALargeOnlyType() {
        // 1024x1024 and 3072x1024 measured slower with the lane; 4096x4096 and up measured faster.
        assertFalse(Int8MatvecTuning.laneIsFaster(ComputeBackend.BackendType.CUDA, GGMLType.Q4_K,
                1024L * 1024));
        assertFalse(Int8MatvecTuning.laneIsFaster(ComputeBackend.BackendType.CUDA, GGMLType.Q6_K,
                3072L * 1024));
        assertTrue(Int8MatvecTuning.laneIsFaster(ComputeBackend.BackendType.CUDA, GGMLType.Q4_K,
                Int8MatvecTuning.LANE_MIN_ELEMENTS));
        assertTrue(Int8MatvecTuning.laneIsFaster(ComputeBackend.BackendType.CUDA, GGMLType.Q4_K,
                4096L * 4096));
        // Q8_0 lost the lane's fixed quantizer cost on the small projections of both backends and
        // measured 1.06x to 1.69x with it from 4096x4096 up on CUDA, 1.11x to 1.36x on OpenCL.
        assertFalse(Int8MatvecTuning.laneIsFaster(ComputeBackend.BackendType.CUDA, GGMLType.Q8_0,
                3072L * 1024));
        assertTrue(Int8MatvecTuning.laneIsFaster(ComputeBackend.BackendType.CUDA, GGMLType.Q8_0,
                4096L * 4096));
        assertFalse(Int8MatvecTuning.laneIsFaster(ComputeBackend.BackendType.OPENCL, GGMLType.Q8_0,
                1024L * 1024));
        assertTrue(Int8MatvecTuning.laneIsFaster(ComputeBackend.BackendType.OPENCL, GGMLType.Q8_0,
                9728L * 2560));
        // The always-slower set stays out at every size.
        assertFalse(Int8MatvecTuning.laneIsFaster(ComputeBackend.BackendType.CUDA, GGMLType.Q4_0,
                Long.MAX_VALUE));
        assertFalse(Int8MatvecTuning.laneIsFaster(ComputeBackend.BackendType.OPENCL, GGMLType.MXFP4,
                Long.MAX_VALUE));
        assertFalse(Int8MatvecTuning.laneIsFaster(ComputeBackend.BackendType.VULKAN, GGMLType.Q4_0,
                Long.MAX_VALUE));
        // A backend and type without evidence keep the lane at any size.
        assertTrue(Int8MatvecTuning.laneIsFaster(ComputeBackend.BackendType.VULKAN, GGMLType.Q4_K, 1));
    }

    @Test
    public void tuningPairsOnlyTheTypesThatMeasuredFasterPaired() {
        // Q4_K and Q5_K lost 1-8% paired on CUDA (medians 0.95x and 0.96x); Q8_K and Q8_0 measured
        // neutral there (Q8_0 medians 0.99x) and Q8_0 never gained on OpenCL either (median 0.97x);
        // Q8_0's paired form measured 1.5-1.9x on the Vulkan device.
        assertFalse(Int8MatvecTuning.pairedIsFaster(ComputeBackend.BackendType.CUDA, GGMLType.Q4_K));
        assertFalse(Int8MatvecTuning.pairedIsFaster(ComputeBackend.BackendType.CUDA, GGMLType.Q5_K));
        assertTrue(Int8MatvecTuning.pairedIsFaster(ComputeBackend.BackendType.CUDA, GGMLType.Q6_K));
        assertFalse(Int8MatvecTuning.pairedIsFaster(ComputeBackend.BackendType.CUDA, GGMLType.Q8_K));
        assertFalse(Int8MatvecTuning.pairedIsFaster(ComputeBackend.BackendType.CUDA, GGMLType.Q8_0));
        assertTrue(Int8MatvecTuning.pairedIsFaster(ComputeBackend.BackendType.OPENCL, GGMLType.Q4_K));
        assertTrue(Int8MatvecTuning.pairedIsFaster(ComputeBackend.BackendType.OPENCL, GGMLType.Q6_K));
        assertFalse(Int8MatvecTuning.pairedIsFaster(ComputeBackend.BackendType.OPENCL, GGMLType.Q8_K));
        assertFalse(Int8MatvecTuning.pairedIsFaster(ComputeBackend.BackendType.OPENCL, GGMLType.Q8_0));
        assertTrue(Int8MatvecTuning.pairedIsFaster(ComputeBackend.BackendType.VULKAN, GGMLType.Q8_0));
        assertTrue(Int8MatvecTuning.pairedIsFaster(ComputeBackend.BackendType.VULKAN, GGMLType.Q8_K));
        assertTrue(Int8MatvecTuning.pairedIsFaster(ComputeBackend.BackendType.HIP, GGMLType.Q8_K));
        assertTrue(Int8MatvecTuning.pairedIsFaster(ComputeBackend.BackendType.CUDA, GGMLType.Q2_K));
    }

    @Test
    public void measuredPairingLossKeepsTheSingleRowInt8Matvec() {
        open(Int8MatvecMode.ON, DeviceFeature.INTEGER_DOT_PRODUCT, ComputeBackend.BackendType.CUDA);
        int rows = 1024;
        GpuWeight w = weight(GGMLType.Q8_K, rows);
        List<String> names = run(() -> ops.matvec(w, backend.allocateBuffer((long) COLS * 4), 0,
                backend.allocateBuffer((long) rows * 4), 0, 1));
        assertTrue(names.toString(), names.contains("Q8_K_MatVec_Q8"));
        assertFalse(names.toString(), names.contains("Q8_K_MatVec2_Q8"));
    }

    @Test
    public void measuredQ4KPairingLossOnCudaKeepsTheSingleRowInt8Matvec() {
        open(Int8MatvecMode.ON, DeviceFeature.INTEGER_DOT_PRODUCT, ComputeBackend.BackendType.CUDA);
        int rows = 1024;
        GpuWeight w = weight(GGMLType.Q4_K, rows);
        List<String> names = run(() -> ops.matvec(w, backend.allocateBuffer((long) COLS * 4), 0,
                backend.allocateBuffer((long) rows * 4), 0, 1));
        assertTrue(names.toString(), names.contains("Q4_K_MatVec_Q8"));
        assertFalse(names.toString(), names.contains("Q4_K_MatVec2_Q8"));
    }

    @Test
    public void measuredQ8_0PairingLossOnCudaKeepsTheSingleRowInt8Matvec() {
        open(Int8MatvecMode.ON, DeviceFeature.INTEGER_DOT_PRODUCT, ComputeBackend.BackendType.CUDA);
        int rows = 1024;
        GpuWeight w = weight(GGMLType.Q8_0, rows);
        List<String> names = run(() -> ops.matvec(w, backend.allocateBuffer((long) COLS * 4), 0,
                backend.allocateBuffer((long) rows * 4), 0, 1));
        assertTrue(names.toString(), names.contains("Q8_0_MatVec_Q8"));
        assertFalse(names.toString(), names.contains("Q8_0_MatVec2_Q8"));
    }

    @Test
    public void smallTokenGroupsQuantizeOnceAndUseTheMultiColumnInt8Matvec() {
        open(Int8MatvecMode.ON);
        GpuWeight w = q4k();
        int tokens = 4;
        List<String> names = run(() -> ops.matvec(w, backend.allocateBuffer((long) tokens * COLS * 4), 0,
                backend.allocateBuffer((long) tokens * ROWS * 4), 0, tokens));
        assertTrue(names.toString(), names.contains(GpuOps.QUANTIZE_Q8_1));
        assertTrue(names.toString(), names.contains("Q4_K_MatVecN_Q8"));
        assertFalse(names.toString(), names.contains("Q4_K_MatVec_Q8"));
        assertFalse(names.toString(), names.stream().anyMatch(n -> n.startsWith("Q4_K_BatchMatMul")));
    }

    @Test
    public void tokenGroupsAboveTheMultiColumnLaneKeepTheFloatBatchMatvec() {
        open(Int8MatvecMode.ON);
        GpuWeight w = q4k();
        int tokens = LlmKernelConstants.MATVEC_N_MAX_TOKENS + 1;
        List<String> names = run(() -> ops.matvec(w, backend.allocateBuffer((long) tokens * COLS * 4), 0,
                backend.allocateBuffer((long) tokens * ROWS * 4), 0, tokens));
        assertTrue(names.toString(), names.stream().anyMatch(n -> n.startsWith("Q4_K_BatchMatMul")));
        assertFalse(names.toString(), names.contains("Q4_K_MatVecN_Q8"));
        assertFalse(names.toString(), names.contains("Q4_K_MatVec_Q8"));
    }

    @Test
    public void modeOnPairsRowsInTheInt8TwoRowMatvecAboveTheThreadFloor() {
        open(Int8MatvecMode.ON);
        int rows = 1024;
        GpuWeight w = weight(GGMLType.Q4_K, rows);
        List<Dispatch> ds = dispatches(() -> ops.matvec(w, backend.allocateBuffer((long) COLS * 4), 0,
                backend.allocateBuffer((long) rows * 4), 0, 1));
        List<String> names = kernelNames(ds);
        assertTrue(names.toString(), names.contains("Q4_K_MatVec2_Q8"));
        assertFalse(names.toString(), names.contains("Q4_K_MatVec_Q8"));
        Dispatch mv = ds.get(names.indexOf("Q4_K_MatVec2_Q8"));
        assertArrayEquals(new long[]{(long) (rows / 2) * ops.matvecLocal()}, mv.global());
        assertArrayEquals(new long[]{ops.matvecLocal()}, mv.local());
    }

    @Test
    public void missingPairedInt8KernelFallsBackToTheSingleRowInt8Matvec() {
        open(Int8MatvecMode.ON, "Q4_K_MatVec2_Q8");
        int rows = 1024;
        GpuWeight w = weight(GGMLType.Q4_K, rows);
        List<String> names = run(() -> ops.matvec(w, backend.allocateBuffer((long) COLS * 4), 0,
                backend.allocateBuffer((long) rows * 4), 0, 1));
        assertTrue(names.toString(), names.contains("Q4_K_MatVec_Q8"));
        assertFalse(names.toString(), names.contains("Q4_K_MatVec2_Q8"));
    }

    @Test
    public void fusedGateUpPairsRowsInTheInt8TwoRowKernelAboveTheThreadFloor() {
        open(Int8MatvecMode.ON);
        int rows = 1024;
        GpuWeight gate = weight(GGMLType.Q4_K, rows);
        GpuWeight up = weight(GGMLType.Q4_K, rows);
        List<String> names = run(() -> ops.gateUpGlu(gate, up, GluOp.SWIGLU, ActParams.NONE,
                backend.allocateBuffer((long) COLS * 4), backend.allocateBuffer((long) rows * 4)));
        assertTrue(names.toString(), names.contains("Q4_K_MatVec2FusedGateUp_Q8"));
        assertFalse(names.toString(), names.contains("Q4_K_MatVecFusedGateUp_Q8"));
    }

    @Test
    public void sharedActivationsQuantizeOneRowOnceForEveryProjection() {
        open(Int8MatvecMode.ON);
        GpuWeight a = weight(GGMLType.Q4_K);
        GpuWeight b = weight(GGMLType.Q5_K);
        GpuBuffer in = backend.allocateBuffer((long) COLS * 4);
        GpuBuffer outA = backend.allocateBuffer((long) ROWS * 4);
        GpuBuffer outB = backend.allocateBuffer((long) ROWS * 4);
        backend.clearLog();
        try (GpuOps.SharedQ8 share = ops.shareQ8Activations(in, 0, COLS)) {
            ops.matvec(a, in, 0, outA, 0, 1);
            ops.matvec(b, in, 0, outB, 0, 1);
        }
        List<String> names = kernelNames(backend.dispatches());
        assertEquals(names.toString(), 1, Collections.frequency(names, GpuOps.QUANTIZE_Q8_1));
        assertEquals(names.toString(), 1, Collections.frequency(names, "Q4_K_MatVec_Q8"));
        assertEquals(names.toString(), 1, Collections.frequency(names, "Q5_K_MatVec_Q8"));
    }

    @Test
    public void theFusedGateUpAndTheAccumulatingMatvecReuseTheSharedRow() {
        open(Int8MatvecMode.ON);
        int rows = 1024;
        GpuWeight w = weight(GGMLType.Q4_K, rows);
        GpuWeight gate = weight(GGMLType.Q4_K, rows);
        GpuWeight up = weight(GGMLType.Q4_K, rows);
        GpuBuffer in = backend.allocateBuffer((long) COLS * 4);
        GpuBuffer out = backend.allocateBuffer((long) rows * 4);
        backend.clearLog();
        try (GpuOps.SharedQ8 share = ops.shareQ8Activations(in, 0, COLS)) {
            ops.matvecAdd(w, in, 0, out, 0, 1);
            ops.gateUpGlu(gate, up, GluOp.SWIGLU, ActParams.NONE, in, out);
        }
        List<String> names = kernelNames(backend.dispatches());
        assertEquals(names.toString(), 1, Collections.frequency(names, GpuOps.QUANTIZE_Q8_1));
        assertEquals(names.toString(), 1, Collections.frequency(names, "Q4_K_MatVecAdd_Q8"));
        assertEquals(names.toString(), 1, Collections.frequency(names, "Q4_K_MatVec2FusedGateUp_Q8"));
    }

    @Test
    public void projectionsOutsideTheSharedRowQuantizeTheirOwnRow() {
        open(Int8MatvecMode.ON);
        GpuWeight a = weight(GGMLType.Q4_K);
        GpuWeight b = weight(GGMLType.Q4_K);
        GpuBuffer in = backend.allocateBuffer((long) COLS * 4);
        GpuBuffer other = backend.allocateBuffer((long) COLS * 4);
        GpuBuffer out = backend.allocateBuffer((long) ROWS * 4);
        backend.clearLog();
        try (GpuOps.SharedQ8 share = ops.shareQ8Activations(in, 0, COLS)) {
            ops.matvec(a, in, 0, out, 0, 1);
            ops.matvec(b, other, 0, out, 0, 1);
        }
        List<String> names = kernelNames(backend.dispatches());
        assertEquals(names.toString(), 2, Collections.frequency(names, GpuOps.QUANTIZE_Q8_1));
    }

    @Test
    public void shareRefusesAnUnshareableRowAndASecondOpenScope() {
        open(Int8MatvecMode.ON);
        GpuBuffer in = backend.allocateBuffer((long) COLS * 4);
        assertThrows(IllegalArgumentException.class, () -> {
            try (GpuOps.SharedQ8 ignored = ops.shareQ8Activations(in, 32, COLS - 32)) {
                // the offset row cannot be handed to the quantizer
            }
        });
        assertThrows(IllegalArgumentException.class, () -> {
            try (GpuOps.SharedQ8 ignored = ops.shareQ8Activations(in, 0, COLS - 8)) {
                // a width that is not a whole number of Q8_1 blocks
            }
        });
        try (GpuOps.SharedQ8 ignored = ops.shareQ8Activations(in, 0, COLS)) {
            assertThrows(IllegalStateException.class, () -> ops.shareQ8Activations(in, 0, COLS));
        }
        assertTrue(GpuOps.shareableQ8Activations(0, COLS));
        assertFalse(GpuOps.shareableQ8Activations(32, COLS - 32));
        assertFalse(GpuOps.shareableQ8Activations(0, COLS - 8));
    }

    private List<Dispatch> dispatches(Runnable r) {
        backend.clearLog();
        r.run();
        return backend.dispatches();
    }

    private static List<String> kernelNames(List<Dispatch> ds) {
        List<String> names = new ArrayList<>();
        for (Dispatch d : ds) {
            names.add(d.kernelName());
        }
        return names;
    }
}

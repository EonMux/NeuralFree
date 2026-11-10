package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities;
import com.eonmux.neuralfree.acceleration.backend.DeviceClass;
import com.eonmux.neuralfree.acceleration.backend.DeviceFeature;
import com.eonmux.neuralfree.acceleration.backend.DeviceIdentity;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.arch.GluOp;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.kernels.LlmKernelConstants;
import com.eonmux.neuralfree.gguf.ops.gemm.GemmKernels;
import com.eonmux.neuralfree.gguf.ops.gemm.MmqMode;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuPlaceholders;
import com.eonmux.neuralfree.gguf.spec.ActParams;
import com.eonmux.neuralfree.gguf.testutil.FakeComputeBackend;
import com.eonmux.neuralfree.gguf.testutil.FakeDispatch;
import org.junit.After;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.EnumSet;
import java.util.List;
import java.util.Locale;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Dispatch traces of the int8-activation (Q8_1) matvec lanes of {@link GpuOps}: with
 * {@link Int8MatvecMode#ON} a single-token projection quantizes the activation row once and launches the
 * {@code {T}_MatVec_Q8} kernel, a token group of two to {@link LlmKernelConstants#MATVEC_N_DISPATCH_MAX}
 * rows quantizes the whole group and launches the {@code {T}_MatVecN_Q8} kernel (its exact-token-count
 * twin {@code {T}_MatVecN<n>_Q8} where the type provides one, its two-row twin where the measured pairing
 * answer selects it) &mdash; for the legacy and
 * 4-bit types as for the K-quants and i-quants &mdash;, {@link Int8MatvecMode#OFF} keeps the
 * F32-activation matvec, and {@link Int8MatvecMode#AUTO} follows the device's packed-dot-products feature
 * and the decode evidence of {@link Int8MatvecTuning} for the device kind and weight type, and a missing
 * int8 kernel falls back to the float form. The AUTO tests' fake devices claim the vendor and class of
 * the measured device per backend, so each backend's traces read that device kind's entries; further
 * fakes of the other measured kinds, and of unknown vendors and classes, show an entry governing only
 * its own kind and unknown kinds keeping the feature decision.
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
    /** A vendor id outside the measured device kinds (NVIDIA, Intel), so its kind keeps the default dispatch. */
    private static final int UNMEASURED_VENDOR_ID = 0x1ED2;

    /**
     * The CUDA modules' warp-row twins ({@link WarpRowMatvec}), which replace the family's single-token forms on the
     * shapes they serve; the tests of the family's selection on the CUDA kind withhold them.
     */
    private static final String[] CUDA_ROW_TWINS = cudaRowTwins();

    private FakeComputeBackend backend;
    private GpuScratch scratch;
    private GpuPlaceholders placeholders;
    private GpuOps ops;

    private static String[] cudaRowTwins() {
        List<String> names = new ArrayList<>();
        for (GGMLType type : WarpRowMatvec.TYPES) {
            for (String form : new String[]{"MatVec", "MatVecAdd", "MatVecFusedGateUp"}) {
                names.add(WarpRowMatvec.twinName(type.name() + "_" + form + "_Q8", type));
            }
        }
        return names.toArray(new String[0]);
    }

    private void open(Int8MatvecMode mode, String... unavailable) {
        open(mode, null, ComputeBackend.BackendType.OPENCL, unavailable);
    }

    private void open(Int8MatvecMode mode, DeviceFeature feature, String... unavailable) {
        open(mode, feature, ComputeBackend.BackendType.OPENCL, unavailable);
    }

    private void open(Int8MatvecMode mode, DeviceFeature feature, ComputeBackend.BackendType type,
                      String... unavailable) {
        open(mode, feature, type, measuredDeviceCapabilities(type, feature), unavailable);
    }

    /** Opens the backend of {@code type} claiming {@code caps} (null keeps the builder default). */
    private void open(Int8MatvecMode mode, DeviceFeature feature, ComputeBackend.BackendType type,
                      DeviceCapabilities caps, String... unavailable) {
        FakeComputeBackend.Builder builder = FakeComputeBackend.builder().strictLlmArgs(true)
                .backendType(type)
                .unavailableKernels(unavailable);
        if (caps != null) {
            builder.capabilities(caps);
        }
        backend = builder.build();
        scratch = GpuScratch.builder(backend, 1).slot(ScratchSlots.X, COLS).build();
        placeholders = GpuPlaceholders.allocate(backend, backend.computeQueue());
        ops = new GpuOps(backend.computeQueue(), LlmKernels.open(backend), scratch, placeholders, mode);
    }

    /**
     * The capabilities of the device whose measured entries {@link Int8MatvecTuning} holds for
     * {@code type}: the NVIDIA RTX 5080 Laptop dGPU over CUDA and OpenCL, the Intel Arrow-Lake iGPU
     * over Vulkan. Per-kind tests below build their own capabilities instead.
     */
    private static DeviceCapabilities measuredDeviceCapabilities(ComputeBackend.BackendType type,
                                                                 DeviceFeature feature) {
        if (feature == null) {
            return null;
        }
        return capabilities(type,
                type == ComputeBackend.BackendType.VULKAN ? FakeComputeBackend.INTEL_VENDOR_ID
                        : FakeComputeBackend.NVIDIA_VENDOR_ID,
                type == ComputeBackend.BackendType.VULKAN ? DeviceClass.INTEGRATED : DeviceClass.DISCRETE,
                feature);
    }

    /** The capabilities of {@code type}'s device claiming {@code vendorId}, {@code deviceClass} and {@code feature}. */
    private static DeviceCapabilities capabilities(ComputeBackend.BackendType type, int vendorId,
                                                   DeviceClass deviceClass, DeviceFeature feature) {
        return capabilitiesBuilder(type, vendorId, deviceClass, feature).build();
    }

    /** The builder of {@link #capabilities}, for tests that add capabilities. */
    private static DeviceCapabilities.Builder capabilitiesBuilder(ComputeBackend.BackendType type, int vendorId,
                                                                  DeviceClass deviceClass, DeviceFeature feature) {
        DeviceCapabilities.Builder b = new DeviceCapabilities.Builder()
                .deviceName("FakeDevice")
                .totalVramBytes(8L << 30)
                .maxAllocSizeBytes(2L << 30)
                .computeUnits(32)
                .maxClockFrequencyMhz(1500)
                .maxWorkGroupSize(1024)
                .supportsAsyncTransfer(true)
                .driverVersion("fake-1.0")
                .maxLocalMemoryBytes(48L * 1024)
                .feature(feature)
                .feature(DeviceFeature.STORAGE_8BIT)
                .feature(DeviceFeature.STORAGE_16BIT)
                .feature(DeviceFeature.TIMELINE_SEMAPHORE)
                .deviceClass(deviceClass)
                .identity(new DeviceIdentity("FakeDevice", vendorId, null, null));
        if (type == ComputeBackend.BackendType.CUDA) {
            b.computeCapability(120);
        }
        return b;
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
        return weight(type, rows, COLS);
    }

    private GpuWeight weight(GGMLType type, int rows, int cols) {
        long rowBytes = (long) (cols / type.getBlockSize()) * type.getTypeSize();
        return GpuWeight.of("blk.0.ffn_up.weight", type, new long[]{cols, rows},
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
        for (FakeDispatch d : backend.dispatches()) {
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
    public void autoKeepsTheFloatMatvecForATypeThatMeasuredSlowerOnTheDeviceKind() {
        open(Int8MatvecMode.AUTO, DeviceFeature.INTEGER_DOT_PRODUCT);
        // MXFP4 measured at most 1.00x with the lane at every projection shape on the RTX 5080's
        // OpenCL device, the kind the fake claims.
        GpuWeight w = autoWeight(GGMLType.MXFP4);
        List<String> names = run(() -> ops.matvec(w, autoInput(), 0, autoOutput(), 0, 1));
        assertTrue(names.toString(), names.contains("MXFP4_MatVec"));
        assertFalse(names.toString(), names.contains("MXFP4_MatVec_Q8"));
        assertFalse(names.toString(), names.contains(GpuOps.QUANTIZE_Q8_1));
    }

    @Test
    public void autoKeepsTheFloatMatvecForAProjectionBelowTheMeasuredLaneThreshold() {
        open(Int8MatvecMode.AUTO, DeviceFeature.INTEGER_DOT_PRODUCT, ComputeBackend.BackendType.CUDA,
                CUDA_ROW_TWINS);
        // Q6_K measured slower with the lane at 1024x2560 elements, faster from 4096x4096 up.
        GpuWeight w = weight(GGMLType.Q6_K);
        List<String> names = run(() -> ops.matvec(w, backend.allocateBuffer((long) COLS * 4), 0,
                backend.allocateBuffer((long) ROWS * 4), 0, 1));
        assertTrue(names.toString(), names.contains("Q6_K_MatVec"));
        assertFalse(names.toString(), names.contains("Q6_K_MatVec_Q8"));
        assertFalse(names.toString(), names.contains(GpuOps.QUANTIZE_Q8_1));
    }

    @Test
    public void autoUsesTheInt8MatvecForATypeWithoutAMeasuredLoss() {
        open(Int8MatvecMode.AUTO, DeviceFeature.INTEGER_DOT_PRODUCT, ComputeBackend.BackendType.CUDA,
                CUDA_ROW_TWINS);
        GpuWeight w = autoWeight(GGMLType.Q4_K);
        List<String> names = run(() -> ops.matvec(w, autoInput(), 0, autoOutput(), 0, 1));
        assertTrue(names.toString(), names.contains("Q4_K_MatVec_Q8"));
        assertFalse(names.toString(), names.contains("Q4_K_MatVec"));
    }

    @Test
    public void autoTakesTheQ8_0LaneOnSmallAndLargeCudaProjections() {
        open(Int8MatvecMode.AUTO, DeviceFeature.INTEGER_DOT_PRODUCT, ComputeBackend.BackendType.CUDA,
                CUDA_ROW_TWINS);
        // The small decoder projections measured faster with the lane once its activation rows cost a
        // warp-per-block quantizer launch or none (norm-emitted), so a 32x256 projection takes it too.
        GpuWeight small = weight(GGMLType.Q8_0);
        List<String> smallNames = run(() -> ops.matvec(small, backend.allocateBuffer((long) COLS * 4), 0,
                backend.allocateBuffer((long) ROWS * 4), 0, 1));
        assertTrue(smallNames.toString(), smallNames.contains("Q8_0_MatVec_Q8"));
        assertFalse(smallNames.toString(), smallNames.contains("Q8_0_MatVec"));
        assertTrue(smallNames.toString(), smallNames.contains(GpuOps.QUANTIZE_Q8_1));
        GpuWeight large = autoWeight(GGMLType.Q8_0);
        List<String> largeNames = run(() -> ops.matvec(large, autoInput(), 0, autoOutput(), 0, 1));
        assertTrue(largeNames.toString(), largeNames.contains("Q8_0_MatVec_Q8"));
        assertFalse(largeNames.toString(), largeNames.contains("Q8_0_MatVec"));
    }

    /**
     * The RTX 5080's Vulkan device kind gates the Q8_0 lane by projection size: its measured 0.61x
     * at 1024x1024 keeps a small projection on the F32 matvec and its measured 1.37-2.27x from
     * 16.8M up takes the lane above it.
     */
    @Test
    public void autoGatesTheVulkanDiscreteKindBelowTheMeasuredLaneThreshold() {
        DeviceCapabilities vulkanRtx = capabilities(ComputeBackend.BackendType.VULKAN,
                FakeComputeBackend.NVIDIA_VENDOR_ID, DeviceClass.DISCRETE,
                DeviceFeature.INTEGER_DOT_PRODUCT);
        open(Int8MatvecMode.AUTO, DeviceFeature.INTEGER_DOT_PRODUCT, ComputeBackend.BackendType.VULKAN,
                vulkanRtx);
        GpuWeight small = weight(GGMLType.Q8_0);
        List<String> smallNames = run(() -> ops.matvec(small, backend.allocateBuffer((long) COLS * 4), 0,
                backend.allocateBuffer((long) ROWS * 4), 0, 1));
        assertTrue(smallNames.toString(), smallNames.contains("Q8_0_MatVec"));
        assertFalse(smallNames.toString(), smallNames.contains("Q8_0_MatVec_Q8"));
        assertFalse(smallNames.toString(), smallNames.contains(GpuOps.QUANTIZE_Q8_1));
        GpuWeight large = autoWeight(GGMLType.Q8_0);
        List<String> largeNames = run(() -> ops.matvec(large, autoInput(), 0, autoOutput(), 0, 1));
        assertTrue(largeNames.toString(), largeNames.contains("Q8_0_MatVec_Q8"));
        assertFalse(largeNames.toString(), largeNames.contains("Q8_0_MatVec"));
    }

    /**
     * The Arrow-Lake iGPU's OpenCL kind follows its own entries under the same backend and feature:
     * Q8_0 measured slower with the lane at every shape (and 0.75-0.85x over the whole 4B decoder
     * step), so it keeps the F32 matvec even above the lane threshold, while its NVFP4 measured
     * 1.08-2.13x faster everywhere and takes the lane.
     */
    @Test
    public void autoFollowsTheOpenCLIntegratedKindEntries() {
        DeviceCapabilities openclIgpu = capabilities(ComputeBackend.BackendType.OPENCL,
                FakeComputeBackend.INTEL_VENDOR_ID, DeviceClass.INTEGRATED,
                DeviceFeature.INTEGER_DOT_PRODUCT);
        open(Int8MatvecMode.AUTO, DeviceFeature.INTEGER_DOT_PRODUCT, ComputeBackend.BackendType.OPENCL,
                openclIgpu);
        GpuWeight q8 = autoWeight(GGMLType.Q8_0);
        List<String> q8Names = run(() -> ops.matvec(q8, autoInput(), 0, autoOutput(), 0, 1));
        assertTrue(q8Names.toString(), q8Names.contains("Q8_0_MatVec"));
        assertFalse(q8Names.toString(), q8Names.contains("Q8_0_MatVec_Q8"));
        assertFalse(q8Names.toString(), q8Names.contains(GpuOps.QUANTIZE_Q8_1));
        GpuWeight nvfp4 = autoWeight(GGMLType.NVFP4);
        List<String> nvfp4Names = run(() -> ops.matvec(nvfp4, autoInput(), 0, autoOutput(), 0, 1));
        assertTrue(nvfp4Names.toString(), nvfp4Names.contains("NVFP4_MatVec_Q8"));
        assertFalse(nvfp4Names.toString(), nvfp4Names.contains("NVFP4_MatVec"));
    }

    /**
     * The CUDA modules' warp-row twins take the single-token Q8_0, Q4_K and Q6_K products, their accumulating form
     * and the fused gate/up on the shapes {@link WarpRowMatvec#warps} gives them, at that many warps per work-group
     * and one warp per row, whatever the family's lane evidence says (a 32x256 Q6_K projection sits below its
     * measured lane threshold); a Q8_0 row wider than {@link WarpRowMatvec#NARROW_COLS} and a wide Q6_K head keep
     * the family, and OFF keeps the float matvec.
     */
    @Test
    public void warpRowTwinsServeTheSingleTokenLaneOnCuda() {
        open(Int8MatvecMode.AUTO, DeviceFeature.INTEGER_DOT_PRODUCT, ComputeBackend.BackendType.CUDA);
        GpuBuffer in = backend.allocateBuffer((long) COLS * 4);
        GpuBuffer out = backend.allocateBuffer((long) ROWS * 4);
        for (GGMLType type : new GGMLType[]{GGMLType.Q8_0, GGMLType.Q4_K, GGMLType.Q6_K}) {
            int warps = WarpRowMatvec.warps(type, ROWS, COLS);
            assertTrue(type.toString(), warps > 0);
            GpuWeight w = weight(type);
            backend.clearLog();
            ops.matvec(w, in, 0, out, 0, 1);
            List<FakeDispatch> ds = backend.dispatches();
            assertEquals(GpuOps.QUANTIZE_Q8_1, ds.get(0).kernelName());
            FakeDispatch row = ds.get(1);
            assertEquals(type.name() + "_MatVecRow_Q8", row.kernelName());
            assertArrayEquals("one warp per row", new long[]{(long) GpuOps.ceilDiv(ROWS, warps) * 32 * warps},
                    row.global());
            assertArrayEquals(new long[]{32L * warps}, row.local());
            assertEquals(Arrays.asList(GpuOps.QUANTIZE_Q8_1, type.name() + "_MatVecAddRow_Q8"),
                    run(() -> ops.matvecAdd(w, in, 0, out, 0, 1)));
            GpuWeight gate = weight(type, 1024);
            GpuWeight up = weight(type, 1024);
            GpuBuffer gateOut = backend.allocateBuffer(1024L * 4);
            assertEquals(Arrays.asList(GpuOps.QUANTIZE_Q8_1, type.name() + "_MatVecFusedGateUpRow_Q8"),
                    run(() -> ops.gateUpGlu(gate, up, GluOp.SWIGLU, ActParams.NONE, in, gateOut, 1)));
        }
        GpuWeight wideQ8 = autoWeight(GGMLType.Q8_0);
        List<String> wide = run(() -> ops.matvec(wideQ8, autoInput(), 0, autoOutput(), 0, 1));
        assertTrue(wide.toString(), wide.contains("Q8_0_MatVec_Q8"));
        int headRows = WarpRowMatvec.HEAD_ROWS;
        int headCols = 5 * GGMLType.Q6_K.getBlockSize();
        GpuWeight head = GpuWeight.of("output.weight", GGMLType.Q6_K, new long[]{headCols, headRows},
                backend.allocateBuffer((long) headRows * (headCols / GGMLType.Q6_K.getBlockSize())
                        * GGMLType.Q6_K.getTypeSize(), true));
        List<String> headNames = run(() -> ops.matvec(head, backend.allocateBuffer((long) headCols * 4), 0,
                backend.allocateBuffer((long) headRows * 4), 0, 1));
        assertFalse(headNames.toString(), headNames.contains("Q6_K_MatVecRow_Q8"));

        open(Int8MatvecMode.OFF, DeviceFeature.INTEGER_DOT_PRODUCT, ComputeBackend.BackendType.CUDA);
        GpuWeight offWeight = weight(GGMLType.Q4_K);
        assertEquals(Collections.singletonList("Q4_K_MatVec"), run(() -> ops.matvec(offWeight,
                backend.allocateBuffer((long) COLS * 4), 0, backend.allocateBuffer((long) ROWS * 4), 0, 1)));
    }

    /**
     * A projection held in shards takes the twin and the warps per work-group its whole shape picks on every shard,
     * the shape the lane was judged on: a Q8_0 head of two shards runs four warps per work-group (a shard alone would
     * count as a decoder projection and run one), and a Q6_K head over more than
     * {@link WarpRowMatvec#NARROW_COLS} columns keeps the family on both shards.
     */
    @Test
    public void aShardedHeadTakesTheFormOfItsWholeShapeOnEveryShard() {
        open(Int8MatvecMode.AUTO, DeviceFeature.INTEGER_DOT_PRODUCT, ComputeBackend.BackendType.CUDA);
        int shardRows = WarpRowMatvec.HEAD_ROWS / 2 + 8;
        int narrow = WarpRowMatvec.NARROW_COLS;
        assertEquals(4, WarpRowMatvec.warps(GGMLType.Q8_0, 2 * shardRows, narrow));
        assertEquals(1, WarpRowMatvec.warps(GGMLType.Q8_0, shardRows, narrow));
        backend.clearLog();
        ops.matvec(shardedHead(GGMLType.Q8_0, narrow, shardRows), backend.allocateBuffer((long) narrow * 4), 0,
                backend.allocateBuffer(2L * shardRows * 4), 0, 1);
        List<FakeDispatch> launches = new ArrayList<>();
        for (FakeDispatch d : backend.dispatches()) {
            if (d.kernelName().equals("Q8_0_MatVecRow_Q8")) {
                launches.add(d);
            }
        }
        assertEquals("one launch per shard", 2, launches.size());
        for (FakeDispatch d : launches) {
            assertArrayEquals(new long[]{4L * 32}, d.local());
            assertArrayEquals(new long[]{(long) GpuOps.ceilDiv(shardRows, 4) * 4 * 32}, d.global());
        }

        int wide = 5 * GGMLType.Q6_K.getBlockSize();
        assertEquals(0, WarpRowMatvec.warps(GGMLType.Q6_K, 2 * shardRows, wide));
        assertEquals(1, WarpRowMatvec.warps(GGMLType.Q6_K, shardRows, wide));
        List<String> names = run(() -> ops.matvec(shardedHead(GGMLType.Q6_K, wide, shardRows),
                backend.allocateBuffer((long) wide * 4), 0, backend.allocateBuffer(2L * shardRows * 4), 0, 1));
        assertFalse(names.toString(), names.contains("Q6_K_MatVecRow_Q8"));
    }

    /** An output head of two shards of {@code shardRows} rows of {@code cols} elements of {@code type}. */
    private GpuWeight shardedHead(GGMLType type, int cols, int shardRows) {
        long shardBytes = (long) shardRows * (cols / type.getBlockSize()) * type.getTypeSize();
        return GpuWeight.sharded("output.weight", type, new long[]{cols, 2L * shardRows},
                Arrays.asList(backend.allocateBuffer(shardBytes, true), backend.allocateBuffer(shardBytes, true)),
                new long[]{shardRows, shardRows});
    }

    /**
     * An OpenCL device whose program builds the subgroup-row forms (subgroups with a required-size range that
     * holds 16 lanes, the Intel GPUs:
     * {@link com.eonmux.neuralfree.acceleration.kernels.OpenClKernelLibrary#subgroupRowSize}) runs the single-token
     * Q8_0, Q4_0, Q4_K, Q5_K and Q6_K products, their accumulating form and the fused gate/up through them under
     * AUTO, overriding the Arrow-Lake kind's family evidence against the Q8_0 lane; other types keep the family, OFF
     * keeps the float matvec, and the same device without the size range keeps today's dispatch.
     */
    @Test
    public void subgroupRowFormsServeTheSingleTokenLaneWhereTheProgramBuildsThem() {
        open(Int8MatvecMode.AUTO, DeviceFeature.INTEGER_DOT_PRODUCT, ComputeBackend.BackendType.OPENCL,
                subgroupRowCapabilities(8, 32));
        GpuBuffer in = backend.allocateBuffer((long) COLS * 4);
        GpuBuffer out = backend.allocateBuffer((long) ROWS * 4);
        for (GGMLType type : SubgroupRowMatvec.TYPES) {
            GpuWeight w = weight(type);
            assertEquals(Arrays.asList(GpuOps.QUANTIZE_Q8_1, type.name() + "_MatVecSg_Q8"),
                    run(() -> ops.matvec(w, in, 0, out, 0, 1)));
            assertEquals(Arrays.asList(GpuOps.QUANTIZE_Q8_1, type.name() + "_MatVecAddSg_Q8"),
                    run(() -> ops.matvecAdd(w, in, 0, out, 0, 1)));
            GpuWeight gate = weight(type, 1024);
            GpuWeight up = weight(type, 1024);
            GpuBuffer gateOut = backend.allocateBuffer(1024L * 4);
            assertEquals(Arrays.asList(GpuOps.QUANTIZE_Q8_1, type.name() + "_MatVecFusedGateUpSg_Q8"),
                    run(() -> ops.gateUpGlu(gate, up, GluOp.SWIGLU, ActParams.NONE, in, gateOut, 1)));
        }
        assertEquals(EnumSet.of(GGMLType.Q8_0, GGMLType.Q4_0, GGMLType.Q4_K, GGMLType.Q5_K, GGMLType.Q6_K),
                SubgroupRowMatvec.TYPES);
        GpuWeight q5_0 = weight(GGMLType.Q5_0);
        List<String> family = run(() -> ops.matvec(q5_0, in, 0, out, 0, 1));
        assertFalse(family.toString(), family.contains("Q5_0_MatVecSg_Q8"));

        open(Int8MatvecMode.OFF, DeviceFeature.INTEGER_DOT_PRODUCT, ComputeBackend.BackendType.OPENCL,
                subgroupRowCapabilities(8, 32));
        GpuWeight offWeight = weight(GGMLType.Q8_0);
        assertEquals(Collections.singletonList("Q8_0_MatVec"), run(() -> ops.matvec(offWeight,
                backend.allocateBuffer((long) COLS * 4), 0, backend.allocateBuffer((long) ROWS * 4), 0, 1)));

        open(Int8MatvecMode.AUTO, DeviceFeature.INTEGER_DOT_PRODUCT, ComputeBackend.BackendType.OPENCL,
                subgroupRowCapabilities(32, 64));
        GpuWeight noRange = weight(GGMLType.Q8_0);
        List<String> names = run(() -> ops.matvec(noRange, backend.allocateBuffer((long) COLS * 4), 0,
                backend.allocateBuffer((long) ROWS * 4), 0, 1));
        assertFalse(names.toString(), names.contains("Q8_0_MatVecSg_Q8"));
    }

    /**
     * A multi-token projection of a {@link SubgroupMmqGemm#TYPES} type on a program built with the subgroup forms
     * runs that type's subgroup-row MMQ GEMM over the same staged records; a type without a subgroup GEMM (Q5_0),
     * and a device whose program does not build the forms, keep the portable MMQ kernel.
     */
    @Test
    public void subgroupMmqGemmServesTheMultiTokenLanesOfItsTypesWhereTheProgramBuildsIt() {
        int tokens = 512;
        open(Int8MatvecMode.AUTO, DeviceFeature.INTEGER_DOT_PRODUCT, ComputeBackend.BackendType.OPENCL,
                subgroupRowCapabilities(8, 32));
        GpuBuffer in = backend.allocateBuffer((long) COLS * tokens * 4);
        GpuBuffer out = backend.allocateBuffer((long) ROWS * tokens * 4);
        assertEquals(EnumSet.of(GGMLType.Q8_0, GGMLType.Q4_0, GGMLType.Q4_K, GGMLType.Q5_K, GGMLType.Q6_K),
                SubgroupMmqGemm.TYPES);
        for (GGMLType type : SubgroupMmqGemm.TYPES) {
            GpuWeight w = weight(type);
            List<String> sg = run(() -> ops.matmul(w, in, 0, out, 0, tokens));
            assertTrue(type + " " + sg, sg.contains(type.name() + "_MatMulSg_MMQ"));
            assertFalse(type + " " + sg, sg.contains(type.name() + "_MatMul_MMQ"));
        }
        GpuWeight q50 = weight(GGMLType.Q5_0);
        List<String> family = run(() -> ops.matmul(q50, in, 0, out, 0, tokens));
        assertTrue(family.toString(), family.contains("Q5_0_MatMul_MMQ"));
        assertFalse(family.toString(), family.contains("Q5_0_MatMulSg_MMQ"));

        open(Int8MatvecMode.AUTO, DeviceFeature.INTEGER_DOT_PRODUCT, ComputeBackend.BackendType.OPENCL,
                subgroupRowCapabilities(32, 64));
        for (GGMLType type : SubgroupMmqGemm.TYPES) {
            GpuWeight noRange = weight(type);
            List<String> portable = run(() -> ops.matmul(noRange, backend.allocateBuffer((long) COLS * tokens * 4), 0,
                    backend.allocateBuffer((long) ROWS * tokens * 4), 0, tokens));
            assertTrue(type + " " + portable, portable.contains(type.name() + "_MatMul_MMQ"));
            assertFalse(type + " " + portable, portable.contains(type.name() + "_MatMulSg_MMQ"));
        }
    }

    /**
     * A subgroup-row GEMM launch splits by its own twin: on a program built with the subgroup forms (32 compute
     * units), a 512 x 2048 projection of 16 tokens, a grid of four 128 x 32 tiles, runs {@code {T}_MatMulSg_MMQ_KSplit}
     * and the reduce, and a 1024 x 2048 one, eight tiles, the plain subgroup grid; neither runs the portable kernel's
     * split, which the second shape's 16 packed-dot tiles would take.
     */
    @Test
    public void subgroupGemmLaunchesSplitByTheirOwnTwinAndNeverByThePortableOne() {
        int tokens = 16;
        int cols = 2048;
        open(Int8MatvecMode.AUTO, DeviceFeature.INTEGER_DOT_PRODUCT, ComputeBackend.BackendType.OPENCL,
                subgroupRowCapabilities(8, 32));
        GpuBuffer in = backend.allocateBuffer((long) cols * tokens * 4);
        for (GGMLType type : SubgroupMmqGemm.TYPES) {
            String sg = type.name() + "_MatMulSg_MMQ";
            GpuWeight narrow = weight(type, 512, cols);
            List<String> split = run(() -> ops.matmul(narrow, in, 0, backend.allocateBuffer(512L * tokens * 4), 0,
                    tokens));
            assertTrue(type + " " + split, split.contains(sg + "_KSplit"));
            assertTrue(type + " " + split, split.contains(GemmKernels.SPLIT_K_REDUCE));
            assertFalse(type + " " + split, split.contains(sg));
            assertFalse(type + " " + split, split.contains(type.name() + GemmKernels.MATMUL_MMQ_KSPLIT_SUFFIX));
            GpuWeight coarse = weight(type, 1024, cols);
            List<String> whole = run(() -> ops.matmul(coarse, in, 0, backend.allocateBuffer(1024L * tokens * 4), 0,
                    tokens));
            assertTrue(type + " " + whole, whole.contains(sg));
            assertFalse(type + " " + whole, whole.contains(sg + "_KSplit"));
            assertFalse(type + " " + whole, whole.contains(type.name() + GemmKernels.MATMUL_MMQ_KSPLIT_SUFFIX));
        }
    }

    /**
     * The subgroup GEMM's row-length gate: Q8_0 rows of whole 64-element stages, K-quant rows of whole 256-element
     * super-blocks, and no form for a type outside {@link SubgroupMmqGemm#TYPES}.
     */
    @Test
    public void subgroupMmqGemmServesWholeStagesOfWholeBlocks() {
        assertTrue(SubgroupMmqGemm.shapeServes(GGMLType.Q8_0, 64));
        assertFalse(SubgroupMmqGemm.shapeServes(GGMLType.Q8_0, 96));
        for (GGMLType type : new GGMLType[]{GGMLType.Q4_K, GGMLType.Q6_K}) {
            assertEquals(type.name() + "_MatMulSg_MMQ", SubgroupMmqGemm.kernel(type));
            assertTrue(type.name(), SubgroupMmqGemm.shapeServes(type, 256));
            assertTrue(type.name(), SubgroupMmqGemm.shapeServes(type, 2560));
            assertFalse(type.name(), SubgroupMmqGemm.shapeServes(type, 192));
            assertFalse(type.name(), SubgroupMmqGemm.shapeServes(type, 0));
        }
        assertNull(SubgroupMmqGemm.kernel(GGMLType.Q5_0));
        assertFalse(SubgroupMmqGemm.shapeServes(GGMLType.Q5_0, 256));
    }

    /** The Arrow-Lake OpenCL kind with subgroups whose required sizes span {@code min..max}. */
    private static DeviceCapabilities subgroupRowCapabilities(int min, int max) {
        return capabilitiesBuilder(ComputeBackend.BackendType.OPENCL, FakeComputeBackend.INTEL_VENDOR_ID,
                DeviceClass.INTEGRATED, DeviceFeature.INTEGER_DOT_PRODUCT)
                .supportsSubgroups(true).subgroupSize(max).minSubgroupSize(min).maxSubgroupSize(max).build();
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

    /**
     * An entry governs only the device kind it was measured on: the Q4_0 loss measured on the RTX
     * 5080's CUDA device keeps that kind's fake on the float matvec, while the same weight over a
     * CUDA fake of another vendor (whose kind nobody measured) takes the lane its packed dot
     * products permit.
     */
    @Test
    public void anEntryGovernsOnlyTheDeviceKindItWasMeasuredOn() {
        open(Int8MatvecMode.AUTO, DeviceFeature.INTEGER_DOT_PRODUCT, ComputeBackend.BackendType.CUDA);
        GpuWeight w = autoWeight(GGMLType.Q4_0);
        List<String> names = run(() -> ops.matvec(w, autoInput(), 0, autoOutput(), 0, 1));
        assertTrue(names.toString(), names.contains("Q4_0_MatVec"));
        assertFalse(names.toString(), names.contains("Q4_0_MatVec_Q8"));
        DeviceCapabilities otherVendor = capabilities(ComputeBackend.BackendType.CUDA, 0x1002,
                DeviceClass.DISCRETE, DeviceFeature.INTEGER_DOT_PRODUCT);
        open(Int8MatvecMode.AUTO, DeviceFeature.INTEGER_DOT_PRODUCT, ComputeBackend.BackendType.CUDA,
                otherVendor);
        GpuWeight same = autoWeight(GGMLType.Q4_0);
        List<String> otherNames = run(() -> ops.matvec(same, autoInput(), 0, autoOutput(), 0, 1));
        assertTrue(otherNames.toString(), otherNames.contains("Q4_0_MatVec_Q8"));
        assertFalse(otherNames.toString(), otherNames.contains("Q4_0_MatVec"));
    }

    /** A device whose vendor or class is unknown matches no entry and keeps the feature decision. */
    @Test
    public void anUnknownVendorOrClassKeepsTheFeatureDecision() {
        DeviceCapabilities unknownVendor = capabilities(ComputeBackend.BackendType.CUDA, 0x1234,
                DeviceClass.DISCRETE, DeviceFeature.INTEGER_DOT_PRODUCT);
        open(Int8MatvecMode.AUTO, DeviceFeature.INTEGER_DOT_PRODUCT, ComputeBackend.BackendType.CUDA,
                unknownVendor);
        GpuWeight w = autoWeight(GGMLType.Q4_0);
        List<String> names = run(() -> ops.matvec(w, autoInput(), 0, autoOutput(), 0, 1));
        assertTrue(names.toString(), names.contains("Q4_0_MatVec_Q8"));
        assertFalse(names.toString(), names.contains("Q4_0_MatVec"));
        DeviceCapabilities unknownClass = capabilities(ComputeBackend.BackendType.CUDA,
                FakeComputeBackend.NVIDIA_VENDOR_ID, DeviceClass.UNKNOWN, DeviceFeature.INTEGER_DOT_PRODUCT);
        open(Int8MatvecMode.AUTO, DeviceFeature.INTEGER_DOT_PRODUCT, ComputeBackend.BackendType.CUDA,
                unknownClass);
        GpuWeight same = autoWeight(GGMLType.Q4_0);
        List<String> otherNames = run(() -> ops.matvec(same, autoInput(), 0, autoOutput(), 0, 1));
        assertTrue(otherNames.toString(), otherNames.contains("Q4_0_MatVec_Q8"));
        assertFalse(otherNames.toString(), otherNames.contains("Q4_0_MatVec"));
    }

    @Test
    public void tuningKeepsTheLaneOnlyForTypesWithoutAMeasuredLoss() {
        DeviceKind openclNvidia = kind(ComputeBackend.BackendType.OPENCL, DeviceKind.Vendor.NVIDIA);
        DeviceKind cudaTensor = DeviceKind.of(ComputeBackend.BackendType.CUDA, DeviceKind.Vendor.NVIDIA,
                DeviceClass.DISCRETE, DeviceKind.CudaBand.TENSOR);
        DeviceKind vulkanIntel = kind(ComputeBackend.BackendType.VULKAN, DeviceKind.Vendor.INTEL);
        DeviceKind vulkanNvidia = kind(ComputeBackend.BackendType.VULKAN, DeviceKind.Vendor.NVIDIA);
        DeviceKind openclIntel = kind(ComputeBackend.BackendType.OPENCL, DeviceKind.Vendor.INTEL);
        // Measured slower with the lane at every projection shape, so never selected.
        assertFalse(Int8MatvecTuning.laneIsFaster(openclNvidia, GGMLType.MXFP4));
        assertFalse(Int8MatvecTuning.laneIsFaster(openclNvidia, GGMLType.NVFP4));
        assertFalse(Int8MatvecTuning.laneIsFaster(openclNvidia, GGMLType.IQ4_XS));
        assertFalse(Int8MatvecTuning.laneIsFaster(cudaTensor, GGMLType.Q4_0));
        assertFalse(Int8MatvecTuning.laneIsFaster(cudaTensor, GGMLType.Q4_1));
        assertFalse(Int8MatvecTuning.laneIsFaster(cudaTensor, GGMLType.Q5_0));
        assertFalse(Int8MatvecTuning.laneIsFaster(cudaTensor, GGMLType.Q5_1));
        assertFalse(Int8MatvecTuning.laneIsFaster(cudaTensor, GGMLType.IQ4_NL));
        assertFalse(Int8MatvecTuning.laneIsFaster(vulkanIntel, GGMLType.Q4_0));
        // Measured faster on the larger projections only.
        assertTrue(Int8MatvecTuning.laneIsFaster(openclNvidia, GGMLType.Q4_0));
        assertTrue(Int8MatvecTuning.laneIsFaster(openclNvidia, GGMLType.Q8_0));
        assertTrue(Int8MatvecTuning.laneIsFaster(openclNvidia, GGMLType.Q4_K));
        assertTrue(Int8MatvecTuning.laneIsFaster(cudaTensor, GGMLType.Q4_K));
        assertTrue(Int8MatvecTuning.laneIsFaster(cudaTensor, GGMLType.Q8_0));
        // No SLOWER loss on the RTX 5080's Vulkan device: every type measured a win from 16.8M up,
        // so the kind records only the large-projection gate.
        assertTrue(Int8MatvecTuning.laneIsFaster(vulkanNvidia, GGMLType.Q8_0));
        assertTrue(Int8MatvecTuning.laneIsFaster(vulkanNvidia, GGMLType.Q4_0));
        // The Arrow-Lake iGPU's OpenCL kind measured eight types slower at every shape, its NVFP4
        // lane 1.08-2.13x faster everywhere.
        assertFalse(Int8MatvecTuning.laneIsFaster(openclIntel, GGMLType.Q8_0));
        assertFalse(Int8MatvecTuning.laneIsFaster(openclIntel, GGMLType.Q4_1));
        assertFalse(Int8MatvecTuning.laneIsFaster(openclIntel, GGMLType.Q5_1));
        assertFalse(Int8MatvecTuning.laneIsFaster(openclIntel, GGMLType.Q3_K));
        assertFalse(Int8MatvecTuning.laneIsFaster(openclIntel, GGMLType.MXFP4));
        assertFalse(Int8MatvecTuning.laneIsFaster(openclIntel, GGMLType.IQ4_XS));
        assertTrue(Int8MatvecTuning.laneIsFaster(openclIntel, GGMLType.NVFP4));
        // A backend nobody measured, and a kind nobody measured, keep the feature decision.
        assertTrue(Int8MatvecTuning.laneIsFaster(DeviceKind.of(ComputeBackend.BackendType.HIP,
                DeviceKind.Vendor.AMD, DeviceClass.DISCRETE, DeviceKind.CudaBand.NONE), GGMLType.Q4_K));
        assertTrue(Int8MatvecTuning.laneIsFaster(DeviceKind.of(ComputeBackend.BackendType.OPENCL,
                DeviceKind.Vendor.OTHER, DeviceClass.DISCRETE, DeviceKind.CudaBand.NONE), GGMLType.MXFP4));
    }

    /** The measured kind of one backend's vendor: NVIDIA discrete outside CUDA banding. */
    private static DeviceKind kind(ComputeBackend.BackendType type, DeviceKind.Vendor vendor) {
        return DeviceKind.of(type, vendor,
                vendor == DeviceKind.Vendor.INTEL ? DeviceClass.INTEGRATED : DeviceClass.DISCRETE,
                DeviceKind.CudaBand.NONE);
    }

    @Test
    public void tuningGatesTheSmallProjectionOfALargeOnlyType() {
        DeviceKind cudaTensor = DeviceKind.of(ComputeBackend.BackendType.CUDA, DeviceKind.Vendor.NVIDIA,
                DeviceClass.DISCRETE, DeviceKind.CudaBand.TENSOR);
        DeviceKind openclNvidia = kind(ComputeBackend.BackendType.OPENCL, DeviceKind.Vendor.NVIDIA);
        DeviceKind vulkanIntel = kind(ComputeBackend.BackendType.VULKAN, DeviceKind.Vendor.INTEL);
        DeviceKind vulkanNvidia = kind(ComputeBackend.BackendType.VULKAN, DeviceKind.Vendor.NVIDIA);
        DeviceKind openclIntel = kind(ComputeBackend.BackendType.OPENCL, DeviceKind.Vendor.INTEL);
        // Q6_K's 1024x2560 rows measured slower with the lane on CUDA; 4096x4096 and up measured faster.
        assertFalse(Int8MatvecTuning.laneIsFaster(cudaTensor, GGMLType.Q6_K, 3072L * 1024));
        assertFalse(Int8MatvecTuning.laneIsFaster(cudaTensor, GGMLType.Q5_K, 1024L * 1024));
        assertTrue(Int8MatvecTuning.laneIsFaster(cudaTensor, GGMLType.Q6_K,
                Int8MatvecTuning.LANE_MIN_ELEMENTS));
        assertTrue(Int8MatvecTuning.laneIsFaster(cudaTensor, GGMLType.Q6_K, 4096L * 4096));
        // With the warp-per-block quantizer and the norm-emitted rows, Q8_0 and Q4_K took the lane faster
        // at the small CUDA projections too; Q8_0 lost the lane's fixed quantizer cost on the small
        // projections of the OpenCL device and measured 1.11x to 1.36x with it from 4096x4096 up there.
        assertTrue(Int8MatvecTuning.laneIsFaster(cudaTensor, GGMLType.Q8_0, 1024L * 1024));
        assertTrue(Int8MatvecTuning.laneIsFaster(cudaTensor, GGMLType.Q4_K, 1024L * 2560));
        assertTrue(Int8MatvecTuning.laneIsFaster(cudaTensor, GGMLType.Q8_0, 4096L * 4096));
        assertFalse(Int8MatvecTuning.laneIsFaster(openclNvidia, GGMLType.Q8_0, 1024L * 1024));
        assertTrue(Int8MatvecTuning.laneIsFaster(openclNvidia, GGMLType.Q8_0, 9728L * 2560));
        // The RTX 5080's Vulkan device measured 0.61x at 1024x1024 and 1.37-2.27x with the lane
        // from 16.8M up, its Arrow-Lake OpenCL iGPU 0.77-1.00x below the threshold and 1.09-1.22x
        // above it for Q4_0.
        assertFalse(Int8MatvecTuning.laneIsFaster(vulkanNvidia, GGMLType.Q8_0, 1024L * 1024));
        assertTrue(Int8MatvecTuning.laneIsFaster(vulkanNvidia, GGMLType.Q8_0, 9728L * 2560));
        assertFalse(Int8MatvecTuning.laneIsFaster(openclIntel, GGMLType.Q4_0, 1024L * 1024));
        assertTrue(Int8MatvecTuning.laneIsFaster(openclIntel, GGMLType.Q4_0, 9728L * 2560));
        // The always-slower set stays out at every size.
        assertFalse(Int8MatvecTuning.laneIsFaster(cudaTensor, GGMLType.Q4_0, Long.MAX_VALUE));
        assertFalse(Int8MatvecTuning.laneIsFaster(openclNvidia, GGMLType.MXFP4, Long.MAX_VALUE));
        assertFalse(Int8MatvecTuning.laneIsFaster(vulkanIntel, GGMLType.Q4_0, Long.MAX_VALUE));
        assertFalse(Int8MatvecTuning.laneIsFaster(openclIntel, GGMLType.Q8_0, Long.MAX_VALUE));
        // A kind and type without evidence keep the lane at any size.
        assertTrue(Int8MatvecTuning.laneIsFaster(vulkanIntel, GGMLType.Q4_K, 1));
        assertTrue(Int8MatvecTuning.laneIsFaster(openclIntel, GGMLType.NVFP4, 1));
    }

    /**
     * The counted multi-column lane's evidence overrides the single-row losses only where its own
     * comparison measured a win: Q8_0 on the CUDA and OpenCL NVIDIA kinds at any size, Q4_0 there from
     * {@link Int8MatvecTuning#LANE_MIN_ELEMENTS} up, IQ4_XS on the OpenCL kind from the threshold up —
     * the same kinds' portable-arithmetic forms measured the wins, so the packed-dot requirement drops
     * for exactly those type-and-kind pairs.
     */
    @Test
    public void countedLaneEvidenceOverridesTheSingleRowLossesWhereItMeasuredWins() {
        DeviceKind cudaTensor = DeviceKind.of(ComputeBackend.BackendType.CUDA, DeviceKind.Vendor.NVIDIA,
                DeviceClass.DISCRETE, DeviceKind.CudaBand.TENSOR);
        DeviceKind openclNvidia = kind(ComputeBackend.BackendType.OPENCL, DeviceKind.Vendor.NVIDIA);
        DeviceKind openclIntel = kind(ComputeBackend.BackendType.OPENCL, DeviceKind.Vendor.INTEL);
        // Q8_0's counted evidence covers the small projections of both kinds.
        assertTrue(Int8MatvecTuning.countedLaneIsFaster(cudaTensor, GGMLType.Q8_0, 1024L * 1024));
        assertTrue(Int8MatvecTuning.countedLaneIsFaster(openclNvidia, GGMLType.Q8_0, 1024L * 1024));
        // Q4_0 and IQ4_XS's counted evidence covers the two measured shapes, both at or above the
        // threshold; below it the single-row losses keep the lane off.
        assertTrue(Int8MatvecTuning.countedLaneIsFaster(cudaTensor, GGMLType.Q4_0,
                Int8MatvecTuning.LANE_MIN_ELEMENTS));
        assertFalse(Int8MatvecTuning.countedLaneIsFaster(cudaTensor, GGMLType.Q4_0, 1024L * 1024));
        assertTrue(Int8MatvecTuning.countedLaneIsFaster(openclNvidia, GGMLType.Q4_0,
                Int8MatvecTuning.LANE_MIN_ELEMENTS));
        assertFalse(Int8MatvecTuning.countedLaneIsFaster(openclNvidia, GGMLType.Q4_0, 1024L * 1024));
        assertTrue(Int8MatvecTuning.countedLaneIsFaster(openclNvidia, GGMLType.IQ4_XS, 4096L * 1024));
        assertFalse(Int8MatvecTuning.countedLaneIsFaster(openclNvidia, GGMLType.IQ4_XS, 1024L * 1024));
        // The large-only types' counted lane follows the single-row size gate.
        assertTrue(Int8MatvecTuning.countedLaneIsFaster(openclNvidia, GGMLType.Q4_K, 4096L * 1024));
        assertFalse(Int8MatvecTuning.countedLaneIsFaster(openclNvidia, GGMLType.Q4_K, 1024L * 1024));
        assertTrue(Int8MatvecTuning.countedLaneIsFaster(cudaTensor, GGMLType.Q5_K, 4096L * 1024));
        // A kind nobody measured for the counted comparison keeps the single-row answer: the
        // OpenCL iGPU's always-slower IQ4_XS gains no counted redemption there.
        assertFalse(Int8MatvecTuning.countedLaneIsFaster(openclIntel, GGMLType.IQ4_XS,
                Int8MatvecTuning.LANE_MIN_ELEMENTS));
        // The portable-arithmetic exception: the measured OpenCL NVIDIA types, and no others.
        for (GGMLType type : new GGMLType[]{GGMLType.Q8_0, GGMLType.Q4_0, GGMLType.Q4_K, GGMLType.Q5_K,
                GGMLType.Q6_K, GGMLType.IQ4_XS}) {
            assertFalse(type + " runs portable arithmetic there",
                    Int8MatvecTuning.countedLaneNeedsPackedDots(openclNvidia, type));
        }
        assertTrue(Int8MatvecTuning.countedLaneNeedsPackedDots(openclNvidia, GGMLType.Q2_K));
        assertTrue(Int8MatvecTuning.countedLaneNeedsPackedDots(cudaTensor, GGMLType.Q8_0));
        assertTrue(Int8MatvecTuning.countedLaneNeedsPackedDots(openclIntel, GGMLType.Q8_0));
    }

    /**
     * The Vulkan NVIDIA kind's measured small projections take the counted lane from
     * {@link Int8MatvecTuning#COUNTED_SMALL_MIN_TOKENS} rows, where the pair beat the batched matvec and the CM2
     * lane, and at two rows from {@link Int8MatvecTuning#COUNTED_SMALL_TWO_ROW_MIN_ELEMENTS} (the batched matvec
     * won two rows on the smaller shapes); the unmeasured types keep the size gate at every count, and the other
     * kinds' answers do not depend on the row count.
     */
    @Test
    public void vulkanNvidiaSmallProjectionsTakeTheCountedLaneFromThreeRows() {
        DeviceKind vulkanNvidia = kind(ComputeBackend.BackendType.VULKAN, DeviceKind.Vendor.NVIDIA);
        DeviceKind vulkanIntel = kind(ComputeBackend.BackendType.VULKAN, DeviceKind.Vendor.INTEL);
        DeviceKind openclNvidia = kind(ComputeBackend.BackendType.OPENCL, DeviceKind.Vendor.NVIDIA);
        long small = 2048L * 1024;
        int three = Int8MatvecTuning.COUNTED_SMALL_MIN_TOKENS;
        for (GGMLType type : new GGMLType[]{GGMLType.Q8_0, GGMLType.Q4_K, GGMLType.Q6_K}) {
            assertFalse(type + " at 2 rows", Int8MatvecTuning.countedLaneIsFaster(vulkanNvidia, type, small, 2));
            assertTrue(type + " 3072x1024 at 2 rows", Int8MatvecTuning.countedLaneIsFaster(vulkanNvidia, type,
                    3072L * 1024, 2));
            assertTrue(type + " at 3 rows", Int8MatvecTuning.countedLaneIsFaster(vulkanNvidia, type, small, three));
            assertTrue(type + " at 16 rows", Int8MatvecTuning.countedLaneIsFaster(vulkanNvidia, type, small,
                    LlmKernelConstants.MATVEC_N_COUNTED_MAX));
            assertTrue(type + " large at 2 rows", Int8MatvecTuning.countedLaneIsFaster(vulkanNvidia, type,
                    Int8MatvecTuning.LANE_MIN_ELEMENTS, 2));
        }
        assertFalse(Int8MatvecTuning.countedLaneIsFaster(vulkanNvidia, GGMLType.Q5_K, small, three));
        assertFalse(Int8MatvecTuning.countedLaneIsFaster(openclNvidia, GGMLType.Q4_K, small, three));
        assertEquals(Int8MatvecTuning.countedLaneIsFaster(vulkanIntel, GGMLType.Q8_0, small),
                Int8MatvecTuning.countedLaneIsFaster(vulkanIntel, GGMLType.Q8_0, small, three));
    }

    @Test
    public void tuningPairsOnlyTheTypesThatMeasuredFasterPaired() {
        DeviceKind cudaTensor = DeviceKind.of(ComputeBackend.BackendType.CUDA, DeviceKind.Vendor.NVIDIA,
                DeviceClass.DISCRETE, DeviceKind.CudaBand.TENSOR);
        DeviceKind openclNvidia = kind(ComputeBackend.BackendType.OPENCL, DeviceKind.Vendor.NVIDIA);
        DeviceKind vulkanIntel = kind(ComputeBackend.BackendType.VULKAN, DeviceKind.Vendor.INTEL);
        DeviceKind vulkanNvidia = kind(ComputeBackend.BackendType.VULKAN, DeviceKind.Vendor.NVIDIA);
        DeviceKind openclIntel = kind(ComputeBackend.BackendType.OPENCL, DeviceKind.Vendor.INTEL);
        // Q4_K and Q5_K lost 1-8% paired on the RTX 5080's CUDA device (medians 0.95x and 0.96x);
        // Q8_K and Q8_0 measured neutral there (Q8_0 medians 0.99x) and Q8_0 never gained on its
        // OpenCL device either (median 0.97x); on its Vulkan device Q8_K measured neutral
        // (0.95-1.09x) while Q8_0 gained 4-8% at the 16.8-50M shapes; the Arrow-Lake iGPU's OpenCL
        // device lost 10-22% with Q5_K and Q6_K at every shape and measured neutral with Q8_0;
        // Q8_0's paired form measured 1.5-1.9x on the iGPU's Vulkan device.
        assertFalse(Int8MatvecTuning.pairedIsFaster(cudaTensor, GGMLType.Q4_K));
        assertFalse(Int8MatvecTuning.pairedIsFaster(cudaTensor, GGMLType.Q5_K));
        assertTrue(Int8MatvecTuning.pairedIsFaster(cudaTensor, GGMLType.Q6_K));
        assertFalse(Int8MatvecTuning.pairedIsFaster(cudaTensor, GGMLType.Q8_K));
        assertFalse(Int8MatvecTuning.pairedIsFaster(cudaTensor, GGMLType.Q8_0));
        assertTrue(Int8MatvecTuning.pairedIsFaster(openclNvidia, GGMLType.Q4_K));
        assertTrue(Int8MatvecTuning.pairedIsFaster(openclNvidia, GGMLType.Q6_K));
        assertFalse(Int8MatvecTuning.pairedIsFaster(openclNvidia, GGMLType.Q8_K));
        assertFalse(Int8MatvecTuning.pairedIsFaster(openclNvidia, GGMLType.Q8_0));
        assertTrue(Int8MatvecTuning.pairedIsFaster(vulkanIntel, GGMLType.Q8_0));
        assertTrue(Int8MatvecTuning.pairedIsFaster(vulkanIntel, GGMLType.Q8_K));
        assertTrue(Int8MatvecTuning.pairedIsFaster(vulkanNvidia, GGMLType.Q8_0));
        assertTrue(Int8MatvecTuning.pairedIsFaster(vulkanNvidia, GGMLType.Q4_K));
        assertFalse(Int8MatvecTuning.pairedIsFaster(vulkanNvidia, GGMLType.Q8_K));
        assertTrue(Int8MatvecTuning.pairedIsFaster(openclIntel, GGMLType.Q4_K));
        assertTrue(Int8MatvecTuning.pairedIsFaster(openclIntel, GGMLType.Q8_K));
        assertFalse(Int8MatvecTuning.pairedIsFaster(openclIntel, GGMLType.Q8_0));
        assertFalse(Int8MatvecTuning.pairedIsFaster(openclIntel, GGMLType.Q5_K));
        assertFalse(Int8MatvecTuning.pairedIsFaster(openclIntel, GGMLType.Q6_K));
        // A backend nobody measured, and a kind nobody measured, keep the paired form.
        assertTrue(Int8MatvecTuning.pairedIsFaster(DeviceKind.of(ComputeBackend.BackendType.HIP,
                DeviceKind.Vendor.AMD, DeviceClass.DISCRETE, DeviceKind.CudaBand.NONE), GGMLType.Q8_K));
        assertTrue(Int8MatvecTuning.pairedIsFaster(DeviceKind.of(ComputeBackend.BackendType.CUDA,
                DeviceKind.Vendor.OTHER, DeviceClass.DISCRETE, DeviceKind.CudaBand.TENSOR), GGMLType.Q4_K));
        assertTrue(Int8MatvecTuning.pairedIsFaster(cudaTensor, GGMLType.Q2_K));
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
        open(Int8MatvecMode.ON, DeviceFeature.INTEGER_DOT_PRODUCT, ComputeBackend.BackendType.CUDA,
                CUDA_ROW_TWINS);
        int rows = 1024;
        GpuWeight w = weight(GGMLType.Q4_K, rows);
        List<String> names = run(() -> ops.matvec(w, backend.allocateBuffer((long) COLS * 4), 0,
                backend.allocateBuffer((long) rows * 4), 0, 1));
        assertTrue(names.toString(), names.contains("Q4_K_MatVec_Q8"));
        assertFalse(names.toString(), names.contains("Q4_K_MatVec2_Q8"));
    }

    @Test
    public void measuredQ8_0PairingLossOnCudaKeepsTheSingleRowInt8Matvec() {
        open(Int8MatvecMode.ON, DeviceFeature.INTEGER_DOT_PRODUCT, ComputeBackend.BackendType.CUDA,
                CUDA_ROW_TWINS);
        int rows = 1024;
        GpuWeight w = weight(GGMLType.Q8_0, rows);
        List<String> names = run(() -> ops.matvec(w, backend.allocateBuffer((long) COLS * 4), 0,
                backend.allocateBuffer((long) rows * 4), 0, 1));
        assertTrue(names.toString(), names.contains("Q8_0_MatVec_Q8"));
        assertFalse(names.toString(), names.contains("Q8_0_MatVec2_Q8"));
    }

    /**
     * The counted two-row pairing answers per kind, size, token count and row count: the OpenCL NVIDIA
     * kind measured losses on the small side of the size split, so its small launches of Q8_0 and of
     * Q4_0, Q4_K, Q5_K and IQ4_XS keep the counted one-row form while their large ones pair (Q6_K
     * measured pairing wins at both sizes and pairs everywhere). At nine or more tokens the tall
     * projections of Q4_K, Q5_K and IQ4_XS measured the counted one-row form faster (Q4_K 9728x2560
     * at 16: 388.6 against the pair's 416.3us) while the 2560-row projections keep the pair (Q4_K
     * 2560x9728 at 9: 129.9 against 168.0), and Q8_0, Q4_0 and Q6_K keep the pair at every large
     * count; the CUDA kind pairs everywhere and the unmeasured kinds answer true.
     */
    @Test
    public void theCountedPairingFollowsTheMeasuredKindAndSize() {
        DeviceKind openclNvidia = DeviceKind.of(ComputeBackend.BackendType.OPENCL, DeviceKind.Vendor.NVIDIA,
                DeviceClass.DISCRETE, DeviceKind.CudaBand.NONE);
        assertFalse(Int8MatvecTuning.pairedCountedIsFaster(openclNvidia, GGMLType.Q8_0, 1024L * 1024, 4, 1024));
        assertTrue(Int8MatvecTuning.pairedCountedIsFaster(openclNvidia, GGMLType.Q8_0,
                SmallBatchCrossover.SHAPE_MIN_ELEMENTS, 4, 9728));
        for (GGMLType type : new GGMLType[]{GGMLType.Q4_0, GGMLType.Q4_K, GGMLType.Q5_K, GGMLType.IQ4_XS}) {
            assertFalse(type + " pairs the small side", Int8MatvecTuning.pairedCountedIsFaster(
                    openclNvidia, type, 1024L * 1024, 4, 1024));
            assertTrue(type + " pairs the large side", Int8MatvecTuning.pairedCountedIsFaster(
                    openclNvidia, type, SmallBatchCrossover.SHAPE_MIN_ELEMENTS, 4, 9728));
        }
        assertTrue(Int8MatvecTuning.pairedCountedIsFaster(openclNvidia, GGMLType.Q6_K, 1024L * 1024, 4, 1024));
        assertTrue(Int8MatvecTuning.pairedCountedIsFaster(openclNvidia, GGMLType.Q6_K,
                SmallBatchCrossover.SHAPE_MIN_ELEMENTS, 16, 151936));
        // The tall 4B-class projections of Q4_K, Q5_K and IQ4_XS flip to the counted one-row form
        // from nine tokens up; the same types' 2560-row projections and the low counts keep the pair.
        assertTrue(Int8MatvecTuning.pairedCountedIsFaster(openclNvidia, GGMLType.Q4_K,
                SmallBatchCrossover.SHAPE_MIN_ELEMENTS, 8, 9728));
        assertFalse(Int8MatvecTuning.pairedCountedIsFaster(openclNvidia, GGMLType.Q4_K,
                SmallBatchCrossover.SHAPE_MIN_ELEMENTS, 9, 9728));
        assertFalse(Int8MatvecTuning.pairedCountedIsFaster(openclNvidia, GGMLType.IQ4_XS,
                SmallBatchCrossover.SHAPE_MIN_ELEMENTS, 16, 151936));
        assertTrue(Int8MatvecTuning.pairedCountedIsFaster(openclNvidia, GGMLType.Q4_K,
                SmallBatchCrossover.SHAPE_MIN_ELEMENTS, 16, 2560));
        assertTrue(Int8MatvecTuning.pairedCountedIsFaster(openclNvidia, GGMLType.Q8_0,
                SmallBatchCrossover.SHAPE_MIN_ELEMENTS, 16, 9728));
        DeviceKind cudaTensor = DeviceKind.of(ComputeBackend.BackendType.CUDA, DeviceKind.Vendor.NVIDIA,
                DeviceClass.DISCRETE, DeviceKind.CudaBand.TENSOR);
        assertTrue(Int8MatvecTuning.pairedCountedIsFaster(cudaTensor, GGMLType.Q8_0, 1024L * 1024, 4, 1024));
        assertTrue(Int8MatvecTuning.pairedCountedIsFaster(cudaTensor, GGMLType.Q4_K,
                SmallBatchCrossover.SHAPE_MIN_ELEMENTS, 16, 9728));
        DeviceKind unmeasured = DeviceKind.of(ComputeBackend.BackendType.OPENCL, DeviceKind.Vendor.OTHER,
                DeviceClass.DISCRETE, DeviceKind.CudaBand.NONE);
        assertTrue(Int8MatvecTuning.pairedCountedIsFaster(unmeasured, GGMLType.Q8_0, 1024L * 1024, 4, 1024));
        // The dispatch reflects it: the measured OpenCL kind keeps the counted one-row form at the
        // small weight.
        open(Int8MatvecMode.ON, DeviceFeature.INTEGER_DOT_PRODUCT, ComputeBackend.BackendType.OPENCL,
                capabilities(ComputeBackend.BackendType.OPENCL, FakeComputeBackend.NVIDIA_VENDOR_ID,
                        DeviceClass.DISCRETE, DeviceFeature.INTEGER_DOT_PRODUCT));
        GpuWeight w = weight(GGMLType.Q8_0);
        List<String> names = run(() -> ops.matvec(w, backend.allocateBuffer((long) 4 * COLS * 4), 0,
                backend.allocateBuffer((long) 4 * ROWS * 4), 0, 4));
        assertTrue(names.toString(), names.contains("Q8_0_MatVecN4_Q8"));
        assertFalse(names.toString(), names.contains("Q8_0_MatVec2N4_Q8"));
    }

    @Test
    public void smallTokenGroupsQuantizeOnceAndUseTheMultiColumnInt8Matvec() {
        open(Int8MatvecMode.ON);
        GpuWeight w = q4k();
        int tokens = 4;
        List<String> names = run(() -> ops.matvec(w, backend.allocateBuffer((long) tokens * COLS * 4), 0,
                backend.allocateBuffer((long) tokens * ROWS * 4), 0, tokens));
        assertTrue(names.toString(), names.contains(GpuOps.QUANTIZE_Q8_1));
        /* The kind's exact-token-count twin serves the group (the counted form of the same launch);
         * its two-row twin stays off the small side of the measured pairing split. */
        assertTrue(names.toString(), names.contains("Q4_K_MatVecN4_Q8"));
        assertFalse(names.toString(), names.contains("Q4_K_MatVecN_Q8"));
        assertFalse(names.toString(), names.contains("Q4_K_MatVec2N4_Q8"));
        assertFalse(names.toString(), names.contains("Q4_K_MatVec_Q8"));
        assertFalse(names.toString(), names.stream().anyMatch(n -> n.startsWith("Q4_K_BatchMatMul")));
    }

    /**
     * An unmeasured device kind keeps the default dispatch: the integer-dot multi-column matvec up to
     * {@link LlmKernelConstants#MATVEC_N_DISPATCH_MAX} rows, the tiled GEMM above it. Every served
     * group runs the counted two-row Q8_0 form instead of the runtime-count kernel (the unmeasured
     * kind answers the pairing question true). The fast quantized-weight GEMM lane is off for the
     * dispatch ({@link MmqMode#PROPERTY}), since its kernel name selects by mode alone and would
     * otherwise cap the multi-column lane at the rows below the lane's first.
     */
    @Test
    public void exactTokenCountsUseTheCountedQ8MatvecForms() {
        String previousMmq = System.getProperty(MmqMode.PROPERTY);
        System.setProperty(MmqMode.PROPERTY, MmqMode.OFF.name().toLowerCase(Locale.ROOT));
        try {
            open(Int8MatvecMode.ON, DeviceFeature.INTEGER_DOT_PRODUCT, ComputeBackend.BackendType.OPENCL,
                    capabilities(ComputeBackend.BackendType.OPENCL, UNMEASURED_VENDOR_ID, DeviceClass.DISCRETE,
                            DeviceFeature.INTEGER_DOT_PRODUCT));
            GpuWeight w = weight(GGMLType.Q8_0);
            for (int tokens = 2; tokens <= LlmKernelConstants.MATVEC_N_MAX_TOKENS; tokens++) {
                final int group = tokens;
                List<String> names = run(() -> ops.matvec(w, backend.allocateBuffer((long) group * COLS * 4), 0,
                        backend.allocateBuffer((long) group * ROWS * 4), 0, group));
                if (tokens <= LlmKernelConstants.MATVEC_N_DISPATCH_MAX) {
                    assertTrue(names.toString(), names.contains("Q8_0_MatVec2N" + group + "_Q8"));
                    assertFalse(names.toString(), names.contains("Q8_0_MatVecN_Q8"));
                    assertFalse(names.toString(), names.contains("Q8_0_MatVecN" + group + "_Q8"));
                } else {
                    assertTrue(names.toString(), names.contains("LLM_Batch_Q8_0_MatMul_2D"));
                }
            }
        } finally {
            if (previousMmq == null) {
                System.clearProperty(MmqMode.PROPERTY);
            } else {
                System.setProperty(MmqMode.PROPERTY, previousMmq);
            }
        }
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
        List<FakeDispatch> ds = dispatches(() -> ops.matvec(w, backend.allocateBuffer((long) COLS * 4), 0,
                backend.allocateBuffer((long) rows * 4), 0, 1));
        List<String> names = kernelNames(ds);
        assertTrue(names.toString(), names.contains("Q4_K_MatVec2_Q8"));
        assertFalse(names.toString(), names.contains("Q4_K_MatVec_Q8"));
        FakeDispatch mv = ds.get(names.indexOf("Q4_K_MatVec2_Q8"));
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
                backend.allocateBuffer((long) COLS * 4), backend.allocateBuffer((long) rows * 4), 1));
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
            ops.gateUpGlu(gate, up, GluOp.SWIGLU, ActParams.NONE, in, out, 1);
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

    private List<FakeDispatch> dispatches(Runnable r) {
        backend.clearLog();
        r.run();
        return backend.dispatches();
    }

    private static List<String> kernelNames(List<FakeDispatch> ds) {
        List<String> names = new ArrayList<>();
        for (FakeDispatch d : ds) {
            names.add(d.kernelName());
        }
        return names;
    }
}

package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend.BackendType;
import com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities;
import com.eonmux.neuralfree.acceleration.backend.DeviceClass;
import com.eonmux.neuralfree.acceleration.backend.DeviceFeature;
import com.eonmux.neuralfree.acceleration.backend.DeviceIdentity;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.BatchInvariance;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.kernels.LlmKernelConstants;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuPlaceholders;
import com.eonmux.neuralfree.gguf.testutil.FakeComputeBackend;
import com.eonmux.neuralfree.gguf.testutil.FakeDispatch;
import org.junit.After;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * The measured small-batch crossover in action: the dispatch trace of {@link GpuOps#matmul} against
 * the host-only fake backend picks the lane {@link SmallBatchCrossover} measured for the fake
 * device's kind and weight type at every token count of the dispatch range, {@link
 * GpuOps#matvecFamilyServes} answers exactly when the dispatch takes a matvec-family kernel, and
 * batch invariance pins the multi-column lanes off like the single-token lane. The fake devices
 * claim the vendor and class of the machine's measured device per backend, so each backend's
 * small-model projections read its small-size-class entries; further fakes of the other measured
 * device kinds over the same backend, and of unknown vendors and classes, show an entry serving
 * only its own kind and unknown kinds keeping the default dispatch.
 */
public class SmallBatchCrossoverTest {

    private static final int COLS = 256;
    private static final int ROWS = 96;

    private FakeComputeBackend backend;
    private GpuScratch scratch;
    private GpuPlaceholders placeholders;
    private GpuOps ops;

    @After
    public void tearDown() {
        BatchInvariance.clearOverride();
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

    private void open(BackendType type) {
        open(type, capabilities(type));
    }

    private void open(BackendType type, DeviceCapabilities caps) {
        backend = FakeComputeBackend.builder().backendType(type).strictLlmArgs(true)
                .capabilities(caps).build();
        scratch = GpuScratch.builder(backend, 16).slot(ScratchSlots.X, 16L * COLS).build();
        placeholders = GpuPlaceholders.allocate(backend, backend.computeQueue());
        ops = new GpuOps(backend.computeQueue(), LlmKernels.open(backend), scratch, placeholders);
    }

    /**
     * A tensor-core device of the type with the packed integer dot the int8 lanes need. The vendor
     * and class default to the measured device of the backend (NVIDIA discrete over CUDA and
     * OpenCL, Intel integrated over Vulkan), so the dispatch reads that device kind's measured
     * entries; tests of the per-kind keying pass another vendor or class explicitly.
     */
    private static DeviceCapabilities capabilities(BackendType type) {
        if (type == BackendType.VULKAN) {
            return capabilities(type, FakeComputeBackend.INTEL_VENDOR_ID, DeviceClass.INTEGRATED);
        }
        return capabilities(type, FakeComputeBackend.NVIDIA_VENDOR_ID, DeviceClass.DISCRETE);
    }

    /** The same device of {@code type} claiming {@code vendorId} and {@code deviceClass}. */
    private static DeviceCapabilities capabilities(BackendType type, int vendorId, DeviceClass deviceClass) {
        DeviceCapabilities.Builder b = new DeviceCapabilities.Builder()
                .deviceName("CrossoverDevice")
                .totalVramBytes(8L << 30)
                .maxAllocSizeBytes(2L << 30)
                .computeUnits(32)
                .maxClockFrequencyMhz(1500)
                .maxWorkGroupSize(1024)
                .driverVersion("fake-1.0")
                .supportsFloat16(true)
                .supportsSubgroups(true)
                .maxLocalMemoryBytes(48L * 1024)
                .feature(DeviceFeature.COOPERATIVE_MATRIX)
                .feature(DeviceFeature.INTEGER_DOT_PRODUCT)
                .deviceClass(deviceClass)
                .identity(new DeviceIdentity("CrossoverDevice", vendorId, null, null));
        if (type == BackendType.CUDA) {
            b.computeCapability(120).feature(DeviceFeature.BF16_COOPERATIVE_MATRIX);
        } else {
            b.feature(DeviceFeature.STORAGE_8BIT).feature(DeviceFeature.STORAGE_16BIT)
                    .feature(DeviceFeature.TIMELINE_SEMAPHORE);
        }
        return b.build();
    }

    private GpuBuffer floats(long n) {
        return backend.allocateBuffer(n * Float.BYTES);
    }

    private GpuWeight weight(GGMLType type) {
        long bytes = (long) ROWS * (COLS / Math.max(1, type.getBlockSize()))
                * Math.max(1, type.getTypeSize());
        return GpuWeight.of("blk.0.attn_q.weight", type, new long[]{COLS, ROWS},
                backend.allocateBuffer(bytes, true));
    }

    /** The kernel names of one matmul of {@code type} over {@code n} token rows. */
    private List<String> run(GGMLType type, int n) {
        GpuWeight w = weight(type);
        GpuBuffer in = floats((long) n * COLS);
        GpuBuffer out = floats((long) n * ROWS);
        backend.clearLog();
        ops.matmul(w, in, 0, out, 0, n);
        List<String> names = new ArrayList<>();
        for (FakeDispatch d : backend.dispatches()) {
            names.add(d.kernelName());
        }
        return names;
    }

    /** Whether every dispatched kernel is of the plain matvec family (the single-row and batched matvecs). */
    private static boolean plainMatvecFamily(List<String> names) {
        for (String name : names) {
            boolean family = name.endsWith("_MatVec") || name.endsWith("_MatVec2")
                    || name.endsWith("_MatVec_Q8") || name.endsWith("_BatchMatMul_2D")
                    || name.equals("LLM_Batch_Q8_0_MatMul_2D");
            if (!family) {
                return false;
            }
        }
        return !names.isEmpty();
    }

    // ------------------------------------------------------------------ dispatch

    /**
     * OpenCL RTX 5080: the counted portable-integer matvec serves Q8_0 from two rows to the cap (the
     * forms ran the byte-wide integer arithmetic without the packed-dot feature and measured faster
     * than the float multi-column matvec at every count), the single-token matvec keeps one row, and
     * the float multi-column matvec serves F16.
     */
    @Test
    public void openclFloatMatvecNServesTheWholeMeasuredRange() {
        open(BackendType.OPENCL);
        assertEquals(Arrays.asList("Q8_0_MatVec"), run(GGMLType.Q8_0, 1));
        assertEquals(Arrays.asList("LLM_QuantizeQ8_1", "Q8_0_MatVecN2_Q8"), run(GGMLType.Q8_0, 2));
        assertEquals(Arrays.asList("LLM_QuantizeQ8_1", "Q8_0_MatVecN9_Q8"), run(GGMLType.Q8_0, 9));
        assertEquals(Arrays.asList("LLM_QuantizeQ8_1", "Q8_0_MatVecN16_Q8"),
                run(GGMLType.Q8_0, LlmKernelConstants.MATVEC_N_MAX_TOKENS));
        assertEquals(Arrays.asList("F16_MatVecN"), run(GGMLType.F16, 9));
    }

    /** The batched matvec keeps the row counts below the float matvec's measured wins when it is faster there. */
    @Test
    public void dispatchesServeTheBatchedMatvecWhereItMeasuredFastest() {
        open(BackendType.VULKAN);
        assertEquals(Arrays.asList("F16_BatchMatMul_2D"), run(GGMLType.F16, 2));
    }

    /**
     * Vulkan Intel iGPU: both multi-column matvecs lose to the batched matvec, so two rows take the
     * batched form, the cooperative-matrices lane keeps its crossover at four rows, and the F16 MMA
     * lane the measurement retired falls through to the tiled GEMM.
     */
    @Test
    public void vulkanServesTheBatchedMatvecWhereTheMatvecsLost() {
        open(BackendType.VULKAN);
        assertEquals(Arrays.asList("LLM_Batch_Q8_0_MatMul_2D"), run(GGMLType.Q8_0, 2));
        assertEquals(Arrays.asList("Q8_0_MatMul_CM"), run(GGMLType.Q8_0, 4));
        assertEquals(Arrays.asList("F16_BatchMatMul_2D"), run(GGMLType.F16, 2));
        assertEquals(Arrays.asList("F16_MatMul_Tiled"), run(GGMLType.F16, 8));
    }

    /**
     * CUDA: the counted two-row matvec serves Q8_0 at every measured count (its entry measured it
     * faster than the MMQ small tile at every one, so the MMQ lane keeps only the rows above the
     * multi-column cap); F16 keeps the batched matvec below MMA's crossover.
     */
    @Test
    public void cudaServesQ8_0WithTheCountedTwoRowMatvec() {
        open(BackendType.CUDA);
        assertEquals(Arrays.asList("LLM_QuantizeQ8_1", "Q8_0_MatVec2N2_Q8"), run(GGMLType.Q8_0, 2));
        assertEquals(Arrays.asList("LLM_QuantizeQ8_1", "Q8_0_MatVec2N3_Q8"), run(GGMLType.Q8_0, 3));
        assertEquals(Arrays.asList("LLM_QuantizeQ8_1", "Q8_0_MatVec2N4_Q8"), run(GGMLType.Q8_0, 4));
        assertEquals(Arrays.asList("F16_BatchMatMul_2D"), run(GGMLType.F16, 2));
        assertEquals(Arrays.asList("F16_MatMul_MMA"), run(GGMLType.F16, 4));
    }

    /**
     * Batch invariance pins the multi-column matvec lanes and the tiled GEMM off: a type with a fast
     * lane takes it at every row count (the MMQ staging and kernel pair here), a type without one
     * keeps the batched matvec at every row count.
     */
    @Test
    public void batchInvariancePinsTheMultiColumnLanesOff() {
        open(BackendType.OPENCL);
        BatchInvariance.setEnabled(Boolean.TRUE);
        try {
            assertEquals(Arrays.asList("LLM_Quant_MMQ_Q8_1", "Q8_0_MatMul_MMQ"), run(GGMLType.Q8_0, 4));
            assertEquals(Arrays.asList("LLM_Quant_MMQ_Q8_1", "Q8_0_MatMul_MMQ"), run(GGMLType.Q8_0, 8));
            assertEquals(Arrays.asList("F16_BatchMatMul_2D"), run(GGMLType.F16, 4));
            assertEquals(Arrays.asList("F16_BatchMatMul_2D"), run(GGMLType.F16, 8));
        } finally {
            BatchInvariance.clearOverride();
        }
    }

    // ------------------------------------------------------------------ device kinds

    /**
     * An entry serves only the device kind it was measured on: over one backend, the Intel
     * integrated fake and the NVIDIA discrete fake read different entries and dispatch differently
     * at the same weight, type and row count.
     */
    @Test
    public void anEntryServesOnlyTheDeviceKindItWasMeasuredOn() {
        // Vulkan: the Intel iGPU's entry retired the F16 MMA lane (the tiled GEMM serves 8 rows),
        // the NVIDIA dGPU's entry keeps it.
        open(BackendType.VULKAN);
        assertEquals(Arrays.asList("F16_MatMul_Tiled"), run(GGMLType.F16, 8));
        open(BackendType.VULKAN, capabilities(BackendType.VULKAN, FakeComputeBackend.NVIDIA_VENDOR_ID,
                DeviceClass.DISCRETE));
        assertEquals(Arrays.asList("F16_MatMul_MMA"), run(GGMLType.F16, 8));
        // The NVIDIA dGPU's paired counted integer-dot matvec serves a small weight from
        // Int8MatvecTuning.COUNTED_SMALL_MIN_TOKENS rows (it beat the batched matvec and the CM2 lane
        // there), and two rows keep the batched matvec, which won them on the small projections.
        assertEquals(Arrays.asList("LLM_Batch_Q8_0_MatMul_2D"), run(GGMLType.Q8_0, 2));
        assertEquals(Arrays.asList("LLM_QuantizeQ8_1", "Q8_0_MatVec2N4_Q8"), run(GGMLType.Q8_0, 4));
        // OpenCL: the NVIDIA dGPU's counted portable-integer matvec serves two rows (its measured
        // wins over the float multi-column matvec hold without the packed-dot feature), while the
        // Intel iGPU's entry keeps the batched matvec there (its measured Q8_0 loss keeps the
        // integer-dot matvec off this kind at every projection size) and its float multi-column
        // matvec serves the F16 counts from two rows up.
        open(BackendType.OPENCL);
        assertEquals(Arrays.asList("LLM_QuantizeQ8_1", "Q8_0_MatVecN2_Q8"), run(GGMLType.Q8_0, 2));
        open(BackendType.OPENCL, capabilities(BackendType.OPENCL, FakeComputeBackend.INTEL_VENDOR_ID,
                DeviceClass.INTEGRATED));
        assertEquals(Arrays.asList("LLM_Batch_Q8_0_MatMul_2D"), run(GGMLType.Q8_0, 2));
        assertEquals(Arrays.asList("F16_MatVecN"), run(GGMLType.F16, 8));
    }

    /** A device whose vendor or class is unknown matches no entry and keeps the default dispatch. */
    @Test
    public void anUnknownVendorOrClassKeepsTheDefaultDispatch() {
        open(BackendType.VULKAN, capabilities(BackendType.VULKAN, 0x1234, DeviceClass.DISCRETE));
        assertEquals(Arrays.asList("LLM_QuantizeQ8_1", "Q8_0_MatVec2N2_Q8"), run(GGMLType.Q8_0, 2));
        assertEquals(Arrays.asList("Q8_0_MatMul_CM"), run(GGMLType.Q8_0, 4));
        assertEquals(Arrays.asList("F16_MatMul_MMA"), run(GGMLType.F16, 8));
        open(BackendType.OPENCL, capabilities(BackendType.OPENCL, FakeComputeBackend.NVIDIA_VENDOR_ID,
                DeviceClass.UNKNOWN));
        // An unknown class matches no tuning or crossover entry, so the device's packed dot
        // products decide and the integer-dot matvec serves the rows below the MMQ lane's first.
        assertEquals(Arrays.asList("LLM_QuantizeQ8_1", "Q8_0_MatVec2N2_Q8"), run(GGMLType.Q8_0, 2));
        assertEquals(Arrays.asList("LLM_Quant_MMQ_Q8_1", "Q8_0_MatMul_MMQ"), run(GGMLType.Q8_0, 8));
    }

    // ------------------------------------------------------------------ family mirror

    /** matvecFamilyServes answers exactly when the dispatch takes a plain matvec-family kernel. */
    @Test
    public void matvecFamilyServesMirrorsTheDispatch() {
        for (BackendType type : new BackendType[]{BackendType.OPENCL, BackendType.VULKAN, BackendType.CUDA}) {
            open(type);
            for (GGMLType weightType : new GGMLType[]{GGMLType.Q8_0, GGMLType.F16}) {
                GpuWeight w = weight(weightType);
                for (int n = 1; n <= LlmKernelConstants.MATVEC_N_MAX_TOKENS + 1; n++) {
                    List<String> names = run(weightType, n);
                    assertEquals(type + " " + weightType + " n=" + n + " dispatch " + names,
                            plainMatvecFamily(names), ops.matvecFamilyServes(w, n));
                }
            }
        }
    }

    // ------------------------------------------------------------------ decision table

    /** The measured decision table itself: each entry's answers, and the defaults without an entry. */
    @Test
    public void theDecisionTableHoldsTheMeasuredAnswers() {
        long small = 1024L * 4096;
        long large = 2560L * 19456;
        DeviceKind openclNvidia = kind(BackendType.OPENCL, DeviceKind.Vendor.NVIDIA, DeviceClass.DISCRETE);
        DeviceKind vulkanIntel = kind(BackendType.VULKAN, DeviceKind.Vendor.INTEL, DeviceClass.INTEGRATED);
        DeviceKind cudaNvidia = kind(BackendType.CUDA, DeviceKind.Vendor.NVIDIA, DeviceClass.DISCRETE);
        DeviceKind vulkanNvidia = kind(BackendType.VULKAN, DeviceKind.Vendor.NVIDIA, DeviceClass.DISCRETE);
        DeviceKind openclIntel = kind(BackendType.OPENCL, DeviceKind.Vendor.INTEL, DeviceClass.INTEGRATED);
        assertEquals(SmallBatchCrossover.FLOAT_MATVEC_N_DISPATCH_MAX,
                SmallBatchCrossover.floatMatvecNMax(openclNvidia, GGMLType.Q8_0, small));
        assertEquals(SmallBatchCrossover.FLOAT_MATVEC_N_DISPATCH_MAX,
                SmallBatchCrossover.floatMatvecNMax(openclNvidia, GGMLType.F16, small));
        assertEquals(SmallBatchCrossover.FLOAT_MATVEC_N_DISPATCH_MAX + 1,
                SmallBatchCrossover.fastGemmMinTokens(openclNvidia, GGMLType.Q8_0, small));
        assertEquals(SmallBatchCrossover.FLOAT_MATVEC_N_DISPATCH_MAX,
                SmallBatchCrossover.floatMatvecNMax(openclNvidia, GGMLType.Q8_0, large));
        assertEquals(0, SmallBatchCrossover.floatMatvecNMax(vulkanIntel, GGMLType.Q8_0, small));
        assertEquals(1, SmallBatchCrossover.int8MatvecNMax(vulkanIntel, GGMLType.Q8_0, small, true));
        assertEquals(0, SmallBatchCrossover.floatMatvecNMax(vulkanIntel, GGMLType.F16, small));
        assertEquals(SmallBatchCrossover.FLOAT_MATVEC_N_DISPATCH_MAX + 1,
                SmallBatchCrossover.fastGemmMinTokens(vulkanIntel, GGMLType.F16, small));
        assertEquals(SmallBatchCrossover.DEFAULT_FAST_GEMM_MIN_TOKENS,
                SmallBatchCrossover.fastGemmMinTokens(vulkanIntel, GGMLType.Q8_0, small));
        assertEquals(0, SmallBatchCrossover.floatMatvecNMax(cudaNvidia, GGMLType.Q8_0, small));
        assertEquals(SmallBatchCrossover.FLOAT_MATVEC_N_DISPATCH_MAX,
                SmallBatchCrossover.int8MatvecNMax(cudaNvidia, GGMLType.Q8_0, small, true));
        assertEquals(SmallBatchCrossover.FLOAT_MATVEC_N_DISPATCH_MAX + 1,
                SmallBatchCrossover.fastGemmMinTokens(cudaNvidia, GGMLType.Q8_0, large));
        assertEquals(0, SmallBatchCrossover.floatMatvecNMax(cudaNvidia, GGMLType.Q8_0, large));
        assertEquals(SmallBatchCrossover.FLOAT_MATVEC_N_DISPATCH_MAX + 1,
                SmallBatchCrossover.fastGemmMinTokens(cudaNvidia, GGMLType.Q8_0, small));
        // Vulkan NVIDIA discrete: the counted integer-dot matvec serves the rows where it measured
        // faster than the CM2 lane (per type and size class), the lane the rest.
        assertEquals(SmallBatchCrossover.DEFAULT_FAST_GEMM_MIN_TOKENS,
                SmallBatchCrossover.fastGemmMinTokens(vulkanNvidia, GGMLType.Q8_0, small));
        assertEquals(8, SmallBatchCrossover.int8MatvecNMax(vulkanNvidia, GGMLType.Q8_0, small, true));
        assertEquals(0, SmallBatchCrossover.floatMatvecNMax(vulkanNvidia, GGMLType.Q8_0, small));
        assertEquals(SmallBatchCrossover.DEFAULT_FAST_GEMM_MIN_TOKENS,
                SmallBatchCrossover.fastGemmMinTokens(vulkanNvidia, GGMLType.Q8_0, large));
        assertEquals(14, SmallBatchCrossover.int8MatvecNMax(vulkanNvidia, GGMLType.Q8_0, large, true));
        assertEquals(14, SmallBatchCrossover.int8MatvecNMax(vulkanNvidia, GGMLType.Q4_K, small, true));
        assertEquals(LlmKernelConstants.MATVEC_N_COUNTED_MAX,
                SmallBatchCrossover.int8MatvecNMax(vulkanNvidia, GGMLType.Q4_K, large, true));
        assertEquals(10, SmallBatchCrossover.int8MatvecNMax(vulkanNvidia, GGMLType.Q6_K, small, true));
        // Its tiny shapes (below TINY_SHAPE_MAX_ELEMENTS) have their own entry: the counted lane serves the
        // whole counted range on Q8_0 and Q4_K and 14 rows on Q6_K; a kind without one keeps its small side.
        long tiny = 3072L * 1024;
        assertEquals(LlmKernelConstants.MATVEC_N_COUNTED_MAX,
                SmallBatchCrossover.int8MatvecNMax(vulkanNvidia, GGMLType.Q8_0, tiny, true));
        assertEquals(LlmKernelConstants.MATVEC_N_COUNTED_MAX,
                SmallBatchCrossover.int8MatvecNMax(vulkanNvidia, GGMLType.Q4_K, tiny, true));
        assertEquals(14, SmallBatchCrossover.int8MatvecNMax(vulkanNvidia, GGMLType.Q6_K, tiny, true));
        assertEquals(SmallBatchCrossover.int8MatvecNMax(cudaNvidia, GGMLType.Q4_K, small, true),
                SmallBatchCrossover.int8MatvecNMax(cudaNvidia, GGMLType.Q4_K, tiny, true));
        assertEquals(LlmKernelConstants.MATVEC_N_COUNTED_MAX,
                SmallBatchCrossover.int8MatvecNMax(vulkanNvidia, GGMLType.Q6_K, large, true));
        assertEquals(SmallBatchCrossover.DEFAULT_FAST_GEMM_MIN_TOKENS,
                SmallBatchCrossover.fastGemmMinTokens(vulkanNvidia, GGMLType.F16, small));
        assertEquals(0, SmallBatchCrossover.floatMatvecNMax(vulkanNvidia, GGMLType.F16, small));
        // OpenCL Intel integrated: the integer-dot matvec keeps the rows below the MMQ lane and the
        // lane keeps the default crossover on Q8_0; on F16 the float multi-column matvec serves the
        // whole measured range and the fast-lane answer is never.
        assertEquals(SmallBatchCrossover.DEFAULT_FAST_GEMM_MIN_TOKENS,
                SmallBatchCrossover.fastGemmMinTokens(openclIntel, GGMLType.Q8_0, small));
        assertEquals(SmallBatchCrossover.DEFAULT_FAST_GEMM_MIN_TOKENS - 1,
                SmallBatchCrossover.int8MatvecNMax(openclIntel, GGMLType.Q8_0, small, true));
        assertEquals(0, SmallBatchCrossover.floatMatvecNMax(openclIntel, GGMLType.Q8_0, small));
        assertEquals(SmallBatchCrossover.DEFAULT_FAST_GEMM_MIN_TOKENS,
                SmallBatchCrossover.fastGemmMinTokens(openclIntel, GGMLType.Q8_0, large));
        assertEquals(SmallBatchCrossover.DEFAULT_FAST_GEMM_MIN_TOKENS - 1,
                SmallBatchCrossover.int8MatvecNMax(openclIntel, GGMLType.Q8_0, large, true));
        assertEquals(SmallBatchCrossover.FLOAT_MATVEC_N_DISPATCH_MAX,
                SmallBatchCrossover.floatMatvecNMax(openclIntel, GGMLType.F16, small));
        assertEquals(SmallBatchCrossover.FLOAT_MATVEC_N_DISPATCH_MAX + 1,
                SmallBatchCrossover.fastGemmMinTokens(openclIntel, GGMLType.F16, small));
        assertTrue(SmallBatchCrossover.hasFloatMatvecN(openclIntel, GGMLType.F16));
        assertFalse(SmallBatchCrossover.hasFloatMatvecN(vulkanNvidia, GGMLType.Q8_0));
        assertFalse(SmallBatchCrossover.hasFloatMatvecN(vulkanNvidia, GGMLType.F16));
        // A kind nobody measured (unknown vendor or class over a measured backend) matches no entry.
        DeviceKind unknownVendor = kind(BackendType.VULKAN, DeviceKind.Vendor.OTHER, DeviceClass.DISCRETE);
        DeviceKind unknownClass = kind(BackendType.OPENCL, DeviceKind.Vendor.NVIDIA, DeviceClass.UNKNOWN);
        assertEquals(SmallBatchCrossover.DEFAULT_FAST_GEMM_MIN_TOKENS,
                SmallBatchCrossover.fastGemmMinTokens(unknownVendor, GGMLType.Q8_0, small));
        assertEquals(LlmKernelConstants.MATVEC_N_DISPATCH_MAX,
                SmallBatchCrossover.int8MatvecNMax(unknownVendor, GGMLType.Q8_0, small, false));
        assertEquals(SmallBatchCrossover.DEFAULT_FAST_GEMM_MIN_TOKENS - 1,
                SmallBatchCrossover.int8MatvecNMax(unknownVendor, GGMLType.Q8_0, small, true));
        assertEquals(0, SmallBatchCrossover.floatMatvecNMax(unknownVendor, GGMLType.Q8_0, small));
        assertEquals(SmallBatchCrossover.DEFAULT_FAST_GEMM_MIN_TOKENS,
                SmallBatchCrossover.fastGemmMinTokens(unknownClass, GGMLType.Q8_0, small));
        assertEquals(0, SmallBatchCrossover.floatMatvecNMax(unknownClass, GGMLType.F16, small));
        // CUDA sm_80+ measured entries for the Q4_K and Q4_0 projections: the counted integer-dot
        // forms beat the MMQ lanes at every count of 2..16 on both size classes (4096x1024, medians
        // of nine interleaved rounds: Q4_K 21.9-35.7us against the tile's 37.0-54.4 over 9..16),
        // so the matvec serves the whole counted range and the tile never takes a projection.
        assertEquals(SmallBatchCrossover.DEFAULT_FAST_GEMM_MIN_TOKENS,
                SmallBatchCrossover.fastGemmMinTokens(cudaNvidia, GGMLType.Q4_K, small));
        assertEquals(0, SmallBatchCrossover.floatMatvecNMax(cudaNvidia, GGMLType.Q4_K, small));
        assertEquals(LlmKernelConstants.MATVEC_N_COUNTED_MAX,
                SmallBatchCrossover.int8MatvecNMax(cudaNvidia, GGMLType.Q4_K, small, false));
        assertEquals(LlmKernelConstants.MATVEC_N_COUNTED_MAX,
                SmallBatchCrossover.int8MatvecNMax(cudaNvidia, GGMLType.Q4_K, small, true));
        assertEquals(SmallBatchCrossover.FLOAT_MATVEC_N_DISPATCH_MAX + 1,
                SmallBatchCrossover.fastGemmMinTokens(cudaNvidia, GGMLType.Q4_0, small));
        assertEquals(SmallBatchCrossover.FLOAT_MATVEC_N_DISPATCH_MAX,
                SmallBatchCrossover.int8MatvecNMax(cudaNvidia, GGMLType.Q4_0, small, true));
        assertEquals(SmallBatchCrossover.DEFAULT_FAST_GEMM_MIN_TOKENS,
                SmallBatchCrossover.fastGemmMinTokens(cudaNvidia, GGMLType.Q4_0, large));
        assertEquals(LlmKernelConstants.MATVEC_N_COUNTED_MAX,
                SmallBatchCrossover.int8MatvecNMax(cudaNvidia, GGMLType.Q4_0, large, true));
        assertTrue(SmallBatchCrossover.hasFloatMatvecN(openclNvidia, GGMLType.Q8_0));
        assertFalse(SmallBatchCrossover.hasFloatMatvecN(cudaNvidia, GGMLType.Q8_0));
        assertFalse(SmallBatchCrossover.hasFloatMatvecN(cudaNvidia, GGMLType.Q4_K));
        // The measured OpenCL NVIDIA entry hands the counted cap's rows to the matvec (the counted
        // forms won the 4B-class shapes summed over the measured projections at every count of
        // 9..15: Q4_K 2560x9728 129.9-208.1us against the fast lane's 730.0-805.4) and keeps the
        // MMQ fast lane above it — the 16-row prefill measured the lane faster at step level (71.1
        // against 75.1ms on the 4B Q4_K_M), so the large side stops one row below the counted cap.
        assertEquals(SmallBatchCrossover.DEFAULT_FAST_GEMM_MIN_TOKENS,
                SmallBatchCrossover.fastGemmMinTokens(openclNvidia, GGMLType.Q4_K, small));
        assertEquals(LlmKernelConstants.MATVEC_N_COUNTED_MAX,
                SmallBatchCrossover.int8MatvecNMax(openclNvidia, GGMLType.Q4_K, small, false));
        assertEquals(LlmKernelConstants.MATVEC_N_COUNTED_MAX - 1,
                SmallBatchCrossover.int8MatvecNMax(openclNvidia, GGMLType.Q4_K, large, true));
    }

    /** The device kind of one backend's vendor and class, banded on the tensor side of sm_80 for CUDA. */
    private static DeviceKind kind(BackendType type, DeviceKind.Vendor vendor, DeviceClass deviceClass) {
        return DeviceKind.of(type, vendor, deviceClass,
                type == BackendType.CUDA ? DeviceKind.CudaBand.TENSOR : DeviceKind.CudaBand.NONE);
    }
}

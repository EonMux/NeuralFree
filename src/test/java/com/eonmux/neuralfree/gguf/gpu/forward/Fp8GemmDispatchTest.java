package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuPlaceholders;
import com.eonmux.neuralfree.gguf.ops.gemm.GemmKernels;
import com.eonmux.neuralfree.gguf.quant.ActQuantScheme;
import com.eonmux.neuralfree.gguf.quant.InputTransform;
import com.eonmux.neuralfree.gguf.testutil.FakeComputeBackend;
import org.junit.After;
import org.junit.Test;

import java.util.List;
import java.util.stream.Collectors;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * The E4M3 lane dispatch: a weight whose checkpoint activation scheme is staged in the E4M3 code form runs the FP8
 * tensor-core GEMM on CUDA sm_89+ (the FP8 weight types over {@code <T>_MatMul_MMQ}, the 4-bit ones over
 * {@code <T>_MatMul_W4A8}), staging the scheme's codes first and skipping the fake-quantization pass. Below sm_89,
 * on the other backends, and for a scheme the lane does not stage, the exact tiled GEMM with the fake-quantization
 * pass serves the projection.
 */
public class Fp8GemmDispatchTest {

    /**
     * Compute capability of the {@link #open} fake CUDA backend: every build of the CUDA kernels covers this one
     * (a PTX image of compute capability N serves every device of N and above), so the dispatch tests need no
     * device. {@link #theFp8GateStartsAtSm89} checks the threshold itself.
     */
    private static final int CUDA_CAPABILITY = 120;

    private static final int TOKENS = 16;
    private static final int COLS = 256;
    private static final int ROWS = 96;

    private FakeComputeBackend backend;
    private GpuScratch scratch;
    private GpuPlaceholders placeholders;
    private GpuOps ops;

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
        backend = null;
        scratch = null;
        placeholders = null;
        ops = null;
    }

    private FakeComputeBackend open(ComputeBackend.BackendType type, String... unavailable) {
        backend = FakeComputeBackend.builder().backendType(type).capabilities(caps())
                .unavailableKernels(unavailable).strictLlmArgs(true).build();
        scratch = GpuScratch.builder(backend, TOKENS).slot(ScratchSlots.X, (long) TOKENS * COLS).build();
        placeholders = GpuPlaceholders.allocate(backend, backend.computeQueue());
        ops = new GpuOps(backend.computeQueue(), LlmKernels.open(backend), scratch, placeholders);
        return backend;
    }

    private static DeviceCapabilities caps() {
        return new DeviceCapabilities.Builder()
                .deviceName("Fp8LaneDevice" + CUDA_CAPABILITY)
                .totalVramBytes(8L << 30)
                .maxAllocSizeBytes(2L << 30)
                .computeUnits(32)
                .maxClockFrequencyMhz(1500)
                .maxWorkGroupSize(1024)
                .supportsAsyncTransfer(true)
                .driverVersion("fake-1.0")
                .supportsFloat16(true)
                .supportsSubgroups(true)
                .maxLocalMemoryBytes(48L * 1024)
                .maxWorkGroupCount(new long[]{65535, 65535, 65535})
                .computeCapability(CUDA_CAPABILITY)
                .build();
    }

    @Test
    public void theFp8GateStartsAtSm89() {
        assertTrue(FastGemmLanes.fp8Tensor(ComputeBackend.BackendType.CUDA, 89));
        assertTrue(FastGemmLanes.fp8Tensor(ComputeBackend.BackendType.CUDA, CUDA_CAPABILITY));
        assertFalse("sm_88 has no E4M3 tensor-core multiply",
                FastGemmLanes.fp8Tensor(ComputeBackend.BackendType.CUDA, 88));
        assertFalse(FastGemmLanes.fp8Tensor(ComputeBackend.BackendType.OPENCL, 120));
        assertFalse(FastGemmLanes.fp8Tensor(ComputeBackend.BackendType.VULKAN, 120));
    }

    @Test
    public void fp8WeightsRunTheE4m3GemmOnSm89() {
        open(ComputeBackend.BackendType.CUDA);
        List<String> names = run(GGMLType.F8_E4M3_B128, ActQuantScheme.FP8_STATIC);
        assertEquals("the scheme's E4M3 codes are staged first", GemmKernels.QUANT_MMQ_FP8_STATIC, names.get(0));
        assertTrue(names.toString(), names.contains("F8_E4M3_B128_MatMul_MMQ"));
        assertTrue("the fake-quantization pass is skipped: " + names, !names.contains(GpuOps.ACT_FAKE_QUANT));
        assertTrue("the int8 staging pass is skipped: " + names, !names.contains(GemmKernels.QUANT_MMQ_Q8_1));
    }

    @Test
    public void aPerRunSchemaStagesItsOwnGeometry() {
        open(ComputeBackend.BackendType.CUDA);
        List<String> names = run(GGMLType.MXFP8, ActQuantScheme.FP8_GROUP128);
        assertEquals(GemmKernels.QUANT_MMQ_FP8_GROUP128, names.get(0));
        assertTrue(names.toString(), names.contains("MXFP8_MatMul_MMQ"));
    }

    @Test
    public void aPowerOfTwoSchemeStagesWithoutAScalar() {
        open(ComputeBackend.BackendType.CUDA);
        List<FakeComputeBackend.Dispatch> dispatched = dispatch(GGMLType.MXFP8, ActQuantScheme.MXFP8);
        assertEquals(GemmKernels.QUANT_MMQ_FP8_MX, dispatched.get(0).kernelName());
        assertEquals("shape, input and records: the per-run exponent needs no scalar",
                7, dispatched.get(0).argCount());
        assertEquals("the static scheme passes its input scale", 8,
                dispatch(GGMLType.MXFP8, ActQuantScheme.FP8_STATIC).get(0).argCount());
    }

    @Test
    public void theTiledGemmKeepsTheSchemeWithoutFp8TensorCores() {
        open(ComputeBackend.BackendType.OPENCL);
        List<String> names = run(GGMLType.MXFP8, ActQuantScheme.MXFP8);
        assertTrue(names.toString(), names.contains("MXFP8_MatMul_Tiled"));
        assertTrue("the fake-quantization pass keeps the scheme: " + names, names.contains(GpuOps.ACT_FAKE_QUANT));
        assertTrue(names.toString(), !names.contains("MXFP8_MatMul_MMQ"));
    }

    @Test
    public void anUnavailableStagingKernelFallsBackToTheTiledGemm() {
        open(ComputeBackend.BackendType.CUDA, GemmKernels.QUANT_MMQ_FP8_STATIC);
        List<String> names = run(GGMLType.F8_E4M3_B128, ActQuantScheme.FP8_STATIC);
        assertTrue("the tiled GEMM (hardware FP8 rows on sm_89+) serves the product: " + names,
                names.stream().anyMatch(n -> n.startsWith("F8_E4M3_B128_") && n.endsWith("_MatMul_Tiled")));
        assertTrue(names.toString(), names.contains(GpuOps.ACT_FAKE_QUANT));
        assertTrue(names.toString(), !names.contains("F8_E4M3_B128_MatMul_MMQ"));
    }

    @Test
    public void fourBitWeightsRunTheW4a8GemmOnSm89() {
        open(ComputeBackend.BackendType.CUDA);
        List<String> names = run(GGMLType.MXFP4, ActQuantScheme.FP8_STATIC);
        assertEquals(GemmKernels.QUANT_MMQ_FP8_STATIC, names.get(0));
        assertTrue(names.toString(), names.contains("MXFP4_MatMul_W4A8"));
        assertTrue("the fake-quantization pass is skipped: " + names, !names.contains(GpuOps.ACT_FAKE_QUANT));
    }

    @Test
    public void fourBitWeightsKeepTheirInt8LaneForAnInt8Scheme() {
        open(ComputeBackend.BackendType.CUDA);
        List<String> names = run(GGMLType.IQ4_XS, ActQuantScheme.INT8_TOKEN);
        assertTrue(names.toString(), names.contains("IQ4_XS_MatMul_MMQ"));
        assertTrue(names.toString(), !names.contains("IQ4_XS_MatMul_W4A8"));
        assertTrue(names.toString(), names.contains(GemmKernels.QUANT_MMQ_INT8_ROW));
    }

    @Test
    public void asymInt8StagesTheSchemeDirectly() {
        open(ComputeBackend.BackendType.CUDA);
        List<String> names = run(GGMLType.I8_B128, ActQuantScheme.INT8_STATIC_ASYM);
        assertEquals(GemmKernels.QUANT_MMQ_INT8_ASYM, names.get(0));
        assertTrue(names.toString(), names.contains("I8_B128_MatMul_MMQ"));
        assertTrue(names.toString(), !names.contains(GpuOps.ACT_FAKE_QUANT));
    }

    private List<String> run(GGMLType type, ActQuantScheme scheme) {
        return dispatch(type, scheme).stream().map(FakeComputeBackend.Dispatch::kernelName)
                .collect(Collectors.toList());
    }

    private List<FakeComputeBackend.Dispatch> dispatch(GGMLType type, ActQuantScheme scheme) {
        GpuBuffer weight = backend.allocateBuffer((long) ROWS * type.getBytesForElements(COLS), true);
        GpuBuffer input = backend.allocateBuffer((long) TOKENS * COLS * Float.BYTES);
        GpuBuffer output = backend.allocateBuffer((long) TOKENS * ROWS * Float.BYTES);
        try {
            backend.clearLog();
            ops.matmul(weight(type, weight, scheme), input, 0, output, 0, TOKENS);
            return backend.dispatches();
        } finally {
            weight.close();
            input.close();
            output.close();
        }
    }

    private static GpuWeight weight(GGMLType type, GpuBuffer weight, ActQuantScheme scheme) {
        float[] scales = scheme.isStatic() ? new float[]{0.125f} : null;
        int[] zps = scheme.needsZeroPoint() ? new int[]{5} : null;
        InputTransform.ActQuant aq = InputTransform.ActQuant.of(scheme, scales, zps, Float.POSITIVE_INFINITY);
        return GpuWeight.of("test.weight", type, new long[]{COLS, ROWS}, weight)
                .withInputTransform(InputTransform.of(null, aq, null));
    }
}

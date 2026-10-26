package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuPlaceholders;
import com.eonmux.neuralfree.gguf.testutil.FakeComputeBackend;
import org.junit.After;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * The hardware-FP8 fast paths of the extension types ({@code F8_E5M2}, {@code F8_E4M3},
 * {@code F8_E4M3_B128} and {@code MXFP8}): GpuOps dispatches a {@code {T}_Hw_*} row only on a CUDA
 * device whose reported compute capability is sm_89 or higher, and the generic rows everywhere else
 * (an OpenCL device never selects them). The {@code _Hw} rows keep the argument lists of the generic
 * rows and decode with the same values on every finite code ({@code kernels/llm/common/quant_decode_ext.h}
 * documents the hardware pair conversions), so the fast path never changes an output.
 */
public class ExtTypeFastPathTest {

    private static final int COLS = 256;
    private static final int ROWS = 96;
    private static final int UBATCH = 4;
    private static final GGMLType[] HW_TYPES = {
        GGMLType.F8_E5M2, GGMLType.F8_E4M3, GGMLType.F8_E4M3_B128, GGMLType.MXFP8
    };

    private FakeComputeBackend backend;
    private GpuScratch scratch;
    private GpuPlaceholders placeholders;
    private GpuOps ops;
    private int local;

    private void open(ComputeBackend.BackendType type, int computeCapability) {
        backend = FakeComputeBackend.builder().backendType(type).capabilities(caps(computeCapability))
                .strictLlmArgs(true).build();
        scratch = GpuScratch.builder(backend, UBATCH).slot(ScratchSlots.X, (long) UBATCH * COLS).build();
        placeholders = GpuPlaceholders.allocate(backend, backend.computeQueue());
        ops = new GpuOps(backend.computeQueue(), LlmKernels.open(backend), scratch, placeholders);
        local = ops.local();
    }

    private static DeviceCapabilities caps(int computeCapability) {
        return new DeviceCapabilities.Builder()
                .deviceName("Fp8Device" + computeCapability)
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
                .computeCapability(computeCapability)
                .build();
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
        backend = null;
        scratch = null;
        placeholders = null;
        ops = null;
    }

    /** A single-buffer weight of {@code type} (1 byte per element on 32-element blocks or its block form). */
    private GpuWeight weight(GGMLType type) {
        long rowBytes = type.getBytesForElements(COLS);
        return GpuWeight.of("blk.0.attn_q.weight", type, new long[]{COLS, ROWS},
                backend.allocateBuffer(rowBytes * ROWS, true));
    }

    private long roundUp(long n) {
        return (n + local - 1) / local * local;
    }

    private List<String> dispatched() {
        List<String> names = new ArrayList<>();
        for (FakeComputeBackend.Dispatch d : backend.dispatches()) {
            names.add(d.kernelName());
        }
        return names;
    }

    // ------------------------------------------------------------------ fast-path selection

    @Test
    public void aSm89CudaDeviceDispatchesTheHwRowsOfEveryFp8Type() {
        for (GGMLType type : HW_TYPES) {
            open(ComputeBackend.BackendType.CUDA, 90);
            GpuWeight w = weight(type);
            GpuBuffer in = backend.allocateBuffer((long) COLS * Float.BYTES, true);
            GpuBuffer out = backend.allocateBuffer((long) ROWS * Float.BYTES, true);
            backend.clearLog();
            ops.matvec(w, in, 0, out, 0, 1);
            List<String> names = dispatched();
            assertEquals("single-token projection of " + type,
                    Arrays.asList(type.name() + "_Hw_MatVec"), names);
            ops.close();
            ops = null;
        }
    }

    @Test
    public void aSm80CudaDeviceKeepsTheGenericRows() {
        for (GGMLType type : HW_TYPES) {
            open(ComputeBackend.BackendType.CUDA, 80);
            ops.matvec(weight(type), backend.allocateBuffer((long) COLS * Float.BYTES, true), 0,
                    backend.allocateBuffer((long) ROWS * Float.BYTES, true), 0, 1);
            assertEquals("single-token projection of " + type + " on sm_80",
                    Arrays.asList(type.name() + "_MatVec"), dispatched());
            ops.close();
            ops = null;
        }
    }

    @Test
    public void anOpenClDeviceKeepsTheGenericRowsWhateverItsCapability() {
        for (GGMLType type : HW_TYPES) {
            open(ComputeBackend.BackendType.OPENCL, 90);
            ops.matvec(weight(type), backend.allocateBuffer((long) COLS * Float.BYTES, true), 0,
                    backend.allocateBuffer((long) ROWS * Float.BYTES, true), 0, 1);
            assertEquals("single-token projection of " + type + " on OpenCL",
                    Arrays.asList(type.name() + "_MatVec"), dispatched());
            ops.close();
            ops = null;
        }
    }

    // ------------------------------------------------------------------ fast-path launches

    /**
     * The fast-path launches keep the generic rows' geometry: the tiled GEMM at {@code Type_Hw_MatMul_Tiled}
     * and the embedding row lookups at {@code Type_Hw_*EmbeddingLookup}.
     */
    @Test
    public void theHwRowsLaunchWithTheGenericGeometryAndSignatures() {
        GGMLType t = GGMLType.F8_E5M2;
        open(ComputeBackend.BackendType.CUDA, 90);
        GpuBuffer in = backend.allocateBuffer((long) COLS * Float.BYTES, true);
        GpuBuffer out = backend.allocateBuffer((long) ROWS * Float.BYTES, true);
        ops.matmul(weight(t), in, 0, out, 0, 1);

        // the embedding lookups prepare and launch on the _Hw rows as well
        GpuWeight table = weight(t);
        backend.clearLog();
        ops.embed(table, new int[]{0, 1}, out, 0);
        List<String> names = dispatched();
        assertTrue("host-id lookup uses the _Hw row", names.contains(t.name() + "_Hw_EmbeddingLookup")
                || names.contains(t.name() + "_Hw_BatchEmbeddingLookup"));

        // and the multi-token batch runs the _Hw _2D kernel at the generic geometry
        backend.clearLog();
        ops.matmul(weight(t), backend.allocateBuffer((long) 9 * COLS * Float.BYTES, true), 0,
                backend.allocateBuffer((long) 9 * ROWS * Float.BYTES, true), 0, 9);
        boolean tiledUsed = dispatched().stream().anyMatch(n -> n.equals(t.name() + "_Hw_MatMul_Tiled"));
        assertTrue("the tiled GEMM of " + t, tiledUsed);
        ops.close();
        ops = null;
    }

    @Test
    public void prewarmOfAnHwTypePreparesTheSelectedRowSet() {
        GGMLType type = GGMLType.F8_E5M2;
        open(ComputeBackend.BackendType.CUDA, 90);
        ops.prewarmWeightKernels(type, true, true);
        List<String> prepared = backend.preparedKernels();
        for (String kernel : Arrays.asList(
                type.name() + "_Hw_MatVec", type.name() + "_Hw_BatchMatMul",
                type.name() + "_Hw_MatVecAdd", type.name() + "_Hw_BatchMatMul_2D",
                type.name() + "_Hw_MatMul_Tiled",
                type.name() + "_Hw_EmbeddingLookup", type.name() + "_Hw_BatchEmbeddingLookup")) {
            assertTrue(kernel + " prepared", prepared.contains(kernel));
        }
        assertTrue("no generic projection of " + type + " is prepared",
                prepared.stream().noneMatch(n -> n.startsWith(type.name() + "_") && !n.contains("_Hw_")));
        ops.close();
        ops = null;
    }
}

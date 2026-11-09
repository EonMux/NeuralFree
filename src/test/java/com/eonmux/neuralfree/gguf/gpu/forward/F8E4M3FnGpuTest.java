package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.WeightStore;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeightUploader;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuPlaceholders;
import com.eonmux.neuralfree.gguf.testutil.FakeComputeBackend;
import com.eonmux.neuralfree.gguf.testutil.FakeDispatch;
import org.junit.After;
import org.junit.Test;

import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * The GPU dispatch of ggml's per-element F8_E4M3FN (id 43): it runs the F8_E4M3 kernel rows (generic, and the
 * {@code _Hw} set on a CUDA device of sm_89 or higher), and the uploader keeps only whole 32-element-run rows
 * native — rows of any other width decode to F32 on upload, because the FP8 kernels walk 32-element blocks.
 */
public class F8E4M3FnGpuTest {

    private static final int COLS = 256;
    private static final int ROWS = 96;
    private static final int UBATCH = 4;

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

    private static com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities caps(int computeCapability) {
        return new com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities.Builder()
                .deviceName("Fp8FnDevice" + computeCapability)
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

    private GpuWeight weight() {
        long rowBytes = GGMLType.F8_E4M3FN.getBytesForElements(COLS);
        return GpuWeight.of("blk.0.attn_q.weight", GGMLType.F8_E4M3FN, new long[]{COLS, ROWS},
                backend.allocateBuffer(rowBytes * ROWS, true));
    }

    private List<String> dispatched() {
        List<String> names = new java.util.ArrayList<>();
        for (FakeDispatch d : backend.dispatches()) {
            names.add(d.kernelName());
        }
        return names;
    }

    @Test
    public void theSharedF8E4M3RowsServeTheUpstreamType() {
        open(ComputeBackend.BackendType.OPENCL, 0);
        GpuBuffer in = backend.allocateBuffer((long) COLS * Float.BYTES, true);
        GpuBuffer out = backend.allocateBuffer((long) ROWS * Float.BYTES, true);
        ops.matvec(weight(), in, 0, out, 0, 1);
        assertEquals("single-token projection", Arrays.asList("F8_E4M3_MatVec"), dispatched());

        backend.clearLog();
        ops.matmul(weight(), backend.allocateBuffer((long) 9 * COLS * Float.BYTES, true), 0,
                backend.allocateBuffer((long) 9 * ROWS * Float.BYTES, true), 0, 9);
        List<String> batch = dispatched();
        assertTrue("multi-token projection of the shared rows (tiled GEMM or batch)",
                batch.contains("F8_E4M3_MatMul_Tiled") || batch.contains("F8_E4M3_BatchMatMul")
                        || batch.contains("F8_E4M3_BatchMatMul_2D"));

        backend.clearLog();
        ops.embed(weight(), new int[]{0, 1},
                backend.allocateBuffer(2L * COLS * Float.BYTES, true), 0);
        List<String> embed = dispatched();
        assertTrue("embedding lookup of the shared rows", embed.contains("F8_E4M3_EmbeddingLookup")
                || embed.contains("F8_E4M3_BatchEmbeddingLookup"));
        ops.close();
        ops = null;
    }

    @Test
    public void aSm89CudaDeviceSelectsTheSharedHwRows() {
        WeightKernels k = GpuWeightFamilies.weightKernels(GGMLType.F8_E4M3FN,
                ComputeBackend.BackendType.CUDA, caps(89));
        assertEquals("hardware FP8 decode of the shared rows", "F8_E4M3_Hw_MatVec", k.matvec);
        assertEquals("F8_E4M3_Hw_MatMul_Tiled", k.tiled);
        assertEquals("F8_E4M3_Hw_EmbeddingLookup", k.embed);
    }

    @Test
    public void wholeBlockRowsStayNativeAndOtherRowsDecodeOnUpload() {
        FakeComputeBackend uploadBackend = FakeComputeBackend.create();
        List<GGUFTensorInfo> tensors = Arrays.asList(
                new GGUFTensorInfo("blk.0.attn_q.weight", new long[]{64, 32}, GGMLType.F8_E4M3FN, 0L),
                new GGUFTensorInfo("blk.0.attn_k.weight", new long[]{48, 32}, GGMLType.F8_E4M3FN, 0L),
                new GGUFTensorInfo("blk.0.attn_norm.weight", new long[]{64}, GGMLType.F8_E4M3FN, 0L));
        try (GpuWeightUploader up = new GpuWeightUploader(uploadBackend, uploadBackend.computeQueue(), tensors,
                new ZeroStore(), "llama", GpuWeightUploader.config((type, use) -> true).build())) {
            assertEquals("rows of 64 whole 32-element blocks keep the code type", GGMLType.F8_E4M3FN,
                    up.deviceType("blk.0.attn_q.weight"));
            assertEquals("rows of 48 decode to F32 on upload", GGMLType.F32,
                    up.deviceType("blk.0.attn_k.weight"));
            assertEquals("element-wise tensors decode to F32", GGMLType.F32,
                    up.deviceType("blk.0.attn_norm.weight"));
        }
    }

    /** Tensor data of zeros. */
    private static final class ZeroStore implements WeightStore {
        @Override
        public void readTensor(long tensorOffset, int size, byte[] dst) {
            Arrays.fill(dst, 0, size, (byte) 0);
        }

        @Override
        public void readTensor(long tensorOffset, int size, byte[] dst, int dstOffset) {
            Arrays.fill(dst, dstOffset, dstOffset + size, (byte) 0);
        }

        @Override
        public void readTensorDirect(long tensorOffset, int size, ByteBuffer dst) {
            for (int i = 0; i < size; i++) {
                dst.put((byte) 0);
            }
        }

        @Override
        public void close() {
        }
    }
}

package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities;
import com.eonmux.neuralfree.acceleration.backend.DeviceClass;
import com.eonmux.neuralfree.acceleration.backend.DeviceFeature;
import com.eonmux.neuralfree.acceleration.backend.DeviceIdentity;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.BatchInvariance;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuPlaceholders;
import com.eonmux.neuralfree.gguf.ops.gemm.GemmKernels;
import com.eonmux.neuralfree.gguf.ops.gemm.MmaGemmPlan;
import com.eonmux.neuralfree.gguf.testutil.FakeComputeBackend;
import com.eonmux.neuralfree.gguf.testutil.FakeDispatch;
import org.junit.After;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertSame;

/**
 * The large MMA tile's route, traced against the host-only fake backend of a CUDA sm_120 device of 60
 * multiprocessors: an F16 or BF16 projection of at least 64 tokens runs {@code <T>_MatMul_MMA_L} with the arguments
 * of the 64 x 32 tile; a launch of fewer tokens, a launch split-K would split (except under batch invariance, which
 * splits none and keeps the large tile, whose bits are the 64 x 32 tile's) and a program without the kernel keep
 * {@code <T>_MatMul_MMA}.
 */
public class LargeMmaTileRoutingTest {

    private static final int COLS = 256;
    private static final int ROWS = 2560;
    private static final int UBATCH = 512;

    private FakeComputeBackend backend;
    private GpuScratch scratch;
    private GpuPlaceholders placeholders;
    private GpuOps ops;

    private void open(String... unavailable) {
        backend = FakeComputeBackend.builder().backendType(ComputeBackend.BackendType.CUDA).strictLlmArgs(true)
                .capabilities(cudaTensorDevice()).unavailableKernels(unavailable).build();
        scratch = GpuScratch.builder(backend, UBATCH).slot(ScratchSlots.X, (long) UBATCH * COLS).build();
        placeholders = GpuPlaceholders.allocate(backend, backend.computeQueue());
        ops = new GpuOps(backend.computeQueue(), LlmKernels.open(backend), scratch, placeholders);
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

    private static DeviceCapabilities cudaTensorDevice() {
        return new DeviceCapabilities.Builder()
                .deviceName("TensorDevice")
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
                .feature(DeviceFeature.BF16_COOPERATIVE_MATRIX)
                .computeCapability(120)
                .deviceClass(DeviceClass.DISCRETE)
                .identity(new DeviceIdentity("TensorDevice", FakeComputeBackend.NVIDIA_VENDOR_ID, null, null))
                .build();
    }

    private GpuWeight dense(GGMLType type) {
        return dense(type, ROWS, COLS);
    }

    private GpuWeight dense(GGMLType type, int rows, int cols) {
        return GpuWeight.of("blk.0.ffn_down.weight", type, new long[]{cols, rows},
                backend.allocateBuffer((long) rows * cols * 2, true));
    }

    private GpuBuffer floats(long n) {
        return backend.allocateBuffer(n * Float.BYTES);
    }

    private List<FakeDispatch> project(GpuWeight w, int n, GpuBuffer in, GpuBuffer out) {
        backend.clearLog();
        ops.matmul(w, in, 0, out, 0, n);
        List<FakeDispatch> gemms = new ArrayList<>();
        for (FakeDispatch d : backend.dispatches()) {
            if (d.kernelName().contains("_MatMul_")) {
                gemms.add(d);
            }
        }
        return gemms;
    }

    /** 512 tokens through 2560 rows: 20 x 8 work-groups of the large tile. */
    @Test
    public void aPrefillLaunchRunsTheLargeTile() {
        open();
        for (GGMLType type : new GGMLType[]{GGMLType.F16, GGMLType.BF16}) {
            GpuWeight w = dense(type);
            GpuBuffer in = backend.allocateBuffer((long) UBATCH * COLS * Float.BYTES);
            GpuBuffer out = backend.allocateBuffer((long) UBATCH * ROWS * Float.BYTES);
            List<FakeDispatch> ds = project(w, UBATCH, in, out);
            assertEquals(1, ds.size());
            FakeDispatch d = ds.get(0);
            assertEquals(GemmKernels.matmulMmaLarge(type), d.kernelName());
            assertArrayEquals(new long[]{20L * MmaGemmPlan.WORKGROUP_SIZE, 8}, d.global());
            int[] args = new int[7];
            for (int i = 0; i < args.length; i++) {
                args[i] = d.intArg(i);
            }
            assertArrayEquals(new int[]{ROWS, COLS, UBATCH, COLS, 0, ROWS, 0}, args);
            assertSame(w.shard(0), d.bufferArg(GemmKernels.ARG_WEIGHT));
            assertSame(in, d.bufferArg(GemmKernels.ARG_X));
            assertSame(out, d.bufferArg(GemmKernels.ARG_Y));
        }
    }

    /** 64 tokens, one large token tile: the large tile, even over a grid of 20 work-groups; 63 tokens keep the
     *  64 x 32 tile. */
    @Test
    public void oneTokenTileIsTheLargeTilesThreshold() {
        open();
        GpuWeight w = dense(GGMLType.F16);
        List<FakeDispatch> at = project(w, MmaGemmPlan.LARGE_TILE_TOKENS, floats(UBATCH * COLS),
                floats(UBATCH * ROWS));
        assertEquals(1, at.size());
        assertEquals("F16_MatMul_MMA_L", at.get(0).kernelName());
        assertArrayEquals(new long[]{20L * MmaGemmPlan.WORKGROUP_SIZE, 1}, at.get(0).global());
        List<FakeDispatch> below = project(w, MmaGemmPlan.LARGE_TILE_TOKENS - 1, floats(UBATCH * COLS),
                floats(UBATCH * ROWS));
        assertEquals("F16_MatMul_MMA", below.get(0).kernelName());
    }

    /**
     * 64 tokens through 1024 rows of 2560: the tall-skinny grid split-K splits three ways (0.040 ms against the
     * large tile's 0.077 ms on the RTX 5080 Laptop) keeps the split; batch invariance, which splits nothing, takes
     * the large tile.
     */
    @Test
    public void aSplitKLaunchKeepsTheSplitUnlessBatchInvariant() {
        open();
        int rows = 1024;
        int cols = 2560;
        int n = MmaGemmPlan.LARGE_TILE_TOKENS;
        GpuWeight w = dense(GGMLType.F16, rows, cols);
        List<FakeDispatch> split = project(w, n, floats((long) n * cols), floats((long) n * rows));
        assertEquals("F16_MatMul_MMA_KSplit", split.get(0).kernelName());
        BatchInvariance.setEnabled(Boolean.TRUE);
        try {
            List<FakeDispatch> ds = project(w, n, floats((long) n * cols), floats((long) n * rows));
            assertEquals(1, ds.size());
            assertEquals("F16_MatMul_MMA_L", ds.get(0).kernelName());
        } finally {
            BatchInvariance.clearOverride();
        }
    }

    /** A program without the large tile (every backend but CUDA sm_80+) keeps the 64 x 32 tile. */
    @Test
    public void programsWithoutTheKernelKeepTheSmallTile() {
        open("F16_MatMul_MMA_L");
        GpuWeight w = dense(GGMLType.F16);
        List<FakeDispatch> ds = project(w, UBATCH, backend.allocateBuffer((long) UBATCH * COLS * Float.BYTES),
                backend.allocateBuffer((long) UBATCH * ROWS * Float.BYTES));
        assertEquals(1, ds.size());
        assertEquals("F16_MatMul_MMA", ds.get(0).kernelName());
        assertArrayEquals(new long[]{40L * MmaGemmPlan.WORKGROUP_SIZE, 16}, ds.get(0).global());
    }
}

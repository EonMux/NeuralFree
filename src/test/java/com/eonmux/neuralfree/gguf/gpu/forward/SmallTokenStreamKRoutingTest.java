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
import com.eonmux.neuralfree.gguf.ops.gemm.MmqConfig;
import com.eonmux.neuralfree.gguf.ops.gemm.StreamKGemmPlan;
import com.eonmux.neuralfree.gguf.testutil.FakeComputeBackend;
import com.eonmux.neuralfree.gguf.testutil.FakeDispatch;
import org.junit.After;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;

/**
 * The small-token stream-k route of the CUDA tensor tile, traced against the host-only fake backend of a CUDA
 * sm_120 device of 60 multiprocessors: a launch of at most 16 tokens that would run the 16-token twin of an int8
 * type runs {@code <T>_MatMul_MMQ_StreamK_S} over the device tile's staged records, one block per multiprocessor,
 * and, when its blocks split tiles, {@code LLM_MMQ_StreamK_Fixup_S}, both through the plan and through
 * {@link GpuOps#matmul}; the small-K prefill launches of 17 to 48 tokens that also take the 16-token twin, a grid of
 * fewer tiles than a quarter of the multiprocessors, the 64-token twin, batch invariance and a program without the
 * kernel keep their grids.
 */
public class SmallTokenStreamKRoutingTest {

    private static final int COLS = 256;
    private static final int ROWS = 1920;
    private static final int UBATCH = 32;
    private static final int MAX_COLS = 1024;

    private FakeComputeBackend backend;
    private GpuScratch scratch;
    private GpuPlaceholders placeholders;
    private GpuOps ops;

    private void open(String... unavailable) {
        backend = FakeComputeBackend.builder().backendType(ComputeBackend.BackendType.CUDA).strictLlmArgs(true)
                .capabilities(cudaTensorDevice()).unavailableKernels(unavailable).build();
        scratch = GpuScratch.builder(backend, UBATCH).slot(ScratchSlots.X, (long) UBATCH * MAX_COLS).build();
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

    private GpuWeight q80(int rows) {
        return GpuWeight.of("blk.0.attn_q.weight", GGMLType.Q8_0, new long[]{COLS, rows},
                backend.allocateBuffer((long) rows * COLS / 32 * 34, true));
    }

    /** A Q5_0 weight, a type without a CUDA multi-column matvec crossover entry: its decode batches take the
     *  MMQ GEMM. */
    private GpuWeight q50(int rows, int cols) {
        return GpuWeight.of("blk.0.attn_q.weight", GGMLType.Q5_0, new long[]{cols, rows},
                backend.allocateBuffer((long) rows * cols / 32 * 22, true));
    }

    /** The GEMM dispatches (the MMQ kernels, their split-K reduce and the stream-k fixups) of one
     *  {@link GpuOps#matmul}, without the activation staging. */
    private List<String> project(GpuWeight w, int n) {
        int cols = (int) w.cols();
        int rows = (int) w.rows();
        GpuBuffer in = floats((long) n * cols);
        GpuBuffer out = floats((long) n * rows);
        List<String> gemms = new ArrayList<>();
        for (FakeDispatch d : run(() -> ops.matmul(w, in, 0, out, 0, n))) {
            if (d.kernelName().contains("_MatMul_") || d.kernelName().startsWith("LLM_MMQ_StreamK")
                    || d.kernelName().equals(GemmKernels.SPLIT_K_REDUCE)) {
                gemms.add(d.kernelName());
            }
        }
        return gemms;
    }

    private GpuBuffer floats(long n) {
        return backend.allocateBuffer(n * Float.BYTES);
    }

    private List<FakeDispatch> run(Runnable r) {
        backend.clearLog();
        r.run();
        return backend.dispatches();
    }

    private static List<String> names(List<FakeDispatch> ds) {
        List<String> out = new ArrayList<>();
        for (FakeDispatch d : ds) {
            out.add(d.kernelName());
        }
        return out;
    }

    private static int[] ints(FakeDispatch d, int from, int to) {
        int[] v = new int[to - from];
        for (int i = from; i < to; i++) {
            v[i - from] = d.intArg(i);
        }
        return v;
    }

    private String twin(GpuWeight w, int n) {
        return ops.lanes.smallTileKernel(ops.lanes.fastGemm(w, COLS), w, n);
    }

    /**
     * Nine tokens through 1920 rows: fifteen 128-row tiles of eight chunks over the 60 resident blocks, two chunks
     * each, so every tile is split and the fixup follows over the same planes, two slices of the 128 x 16 outputs
     * per block.
     */
    @Test
    public void aSixteenTokenTwinLaunchRunsTheSmallTokenStreamK() {
        open();
        int n = 9;
        GpuWeight w = q80(ROWS);
        GpuBuffer in = floats((long) n * COLS);
        GpuBuffer out = floats((long) ROWS * n);
        String small = twin(w, n);
        assertEquals("Q8_0_MatMul_MMQ_S", small);
        StreamKGemmPlan plan = ops.lanes.streamKSmall(small, w, ROWS, COLS, n, 0, 0, ROWS);
        assertNotNull(plan);
        assertEquals(60, plan.blocks());
        List<FakeDispatch> ds = run(() -> ops.lanes.streamKLaunch(plan, w.shard(0), in, out));
        assertEquals(Arrays.asList("Q8_0_MatMul_MMQ_StreamK_S", GemmKernels.STREAM_K_FIXUP_SMALL), names(ds));
        FakeDispatch sk = ds.get(0);
        assertArrayEquals(new int[]{ROWS, COLS, n, MmqConfig.INT8_TENSOR.rowRecords(COLS), 0, ROWS, 0},
                ints(sk, 0, 7));
        assertArrayEquals(new long[]{60L * MmqConfig.INT8_TENSOR_SMALL.workGroupSize(), 1}, sk.global());
        assertSame(w.shard(0), sk.bufferArg(GemmKernels.ARG_WEIGHT));
        assertSame(in, sk.bufferArg(GemmKernels.ARG_X));
        assertSame(out, sk.bufferArg(GemmKernels.ARG_Y));
        FakeDispatch fix = ds.get(1);
        assertSame(sk.bufferArg(StreamKGemmPlan.ARG_FIXUP), fix.bufferArg(StreamKGemmPlan.FIXUP_ARG_FIXUP));
        assertSame(out, fix.bufferArg(StreamKGemmPlan.FIXUP_ARG_Y));
        assertArrayEquals(new long[]{60L * MmqConfig.INT8_TENSOR_SMALL.workGroupSize(), 2}, fix.global());
    }

    /**
     * A decode batch of nine tokens through a 2048-row Q5_0 projection, through {@link GpuOps#matmul}: the dispatch
     * takes the 16-token twin's route and runs the stream-k form and its fixup in place of the twin.
     */
    @Test
    public void aDecodeBatchThroughMatmulRunsTheSmallTokenStreamK() {
        open();
        GpuWeight w = q50(2048, COLS);
        assertEquals("Q5_0_MatMul_MMQ_S", twin(w, 9));
        assertEquals(Arrays.asList("Q5_0_MatMul_MMQ_StreamK_S", GemmKernels.STREAM_K_FIXUP_SMALL), project(w, 9));
        assertEquals(Arrays.asList("Q5_0_MatMul_MMQ_StreamK_S", GemmKernels.STREAM_K_FIXUP_SMALL),
                project(w, MmqConfig.TOKEN_LANES));
    }

    /**
     * 24 tokens through a 2048-row Q5_0 projection over K = 1024, as a Qwen3 0.6B prefill's q projection: below
     * {@code SplitKGemmPlan.MIN_SPLIT_K} the ladder gives the 16-token twin up to 48 tokens, and that prefill launch
     * keeps the twin rather than the decode batches' stream-k form, as does a 17-token one.
     */
    @Test
    public void aSmallKPrefillLaunchPastSixteenTokensKeepsTheTwin() {
        open();
        int cols = MAX_COLS;
        GpuWeight w = q50(2048, cols);
        for (int n : new int[]{MmqConfig.TOKEN_LANES + 1, 24}) {
            String small = twin(w, n);
            assertEquals("Q5_0_MatMul_MMQ_S", small);
            assertNull(ops.lanes.streamKSmall(small, w, 2048, cols, n, 0, 0, 2048));
            List<String> gemms = project(w, n);
            assertEquals(n + " tokens: " + gemms, "Q5_0_MatMul_MMQ_S", gemms.get(0));
            for (String k : gemms) {
                assertFalse(n + " tokens: " + gemms, k.contains("StreamK"));
            }
        }
    }

    /**
     * Grids of fewer tiles than a quarter of the 60 units keep the twin and its split-K: three tiles, and up to
     * fourteen (1792 rows), while fifteen (1920 rows) take the stream-k form; through {@link GpuOps#matmul} a
     * 1024-row projection of eight tiles keeps the twin.
     */
    @Test
    public void aGridOfFewTilesKeepsTheTwin() {
        open();
        for (int rows : new int[]{320, 1792}) {
            GpuWeight w = q80(rows);
            String small = twin(w, 9);
            assertEquals("Q8_0_MatMul_MMQ_S", small);
            assertNull(rows + " rows", ops.lanes.streamKSmall(small, w, rows, COLS, 9, 0, 0, rows));
        }
        assertNotNull(ops.lanes.streamKSmall("Q8_0_MatMul_MMQ_S", q80(ROWS), ROWS, COLS, 9, 0, 0, ROWS));
        List<String> gemms = project(q50(1024, COLS), 9);
        assertTrue(gemms.toString(), gemms.get(0).startsWith("Q5_0_MatMul_MMQ_S"));
        for (String k : gemms) {
            assertFalse(gemms.toString(), k.contains("StreamK"));
        }
    }

    /** The 64-token twin keeps its grid, and batch invariance takes no twin at all. */
    @Test
    public void theWideTwinAndBatchInvarianceKeepTheirGrids() {
        open();
        GpuWeight w = q80(ROWS);
        assertNull(ops.lanes.streamKSmall(GemmKernels.matmulMmqSmall(GGMLType.Q8_0, MmqConfig.TWIN_MAX_TOKENS), w,
                ROWS, COLS, 40, 0, 0, ROWS));
        BatchInvariance.setEnabled(Boolean.TRUE);
        try {
            assertNull(twin(w, 9));
        } finally {
            BatchInvariance.clearOverride();
        }
    }

    /** A CUDA program without the small-token stream-k kernel keeps the 16-token twin. */
    @Test
    public void programsWithoutTheKernelKeepTheTwin() {
        open("Q8_0_MatMul_MMQ_StreamK_S");
        GpuWeight w = q80(ROWS);
        String small = twin(w, 9);
        assertEquals("Q8_0_MatMul_MMQ_S", small);
        assertNull(ops.lanes.streamKSmall(small, w, ROWS, COLS, 9, 0, 0, ROWS));
    }
}

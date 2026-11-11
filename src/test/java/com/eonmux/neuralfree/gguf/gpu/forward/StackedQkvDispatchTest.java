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
import com.eonmux.neuralfree.gguf.testutil.FakeComputeBackend;
import com.eonmux.neuralfree.gguf.testutil.FakeDispatch;
import org.junit.After;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Selection and launch of the stacked Q/K/V matvec ({@link GpuOps#stackedQkv}) on faked devices: one launch of
 * {@code {T}_MatVecQKV} for three single-shard weights of one type on Vulkan, its arguments in the kernel's order,
 * the integer-dot form once the three weights together reach the lane's size, and the separate products wherever a
 * condition fails (another backend, mixed types or widths, a sharded or scaled weight, more than one token, fusion
 * turned off, batch invariance).
 */
public class StackedQkvDispatchTest {

    private static final int COLS = 256;
    private static final int UBATCH = 4;

    private FakeComputeBackend backend;
    private GpuScratch scratch;
    private GpuPlaceholders placeholders;
    private GpuOps ops;
    private int matvecLocal;

    @After
    public void tearDown() {
        BatchInvariance.clearOverride();
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

    private void open(ComputeBackend.BackendType type, FusedMode mode, int cols) {
        close();
        backend = FakeComputeBackend.builder().backendType(type).strictLlmArgs(true)
                .capabilities(nvidiaDiscrete(type)).build();
        scratch = GpuScratch.builder(backend, UBATCH).slot(ScratchSlots.X, (long) UBATCH * cols).build();
        placeholders = GpuPlaceholders.allocate(backend, backend.computeQueue());
        LlmKernels kernels = LlmKernels.open(backend);
        matvecLocal = kernels.matvecLocal();
        ops = new GpuOps(backend.computeQueue(), kernels, scratch, placeholders, Int8MatvecMode.configured(), mode);
    }

    private void open() {
        open(ComputeBackend.BackendType.VULKAN, FusedMode.AUTO, COLS);
    }

    /** An RTX-class discrete NVIDIA device with the packed integer dot product. */
    private static DeviceCapabilities nvidiaDiscrete(ComputeBackend.BackendType type) {
        DeviceCapabilities.Builder b = new DeviceCapabilities.Builder()
                .deviceName("QkvDevice")
                .totalVramBytes(8L << 30)
                .maxAllocSizeBytes(2L << 30)
                .computeUnits(60)
                .maxClockFrequencyMhz(1500)
                .maxWorkGroupSize(1024)
                .driverVersion("fake-1.0")
                .supportsFloat16(true)
                .supportsSubgroups(true)
                .maxLocalMemoryBytes(48L * 1024)
                .feature(DeviceFeature.INTEGER_DOT_PRODUCT)
                .deviceClass(DeviceClass.DISCRETE)
                .identity(new DeviceIdentity("QkvDevice", FakeComputeBackend.NVIDIA_VENDOR_ID, null, null));
        if (type == ComputeBackend.BackendType.VULKAN) {
            b.feature(DeviceFeature.STORAGE_8BIT).feature(DeviceFeature.STORAGE_16BIT)
                    .feature(DeviceFeature.TIMELINE_SEMAPHORE);
        } else if (type == ComputeBackend.BackendType.CUDA) {
            b.computeCapability(120);
        }
        return b.build();
    }

    private GpuWeight weight(String name, GGMLType type, int rows, int cols) {
        long bytes = (long) rows * (cols / type.getBlockSize()) * type.getTypeSize();
        return GpuWeight.of(name, type, new long[]{cols, rows}, backend.allocateBuffer(bytes, true));
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

    @Test
    public void threeWeightsOfOneTypeRunAsOneLaunchInTheKernelArgumentOrder() {
        open();
        GpuWeight wq = weight("blk.0.attn_q.weight", GGMLType.Q8_0, 96, COLS);
        GpuWeight wk = weight("blk.0.attn_k.weight", GGMLType.Q8_0, 32, COLS);
        GpuWeight wv = weight("blk.0.attn_v.weight", GGMLType.Q8_0, 64, COLS);
        GpuBuffer x = floats(COLS);
        GpuBuffer q = floats(96);
        GpuBuffer k = floats(32);
        GpuBuffer v = floats(64);
        assertTrue(ops.stackedQkvUsable(wq, wk, wv, 1));
        List<FakeDispatch> ds = run(() -> ops.stackedQkv(wq, wk, wv, x, 0, q, k, v));

        assertEquals(Arrays.asList("Q8_0_MatVecQKV"), names(ds));
        FakeDispatch d = ds.get(0);
        assertEquals(96, d.intArg(0));
        assertEquals(COLS, d.intArg(1));
        assertEquals(32, d.intArg(2));
        assertEquals(64, d.intArg(3));
        assertSame(wq.shard(0), d.bufferArg(4));
        assertSame(wk.shard(0), d.bufferArg(5));
        assertSame(wv.shard(0), d.bufferArg(6));
        assertSame(x, d.bufferArg(7));
        assertSame(q, d.bufferArg(8));
        assertSame(k, d.bufferArg(9));
        assertSame(v, d.bufferArg(10));
        assertEquals("one work-group per row of the three weights", (96L + 32 + 64) * matvecLocal,
                d.global()[0]);
    }

    /**
     * A stack as large as the integer-dot lane's threshold takes the Q8_1 row with one row per work-group, also on
     * a device kind whose separate products pair their rows.
     */
    @Test
    public void aStackAsLargeAsTheInt8LaneTakesTheQ8RowForm() {
        int cols = 2560;
        open(ComputeBackend.BackendType.VULKAN, FusedMode.AUTO, cols);
        GpuWeight wq = weight("blk.0.attn_q.weight", GGMLType.Q8_0, 4095, cols);
        GpuWeight wk = weight("blk.0.attn_k.weight", GGMLType.Q8_0, 1025, cols);
        GpuWeight wv = weight("blk.0.attn_v.weight", GGMLType.Q8_0, 1024, cols);
        GpuBuffer x = floats(cols);
        List<FakeDispatch> ds = run(() -> ops.stackedQkv(wq, wk, wv, x, 0, floats(4095), floats(1025),
                floats(1024)));

        assertEquals(Arrays.asList(GpuOps.QUANTIZE_Q8_1, "Q8_0_MatVecQKV_Q8"), names(ds));
        assertEquals((4095L + 1025 + 1024) * matvecLocal, ds.get(1).global()[0]);
    }

    @Test
    public void anInputAtAnOffsetIsStagedOnce() {
        open();
        GpuWeight wq = weight("blk.0.attn_q.weight", GGMLType.Q4_K, 64, COLS);
        GpuWeight wk = weight("blk.0.attn_k.weight", GGMLType.Q4_K, 32, COLS);
        GpuWeight wv = weight("blk.0.attn_v.weight", GGMLType.Q4_K, 32, COLS);
        GpuBuffer x = floats(2L * COLS);
        List<FakeDispatch> ds = run(() -> ops.stackedQkv(wq, wk, wv, x, COLS, floats(64), floats(32), floats(32)));

        assertEquals(2, ds.size());
        assertEquals("Q4_K_MatVecQKV", ds.get(1).kernelName());
        assertFalse("the stacked launch reads the staged copy", ds.get(1).bufferArg(7) == x);
    }

    @Test
    public void everyUnmetConditionKeepsTheSeparateProducts() {
        open();
        GpuWeight wq = weight("blk.0.attn_q.weight", GGMLType.Q8_0, 96, COLS);
        GpuWeight wk = weight("blk.0.attn_k.weight", GGMLType.Q8_0, 32, COLS);
        GpuWeight wv = weight("blk.0.attn_v.weight", GGMLType.Q8_0, 64, COLS);
        assertFalse("more than one token", ops.stackedQkvUsable(wq, wk, wv, 2));
        assertFalse("a missing weight", ops.stackedQkvUsable(wq, wk, null, 1));
        assertFalse("mixed types",
                ops.stackedQkvUsable(wq, wk, weight("blk.0.attn_v.weight", GGMLType.Q6_K, 64, COLS), 1));
        assertFalse("mixed widths",
                ops.stackedQkvUsable(wq, wk, weight("blk.0.attn_v.weight", GGMLType.Q8_0, 64, 2 * COLS), 1));
        List<GpuBuffer> shards = Arrays.asList(backend.allocateBuffer(32L * 8 * 34, true),
                backend.allocateBuffer(32L * 8 * 34, true));
        GpuWeight sharded = GpuWeight.sharded("blk.0.attn_k.weight", GGMLType.Q8_0, new long[]{COLS, 64}, shards,
                new long[]{32, 32});
        assertFalse("a sharded weight", ops.stackedQkvUsable(wq, sharded, wv, 1));
        GpuBuffer x = floats(COLS);
        assertThrows(UnsupportedOperationException.class,
                () -> ops.stackedQkv(wq, wk, null, x, 0, floats(96), floats(32), floats(64)));

        BatchInvariance.setEnabled(Boolean.TRUE);
        assertFalse("batch invariance", ops.stackedQkvUsable(wq, wk, wv, 1));
        BatchInvariance.clearOverride();

        open(ComputeBackend.BackendType.VULKAN, FusedMode.OFF, COLS);
        assertFalse("fusion off", ops.stackedQkvUsable(weight("q", GGMLType.Q8_0, 96, COLS),
                weight("k", GGMLType.Q8_0, 32, COLS), weight("v", GGMLType.Q8_0, 64, COLS), 1));

        open(ComputeBackend.BackendType.CUDA, FusedMode.AUTO, COLS);
        assertFalse("a backend without the form", ops.stackedQkvUsable(weight("q", GGMLType.Q8_0, 96, COLS),
                weight("k", GGMLType.Q8_0, 32, COLS), weight("v", GGMLType.Q8_0, 64, COLS), 1));
    }

    /**
     * The stacked forms are prepared for the attention Q weights' types alone ({@link GpuOps#prewarmStackedQkv}),
     * not with every projected type's kernels, and only where the backend holds them and fusion is permitted.
     */
    @Test
    public void prewarmPreparesBothStackedLanesForAttentionTypesOnVulkanOnly() {
        open();
        ops.prewarmWeightKernels(GGMLType.Q8_0, true, false);
        assertFalse(backend.preparedKernels().toString(), backend.preparedKernels().contains("Q8_0_MatVecQKV"));
        ops.prewarmStackedQkv(GGMLType.Q8_0);
        List<String> prepared = backend.preparedKernels();
        assertTrue(prepared.toString(), prepared.contains("Q8_0_MatVecQKV"));
        assertTrue(prepared.toString(), prepared.contains("Q8_0_MatVecQKV_Q8"));

        open(ComputeBackend.BackendType.VULKAN, FusedMode.OFF, COLS);
        ops.prewarmStackedQkv(GGMLType.Q8_0);
        assertFalse(backend.preparedKernels().toString(), backend.preparedKernels().contains("Q8_0_MatVecQKV"));

        open(ComputeBackend.BackendType.CUDA, FusedMode.AUTO, COLS);
        ops.prewarmStackedQkv(GGMLType.Q8_0);
        assertFalse(backend.preparedKernels().toString(), backend.preparedKernels().contains("Q8_0_MatVecQKV"));
    }

    @Test
    public void overlappingViewsAreRefused() {
        open();
        GpuWeight wq = weight("blk.0.attn_q.weight", GGMLType.Q8_0, 96, COLS);
        GpuWeight wk = weight("blk.0.attn_k.weight", GGMLType.Q8_0, 32, COLS);
        GpuWeight wv = weight("blk.0.attn_v.weight", GGMLType.Q8_0, 64, COLS);
        GpuBuffer shared = floats(COLS + 96);
        assertThrows(IllegalArgumentException.class,
                () -> ops.stackedQkv(wq, wk, wv, shared, 0, shared, floats(32), floats(64)));
        GpuBuffer out = floats(96);
        assertThrows(IllegalArgumentException.class,
                () -> ops.stackedQkv(wq, wk, wv, floats(COLS), 0, out, out, floats(64)));
    }
}

package com.eonmux.neuralfree.gguf.gpu.head;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend.BackendType;
import com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities;
import com.eonmux.neuralfree.acceleration.backend.DeviceFeature;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuScratch;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlots;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuPlaceholders;
import com.eonmux.neuralfree.gguf.testutil.FakeComputeBackend;
import com.eonmux.neuralfree.gguf.testutil.FakeDispatch;
import org.junit.After;
import org.junit.Test;

import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link GpuArgmax}'s launch plan on fake backends whose argument rules follow the kernel inventory: a row longer than
 * one chunk of {@code UNROLL * L} elements runs the split pair (the chunk pass over {@code rows * chunks} work-groups,
 * then the per-row fold), a row of at most one chunk and a backend without the split pair run the one-pass kernel.
 */
public class GpuArgmaxTest {

    private static final int LOCAL = 256;

    private FakeComputeBackend backend;
    private GpuScratch scratch;
    private GpuPlaceholders placeholders;
    private GpuOps ops;

    @After
    public void tearDown() {
        if (ops != null) {
            ops.close();
            ops = null;
        }
        if (placeholders != null) {
            placeholders.close();
            placeholders = null;
        }
        if (scratch != null) {
            scratch.close();
            scratch = null;
        }
        if (backend != null) {
            backend.close();
            backend = null;
        }
    }

    private void open(BackendType type, String... unavailable) {
        backend = FakeComputeBackend.builder().backendType(type).strictLlmArgs(true)
                .capabilities(capabilities(type)).unavailableKernels(unavailable).build();
        scratch = GpuScratch.builder(backend, 4).slot(ScratchSlots.X, 4L * LOCAL).build();
        placeholders = GpuPlaceholders.allocate(backend, backend.computeQueue());
        ops = new GpuOps(backend.computeQueue(), LlmKernels.open(backend), scratch, placeholders);
    }

    /** A device of {@code type} with what the LLM kernel set needs there. */
    private static DeviceCapabilities capabilities(BackendType type) {
        DeviceCapabilities.Builder b = new DeviceCapabilities.Builder()
                .deviceName("ArgmaxDevice")
                .totalVramBytes(8L << 30)
                .maxAllocSizeBytes(2L << 30)
                .computeUnits(32)
                .maxClockFrequencyMhz(1500)
                .maxWorkGroupSize(1024)
                .driverVersion("fake-1.0")
                .supportsFloat16(true)
                .supportsSubgroups(true)
                .maxLocalMemoryBytes(48L * 1024);
        if (type == BackendType.CUDA) {
            b.computeCapability(120);
        } else if (type == BackendType.VULKAN) {
            b.feature(DeviceFeature.STORAGE_8BIT).feature(DeviceFeature.STORAGE_16BIT)
                    .feature(DeviceFeature.TIMELINE_SEMAPHORE);
        }
        return b.build();
    }

    @Test
    public void chunksCoverTheRowInUnrollTimesLocalSpans() {
        int chunk = GpuArgmax.UNROLL * LOCAL;
        assertEquals(1, GpuArgmax.chunks(1, LOCAL));
        assertEquals(1, GpuArgmax.chunks(chunk, LOCAL));
        assertEquals(2, GpuArgmax.chunks(chunk + 1, LOCAL));
        assertEquals(75, GpuArgmax.chunks(151936, LOCAL));
        assertEquals(297, GpuArgmax.chunks(151936, 64));
        assertThrows(IllegalArgumentException.class, () -> GpuArgmax.chunks(0, LOCAL));
        assertThrows(IllegalArgumentException.class, () -> GpuArgmax.chunks((1 << 30) + 1, LOCAL));
        assertThrows(IllegalArgumentException.class, () -> GpuArgmax.chunks(10, 0));
    }

    @Test
    public void aVocabularyRowRunsTheSplitPair() {
        for (BackendType type : new BackendType[]{BackendType.CUDA, BackendType.OPENCL, BackendType.VULKAN}) {
            open(type);
            int vocab = 151936;
            int rows = 3;
            GpuArgmax plan = GpuArgmax.plan(ops, vocab, LOCAL);
            assertTrue(type + " splits", plan.split());
            assertEquals(75, plan.chunksPerRow());
            assertEquals((long) rows * 75 * 2, plan.partialInts(rows));
            GpuBuffer logits = backend.allocateBuffer((long) rows * vocab * Float.BYTES);
            GpuBuffer ids = backend.allocateBuffer(rows * Integer.BYTES);
            GpuBuffer parts = backend.allocateBuffer(plan.partialInts(rows) * Integer.BYTES);
            backend.clearLog();
            plan.enqueue(ops, logits, ids, rows, parts);
            List<FakeDispatch> d = backend.dispatches();
            assertEquals(type + " launches", 2, d.size());
            assertEquals(GpuArgmax.SPLIT, d.get(0).kernelName());
            assertArrayEquals(new long[]{(long) rows * 75 * LOCAL}, d.get(0).global());
            assertEquals(vocab, d.get(0).intArg(0));
            assertEquals(75, d.get(0).intArg(1));
            assertSame(logits, d.get(0).arg(2));
            assertSame(parts, d.get(0).arg(3));
            assertEquals(rows, d.get(0).intArg(4));
            assertEquals(GpuArgmax.COMBINE, d.get(1).kernelName());
            assertArrayEquals(new long[]{(long) rows * LOCAL}, d.get(1).global());
            assertEquals(75, d.get(1).intArg(0));
            assertSame(parts, d.get(1).arg(1));
            assertSame(ids, d.get(1).arg(2));
            assertEquals(rows, d.get(1).intArg(3));
            assertThrows(NullPointerException.class, () -> plan.enqueue(ops, logits, ids, rows, null));
            tearDown();
        }
    }

    @Test
    public void aRowOfOneChunkRunsTheOnePassKernel() {
        open(BackendType.CUDA);
        int size = GpuArgmax.UNROLL * LOCAL;
        GpuArgmax plan = GpuArgmax.plan(ops, size, LOCAL);
        assertFalse(plan.split());
        assertEquals(0, plan.partialInts(4));
        GpuBuffer logits = backend.allocateBuffer(2L * size * Float.BYTES);
        GpuBuffer ids = backend.allocateBuffer(2 * Integer.BYTES);
        backend.clearLog();
        plan.enqueue(ops, logits, ids, 2, null);
        List<FakeDispatch> d = backend.dispatches();
        assertEquals(1, d.size());
        assertEquals(GpuArgmax.ROW, d.get(0).kernelName());
        assertArrayEquals(new long[]{2L * LOCAL}, d.get(0).global());
        assertEquals(size, d.get(0).intArg(0));
        assertEquals(2, d.get(0).intArg(3));
    }

    @Test
    public void aBackendWithoutTheSplitPairKeepsTheOnePassKernel() {
        open(BackendType.VULKAN, GpuArgmax.COMBINE);
        GpuArgmax plan = GpuArgmax.plan(ops, 151936, LOCAL);
        assertFalse(plan.split());
        GpuBuffer logits = backend.allocateBuffer(151936L * Float.BYTES);
        GpuBuffer ids = backend.allocateBuffer(Integer.BYTES);
        backend.clearLog();
        plan.enqueue(ops, logits, ids, 1, null);
        assertEquals(GpuArgmax.ROW, backend.dispatches().get(0).kernelName());
    }
}

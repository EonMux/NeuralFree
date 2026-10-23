package com.eonmux.neuralfree.tensor.gpu;

import com.eonmux.neuralfree.acceleration.backend.BackendResources;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.acceleration.kernels.Availability;
import com.eonmux.neuralfree.tensor.DType;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;
import com.eonmux.neuralfree.tensor.TensorStorage;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.concurrent.locks.ReentrantLock;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNotSame;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The per-backend train context on each GPU API: cached availability, prewarmed kernels, shared buffers, launches,
 * host-read ordering and release. A host-backend case checks the cached negative result.
 */
@RunWith(Parameterized.class)
public class TrainGpuTest {

    @Parameterized.Parameters(name = "{0}")
    public static Collection<Object[]> data() {
        return TrainParity.apis();
    }

    private final TrainParity.Api api;

    public TrainGpuTest(TrainParity.Api api) {
        this.api = api;
    }

    @Test
    public void hostBackendIsUnavailableAndTheResultIsCached() {
        ComputeBackend host = TrainParity.hostBackend();
        try {
            Availability first = TrainGpu.availability(host);
            assertFalse(first.available());
            assertNotNull(first.reason());
            assertSame(first, TrainGpu.availability(host));
            assertNull(TrainGpu.forBackend(host));
            assertFalse(TrainKernels.availability(host).available());
            List<GpuEvent> waits = new ArrayList<>();
            TrainGpu.beforeHostRead(host, host.transferQueue(), waits);
            assertTrue(waits.isEmpty());
        } finally {
            BackendResources.closeAll(host);
        }
    }

    @Test
    public void contextIsCachedPerBackend() {
        try (TrainParity.Session s = TrainParity.open(api)) {
            TrainGpu g = s.gpu();
            assertNotNull(g);
            assertSame(g, TrainGpu.forBackend(s.backend()));
            assertSame(s.backend(), g.backend());
            assertTrue(TrainGpu.availability(s.backend()).available());
            assertSame(TrainGpu.availability(s.backend()), TrainKernels.availability(s.backend()));
            assertTrue(s.backend().capabilities().getMaxWorkGroupSize() >= TrainGpu.LINEAR_LOCAL);
        }
    }

    @Test
    public void everyKernelIsLoaded() {
        try (TrainParity.Session s = TrainParity.open(api)) {
            TrainGpu g = s.gpu();
            for (String name : TrainOps.Kernels.ALL) {
                assertNotNull(name, g.kernel(name));
            }
        }
    }

    @Test
    public void sharedBuffersAreZeroed() {
        try (TrainParity.Session s = TrainParity.open(api)) {
            TrainGpu g = s.gpu();
            assertTrue(g.placeholder().sizeInBytes() >= 16);
            assertTrue(g.errors().sizeInBytes() >= 48);
            for (int i = 0; i < 4; i++) {
                assertEquals(0, g.readFlag(g.placeholder(), i));
            }
            for (int i = 0; i < 12; i++) {
                assertEquals(0, g.readFlag(g.errors(), i));
            }
            g.checkErrors();
        }
    }

    @Test
    public void fillAndReadScalar() {
        try (TrainParity.Session s = TrainParity.open(api)) {
            TrainGpu g = s.gpu();
            try (TensorStorage st = g.allocate(1000, DType.FLOAT32)) {
                g.zero(st.gpuBuffer());
                g.fill(st.gpuBuffer(), 10, 513, 2.5f);
                assertEquals(0.0f, g.readScalar(st.gpuBuffer(), 9), 0.0f);
                assertEquals(2.5f, g.readScalar(st.gpuBuffer(), 10), 0.0f);
                assertEquals(2.5f, g.readScalar(st.gpuBuffer(), 522), 0.0f);
                assertEquals(0.0f, g.readScalar(st.gpuBuffer(), 523), 0.0f);
            }
        }
    }

    @Test
    public void transferQueueReadSeesTrainWork() {
        try (TrainParity.Session s = TrainParity.open(api)) {
            TrainGpu g = s.gpu();
            int n = 300_000;
            try (TensorStorage st = g.allocate(n, DType.FLOAT32)) {
                g.fill(st.gpuBuffer(), 0, n, 7.0f);
                float[] host = st.copyToHostFloat(s.backend().transferQueue());
                for (int i = 0; i < n; i += 997) {
                    assertEquals(7.0f, host[i], 0.0f);
                }
                assertEquals(7.0f, host[n - 1], 0.0f);
            }
        }
    }

    @Test
    public void uploadIsVisibleToLaterLaunches() {
        try (TrainParity.Session s = TrainParity.open(api)) {
            int rows = 61;
            int cols = 67;
            float[] data = TrainParity.uniform(3, rows * cols, -4, 4);
            float[] expected = new float[rows * cols];
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    expected[j * rows + i] = data[i * cols + j];
                }
            }
            try (Tensor src = Tensor.fromArray(s.backend(), data);
                 Tensor matrix = src.reshape(Shape.of(rows, cols));
                 Tensor view = matrix.transpose()) {
                assertFalse(view.isDense());
                try (Tensor copy = view.contiguous()) {
                    assertNotSame(view, copy);
                    TrainParity.assertBitIdentical("upload then strided copy", expected, copy.toFloatArray());
                }
            }
        }
    }

    @Test
    public void beforeHostReadAddsMarkerOnlyForOtherQueuesWithPendingWork() {
        try (TrainParity.Session s = TrainParity.open(api)) {
            TrainGpu g = s.gpu();
            ComputeBackend b = s.backend();
            try (TensorStorage st = g.allocate(64, DType.FLOAT32)) {
                ReentrantLock lock = BackendResources.lock(b);
                lock.lock();
                try {
                    g.fill(st.gpuBuffer(), 0, 64, 1.0f);
                    List<GpuEvent> computeWaits = new ArrayList<>();
                    TrainGpu.beforeHostRead(b, b.computeQueue(), computeWaits);
                    assertTrue(computeWaits.isEmpty());
                    if (b.transferQueue() != b.computeQueue()) {
                        List<GpuEvent> waits = new ArrayList<>();
                        TrainGpu.beforeHostRead(b, b.transferQueue(), waits);
                        assertEquals(1, waits.size());
                        b.computeQueue().finish();
                        waits.get(0).close();
                    }
                    g.checkErrors();
                    List<GpuEvent> idle = new ArrayList<>();
                    TrainGpu.beforeHostRead(b, b.transferQueue(), idle);
                    assertTrue(idle.isEmpty());
                } finally {
                    lock.unlock();
                }
            }
        }
    }

    @Test
    public void scratchGrowsAndIsReused() {
        try (TrainParity.Session s = TrainParity.open(api)) {
            TrainGpu g = s.gpu();
            GpuBuffer a = g.scratch(0, 100);
            assertTrue(a.sizeInBytes() >= 100);
            assertEquals(0, a.sizeInBytes() % 4);
            assertSame(a, g.scratch(0, 64));
            GpuBuffer b = g.scratch(0, 101 + a.sizeInBytes());
            assertNotSame(a, b);
            assertTrue(b.sizeInBytes() >= 101 + a.sizeInBytes());
            assertNotSame(b, g.scratch(1, 16));
        }
    }

    @Test
    public void releaseOpensANewContext() {
        try (TrainParity.Session s = TrainParity.open(api)) {
            TrainGpu first = s.gpu();
            TrainGpu.release(s.backend());
            TrainGpu second = TrainGpu.forBackend(s.backend());
            assertNotNull(second);
            assertNotSame(first, second);
            first.close();
            assertSame(second, TrainGpu.forBackend(s.backend()));
            second.close();
            assertNotSame(second, TrainGpu.forBackend(s.backend()));
        }
    }

    @Test
    public void backendCloseReleasesTheContext() {
        TrainParity.Session s = TrainParity.open(api);
        ComputeBackend b = s.backend();
        ReentrantLock before;
        try {
            assertNotNull(s.gpu());
            before = BackendResources.lock(b);
        } finally {
            s.close();
        }
        assertNotSame(before, BackendResources.lock(b));
        BackendResources.closeAll(b);
    }

    @Test
    public void zeroElementLaunchesAreNoOps() {
        try (TrainParity.Session s = TrainParity.open(api)) {
            TrainGpu g = s.gpu();
            try (TensorStorage st = g.allocate(0, DType.FLOAT32)) {
                assertEquals(0, st.lengthBytes());
                assertTrue(st.gpuBuffer().sizeInBytes() >= 4);
                assertEquals(0, st.gpuBuffer().sizeInBytes() % 4);
                g.fill(st.gpuBuffer(), 0, 0, 1.0f);
                g.checkErrors();
                assertArrayEquals(new float[0], st.copyToHostFloat(s.backend().transferQueue()), 0.0f);
            }
        }
    }

    @Test
    public void outOfRangeGatherIndexThrowsTheCpuMessage() {
        float[] values = {1f, 2f, 3f};
        int[] indices = {0, 3, 1};
        ComputeBackend host = TrainParity.hostBackend();
        try (Tensor in = Tensor.fromArray(host, values); Tensor idx = Tensor.fromArray(host, indices)) {
            IndexOutOfBoundsException cpu = assertThrows(IndexOutOfBoundsException.class,
                    () -> TensorOps.gather(in, 0, idx));
            assertEquals(TrainGpu.gatherIndexMessage(3, 0, 3), cpu.getMessage());
            assertEquals("Gather index 3 out of bounds for dimension 0 with size 3", cpu.getMessage());
        } finally {
            BackendResources.closeAll(host);
        }
        try (TrainParity.Session s = TrainParity.open(api);
             Tensor in = Tensor.fromArray(s.backend(), values);
             Tensor idx = Tensor.fromArray(s.backend(), indices);
             Tensor out = TensorOps.gather(in, 0, idx)) {
            IndexOutOfBoundsException device = assertThrows(IndexOutOfBoundsException.class, out::toFloatArray);
            String prefix = TrainGpu.gatherIndexMessage(3, 0, 3) + " (reported by a device kernel";
            assertTrue(device.getMessage(), device.getMessage().startsWith(prefix));
        }
    }

    /**
     * Each deferred index error carries the index, the caller's dimension and its size, so the device message equals
     * the CPU path's, including a gather of rank above 6 whose launch merges the dimensions.
     */
    @Test
    public void deviceIndexErrorsReportIndexDimensionAndSize() {
        float[] values = {1f, 2f, 3f, 4f, 5f, 6f};
        try (TrainParity.Session s = TrainParity.open(api);
             Tensor flat = Tensor.fromArray(s.backend(), values);
             Tensor in = flat.reshape(2, 3);
             Tensor flatIdx = Tensor.fromArray(s.backend(), new int[]{0, 5, 1, 2});
             Tensor idx = flatIdx.reshape(2, 2);
             Tensor out = TensorOps.gather(in, 1, idx)) {
            IndexOutOfBoundsException e = assertThrows(IndexOutOfBoundsException.class, out::toFloatArray);
            assertTrue(e.getMessage(), e.getMessage().startsWith(TrainGpu.gatherIndexMessage(5, 1, 3) + " ("));
        }
        try (TrainParity.Session s = TrainParity.open(api);
             Tensor flat = Tensor.fromArray(s.backend(), values);
             Tensor in = flat.reshape(2, 3);
             Tensor flatIdx = Tensor.fromArray(s.backend(), new int[]{2, 0, -1, 1});
             Tensor idx = flatIdx.reshape(2, 2);
             Tensor flatSrc = Tensor.fromArray(s.backend(), new float[]{7f, 8f, 9f, 10f});
             Tensor src = flatSrc.reshape(2, 2);
             Tensor out = TensorOps.scatter(in, -1, idx, src)) {
            IndexOutOfBoundsException e = assertThrows(IndexOutOfBoundsException.class, out::toFloatArray);
            assertTrue(e.getMessage(), e.getMessage().startsWith(TrainGpu.scatterIndexMessage(-1, 1, 3) + " ("));
        }
        try (TrainParity.Session s = TrainParity.open(api);
             Tensor flat = Tensor.fromArray(s.backend(), values);
             Tensor in = flat.reshape(1, 1, 1, 1, 1, 2, 3);
             Tensor flatIdx = Tensor.fromArray(s.backend(), new int[]{1, 0, 2, 4});
             Tensor idx = flatIdx.reshape(1, 1, 1, 1, 1, 2, 2);
             Tensor out = TensorOps.gather(in, 6, idx)) {
            IndexOutOfBoundsException e = assertThrows(IndexOutOfBoundsException.class, out::toFloatArray);
            assertTrue(e.getMessage(), e.getMessage().startsWith(TrainGpu.gatherIndexMessage(4, 6, 3) + " ("));
        }
    }
}

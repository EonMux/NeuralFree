package com.eonmux.neuralfree.tensor;

import com.eonmux.neuralfree.acceleration.UnsupportedKernelException;
import com.eonmux.neuralfree.acceleration.backend.*;
import org.junit.Test;
import org.junit.Before;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import static org.junit.Assert.*;

public class TensorStorageTest {

    private MockComputeBackend mockBackend;

    @Before
    public void setUp() {
        mockBackend = new MockComputeBackend();
    }

    @Test
    public void testAllocate() {
        TensorStorage storage = TensorStorage.allocate(mockBackend, 100, DType.FLOAT32);
        
        assertEquals(400, storage.lengthBytes());
        assertEquals(100, storage.numElements());
        assertEquals(DType.FLOAT32, storage.dtype());
        assertEquals(0, storage.offsetBytes());
        assertTrue(storage.ownsBuffer());
        assertFalse(storage.isView());
    }

    @Test
    public void testAllocateInt32() {
        TensorStorage storage = TensorStorage.allocate(mockBackend, 50, DType.INT32);
        
        assertEquals(200, storage.lengthBytes());
        assertEquals(50, storage.numElements());
        assertEquals(DType.INT32, storage.dtype());
    }

    @Test
    public void testAllocateZero() {
        TensorStorage storage = TensorStorage.allocate(mockBackend, 0, DType.FLOAT32);
        
        assertEquals(0, storage.lengthBytes());
        assertEquals(0, storage.numElements());
    }

    @Test
    public void testFromHostFloat() {
        float[] data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        TensorStorage storage = TensorStorage.fromHostFloat(mockBackend, data);
        
        assertEquals(20, storage.lengthBytes());
        assertEquals(5, storage.numElements());
        assertEquals(DType.FLOAT32, storage.dtype());
        assertTrue(storage.ownsBuffer());
    }

    @Test
    public void testFromHostByteBuffer() {
        byte[] data = {1, 2, 3, 4, 5, 6, 7, 8};
        ByteBuffer buffer = ByteBuffer.wrap(data).order(ByteOrder.nativeOrder());
        
        TensorStorage storage = TensorStorage.fromHost(mockBackend, buffer, DType.INT8);
        
        assertEquals(8, storage.lengthBytes());
        assertEquals(8, storage.numElements());
        assertEquals(DType.INT8, storage.dtype());
    }

    @Test
    public void testSlice() {
        TensorStorage storage = TensorStorage.allocate(mockBackend, 100, DType.FLOAT32);
        TensorStorage slice = storage.slice(10, 20);
        
        assertEquals(40, slice.offsetBytes());
        assertEquals(80, slice.lengthBytes());
        assertEquals(20, slice.numElements());
        assertFalse(slice.ownsBuffer());
        assertTrue(slice.isView());
        assertEquals(DType.FLOAT32, slice.dtype());
    }

    @Test
    public void testSliceZeroOffset() {
        TensorStorage storage = TensorStorage.allocate(mockBackend, 100, DType.FLOAT32);
        TensorStorage slice = storage.slice(0, 50);
        
        assertEquals(0, slice.offsetBytes());
        assertEquals(200, slice.lengthBytes());
        assertFalse(slice.ownsBuffer());
    }

    @Test(expected = IllegalArgumentException.class)
    public void testSliceOutOfBounds() {
        TensorStorage storage = TensorStorage.allocate(mockBackend, 100, DType.FLOAT32);
        storage.slice(50, 100);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testSliceNegativeOffset() {
        TensorStorage storage = TensorStorage.allocate(mockBackend, 100, DType.FLOAT32);
        storage.slice(-1, 10);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testSliceNegativeLength() {
        TensorStorage storage = TensorStorage.allocate(mockBackend, 100, DType.FLOAT32);
        storage.slice(0, -1);
    }

    @Test
    public void testCopyToHostFloat() {
        float[] data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        TensorStorage storage = TensorStorage.fromHostFloat(mockBackend, data);
        
        float[] result = storage.copyToHostFloat(mockBackend.transferQueue());
        
        assertEquals(5, result.length);
    }

    @Test
    public void testCopyFromHostFloat() {
        TensorStorage storage = TensorStorage.allocate(mockBackend, 5, DType.FLOAT32);
        float[] data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        
        storage.copyFromHostFloat(data, mockBackend.transferQueue());
    }

    @Test(expected = IllegalArgumentException.class)
    public void testCopyFromHostFloatTooLarge() {
        TensorStorage storage = TensorStorage.allocate(mockBackend, 3, DType.FLOAT32);
        float[] data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        
        storage.copyFromHostFloat(data, mockBackend.transferQueue());
    }

    @Test
    public void testCopyToHostByteBuffer() {
        TensorStorage storage = TensorStorage.allocate(mockBackend, 8, DType.INT8);
        ByteBuffer dst = ByteBuffer.allocate(8).order(ByteOrder.nativeOrder());
        
        storage.copyToHost(dst, mockBackend.transferQueue());
        
        assertEquals(8, dst.position());
    }

    @Test
    public void testGpuBuffer() {
        TensorStorage storage = TensorStorage.allocate(mockBackend, 100, DType.FLOAT32);
        
        GpuBuffer buffer = storage.gpuBuffer();
        
        assertNotNull(buffer);
        assertEquals(400, buffer.sizeInBytes());
    }

    @Test
    public void testClose() {
        TensorStorage storage = TensorStorage.allocate(mockBackend, 100, DType.FLOAT32);
        
        assertFalse(storage.toString().contains("closed=true"));
        storage.close();
    }

    @Test
    public void testCloseViewDoesNotCloseBuffer() {
        TensorStorage storage = TensorStorage.allocate(mockBackend, 100, DType.FLOAT32);
        TensorStorage view = storage.slice(0, 50);
        
        view.close();
        
        assertNotNull(storage.gpuBuffer());
    }

    @Test(expected = IllegalStateException.class)
    public void testOperationAfterCloseThrows() {
        TensorStorage storage = TensorStorage.allocate(mockBackend, 100, DType.FLOAT32);
        storage.close();
        
        storage.slice(0, 10);
    }

    @Test(expected = IllegalStateException.class)
    public void testCopyAfterCloseThrows() {
        TensorStorage storage = TensorStorage.allocate(mockBackend, 100, DType.FLOAT32);
        storage.close();
        
        storage.copyToHostFloat(mockBackend.transferQueue());
    }

    @Test
    public void testSliceOfSlice() {
        TensorStorage storage = TensorStorage.allocate(mockBackend, 100, DType.FLOAT32);
        TensorStorage slice1 = storage.slice(10, 40);
        TensorStorage slice2 = slice1.slice(10, 20);
        
        assertEquals(80, slice2.offsetBytes());
        assertEquals(80, slice2.lengthBytes());
        assertFalse(slice2.ownsBuffer());
    }

    @Test
    public void testToString() {
        TensorStorage storage = TensorStorage.allocate(mockBackend, 100, DType.FLOAT32);
        String str = storage.toString();
        
        assertTrue(str.contains("TensorStorage"));
        assertTrue(str.contains("lengthBytes=400"));
        assertTrue(str.contains("dtype=FLOAT32"));
        assertTrue(str.contains("isPrimaryOwner=true"));
    }

    @Test(expected = NullPointerException.class)
    public void testAllocateNullBackend() {
        TensorStorage.allocate(null, 100, DType.FLOAT32);
    }

    @Test(expected = NullPointerException.class)
    public void testAllocateNullDType() {
        TensorStorage.allocate(mockBackend, 100, null);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testAllocateNegativeElements() {
        TensorStorage.allocate(mockBackend, -1, DType.FLOAT32);
    }

    @Test(expected = NullPointerException.class)
    public void testFromHostFloatNullBackend() {
        TensorStorage.fromHostFloat(null, new float[]{1.0f});
    }

    @Test(expected = NullPointerException.class)
    public void testFromHostFloatNullData() {
        TensorStorage.fromHostFloat(mockBackend, null);
    }

    @Test
    public void testRefCountingInitialValue() {
        TensorStorage storage = TensorStorage.allocate(mockBackend, 100, DType.FLOAT32);
        assertEquals(1, storage.refCount());
        storage.close();
    }

    @Test
    public void testRefCountingSliceIncrements() {
        TensorStorage storage = TensorStorage.allocate(mockBackend, 100, DType.FLOAT32);
        assertEquals(1, storage.refCount());
        
        TensorStorage slice = storage.slice(10, 20);
        assertEquals(2, storage.refCount());
        assertEquals(2, slice.refCount());
        
        slice.close();
        assertEquals(1, storage.refCount());
        storage.close();
    }

    @Test
    public void testBufferNotFreedWhileViewExists() {
        TensorStorage storage = TensorStorage.allocate(mockBackend, 100, DType.FLOAT32);
        MockGpuBuffer buffer = (MockGpuBuffer) storage.gpuBuffer();
        
        TensorStorage view = storage.slice(0, 50);
        
        storage.close();
        
        assertFalse("Buffer should not be closed while view exists", buffer.isClosed());
        assertEquals(1, view.refCount());
        
        assertNotNull("View should still work after original closed", view.gpuBuffer());
        view.close();
        assertTrue("Buffer should be closed after all references", buffer.isClosed());
    }

    @Test
    public void testMultipleViewsKeepBufferAlive() {
        TensorStorage storage = TensorStorage.allocate(mockBackend, 100, DType.FLOAT32);
        MockGpuBuffer buffer = (MockGpuBuffer) storage.gpuBuffer();
        
        TensorStorage view1 = storage.slice(0, 25);
        TensorStorage view2 = storage.slice(25, 25);
        TensorStorage view3 = storage.slice(50, 25);
        
        assertEquals(4, storage.refCount());
        
        storage.close();
        assertFalse("Buffer alive after original closed", buffer.isClosed());
        
        view1.close();
        assertFalse("Buffer alive after view1 closed", buffer.isClosed());
        
        view2.close();
        assertFalse("Buffer alive after view2 closed", buffer.isClosed());
        
        view3.close();
        assertTrue("Buffer freed after all references closed", buffer.isClosed());
    }

    @Test
    public void testSliceOfSliceRefCounting() {
        TensorStorage storage = TensorStorage.allocate(mockBackend, 100, DType.FLOAT32);
        MockGpuBuffer buffer = (MockGpuBuffer) storage.gpuBuffer();
        
        TensorStorage slice1 = storage.slice(10, 40);
        TensorStorage slice2 = slice1.slice(10, 20);
        
        assertEquals(3, storage.refCount());
        assertEquals(3, slice1.refCount());
        assertEquals(3, slice2.refCount());
        
        storage.close();
        assertEquals(2, slice1.refCount());
        assertFalse("Buffer alive after original closed", buffer.isClosed());
        
        slice1.close();
        assertEquals(1, slice2.refCount());
        assertFalse("Buffer alive after slice1 closed", buffer.isClosed());
        
        slice2.close();
        assertTrue("Buffer freed after all closed", buffer.isClosed());
    }

    @Test
    public void testStressCreateClose1000Tensors() {
        long totalBefore = mockBackend.aliveBufferCount();
        
        for (int i = 0; i < 1000; i++) {
            TensorStorage storage = TensorStorage.allocate(mockBackend, 100, DType.FLOAT32);
            storage.close();
        }
        
        long totalAfter = mockBackend.aliveBufferCount();
        assertEquals("All buffers should be freed", totalBefore, totalAfter);
    }

    @Test
    public void testStressCreateClose1000TensorsWithViews() {
        long totalBefore = mockBackend.aliveBufferCount();
        
        for (int i = 0; i < 100; i++) {
            TensorStorage storage = TensorStorage.allocate(mockBackend, 100, DType.FLOAT32);
            TensorStorage view1 = storage.slice(0, 25);
            TensorStorage view2 = storage.slice(25, 25);
            TensorStorage view3 = storage.slice(50, 25);
            
            storage.close();
            view1.close();
            view2.close();
            view3.close();
        }
        
        long totalAfter = mockBackend.aliveBufferCount();
        assertEquals("All buffers should be freed with views", totalBefore, totalAfter);
    }

    @Test
    public void testViewLifecycleAfterOriginalClosed() {
        TensorStorage storage = TensorStorage.allocate(mockBackend, 100, DType.FLOAT32);
        TensorStorage view = storage.slice(0, 3);
        
        assertEquals(2, storage.refCount());
        assertEquals(2, view.refCount());
        
        storage.close();
        
        assertTrue("Storage should be marked closed", storage.toString().contains("closed=true"));
        assertEquals(1, view.refCount());
        
        assertFalse("View should still be usable", view.toString().contains("closed=true"));
        assertEquals(3, view.numElements());
        
        view.close();
        assertEquals(0, mockBackend.aliveBufferCount());
    }

    @Test
    public void testIsViewMethod() {
        TensorStorage storage = TensorStorage.allocate(mockBackend, 100, DType.FLOAT32);
        assertFalse("Full storage is not a view", storage.isView());
        assertTrue("Full storage owns buffer", storage.ownsBuffer());
        
        TensorStorage slice = storage.slice(0, 50);
        assertTrue("Partial slice is a view", slice.isView());
        assertFalse("Partial slice does not own buffer", slice.ownsBuffer());
        
        TensorStorage fullSlice = storage.slice(0, 100);
        assertTrue("Full slice is still a view (shares buffer)", fullSlice.isView());
        assertFalse("Full slice does not own buffer", fullSlice.ownsBuffer());
        
        TensorStorage share = storage.share();
        assertTrue("Share is a view", share.isView());
        assertFalse("Share does not own buffer", share.ownsBuffer());
        
        storage.close();
        slice.close();
        fullSlice.close();
        share.close();
    }

    /** Declines kernel artifacts so tensor ops take their CPU path. */
    private static class MockComputeBackend implements ComputeBackend {
        private final MockComputeQueue computeQueue = new MockComputeQueue();
        private final MockComputeQueue transferQueue = new MockComputeQueue();
        private volatile boolean closed = false;
        private final java.util.concurrent.atomic.AtomicLong aliveBufferCount = new java.util.concurrent.atomic.AtomicLong(0);

        @Override
        public BackendType backendType() {
            return BackendType.OPENCL;
        }

        @Override
        public DeviceCapabilities capabilities() {
            return new DeviceCapabilities.Builder()
                    .deviceName("MockDevice")
                    .totalVramBytes(1024 * 1024 * 1024)
                    .computeUnits(8)
                    .maxClockFrequencyMhz(1000)
                    .maxWorkGroupSize(256)
                    .supportsAsyncTransfer(true)
                    .driverVersion("1.0")
                    .build();
        }

        @Override
        public ComputeQueue computeQueue() {
            return computeQueue;
        }

        @Override
        public ComputeQueue transferQueue() {
            return transferQueue;
        }

        @Override
        public GpuBuffer allocateBuffer(long sizeInBytes) {
            aliveBufferCount.incrementAndGet();
            return new MockGpuBuffer(sizeInBytes, aliveBufferCount);
        }

        @Override
        public GpuBuffer allocateBuffer(long sizeInBytes, boolean readOnly) {
            aliveBufferCount.incrementAndGet();
            return new MockGpuBuffer(sizeInBytes, aliveBufferCount);
        }

        @Override
        public KernelHandle compileKernel(String kernelName, String kernelSource, String buildOptions) {
            return new MockKernelHandle(kernelName);
        }

        @Override
        public boolean supportsArtifact(KernelArtifact.Kind kind) {
            return false;
        }

        @Override
        public KernelHandle loadKernel(KernelArtifact artifact) {
            if (artifact == null) {
                throw new IllegalArgumentException("kernel artifact must not be null");
            }
            throw new UnsupportedKernelException("mock backend declines " + artifact.kind()
                    + " artifact for kernel " + artifact.kernelName());
        }

        @Override
        public void close() {
            closed = true;
        }

        long aliveBufferCount() {
            return aliveBufferCount.get();
        }
    }

    private static class MockComputeQueue implements ComputeQueue {
        @Override
        public QueueType getQueueType() {
            return QueueType.TRANSFER;
        }

        @Override
        public GpuEvent enqueueWriteBuffer(GpuBuffer dstBuffer, long dstOffset, long sizeInBytes, byte[] srcData, GpuEvent... waitOnEvents) {
            return new MockGpuEvent();
        }

        @Override
        public GpuEvent enqueueWriteBuffer(GpuBuffer dstBuffer, long dstOffset, long sizeInBytes, float[] srcData, GpuEvent... waitOnEvents) {
            return new MockGpuEvent();
        }

        @Override
        public GpuEvent enqueueReadBuffer(GpuBuffer srcBuffer, long srcOffset, long sizeInBytes, byte[] dstData, GpuEvent... waitOnEvents) {
            return new MockGpuEvent();
        }

        @Override
        public GpuEvent enqueueReadBuffer(GpuBuffer srcBuffer, long srcOffset, long sizeInBytes, float[] dstData, GpuEvent... waitOnEvents) {
            return new MockGpuEvent();
        }

        @Override
        public GpuEvent enqueueKernel(KernelHandle kernel, long[] globalWorkSize, long[] localWorkSize, Object[] args, GpuEvent... waitOnEvents) {
            return new MockGpuEvent();
        }

        @Override
        public void flush() {
        }

        @Override
        public void finish() {
        }

        @Override
        public void close() {
        }
    }

    private static class MockGpuBuffer implements GpuBuffer {
        private final long sizeInBytes;
        private volatile boolean closed = false;
        private final java.util.concurrent.atomic.AtomicLong aliveCounter;

        MockGpuBuffer(long sizeInBytes, java.util.concurrent.atomic.AtomicLong aliveCounter) {
            this.sizeInBytes = sizeInBytes;
            this.aliveCounter = aliveCounter;
        }

        @Override
        public long sizeInBytes() {
            return sizeInBytes;
        }

        @Override
        public long bufferId() {
            return System.identityHashCode(this);
        }

        @Override
        public void close() {
            if (!closed) {
                closed = true;
                aliveCounter.decrementAndGet();
            }
        }

        boolean isClosed() {
            return closed;
        }
    }

    private static class MockGpuEvent implements GpuEvent {
        private volatile boolean complete = true;
        private volatile boolean closed = false;

        @Override
        public void awaitCompletion() throws InterruptedException {
            complete = true;
        }

        @Override
        public boolean isComplete() {
            return complete;
        }

        @Override
        public void close() {
            closed = true;
        }
    }

    private static class MockKernelHandle implements KernelHandle {
        private final String kernelName;
        private volatile boolean closed = false;

        MockKernelHandle(String kernelName) {
            this.kernelName = kernelName;
        }

        @Override
        public String kernelName() {
            return kernelName;
        }

        @Override
        public void close() {
            closed = true;
        }
    }
}

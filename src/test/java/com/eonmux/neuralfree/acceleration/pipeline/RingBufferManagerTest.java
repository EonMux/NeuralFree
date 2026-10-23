package com.eonmux.neuralfree.acceleration.pipeline;

import com.eonmux.neuralfree.acceleration.UnsupportedKernelException;
import com.eonmux.neuralfree.acceleration.backend.*;
import com.eonmux.neuralfree.acceleration.memory.GpuMemoryPool;
import com.eonmux.neuralfree.acceleration.memory.MemoryBlock;
import com.eonmux.neuralfree.acceleration.memory.PoolConfig;
import com.eonmux.neuralfree.acceleration.pipeline.schedule.PipelineConfig;
import org.junit.Test;

import java.util.List;

import static org.junit.Assert.*;

public class RingBufferManagerTest {

    @Test
    public void testConstructionWithDoubleBuffer() {
        ComputeBackend backend = new MockComputeBackend();
        PipelineConfig config = PipelineConfig.builder()
                .enablePrefetch(true)
                .prefetchDepth(1)
                .build();
        
        RingBufferManager manager = new RingBufferManager(backend, config);
        
        assertEquals("Double buffer should have 2 slots", 2, manager.getSlotCount());
        assertEquals("Initial compute slot index should be 0", 0, manager.getComputeSlotIndex());
        assertEquals("Prefetch depth should be 1", 1, manager.getPrefetchDepth());
        assertTrue("Prefetch should be enabled", manager.isPrefetchEnabled());
        
        manager.close();
    }

    @Test
    public void testConstructionWithTripleBuffer() {
        ComputeBackend backend = new MockComputeBackend();
        PipelineConfig config = PipelineConfig.builder()
                .enablePrefetch(true)
                .prefetchDepth(2)
                .build();
        
        RingBufferManager manager = new RingBufferManager(backend, config);
        
        assertEquals("Triple buffer should have 3 slots", 3, manager.getSlotCount());
        assertEquals("Prefetch depth should be 2", 2, manager.getPrefetchDepth());
        
        manager.close();
    }

    @Test
    public void testConstructionWithPrefetchDisabled() {
        ComputeBackend backend = new MockComputeBackend();
        PipelineConfig config = PipelineConfig.builder()
                .enablePrefetch(false)
                .build();
        
        RingBufferManager manager = new RingBufferManager(backend, config);
        
        assertEquals("Disabled prefetch should have 1 slot", 1, manager.getSlotCount());
        assertEquals("Prefetch depth should be 0", 0, manager.getPrefetchDepth());
        assertFalse("Prefetch should be disabled", manager.isPrefetchEnabled());
        
        manager.close();
    }

    @Test
    public void testActiveBufferReturnsCurrentSlot() {
        ComputeBackend backend = new MockComputeBackend();
        PipelineConfig config = PipelineConfig.defaultConfig();
        
        RingBufferManager manager = new RingBufferManager(backend, config);
        
        RingBufferManager.BufferSlot slot = manager.activeBuffer();
        assertNotNull("Active buffer should return a slot", slot);
        assertEquals("Active slot should have index 0 initially", 0, slot.slotIndex());
        
        manager.close();
    }

    @Test
    public void testPrefetchSlotWithValidAhead() {
        ComputeBackend backend = new MockComputeBackend();
        PipelineConfig config = PipelineConfig.builder()
                .enablePrefetch(true)
                .prefetchDepth(1)
                .build();
        
        RingBufferManager manager = new RingBufferManager(backend, config);
        
        RingBufferManager.BufferSlot slot1 = manager.prefetchSlot(1);
        assertNotNull("Prefetch slot should return a slot", slot1);
        assertEquals("Prefetch slot 1 ahead should have index 1", 1, slot1.slotIndex());
        
        manager.close();
    }

    @Test(expected = IllegalArgumentException.class)
    public void testPrefetchSlotWithInvalidAheadTooHigh() {
        ComputeBackend backend = new MockComputeBackend();
        PipelineConfig config = PipelineConfig.builder()
                .enablePrefetch(true)
                .prefetchDepth(1)
                .build();
        
        RingBufferManager manager = new RingBufferManager(backend, config);
        try {
            manager.prefetchSlot(2);
        } finally {
            manager.close();
        }
    }

    @Test(expected = IllegalArgumentException.class)
    public void testPrefetchSlotWithInvalidAheadZero() {
        ComputeBackend backend = new MockComputeBackend();
        PipelineConfig config = PipelineConfig.builder()
                .enablePrefetch(true)
                .prefetchDepth(1)
                .build();
        
        RingBufferManager manager = new RingBufferManager(backend, config);
        try {
            manager.prefetchSlot(0);
        } finally {
            manager.close();
        }
    }

    @Test
    public void testAdvanceRotatesRingBuffer() {
        ComputeBackend backend = new MockComputeBackend();
        PipelineConfig config = PipelineConfig.builder()
                .enablePrefetch(true)
                .prefetchDepth(1)
                .build();
        
        RingBufferManager manager = new RingBufferManager(backend, config);
        
        assertEquals("Initial compute slot index should be 0", 0, manager.getComputeSlotIndex());
        
        manager.advance();
        assertEquals("After first advance, compute slot index should be 1", 1, manager.getComputeSlotIndex());
        
        manager.advance();
        assertEquals("After second advance, compute slot index should wrap to 0", 0, manager.getComputeSlotIndex());
        
        manager.close();
    }

    @Test
    public void testAdvanceAsyncRotatesRingBuffer() {
        ComputeBackend backend = new MockComputeBackend();
        PipelineConfig config = PipelineConfig.builder()
                .enablePrefetch(true)
                .prefetchDepth(1)
                .build();
        
        RingBufferManager manager = new RingBufferManager(backend, config);
        
        manager.advanceAsync();
        assertEquals("After advanceAsync, compute slot index should be 1", 1, manager.getComputeSlotIndex());
        
        manager.advanceAsync();
        assertEquals("After second advanceAsync, compute slot index should wrap to 0", 0, manager.getComputeSlotIndex());
        
        manager.close();
    }

    @Test
    public void testAllocateSlotBuffer() {
        ComputeBackend backend = new MockComputeBackend();
        PipelineConfig config = PipelineConfig.defaultConfig();
        
        RingBufferManager manager = new RingBufferManager(backend, config);
        
        long bufferSize = 1024;
        manager.allocateSlotBuffer(0, bufferSize);
        
        RingBufferManager.BufferSlot slot = manager.activeBuffer();
        assertTrue("Slot should have buffer after allocation", slot.hasBuffer());
        assertEquals("Buffer size should match", bufferSize, slot.getBufferSizeBytes());
        
        manager.close();
    }

    @Test
    public void testAllocateSlotBufferFromPool() {
        MockComputeBackend backend = new MockComputeBackend();
        PipelineConfig config = PipelineConfig.defaultConfig();
        GpuMemoryPool pool = new GpuMemoryPool(backend, 
                PoolConfig.builder().maxPoolSizeBytes(1024 * 1024).build());
        
        RingBufferManager manager = new RingBufferManager(backend, config);
        
        long bufferSize = 1024;
        manager.allocateSlotBufferFromPool(0, bufferSize, pool);
        
        RingBufferManager.BufferSlot slot = manager.activeBuffer();
        assertTrue("Slot should have buffer after allocation from pool", slot.hasBuffer());
        
        manager.close();
    }

    @Test(expected = IllegalArgumentException.class)
    public void testAllocateSlotBufferInvalidIndex() {
        ComputeBackend backend = new MockComputeBackend();
        PipelineConfig config = PipelineConfig.builder()
                .enablePrefetch(true)
                .prefetchDepth(1)
                .build();
        
        RingBufferManager manager = new RingBufferManager(backend, config);
        try {
            manager.allocateSlotBuffer(5, 1024);
        } finally {
            manager.close();
        }
    }

    @Test
    public void testReset() {
        ComputeBackend backend = new MockComputeBackend();
        PipelineConfig config = PipelineConfig.builder()
                .enablePrefetch(true)
                .prefetchDepth(1)
                .build();
        
        RingBufferManager manager = new RingBufferManager(backend, config);
        
        manager.advance();
        manager.advance();
        assertEquals("After two advances, compute slot index should be 0", 0, manager.getComputeSlotIndex());
        
        manager.advance();
        assertEquals("After three advances, compute slot index should be 1", 1, manager.getComputeSlotIndex());
        
        manager.reset();
        assertEquals("After reset, compute slot index should be 0", 0, manager.getComputeSlotIndex());
        
        manager.close();
    }

    @Test
    public void testGetAllSlots() {
        ComputeBackend backend = new MockComputeBackend();
        PipelineConfig config = PipelineConfig.builder()
                .enablePrefetch(true)
                .prefetchDepth(2)
                .build();
        
        RingBufferManager manager = new RingBufferManager(backend, config);
        
        List<RingBufferManager.BufferSlot> allSlots = manager.getAllSlots();
        assertEquals("Should have 3 slots", 3, allSlots.size());
        
        for (int i = 0; i < 3; i++) {
            assertEquals("Slot index should match", i, allSlots.get(i).slotIndex());
        }
        
        manager.close();
    }

    @Test
    public void testBufferSlotInitiallyReady() {
        ComputeBackend backend = new MockComputeBackend();
        PipelineConfig config = PipelineConfig.defaultConfig();
        
        RingBufferManager manager = new RingBufferManager(backend, config);
        
        RingBufferManager.BufferSlot slot = manager.activeBuffer();
        assertTrue("Slot should be initially ready", slot.isReady());
        assertFalse("Slot should not be prefetching initially", slot.isPrefetchInProgress());
        
        manager.close();
    }

    @Test
    public void testBufferSlotToString() {
        ComputeBackend backend = new MockComputeBackend();
        PipelineConfig config = PipelineConfig.defaultConfig();
        
        RingBufferManager manager = new RingBufferManager(backend, config);
        
        RingBufferManager.BufferSlot slot = manager.activeBuffer();
        String str = slot.toString();
        
        assertTrue("toString should contain index", str.contains("index=0"));
        assertTrue("toString should contain hasBuffer", str.contains("hasBuffer"));
        assertTrue("toString should contain ready", str.contains("ready"));
        
        manager.close();
    }

    @Test(expected = IllegalStateException.class)
    public void testBeginPrefetchWithoutBuffer() {
        ComputeBackend backend = new MockComputeBackend();
        PipelineConfig config = PipelineConfig.builder()
                .enablePrefetch(true)
                .prefetchDepth(1)
                .build();
        
        RingBufferManager manager = new RingBufferManager(backend, config);
        try {
            float[] data = {1.0f, 2.0f, 3.0f};
            manager.beginPrefetch(1, data, 0);
        } finally {
            manager.close();
        }
    }

    @Test(expected = IllegalStateException.class)
    public void testActiveBufferAfterClose() {
        ComputeBackend backend = new MockComputeBackend();
        PipelineConfig config = PipelineConfig.defaultConfig();
        
        RingBufferManager manager = new RingBufferManager(backend, config);
        manager.close();
        
        manager.activeBuffer();
    }

    @Test(expected = IllegalStateException.class)
    public void testAdvanceAfterClose() {
        ComputeBackend backend = new MockComputeBackend();
        PipelineConfig config = PipelineConfig.defaultConfig();
        
        RingBufferManager manager = new RingBufferManager(backend, config);
        manager.close();
        
        manager.advance();
    }

    @Test
    public void testCloseIsIdempotent() {
        ComputeBackend backend = new MockComputeBackend();
        PipelineConfig config = PipelineConfig.defaultConfig();
        
        RingBufferManager manager = new RingBufferManager(backend, config);
        manager.close();
        manager.close();
    }

    @Test
    public void testRingBufferWrapsCorrectly() {
        ComputeBackend backend = new MockComputeBackend();
        PipelineConfig config = PipelineConfig.builder()
                .enablePrefetch(true)
                .prefetchDepth(1)
                .build();
        
        RingBufferManager manager = new RingBufferManager(backend, config);
        
        for (int i = 0; i < 10; i++) {
            int expectedIndex = i % 2;
            assertEquals("Compute slot index should cycle correctly", 
                    expectedIndex, manager.getComputeSlotIndex());
            manager.advance();
        }
        
        manager.close();
    }

    @Test
    public void testPrefetchSlotWrapsCorrectly() {
        ComputeBackend backend = new MockComputeBackend();
        PipelineConfig config = PipelineConfig.builder()
                .enablePrefetch(true)
                .prefetchDepth(1)
                .build();
        
        RingBufferManager manager = new RingBufferManager(backend, config);
        
        assertEquals("Initial: prefetch 1 ahead should be slot 1", 1, manager.prefetchSlot(1).slotIndex());
        
        manager.advance();
        assertEquals("After 1 advance: prefetch 1 ahead should wrap to slot 0", 0, manager.prefetchSlot(1).slotIndex());
        
        manager.advance();
        assertEquals("After 2 advances: prefetch 1 ahead should be slot 1 again", 1, manager.prefetchSlot(1).slotIndex());
        
        manager.close();
    }

    @Test
    public void testTripleBufferPrefetchSlots() {
        ComputeBackend backend = new MockComputeBackend();
        PipelineConfig config = PipelineConfig.builder()
                .enablePrefetch(true)
                .prefetchDepth(2)
                .build();
        
        RingBufferManager manager = new RingBufferManager(backend, config);
        
        assertEquals("Triple buffer: prefetch 1 ahead should be slot 1", 1, manager.prefetchSlot(1).slotIndex());
        assertEquals("Triple buffer: prefetch 2 ahead should be slot 2", 2, manager.prefetchSlot(2).slotIndex());
        
        manager.close();
    }

    /** Declines kernel artifacts so tensor ops take their CPU path. */
    private static class MockComputeBackend implements ComputeBackend {
        private final MockComputeQueue computeQueue = new MockComputeQueue();
        private final MockComputeQueue transferQueue = new MockComputeQueue();
        private volatile boolean closed = false;

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
            return new MockGpuBuffer(sizeInBytes);
        }

        @Override
        public GpuBuffer allocateBuffer(long sizeInBytes, boolean readOnly) {
            return new MockGpuBuffer(sizeInBytes);
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
    }

    private static class MockComputeQueue implements ComputeQueue {
        @Override
        public QueueType getQueueType() {
            return QueueType.COMPUTE;
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

        MockGpuBuffer(long sizeInBytes) {
            this.sizeInBytes = sizeInBytes;
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
            closed = true;
        }
    }

    private static class MockGpuEvent implements GpuEvent {
        private volatile boolean complete = false;
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

package com.eonmux.neuralfree.acceleration.memory;

import com.eonmux.neuralfree.acceleration.GpuOutOfMemoryException;
import com.eonmux.neuralfree.acceleration.UnsupportedKernelException;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.KernelArtifact;
import com.eonmux.neuralfree.acceleration.backend.KernelHandle;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

import static org.junit.Assert.*;

public class GpuMemoryPoolTest {

    @Test
    public void testAcquireReturnsCorrectSize() {
        MockBackend backend = new MockBackend(1024 * 1024 * 100);
        GpuMemoryPool pool = new GpuMemoryPool(backend, PoolConfig.builder()
                .maxPoolSizeBytes(1024 * 1024 * 80)
                .minBlockSizeBytes(4096)
                .defaultBlockSizeBytes(1024 * 1024)
                .build());

        MemoryBlock block = pool.acquire(8192);
        assertNotNull(block);
        assertTrue(block.isInUse());
        assertEquals(8192, block.getSizeBytes());
        pool.close();
    }

    @Test
    public void testReleaseAllowsReuse() {
        MockBackend backend = new MockBackend(1024 * 1024 * 100);
        GpuMemoryPool pool = new GpuMemoryPool(backend, PoolConfig.builder()
                .maxPoolSizeBytes(1024 * 1024 * 80)
                .minBlockSizeBytes(4096)
                .defaultBlockSizeBytes(1024 * 1024)
                .build());

        MemoryBlock block1 = pool.acquire(8192);
        long bufferId = block1.bufferId();
        pool.release(block1);

        MemoryBlock block2 = pool.acquire(8192);
        assertEquals(bufferId, block2.bufferId());
        assertEquals(1, pool.getBlockCount());
        pool.close();
    }

    @Test
    public void testPoolRespectsMaxSizeLimit() {
        MockBackend backend = new MockBackend(1024 * 1024 * 100);
        GpuMemoryPool pool = new GpuMemoryPool(backend, PoolConfig.builder()
                .maxPoolSizeBytes(16384)
                .minBlockSizeBytes(4096)
                .defaultBlockSizeBytes(4096)
                .build());

        MemoryBlock block1 = pool.acquire(4096);
        assertNotNull(block1);
        assertEquals(4096, pool.getTotalAllocatedBytes());

        MemoryBlock block2 = pool.acquire(4096);
        assertNotNull(block2);
        assertEquals(8192, pool.getTotalAllocatedBytes());

        MemoryBlock block3 = pool.acquire(4096);
        assertNotNull(block3);
        assertEquals(12288, pool.getTotalAllocatedBytes());

        MemoryBlock block4 = pool.acquire(4096);
        assertNotNull(block4);
        assertEquals(16384, pool.getTotalAllocatedBytes());

        try {
            pool.acquire(4096);
            fail("Expected GpuOutOfMemoryException due to pool limit");
        } catch (GpuOutOfMemoryException e) {
            assertTrue(e.getMessage().contains("Memory pool limit exceeded"));
        }

        pool.close();
    }

    @Test
    public void testBestFitPicksSmallestAdequate() {
        MockBackend backend = new MockBackend(1024 * 1024 * 100);
        GpuMemoryPool pool = new GpuMemoryPool(backend, PoolConfig.builder()
                .maxPoolSizeBytes(1024 * 1024 * 80)
                .minBlockSizeBytes(1024)
                .defaultBlockSizeBytes(1024 * 1024)
                .build());

        MemoryBlock small = pool.acquire(4096);
        MemoryBlock large = pool.acquire(16384);
        MemoryBlock huge = pool.acquire(65536);

        pool.release(small);
        pool.release(large);
        pool.release(huge);

        assertEquals(3, pool.getAvailableBlockCount());

        MemoryBlock reused = pool.acquire(8192);
        assertEquals(large.bufferId(), reused.bufferId());
        assertEquals(16384, reused.getSizeBytes());

        pool.close();
    }

    @Test
    public void testSizeAlignment() {
        MockBackend backend = new MockBackend(1024 * 1024 * 100);
        GpuMemoryPool pool = new GpuMemoryPool(backend, PoolConfig.builder()
                .maxPoolSizeBytes(1024 * 1024 * 80)
                .minBlockSizeBytes(4096)
                .defaultBlockSizeBytes(1024 * 1024)
                .build());

        MemoryBlock block1 = pool.acquire(100);
        assertEquals(4096, block1.getSizeBytes());

        MemoryBlock block2 = pool.acquire(4096);
        assertEquals(4096, block2.getSizeBytes());

        MemoryBlock block3 = pool.acquire(5000);
        assertEquals(8192, block3.getSizeBytes());

        pool.close();
    }

    @Test
    public void testCloseReleasesAllBuffers() {
        MockBackend backend = new MockBackend(1024 * 1024 * 100);
        GpuMemoryPool pool = new GpuMemoryPool(backend, PoolConfig.builder()
                .maxPoolSizeBytes(1024 * 1024 * 80)
                .minBlockSizeBytes(4096)
                .defaultBlockSizeBytes(1024 * 1024)
                .build());

        pool.acquire(4096);
        pool.acquire(8192);
        assertEquals(2, backend.getAllocatedBufferCount());

        pool.close();
        assertEquals(0, backend.getAllocatedBufferCount());
        assertEquals(0, pool.getBlockCount());
    }

    @Test
    public void testMultipleReleasesAndAcquires() {
        MockBackend backend = new MockBackend(1024 * 1024 * 100);
        GpuMemoryPool pool = new GpuMemoryPool(backend, PoolConfig.builder()
                .maxPoolSizeBytes(1024 * 1024 * 80)
                .minBlockSizeBytes(4096)
                .defaultBlockSizeBytes(1024 * 1024)
                .build());

        MemoryBlock b1 = pool.acquire(4096);
        MemoryBlock b2 = pool.acquire(4096);
        MemoryBlock b3 = pool.acquire(4096);
        assertEquals(3, pool.getBlockCount());
        assertEquals(0, pool.getAvailableBlockCount());

        pool.release(b1);
        pool.release(b3);
        assertEquals(2, pool.getAvailableBlockCount());

        MemoryBlock b4 = pool.acquire(4096);
        MemoryBlock b5 = pool.acquire(4096);
        assertTrue(b4.bufferId() == b1.bufferId() || b4.bufferId() == b3.bufferId());
        assertTrue(b5.bufferId() == b1.bufferId() || b5.bufferId() == b3.bufferId());

        assertEquals(3, pool.getBlockCount());
        assertEquals(0, pool.getAvailableBlockCount());

        pool.close();
    }

    @Test(expected = IllegalStateException.class)
    public void testAcquireAfterCloseThrows() {
        MockBackend backend = new MockBackend(1024 * 1024 * 100);
        GpuMemoryPool pool = new GpuMemoryPool(backend);
        pool.close();
        pool.acquire(4096);
    }

    private static class MockBackend implements ComputeBackend {
        private final DeviceCapabilities capabilities;
        private final AtomicLong bufferIdCounter = new AtomicLong(1);
        private final List<GpuBuffer> allocatedBuffers = new ArrayList<>();

        MockBackend(long vramBytes) {
            this.capabilities = new DeviceCapabilities.Builder()
                    .deviceName("MockDevice")
                    .totalVramBytes(vramBytes)
                    .computeUnits(16)
                    .maxClockFrequencyMhz(1500)
                    .maxWorkGroupSize(256)
                    .supportsAsyncTransfer(true)
                    .driverVersion("1.0")
                    .build();
        }

        @Override
        public GpuBuffer allocateBuffer(long sizeInBytes, boolean readOnly) {
            MockBuffer buffer = new MockBuffer(bufferIdCounter.getAndIncrement(), sizeInBytes);
            synchronized (allocatedBuffers) {
                allocatedBuffers.add(buffer);
            }
            return buffer;
        }

        @Override
        public GpuBuffer allocateBuffer(long sizeInBytes) {
            return allocateBuffer(sizeInBytes, false);
        }

        int getAllocatedBufferCount() {
            synchronized (allocatedBuffers) {
                return (int) allocatedBuffers.stream().filter(b -> !((MockBuffer) b).closed).count();
            }
        }

        @Override
        public BackendType backendType() { return BackendType.OPENCL; }
        @Override
        public DeviceCapabilities capabilities() { return capabilities; }
        @Override
        public ComputeQueue computeQueue() { return null; }
        @Override
        public ComputeQueue transferQueue() { return null; }
        @Override
        public KernelHandle compileKernel(String kernelName, String kernelSource, String buildOptions) { return null; }
        /** Declines kernel artifacts so tensor ops take their CPU path. */
        @Override
        public boolean supportsArtifact(KernelArtifact.Kind kind) { return false; }
        @Override
        public KernelHandle loadKernel(KernelArtifact artifact) {
            if (artifact == null) {
                throw new IllegalArgumentException("kernel artifact must not be null");
            }
            throw new UnsupportedKernelException("mock backend declines " + artifact.kind()
                    + " artifact for kernel " + artifact.kernelName());
        }
        @Override
        public void close() {}
    }

    private static class MockBuffer implements GpuBuffer {
        private final long id;
        private final long size;
        private volatile boolean closed;

        MockBuffer(long id, long size) {
            this.id = id;
            this.size = size;
        }

        @Override
        public long sizeInBytes() { return size; }
        @Override
        public long bufferId() { return id; }
        @Override
        public void close() { closed = true; }
    }
}

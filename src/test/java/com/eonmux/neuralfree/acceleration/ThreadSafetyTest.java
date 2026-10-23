package com.eonmux.neuralfree.acceleration;

import com.eonmux.neuralfree.acceleration.cache.CacheKey;
import com.eonmux.neuralfree.acceleration.cache.KernelCache;
import com.eonmux.neuralfree.acceleration.backend.KernelHandle;
import com.eonmux.neuralfree.acceleration.backend.KernelArtifact;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.memory.GpuMemoryPool;
import com.eonmux.neuralfree.acceleration.memory.MemoryBlock;
import com.eonmux.neuralfree.acceleration.memory.PoolConfig;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.Assert.*;

public class ThreadSafetyTest {

    @Test
    public void testKernelCacheConcurrentGetOrCompile() throws InterruptedException {
        KernelCache cache = new KernelCache();
        int numThreads = 20;
        int iterationsPerThread = 100;
        AtomicInteger compileCount = new AtomicInteger(0);
        
        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        CountDownLatch startLatch = new CountDownLatch(1);
        CountDownLatch doneLatch = new CountDownLatch(numThreads);
        
        List<Throwable> errors = Collections.synchronizedList(new ArrayList<>());
        
        for (int i = 0; i < numThreads; i++) {
            final int threadId = i;
            executor.submit(() -> {
                try {
                    startLatch.await();
                    for (int j = 0; j < iterationsPerThread; j++) {
                        String kernelName = "kernel_" + (j % 5);
                        CacheKey key = new CacheKey(kernelName, "hash_" + kernelName, "");
                        
                        KernelHandle handle = cache.getOrCompile(key, k -> {
                            compileCount.incrementAndGet();
                            return new MockKernelHandle(k.kernelName());
                        });
                        
                        assertNotNull(handle);
                        assertEquals(kernelName, handle.kernelName());
                    }
                } catch (Throwable t) {
                    errors.add(t);
                } finally {
                    doneLatch.countDown();
                }
            });
        }
        
        startLatch.countDown();
        doneLatch.await(30, TimeUnit.SECONDS);
        executor.shutdown();
        
        assertTrue("Concurrent access errors: " + errors, errors.isEmpty());
        assertEquals("Should compile exactly 5 unique kernels (one per unique key)", 5, compileCount.get());
        assertEquals("Cache should contain exactly 5 entries", 5, cache.size());
        cache.clear();
    }

    @Test
    public void testKernelCacheConcurrentClear() throws InterruptedException {
        KernelCache cache = new KernelCache();
        
        for (int i = 0; i < 10; i++) {
            CacheKey key = new CacheKey("kernel_" + i, "hash_" + i, "");
            cache.getOrCompile(key, k -> new MockKernelHandle(k.kernelName()));
        }
        
        assertEquals(10, cache.size());
        
        int numThreads = 10;
        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        CountDownLatch startLatch = new CountDownLatch(1);
        CountDownLatch doneLatch = new CountDownLatch(numThreads * 2);
        
        List<Throwable> errors = Collections.synchronizedList(new ArrayList<>());
        
        for (int i = 0; i < numThreads; i++) {
            executor.submit(() -> {
                try {
                    startLatch.await();
                    cache.clear();
                } catch (Throwable t) {
                    errors.add(t);
                } finally {
                    doneLatch.countDown();
                }
            });
            executor.submit(() -> {
                try {
                    startLatch.await();
                    cache.getOrCompile(new CacheKey("new_kernel", "new_hash", ""), 
                            k -> new MockKernelHandle(k.kernelName()));
                } catch (Throwable t) {
                    errors.add(t);
                } finally {
                    doneLatch.countDown();
                }
            });
        }
        
        startLatch.countDown();
        doneLatch.await(30, TimeUnit.SECONDS);
        executor.shutdown();
        
        assertTrue("Concurrent clear errors: " + errors, errors.isEmpty());
    }

    @Test
    public void testGpuMemoryPoolConcurrentAcquireRelease() throws InterruptedException {
        MockComputeBackend backend = new MockComputeBackend();
        PoolConfig config = PoolConfig.builder()
                .maxPoolSizeBytes(10 * 1024 * 1024)
                .minBlockSizeBytes(1024)
                .defaultBlockSizeBytes(4096)
                .build();
        GpuMemoryPool pool = new GpuMemoryPool(backend, config);
        
        int numThreads = 10;
        int iterationsPerThread = 100;
        
        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        CountDownLatch startLatch = new CountDownLatch(1);
        CountDownLatch doneLatch = new CountDownLatch(numThreads);
        
        List<Throwable> errors = Collections.synchronizedList(new ArrayList<>());
        AtomicInteger totalAcquires = new AtomicInteger(0);
        AtomicInteger totalReleases = new AtomicInteger(0);
        
        for (int i = 0; i < numThreads; i++) {
            executor.submit(() -> {
                try {
                    startLatch.await();
                    List<MemoryBlock> acquired = new ArrayList<>();
                    for (int j = 0; j < iterationsPerThread; j++) {
                        MemoryBlock block = pool.acquire(2048);
                        assertNotNull(block);
                        totalAcquires.incrementAndGet();
                        acquired.add(block);
                        
                        if (j % 10 == 0 && !acquired.isEmpty()) {
                            MemoryBlock toRelease = acquired.remove(0);
                            pool.release(toRelease);
                            totalReleases.incrementAndGet();
                        }
                    }
                    for (MemoryBlock block : acquired) {
                        pool.release(block);
                        totalReleases.incrementAndGet();
                    }
                } catch (Throwable t) {
                    errors.add(t);
                } finally {
                    doneLatch.countDown();
                }
            });
        }
        
        startLatch.countDown();
        doneLatch.await(30, TimeUnit.SECONDS);
        executor.shutdown();
        
        assertTrue("Concurrent acquire/release errors: " + errors, errors.isEmpty());
        assertEquals(numThreads * iterationsPerThread, totalAcquires.get());
        assertEquals(numThreads * iterationsPerThread, totalReleases.get());
        pool.close();
    }

    @Test
    public void testGpuMemoryPoolConcurrentAcquireNoDeadlock() throws InterruptedException {
        MockComputeBackend backend = new MockComputeBackend();
        PoolConfig config = PoolConfig.builder()
                .maxPoolSizeBytes(100 * 1024)
                .minBlockSizeBytes(1024)
                .defaultBlockSizeBytes(4096)
                .build();
        GpuMemoryPool pool = new GpuMemoryPool(backend, config);
        
        int numThreads = 20;
        
        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        CountDownLatch startLatch = new CountDownLatch(1);
        CountDownLatch doneLatch = new CountDownLatch(numThreads);
        
        List<Throwable> errors = Collections.synchronizedList(new ArrayList<>());
        
        for (int i = 0; i < numThreads; i++) {
            final int threadId = i;
            executor.submit(() -> {
                try {
                    startLatch.await();
                    for (int j = 0; j < 50; j++) {
                        MemoryBlock block = pool.acquire(1024);
                        assertNotNull(block);
                        Thread.sleep(1);
                        pool.release(block);
                    }
                } catch (Throwable t) {
                    errors.add(t);
                } finally {
                    doneLatch.countDown();
                }
            });
        }
        
        startLatch.countDown();
        boolean finished = doneLatch.await(10, TimeUnit.SECONDS);
        executor.shutdown();
        
        assertTrue("Deadlock detected: threads did not complete in time", finished);
        assertTrue("Concurrent access errors: " + errors, errors.isEmpty());
        pool.close();
    }

    @Test
    public void testCacheKeyThreadSafety() throws InterruptedException {
        int numThreads = 20;
        int iterationsPerThread = 1000;
        
        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        CountDownLatch startLatch = new CountDownLatch(1);
        CountDownLatch doneLatch = new CountDownLatch(numThreads);
        
        List<Throwable> errors = Collections.synchronizedList(new ArrayList<>());
        
        for (int i = 0; i < numThreads; i++) {
            executor.submit(() -> {
                try {
                    startLatch.await();
                    for (int j = 0; j < iterationsPerThread; j++) {
                        CacheKey key1 = new CacheKey("kernel", "hash", "options");
                        CacheKey key2 = new CacheKey("kernel", "hash", "options");
                        
                        assertEquals(key1, key2);
                        assertEquals(key1.hashCode(), key2.hashCode());
                        
                        CacheKey key3 = new CacheKey("kernel", "different_hash", "options");
                        assertNotEquals(key1, key3);
                    }
                } catch (Throwable t) {
                    errors.add(t);
                } finally {
                    doneLatch.countDown();
                }
            });
        }
        
        startLatch.countDown();
        doneLatch.await(30, TimeUnit.SECONDS);
        executor.shutdown();
        
        assertTrue("CacheKey thread safety errors: " + errors, errors.isEmpty());
    }

    private static class MockKernelHandle implements KernelHandle {
        private final String kernelName;
        private boolean closed = false;
        
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

    /** Declines kernel artifacts so tensor ops take their CPU path. */
    private static class MockComputeBackend implements ComputeBackend {
        private final AtomicInteger bufferCount = new AtomicInteger(0);
        
        @Override
        public ComputeBackend.BackendType backendType() {
            return ComputeBackend.BackendType.OPENCL;
        }

        @Override
        public DeviceCapabilities capabilities() {
            return new DeviceCapabilities.Builder()
                    .deviceName("MockDevice")
                    .totalVramBytes(1024 * 1024 * 1024)
                    .computeUnits(10)
                    .maxClockFrequencyMhz(1500)
                    .maxWorkGroupSize(256)
                    .supportsAsyncTransfer(true)
                    .driverVersion("1.0")
                    .build();
        }

        @Override
        public ComputeQueue computeQueue() {
            return null;
        }

        @Override
        public ComputeQueue transferQueue() {
            return null;
        }

        @Override
        public GpuBuffer allocateBuffer(long sizeInBytes) {
            return new MockGpuBuffer(bufferCount.incrementAndGet(), sizeInBytes);
        }

        @Override
        public GpuBuffer allocateBuffer(long sizeInBytes, boolean readOnly) {
            return allocateBuffer(sizeInBytes);
        }

        @Override
        public KernelHandle compileKernel(String kernelName, String source, String buildOptions) {
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
        }
    }

    private static class MockGpuBuffer implements GpuBuffer {
        private final long id;
        private final long size;
        private volatile boolean closed = false;
        
        MockGpuBuffer(int id, long size) {
            this.id = id;
            this.size = size;
        }
        
        @Override
        public long sizeInBytes() {
            return size;
        }

        @Override
        public long bufferId() {
            return id;
        }

        @Override
        public void close() {
            closed = true;
        }
    }
}

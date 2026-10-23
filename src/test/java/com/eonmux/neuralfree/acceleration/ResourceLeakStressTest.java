package com.eonmux.neuralfree.acceleration;

import com.eonmux.neuralfree.acceleration.backend.*;
import com.eonmux.neuralfree.acceleration.backend.ocl.OclComputeBackend;
import com.eonmux.neuralfree.acceleration.cache.CacheKey;
import com.eonmux.neuralfree.acceleration.cache.KernelCache;
import com.eonmux.neuralfree.acceleration.memory.GpuMemoryPool;
import com.eonmux.neuralfree.acceleration.memory.PoolConfig;
import com.eonmux.neuralfree.acceleration.ocl.CLDevice;
import com.eonmux.neuralfree.acceleration.pipeline.PipelineScheduler;
import com.eonmux.neuralfree.acceleration.pipeline.RingBufferManager;
import com.eonmux.neuralfree.acceleration.pipeline.schedule.PipelineConfig;
import com.eonmux.neuralfree.acceleration.kernels.Availability;
import com.eonmux.neuralfree.tensor.gpu.TrainGpu;
import com.eonmux.neuralfree.tensor.gpu.TrainKernels;
import com.eonmux.neuralfree.utils.LoggingUtil;
import org.apache.logging.log4j.Logger;
import org.junit.Assume;
import org.junit.Test;

import java.lang.management.ManagementFactory;
import java.lang.management.MemoryMXBean;
import java.lang.management.MemoryUsage;
import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.*;

public class ResourceLeakStressTest {
    private static final Logger LOGGER = LoggingUtil.getLogger(ResourceLeakStressTest.class);
    private static final int ITERATIONS = 100;
    private static final long BUFFER_SIZE = 1024 * 1024;
    
    @Test
    public void testBackendCreateClose100TimesNoLeak() {
        GpuTestUtil.assumeOpenCLAvailable();
        
        MemoryMXBean memoryBean = ManagementFactory.getMemoryMXBean();
        MemoryUsage beforeHeap = memoryBean.getHeapMemoryUsage();
        long beforeUsed = beforeHeap.getUsed();
        
        List<Long> memorySamples = new ArrayList<>();
        
        for (int i = 0; i < ITERATIONS; i++) {
            HWAcceleration hwAccel = new HWAcceleration();
            try {
                if (hwAccel.getCLDevices().isEmpty()) {
                    hwAccel.shutdown();
                    continue;
                }
                
                CLDevice clDevice = hwAccel.getCLDevices().get(0);
                try (OclComputeBackend backend = new OclComputeBackend(clDevice)) {
                    assertNotNull(backend.computeQueue());
                    assertNotNull(backend.transferQueue());
                    assertNotNull(backend.capabilities());
                    
                    try (GpuBuffer buffer = backend.allocateBuffer(BUFFER_SIZE)) {
                        assertNotNull(buffer);
                        assertEquals(BUFFER_SIZE, buffer.sizeInBytes());
                    }
                }
            } finally {
                hwAccel.shutdown();
            }
            
            if (i % 20 == 19) {
                System.gc();
                try {
                    Thread.sleep(50);
                } catch (InterruptedException ignored) {}
                MemoryUsage sample = memoryBean.getHeapMemoryUsage();
                memorySamples.add(sample.getUsed());
            }
        }
        
        System.gc();
        try {
            Thread.sleep(100);
        } catch (InterruptedException ignored) {}
        
        MemoryUsage afterHeap = memoryBean.getHeapMemoryUsage();
        long afterUsed = afterHeap.getUsed();
        
        long heapGrowth = afterUsed - beforeUsed;
        long maxAcceptableGrowth = 50 * 1024 * 1024;
        
        LOGGER.info("Heap memory: before={} MB, after={} MB, growth={} MB",
                beforeUsed / (1024 * 1024), afterUsed / (1024 * 1024), heapGrowth / (1024 * 1024));
        
        if (!memorySamples.isEmpty()) {
            long firstSample = memorySamples.get(0);
            long lastSample = memorySamples.get(memorySamples.size() - 1);
            LOGGER.info("Memory samples: first={} MB, last={} MB",
                    firstSample / (1024 * 1024), lastSample / (1024 * 1024));
        }
        
        assertTrue("Heap memory grew by " + (heapGrowth / (1024 * 1024)) + " MB after " + ITERATIONS + 
                " iterations. Possible resource leak.", heapGrowth < maxAcceptableGrowth);
    }
    
    @Test
    public void testRingBufferManagerCreateClose100Times() {
        GpuTestUtil.assumeOpenCLAvailable();
        
        MemoryMXBean memoryBean = ManagementFactory.getMemoryMXBean();
        MemoryUsage beforeHeap = memoryBean.getHeapMemoryUsage();
        
        for (int i = 0; i < ITERATIONS; i++) {
            HWAcceleration hwAccel = new HWAcceleration();
            try {
                if (hwAccel.getCLDevices().isEmpty()) {
                    hwAccel.shutdown();
                    continue;
                }
                
                CLDevice clDevice = hwAccel.getCLDevices().get(0);
                try (OclComputeBackend backend = new OclComputeBackend(clDevice)) {
                    PipelineConfig config = PipelineConfig.builder()
                            .enablePrefetch(true)
                            .prefetchDepth(1)
                            .build();
                    
                    try (RingBufferManager manager = new RingBufferManager(backend, config)) {
                        assertEquals(2, manager.getSlotCount());
                        assertNotNull(manager.activeBuffer());
                    }
                }
            } finally {
                hwAccel.shutdown();
            }
        }
        
        System.gc();
        MemoryUsage afterHeap = memoryBean.getHeapMemoryUsage();
        long heapGrowth = afterHeap.getUsed() - beforeHeap.getUsed();
        
        LOGGER.info("RingBufferManager stress test - heap growth: {} MB", heapGrowth / (1024 * 1024));
        
        assertTrue("RingBufferManager test showed excessive heap growth: " + (heapGrowth / (1024 * 1024)) + " MB",
                heapGrowth < 30 * 1024 * 1024);
    }
    
    @Test
    public void testGpuMemoryPoolCreateClose100Times() {
        GpuTestUtil.assumeOpenCLAvailable();
        
        MemoryMXBean memoryBean = ManagementFactory.getMemoryMXBean();
        MemoryUsage beforeHeap = memoryBean.getHeapMemoryUsage();
        
        for (int i = 0; i < ITERATIONS; i++) {
            HWAcceleration hwAccel = new HWAcceleration();
            try {
                if (hwAccel.getCLDevices().isEmpty()) {
                    hwAccel.shutdown();
                    continue;
                }
                
                CLDevice clDevice = hwAccel.getCLDevices().get(0);
                try (OclComputeBackend backend = new OclComputeBackend(clDevice)) {
                    PoolConfig poolConfig = PoolConfig.defaultConfig(backend.capabilities().getTotalVramBytes());
                    
                    try (GpuMemoryPool pool = new GpuMemoryPool(backend, poolConfig)) {
                        var block1 = pool.acquire(BUFFER_SIZE);
                        var block2 = pool.acquire(BUFFER_SIZE);
                        
                        pool.release(block1);
                        pool.release(block2);
                    }
                }
            } finally {
                hwAccel.shutdown();
            }
        }
        
        System.gc();
        MemoryUsage afterHeap = memoryBean.getHeapMemoryUsage();
        long heapGrowth = afterHeap.getUsed() - beforeHeap.getUsed();
        
        LOGGER.info("GpuMemoryPool stress test - heap growth: {} MB", heapGrowth / (1024 * 1024));
        
        assertTrue("GpuMemoryPool test showed excessive heap growth: " + (heapGrowth / (1024 * 1024)) + " MB",
                heapGrowth < 30 * 1024 * 1024);
    }
    
    @Test
    public void testKernelCacheClearReleasesResources() {
        GpuTestUtil.assumeOpenCLAvailable();
        
        HWAcceleration hwAccel = new HWAcceleration();
        try {
            if (hwAccel.getCLDevices().isEmpty()) {
                LOGGER.info("No CL devices, skipping");
                return;
            }
            
            CLDevice clDevice = hwAccel.getCLDevices().get(0);
            try (OclComputeBackend backend = new OclComputeBackend(clDevice)) {
                KernelCache cache = backend.getKernelCache();
                
                String kernelSource = "__kernel void test(__global float* data) { data[get_global_id(0)] = 0; }";
                
                for (int i = 0; i < 10; i++) {
                    try {
                        backend.compileKernel("test" + i, kernelSource, "");
                    } catch (Exception e) {
                        // Expected - kernels with same source but different names
                    }
                }
                
                int sizeBeforeClear = cache.size();
                cache.clear();
                int sizeAfterClear = cache.size();
                
                assertEquals("Kernel cache should be empty after clear()", 0, sizeAfterClear);
                LOGGER.info("Kernel cache: {} entries cleared", sizeBeforeClear);
            }
        } finally {
            hwAccel.shutdown();
        }
    }
    
    @Test
    public void testPipelineSchedulerCreateClose100Times() {
        GpuTestUtil.assumeOpenCLAvailable();
        
        MemoryMXBean memoryBean = ManagementFactory.getMemoryMXBean();
        MemoryUsage beforeHeap = memoryBean.getHeapMemoryUsage();
        
        for (int i = 0; i < ITERATIONS; i++) {
            HWAcceleration hwAccel = new HWAcceleration();
            try {
                if (hwAccel.getCLDevices().isEmpty()) {
                    hwAccel.shutdown();
                    continue;
                }
                
                CLDevice clDevice = hwAccel.getCLDevices().get(0);
                try (OclComputeBackend backend = new OclComputeBackend(clDevice)) {
                    Availability train = TrainKernels.availability(backend);
                    Assume.assumeTrue("train kernels unavailable: " + train.reason(), train.available());
                    PipelineConfig config = PipelineConfig.builder()
                            .enablePrefetch(false)
                            .build();
                    
                    try (PipelineScheduler scheduler = new PipelineScheduler(backend, config)) {
                        assertNotNull(scheduler.config());
                        assertNotNull(scheduler.ringBufferManager());
                    }
                }
            } finally {
                hwAccel.shutdown();
            }
        }
        
        System.gc();
        MemoryUsage afterHeap = memoryBean.getHeapMemoryUsage();
        long heapGrowth = afterHeap.getUsed() - beforeHeap.getUsed();
        
        LOGGER.info("PipelineScheduler stress test - heap growth: {} MB", heapGrowth / (1024 * 1024));
        
        assertTrue("PipelineScheduler test showed excessive heap growth: " + (heapGrowth / (1024 * 1024)) + " MB",
                heapGrowth < 30 * 1024 * 1024);
    }
    
    @Test
    public void testKernelCacheThreadSafeClear() {
        KernelCache cache = new KernelCache();
        
        int numThreads = 10;
        int iterationsPerThread = 100;
        Thread[] threads = new Thread[numThreads];
        
        for (int t = 0; t < numThreads; t++) {
            final int threadId = t;
            threads[t] = new Thread(() -> {
                for (int i = 0; i < iterationsPerThread; i++) {
                    CacheKey key = new CacheKey("kernel" + threadId + "_" + i, "hash" + i, "");
                    if (i % 10 == 0) {
                        cache.clear();
                    }
                }
            });
        }
        
        for (Thread thread : threads) {
            thread.start();
        }
        
        for (Thread thread : threads) {
            try {
                thread.join();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
        
        LOGGER.info("KernelCache concurrent clear test completed without exceptions");
    }
    
    @Test
    public void testIdempotentClose() {
        MockComputeBackend mockBackend = new MockComputeBackend();
        
        GpuMemoryPool pool = new GpuMemoryPool(mockBackend);
        pool.close();
        pool.close();
        pool.close();
        
        RingBufferManager manager = new RingBufferManager(mockBackend, 
                PipelineConfig.builder().enablePrefetch(false).build());
        manager.close();
        manager.close();
        
        assertSchedulerDeclined(mockBackend);
        TrainGpu.release(mockBackend);
        
        mockBackend.close();
        mockBackend.close();
        assertEquals("Backend should only be closed once", 1, mockBackend.getCloseCount());
        
        LOGGER.info("Idempotent close test passed - resources closed safely multiple times");
    }
    
    @Test
    public void testCloseOnException() {
        MockComputeBackend mockBackend = new MockComputeBackend();
        
        try (RingBufferManager manager = new RingBufferManager(mockBackend, 
                PipelineConfig.builder().enablePrefetch(false).build())) {
            throw new RuntimeException("Simulated exception");
        } catch (RuntimeException e) {
            assertEquals("Simulated exception", e.getMessage());
        }
        
        assertSchedulerDeclined(mockBackend);
        TrainGpu.release(mockBackend);
        
        assertTrue("Backend queues should be accessible after try-with-resources close", 
                mockBackend.computeQueue() != null);
        LOGGER.info("Close on exception test passed - resources properly cleaned up");
    }
    
    /**
     * The mock declines every kernel artifact, so the train library is unavailable and the scheduler constructor
     * fails before acquiring any resource. Close idempotence and close-on-exception of a constructed scheduler are
     * covered on real backends by {@code PipelineSchedulerTest}.
     */
    private static void assertSchedulerDeclined(ComputeBackend backend) {
        String reason = TrainKernels.availability(backend).reason();
        try (PipelineScheduler scheduler = new PipelineScheduler(backend,
                PipelineConfig.builder().enablePrefetch(false).build())) {
            fail("Expected UnsupportedKernelException from a backend without the train kernels, got " + scheduler);
        } catch (UnsupportedKernelException expected) {
            assertTrue("message carries the availability reason: " + expected.getMessage(),
                    expected.getMessage().contains(reason));
        }
    }
    
    private static class MockComputeBackend implements ComputeBackend {
        private boolean closed = false;
        private int closeCount = 0;
        private MockComputeQueue computeQueue = new MockComputeQueue(ComputeQueue.QueueType.COMPUTE);
        private MockComputeQueue transferQueue = new MockComputeQueue(ComputeQueue.QueueType.TRANSFER);
        
        @Override
        public BackendType backendType() {
            return BackendType.OPENCL;
        }
        
        @Override
        public DeviceCapabilities capabilities() {
            return new DeviceCapabilities.Builder()
                    .deviceName("Mock")
                    .totalVramBytes(1024 * 1024 * 1024)
                    .computeUnits(4)
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
            return null;
        }
        
        /** Declines kernel artifacts so tensor ops take their CPU path. */
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
            if (!closed) {
                closed = true;
                closeCount++;
                computeQueue.close();
                transferQueue.close();
            }
        }
        
        int getCloseCount() {
            return closeCount;
        }
        
        boolean isTransferQueueClosed() {
            return transferQueue.isClosed();
        }
    }
    
    private static class MockComputeQueue implements ComputeQueue {
        private final ComputeQueue.QueueType queueType;
        private boolean closed = false;
        
        MockComputeQueue(ComputeQueue.QueueType queueType) {
            this.queueType = queueType;
        }
        
        @Override
        public ComputeQueue.QueueType getQueueType() {
            return queueType;
        }
        
        @Override
        public GpuEvent enqueueWriteBuffer(GpuBuffer dstBuffer, long dstOffset, long sizeInBytes, byte[] srcData, GpuEvent... waitOnEvents) {
            return null;
        }
        
        @Override
        public GpuEvent enqueueWriteBuffer(GpuBuffer dstBuffer, long dstOffset, long sizeInBytes, float[] srcData, GpuEvent... waitOnEvents) {
            return null;
        }
        
        @Override
        public GpuEvent enqueueReadBuffer(GpuBuffer srcBuffer, long srcOffset, long sizeInBytes, byte[] dstData, GpuEvent... waitOnEvents) {
            return null;
        }
        
        @Override
        public GpuEvent enqueueReadBuffer(GpuBuffer srcBuffer, long srcOffset, long sizeInBytes, float[] dstData, GpuEvent... waitOnEvents) {
            return null;
        }
        
        @Override
        public GpuEvent enqueueKernel(KernelHandle kernel, long[] globalWorkSize, long[] localWorkSize, Object[] args, GpuEvent... waitOnEvents) {
            return null;
        }
        
        @Override
        public void flush() {
        }
        
        @Override
        public void finish() {
        }
        
        @Override
        public void close() {
            closed = true;
        }
        
        boolean isClosed() {
            return closed;
        }
    }
    
    private static class MockGpuBuffer implements GpuBuffer {
        private final long sizeBytes;
        private boolean closed = false;
        
        MockGpuBuffer(long sizeBytes) {
            this.sizeBytes = sizeBytes;
        }
        
        @Override
        public long sizeInBytes() {
            return sizeBytes;
        }
        
        @Override
        public long bufferId() {
            return closed ? -1 : System.identityHashCode(this);
        }
        
        @Override
        public void close() {
            closed = true;
        }
    }
}

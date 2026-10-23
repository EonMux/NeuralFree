package com.eonmux.neuralfree.acceleration.backend.vulkan;

import com.eonmux.neuralfree.acceleration.backend.*;
import com.eonmux.neuralfree.acceleration.cache.KernelCache;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

public class VkComputeBackendIntegrationTest {

    private VkComputeBackend backend;
    private boolean gpuAvailable = false;

    @Before
    public void setUp() {
        try {
            backend = new VkComputeBackend();
            gpuAvailable = true;

            System.out.println("Vulkan GPU available: " + backend.capabilities().getDeviceName() +
                    ", VRAM: " + (backend.capabilities().getTotalVramBytes() / (1024 * 1024)) + " MB" +
                    ", Compute Units: " + backend.capabilities().getComputeUnits() +
                    ", Supports Async Transfer: " + backend.capabilities().supportsAsyncTransfer());
        } catch (Throwable e) {
            System.out.println("Vulkan GPU initialization failed: " + e.getMessage());
            e.printStackTrace();
        }
        Assume.assumeTrue("no Vulkan GPU", gpuAvailable);
    }

    @After
    public void tearDown() {
        if (backend != null) {
            backend.close();
        }
    }

    @Test
    public void testFullLifecycle() {
        KernelHandle kernel = backend.compileKernel("increment", "shaders/test/increment.spv", null);
        assertNotNull("Kernel should not be null", kernel);
        assertEquals("Kernel name should match", "increment", kernel.kernelName());

        int size = 1024;
        long bufferSize = size * Float.BYTES;

        GpuBuffer buffer = backend.allocateBuffer(bufferSize);
        assertNotNull("Buffer should not be null", buffer);
        assertEquals("Buffer size should match", bufferSize, buffer.sizeInBytes());

        float[] data = new float[size];
        float[] expected = new float[size];
        for (int i = 0; i < size; i++) {
            data[i] = i * 1.5f;
            expected[i] = data[i] + 1.0f;
        }

        ComputeQueue transferQueue = backend.transferQueue();
        ComputeQueue computeQueue = backend.computeQueue();

        GpuEvent writeEvent = transferQueue.enqueueWriteBuffer(buffer, 0, bufferSize, data);
        assertNotNull("Write event should not be null", writeEvent);

        Object[] kernelArgs = new Object[]{buffer, size};
        int blockSize = 256;
        int gridSize = (size + blockSize - 1) / blockSize;
        GpuEvent kernelEvent = computeQueue.enqueueKernel(kernel, new long[]{(long) gridSize * blockSize}, new long[]{blockSize}, kernelArgs, writeEvent);
        assertNotNull("Kernel event should not be null", kernelEvent);

        float[] result = new float[size];
        GpuEvent readEvent = computeQueue.enqueueReadBuffer(buffer, 0, bufferSize, result, kernelEvent);
        assertNotNull("Read event should not be null", readEvent);

        try {
            readEvent.awaitCompletion();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            fail("Test interrupted");
            return;
        }

        for (int i = 0; i < Math.min(20, size); i++) {
            assertEquals("Result mismatch at index " + i, expected[i], result[i], 1e-4f);
        }

        buffer.close();
        kernel.close();
    }

    @Test
    public void testKernelCachePreventsRecompilation() {
        KernelCache cache = backend.getKernelCache();
        int initialSize = cache.size();

        KernelHandle kernel1 = backend.compileKernel("increment", "shaders/test/increment.spv", null);
        assertNotNull("First compilation should succeed", kernel1);
        assertEquals("Cache should have 1 new entry", initialSize + 1, cache.size());

        int cacheSizeAfterFirst = cache.size();

        KernelHandle kernel2 = backend.compileKernel("increment", "shaders/test/increment.spv", null);
        assertNotNull("Second compilation should succeed", kernel2);
        assertEquals("Cache size should not change (cache hit)", cacheSizeAfterFirst, cache.size());

        assertSame("Both handles should be the same cached instance", kernel1, kernel2);

        kernel1.close();
    }

    @Test
    public void testComputeQueueWorksIndependently() {
        ComputeQueue computeQueue = backend.computeQueue();
        assertEquals("Compute queue type should be COMPUTE", ComputeQueue.QueueType.COMPUTE, computeQueue.getQueueType());

        int size = 256;
        float[] data = new float[size];
        for (int i = 0; i < size; i++) {
            data[i] = i;
        }

        GpuBuffer buffer = backend.allocateBuffer(size * Float.BYTES);
        computeQueue.enqueueWriteBuffer(buffer, 0, size * Float.BYTES, data);
        computeQueue.finish();

        KernelHandle kernel = backend.compileKernel("increment", "shaders/test/increment.spv", null);
        Object[] args = new Object[]{buffer, size};
        int blockSize = 32;
        int gridSize = (size + blockSize - 1) / blockSize;
        computeQueue.enqueueKernel(kernel, new long[]{(long) gridSize * blockSize}, new long[]{blockSize}, args);
        computeQueue.finish();

        float[] result = new float[size];
        computeQueue.enqueueReadBuffer(buffer, 0, size * Float.BYTES, result);
        computeQueue.finish();

        for (int i = 0; i < size; i++) {
            assertEquals("Compute queue result mismatch at index " + i, data[i] + 1.0f, result[i], 1e-4f);
        }

        buffer.close();
        kernel.close();
    }

    @Test
    public void testTransferQueueWorksIndependently() {
        ComputeQueue transferQueue = backend.transferQueue();
        assertEquals("Transfer queue type should be TRANSFER", ComputeQueue.QueueType.TRANSFER, transferQueue.getQueueType());

        int size = 512;
        float[] writeData = new float[size];
        for (int i = 0; i < size; i++) {
            writeData[i] = i * 2.5f;
        }

        GpuBuffer buffer = backend.allocateBuffer(size * Float.BYTES);

        transferQueue.enqueueWriteBuffer(buffer, 0, size * Float.BYTES, writeData);
        transferQueue.finish();

        float[] readData = new float[size];
        transferQueue.enqueueReadBuffer(buffer, 0, size * Float.BYTES, readData);
        transferQueue.finish();

        for (int i = 0; i < size; i++) {
            assertEquals("Transfer queue data mismatch at index " + i, writeData[i], readData[i], 1e-5f);
        }

        buffer.close();
    }

    @Test
    public void testComputeAndTransferQueuesBothFunctional() {
        ComputeQueue computeQueue = backend.computeQueue();
        ComputeQueue transferQueue = backend.transferQueue();

        assertNotNull("Compute queue should not be null", computeQueue);
        assertNotNull("Transfer queue should not be null", transferQueue);

        assertNotSame("Compute and transfer queues should be different instances", computeQueue, transferQueue);

        int size = 128;
        float[] data = new float[size];
        for (int i = 0; i < size; i++) {
            data[i] = i;
        }

        GpuBuffer buffer = backend.allocateBuffer(size * Float.BYTES);

        transferQueue.enqueueWriteBuffer(buffer, 0, size * Float.BYTES, data);
        transferQueue.finish();

        KernelHandle kernel = backend.compileKernel("increment", "shaders/test/increment.spv", null);
        Object[] args = new Object[]{buffer, size};
        int blockSize = 32;
        int gridSize = (size + blockSize - 1) / blockSize;
        computeQueue.enqueueKernel(kernel, new long[]{(long) gridSize * blockSize}, new long[]{blockSize}, args);
        computeQueue.finish();

        float[] result = new float[size];
        transferQueue.enqueueReadBuffer(buffer, 0, size * Float.BYTES, result);
        transferQueue.finish();

        for (int i = 0; i < size; i++) {
            assertEquals("Combined queue result mismatch at index " + i, data[i] + 1.0f, result[i], 1e-4f);
        }

        buffer.close();
        kernel.close();
    }

    @Test
    public void testBackendCapabilities() {
        DeviceCapabilities caps = backend.capabilities();
        assertNotNull("Capabilities should not be null", caps);

        assertNotNull("Device name should not be null", caps.getDeviceName());
        assertFalse("Device name should not be empty", caps.getDeviceName().isEmpty());

        assertTrue("VRAM should be positive", caps.getTotalVramBytes() > 0);
        assertTrue("Compute units should not be negative (0 when the driver does not report them)",
                caps.getComputeUnits() >= 0);
        assertTrue("Max work group size should be positive", caps.getMaxWorkGroupSize() > 0);

        System.out.println("Vulkan Device: " + caps.getDeviceName());
        System.out.println("VRAM: " + (caps.getTotalVramBytes() / (1024 * 1024)) + " MB");
        System.out.println("Compute Units: " + caps.getComputeUnits());
        System.out.println("Max Work Group Size: " + caps.getMaxWorkGroupSize());
        System.out.println("Supports Async Transfer: " + caps.supportsAsyncTransfer());
    }

    @Test
    public void testBackendType() {
        assertEquals("Backend type should be VULKAN", ComputeBackend.BackendType.VULKAN, backend.backendType());
    }

    @Test
    public void testEventSynchronization() {
        int size = 1024;
        float[] data = new float[size];
        for (int i = 0; i < size; i++) {
            data[i] = i;
        }

        KernelHandle kernel = backend.compileKernel("increment", "shaders/test/increment.spv", null);
        GpuBuffer buffer = backend.allocateBuffer(size * Float.BYTES);

        GpuEvent writeEvent = backend.transferQueue().enqueueWriteBuffer(buffer, 0, size * Float.BYTES, data);

        assertNotNull("Write event should not be null", writeEvent);

        Object[] args = new Object[]{buffer, size};
        int blockSize = 64;
        int gridSize = (size + blockSize - 1) / blockSize;
        GpuEvent kernelEvent = backend.computeQueue().enqueueKernel(
                kernel, new long[]{(long) gridSize * blockSize}, new long[]{blockSize}, args, writeEvent);

        float[] result = new float[size];
        GpuEvent readEvent = backend.computeQueue().enqueueReadBuffer(buffer, 0, size * Float.BYTES, result, kernelEvent);

        try {
            readEvent.awaitCompletion();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            fail("Test interrupted");
            return;
        }

        for (int i = 0; i < Math.min(10, size); i++) {
            assertEquals("Event-synced result mismatch at index " + i, data[i] + 1.0f, result[i], 1e-4f);
        }

        buffer.close();
        kernel.close();
    }

    @Test
    public void testMultipleKernelDispatches() {
        int size = 256;
        float[] data = new float[size];
        for (int i = 0; i < size; i++) {
            data[i] = 1.0f;
        }

        KernelHandle kernel = backend.compileKernel("increment", "shaders/test/increment.spv", null);
        GpuBuffer buffer = backend.allocateBuffer(size * Float.BYTES);

        backend.transferQueue().enqueueWriteBuffer(buffer, 0, size * Float.BYTES, data);
        backend.transferQueue().finish();

        int blockSize = 32;
        int gridSize = (size + blockSize - 1) / blockSize;
        
        Object[] args = new Object[]{buffer, size};
        backend.computeQueue().enqueueKernel(kernel, new long[]{(long) gridSize * blockSize}, new long[]{blockSize}, args);
        backend.computeQueue().finish();

        backend.computeQueue().enqueueKernel(kernel, new long[]{(long) gridSize * blockSize}, new long[]{blockSize}, args);
        backend.computeQueue().finish();

        float[] result = new float[size];
        backend.transferQueue().enqueueReadBuffer(buffer, 0, size * Float.BYTES, result);
        backend.transferQueue().finish();

        float expected = 1.0f + 1.0f + 1.0f;
        for (int i = 0; i < size; i++) {
            assertEquals("Multiple dispatch result mismatch at index " + i, expected, result[i], 1e-4f);
        }

        buffer.close();
        kernel.close();
    }

    @Test
    public void testCloseIsIdempotent() {
        backend.close();
        backend.close();
        backend.close();
    }

    @Test(expected = IllegalStateException.class)
    public void testOperationsAfterCloseThrow() {
        backend.close();
        backend.allocateBuffer(1024);
    }

    @Test(expected = IllegalStateException.class)
    public void testQueueOperationsAfterBackendCloseThrow() {
        GpuBuffer buffer = backend.allocateBuffer(1024);
        backend.close();

        float[] data = new float[256];
        backend.computeQueue().enqueueWriteBuffer(buffer, 0, 1024, data);
    }

    @Test
    public void testBufferIdIsUnique() {
        GpuBuffer buffer1 = backend.allocateBuffer(1024);
        GpuBuffer buffer2 = backend.allocateBuffer(1024);
        GpuBuffer buffer3 = backend.allocateBuffer(2048);

        assertNotEquals("Buffer IDs should be unique", buffer1.bufferId(), buffer2.bufferId());
        assertNotEquals("Buffer IDs should be unique", buffer1.bufferId(), buffer3.bufferId());
        assertNotEquals("Buffer IDs should be unique", buffer2.bufferId(), buffer3.bufferId());

        buffer1.close();
        buffer2.close();
        buffer3.close();
    }

    @Test
    public void testLargeBufferAllocation() {
        int size = 1024 * 1024;
        long bufferSize = size * Float.BYTES;

        GpuBuffer buffer = backend.allocateBuffer(bufferSize);
        assertNotNull("Large buffer should allocate", buffer);
        assertEquals("Buffer size should match", bufferSize, buffer.sizeInBytes());

        float[] data = new float[size];
        for (int i = 0; i < size; i++) {
            data[i] = i % 100;
        }

        backend.transferQueue().enqueueWriteBuffer(buffer, 0, bufferSize, data);
        backend.transferQueue().finish();

        float[] result = new float[size];
        backend.transferQueue().enqueueReadBuffer(buffer, 0, bufferSize, result);
        backend.transferQueue().finish();

        for (int i = 0; i < 100; i++) {
            int idx = (int) (Math.random() * size);
            assertEquals("Large buffer data mismatch at random index " + idx, data[idx], result[idx], 1e-5f);
        }

        buffer.close();
    }

    @Test
    public void testReadOnlyBufferAllocation() {
        int size = 256;
        GpuBuffer buffer = backend.allocateBuffer(size * Float.BYTES, true);

        assertNotNull("Read-only buffer should allocate", buffer);
        assertEquals("Buffer size should match", size * Float.BYTES, buffer.sizeInBytes());

        float[] data = new float[size];
        for (int i = 0; i < size; i++) {
            data[i] = i;
        }

        backend.transferQueue().enqueueWriteBuffer(buffer, 0, size * Float.BYTES, data);
        backend.transferQueue().finish();

        buffer.close();
    }

    @Test
    public void testDescriptorStrategyMatchesPushDescriptorFeature() {
        VkDescriptorStrategy strategy = backend.descriptorStrategy();
        assertNotNull("Descriptor strategy should not be null", strategy);

        boolean pushDescriptors = backend.capabilities().hasFeature(DeviceFeature.PUSH_DESCRIPTORS);
        assertEquals("Push descriptors are used exactly when the device enables them",
                pushDescriptors, strategy instanceof VkPushDescriptorStrategy);
        assertEquals("Pooled descriptors are the fallback without push descriptors",
                !pushDescriptors, strategy instanceof VkPooledDescriptorStrategy);
    }

    @Test
    public void testPipelineCacheIntegration() {
        VkPipelineCache vkPipelineCache = backend.getVkPipelineCache();
        assertNotNull("Pipeline cache should not be null", vkPipelineCache);

        KernelHandle kernel1 = backend.compileKernel("increment", "shaders/test/increment.spv", null);
        assertNotNull("First kernel should compile", kernel1);

        backend.prepareKernel(kernel1, new long[]{256});
        byte[] cacheData = vkPipelineCache.data();
        assertNotNull("Pipeline cache data should not be null after the pipeline is created", cacheData);
        assertTrue("Pipeline cache data should hold at least its header", cacheData.length > 0);

        KernelHandle kernel2 = backend.compileKernel("increment", "shaders/test/increment.spv", null);
        assertNotNull("Second kernel should compile (from cache)", kernel2);

        assertSame("Kernels should be the same (from kernel cache)", kernel1, kernel2);

        kernel1.close();
    }
}

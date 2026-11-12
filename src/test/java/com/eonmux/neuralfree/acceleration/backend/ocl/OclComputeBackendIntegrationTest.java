package com.eonmux.neuralfree.acceleration.backend.ocl;

import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.*;
import com.eonmux.neuralfree.acceleration.ocl.CLDevice;
import com.eonmux.neuralfree.acceleration.cache.KernelCache;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.*;

public class OclComputeBackendIntegrationTest {

    private static final String SIMPLE_ADD_KERNEL =
            "__kernel void simple_add(__global float *a, __global float *b, __global float *c, const int n) {\n" +
            "    int i = get_global_id(0);\n" +
            "    if (i < n) {\n" +
            "        c[i] = a[i] + b[i];\n" +
            "    }\n" +
            "}\n";

    private static final String MATRIX_MULTIPLY_KERNEL =
            "__kernel void matmul(__global float *A, __global float *B, __global float *C, const int M, const int K, const int N) {\n" +
            "    int row = get_global_id(0);\n" +
            "    int col = get_global_id(1);\n" +
            "    if (row < M && col < N) {\n" +
            "        float sum = 0.0f;\n" +
            "        for (int k = 0; k < K; k++) {\n" +
            "            sum += A[row * K + k] * B[k * N + col];\n" +
            "        }\n" +
            "        C[row * N + col] = sum;\n" +
            "    }\n" +
            "}\n";

    private static final String SCALE_KERNEL =
            "__kernel void scale(__global float *data, const float factor, const int n) {\n" +
            "    int i = get_global_id(0);\n" +
            "    if (i < n) {\n" +
            "        data[i] = data[i] * factor;\n" +
            "    }\n" +
            "}\n";

    private HWAcceleration hwAccel;
    private CLDevice clDevice;
    private OclComputeBackend backend;
    private boolean gpuAvailable = false;
    /** GpuEvents of the current test; each pins its queue and context until it is released in tearDown. */
    private final List<GpuEvent> events = new ArrayList<>();

    /** Registers an event for release in tearDown, so no cl_event outlives the test method. */
    private GpuEvent track(GpuEvent event) {
        if (event != null) {
            events.add(event);
        }
        return event;
    }

    @Before
    public void setUp() {
        try {
            hwAccel = new HWAcceleration();

            if (!hwAccel.getCLDevices().isEmpty()) {
                clDevice = hwAccel.getCLDevices().get(0);
                backend = new OclComputeBackend(clDevice);
                gpuAvailable = true;

                System.out.println("GPU available: " + backend.capabilities().getDeviceName() +
                        ", VRAM: " + (backend.capabilities().getTotalVramBytes() / (1024 * 1024)) + " MB" +
                        ", Compute Units: " + backend.capabilities().getComputeUnits());
            }
        } catch (Exception e) {
            System.out.println("GPU initialization failed: " + e.getMessage());
        }
        Assume.assumeTrue("no OpenCL GPU", gpuAvailable);
    }

    @After
    public void tearDown() {
        for (GpuEvent event : events) {
            try {
                event.close();
            } catch (RuntimeException ignored) {
            }
        }
        events.clear();
        if (backend != null) {
            backend.close();
        }
        if (hwAccel != null) {
            hwAccel.shutdown();
        }
    }

    @Test
    public void testFullLifecycle() throws InterruptedException {
        KernelHandle kernel = backend.compileKernel("simple_add", SIMPLE_ADD_KERNEL, null);
        assertNotNull("Kernel should not be null", kernel);
        assertEquals("Kernel name should match", "simple_add", kernel.kernelName());

        int size = 1024;
        long bufferSize = size * Float.BYTES;

        GpuBuffer bufferA = backend.allocateBuffer(bufferSize);
        GpuBuffer bufferB = backend.allocateBuffer(bufferSize);
        GpuBuffer bufferC = backend.allocateBuffer(bufferSize);

        assertNotNull("Buffer A should not be null", bufferA);
        assertNotNull("Buffer B should not be null", bufferB);
        assertNotNull("Buffer C should not be null", bufferC);
        assertEquals("Buffer A size should match", bufferSize, bufferA.sizeInBytes());
        assertEquals("Buffer B size should match", bufferSize, bufferB.sizeInBytes());
        assertEquals("Buffer C size should match", bufferSize, bufferC.sizeInBytes());

        float[] a = new float[size];
        float[] b = new float[size];
        float[] expected = new float[size];
        for (int i = 0; i < size; i++) {
            a[i] = i * 1.5f;
            b[i] = i * 2.5f;
            expected[i] = a[i] + b[i];
        }

        ComputeQueue transferQueue = backend.transferQueue();
        ComputeQueue computeQueue = backend.computeQueue();

        GpuEvent writeEventA = track(transferQueue.enqueueWriteBuffer(bufferA, 0, bufferSize, a));
        GpuEvent writeEventB = track(transferQueue.enqueueWriteBuffer(bufferB, 0, bufferSize, b));

        assertNotNull("Write event A should not be null", writeEventA);
        assertNotNull("Write event B should not be null", writeEventB);

        Object[] kernelArgs = new Object[]{bufferA, bufferB, bufferC, size};
        GpuEvent kernelEvent = track(computeQueue.enqueueKernel(kernel, new long[]{size}, new long[]{64}, kernelArgs, writeEventA, writeEventB));

        assertNotNull("Kernel event should not be null", kernelEvent);

        float[] result = new float[size];
        GpuEvent readEvent = track(computeQueue.enqueueReadBuffer(bufferC, 0, bufferSize, result, kernelEvent));

        assertNotNull("Read event should not be null", readEvent);

        readEvent.awaitCompletion();

        for (int i = 0; i < Math.min(20, size); i++) {
            assertEquals("Result mismatch at index " + i, expected[i], result[i], 1e-4f);
        }

        bufferA.close();
        bufferB.close();
        bufferC.close();

        kernel.close();
    }

    @Test
    public void testMatrixMultiplyCorrectOutput() throws InterruptedException {
        int M = 4, K = 4, N = 4;

        float[] A = new float[M * K];
        float[] B = new float[K * N];
        float[] expected = new float[M * N];

        for (int i = 0; i < M; i++) {
            for (int j = 0; j < K; j++) {
                A[i * K + j] = i + j;
            }
        }
        for (int i = 0; i < K; i++) {
            for (int j = 0; j < N; j++) {
                B[i * N + j] = i - j + 10;
            }
        }
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0;
                for (int k = 0; k < K; k++) {
                    sum += A[i * K + k] * B[k * N + j];
                }
                expected[i * N + j] = sum;
            }
        }

        KernelHandle kernel = backend.compileKernel("matmul", MATRIX_MULTIPLY_KERNEL, null);
        assertNotNull("Kernel should compile", kernel);

        GpuBuffer bufferA = backend.allocateBuffer(M * K * Float.BYTES);
        GpuBuffer bufferB = backend.allocateBuffer(K * N * Float.BYTES);
        GpuBuffer bufferC = backend.allocateBuffer(M * N * Float.BYTES);

        ComputeQueue tQueue = backend.transferQueue();
        ComputeQueue cQueue = backend.computeQueue();

        GpuEvent writeEventA = track(tQueue.enqueueWriteBuffer(bufferA, 0, A.length * Float.BYTES, A));
        GpuEvent writeEventB = track(tQueue.enqueueWriteBuffer(bufferB, 0, B.length * Float.BYTES, B));

        Object[] args = new Object[]{bufferA, bufferB, bufferC, M, K, N};
        GpuEvent kernelEvent = track(cQueue.enqueueKernel(kernel, new long[]{M, N}, new long[]{2, 2}, args, writeEventA, writeEventB));

        float[] result = new float[M * N];
        GpuEvent readEvent = track(cQueue.enqueueReadBuffer(bufferC, 0, result.length * Float.BYTES, result, kernelEvent));
        readEvent.awaitCompletion();

        for (int i = 0; i < M * N; i++) {
            assertEquals("Matrix multiply result mismatch at index " + i, expected[i], result[i], 1e-4f);
        }

        bufferA.close();
        bufferB.close();
        bufferC.close();
        kernel.close();
    }

    @Test
    public void testKernelCachePreventsRecompilation() {
        KernelCache cache = backend.getKernelCache();
        int initialSize = cache.size();

        KernelHandle kernel1 = backend.compileKernel("simple_add", SIMPLE_ADD_KERNEL, null);
        assertNotNull("First compilation should succeed", kernel1);
        assertEquals("Cache should have 1 new entry", initialSize + 1, cache.size());

        int cacheSizeAfterFirst = cache.size();

        KernelHandle kernel2 = backend.compileKernel("simple_add", SIMPLE_ADD_KERNEL, null);
        assertNotNull("Second compilation should succeed", kernel2);
        assertEquals("Cache size should not change (cache hit)", cacheSizeAfterFirst, cache.size());

        assertSame("Both handles should be the same cached instance", kernel1, kernel2);

        kernel1.close();
    }

    @Test
    public void testComputeQueueWorksIndependently() throws InterruptedException {
        ComputeQueue computeQueue = backend.computeQueue();
        assertEquals("Compute queue type should be COMPUTE", ComputeQueue.QueueType.COMPUTE, computeQueue.getQueueType());

        int size = 256;
        float[] data = new float[size];
        float[] expected = new float[size];
        for (int i = 0; i < size; i++) {
            data[i] = i;
            expected[i] = i * 3.0f;
        }

        KernelHandle kernel = backend.compileKernel("scale", SCALE_KERNEL, null);
        GpuBuffer buffer = backend.allocateBuffer(size * Float.BYTES);

        GpuEvent writeEvent = track(computeQueue.enqueueWriteBuffer(buffer, 0, size * Float.BYTES, data));

        Object[] args = new Object[]{buffer, 3.0f, size};
        GpuEvent kernelEvent = track(computeQueue.enqueueKernel(kernel, new long[]{size}, new long[]{32}, args, writeEvent));

        float[] result = new float[size];
        GpuEvent readEvent = track(computeQueue.enqueueReadBuffer(buffer, 0, size * Float.BYTES, result, kernelEvent));
        readEvent.awaitCompletion();

        for (int i = 0; i < size; i++) {
            assertEquals("Compute queue result mismatch at index " + i, expected[i], result[i], 1e-4f);
        }

        buffer.close();
        kernel.close();
    }

    @Test
    public void unchangedArgumentsStayOnTheKernelObjectAndChangedOnesReachIt() throws InterruptedException {
        ComputeQueue compute = backend.computeQueue();
        ComputeQueue transfer = backend.transferQueue();
        int size = 64;
        float[] ones = new float[size];
        java.util.Arrays.fill(ones, 1f);
        KernelHandle kernel = backend.compileKernel("scale", SCALE_KERNEL, null);
        GpuBuffer a = backend.allocateBuffer(size * Float.BYTES);
        GpuBuffer b = backend.allocateBuffer(size * Float.BYTES);
        track(compute.enqueueWriteBuffer(a, 0, size * Float.BYTES, ones));
        track(compute.enqueueWriteBuffer(b, 0, size * Float.BYTES, ones));
        long[] global = {size};
        long[] local = {32};
        // a: x2, x2 (same arguments again), x3 (one scalar changed); b: x5 through the other queue, then a
        // pre-bound dispatch keeps b and x5, and a set argument switches it back to a
        compute.enqueueKernelNoEvent(kernel, global, local, new Object[]{a, 2f, size});
        compute.enqueueKernelNoEvent(kernel, global, local, new Object[]{a, 2f, size});
        compute.enqueueKernelNoEvent(kernel, global, local, new Object[]{a, 3f, size});
        compute.finish();
        transfer.enqueueKernelNoEvent(kernel, global, local, new Object[]{b, 5f, size});
        transfer.enqueueKernelPreBound(kernel, global, local);
        transfer.finish();
        transfer.setKernelArg(kernel, 0, a);
        transfer.enqueueKernelPreBound(kernel, global, local);
        transfer.finish();
        float[] ra = new float[size];
        float[] rb = new float[size];
        track(compute.enqueueReadBuffer(a, 0, size * Float.BYTES, ra)).awaitCompletion();
        track(compute.enqueueReadBuffer(b, 0, size * Float.BYTES, rb)).awaitCompletion();
        for (int i = 0; i < size; i++) {
            assertEquals("a[" + i + "]", 2f * 2f * 3f * 5f, ra[i], 0f);
            assertEquals("b[" + i + "]", 5f * 5f, rb[i], 0f);
        }
        compute.enqueueKernelNoEvent(kernel, global, local, new Object[]{b, 1f, size});
        compute.finish();
        b.close();
        try {
            compute.enqueueKernelNoEvent(kernel, global, local, new Object[]{b, 1f, size});
            fail("a closed buffer argument must be refused even when it is the bound one");
        } catch (IllegalStateException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains("closed"));
        }
        a.close();
        kernel.close();
    }

    @Test
    public void anUnchangedArgumentIsNotSentToTheKernelObjectAgain() throws InterruptedException {
        ComputeQueue compute = backend.computeQueue();
        int size = 64;
        float[] ones = new float[size];
        java.util.Arrays.fill(ones, 1f);
        KernelHandle kernel = backend.compileKernel("scale", SCALE_KERNEL, null);
        GpuBuffer a = backend.allocateBuffer(size * Float.BYTES);
        GpuBuffer b = backend.allocateBuffer(size * Float.BYTES);
        track(compute.enqueueWriteBuffer(a, 0, size * Float.BYTES, ones));
        track(compute.enqueueWriteBuffer(b, 0, size * Float.BYTES, ones));
        long[] global = {size};
        long[] local = {32};
        compute.enqueueKernelNoEvent(kernel, global, local, new Object[]{a, 2f, size});
        compute.finish();
        // behind the queue's back, the kernel object now holds b and 7 in the arguments the queue set to a and 2
        org.jocl.cl_kernel clKernel = ((OclKernelHandle) kernel).getCLKernel();
        org.jocl.CL.clSetKernelArg(clKernel, 0, org.jocl.Sizeof.cl_mem,
                org.jocl.Pointer.to(((OclGpuBuffer) b).getCLMem()));
        org.jocl.CL.clSetKernelArg(clKernel, 1, org.jocl.Sizeof.cl_float, org.jocl.Pointer.to(new float[]{7f}));
        compute.enqueueKernelNoEvent(kernel, global, local, new Object[]{a, 2f, size});
        compute.finish();
        assertScaled(compute, a, b, size, 2f, 7f);
        compute.enqueueKernelNoEvent(kernel, global, local, new Object[]{a, 3f, size});
        compute.finish();
        assertScaled(compute, a, b, size, 2f, 21f);
        a.close();
        b.close();
        kernel.close();
    }

    /** Asserts every element of {@code a} and {@code b}, both filled with ones before the scalings. */
    private void assertScaled(ComputeQueue queue, GpuBuffer a, GpuBuffer b, int size, float expectA, float expectB)
            throws InterruptedException {
        float[] ra = new float[size];
        float[] rb = new float[size];
        track(queue.enqueueReadBuffer(a, 0, size * Float.BYTES, ra)).awaitCompletion();
        track(queue.enqueueReadBuffer(b, 0, size * Float.BYTES, rb)).awaitCompletion();
        for (int i = 0; i < size; i++) {
            assertEquals("a[" + i + "]", expectA, ra[i], 0f);
            assertEquals("b[" + i + "]", expectB, rb[i], 0f);
        }
    }

    @Test
    public void testTransferQueueWorksIndependently() throws InterruptedException {
        ComputeQueue transferQueue = backend.transferQueue();
        assertEquals("Transfer queue type should be TRANSFER", ComputeQueue.QueueType.TRANSFER, transferQueue.getQueueType());

        int size = 512;
        float[] writeData = new float[size];
        for (int i = 0; i < size; i++) {
            writeData[i] = i * 2.5f;
        }

        GpuBuffer buffer = backend.allocateBuffer(size * Float.BYTES);

        GpuEvent writeEvent = track(transferQueue.enqueueWriteBuffer(buffer, 0, size * Float.BYTES, writeData));

        float[] readData = new float[size];
        GpuEvent readEvent = track(transferQueue.enqueueReadBuffer(buffer, 0, size * Float.BYTES, readData, writeEvent));
        readEvent.awaitCompletion();

        for (int i = 0; i < size; i++) {
            assertEquals("Transfer queue data mismatch at index " + i, writeData[i], readData[i], 1e-5f);
        }

        buffer.close();
    }

    @Test
    public void testComputeAndTransferQueuesBothFunctional() throws InterruptedException {
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

        track(transferQueue.enqueueWriteBuffer(buffer, 0, size * Float.BYTES, data));
        transferQueue.finish();

        KernelHandle kernel = backend.compileKernel("scale", SCALE_KERNEL, null);
        Object[] args = new Object[]{buffer, 2.0f, size};
        track(computeQueue.enqueueKernel(kernel, new long[]{size}, new long[]{32}, args));
        computeQueue.finish();

        float[] result = new float[size];
        GpuEvent readEvent = track(transferQueue.enqueueReadBuffer(buffer, 0, size * Float.BYTES, result));
        readEvent.awaitCompletion();

        for (int i = 0; i < size; i++) {
            assertEquals("Combined queue result mismatch at index " + i, data[i] * 2.0f, result[i], 1e-4f);
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
        assertTrue("Compute units should be positive", caps.getComputeUnits() > 0);
        assertTrue("Max work group size should be positive", caps.getMaxWorkGroupSize() > 0);

        System.out.println("Device: " + caps.getDeviceName());
        System.out.println("VRAM: " + (caps.getTotalVramBytes() / (1024 * 1024)) + " MB");
        System.out.println("Compute Units: " + caps.getComputeUnits());
        System.out.println("Max Work Group Size: " + caps.getMaxWorkGroupSize());
        System.out.println("Supports Async Transfer: " + caps.supportsAsyncTransfer());
    }

    @Test
    public void testBackendType() {
        assertEquals("Backend type should be OPENCL", ComputeBackend.BackendType.OPENCL, backend.backendType());
    }

    @Test
    public void testEventSynchronization() throws InterruptedException {
        int size = 1024;
        float[] a = new float[size];
        float[] b = new float[size];
        for (int i = 0; i < size; i++) {
            a[i] = i;
            b[i] = i * 2;
        }

        KernelHandle kernel = backend.compileKernel("simple_add", SIMPLE_ADD_KERNEL, null);
        GpuBuffer bufferA = backend.allocateBuffer(size * Float.BYTES);
        GpuBuffer bufferB = backend.allocateBuffer(size * Float.BYTES);
        GpuBuffer bufferC = backend.allocateBuffer(size * Float.BYTES);

        GpuEvent writeEventA = track(backend.transferQueue().enqueueWriteBuffer(bufferA, 0, size * Float.BYTES, a));
        GpuEvent writeEventB = track(backend.transferQueue().enqueueWriteBuffer(bufferB, 0, size * Float.BYTES, b));

        assertNotNull("Write event A should not be null", writeEventA);
        assertNotNull("Write event B should not be null", writeEventB);

        Object[] args = new Object[]{bufferA, bufferB, bufferC, size};
        GpuEvent kernelEvent = track(backend.computeQueue().enqueueKernel(
                kernel, new long[]{size}, new long[]{64}, args, writeEventA, writeEventB));

        float[] result = new float[size];
        GpuEvent readEvent = track(backend.computeQueue().enqueueReadBuffer(bufferC, 0, size * Float.BYTES, result, kernelEvent));

        readEvent.awaitCompletion();

        for (int i = 0; i < Math.min(10, size); i++) {
            assertEquals("Event-synced result mismatch at index " + i, a[i] + b[i], result[i], 1e-4f);
        }

        bufferA.close();
        bufferB.close();
        bufferC.close();
        kernel.close();
    }

    @Test
    public void testMultipleKernelDispatches() throws InterruptedException {
        int size = 256;
        float[] data = new float[size];
        for (int i = 0; i < size; i++) {
            data[i] = 1.0f;
        }

        KernelHandle kernel = backend.compileKernel("scale", SCALE_KERNEL, null);
        GpuBuffer buffer = backend.allocateBuffer(size * Float.BYTES);

        GpuEvent writeEvent = track(backend.transferQueue().enqueueWriteBuffer(buffer, 0, size * Float.BYTES, data));

        Object[] args = new Object[]{buffer, 2.0f, size};
        GpuEvent kernelEvent1 = track(backend.computeQueue().enqueueKernel(kernel, new long[]{size}, new long[]{32}, args, writeEvent));

        args = new Object[]{buffer, 3.0f, size};
        GpuEvent kernelEvent2 = track(backend.computeQueue().enqueueKernel(kernel, new long[]{size}, new long[]{32}, args, kernelEvent1));

        float[] result = new float[size];
        GpuEvent readEvent = track(backend.transferQueue().enqueueReadBuffer(buffer, 0, size * Float.BYTES, result, kernelEvent2));
        readEvent.awaitCompletion();

        float expected = 1.0f * 2.0f * 3.0f;
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

    @Test
    public void testCloseReleasesBuffersLeftOpen() {
        OclGpuBuffer leftOpen = (OclGpuBuffer) backend.allocateBuffer(1024);
        OclGpuBuffer closedByHolder = (OclGpuBuffer) backend.allocateBuffer(4096);
        closedByHolder.close();
        assertEquals(1024, backend.getAllocatedBytes());

        backend.close();

        assertNull("a buffer left open is released with its backend", leftOpen.getCLMem());
        assertEquals(-1, leftOpen.bufferId());
        assertEquals(0, backend.getAllocatedBytes());
        leftOpen.close();
        assertEquals("closing a released buffer again changes nothing", 0, backend.getAllocatedBytes());
    }

    @Test
    public void testCloseReleasesCallerOwnedQueues() {
        OclComputeQueue leftOpen = (OclComputeQueue) backend.createComputeQueue();
        OclComputeQueue closedByCaller = (OclComputeQueue) backend.createComputeQueue();
        closedByCaller.close();
        assertTrue(closedByCaller.isReleased());
        assertFalse(leftOpen.isReleased());

        backend.close();

        assertTrue("a caller-owned queue left open is released with its backend", leftOpen.isReleased());
        leftOpen.close();
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
        track(backend.computeQueue().enqueueWriteBuffer(buffer, 0, 1024, data));
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
    public void testLargeBufferAllocation() throws InterruptedException {
        int size = 1024 * 1024;
        long bufferSize = size * Float.BYTES;

        GpuBuffer buffer = backend.allocateBuffer(bufferSize);
        assertNotNull("Large buffer should allocate", buffer);
        assertEquals("Buffer size should match", bufferSize, buffer.sizeInBytes());

        float[] data = new float[size];
        for (int i = 0; i < size; i++) {
            data[i] = i % 100;
        }

        GpuEvent writeEvent = track(backend.transferQueue().enqueueWriteBuffer(buffer, 0, bufferSize, data));

        float[] result = new float[size];
        GpuEvent readEvent = track(backend.transferQueue().enqueueReadBuffer(buffer, 0, bufferSize, result, writeEvent));
        readEvent.awaitCompletion();

        for (int i = 0; i < 100; i++) {
            int idx = (int) (Math.random() * size);
            assertEquals("Large buffer data mismatch at random index " + idx, data[idx], result[idx], 1e-5f);
        }

        buffer.close();
    }

    @Test
    public void testReadOnlyBufferAllocation() throws InterruptedException {
        int size = 256;
        GpuBuffer buffer = backend.allocateBuffer(size * Float.BYTES, true);

        assertNotNull("Read-only buffer should allocate", buffer);
        assertEquals("Buffer size should match", size * Float.BYTES, buffer.sizeInBytes());

        float[] data = new float[size];
        for (int i = 0; i < size; i++) {
            data[i] = i;
        }

        GpuEvent writeEvent = track(backend.transferQueue().enqueueWriteBuffer(buffer, 0, size * Float.BYTES, data));
        writeEvent.awaitCompletion();

        buffer.close();
    }
}

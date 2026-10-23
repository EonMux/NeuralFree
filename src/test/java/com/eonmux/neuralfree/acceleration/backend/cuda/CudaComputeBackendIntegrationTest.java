package com.eonmux.neuralfree.acceleration.backend.cuda;

import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.*;
import com.eonmux.neuralfree.acceleration.cuda.CUDevice;
import com.eonmux.neuralfree.acceleration.cache.KernelCache;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

public class CudaComputeBackendIntegrationTest {

    private static final String SIMPLE_ADD_PTX =
            ".version 7.5\n" +
            ".target sm_50\n" +
            ".address_size 64\n" +
            "\n" +
            ".visible .entry simple_add(\n" +
            "    .param .u64 a,\n" +
            "    .param .u64 b,\n" +
            "    .param .u64 c,\n" +
            "    .param .u32 n\n" +
            ") {\n" +
            "    .reg .f32   %f<4>;\n" +
            "    .reg .b32   %r<5>;\n" +
            "    .reg .b64   %rd<10>;\n" +
            "    .reg .pred  p0;\n" +
            "\n" +
            "    ld.param.u64    %rd1, [a];\n" +
            "    ld.param.u64    %rd2, [b];\n" +
            "    ld.param.u64    %rd3, [c];\n" +
            "    ld.param.u32    %r1, [n];\n" +
            "\n" +
            "    mov.u32         %r2, %tid.x;\n" +
            "    mov.u32         %r3, %ntid.x;\n" +
            "    mov.u32         %r4, %ctaid.x;\n" +
            "    mad.lo.u32      %r2, %r4, %r3, %r2;\n" +
            "\n" +
            "    setp.ge.u32     p0, %r2, %r1;\n" +
            "    @p0 bra         L_exit;\n" +
            "\n" +
            "    mul.wide.u32    %rd4, %r2, 4;\n" +
            "    add.s64         %rd5, %rd1, %rd4;\n" +
            "    add.s64         %rd6, %rd2, %rd4;\n" +
            "    add.s64         %rd7, %rd3, %rd4;\n" +
            "\n" +
            "    ld.global.f32   %f1, [%rd5];\n" +
            "    ld.global.f32   %f2, [%rd6];\n" +
            "    add.f32         %f3, %f1, %f2;\n" +
            "    st.global.f32   [%rd7], %f3;\n" +
            "\n" +
            "L_exit:\n" +
            "    ret;\n" +
            "}\n";

    private static final String SCALE_PTX =
            ".version 7.5\n" +
            ".target sm_50\n" +
            ".address_size 64\n" +
            "\n" +
            ".visible .entry scale(\n" +
            "    .param .u64 data,\n" +
            "    .param .f32 factor,\n" +
            "    .param .u32 n\n" +
            ") {\n" +
            "    .reg .f32   %f<4>;\n" +
            "    .reg .b32   %r<5>;\n" +
            "    .reg .b64   %rd<8>;\n" +
            "    .reg .pred  p0;\n" +
            "\n" +
            "    ld.param.u64    %rd1, [data];\n" +
            "    ld.param.f32    %f1, [factor];\n" +
            "    ld.param.u32    %r1, [n];\n" +
            "\n" +
            "    mov.u32         %r2, %tid.x;\n" +
            "    mov.u32         %r3, %ntid.x;\n" +
            "    mov.u32         %r4, %ctaid.x;\n" +
            "    mad.lo.u32      %r2, %r4, %r3, %r2;\n" +
            "\n" +
            "    setp.ge.u32     p0, %r2, %r1;\n" +
            "    @p0 bra         L_exit;\n" +
            "\n" +
            "    mul.wide.u32    %rd2, %r2, 4;\n" +
            "    add.s64         %rd3, %rd1, %rd2;\n" +
            "\n" +
            "    ld.global.f32   %f2, [%rd3];\n" +
            "    mul.f32         %f3, %f2, %f1;\n" +
            "    st.global.f32   [%rd3], %f3;\n" +
            "\n" +
            "L_exit:\n" +
            "    ret;\n" +
            "}\n";

    private static final String MATMUL_PTX =
            ".version 7.5\n" +
            ".target sm_50\n" +
            ".address_size 64\n" +
            "\n" +
            ".visible .entry matmul(\n" +
            "    .param .u64 A,\n" +
            "    .param .u64 B,\n" +
            "    .param .u64 C,\n" +
            "    .param .u32 M,\n" +
            "    .param .u32 K,\n" +
            "    .param .u32 N\n" +
            ") {\n" +
            "    .reg .f32   %f<8>;\n" +
            "    .reg .b32   %r<12>;\n" +
            "    .reg .b64   %rd<16>;\n" +
            "    .reg .pred  p0;\n" +
            "\n" +
            "    ld.param.u64    %rd1, [A];\n" +
            "    ld.param.u64    %rd2, [B];\n" +
            "    ld.param.u64    %rd3, [C];\n" +
            "    ld.param.u32    %r1, [M];\n" +
            "    ld.param.u32    %r2, [K];\n" +
            "    ld.param.u32    %r3, [N];\n" +
            "\n" +
            "    mov.u32         %r4, %tid.x;\n" +
            "    mov.u32         %r5, %ntid.x;\n" +
            "    mov.u32         %r6, %ctaid.x;\n" +
            "    mad.lo.u32      %r4, %r6, %r5, %r4;\n" +
            "\n" +
            "    mov.u32         %r7, %tid.y;\n" +
            "    mov.u32         %r8, %ntid.y;\n" +
            "    mov.u32         %r9, %ctaid.y;\n" +
            "    mad.lo.u32      %r7, %r9, %r8, %r7;\n" +
            "\n" +
            "    setp.ge.u32     p0, %r4, %r1;\n" +
            "    @p0 bra         L_exit;\n" +
            "    setp.ge.u32     p0, %r7, %r3;\n" +
            "    @p0 bra         L_exit;\n" +
            "\n" +
            "    mov.f32         %f1, 0f00000000;\n" +
            "    mov.u32         %r10, 0;\n" +
            "\n" +
            "L_loop:\n" +
            "    setp.ge.u32     p0, %r10, %r2;\n" +
            "    @p0 bra         L_store;\n" +
            "\n" +
            "    mul.lo.u32      %r11, %r4, %r2;\n" +
            "    add.u32         %r11, %r11, %r10;\n" +
            "    mul.wide.u32    %rd4, %r11, 4;\n" +
            "    add.s64         %rd5, %rd1, %rd4;\n" +
            "    ld.global.f32   %f2, [%rd5];\n" +
            "\n" +
            "    mul.lo.u32      %r11, %r10, %r3;\n" +
            "    add.u32         %r11, %r11, %r7;\n" +
            "    mul.wide.u32    %rd6, %r11, 4;\n" +
            "    add.s64         %rd7, %rd2, %rd6;\n" +
            "    ld.global.f32   %f3, [%rd7];\n" +
            "\n" +
            "    mul.f32         %f4, %f2, %f3;\n" +
            "    add.f32         %f1, %f1, %f4;\n" +
            "\n" +
            "    add.u32         %r10, %r10, 1;\n" +
            "    bra             L_loop;\n" +
            "\n" +
            "L_store:\n" +
            "    mul.lo.u32      %r11, %r4, %r3;\n" +
            "    add.u32         %r11, %r11, %r7;\n" +
            "    mul.wide.u32    %rd8, %r11, 4;\n" +
            "    add.s64         %rd9, %rd3, %rd8;\n" +
            "    st.global.f32   [%rd9], %f1;\n" +
            "\n" +
            "L_exit:\n" +
            "    ret;\n" +
            "}\n";

    private HWAcceleration hwAccel;
    private CUDevice cuDevice;
    private CudaComputeBackend backend;
    private boolean gpuAvailable = false;

    @Before
    public void setUp() {
        try {
            HWAcceleration.enableCuda = true;
            hwAccel = new HWAcceleration();

            if (!hwAccel.getCUDevices().isEmpty()) {
                cuDevice = hwAccel.getCUDevices().get(0);
                backend = new CudaComputeBackend(cuDevice);
                gpuAvailable = true;

                System.out.println("CUDA GPU available: " + backend.capabilities().getDeviceName() +
                        ", VRAM: " + (backend.capabilities().getTotalVramBytes() / (1024 * 1024)) + " MB" +
                        ", Compute Units: " + backend.capabilities().getComputeUnits());
            }
        } catch (Throwable e) {
            System.out.println("CUDA GPU initialization failed: " + e.getMessage());
            e.printStackTrace();
        }
        Assume.assumeTrue("no CUDA GPU", gpuAvailable);
    }

    @After
    public void tearDown() {
        if (backend != null) {
            backend.close();
        }
        if (hwAccel != null) {
            hwAccel.shutdown();
        }
    }

    @Test
    public void testFullLifecycle() {
        KernelHandle kernel = backend.compileKernel("simple_add", SIMPLE_ADD_PTX, null);
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

        GpuEvent writeEventA = transferQueue.enqueueWriteBuffer(bufferA, 0, bufferSize, a);
        GpuEvent writeEventB = transferQueue.enqueueWriteBuffer(bufferB, 0, bufferSize, b);

        assertNotNull("Write event A should not be null", writeEventA);
        assertNotNull("Write event B should not be null", writeEventB);

        Object[] kernelArgs = new Object[]{bufferA, bufferB, bufferC, size};
        int blockSize = 256;
        int gridSize = (size + blockSize - 1) / blockSize;
        GpuEvent kernelEvent = computeQueue.enqueueKernel(kernel, new long[]{(long) gridSize * blockSize}, new long[]{blockSize}, kernelArgs, writeEventA, writeEventB);

        assertNotNull("Kernel event should not be null", kernelEvent);

        float[] result = new float[size];
        GpuEvent readEvent = computeQueue.enqueueReadBuffer(bufferC, 0, bufferSize, result, kernelEvent);

        assertNotNull("Read event should not be null", readEvent);
        computeQueue.finish();

        for (int i = 0; i < Math.min(20, size); i++) {
            assertEquals("Result mismatch at index " + i, expected[i], result[i], 1e-4f);
        }

        bufferA.close();
        bufferB.close();
        bufferC.close();

        kernel.close();
    }

    @Test
    public void testMatrixMultiplyCorrectOutput() {
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

        KernelHandle kernel = backend.compileKernel("matmul", MATMUL_PTX, null);
        assertNotNull("Kernel should compile", kernel);

        GpuBuffer bufferA = backend.allocateBuffer(M * K * Float.BYTES);
        GpuBuffer bufferB = backend.allocateBuffer(K * N * Float.BYTES);
        GpuBuffer bufferC = backend.allocateBuffer(M * N * Float.BYTES);

        ComputeQueue tQueue = backend.transferQueue();
        ComputeQueue cQueue = backend.computeQueue();

        tQueue.enqueueWriteBuffer(bufferA, 0, A.length * Float.BYTES, A);
        tQueue.enqueueWriteBuffer(bufferB, 0, B.length * Float.BYTES, B);
        tQueue.finish();

        Object[] args = new Object[]{bufferA, bufferB, bufferC, M, K, N};
        cQueue.enqueueKernel(kernel, new long[]{M, N}, new long[]{1, 1}, args);
        cQueue.finish();

        float[] result = new float[M * N];
        cQueue.enqueueReadBuffer(bufferC, 0, result.length * Float.BYTES, result);
        cQueue.finish();

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

        KernelHandle kernel1 = backend.compileKernel("simple_add", SIMPLE_ADD_PTX, null);
        assertNotNull("First compilation should succeed", kernel1);
        assertEquals("Cache should have 1 new entry", initialSize + 1, cache.size());

        int cacheSizeAfterFirst = cache.size();

        KernelHandle kernel2 = backend.compileKernel("simple_add", SIMPLE_ADD_PTX, null);
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
        float[] expected = new float[size];
        for (int i = 0; i < size; i++) {
            data[i] = i;
            expected[i] = i * 3.0f;
        }

        KernelHandle kernel = backend.compileKernel("scale", SCALE_PTX, null);
        GpuBuffer buffer = backend.allocateBuffer(size * Float.BYTES);

        computeQueue.enqueueWriteBuffer(buffer, 0, size * Float.BYTES, data);
        computeQueue.finish();

        Object[] args = new Object[]{buffer, 3.0f, size};
        int blockSize = 32;
        int gridSize = (size + blockSize - 1) / blockSize;
        computeQueue.enqueueKernel(kernel, new long[]{(long) gridSize * blockSize}, new long[]{blockSize}, args);
        computeQueue.finish();

        float[] result = new float[size];
        computeQueue.enqueueReadBuffer(buffer, 0, size * Float.BYTES, result);
        computeQueue.finish();

        for (int i = 0; i < size; i++) {
            assertEquals("Compute queue result mismatch at index " + i, expected[i], result[i], 1e-4f);
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

        KernelHandle kernel = backend.compileKernel("scale", SCALE_PTX, null);
        Object[] args = new Object[]{buffer, 2.0f, size};
        int blockSize = 32;
        int gridSize = (size + blockSize - 1) / blockSize;
        computeQueue.enqueueKernel(kernel, new long[]{(long) gridSize * blockSize}, new long[]{blockSize}, args);
        computeQueue.finish();

        float[] result = new float[size];
        transferQueue.enqueueReadBuffer(buffer, 0, size * Float.BYTES, result);
        transferQueue.finish();

        for (int i = 0; i < size; i++) {
            assertEquals("Combined queue result mismatch at index " + i, data[i] * 2.0f, result[i], 1e-4f);
        }

        buffer.close();
        kernel.close();
    }

    @Test
    public void testDualStreamAsyncOperations() {
        int size = 2048;
        float[] a = new float[size];
        float[] b = new float[size];
        for (int i = 0; i < size; i++) {
            a[i] = i * 0.5f;
            b[i] = i * 1.5f;
        }

        KernelHandle kernel = backend.compileKernel("simple_add", SIMPLE_ADD_PTX, null);
        GpuBuffer bufferA = backend.allocateBuffer(size * Float.BYTES);
        GpuBuffer bufferB = backend.allocateBuffer(size * Float.BYTES);
        GpuBuffer bufferC = backend.allocateBuffer(size * Float.BYTES);

        ComputeQueue transferQueue = backend.transferQueue();
        ComputeQueue computeQueue = backend.computeQueue();

        GpuEvent writeEventA = transferQueue.enqueueWriteBuffer(bufferA, 0, size * Float.BYTES, a);
        GpuEvent writeEventB = transferQueue.enqueueWriteBuffer(bufferB, 0, size * Float.BYTES, b);

        Object[] args = new Object[]{bufferA, bufferB, bufferC, size};
        int blockSize = 256;
        int gridSize = (size + blockSize - 1) / blockSize;
        GpuEvent kernelEvent = computeQueue.enqueueKernel(kernel, new long[]{(long) gridSize * blockSize}, new long[]{blockSize}, args, writeEventA, writeEventB);

        float[] result = new float[size];
        GpuEvent readEvent = computeQueue.enqueueReadBuffer(bufferC, 0, size * Float.BYTES, result, kernelEvent);

        try {
            readEvent.awaitCompletion();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            fail("Test interrupted");
            return;
        }

        for (int i = 0; i < Math.min(50, size); i++) {
            float expected = a[i] + b[i];
            assertEquals("Dual stream result mismatch at index " + i, expected, result[i], 1e-4f);
        }

        bufferA.close();
        bufferB.close();
        bufferC.close();
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

        System.out.println("CUDA Device: " + caps.getDeviceName());
        System.out.println("VRAM: " + (caps.getTotalVramBytes() / (1024 * 1024)) + " MB");
        System.out.println("Compute Units: " + caps.getComputeUnits());
        System.out.println("Max Work Group Size: " + caps.getMaxWorkGroupSize());
        System.out.println("Supports Async Transfer: " + caps.supportsAsyncTransfer());
    }

    @Test
    public void testBackendType() {
        assertEquals("Backend type should be CUDA", ComputeBackend.BackendType.CUDA, backend.backendType());
    }

    @Test
    public void testEventSynchronization() {
        int size = 1024;
        float[] a = new float[size];
        float[] b = new float[size];
        for (int i = 0; i < size; i++) {
            a[i] = i;
            b[i] = i * 2;
        }

        KernelHandle kernel = backend.compileKernel("simple_add", SIMPLE_ADD_PTX, null);
        GpuBuffer bufferA = backend.allocateBuffer(size * Float.BYTES);
        GpuBuffer bufferB = backend.allocateBuffer(size * Float.BYTES);
        GpuBuffer bufferC = backend.allocateBuffer(size * Float.BYTES);

        GpuEvent writeEventA = backend.transferQueue().enqueueWriteBuffer(bufferA, 0, size * Float.BYTES, a);
        GpuEvent writeEventB = backend.transferQueue().enqueueWriteBuffer(bufferB, 0, size * Float.BYTES, b);

        assertNotNull("Write event A should not be null", writeEventA);
        assertNotNull("Write event B should not be null", writeEventB);

        Object[] args = new Object[]{bufferA, bufferB, bufferC, size};
        int blockSize = 64;
        int gridSize = (size + blockSize - 1) / blockSize;
        GpuEvent kernelEvent = backend.computeQueue().enqueueKernel(
                kernel, new long[]{(long) gridSize * blockSize}, new long[]{blockSize}, args, writeEventA, writeEventB);

        float[] result = new float[size];
        GpuEvent readEvent = backend.computeQueue().enqueueReadBuffer(bufferC, 0, size * Float.BYTES, result, kernelEvent);

        try {
            readEvent.awaitCompletion();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            fail("Test interrupted");
            return;
        }

        for (int i = 0; i < Math.min(10, size); i++) {
            assertEquals("Event-synced result mismatch at index " + i, a[i] + b[i], result[i], 1e-4f);
        }

        bufferA.close();
        bufferB.close();
        bufferC.close();
        kernel.close();
    }

    @Test
    public void testMultipleKernelDispatches() {
        int size = 256;
        float[] data = new float[size];
        for (int i = 0; i < size; i++) {
            data[i] = 1.0f;
        }

        KernelHandle kernel = backend.compileKernel("scale", SCALE_PTX, null);
        GpuBuffer buffer = backend.allocateBuffer(size * Float.BYTES);

        backend.transferQueue().enqueueWriteBuffer(buffer, 0, size * Float.BYTES, data);
        backend.transferQueue().finish();

        int blockSize = 32;
        int gridSize = (size + blockSize - 1) / blockSize;
        
        Object[] args = new Object[]{buffer, 2.0f, size};
        backend.computeQueue().enqueueKernel(kernel, new long[]{(long) gridSize * blockSize}, new long[]{blockSize}, args);
        backend.computeQueue().finish();

        args = new Object[]{buffer, 3.0f, size};
        backend.computeQueue().enqueueKernel(kernel, new long[]{(long) gridSize * blockSize}, new long[]{blockSize}, args);
        backend.computeQueue().finish();

        float[] result = new float[size];
        backend.transferQueue().enqueueReadBuffer(buffer, 0, size * Float.BYTES, result);
        backend.transferQueue().finish();

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
    public void testWorkItemGridRoundsUpAndKernelBoundsChecks() {
        int capacity = 1024;
        int n = 1000;
        float[] data = new float[capacity];
        java.util.Arrays.fill(data, 1.0f);

        KernelHandle kernel = backend.compileKernel("scale", SCALE_PTX, null);
        GpuBuffer buffer = backend.allocateBuffer(capacity * Float.BYTES);
        ComputeQueue queue = backend.computeQueue();
        queue.enqueueWriteBuffer(buffer, 0, capacity * Float.BYTES, data);

        // 1000 work-items in blocks of 256 launch 4 blocks; the kernel skips ids >= n.
        queue.enqueueKernelNoEvent(kernel, new long[]{n}, new long[]{256}, new Object[]{buffer, 2.0f, n});

        float[] result = new float[capacity];
        queue.enqueueReadBuffer(buffer, 0, capacity * Float.BYTES, result);
        queue.finish();

        for (int i = 0; i < n; i++) {
            assertEquals("scaled element " + i, 2.0f, result[i], 0f);
        }
        for (int i = n; i < capacity; i++) {
            assertEquals("element " + i + " beyond n must be untouched", 1.0f, result[i], 0f);
        }

        buffer.close();
        kernel.close();
    }

    @Test
    public void testDefaultLocalSizeCoversGlobal() {
        int n = 768;
        float[] data = new float[n];
        java.util.Arrays.fill(data, 3.0f);

        KernelHandle kernel = backend.compileKernel("scale", SCALE_PTX, null);
        GpuBuffer buffer = backend.allocateBuffer(n * Float.BYTES);
        ComputeQueue queue = backend.computeQueue();
        queue.enqueueWriteBuffer(buffer, 0, n * Float.BYTES, data);
        queue.enqueueKernelNoEvent(kernel, new long[]{n}, null, new Object[]{buffer, 2.0f, n});

        float[] result = new float[n];
        queue.enqueueReadBuffer(buffer, 0, n * Float.BYTES, result);
        queue.finish();

        for (int i = 0; i < n; i++) {
            assertEquals("default-local dispatch element " + i, 6.0f, result[i], 0f);
        }

        buffer.close();
        kernel.close();
    }

    @Test
    public void testWriteCopiesHostDataAtEnqueue() {
        int n = 4096;
        float[] data = new float[n];
        for (int i = 0; i < n; i++) {
            data[i] = i;
        }
        java.nio.ByteBuffer bytes = java.nio.ByteBuffer.allocateDirect(n).order(java.nio.ByteOrder.nativeOrder());
        for (int i = 0; i < n; i++) {
            bytes.put(i, (byte) i);
        }
        bytes.position(17).limit(99);

        GpuBuffer floats = backend.allocateBuffer(n * Float.BYTES);
        GpuBuffer raw = backend.allocateBuffer(n);
        ComputeQueue queue = backend.transferQueue();

        GpuEvent floatWrite = queue.enqueueWriteBuffer(floats, 0, n * Float.BYTES, data);
        java.util.Arrays.fill(data, -1.0f);
        GpuEvent byteWrite = queue.enqueueWriteBuffer(raw, 0, n, bytes);
        assertEquals("ByteBuffer position must be preserved", 17, bytes.position());
        assertEquals("ByteBuffer limit must be preserved", 99, bytes.limit());
        for (int i = 0; i < n; i++) {
            bytes.put(i, (byte) 0);
        }
        floatWrite.close();
        byteWrite.close();

        float[] floatResult = new float[n];
        byte[] byteResult = new byte[n];
        queue.enqueueReadBuffer(floats, 0, n * Float.BYTES, floatResult);
        queue.enqueueReadBuffer(raw, 0, n, byteResult);
        queue.finish();

        for (int i = 0; i < n; i++) {
            assertEquals("float written before the host array changed, index " + i, i, floatResult[i], 0f);
            assertEquals("byte written before the host buffer changed, index " + i, (byte) i, byteResult[i]);
        }

        floats.close();
        raw.close();
    }

    @Test
    public void testChunkedTransfersLargerThanStagingSlices() {
        // 40 MiB: more than one slice of the 64 MiB upload ring and several of the 16 MiB readback ring.
        int n = 10 * 1024 * 1024;
        float[] data = new float[n];
        for (int i = 0; i < n; i++) {
            data[i] = (i % 9973) * 0.25f;
        }

        GpuBuffer buffer = backend.allocateBuffer((long) n * Float.BYTES);
        ComputeQueue queue = backend.transferQueue();
        queue.enqueueWriteBuffer(buffer, 0, (long) n * Float.BYTES, data).close();

        float[] result = new float[n];
        GpuEvent read = queue.enqueueReadBuffer(buffer, 0, (long) n * Float.BYTES, result);
        try {
            read.awaitCompletion();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            fail("Test interrupted");
            return;
        }
        read.close();

        assertArrayEquals("chunked round trip", data, result, 0f);

        buffer.close();
    }

    @Test
    public void testFillBufferWritesPatternInRange() {
        int n = 256;
        GpuBuffer buffer = backend.allocateBuffer(n * Float.BYTES);
        ComputeQueue queue = backend.computeQueue();
        queue.enqueueFillBuffer(buffer, 0, n * Float.BYTES, 0);
        queue.enqueueFillBuffer(buffer, 16 * Float.BYTES, 64 * Float.BYTES, Float.floatToIntBits(1.5f));

        float[] result = new float[n];
        queue.enqueueReadBuffer(buffer, 0, n * Float.BYTES, result);
        queue.finish();

        for (int i = 0; i < n; i++) {
            float expected = i >= 16 && i < 80 ? 1.5f : 0.0f;
            assertEquals("fill element " + i, expected, result[i], 0f);
        }

        buffer.close();
    }

    @Test
    public void testCopyBufferBetweenOffsets() {
        int n = 512;
        float[] data = new float[n];
        for (int i = 0; i < n; i++) {
            data[i] = i + 0.5f;
        }
        GpuBuffer src = backend.allocateBuffer(n * Float.BYTES);
        GpuBuffer dst = backend.allocateBuffer(n * Float.BYTES);
        ComputeQueue queue = backend.computeQueue();
        queue.enqueueWriteBuffer(src, 0, n * Float.BYTES, data);
        queue.enqueueFillBuffer(dst, 0, n * Float.BYTES, 0);
        queue.enqueueCopyBuffer(src, 100 * Float.BYTES, dst, 10 * Float.BYTES, 200 * Float.BYTES);

        float[] result = new float[n];
        queue.enqueueReadBuffer(dst, 0, n * Float.BYTES, result);
        queue.finish();

        for (int i = 0; i < n; i++) {
            float expected = i >= 10 && i < 210 ? data[i + 90] : 0.0f;
            assertEquals("copied element " + i, expected, result[i], 0f);
        }

        src.close();
        dst.close();
    }

    @Test(expected = IllegalArgumentException.class)
    public void testOverlappingCopyIsRejected() {
        GpuBuffer buffer = backend.allocateBuffer(1024);
        try {
            backend.computeQueue().enqueueCopyBuffer(buffer, 0, buffer, 256, 512);
        } finally {
            buffer.close();
        }
    }

    @Test
    public void testMarkerOrdersWorkAcrossQueues() {
        int n = 1 << 20;
        float[] data = new float[n];
        java.util.Arrays.fill(data, 4.0f);

        KernelHandle kernel = backend.compileKernel("scale", SCALE_PTX, null);
        GpuBuffer buffer = backend.allocateBuffer((long) n * Float.BYTES);

        ComputeQueue transfer = backend.transferQueue();
        ComputeQueue compute = backend.computeQueue();
        transfer.enqueueWriteBuffer(buffer, 0, (long) n * Float.BYTES, data).close();
        GpuEvent uploaded = transfer.enqueueMarker();

        compute.enqueueKernelNoEvent(kernel, new long[]{n}, new long[]{256}, new Object[]{buffer, 0.5f, n}, uploaded);
        uploaded.close();

        float[] result = new float[n];
        compute.enqueueReadBuffer(buffer, 0, (long) n * Float.BYTES, result);
        compute.finish();

        for (int i = 0; i < n; i += 4099) {
            assertEquals("cross-queue ordered element " + i, 2.0f, result[i], 0f);
        }

        buffer.close();
        kernel.close();
    }

    @Test
    public void testPreBoundClonesKeepIndependentArguments() {
        int n = 512;
        float[] ones = new float[n];
        java.util.Arrays.fill(ones, 1.0f);

        KernelHandle template = backend.compileKernel("scale", SCALE_PTX, null);
        KernelHandle first = backend.cloneKernel(template);
        KernelHandle second = backend.cloneKernel(template);
        GpuBuffer a = backend.allocateBuffer(n * Float.BYTES);
        GpuBuffer b = backend.allocateBuffer(n * Float.BYTES);

        ComputeQueue transfer = backend.transferQueue();
        ComputeQueue compute = backend.computeQueue();
        transfer.enqueueWriteBuffer(a, 0, n * Float.BYTES, ones).close();
        transfer.enqueueWriteBuffer(b, 0, n * Float.BYTES, ones).close();
        transfer.finish();

        // Arguments are set through one queue and launched on another: they live on the handle.
        transfer.setKernelArg(first, 0, a);
        transfer.setKernelArg(first, 1, 5.0f);
        transfer.setKernelArg(first, 2, n);
        transfer.setKernelArg(second, 0, b);
        transfer.setKernelArg(second, 1, 7.0f);
        transfer.setKernelArg(second, 2, n);

        compute.enqueueKernelPreBound(first, new long[]{n}, new long[]{128});
        GpuEvent done = compute.enqueueKernelPreBoundWithEvent(second, new long[]{n}, new long[]{128});

        float[] resultA = new float[n];
        float[] resultB = new float[n];
        compute.enqueueReadBuffer(a, 0, n * Float.BYTES, resultA, done);
        GpuEvent readB = compute.enqueueReadBuffer(b, 0, n * Float.BYTES, resultB, done);
        try {
            readB.awaitCompletion();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            fail("Test interrupted");
            return;
        }
        done.close();
        readB.close();

        for (int i = 0; i < n; i++) {
            assertEquals("first clone element " + i, 5.0f, resultA[i], 0f);
            assertEquals("second clone element " + i, 7.0f, resultB[i], 0f);
        }

        first.close();
        second.close();
        a.close();
        b.close();
    }

    @Test(expected = IllegalArgumentException.class)
    public void testWaitOnForeignEventIsRejected() {
        GpuEvent foreign = new GpuEvent() {
            @Override
            public void awaitCompletion() {
            }

            @Override
            public boolean isComplete() {
                return true;
            }

            @Override
            public void close() {
            }
        };
        backend.computeQueue().enqueueMarker(foreign);
    }

    @Test
    public void testBackendOwnedQueueCloseOnlyFinishes() {
        ComputeQueue queue = backend.computeQueue();
        queue.close();

        float[] data = {1.0f, 2.0f, 3.0f, 4.0f};
        GpuBuffer buffer = backend.allocateBuffer(data.length * Float.BYTES);
        queue.enqueueWriteBuffer(buffer, 0, data.length * Float.BYTES, data).close();
        float[] result = new float[data.length];
        queue.enqueueReadBuffer(buffer, 0, data.length * Float.BYTES, result).close();
        queue.finish();

        assertArrayEquals("backend-owned queue stays usable after close()", data, result, 0f);
        buffer.close();
    }

    @Test
    public void testCreatedQueueIsCallerOwned() {
        ComputeQueue queue = backend.createComputeQueue();
        assertNotSame("created queue is a new stream", backend.computeQueue(), queue);

        float[] data = {5.0f, 6.0f, 7.0f, 8.0f};
        GpuBuffer buffer = backend.allocateBuffer(data.length * Float.BYTES);
        queue.enqueueWriteBuffer(buffer, 0, data.length * Float.BYTES, data).close();
        float[] result = new float[data.length];
        queue.enqueueReadBuffer(buffer, 0, data.length * Float.BYTES, result).close();
        queue.close();

        assertArrayEquals("close() of a caller-owned queue completes its reads", data, result, 0f);
        try {
            queue.finish();
            fail("a released queue must reject further use");
        } catch (IllegalStateException expected) {
            // released by close()
        }
        buffer.close();
    }

    @Test
    public void closeFreesOpenBuffersAndCallerQueuesWhileASiblingKeepsTheContext() {
        long bytes = 256L << 20;
        long freeBefore = backend.getFreeVramBytes();
        Assume.assumeTrue("needs " + (4 * bytes >> 20) + " MiB free VRAM, " + (freeBefore >> 20) + " MiB free",
                freeBefore >= 4 * bytes);

        CudaComputeBackend sibling = new CudaComputeBackend(cuDevice);
        GpuBuffer buffer;
        ComputeQueue queue;
        try {
            buffer = sibling.allocateBuffer(bytes);
            queue = sibling.createComputeQueue();
            float[] data = {1.0f, 2.0f, 3.0f, 4.0f};
            queue.enqueueWriteBuffer(buffer, 0, data.length * Float.BYTES, data).close();
            queue.finish();
            assertTrue("the open buffer holds device memory",
                    backend.getFreeVramBytes() <= freeBefore - bytes / 2);
        } finally {
            sibling.close();
        }

        long freeAfter = backend.getFreeVramBytes();
        assertTrue("closing a backend frees its open buffers while another backend keeps the primary context "
                + "(free before " + freeBefore + ", after " + freeAfter + ")", freeAfter >= freeBefore - bytes / 2);
        buffer.close();
        queue.close();
        try {
            queue.finish();
            fail("the backend's close released its caller-owned queue");
        } catch (IllegalStateException expected) {
            // released by the backend's close
        }
    }
}

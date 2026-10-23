package com.eonmux.neuralfree.acceleration;

import com.eonmux.neuralfree.Config.ACCELERATION_MODE;
import com.eonmux.neuralfree.acceleration.backend.*;
import com.eonmux.neuralfree.acceleration.backend.ocl.OclComputeBackend;
import com.eonmux.neuralfree.acceleration.memory.GpuMemoryPool;
import com.eonmux.neuralfree.acceleration.memory.MemoryBlock;
import com.eonmux.neuralfree.acceleration.memory.PoolConfig;
import com.eonmux.neuralfree.acceleration.pipeline.RingBufferManager;
import com.eonmux.neuralfree.acceleration.pipeline.schedule.PipelineConfig;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.*;

public class PerformanceBenchmarkTest {

    private static final String SIMPLE_ADD_KERNEL =
            "__kernel void simple_add(__global float *a, __global float *b, __global float *c, const int n) {\n" +
            "    int i = get_global_id(0);\n" +
            "    if (i < n) {\n" +
            "        c[i] = a[i] + b[i];\n" +
            "    }\n" +
            "}\n";

    private static final String DELAY_KERNEL =
            "__kernel void delay_kernel(__global float *data, const int iterations, const int n) {\n" +
            "    int i = get_global_id(0);\n" +
            "    if (i < n) {\n" +
            "        float val = data[i];\n" +
            "        for (int j = 0; j < iterations; j++) {\n" +
            "            val = sqrt(val * val + 1.0f);\n" +
            "        }\n" +
            "        data[i] = val;\n" +
            "    }\n" +
            "}\n";

    private static final String SCALE_KERNEL =
            "__kernel void scale(__global float *data, const float factor, const int n) {\n" +
            "    int i = get_global_id(0);\n" +
            "    if (i < n) {\n" +
            "        data[i] = data[i] * factor;\n" +
            "    }\n" +
            "}\n";

    private static final String COPY_KERNEL =
            "__kernel void copy_kernel(__global float *src, __global float *dst, const int n) {\n" +
            "    int i = get_global_id(0);\n" +
            "    if (i < n) {\n" +
            "        dst[i] = src[i];\n" +
            "    }\n" +
            "}\n";

    private static final int WARMUP_ITERATIONS = 3;
    private static final int BENCHMARK_ITERATIONS = 10;

    private HWAcceleration hwAccel;
    private OclComputeBackend backend;
    private boolean gpuAvailable = false;

    @Before
    public void setUp() {
        try {
            hwAccel = HWAcceleration.forMode(ACCELERATION_MODE.OPENCL);

            if (!hwAccel.getCLDevices().isEmpty()) {
                backend = (OclComputeBackend) hwAccel.createBackend(ACCELERATION_MODE.OPENCL);
                gpuAvailable = true;

                System.out.println("GPU available for benchmarks: " + backend.capabilities().getDeviceName() +
                        ", VRAM: " + (backend.capabilities().getTotalVramBytes() / (1024 * 1024)) + " MB");
            }
        } catch (Exception e) {
            System.out.println("GPU initialization failed: " + e.getMessage());
        }
        Assume.assumeTrue("no OpenCL GPU", gpuAvailable);
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
    public void testKernelCacheBenchmark() {
        System.out.println("\n========== KERNEL CACHE BENCHMARK ==========");

        long firstCallTotalNs = 0;
        for (int i = 0; i < WARMUP_ITERATIONS + BENCHMARK_ITERATIONS; i++) {
            String uniqueSource = SIMPLE_ADD_KERNEL + "\n// unique_id: " + i + "_" + System.nanoTime();
            String kernelName = "simple_add";

            long start = System.nanoTime();
            KernelHandle handle = backend.compileKernel(kernelName, uniqueSource, null);
            long elapsed = System.nanoTime() - start;

            if (i >= WARMUP_ITERATIONS) {
                firstCallTotalNs += elapsed;
            }
            handle.close();
            backend.getKernelCache().clear();
        }
        long avgFirstCallNs = firstCallTotalNs / BENCHMARK_ITERATIONS;
        double avgFirstCallMs = avgFirstCallNs / 1_000_000.0;
        System.out.printf("Average FIRST compile time: %.3f ms%n", avgFirstCallMs);

        backend.getKernelCache().clear();
        long singleCompileStart = System.nanoTime();
        KernelHandle firstHandle = backend.compileKernel("simple_add", SIMPLE_ADD_KERNEL, null);
        long singleCompileNs = System.nanoTime() - singleCompileStart;
        System.out.printf("Single FIRST compile time: %.3f ms%n", singleCompileNs / 1_000_000.0);

        long cachedCallTotalNs = 0;
        for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
            long start = System.nanoTime();
            KernelHandle cachedHandle = backend.compileKernel("simple_add", SIMPLE_ADD_KERNEL, null);
            long elapsed = System.nanoTime() - start;
            cachedCallTotalNs += elapsed;

            assertSame("Cached call should return same handle", firstHandle, cachedHandle);
        }
        long avgCachedCallNs = cachedCallTotalNs / BENCHMARK_ITERATIONS;
        double avgCachedCallMs = avgCachedCallNs / 1_000_000.0;
        System.out.printf("Average CACHED compile time: %.3f ms%n", avgCachedCallMs);

        double speedup = (double) avgFirstCallNs / Math.max(avgCachedCallNs, 1);
        System.out.printf("Cache speedup: %.1fx%n", speedup);

        firstHandle.close();
    }

    @Test
    public void testMemoryPoolBenchmark() {
        System.out.println("\n========== MEMORY POOL BENCHMARK ==========");

        long poolSizeBytes = backend.capabilities().getTotalVramBytes() / 4;
        PoolConfig config = new PoolConfig.Builder()
                .maxPoolSizeBytes(poolSizeBytes)
                .minBlockSizeBytes(4096)
                .defaultBlockSizeBytes(1024 * 1024)
                .build();

        GpuMemoryPool pool = new GpuMemoryPool(backend, config);
        long bufferSize = 64 * 1024;

        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            GpuBuffer buffer = backend.allocateBuffer(bufferSize);
            buffer.close();
        }

        long directAllocTotalNs = 0;
        List<GpuBuffer> directBuffers = new ArrayList<>();
        for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
            long start = System.nanoTime();
            GpuBuffer buffer = backend.allocateBuffer(bufferSize);
            long elapsed = System.nanoTime() - start;
            directAllocTotalNs += elapsed;
            directBuffers.add(buffer);
        }
        for (GpuBuffer b : directBuffers) {
            b.close();
        }
        long avgDirectNs = directAllocTotalNs / BENCHMARK_ITERATIONS;
        double avgDirectMs = avgDirectNs / 1_000_000.0;
        System.out.printf("Average DIRECT allocation time: %.3f ms%n", avgDirectMs);

        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            MemoryBlock block = pool.acquire(bufferSize);
            pool.release(block);
        }

        List<MemoryBlock> poolBlocks = new ArrayList<>();
        for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
            poolBlocks.add(pool.acquire(bufferSize));
        }

        long poolAcquireTotalNs = 0;
        for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
            pool.release(poolBlocks.get(i));

            long start = System.nanoTime();
            MemoryBlock block = pool.acquire(bufferSize);
            long elapsed = System.nanoTime() - start;
            poolAcquireTotalNs += elapsed;

            assertNotNull("pool returned no block", block);
            assertTrue("pooled block smaller than requested", block.getSizeBytes() >= bufferSize);
            poolBlocks.set(i, block);
        }

        long poolNewAllocTotalNs = 0;
        pool.close();
        pool = new GpuMemoryPool(backend, config);

        for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
            long start = System.nanoTime();
            MemoryBlock block = pool.acquire(bufferSize);
            long elapsed = System.nanoTime() - start;
            poolNewAllocTotalNs += elapsed;
            pool.release(block);
        }

        for (MemoryBlock block : poolBlocks) {
            pool.release(block);
        }
        pool.close();

        long avgPoolNs = poolAcquireTotalNs / BENCHMARK_ITERATIONS;
        double avgPoolMs = avgPoolNs / 1_000_000.0;
        System.out.printf("Average POOL acquire time (reuse): %.3f ms%n", avgPoolMs);

        long avgPoolNewNs = poolNewAllocTotalNs / BENCHMARK_ITERATIONS;
        double avgPoolNewMs = avgPoolNewNs / 1_000_000.0;
        System.out.printf("Average POOL acquire time (new alloc): %.3f ms%n", avgPoolNewMs);

    }

    @Test
    public void testComputeTransferOverlap() throws Exception {
        System.out.println("\n========== COMPUTE-TRANSFER OVERLAP BENCHMARK ==========");

        int dataSize = 256 * 1024;
        int iterations = 5000;

        float[] dataA = new float[dataSize];
        float[] dataB = new float[dataSize];
        for (int i = 0; i < dataSize; i++) {
            dataA[i] = 1.0f + (i % 100);
            dataB[i] = 1.0f + (i % 100);
        }

        KernelHandle kernel = backend.compileKernel("delay_kernel", DELAY_KERNEL, null);
        assertNotNull("Kernel should compile", kernel);

        ComputeQueue computeQueue = backend.computeQueue();
        ComputeQueue transferQueue = backend.transferQueue();

        GpuBuffer bufferA = backend.allocateBuffer((long) dataSize * Float.BYTES);
        GpuBuffer bufferB = backend.allocateBuffer((long) dataSize * Float.BYTES);

        transferQueue.enqueueWriteBuffer(bufferA, 0, (long) dataSize * Float.BYTES, dataA);
        transferQueue.enqueueWriteBuffer(bufferB, 0, (long) dataSize * Float.BYTES, dataB);
        transferQueue.finish();

        long serializedStart = System.nanoTime();

        GpuEvent writeEvent1 = transferQueue.enqueueWriteBuffer(bufferA, 0, (long) dataSize * Float.BYTES, dataA);
        writeEvent1.awaitCompletion();

        Object[] args = new Object[]{bufferA, iterations, dataSize};
        GpuEvent computeEvent1 = computeQueue.enqueueKernel(kernel, new long[]{dataSize}, new long[]{256}, args);
        computeEvent1.awaitCompletion();

        float[] result1 = new float[dataSize];
        GpuEvent readEvent1 = computeQueue.enqueueReadBuffer(bufferA, 0, (long) dataSize * Float.BYTES, result1);
        readEvent1.awaitCompletion();

        long serializedTime = System.nanoTime() - serializedStart;
        System.out.printf("SERIALIZED time: %.3f ms%n", serializedTime / 1_000_000.0);

        for (int warmup = 0; warmup < 2; warmup++) {
            GpuEvent w1 = transferQueue.enqueueWriteBuffer(bufferA, 0, (long) dataSize * Float.BYTES, dataA);
            Object[] wargs = new Object[]{bufferA, iterations, dataSize};
            GpuEvent c1 = computeQueue.enqueueKernel(kernel, new long[]{dataSize}, new long[]{256}, wargs, w1);
            float[] dummy = new float[dataSize];
            GpuEvent r1 = computeQueue.enqueueReadBuffer(bufferA, 0, (long) dataSize * Float.BYTES, dummy, c1);
            r1.awaitCompletion();
        }

        long overlappedStart = System.nanoTime();

        GpuEvent writeEvent2 = transferQueue.enqueueWriteBuffer(bufferA, 0, (long) dataSize * Float.BYTES, dataA);
        GpuEvent writeEventB = transferQueue.enqueueWriteBuffer(bufferB, 0, (long) dataSize * Float.BYTES, dataB);

        Object[] args2 = new Object[]{bufferA, iterations, dataSize};
        GpuEvent computeEvent2 = computeQueue.enqueueKernel(kernel, new long[]{dataSize}, new long[]{256}, args2, writeEvent2);

        float[] result2 = new float[dataSize];
        GpuEvent readEvent2 = computeQueue.enqueueReadBuffer(bufferA, 0, (long) dataSize * Float.BYTES, result2, computeEvent2);

        writeEventB.awaitCompletion();
        readEvent2.awaitCompletion();

        long overlappedTime = System.nanoTime() - overlappedStart;
        System.out.printf("OVERLAPPED time (transfer B concurrent with compute A): %.3f ms%n",
                overlappedTime / 1_000_000.0);

        double overlapRatio = (double) serializedTime / overlappedTime;
        System.out.printf("Overlap ratio: %.2fx (ideal: 2.0x if fully overlapped)%n", overlapRatio);

        System.out.printf("Verify result[0] changed from 1.0 to: %.4f%n", result2[0]);
        assertTrue("Compute should have modified data", result2[0] > 1.0f);

        bufferA.close();
        bufferB.close();
        kernel.close();
    }

    @Test
    public void testPipelineOverlapBenchmark() throws Exception {
        System.out.println("\n========== PIPELINE OVERLAP BENCHMARK ==========");

        int bufferSize = 64 * 1024;
        int numOperations = 10;
        float[] testData = new float[bufferSize];
        for (int i = 0; i < bufferSize; i++) {
            testData[i] = i + 1.0f;
        }

        ComputeQueue transferQueue = backend.transferQueue();
        long directTransferTotalNs = 0;

        GpuBuffer directBuffer = backend.allocateBuffer((long) bufferSize * Float.BYTES);
        for (int i = 0; i < numOperations; i++) {
            long start = System.nanoTime();
            GpuEvent event = transferQueue.enqueueWriteBuffer(directBuffer, 0, (long) bufferSize * Float.BYTES, testData);
            event.awaitCompletion();
            long elapsed = System.nanoTime() - start;
            directTransferTotalNs += elapsed;
        }
        directBuffer.close();

        double directTransferMs = directTransferTotalNs / 1_000_000.0 / numOperations;
        System.out.printf("Average DIRECT transfer (blocking): %.3f ms per operation%n", directTransferMs);

        PipelineConfig prefetchConfig = new PipelineConfig.Builder()
                .enablePrefetch(true)
                .prefetchDepth(1)
                .enableKernelCache(true)
                .enableMemoryPool(true)
                .build();

        RingBufferManager prefetchMgr = new RingBufferManager(backend, prefetchConfig);
        prefetchMgr.allocateSlotBuffer(0, (long) bufferSize * Float.BYTES);
        prefetchMgr.allocateSlotBuffer(1, (long) bufferSize * Float.BYTES);

        long prefetchTotalNs = 0;

        for (int i = 0; i < numOperations; i++) {
            long start = System.nanoTime();
            GpuEvent event = prefetchMgr.beginPrefetch(1, testData, 0);
            prefetchMgr.advance();
            if (event != null) {
                event.awaitCompletion();
            }
            long elapsed = System.nanoTime() - start;
            prefetchTotalNs += elapsed;
        }
        prefetchMgr.close();

        double prefetchMs = prefetchTotalNs / 1_000_000.0 / numOperations;
        System.out.printf("Average WITH prefetch (double buffer): %.3f ms per operation%n", prefetchMs);

        double speedup = directTransferMs / prefetchMs;
        System.out.printf("Prefetch vs direct speedup: %.2fx%n", speedup);
    }

    @Test
    public void testEventSynchronizationBenchmark() throws Exception {
        System.out.println("\n========== EVENT SYNCHRONIZATION BENCHMARK ==========");

        int dataSize = 128 * 1024;
        float[] data = new float[dataSize];
        for (int i = 0; i < dataSize; i++) {
            data[i] = i + 1.0f;
        }

        KernelHandle scaleKernel = backend.compileKernel("scale", SCALE_KERNEL, null);
        KernelHandle copyKernel = backend.compileKernel("copy_kernel", COPY_KERNEL, null);

        ComputeQueue computeQueue = backend.computeQueue();
        ComputeQueue transferQueue = backend.transferQueue();

        GpuBuffer bufferA = backend.allocateBuffer((long) dataSize * Float.BYTES);
        GpuBuffer bufferB = backend.allocateBuffer((long) dataSize * Float.BYTES);
        GpuBuffer bufferC = backend.allocateBuffer((long) dataSize * Float.BYTES);

        GpuEvent writeEvent = transferQueue.enqueueWriteBuffer(bufferA, 0, (long) dataSize * Float.BYTES, data);

        Object[] scaleArgs = new Object[]{bufferA, 2.0f, dataSize};
        GpuEvent scaleEvent = computeQueue.enqueueKernel(scaleKernel, new long[]{dataSize}, new long[]{256}, scaleArgs, writeEvent);

        Object[] copyArgs = new Object[]{bufferA, bufferB, dataSize};
        GpuEvent copyEvent = computeQueue.enqueueKernel(copyKernel, new long[]{dataSize}, new long[]{256}, copyArgs, scaleEvent);

        float[] result = new float[dataSize];
        GpuEvent readEvent = computeQueue.enqueueReadBuffer(bufferB, 0, (long) dataSize * Float.BYTES, result, copyEvent);

        long start = System.nanoTime();
        readEvent.awaitCompletion();
        long elapsed = System.nanoTime() - start;

        System.out.printf("Event chain completion time: %.3f ms%n", elapsed / 1_000_000.0);

        for (int i = 0; i < 10; i++) {
            assertEquals("Result should be scaled", data[i] * 2.0f, result[i], 1e-4f);
        }

        bufferA.close();
        bufferB.close();
        bufferC.close();
        scaleKernel.close();
        copyKernel.close();
    }
}

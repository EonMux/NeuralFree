package com.eonmux.neuralfree.acceleration.backend.metal;

import com.eonmux.neuralfree.acceleration.UnsupportedKernelException;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.acceleration.backend.KernelArtifact;
import com.eonmux.neuralfree.acceleration.backend.KernelHandle;
import com.eonmux.neuralfree.acceleration.kernels.KernelLibraryConfig;
import com.eonmux.neuralfree.acceleration.kernels.MetalKernelLibrary;
import com.eonmux.neuralfree.acceleration.metal.MetalDevice;
import com.eonmux.neuralfree.acceleration.metal.MetalDeviceRanking;
import com.eonmux.neuralfree.acceleration.metal.MetalRuntime;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;

import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * The Metal backend end to end on an Apple GPU: device enumeration, MSL compilation, buffer
 * transfers, dispatch and event completion through the four test kernels. Skips on a host without
 * the Metal shim.
 */
public class MetalComputeBackendIntegrationTest {

    private MetalComputeBackend backend;
    private MetalKernelLibrary library;
    private boolean metalAvailable;

    @Before
    public void setUp() {
        metalAvailable = MetalRuntime.available();
        Assume.assumeTrue("no Metal shim", metalAvailable);
        List<MetalDevice> devices = MetalRuntime.enumerate();
        Assume.assumeFalse("no Metal device", devices.isEmpty());
        backend = new MetalComputeBackend(MetalDeviceRanking.fastest(devices));
        library = new MetalKernelLibrary(new KernelLibraryConfig("metal-test", "shaders/test",
                "shaders/test/cuda", "shaders/test", false));
        System.out.println("Metal GPU available: " + backend.capabilities().getDeviceName()
                + ", working set: " + (backend.capabilities().getTotalVramBytes() / (1024 * 1024)) + " MB");
    }

    @After
    public void tearDown() {
        if (backend != null) {
            backend.close();
        }
    }

    @Test
    public void incrementRunsEndToEnd() {
        int size = 1024;
        float[] data = sequence(size);
        float[] result = new float[size];
        try (GpuBuffer buffer = backend.allocateBuffer(size * Float.BYTES)) {
            KernelHandle kernel = kernel("increment");
            write(buffer, data);
            try (GpuEvent done = compute().enqueueKernel(kernel, new long[]{size}, new long[]{256},
                    new Object[]{buffer, size})) {
                readAfter(done, buffer, result);
            }
        }
        for (int i = 0; i < size; i++) {
            assertEquals("element " + i, data[i] + 1.0f, result[i], 0.0f);
        }
    }

    @Test
    public void scaleRunsEndToEnd() {
        int size = 1000;
        float[] data = sequence(size);
        float[] result = new float[size];
        try (GpuBuffer buffer = backend.allocateBuffer(size * Float.BYTES)) {
            KernelHandle kernel = kernel("scale");
            write(buffer, data);
            try (GpuEvent done = compute().enqueueKernel(kernel, new long[]{size}, new long[]{256},
                    new Object[]{buffer, 2.5f, size})) {
                readAfter(done, buffer, result);
            }
        }
        for (int i = 0; i < size; i++) {
            assertEquals("element " + i, data[i] * 2.5f, result[i], 1e-5f);
        }
    }

    @Test
    public void simpleAddReadsTwoBuffersAndWritesAThird() {
        int size = 512;
        float[] a = sequence(size);
        float[] b = sequence(size);
        float[] result = new float[size];
        try (GpuBuffer bufferA = backend.allocateBuffer(size * Float.BYTES);
             GpuBuffer bufferB = backend.allocateBuffer(size * Float.BYTES);
             GpuBuffer bufferC = backend.allocateBuffer(size * Float.BYTES)) {
            KernelHandle kernel = kernel("simple_add");
            write(bufferA, a);
            write(bufferB, b);
            try (GpuEvent done = compute().enqueueKernel(kernel, new long[]{size}, new long[]{64},
                    new Object[]{bufferA, bufferB, bufferC, size})) {
                readAfter(done, bufferC, result);
            }
        }
        for (int i = 0; i < size; i++) {
            assertEquals("element " + i, a[i] + b[i], result[i], 1e-5f);
        }
    }

    @Test
    public void matmulDispatchesA2dGrid() {
        int m = 16;
        int k = 8;
        int n = 12;
        float[] a = new float[m * k];
        float[] b = new float[k * n];
        for (int i = 0; i < a.length; i++) {
            a[i] = (i % 7) - 3;
        }
        for (int i = 0; i < b.length; i++) {
            b[i] = (i % 5) - 2;
        }
        float[] result = new float[m * n];
        try (GpuBuffer bufferA = backend.allocateBuffer(a.length * Float.BYTES);
             GpuBuffer bufferB = backend.allocateBuffer(b.length * Float.BYTES);
             GpuBuffer bufferC = backend.allocateBuffer(result.length * Float.BYTES)) {
            KernelHandle kernel = kernel("matmul");
            write(bufferA, a);
            write(bufferB, b);
            try (GpuEvent done = compute().enqueueKernel(kernel,
                    new long[]{n, m}, new long[]{16, 16},
                    new Object[]{bufferA, bufferB, bufferC, m, k, n})) {
                readAfter(done, bufferC, result);
            }
        }
        for (int row = 0; row < m; row++) {
            for (int col = 0; col < n; col++) {
                float expected = 0.0f;
                for (int i = 0; i < k; i++) {
                    expected += a[row * k + i] * b[i * n + col];
                }
                assertEquals("element " + row + "," + col, expected, result[row * n + col], 1e-4f);
            }
        }
    }

    @Test
    public void aLaunchThatDisagreesWithTheSignatureIsRejectedByName() {
        KernelHandle kernel = kernel("increment");
        try (GpuBuffer buffer = backend.allocateBuffer(64)) {
            compute().enqueueKernel(kernel, new long[]{16}, new long[]{1},
                    new Object[]{buffer, 16, 32});
            fail("the extra scalar must be rejected");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("increment"));
            assertTrue(expected.getMessage().contains("scalar"));
        }
    }

    @Test
    public void theBackendRefusesForeignArtifactKinds() {
        try {
            backend.loadKernel(KernelArtifact.openClSource("foreign", "program", () -> "__kernel void foreign() {}",
                    null));
            fail("a Metal backend must refuse OpenCL source");
        } catch (UnsupportedKernelException expected) {
            assertTrue(expected.getMessage().contains("OPENCL_SOURCE"));
        }
    }

    private KernelHandle kernel(String name) {
        KernelArtifact artifact = library.resolve(name, backend);
        KernelHandle handle = backend.loadKernel(artifact);
        assertNotNull(handle);
        return handle;
    }

    private ComputeQueue compute() {
        return backend.computeQueue();
    }

    private void write(GpuBuffer buffer, float[] data) {
        backend.transferQueue().enqueueWriteBuffer(buffer, 0, (long) data.length * Float.BYTES, data).close();
        backend.transferQueue().finish();
    }

    private void readAfter(GpuEvent done, GpuBuffer buffer, float[] result) {
        try {
            done.awaitCompletion();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            fail("interrupted while waiting for the kernel");
        }
        compute().enqueueReadBuffer(buffer, 0, (long) result.length * Float.BYTES, result).close();
        compute().finish();
    }

    private static float[] sequence(int size) {
        float[] values = new float[size];
        for (int i = 0; i < size; i++) {
            values[i] = i * 0.5f - 3.0f;
        }
        return values;
    }
}

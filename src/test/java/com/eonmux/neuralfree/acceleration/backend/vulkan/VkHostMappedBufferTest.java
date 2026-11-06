package com.eonmux.neuralfree.acceleration.backend.vulkan;

import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.HostMappedBuffer;
import com.eonmux.neuralfree.acceleration.backend.KernelHandle;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Vulkan host-visible weight memory ({@link VkGpuBuffer#hostView(long, int)}): the device reads a persistently
 * mapped host-visible allocation through a copy and through a kernel argument, and the buffer reports its mapping.
 */
public class VkHostMappedBufferTest {

    private VkComputeBackend backend;
    private ComputeQueue queue;
    private boolean gpuAvailable;

    @Before
    public void setUp() {
        try {
            backend = new VkComputeBackend();
            queue = backend.computeQueue();
            gpuAvailable = true;
        } catch (Throwable e) {
            gpuAvailable = false;
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
    public void aHostMappedBufferIsPersistentlyMappedHostMemory() {
        assertTrue(backend.supportsHostMappedBuffers());
        HostMappedBuffer mapped = backend.allocateHostMappedBuffer(64);
        try {
            assertTrue(mapped instanceof VkGpuBuffer);
            VkGpuBuffer vk = (VkGpuBuffer) mapped;
            assertTrue(vk.isHostVisible());
            assertEquals(64, mapped.sizeInBytes());

            ByteBuffer view = mapped.hostView(8, 16);
            assertEquals(ByteOrder.LITTLE_ENDIAN, view.order());
            assertEquals(0, view.position());
            assertEquals(16, view.limit());
            assertThrows(IllegalArgumentException.class, () -> mapped.hostView(0, 65));
        } finally {
            mapped.close();
            mapped.close();
        }
    }

    @Test
    public void theDeviceCopiesFromHostVisibleMemory() {
        int bytes = 128 * Float.BYTES;
        HostMappedBuffer mapped = backend.allocateHostMappedBuffer(bytes);
        GpuBuffer device = backend.allocateBuffer(bytes);
        try {
            float[] src = new float[128];
            for (int i = 0; i < src.length; i++) {
                src[i] = i * 0.25f;
            }
            mapped.hostView(0, bytes).asFloatBuffer().put(src);
            mapped.flush();

            queue.enqueueCopyBuffer(mapped, 0, device, 0, bytes);
            float[] dst = new float[128];
            queue.enqueueReadBuffer(device, 0, bytes, dst);
            queue.finish();

            assertArrayEquals(src, dst, 0.0f);
        } finally {
            device.close();
            mapped.close();
        }
    }

    @Test
    public void aKernelReadsAndWritesHostVisibleMemory() {
        int n = 64;
        int bytes = n * Float.BYTES;
        HostMappedBuffer mapped = backend.allocateHostMappedBuffer(bytes);
        GpuBuffer device = backend.allocateBuffer(bytes);
        try {
            float[] src = new float[n];
            float[] expected = new float[n];
            for (int i = 0; i < n; i++) {
                src[i] = i;
                expected[i] = i + 1.0f;
            }
            mapped.hostView(0, bytes).asFloatBuffer().put(src);
            mapped.flush();

            KernelHandle kernel = backend.compileKernel("increment", "shaders/test/increment.spv", null);
            queue.enqueueKernel(kernel, new long[]{256}, new long[]{256}, new Object[]{mapped, n});
            queue.enqueueCopyBuffer(mapped, 0, device, 0, bytes);
            float[] dst = new float[n];
            queue.enqueueReadBuffer(device, 0, bytes, dst);
            queue.finish();

            assertArrayEquals(expected, dst, 0.0f);
        } finally {
            device.close();
            mapped.close();
        }
    }
}

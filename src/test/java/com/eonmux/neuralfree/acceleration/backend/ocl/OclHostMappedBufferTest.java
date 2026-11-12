package com.eonmux.neuralfree.acceleration.backend.ocl;

import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.DeviceClass;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.HostMappedBuffer;
import com.eonmux.neuralfree.acceleration.backend.KernelHandle;
import com.eonmux.neuralfree.acceleration.ocl.CLDevice;
import org.junit.After;
import org.junit.Assume;
import org.junit.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;

/**
 * OpenCL host-mapped weight memory ({@link OclHostMappedBuffer}): an integrated device fills a
 * {@code CL_MEM_ALLOC_HOST_PTR} buffer through its mapping, and kernels and copies read it in place after the flush; a
 * discrete device reports no host-mapped memory and refuses the allocation.
 */
public class OclHostMappedBufferTest {

    private static final String SCALE_KERNEL =
            "__kernel void scale(__global float *data, const float factor, const int n) {\n"
                    + "    int i = get_global_id(0);\n"
                    + "    if (i < n) data[i] *= factor;\n"
                    + "}\n";

    private final List<OclComputeBackend> backends = new ArrayList<>();

    @After
    public void tearDown() {
        for (OclComputeBackend b : backends) {
            b.close();
        }
    }

    private List<OclComputeBackend> open() {
        List<CLDevice> devices;
        try {
            devices = new HWAcceleration().getCLDevices();
        } catch (RuntimeException | UnsatisfiedLinkError e) {
            devices = new ArrayList<>();
        }
        Assume.assumeFalse("no OpenCL device", devices.isEmpty());
        for (CLDevice d : devices) {
            backends.add(new OclComputeBackend(d));
        }
        return backends;
    }

    private OclComputeBackend integrated() {
        for (OclComputeBackend b : open()) {
            if (b.capabilities().getDeviceClass() == DeviceClass.INTEGRATED) {
                return b;
            }
        }
        Assume.assumeTrue("no integrated OpenCL device", false);
        return null;
    }

    @Test
    public void onlyDevicesSharingHostMemoryAllocateHostMappedBuffers() {
        for (OclComputeBackend b : open()) {
            boolean integrated = b.capabilities().getDeviceClass() == DeviceClass.INTEGRATED;
            assertEquals(b.capabilities().getDeviceName(), integrated, b.supportsHostMappedBuffers());
            if (!integrated) {
                assertThrows(UnsupportedOperationException.class, () -> b.allocateHostMappedBuffer(64));
            }
        }
    }

    @Test
    public void aKernelReadsAndWritesTheFlushedBufferInPlace() {
        OclComputeBackend backend = integrated();
        ComputeQueue queue = backend.computeQueue();
        int n = 1000;
        int bytes = n * Float.BYTES;
        long allocatedBefore = backend.getAllocatedBytes();
        HostMappedBuffer mapped = backend.allocateHostMappedBuffer(bytes);
        GpuBuffer device = backend.allocateBuffer(bytes);
        try {
            assertEquals("host-mapped memory is not device memory", allocatedBefore + device.sizeInBytes(),
                    backend.getAllocatedBytes());
            ByteBuffer view = mapped.hostView(16, 64);
            assertEquals(ByteOrder.LITTLE_ENDIAN, view.order());
            assertEquals(64, view.remaining());
            assertThrows(IllegalArgumentException.class, () -> mapped.hostView(0, (int) mapped.sizeInBytes() + 1));

            float[] src = new float[n];
            float[] expected = new float[n];
            for (int i = 0; i < n; i++) {
                src[i] = i * 0.5f;
                expected[i] = src[i] * 3f;
            }
            mapped.hostView(0, bytes).asFloatBuffer().put(src);
            mapped.flush();

            KernelHandle kernel = backend.compileKernel("scale", SCALE_KERNEL, null);
            queue.enqueueKernel(kernel, new long[]{1024}, new long[]{64}, new Object[]{mapped, 3f, n});
            queue.enqueueCopyBuffer(mapped, 0, device, 0, bytes);
            float[] copied = new float[n];
            queue.enqueueReadBuffer(device, 0, bytes, copied);
            queue.finish();
            assertArrayEquals(expected, copied, 0f);

            float[] inPlace = new float[n];
            mapped.hostView(0, bytes).asFloatBuffer().get(inPlace);
            assertArrayEquals("a later view maps the device's writes", expected, inPlace, 0f);
            mapped.flush();
        } finally {
            device.close();
            mapped.close();
            mapped.close();
        }
        assertFalse(backend.getAllocatedBytes() > allocatedBefore);
    }

    @Test
    public void aBufferClosedWhileMappedIsReleased() {
        OclComputeBackend backend = integrated();
        HostMappedBuffer mapped = backend.allocateHostMappedBuffer(256);
        mapped.hostView(0, 256).put(0, (byte) 7);
        mapped.close();
        assertThrows(IllegalStateException.class, () -> mapped.hostView(0, 4));
        assertEquals(-1L, mapped.bufferId());
    }
}

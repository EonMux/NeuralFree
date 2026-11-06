package com.eonmux.neuralfree.acceleration.backend.cuda;

import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.HostMappedBuffer;
import com.eonmux.neuralfree.acceleration.backend.KernelHandle;
import com.eonmux.neuralfree.acceleration.cuda.CUDevice;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.EnumSet;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * CUDA mapped host memory ({@link CudaHostMappedBuffer}): the device reads a page-locked host allocation through
 * {@code cuMemHostGetDevicePointer}, both through a copy and through a kernel argument, and closing returns the
 * memory (also when the backend closes first).
 */
public class CudaHostMappedBufferTest {

    private static final String COPY_PTX =
            ".version 7.5\n"
                    + ".target sm_50\n"
                    + ".address_size 64\n"
                    + "\n"
                    + ".visible .entry copy_host(\n"
                    + "    .param .u64 src,\n"
                    + "    .param .u64 dst,\n"
                    + "    .param .u32 n\n"
                    + ") {\n"
                    + "    .reg .f32   %f<2>;\n"
                    + "    .reg .b32   %r<5>;\n"
                    + "    .reg .b64   %rd<6>;\n"
                    + "    .reg .pred  p0;\n"
                    + "\n"
                    + "    ld.param.u64    %rd1, [src];\n"
                    + "    ld.param.u64    %rd2, [dst];\n"
                    + "    ld.param.u32    %r1, [n];\n"
                    + "    mov.u32         %r2, %tid.x;\n"
                    + "    mov.u32         %r3, %ntid.x;\n"
                    + "    mov.u32         %r4, %ctaid.x;\n"
                    + "    mad.lo.u32      %r2, %r4, %r3, %r2;\n"
                    + "    setp.ge.u32     p0, %r2, %r1;\n"
                    + "    @p0 bra         L_exit;\n"
                    + "    mul.wide.u32    %rd3, %r2, 4;\n"
                    + "    add.s64         %rd4, %rd1, %rd3;\n"
                    + "    add.s64         %rd5, %rd2, %rd3;\n"
                    + "    ld.global.f32   %f1, [%rd4];\n"
                    + "    st.global.f32   [%rd5], %f1;\n"
                    + "L_exit:\n"
                    + "    ret;\n"
                    + "}\n";

    private HWAcceleration acceleration;
    private CudaComputeBackend backend;
    private ComputeQueue queue;
    private boolean gpuAvailable;

    @Before
    public void setUp() {
        try {
            acceleration = new HWAcceleration(EnumSet.of(ComputeBackend.BackendType.CUDA));
            if (!acceleration.getCUDevices().isEmpty()) {
                CUDevice device = acceleration.getCUDevices().get(0);
                backend = new CudaComputeBackend(device);
                queue = backend.computeQueue();
                gpuAvailable = true;
            }
        } catch (Throwable e) {
            gpuAvailable = false;
        }
        Assume.assumeTrue("no CUDA GPU", gpuAvailable);
    }

    @After
    public void tearDown() {
        if (backend != null) {
            backend.close();
        }
        if (acceleration != null) {
            acceleration.shutdown();
        }
    }

    @Test
    public void theBackendSupportsHostMappedBuffers() {
        assertTrue(backend.supportsHostMappedBuffers());
    }

    @Test
    public void theDeviceCopiesFromMappedHostMemory() {
        int bytes = 256 * Float.BYTES;
        HostMappedBuffer mapped = backend.allocateHostMappedBuffer(bytes);
        GpuBuffer device = backend.allocateBuffer(bytes);
        try {
            float[] src = new float[256];
            for (int i = 0; i < src.length; i++) {
                src[i] = i * 0.5f;
            }
            ByteBuffer view = mapped.hostView(0, bytes);
            view.asFloatBuffer().put(src);
            mapped.flush();

            queue.enqueueCopyBuffer(mapped, 0, device, 0, bytes);
            float[] dst = new float[256];
            queue.enqueueReadBuffer(device, 0, bytes, dst);
            queue.finish();

            assertArrayEquals(src, dst, 0.0f);
            assertEquals(bytes, mapped.sizeInBytes());
            assertTrue(mapped instanceof CudaHostMappedBuffer);
        } finally {
            device.close();
            mapped.close();
        }
    }

    @Test
    public void aKernelReadsMappedHostMemory() {
        int bytes = 64 * Float.BYTES;
        HostMappedBuffer mapped = backend.allocateHostMappedBuffer(bytes);
        GpuBuffer device = backend.allocateBuffer(bytes);
        try {
            float[] src = new float[64];
            for (int i = 0; i < src.length; i++) {
                src[i] = 1.0f + i;
            }
            mapped.hostView(0, bytes).asFloatBuffer().put(src);
            mapped.flush();

            KernelHandle kernel = backend.compileKernel("copy_host", COPY_PTX, null);
            queue.enqueueKernel(kernel, new long[]{64}, new long[]{64}, new Object[]{mapped, device, 64});
            float[] dst = new float[64];
            queue.enqueueReadBuffer(device, 0, bytes, dst);
            queue.finish();

            assertArrayEquals(src, dst, 0.0f);
        } finally {
            device.close();
            mapped.close();
        }
    }

    @Test
    public void hostViewsAreBoundedAndLittleEndian() {
        HostMappedBuffer mapped = backend.allocateHostMappedBuffer(64);
        try {
            ByteBuffer view = mapped.hostView(8, 16);
            assertEquals(ByteOrder.LITTLE_ENDIAN, view.order());
            assertEquals(0, view.position());
            assertEquals(16, view.limit());
            assertThrows(IllegalArgumentException.class, () -> mapped.hostView(0, 65));
            assertThrows(IllegalArgumentException.class, () -> mapped.hostView(-1, 4));
        } finally {
            mapped.close();
        }
    }

    @Test
    public void closingIsIdempotentAndInvalidatesTheBuffer() {
        HostMappedBuffer mapped = backend.allocateHostMappedBuffer(64);
        long id = mapped.bufferId();
        assertTrue(id > 0);

        mapped.close();
        mapped.close();

        assertEquals(-1, mapped.bufferId());
        assertTrue(((CudaHostMappedBuffer) mapped).isClosed());
        assertThrows(IllegalStateException.class, () -> mapped.hostView(0, 4));
    }

    @Test
    public void theBackendFreesABufferStillOpenWhenItCloses() {
        CudaComputeBackend second = new CudaComputeBackend(backend.getCUDevice());
        HostMappedBuffer mapped = second.allocateHostMappedBuffer(64);
        mapped.hostView(0, 4).putInt(7);
        mapped.flush();

        second.close();
        mapped.close();

        assertEquals(-1, mapped.bufferId());
    }
}

package com.eonmux.neuralfree.acceleration.backend.hip;

import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.DeviceFeature;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.acceleration.backend.KernelArtifact;
import com.eonmux.neuralfree.acceleration.backend.KernelHandle;
import com.eonmux.neuralfree.acceleration.UnsupportedKernelException;
import com.eonmux.neuralfree.acceleration.backend.KernelSignature;
import com.eonmux.neuralfree.acceleration.hip.FakeHipDriver;
import com.eonmux.neuralfree.acceleration.hip.HipDevice;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNotSame;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * The HIP backend end to end over the simulated device of {@link FakeHipDriver}: allocation,
 * staged transfers, dispatch and readback of the fake scale kernel, events across queues,
 * pre-bound arguments, signature checks, buffer pooling and close semantics.
 */
public class HipComputeBackendIntegrationTest {

    private FakeHipDriver driver;
    private HipComputeBackend backend;

    @Before
    public void openBackend() {
        driver = new FakeHipDriver();
        HipDevice device = new HipDevice(0, driver);
        backend = new HipComputeBackend(device);
    }

    @After
    public void closeBackend() {
        if (!backend.isClosed()) {
            backend.close();
        }
    }

    @Test
    public void writeDispatchAndReadBackThroughTheScaleKernel() throws Exception {
        ComputeQueue queue = backend.computeQueue();
        float[] input = {1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f};
        HipGpuBuffer buffer = (HipGpuBuffer) backend.allocateBuffer(input.length * Float.BYTES);
        queue.enqueueWriteBuffer(buffer, 0, input.length * Float.BYTES, input).awaitCompletion();

        KernelHandle kernel = backend.loadKernel(scaleArtifact());
        queue.enqueueKernel(kernel, new long[]{input.length}, new long[]{64},
                new Object[]{buffer, 2f, input.length}).awaitCompletion();

        float[] output = new float[input.length];
        queue.enqueueReadBuffer(buffer, 0, output.length * Float.BYTES, output).awaitCompletion();
        assertArrayEquals(new float[]{2f, 4f, 6f, 8f, 10f, 12f, 14f, 16f}, output, 1e-6f);

        // The launch ran 1 block of 64 threads with the marshalled (pointer, float, int) parameters.
        List<FakeHipDriver.Launch> launches = driver.launches();
        assertEquals(1, launches.size());
        FakeHipDriver.Launch launch = launches.get(0);
        assertEquals(FakeHipDriver.SCALE_KERNEL, launch.kernel);
        assertEquals(1, launch.gridX);
        assertEquals(64, launch.blockX);
        assertEquals(3, launch.params.length);
        assertEquals(buffer.getDevicePointer(), launch.params[0]);
        assertEquals(2f, (Float) launch.params[1], 0f);
        assertEquals(input.length, ((Integer) launch.params[2]).intValue());
    }

    @Test
    public void fillAndDeviceCopy() throws Exception {
        ComputeQueue queue = backend.computeQueue();
        GpuBuffer src = backend.allocateBuffer(64);
        GpuBuffer dst = backend.allocateBuffer(64);
        queue.enqueueFillBuffer(src, 0, 64, 0x01020304);
        queue.enqueueCopyBuffer(src, 0, dst, 0, 64);
        queue.finish();

        byte[] out = new byte[64];
        queue.enqueueReadBuffer(dst, 0, out.length, out).awaitCompletion();
        // The fake fills 32-bit words little-endian, so 0x01020304 is stored as 04 03 02 01.
        assertEquals(0x04, out[0] & 0xFF);
        assertEquals(0x03, out[1] & 0xFF);
        assertEquals(0x02, out[2] & 0xFF);
        assertEquals(0x01, out[3] & 0xFF);
    }

    @Test
    public void crossQueueWaitsOrderTheWork() throws Exception {
        ComputeQueue compute = backend.computeQueue();
        ComputeQueue transfer = backend.transferQueue();
        GpuEvent marker = transfer.enqueueMarker();
        compute.enqueueWriteBuffer(writeTarget(), 0, 8, new byte[8], marker);
        compute.finish();
    }

    @Test
    public void preBoundArgumentsLaunchWithTheStoredSnapshot() {
        ComputeQueue queue = backend.computeQueue();
        KernelHandle owned = backend.loadKernel(scaleArtifact());
        KernelHandle clone = backend.cloneKernel(owned);

        GpuBuffer buffer = backend.allocateBuffer(16);
        queue.setKernelArg(clone, 0, buffer);
        queue.setKernelArg(clone, 1, 3f);
        queue.setKernelArg(clone, 2, 4);
        queue.enqueueKernelPreBound(clone, new long[]{4}, new long[]{64});
        queue.finish();

        List<FakeHipDriver.Launch> launches = driver.launches();
        assertEquals(1, launches.size());
        assertEquals(3f, (Float) launches.get(0).params[1], 0f);
        clone.close();
    }

    @Test
    public void signatureChecksRejectWrongArguments() {
        // The kernel library registers the inventory signature at resolve time; this test
        // registers the same signature the test library's inventory records.
        HipKernelSignatures.register("hip-test-lib/src/hip/nf_test_scale.hsaco",
                FakeHipDriver.SCALE_KERNEL, KernelSignature.parseInventory("1", "2", "f,i", "Bfi"));
        ComputeQueue queue = backend.computeQueue();
        KernelHandle kernel = backend.loadKernel(scaleArtifact());
        GpuBuffer buffer = backend.allocateBuffer(16);
        try {
            // (buffer, float, int) declared; the int where the float belongs is rejected.
            queue.enqueueKernel(kernel, new long[]{4}, new long[]{64}, new Object[]{buffer, 2, 1f});
            fail("a wrong scalar kind must be rejected");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("nf_test_scale"));
        }
        try {
            queue.enqueueKernel(kernel, new long[]{4}, new long[]{64}, new Object[]{buffer, 2f});
            fail("a short argument list must be rejected");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("1 buffers and 2 scalars"));
        }
        try {
            // A gap in the pre-bound list fails before the signature, at the completeness check.
            KernelHandle clone = backend.cloneKernel(kernel);
            queue.setKernelArg(clone, 0, buffer);
            queue.setKernelArg(clone, 2, 4);
            queue.enqueueKernelPreBound(clone, new long[]{4}, new long[]{64});
            fail("an argument list with a gap must be rejected");
        } catch (IllegalStateException expected) {
            assertTrue(expected.getMessage().contains("arg 1"));
        }
    }

    @Test
    public void closedBuffersReturnToThePoolAndAreReused() {
        ComputeQueue queue = backend.computeQueue();
        float[] data = {1f, 2f, 3f, 4f};
        HipGpuBuffer first = (HipGpuBuffer) backend.allocateBuffer(data.length * Float.BYTES);
        queue.enqueueWriteBuffer(first, 0, data.length * Float.BYTES, data);
        queue.finish();
        Long firstPointer = first.getDevicePointer();
        first.close();
        // The release fences complete, so the block becomes ready for reuse.
        driver.completeAll();

        HipGpuBuffer second = (HipGpuBuffer) backend.allocateBuffer(data.length * Float.BYTES);
        assertEquals("the pooled block is reused", firstPointer, second.getDevicePointer());
        // getAllocatedBytes counts the 256-byte aligned block, not the requested 16 bytes.
        assertEquals(HipComputeBackend.BUFFER_ALIGNMENT, backend.getAllocatedBytes());
        second.close();
    }

    @Test
    public void allocationFreesPooledSiblingsWhenTheDeviceIsFull() {
        ComputeQueue queue = backend.computeQueue();
        backend.trimPool();
        driver.capacity = 3 * 256;
        GpuBuffer small = backend.allocateBuffer(256);
        queue.enqueueWriteBuffer(small, 0, 8, new byte[8]);
        queue.finish();
        small.close();
        // The 256-byte block is pooled with its fence in flight; a full-capacity allocation
        // synchronizes, frees the pool and retries.
        HipGpuBuffer full = (HipGpuBuffer) backend.allocateBuffer(768);
        assertEquals(768, full.allocationBytes());
        assertEquals(1, backend.bufferPoolStats().oomRetries());
        full.close();
    }

    @Test
    public void compileKernelRefusesSourceTextByName() {
        try {
            backend.compileKernel("scale", "__global__ void scale() {}", "");
            fail("source text must be refused");
        } catch (UnsupportedKernelException expected) {
            assertTrue(expected.getMessage().contains("hipImage"));
        }
    }

    @Test
    public void loadKernelRejectsForeignArtifactKinds() {
        KernelArtifact foreign = KernelArtifact.cudaImage("scale", "kernels/llm/cuda/scale.fatbin");
        try {
            backend.loadKernel(foreign);
            fail("a CUDA artifact must be refused");
        } catch (UnsupportedKernelException expected) {
            assertTrue(expected.getMessage().contains("CUDA_IMAGE"));
        }
    }

    @Test
    public void closedBackendRejectsWorkAndBuffersCloseQuietly() {
        HipGpuBuffer buffer = (HipGpuBuffer) backend.allocateBuffer(64);
        backend.close();
        try {
            backend.allocateBuffer(64);
            fail("a closed backend must refuse allocations");
        } catch (IllegalStateException expected) {
            // expected
        }
        buffer.close(); // the backend already freed the memory; closing must not throw
    }

    @Test
    public void freeVramReportsTheDriverFigurePlusPooledBlocks() {
        ComputeQueue queue = backend.computeQueue();
        driver.capacity = 1024 * 1024;
        GpuBuffer buffer = backend.allocateBuffer(256);
        queue.enqueueWriteBuffer(buffer, 0, 8, new byte[8]);
        queue.finish();
        buffer.close();
        // hipMemGetInfo of the fake reports capacity minus its live blocks; the trimmed 256-byte
        // block is pooled by this backend and counted on top.
        assertEquals(1024 * 1024, backend.getFreeVramBytes());
    }

    @Test
    public void kernelOfAnotherBackendIsRejected() {
        FakeHipDriver otherDriver = new FakeHipDriver();
        HipComputeBackend other = new HipComputeBackend(new HipDevice(0, otherDriver));
        try {
            other.computeQueue().enqueueKernel(backend.loadKernel(scaleArtifact()),
                    new long[]{4}, new long[]{64}, new Object[0]);
            fail("a foreign handle must be rejected");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("another backend"));
        } finally {
            other.close();
        }
    }

    @Test
    public void repeatedDispatchSizesReuseTheGeometryAndChangedOnesDeriveIt() throws Exception {
        ComputeQueue queue = backend.computeQueue();
        HipGpuBuffer buffer = (HipGpuBuffer) backend.allocateBuffer(1024 * Float.BYTES);
        KernelHandle kernel = backend.loadKernel(scaleArtifact());
        long[] global = {128};
        long[] local = {64};
        queue.enqueueKernel(kernel, global, local, new Object[]{buffer, 2f, 128});
        queue.enqueueKernel(kernel, global, local, new Object[]{buffer, 2f, 128});
        // the caller's arrays change in place: the kept geometry is keyed by their values, not the arrays
        global[0] = 512;
        queue.enqueueKernel(kernel, global, local, new Object[]{buffer, 2f, 512});
        local[0] = 128;
        queue.enqueueKernel(kernel, global, local, new Object[]{buffer, 2f, 512});
        queue.enqueueKernel(kernel, global, null, new Object[]{buffer, 2f, 512});
        queue.finish();

        List<FakeHipDriver.Launch> launches = driver.launches();
        assertEquals(5, launches.size());
        int[][] gridBlock = {{2, 64}, {2, 64}, {8, 64}, {4, 128}, {2, 256}};
        for (int i = 0; i < gridBlock.length; i++) {
            FakeHipDriver.Launch launch = launches.get(i);
            assertEquals("launch " + i + " grid", gridBlock[i][0], launch.gridX);
            assertEquals("launch " + i + " block", gridBlock[i][1], launch.blockX);
            assertEquals(1, launch.gridY);
            assertEquals(1, launch.blockZ);
        }
        try {
            queue.enqueueKernel(kernel, new long[]{64, 65536}, new long[]{64, 1},
                    new Object[]{buffer, 2f, 64});
            fail("a grid beyond the device's y limit must be refused");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains("exceed the device limit of 65535"));
        }
        assertTrue(driver.launches().isEmpty());
    }

    @Test
    public void launchParamsAreBuiltOnceWhileTheArgumentsRepeat() {
        HipKernelHandle kernel = (HipKernelHandle) backend.loadKernel(scaleArtifact());
        HipGpuBuffer buffer = (HipGpuBuffer) backend.allocateBuffer(64);
        HipGpuBuffer other = (HipGpuBuffer) backend.allocateBuffer(64);
        HipArgState args = kernel.args();
        args.setAll(new Object[]{buffer, 2f, 16});
        Object[] first = args.launchParams();
        args.setAll(new Object[]{buffer, 2f, 16});
        assertSame("unchanged arguments reuse the table", first, args.launchParams());
        assertEquals(buffer.getDevicePointer(), first[0]);

        args.set(1, 3f);
        Object[] changed = args.launchParams();
        assertNotSame(first, changed);
        assertEquals(3f, changed[1]);
        assertEquals("a table handed out is never rewritten", 2f, first[1]);

        args.set(0, other);
        assertEquals(other.getDevicePointer(), args.launchParams()[0]);
        other.close();
        IllegalStateException closed = assertThrows(IllegalStateException.class, args::launchParams);
        assertTrue(closed.getMessage(), closed.getMessage().contains("closed after it was set"));
        buffer.close();
    }

    @Test
    public void capabilitiesCarryTheGfxTargetAndItsFeatures() {
        assertEquals(0x1000000 | 0x1100, backend.capabilities().getComputeCapability());
        assertEquals("AMD Radeon Test GPU", backend.capabilities().getDeviceName());
        assertTrue("RDNA3 reports WMMA", backend.capabilities().hasFeature(DeviceFeature.COOPERATIVE_MATRIX));
        assertEquals(32, backend.capabilities().getSubgroupSize());
    }

    @Test
    public void readAwaitsItsEventAndRetiresTheDestination() throws Exception {
        ComputeQueue queue = backend.computeQueue();
        GpuBuffer buffer = backend.allocateBuffer(8);
        queue.enqueueWriteBuffer(buffer, 0, 8, new byte[]{1, 2, 3, 4, 5, 6, 7, 8});
        byte[] out = new byte[8];
        GpuEvent read = queue.enqueueReadBuffer(buffer, 0, 8, out);
        assertNotNull(read);
        read.awaitCompletion();
        assertArrayEquals(new byte[]{1, 2, 3, 4, 5, 6, 7, 8}, out);
        read.close();
        assertTrue("a closed handle reports complete", read.isComplete());
        buffer.close();
    }

    @Test
    public void transferQueueRejectsNothingButDispatchStaysOnCompute() {
        assertEquals(ComputeQueue.QueueType.COMPUTE, backend.computeQueue().getQueueType());
        assertEquals(ComputeQueue.QueueType.TRANSFER, backend.transferQueue().getQueueType());
    }

    /** A buffer for waits that need only a destination; closed by the pool accounting of the test. */
    private GpuBuffer writeTarget() {
        return backend.allocateBuffer(8);
    }

    private static KernelArtifact scaleArtifact() {
        return KernelArtifact.hipImage(FakeHipDriver.SCALE_KERNEL,
                "hip-test-lib/src/hip/nf_test_scale.hsaco");
    }
}

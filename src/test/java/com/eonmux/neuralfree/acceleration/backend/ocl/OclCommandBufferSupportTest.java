package com.eonmux.neuralfree.acceleration.backend.ocl;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.DeviceFeature;
import com.eonmux.neuralfree.acceleration.ocl.CLDevice;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The OpenCL capture capability is reported honestly: a queue supports capture exactly when the
 * device reports {@link DeviceFeature#COMMAND_BUFFER}, and a device without
 * {@code cl_khr_command_buffer} keeps running directly and names the missing extension when asked
 * to capture. These assertions hold on every OpenCL device, extension or not.
 */
public class OclCommandBufferSupportTest {

    private HWAcceleration acceleration;
    private OclComputeBackend backend;

    @Before
    public void setUp() {
        Assume.assumeTrue("no OpenCL GPU", GpuTestUtil.isOpenCLAvailable());
        acceleration = new HWAcceleration();
        CLDevice device = acceleration.getCLDevices().get(0);
        backend = new OclComputeBackend(device);
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
    public void captureSupportMatchesTheDeviceCapability() {
        boolean feature = backend.capabilities().hasFeature(DeviceFeature.COMMAND_BUFFER);
        assertEquals(feature, backend.computeQueue().supportsGraphCapture());
        assertEquals(feature, backend.transferQueue().supportsGraphCapture());
    }

    @Test
    public void aDeviceWithoutTheExtensionRefusesCaptureByName() {
        Assume.assumeFalse("the device advertises cl_khr_command_buffer",
                backend.capabilities().hasFeature(DeviceFeature.COMMAND_BUFFER));
        ComputeQueue queue = backend.computeQueue();
        UnsupportedOperationException begin = assertThrows(UnsupportedOperationException.class,
                queue::beginCapture);
        assertTrue(begin.getMessage(), begin.getMessage().contains("cl_khr_command_buffer"));
        UnsupportedOperationException end = assertThrows(UnsupportedOperationException.class,
                queue::endCapture);
        assertTrue(end.getMessage(), end.getMessage().contains("graph capture"));
        UnsupportedOperationException abort = assertThrows(UnsupportedOperationException.class,
                queue::abortCapture);
        assertTrue(abort.getMessage(), abort.getMessage().contains("graph capture"));
        assertFalse(queue.isCapturing());
    }

    @Test
    public void aDeviceWithTheExtensionStartsAndEndsAnEmptyCapture() {
        Assume.assumeTrue("the device does not advertise cl_khr_command_buffer",
                backend.capabilities().hasFeature(DeviceFeature.COMMAND_BUFFER));
        ComputeQueue queue = backend.computeQueue();
        queue.beginCapture();
        assertTrue(queue.isCapturing());
        assertEquals(null, queue.endCapture());
        assertFalse(queue.isCapturing());
    }
}

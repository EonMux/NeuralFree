package com.eonmux.neuralfree.acceleration.backend.ocl;

import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import org.jocl.cl_event;
import org.junit.Test;

import java.util.EnumSet;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

public class OclGpuEventTest {

    @Test
    public void unprofiledEventReportsMinusOneWithoutQueryingTheDriver() {
        OclGpuEvent event = new OclGpuEvent(new cl_event(), null, false);
        assertEquals(-1L, event.getElapsedNanos());
    }

    @Test
    public void nullEventIsRejected() {
        assertThrows(IllegalArgumentException.class, () -> new OclGpuEvent(null));
    }

    @Test
    public void profiledEventReportsElapsedTimeAndRejectsUseAfterClose() throws InterruptedException {
        String previous = System.getProperty(OclComputeBackend.PROFILE_PROPERTY);
        System.setProperty(OclComputeBackend.PROFILE_PROPERTY, "true");
        HWAcceleration hwAccel = null;
        OclComputeBackend backend = null;
        try {
            hwAccel = new HWAcceleration(EnumSet.of(ComputeBackend.BackendType.OPENCL));
            assumeTrue("No OpenCL device available", !hwAccel.getCLDevices().isEmpty());
            backend = new OclComputeBackend(hwAccel.getCLDevices().get(0));

            int size = 256;
            float[] data = new float[size];
            try (GpuBuffer buffer = backend.allocateBuffer((long) size * Float.BYTES)) {
                GpuEvent write = backend.transferQueue().enqueueWriteBuffer(buffer, 0, (long) size * Float.BYTES,
                        data);
                write.awaitCompletion();
                assertTrue("A completed profiled command reports a non-negative duration",
                        write.getElapsedNanos() >= 0);
                write.close();
                assertThrows(IllegalStateException.class, write::getElapsedNanos);
            }
        } finally {
            if (backend != null) {
                backend.close();
            }
            if (hwAccel != null) {
                hwAccel.shutdown();
            }
            if (previous == null) {
                System.clearProperty(OclComputeBackend.PROFILE_PROPERTY);
            } else {
                System.setProperty(OclComputeBackend.PROFILE_PROPERTY, previous);
            }
        }
    }
}

package com.eonmux.neuralfree.acceleration.ocl;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend.BackendType;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_context;
import org.junit.Test;

import java.util.EnumSet;

import static org.jocl.CL.CL_CONTEXT_REFERENCE_COUNT;
import static org.jocl.CL.CL_SUCCESS;
import static org.jocl.CL.clGetContextInfo;
import static org.jocl.CL.clReleaseContext;
import static org.jocl.CL.clRetainContext;
import static org.junit.Assert.assertEquals;
import static org.junit.Assume.assumeFalse;

/**
 * Release semantics of an enumerated {@link CLDevice}: it drops its context reference exactly once, however often it
 * is released (a repeated {@link HWAcceleration#shutdown()} included), so it never releases a reference another
 * holder owns. Observed through {@code CL_CONTEXT_REFERENCE_COUNT} while the test holds its own retain.
 */
public class CLDeviceReleaseTest {

    @Test
    public void releaseDropsTheContextReferenceOnce() {
        GpuTestUtil.assumeOpenCLAvailable();
        HWAcceleration hw = new HWAcceleration(EnumSet.of(BackendType.OPENCL));
        try {
            assumeFalse("no OpenCL device enumerated", hw.getCLDevices().isEmpty());
            CLDevice device = hw.getCLDevices().get(0);
            cl_context context = device.getDeviceContext();
            assertEquals(CL_SUCCESS, clRetainContext(context));
            try {
                int held = referenceCount(context);
                device.release();
                assertEquals("release drops the enumeration's reference", held - 1, referenceCount(context));
                device.release();
                hw.shutdown();
                assertEquals("a repeated release must not drop a reference the device no longer holds",
                        held - 1, referenceCount(context));
            } finally {
                clReleaseContext(context);
            }
        } finally {
            hw.shutdown();
        }
    }

    private static int referenceCount(cl_context context) {
        int[] count = new int[1];
        assertEquals(CL_SUCCESS, clGetContextInfo(context, CL_CONTEXT_REFERENCE_COUNT, Sizeof.cl_uint,
                Pointer.to(count), null));
        return count[0];
    }
}

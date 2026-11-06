package com.eonmux.neuralfree.acceleration.backend.ocl;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.acceleration.backend.KernelHandle;
import com.eonmux.neuralfree.training.LogCapture;
import org.apache.logging.log4j.Level;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_event;
import org.junit.Test;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.List;

import static org.jocl.CL.*;
import static org.junit.Assert.*;

/** OpenCL events cannot retain a closed backend's queues and context, even when callers drop every event. */
public class OclEventOwnershipTest {

    private static final int CYCLES = 100;
    private static final int BUFFER_FLOATS = 64;
    private static final String ADD_ONE_SOURCE =
            "__kernel void add_one(__global float *x, float v) { x[get_global_id(0)] += v; }";

    @Test
    public void backendCloseReleasesTheEventsCallersLeftOpen() {
        GpuTestUtil.assumeOpenCLAvailable();
        HWAcceleration hw = new HWAcceleration(EnumSet.of(ComputeBackend.BackendType.OPENCL));
        try (OclComputeBackend backend = new OclComputeBackend(hw.getCLDevices().get(0));
             LogCapture logs = new LogCapture(OclComputeBackend.class)) {
            List<GpuEvent> leaked = issueUnclosedEvents(backend);
            assertEquals(4, backend.openEvents());
            backend.computeQueue().finish();
            cl_event nativeEvent = ((OclGpuEvent) leaked.get(3)).getCLEvent();
            assertEquals(CL_SUCCESS, clRetainEvent(nativeEvent));
            try {
                int refs = references(nativeEvent);
                backend.close();
                backend.close();
                assertEquals("close releases the native reference, not just its registry entry",
                        refs - 1, references(nativeEvent));
                assertEquals(0, backend.openEvents());
                for (GpuEvent event : leaked) {
                    OclGpuEvent ocl = (OclGpuEvent) event;
                    assertTrue(ocl.isClosed());
                    assertThrows(IllegalStateException.class, ocl::getCLEvent);
                    event.close();
                }
                List<String> warnings = logs.messages(Level.WARN);
                assertEquals(warnings.toString(), 1, warnings.size());
                assertTrue(warnings.get(0), warnings.get(0).contains("4 event(s)"));
                assertTrue(warnings.get(0), warnings.get(0).contains("OpenCL backend"));
            } finally {
                assertEquals(CL_SUCCESS, clReleaseEvent(nativeEvent));
            }
        } finally {
            hw.shutdown();
        }
    }

    @Test
    public void closedEventsLeaveTheRegistryAndDoNotWarn() {
        GpuTestUtil.assumeOpenCLAvailable();
        HWAcceleration hw = new HWAcceleration(EnumSet.of(ComputeBackend.BackendType.OPENCL));
        try (OclComputeBackend backend = new OclComputeBackend(hw.getCLDevices().get(0));
             GpuBuffer buffer = backend.allocateBuffer(16);
             LogCapture logs = new LogCapture(OclComputeBackend.class)) {
            ComputeQueue transfer = backend.transferQueue();
            try (GpuEvent bytes = transfer.enqueueWriteBuffer(buffer, 0, 16, new byte[16]);
                 GpuEvent direct = transfer.enqueueWriteBuffer(buffer, 0, 16, ByteBuffer.allocateDirect(16));
                 GpuEvent floats = transfer.enqueueWriteBuffer(buffer, 0, 16, new float[4])) {
                assertEquals(3, backend.openEvents());
                bytes.close();
                bytes.close();
                assertEquals(2, backend.openEvents());
            }
            assertEquals(0, backend.openEvents());
            backend.close();
            assertTrue(logs.messages(Level.WARN).toString(), logs.messages(Level.WARN).isEmpty());
        } finally {
            hw.shutdown();
        }
    }

    @Test
    public void backendOwnsEventsFromClosedCallerQueuesAndTransferQueues() {
        GpuTestUtil.assumeOpenCLAvailable();
        HWAcceleration hw = new HWAcceleration(EnumSet.of(ComputeBackend.BackendType.OPENCL));
        try (OclComputeBackend backend = new OclComputeBackend(hw.getCLDevices().get(0));
             GpuBuffer buffer = backend.allocateBuffer(16)) {
            ComputeQueue caller = backend.createComputeQueue();
            caller.enqueueFillBuffer(buffer, 0, 16, 0);
            KernelHandle kernel = backend.compileKernel("add_one", ADD_ONE_SOURCE, null);
            caller.setKernelArg(kernel, 0, buffer);
            caller.setKernelArg(kernel, 1, 1.0f);
            GpuEvent dispatch = caller.enqueueKernelPreBoundWithEvent(kernel, new long[]{4}, new long[]{4});
            byte[] out = new byte[16];
            GpuEvent read = backend.transferQueue().enqueueReadBuffer(buffer, 0, 16, out, dispatch);
            GpuEvent marker = backend.transferQueue().enqueueMarker(read);
            backend.transferQueue().finish();
            caller.close();
            assertEquals(3, backend.openEvents());
            backend.close();
            assertEquals(0, backend.openEvents());
            assertTrue(((OclGpuEvent) dispatch).isClosed());
            assertTrue(((OclGpuEvent) read).isClosed());
            assertTrue(((OclGpuEvent) marker).isClosed());
            assertThrows(IllegalStateException.class, caller::finish);
        } finally {
            hw.shutdown();
        }
    }

    @Test
    public void closeCyclesReleaseTheDroppedEvents() {
        GpuTestUtil.assumeOpenCLAvailable();
        boolean cudaAvailable = GpuTestUtil.isCudaAvailable();
        for (int cycle = 0; cycle < CYCLES; cycle++) {
            HWAcceleration hw = new HWAcceleration(EnumSet.of(ComputeBackend.BackendType.OPENCL));
            try (OclComputeBackend backend = new OclComputeBackend(hw.getCLDevices().get(0))) {
                issueUnclosedEvents(backend);
                backend.close();
                assertEquals("dropped events at cycle " + cycle, 0, backend.openEvents());
            } finally {
                hw.shutdown();
            }
        }
        if (cudaAvailable) {
            retainCudaPrimaryContext();
        }
    }

    /** Issues a write, kernel, read and marker whose events and buffer are deliberately left for backend close. */
    private static List<GpuEvent> issueUnclosedEvents(OclComputeBackend backend) {
        List<GpuEvent> events = new ArrayList<>();
        GpuBuffer buffer = backend.allocateBuffer(BUFFER_FLOATS * (long) Float.BYTES);
        KernelHandle kernel = backend.compileKernel("add_one", ADD_ONE_SOURCE, null);
        float[] in = new float[BUFFER_FLOATS];
        events.add(backend.computeQueue().enqueueWriteBuffer(buffer, 0, in.length * (long) Float.BYTES, in));
        events.add(backend.computeQueue().enqueueKernel(kernel, new long[]{BUFFER_FLOATS},
                new long[]{BUFFER_FLOATS}, new Object[]{buffer, 1.0f}));
        float[] out = new float[BUFFER_FLOATS];
        events.add(backend.computeQueue().enqueueReadBuffer(buffer, 0, out.length * (long) Float.BYTES, out));
        events.add(backend.computeQueue().enqueueMarker());
        return events;
    }

    private static int references(cl_event event) {
        int[] count = new int[1];
        assertEquals(CL_SUCCESS, clGetEventInfo(event, CL_EVENT_REFERENCE_COUNT, Sizeof.cl_uint,
                Pointer.to(count), null));
        return count[0];
    }

    /** Must not turn an allocation failure after the cycles into an availability skip. */
    private static void retainCudaPrimaryContext() {
        HWAcceleration hw = new HWAcceleration(EnumSet.of(ComputeBackend.BackendType.CUDA));
        try {
            assertFalse("the CUDA device present before the cycles must still be available", hw.getCUDevices().isEmpty());
            try (ComputeBackend backend = hw.createBackend(hw.getCUDevices().get(0))) {
                assertNotNull(backend);
            }
        } finally {
            hw.shutdown();
        }
    }
}

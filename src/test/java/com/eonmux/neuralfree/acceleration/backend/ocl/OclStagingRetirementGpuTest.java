package com.eonmux.neuralfree.acceleration.backend.ocl;

import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.acceleration.backend.KernelHandle;
import com.eonmux.neuralfree.acceleration.backend.KernelTiming;
import com.eonmux.neuralfree.acceleration.ocl.CLDevice;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;

import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * An OpenCL queue that runs for many transfers without a {@code finish()}, as chained greedy decode steps do (each
 * step writes its inputs, dispatches and reads its id, and the host waits only on that read's event): on every
 * OpenCL device here, the staging buffers and pending reads stay bounded by the reads still in flight, every read
 * delivers its own data, and a profiled queue moves the dispatches timed before a completed read into its report
 * without waiting for the finish.
 */
public class OclStagingRetirementGpuTest {

    private static final String INCREMENT =
            "__kernel void increment(__global int *v) {\n"
                    + "    v[get_global_id(0)] += 1;\n"
                    + "}\n";
    private static final int STEPS = 2000;
    private static final int BYTES = 16;
    /** The write and read of the last step stay in flight until a later transfer or a finish. */
    private static final int MAX_IN_FLIGHT = 4;

    private HWAcceleration hw;
    private List<CLDevice> devices;

    @Before
    public void setUp() {
        hw = new HWAcceleration();
        devices = hw.getCLDevices();
        Assume.assumeTrue("no OpenCL device", !devices.isEmpty());
    }

    @After
    public void tearDown() {
        System.clearProperty(KernelTiming.PROFILE_PROPERTY);
        if (hw != null) {
            hw.shutdown();
        }
    }

    @Test
    public void transfersWaitedOnThroughTheirReadsRecycleTheirStagingWithoutAFinish() throws InterruptedException {
        for (CLDevice device : devices) {
            try (OclComputeBackend backend = new OclComputeBackend(device)) {
                runSteps(backend, false);
            }
        }
    }

    @Test
    public void aProfiledQueueReportsTheDispatchesBeforeACompletedRead() throws InterruptedException {
        System.setProperty(KernelTiming.PROFILE_PROPERTY, "true");
        for (CLDevice device : devices) {
            try (OclComputeBackend backend = new OclComputeBackend(device)) {
                runSteps(backend, true);
            }
        }
    }

    /**
     * {@link #STEPS} steps of a host write, an increment dispatch and a read the host waits on, then checks the
     * queue's bounds before and after the closing finish.
     */
    private static void runSteps(OclComputeBackend backend, boolean profiled) throws InterruptedException {
        OclComputeQueue queue = (OclComputeQueue) backend.computeQueue();
        String name = backend.capabilities().getDeviceName();
        KernelHandle kernel = backend.compileKernel("increment", INCREMENT, null);
        GpuBuffer buffer = backend.allocateBuffer(BYTES);
        try {
            int maxInFlight = 0;
            int maxPending = 0;
            byte[] in = new byte[BYTES];
            byte[] out = new byte[BYTES];
            for (int step = 0; step < STEPS; step++) {
                int value = step * 3;
                in[0] = (byte) value;
                in[1] = (byte) (value >>> 8);
                try (GpuEvent write = queue.enqueueWriteBuffer(buffer, 0, BYTES, in)) {
                    queue.enqueueKernelNoEvent(kernel, new long[]{1}, new long[]{1}, new Object[]{buffer});
                    try (GpuEvent read = queue.enqueueReadBuffer(buffer, 0, BYTES, out)) {
                        queue.flush();
                        read.awaitCompletion();
                    }
                }
                int got = (out[0] & 0xff) | (out[1] & 0xff) << 8;
                assertEquals(name + ": step " + step + " reads its own data", (value + 1) & 0xffff, got);
                maxInFlight = Math.max(maxInFlight, queue.stagingBuffersInFlight());
                maxPending = Math.max(maxPending, queue.pendingReadCount());
            }
            if (profiled) {
                assertTrue(name + ": the dispatches before the last completed read are reported before the finish, "
                        + queue.kernelTiming().totalCount(), queue.kernelTiming().totalCount() >= STEPS - 1);
            }
            assertTrue(name + ": staging buffers in flight stay bounded, at most " + maxInFlight,
                    maxInFlight <= MAX_IN_FLIGHT);
            assertTrue(name + ": pending reads stay bounded, at most " + maxPending, maxPending <= 2);
            queue.finish();
            assertEquals(name + ": a finish recycles the rest", 0, queue.stagingBuffersInFlight());
            assertEquals(name + ": a finish retires every read", 0, queue.pendingReadCount());
            if (profiled) {
                assertEquals(name + ": every dispatch is reported after the finish", STEPS,
                        queue.kernelTiming().totalCount());
            }
        } finally {
            buffer.close();
            kernel.close();
        }
    }
}

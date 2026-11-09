package com.eonmux.neuralfree.acceleration.backend.ocl;

import com.eonmux.neuralfree.acceleration.GpuBackendException;
import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.acceleration.backend.KernelHandle;
import com.eonmux.neuralfree.acceleration.ocl.CLDevice;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;

import java.util.EnumSet;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * A kernel that never ends must surface as an exception from the queue's waits once the kernel driver resets the hung
 * context, never as a wait that does not return: a read queued behind it fails, and so does the next
 * {@code finish()}. On the Intel compute runtime the read carries {@link OclErrors#EXECUTION_ABORTED} (measured on
 * the Arrow Lake iGPU under i915: both waits return about 9 s after launch, with {@code preempt_timeout_ms} 7500).
 *
 * <p>The test hangs the GPU on purpose, which makes the kernel driver reset a context and log a GPU hang, so it is
 * opt-in through {@code -Dneuralfree.test.devicefault=true}; {@code -Dneuralfree.test.devicefault.device} picks the
 * OpenCL device by index (default 0).</p>
 */
public class OclDeviceFaultGpuTest {

    private static final String PROPERTY = "neuralfree.test.devicefault";
    /** Longest wait accepted for the driver's reset; far above the 7.5 s preemption timeout of i915. */
    private static final long DEADLINE_SECONDS = 120;
    private static final int INTEL_VENDOR_ID = 0x8086;
    private static final String SOURCE = "__kernel void LLM_TestSpin(__global volatile int* flag) {\n"
            + "    while (flag[0] == 0) { flag[1] = flag[1] + 1; }\n"
            + "}\n";

    private HWAcceleration hw;
    private OclComputeBackend backend;
    private boolean intel;

    @Before
    public void open() {
        Assume.assumeTrue("opt-in (" + PROPERTY + "=true)", Boolean.getBoolean(PROPERTY));
        GpuTestUtil.assumeOpenCLAvailable();
        hw = new HWAcceleration(EnumSet.of(ComputeBackend.BackendType.OPENCL));
        List<CLDevice> devices = hw.getCLDevices();
        int index = Integer.getInteger(PROPERTY + ".device", 0);
        Assume.assumeTrue("no OpenCL device " + index, index >= 0 && index < devices.size());
        CLDevice device = devices.get(index);
        intel = device.getIdentity().vendorId() == INTEL_VENDOR_ID;
        backend = new OclComputeBackend(device);
    }

    @After
    public void close() {
        // The context the driver reset is unusable, so its release may report the fault the test asserted.
        if (backend != null) {
            try {
                backend.close();
            } catch (RuntimeException e) {
                System.err.println("release of the reset context reported: " + e);
            }
        }
        if (hw != null) {
            hw.shutdown();
        }
    }

    @Test
    public void aHungKernelFailsTheWaitsBehindItInsteadOfBlockingThem() throws Exception {
        KernelHandle spin = backend.compileKernel("LLM_TestSpin", SOURCE, "");
        GpuBuffer flag = backend.allocateBuffer(64);
        ComputeQueue queue = backend.computeQueue();
        queue.enqueueWriteBuffer(flag, 0, 8, new byte[8]).close();
        queue.finish();

        queue.enqueueKernelNoEvent(spin, new long[]{1}, new long[]{1}, new Object[]{flag});
        GpuEvent read = queue.enqueueReadBuffer(flag, 0, 8, new byte[8]);
        queue.flush();

        Throwable readFailure = failureWithin(() -> {
            read.awaitCompletion();
            return null;
        });
        assertTrue("the read behind the hung kernel failed: " + readFailure,
                readFailure instanceof GpuBackendException);
        if (intel) {
            assertEquals(OclErrors.EXECUTION_ABORTED, ((GpuBackendException) readFailure).getCode());
        }
        Throwable finishFailure = failureWithin(() -> {
            queue.finish();
            return null;
        });
        assertTrue("finish after the reset failed: " + finishFailure, finishFailure instanceof RuntimeException);
    }

    /** Runs {@code wait} off the test thread and returns what it threw, failing when it neither threw nor returned. */
    private static Throwable failureWithin(java.util.concurrent.Callable<Void> wait) throws InterruptedException {
        CompletableFuture<Void> done = CompletableFuture.supplyAsync(() -> {
            try {
                return wait.call();
            } catch (Exception e) {
                throw new java.util.concurrent.CompletionException(e);
            }
        });
        try {
            done.get(DEADLINE_SECONDS, TimeUnit.SECONDS);
            fail("the wait returned normally behind a kernel that never ends");
            return null;
        } catch (java.util.concurrent.TimeoutException e) {
            fail("the wait did not return within " + DEADLINE_SECONDS + " s of the hang");
            return null;
        } catch (ExecutionException e) {
            return e.getCause();
        }
    }
}

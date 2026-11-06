package com.eonmux.neuralfree.gguf.ops.moe;

import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ocl.OclComputeBackend;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;

/** {@link MoeGroupedPathsParity} on an OpenCL device. Skips without an OpenCL GPU. */
public class MoeGroupedPathsOpenClTest extends MoeGroupedPathsParity {

    private HWAcceleration hwAccel;
    private ComputeBackend backend;

    @Before
    public void setUp() {
        try {
            hwAccel = new HWAcceleration();
            if (!hwAccel.getCLDevices().isEmpty()) {
                backend = new OclComputeBackend(hwAccel.getCLDevices().get(0));
                h = new GroupedGemmHarness(backend, LlmKernels.open(backend));
            }
        } catch (Throwable e) {
            backend = null;
        }
        Assume.assumeTrue("no OpenCL GPU", backend != null);
    }

    @After
    public void tearDown() {
        if (h != null) {
            h.close();
        }
        if (backend != null) {
            backend.close();
        }
        if (hwAccel != null) {
            try {
                hwAccel.shutdown();
            } catch (Throwable ignored) {
                // nothing left to release
            }
        }
    }
}

package com.eonmux.neuralfree.gguf.ops.moe;

import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.cuda.CudaComputeBackend;
import com.eonmux.neuralfree.acceleration.kernels.Availability;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;

/** {@link MoeGroupedPathsParity} on a CUDA device (the scalar grouped GEMM). Skips without a CUDA device. */
public class MoeGroupedPathsCudaTest extends MoeGroupedPathsParity {

    private HWAcceleration hwAccel;
    private CudaComputeBackend backend;

    @Before
    public void setUp() {
        HWAcceleration.enableCuda = true;
        HWAcceleration hw = null;
        try {
            hw = new HWAcceleration();
            Assume.assumeTrue("no CUDA device", !hw.getCUDevices().isEmpty());
            CudaComputeBackend b = new CudaComputeBackend(hw.getCUDevices().get(0));
            Availability availability = LlmKernels.availability(b);
            Assume.assumeTrue("LLM kernels unavailable: " + availability.reason(), availability.available());
            hwAccel = hw;
            backend = b;
            h = new GroupedGemmHarness(b, LlmKernels.open(b));
        } catch (Throwable t) {
            if (hw != null && hwAccel == null) {
                try {
                    hw.shutdown();
                } catch (Throwable ignored) {
                    // the device never opened
                }
            }
            Assume.assumeTrue("CUDA init failed: " + t.getMessage(), false);
        }
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
            hwAccel.shutdown();
        }
    }
}

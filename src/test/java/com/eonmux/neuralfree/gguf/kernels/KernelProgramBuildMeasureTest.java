package com.eonmux.neuralfree.gguf.kernels;

import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.cuda.CudaComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ocl.OclComputeBackend;
import com.eonmux.neuralfree.acceleration.cuda.CUDevice;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.EnumSet;
import java.util.List;
import java.util.Locale;

/**
 * Opt-in wall-clock timing of the first {@link LlmKernels#open} per backend in a fresh process
 * ({@code -Dneuralfree.test.kernels.buildtime=true}): the OpenCL program build the device open pays
 * and the CUDA module loads. Run one backend set per invocation so the open is the process's first,
 * and compare equal-state runs (the OpenCL compiler caches binaries by source, so a repeated source
 * builds warm; the numbers matter cold). Asserts nothing; the numbers bound the model-load cost of
 * the kernel set.
 */
@RunWith(Parameterized.class)
public class KernelProgramBuildMeasureTest {

    /** System property that turns the measurement on. */
    public static final String MEASURE_PROPERTY = "neuralfree.test.kernels.buildtime";

    @Parameterized.Parameters(name = "{0}")
    public static List<Object[]> backends() {
        return Arrays.asList(new Object[]{"opencl"}, new Object[]{"cuda"});
    }

    private final String api;
    private HWAcceleration hwAccel;
    private ComputeBackend backend;
    private boolean previousCuda;

    public KernelProgramBuildMeasureTest(String api) {
        this.api = api;
    }

    @Before
    public void openDevice() {
        Assume.assumeTrue("set " + MEASURE_PROPERTY + "=true to run", Boolean.getBoolean(MEASURE_PROPERTY));
        previousCuda = HWAcceleration.enableCuda;
        try {
            HWAcceleration.enableCuda = api.equals("cuda");
            HWAcceleration accel = new HWAcceleration(EnumSet.of(api.equals("cuda")
                    ? ComputeBackend.BackendType.CUDA : ComputeBackend.BackendType.OPENCL));
            hwAccel = accel;
            if (api.equals("cuda")) {
                Assume.assumeFalse("no CUDA device", accel.getCUDevices().isEmpty());
                backend = new CudaComputeBackend(accel.getCUDevices().get(0));
            } else {
                Assume.assumeFalse("no OpenCL device", accel.getCLDevices().isEmpty());
                backend = new OclComputeBackend(accel.getCLDevices().get(0));
            }
        } catch (Throwable e) {
            closeDevice();
            Assume.assumeTrue("no usable " + api + " GPU: " + e.getMessage(), false);
        }
    }

    @After
    public void closeDevice() {
        if (backend != null) {
            try {
                backend.close();
            } catch (RuntimeException ignored) {
            }
            backend = null;
        }
        if (hwAccel != null) {
            hwAccel.shutdown();
            hwAccel = null;
        }
        HWAcceleration.enableCuda = previousCuda;
    }

    @Test
    public void firstOpenWallClock() {
        long start = System.nanoTime();
        LlmKernels kernels = LlmKernels.open(backend);
        long openNanos = System.nanoTime() - start;
        List<String> names = new ArrayList<>(kernels.kernelNames());
        System.out.println(String.format(Locale.ROOT,
                "%s open %.3f s, %d kernels, matvec local %d", api, openNanos / 1e9, names.size(),
                kernels.matvecLocal()));
    }
}

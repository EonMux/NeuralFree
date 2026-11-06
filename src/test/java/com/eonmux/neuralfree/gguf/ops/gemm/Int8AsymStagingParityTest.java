package com.eonmux.neuralfree.gguf.ops.gemm;

import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.acceleration.backend.cuda.CudaComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ocl.OclComputeBackend;
import com.eonmux.neuralfree.acceleration.cuda.CUDevice;
import com.eonmux.neuralfree.acceleration.kernels.Availability;
import com.eonmux.neuralfree.acceleration.ocl.CLDevice;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.quant.ActFakeQuant;
import com.eonmux.neuralfree.gguf.quant.ActQuantScheme;
import com.eonmux.neuralfree.gguf.quant.CpuBlockDecoders;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.assertTrue;

/**
 * The asymmetric static int8 lane ({@link ActQuantScheme#INT8_STATIC_ASYM}): the staging kernel applies the
 * checkpoint's zero point to the int8 codes and records it, and {@code I8_B128_MatMul_MMQ} subtracts the zero-point
 * term from every stage dot through the weight code sums it reduces, so the product of the checkpoint's asymmetric
 * int8 activations with the int8 weight codes runs in int8 rather than from fake-quantized floats. The result must
 * agree with {@link ActFakeQuant} within fp32 accumulation rounding.
 */
public class Int8AsymStagingParityTest {

    private static final int M = 70;
    private static final int K = 512;
    private static final int N = 21;
    private static final float STATIC_SCALE = 0.25f;
    private static final int ZERO_POINT = 17;

    private HWAcceleration hwAccel;
    private ComputeBackend backend;
    private ComputeQueue queue;
    private LlmKernels kernels;

    @Before
    public void setUp() {
        hwAccel = null;
        backend = null;
    }

    @After
    public void tearDown() {
        if (backend != null) {
            backend.close();
        }
        if (hwAccel != null) {
            hwAccel.shutdown();
        }
    }

    @Test
    public void openClAsymStagingMatchesTheFakeQuantizedDotProduct() throws Exception {
        openOpenCl();
        check(MmqConfig.PACKED_DOT);
    }

    @Test
    public void cudaAsymStagingMatchesTheFakeQuantizedDotProduct() throws Exception {
        openCuda();
        check(MmqConfig.forDevice(backend.backendType(),
                backend.capabilities().getComputeCapability()));
    }

    private void openOpenCl() {
        HWAcceleration hw = null;
        try {
            hw = new HWAcceleration();
            Assume.assumeTrue("no OpenCL device", !hw.getCLDevices().isEmpty());
            OclComputeBackend b = new OclComputeBackend(hw.getCLDevices().get(0));
            Availability availability = LlmKernels.availability(b);
            Assume.assumeTrue("LLM kernels unavailable: " + availability.reason(), availability.available());
            hwAccel = hw;
            backend = b;
            queue = b.computeQueue();
            kernels = LlmKernels.open(b);
        } catch (Throwable t) {
            if (hw != null) {
                try {
                    hw.shutdown();
                } catch (Throwable ignored) {
                }
            }
            Assume.assumeTrue("OpenCL init failed: " + t.getMessage(), false);
        }
    }

    private void openCuda() {
        HWAcceleration.enableCuda = true;
        HWAcceleration hw = null;
        try {
            hw = new HWAcceleration();
            Assume.assumeTrue("no CUDA device", !hw.getCUDevices().isEmpty());
            CUDevice device = hw.getCUDevices().get(0);
            CudaComputeBackend b = new CudaComputeBackend(device);
            Availability availability = LlmKernels.availability(b);
            Assume.assumeTrue("LLM kernels unavailable: " + availability.reason(), availability.available());
            hwAccel = hw;
            backend = b;
            queue = b.computeQueue();
            kernels = LlmKernels.open(b);
        } catch (Throwable t) {
            if (hw != null) {
                try {
                    hw.shutdown();
                } catch (Throwable ignored) {
                }
            }
            Assume.assumeTrue("CUDA init failed: " + t.getMessage(), false);
        }
    }

    private void check(MmqConfig config) throws Exception {
        String stageKernel = GemmKernels.int8ActivationStaging(ActQuantScheme.INT8_STATIC_ASYM);
        Assume.assumeTrue(ActQuantScheme.INT8_STATIC_ASYM + " has no direct staging kernel", stageKernel != null);
        Assume.assumeTrue(stageKernel + " is unavailable", kernels.has(stageKernel));
        MmqGemmPlan plan = MmqGemmPlan.contiguous(config, GGMLType.I8_B128, M, K, N);
        byte[] weight = OpenClMmqGemmParityTest.buildWeight(GGMLType.I8_B128, M, K);
        float[] x = OpenClMmqGemmParityTest.buildX(N, K);
        float[] w = CpuBlockDecoders.dequantize(weight, GGMLType.I8_B128, (long) M * K);
        float[] fake = x.clone();
        ActFakeQuant.apply(ActQuantScheme.INT8_STATIC_ASYM, STATIC_SCALE, ZERO_POINT, 0.0f, fake, 0, N, K);

        int tk = config.stageDepth();
        int local = kernels.matvecLocal();
        GpuBuffer wbuf = backend.allocateBuffer(plan.weightBytes(), true);
        GpuBuffer xbuf = backend.allocateBuffer((long) x.length * Float.BYTES, true);
        GpuBuffer qbuf = backend.allocateBuffer(plan.q8ElementsRequired() * Integer.BYTES);
        GpuBuffer ybuf = backend.allocateBuffer(plan.yElementsRequired() * Float.BYTES);
        try {
            try (GpuEvent wv = queue.enqueueWriteBuffer(wbuf, 0, weight.length, weight);
                 GpuEvent xv = queue.enqueueWriteBuffer(xbuf, 0, (long) x.length * Float.BYTES, x, wv)) {
                long records = (long) plan.n() * config.rowRecords(plan.k());
                Object[] stageArgs = {plan.n(), plan.k(), tk, plan.k(), 0, STATIC_SCALE, ZERO_POINT, xbuf, qbuf};
                queue.enqueueKernelNoEvent(kernels.get(stageKernel), new long[]{roundUp(records, local)},
                        new long[]{local}, stageArgs, xv);
                queue.enqueueKernelNoEvent(kernels.get(plan.kernelName()), plan.globalWorkSize(),
                        plan.localWorkSize(), plan.args(wbuf, qbuf, ybuf));
                float[] out = new float[(int) plan.yElementsRequired()];
                try (GpuEvent read = queue.enqueueReadBuffer(ybuf, 0, (long) out.length * Float.BYTES, out)) {
                    read.awaitCompletion();
                }
                for (int t = 0; t < N; t++) {
                    for (int r = 0; r < M; r++) {
                        double acc = 0.0;
                        double terms = 0.0;
                        for (int k = 0; k < K; k++) {
                            final double term = (double) w[r * K + k] * fake[t * K + k];
                            acc += term;
                            terms += Math.abs(term);
                        }
                        double tol = 1e-5 * terms + 1e-3;
                        assertTrue("token " + t + " row " + r + ": staged kernel " + out[t * M + r]
                                        + " expected " + acc + " (|terms| " + terms + ")",
                                Math.abs(out[t * M + r] - acc) <= tol);
                    }
                }
            }
        } finally {
            wbuf.close();
            xbuf.close();
            qbuf.close();
            ybuf.close();
        }
    }

    private static long roundUp(long n, long m) {
        return (n + m - 1) / m * m;
    }
}

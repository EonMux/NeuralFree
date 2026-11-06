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
 * The checkpoint-scheme int8 staging behind the int8 MMQ GEMM: with an {@link ActQuantScheme#INT8_STATIC},
 * {@link ActQuantScheme#INT8_TOKEN} or {@link ActQuantScheme#INT8_GROUP128} weight, the staging kernel applies the
 * scheme's own scale and codes, and the {@code I8_B128_MatMul_MMQ} kernel then computes the product of the
 * checkpoint's int8 activation codes with the int8 weight codes. The result must agree with the fake-quantized
 * activations ({@link ActFakeQuant}, the host twin) multiplied by the decoded weight, within one activation code
 * step per element plus fp32 accumulation rounding.
 */
public class Int8ActStagingParityTest {

    private static final int M = 70;   /* two 64-row tiles, the second partial */
    private static final int K = 512;  /* four I8_B128 blocks per row */
    private static final int N = 21;   /* two token tiles, the second partial */
    private static final float STATIC_SCALE = 0.25f;

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
    public void openClStagingMatchesTheFakeQuantizedDotProduct() throws Exception {
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
        checkSchemes(MmqConfig.PACKED_DOT);
    }

    @Test
    public void cudaStagingMatchesTheFakeQuantizedDotProduct() throws Exception {
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
        checkSchemes(MmqConfig.forDevice(backend.backendType(),
                backend.capabilities().getComputeCapability()));
    }

    private void checkSchemes(MmqConfig config) throws Exception {
        for (ActQuantScheme scheme : new ActQuantScheme[]{ActQuantScheme.INT8_STATIC,
                ActQuantScheme.INT8_TOKEN, ActQuantScheme.INT8_GROUP128}) {
            checkScheme(config, scheme);
        }
    }

    private void checkScheme(MmqConfig config, ActQuantScheme scheme) throws Exception {
        String stageKernel = GemmKernels.int8ActivationStaging(scheme);
        Assume.assumeTrue(scheme + " has no direct staging kernel", stageKernel != null);
        Assume.assumeTrue(stageKernel + " is unavailable", kernels.has(stageKernel));
        MmqGemmPlan plan = MmqGemmPlan.contiguous(config, GGMLType.I8_B128, M, K, N);
        byte[] weight = OpenClMmqGemmParityTest.buildWeight(GGMLType.I8_B128, M, K);
        float[] x = OpenClMmqGemmParityTest.buildX(N, K);
        float[] w = CpuBlockDecoders.dequantize(weight, GGMLType.I8_B128, (long) M * K);
        float[] fake = x.clone();
        ActFakeQuant.apply(scheme, staticScale(scheme), 0, 0.0f, fake, 0, N, K);

        float[] got = launch(config, stageKernel, scheme, plan, weight, x);
        for (int t = 0; t < N; t++) {
            double[] codeStep = codeStepScale(scheme, x, t * K);
            for (int r = 0; r < M; r++) {
                double want = 0.0;
                double terms = 0.0;
                double bound = 0.0;
                for (int k = 0; k < K; k++) {
                    want += (double) w[r * K + k] * fake[t * K + k];
                    terms += Math.abs((double) w[r * K + k] * fake[t * K + k]);
                    bound += Math.abs((double) w[r * K + k]) * codeStep[k];
                }
                double tol = bound + 1e-4 * terms + 1e-3;
                assertTrue(scheme + " token " + t + " row " + r + ": staged kernel " + got[t * M + r]
                                + " expected " + want + " (bound " + tol + ")",
                        Math.abs(got[t * M + r] - want) <= tol);
            }
        }
    }

    private static float staticScale(ActQuantScheme scheme) {
        return scheme.isStatic() ? STATIC_SCALE : 0.0f;
    }

    /**
     * One activation code step of each element: the scheme's dequant multiplier of the run holding it, i.e.
     * {@code input_scale}, {@code amax / 127} of the token row, or {@code max(amax, 1e-10f) / 127} per 128-element
     * run.
     */
    private static double[] codeStepScale(ActQuantScheme scheme, float[] x, int base) {
        double[] d = new double[K];
        if (scheme == ActQuantScheme.INT8_STATIC) {
            for (int k = 0; k < K; k++) {
                d[k] = STATIC_SCALE;
            }
            return d;
        }
        if (scheme == ActQuantScheme.INT8_TOKEN) {
            float amax = 0.0f;
            for (int k = 0; k < K; k++) {
                amax = Math.max(amax, Math.abs(x[base + k]));
            }
            for (int k = 0; k < K; k++) {
                d[k] = amax / 127.0;
            }
            return d;
        }
        for (int c = 0; c < K; c += 128) {
            float amax = 0.0f;
            for (int k = c; k < c + 128; k++) {
                amax = Math.max(amax, Math.abs(x[base + k]));
            }
            double s = (double) Math.max(amax, 1e-10f) / 127.0;
            for (int k = c; k < c + 128; k++) {
                d[k] = s;
            }
        }
        return d;
    }

    private float[] launch(MmqConfig config, String stageKernel, ActQuantScheme scheme, MmqGemmPlan plan,
                           byte[] weight, float[] x) throws InterruptedException {
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
                Object[] stageArgs;
                long global;
                if (scheme == ActQuantScheme.INT8_STATIC) {
                    stageArgs = new Object[]{plan.n(), plan.k(), tk, plan.k(), 0, STATIC_SCALE, xbuf, qbuf};
                    global = records;
                } else if (scheme == ActQuantScheme.INT8_TOKEN) {
                    stageArgs = new Object[]{plan.n(), plan.k(), tk, plan.k(), 0, xbuf, qbuf};
                    global = (long) plan.n() * local;
                } else {
                    long runs = (long) plan.n() * (plan.k() / 128);
                    stageArgs = new Object[]{plan.n(), plan.k(), tk, plan.k(), 0, xbuf, qbuf};
                    global = runs * local;
                }
                queue.enqueueKernelNoEvent(kernels.get(stageKernel), new long[]{roundUp(global, local)},
                        new long[]{local}, stageArgs, xv);
                queue.enqueueKernelNoEvent(kernels.get(plan.kernelName()), plan.globalWorkSize(),
                        plan.localWorkSize(), plan.args(wbuf, qbuf, ybuf));
                float[] out = new float[(int) plan.yElementsRequired()];
                try (GpuEvent read = queue.enqueueReadBuffer(ybuf, 0, (long) out.length * Float.BYTES, out)) {
                    read.awaitCompletion();
                }
                return out;
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

package com.eonmux.neuralfree.gguf.ops.gemm;

import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.acceleration.backend.cuda.CudaComputeBackend;
import com.eonmux.neuralfree.acceleration.cuda.CUDevice;
import com.eonmux.neuralfree.acceleration.kernels.Availability;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.quant.ActFakeQuant;
import com.eonmux.neuralfree.gguf.quant.ActQuantScheme;
import com.eonmux.neuralfree.gguf.quant.CpuBlockDecoders;
import com.eonmux.neuralfree.utils.numeric.Fp8Codec;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import static org.junit.Assert.assertTrue;

/**
 * The W4A8 lane: a 4-bit weight (MXFP4, NVFP4, IQ4_NL, IQ4_XS) whose codes are mapped to E4M3 codes and multiplied
 * by the staged E4M3 activation codes of an E4M3-stageable activation scheme ({@code <T>_MatMul_W4A8}). The MXFP4
 * and NVFP4 codebooks are E2M1 values, which E4M3 represents exactly, so their lane matches the fake-quantized
 * reference up to the fp32 accumulation; the non-linear IQ4 codebooks and the NVFP4 per-16 weight scales need one
 * E4M3 rounding per weight element, for which the comparison states its bound.
 */
public class W4a8GemmParityTest {

    private static final int M = 70;
    private static final int K = 512;
    private static final int N = 21;
    private static final float STATIC_SCALE = 0.25f;

    private HWAcceleration hwAccel;
    private CudaComputeBackend backend;
    private ComputeQueue queue;
    private LlmKernels kernels;

    @Before
    public void setUp() {
        HWAcceleration.enableCuda = true;
        HWAcceleration hw = null;
        try {
            hw = new HWAcceleration();
            Assume.assumeTrue("no CUDA device", !hw.getCUDevices().isEmpty());
            CUDevice device = hw.getCUDevices().get(0);
            CudaComputeBackend b = new CudaComputeBackend(device);
            Availability availability = LlmKernels.availability(b);
            Assume.assumeTrue("LLM kernels unavailable: " + availability.reason(), availability.available());
            Assume.assumeTrue("the E4M3 GEMM needs sm_89+", b.capabilities().getComputeCapability() >= 89);
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
    public void mxFp4WeightsRunTheE4m3Gemm() throws Exception {
        check(GGMLType.MXFP4, 0.0);
    }

    @Test
    public void iq4WeightsRunTheE4m3Gemm() throws Exception {
        check(GGMLType.IQ4_NL, 0.06);
        check(GGMLType.IQ4_XS, 0.06);
    }

    @Test
    public void nvFp4WeightsRunTheE4m3Gemm() throws Exception {
        check(GGMLType.NVFP4, 0.08);
    }

    /**
     * The lane equals the fake-quantized reference within {@code weightTolerance} of every term (the E4M3 rounding
     * of the weight code the lane performs) plus fp32 accumulation.
     */
    private void check(GGMLType type, double weightTolerance) throws Exception {
        MmqConfig config = MmqConfig.forDevice(backend.backendType(),
                backend.capabilities().getComputeCapability());
        String gemm = GemmKernels.matmulW4a8(type);
        Assume.assumeTrue(gemm + " is unavailable", kernels.has(gemm));
        for (ActQuantScheme scheme : new ActQuantScheme[]{ActQuantScheme.FP8_STATIC, ActQuantScheme.MXFP8,
                ActQuantScheme.MXFP4}) {
            String stageKernel = GemmKernels.fp8ActivationStaging(scheme);
            Assume.assumeTrue(stageKernel + " is unavailable", kernels.has(stageKernel));
            int tk = config.stageDepth();
            int units = config.recordUnits();
            int tileTokens = config.tileTokens();
            long qUnits = (long) N * config.rowRecords(K) * units;
            byte[] weight = OpenClMmqGemmParityTest.buildWeight(type, M, K);
            float[] x = OpenClMmqGemmParityTest.buildX(N, K);
            float[] w = CpuBlockDecoders.dequantize(weight, type, (long) M * K);
            float[] fake = x.clone();
            ActFakeQuant.apply(scheme, scheme.isStatic() ? STATIC_SCALE : 0.0f, 0, 0.0f, fake, 0, N, K);

            GpuBuffer wbuf = backend.allocateBuffer((long) weight.length, true);
            GpuBuffer xbuf = backend.allocateBuffer((long) x.length * Float.BYTES, true);
            GpuBuffer qbuf = backend.allocateBuffer(qUnits * Integer.BYTES);
            GpuBuffer ybuf = backend.allocateBuffer((long) N * M * Float.BYTES);
            try {
                try (GpuEvent wv = queue.enqueueWriteBuffer(wbuf, 0, weight.length, weight);
                     GpuEvent xv = queue.enqueueWriteBuffer(xbuf, 0, (long) x.length * Float.BYTES, x, wv)) {
                    stage(stageKernel, scheme, tk, xbuf, qbuf, xv);
                    Object[] args = {M, K, N, config.rowRecords(K), 0, M, 0, wbuf, qbuf, ybuf};
                    long[] global = {((long) (M + 63) / 64) * 256, (N + tileTokens - 1) / tileTokens};
                    queue.enqueueKernelNoEvent(kernels.get(gemm), global, new long[]{256, 1}, args);
                    float[] out = new float[N * M];
                    try (GpuEvent read = queue.enqueueReadBuffer(ybuf, 0, (long) out.length * Float.BYTES, out)) {
                        read.awaitCompletion();
                    }
                    for (int t = 0; t < N; t++) {
                        for (int r = 0; r < M; r++) {
                            double acc = 0.0;
                            double bound = 0.0;
                            for (int k = 0; k < K; k++) {
                                final double term = (double) w[r * K + k] * fake[t * K + k];
                                acc += term;
                                bound += Math.abs(term) * weightTolerance;
                            }
                            double tol = bound + 1e-4 * Math.abs(acc) + 1e-3;
                            assertTrue(type + " " + scheme + " token " + t + " row " + r + ": kernel "
                                            + out[t * M + r] + " expected " + acc + " (bound " + tol + ")",
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
    }

    private void stage(String kernel, ActQuantScheme scheme, int tk, GpuBuffer x, GpuBuffer q8, GpuEvent after) {
        int local = kernels.matvecLocal();
        long records = (long) N * (K / tk);
        Object[] args;
        long global;
        if (scheme.isStatic()) {
            args = new Object[]{N, K, tk, K, 0, STATIC_SCALE, x, q8};
            global = records;
        } else {
            args = new Object[]{N, K, tk, K, 0, x, q8};
            global = records;
        }
        queue.enqueueKernelNoEvent(kernels.get(kernel), new long[]{roundUp(global, local)}, new long[]{local},
                args, after);
    }

    private static long roundUp(long n, long m) {
        return (n + m - 1) / m * m;
    }
}

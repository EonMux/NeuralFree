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
import com.eonmux.neuralfree.gguf.quant.ActQuantScheme;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;

import java.util.Locale;

/**
 * Opt-in timing of the E4M3 lanes against the tiled GEMM over fake-quantized activations (environment
 * {@code NEURALFREE_BENCH_FP8=1}): the FP8 W8A8 lane ({@code <T>_MatMul_MMQ} over staged E4M3 codes) and the W4A8
 * lane ({@code <T>_MatMul_W4A8}) against {@code <T>_MatMul_Tiled} at prefill and batched-decode token counts.
 * Asserts nothing; prints the mean kernel time and the achieved TFLOP/s of each form.
 */
public class Fp8GemmBenchmarkTest {

    private static final String SWITCH = "NEURALFREE_BENCH_FP8";
    private static final int WARMUP = 3;
    private static final int RUNS = 20;
    private static final int M = 4096;
    private static final int K = 4096;

    private HWAcceleration hwAccel;
    private CudaComputeBackend backend;
    private ComputeQueue queue;
    private LlmKernels kernels;

    @Before
    public void setUp() {
        Assume.assumeTrue("set " + SWITCH + "=1 to run", "1".equals(System.getenv(SWITCH)));
        HWAcceleration.enableCuda = true;
        HWAcceleration hw = new HWAcceleration();
        if (hw.getCUDevices().isEmpty()) {
            hw.shutdown();
            Assume.assumeTrue("no CUDA device", false);
        }
        CUDevice device = hw.getCUDevices().get(0);
        CudaComputeBackend b = new CudaComputeBackend(device);
        Availability availability = LlmKernels.availability(b);
        if (!availability.available()) {
            b.close();
            hw.shutdown();
            Assume.assumeTrue("LLM kernels unavailable: " + availability.reason(), false);
        }
        hwAccel = hw;
        backend = b;
        queue = b.computeQueue();
        kernels = LlmKernels.open(b);
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
    public void fp8AndW4a8LanesAgainstTheTiledGemm() throws Exception {
        MmqConfig config = MmqConfig.forDevice(backend.backendType(),
                backend.capabilities().getComputeCapability());
        for (int n : new int[]{16, 512}) {
            bench(GGMLType.F8_E4M3_B128, ActQuantScheme.FP8_STATIC, n, config);
        }
        for (int n : new int[]{16, 512}) {
            bench(GGMLType.MXFP4, ActQuantScheme.FP8_STATIC, n, config);
        }
    }

    private void bench(GGMLType type, ActQuantScheme scheme, int n, MmqConfig config) throws Exception {
        int tk = config.stageDepth();
        int units = config.recordUnits();
        long qUnits = (long) n * config.rowRecords(K) * units;
        byte[] weight = OpenClMmqGemmParityTest.buildWeight(type, M, K);
        float[] x = new float[n * K];
        for (int i = 0; i < x.length; i++) {
            x[i] = (float) Math.sin(i * 0.017) * 2.0f;
        }
        String stageKernel = GemmKernels.fp8ActivationStaging(scheme);
        String lane = GemmKernels.isFp8(type) ? GemmKernels.matmulMmq(type) : GemmKernels.matmulW4a8(type);
        String tiled = GemmKernels.matmulTiled(type);
        GpuBuffer wbuf = backend.allocateBuffer((long) weight.length, true);
        GpuBuffer xbuf = backend.allocateBuffer((long) x.length * Float.BYTES, true);
        GpuBuffer qbuf = backend.allocateBuffer(qUnits * Integer.BYTES);
        GpuBuffer ybuf = backend.allocateBuffer((long) n * M * Float.BYTES);
        try {
            try (GpuEvent wv = queue.enqueueWriteBuffer(wbuf, 0, weight.length, weight);
                 GpuEvent xv = queue.enqueueWriteBuffer(xbuf, 0, (long) x.length * Float.BYTES, x, wv)) {
                xv.awaitCompletion();
            }
            int local = kernels.matvecLocal();
            long records = (long) n * (K / tk);
            Object[] stageArgs = {n, K, tk, K, 0, 0.25f, xbuf, qbuf};
            Object[] laneArgs = {M, K, n, config.rowRecords(K), 0, M, 0, wbuf, qbuf, ybuf};
            long[] laneGlobal = {((long) (M + 63) / 64) * 256, (n + config.tileTokens() - 1) / config.tileTokens()};
            Object[] tiledArgs = {M, K, n, K, 0, M, 0, wbuf, xbuf, ybuf};
            long[] tiledGlobal = {((long) (M + 63) / 64) * 256, (n + 31) / 32};

            double staged = time(() -> {
                queue.enqueueKernelNoEvent(kernels.get(stageKernel), new long[]{roundUp(records, local)},
                        new long[]{local}, stageArgs);
                queue.enqueueKernelNoEvent(kernels.get(lane), laneGlobal, new long[]{256, 1}, laneArgs);
            });
            double tiledMs = time(() -> queue.enqueueKernelNoEvent(kernels.get(tiled), tiledGlobal,
                    new long[]{256, 1}, tiledArgs));
            System.out.printf(Locale.ROOT, "%s n=%d: %s+%s %.3f ms (%.1f TFLOP/s); %s %.3f ms (%.1f TFLOP/s);"
                            + " speedup %.2fx%n", type, n, stageKernel, lane, staged,
                    2.0 * M * K * n / staged * 1e-9, tiled, tiledMs, 2.0 * M * K * n / tiledMs * 1e-9,
                    tiledMs / staged);
        } finally {
            wbuf.close();
            xbuf.close();
            qbuf.close();
            ybuf.close();
        }
    }

    private interface Launch {
        void run();
    }

    private double time(Launch launch) throws Exception {
        for (int i = 0; i < WARMUP; i++) {
            launch.run();
        }
        long start = System.nanoTime();
        for (int i = 0; i < RUNS; i++) {
            launch.run();
        }
        try (GpuEvent done = queue.enqueueMarker()) {
            done.awaitCompletion();
        }
        return (System.nanoTime() - start) / 1e6 / RUNS;
    }

    private static long roundUp(long n, long m) {
        return (n + m - 1) / m * m;
    }
}

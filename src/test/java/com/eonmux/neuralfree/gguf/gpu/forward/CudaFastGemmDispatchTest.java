package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.acceleration.backend.cuda.CudaComputeBackend;
import com.eonmux.neuralfree.acceleration.cuda.CUDevice;
import com.eonmux.neuralfree.acceleration.kernels.Availability;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuPlaceholders;
import com.eonmux.neuralfree.gguf.ops.gemm.MmqMode;
import com.eonmux.neuralfree.gguf.ops.gemm.OpenClMmqGemmParityTest;
import com.eonmux.neuralfree.gguf.quant.CpuBlockDecoders;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import static org.junit.Assert.assertTrue;

/**
 * {@link GpuOps#matmul} through the fast GEMM lanes on the CUDA device against the batched matvec of
 * the same projection: the int8 MMQ GEMM under the device's tile config, the F16 MMA GEMM, its
 * split-K form on a tall-skinny launch, and {@code neuralfree.gemm.mmq=off}, which keeps the exact
 * tiled GEMM. Every comparison allows the rounding of the lane taken.
 */
public class CudaFastGemmDispatchTest {

    private static final int UBATCH = 64;

    private HWAcceleration hwAccel;
    private CudaComputeBackend backend;
    private ComputeQueue queue;
    private GpuScratch scratch;
    private GpuPlaceholders placeholders;
    private GpuOps ops;

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
            Assume.assumeTrue("the fast lanes need sm_80+", b.capabilities().getComputeCapability() >= 80);
            hwAccel = hw;
            backend = b;
            queue = b.computeQueue();
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

    private void openOps() {
        closeOps();
        scratch = GpuScratch.builder(backend, UBATCH).build();
        placeholders = GpuPlaceholders.allocate(backend, queue);
        ops = new GpuOps(queue, LlmKernels.open(backend), scratch, placeholders);
    }

    private void closeOps() {
        if (ops != null) {
            ops.close();
        }
        if (placeholders != null) {
            placeholders.close();
        }
        if (scratch != null) {
            scratch.close();
        }
        ops = null;
        placeholders = null;
        scratch = null;
    }

    @After
    public void tearDown() {
        closeOps();
        System.clearProperty(MmqMode.PROPERTY);
        if (backend != null) {
            backend.close();
        }
        if (hwAccel != null) {
            hwAccel.shutdown();
        }
    }

    @Test
    public void mmqLaneMatchesTheBatchedMatvecWithinTheActivationCodeStep() throws Exception {
        openOps();
        for (GGMLType type : new GGMLType[]{GGMLType.Q4_K, GGMLType.Q6_K, GGMLType.Q8_0}) {
            int rows = 200;
            int cols = 512;
            int n = 45;
            byte[] bytes = OpenClMmqGemmParityTest.buildWeight(type, rows, cols);
            float[] w = CpuBlockDecoders.dequantize(bytes, type, (long) rows * cols);
            float[] x = OpenClMmqGemmParityTest.buildX(n, cols);
            float[] fast = project(type, bytes, rows, cols, x, n, true);
            float[] exact = project(type, bytes, rows, cols, x, n, false);
            assertWithinCodeStep(type + " MMQ lane", fast, exact, w, x, rows, cols, n);
            /* the wave-fill split-K regime of a staged launch of fewer tokens than the 64-token tile */
            int deepCols = 2048;
            byte[] deepBytes = OpenClMmqGemmParityTest.buildWeight(type, rows, deepCols);
            float[] dw = CpuBlockDecoders.dequantize(deepBytes, type, (long) rows * deepCols);
            float[] dx = OpenClMmqGemmParityTest.buildX(n, deepCols);
            float[] deepFast = project(type, deepBytes, rows, deepCols, dx, n, true);
            float[] deepExact = project(type, deepBytes, rows, deepCols, dx, n, false);
            assertWithinCodeStep(type + " MMQ lane over K=2048", deepFast, deepExact, dw, dx, rows, deepCols, n);
        }
    }

    @Test
    public void mmqOffKeepsTheExactTiledGemm() throws Exception {
        System.setProperty(MmqMode.PROPERTY, "off");
        openOps();
        int rows = 200;
        int cols = 512;
        int n = 45;
        byte[] bytes = OpenClMmqGemmParityTest.buildWeight(GGMLType.Q4_K, rows, cols);
        float[] w = CpuBlockDecoders.dequantize(bytes, GGMLType.Q4_K, (long) rows * cols);
        float[] x = OpenClMmqGemmParityTest.buildX(n, cols);
        float[] tiled = project(GGMLType.Q4_K, bytes, rows, cols, x, n, true);
        float[] exact = project(GGMLType.Q4_K, bytes, rows, cols, x, n, false);
        assertClose("Q4_K with the MMQ lane off", tiled, exact, w, x, rows, cols, n, 1e-5);
    }

    @Test
    public void mmaAndSplitKLanesMatchTheBatchedMatvecWithinFp16Rounding() throws Exception {
        openOps();
        int[][] shapes = {{200, 512, 45}, {128, 4096, 8}};   /* plain grid; split-K tall-skinny grid */
        for (int[] s : shapes) {
            float[] w = denseValues(s[0] * s[1], 0x51ED);
            float[] x = denseValues(s[2] * s[1], 0xC0DE);
            byte[] bytes = f16Bytes(w);
            float[] fast = project(GGMLType.F16, bytes, s[0], s[1], x, s[2], true);
            float[] exact = project(GGMLType.F16, bytes, s[0], s[1], x, s[2], false);
            assertClose("F16 M=" + s[0] + " K=" + s[1] + " N=" + s[2], fast, exact, w, x, s[0], s[1], s[2], 1e-5);
        }
    }

    /** {@code matmul} (fast lanes) or {@code matvec} (batched matvec) of one weight over {@code n} tokens. */
    private float[] project(GGMLType type, byte[] bytes, int rows, int cols, float[] x, int n, boolean gemm)
            throws InterruptedException {
        GpuBuffer wbuf = backend.allocateBuffer(bytes.length, true);
        GpuBuffer in = backend.allocateBuffer((long) x.length * Float.BYTES, true);
        GpuBuffer out = backend.allocateBuffer((long) n * rows * Float.BYTES);
        try {
            try (GpuEvent wv = queue.enqueueWriteBuffer(wbuf, 0, bytes.length, bytes);
                 GpuEvent xv = queue.enqueueWriteBuffer(in, 0, (long) x.length * Float.BYTES, x, wv)) {
                xv.awaitCompletion();
                GpuWeight weight = GpuWeight.of("blk.0.ffn_down.weight", type, new long[]{cols, rows}, wbuf);
                if (gemm) {
                    ops.matmul(weight, in, 0, out, 0, n);
                } else {
                    ops.matvec(weight, in, 0, out, 0, n);
                }
                float[] y = new float[n * rows];
                try (GpuEvent read = queue.enqueueReadBuffer(out, 0, (long) y.length * Float.BYTES, y)) {
                    read.awaitCompletion();
                }
                return y;
            }
        } finally {
            wbuf.close();
            in.close();
            out.close();
        }
    }

    /**
     * |got - want| <= sum_k |W[r][k]| * d_k + 1e-4 * sum_k |W[r][k] * x[t][k]| + 1e-3 at every output, with
     * d_k = maxabs / 127 of the 32-element activation chunk holding k: one 8-bit code step per element.
     */
    private static void assertWithinCodeStep(String what, float[] got, float[] want, float[] w, float[] x, int rows,
                                             int cols, int n) {
        for (int t = 0; t < n; t++) {
            double[] d = new double[cols];
            for (int c = 0; c < cols; c += 32) {
                float maxabs = 0.0f;
                for (int k = c; k < c + 32; k++) {
                    maxabs = Math.max(maxabs, Math.abs(x[t * cols + k]));
                }
                for (int k = c; k < c + 32; k++) {
                    d[k] = maxabs / 127.0;
                }
            }
            for (int r = 0; r < rows; r++) {
                double bound = 0.0;
                double terms = 0.0;
                for (int k = 0; k < cols; k++) {
                    bound += Math.abs((double) w[r * cols + k]) * d[k];
                    terms += Math.abs((double) w[r * cols + k] * x[t * cols + k]);
                }
                double tol = bound + 1e-4 * terms + 1e-3;
                assertTrue(what + " token " + t + " row " + r + ": " + got[t * rows + r] + " vs "
                        + want[t * rows + r], Math.abs(got[t * rows + r] - want[t * rows + r]) <= tol);
            }
        }
    }

    /** |got - want| <= relative * sum_k |W[r][k] * x[t][k]| + 1e-3 at every output. */
    private static void assertClose(String what, float[] got, float[] want, float[] w, float[] x, int rows, int cols,
                                    int n, double relative) {
        for (int t = 0; t < n; t++) {
            for (int r = 0; r < rows; r++) {
                double terms = 0.0;
                for (int k = 0; k < cols; k++) {
                    terms += Math.abs((double) w[r * cols + k] * x[t * cols + k]);
                }
                double tol = relative * terms + 1e-3;
                assertTrue(what + " token " + t + " row " + r + ": " + got[t * rows + r] + " vs "
                        + want[t * rows + r], Math.abs(got[t * rows + r] - want[t * rows + r]) <= tol);
            }
        }
    }

    /** Deterministic values with at most 7 explicit mantissa bits in [-4, -0.5] and [0.5, 4] (exact in fp16). */
    private static float[] denseValues(int count, int seed) {
        int state = seed | 1;
        float[] v = new float[count];
        for (int i = 0; i < count; i++) {
            state ^= state << 13;
            state ^= state >>> 17;
            state ^= state << 5;
            int bits = state >>> 8;
            float sign = ((bits >>> 10) & 1) == 0 ? 1.0f : -1.0f;
            v[i] = sign * (float) Math.scalb(1.0 + (bits & 0x7F) / 128.0, (bits >>> 7) % 3 - 1);
        }
        return v;
    }

    private static byte[] f16Bytes(float[] v) {
        ByteBuffer b = ByteBuffer.allocate(v.length * 2).order(ByteOrder.LITTLE_ENDIAN);
        for (float f : v) {
            int bits = Float.floatToRawIntBits(f);
            int exp = ((bits >>> 23) & 0xFF) - 127 + 15;
            b.putShort((short) (((bits >>> 16) & 0x8000) | (exp << 10) | ((bits >>> 13) & 0x3FF)));
        }
        return b.array();
    }
}

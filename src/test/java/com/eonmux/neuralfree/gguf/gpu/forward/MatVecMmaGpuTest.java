package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.cuda.CudaComputeBackend;
import com.eonmux.neuralfree.acceleration.kernels.Availability;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.ops.gemm.MatVecMmaPlan;
import com.eonmux.neuralfree.gguf.quant.FpBlockDecoders;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;

import java.util.EnumSet;
import java.util.Locale;
import java.util.Random;

import static org.junit.Assert.assertTrue;

/**
 * The tensor-core multi-column matvec ({@code {T}_MatVecMma{8,16}_Q8}, kernels/llm/matvec_mma.cl) against the
 * counted integer-dot matvec ({@code {T}_MatVecN<n>_Q8}) over the same weights and the same Q8_1 activation rows:
 * both take the exact integer products of every 32-element group and differ only in the float summation order of
 * the scaled group terms, so every output agrees to a float-rounding bound of its row's absolute term sum. Covers
 * every type the kernel serves, every token count 2..16 under both token tiles, a row count that leaves a partial
 * row tile, and every warp count a plan can take. CUDA sm_80+ only; skipped elsewhere.
 */
public class MatVecMmaGpuTest {

    /** Rows: a partial last 32-row tile. Cols: two Q4_K/Q6_K block pairs, sixteen Q8_0 block pairs. */
    private static final int ROWS = 77;
    private static final int COLS = 1024;
    private static final GGMLType[] TYPES = {GGMLType.Q8_0, GGMLType.Q4_K, GGMLType.Q6_K};
    private static final int MAX_TOKENS = 16;

    private HWAcceleration hwAccel;
    private ComputeBackend backend;
    private LlmKernels kernels;
    private int local;

    @Before
    public void openDevice() {
        try {
            HWAcceleration accel = new HWAcceleration(EnumSet.of(ComputeBackend.BackendType.CUDA));
            hwAccel = accel;
            Assume.assumeFalse("no CUDA device", accel.getCUDevices().isEmpty());
            backend = new CudaComputeBackend(accel.getCUDevices().get(0));
            Availability availability = LlmKernels.availability(backend);
            Assume.assumeTrue("LLM kernels unavailable: " + availability.reason(), availability.available());
            kernels = LlmKernels.open(backend);
            local = kernels.matvecLocal();
        } catch (Throwable e) {
            closeDevice();
            Assume.assumeTrue("no usable CUDA GPU: " + e.getMessage(), false);
        }
        Assume.assumeTrue("no tensor-core matvec on this device",
                kernels.has(MatVecMmaPlan.kernelName(GGMLType.Q8_0, 16)));
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
    }

    @Test
    public void everyTokenCountMatchesTheCountedMatvec() throws InterruptedException {
        for (GGMLType type : TYPES) {
            int rowBytes = COLS / type.getBlockSize() * type.getTypeSize();
            byte[] weights = weights(type, ROWS, rowBytes, 0x5EED + type.ordinal());
            float[] x = activations(MAX_TOKENS * COLS);
            GpuBuffer wBuf = backend.allocateBuffer((long) ROWS * rowBytes);
            GpuBuffer xBuf = backend.allocateBuffer((long) x.length * 4);
            GpuBuffer qBuf = backend.allocateBuffer((long) MAX_TOKENS * (COLS / 32) * GpuOps.Q8_1_FLOATS * 4);
            GpuBuffer yRef = backend.allocateBuffer((long) MAX_TOKENS * ROWS * 4);
            GpuBuffer yMma = backend.allocateBuffer((long) MAX_TOKENS * ROWS * 4);
            try {
                backend.computeQueue().enqueueWriteBuffer(wBuf, 0, weights.length, weights).awaitCompletion();
                backend.computeQueue().enqueueWriteBuffer(xBuf, 0, (long) x.length * 4, x).awaitCompletion();
                backend.computeQueue().enqueueKernel(kernels.get(GpuOps.QUANTIZE_Q8_1),
                        new long[]{roundUp((long) MAX_TOKENS * (COLS / 32))}, new long[]{local},
                        new Object[]{MAX_TOKENS * COLS, xBuf, qBuf});
                for (int n = 2; n <= MAX_TOKENS; n++) {
                    float[] ref = run(type.name() + "_MatVecN" + n + "_Q8", new long[]{(long) ROWS * local},
                            new long[]{local}, new Object[]{ROWS, COLS, wBuf, qBuf, yRef}, yRef, n);
                    float[] scale = rowTermScale(type, weights, rowBytes, x, n);
                    for (int tile : new int[]{8, 16}) {
                        String kernel = MatVecMmaPlan.kernelName(type, tile);
                        if (n > tile) {
                            continue;
                        }
                        int tileRows = MatVecMmaPlan.tileRows(type, tile);
                        for (int warps = 1; warps <= MatVecMmaPlan.MAX_WARPS; warps *= 2) {
                            long groups = (ROWS + tileRows - 1) / tileRows;
                            float[] got = run(kernel, new long[]{groups * 32L * warps}, new long[]{32L * warps},
                                    new Object[]{ROWS, COLS, n, wBuf, qBuf, yMma}, yMma, n);
                            compare(kernel + " n=" + n + " warps=" + warps, ref, got, scale);
                        }
                    }
                }
            } finally {
                wBuf.close();
                xBuf.close();
                qBuf.close();
                yRef.close();
                yMma.close();
            }
        }
    }

    private float[] run(String kernel, long[] global, long[] localSize, Object[] args, GpuBuffer out, int n)
            throws InterruptedException {
        float[] poison = new float[MAX_TOKENS * ROWS];
        java.util.Arrays.fill(poison, Float.NaN);
        backend.computeQueue().enqueueWriteBuffer(out, 0, (long) poison.length * 4, poison).awaitCompletion();
        backend.computeQueue().enqueueKernel(kernels.get(kernel), global, localSize, args);
        float[] y = new float[n * ROWS];
        backend.computeQueue().enqueueReadBuffer(out, 0, (long) y.length * 4, y).awaitCompletion();
        return y;
    }

    /**
     * A per-output bound of the summation-order difference: the activation magnitude times the row's absolute
     * weight-byte sum stands in for the absolute term sum (scales are of order 0.01, codes below 128).
     */
    private static float[] rowTermScale(GGMLType type, byte[] w, int rowBytes, float[] x, int n) {
        float[] s = new float[n * ROWS];
        for (int t = 0; t < n; t++) {
            float xm = 0f;
            for (int k = 0; k < COLS; k++) {
                xm = Math.max(xm, Math.abs(x[t * COLS + k]));
            }
            for (int r = 0; r < ROWS; r++) {
                s[t * ROWS + r] = xm * COLS * 0.05f * 128f;
            }
        }
        return s;
    }

    private static void compare(String what, float[] ref, float[] got, float[] scale) {
        for (int i = 0; i < ref.length; i++) {
            assertTrue(what + " output " + i + " is not finite: " + got[i], Float.isFinite(got[i]));
            float tol = 2e-6f * scale[i] + 1e-6f;
            assertTrue(String.format(Locale.ROOT, "%s output %d (token %d row %d): counted %.7g, mma %.7g", what, i,
                    i / ROWS, i % ROWS, ref[i], got[i]), Math.abs(ref[i] - got[i]) <= tol);
        }
    }

    private long roundUp(long n) {
        return (n + local - 1) / local * local;
    }

    /** Random blocks of {@code type} with finite block scales. */
    static byte[] weights(GGMLType type, int rows, int rowBytes, long seed) {
        Random rnd = new Random(seed);
        byte[] data = new byte[rows * rowBytes];
        rnd.nextBytes(data);
        int blockBytes = type.getTypeSize();
        for (int off = 0; off + blockBytes <= data.length; off += blockBytes) {
            switch (type) {
                case Q6_K:
                    putHalf(data, off + 208, 0.01f);
                    break;
                case Q4_K:
                    putHalf(data, off, 0.02f);
                    putHalf(data, off + 2, 0.01f);
                    break;
                default:
                    putHalf(data, off, 0.02f);
                    break;
            }
        }
        return data;
    }

    private static void putHalf(byte[] data, int off, float value) {
        int bits = FpBlockDecoders.fp32ToFp16(value);
        data[off] = (byte) bits;
        data[off + 1] = (byte) (bits >>> 8);
    }

    static float[] activations(int n) {
        Random rnd = new Random(0xC0DE);
        float[] x = new float[n];
        for (int i = 0; i < n; i++) {
            x[i] = rnd.nextFloat() * 2f - 1f;
        }
        return x;
    }
}

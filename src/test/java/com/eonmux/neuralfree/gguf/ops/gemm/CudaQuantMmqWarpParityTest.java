package com.eonmux.neuralfree.gguf.ops.gemm;

import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.acceleration.backend.cuda.CudaComputeBackend;
import com.eonmux.neuralfree.acceleration.cuda.CUDevice;
import com.eonmux.neuralfree.acceleration.kernels.Availability;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;

import java.util.Random;

import static org.junit.Assert.assertEquals;

/**
 * The warp-cooperative MMQ activation staging ({@code LLM_Quant_MMQ_Q8_1_Warp}) against the per-record kernel
 * ({@code LLM_Quant_MMQ_Q8_1}) of the same CUDA program: every record unit (codes, chunk scale, half code sums,
 * zero point) bit for bit, at both stage depths, over a strided view with an offset and a token count whose
 * work-items do not fill the last work-group, with chunks that are all zero, that hold a NaN or an infinity,
 * and that hold values far apart in magnitude.
 */
public class CudaQuantMmqWarpParityTest {

    private static final int LOCAL = 256;

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
    public void tensorStageDepthRecordsMatchThePerRecordKernel() throws InterruptedException {
        assertSameRecords(32, 77, 640, 652, 8);
    }

    @Test
    public void packedDotStageDepthRecordsMatchThePerRecordKernel() throws InterruptedException {
        assertSameRecords(16, 45, 208, 208, 0);
    }

    /** One wide row set at the 4B FFN width over a contiguous view, as the prefill stages it. */
    @Test
    public void prefillWidthRecordsMatchThePerRecordKernel() throws InterruptedException {
        assertSameRecords(32, 33, 9728, 9728, 0);
    }

    private void assertSameRecords(int tk, int n, int k, int stride, int off) throws InterruptedException {
        float[] x = buildX(n, k, stride, off);
        int recUnits = tk / 4 + 2 + tk / 16;
        long records = (long) n * (k / tk);
        int units = (int) (records * recUnits);
        GpuBuffer xbuf = backend.allocateBuffer((long) x.length * Float.BYTES, true);
        GpuBuffer plain = backend.allocateBuffer((long) units * Integer.BYTES);
        GpuBuffer warp = backend.allocateBuffer((long) units * Integer.BYTES);
        try {
            float[] a = new float[units];
            float[] b = new float[units];
            try (GpuEvent xv = queue.enqueueWriteBuffer(xbuf, 0, (long) x.length * Float.BYTES, x)) {
                Object[] argsPlain = {n, k, tk, stride, off, xbuf, plain};
                Object[] argsWarp = {n, k, tk, stride, off, xbuf, warp};
                queue.enqueueKernelNoEvent(kernels.get(GemmKernels.QUANT_MMQ_Q8_1),
                        new long[]{roundUp(records)}, new long[]{LOCAL}, argsPlain, xv);
                queue.enqueueKernelNoEvent(kernels.get(GemmKernels.QUANT_MMQ_Q8_1_WARP),
                        new long[]{roundUp((long) n * k / 4)}, new long[]{LOCAL}, argsWarp);
                try (GpuEvent ra = queue.enqueueReadBuffer(plain, 0, (long) units * Float.BYTES, a);
                     GpuEvent rb = queue.enqueueReadBuffer(warp, 0, (long) units * Float.BYTES, b, ra)) {
                    rb.awaitCompletion();
                }
            }
            for (int u = 0; u < units; u++) {
                assertEquals("tk " + tk + " record " + u / recUnits + " unit " + u % recUnits,
                        Integer.toHexString(Float.floatToRawIntBits(a[u])),
                        Integer.toHexString(Float.floatToRawIntBits(b[u])));
            }
        } finally {
            xbuf.close();
            plain.close();
            warp.close();
        }
    }

    private static long roundUp(long n) {
        return (n + LOCAL - 1) / LOCAL * LOCAL;
    }

    /**
     * Token rows of {@code k} values at {@code off + t * stride}: Gaussian values with per-chunk magnitudes spread
     * over eight decades, an all-zero chunk, a chunk holding a NaN, one holding an infinity and one with a single
     * non-zero value of each sign; the padding between rows holds values the view must not read (huge, so a
     * stray read changes the chunk scale).
     */
    private static float[] buildX(int n, int k, int stride, int off) {
        Random rnd = new Random(0x51A7E + k);
        float[] x = new float[off + n * stride];
        java.util.Arrays.fill(x, 1e30f);
        for (int t = 0; t < n; t++) {
            for (int i = 0; i < k; i++) {
                float scale = (float) Math.pow(10.0, ((t * 7 + i / 32) % 9) - 4);
                x[off + t * stride + i] = (float) rnd.nextGaussian() * scale;
            }
        }
        for (int i = 0; i < 32; i++) {
            x[off + i] = 0.0f;
        }
        if (n > 1 && k >= 128) {
            x[off + stride + 33] = Float.NaN;
            x[off + stride + 70] = Float.POSITIVE_INFINITY;
            for (int i = 96; i < 128; i++) {
                x[off + stride + i] = i == 100 ? -3.5f : i == 120 ? 0.25f : 0.0f;
            }
        }
        return x;
    }
}

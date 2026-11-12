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
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.Random;

import static org.junit.Assert.assertEquals;

/**
 * {@code <T>_MatMul_MMA_L} on the CUDA device against {@code <T>_MatMul_MMA} over the same views: the large tile
 * adds the same 16-deep chunk products in the same order, so every output must match bit for bit, and positions
 * outside the output view must keep their contents. The shapes cover the 16-byte weight and float4 activation
 * loads (K a multiple of 8, aligned views), the per-element loads (K off a multiple of 8, activation offsets and
 * strides off a multiple of 4), a partial last stage (K off a multiple of 32), ragged row and token tiles, and
 * output views with an offset and a padded stride.
 */
public class CudaMmaLargeTileParityTest {

    private static final float SENTINEL = -12345.0f;

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
            Assume.assumeTrue("the large MMA tile needs sm_80+", b.capabilities().getComputeCapability() >= 80);
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

    /** Aligned views over whole 32-deep stages: the 16-byte weight and float4 activation loads. */
    @Test
    public void alignedViewsMatchTheSmallTileBitForBit() throws Exception {
        assertSameBits(GGMLType.F16, 333, 2560, 141, 2560, 0, 333, 0);
        assertSameBits(GGMLType.BF16, 333, 2560, 141, 2560, 0, 333, 0);
    }

    /** K off a multiple of 8 (and of 4): per-element weight loads and the activation tail of every row. */
    @Test
    public void unalignedReductionMatchesTheSmallTileBitForBit() throws Exception {
        assertSameBits(GGMLType.F16, 200, 2425, 70, 2425, 0, 200, 0);
        assertSameBits(GGMLType.BF16, 200, 2425, 70, 2425, 0, 200, 0);
    }

    /**
     * Strided views: activation rows at an offset and stride off a multiple of 4 (per-element activation loads), a
     * partial last stage (K = 1000), one row past a row tile and an output view with an offset and padded stride.
     */
    @Test
    public void stridedViewsMatchTheSmallTileBitForBit() throws Exception {
        assertSameBits(GGMLType.F16, 129, 1000, 64, 1003, 5, 136, 11);
        assertSameBits(GGMLType.BF16, 129, 1000, 64, 1003, 5, 136, 11);
        assertSameBits(GGMLType.F16, 129, 1000, 65, 1004, 8, 129, 0);
    }

    private void assertSameBits(GGMLType type, int m, int k, int n, int xStride, int xOff, int yStride, int yOff)
            throws InterruptedException {
        Random rnd = new Random(31L * m + k + 7L * n);
        byte[] weight = weightBytes(type, m * k, rnd);
        float[] x = new float[xOff + (n - 1) * xStride + k];
        for (int i = 0; i < x.length; i++) {
            x[i] = (float) (rnd.nextGaussian() * 2.0);
        }
        int yLength = yOff + (n - 1) * yStride + m + 3;
        float[] small = launch(GemmKernels.matmulMma(type), MmaGemmPlan.TILE_ROWS, MmaGemmPlan.TILE_TOKENS,
                m, k, n, xStride, xOff, yStride, yOff, weight, x, yLength);
        float[] large = launch(GemmKernels.matmulMmaLarge(type), MmaGemmPlan.LARGE_TILE_ROWS,
                MmaGemmPlan.LARGE_TILE_TOKENS, m, k, n, xStride, xOff, yStride, yOff, weight, x, yLength);
        String what = type + " M=" + m + " K=" + k + " N=" + n + " x_stride=" + xStride + " x_off=" + xOff
                + " y_stride=" + yStride + " y_off=" + yOff;
        int live = 0;
        for (int i = 0; i < yLength; i++) {
            int rel = i - yOff;
            boolean inView = rel >= 0 && rel / yStride < n && rel % yStride < m;
            if (inView) {
                live++;
            } else {
                assertEquals(what + ": position " + i + " outside the view", SENTINEL, large[i], 0.0f);
            }
            assertEquals(what + ": position " + i + " small tile " + small[i] + " large tile " + large[i],
                    Float.floatToRawIntBits(small[i]), Float.floatToRawIntBits(large[i]));
        }
        assertEquals(what + ": live outputs", m * n, live);
    }

    private static byte[] weightBytes(GGMLType type, int count, Random rnd) {
        ByteBuffer b = ByteBuffer.allocate(count * 2).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < count; i++) {
            float v = (float) (rnd.nextGaussian() * 0.25);
            b.putShort(type == GGMLType.BF16 ? CudaMmaGemmParityTest.floatToBf16Bits(v)
                    : CudaMmaGemmParityTest.floatToHalfBits(v));
        }
        return b.array();
    }

    private float[] launch(String kernel, int tileRows, int tileTokens, int m, int k, int n, int xStride, int xOff,
                           int yStride, int yOff, byte[] weight, float[] x, int yLength) throws InterruptedException {
        float[] init = new float[yLength];
        Arrays.fill(init, SENTINEL);
        GpuBuffer wbuf = backend.allocateBuffer(weight.length, true);
        GpuBuffer xbuf = backend.allocateBuffer((long) x.length * Float.BYTES, true);
        GpuBuffer ybuf = backend.allocateBuffer((long) yLength * Float.BYTES);
        try {
            try (GpuEvent wv = queue.enqueueWriteBuffer(wbuf, 0, weight.length, weight);
                 GpuEvent xv = queue.enqueueWriteBuffer(xbuf, 0, (long) x.length * Float.BYTES, x, wv);
                 GpuEvent yv = queue.enqueueWriteBuffer(ybuf, 0, (long) yLength * Float.BYTES, init, xv)) {
                long[] global = {(long) ((m + tileRows - 1) / tileRows) * MmaGemmPlan.WORKGROUP_SIZE,
                        (n + tileTokens - 1) / tileTokens};
                long[] local = {MmaGemmPlan.WORKGROUP_SIZE, 1};
                Object[] args = {m, k, n, xStride, xOff, yStride, yOff, wbuf, xbuf, ybuf};
                queue.enqueueKernelNoEvent(kernels.get(kernel), global, local, args, yv);
                float[] out = new float[yLength];
                try (GpuEvent read = queue.enqueueReadBuffer(ybuf, 0, (long) yLength * Float.BYTES, out)) {
                    read.awaitCompletion();
                }
                return out;
            }
        } finally {
            wbuf.close();
            xbuf.close();
            ybuf.close();
        }
    }
}

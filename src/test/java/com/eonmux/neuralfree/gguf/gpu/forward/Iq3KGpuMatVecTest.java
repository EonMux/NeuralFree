package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.acceleration.backend.ocl.OclComputeBackend;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * The real OpenCL {T}_MatVec, {T}_MatVecFusedGateUp and {T}_EmbeddingLookup kernels of IQ3_K and IQ3_KS
 * against the CPU decoders and vec-dots, on the device the test finds. Blocks are random bytes with the
 * half scales forced finite; the tolerance covers the different float summation order of the work-group
 * reduction. Skipped when no OpenCL GPU is available.
 */
public class Iq3KGpuMatVecTest {

    private static final int COLS = 512;   // two 256-element super-blocks per row
    private static final int ROWS = 8;

    private HWAcceleration hwAccel;
    private OclComputeBackend backend;
    private LlmKernels kernels;
    private int local;

    @Before
    public void openDevice() {
        HWAcceleration accel = new HWAcceleration();
        try {
            if (accel.getCLDevices().isEmpty()) {
                accel.shutdown();
                Assume.assumeTrue("no OpenCL device", false);
            }
            hwAccel = accel;
            backend = new OclComputeBackend(accel.getCLDevices().get(0));
            kernels = LlmKernels.open(backend);
            local = kernels.matvecLocal();
        } catch (Exception e) {
            // The backend must close before the enumeration: it was built directly on the enumerated
            // device, so the shutdown below does not track it.
            if (backend != null) {
                try {
                    backend.close();
                } catch (RuntimeException ignored) {
                }
                backend = null;
            }
            if (accel != null) {
                accel.shutdown();
            }
            Assume.assumeTrue("no usable OpenCL GPU: " + e.getMessage(), false);
        }
    }

    @After
    public void closeDevice() {
        if (backend != null) {
            backend.close();
        }
        if (hwAccel != null) {
            hwAccel.shutdown();
        }
    }

    /** Random blocks of {@code type} with a finite half scale in every block. */
    private static byte[] randomWeights(GGMLType type, long seed) {
        Random rnd = new Random(seed);
        int blockBytes = type.getTypeSize();
        int blocks = ROWS * (COLS / 256);
        byte[] data = new byte[blocks * blockBytes];
        rnd.nextBytes(data);
        for (int b = 0; b < blocks; b++) {
            int h = com.eonmux.neuralfree.gguf.quant.FpBlockDecoders.fp32ToFp16(0.5f + rnd.nextFloat() * 0.5f);
            data[b * blockBytes] = (byte) h;
            data[b * blockBytes + 1] = (byte) (h >>> 8);
        }
        return data;
    }

    /**
     * The F32 reference of the GPU matvec: {@code sum_j decode(w)[j] * x[j]} over the whole row. The CPU
     * vec-dot of these types quantizes activations to Q8_K first (ggml parity), which the GPU matvec
     * kernels do not — they read F32 activations — so the two differ by the Q8_K rounding of about
     * {@code 1/127} per element.
     */
    private static float[] f32Reference(GGMLType type, byte[] weights, float[] x) {
        CpuTensorRef w = CpuTensorRef.of(type, ByteBuffer.wrap(weights).order(ByteOrder.LITTLE_ENDIAN),
                ROWS, COLS);
        float[] decoded = new float[COLS];
        float[] y = new float[ROWS];
        for (int r = 0; r < ROWS; r++) {
            w.decodeRow(r, decoded, 0);
            double sum = 0;
            for (int j = 0; j < COLS; j++) {
                sum += (double) decoded[j] * x[j];
            }
            y[r] = (float) sum;
        }
        return y;
    }

    private void assertMatvecMatchesCpu(GGMLType type, long seed) throws Exception {
        byte[] weights = randomWeights(type, seed);
        Random rnd = new Random(seed + 1);
        float[] x = new float[COLS];
        for (int i = 0; i < COLS; i++) {
            x[i] = rnd.nextFloat() * 2f - 1f;
        }
        float[] expected = f32Reference(type, weights, x);

        GpuBuffer wBuf = backend.allocateBuffer(weights.length);
        GpuBuffer xBuf = backend.allocateBuffer((long) COLS * 4);
        GpuBuffer yBuf = backend.allocateBuffer((long) ROWS * 4);
        try (GpuEvent w = backend.computeQueue().enqueueWriteBuffer(wBuf, 0, weights.length, weights);
             GpuEvent xw = backend.computeQueue().enqueueWriteBuffer(xBuf, 0, (long) COLS * 4, x, w)) {
            xw.awaitCompletion();
        }

        String kernel = type.name() + "_MatVec";
        long[] global = {ROWS * local};
        long[] lcl = {local};
        backend.computeQueue().enqueueKernelNoEvent(kernels.get(kernel), global, lcl,
                new Object[]{COLS, COLS, wBuf, xBuf, yBuf});
        backend.computeQueue().finish();

        float[] got = new float[ROWS];
        try (GpuEvent read = backend.computeQueue().enqueueReadBuffer(yBuf, 0, (long) ROWS * 4, got)) {
            read.awaitCompletion();
        }

        double magnitude = 0;
        for (int r = 0; r < ROWS; r++) {
            magnitude += Math.abs(expected[r]);
            assertEquals(type + " row " + r, expected[r], got[r], 2e-3 * magnitude / ROWS + 1e-2);
        }
        wBuf.close();
        xBuf.close();
        yBuf.close();
    }

    @Test
    public void iq3KMatVecMatchesCpu() throws Exception {
        assertMatvecMatchesCpu(GGMLType.IQ3_K, 20260501L);
    }

    @Test
    public void iq3KsMatVecMatchesCpu() throws Exception {
        assertMatvecMatchesCpu(GGMLType.IQ3_KS, 20260502L);
    }

    private void assertEmbeddingMatchesDecoder(GGMLType type, long seed) throws Exception {
        byte[] weights = randomWeights(type, seed);
        float[] decoded = new float[COLS];
        CpuTensorRef w = CpuTensorRef.of(type, ByteBuffer.wrap(weights).order(ByteOrder.LITTLE_ENDIAN),
                1, COLS);
        w.decodeRow(0, decoded, 0);

        GpuBuffer wBuf = backend.allocateBuffer(weights.length);
        GpuBuffer yBuf = backend.allocateBuffer((long) COLS * 4);
        try (GpuEvent wv = backend.computeQueue().enqueueWriteBuffer(wBuf, 0, weights.length, weights)) {
            wv.awaitCompletion();
        }

        String kernel = type.name() + "_EmbeddingLookup";
        long[] global = {COLS};
        backend.computeQueue().enqueueKernelNoEvent(kernels.get(kernel), global, new long[]{local},
                new Object[]{COLS, 0, wBuf, yBuf});
        backend.computeQueue().finish();

        float[] got = new float[COLS];
        try (GpuEvent read = backend.computeQueue().enqueueReadBuffer(yBuf, 0, (long) COLS * 4, got)) {
            read.awaitCompletion();
        }
        for (int i = 0; i < COLS; i++) {
            assertTrue(type + " element " + i + ": " + got[i] + " vs " + decoded[i],
                    Math.abs(got[i] - decoded[i]) <= 1e-4 * Math.abs(decoded[i]) + 1e-4);
        }
        wBuf.close();
        yBuf.close();
    }

    @Test
    public void iq3KEmbeddingLookupMatchesDecoder() throws Exception {
        assertEmbeddingMatchesDecoder(GGMLType.IQ3_K, 20260503L);
    }

    @Test
    public void iq3KsEmbeddingLookupMatchesDecoder() throws Exception {
        assertEmbeddingMatchesDecoder(GGMLType.IQ3_KS, 20260504L);
    }

    private void assertFusedGateUpMatchesCpu(GGMLType type, long seed) throws Exception {
        byte[] gate = randomWeights(type, seed);
        byte[] up = randomWeights(type, seed + 100);
        Random rnd = new Random(seed + 200);
        float[] x = new float[COLS];
        for (int i = 0; i < COLS; i++) {
            x[i] = rnd.nextFloat() * 2f - 1f;
        }
        float[] g = f32Reference(type, gate, x);
        float[] u = f32Reference(type, up, x);

        GpuBuffer gBuf = backend.allocateBuffer(gate.length);
        GpuBuffer uBuf = backend.allocateBuffer(up.length);
        GpuBuffer xBuf = backend.allocateBuffer((long) COLS * 4);
        GpuBuffer yBuf = backend.allocateBuffer((long) ROWS * 4);
        try (GpuEvent gw = backend.computeQueue().enqueueWriteBuffer(gBuf, 0, gate.length, gate);
             GpuEvent uw = backend.computeQueue().enqueueWriteBuffer(uBuf, 0, up.length, up, gw);
             GpuEvent xw = backend.computeQueue().enqueueWriteBuffer(xBuf, 0, (long) COLS * 4, x, uw)) {
            xw.awaitCompletion();
        }

        String kernel = type.name() + "_MatVecFusedGateUp";
        backend.computeQueue().enqueueKernelNoEvent(kernels.get(kernel), new long[]{ROWS * local},
                new long[]{local}, new Object[]{ROWS, COLS, gBuf, uBuf, xBuf, yBuf});
        backend.computeQueue().finish();

        float[] got = new float[ROWS];
        try (GpuEvent read = backend.computeQueue().enqueueReadBuffer(yBuf, 0, (long) ROWS * 4, got)) {
            read.awaitCompletion();
        }
        for (int r = 0; r < ROWS; r++) {
            float expected = (float) (g[r] / (1f + Math.exp(-g[r]))) * u[r];
            assertTrue(type + " row " + r + ": " + got[r] + " vs " + expected,
                    Math.abs(got[r] - expected) <= 2e-3 * (Math.abs(expected) + 1e-2));
        }
        gBuf.close();
        uBuf.close();
        xBuf.close();
        yBuf.close();
    }

    @Test
    public void iq3KFusedGateUpMatchesCpu() throws Exception {
        assertFusedGateUpMatchesCpu(GGMLType.IQ3_K, 20260505L);
    }

    @Test
    public void iq3KsFusedGateUpMatchesCpu() throws Exception {
        assertFusedGateUpMatchesCpu(GGMLType.IQ3_KS, 20260506L);
    }
}

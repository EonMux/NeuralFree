package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.acceleration.backend.cuda.CudaComputeBackend;
import com.eonmux.neuralfree.acceleration.kernels.Availability;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.tensor.HalfFloats;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.SplittableRandom;

import static org.junit.Assert.assertEquals;

/**
 * The dense F32, F16 and BF16 matvecs and their fused gate/up forms on CUDA against a double-precision CPU dot,
 * over both load paths of their partials: a row length that is a multiple of four starts every weight row on a
 * vector boundary (the aligned 64-bit / 128-bit loads), and an odd multiple of two starts every other row two
 * elements off one (the element loads), so one launch runs both. The tail loop covers the remainder of a row
 * length that is no multiple of four.
 */
public class DenseMatVecAlignmentGpuTest {

    private static final int ROWS = 37;
    private static final int[] IN_DIMS = {512, 514, 515};

    private HWAcceleration hwAccel;
    private CudaComputeBackend backend;
    private ComputeQueue queue;
    private LlmKernels kernels;

    private enum Dense {
        FLOAT("Float", 4), F16("F16", 2), BF16("BF16", 2);

        final String prefix;
        final int bytes;

        Dense(String prefix, int bytes) {
            this.prefix = prefix;
            this.bytes = bytes;
        }

        /** The stored element nearest {@code v} and its value. */
        float round(float v) {
            switch (this) {
                case F16:
                    return HalfFloats.halfToFloat(HalfFloats.floatToHalf(v));
                case BF16:
                    return HalfFloats.bfloat16ToFloat(HalfFloats.floatToBFloat16(v));
                default:
                    return v;
            }
        }

        void put(ByteBuffer b, float v) {
            switch (this) {
                case F16:
                    b.putShort((short) HalfFloats.floatToHalf(v));
                    break;
                case BF16:
                    b.putShort((short) HalfFloats.floatToBFloat16(v));
                    break;
                default:
                    b.putFloat(v);
            }
        }
    }

    @Before
    public void setUp() {
        HWAcceleration.enableCuda = true;
        HWAcceleration hw = null;
        try {
            hw = new HWAcceleration();
            Assume.assumeTrue("no CUDA device", !hw.getCUDevices().isEmpty());
            CudaComputeBackend b = new CudaComputeBackend(hw.getCUDevices().get(0));
            Availability availability = LlmKernels.availability(b);
            Assume.assumeTrue("LLM kernels unavailable: " + availability.reason(), availability.available());
            hwAccel = hw;
            backend = b;
            queue = b.computeQueue();
            kernels = LlmKernels.open(b);
        } catch (Throwable t) {
            if (hw != null) {
                hw.shutdown();
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
    public void matvecsMatchTheCpuDotOnAlignedAndUnalignedRows() throws InterruptedException {
        for (Dense type : Dense.values()) {
            for (int inDim : IN_DIMS) {
                SplittableRandom rnd = new SplittableRandom(31L * inDim + type.ordinal());
                float[] gate = values(rnd, ROWS * inDim, type);
                float[] up = values(rnd, ROWS * inDim, type);
                float[] x = values(rnd, inDim, Dense.FLOAT);
                float[] mv = run(type.prefix + "_MatVec", type, inDim, x, gate);
                float[] fused = run(type.prefix + "_MatVecFusedGateUp", type, inDim, x, gate, up);
                for (int r = 0; r < ROWS; r++) {
                    double g = dot(gate, r, x);
                    double u = dot(up, r, x);
                    double silu = g / (1.0 + Math.exp(-g));
                    String at = type + " in_dim " + inDim + " row " + r;
                    assertEquals(at, g, mv[r], 1e-4 * (1.0 + Math.abs(g)));
                    assertEquals(at + " fused", silu * u, fused[r], 1e-4 * (1.0 + Math.abs(silu * u)));
                }
            }
        }
    }

    private static float[] values(SplittableRandom rnd, int n, Dense type) {
        float[] v = new float[n];
        for (int i = 0; i < n; i++) {
            v[i] = type.round((float) (rnd.nextDouble() * 2.0 - 1.0));
        }
        return v;
    }

    private static double dot(float[] w, int row, float[] x) {
        double s = 0.0;
        for (int k = 0; k < x.length; k++) {
            s += (double) w[row * x.length + k] * x[k];
        }
        return s;
    }

    /** Launches {@code kernel} (rows x L work-items) over the weights, the last of which is the output. */
    private float[] run(String kernel, Dense type, int inDim, float[] x, float[]... weights)
            throws InterruptedException {
        int local = kernels.matvecLocal();
        GpuBuffer[] wbufs = new GpuBuffer[weights.length];
        GpuBuffer xbuf = backend.allocateBuffer((long) x.length * Float.BYTES, true);
        GpuBuffer ybuf = backend.allocateBuffer((long) ROWS * Float.BYTES);
        try {
            Object[] args = new Object[weights.length + 4];
            args[0] = ROWS;
            args[1] = inDim;
            for (int i = 0; i < weights.length; i++) {
                ByteBuffer b = ByteBuffer.allocate(weights[i].length * type.bytes).order(ByteOrder.LITTLE_ENDIAN);
                for (float v : weights[i]) {
                    type.put(b, v);
                }
                wbufs[i] = backend.allocateBuffer(b.capacity(), true);
                try (GpuEvent e = queue.enqueueWriteBuffer(wbufs[i], 0, b.capacity(), b.array())) {
                    e.awaitCompletion();
                }
                args[2 + i] = wbufs[i];
            }
            args[2 + weights.length] = xbuf;
            args[3 + weights.length] = ybuf;
            float[] out = new float[ROWS];
            try (GpuEvent xv = queue.enqueueWriteBuffer(xbuf, 0, (long) x.length * Float.BYTES, x)) {
                queue.enqueueKernelNoEvent(kernels.get(kernel), new long[]{(long) ROWS * local},
                        new long[]{local}, args, xv);
                try (GpuEvent read = queue.enqueueReadBuffer(ybuf, 0, (long) ROWS * Float.BYTES, out)) {
                    read.awaitCompletion();
                }
            }
            return out;
        } finally {
            for (GpuBuffer b : wbufs) {
                if (b != null) {
                    b.close();
                }
            }
            xbuf.close();
            ybuf.close();
        }
    }
}

package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.acceleration.backend.ocl.OclComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.vulkan.VkComputeBackend;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.quant.BlockLayout;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static org.junit.Assert.assertEquals;

/**
 * The real {@code F8_E4M3_B32_MatVec}, {@code F8_E4M3_B32_MatMul_Tiled} and {@code F8_E4M3_B32_EmbeddingLookup}
 * kernels of the OpenCL and Vulkan libraries against the CPU decoder, on rows of 192 values (six 32-value runs, no
 * whole 128-value block: the down projection width F8_E4M3_B32 exists for). Each backend is skipped when its GPU
 * is unavailable.
 */
@RunWith(Parameterized.class)
public class F8E4M3B32GpuTest {

    private static final GGMLType TYPE = GGMLType.F8_E4M3_B32;
    private static final int COLS = 192;
    private static final int ROWS = 70;

    @Parameterized.Parameters(name = "{0}")
    public static List<Object[]> backends() {
        return Arrays.asList(new Object[]{"opencl"}, new Object[]{"vulkan"});
    }

    private final String api;
    private HWAcceleration hwAccel;
    private ComputeBackend backend;
    private LlmKernels kernels;
    private int local;

    public F8E4M3B32GpuTest(String api) {
        this.api = api;
    }

    @Before
    public void openDevice() {
        try {
            backend = api.equals("vulkan") ? new VkComputeBackend() : openCl();
            kernels = LlmKernels.open(backend);
            local = kernels.matvecLocal();
        } catch (Throwable e) {
            closeDevice();
            Assume.assumeTrue("no usable " + api + " GPU: " + e.getMessage(), false);
        }
    }

    private ComputeBackend openCl() {
        HWAcceleration accel = new HWAcceleration();
        hwAccel = accel;
        Assume.assumeFalse("no OpenCL device", accel.getCLDevices().isEmpty());
        return new OclComputeBackend(accel.getCLDevices().get(0));
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

    /** {@code rows} rows of F8_E4M3_B32 runs: a distinct finite scale per run and no E4M3 NaN code. */
    private static byte[] weights(int rows, long seed) {
        Random rnd = new Random(seed);
        int blocks = rows * COLS / BlockLayout.QK_F8B32;
        byte[] data = new byte[blocks * BlockLayout.F8B32_BYTES];
        rnd.nextBytes(data);
        ByteBuffer b = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < blocks; i++) {
            int o = i * BlockLayout.F8B32_BYTES;
            b.putFloat(o + BlockLayout.F8B32_D, (0.001f + rnd.nextFloat() * 0.01f) * (1 + i % 3));
            for (int j = o + BlockLayout.F8B32_QS; j < o + BlockLayout.F8B32_BYTES; j++) {
                if ((data[j] & 0x7F) == 0x7F) {
                    data[j] = (byte) (data[j] & 0xFE);
                }
            }
        }
        return data;
    }

    private static float[] decoded(byte[] w, int rows) {
        CpuTensorRef ref = CpuTensorRef.of(TYPE, ByteBuffer.wrap(w).order(ByteOrder.LITTLE_ENDIAN), rows, COLS);
        float[] out = new float[rows * COLS];
        for (int r = 0; r < rows; r++) {
            ref.decodeRow(r, out, r * COLS);
        }
        return out;
    }

    private static float[] activations(int n, long seed) {
        Random rnd = new Random(seed);
        float[] x = new float[n];
        for (int i = 0; i < n; i++) {
            x[i] = rnd.nextFloat() * 2f - 1f;
        }
        return x;
    }

    /** {@code y[t * rows + r] = sum_k w[r][k] * x[t * COLS + k]} in double, with the sum of magnitudes. */
    private static double[][] reference(float[] w, int rows, float[] x, int tokens) {
        double[] y = new double[rows * tokens];
        double[] mag = new double[rows * tokens];
        for (int t = 0; t < tokens; t++) {
            for (int r = 0; r < rows; r++) {
                for (int k = 0; k < COLS; k++) {
                    double p = (double) w[r * COLS + k] * x[t * COLS + k];
                    y[t * rows + r] += p;
                    mag[t * rows + r] += Math.abs(p);
                }
            }
        }
        return new double[][]{y, mag};
    }

    private GpuBuffer upload(byte[] data) throws InterruptedException {
        GpuBuffer buf = backend.allocateBuffer(data.length);
        try (GpuEvent e = backend.computeQueue().enqueueWriteBuffer(buf, 0, data.length, data)) {
            e.awaitCompletion();
        }
        return buf;
    }

    private GpuBuffer upload(float[] data) throws InterruptedException {
        GpuBuffer buf = backend.allocateBuffer((long) data.length * 4);
        try (GpuEvent e = backend.computeQueue().enqueueWriteBuffer(buf, 0, (long) data.length * 4, data)) {
            e.awaitCompletion();
        }
        return buf;
    }

    private float[] read(GpuBuffer buf, int n) throws InterruptedException {
        float[] out = new float[n];
        try (GpuEvent e = backend.computeQueue().enqueueReadBuffer(buf, 0, (long) n * 4, out)) {
            e.awaitCompletion();
        }
        return out;
    }

    private void assertClose(String what, double[][] ref, float[] got) {
        for (int i = 0; i < got.length; i++) {
            assertEquals(api + " " + what + " output " + i, ref[0][i], got[i], 1e-5 * ref[1][i] + 1e-6);
        }
    }

    @Test
    public void matVecOfRowsOfNoWhole128ValueBlockMatchesTheDecoder() throws InterruptedException {
        byte[] w = weights(ROWS, 20261101L);
        float[] x = activations(COLS, 20261102L);
        GpuBuffer wBuf = upload(w);
        GpuBuffer xBuf = upload(x);
        GpuBuffer yBuf = backend.allocateBuffer((long) ROWS * 4);
        try {
            backend.computeQueue().enqueueKernelNoEvent(kernels.get(TYPE.name() + "_MatVec"), new long[]{(long) ROWS * local},
                    new long[]{local}, new Object[]{ROWS, COLS, wBuf, xBuf, yBuf});
            backend.computeQueue().finish();
            assertClose("MatVec", reference(decoded(w, ROWS), ROWS, x, 1), read(yBuf, ROWS));
        } finally {
            wBuf.close();
            xBuf.close();
            yBuf.close();
        }
    }

    @Test
    public void tiledGemmOfRowsOfNoWhole128ValueBlockMatchesTheDecoder() throws InterruptedException {
        int tokens = 5;
        byte[] w = weights(ROWS, 20261103L);
        float[] x = activations(tokens * COLS, 20261104L);
        GpuBuffer wBuf = upload(w);
        GpuBuffer xBuf = upload(x);
        GpuBuffer yBuf = backend.allocateBuffer((long) ROWS * tokens * 4);
        try {
            long groups = (ROWS + 63) / 64;
            backend.computeQueue().enqueueKernelNoEvent(kernels.get(TYPE.name() + "_MatMul_Tiled"),
                    new long[]{groups * 256, (tokens + 31) / 32}, new long[]{256, 1},
                    new Object[]{ROWS, COLS, tokens, COLS, 0, ROWS, 0, wBuf, xBuf, yBuf});
            backend.computeQueue().finish();
            assertClose("MatMul_Tiled", reference(decoded(w, ROWS), ROWS, x, tokens), read(yBuf, ROWS * tokens));
        } finally {
            wBuf.close();
            xBuf.close();
            yBuf.close();
        }
    }

    @Test
    public void embeddingLookupDecodesTheRowOfTheToken() throws InterruptedException {
        int rows = 3;
        byte[] w = weights(rows, 20261105L);
        float[] want = decoded(w, rows);
        GpuBuffer wBuf = upload(w);
        GpuBuffer yBuf = backend.allocateBuffer((long) COLS * 4);
        try {
            long global = (COLS + local - 1) / local * (long) local;
            backend.computeQueue().enqueueKernelNoEvent(kernels.get(TYPE.name() + "_EmbeddingLookup"), new long[]{global},
                    new long[]{local}, new Object[]{COLS, 2, wBuf, yBuf});
            backend.computeQueue().finish();
            float[] got = read(yBuf, COLS);
            for (int i = 0; i < COLS; i++) {
                assertEquals(api + " element " + i, want[2 * COLS + i], got[i], 0f);
            }
        } finally {
            wBuf.close();
            yBuf.close();
        }
    }
}

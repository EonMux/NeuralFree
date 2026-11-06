package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.cuda.CudaComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ocl.OclComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.vulkan.VkComputeBackend;
import com.eonmux.neuralfree.acceleration.cuda.CUDevice;
import com.eonmux.neuralfree.acceleration.kernels.Availability;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuPlaceholders;
import com.eonmux.neuralfree.gguf.quant.W12Codec;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * The W12 projections of {@link GpuOps} against a real GPU: an escape-free weight's exact matvec equals the
 * BF16 matvec of the same weights bit for bit (the same lanes, the same reduction), a weight full of zeros,
 * subnormals, specials and window outliers still matches the host decode of its row records, the whole-tensor
 * decode of a wide prefill writes exactly the mirror the host decodes (escapes scattered), and the Q-view
 * scope's output stays within the host bound of the 8-bit view. A backend is skipped when its GPU or its
 * kernel library is unavailable.
 */
@RunWith(Parameterized.class)
public class W12MatvecGpuTest {

    private static final int COLS = 256;
    private static final int ROWS = 32;
    private static final int TOKENS = 96;
    private static final String T = "blk.0.attn_q.weight";

    @Parameterized.Parameters(name = "{0}")
    public static List<Object[]> backends() {
        return Arrays.asList(new Object[]{"opencl"}, new Object[]{"cuda"}, new Object[]{"vulkan"});
    }

    private final String api;
    private HWAcceleration hwAccel;
    private ComputeBackend backend;
    private LlmKernels kernels;
    private GpuScratch scratch;
    private GpuPlaceholders placeholders;
    private GpuOps ops;
    private int local;
    private boolean previousCuda;

    /** Escape-free weights (one window covers every element) and their W12 record and decoded mirror. */
    private final float[] values;
    private final byte[] encoded;
    private final float[] mirror;
    private final byte[] mirrorBits;

    /** Weights packed with zeros, subnormals, specials and out-of-window outliers, and their record and
     * mirror; every row carries escapes the exact table must correct. */
    private final float[] escapeValues;
    private final byte[] escapeEncoded;
    private final float[] escapeMirror;
    private final byte[] escapeMirrorBits;

    public W12MatvecGpuTest(String api) {
        this.api = api;
        Random r = new Random(5);
        values = new float[ROWS * COLS];
        for (int i = 0; i < values.length; i++) {
            // the binades around 2^-4: one window covers every element, so no row needs escapes
            int sign = r.nextBoolean() ? 0x80000000 : 0;
            int e = 118 + r.nextInt(16);
            int m = r.nextInt(128) << 16;
            values[i] = Float.intBitsToFloat(sign | (e << 23) | m);
        }
        encoded = W12Codec.encode(T, values, ROWS, COLS);
        mirror = new float[values.length];
        W12Codec.decode(ByteBuffer.wrap(encoded), mirror, ROWS, COLS);
        mirrorBits = bf16Bits(mirror);

        escapeValues = new float[ROWS * COLS];
        for (int row = 0; row < ROWS; row++) {
            for (int k = 0; k < COLS; k++) {
                int sign = r.nextBoolean() ? 0x80000000 : 0;
                int e = 118 + r.nextInt(16);
                int m = r.nextInt(128) << 16;
                int i = row * COLS + k;
                if (k % 16 == 0) {
                    escapeValues[i] = 0.0f;
                } else if (row > 0 && k % 64 == 1) {
                    escapeValues[i] = Float.intBitsToFloat(0x00030000); // a bf16 subnormal
                } else if (row > 0 && k % 64 == 2) {
                    escapeValues[i] = Float.intBitsToFloat(sign | 0x7F800000); // an infinity
                } else if (row > 0 && k % 64 == 3) {
                    escapeValues[i] = Float.intBitsToFloat(0x7FC00000); // a NaN
                } else if (k % 16 == 8) {
                    escapeValues[i] = Float.intBitsToFloat(sign | (153 << 23) | m); // above the window
                } else if (k % 16 == 9) {
                    escapeValues[i] = Float.intBitsToFloat(sign | (98 << 23) | m); // below the window
                } else {
                    escapeValues[i] = Float.intBitsToFloat(sign | (e << 23) | m);
                }
            }
        }
        escapeEncoded = W12Codec.encode(T, escapeValues, ROWS, COLS);
        escapeMirror = new float[escapeValues.length];
        W12Codec.decode(ByteBuffer.wrap(escapeEncoded), escapeMirror, ROWS, COLS);
        escapeMirrorBits = bf16Bits(escapeMirror);
    }

    private static byte[] bf16Bits(float[] v) {
        ByteBuffer bits = ByteBuffer.allocate(v.length * 2).order(java.nio.ByteOrder.LITTLE_ENDIAN);
        for (float value : v) {
            bits.putShort((short) (Float.floatToRawIntBits(value) >>> 16));
        }
        return bits.array();
    }

    @Before
    public void openDevice() throws Exception {
        previousCuda = HWAcceleration.enableCuda;
        try {
            backend = openBackend();
            Availability availability = LlmKernels.availability(backend);
            Assume.assumeTrue("LLM kernels unavailable on " + api + ": " + availability.reason(),
                    availability.available());
            kernels = LlmKernels.open(backend);
            local = kernels.matvecLocal();
            scratch = GpuScratch.builder(backend, TOKENS).slot(ScratchSlots.X, (long) TOKENS * COLS).build();
            placeholders = GpuPlaceholders.allocate(backend, backend.computeQueue());
            ops = new GpuOps(backend.computeQueue(), kernels, scratch, placeholders);
        } catch (Throwable e) {
            closeDevice();
            Assume.assumeTrue("no usable " + api + " GPU: " + e.getMessage(), false);
        }
    }

    private ComputeBackend openBackend() {
        if (api.equals("vulkan")) {
            return new VkComputeBackend();
        }
        if (api.equals("cuda")) {
            HWAcceleration.enableCuda = true;
            HWAcceleration accel = new HWAcceleration();
            hwAccel = accel;
            Assume.assumeFalse("no CUDA device", accel.getCUDevices().isEmpty());
            CUDevice device = accel.getCUDevices().get(0);
            return new CudaComputeBackend(device);
        }
        HWAcceleration accel = new HWAcceleration();
        hwAccel = accel;
        Assume.assumeFalse("no OpenCL device", accel.getCLDevices().isEmpty());
        return new OclComputeBackend(accel.getCLDevices().get(0));
    }

    @After
    public void closeDevice() {
        if (ops != null) {
            ops.close();
            ops = null;
        }
        if (placeholders != null) {
            placeholders.close();
            placeholders = null;
        }
        if (scratch != null) {
            scratch.close();
            scratch = null;
        }
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
        HWAcceleration.enableCuda = previousCuda;
    }

    // ------------------------------------------------------------------ fixtures

    private GpuBuffer upload(byte[] data) throws Exception {
        GpuBuffer buf = backend.allocateBuffer(data.length);
        backend.computeQueue().enqueueWriteBuffer(buf, 0, data.length, data).awaitCompletion();
        return buf;
    }

    private GpuBuffer upload(float[] data) throws Exception {
        GpuBuffer buf = backend.allocateBuffer((long) data.length * Float.BYTES);
        backend.computeQueue().enqueueWriteBuffer(buf, 0, (long) data.length * Float.BYTES, data)
                .awaitCompletion();
        return buf;
    }

    private float[] read(GpuBuffer buf, int floats) throws Exception {
        float[] out = new float[floats];
        backend.computeQueue().enqueueReadBuffer(buf, 0, (long) floats * Float.BYTES, out).awaitCompletion();
        return out;
    }

    private GpuWeight w12() throws Exception {
        return GpuWeight.of(T, GGMLType.W12, new long[]{COLS, ROWS}, upload(encoded));
    }

    private GpuWeight w12WithEscapes() throws Exception {
        return GpuWeight.of(T, GGMLType.W12, new long[]{COLS, ROWS}, upload(escapeEncoded));
    }

    private GpuWeight bf16(byte[] bits) throws Exception {
        return GpuWeight.of(T, GGMLType.BF16, new long[]{COLS, ROWS}, upload(bits));
    }

    /** out[t * ROWS + r] = weight row r dot input row t, reduced in double. */
    private double[] reference(float[] weights, float[] input, int nTokens) {
        double[] out = new double[nTokens * ROWS];
        for (int t = 0; t < nTokens; t++) {
            for (int r2 = 0; r2 < ROWS; r2++) {
                double sum = 0;
                for (int j = 0; j < COLS; j++) {
                    sum += (double) weights[r2 * COLS + j] * (double) input[t * COLS + j];
                }
                out[t * ROWS + r2] = sum;
            }
        }
        return out;
    }

    private static void assertClose(float[] got, double[] want, String what, float tol) {
        for (int i = 0; i < want.length; i++) {
            assertTrue(what + "[" + i + "]: got " + got[i] + ", want " + want[i],
                    Math.abs(got[i] - want[i]) <= tol * Math.max(1f, (float) Math.abs(want[i])));
        }
    }

    /** {@link #assertClose} over rows whose weights hold infinities or NaNs: an infinite or NaN reference
     * asks for the same class of result, not a finite distance. */
    private static void assertCloseWithSpecials(float[] got, double[] want, String what, float tol) {
        for (int i = 0; i < want.length; i++) {
            double w = want[i];
            if (Double.isNaN(w)) {
                assertTrue(what + "[" + i + "] is not NaN: " + got[i], Float.isNaN(got[i]));
            } else if (Double.isInfinite(w)) {
                assertEquals(what + "[" + i + "]", (float) w, got[i], 0.0f);
            } else {
                assertTrue(what + "[" + i + "]: got " + got[i] + ", want " + w,
                        Math.abs(got[i] - w) <= tol * Math.max(1f, (float) Math.abs(w)));
            }
        }
    }

    private static float[] randomInput(long seed, int nTokens) {
        Random r = new Random(seed);
        float[] x = new float[nTokens * COLS];
        for (int i = 0; i < x.length; i++) {
            x[i] = r.nextFloat() * 2f - 1f;
        }
        return x;
    }

    // ------------------------------------------------------------------ the row kernels

    @Test
    public void matvecMatchesTheHostDecodeAndTheBf16Kernel() throws Exception {
        float[] x = randomInput(7, 4);
        double[] want = reference(mirror, x, 4);
        GpuBuffer in = upload(x);
        GpuBuffer wOut = backend.allocateBuffer((long) 4 * ROWS * Float.BYTES);
        GpuBuffer bOut = backend.allocateBuffer((long) 4 * ROWS * Float.BYTES);
        ops.matvec(w12(), in, 0, wOut, 0, 4);
        ops.matvec(bf16(mirrorBits), in, 0, bOut, 0, 4);
        float[] got = read(wOut, 4 * ROWS);
        float[] fromBf16 = read(bOut, 4 * ROWS);
        assertClose(got, want, "w12", 1e-4f);
        assertClose(fromBf16, want, "bf16", 1e-4f);
        // escape-free rows: the same weights, the same lanes and reduction — bit for bit
        assertArrayEquals(got, fromBf16, 0.0f);
    }

    @Test
    public void matvecWithEscapesMatchesTheHostDecode() throws Exception {
        float[] x = randomInput(11, 4);
        double[] want = reference(escapeMirror, x, 4);
        GpuBuffer in = upload(x);
        GpuBuffer wOut = backend.allocateBuffer((long) 4 * ROWS * Float.BYTES);
        GpuBuffer bOut = backend.allocateBuffer((long) 4 * ROWS * Float.BYTES);
        ops.matvec(w12WithEscapes(), in, 0, wOut, 0, 4);
        ops.matvec(bf16(escapeMirrorBits), in, 0, bOut, 0, 4);
        float[] got = read(wOut, 4 * ROWS);
        float[] fromBf16 = read(bOut, 4 * ROWS);
        // the escaped zeros, subnormals, specials and outliers are corrected from the exact table
        assertCloseWithSpecials(got, want, "w12 with escapes", 1e-4f);
        assertCloseWithSpecials(got, toDoubles(fromBf16), "against the bf16 kernel", 1e-4f);
    }

    private static double[] toDoubles(float[] v) {
        double[] out = new double[v.length];
        for (int i = 0; i < v.length; i++) {
            out[i] = v[i];
        }
        return out;
    }

    @Test
    public void matvecAddAccumulatesTheHostReference() throws Exception {
        float[] x = randomInput(13, 1);
        double[] want = reference(mirror, x, 1);
        float[] base = new float[ROWS];
        for (int r2 = 0; r2 < ROWS; r2++) {
            base[r2] = 1.5f + r2;
        }
        GpuBuffer in = upload(x);
        GpuBuffer accum = upload(base);
        ops.matvecAdd(w12(), in, 0, accum, 0, 1);
        float[] got = read(accum, ROWS);
        for (int r2 = 0; r2 < ROWS; r2++) {
            assertEquals("add[" + r2 + "]", (float) want[r2] + base[r2], got[r2], 1e-4f * Math.max(1f, base[r2]));
        }
    }

    @Test
    public void theQViewStaysWithinTheHostBound() throws Exception {
        float[] x = randomInput(17, 1);
        ByteBuffer enc = ByteBuffer.wrap(escapeEncoded);
        double[] qView = new double[ROWS];
        double[] bound = new double[ROWS];
        for (int r2 = 0; r2 < ROWS; r2++) {
            for (int j = 0; j < COLS; j++) {
                float w = W12Codec.qViewElement(enc, ROWS, COLS, r2, j);
                qView[r2] += (double) w * x[j];
                bound[r2] += (double) Math.abs(escapeMirror[r2 * COLS + j] - w) * Math.abs(x[j]);
            }
        }
        GpuBuffer in = upload(x);
        GpuBuffer out = backend.allocateBuffer((long) ROWS * Float.BYTES);
        try (GpuOps.W12QView scope = ops.openW12QView()) {
            ops.matvec(w12WithEscapes(), in, 0, out, 0, 1);
        }
        float[] got = read(out, ROWS);
        // row 0 carries no NaN weights, so its bound is finite: the Q view stays inside it
        assertEquals("row 0 of the Q view", qView[0], got[0],
                1e-4f + 1.01f * (float) bound[0]);
        // every other row reads its own row of the record against this input: the same host view value,
        // where a NaN-weighted row is finite in the view (NaNs are not Q escapes) and an infinite one
        // keeps the infinity's sign
        for (int r2 = 1; r2 < ROWS; r2++) {
            String what = "row " + r2 + " of the Q view";
            if (Double.isNaN(qView[r2])) {
                assertTrue(what + " is not NaN: " + got[r2], Float.isNaN(got[r2]));
            } else if (Double.isInfinite(qView[r2])) {
                assertEquals(what, (float) qView[r2], got[r2], 0.0f);
            } else {
                assertEquals(what, qView[r2], got[r2],
                        1e-3f * Math.max(1f, (float) Math.abs(qView[r2])));
            }
        }
    }

    // ------------------------------------------------------------------ the wide prefill

    @Test
    public void aWidePrefillRunsOnAMirrorTheHostDecodes() throws Exception {
        float[] x = randomInput(19, TOKENS);
        double[] want = reference(escapeMirror, x, TOKENS);
        GpuBuffer in = upload(x);
        GpuBuffer out = backend.allocateBuffer((long) TOKENS * ROWS * Float.BYTES);
        ops.matvec(w12WithEscapes(), in, 0, out, 0, TOKENS);
        assertCloseWithSpecials(read(out, TOKENS * ROWS), want, "prefill", 1e-4f);
    }

    /** The decode kernel alone: its u16 mirror equals the host decode's bf16 bits of every row, escapes
     * scattered over the plane codes. */
    @Test
    public void theDecodeKernelWritesTheHostMirrorBits() throws Exception {
        GpuBuffer weight = upload(escapeEncoded);
        GpuBuffer mirrorBuf = backend.allocateBuffer((long) ROWS * COLS * 2L);
        long global = (long) ROWS * local;
        backend.computeQueue()
                .enqueueKernel(kernels.get("W12_DecodeBF16"), new long[]{global}, new long[]{local},
                        new Object[]{ROWS, COLS, 0, weight, mirrorBuf})
                .awaitCompletion();
        byte[] got = new byte[ROWS * COLS * 2];
        backend.computeQueue().enqueueReadBuffer(mirrorBuf, 0, (long) got.length, got).awaitCompletion();
        assertArrayEquals("the device mirror bits", escapeMirrorBits, got);
    }
}

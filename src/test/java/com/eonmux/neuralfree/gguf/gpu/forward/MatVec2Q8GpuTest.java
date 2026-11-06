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
import com.eonmux.neuralfree.gguf.cpu.quant.CpuActivationQuantizer;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuVecDot.QuantizedActivations;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.quant.FpBlockDecoders;
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
import static org.junit.Assert.assertTrue;

/**
 * The two-row integer-dot matvec kernels ({@code {T}_MatVec2_Q8}) and their fused gate/up form
 * ({@code {T}_MatVec2FusedGateUp_Q8}) of every supported weight type against a float reference over
 * the CPU-decoded weights and the CPU Q8_1 quantizer's dequantized activations, on every available GPU
 * backend, next to the single-row integer-dot kernels on the same weights and activation row. Weights
 * are random bytes with finite block scales; the tolerance covers the float summation order of the
 * work-group reduction and the dual decode's association. A backend is skipped when its GPU or its
 * kernel library is unavailable.
 */
@RunWith(Parameterized.class)
public class MatVec2Q8GpuTest {

    private static final int COLS = 512;   // two 256-element super-blocks (sixteen 32-element blocks)
    private static final int ROWS = 8;     // four row pairs
    private static final int ODD_ROWS = 7; // odd row count: the tail pair repeats the last row

    /** Every weight type whose family instantiation provides a {@code {T}_MatVec2_Q8} kernel. */
    private static final GGMLType[] TYPES = {
        GGMLType.Q4_0, GGMLType.Q4_1, GGMLType.Q5_0, GGMLType.Q5_1, GGMLType.Q8_0,
        GGMLType.MXFP4, GGMLType.NVFP4,
        GGMLType.Q2_K, GGMLType.Q3_K, GGMLType.Q4_K, GGMLType.Q5_K, GGMLType.Q6_K, GGMLType.Q8_K,
        GGMLType.IQ4_NL, GGMLType.IQ4_XS,
        GGMLType.IQ2_XXS, GGMLType.IQ2_XS, GGMLType.IQ2_S,
        GGMLType.IQ3_XXS, GGMLType.IQ3_S, GGMLType.IQ3_K, GGMLType.IQ3_KS,
        GGMLType.IQ1_S, GGMLType.IQ1_M,
    };

    @Parameterized.Parameters(name = "{0}")
    public static List<Object[]> backends() {
        return Arrays.asList(new Object[]{"opencl"}, new Object[]{"cuda"}, new Object[]{"vulkan"});
    }

    private final String api;
    private HWAcceleration hwAccel;
    private ComputeBackend backend;
    private LlmKernels kernels;
    private int local;
    private boolean previousCuda;

    public MatVec2Q8GpuTest(String api) {
        this.api = api;
    }

    @Before
    public void openDevice() {
        previousCuda = HWAcceleration.enableCuda;
        try {
            backend = openBackend();
            Availability availability = LlmKernels.availability(backend);
            Assume.assumeTrue("LLM kernels unavailable on " + api + ": " + availability.reason(),
                    availability.available());
            kernels = LlmKernels.open(backend);
            local = kernels.matvecLocal();
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

    /** Random blocks of {@code type} with finite block scales. */
    private static byte[] randomWeights(GGMLType type, long seed) {
        Random rnd = new Random(seed);
        int blockBytes = type.getTypeSize();
        int blockSize = type.getBlockSize();
        int blocks = ROWS * (COLS / blockSize);
        byte[] data = new byte[blocks * blockBytes];
        rnd.nextBytes(data);
        for (int b = 0; b < blocks; b++) {
            int base = b * blockBytes;
            switch (type) {
                case Q4_0:
                case Q5_0:
                    putHalf(data, base, 0.02f);
                    break;
                case Q4_1:
                case Q5_1:
                    putHalf(data, base, 0.02f);
                    putHalf(data, base + 2, 0.01f);
                    break;
                case MXFP4:
                    data[base] = (byte) 127;      // e8m0: half scale 0.5
                    break;
                case NVFP4:
                    for (int i = 0; i < 4; i++) {
                        data[base + i] = (byte) 0x40;   // ue4m3: 1.0
                    }
                    break;
                case Q2_K:
                    putHalf(data, base + 80, 0.02f);
                    putHalf(data, base + 82, 0.01f);
                    break;
                case Q3_K:
                    putHalf(data, base + 108, 0.02f);
                    break;
                case Q4_K:
                case Q5_K:
                    putHalf(data, base, 0.02f);
                    putHalf(data, base + 2, 0.01f);
                    break;
                case Q6_K:
                    putHalf(data, base + 208, 0.01f);
                    break;
                case Q8_K:
                    ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN).putFloat(base, 0.01f);
                    break;
                case IQ1_M:
                    // Rebuild d = 1.0 from the top nibbles of the four scale words (48..55).
                    data[base + 53] = (byte) ((data[base + 53] & 0x0F) | 0xC0);
                    data[base + 55] = (byte) ((data[base + 55] & 0x0F) | 0x30);
                    break;
                default:
                    putHalf(data, base, 0.02f);
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

    private static float[] randomActivations(long seed) {
        Random rnd = new Random(seed);
        float[] x = new float[COLS];
        for (int i = 0; i < COLS; i++) {
            x[i] = rnd.nextFloat() * 2f - 1f;
        }
        return x;
    }

    /**
     * The float reference: decode each weight row and dot it with the CPU Q8_1 quantizer's dequantized
     * activations ({@code d8 * q8}), which is what the integer-dot kernels compute from codes.
     */
    private static float[] reference(GGMLType type, byte[] weights, float[] x, int rows) {
        CpuTensorRef w = CpuTensorRef.of(type, ByteBuffer.wrap(weights).order(ByteOrder.LITTLE_ENDIAN),
                rows, COLS);
        QuantizedActivations q = CpuActivationQuantizer.quantizeQ8_1(x, 0, COLS);
        float[] decoded = new float[COLS];
        float[] y = new float[rows];
        for (int r = 0; r < rows; r++) {
            w.decodeRow(r, decoded, 0);
            double sum = 0;
            for (int j = 0; j < COLS; j++) {
                sum += (double) decoded[j] * ((double) q.scale(j / 32) * q.quant(j));
            }
            y[r] = (float) sum;
        }
        return y;
    }

    /** A device buffer holding {@code data}. */
    private GpuBuffer upload(byte[] data) throws Exception {
        GpuBuffer buf = backend.allocateBuffer(data.length);
        backend.computeQueue().enqueueWriteBuffer(buf, 0, data.length, data).awaitCompletion();
        return buf;
    }

    private GpuBuffer upload(float[] data) throws Exception {
        GpuBuffer buf = backend.allocateBuffer((long) data.length * 4);
        backend.computeQueue().enqueueWriteBuffer(buf, 0, (long) data.length * 4, data).awaitCompletion();
        return buf;
    }

    /** The Q8_1 blocks of {@code x} as the quantizer kernel writes them. */
    private GpuBuffer q8_1(float[] x) throws Exception {
        int blocks = COLS / 32;
        GpuBuffer xBuf = upload(x);
        GpuBuffer qBuf = backend.allocateBuffer((long) blocks * GpuOps.Q8_1_FLOATS * 4);
        backend.computeQueue().enqueueKernel(kernels.get(GpuOps.QUANTIZE_Q8_1),
                new long[]{roundUp(blocks)}, new long[]{local}, new Object[]{COLS, xBuf, qBuf});
        backend.computeQueue().finish();
        xBuf.close();
        return qBuf;
    }

    private long roundUp(long n) {
        return (n + local - 1) / local * local;
    }

    private String requireKernel(String name) {
        assertTrue(api + " build has no " + name + " kernel", kernels.has(name));
        return name;
    }

    /** The matvec output of {@code kernel} over {@code rows} rows, at its own launch geometry. */
    private float[] matvec(String kernel, byte[] weights, GpuBuffer q8, int rows) throws Exception {
        GpuBuffer wBuf = upload(weights);
        GpuBuffer yBuf = backend.allocateBuffer((long) rows * 4);
        long groups = kernel.contains("_MatVec2") ? ceilDiv(rows, 2) : rows;
        backend.computeQueue().enqueueKernel(kernels.get(kernel), new long[]{groups * local},
                new long[]{local}, new Object[]{rows, COLS, wBuf, q8, yBuf});
        backend.computeQueue().finish();
        float[] got = new float[rows];
        backend.computeQueue().enqueueReadBuffer(yBuf, 0, (long) rows * 4, got).awaitCompletion();
        wBuf.close();
        yBuf.close();
        return got;
    }

    /** The fused gate/up output of {@code kernel} over {@code rows} rows, at its own launch geometry. */
    private float[] fusedGateUp(String kernel, byte[] gate, byte[] up, GpuBuffer q8, int rows) throws Exception {
        GpuBuffer gBuf = upload(gate);
        GpuBuffer uBuf = upload(up);
        GpuBuffer yBuf = backend.allocateBuffer((long) rows * 4);
        long groups = kernel.contains("_MatVec2") ? ceilDiv(rows, 2) : rows;
        backend.computeQueue().enqueueKernel(kernels.get(kernel), new long[]{groups * local},
                new long[]{local}, new Object[]{rows, COLS, gBuf, uBuf, q8, yBuf});
        backend.computeQueue().finish();
        float[] got = new float[rows];
        backend.computeQueue().enqueueReadBuffer(yBuf, 0, (long) rows * 4, got).awaitCompletion();
        gBuf.close();
        uBuf.close();
        yBuf.close();
        return got;
    }

    private static float silu(float v) {
        return v / (1f + (float) Math.exp(-v));
    }

    /** {@code silu(gate . x) * (up . x)} per row from the reference dots. */
    private static float[] fusedReference(float[] gateDots, float[] upDots) {
        float[] y = new float[gateDots.length];
        for (int r = 0; r < y.length; r++) {
            y[r] = silu(gateDots[r]) * upDots[r];
        }
        return y;
    }

    private static int ceilDiv(int n, int d) {
        return (n + d - 1) / d;
    }

    private static void assertClose(String what, float[] expected, float[] got) {
        double magnitude = 0;
        for (float v : expected) {
            magnitude += Math.abs(v);
        }
        double tolerance = 1e-5 + 2e-4 * magnitude;
        for (int r = 0; r < expected.length; r++) {
            assertEquals(what + " row " + r, expected[r], got[r], tolerance / expected.length);
        }
    }

    // ------------------------------------------------------------------ tests

    @Test
    public void twoRowMatvecMatchesTheDequantizedActivationReference() throws Exception {
        for (GGMLType type : TYPES) {
            byte[] weights = randomWeights(type, 20000 + type.ordinal());
            float[] x = randomActivations(21000 + type.ordinal());
            float[] expected = reference(type, weights, x, ROWS);
            GpuBuffer q8 = q8_1(x);
            try {
                String kernel = requireKernel(type.name() + "_MatVec2_Q8");
                assertClose(api + " " + type + " " + kernel, expected, matvec(kernel, weights, q8, ROWS));

                String single = requireKernel(type.name() + "_MatVec_Q8");
                assertClose(api + " " + type + " " + single, expected, matvec(single, weights, q8, ROWS));
            } finally {
                q8.close();
            }
        }
    }

    @Test
    public void oddRowCountTailPairStoresEveryRowOnce() throws Exception {
        for (GGMLType type : TYPES) {
            byte[] weights = randomWeights(type, 22000 + type.ordinal());
            float[] x = randomActivations(23000 + type.ordinal());
            float[] expected = reference(type, weights, x, ODD_ROWS);
            GpuBuffer q8 = q8_1(x);
            try {
                String kernel = requireKernel(type.name() + "_MatVec2_Q8");
                assertClose(api + " " + type + " " + kernel + " odd", expected,
                        matvec(kernel, weights, q8, ODD_ROWS));
            } finally {
                q8.close();
            }
        }
    }

    @Test
    public void fusedGateUpMatchesSiluOfBothRows() throws Exception {
        for (GGMLType type : TYPES) {
            byte[] gate = randomWeights(type, 24000 + type.ordinal());
            byte[] up = randomWeights(type, 25000 + type.ordinal());
            float[] x = randomActivations(26000 + type.ordinal());
            GpuBuffer q8 = q8_1(x);
            try {
                String kernel = requireKernel(type.name() + "_MatVec2FusedGateUp_Q8");
                float[] expected = fusedReference(reference(type, gate, x, ROWS),
                        reference(type, up, x, ROWS));
                assertClose(api + " " + type + " " + kernel, expected, fusedGateUp(kernel, gate, up, q8, ROWS));

                String single = requireKernel(type.name() + "_MatVecFusedGateUp_Q8");
                assertClose(api + " " + type + " " + single, expected, fusedGateUp(single, gate, up, q8, ROWS));

                float[] oddExpected = fusedReference(reference(type, gate, x, ODD_ROWS),
                        reference(type, up, x, ODD_ROWS));
                assertClose(api + " " + type + " " + kernel + " odd", oddExpected,
                        fusedGateUp(kernel, gate, up, q8, ODD_ROWS));
            } finally {
                q8.close();
            }
        }
    }
}

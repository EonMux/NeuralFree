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
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuPlaceholders;
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

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * The Q8_1 activation quantizer ({@code LLM_QuantizeQ8_1}) and the integer-dot matvec kernels
 * ({@code {T}_MatVec_Q8}, {@code {T}_MatVecAdd_Q8}, {@code {T}_MatVecFusedGateUp_Q8}) of every supported
 * weight type against the CPU quantizer and a float reference over the dequantized activations, on every
 * available GPU backend. Weights are random bytes with finite block scales; the tolerance covers the
 * float summation order of the work-group reduction. A backend is skipped when its GPU or its kernel
 * library is unavailable.
 */
@RunWith(Parameterized.class)
public class Int8MatvecGpuTest {

    private static final int COLS = 512;   // two 256-element super-blocks (sixteen 32-element blocks)
    private static final int ROWS = 8;
    private static final int Q8_1_BYTES = 36;

    /** Every weight type with a {@code {T}_MatVec_Q8} kernel. */
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

    public Int8MatvecGpuTest(String api) {
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
     * activations ({@code d8 * q8}), which is exactly what the integer-dot kernel computes from codes.
     */
    private static float[] reference(GGMLType type, byte[] weights, float[] x) {
        CpuTensorRef w = CpuTensorRef.of(type, ByteBuffer.wrap(weights).order(ByteOrder.LITTLE_ENDIAN),
                ROWS, COLS);
        QuantizedActivations q = CpuActivationQuantizer.quantizeQ8_1(x, 0, COLS);
        float[] decoded = new float[COLS];
        float[] y = new float[ROWS];
        for (int r = 0; r < ROWS; r++) {
            w.decodeRow(r, decoded, 0);
            double sum = 0;
            for (int j = 0; j < COLS; j++) {
                sum += (double) decoded[j] * ((double) q.scale(j / 32) * q.quant(j));
            }
            y[r] = (float) sum;
        }
        return y;
    }

    private ByteBuffer q8_1(float[] x) throws Exception {
        return q8_1(x, COLS);
    }

    private ByteBuffer q8_1(float[] x, int size) throws Exception {
        int blocks = size / 32;
        GpuBuffer xBuf = backend.allocateBuffer((long) COLS * 4);
        GpuBuffer qBuf = backend.allocateBuffer((long) blocks * Q8_1_BYTES);
        backend.computeQueue().enqueueWriteBuffer(xBuf, 0, (long) COLS * 4, x).awaitCompletion();
        long[] global = {roundUp(blocks)};
        backend.computeQueue().enqueueKernel(kernels.get(GpuOps.QUANTIZE_Q8_1), global, new long[]{local},
                new Object[]{size, xBuf, qBuf});
        backend.computeQueue().finish();
        byte[] out = new byte[blocks * Q8_1_BYTES];
        backend.computeQueue().enqueueReadBuffer(qBuf, 0, out.length, out).awaitCompletion();
        xBuf.close();
        qBuf.close();
        return ByteBuffer.wrap(out).order(ByteOrder.LITTLE_ENDIAN);
    }

    private long roundUp(long n) {
        return (n + local - 1) / local * local;
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

    private GpuBuffer upload(ByteBuffer q8) throws Exception {
        byte[] bytes = new byte[q8.capacity()];
        q8.duplicate().get(bytes);
        return upload(bytes);
    }

    private String requireKernel(String name) {
        assertTrue(api + " build has no " + name + " kernel", kernels.has(name));
        return name;
    }

    private float[] matvec(byte[] weights, ByteBuffer q8, String op, boolean add) throws Exception {
        GpuBuffer wBuf = upload(weights);
        GpuBuffer xBuf = upload(q8);
        GpuBuffer yBuf = backend.allocateBuffer((long) ROWS * 4);
        if (add) {
            backend.computeQueue().enqueueFillBuffer(yBuf, 0, (long) ROWS * 4, 0);
        }
        long[] global = {(long) ROWS * local};
        backend.computeQueue().enqueueKernel(kernels.get(op), global, new long[]{local},
                new Object[]{ROWS, COLS, wBuf, xBuf, yBuf});
        backend.computeQueue().finish();
        float[] got = new float[ROWS];
        backend.computeQueue().enqueueReadBuffer(yBuf, 0, (long) ROWS * 4, got).awaitCompletion();
        wBuf.close();
        xBuf.close();
        yBuf.close();
        return got;
    }

    private float[] fusedGateUp(String kernel, byte[] gate, byte[] up, ByteBuffer q8) throws Exception {
        GpuBuffer gBuf = upload(gate);
        GpuBuffer uBuf = upload(up);
        GpuBuffer xBuf = upload(q8);
        GpuBuffer yBuf = backend.allocateBuffer((long) ROWS * 4);
        long[] global = {(long) ROWS * local};
        backend.computeQueue().enqueueKernel(kernels.get(kernel),
                global, new long[]{local}, new Object[]{ROWS, COLS, gBuf, uBuf, xBuf, yBuf});
        backend.computeQueue().finish();
        float[] got = new float[ROWS];
        backend.computeQueue().enqueueReadBuffer(yBuf, 0, (long) ROWS * 4, got).awaitCompletion();
        gBuf.close();
        uBuf.close();
        xBuf.close();
        yBuf.close();
        return got;
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
    public void quantizerMatchesTheCpuQ8_1Reference() throws Exception {
        float[] x = randomActivations(41);
        QuantizedActivations cpu = CpuActivationQuantizer.quantizeQ8_1(x, 0, COLS);
        ByteBuffer gpu = q8_1(x);
        for (int b = 0; b < COLS / 32; b++) {
            int off = b * Q8_1_BYTES;
            assertEquals("block " + b + " d", FpBlockDecoders.fp32ToFp16(cpu.scale(b)),
                    gpu.getShort(off) & 0xFFFF);
            assertEquals("block " + b + " s", FpBlockDecoders.fp32ToFp16(cpu.blockSum(b)),
                    gpu.getShort(off + 2) & 0xFFFF);
            for (int j = 0; j < 32; j++) {
                assertEquals("block " + b + " code " + j, cpu.quant(b * 32 + j),
                        gpu.get(off + 4 + j));
            }
        }
    }

    @Test
    public void integerDotMatvecMatchesTheDequantizedActivationReference() throws Exception {
        for (GGMLType type : TYPES) {
            byte[] weights = randomWeights(type, 1000 + type.ordinal());
            float[] x = randomActivations(2000 + type.ordinal());
            float[] expected = reference(type, weights, x);
            ByteBuffer q8 = q8_1(x);

            String kernel = requireKernel(type.name() + "_MatVec_Q8");
            assertClose(api + " " + type + " " + kernel, expected, matvec(weights, q8, kernel, false));

            String addKernel = requireKernel(type.name() + "_MatVecAdd_Q8");
            assertClose(api + " " + type + " " + addKernel, expected, matvec(weights, q8, addKernel, true));
        }
    }

    @Test
    public void fusedGateUpMatchesSiluOfBothRows() throws Exception {
        for (GGMLType type : TYPES) {
            byte[] gate = randomWeights(type, 3000 + type.ordinal());
            byte[] up = randomWeights(type, 4000 + type.ordinal());
            float[] x = randomActivations(5000 + type.ordinal());
            float[] g = reference(type, gate, x);
            float[] u = reference(type, up, x);
            float[] expected = new float[ROWS];
            for (int r = 0; r < ROWS; r++) {
                expected[r] = (float) (g[r] / (1.0 + Math.exp(-g[r]))) * u[r];
            }
            String kernel = requireKernel(type.name() + "_MatVecFusedGateUp_Q8");
            assertClose(api + " " + type + " " + kernel, expected, fusedGateUp(kernel, gate, up, q8_1(x)));
        }
    }

    /**
     * The Q8_0 integer-dot GELU gate/up form, which only the Q8_0 family has next to its F32 twin: the
     * tanh GELU of both rows' dots over the shared Q8_1 activation row.
     */
    @Test
    public void q8_0FusedGateUpGeluMatchesGeluOfBothRows() throws Exception {
        byte[] gate = randomWeights(GGMLType.Q8_0, 3000 + GGMLType.Q8_0.ordinal());
        byte[] up = randomWeights(GGMLType.Q8_0, 4000 + GGMLType.Q8_0.ordinal());
        float[] x = randomActivations(5000 + GGMLType.Q8_0.ordinal());
        float[] g = reference(GGMLType.Q8_0, gate, x);
        float[] u = reference(GGMLType.Q8_0, up, x);
        float[] expected = new float[ROWS];
        for (int r = 0; r < ROWS; r++) {
            float gelu = (float) (0.5 * g[r] * (1.0 + Math.tanh(0.7978845608028654
                    * (g[r] + 0.044715 * g[r] * g[r] * g[r]))));
            expected[r] = gelu * u[r];
        }
        String kernel = requireKernel("Q8_0_MatVecFusedGateUpGelu_Q8");
        assertClose(api + " Q8_0 " + kernel, expected, fusedGateUp(kernel, gate, up, q8_1(x)));
    }

    @Test
    public void sharedActivationRowQuantizesOnceAndKeepsTheProjectionResults() throws Exception {
        long rowBytes = (long) (COLS / GGMLType.Q4_K.getBlockSize()) * GGMLType.Q4_K.getTypeSize();
        GpuWeight a = GpuWeight.of("blk.0.attn_q.weight", GGMLType.Q4_K, new long[]{COLS, ROWS},
                backend.allocateBuffer(ROWS * rowBytes, true));
        GpuWeight b = GpuWeight.of("blk.0.attn_k.weight", GGMLType.Q4_K, new long[]{COLS, ROWS},
                backend.allocateBuffer(ROWS * rowBytes, true));
        GpuBuffer in = upload(randomActivations(61));
        GpuBuffer outA = backend.allocateBuffer((long) ROWS * 4);
        GpuBuffer outB = backend.allocateBuffer((long) ROWS * 4);
        GpuScratch scratch = GpuScratch.builder(backend, 1).slot(ScratchSlots.X, COLS).build();
        GpuPlaceholders placeholders = GpuPlaceholders.allocate(backend, backend.computeQueue());
        try (GpuOps ops = new GpuOps(backend.computeQueue(), kernels, scratch, placeholders, Int8MatvecMode.ON)) {
            ops.matvec(a, in, 0, outA, 0, 1);
            ops.matvec(b, in, 0, outB, 0, 1);
            float[] directA = read(outA);
            float[] directB = read(outB);

            try (GpuOps.SharedQ8 share = ops.shareQ8Activations(in, 0, COLS)) {
                ops.matvec(a, in, 0, outA, 0, 1);
                ops.matvec(b, in, 0, outB, 0, 1);
            }
            backend.computeQueue().finish();
            assertArrayEquals(api + " shared row projection A", directA, read(outA), 0f);
            assertArrayEquals(api + " shared row projection B", directB, read(outB), 0f);
        } finally {
            placeholders.close();
            scratch.close();
            in.close();
            outA.close();
            outB.close();
        }
    }

    private float[] read(GpuBuffer buffer) throws Exception {
        backend.computeQueue().finish();
        float[] values = new float[ROWS];
        backend.computeQueue().enqueueReadBuffer(buffer, 0, (long) ROWS * 4, values).awaitCompletion();
        return values;
    }

    @Test
    public void quantizerWritesOnlyTheWholeBlocksOfTheRequestedLength() throws Exception {
        final float[] x = randomActivations(7);
        final int size = COLS - 16;          // 15 whole 32-element blocks and one trailing half block
        final int blocks = size / 32;
        QuantizedActivations whole = CpuActivationQuantizer.quantizeQ8_1(x, 0, blocks * 32);
        ByteBuffer gpu = q8_1(x, size);
        assertEquals("whole blocks only", blocks * Q8_1_BYTES, gpu.capacity());
        for (int b = 0; b < blocks; b++) {
            int off = b * Q8_1_BYTES;
            assertEquals("block " + b + " d", FpBlockDecoders.fp32ToFp16(whole.scale(b)),
                    gpu.getShort(off) & 0xFFFF);
            for (int j = 0; j < 32; j++) {
                assertEquals("block " + b + " code " + j, whole.quant(b * 32 + j), gpu.get(off + 4 + j));
            }
        }
    }
}

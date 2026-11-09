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
import com.eonmux.neuralfree.gguf.kernels.LlmKernelConstants;
import com.eonmux.neuralfree.gguf.quant.FpBlockDecoders;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.EnumSet;
import java.util.List;
import java.util.Random;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * The multi-column integer-dot matvec kernels ({@code {T}_MatVecN_Q8}) of every supported weight type
 * against a float reference over the CPU-decoded weights and the CPU Q8_1 quantizer's dequantized
 * activations, on every available GPU backend, and against the single-row integer-dot kernels
 * ({@code {T}_MatVec_Q8}) stacked per token row on the same weights &mdash; the lane the projections
 * take for token groups of two to {@link LlmKernelConstants#MATVEC_N_MAX_TOKENS} rows, where one
 * launch reads each weight row once for the whole group. The exact-token-count forms ({@code
 * {T}_MatVecN<n>_Q8}, {@code {T}_MatVec2N<n>_Q8}) meet the same contract at every count their type
 * instantiates. Token counts cover odd groups and the lane's upper bound. Weights are random bytes
 * with finite block scales; the tolerance covers the float summation order of the work-group
 * reduction. A backend is skipped when its GPU or its kernel library is unavailable.
 */
@RunWith(Parameterized.class)
public class MatVecNQ8GpuTest {

    private static final int COLS = 512;   // two 256-element super-blocks (sixteen 32-element blocks)
    private static final int ROWS = 8;
    private static final int TOKENS = 9;   // the speculative verify shape: draft maximum of 8 plus one
    /** Token group sizes the parity test covers: even, odd and the lane's upper bound. */
    private static final int[] TOKEN_COUNTS = {2, 3, 5, LlmKernelConstants.MATVEC_N_MAX_TOKENS};

    /** Every weight type with a {@code {T}_MatVecN_Q8} kernel. */
    private static final GGMLType[] TYPES = {
        GGMLType.Q4_0, GGMLType.Q4_1, GGMLType.Q5_0, GGMLType.Q5_1, GGMLType.Q8_0,
        GGMLType.MXFP4, GGMLType.NVFP4,
        GGMLType.Q2_K, GGMLType.Q3_K, GGMLType.Q4_K, GGMLType.Q5_K, GGMLType.Q6_K, GGMLType.Q8_K,
        GGMLType.IQ4_NL, GGMLType.IQ4_XS,
        GGMLType.IQ2_XXS, GGMLType.IQ2_XS, GGMLType.IQ2_S,
        GGMLType.IQ3_XXS, GGMLType.IQ3_S, GGMLType.IQ3_K, GGMLType.IQ3_KS,
        GGMLType.IQ1_S, GGMLType.IQ1_M,
    };

    /** Every OpenCL device (their subgroup support differs, and with it the reduction form), CUDA and Vulkan. */
    @Parameterized.Parameters(name = "{0}")
    public static List<Object[]> backends() {
        List<Object[]> apis = new ArrayList<>();
        int openClDevices = Math.max(1, openClDeviceCount());
        for (int i = 0; i < openClDevices; i++) {
            apis.add(new Object[]{OPENCL_PREFIX + i});
        }
        apis.add(new Object[]{"cuda"});
        apis.add(new Object[]{"vulkan"});
        return apis;
    }

    private static int openClDeviceCount() {
        HWAcceleration hw = new HWAcceleration(EnumSet.of(ComputeBackend.BackendType.OPENCL));
        try {
            return hw.getCLDevices().size();
        } catch (RuntimeException e) {
            return 0;
        } finally {
            hw.shutdown();
        }
    }

    private static final String OPENCL_PREFIX = "opencl:";

    private final String api;
    private HWAcceleration hwAccel;
    private ComputeBackend backend;
    private LlmKernels kernels;
    private int local;
    private boolean previousCuda;

    public MatVecNQ8GpuTest(String api) {
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
        int index = Integer.parseInt(api.substring(OPENCL_PREFIX.length()));
        HWAcceleration accel = new HWAcceleration();
        hwAccel = accel;
        Assume.assumeTrue("no OpenCL device " + index, index < accel.getCLDevices().size());
        return new OclComputeBackend(accel.getCLDevices().get(index));
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

    /** {@code n} activation rows of {@code cols} elements. */
    private static float[][] randomActivations(long seed, int n) {
        Random rnd = new Random(seed);
        float[][] x = new float[n][COLS];
        for (int t = 0; t < n; t++) {
            for (int i = 0; i < COLS; i++) {
                x[t][i] = rnd.nextFloat() * 2f - 1f;
            }
        }
        return x;
    }

    /**
     * The float reference: decode each weight row and dot it with the CPU Q8_1 quantizer's dequantized
     * activations ({@code d8 * q8}) of every token row, which is what the integer-dot kernels compute
     * from codes. Token-major like the kernel's output layout: element {@code t * rows + r}.
     */
    private static float[] reference(GGMLType type, byte[] weights, float[][] x, int rows) {
        return reference(type, weights, x, rows, new float[x.length * rows]);
    }

    /**
     * {@link #reference} with the L1 magnitude of every row's summed terms in {@code termAbs[t * rows
     * + r]}: the kernel's float reduction rounds against the partial sums' size, which for a
     * strongly cancelling row is far above the row's result, so the per-row tolerance derives from
     * this magnitude instead of the result.
     */
    private static float[] reference(GGMLType type, byte[] weights, float[][] x, int rows,
                                     float[] termAbs) {
        CpuTensorRef w = CpuTensorRef.of(type, ByteBuffer.wrap(weights).order(ByteOrder.LITTLE_ENDIAN),
                rows, COLS);
        float[] y = new float[x.length * rows];
        float[] decoded = new float[COLS];
        for (int t = 0; t < x.length; t++) {
            QuantizedActivations q = CpuActivationQuantizer.quantizeQ8_1(x[t], 0, COLS);
            for (int r = 0; r < rows; r++) {
                w.decodeRow(r, decoded, 0);
                double sum = 0;
                double absTerms = 0;
                for (int j = 0; j < COLS; j++) {
                    double term = (double) decoded[j] * ((double) q.scale(j / 32) * q.quant(j));
                    sum += term;
                    absTerms += Math.abs(term);
                }
                y[t * rows + r] = (float) sum;
                termAbs[t * rows + r] = (float) absTerms;
            }
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

    /**
     * The Q8_1 blocks of {@code n} consecutive activation rows as one {@code LLM_QuantizeQ8_1} launch
     * of the whole group writes them: row {@code t}'s blocks follow row {@code t - 1}'s.
     */
    private GpuBuffer q8_1(float[][] x) throws Exception {
        int n = x.length;
        int blocks = COLS / 32;
        float[] flat = new float[n * COLS];
        for (int t = 0; t < n; t++) {
            System.arraycopy(x[t], 0, flat, t * COLS, COLS);
        }
        GpuBuffer xBuf = upload(flat);
        GpuBuffer qBuf = backend.allocateBuffer((long) n * blocks * GpuOps.Q8_1_FLOATS * 4);
        backend.computeQueue().enqueueKernel(kernels.get(GpuOps.QUANTIZE_Q8_1),
                new long[]{roundUp((long) n * blocks)}, new long[]{local},
                new Object[]{n * COLS, xBuf, qBuf});
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

    /** The multi-column matvec output of {@code kernel} over {@code rows} rows and {@code n} token rows. */
    private float[] matvecN(String kernel, byte[] weights, GpuBuffer q8, int n) throws Exception {
        GpuBuffer wBuf = upload(weights);
        GpuBuffer yBuf = backend.allocateBuffer((long) n * ROWS * 4);
        backend.computeQueue().enqueueKernel(kernels.get(kernel), new long[]{(long) ROWS * local},
                new long[]{local}, new Object[]{ROWS, COLS, n, wBuf, q8, yBuf});
        backend.computeQueue().finish();
        float[] got = new float[n * ROWS];
        backend.computeQueue().enqueueReadBuffer(yBuf, 0, (long) got.length * 4, got).awaitCompletion();
        wBuf.close();
        yBuf.close();
        return got;
    }

    /** The single-row integer-dot matvec output over {@code rows} weight rows and one Q8_1 row. */
    private float[] matvecSingle(String kernel, byte[] weights, GpuBuffer q8) throws Exception {
        GpuBuffer wBuf = upload(weights);
        GpuBuffer yBuf = backend.allocateBuffer((long) ROWS * 4);
        backend.computeQueue().enqueueKernel(kernels.get(kernel), new long[]{(long) ROWS * local},
                new long[]{local}, new Object[]{ROWS, COLS, wBuf, q8, yBuf});
        backend.computeQueue().finish();
        float[] got = new float[ROWS];
        backend.computeQueue().enqueueReadBuffer(yBuf, 0, (long) ROWS * 4, got).awaitCompletion();
        wBuf.close();
        yBuf.close();
        return got;
    }

    /** The counted multi-column matvec output ({@code kernel}'s token count is compiled in). */
    private float[] matvecCounted(String kernel, byte[] weights, GpuBuffer q8, int n) throws Exception {
        GpuBuffer wBuf = upload(weights);
        GpuBuffer yBuf = backend.allocateBuffer((long) n * ROWS * 4);
        backend.computeQueue().enqueueKernel(kernels.get(kernel), new long[]{(long) ROWS * local},
                new long[]{local}, new Object[]{ROWS, COLS, wBuf, q8, yBuf});
        backend.computeQueue().finish();
        float[] got = new float[n * ROWS];
        backend.computeQueue().enqueueReadBuffer(yBuf, 0, (long) got.length * 4, got).awaitCompletion();
        wBuf.close();
        yBuf.close();
        return got;
    }

    /**
     * The paired counted multi-column matvec output: two weight rows per workgroup over {@code n}
     * token rows ({@code kernel}'s token count is compiled in), global = ceil(ROWS / 2) * local.
     */
    private float[] matvecPairedCounted(String kernel, byte[] weights, GpuBuffer q8, int n) throws Exception {
        GpuBuffer wBuf = upload(weights);
        GpuBuffer yBuf = backend.allocateBuffer((long) n * ROWS * 4);
        long groups = (ROWS + 1) / 2;
        backend.computeQueue().enqueueKernel(kernels.get(kernel), new long[]{groups * local},
                new long[]{local}, new Object[]{ROWS, COLS, wBuf, q8, yBuf});
        backend.computeQueue().finish();
        float[] got = new float[n * ROWS];
        backend.computeQueue().enqueueReadBuffer(yBuf, 0, (long) got.length * 4, got).awaitCompletion();
        wBuf.close();
        yBuf.close();
        return got;
    }

    /**
     * Per-row reference check: the tolerance scales with the row's summed-term magnitude, which bounds
     * the float rounding of the kernels' factored integer dots (the d-term and min-term of one slice
     * resolve their cancellation in a float subtraction of products much larger than the row's result).
     */
    private static void assertClose(String what, float[] expected, float[] termAbs, float[] got) {
        assertEquals(what + " length", expected.length, got.length);
        for (int r = 0; r < expected.length; r++) {
            assertEquals(what + " row " + r + " (termAbs " + termAbs[r] + ", delta "
                            + Math.abs(expected[r] - got[r]) + ")",
                    expected[r], got[r], 1e-5 + 5e-4 * termAbs[r]);
        }
    }

    // ------------------------------------------------------------------ tests

    @Test
    public void multiColumnMatvecMatchesTheDequantizedActivationReference() throws Exception {
        for (GGMLType type : TYPES) {
            byte[] weights = randomWeights(type, 30000 + type.ordinal());
            float[][] x = randomActivations(31000 + type.ordinal(), TOKENS);
            float[] termAbs = new float[TOKENS * ROWS];
            float[] expected = reference(type, weights, x, ROWS, termAbs);
            GpuBuffer q8 = q8_1(x);
            try {
                String kernel = requireKernel(type.name() + "_MatVecN_Q8");
                assertClose(api + " " + type + " " + kernel, expected, termAbs,
                        matvecN(kernel, weights, q8, TOKENS));
            } finally {
                q8.close();
            }
        }
    }

    @Test
    public void oddTokenGroupsMatchTheStackedSingleRowMatvecs() throws Exception {
        for (GGMLType type : TYPES) {
            for (int n : TOKEN_COUNTS) {
                byte[] weights = randomWeights(type, 32000 + type.ordinal());
                float[][] x = randomActivations(33000 + 100 * n + type.ordinal(), n);
                float[] termAbs = new float[n * ROWS];
                float[] expected = reference(type, weights, x, ROWS, termAbs);
                GpuBuffer q8 = q8_1(x);
                try {
                    String kernel = requireKernel(type.name() + "_MatVecN_Q8");
                    String single = requireKernel(type.name() + "_MatVec_Q8");
                    float[] got = matvecN(kernel, weights, q8, n);
                    for (int t = 0; t < n; t++) {
                        GpuBuffer row = q8_1(new float[][]{x[t]});
                        try {
                            float[] singleGot = matvecSingle(single, weights, row);
                            for (int r = 0; r < ROWS; r++) {
                                assertEquals(api + " " + type + " n=" + n + " token " + t + " row " + r
                                                + " vs stacked single-row " + single,
                                        got[t * ROWS + r], singleGot[r], 1e-5 + 2e-4
                                                * (Math.abs(got[t * ROWS + r]) + Math.abs(singleGot[r]) + 1f));
                            }
                        } finally {
                            row.close();
                        }
                    }
                    assertClose(api + " " + type + " n=" + n + " reference", expected, termAbs, got);
                } finally {
                    q8.close();
                }
            }
        }
    }

    /**
     * The one-launch multi-column output equals the stacked single-row outputs bit for bit: token
     * t's partial sums walk the same weight blocks in the same order as the single-row kernel and
     * reduce through the exact same steps, so any reordering shows as a differing bit pattern (the
     * documented contract of the multi-column form). The Q8_1 rows are the group quantizer's own
     * blocks, sliced per token, so the comparison isolates the matvec kernels.
     */
    @Test
    public void multiColumnMatvecIsBitwiseTheStackedSingleRowMatvecs() throws Exception {
        for (GGMLType type : TYPES) {
            String kernel = requireKernel(type.name() + "_MatVecN_Q8");
            String single = requireKernel(type.name() + "_MatVec_Q8");
            for (int n : TOKEN_COUNTS) {
                byte[] weights = randomWeights(type, 32000 + type.ordinal());
                float[][] x = randomActivations(33000 + 100 * n + type.ordinal(), n);
                GpuBuffer q8 = q8_1(x);
                try {
                    int rowBytes = (COLS / 32) * GpuOps.Q8_1_FLOATS * 4;
                    byte[] groupBlocks = new byte[n * rowBytes];
                    backend.computeQueue().enqueueReadBuffer(q8, 0, groupBlocks.length, groupBlocks)
                            .awaitCompletion();
                    float[] got = matvecN(kernel, weights, q8, n);
                    for (int t = 0; t < n; t++) {
                        GpuBuffer row = upload(Arrays.copyOfRange(groupBlocks, t * rowBytes, (t + 1) * rowBytes));
                        try {
                            float[] singleGot = matvecSingle(single, weights, row);
                            for (int r = 0; r < ROWS; r++) {
                                if (got[t * ROWS + r] != singleGot[r]) {
                                    assertArrayEquals(api + " " + type + " n=" + n + " token " + t
                                                    + " row " + r + " bitwise vs stacked " + single,
                                            new float[]{singleGot[r]}, new float[]{got[t * ROWS + r]}, 0f);
                                }
                            }
                        } finally {
                            row.close();
                        }
                    }
                } finally {
                    q8.close();
                }
            }
        }
    }

    /**
     * The exact-token-count forms of every served type &mdash; {@code {T}_MatVecN<n>_Q8} (the generic
     * kernel with the token count folded to compile time) and the paired {@code {T}_MatVec2N<n>_Q8}
     * (two weight rows per workgroup) &mdash; produce the stacked single-row outputs bit for bit at
     * their own token count, the runtime-count form's contract, at every count the type instantiates
     * ({@link LlmKernelConstants#countedMaxTokens}): [2, {@link LlmKernelConstants#MATVEC_N_DISPATCH_MAX}]
     * for most family types and up to {@link LlmKernelConstants#MATVEC_N_COUNTED_MAX} for the types
     * whose measured dispatch serves the counts above it (Q8_0, and the K-quant, Q4_0 and IQ4_XS
     * types on the kinds whose entries serve them). Each (type, count) quantizes once, computes the
     * single-row outputs once and holds them against both forms.
     */
    @Test
    public void countedQ8MatvecFormsAreBitwiseTheStackedSingleRowMatvecs() throws Exception {
        for (GGMLType type : TYPES) {
            String single = requireKernel(type.name() + "_MatVec_Q8");
            int maxCount = LlmKernelConstants.countedMaxTokens(type);
            for (int n = 2; n <= maxCount; n++) {
                byte[] weights = randomWeights(type, 32000 + type.ordinal());
                float[][] x = randomActivations(33000 + 100 * n + type.ordinal(), n);
                GpuBuffer q8 = q8_1(x);
                try {
                    int rowBytes = (COLS / 32) * GpuOps.Q8_1_FLOATS * 4;
                    byte[] groupBlocks = new byte[n * rowBytes];
                    backend.computeQueue().enqueueReadBuffer(q8, 0, groupBlocks.length, groupBlocks)
                            .awaitCompletion();
                    float[][] singleGot = new float[n][];
                    for (int t = 0; t < n; t++) {
                        GpuBuffer row = upload(Arrays.copyOfRange(groupBlocks, t * rowBytes,
                                (t + 1) * rowBytes));
                        try {
                            singleGot[t] = matvecSingle(single, weights, row);
                        } finally {
                            row.close();
                        }
                    }
                    for (int form = 0; form < 2; form++) {
                        String label = form == 0 ? "_MatVecN" : "_MatVec2N";
                        String counted = requireKernel(type.name() + label + n + "_Q8");
                        float[] got = form == 0 ? matvecCounted(counted, weights, q8, n)
                                : matvecPairedCounted(counted, weights, q8, n);
                        for (int t = 0; t < n; t++) {
                            for (int r = 0; r < ROWS; r++) {
                                if (got[t * ROWS + r] != singleGot[t][r]) {
                                    assertArrayEquals(api + " " + type + " " + label + n + " token " + t
                                                    + " row " + r + " bitwise vs stacked " + single,
                                            new float[]{singleGot[t][r]},
                                            new float[]{got[t * ROWS + r]}, 0f);
                                }
                            }
                        }
                    }
                } finally {
                    q8.close();
                }
            }
        }
    }

    @Test
    public void oneQuantizerLaunchWritesTheSameBlocksAsPerRowLaunches() throws Exception {
        for (int n : new int[]{TOKENS, LlmKernelConstants.MATVEC_N_MAX_TOKENS}) {
            float[][] x = randomActivations(34000, n);
            int blocks = COLS / 32;
            GpuBuffer grouped = q8_1(x);
            try {
                byte[] groupedBytes = new byte[n * blocks * GpuOps.Q8_1_FLOATS * 4];
                backend.computeQueue().enqueueReadBuffer(grouped, 0, groupedBytes.length, groupedBytes)
                        .awaitCompletion();
                for (int t = 0; t < n; t++) {
                    GpuBuffer row = q8_1(new float[][]{x[t]});
                    try {
                        byte[] rowBytes = new byte[blocks * GpuOps.Q8_1_FLOATS * 4];
                        backend.computeQueue().enqueueReadBuffer(row, 0, rowBytes.length, rowBytes)
                                .awaitCompletion();
                        assertArrayEquals(api + " n=" + n + " token " + t + " blocks", rowBytes,
                                Arrays.copyOfRange(groupedBytes, t * rowBytes.length, (t + 1) * rowBytes.length));
                    } finally {
                        row.close();
                    }
                }
            } finally {
                grouped.close();
            }
        }
    }
}

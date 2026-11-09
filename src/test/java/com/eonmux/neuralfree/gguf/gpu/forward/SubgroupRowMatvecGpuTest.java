package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.cuda.CudaComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ocl.OclComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.vulkan.VkComputeBackend;
import com.eonmux.neuralfree.acceleration.kernels.Availability;
import com.eonmux.neuralfree.acceleration.kernels.OpenClKernelLibrary;
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
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.List;
import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * The subgroup-row integer-dot matvec forms ({@code {T}_MatVecSg_Q8}, {@code {T}_MatVecAddSg_Q8},
 * {@code {T}_MatVecFusedGateUpSg_Q8}, {@link SubgroupRowMatvec}) against a float reference over the CPU-decoded
 * weights and the CPU Q8_1 quantizer's dequantized activations, on every OpenCL device whose program builds
 * them ({@link OpenClKernelLibrary#subgroupRowSize} positive). Shapes cover a single row, row counts that end
 * inside a subgroup and inside a work-group, rows of an odd Q6_K block count (whose odd rows and odd blocks start
 * 2 bytes off a word) and the 2560-column rows of a 4B decoder. Every other device and backend must lack the forms,
 * so the projections keep the family there. The tolerance derives from the L1 magnitude of each row's terms,
 * which bounds the float summation-order error.
 */
@RunWith(Parameterized.class)
public class SubgroupRowMatvecGpuTest {

    private static final int[] ROW_COUNTS = {1, 3, 13};
    private static final int[] COL_COUNTS = {256, 1280, 2560};
    private static final GGMLType[] TYPES = {GGMLType.Q8_0, GGMLType.Q4_K, GGMLType.Q6_K};
    private static final String OPENCL_PREFIX = "opencl:";

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

    private final String api;
    private HWAcceleration hwAccel;
    private ComputeBackend backend;
    private LlmKernels kernels;
    private int local;
    private boolean previousCuda;

    public SubgroupRowMatvecGpuTest(String api) {
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
            return new CudaComputeBackend(accel.getCUDevices().get(0));
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

    /** The forms exist exactly where the OpenCL program defines NF_SG_ROW_SIZE. */
    @Test
    public void formsExistOnlyWhereTheProgramBuildsThem() {
        boolean expected = backend.backendType() == ComputeBackend.BackendType.OPENCL
                && OpenClKernelLibrary.subgroupRowSize(backend.capabilities(),
                        OpenClKernelLibrary.useSubgroups(backend.capabilities())) > 0;
        for (GGMLType type : TYPES) {
            for (String form : new String[]{"MatVec", "MatVecAdd", "MatVecFusedGateUp"}) {
                String name = SubgroupRowMatvec.twinName(type.name() + "_" + form + "_Q8", type);
                assertEquals(api + ": " + name, expected, kernels.has(name));
            }
        }
    }

    @Test
    public void matvecMatchesTheReference() throws Exception {
        assumeForms();
        for (GGMLType type : TYPES) {
            for (int cols : COL_COUNTS) {
                if (!SubgroupRowMatvec.shapeServes(type, cols)) {
                    continue;
                }
                for (int rows : ROW_COUNTS) {
                    byte[] w = randomWeights(type, rows, cols, 31L * rows + cols);
                    float[] x = randomActivations(cols, 7L + cols);
                    float[] termAbs = new float[rows];
                    float[] want = reference(type, w, rows, cols, x, termAbs);
                    float[] seed = new float[rows];
                    float[] got = run(type.name() + "_MatVecSg_Q8", rows, cols, x, seed, w);
                    assertClose(type + " " + rows + "x" + cols + " MatVecSg", want, got, termAbs);
                    Random rnd = new Random(rows * 13L + cols);
                    for (int r = 0; r < rows; r++) {
                        seed[r] = rnd.nextFloat() * 4f - 2f;
                    }
                    float[] added = run(type.name() + "_MatVecAddSg_Q8", rows, cols, x, seed, w);
                    float[] wantAdded = new float[rows];
                    for (int r = 0; r < rows; r++) {
                        wantAdded[r] = want[r] + seed[r];
                    }
                    assertClose(type + " " + rows + "x" + cols + " MatVecAddSg", wantAdded, added, termAbs);
                }
            }
        }
    }

    @Test
    public void fusedGateUpMatchesTheReference() throws Exception {
        assumeForms();
        for (GGMLType type : TYPES) {
            for (int cols : COL_COUNTS) {
                if (!SubgroupRowMatvec.shapeServes(type, cols)) {
                    continue;
                }
                for (int rows : ROW_COUNTS) {
                    byte[] gate = randomWeights(type, rows, cols, 101L * rows + cols);
                    byte[] up = randomWeights(type, rows, cols, 103L * rows + cols);
                    float[] x = randomActivations(cols, 11L + cols);
                    float[] gAbs = new float[rows];
                    float[] uAbs = new float[rows];
                    float[] g = reference(type, gate, rows, cols, x, gAbs);
                    float[] u = reference(type, up, rows, cols, x, uAbs);
                    float[] got = run(type.name() + "_MatVecFusedGateUpSg_Q8", rows, cols, x, new float[rows],
                            gate, up);
                    for (int r = 0; r < rows; r++) {
                        double silu = g[r] / (1.0 + Math.exp(-g[r]));
                        double tol = 2e-4 * (gAbs[r] + 1.0) * (uAbs[r] + 1.0);
                        assertEquals(type + " " + rows + "x" + cols + " FusedGateUpSg row " + r,
                                silu * u[r], got[r], tol);
                    }
                }
            }
        }
    }

    private void assumeForms() {
        Assume.assumeTrue(api + " builds no subgroup-row forms", kernels.has("Q8_0_MatVecSg_Q8"));
    }

    private static void assertClose(String what, float[] want, float[] got, float[] termAbs) {
        for (int r = 0; r < want.length; r++) {
            assertEquals(what + " row " + r, want[r], got[r], 1e-4 * termAbs[r] + 1e-5);
        }
    }

    /**
     * One launch of {@code kernel} over {@code rows} rows with the Q8_1 blocks of {@code x}; the output buffer
     * starts as {@code seed} (the accumulate form adds into it). {@code weights} are one or two weight matrices.
     */
    private float[] run(String kernel, int rows, int cols, float[] x, float[] seed, byte[]... weights)
            throws Exception {
        assertTrue(api + " has no " + kernel, kernels.has(kernel));
        List<GpuBuffer> owned = new ArrayList<>();
        try {
            GpuBuffer xq = q8_1(x, owned);
            GpuBuffer y = upload(seed, owned);
            Object[] args = new Object[weights.length + 4];
            args[0] = rows;
            args[1] = cols;
            for (int i = 0; i < weights.length; i++) {
                args[2 + i] = upload(weights[i], owned);
            }
            args[2 + weights.length] = xq;
            args[3 + weights.length] = y;
            backend.computeQueue().enqueueKernel(kernels.get(kernel),
                    new long[]{SubgroupRowMatvec.globalSize(rows), 1},
                    new long[]{SubgroupRowMatvec.WORK_GROUP, 1}, args);
            backend.computeQueue().finish();
            float[] out = new float[rows];
            backend.computeQueue().enqueueReadBuffer(y, 0, (long) rows * 4, out).awaitCompletion();
            return out;
        } finally {
            for (GpuBuffer b : owned) {
                b.close();
            }
        }
    }

    private GpuBuffer q8_1(float[] x, List<GpuBuffer> owned) throws Exception {
        int blocks = x.length / 32;
        GpuBuffer xBuf = upload(x, owned);
        GpuBuffer qBuf = backend.allocateBuffer((long) blocks * GpuOps.Q8_1_FLOATS * 4);
        owned.add(qBuf);
        backend.computeQueue().enqueueKernel(kernels.get(GpuOps.QUANTIZE_Q8_1),
                new long[]{(blocks + local - 1) / local * (long) local}, new long[]{local},
                new Object[]{x.length, xBuf, qBuf});
        backend.computeQueue().finish();
        return qBuf;
    }

    private GpuBuffer upload(byte[] data, List<GpuBuffer> owned) throws Exception {
        GpuBuffer buf = backend.allocateBuffer(data.length);
        owned.add(buf);
        backend.computeQueue().enqueueWriteBuffer(buf, 0, data.length, data).awaitCompletion();
        return buf;
    }

    private GpuBuffer upload(float[] data, List<GpuBuffer> owned) throws Exception {
        GpuBuffer buf = backend.allocateBuffer((long) data.length * 4);
        owned.add(buf);
        backend.computeQueue().enqueueWriteBuffer(buf, 0, (long) data.length * 4, data).awaitCompletion();
        return buf;
    }

    /** Random blocks of {@code type} with finite block scales. */
    private static byte[] randomWeights(GGMLType type, int rows, int cols, long seed) {
        Random rnd = new Random(seed);
        int blockBytes = type.getTypeSize();
        int blocks = rows * (cols / type.getBlockSize());
        byte[] data = new byte[blocks * blockBytes];
        rnd.nextBytes(data);
        for (int b = 0; b < blocks; b++) {
            int base = b * blockBytes;
            if (type == GGMLType.Q6_K) {
                putHalf(data, base + 208, 0.01f);
            } else if (type == GGMLType.Q4_K) {
                putHalf(data, base, 0.02f);
                putHalf(data, base + 2, 0.01f);
            } else {
                putHalf(data, base, 0.02f);
            }
        }
        return data;
    }

    private static void putHalf(byte[] data, int off, float value) {
        int bits = FpBlockDecoders.fp32ToFp16(value);
        data[off] = (byte) bits;
        data[off + 1] = (byte) (bits >>> 8);
    }

    private static float[] randomActivations(int cols, long seed) {
        Random rnd = new Random(seed);
        float[] x = new float[cols];
        for (int i = 0; i < cols; i++) {
            x[i] = rnd.nextFloat() * 2f - 1f;
        }
        return x;
    }

    /** Decoded weight rows dotted with the CPU Q8_1 quantizer's dequantized activations, with each row's L1 term sum. */
    private static float[] reference(GGMLType type, byte[] weights, int rows, int cols, float[] x, float[] termAbs) {
        CpuTensorRef w = CpuTensorRef.of(type, ByteBuffer.wrap(weights).order(ByteOrder.LITTLE_ENDIAN), rows, cols);
        QuantizedActivations q = CpuActivationQuantizer.quantizeQ8_1(x, 0, cols);
        float[] decoded = new float[cols];
        float[] y = new float[rows];
        for (int r = 0; r < rows; r++) {
            w.decodeRow(r, decoded, 0);
            double sum = 0;
            double abs = 0;
            for (int j = 0; j < cols; j++) {
                double term = (double) decoded[j] * ((double) q.scale(j / 32) * q.quant(j));
                sum += term;
                abs += Math.abs(term);
            }
            y[r] = (float) sum;
            termAbs[r] = (float) abs;
        }
        return y;
    }
}

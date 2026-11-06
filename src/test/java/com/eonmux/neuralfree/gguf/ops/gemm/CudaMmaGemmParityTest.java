package com.eonmux.neuralfree.gguf.ops.gemm;

import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.acceleration.backend.cuda.CudaComputeBackend;
import com.eonmux.neuralfree.acceleration.cuda.CUDevice;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import static org.junit.Assert.assertTrue;

/**
 * The MMA and split-K GEMM families on the CUDA device, against exact references: the test builds
 * dense 16-bit weights and activations whose values are exact in fp16 (or bf16), so every product of
 * the tensor-core path (16-bit multiply, fp32 accumulate) is exact and the kernel must match the
 * fp64 dot up to fp32 accumulation rounding; the int8 MMQ GEMM runs the {@code mma.m16n8k32.s8}
 * tensor path on sm_80+ (dp4a below) and is checked against the CPU decoder with the activation
 * quantization of the arch's stage depth; the split-K twin must reproduce the plain kernel's planes
 * summed by {@code LLM_SplitK_Reduce}. Skipped on devices below CUDA sm_80, where the MMA kernels
 * take their scalar path.
 */
public class CudaMmaGemmParityTest {

    private static final int M = 70;   /* two 64-row tiles, the second partial */
    private static final int K = 96;   /* six 16-deep stages, one partial */
    private static final int N = 40;   /* two 32-token tiles, the second partial */

    private HWAcceleration hwAccel;
    private CudaComputeBackend backend;
    private ComputeQueue queue;
    private LlmKernels kernels;
    private DeviceCapabilities caps;

    @Before
    public void setUp() {
        HWAcceleration.enableCuda = true;
        HWAcceleration hw = null;
        try {
            hw = new HWAcceleration();
            Assume.assumeTrue("no CUDA device", !hw.getCUDevices().isEmpty());
            CUDevice device = hw.getCUDevices().get(0);
            CudaComputeBackend b = new CudaComputeBackend(device);
            com.eonmux.neuralfree.acceleration.kernels.Availability availability = LlmKernels.availability(b);
            Assume.assumeTrue("LLM kernels unavailable on " + b.capabilities().getDeviceName() + ": "
                    + availability.reason(), availability.available());
            hwAccel = hw;
            backend = b;
            queue = b.computeQueue();
            kernels = LlmKernels.open(b);
            caps = b.capabilities();
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
    public void mmaGemmMatchesTheExactFp16ProductDot() throws Exception {
        Assume.assumeTrue("MMA GEMM needs CUDA sm_80+; device is sm_" + caps.getComputeCapability(),
                caps.getComputeCapability() >= 80);
        MmaGemmPlan plan = MmaGemmPlan.contiguous(GGMLType.F16, M, K, N);
        short[] weight = buildWeight();
        float[] x = buildX();

        float[] out = launch(plan, halfBytes(weight));
        double[] q = new double[K];
        for (int t = 0; t < N; t++) {
            toHalf(x, t * K, q);
            for (int r = 0; r < M; r++) {
                double acc = 0.0;
                double terms = 0.0;
                for (int k = 0; k < K; k++) {
                    double w = halfBitsToFloat(weight[r * K + k]);
                    acc += w * q[k];
                    terms += Math.abs(w * q[k]);
                }
                double tol = 1e-4 * terms + 1e-2;
                assertTrue("token " + t + " row " + r + ": kernel " + out[t * M + r] + " expected " + acc
                                + " (|terms| " + terms + ")",
                        Math.abs(out[t * M + r] - acc) <= tol);
            }
        }
    }

    /**
     * The BF16 row of the MMA family: bf16 products are exact in fp32 (8-bit mantissas), so the
     * kernel must match the exact dot of the bf16-decoded weight with the bf16-rounded activations
     * up to fp32 accumulation rounding, as the F16 test does for its format.
     */
    @Test
    public void bf16MmaGemmMatchesTheExactBf16ProductDot() throws Exception {
        Assume.assumeTrue("the bf16 mma needs CUDA sm_80+; device is sm_" + caps.getComputeCapability(),
                caps.getComputeCapability() >= 80);
        MmaGemmPlan plan = MmaGemmPlan.contiguous(GGMLType.BF16, M, K, N);
        short[] weight = buildWeight();
        float[] x = buildX();

        float[] out = launch(plan, bf16Bytes(weight));
        double[] q = new double[K];
        for (int t = 0; t < N; t++) {
            for (int k = 0; k < K; k++) {
                q[k] = bf16BitsToFloat(floatToBf16Bits(x[t * K + k]));
            }
            for (int r = 0; r < M; r++) {
                double acc = 0.0;
                double terms = 0.0;
                for (int k = 0; k < K; k++) {
                    double w = halfBitsToFloat(weight[r * K + k]);
                    acc += w * q[k];
                    terms += Math.abs(w * q[k]);
                }
                double tol = 1e-4 * terms + 1e-2;
                assertTrue("bf16 token " + t + " row " + r + ": kernel " + out[t * M + r] + " expected " + acc
                                + " (|terms| " + terms + ")",
                        Math.abs(out[t * M + r] - acc) <= tol);
            }
        }
    }

    /**
     * The int8 MMQ GEMM over Q8_0 on the CUDA device: the m16n8k32.s8 tensor path on sm_80+ (dp4a
     * below), checked against the CPU decoder with the activation quantization of the arch's stage
     * depth ({@code NF_MMQ_TK}: 32 tensor, 16 packed-dot), as in {@code OpenClMmqGemmParityTest}.
     */
    @Test
    public void mmqGemmMatchesTheDecodedWeightDotProduct() throws Exception {
        GGMLType type = GGMLType.Q8_0;
        MmqConfig config = MmqConfig.forDevice(backend.backendType(), caps.getComputeCapability());
        int chunk = config.stageDepth();
        int m = M, k = 256, n = 21;
        ByteBuffer b = ByteBuffer.allocate(m * 8 * 34).order(ByteOrder.LITTLE_ENDIAN);
        Bits bits = new Bits(0x77AA);
        for (int r = 0; r < m; r++) {
            for (int blk = 0; blk < 8; blk++) {
                putF16(b, 0.5f + 0.25f * (r % 5));
                for (int e = 0; e < 32; e++) {
                    b.put((byte) (bits.next(256) - 128));
                }
            }
        }
        byte[] weight = b.array();
        float[] x = new float[n * k];
        for (int i = 0; i < x.length; i++) {
            x[i] = 4.0f * (bits.next(1024) - 512) / 1024.0f;
        }
        MmqGemmPlan plan = MmqGemmPlan.contiguous(config, type, m, k, n);
        float[] w = com.eonmux.neuralfree.gguf.quant.CpuBlockDecoders.dequantize(weight, type, (long) m * k);
        float[] out = launch(plan, weight, x);
        double[] q = new double[k];
        double[] d = new double[k];
        for (int t = 0; t < n; t++) {
            quantizeChunked(x, t * k, chunk, q, d);
            for (int r = 0; r < m; r++) {
                double acc = 0.0;
                double terms = 0.0;
                double bound = 0.0;
                for (int e = 0; e < k; e++) {
                    double term = (double) w[r * k + e] * q[e];
                    acc += term;
                    terms += Math.abs(term);
                    /* A backend's activation-quantization division (CUDA fast-math, GLSL ULP rules) may
                     * round one code up or down against this reference; one code step per element. */
                    bound += Math.abs((double) w[r * k + e] * d[e]);
                }
                double tol = bound + 1e-4 * terms + 1e-3;
                assertTrue("token " + t + " row " + r + ": kernel " + out[t * m + r] + " expected " + acc
                                + " (|terms| " + terms + ", delta " + Math.abs(out[t * m + r] - acc) + ")",
                        Math.abs(out[t * m + r] - acc) <= tol);
            }
        }
    }

    /**
     * The split-K twin over three k-slices (one ragged tail) against the plain kernel: every slice
     * covers whole 16-deep stages, the reduce sums the planes in slice order, and the fp32 plane-sum
     * may round differently from the plain kernel's sequential accumulation by fp32 addition only.
     */
    @Test
    public void splitKGemmReproducesThePlainKernelWithinFp32Summation() throws Exception {
        Assume.assumeTrue("split-K MMA needs CUDA sm_80+; device is sm_" + caps.getComputeCapability(),
                caps.getComputeCapability() >= 80);
        int splits = 3;
        int k = 2425;                       /* ceil(2425/3) = 809 -> 816: the third slice is ragged */
        int kSlice = SplitKGemmPlan.sliceLength(k, splits);
        assertTrue((long) kSlice * (splits - 1) < k);
        SplitKGemmPlan plan = SplitKGemmPlan.of(GGMLType.F16, M, k, N, k, 0, splits, kSlice, 256);
        MmaGemmPlan plain = MmaGemmPlan.contiguous(GGMLType.F16, M, k, N);
        short[] weight = buildWeight(k);
        float[] x = buildX(k);

        float[] split = launchSplit(plan, halfBytes(weight), x);
        float[] plainOut = launch(plain, halfBytes(weight));
        double[] q = new double[k];
        toHalf(x, 0, q);
        for (int t = 0; t < N; t++) {
            for (int r = 0; r < M; r++) {
                double terms = 0.0;
                for (int i = 0; i < k; i++) {
                    terms += Math.abs(halfBitsToFloat(weight[r * k + i]) * q[i]);
                }
                double tol = 1e-3 * terms + 1e-2;
                assertTrue("split token " + t + " row " + r + ": kernel " + split[t * M + r] + " plain "
                                + plainOut[t * M + r] + " (|terms| " + terms + ")",
                        Math.abs(split[t * M + r] - plainOut[t * M + r]) <= tol);
            }
        }
    }

    /**
     * The kernel's activation quantization: per {@code chunk}-deep block, d = maxabs * (1f/127) and
     * the codes round to nearest even of x * (127f / maxabs); writes the code values d * q and the
     * block scales d (the per-code uncertainty bound).
     */
    private static void quantizeChunked(float[] x, int base, int chunk, double[] out, double[] dOut) {
        for (int c = 0; c < 256; c += chunk) {
            float maxabs = 0.0f;
            for (int e = c; e < c + chunk; e++) {
                maxabs = Math.max(maxabs, Math.abs(x[base + e]));
            }
            float id = maxabs > 0.0f ? 127.0f / maxabs : 0.0f;
            float d = maxabs > 0.0f ? maxabs * (1.0f / 127.0f) : 1.0f;
            for (int e = c; e < c + chunk; e++) {
                out[e] = (double) d * Math.rint((double) (x[base + e] * id));
                dOut[e] = d;
            }
        }
    }

    private static void putF16(ByteBuffer b, float v) {
        b.putShort(floatToHalfBits(v));
    }

    private float[] launch(MmqGemmPlan plan, byte[] weight, float[] x) throws Exception {
        GpuBuffer wbuf = backend.allocateBuffer(plan.weightBytes(), true);
        GpuBuffer xbuf = backend.allocateBuffer((long) x.length * Float.BYTES, true);
        GpuBuffer qbuf = backend.allocateBuffer(plan.q8ElementsRequired() * Integer.BYTES);
        GpuBuffer ybuf = backend.allocateBuffer(plan.yElementsRequired() * Float.BYTES);
        try {
            try (GpuEvent wv = queue.enqueueWriteBuffer(wbuf, 0, weight.length, weight);
                 GpuEvent xv = queue.enqueueWriteBuffer(xbuf, 0, (long) x.length * Float.BYTES, x, wv)) {
                int tk = plan.config().stageDepth();
                long records = (long) plan.n() * plan.config().rowRecords(plan.k());
                Object[] quantArgs = {plan.n(), plan.k(), tk, plan.k(), 0, xbuf, qbuf};
                queue.enqueueKernelNoEvent(kernels.get(GemmKernels.QUANT_MMQ_Q8_1),
                        new long[]{((records + 63) / 64) * 64}, new long[]{64}, quantArgs, xv);
                Object[] args = plan.args(wbuf, qbuf, ybuf);
                queue.enqueueKernelNoEvent(kernels.get(plan.kernelName()), plan.globalWorkSize(),
                        plan.localWorkSize(), args);
                float[] out = new float[(int) plan.yElementsRequired()];
                try (GpuEvent read = queue.enqueueReadBuffer(ybuf, 0, out.length * Float.BYTES, out)) {
                    read.awaitCompletion();
                }
                return out;
            }
        } finally {
            wbuf.close();
            xbuf.close();
            qbuf.close();
            ybuf.close();
        }
    }

    /** Writes each float rounded to nearest even into fp16 bits, as the kernel's fp32-to-fp16 tile conversion. */
    private static void toHalf(float[] x, int base, double[] out) {
        for (int k = 0; k < K; k++) {
            out[k] = halfBitsToFloat(floatToHalfBits(x[base + k]));
        }
    }

    private float[] launch(MmaGemmPlan plan, byte[] weightBytes) throws Exception {
        GpuBuffer wbuf = backend.allocateBuffer(plan.weightBytes(), true);
        GpuBuffer xbuf = backend.allocateBuffer(plan.xElementsRequired() * Float.BYTES, true);
        GpuBuffer ybuf = backend.allocateBuffer(plan.yElementsRequired() * Float.BYTES);
        try {
            float[] x = buildX(plan.k());
            try (GpuEvent wv = queue.enqueueWriteBuffer(wbuf, 0, weightBytes.length, weightBytes);
                 GpuEvent xv = queue.enqueueWriteBuffer(xbuf, 0, x.length * Float.BYTES, x, wv)) {
                Object[] args = plan.args(wbuf, xbuf, ybuf);
                queue.enqueueKernelNoEvent(kernels.get(plan.kernelName()), plan.globalWorkSize(),
                        plan.localWorkSize(), args, xv);
                float[] out = new float[(int) plan.yElementsRequired()];
                try (GpuEvent read = queue.enqueueReadBuffer(ybuf, 0, out.length * Float.BYTES, out)) {
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

    private float[] launchSplit(SplitKGemmPlan plan, byte[] weightBytes, float[] x) throws Exception {
        GpuBuffer wbuf = backend.allocateBuffer(plan.weightBytes(), true);
        GpuBuffer xbuf = backend.allocateBuffer(plan.xElementsRequired() * Float.BYTES, true);
        GpuBuffer pbuf = backend.allocateBuffer(plan.partialElements() * Float.BYTES);
        GpuBuffer ybuf = backend.allocateBuffer(plan.yElementsRequired(plan.m(), 0) * Float.BYTES);
        try {
            try (GpuEvent wv = queue.enqueueWriteBuffer(wbuf, 0, weightBytes.length, weightBytes);
                 GpuEvent xv = queue.enqueueWriteBuffer(xbuf, 0, x.length * Float.BYTES, x, wv)) {
                Object[] partial = plan.partialArgs(wbuf, xbuf, pbuf);
                queue.enqueueKernelNoEvent(kernels.get(plan.kernelName()), plan.partialGlobalWorkSize(),
                        plan.partialLocalWorkSize(), partial, xv);
                queue.enqueueKernelNoEvent(kernels.get(GemmKernels.SPLIT_K_REDUCE), plan.reduceGlobalWorkSize(),
                        plan.reduceLocalWorkSize(), plan.reduceArgs(pbuf, ybuf, plan.m(), 0));
                float[] out = new float[(int) plan.yElementsRequired(plan.m(), 0)];
                try (GpuEvent read = queue.enqueueReadBuffer(ybuf, 0, out.length * Float.BYTES, out)) {
                    read.awaitCompletion();
                }
                return out;
            }
        } finally {
            wbuf.close();
            xbuf.close();
            pbuf.close();
            ybuf.close();
        }
    }

    // ------------------------------------------------------------------ fixtures

    /** Deterministic pseudo-random integers in [0, bound) (xorshift). */
    private static final class Bits {
        private int state;

        Bits(int seed) {
            state = seed | 1;
        }

        int next(int bound) {
            state ^= state << 13;
            state ^= state >>> 17;
            state ^= state << 5;
            return Math.floorMod(state >>> 8, bound);
        }
    }

    /** fp16 bits of values with at most 7 explicit mantissa bits, exact in fp16 and bf16. */
    private static short buildHalf(Bits bits) {
        int sign = bits.next(2) == 0 ? 0x0000 : 0x8000;
        int mant = bits.next(8);
        int exp = 14 + bits.next(3);   /* 2^-1 .. 2^1 leading */
        return (short) (sign | (exp << 10) | (mant << 3));
    }

    private static short[] buildWeight() {
        return buildWeight(K);
    }

    private static short[] buildWeight(int k) {
        Bits bits = new Bits(0x51ED);
        short[] w = new short[M * k];
        for (int i = 0; i < w.length; i++) {
            w[i] = buildHalf(bits);
        }
        return w;
    }

    private static float[] buildX() {
        return buildX(K);
    }

    private static float[] buildX(int k) {
        Bits bits = new Bits(0xC0DE);
        float[] x = new float[N * k];
        for (int i = 0; i < x.length; i++) {
            x[i] = halfBitsToFloat(buildHalf(bits));
        }
        return x;
    }

    /** fp16 weight bits, little-endian per element. */
    private static byte[] halfBytes(short[] w) {
        ByteBuffer b = ByteBuffer.allocate(w.length * 2).order(ByteOrder.LITTLE_ENDIAN);
        for (short v : w) {
            b.putShort(v);
        }
        return b.array();
    }

    /** The same values as bf16 bits (exact for the fixture values), little-endian per element. */
    private static byte[] bf16Bytes(short[] w) {
        ByteBuffer b = ByteBuffer.allocate(w.length * 2).order(ByteOrder.LITTLE_ENDIAN);
        for (short v : w) {
            b.putShort(floatToBf16Bits(halfBitsToFloat(v)));
        }
        return b.array();
    }

    /** fp16 bits to float (exact). */
    static float halfBitsToFloat(short h) {
        int sign = (h & 0x8000) << 16;
        int exp = (h >>> 10) & 0x1F;
        int mant = h & 0x3FF;
        if (exp == 0) {
            if (mant == 0) {
                return Float.intBitsToFloat(sign);
            }
            return Float.intBitsToFloat(sign | ((exp + 112) << 23) | (mant << 13));
        }
        if (exp == 31) {
            return Float.intBitsToFloat(sign | 0x7F800000 | (mant << 13));
        }
        return Float.intBitsToFloat(sign | ((exp + 112) << 23) | (mant << 13));
    }

    /** float to fp16 bits, round to nearest even (normals in fp16 range, no NaN handling needed). */
    static short floatToHalfBits(float v) {
        int bits = Float.floatToRawIntBits(v);
        int sign = (bits >>> 16) & 0x8000;
        int exp = ((bits >>> 23) & 0xFF) - 127 + 15;
        int mant = (bits >>> 13) & 0x3FF;
        if (exp >= 31) {
            return (short) (sign | 0x7C00);   /* overflow of the test range, not taken */
        }
        if (exp <= 0) {
            return (short) sign;               /* subnormals do not occur in the fixtures */
        }
        return (short) (sign | (exp << 10) | mant);
    }

    /** bf16 bits to float (zero-padded mantissa, exact). */
    static float bf16BitsToFloat(short b) {
        return Float.intBitsToFloat((b & 0xFFFF) << 16);
    }

    /** float to bf16 bits, round to nearest even (finite inputs). */
    static short floatToBf16Bits(float v) {
        int bits = Float.floatToRawIntBits(v);
        return (short) ((bits + 0x7FFF + ((bits >>> 16) & 1)) >>> 16);
    }
}

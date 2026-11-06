package com.eonmux.neuralfree.gguf.ops.gemm;

import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.acceleration.backend.ocl.OclComputeBackend;
import com.eonmux.neuralfree.acceleration.kernels.Availability;
import com.eonmux.neuralfree.acceleration.ocl.CLDevice;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * The scalar twins of the MMA, split-K and CM GEMM families on the OpenCL device: the F16/BF16 MMA
 * kernels against exact 16-bit-product references, the split-K twin against the plain kernel's
 * output, and the {@code <T>_MatMul_CM} rows against the tiled GEMM of the same weight type (the
 * same dequantizing loaders and the same ascending-k accumulation, so the results agree to fp32
 * rounding).
 */
public class OpenClMmaGemmParityTest {

    private static final int M = 70;   /* two 64-row tiles, the second partial */
    private static final int K = 96;   /* six 16-deep stages, one partial */
    private static final int N = 40;   /* two 32-token tiles, the second partial */

    private HWAcceleration hwAccel;
    private OclComputeBackend backend;
    private ComputeQueue queue;
    private LlmKernels kernels;

    @Before
    public void setUp() {
        HWAcceleration hw = null;
        try {
            hw = new HWAcceleration();
            Assume.assumeTrue("no OpenCL device", !hw.getCLDevices().isEmpty());
            CLDevice device = hw.getCLDevices().get(0);
            OclComputeBackend b = new OclComputeBackend(device);
            Availability availability = LlmKernels.availability(b);
            Assume.assumeTrue("LLM kernels unavailable on " + b.capabilities().getDeviceName() + ": "
                    + availability.reason(), availability.available());
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
            Assume.assumeTrue("OpenCL init failed: " + t.getMessage(), false);
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

    /** The F16 MMA scalar twin: fp16 products with an exact reference, as its CUDA tensor path. */
    @Test
    public void f16MmaScalarTwinMatchesTheExactFp16ProductDot() throws Exception {
        MmaGemmPlan plan = MmaGemmPlan.contiguous(GGMLType.F16, M, K, N);
        short[] weight = buildWeight(K);
        float[] x = buildX(K);
        float[] out = launch(plan.kernelName(), plan, halfBytes(weight), x);

        for (int t = 0; t < N; t++) {
            for (int r = 0; r < M; r++) {
                double acc = 0.0;
                double terms = 0.0;
                for (int k = 0; k < K; k++) {
                    double w = halfBitsToFloat(weight[r * K + k]);
                    double q = halfBitsToFloat(floatToHalfBits(x[t * K + k]));
                    acc += w * q;
                    terms += Math.abs(w * q);
                }
                double tol = 1e-4 * terms + 1e-2;
                assertTrue("token " + t + " row " + r + ": kernel " + out[t * M + r] + " expected " + acc
                                + " (|terms| " + terms + ")",
                        Math.abs(out[t * M + r] - acc) <= tol);
            }
        }
    }

    /** The BF16 MMA scalar twin: bf16 products are exact in fp32, so the reference is exact. */
    @Test
    public void bf16MmaScalarTwinMatchesTheExactBf16ProductDot() throws Exception {
        MmaGemmPlan plan = MmaGemmPlan.contiguous(GGMLType.BF16, M, K, N);
        short[] weight = buildWeight(K);
        float[] x = buildX(K);
        float[] out = launch(plan.kernelName(), plan, bf16Bytes(weight), x);

        for (int t = 0; t < N; t++) {
            for (int r = 0; r < M; r++) {
                double acc = 0.0;
                double terms = 0.0;
                for (int k = 0; k < K; k++) {
                    double w = halfBitsToFloat(weight[r * K + k]);
                    acc += w * x[t * K + k];
                    terms += Math.abs(w * x[t * K + k]);
                }
                double tol = 1e-5 * terms + 1e-2;
                assertTrue("bf16 token " + t + " row " + r + ": kernel " + out[t * M + r] + " expected " + acc
                                + " (|terms| " + terms + ")",
                        Math.abs(out[t * M + r] - acc) <= tol);
            }
        }
    }

    /** The split-K twin of the F16 MMA kernel: three k-slices (one ragged) reduce to the plain kernel. */
    @Test
    public void splitKTwinReproducesThePlainKernelWithinFp32Summation() throws Exception {
        int splits = 3;
        int k = 2425;
        int kSlice = SplitKGemmPlan.sliceLength(k, splits);
        SplitKGemmPlan plan = SplitKGemmPlan.of(GGMLType.F16, M, k, N, k, 0, splits, kSlice, 256);
        MmaGemmPlan plain = MmaGemmPlan.contiguous(GGMLType.F16, M, k, N);
        byte[] weightBytes = halfBytes(buildWeight(k));
        float[] x = buildX(k);

        float[] split = launchSplit(plan, weightBytes, x);
        float[] plainOut = launch(plain.kernelName(), plain, weightBytes, x);
        for (int t = 0; t < N; t++) {
            for (int r = 0; r < M; r++) {
                double terms = 0.0;
                for (int i = 0; i < k; i++) {
                    terms += Math.abs(halfBitsToFloat(shortAt(weightBytes, i * 2))
                            * halfBitsToFloat(floatToHalfBits(x[t * k + i])));
                }
                double tol = 1e-3 * terms + 1e-2;
                assertTrue("split token " + t + " row " + r + ": kernel " + split[t * M + r] + " plain "
                                + plainOut[t * M + r] + " (|terms| " + terms + ")",
                        Math.abs(split[t * M + r] - plainOut[t * M + r]) <= tol);
            }
        }
    }

    /** The CM twin of Q4_K against the tiled GEMM of the same weight type. */
    @Test
    public void cmTwinMatchesTheTiledGemmOfTheSameType() throws Exception {
        GGMLType type = GGMLType.Q4_K;
        int k = 256;
        byte[] weight = OpenClMmqGemmParityTest.buildWeight(type, M, k);
        float[] x = OpenClMmqGemmParityTest.buildX(N, k);
        float[] cm = launchShaped(GemmKernels.matmulCm(type), M, k, N, type, weight, x);
        float[] tiled = launchShaped(GemmKernels.matmulTiled(type), M, k, N, type, weight, x);

        for (int t = 0; t < N; t++) {
            for (int r = 0; r < M; r++) {
                double terms = 0.0;
                for (int i = 0; i < k; i++) {
                    terms += Math.abs(x[t * k + i]);
                }
                double tol = 1e-5 * terms + 1e-3;
                assertTrue("CM token " + t + " row " + r + ": " + cm[t * M + r] + " tiled " + tiled[t * M + r],
                        Math.abs(cm[t * M + r] - tiled[t * M + r]) <= tol);
            }
        }
    }

    // ------------------------------------------------------------------ launches

    private float[] launch(String kernel, MmaGemmPlan plan, byte[] weightBytes, float[] x) throws Exception {
        GpuBuffer wbuf = backend.allocateBuffer(plan.weightBytes(), true);
        GpuBuffer xbuf = backend.allocateBuffer(plan.xElementsRequired() * Float.BYTES, true);
        GpuBuffer ybuf = backend.allocateBuffer(plan.yElementsRequired() * Float.BYTES);
        try {
            try (GpuEvent wv = queue.enqueueWriteBuffer(wbuf, 0, weightBytes.length, weightBytes);
                 GpuEvent xv = queue.enqueueWriteBuffer(xbuf, 0, x.length * Float.BYTES, x, wv)) {
                Object[] args = plan.args(wbuf, xbuf, ybuf);
                queue.enqueueKernelNoEvent(kernels.get(kernel), plan.globalWorkSize(), plan.localWorkSize(), args, xv);
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
                queue.enqueueKernelNoEvent(kernels.get(plan.kernelName()), plan.partialGlobalWorkSize(),
                        plan.partialLocalWorkSize(), plan.partialArgs(wbuf, xbuf, pbuf), xv);
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

    /** Launches a 10-argument GEMM row of {@code type} over contiguous views. */
    private float[] launchShaped(String kernel, int m, int k, int n, GGMLType type, byte[] weight, float[] x)
            throws Exception {
        long rowBytes = (long) (k / type.getBlockSize()) * type.getTypeSize();
        GpuBuffer wbuf = backend.allocateBuffer(rowBytes * m, true);
        GpuBuffer xbuf = backend.allocateBuffer((long) n * k * Float.BYTES, true);
        GpuBuffer ybuf = backend.allocateBuffer((long) n * m * Float.BYTES);
        try {
            try (GpuEvent wv = queue.enqueueWriteBuffer(wbuf, 0, weight.length, weight);
                 GpuEvent xv = queue.enqueueWriteBuffer(xbuf, 0, x.length * Float.BYTES, x, wv)) {
                Object[] args = {m, k, n, k, 0, m, 0, wbuf, xbuf, ybuf};
                queue.enqueueKernelNoEvent(kernels.get(kernel), new long[]{(long) ((m + 63) / 64) * 256,
                        (n + 31) / 32}, new long[]{256, 1}, args, xv);
                float[] out = new float[n * m];
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

    private static short shortAt(byte[] bytes, int shortIndex) {
        return ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).getShort(shortIndex * 2);
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

    private static short[] buildWeight(int k) {
        Bits bits = new Bits(0x51ED);
        short[] w = new short[M * k];
        for (int i = 0; i < w.length; i++) {
            w[i] = buildHalf(bits);
        }
        return w;
    }

    private static float[] buildX(int k) {
        Bits bits = new Bits(0xC0DE);
        float[] x = new float[N * k];
        for (int i = 0; i < x.length; i++) {
            x[i] = halfBitsToFloat(buildHalf(bits));
        }
        return x;
    }

    private static byte[] halfBytes(short[] w) {
        ByteBuffer b = ByteBuffer.allocate(w.length * 2).order(ByteOrder.LITTLE_ENDIAN);
        for (short v : w) {
            b.putShort(v);
        }
        return b.array();
    }

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
            return (short) (sign | 0x7C00);
        }
        if (exp <= 0) {
            return (short) sign;
        }
        return (short) (sign | (exp << 10) | mant);
    }

    /** float to bf16 bits, round to nearest even (finite inputs). */
    static short floatToBf16Bits(float v) {
        int bits = Float.floatToRawIntBits(v);
        return (short) ((bits + 0x7FFF + ((bits >>> 16) & 1)) >>> 16);
    }
}

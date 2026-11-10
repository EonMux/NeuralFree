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
import com.eonmux.neuralfree.gguf.quant.CpuBlockDecoders;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Collection;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Every {@code <T>_MatMul_MMQ} kernel against the CPU decoder of the same weight type on the OpenCL
 * device: the test hand-builds quantized rows whose codes and (scale, min) fields follow the ggml
 * block layouts (the llama.cpp MMQ types and the i-quants iq1_s..iq4_xs), decodes them with
 * {@link CpuBlockDecoders} (the reference), stages the activations once with
 * {@link GemmKernels#QUANT_MMQ_Q8_1} exactly as the dispatch does, and requires the kernel result to
 * match the exact dot of the decoded row with those codes within fp32 rounding of the stage combine;
 * the split-K twins of the MMQ and CM rows run against their plain kernels.
 */
@RunWith(Parameterized.class)
public class OpenClMmqGemmParityTest {

    private static final int M = 70;   /* two 64-row tiles, the second partial */
    private static final int K = 256;  /* one K-quant super-block per row, eight legacy blocks */
    private static final int N = 21;   /* two 16-token tiles, the second partial */

    @Parameterized.Parameters(name = "{0}")
    public static Collection<GGMLType> types() {
        return new ArrayList<>(GemmKernels.MMQ_TYPES);
    }

    private final GGMLType type;

    private HWAcceleration hwAccel;
    private OclComputeBackend backend;
    private ComputeQueue queue;
    private LlmKernels kernels;

    public OpenClMmqGemmParityTest(GGMLType type) {
        this.type = type;
    }

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

    @Test
    public void mmqGemmMatchesTheDecodedWeightDotProduct() throws InterruptedException {
        MmqGemmPlan plan = MmqGemmPlan.contiguous(MmqConfig.PACKED_DOT, type, M, K, N);
        byte[] weight = buildWeight(type, M, K);
        float[] x = buildX(N, K);

        float[] w = CpuBlockDecoders.dequantize(weight, type, (long) M * K);
        double[] q = new double[K];
        double[] d = new double[K];
        float[] out = launch(plan, weight, x);
        for (int t = 0; t < N; t++) {
            quantizeChunked(x, t * K, q, d);
            for (int r = 0; r < M; r++) {
                double acc = 0.0;
                double terms = 0.0;
                double bound = 0.0;
                for (int k = 0; k < K; k++) {
                    double term = (double) w[r * K + k] * q[k];
                    acc += term;
                    terms += Math.abs(term);
                    /* A backend's activation-quantization division (CUDA fast-math, GLSL ULP rules) may
                     * round one code up or down against this reference; one code step per element. */
                    bound += Math.abs((double) w[r * K + k] * d[k]);
                }
                double tol = bound + 1e-4 * terms + 1e-3;
                assertTrue(type + " token " + t + " row " + r + ": kernel " + out[t * M + r] + " expected " + acc
                                + " (|terms| " + terms + ")",
                        Math.abs(out[t * M + r] - acc) <= tol);
            }
        }
    }

    /**
     * The staging pass itself: {@link GemmKernels#QUANT_MMQ_Q8_1} writes, per token and 16-element
     * chunk, exactly the codes, chunk scale d and code sum the CPU quantization computes.
     */
    @Test
    public void theStagingPassWritesTheCpuCodesScalesAndSums() throws InterruptedException {
        MmqGemmPlan plan = MmqGemmPlan.contiguous(MmqConfig.PACKED_DOT, type, M, K, N);
        float[] x = buildX(N, K);
        int tk = MmqConfig.PACKED_DOT.stageDepth();
        int units = MmqConfig.PACKED_DOT.recordUnits();
        GpuBuffer xbuf = backend.allocateBuffer((long) x.length * Float.BYTES, true);
        GpuBuffer qbuf = backend.allocateBuffer(plan.q8ElementsRequired() * Integer.BYTES);
        try {
            try (GpuEvent write = queue.enqueueWriteBuffer(xbuf, 0, (long) x.length * Float.BYTES, x)) {
                write.awaitCompletion();
            }
            long records = (long) N * (K / tk);
            Object[] args = {N, K, tk, K, 0, xbuf, qbuf};
            queue.enqueueKernelNoEvent(kernels.get(GemmKernels.QUANT_MMQ_Q8_1),
                    new long[]{((records + 63) / 64) * 64}, new long[]{64}, args);
            byte[] raw = new byte[(int) plan.q8ElementsRequired() * Integer.BYTES];
            try (GpuEvent read = queue.enqueueReadBuffer(qbuf, 0, raw.length, raw)) {
                read.awaitCompletion();
            }
            int[] staged = new int[(int) plan.q8ElementsRequired()];
            ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN).asIntBuffer().get(staged);
            for (int t = 0; t < N; t++) {
                for (int c = 0; c < K / tk; c++) {
                    float maxabs = 0.0f;
                    for (int i = 0; i < tk; i++) {
                        maxabs = Math.max(maxabs, Math.abs(x[t * K + c * tk + i]));
                    }
                    float id = maxabs > 0.0f ? 127.0f / maxabs : 0.0f;
                    float d = maxabs > 0.0f ? maxabs * (1.0f / 127.0f) : 1.0f;
                    long rec = (long) (t * (K / tk) + c) * units;
                    for (int u = 0; u < tk / 4; u++) {
                        int packed = staged[(int) rec + u];
                        for (int b = 0; b < 4; b++) {
                            int code = (packed >>> (8 * b)) & 0xFF;
                            code = code >= 128 ? code - 256 : code;
                            int want = (int) Math.rint((double) (x[t * K + c * tk + 4 * u + b] * id));
                            assertEquals(type + " token " + t + " chunk " + c + " depth " + (4 * u + b),
                                    want, code);
                        }
                    }
                    int dBits = staged[(int) rec + tk / 4];
                    float stagedD = Float.intBitsToFloat(dBits);
                    assertEquals(type + " token " + t + " chunk " + c + " scale", d, stagedD, 0.0f);
                    int wantSum = 0;
                    for (int i = 0; i < tk; i++) {
                        wantSum += (int) Math.rint((double) (x[t * K + c * tk + i] * id));
                    }
                    assertEquals(type + " token " + t + " chunk " + c + " code sum", wantSum,
                            staged[(int) rec + tk / 4 + 1]);
                }
            }
        } finally {
            xbuf.close();
            qbuf.close();
        }
    }

    /**
     * The split-K twin of every int8 MMQ kernel against its plain kernel over the same staged
     * activations (the OpenCL scalar twins of kernels/llm/matmul_mmq.cl): fp32 plane sums only.
     */
    @Test
    public void mmqSplitKTwinMatchesThePlainKernel() throws Exception {
        FastGemmParity parity = new FastGemmParity(backend, kernels);
        for (GGMLType t : GemmKernels.MMQ_TYPES) {
            parity.mmqSplitKMatchesPlain(t, M, 768, N, 3);
        }
    }

    /**
     * The split-K twin of every CM kernel against its plain kernel (the OpenCL scalar twins of
     * kernels/llm/matmul_cm.cl): fp32 plane sums only.
     */
    @Test
    public void cmSplitKTwinMatchesThePlainKernel() throws Exception {
        FastGemmParity parity = new FastGemmParity(backend, kernels);
        for (GGMLType t : GemmKernels.MMQ_TYPES) {
            parity.cmSplitKMatchesPlain(t, M, 768, N, 3);
        }
    }

    /** Runs the staging pass and then the MMQ GEMM of {@code plan} over the staged records. */
    private float[] launch(MmqGemmPlan plan, byte[] weight, float[] x) throws InterruptedException {        GpuBuffer wbuf = backend.allocateBuffer(plan.weightBytes(), true);
        GpuBuffer xbuf = backend.allocateBuffer((long) x.length * Float.BYTES, true);
        GpuBuffer qbuf = backend.allocateBuffer(plan.q8ElementsRequired() * Integer.BYTES);
        GpuBuffer ybuf = backend.allocateBuffer(plan.yElementsRequired() * Float.BYTES);
        try {
            try (GpuEvent wv = queue.enqueueWriteBuffer(wbuf, 0, weight.length, weight);
                 GpuEvent xv = queue.enqueueWriteBuffer(xbuf, 0, x.length * Float.BYTES, x, wv)) {
                int tk = MmqConfig.PACKED_DOT.stageDepth();
                long records = (long) plan.n() * (plan.k() / tk);
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

    /**
     * The kernel's activation quantization: per 16-element chunk, d = maxabs * (1f/127) and the codes
     * round to nearest even of x * (127f / maxabs); writes the code values d * q the kernel multiplies
     * with the decoded weight, indexed by element offset 0..K-1 of the token, and the chunk scales d.
     */
    private static void quantizeChunked(float[] x, int base, double[] out, double[] dOut) {
        for (int c = 0; c < K; c += 16) {
            float maxabs = 0.0f;
            for (int k = c; k < c + 16; k++) {
                maxabs = Math.max(maxabs, Math.abs(x[base + k]));
            }
            float id = maxabs > 0.0f ? 127.0f / maxabs : 0.0f;
            float d = maxabs > 0.0f ? maxabs * (1.0f / 127.0f) : 1.0f;
            for (int k = c; k < c + 16; k++) {
                out[k] = (double) d * Math.rint((double) (x[base + k] * id));
                dOut[k] = d;
            }
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

    /**
     * Builds {@code rows} rows of {@code k} elements of {@code type}: deterministic codes covering the
     * code range and exact-in-fp16 scales, in the ggml block layouts of kernels/llm/common/quant_blocks.h.
     */
    public static byte[] buildWeight(GGMLType type, int rows, int k) {
        int blocksPerRow = k / type.getBlockSize();
        int blockBytes = type.getTypeSize();
        ByteBuffer b = ByteBuffer.allocate(rows * blocksPerRow * blockBytes).order(ByteOrder.LITTLE_ENDIAN);
        Bits bits = new Bits(0x9E3779B9);
        for (int r = 0; r < rows; r++) {
            for (int blk = 0; blk < blocksPerRow; blk++) {
                int start = b.position();
                float d = 0.5f + 0.25f * (r % 5);
                if (type == GGMLType.I8_B128) {
                    b.putFloat(d);
                    for (int e = 0; e < 128; e++) {
                        b.put((byte) (bits.next(256) - 128));
                    }
                } else if (type == GGMLType.Q4_0 || type == GGMLType.Q4_1 || type == GGMLType.Q5_0
                        || type == GGMLType.Q5_1 || type == GGMLType.Q8_0) {
                    legacyBlock(b, bits, type, d);
                } else if (type == GGMLType.Q2_K || type == GGMLType.Q3_K) {
                    q2q3Block(b, bits, type, d);
                } else if (type == GGMLType.Q4_K || type == GGMLType.Q5_K) {
                    q4kBlock(b, bits, type, d);
                } else if (type == GGMLType.Q6_K) {
                    q6kBlock(b, bits, d);
                } else if (type == GGMLType.F8_E4M3_B128) {
                    b.putFloat(d);
                    e4m3Codes(b, 128, bits);
                } else if (type == GGMLType.MXFP4) {
                    b.put((byte) 127);                 /* e8m0 exponent of 2^0 (half scale 0.5) */
                    nibbleBlock(b, bits, 16);
                } else if (type == GGMLType.NVFP4) {
                    for (int i = 0; i < 4; i++) {
                        b.put((byte) 0x38);            /* ue4m3 sub-block scale 0.5 */
                    }
                    nibbleBlock(b, bits, 32);
                } else if (type == GGMLType.IQ1_S || type == GGMLType.IQ2_XXS || type == GGMLType.IQ2_XS
                        || type == GGMLType.IQ2_S || type == GGMLType.IQ3_XXS || type == GGMLType.IQ3_S
                        || type == GGMLType.IQ4_NL || type == GGMLType.IQ4_XS) {
                    /* every byte of these layouts is a valid grid index, sign pattern or scale nibble */
                    iqBlock(b, bits, blockBytes);
                } else {
                    throw new IllegalStateException("no builder for " + type);
                }
                if (b.position() - start != blockBytes) {
                    throw new IllegalStateException(type + " block builder wrote " + (b.position() - start)
                            + " bytes, the layout needs " + blockBytes);
                }
            }
        }
        return b.array();
    }

    /** Finite E4M3 codes in [0x30, 0x4F] (magnitudes 0.5 .. 15), so no NaN code can occur. */
    private static void e4m3Codes(ByteBuffer b, int count, Bits bits) {
        for (int e = 0; e < count; e++) {
            b.put((byte) (0x30 + bits.next(32)));
        }
    }

    /** The packed 4-bit code bytes of an E2M1 block: deterministic nibbles (every code decodes). */
    private static void nibbleBlock(ByteBuffer b, Bits bits, int bytes) {
        for (int i = 0; i < bytes; i++) {
            b.put((byte) bits.next(256));
        }
    }

    /** One i-quant block: an exact fp16 scale then deterministic bytes (grid indexes, sign bytes and
     *  scale nibbles; every value decodes). */
    private static void iqBlock(ByteBuffer b, Bits bits, int blockBytes) {
        putF16(b, 0.5f + 0.25f * bits.next(3));
        for (int i = 2; i < blockBytes; i++) {
            b.put((byte) bits.next(256));
        }
    }

    /** One legacy 32-element block (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0); nibbles per ggml SPLIT layout. */
    private static void legacyBlock(ByteBuffer b, Bits bits, GGMLType type, float d) {
        putF16(b, d);
        if (type == GGMLType.Q4_1 || type == GGMLType.Q5_1) {
            putF16(b, 0.25f);
        }
        if (type == GGMLType.Q8_0) {
            for (int e = 0; e < 32; e++) {
                b.put((byte) (bits.next(256) - 128));
            }
            return;
        }
        int qh = -1;
        if (type == GGMLType.Q5_0 || type == GGMLType.Q5_1) {
            qh = b.position();
            b.putInt(0);
        }
        int qs = b.position();
        b.position(qs + 16);
        for (int e = 0; e < 32; e++) {
            int lo = bits.next(16);
            int idx = qs + (e & 15);
            int cur = b.get(idx) & 0xFF;
            cur |= (e < 16 ? lo : lo << 4);
            b.put(idx, (byte) cur);
            if (qh >= 0 && bits.next(4) == 0) {
                b.putInt(qh, b.getInt(qh) | (1 << e));
            }
        }
    }

    /** One Q2_K / Q3_K super-block: 16 sub-blocks of 16 elements on four 2-bit planes. */
    private static void q2q3Block(ByteBuffer b, Bits bits, GGMLType type, float d) {
        int hmask = -1;
        if (type == GGMLType.Q2_K) {
            for (int j = 0; j < 16; j++) {
                b.put((byte) ((1 + bits.next(15)) | ((1 + bits.next(15)) << 4)));
            }
        } else {
            hmask = b.position();
            b.position(b.position() + 32);
        }
        int qs = b.position();
        b.position(b.position() + 64);
        if (type == GGMLType.Q3_K) {
            putQ3kScales(b, bits);
        }
        putF16(b, d);
        if (type == GGMLType.Q2_K) {
            putF16(b, 0.25f);
        }
        for (int e = 0; e < 256; e++) {
            int hf = e >> 7, j = (e >> 5) & 3, h = (e & 16) >> 4, l = e & 15;
            int idx = qs + hf * 32 + h * 16 + l;
            int cur = b.get(idx) & 0xFF;
            cur |= bits.next(4) << (2 * j);
            b.put(idx, (byte) cur);
            if (hmask >= 0) {
                /* element e's high bit is bit 4 hf + j of its hmask byte, so every bit of the byte is in use */
                int hidx = hmask + h * 16 + l;
                b.put(hidx, (byte) ((b.get(hidx) & 0xFF) | (bits.next(2) << (4 * hf + j))));
            }
        }
    }

    /** 12 scale bytes of one Q3_K block: sixteen packed 6-bit signed scales (llama.cpp aux repacking). */
    private static void putQ3kScales(ByteBuffer b, Bits bits) {
        int sc = b.position();
        b.position(b.position() + 12);
        for (int si = 0; si < 16; si++) {
            int v = bits.next(64);
            int lo = v & 0xF, hi = (v >> 4) & 3;
            int idx = sc + (si & 7);
            int cur = b.get(idx) & 0xFF;
            cur |= lo << ((si >> 3) * 4);
            b.put(idx, (byte) cur);
            idx = sc + 8 + (si & 3);
            cur = b.get(idx) & 0xFF;
            cur |= hi << ((si >> 2) * 2);
            b.put(idx, (byte) cur);
        }
    }

    /** One Q4_K / Q5_K super-block: 8 groups of 32 elements with get_scale_min_k4 scale/min pairs. */
    private static void q4kBlock(ByteBuffer b, Bits bits, GGMLType type, float d) {
        boolean fifth = type == GGMLType.Q5_K;
        putF16(b, d);
        putF16(b, 0.25f);
        int sc = b.position();
        b.position(b.position() + 12);
        int qh = 0;
        if (fifth) {
            qh = b.position();
            b.position(b.position() + 32);
        }
        int qs = b.position();
        b.position(b.position() + 128);
        int[] groupScale = new int[8];
        int[] groupMin = new int[8];
        for (int j = 0; j < 8; j++) {
            groupScale[j] = 1 + bits.next(31);
            groupMin[j] = 1 + bits.next(31);
        }
        for (int j = 0; j < 4; j++) {
            b.put(sc + j, (byte) groupScale[j]);
            b.put(sc + j + 4, (byte) groupMin[j]);
        }
        for (int j = 4; j < 8; j++) {
            int low = b.get(sc + j + 4) & 0xFF;
            b.put(sc + j + 4, (byte) ((groupScale[j] & 0xF) | ((groupMin[j] & 0xF) << 4)));
            int up = b.get(sc + j - 4) & 0xFF;
            b.put(sc + j - 4, (byte) (up | ((groupScale[j] >> 4) << 6)));
            up = b.get(sc + j) & 0xFF;
            b.put(sc + j, (byte) (up | ((groupMin[j] >> 4) << 6)));
        }
        for (int e = 0; e < 256; e++) {
            int si = e >> 5, l = e & 31;
            int nibble = bits.next(16);
            int bit5 = fifth ? bits.next(2) : 0;
            int idx = qs + (si >> 1) * 32 + l;
            int cur = b.get(idx) & 0xFF;
            cur |= nibble << ((si & 1) * 4);
            b.put(idx, (byte) cur);
            if (fifth && bit5 != 0 && si > 0) {
                int qcur = b.get(qh + l) & 0xFF;
                b.put(qh + l, (byte) (qcur | (1 << si)));
            }
        }
    }

    /** One Q6_K super-block: 256 6-bit codes over ql/qh planes with signed per-16 scales. */
    private static void q6kBlock(ByteBuffer b, Bits bits, float d) {
        int ql = b.position();
        b.position(b.position() + 128);
        int qh = b.position();
        b.position(b.position() + 64);
        int sc = b.position();
        b.position(b.position() + 16);
        for (int e = 0; e < 256; e++) {
            int hf = e >> 7, t = (e >> 5) & 3, l = e & 31;
            int code = bits.next(64);
            int idx = ql + hf * 64 + (t & 1) * 32 + l;
            int cur = b.get(idx) & 0xFF;
            cur |= (code & 15) << ((t >> 1) * 4);
            b.put(idx, (byte) cur);
            idx = qh + hf * 32 + l;
            cur = b.get(idx) & 0xFF;
            cur |= ((code >> 4) & 3) << (t * 2);
            b.put(idx, (byte) cur);
        }
        for (int s = 0; s < 16; s++) {
            b.put(sc + s, (byte) (bits.next(9) - 4));
        }
        putF16(b, d);
    }

    /** Deterministic activations in [-2, 2). */
    public static float[] buildX(int n, int k) {
        Bits bits = new Bits(0xC0FFEE);
        float[] x = new float[n * k];
        for (int i = 0; i < x.length; i++) {
            x[i] = 4.0f * (bits.next(1024) - 512) / 1024.0f;
        }
        return x;
    }

    /** IEEE half bits of a float (the builders only use exact small multiples of 1/8). */
    private static void putF16(ByteBuffer b, float v) {
        int bits = Float.floatToIntBits(v);
        int sign = (bits >>> 16) & 0x8000;
        int exp = ((bits >>> 23) & 0xFF) - 127 + 15;
        int mant = (bits >>> 13) & 0x3FF;
        b.putShort((short) (sign | (exp << 10) | mant));
    }
}

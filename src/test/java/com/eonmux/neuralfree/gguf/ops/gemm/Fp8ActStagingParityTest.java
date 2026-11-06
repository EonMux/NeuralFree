package com.eonmux.neuralfree.gguf.ops.gemm;

import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.acceleration.backend.cuda.CudaComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ocl.OclComputeBackend;
import com.eonmux.neuralfree.acceleration.cuda.CUDevice;
import com.eonmux.neuralfree.acceleration.kernels.Availability;
import com.eonmux.neuralfree.acceleration.ocl.CLDevice;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.quant.ActFakeQuant;
import com.eonmux.neuralfree.gguf.quant.ActQuantScheme;
import com.eonmux.neuralfree.gguf.quant.CpuBlockDecoders;
import com.eonmux.neuralfree.utils.numeric.Fp8Codec;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import static org.junit.Assert.assertTrue;

/**
 * The checkpoint-scheme E4M3 staging and the E4M3 GEMM behind the FP8 W8A8 lane: with an
 * {@link ActQuantScheme#FP8_STATIC}, {@link ActQuantScheme#FP8_TOKEN}, {@link ActQuantScheme#FP8_GROUP128},
 * {@link ActQuantScheme#MXFP8}, {@link ActQuantScheme#MXFP4} or {@link ActQuantScheme#NVFP4} weight, the staging
 * kernel writes the scheme's own codes as E4M3 and one chunk scale, and {@code <T>_MatMul_MMQ} then computes the
 * product of those activation codes with the FP8 weight codes. The staged value equals the fake-quantized
 * activation ({@link ActFakeQuant}, the host twin) exactly for every scheme whose quantized values are E4M3 codes,
 * and up to one E4M3 rounding of the chunk maximum for {@link ActQuantScheme#NVFP4}, whose per-16 sub-block scales
 * an E4M3 MMA cannot carry; the GEMM output must equal the dot of the decoded weight with the staged values.
 */
public class Fp8ActStagingParityTest {

    private static final int M = 70;   /* two 64-row tiles, the second partial */
    private static final int K = 512;
    private static final int N = 21;   /* two 64-token tiles (tensor config), the second partial */
    private static final float STATIC_SCALE = 0.25f;
    private static final float GLOBAL_SCALE = 0.25f;

    private HWAcceleration hwAccel;
    private ComputeBackend backend;
    private ComputeQueue queue;
    private LlmKernels kernels;

    @Before
    public void setUp() {
        hwAccel = null;
        backend = null;
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
    public void theE4m3LaneIsCudaOnly() throws Exception {
        openOpenCl();
        for (ActQuantScheme scheme : schemes()) {
            String kernel = GemmKernels.fp8ActivationStaging(scheme);
            assertTrue(kernel + " must not exist on OpenCL", !kernels.has(kernel));
        }
        assertTrue("F8_E4M3_MatMul_MMQ must not exist on OpenCL",
                !kernels.has(GemmKernels.matmulMmq(GGMLType.F8_E4M3)));
    }

    @Test
    public void cudaStagingAndGemmMatchTheFakeQuantizedDotProduct() throws Exception {
        openCuda();
        MmqConfig config = MmqConfig.forDevice(backend.backendType(),
                backend.capabilities().getComputeCapability());
        for (ActQuantScheme scheme : schemes()) {
            checkStaging(config, scheme);
        }
        Assume.assumeTrue("the E4M3 GEMM needs sm_89+",
                backend.capabilities().getComputeCapability() >= 89);
        for (GGMLType type : new GGMLType[]{GGMLType.F8_E4M3, GGMLType.F8_E4M3_B32, GGMLType.F8_E4M3_B128,
                GGMLType.MXFP8}) {
            for (ActQuantScheme scheme : schemes()) {
                checkGemm(config, type, scheme);
            }
        }
    }

    private static ActQuantScheme[] schemes() {
        return new ActQuantScheme[]{ActQuantScheme.FP8_STATIC, ActQuantScheme.FP8_TOKEN,
                ActQuantScheme.FP8_GROUP128, ActQuantScheme.MXFP8, ActQuantScheme.MXFP4, ActQuantScheme.NVFP4};
    }

    private void openOpenCl() {
        HWAcceleration hw = null;
        try {
            hw = new HWAcceleration();
            Assume.assumeTrue("no OpenCL device", !hw.getCLDevices().isEmpty());
            OclComputeBackend b = new OclComputeBackend(hw.getCLDevices().get(0));
            Availability availability = LlmKernels.availability(b);
            Assume.assumeTrue("LLM kernels unavailable: " + availability.reason(), availability.available());
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

    private void openCuda() {
        HWAcceleration.enableCuda = true;
        HWAcceleration hw = null;
        try {
            hw = new HWAcceleration();
            Assume.assumeTrue("no CUDA device", !hw.getCUDevices().isEmpty());
            CUDevice device = hw.getCUDevices().get(0);
            CudaComputeBackend b = new CudaComputeBackend(device);
            Availability availability = LlmKernels.availability(b);
            Assume.assumeTrue("LLM kernels unavailable: " + availability.reason(), availability.available());
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
            Assume.assumeTrue("CUDA init failed: " + t.getMessage(), false);
        }
    }

    /** The staging kernel of a scheme writes codes and scales whose product is the fake-quantized value. */
    private void checkStaging(MmqConfig config, ActQuantScheme scheme) throws Exception {
        String stageKernel = GemmKernels.fp8ActivationStaging(scheme);
        Assume.assumeTrue(scheme + " has no E4M3 staging kernel", stageKernel != null);
        Assume.assumeTrue(stageKernel + " is unavailable", kernels.has(stageKernel));
        MmqGemmPlan plan = MmqGemmPlan.contiguous(config, GGMLType.F8_E4M3_B128, M, K, N);
        float[] x = OpenClMmqGemmParityTest.buildX(N, K);
        float[] fake = x.clone();
        ActFakeQuant.apply(scheme, staticScale(scheme), 0, GLOBAL_SCALE, fake, 0, N, K);
        Staged staged = stagedValues(config, stageKernel, scheme, plan, x);
        for (int i = 0; i < N * K; i++) {
            /* NVFP4 carries per-16 sub-block scales the chunk scale cannot express, so its staged value is the
             * scheme's value rounded once to an E4M3 code of the chunk maximum (at most half a code step). */
            double step = scheme == ActQuantScheme.NVFP4 ? staged.chunkScale[i] * 0.0625 + 1e-6 : 0.0;
            assertTrue(scheme + " element " + i + ": staged " + staged.value[i] + " expected " + fake[i],
                    Math.abs(staged.value[i] - fake[i]) <= step);
        }
    }

    /** The E4M3 GEMM of an FP8 weight type equals the dot of the decoded weight with the staged values. */
    private void checkGemm(MmqConfig config, GGMLType type, ActQuantScheme scheme) throws Exception {
        String stageKernel = GemmKernels.fp8ActivationStaging(scheme);
        Assume.assumeTrue(scheme + " has no E4M3 staging kernel", stageKernel != null);
        Assume.assumeTrue(stageKernel + " is unavailable", kernels.has(stageKernel));
        MmqGemmPlan plan = MmqGemmPlan.contiguous(config, type, M, K, N);
        Assume.assumeTrue(plan.kernelName() + " is unavailable", kernels.has(plan.kernelName()));
        byte[] weight = fp8Weight(type, M, K);
        float[] x = OpenClMmqGemmParityTest.buildX(N, K);
        float[] w = CpuBlockDecoders.dequantize(weight, type, (long) M * K);

        int tk = config.stageDepth();
        int units = config.recordUnits();
        GpuBuffer wbuf = backend.allocateBuffer(plan.weightBytes(), true);
        GpuBuffer xbuf = backend.allocateBuffer((long) x.length * Float.BYTES, true);
        GpuBuffer qbuf = backend.allocateBuffer(plan.q8ElementsRequired() * Integer.BYTES);
        GpuBuffer ybuf = backend.allocateBuffer(plan.yElementsRequired() * Float.BYTES);
        try {
            try (GpuEvent wv = queue.enqueueWriteBuffer(wbuf, 0, weight.length, weight);
                 GpuEvent xv = queue.enqueueWriteBuffer(xbuf, 0, (long) x.length * Float.BYTES, x, wv)) {
                stage(stageKernel, scheme, plan, xbuf, qbuf, xv);
                queue.enqueueKernelNoEvent(kernels.get(plan.kernelName()), plan.globalWorkSize(),
                        plan.localWorkSize(), plan.args(wbuf, qbuf, ybuf));
                float[] out = new float[(int) plan.yElementsRequired()];
                try (GpuEvent read = queue.enqueueReadBuffer(ybuf, 0, (long) out.length * Float.BYTES, out)) {
                    read.awaitCompletion();
                }
                int[] staged = readInts(qbuf, plan.q8ElementsRequired());
                float[] value = stagedFloatValues(staged, tk, units, N, K).value;
                for (int t = 0; t < N; t++) {
                    for (int r = 0; r < M; r++) {
                        double acc = 0.0;
                        double terms = 0.0;
                        for (int k = 0; k < K; k++) {
                            final double term = (double) w[r * K + k] * value[t * K + k];
                            acc += term;
                            terms += Math.abs(term);
                        }
                        double tol = 1e-4 * terms + 1e-3;
                        assertTrue(type + " " + scheme + " token " + t + " row " + r + ": kernel "
                                        + out[t * M + r] + " expected " + acc + " (|terms| " + terms + ")",
                                Math.abs(out[t * M + r] - acc) <= tol);
                    }
                }
            }
        } finally {
            wbuf.close();
            xbuf.close();
            qbuf.close();
            ybuf.close();
        }
    }

    /** The staged value of every element and the chunk scale it was encoded with. */
    private static final class Staged {
        final float[] value;
        final float[] chunkScale;

        Staged(float[] value, float[] chunkScale) {
            this.value = value;
            this.chunkScale = chunkScale;
        }
    }

    /** Stages {@code x} with the scheme's kernel and returns the staged value of every element (code * chunk scale). */
    private Staged stagedValues(MmqConfig config, String stageKernel, ActQuantScheme scheme, MmqGemmPlan plan,
                                float[] x) throws Exception {
        GpuBuffer xbuf = backend.allocateBuffer((long) x.length * Float.BYTES, true);
        GpuBuffer qbuf = backend.allocateBuffer(plan.q8ElementsRequired() * Integer.BYTES);
        try {
            try (GpuEvent xv = queue.enqueueWriteBuffer(xbuf, 0, (long) x.length * Float.BYTES, x)) {
                stage(stageKernel, scheme, plan, xbuf, qbuf, xv);
                int[] staged = readInts(qbuf, plan.q8ElementsRequired());
                return stagedFloatValues(staged, config.stageDepth(), config.recordUnits(), plan.n(), plan.k());
            }
        } finally {
            xbuf.close();
            qbuf.close();
        }
    }

    private void stage(String kernel, ActQuantScheme scheme, MmqGemmPlan plan, GpuBuffer x, GpuBuffer q8,
                       GpuEvent after) {
        int tk = plan.config().stageDepth();
        int local = kernels.matvecLocal();
        long records = (long) plan.n() * plan.config().rowRecords(plan.k());
        float argument = staticScale(scheme);
        Object[] args;
        long global;
        if (scheme == ActQuantScheme.FP8_TOKEN) {
            args = new Object[]{plan.n(), plan.k(), tk, plan.k(), 0, argument, x, q8};
            global = (long) plan.n() * local;
        } else if (scheme == ActQuantScheme.FP8_GROUP128) {
            long runs = (long) plan.n() * (plan.k() / 128);
            args = new Object[]{plan.n(), plan.k(), tk, plan.k(), 0, x, q8};
            global = runs * local;
        } else if (scheme == ActQuantScheme.NVFP4) {
            args = new Object[]{plan.n(), plan.k(), tk, plan.k(), 0, GLOBAL_SCALE, x, q8};
            global = records * local;
        } else if (scheme.isStatic()) {
            args = new Object[]{plan.n(), plan.k(), tk, plan.k(), 0, argument, x, q8};
            global = records;
        } else {
            args = new Object[]{plan.n(), plan.k(), tk, plan.k(), 0, x, q8};
            global = records;
        }
        queue.enqueueKernelNoEvent(kernels.get(kernel), new long[]{roundUp(global, local)}, new long[]{local},
                args, after);
    }

    private int[] readInts(GpuBuffer buffer, long units) throws Exception {
        byte[] raw = new byte[(int) units * Integer.BYTES];
        try (GpuEvent read = queue.enqueueReadBuffer(buffer, 0, raw.length, raw)) {
            read.awaitCompletion();
        }
        int[] values = new int[(int) units];
        ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN).asIntBuffer().get(values);
        return values;
    }

    /** Per-element staged values of the E4M3 records: E4M3 code * the chunk scale. */
    private static Staged stagedFloatValues(int[] staged, int tk, int units, int n, int k) {
        float[] value = new float[n * k];
        float[] chunkScale = new float[n * k];
        int chunks = k / tk;
        for (int t = 0; t < n; t++) {
            for (int c = 0; c < chunks; c++) {
                int rec = (t * chunks + c) * units;
                float d = Float.intBitsToFloat(staged[rec + tk / 4]);
                for (int i = 0; i < tk; i++) {
                    int code = (staged[rec + (i >> 2)] >>> (8 * (i & 3))) & 0xFF;
                    value[t * k + c * tk + i] = Fp8Codec.E4M3FN.decode(code) * d;
                    chunkScale[t * k + c * tk + i] = d;
                }
            }
        }
        return new Staged(value, chunkScale);
    }

    private static float staticScale(ActQuantScheme scheme) {
        if (scheme == ActQuantScheme.FP8_STATIC) {
            return STATIC_SCALE;
        }
        if (scheme == ActQuantScheme.FP8_TOKEN) {
            return Float.POSITIVE_INFINITY;
        }
        return 0.0f;
    }

    /** Rows of an FP8 weight type: deterministic finite E4M3 codes and exact block scales. */
    private static byte[] fp8Weight(GGMLType type, int rows, int k) {
        int blocksPerRow = k / type.getBlockSize();
        int blockBytes = type.getTypeSize();
        ByteBuffer b = ByteBuffer.allocate(rows * blocksPerRow * blockBytes).order(ByteOrder.LITTLE_ENDIAN);
        for (int r = 0; r < rows; r++) {
            for (int blk = 0; blk < blocksPerRow; blk++) {
                float d = 0.5f + 0.25f * (r % 5);
                if (type == GGMLType.F8_E4M3) {
                    putCodes(b, type.getBlockSize(), (r + blk) * 7);
                } else if (type == GGMLType.F8_E4M3_B32 || type == GGMLType.F8_E4M3_B128) {
                    b.putFloat(d);
                    putCodes(b, type.getBlockSize(), (r + blk) * 7);
                } else if (type == GGMLType.MXFP8) {
                    b.put((byte) 127);                     /* e8m0 exponent of 2^0 */
                    putCodes(b, type.getBlockSize(), (r + blk) * 7);
                } else {
                    throw new IllegalStateException("no FP8 weight builder for " + type);
                }
            }
        }
        return b.array();
    }

    /** Finite E4M3 codes in [0x30, 0x4F] (magnitudes 0.5 .. 15) so no NaN can occur. */
    private static void putCodes(ByteBuffer b, int count, int seed) {
        for (int e = 0; e < count; e++) {
            b.put((byte) (0x30 + (seed + e * 3) % 32));
        }
    }

    private static long roundUp(long n, long m) {
        return (n + m - 1) / m * m;
    }
}

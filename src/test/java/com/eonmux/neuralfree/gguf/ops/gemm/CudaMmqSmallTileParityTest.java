package com.eonmux.neuralfree.gguf.ops.gemm;

import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.acceleration.backend.cuda.CudaComputeBackend;
import com.eonmux.neuralfree.acceleration.cuda.CUDevice;
import com.eonmux.neuralfree.acceleration.kernels.Availability;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.ArrayList;
import java.util.Collection;

import static org.junit.Assert.assertTrue;

/**
 * The small-token twins ({@code <T>_MatMul_MMQ_S} and the 64-token {@code _S64}) and their split-K
 * forms ({@code <T>_MatMul_MMQ_SKSplit} / {@code _SKSplit64}) against the plain kernel of the same
 * CUDA fatbin, at the tile {@link MmqConfig#forDevice} picks for the device: the twins tile the
 * outputs 16 or 64 tokens wide and stream whole weight blocks, with the same staged codes, the same
 * decoded tiles and the same per-output accumulation order, so every twin result equals the plain
 * kernel's bit for bit. The split-K planes add only the fp32 plane sums of the host reduce, as
 * {@link FastGemmParity#mmqSplitKMatchesPlain} tolerates. {@link OpenClMmqSmallTileParityTest} is the
 * OpenCL counterpart.
 */
@RunWith(Parameterized.class)
public class CudaMmqSmallTileParityTest {

    private static final int M = 70;  /* one full 64-row tile plus a 6-row partial */

    @Parameterized.Parameters(name = "{0}")
    public static Collection<GGMLType> types() {
        return new ArrayList<>(GemmKernels.MMQ_TYPES);
    }

    private final GGMLType type;

    private HWAcceleration hwAccel;
    private CudaComputeBackend backend;
    private ComputeQueue queue;
    private LlmKernels kernels;
    private MmqConfig config;

    public CudaMmqSmallTileParityTest(GGMLType type) {
        this.type = type;
    }

    @Before
    public void setUp() {
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
            config = MmqConfig.forDevice(ComputeBackend.BackendType.CUDA, b.capabilities().getComputeCapability());
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

    /**
     * The small-token twin at the step sizes of a speculative verify and across whole and partial
     * 16-token tiles (9, 16, 31 tokens; 13 and 70 weight rows; K of one and three blocks) equals the
     * plain kernel of the same fatbin exactly, bit for bit.
     */
    @Test
    public void theSmallTileTwinMatchesThePlainKernelBitForBit() throws InterruptedException {
        FastGemmParity parity = new FastGemmParity(backend, kernels);
        for (int blocks : new int[]{1, 3}) {
            int k = blocks * type.getBlockSize();
            for (int n : new int[]{9, 16, 31}) {
                for (int m : new int[]{13, M}) {
                    MmqGemmPlan plain = MmqGemmPlan.contiguous(config, type, m, k, n);
                    MmqGemmPlan small = smallPlan(m, k, n);
                    assertTrue(small.kernelName() + " is no small-tile twin",
                            small.kernelName().endsWith(GemmKernels.MATMUL_MMQ_S_SUFFIX));
                    byte[] weight = OpenClMmqGemmParityTest.buildWeight(type, m, k);
                    float[] x = OpenClMmqGemmParityTest.buildX(n, k);
                    float[] plainOut = parity.launchMmq(plain, weight, x);
                    float[] twinOut = parity.launchMmq(small, weight, x);
                    assertBits(type + " n=" + n + " m=" + m + " k=" + k, plainOut, twinOut);
                }
            }
        }
    }

    /**
     * The 64-token twin ({@code _S64}) at the mid-batch steps across whole and partial 64-token
     * tiles (17, 32, 64 tokens; 13 and 70 weight rows; K of one and three blocks) equals the plain
     * kernel of the same fatbin exactly, bit for bit.
     */
    @Test
    public void theTwin64MatchesThePlainKernelBitForBit() throws InterruptedException {
        FastGemmParity parity = new FastGemmParity(backend, kernels);
        for (int blocks : new int[]{1, 3}) {
            int k = blocks * type.getBlockSize();
            for (int n : new int[]{17, 32, 64}) {
                for (int m : new int[]{13, M}) {
                    MmqGemmPlan plain = MmqGemmPlan.contiguous(config, type, m, k, n);
                    MmqGemmPlan small = smallPlan(m, k, n, MmqConfig.TWIN_MAX_TOKENS);
                    assertTrue(small.kernelName() + " is no 64-token twin",
                            small.kernelName().endsWith(GemmKernels.MATMUL_MMQ_S64_SUFFIX));
                    byte[] weight = OpenClMmqGemmParityTest.buildWeight(type, m, k);
                    float[] x = OpenClMmqGemmParityTest.buildX(n, k);
                    float[] plainOut = parity.launchMmq(plain, weight, x);
                    float[] twinOut = parity.launchMmq(small, weight, x);
                    assertBits(type + " S64 n=" + n + " m=" + m + " k=" + k, plainOut, twinOut);
                }
            }
        }
    }

    /**
     * The split-K twin over one whole-K slice equals the plain kernel bit for bit: with
     * {@code k_slice = K} the single plane is the output, so the combine faces no plane sum at all.
     */
    @Test
    public void theSplitKTwinMatchesThePlainKernelBitForBit() throws InterruptedException {
        FastGemmParity parity = new FastGemmParity(backend, kernels);
        int k = 3 * type.getBlockSize();
        int qStride = config.rowRecords(k);
        for (int n : new int[]{9, 16}) {
            MmqGemmPlan plain = MmqGemmPlan.contiguous(config, type, M, k, n);
            SplitKGemmPlan split = SplitKGemmPlan.ofMmq(type, M, k, n, qStride, 0, 1, k,
                    config, false, 256);
            assertTrue(split.kernelName() + " is no split-K twin",
                    split.kernelName().endsWith(GemmKernels.MATMUL_MMQ_KSPLIT_SUFFIX));
            byte[] weight = OpenClMmqGemmParityTest.buildWeight(type, M, k);
            float[] x = OpenClMmqGemmParityTest.buildX(n, k);
            float[] plainOut = parity.launchMmq(plain, weight, x);
            float[] partial = launchSplitPartials(split, weight, x);
            assertTrue("partial plane of " + split.kernelName() + " n=" + n, partial.length == n * M);
            assertBits(type + " split-K n=" + n + " k=" + k, plainOut, partial);
        }
    }

    /**
     * The split-K twin over two block-aligned slices, reduced by the host reduce in slice order,
     * matches the plain kernel within the fp32 plane sums the split-K form adds
     * ({@link FastGemmParity#mmqSplitKMatchesPlain}).
     */
    @Test
    public void theSplitKPlanesSumToThePlainKernel() throws InterruptedException {
        FastGemmParity parity = new FastGemmParity(backend, kernels);
        int k = 4 * type.getBlockSize();
        int n = 9;
        int qStride = config.rowRecords(k);
        MmqGemmPlan plain = MmqGemmPlan.contiguous(config, type, M, k, n);
        SplitKGemmPlan split = SplitKGemmPlan.ofMmq(type, M, k, n, qStride, 0, 2, k / 2,
                config, false, 256);
        byte[] weight = OpenClMmqGemmParityTest.buildWeight(type, M, k);
        float[] x = OpenClMmqGemmParityTest.buildX(n, k);
        float[] plainOut = parity.launchMmq(plain, weight, x);
        float[] splitOut = parity.launchMmqSplit(split, weight, x);
        for (int i = 0; i < n * M; i++) {
            assertTrue(split.kernelName() + " element " + i + ": " + splitOut[i] + " plain "
                            + plainOut[i],
                    Math.abs(splitOut[i] - plainOut[i]) <= 5e-3 * (1.0 + Math.abs(plainOut[i])));
        }
    }

    /**
     * The small tiles' split-K twin ({@code _SKSplit} over the twin tile of
     * {@link MmqConfig#smallTile}) equals the plain kernel bit for bit over one whole-K slice, and
     * its two-slice form matches within the fp32 plane sums the split adds.
     */
    @Test
    public void theSmallTileSplitKTwinMatchesThePlainKernel() throws InterruptedException {
        theSmallTileSplitKMatchesThePlainKernel(MmqConfig.TOKEN_LANES, 9);
    }

    /**
     * The 64-token twin's split-K ({@code _SKSplit64}) equals the plain kernel bit for bit over one
     * whole-K slice, and its two-slice form matches within the fp32 plane sums the split adds.
     */
    @Test
    public void theTwin64SplitKMatchesThePlainKernel() throws InterruptedException {
        theSmallTileSplitKMatchesThePlainKernel(MmqConfig.TWIN_MAX_TOKENS, 32);
    }

    /** The split-K twin of the twin tile of {@code tokens} tokens: whole-K bit for bit, two slices
     *  within the fp32 plane sums. */
    private void theSmallTileSplitKMatchesThePlainKernel(int tokens, int n) throws InterruptedException {
        FastGemmParity parity = new FastGemmParity(backend, kernels);
        int k = 4 * type.getBlockSize();
        MmqConfig twin = MmqConfig.smallTile(config, tokens);
        int qStride = twin.rowRecords(k);
        MmqGemmPlan plain = MmqGemmPlan.contiguous(config, type, M, k, n);
        SplitKGemmPlan split = SplitKGemmPlan.ofMmq(type, M, k, n, qStride, 0, 1, k, twin, true, 256);
        String splitSuffix = tokens == MmqConfig.TWIN_MAX_TOKENS ? GemmKernels.MATMUL_MMQ_SKSPLIT64_SUFFIX
                : GemmKernels.MATMUL_MMQ_SKSPLIT_SUFFIX;
        assertTrue(split.kernelName() + " is no small-tile split-K twin",
                split.kernelName().endsWith(splitSuffix));
        byte[] weight = OpenClMmqGemmParityTest.buildWeight(type, M, k);
        float[] x = OpenClMmqGemmParityTest.buildX(n, k);
        float[] plainOut = parity.launchMmq(plain, weight, x);
        float[] partial = launchSplitPartials(split, weight, x);
        assertBits(type + " SKSplit" + tokens + " n=" + n + " k=" + k, plainOut, partial);
        SplitKGemmPlan two = SplitKGemmPlan.ofMmq(type, M, k, n, qStride, 0, 2, k / 2, twin, true, 256);
        float[] splitOut = parity.launchMmqSplit(two, weight, x);
        for (int i = 0; i < n * M; i++) {
            assertTrue(two.kernelName() + " element " + i + ": " + splitOut[i] + " plain " + plainOut[i],
                    Math.abs(splitOut[i] - plainOut[i]) <= 5e-3 * (1.0 + Math.abs(plainOut[i])));
        }
    }

    /** The twin launch over contiguous staged rows and output rows at the 16-token twin tile. */
    private MmqGemmPlan smallPlan(int m, int k, int n) {
        return smallPlan(m, k, n, MmqConfig.TOKEN_LANES);
    }

    /** The twin launch of the twin tile of {@code tokens} tokens. */
    private MmqGemmPlan smallPlan(int m, int k, int n, int tokens) {
        return MmqGemmPlan.smallOf(config, type, tokens, m, k, n,
                MmqConfig.smallTile(config, tokens).rowRecords(k), 0, m, 0);
    }

    /**
     * The split-K kernel's raw slice planes over the staged records: {@code LLM_Quant_MMQ_Q8_1}
     * stages the activations (the launch contract of {@link FastGemmParity#launchMmqSplit}), then
     * the slice kernel runs without the host reduce and the planes are read directly.
     */
    private float[] launchSplitPartials(SplitKGemmPlan split, byte[] weight, float[] x)
            throws InterruptedException {
        int k = split.k();
        int n = split.n();
        int tk = config.stageDepth();
        long records = (long) n * config.rowRecords(k);
        GpuBuffer wbuf = backend.allocateBuffer(weight.length, true);
        GpuBuffer xbuf = backend.allocateBuffer((long) x.length * Float.BYTES, true);
        GpuBuffer qbuf = backend.allocateBuffer(records * config.recordUnits()
                * Integer.BYTES);
        GpuBuffer pbuf = backend.allocateBuffer(split.partialElements() * Float.BYTES);
        try {
            try (GpuEvent wv = queue.enqueueWriteBuffer(wbuf, 0, weight.length, weight);
                 GpuEvent xv = queue.enqueueWriteBuffer(xbuf, 0, (long) x.length * Float.BYTES, x, wv)) {
                Object[] quantArgs = {n, k, tk, k, 0, xbuf, qbuf};
                queue.enqueueKernelNoEvent(kernels.get(GemmKernels.QUANT_MMQ_Q8_1),
                        new long[]{roundUp(records, 64)}, new long[]{64}, quantArgs, xv);
                queue.enqueueKernelNoEvent(kernels.get(split.kernelName()), split.partialGlobalWorkSize(),
                        split.partialLocalWorkSize(), split.partialArgs(wbuf, qbuf, pbuf));
                float[] out = new float[(int) split.partialElements()];
                try (GpuEvent read = queue.enqueueReadBuffer(pbuf, 0,
                        (long) out.length * Float.BYTES, out)) {
                    read.awaitCompletion();
                }
                return out;
            }
        } finally {
            wbuf.close();
            xbuf.close();
            qbuf.close();
            pbuf.close();
        }
    }

    /** Every output of the twin launch is the plain kernel's exact bit pattern. */
    private static void assertBits(String what, float[] plain, float[] twin) {
        assertTrue(what + ": output lengths differ, " + twin.length + " vs " + plain.length,
                twin.length == plain.length);
        for (int i = 0; i < plain.length; i++) {
            assertTrue(what + " element " + i + ": twin " + Float.toHexString(twin[i]) + " ("
                            + Integer.toHexString(Float.floatToRawIntBits(twin[i])) + ") plain "
                            + Float.toHexString(plain[i]) + " ("
                            + Integer.toHexString(Float.floatToRawIntBits(plain[i])) + ")",
                    Float.floatToRawIntBits(twin[i]) == Float.floatToRawIntBits(plain[i]));
        }
    }

    private static long roundUp(long v, long multiple) {
        return (v + multiple - 1) / multiple * multiple;
    }
}

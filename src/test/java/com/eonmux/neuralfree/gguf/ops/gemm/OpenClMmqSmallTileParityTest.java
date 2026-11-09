package com.eonmux.neuralfree.gguf.ops.gemm;

import com.eonmux.neuralfree.acceleration.HWAcceleration;
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

import java.util.ArrayList;
import java.util.Collection;

import static org.junit.Assert.assertTrue;

/**
 * The small-token-tile twins {@code <T>_MatMul_MMQ_S} and {@code _S64} on the OpenCL device, under
 * the forced {@code rdna3} tile ({@link MmqConfig#TILE_PROPERTY}) whose 128-token tile makes the
 * OpenCL program generate the twins: every MMQ weight type's twin at the multi-row step sizes (9,
 * 13, 16, 24, 31, 64 tokens, the partial tiles included) must match the exact dot of the hand-built
 * weight's decoded codes with the staged activations, and the plain kernel of the same tile over the
 * same staged records. The token-tile change must not move the results, only the tile the weight
 * stream fills ({@code MmqGemmPlan.smallOf} plans the launches). The rdna3 tile is the widest
 * twin-generating tile whose 256-item workgroups the OpenCL NVIDIA kernels dispatch; the int8-tensor
 * tile needs 512 (the CUDA twin runs in {@link CudaMmqSmallTileParityTest}).
 */
@RunWith(Parameterized.class)
public class OpenClMmqSmallTileParityTest {

    private static final int M = 70;  /* one full 64-row tile plus a 6-row partial */

    @Parameterized.Parameters(name = "{0}")
    public static Collection<GGMLType> types() {
        return new ArrayList<>(GemmKernels.MMQ_TYPES);
    }

    private final GGMLType type;

    private HWAcceleration hwAccel;
    private OclComputeBackend backend;
    private LlmKernels kernels;

    public OpenClMmqSmallTileParityTest(GGMLType type) {
        this.type = type;
    }

    @Before
    public void setUp() {
        HWAcceleration hw = null;
        try {
            hw = new HWAcceleration();
            Assume.assumeTrue("no OpenCL device", !hw.getCLDevices().isEmpty());
            CLDevice device = hw.getCLDevices().get(0);
            System.setProperty(MmqConfig.TILE_PROPERTY, MmqConfig.RDNA3.name());
            OclComputeBackend b = new OclComputeBackend(device);
            Availability availability = LlmKernels.availability(b);
            Assume.assumeTrue("LLM kernels unavailable on " + b.capabilities().getDeviceName() + ": "
                    + availability.reason(), availability.available());
            hwAccel = hw;
            backend = b;
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
        System.clearProperty(MmqConfig.TILE_PROPERTY);
        if (backend != null) {
            backend.close();
        }
        if (hwAccel != null) {
            hwAccel.shutdown();
        }
    }

    /**
     * The twin at the step sizes of a speculative verify: 9 tokens (one partial 16-token tile) and
     * 13 tokens, against the exact dot of the decoded weight with the staged codes; K is one whole
     * weight block, the shortest reduction the stage depth covers.
     */
    @Test
    public void theSmallTileTwinMatchesTheDecodedDotAtVerifyStepSizes() throws InterruptedException {
        int k = type.getBlockSize();
        for (int n = 9; n <= 13; n += 4) {
            MmqGemmPlan plan = smallPlan(M, k, n);
            assertMatchesDecodedDot(plan, n, k);
        }
    }

    /**
     * The twin across several 16-token tiles: 16 tokens (one full tile), 31 (two, the second
     * partial) and 64 (four), at row counts inside and past the row tile; K = 3 weight blocks.
     */
    @Test
    public void theSmallTileTwinMatchesTheDecodedDotAcrossTiles() throws InterruptedException {
        int k = 3 * type.getBlockSize();
        for (int n : new int[]{16, 31, 64}) {
            for (int m : new int[]{13, M}) {
                assertMatchesDecodedDot(smallPlan(m, k, n), n, k);
            }
        }
    }

    /**
     * The 64-token twin across whole and partial 64-token tiles: 24 tokens (one partial tile) and
     * 64 (one full), at row counts inside and past the row tile; K = 3 weight blocks.
     */
    @Test
    public void theTwin64MatchesTheDecodedDotAcrossTiles() throws InterruptedException {
        int k = 3 * type.getBlockSize();
        for (int n : new int[]{24, 64}) {
            for (int m : new int[]{13, M}) {
                assertMatchesDecodedDot(smallPlan(m, k, n, MmqConfig.TWIN_MAX_TOKENS), n, k);
            }
        }
    }

    /**
     * Twin and plain kernel of the same forced tile agree over the same staged records: the tile only
     * changes which tokens a workgroup covers, never a result's arithmetic.
     */
    @Test
    public void theSmallTileTwinMatchesThePlainKernel() throws InterruptedException {
        FastGemmParity parity = new FastGemmParity(backend, kernels);
        int k = 3 * type.getBlockSize();
        for (int n : new int[]{9, 16}) {
            MmqGemmPlan plain = MmqGemmPlan.contiguous(MmqConfig.RDNA3, type, M, k, n);
            MmqGemmPlan small = smallPlan(M, k, n);
            assertTrue(small.kernelName() + " is no small-tile twin",
                    small.kernelName().endsWith(GemmKernels.MATMUL_MMQ_S_SUFFIX));
            byte[] weight = OpenClMmqGemmParityTest.buildWeight(type, M, k);
            float[] x = OpenClMmqGemmParityTest.buildX(n, k);
            float[] plainOut = parity.launchMmq(plain, weight, x);
            float[] smallOut = parity.launchMmq(small, weight, x);
            for (int i = 0; i < n * M; i++) {
                assertTrue(small.kernelName() + " element " + i + ": " + smallOut[i] + " plain "
                        + plainOut[i], Math.abs(smallOut[i] - plainOut[i]) <= 1e-3 * (1.0 + Math.abs(plainOut[i])));
            }
        }
    }

    /**
     * The wave-fill split the dispatch takes for a staged launch of fewer tokens than the device tile
     * ({@link SplitKGemmPlan#ofSmallTiles}) matches the plain kernel over the same staged records: the
     * fp32 plane sums of the slices differ from the plain kernel's sequential accumulation by rounding
     * only, as {@link FastGemmParity#mmqSplitKMatchesPlain} checks for the tall-skinny heuristic.
     */
    @Test
    public void theSmallTileSplitKTwinMatchesThePlainKernel() throws InterruptedException {
        theSmallTileSplitKMatchesThePlainKernel(MmqConfig.TOKEN_LANES, 9);
    }

    /** The dispatch's 64-token twin split ({@code _SKSplit64} over the S64 tile of a 24-token
     *  launch) matches the plain kernel within the fp32 plane sums. */
    @Test
    public void theTwin64SplitKMatchesThePlainKernel() throws InterruptedException {
        theSmallTileSplitKMatchesThePlainKernel(MmqConfig.TWIN_MAX_TOKENS, 24);
    }

    /** The dispatch's split of a staged launch of {@code n} tokens, at the twin tile its ladder row
     *  picks, against the plain kernel over the same staged records. */
    private void theSmallTileSplitKMatchesThePlainKernel(int tokens, int n) throws InterruptedException {
        Assume.assumeTrue("the device does not report compute units",
                backend.capabilities().getComputeUnits() > 0);
        FastGemmParity parity = new FastGemmParity(backend, kernels);
        int k = Math.max(2048 / type.getBlockSize(), 2) * type.getBlockSize();
        MmqGemmPlan base = MmqGemmPlan.of(MmqConfig.RDNA3, type, M, k, n,
                MmqConfig.RDNA3.rowRecords(k), 0, M, 0);
        SplitKGemmPlan split = SplitKGemmPlan.ofSmallTiles(base, backend.capabilities().getComputeUnits(), 256);
        Assume.assumeTrue("the wave-fill heuristic does not split on this device", split != null);
        assertTrue(split.kernelName().endsWith(tokens == MmqConfig.TWIN_MAX_TOKENS
                ? GemmKernels.MATMUL_MMQ_SKSPLIT64_SUFFIX : GemmKernels.MATMUL_MMQ_SKSPLIT_SUFFIX));
        byte[] weight = OpenClMmqGemmParityTest.buildWeight(type, M, k);
        float[] x = OpenClMmqGemmParityTest.buildX(n, k);
        float[] plainOut = parity.launchMmq(MmqGemmPlan.contiguous(MmqConfig.RDNA3, type, M, k, n), weight, x);
        float[] splitOut = parity.launchMmqSplit(split, weight, x);
        for (int i = 0; i < n * M; i++) {
            assertTrue(split.kernelName() + " element " + i + ": " + splitOut[i] + " plain " + plainOut[i],
                    Math.abs(splitOut[i] - plainOut[i]) <= 5e-3 * (1.0 + Math.abs(plainOut[i])));
        }
    }

    /** The twin launch over contiguous staged rows and output rows at the 16-token twin tile. */
    private MmqGemmPlan smallPlan(int m, int k, int n) {
        return smallPlan(m, k, n, MmqConfig.TOKEN_LANES);
    }

    /** The twin launch of the twin tile of {@code tokens} tokens. */
    private MmqGemmPlan smallPlan(int m, int k, int n, int tokens) {
        return MmqGemmPlan.smallOf(MmqConfig.RDNA3, type, tokens, m, k, n,
                MmqConfig.smallTile(MmqConfig.RDNA3, tokens).rowRecords(k), 0, m, 0);
    }

    /**
     * The twin's [n x m] output against the exact dot of the decoded weight with the staged codes, as
     * {@link OpenClMmqGemmParityTest} checks its plain kernel: the reference quantizes per chunk of
     * the plan's stage depth (the rdna3 tile stages 16-deep).
     */
    private void assertMatchesDecodedDot(MmqGemmPlan plan, int n, int k) throws InterruptedException {
        FastGemmParity parity = new FastGemmParity(backend, kernels);
        int tk = plan.config().stageDepth();
        int m = plan.m();
        byte[] weight = OpenClMmqGemmParityTest.buildWeight(type, m, k);
        float[] x = OpenClMmqGemmParityTest.buildX(n, k);
        float[] w = CpuBlockDecoders.dequantize(weight, type, (long) m * k);
        double[] q = new double[k];
        double[] d = new double[k];
        float[] out = parity.launchMmq(plan, weight, x);
        for (int t = 0; t < n; t++) {
            quantizeChunked(x, t * k, q, d, tk);
            for (int r = 0; r < m; r++) {
                double acc = 0.0;
                double terms = 0.0;
                double bound = 0.0;
                for (int i = 0; i < k; i++) {
                    double term = (double) w[(int) ((long) r * k + i)] * q[i];
                    acc += term;
                    terms += Math.abs(term);
                    /* one activation code step per element (the quantization rounding) */
                    bound += Math.abs((double) w[(int) ((long) r * k + i)] * d[i]);
                }
                double tol = bound + 1e-4 * terms + 1e-3;
                assertTrue(type + " n=" + n + " m=" + m + " token " + t + " row " + r + ": kernel "
                        + out[t * m + r] + " expected " + acc + " (|terms| " + terms + ")",
                        Math.abs(out[t * m + r] - acc) <= tol);
            }
        }
    }

    /**
     * The kernel's activation quantization of one token: per {@code tk}-element chunk, d = maxabs *
     * (1f/127) and the codes round to nearest even of x * (127f / maxabs); writes the code values
     * d * q, indexed by element offset of the token, and the chunk scales.
     */
    private static void quantizeChunked(float[] x, int base, double[] out, double[] dOut, int tk) {
        for (int c = 0; c < out.length; c += tk) {
            float maxabs = 0.0f;
            for (int i = c; i < c + tk; i++) {
                maxabs = Math.max(maxabs, Math.abs(x[base + i]));
            }
            float id = maxabs > 0.0f ? 127.0f / maxabs : 0.0f;
            float d = maxabs > 0.0f ? maxabs * (1.0f / 127.0f) : 1.0f;
            for (int i = c; i < c + tk; i++) {
                out[i] = (double) d * Math.rint((double) (x[base + i] * id));
                dOut[i] = d;
            }
        }
    }
}

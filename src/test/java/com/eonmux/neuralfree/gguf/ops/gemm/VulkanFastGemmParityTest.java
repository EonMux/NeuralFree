package com.eonmux.neuralfree.gguf.ops.gemm;

import com.eonmux.neuralfree.acceleration.backend.DeviceFeature;
import com.eonmux.neuralfree.acceleration.backend.vulkan.VkComputeBackend;
import com.eonmux.neuralfree.acceleration.vulkan.VkInstanceManager;
import com.eonmux.neuralfree.acceleration.vulkan.VkPhysicalDeviceInfo;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;

/**
 * The GLSL twins of the fast GEMM kernels on a Vulkan device against the tiled GEMM of the same
 * weight type ({@link FastGemmParity}): every int8 MMQ kernel on any device, and on devices with
 * {@link DeviceFeature#COOPERATIVE_MATRIX} every fp16-tile CM kernel, the F16 and BF16 MMA rows and
 * the split-K pair of both dense 16-bit types, and on devices with
 * {@link DeviceFeature#COOPERATIVE_MATRIX_2} every workgroup-scope CM2 kernel and its split-K twin, the dense
 * F16 one included.
 * Skips without a Vulkan GPU.
 */
public class VulkanFastGemmParityTest {

    private static final int M = 70;   /* two 64-row tiles, the second partial */
    private static final int K = 512;  /* two K-quant super-blocks per row */
    private static final int N = 40;   /* MMQ: three 16-token tiles; CM/MMA: two 32-token tiles */
    private static final int M2 = 150; /* two 128-row CM2 tiles, the second partial */
    private static final int N2 = 300; /* a full 256-token CM2 tile and a quarter-width tail */
    private static final int M3 = 256; /* two whole 128-row CM2 tiles: the unclamped interior path */
    private static final int N3 = 100; /* one CM2 token tile ending in its second quarter: half width */

    private VkComputeBackend backend;
    private FastGemmParity parity;

    @Before
    public void setUp() {
        try {
            backend = openPreferringCooperativeMatrix();
            parity = new FastGemmParity(backend, LlmKernels.open(backend));
        } catch (Throwable e) {
            if (backend != null) {
                backend.close();
            }
            backend = null;
        }
        Assume.assumeTrue("no Vulkan GPU", backend != null);
    }

    /** A device reporting cooperative matrices when one is enumerated, else the fastest device. */
    private static VkComputeBackend openPreferringCooperativeMatrix() {
        VkInstanceManager manager = VkInstanceManager.acquire();
        try {
            for (VkPhysicalDeviceInfo info : manager.devices()) {
                if (info.supports(VkPhysicalDeviceInfo.Feature.COOPERATIVE_MATRIX_2)) {
                    return new VkComputeBackend(manager, info);
                }
            }
            for (VkPhysicalDeviceInfo info : manager.devices()) {
                if (info.supports(VkPhysicalDeviceInfo.Feature.COOPERATIVE_MATRIX)) {
                    return new VkComputeBackend(manager, info);
                }
            }
        } finally {
            manager.release();
        }
        return new VkComputeBackend();
    }

    @After
    public void tearDown() {
        if (backend != null) {
            backend.close();
        }
    }

    private void assumeCooperativeMatrix() {
        Assume.assumeTrue("no cooperative-matrix support on " + backend.capabilities().getDeviceName(),
                backend.capabilities().hasFeature(DeviceFeature.COOPERATIVE_MATRIX));
    }

    private void assumeCooperativeMatrix2() {
        Assume.assumeTrue("no VK_NV_cooperative_matrix2 support on " + backend.capabilities().getDeviceName(),
                backend.capabilities().hasFeature(DeviceFeature.COOPERATIVE_MATRIX_2));
    }

    @Test
    public void everyMmqKernelMatchesTheTiledGemmWithinTheActivationCodeStep() throws Exception {
        for (GGMLType type : GemmKernels.MMQ_TYPES) {
            parity.mmqMatchesTiled(type, M, K, N);
        }
    }

    /** The split-K twins of the int8 MMQ and CM kernels against their plain kernels. */
    @Test
    public void mmqAndCmSplitKTwinsMatchTheirPlainKernels() throws Exception {
        for (GGMLType type : GemmKernels.MMQ_TYPES) {
            parity.mmqSplitKMatchesPlain(type, M, 768, N, 3);
        }
        assumeCooperativeMatrix();
        for (GGMLType type : GemmKernels.MMQ_TYPES) {
            parity.cmSplitKMatchesPlain(type, M, 768, N, 3);
        }
    }

    @Test
    public void everyCmKernelMatchesTheTiledGemmWithinFp16Rounding() throws Exception {
        assumeCooperativeMatrix();
        for (GGMLType type : GemmKernels.MMQ_TYPES) {
            parity.cmMatchesTiled(type, M, K, N);
        }
    }

    /**
     * The workgroup-scope CM2 kernels over every type that has one: a shape inside one 128 x 256 tile
     * (partial rows through the clamped layouts, a quarter-width token tile), two row tiles (one interior,
     * one partial) across a full token tile and a quarter-width tail, and whole row tiles through the
     * unrolled interior path at half width.
     */
    @Test
    public void everyCm2KernelMatchesTheTiledGemmWithinFp16Rounding() throws Exception {
        assumeCooperativeMatrix2();
        for (GGMLType type : quantizedCm2Types()) {
            parity.cm2OneHotColumnsMatchCpu(type);
            parity.cm2MatchesTiled(type, M, K, N);
            parity.cm2MatchesTiled(type, M2, 256, N2);
            parity.cm2MatchesTiled(type, M3, K, N3);
        }
    }

    /** The split-K twin of the CM2 GEMM against its plain kernel, on clamped and interior row tiles and on
     *  192-element slices that start inside a K-quant block (the staged Q4_K / Q5_K scales of a slice's first,
     *  partial block): one-hot products to the fp32 plane sum, full reductions within the fp16-accumulation
     *  bound. */
    @Test
    public void cm2SplitKTwinsMatchTheirPlainKernels() throws Exception {
        assumeCooperativeMatrix2();
        for (GGMLType type : quantizedCm2Types()) {
            parity.cm2SplitKMatchesPlain(type, M, 768, N, 3);
            parity.cm2SplitKMatchesPlain(type, M3, 768, N3, 3);
            parity.cm2SplitKMatchesPlain(type, M3, 768, N3, 4);
        }
    }

    /**
     * The dense F16 CM2 kernel over fp16-exact operands, within the fp32 accumulation bound: one tile with
     * partial rows through the clamped layouts and a quarter-width token tile, a full token tile plus a tail over a
     * reduction that ends inside a stage (clamped K), whole row tiles through the unrolled interior path at half
     * width, and the split-K twin on clamped and interior tiles.
     */
    @Test
    public void denseF16Cm2MatchesTheTiledGemmAndItsSplitKTwin() throws Exception {
        assumeCooperativeMatrix2();
        int[][] shapes = {{M, K, N}, {M2, 520, N2}, {M3, 1024, N3}};
        for (int[] s : shapes) {
            float[] w = FastGemmParity.denseValues(s[0] * s[1], 0x51ED);
            float[] x = FastGemmParity.denseValues(s[2] * s[1], 0xC0DE);
            parity.cm2DenseMatchesTiled(FastGemmParity.f16Bytes(w), w, x, s[0], s[1], s[2]);
        }
        float[] w = FastGemmParity.denseValues(M3 * 1536, 0x51ED);
        float[] x = FastGemmParity.denseValues(N * 1536, 0xC0DE);
        parity.cm2DenseSplitKMatchesPlain(FastGemmParity.f16Bytes(w), w, x, M3, 1536, N, 3);
        float[] w2 = FastGemmParity.denseValues(M * 1536, 0x51ED);
        parity.cm2DenseSplitKMatchesPlain(FastGemmParity.f16Bytes(w2), w2, x, M, 1536, N, 4);
    }

    /**
     * The dense F16 CM2 kernel, plain and split-K, and the F16 MMA GEMM against the exact products of a
     * feed-forward down projection's shape (K = 3072, two whole row tiles, a full token tile and a quarter-width
     * tail) whose outlier activations drive one row's sums to 8.6e4 .. 1.7e5: every sum, in range or past the fp16
     * range, matches within the fp32 accumulation bound.
     */
    @Test
    public void denseF16Cm2KeepsSumsPastTheFp16RangeWithinTheFp32Bound() throws Exception {
        assumeCooperativeMatrix2();
        int k = 3072;
        float[] w = FastGemmParity.outlierWeights(M3, k, 0x51ED);
        float[] x = FastGemmParity.outlierActivations(N2, k, 0xC0DE);
        parity.cm2DenseMatchesExactProducts(FastGemmParity.f16Bytes(w), w, x, M3, k, N2, 3);
    }

    /** The CM2 types whose weights are quantized blocks (every one but the dense F16). */
    private static java.util.List<GGMLType> quantizedCm2Types() {
        java.util.List<GGMLType> types = new java.util.ArrayList<>();
        for (GGMLType type : GemmKernels.CM2_TYPES) {
            if (GemmKernels.hasMmq(type)) {
                types.add(type);
            }
        }
        return types;
    }

    @Test
    public void denseMmaRowsMatchTheTiledGemm() throws Exception {
        assumeCooperativeMatrix();
        float[] w = FastGemmParity.denseValues(M * K, 0x51ED);
        float[] x = FastGemmParity.denseValues(N * K, 0xC0DE);
        parity.mmaMatchesTiled(GGMLType.F16, FastGemmParity.f16Bytes(w), w, x, M, K, N, 1e-5);
        parity.mmaMatchesTiled(GGMLType.BF16, FastGemmParity.bf16Bytes(w), w, x, M, K, N, 1e-5);
    }

    @Test
    public void splitKPairMatchesThePlainMmaKernel() throws Exception {
        assumeCooperativeMatrix();
        int k = 2425;
        float[] w = FastGemmParity.denseValues(M * k, 0x51ED);
        float[] x = FastGemmParity.denseValues(N * k, 0xC0DE);
        parity.splitKMatchesPlain(GGMLType.F16, FastGemmParity.f16Bytes(w), w, x, M, k, N, 3);
        parity.splitKMatchesPlain(GGMLType.BF16, FastGemmParity.bf16Bytes(w), w, x, M, k, N, 3);
    }
}

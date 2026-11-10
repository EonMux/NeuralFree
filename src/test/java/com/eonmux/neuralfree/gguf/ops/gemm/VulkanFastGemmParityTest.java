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
 * {@link DeviceFeature#COOPERATIVE_MATRIX_2} every workgroup-scope CM2 kernel and its split-K twin.
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
        for (GGMLType type : GemmKernels.CM2_TYPES) {
            parity.cm2OneHotColumnsMatchCpu(type);
            parity.cm2MatchesTiled(type, M, K, N);
            parity.cm2MatchesTiled(type, M2, 256, N2);
            parity.cm2MatchesTiled(type, M3, K, N3);
        }
    }

    /** The split-K twin of the CM2 GEMM against its plain kernel, on clamped and interior row tiles: one-hot
     *  products to the fp32 plane sum, full reductions within the fp16-accumulation bound. */
    @Test
    public void cm2SplitKTwinsMatchTheirPlainKernels() throws Exception {
        assumeCooperativeMatrix2();
        for (GGMLType type : GemmKernels.CM2_TYPES) {
            parity.cm2SplitKMatchesPlain(type, M, 768, N, 3);
            parity.cm2SplitKMatchesPlain(type, M3, 768, N3, 3);
        }
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

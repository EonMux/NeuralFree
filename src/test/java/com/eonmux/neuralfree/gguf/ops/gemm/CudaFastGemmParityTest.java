package com.eonmux.neuralfree.gguf.ops.gemm;

import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.cuda.CudaComputeBackend;
import com.eonmux.neuralfree.acceleration.cuda.CUDevice;
import com.eonmux.neuralfree.acceleration.kernels.Availability;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;

/**
 * The fast GEMM kernels on the CUDA device against the tiled GEMM of the same weight type
 * ({@link FastGemmParity}): every int8 MMQ kernel (the {@code mma.m16n8k32} / {@code m16n8k16}
 * int8 tensor path on sm_80+, dp4a below), the F16 and BF16 MMA kernels and the split-K pair.
 */
public class CudaFastGemmParityTest {

    private static final int M = 70;   /* one partial row tile of the MMQ tensor tile (two 64-row packed-dot tiles) */
    private static final int K = 512;  /* two K-quant super-blocks per row */
    private static final int N = 40;   /* MMQ: one partial token tile; MMA: two 32-token tiles */
    /* The whole MMQ tensor tile: three 128-row tiles and two 128-token tiles, each the last ragged, so every warp's
     * row group and both warp token groups (tokens 0-63 and 64-127) carry live outputs. */
    private static final int WIDE_M = 333;
    private static final int WIDE_N = 141;

    private HWAcceleration hwAccel;
    private CudaComputeBackend backend;
    private FastGemmParity parity;
    private int computeCapability;

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
            Assume.assumeTrue("LLM kernels unavailable on " + b.capabilities().getDeviceName() + ": "
                    + availability.reason(), availability.available());
            hwAccel = hw;
            backend = b;
            parity = new FastGemmParity(b, LlmKernels.open(b));
            computeCapability = b.capabilities().getComputeCapability();
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
    public void everyMmqKernelMatchesTheTiledGemmWithinTheActivationCodeStep() throws Exception {
        StringBuilder bad = new StringBuilder();
        for (GGMLType type : GemmKernels.MMQ_TYPES) {
            try {
                parity.mmqMatchesTiled(type, M, K, N);
            } catch (AssertionError | RuntimeException e) {
                bad.append(e.getMessage()).append(';').append('\n');
            }
        }
        org.junit.Assert.assertTrue("failed types: " + bad, bad.length() == 0);
    }

    @Test
    public void everyMmqKernelMatchesTheTiledGemmOverWholeTiles() throws Exception {
        StringBuilder bad = new StringBuilder();
        for (GGMLType type : GemmKernels.MMQ_TYPES) {
            try {
                parity.mmqMatchesTiled(type, WIDE_M, K, WIDE_N);
            } catch (AssertionError | RuntimeException e) {
                bad.append(e.getMessage()).append(';').append('\n');
            }
        }
        org.junit.Assert.assertTrue("failed types: " + bad, bad.length() == 0);
    }

    /** Records off 16-byte alignment (a 1-unit offset) take the tile's 4-byte record copy; 4 units, its 16-byte copy. */
    @Test
    public void everyMmqKernelReadsOffsetRecordsBitIdentically() throws Exception {
        StringBuilder bad = new StringBuilder();
        for (GGMLType type : GemmKernels.MMQ_TYPES) {
            for (int qOff : new int[]{1, 4}) {
                try {
                    parity.mmqRecordOffsetMatchesAligned(type, WIDE_M, K, WIDE_N, qOff);
                } catch (AssertionError | RuntimeException e) {
                    bad.append(e.getMessage()).append(';').append('\n');
                }
            }
        }
        org.junit.Assert.assertTrue("failed types: " + bad, bad.length() == 0);
    }

    @Test
    public void denseMmaKernelsMatchTheTiledGemm() throws Exception {
        Assume.assumeTrue("the MMA tensor path needs sm_80+", computeCapability >= 80);
        float[] w = FastGemmParity.denseValues(M * K, 0x51ED);
        float[] x = FastGemmParity.denseValues(N * K, 0xC0DE);
        parity.mmaMatchesTiled(GGMLType.F16, FastGemmParity.f16Bytes(w), w, x, M, K, N, 1e-5);
        parity.mmaMatchesTiled(GGMLType.BF16, FastGemmParity.bf16Bytes(w), w, x, M, K, N, 1e-5);
    }

    @Test
    public void splitKPairMatchesThePlainMmaKernel() throws Exception {
        Assume.assumeTrue("the MMA tensor path needs sm_80+", computeCapability >= 80);
        int k = 2425;
        float[] w = FastGemmParity.denseValues(M * k, 0x51ED);
        float[] x = FastGemmParity.denseValues(N * k, 0xC0DE);
        parity.splitKMatchesPlain(GGMLType.F16, FastGemmParity.f16Bytes(w), w, x, M, k, N, 3);
        parity.splitKMatchesPlain(GGMLType.BF16, FastGemmParity.bf16Bytes(w), w, x, M, k, N, 8);
    }
}

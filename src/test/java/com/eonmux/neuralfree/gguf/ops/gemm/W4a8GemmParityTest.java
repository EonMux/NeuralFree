package com.eonmux.neuralfree.gguf.ops.gemm;

import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.acceleration.backend.cuda.CudaComputeBackend;
import com.eonmux.neuralfree.acceleration.cuda.CUDevice;
import com.eonmux.neuralfree.acceleration.kernels.Availability;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.quant.ActFakeQuant;
import com.eonmux.neuralfree.gguf.quant.ActQuantScheme;
import com.eonmux.neuralfree.gguf.quant.CpuBlockDecoders;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertTrue;

/**
 * The W4A8 lane: a 4-bit weight (MXFP4, NVFP4, IQ4_NL, IQ4_XS) whose codes are mapped to E4M3 codes and multiplied
 * by the staged E4M3 activation codes of an E4M3-stageable activation scheme ({@code <T>_MatMul_W4A8}). The MXFP4
 * and NVFP4 codebooks are E2M1 values, which E4M3 represents exactly, so their lane matches the fake-quantized
 * reference up to the fp32 accumulation; the non-linear IQ4 codebooks and the NVFP4 per-16 weight scales need one
 * E4M3 rounding per weight element, for which the comparison states its bound.
 *
 * <p>Every launch takes the geometry of the tile the kernel is compiled with
 * ({@link #tileOf}, the {@code FastGemmLanes.fastTileRows} / {@code fastTileTokens} / {@code fastWorkGroup}
 * derivation the dispatch itself uses: the device's {@link MmqConfig} for the plain kernel,
 * {@link MmqConfig#smallTile} for the small-token-tile twins), never literals &mdash; the tile the CUDA fatbin
 * builds a kernel with follows the device ({@code MmqConfig.INT8_TENSOR}, 128 rows x 128 tokens over 512
 * work-items, on sm_80+), and a launch planned for another tile leaves whole row groups unwritten.</p>
 */
public class W4a8GemmParityTest {

    private static final int M = 70;
    private static final int K = 512;
    private static final float STATIC_SCALE = 0.25f;

    private HWAcceleration hwAccel;
    private CudaComputeBackend backend;
    private ComputeQueue queue;
    private LlmKernels kernels;

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
            Assume.assumeTrue("the E4M3 GEMM needs sm_89+", b.capabilities().getComputeCapability() >= 89);
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
    public void mxFp4WeightsRunTheE4m3Gemm() throws Exception {
        check(GGMLType.MXFP4, 0.0);
    }

    @Test
    public void iq4WeightsRunTheE4m3Gemm() throws Exception {
        check(GGMLType.IQ4_NL, 0.06);
        check(GGMLType.IQ4_XS, 0.06);
    }

    @Test
    public void nvFp4WeightsRunTheE4m3Gemm() throws Exception {
        check(GGMLType.NVFP4, 0.08);
    }

    /** One launch of the parity matrix: the staged GEMM kernel and its token count. */
    private static final class Launch {

        final String kernel;
        final int tokens;

        Launch(String kernel, int tokens) {
            this.kernel = kernel;
            this.tokens = tokens;
        }
    }

    /**
     * The kernels and token counts of one type's parity matrix: the plain kernel at the device tile over a
     * partial token tile (21) and across two device tiles (129), and &mdash; where the device tile is wider
     * and the fatbin generates them &mdash; the 16-token twin at a partial (9) and a two-tile (21) token
     * count, the 64-token twin at a partial (48) and a two-tile (100) token count: a token count in every
     * band of the {@link MmqConfig#twinTokens(int, int)} ladder, as {@link CudaMmqSmallTileParityTest}
     * covers for the int8 MMQ types. A wider device tile always carries the twins on the sm_89+ device
     * this test assumes (the CUDA dialect generates them for every tensor-path segment), so a missing
     * twin fails instead of silently shrinking the matrix.
     */
    private List<Launch> launches(GGMLType type, MmqConfig config) {
        List<Launch> launches = new ArrayList<>();
        String plain = GemmKernels.matmulW4a8(type);
        launches.add(new Launch(plain, 21));
        launches.add(new Launch(plain, 129));
        if (config.tileTokens() > MmqConfig.TOKEN_LANES) {
            String twin = GemmKernels.matmulW4a8Small(type);
            assertTrue(twin + " is missing although the device tile is " + config.tileTokens()
                    + " tokens wide", kernels.has(twin));
            launches.add(new Launch(twin, 9));
            launches.add(new Launch(twin, 21));
            if (config.tileTokens() > MmqConfig.TWIN_MAX_TOKENS) {
                String twin64 = GemmKernels.matmulW4a8Small(type, MmqConfig.TWIN_MAX_TOKENS);
                assertTrue(twin64 + " is missing although the device tile is " + config.tileTokens()
                        + " tokens wide", kernels.has(twin64));
                launches.add(new Launch(twin64, 48));
                launches.add(new Launch(twin64, 100));
            }
        }
        return launches;
    }

    /**
     * The lane equals the fake-quantized reference within {@code weightTolerance} of every term (the E4M3 rounding
     * of the weight code the lane performs) plus fp32 accumulation, at every launch of the parity matrix.
     */
    private void check(GGMLType type, double weightTolerance) throws Exception {
        MmqConfig config = MmqConfig.forDevice(backend.backendType(),
                backend.capabilities().getComputeCapability());
        String plain = GemmKernels.matmulW4a8(type);
        Assume.assumeTrue(plain + " is unavailable", kernels.has(plain));
        for (ActQuantScheme scheme : new ActQuantScheme[]{ActQuantScheme.FP8_STATIC, ActQuantScheme.MXFP8,
                ActQuantScheme.MXFP4}) {
            String stageKernel = GemmKernels.fp8ActivationStaging(scheme);
            Assume.assumeTrue(stageKernel + " is unavailable", kernels.has(stageKernel));
            byte[] weight = OpenClMmqGemmParityTest.buildWeight(type, M, K);
            float[] w = CpuBlockDecoders.dequantize(weight, type, (long) M * K);
            for (Launch launch : launches(type, config)) {
                run(type, scheme, stageKernel, weight, w, config, launch, weightTolerance);
            }
        }
    }

    /** One staged launch of {@code launch} over contiguous views, checked against the fake-quantized reference.
     *  The output is prefilled with NaN, so a row no work-group covers fails instead of reading as zero. */
    private void run(GGMLType type, ActQuantScheme scheme, String stageKernel, byte[] weight, float[] w,
                     MmqConfig config, Launch launch, double weightTolerance) throws Exception {
        int tokens = launch.tokens;
        String gemm = launch.kernel;
        MmqConfig tile = tileOf(gemm, config);
        int tk = config.stageDepth();
        long qUnits = (long) tokens * config.rowRecords(K) * config.recordUnits();
        float[] x = OpenClMmqGemmParityTest.buildX(tokens, K);
        float[] fake = x.clone();
        ActFakeQuant.apply(scheme, scheme.isStatic() ? STATIC_SCALE : 0.0f, 0, 0.0f, fake, 0, tokens, K);
        float[] prefilled = new float[tokens * M];
        Arrays.fill(prefilled, Float.NaN);

        GpuBuffer wbuf = backend.allocateBuffer((long) weight.length, true);
        GpuBuffer xbuf = backend.allocateBuffer((long) x.length * Float.BYTES, true);
        GpuBuffer qbuf = backend.allocateBuffer(qUnits * Integer.BYTES);
        GpuBuffer ybuf = backend.allocateBuffer((long) tokens * M * Float.BYTES);
        try {
            try (GpuEvent wv = queue.enqueueWriteBuffer(wbuf, 0, weight.length, weight);
                 GpuEvent xv = queue.enqueueWriteBuffer(xbuf, 0, (long) x.length * Float.BYTES, x, wv);
                 GpuEvent yv = queue.enqueueWriteBuffer(ybuf, 0, (long) prefilled.length * Float.BYTES,
                         prefilled, xv)) {
                stage(stageKernel, scheme, tk, tokens, xbuf, qbuf, yv);
                Object[] args = {M, K, tokens, config.rowRecords(K), 0, M, 0, wbuf, qbuf, ybuf};
                long[] global = {(long) ((M + tile.tileRows() - 1) / tile.tileRows()) * tile.workGroupSize(),
                        (tokens + tile.tileTokens() - 1) / tile.tileTokens()};
                queue.enqueueKernelNoEvent(kernels.get(gemm), global, new long[]{tile.workGroupSize(), 1}, args);
                float[] out = new float[tokens * M];
                try (GpuEvent read = queue.enqueueReadBuffer(ybuf, 0, (long) out.length * Float.BYTES, out)) {
                    read.awaitCompletion();
                }
                assertWithin(type, scheme, gemm, w, fake, out, tokens, weightTolerance);
            }
        } finally {
            wbuf.close();
            xbuf.close();
            qbuf.close();
            ybuf.close();
        }
    }

    /**
     * The tile the staged GEMM kernel is compiled with, the derivation the dispatch launches it with
     * ({@code FastGemmLanes.fastTileRows} / {@code fastTileTokens} / {@code fastWorkGroup}): the device's
     * tile for the plain kernel, {@link MmqConfig#smallTile} for the {@code _S} (16-token) and
     * {@code _S64} (64-token) twins.
     */
    private static MmqConfig tileOf(String gemm, MmqConfig device) {
        if (gemm.endsWith(GemmKernels.MATMUL_W4A8_S64_SUFFIX)) {
            return MmqConfig.smallTile(device, MmqConfig.TWIN_MAX_TOKENS);
        }
        if (gemm.endsWith(GemmKernels.MATMUL_W4A8_S_SUFFIX)) {
            return MmqConfig.smallTile(device);
        }
        return device;
    }

    /** |kernel - reference| <= one E4M3 weight rounding per term plus fp32 accumulation, at every output. */
    private static void assertWithin(GGMLType type, ActQuantScheme scheme, String gemm, float[] w, float[] fake,
                                     float[] out, int tokens, double weightTolerance) {
        for (int t = 0; t < tokens; t++) {
            for (int r = 0; r < M; r++) {
                double acc = 0.0;
                double bound = 0.0;
                for (int k = 0; k < K; k++) {
                    final double term = (double) w[r * K + k] * fake[t * K + k];
                    acc += term;
                    bound += Math.abs(term) * weightTolerance;
                }
                double tol = bound + 1e-4 * Math.abs(acc) + 1e-3;
                assertTrue(type + " " + scheme + " " + gemm + " token " + t + " row " + r + ": kernel "
                                + out[t * M + r] + " expected " + acc + " (bound " + tol + ")",
                        Math.abs(out[t * M + r] - acc) <= tol);
            }
        }
    }

    private void stage(String kernel, ActQuantScheme scheme, int tk, int tokens, GpuBuffer x, GpuBuffer q8,
                       GpuEvent after) {
        int local = kernels.matvecLocal();
        long records = (long) tokens * (K / tk);
        Object[] args;
        if (scheme.isStatic()) {
            args = new Object[]{tokens, K, tk, K, 0, STATIC_SCALE, x, q8};
        } else {
            args = new Object[]{tokens, K, tk, K, 0, x, q8};
        }
        queue.enqueueKernelNoEvent(kernels.get(kernel), new long[]{roundUp(records, local)}, new long[]{local},
                args, after);
    }

    private static long roundUp(long n, long m) {
        return (n + m - 1) / m * m;
    }
}

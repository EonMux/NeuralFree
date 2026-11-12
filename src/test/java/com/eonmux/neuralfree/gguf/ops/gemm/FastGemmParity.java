package com.eonmux.neuralfree.gguf.ops.gemm;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.quant.CpuBlockDecoders;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Device harness of the fast GEMM parity tests: runs a fast GEMM kernel and the exact
 * {@code <T>_MatMul_Tiled} kernel of the same weight type on one backend over the same inputs and
 * compares them within the rounding the fast form adds (the 8-bit activation codes of the MMQ GEMM,
 * the fp16 tiles of the cooperative-matrix forms, the fp32 plane sums of split-K). Fixtures are the
 * hand-built ggml block layouts of {@link OpenClMmqGemmParityTest}.
 */
final class FastGemmParity {

    private final ComputeBackend backend;
    private final ComputeQueue queue;
    private final LlmKernels kernels;
    private final MmqConfig mmqConfig;

    FastGemmParity(ComputeBackend backend, LlmKernels kernels) {
        this.backend = backend;
        this.queue = backend.computeQueue();
        this.kernels = kernels;
        this.mmqConfig = MmqConfig.forDevice(backend.backendType(), backend.capabilities().getComputeCapability());
    }

    /**
     * The int8 MMQ GEMM of {@code type} against the tiled GEMM: the activations are staged once by
     * {@link GemmKernels#QUANT_MMQ_Q8_1} and the difference is bounded by one activation code step per
     * element ({@code sum_k |W[r][k]| * d_k}, with d_k the 32-element chunk scale, an upper bound for
     * the 16-element chunks too) plus fp32 rounding.
     */
    void mmqMatchesTiled(GGMLType type, int m, int k, int n) throws InterruptedException {
        mmqMatchesTiledWithConfig(type, m, k, n, "device", mmqConfig);
    }

    void mmqMatchesTiledWithConfig(GGMLType type, int m, int k, int n, String label, MmqConfig cfg)
            throws InterruptedException {
        MmqGemmPlan plan = MmqGemmPlan.contiguous(cfg, type, m, k, n);
        byte[] weight = OpenClMmqGemmParityTest.buildWeight(type, m, k);
        float[] x = OpenClMmqGemmParityTest.buildX(n, k);
        float[] w = CpuBlockDecoders.dequantize(weight, type, (long) m * k);
        float[] fast = launchMmq(plan, weight, x);
        float[] tiled = tiled(type, m, k, n, weight, x);
        for (int t = 0; t < n; t++) {
            double[] d = chunkScales(x, t * k, k, 32);
            for (int r = 0; r < m; r++) {
                double bound = 0.0;
                double terms = 0.0;
                for (int i = 0; i < k; i++) {
                    bound += Math.abs((double) w[r * k + i]) * d[i];
                    terms += Math.abs((double) w[r * k + i] * x[t * k + i]);
                }
                double tol = bound + 1e-4 * terms + 1e-3;
                float got = fast[t * m + r];
                float want = tiled[t * m + r];
                assertTrue(label + " " + plan.kernelName() + " tk" + cfg.stageDepth() + " token " + t + " row " + r
                        + ": " + got + " tiled " + want + " (bound " + tol + ")", Math.abs(got - want) <= tol);
            }
        }
    }

    /**
     * The plain MMQ GEMM of {@code type} over activation records that start {@code qOff} units into their buffer
     * against the same GEMM over records at offset 0: the record offset changes only how the records are read
     * (an offset that is not a multiple of 4 units leaves them off 16-byte alignment), so the two agree bit for bit.
     */
    void mmqRecordOffsetMatchesAligned(GGMLType type, int m, int k, int n, int qOff) throws InterruptedException {
        MmqGemmPlan aligned = MmqGemmPlan.contiguous(mmqConfig, type, m, k, n);
        MmqGemmPlan shifted = MmqGemmPlan.of(mmqConfig, type, m, k, n, aligned.qStride(), qOff, m, 0);
        byte[] weight = OpenClMmqGemmParityTest.buildWeight(type, m, k);
        float[] x = OpenClMmqGemmParityTest.buildX(n, k);
        float[] want = launchMmq(aligned, weight, x);
        float[] got = launchMmq(shifted, weight, x);
        for (int i = 0; i < m * n; i++) {
            assertTrue(shifted.kernelName() + " qOff " + qOff + " token " + i / m + " row " + i % m + ": " + got[i]
                    + " at qOff 0: " + want[i], Float.floatToRawIntBits(got[i]) == Float.floatToRawIntBits(want[i]));
        }
    }

    /**
     * The int8 MMQ GEMM over operands whose every step is exact in fp32: Q8_0 blocks of scale 1 holding random codes,
     * and integer activations in [-127, 127] whose every 16-element chunk reaches 127 in magnitude, so the staging
     * scale is 1 and the staged codes are the activations themselves. Each output is then an integer below 2^24 that
     * any correct evaluation stores exactly, whatever its integer-dot instruction, staging order or float
     * contraction: the plain kernel and its split-K twin must equal the CPU's integer sum bit for bit.
     */
    void mmqExactOnIntegerOperands(int m, int k, int n, int splits) throws InterruptedException {
        GGMLType type = GGMLType.Q8_0;
        java.util.Random rnd = new java.util.Random(0x1A7E5L + 31L * m + k + n);
        byte[] weight = new byte[m * (k / 32) * 34];
        int[] codes = new int[m * k];
        for (int r = 0; r < m; r++) {
            for (int b = 0; b < k / 32; b++) {
                int base = (r * (k / 32) + b) * 34;
                weight[base] = 0x00;
                weight[base + 1] = 0x3C;
                for (int e = 0; e < 32; e++) {
                    int c = rnd.nextInt(255) - 127;
                    codes[r * k + b * 32 + e] = c;
                    weight[base + 2 + e] = (byte) c;
                }
            }
        }
        float[] x = new float[n * k];
        for (int t = 0; t < n; t++) {
            for (int i = 0; i < k; i++) {
                x[t * k + i] = i % 16 == rnd.nextInt(16) || i % 16 == 15 ? (rnd.nextBoolean() ? 127 : -127)
                        : rnd.nextInt(255) - 127;
            }
        }
        float[] want = new float[n * m];
        for (int t = 0; t < n; t++) {
            for (int r = 0; r < m; r++) {
                long dot = 0;
                for (int i = 0; i < k; i++) {
                    dot += (long) codes[r * k + i] * (long) x[t * k + i];
                }
                want[t * m + r] = dot;
            }
        }
        MmqGemmPlan base = MmqGemmPlan.contiguous(mmqConfig, type, m, k, n);
        assertIdentical(base.kernelName() + " exact", java.util.Arrays.copyOf(launchMmq(base, weight, x), n * m),
                want);
        SplitKGemmPlan split = SplitKGemmPlan.ofMmq(type, m, k, n, base.qStride(), 0, splits,
                SplitKGemmPlan.sliceLength(k, splits, type.getBlockSize()), mmqConfig, false, 256);
        assertIdentical(split.kernelName() + " exact", java.util.Arrays.copyOf(launchMmqSplit(split, weight, x),
                n * m), want);
    }

    /**
     * The split-K twin of the int8 MMQ GEMM against the plain kernel of the same type, both over the
     * same staged activations: fp32 plane sums only.
     */
    void mmqSplitKMatchesPlain(GGMLType type, int m, int k, int n, int splits) throws InterruptedException {
        MmqGemmPlan base = MmqGemmPlan.contiguous(mmqConfig, type, m, k, n);
        int depth = type.getBlockSize();
        while (splits > 1 && (long) SplitKGemmPlan.sliceLength(k, splits, depth) * (splits - 1) >= k) {
            splits--;
        }
        SplitKGemmPlan split = SplitKGemmPlan.ofMmq(type, m, k, n, base.qStride(), 0, splits,
                SplitKGemmPlan.sliceLength(k, splits, depth), mmqConfig, false, 256);
        byte[] weight = OpenClMmqGemmParityTest.buildWeight(type, m, k);
        float[] x = OpenClMmqGemmParityTest.buildX(n, k);
        float[] plain = launchMmq(base, weight, x);
        float[] splitOut = launchMmqSplit(split, weight, x);
        assertWithin(split.kernelName(), splitOut, plain, m * n, 5e-3);
    }

    /**
     * The split-K twin of the CM GEMM against the plain CM kernel of the same type: fp32 plane sums
     * only (both dequantize into fp16 tiles, so their plain forms agree exactly).
     */
    void cmSplitKMatchesPlain(GGMLType type, int m, int k, int n, int splits) throws InterruptedException {
        CmGemmPlan base = CmGemmPlan.contiguous(type, m, k, n);
        int depth = type.getBlockSize();
        while (splits > 1 && (long) SplitKGemmPlan.sliceLength(k, splits, depth) * (splits - 1) >= k) {
            splits--;
        }
        SplitKGemmPlan split = SplitKGemmPlan.of(SplitKGemmPlan.Family.CM, type, m, k, n, k, 0, splits,
                SplitKGemmPlan.sliceLength(k, splits, depth), CmGemmPlan.TILE_TOKENS, 256);
        byte[] weight = OpenClMmqGemmParityTest.buildWeight(type, m, k);
        float[] x = OpenClMmqGemmParityTest.buildX(n, k);
        float[] plain = launch(base.kernelName(), base.globalWorkSize(), base.localWorkSize(), m, k, n, weight, x);
        float[] splitOut = launchSplit(split, weight, x);
        assertWithin(split.kernelName(), splitOut, plain, m * n, 2e-3);
    }

    /**
     * The fp16-tile CM GEMM of {@code type} against the tiled GEMM: weight values and activations
     * round to fp16 (2^-11 relative each) before the fp32 accumulation.
     */
    void cmMatchesTiled(GGMLType type, int m, int k, int n) throws InterruptedException {
        CmGemmPlan plan = CmGemmPlan.contiguous(type, m, k, n);
        byte[] weight = OpenClMmqGemmParityTest.buildWeight(type, m, k);
        float[] x = OpenClMmqGemmParityTest.buildX(n, k);
        float[] w = CpuBlockDecoders.dequantize(weight, type, (long) m * k);
        float[] fast = launch(plan.kernelName(), plan.globalWorkSize(), plan.localWorkSize(), m, k, n, weight, x);
        float[] tiled = tiled(type, m, k, n, weight, x);
        assertWithinRelative(plan.kernelName(), fast, tiled, w, x, m, k, n, 2.5e-3);
    }

    /**
     * The split-K twin of the CM2 GEMM against the plain CM2 kernel of the same type. Both decode the weight
     * blocks into the same fp16 tiles, so a one-hot activation (one product per output) agrees to the fp32
     * plane sum; over a full reduction the two accumulate in fp16 over different spans (the whole K against
     * one slice each, the slices summed in fp32), so they agree within the fp16-accumulation bound both keep
     * against the tiled GEMM ({@link #cm2MatchesTiled}: 2.5e-3 of the absolute product sum).
     */
    void cm2SplitKMatchesPlain(GGMLType type, int m, int k, int n, int splits) throws InterruptedException {
        Cm2GemmPlan base = Cm2GemmPlan.contiguous(type, m, k, n);
        int depth = Cm2GemmPlan.TILE_DEPTH;
        while (splits > 1 && (long) SplitKGemmPlan.sliceLength(k, splits, depth) * (splits - 1) >= k) {
            splits--;
        }
        SplitKGemmPlan split = SplitKGemmPlan.of(SplitKGemmPlan.Family.CM2, type, m, k, n, k, 0, splits,
                SplitKGemmPlan.sliceLength(k, splits, depth), Cm2GemmPlan.TILE_TOKENS, 256);
        byte[] weight = OpenClMmqGemmParityTest.buildWeight(type, m, k);
        for (int selectedK : new int[]{0, split.kSlice(), 2 * split.kSlice()}) {
            float[] oneHot = new float[n * k];
            for (int t = 0; t < n; t++) {
                oneHot[t * k + selectedK] = 1.0f;
            }
            float[] plainOneHot = launchCm2(base, null, weight, oneHot);
            float[] splitOneHot = launchCm2(base, split, weight, oneHot);
            assertWithin(split.kernelName() + " K=" + selectedK, splitOneHot, plainOneHot, m * n, 2e-3);
        }
        float[] x = OpenClMmqGemmParityTest.buildX(n, k);
        float[] w = CpuBlockDecoders.dequantize(weight, type, (long) m * k);
        float[] plain = launchCm2(base, null, weight, x);
        float[] splitOut = launchCm2(base, split, weight, x);
        assertWithinRelative(split.kernelName(), splitOut, plain, w, x, m, k, n, 2.5e-3);
    }

    /**
     * The split-K launch the dispatch's heuristic makes of a CM2 launch ({@link SplitKGemmPlan#of(Cm2GemmPlan, int,
     * int)}) on a device of {@code computeUnits}, against the plain kernel of the same form: the plan must take
     * {@code splits} slices of whole {@link Cm2GemmPlan#SPLIT_SLICE_ALIGN}-element groups, so slices past the first
     * start at a non-zero 256-multiple and, for the 32-element types, run the unrolled interior groups from there. A
     * one-hot activation at each slice's first and last element and at the reduction's last one selects one decoded
     * weight per output, which both launches store exactly; a full reduction agrees within the fp16-accumulation
     * bound ({@link #cm2SplitKMatchesPlain}). The full reduction's activations are scaled by a power of two so that no
     * output's sum of absolute products reaches 2^14: every running fp16 sum then stays in range and the comparison
     * covers the slice geometry alone. Unscaled, the fixture's 9728-deep sums reach past 65504 in the whole launch's
     * single fp16 chain, which saturates there while the slices' shorter chains do not.
     */
    void cm2HeuristicSplitMatchesPlain(GGMLType type, boolean decodeVector, int m, int k, int n, int computeUnits,
                                       int splits) throws InterruptedException {
        Cm2GemmPlan base = Cm2GemmPlan.of(type, decodeVector, m, k, n, k, 0, m, 0);
        SplitKGemmPlan split = SplitKGemmPlan.of(base, computeUnits, 256);
        String what = (split == null ? base.kernelName() : split.kernelName()) + " M=" + m + " K=" + k + " N=" + n;
        assertTrue(what + " splits on " + computeUnits + " units", split != null);
        assertEquals(what + " slices", splits, split.splits());
        assertEquals(what + " slice alignment", 0, split.kSlice() % Cm2GemmPlan.SPLIT_SLICE_ALIGN);
        byte[] weight = OpenClMmqGemmParityTest.buildWeight(type, m, k);
        for (int s = 0; s < splits; s++) {
            int start = s * split.kSlice();
            int end = Math.min(k, start + split.kSlice());
            for (int selectedK : new int[]{start, end - 1}) {
                float[] oneHot = new float[n * k];
                for (int t = 0; t < n; t++) {
                    oneHot[t * k + selectedK] = 1.0f;
                }
                assertEqualValues(what + " one-hot K=" + selectedK, launchCm2(base, split, weight, oneHot),
                        launchCm2(base, null, weight, oneHot));
            }
        }
        float[] x = OpenClMmqGemmParityTest.buildX(n, k);
        float[] w = CpuBlockDecoders.dequantize(weight, type, (long) m * k);
        double maxTerms = 0.0;
        for (int t = 0; t < n; t++) {
            for (int r = 0; r < m; r++) {
                double terms = 0.0;
                for (int i = 0; i < k; i++) {
                    terms += Math.abs((double) w[r * k + i] * x[t * k + i]);
                }
                maxTerms = Math.max(maxTerms, terms);
            }
        }
        float scale = 1.0f;
        while (maxTerms * scale >= 16384.0) {
            scale *= 0.5f;
        }
        for (int i = 0; i < x.length; i++) {
            x[i] *= scale;
        }
        assertWithinRelative(what, launchCm2(base, split, weight, x), launchCm2(base, null, weight, x), w, x, m, k, n,
                2.5e-3);
    }

    /** Equal values element for element ({@code -0.0} equals {@code 0.0}: a slice sum turns a negative zero
     *  positive). */
    private static void assertEqualValues(String what, float[] actual, float[] expected) {
        assertEquals(what + " length", expected.length, actual.length);
        for (int i = 0; i < expected.length; i++) {
            if (actual[i] != expected[i]) {
                throw new AssertionError(what + " element " + i + ": " + actual[i] + " expected " + expected[i]);
            }
        }
    }

    /**
     * The four-element decode form {@code <T>_MatMul_CM2V} against {@code <T>_MatMul_CM2}: an identity activation over
     * every reduction position (token t selects column t, so each output is one decoded weight) must agree bit for
     * bit, since each vector component takes the scalar callback's formula; over a full reduction both feed the same
     * fp16 tiles to the same products and must agree exactly as well, plain and split-K. Rows {@code m} past a whole
     * tile take the clamped layouts.
     */
    void cm2DecodeVectorMatchesScalar(GGMLType type, int m, int k) throws InterruptedException {
        Cm2GemmPlan scalar = Cm2GemmPlan.contiguous(type, m, k, k);
        Cm2GemmPlan vector = Cm2GemmPlan.of(type, true, m, k, k, k, 0, m, 0);
        byte[] weight = OpenClMmqGemmParityTest.buildWeight(type, m, k);
        float[] identity = new float[k * k];
        for (int t = 0; t < k; t++) {
            identity[t * k + t] = 1.0f;
        }
        assertIdentical(vector.kernelName() + " identity", launchCm2(vector, null, weight, identity),
                launchCm2(scalar, null, weight, identity));
        int n = 40;
        float[] x = OpenClMmqGemmParityTest.buildX(n, k);
        Cm2GemmPlan scalarN = Cm2GemmPlan.contiguous(type, m, k, n);
        Cm2GemmPlan vectorN = Cm2GemmPlan.of(type, true, m, k, n, k, 0, m, 0);
        assertIdentical(vectorN.kernelName(), launchCm2(vectorN, null, weight, x),
                launchCm2(scalarN, null, weight, x));
        int splits = 3;
        int kSlice = SplitKGemmPlan.sliceLength(k, splits, Cm2GemmPlan.TILE_DEPTH);
        SplitKGemmPlan scalarSplit = SplitKGemmPlan.of(SplitKGemmPlan.Family.CM2, type, m, k, n, k, 0, splits, kSlice,
                Cm2GemmPlan.TILE_TOKENS, 256);
        SplitKGemmPlan vectorSplit = SplitKGemmPlan.of(SplitKGemmPlan.Family.CM2V, type, m, k, n, k, 0, splits, kSlice,
                Cm2GemmPlan.TILE_TOKENS, 256);
        assertIdentical(vectorSplit.kernelName(), launchCm2(vectorN, vectorSplit, weight, x),
                launchCm2(scalarN, scalarSplit, weight, x));
    }

    /**
     * The 128-token tile ({@code <T>_MatMul_CM2[V]_N128}) against the 256-token tile of the same form over the same
     * launch, bit for bit: both accumulate every output element over the same reduction stages, and differ only in the
     * grid and the token width of the narrower tail tiles.
     */
    void cm2MediumTileMatchesLarge(GGMLType type, boolean decodeVector, byte[] weight, float[] x, int m, int k, int n)
            throws InterruptedException {
        Cm2GemmPlan large = Cm2GemmPlan.of(type, decodeVector, m, k, n, k, 0, m, 0);
        Cm2GemmPlan medium = Cm2GemmPlan.of(type, decodeVector, Cm2GemmPlan.MEDIUM_TILE_TOKENS, m, k, n, k, 0, m, 0);
        assertIdentical(medium.kernelName() + " M=" + m + " K=" + k + " N=" + n, launchCm2(medium, null, weight, x),
                launchCm2(large, null, weight, x));
    }

    private static void assertIdentical(String what, float[] actual, float[] expected) {
        assertEquals(what + " length", expected.length, actual.length);
        for (int i = 0; i < expected.length; i++) {
            if (Float.floatToRawIntBits(actual[i]) != Float.floatToRawIntBits(expected[i])) {
                throw new AssertionError(what + " element " + i + ": " + actual[i] + " expected " + expected[i]);
            }
        }
    }

    /**
     * Decodes one value per matrix output through an identity-column activation tile. This isolates
     * the CM2 tensor-layout weight callback from a reduction over many K values and catches block
     * addressing errors that can otherwise be obscured by cancellation.
     */
    void cm2OneHotColumnsMatchCpu(GGMLType type) throws InterruptedException {
        int m = 128, k = 256, n = 128;
        Cm2GemmPlan plan = Cm2GemmPlan.contiguous(type, m, k, n);
        byte[] weight = OpenClMmqGemmParityTest.buildWeight(type, m, k);
        float[] w = CpuBlockDecoders.dequantize(weight, type, (long) m * k);
        float[] x = new float[n * k];
        for (int t = 0; t < n; t++) {
            x[t * k] = 1.0f;
        }
        float[] got = launchCm2(plan, null, weight, x);
        float[] expectedK0 = new float[m * n];
        for (int t = 0; t < n; t++) {
            for (int r = 0; r < m; r++) {
                expectedK0[t * m + r] = w[r * k];
            }
        }
        assertWithinRelative(plan.kernelName() + " k0", got, expectedK0, w, x, m, k, n, 2.5e-3);

        java.util.Arrays.fill(x, 0.0f);
        for (int t = 0; t < n; t++) {
            x[t * k + t] = 1.0f;
        }
        got = launchCm2(plan, null, weight, x);
        for (int t = 0; t < n; t++) {
            for (int r = 0; r < m; r++) {
                float expected = CudaMmaGemmParityTest.halfBitsToFloat(
                        CudaMmaGemmParityTest.floatToHalfBits(w[r * k + t]));
                float actual = got[t * m + r];
                assertTrue(plan.kernelName() + " token " + t + " row " + r + ": " + actual + " reference "
                                + expected,
                        Math.abs(actual - expected) <= 1e-3f + 2e-3f * Math.abs(expected));
            }
        }
    }

    /**
     * The workgroup-scope CM2 GEMM of {@code type} against the tiled GEMM: weight values and activations
     * round to fp16 (2^-11 relative each) as in the CM form, and the products accumulate in fp16
     * (llama.cpp's coopmat2 contract), whose rounding grows with the absolute product sum the bound scales.
     */
    void cm2MatchesTiled(GGMLType type, int m, int k, int n) throws InterruptedException {
        Cm2GemmPlan plan = Cm2GemmPlan.contiguous(type, m, k, n);
        byte[] weight = OpenClMmqGemmParityTest.buildWeight(type, m, k);
        float[] x = OpenClMmqGemmParityTest.buildX(n, k);
        float[] w = CpuBlockDecoders.dequantize(weight, type, (long) m * k);
        float[] fast = launchCm2(plan, null, weight, x);
        float[] tiled = tiled(type, m, k, n, weight, x);
        assertWithinRelative(plan.kernelName(), fast, tiled, w, x, m, k, n, 2.5e-3);
    }

    /**
     * The dense F16 CM2 GEMM against the tiled GEMM: the weights and activations are exact in fp16 and both
     * accumulate in fp32, so they agree within the MMA GEMM's fp32 bound ({@code 1e-5} of the absolute product
     * sum, {@link #mmaMatchesTiled}).
     */
    void cm2DenseMatchesTiled(byte[] weight, float[] wValues, float[] x, int m, int k, int n)
            throws InterruptedException {
        Cm2GemmPlan plan = Cm2GemmPlan.contiguous(GGMLType.F16, m, k, n);
        float[] fast = launchCm2(plan, null, weight, x);
        float[] tiled = tiled(GGMLType.F16, m, k, n, weight, x);
        assertWithinRelative(plan.kernelName(), fast, tiled, wValues, x, m, k, n, 1e-5);
    }

    /**
     * The split-K pair of the dense F16 CM2 GEMM against its plain kernel: fp32 slices summed in fp32 against one
     * fp32 sum, within the fp32 plane-sum bound of the MMA pair ({@link #splitKMatchesPlain}).
     */
    void cm2DenseSplitKMatchesPlain(byte[] weight, float[] wValues, float[] x, int m, int k, int n, int splits)
            throws InterruptedException {
        Cm2GemmPlan base = Cm2GemmPlan.contiguous(GGMLType.F16, m, k, n);
        float[] plain = launchCm2(base, null, weight, x);
        float[] splitOut = launchCm2(base, denseSplit(m, k, n, splits), weight, x);
        assertWithinRelative(GemmKernels.matmulCm2KSplit(GGMLType.F16), splitOut, plain, wValues, x, m, k, n, 1e-5);
    }

    /**
     * The dense F16 CM2 GEMM, plain and split into {@code splits} slices, and the F16 MMA GEMM it replaces, each
     * against the exact double-precision product, within the MMA GEMM's fp32 bound ({@code 1e-5} of the absolute
     * product sum). The operands are exact in fp16, so the bound measures the accumulation alone: an fp16
     * accumulator misses it on every row, and a sum past the fp16 range (the fixtures of
     * {@link #outlierActivations}) must store unclipped.
     */
    void cm2DenseMatchesExactProducts(byte[] weight, float[] wValues, float[] x, int m, int k, int n, int splits)
            throws InterruptedException {
        double[] exact = exactProducts(wValues, x, m, k, n);
        Cm2GemmPlan plan = Cm2GemmPlan.contiguous(GGMLType.F16, m, k, n);
        MmaGemmPlan mma = MmaGemmPlan.contiguous(GGMLType.F16, m, k, n);
        assertWithinExact(mma.kernelName(), launch(mma.kernelName(), mma.globalWorkSize(), mma.localWorkSize(), m, k,
                n, weight, x), exact, wValues, x, m, k, n);
        assertWithinExact(plan.kernelName(), launchCm2(plan, null, weight, x), exact, wValues, x, m, k, n);
        assertWithinExact(GemmKernels.matmulCm2KSplit(GGMLType.F16),
                launchCm2(plan, denseSplit(m, k, n, splits), weight, x), exact, wValues, x, m, k, n);
    }

    private static SplitKGemmPlan denseSplit(int m, int k, int n, int splits) {
        return SplitKGemmPlan.of(SplitKGemmPlan.Family.CM2, GGMLType.F16, m, k, n, k, 0, splits,
                SplitKGemmPlan.sliceLength(k, splits, Cm2GemmPlan.TILE_DEPTH), Cm2GemmPlan.TILE_TOKENS, 256);
    }

    private static double[] exactProducts(float[] w, float[] x, int m, int k, int n) {
        double[] y = new double[n * m];
        for (int t = 0; t < n; t++) {
            for (int r = 0; r < m; r++) {
                double acc = 0.0;
                for (int i = 0; i < k; i++) {
                    acc += (double) w[r * k + i] * x[t * k + i];
                }
                y[t * m + r] = acc;
            }
        }
        return y;
    }

    private static void assertWithinExact(String kernel, float[] got, double[] exact, float[] w, float[] x, int m,
                                          int k, int n) {
        for (int t = 0; t < n; t++) {
            for (int r = 0; r < m; r++) {
                double terms = 0.0;
                for (int i = 0; i < k; i++) {
                    terms += Math.abs((double) w[r * k + i] * x[t * k + i]);
                }
                double tol = 1e-5 * terms + 1e-3;
                assertTrue(kernel + " token " + t + " row " + r + ": " + got[t * m + r] + " exact " + exact[t * m + r]
                        + " (|terms| " + terms + ")", Math.abs(got[t * m + r] - exact[t * m + r]) <= tol);
            }
        }
    }

    /**
     * fp16-exact activations of {@code n} tokens of {@code k} channels with the massive-activation pattern of an F16
     * checkpoint's feed-forward input: {@link #denseValues} everywhere except {@link #OUTLIER_CHANNELS}, which carry
     * positive values of 1024 to 2032 that grow with the token. Through {@link #outlierWeights}' row
     * {@link #OUTLIER_ROW} they sum to 8.6e4 .. 1.7e5, past the fp16 range: the magnitude of pplx-embed-v1 0.6B's
     * layer-19 feed-forward output, whose channel 35 reaches 1.5e5 and 1.7e5 through the fp32-accumulating MMA GEMM.
     */
    static float[] outlierActivations(int n, int k, int seed) {
        float[] x = denseValues(n * k, seed);
        for (int t = 0; t < n; t++) {
            for (int c : OUTLIER_CHANNELS) {
                x[t * k + c] = 1024.0f + 16.0f * (t % 64);
            }
        }
        return x;
    }

    /**
     * fp16-exact weights for {@link #outlierActivations}: {@link #denseValues} scaled by 1/32 (2^-6 .. 2^-4), with
     * {@link #OUTLIER_ROW} weighting the outlier channels by 18 to 24.
     */
    static float[] outlierWeights(int m, int k, int seed) {
        float[] w = denseValues(m * k, seed);
        for (int i = 0; i < w.length; i++) {
            w[i] = w[i] / 32.0f;
        }
        float[] gains = {20.0f, 24.0f, 18.0f, 22.0f};
        for (int j = 0; j < OUTLIER_CHANNELS.length; j++) {
            w[OUTLIER_ROW * k + OUTLIER_CHANNELS[j]] = gains[j];
        }
        return w;
    }

    /** The outlier channels of {@link #outlierActivations} (below 3072, one of them in the last 64-deep stage). */
    static final int[] OUTLIER_CHANNELS = {7, 911, 1500, 3050};

    /** The output row of {@link #outlierWeights} whose sums pass the fp16 range. */
    static final int OUTLIER_ROW = 35;

    /**
     * The MMA GEMM of a dense 16-bit type against the tiled GEMM of the same type: the activations
     * round to the tile format (fp16, or bf16 on the bf16 instruction) before exact products.
     */
    void mmaMatchesTiled(GGMLType type, byte[] weight, float[] wValues, float[] x, int m, int k, int n,
                         double relative) throws InterruptedException {
        MmaGemmPlan plan = MmaGemmPlan.contiguous(type, m, k, n);
        float[] fast = launch(plan.kernelName(), plan.globalWorkSize(), plan.localWorkSize(), m, k, n, weight, x);
        float[] tiled = tiled(type, m, k, n, weight, x);
        assertWithinRelative(plan.kernelName(), fast, tiled, wValues, x, m, k, n, relative);
    }

    /** The split-K pair against the plain MMA kernel: fp32 plane sums only. */
    void splitKMatchesPlain(GGMLType type, byte[] weight, float[] wValues, float[] x, int m, int k, int n,
                            int splits) throws InterruptedException {
        SplitKGemmPlan split = SplitKGemmPlan.of(type, m, k, n, k, 0, splits, SplitKGemmPlan.sliceLength(k, splits),
                256);
        MmaGemmPlan plain = MmaGemmPlan.contiguous(type, m, k, n);
        float[] plainOut = launch(plain.kernelName(), plain.globalWorkSize(), plain.localWorkSize(), m, k, n,
                weight, x);
        float[] splitOut = launchSplit(split, weight, x);
        assertWithinRelative(split.kernelName(), splitOut, plainOut, wValues, x, m, k, n, 1e-5);
    }

    private static void assertWithinRelative(String kernel, float[] got, float[] want, float[] w, float[] x,
                                             int m, int k, int n, double relative) {
        for (int t = 0; t < n; t++) {
            for (int r = 0; r < m; r++) {
                double terms = 0.0;
                for (int i = 0; i < k; i++) {
                    terms += Math.abs((double) w[r * k + i] * x[t * k + i]);
                }
                double tol = relative * terms + 1e-3;
                assertTrue(kernel + " token " + t + " row " + r + ": " + got[t * m + r] + " reference "
                                + want[t * m + r] + " (|terms| " + terms + ")",
                        Math.abs(got[t * m + r] - want[t * m + r]) <= tol);
            }
        }
    }

    /** Elementwise agreement of two flat [n] outputs within a tolerance scaled by the magnitude. */
    private static void assertWithin(String kernel, float[] got, float[] want, int count, double tol) {
        for (int i = 0; i < count; i++) {
            assertTrue(kernel + " element " + i + ": " + got[i] + " reference " + want[i],
                    Math.abs(got[i] - want[i]) <= tol * (1.0 + Math.abs(want[i])));
        }
    }

    /** Per-element scale d = maxabs / 127 of the {@code chunk}-element activation chunk holding it. */
    private static double[] chunkScales(float[] x, int base, int k, int chunk) {
        double[] d = new double[k];
        for (int c = 0; c < k; c += chunk) {
            float maxabs = 0.0f;
            for (int i = c; i < Math.min(k, c + chunk); i++) {
                maxabs = Math.max(maxabs, Math.abs(x[base + i]));
            }
            for (int i = c; i < Math.min(k, c + chunk); i++) {
                d[i] = maxabs / 127.0;
            }
        }
        return d;
    }

    float[] tiled(GGMLType type, int m, int k, int n, byte[] weight, float[] x) throws InterruptedException {
        TiledGemmPlan plan = TiledGemmPlan.contiguous(type, m, k, n);
        return launch(plan.kernelName(), plan.globalWorkSize(), plan.localWorkSize(), m, k, n, weight, x);
    }

    /** One 10-argument GEMM launch over contiguous views ({@code x_stride = K}, {@code y_stride = M}). */
    float[] launch(String kernel, long[] global, long[] local, int m, int k, int n, byte[] weight, float[] x)
            throws InterruptedException {
        GpuBuffer wbuf = backend.allocateBuffer(weight.length, true);
        GpuBuffer xbuf = backend.allocateBuffer((long) x.length * Float.BYTES, true);
        GpuBuffer ybuf = backend.allocateBuffer((long) n * m * Float.BYTES);
        try {
            try (GpuEvent wv = queue.enqueueWriteBuffer(wbuf, 0, weight.length, weight);
                 GpuEvent xv = queue.enqueueWriteBuffer(xbuf, 0, (long) x.length * Float.BYTES, x, wv)) {
                Object[] args = {m, k, n, k, 0, m, 0, wbuf, xbuf, ybuf};
                queue.enqueueKernelNoEvent(kernels.get(kernel), global, local, args, xv);
                float[] out = new float[n * m];
                try (GpuEvent read = queue.enqueueReadBuffer(ybuf, 0, (long) out.length * Float.BYTES, out)) {
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

    private float[] launchSplit(SplitKGemmPlan plan, byte[] weight, float[] x) throws InterruptedException {
        GpuBuffer wbuf = backend.allocateBuffer(weight.length, true);
        GpuBuffer xbuf = backend.allocateBuffer((long) x.length * Float.BYTES, true);
        GpuBuffer pbuf = backend.allocateBuffer(plan.partialElements() * Float.BYTES);
        GpuBuffer ybuf = backend.allocateBuffer(plan.yElementsRequired(plan.m(), 0) * Float.BYTES);
        try {
            try (GpuEvent wv = queue.enqueueWriteBuffer(wbuf, 0, weight.length, weight);
                 GpuEvent xv = queue.enqueueWriteBuffer(xbuf, 0, (long) x.length * Float.BYTES, x, wv)) {
                queue.enqueueKernelNoEvent(kernels.get(plan.kernelName()), plan.partialGlobalWorkSize(),
                        plan.partialLocalWorkSize(), plan.partialArgs(wbuf, xbuf, pbuf), xv);
                queue.enqueueKernelNoEvent(kernels.get(GemmKernels.SPLIT_K_REDUCE), plan.reduceGlobalWorkSize(),
                        plan.reduceLocalWorkSize(), plan.reduceArgs(pbuf, ybuf, plan.m(), 0));
                float[] out = new float[(int) plan.yElementsRequired(plan.m(), 0)];
                try (GpuEvent read = queue.enqueueReadBuffer(ybuf, 0, (long) out.length * Float.BYTES, out)) {
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

    /**
     * One CM2 GEMM launch over contiguous views, the plain kernel of {@code plan} or, with {@code split}, the
     * split-K pair: {@link Cm2GemmPlan#STAGE_KERNEL} stages the fp16 rows once (the launch contract the dispatch
     * uses), then the GEMM reads them.
     */
    float[] launchCm2(Cm2GemmPlan plan, SplitKGemmPlan split, byte[] weight, float[] x) throws InterruptedException {
        GpuBuffer wbuf = backend.allocateBuffer(weight.length, true);
        GpuBuffer xbuf = backend.allocateBuffer((long) x.length * Float.BYTES, true);
        GpuBuffer hbuf = backend.allocateBuffer(plan.stagedHalves() * Short.BYTES);
        GpuBuffer pbuf = split == null ? null : backend.allocateBuffer(split.partialElements() * Float.BYTES);
        GpuBuffer ybuf = backend.allocateBuffer(plan.yElementsRequired() * Float.BYTES);
        try {
            try (GpuEvent wv = queue.enqueueWriteBuffer(wbuf, 0, weight.length, weight);
                 GpuEvent xv = queue.enqueueWriteBuffer(xbuf, 0, (long) x.length * Float.BYTES, x, wv)) {
                queue.enqueueKernelNoEvent(kernels.get(Cm2GemmPlan.STAGE_KERNEL), plan.stageGlobalWorkSize(),
                        Cm2GemmPlan.stageLocalWorkSize(), plan.stageArgs(xbuf, hbuf), xv);
                if (split == null) {
                    queue.enqueueKernelNoEvent(kernels.get(plan.kernelName()), plan.globalWorkSize(),
                            plan.localWorkSize(), plan.args(wbuf, hbuf, ybuf));
                } else {
                    queue.enqueueKernelNoEvent(kernels.get(split.kernelName()), split.partialGlobalWorkSize(),
                            split.partialLocalWorkSize(), split.partialArgs(wbuf, hbuf, pbuf));
                    queue.enqueueKernelNoEvent(kernels.get(GemmKernels.SPLIT_K_REDUCE), split.reduceGlobalWorkSize(),
                            split.reduceLocalWorkSize(), split.reduceArgs(pbuf, ybuf, plan.m(), 0));
                }
                float[] out = new float[(int) plan.yElementsRequired()];
                try (GpuEvent read = queue.enqueueReadBuffer(ybuf, 0, (long) out.length * Float.BYTES, out)) {
                    read.awaitCompletion();
                }
                return out;
            }
        } finally {
            wbuf.close();
            xbuf.close();
            hbuf.close();
            if (pbuf != null) {
                pbuf.close();
            }
            ybuf.close();
        }
    }

    /**
     * Stages {@code x} and runs the MMQ GEMM of {@code plan}. The records are staged contiguously (the plan's qStride
     * must be the row's record count) and, for a plan with a record offset, copied to that unit offset of the
     * GEMM's activation buffer, so the kernel reads them through a view that starts {@code qOff} units in.
     */
    float[] launchMmq(MmqGemmPlan plan, byte[] weight, float[] x) throws InterruptedException {
        if (plan.qStride() != plan.config().rowRecords(plan.k())) {
            throw new IllegalArgumentException("launchMmq stages contiguous records: qStride " + plan.qStride()
                    + " is not the row's " + plan.config().rowRecords(plan.k()) + " records");
        }
        long stagedUnits = (long) plan.n() * plan.config().rowRecords(plan.k()) * plan.config().recordUnits();
        GpuBuffer wbuf = backend.allocateBuffer(weight.length, true);
        GpuBuffer xbuf = backend.allocateBuffer((long) x.length * Float.BYTES, true);
        GpuBuffer qbuf = backend.allocateBuffer(plan.q8ElementsRequired() * Integer.BYTES);
        GpuBuffer sbuf = plan.qOff() == 0 ? qbuf : backend.allocateBuffer(stagedUnits * Integer.BYTES);
        GpuBuffer ybuf = backend.allocateBuffer(plan.yElementsRequired() * Float.BYTES);
        try {
            try (GpuEvent wv = queue.enqueueWriteBuffer(wbuf, 0, weight.length, weight);
                 GpuEvent xv = queue.enqueueWriteBuffer(xbuf, 0, (long) x.length * Float.BYTES, x, wv)) {
                int tk = plan.config().stageDepth();
                long records = (long) plan.n() * plan.config().rowRecords(plan.k());
                Object[] quantArgs = {plan.n(), plan.k(), tk, plan.k(), 0, xbuf, sbuf};
                queue.enqueueKernelNoEvent(kernels.get(GemmKernels.QUANT_MMQ_Q8_1),
                        new long[]{roundUp(records, 64)}, new long[]{64}, quantArgs, xv);
                if (sbuf != qbuf) {
                    queue.enqueueCopyBuffer(sbuf, 0, qbuf, (long) plan.qOff() * Integer.BYTES,
                            stagedUnits * Integer.BYTES);
                }
                queue.enqueueKernelNoEvent(kernels.get(plan.kernelName()), plan.globalWorkSize(),
                        plan.localWorkSize(), plan.args(wbuf, qbuf, ybuf));
                float[] out = new float[(int) plan.yElementsRequired()];
                try (GpuEvent read = queue.enqueueReadBuffer(ybuf, 0, (long) out.length * Float.BYTES, out)) {
                    read.awaitCompletion();
                }
                return out;
            }
        } finally {
            wbuf.close();
            xbuf.close();
            qbuf.close();
            if (sbuf != qbuf) {
                sbuf.close();
            }
            ybuf.close();
        }
    }

    /** The split-K pair of one staged MMQ launch: the same staging pass feeds the slice planes. */
    float[] launchMmqSplit(SplitKGemmPlan plan, byte[] weight, float[] x) throws InterruptedException {
        MmqConfig cfg = plan.stagedConfig() != null ? plan.stagedConfig() : mmqConfig;
        MmqGemmPlan plain = MmqGemmPlan.contiguous(cfg, plan.type(), plan.m(), plan.k(), plan.n());
        GpuBuffer wbuf = backend.allocateBuffer(weight.length, true);
        GpuBuffer xbuf = backend.allocateBuffer((long) x.length * Float.BYTES, true);
        GpuBuffer qbuf = backend.allocateBuffer(plain.q8ElementsRequired() * Integer.BYTES);
        GpuBuffer pbuf = backend.allocateBuffer(plan.partialElements() * Float.BYTES);
        GpuBuffer ybuf = backend.allocateBuffer(plan.yElementsRequired(plan.m(), 0) * Float.BYTES);
        try {
            try (GpuEvent wv = queue.enqueueWriteBuffer(wbuf, 0, weight.length, weight);
                 GpuEvent xv = queue.enqueueWriteBuffer(xbuf, 0, (long) x.length * Float.BYTES, x, wv)) {
                int tk = plain.config().stageDepth();
                long records = (long) plain.n() * plain.config().rowRecords(plain.k());
                Object[] quantArgs = {plain.n(), plain.k(), tk, plain.k(), 0, xbuf, qbuf};
                queue.enqueueKernelNoEvent(kernels.get(GemmKernels.QUANT_MMQ_Q8_1),
                        new long[]{roundUp(records, 64)}, new long[]{64}, quantArgs, xv);
                queue.enqueueKernelNoEvent(kernels.get(plan.kernelName()), plan.partialGlobalWorkSize(),
                        plan.partialLocalWorkSize(), plan.partialArgs(wbuf, qbuf, pbuf));
                queue.enqueueKernelNoEvent(kernels.get(GemmKernels.SPLIT_K_REDUCE), plan.reduceGlobalWorkSize(),
                        plan.reduceLocalWorkSize(), plan.reduceArgs(pbuf, ybuf, plan.m(), 0));
                float[] out = new float[(int) plan.yElementsRequired(plan.m(), 0)];
                try (GpuEvent read = queue.enqueueReadBuffer(ybuf, 0, (long) out.length * Float.BYTES, out)) {
                    read.awaitCompletion();
                }
                return out;
            }
        } finally {
            wbuf.close();
            xbuf.close();
            qbuf.close();
            pbuf.close();
            ybuf.close();
        }
    }

    private static long roundUp(long n, long m) {
        return (n + m - 1) / m * m;
    }

    // ------------------------------------------------------------------ dense 16-bit fixtures

    /** Deterministic dense weight values with at most 7 explicit mantissa bits (exact in fp16 and bf16). */
    static float[] denseValues(int count, int seed) {
        int state = seed | 1;
        float[] v = new float[count];
        for (int i = 0; i < count; i++) {
            state ^= state << 13;
            state ^= state >>> 17;
            state ^= state << 5;
            int bits = state >>> 8;
            int mant = bits & 0x7F;
            int exp = (bits >>> 7) % 3 - 1;          /* 2^-1 .. 2^1 */
            float sign = ((bits >>> 10) & 1) == 0 ? 1.0f : -1.0f;
            v[i] = sign * (float) Math.scalb(1.0 + mant / 128.0, exp);
        }
        return v;
    }

    /** Little-endian fp16 bits of values exact in fp16. */
    static byte[] f16Bytes(float[] v) {
        ByteBuffer b = ByteBuffer.allocate(v.length * 2).order(ByteOrder.LITTLE_ENDIAN);
        for (float f : v) {
            b.putShort(CudaMmaGemmParityTest.floatToHalfBits(f));
        }
        return b.array();
    }

    /** Little-endian bf16 bits of values exact in bf16. */
    static byte[] bf16Bytes(float[] v) {
        ByteBuffer b = ByteBuffer.allocate(v.length * 2).order(ByteOrder.LITTLE_ENDIAN);
        for (float f : v) {
            b.putShort(CudaMmaGemmParityTest.floatToBf16Bits(f));
        }
        return b.array();
    }
}

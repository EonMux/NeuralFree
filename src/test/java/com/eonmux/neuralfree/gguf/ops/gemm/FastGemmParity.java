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
     * One int8 MMQ GEMM launch over contiguous views: {@code LLM_Quant_MMQ_Q8_1} stages the input once
     * (the launch contract the dispatch uses), then the GEMM reads the staged records.
     */
    float[] launchMmq(MmqGemmPlan plan, byte[] weight, float[] x) throws InterruptedException {
        GpuBuffer wbuf = backend.allocateBuffer(weight.length, true);
        GpuBuffer xbuf = backend.allocateBuffer((long) x.length * Float.BYTES, true);
        GpuBuffer qbuf = backend.allocateBuffer(plan.q8ElementsRequired() * Integer.BYTES);
        GpuBuffer ybuf = backend.allocateBuffer(plan.yElementsRequired() * Float.BYTES);
        try {
            try (GpuEvent wv = queue.enqueueWriteBuffer(wbuf, 0, weight.length, weight);
                 GpuEvent xv = queue.enqueueWriteBuffer(xbuf, 0, (long) x.length * Float.BYTES, x, wv)) {
                int tk = plan.config().stageDepth();
                long records = (long) plan.n() * plan.config().rowRecords(plan.k());
                Object[] quantArgs = {plan.n(), plan.k(), tk, plan.k(), 0, xbuf, qbuf};
                queue.enqueueKernelNoEvent(kernels.get(GemmKernels.QUANT_MMQ_Q8_1),
                        new long[]{roundUp(records, 64)}, new long[]{64}, quantArgs, xv);
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

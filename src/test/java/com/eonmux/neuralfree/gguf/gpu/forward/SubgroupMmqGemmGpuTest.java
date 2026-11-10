package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend.BackendType;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.ops.gemm.MmqConfig;
import com.eonmux.neuralfree.gguf.ops.gemm.GemmKernels;
import com.eonmux.neuralfree.gguf.ops.gemm.MmqGemmPlan;
import com.eonmux.neuralfree.gguf.ops.gemm.SplitKGemmPlan;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * The subgroup-row MMQ GEMM ({@code {T}_MatMulSg_MMQ}, {@link SubgroupMmqGemm}) of every weight type in
 * {@link SubgroupMmqGemm#TYPES}, on every device whose kernels include it ({@link SubgroupFormBackends}): the OpenCL
 * kernel of kernels/llm/matmul_mmq_sg.cl and its Vulkan port shaders/llm/matmul_mmq_sg.comp (the Intel GPUs; other
 * devices skip). Each case hand-builds weight rows of the type (Q8_0: codes over the full -128..127 range and f16
 * scales; Q4_0: every nibble, f16 scales; Q4_K: random 6-bit scales and minimums in the packed 12-byte table, f16 d and
 * dmin, every nibble; Q5_K: the same with every fifth bit; Q6_K: every 6-bit code, int8 sub-block scales over
 * -128..127, f16 d, and 210-byte blocks that leave rows and odd blocks 2 bytes off a word) and staged 16-deep Q8_1
 * records (codes, scale, code sum, zero point 0) at a record offset and row stride, runs the kernel at its launch
 * geometry and requires:
 * <ul>
 *   <li>every output within the fp32 summation bound of the exact reference computed in double from the same
 *       bytes (the -8 code offset of Q4_0, the K-quant scales decoded as ggml does: get_scale_min_k4 for Q4_K and
 *       Q5_K, the int8 scales and the -32 code offset for Q6_K), and within twice that bound of the portable
 *       {@code {T}_MatMul_MMQ} over the same records;</li>
 *   <li>output cells outside the {@code [y_off, y_off + (N-1) y_stride + M)} rows and the gaps of a wide
 *       {@code y_stride} untouched;</li>
 *   <li>bit-identical results across two launches, and a token's outputs bit-identical whether it runs alone or in
 *       a wide launch (the tile position does not change a row's summation order).</li>
 * </ul>
 * The shapes cover partial row tiles, one and several stages and super-blocks (Q8_0 and Q4_0 K 64, 192, 1024; the
 * K-quants K 256, 512, 768, 2560), and token counts on both sides of the 32-token tile (1, 5, 31, 32, 33, 77). The
 * split-K twin {@code {T}_MatMulSg_MMQ_KSplit} (OpenCL) must hold each slice's exact product in its plane.
 *
 * <p>With {@code -Dneuralfree.test.sgmmq.measure=true} the timing case runs the portable and the subgroup kernel on
 * the 512-token projection shapes of the 0.6B and 4B Qwen3 models and logs both rates.</p>
 */
@RunWith(Parameterized.class)
public class SubgroupMmqGemmGpuTest {

    private static final String MEASURE_PROPERTY = "neuralfree.test.sgmmq.measure";
    private static final String SPLIT_MEASURE_PROPERTY = "neuralfree.test.sgmmq.splitk.measure";
    private static final int RC = 7;
    private static final int TK = 16;
    private static final float SENTINEL = -7777.0f;
    /** Unit roundoff of fp32. */
    private static final double FP32_U = 0x1.0p-24;

    private final BackendType api;
    private final GGMLType type;
    private SubgroupFormBackends devices;
    private List<ComputeBackend> backends;

    public SubgroupMmqGemmGpuTest(BackendType api, GGMLType type) {
        this.api = api;
        this.type = type;
    }

    @Parameterized.Parameters(name = "{0} {1}")
    public static Collection<Object[]> types() {
        List<Object[]> types = new ArrayList<>();
        for (BackendType api : new BackendType[]{BackendType.OPENCL, BackendType.VULKAN}) {
            for (GGMLType t : SubgroupMmqGemm.TYPES) {
                types.add(new Object[]{api, t});
            }
        }
        return types;
    }

    @Before
    public void open() {
        devices = SubgroupFormBackends.open(api);
        backends = devices.backends();
    }

    @After
    public void close() {
        if (devices != null) {
            devices.close();
        }
    }

    @Test
    public void matchesTheExactReferenceAndThePortableKernel() {
        int served = 0;
        for (ComputeBackend backend : backends) {
            LlmKernels kernels = kernelsServing(backend);
            if (kernels == null) {
                continue;
            }
            served++;
            String dev = backend.capabilities().getDeviceName();
            for (int[] s : shapes()) {
                run(backend, kernels, dev, s[0], s[1], s[2], s[1] / TK, 0, s[0], 0);
            }
            // record offset and wide record stride, output offset and wide output stride
            int k = pairStaged() ? 192 : 512;
            run(backend, kernels, dev, 70, k, 33, k / TK + 3, 5 * RC, 70 + 9, 11);
            run(backend, kernels, dev, 77, k, 40, k / TK + 1, 3 * RC, 77 + 2, 5);
        }
        Assume.assumeTrue("no " + api + " device builds " + SubgroupMmqGemm.kernel(type), served > 0);
    }

    /** Whether a stage of the type is a 32-element block pair (Q8_0, Q4_0) rather than a super-block quarter. */
    private boolean pairStaged() {
        return type.getBlockSize() == 32;
    }

    /** {M, K, N} of the reference cases: ragged rows and tokens around the tiles, one to many stages. */
    private int[][] shapes() {
        if (pairStaged()) {
            return new int[][]{{70, 64, 1}, {70, 64, 5}, {200, 192, 31}, {200, 192, 32}, {200, 192, 33},
                    {130, 1024, 77}, {128, 256, 64}};
        }
        return new int[][]{{70, 256, 1}, {77, 256, 5}, {200, 512, 31}, {200, 512, 32}, {200, 512, 33},
                {129, 768, 40}, {130, 2560, 77}, {128, 256, 64}};
    }

    @Test
    public void aTokenSumsAloneAsInAWideLaunchAndRunsRepeatBitForBit() {
        int served = 0;
        for (ComputeBackend backend : backends) {
            LlmKernels kernels = kernelsServing(backend);
            if (kernels == null) {
                continue;
            }
            served++;
            int m = 200;
            int k = type == GGMLType.Q8_0 ? 1024 : 2560;
            int n = 77;
            Random rnd = new Random(5);
            byte[] weight = weight(type, rnd, m, k);
            int[] records = records(rnd, n, k, k / TK, 0);
            float[] wide = launchSg(backend, kernels, weight, records, m, k, n, k / TK, 0, m, 0);
            float[] again = launchSg(backend, kernels, weight, records, m, k, n, k / TK, 0, m, 0);
            for (int i = 0; i < m * n; i++) {
                assertEquals(type + " repeat " + i, Float.floatToRawIntBits(wide[i]),
                        Float.floatToRawIntBits(again[i]));
            }
            // token 40 alone: its records at record offset 40 rows in, one-token launch
            float[] alone = launchSg(backend, kernels, weight, records, m, k, 1, k / TK, 40 * (k / TK) * RC, m, 0);
            for (int r = 0; r < m; r++) {
                assertEquals(type + " token 40 row " + r, Float.floatToRawIntBits(wide[40 * m + r]),
                        Float.floatToRawIntBits(alone[r]));
            }
        }
        Assume.assumeTrue("no " + api + " device builds " + SubgroupMmqGemm.kernel(type), served > 0);
    }

    @Test
    public void measureAgainstThePortableKernel() {
        Assume.assumeTrue("opt-in: -D" + MEASURE_PROPERTY + "=true", Boolean.getBoolean(MEASURE_PROPERTY));
        int[][] shapes = {{1024, 1024}, {2048, 1024}, {3072, 1024}, {1024, 3072}, {4096, 2560}, {2560, 9728},
                {19456, 2560}};
        int[] tokens = {16, 32, 64, 128, 512};
        for (ComputeBackend backend : backends) {
            LlmKernels kernels = kernelsServing(backend);
            if (kernels == null) {
                continue;
            }
            for (int[] s : shapes) {
                for (int n : tokens) {
                    measure(backend, kernels, s[0], s[1], n);
                }
            }
        }
    }

    /**
     * The split-K twin {@code {T}_MatMulSg_MMQ_KSplit}: every slice plane holds the exact product over its own stages
     * (slices of 1, 3 and 5 stages, so K-quant slices start at every quarter of a super-block, and a last slice
     * shorter than the others), within the bound of {@link #run}, over a ragged row tile and token tile.
     */
    @Test
    public void kSplitPlanesHoldTheirSlicesProducts() {
        int served = 0;
        String name = type.name() + GemmKernels.MATMUL_MMQ_SG_KSPLIT_SUFFIX;
        for (ComputeBackend backend : backends) {
            LlmKernels kernels = kernelsServing(backend);
            if (kernels == null || !kernels.has(name)) {
                continue;
            }
            served++;
            String dev = backend.capabilities().getDeviceName();
            int k = pairStaged() ? 1024 : 2560;
            for (int sliceStages : new int[]{1, 3, 5}) {
                int kSlice = sliceStages * SplitKGemmPlan.SUBGROUP_MMQ_SLICE_DEPTH;
                int splits = (k + kSlice - 1) / kSlice;
                runSplit(backend, kernels, dev, name, 200, k, 37, k / TK + 2, 3 * RC, kSlice, splits);
            }
        }
        Assume.assumeTrue("no " + api + " device builds " + name, served > 0);
    }

    private void runSplit(ComputeBackend backend, LlmKernels kernels, String dev, String name, int m, int k, int n,
                          int qStride, int qOff, int kSlice, int splits) {
        String what = dev + " " + name + " M=" + m + " K=" + k + " N=" + n + " kSlice=" + kSlice + " splits=" + splits;
        Random rnd = new Random(13L * m + k + n + kSlice + type.ordinal());
        byte[] weight = weight(type, rnd, m, k);
        int[] records = records(rnd, n, k, qStride, qOff);
        int group = SubgroupMmqGemm.workGroup(kernels, type);
        long rowGroups = SubgroupMmqGemm.globalRows(m, group) / group;
        ComputeQueue queue = backend.computeQueue();
        int planes = splits * m * n;
        GpuBuffer w = backend.allocateBuffer(weight.length, true);
        GpuBuffer q = backend.allocateBuffer((long) records.length * 4, true);
        GpuBuffer p = backend.allocateBuffer((long) planes * 4, true);
        float[] out = new float[planes];
        try {
            float[] init = new float[planes];
            Arrays.fill(init, SENTINEL);
            close(queue.enqueueWriteBuffer(w, 0, weight.length, weight));
            close(queue.enqueueWriteBuffer(q, 0, (long) records.length * 4, bytes(records)));
            close(queue.enqueueWriteBuffer(p, 0, (long) planes * 4, init));
            close(queue.enqueueKernel(kernels.get(name),
                    new long[]{rowGroups * group * splits, SubgroupMmqGemm.tokenGroups(n)}, new long[]{group, 1},
                    new Object[]{m, k, n, qStride, qOff, kSlice, w, q, p}));
            close(queue.enqueueReadBuffer(p, 0, (long) planes * 4, out));
            queue.finish();
        } finally {
            w.close();
            q.close();
            p.close();
        }
        double bound = (2.0 * k / TK + 8) * FP32_U;
        for (int s = 0; s < splits; s++) {
            int c0 = s * kSlice / TK;
            int c1 = Math.min((s + 1) * kSlice, k) / TK;
            for (int t = 0; t < n; t++) {
                for (int r = 0; r < m; r++) {
                    float got = out[(s * n + t) * m + r];
                    double[] ref = reference(type, weight, records, k, qStride, qOff, t, r, c0, c1);
                    assertTrue(what + " slice " + s + " token " + t + " row " + r + ": kernel " + got
                            + " expected " + ref[0], Math.abs(got - ref[0]) <= bound * ref[1] + 1e-6);
                }
            }
        }
    }

    /**
     * Opt-in ({@code -Dneuralfree.test.sgmmq.splitk.measure=true}): at the token counts that split on the device
     * ({@link SplitKGemmPlan#ofSubgroupMmq} with its compute units), the plain subgroup kernel against its split-K
     * twin with the reduce, and the portable kernel's split-K launch the dispatcher ran before the twin existed.
     */
    @Test
    public void measureTheSplitKTwin() {
        Assume.assumeTrue("opt-in: -D" + SPLIT_MEASURE_PROPERTY + "=true", Boolean.getBoolean(SPLIT_MEASURE_PROPERTY));
        int[][] shapes = {{1024, 2048}, {1024, 3072}, {2048, 1024}, {3072, 1024}, {1024, 1024}, {4096, 2560},
                {2560, 9728}, {9728, 2560}};
        int[] tokens = {2, 8, 16, 32, 48, 64, 96};
        for (ComputeBackend backend : backends) {
            LlmKernels kernels = kernelsServing(backend);
            if (kernels == null || !kernels.has(type.name() + GemmKernels.MATMUL_MMQ_SG_KSPLIT_SUFFIX)) {
                continue;
            }
            for (int[] s : shapes) {
                for (int n : tokens) {
                    measureSplit(backend, kernels, s[0], s[1], n);
                }
            }
        }
    }

    private void measureSplit(ComputeBackend backend, LlmKernels kernels, int m, int k, int n) {
        int cu = backend.capabilities().getComputeUnits();
        int reduceLocal = kernels.matvecLocal();
        int group = SubgroupMmqGemm.workGroup(kernels, type);
        MmqGemmPlan base = MmqGemmPlan.contiguous(MmqConfig.PACKED_DOT, type, m, k, n);
        SplitKGemmPlan sgSplit = SplitKGemmPlan.ofSubgroupMmq(base, SubgroupMmqGemm.TILE_ROWS,
                SubgroupMmqGemm.TILE_TOKENS, group, cu, reduceLocal);
        SplitKGemmPlan portableSplit = SplitKGemmPlan.of(base, cu, reduceLocal);
        Random rnd = new Random(m + k + n);
        byte[] weight = weight(type, rnd, m, k);
        int[] records = records(rnd, n, k, k / TK, 0);
        ComputeQueue queue = backend.computeQueue();
        int planes = SplitKGemmPlan.MAX_SPLITS * m * n;
        GpuBuffer w = backend.allocateBuffer(weight.length, true);
        GpuBuffer q = backend.allocateBuffer((long) records.length * 4, true);
        GpuBuffer y = backend.allocateBuffer((long) m * n * 4, true);
        GpuBuffer p = backend.allocateBuffer((long) planes * 4, true);
        try {
            close(queue.enqueueWriteBuffer(w, 0, weight.length, weight));
            close(queue.enqueueWriteBuffer(q, 0, (long) records.length * 4, bytes(records)));
            double plain = time(queue, kernels, SubgroupMmqGemm.kernel(type),
                    new long[]{SubgroupMmqGemm.globalRows(m, group), SubgroupMmqGemm.tokenGroups(n)},
                    new long[]{group, 1}, new Object[]{m, k, n, k / TK, 0, m, 0, w, q, y});
            double sg = sgSplit == null ? Double.NaN : timeSplit(queue, kernels, sgSplit, w, q, p, y, m);
            double portable = portableSplit == null ? Double.NaN
                    : timeSplit(queue, kernels, portableSplit, w, q, p, y, m);
            System.out.printf("SGSPLIT %s %s M=%d K=%d N=%d CU=%d sg %.1f us, sg split x%d %.1f us,"
                            + " portable split x%d %.1f us%n", backend.capabilities().getDeviceName(), type, m, k, n,
                    cu, plain, sgSplit == null ? 1 : sgSplit.splits(), sg,
                    portableSplit == null ? 1 : portableSplit.splits(), portable);
        } finally {
            w.close();
            q.close();
            y.close();
            p.close();
        }
    }

    /** Mean microseconds of the partial launch and its reduce, as {@link #time}. */
    private static double timeSplit(ComputeQueue queue, LlmKernels kernels, SplitKGemmPlan plan, GpuBuffer w,
                                    GpuBuffer q, GpuBuffer p, GpuBuffer y, int m) {
        Object[] partialArgs = plan.partialArgs(w, q, p);
        Object[] reduceArgs = plan.reduceArgs(p, y, m, 0);
        Runnable step = () -> {
            queue.enqueueKernelNoEvent(kernels.get(plan.kernelName()), plan.partialGlobalWorkSize(),
                    plan.partialLocalWorkSize(), partialArgs);
            queue.enqueueKernelNoEvent(kernels.get(GemmKernels.SPLIT_K_REDUCE), plan.reduceGlobalWorkSize(),
                    plan.reduceLocalWorkSize(), reduceArgs);
        };
        for (int i = 0; i < 3; i++) {
            step.run();
        }
        queue.finish();
        int reps = 20;
        long t0 = System.nanoTime();
        for (int i = 0; i < reps; i++) {
            step.run();
        }
        queue.finish();
        return (System.nanoTime() - t0) / 1e3 / reps;
    }

    private LlmKernels kernelsServing(ComputeBackend backend) {
        if (!LlmKernels.availability(backend).available()) {
            return null;
        }
        LlmKernels kernels = LlmKernels.open(backend);
        return SubgroupMmqGemm.programBuilds(kernels) && kernels.has(SubgroupMmqGemm.kernel(type)) ? kernels : null;
    }

    private void run(ComputeBackend backend, LlmKernels kernels, String dev, int m, int k, int n, int qStride,
                     int qOff, int yStride, int yOff) {
        String what = dev + " " + type + " M=" + m + " K=" + k + " N=" + n + " qStride=" + qStride + " qOff=" + qOff
                + " yStride=" + yStride + " yOff=" + yOff;
        assertTrue(what, SubgroupMmqGemm.shapeServes(type, k));
        Random rnd = new Random(31L * m + 7L * k + n + type.ordinal());
        byte[] weight = weight(type, rnd, m, k);
        int[] records = records(rnd, n, k, qStride, qOff);
        float[] sg = launchSg(backend, kernels, weight, records, m, k, n, qStride, qOff, yStride, yOff);
        MmqGemmPlan plan = MmqGemmPlan.of(MmqConfig.PACKED_DOT, type, m, k, n, qStride, qOff, yStride, yOff);
        float[] portable = launch(backend, kernels, plan.kernelName(), plan.globalWorkSize(), plan.localWorkSize(),
                weight, records, m, k, n, qStride, qOff, yStride, yOff);
        // first-order bound of the kernel's fp32 evaluation: a few roundings per record term and one per addition
        double bound = (2.0 * k / TK + 8) * FP32_U;
        int yLen = yOff + (n - 1) * yStride + m;
        for (int i = 0; i < yLen; i++) {
            int rel = i - yOff;
            int t = rel < 0 ? -1 : rel / yStride;
            int r = rel < 0 ? -1 : rel % yStride;
            if (rel < 0 || r >= m) {
                assertEquals(what + " untouched cell " + i, SENTINEL, sg[i], 0.0f);
                continue;
            }
            double[] ref = reference(type, weight, records, k, qStride, qOff, t, r);
            double tol = bound * ref[1] + 1e-6;
            assertTrue(what + " token " + t + " row " + r + ": kernel " + sg[i] + " expected " + ref[0],
                    Math.abs(sg[i] - ref[0]) <= tol);
            assertTrue(what + " token " + t + " row " + r + ": kernel " + sg[i] + " portable " + portable[i],
                    Math.abs(sg[i] - portable[i]) <= 2 * tol);
        }
    }

    /**
     * {exact sum, sum of |terms|} of row r against token t, from the same bytes the kernel reads: per 16-deep record,
     * dx times the dot of the record's codes with the decoded weights.
     */
    private static double[] reference(GGMLType type, byte[] weight, int[] records, int k, int qStride, int qOff,
                                      int t, int r) {
        return reference(type, weight, records, k, qStride, qOff, t, r, 0, k / TK);
    }

    /** {@link #reference} over the 16-deep records {@code [c0, c1)} of the row. */
    private static double[] reference(GGMLType type, byte[] weight, int[] records, int k, int qStride, int qOff,
                                      int t, int r, int c0, int c1) {
        double acc = 0.0;
        double mag = 0.0;
        long rowBytes = rowBytes(type, k);
        for (int c = c0; c < c1; c++) {
            int rec = qOff + (t * qStride + c) * RC;
            double dx = Float.intBitsToFloat(records[rec + 4]);
            for (int i = 0; i < TK; i++) {
                int code = (byte) (records[rec + i / 4] >>> (8 * (i % 4)));
                double[] w = weightTerms(type, weight, r * rowBytes, c * TK + i);
                for (double term : w) {
                    double v = dx * term * code;
                    acc += v;
                    mag += Math.abs(v);
                }
            }
        }
        return new double[]{acc, mag};
    }

    /**
     * The terms whose sum is weight element e of the row at byte rowBase: the scaled code and, for Q4_K and Q5_K, the
     * negated minimum (kept apart so the magnitude bound sees both).
     */
    private static double[] weightTerms(GGMLType type, byte[] w, long rowBase, int e) {
        switch (type) {
            case Q8_0: {
                int base = (int) (rowBase + (long) (e / 32) * 34);
                return new double[]{f16(u16(w, base)) * w[base + 2 + e % 32]};
            }
            case Q4_0: {
                int base = (int) (rowBase + (long) (e / 32) * 18);
                int qs = w[base + 2 + e % 16] & 0xFF;
                int code = (e % 32) < 16 ? qs & 0xF : qs >>> 4;
                return new double[]{f16(u16(w, base)) * (code - 8)};
            }
            case Q5_K: {
                int base = (int) (rowBase + (long) (e / 256) * 176);
                int el = e % 256;
                int j = el / 32;
                int[] sm = scaleMinK4(w, base + 4, j);
                int qs = w[base + 48 + 32 * (el / 64) + el % 32] & 0xFF;
                int qh = w[base + 16 + el % 32] & 0xFF;
                int code = ((el % 64) < 32 ? qs & 0xF : qs >>> 4) | (((qh >>> j) & 1) << 4);
                return new double[]{f16(u16(w, base)) * sm[0] * code, -f16(u16(w, base + 2)) * sm[1]};
            }
            case Q4_K: {
                int base = (int) (rowBase + (long) (e / 256) * 144);
                int el = e % 256;
                int j = el / 32;
                int[] sm = scaleMinK4(w, base + 4, j);
                int qs = w[base + 16 + 32 * (el / 64) + el % 32] & 0xFF;
                int code = (el % 64) < 32 ? qs & 0xF : qs >>> 4;
                return new double[]{f16(u16(w, base)) * sm[0] * code, -f16(u16(w, base + 2)) * sm[1]};
            }
            case Q6_K: {
                int base = (int) (rowBase + (long) (e / 256) * 210);
                int el = e % 256;
                int h = el / 128;
                int l = el % 32;
                int g = (el % 128) / 32;
                int ql = w[base + 64 * h + 32 * (g % 2) + l] & 0xFF;
                int qh = w[base + 128 + 32 * h + l] & 0xFF;
                int code = ((g < 2 ? ql & 0xF : ql >>> 4) | (((qh >>> (2 * g)) & 3) << 4)) - 32;
                int scale = w[base + 192 + 8 * h + 2 * g + l / 16];
                return new double[]{f16(u16(w, base + 208)) * scale * code};
            }
            default:
                throw new IllegalArgumentException("no reference for " + type);
        }
    }

    /** {scale, min} of Q4_K sub-block j from the 12-byte table at q, as ggml's get_scale_min_k4. */
    private static int[] scaleMinK4(byte[] w, int q, int j) {
        if (j < 4) {
            return new int[]{w[q + j] & 63, w[q + j + 4] & 63};
        }
        int sc = (w[q + j + 4] & 0xF) | (((w[q + j - 4] & 0xFF) >>> 6) << 4);
        int mn = ((w[q + j + 4] & 0xFF) >>> 4) | (((w[q + j] & 0xFF) >>> 6) << 4);
        return new int[]{sc, mn};
    }

    private float[] launchSg(ComputeBackend backend, LlmKernels kernels, byte[] weight, int[] records, int m, int k,
                             int n, int qStride, int qOff, int yStride, int yOff) {
        return launch(backend, kernels, SubgroupMmqGemm.kernel(type),
                new long[]{SubgroupMmqGemm.globalRows(m, SubgroupMmqGemm.workGroup(kernels, type)),
                        SubgroupMmqGemm.tokenGroups(n)},
                new long[]{SubgroupMmqGemm.workGroup(kernels, type), 1}, weight, records, m, k, n, qStride, qOff,
                yStride, yOff);
    }

    private static float[] launch(ComputeBackend backend, LlmKernels kernels, String kernel, long[] global,
                                  long[] local, byte[] weight, int[] records, int m, int k, int n, int qStride,
                                  int qOff, int yStride, int yOff) {
        ComputeQueue queue = backend.computeQueue();
        int yLen = yOff + (n - 1) * yStride + m;
        float[] init = new float[yLen];
        Arrays.fill(init, SENTINEL);
        GpuBuffer w = backend.allocateBuffer(weight.length, true);
        GpuBuffer q = backend.allocateBuffer((long) records.length * 4, true);
        GpuBuffer y = backend.allocateBuffer((long) yLen * 4, true);
        try {
            close(queue.enqueueWriteBuffer(w, 0, weight.length, weight));
            close(queue.enqueueWriteBuffer(q, 0, (long) records.length * 4, bytes(records)));
            close(queue.enqueueWriteBuffer(y, 0, (long) yLen * 4, init));
            close(queue.enqueueKernel(kernels.get(kernel), global, local,
                    new Object[]{m, k, n, qStride, qOff, yStride, yOff, w, q, y}));
            float[] out = new float[yLen];
            close(queue.enqueueReadBuffer(y, 0, (long) yLen * 4, out));
            queue.finish();
            return out;
        } finally {
            w.close();
            q.close();
            y.close();
        }
    }

    private void measure(ComputeBackend backend, LlmKernels kernels, int m, int k, int n) {
        Random rnd = new Random(m + k + n);
        byte[] weight = weight(type, rnd, m, k);
        int[] records = records(rnd, n, k, k / TK, 0);
        MmqGemmPlan plan = MmqGemmPlan.contiguous(MmqConfig.PACKED_DOT, type, m, k, n);
        ComputeQueue queue = backend.computeQueue();
        GpuBuffer w = backend.allocateBuffer(weight.length, true);
        GpuBuffer q = backend.allocateBuffer((long) records.length * 4, true);
        GpuBuffer y = backend.allocateBuffer((long) m * n * 4, true);
        try {
            close(queue.enqueueWriteBuffer(w, 0, weight.length, weight));
            close(queue.enqueueWriteBuffer(q, 0, (long) records.length * 4, bytes(records)));
            Object[] args = {m, k, n, k / TK, 0, m, 0, w, q, y};
            double portable = time(queue, kernels, plan.kernelName(), plan.globalWorkSize(), plan.localWorkSize(),
                    args);
            double sg = time(queue, kernels, SubgroupMmqGemm.kernel(type),
                    new long[]{SubgroupMmqGemm.globalRows(m, SubgroupMmqGemm.workGroup(kernels, type)),
                            SubgroupMmqGemm.tokenGroups(n)},
                    new long[]{SubgroupMmqGemm.workGroup(kernels, type), 1}, args);
            double flop = 2.0 * m * k * n;
            System.out.printf("SGMMQ %s %s M=%d K=%d N=%d portable %.1f us (%.0f GFLOPS) sg %.1f us (%.0f GFLOPS)"
                            + " speedup %.2fx%n", backend.capabilities().getDeviceName(), type, m, k, n, portable,
                    flop / portable / 1e3, sg, flop / sg / 1e3, portable / sg);
        } finally {
            w.close();
            q.close();
            y.close();
        }
    }

    /** Mean microseconds per launch over 20 launches after 3 warm-up launches. */
    private static double time(ComputeQueue queue, LlmKernels kernels, String kernel, long[] global, long[] local,
                               Object[] args) {
        for (int i = 0; i < 3; i++) {
            queue.enqueueKernelNoEvent(kernels.get(kernel), global, local, args);
        }
        queue.finish();
        int reps = 20;
        long t0 = System.nanoTime();
        for (int i = 0; i < reps; i++) {
            queue.enqueueKernelNoEvent(kernels.get(kernel), global, local, args);
        }
        queue.finish();
        return (System.nanoTime() - t0) / 1e3 / reps;
    }

    /** Bytes of one row of {@code k} elements of {@code type}. */
    private static long rowBytes(GGMLType type, int k) {
        return (long) (k / type.getBlockSize()) * type.getTypeSize();
    }

    /** {@code m} rows of {@code k} elements of {@code type}, every block filled by {@link #block}. */
    private static byte[] weight(GGMLType type, Random rnd, int m, int k) {
        int blockBytes = type.getTypeSize();
        byte[] b = new byte[Math.toIntExact(m * rowBytes(type, k))];
        for (int base = 0; base < b.length; base += blockBytes) {
            block(type, rnd, b, base);
        }
        return b;
    }

    /**
     * One block at base. Q8_0: f16 scale 2^-6 (1 + j/64), codes over -128..127 with both extremes forced. Q4_0: the
     * same scale, the 16 code bytes uniformly random. Q4_K and Q5_K: f16 d and dmin of the same form, the 12 scale
     * bytes and the code bytes (and Q5_K's 32 qh bytes) uniformly random (every 6-bit scale and minimum, every
     * code). Q6_K: ql and qh uniformly random (every 6-bit code), int8 scales over -128..127 with
     * both extremes forced, f16 d 2^-9 (1 + j/64).
     */
    private static void block(GGMLType type, Random rnd, byte[] b, int base) {
        switch (type) {
            case Q8_0:
                putF16(b, base, (9 << 10) | (rnd.nextInt(64) << 4));
                for (int i = 0; i < 32; i++) {
                    b[base + 2 + i] = (byte) (rnd.nextInt(256) - 128);
                }
                b[base + 2] = (byte) -128;
                b[base + 3] = (byte) 127;
                break;
            case Q4_0:
                putF16(b, base, (9 << 10) | (rnd.nextInt(64) << 4));
                for (int i = 2; i < 18; i++) {
                    b[base + i] = (byte) rnd.nextInt(256);
                }
                break;
            case Q4_K:
            case Q5_K:
                putF16(b, base, (9 << 10) | (rnd.nextInt(64) << 4));
                putF16(b, base + 2, (8 << 10) | (rnd.nextInt(64) << 4));
                for (int i = 4; i < type.getTypeSize(); i++) {
                    b[base + i] = (byte) rnd.nextInt(256);
                }
                break;
            case Q6_K:
                for (int i = 0; i < 192; i++) {
                    b[base + i] = (byte) rnd.nextInt(256);
                }
                for (int i = 0; i < 16; i++) {
                    b[base + 192 + i] = (byte) (rnd.nextInt(256) - 128);
                }
                b[base + 192] = (byte) -128;
                b[base + 193] = (byte) 127;
                putF16(b, base + 208, (6 << 10) | (rnd.nextInt(64) << 4));
                break;
            default:
                throw new IllegalArgumentException("no block generator for " + type);
        }
    }

    /**
     * Staged records of {@code n} tokens at {@code qOff} units with {@code qStride} records between tokens: codes
     * over -127..127, a positive scale, the exact code sum and zero point 0; padding units hold garbage the kernel
     * must not read.
     */
    private static int[] records(Random rnd, int n, int k, int qStride, int qOff) {
        int[] r = new int[qOff + ((n - 1) * qStride + k / TK) * RC];
        for (int i = 0; i < r.length; i++) {
            r[i] = rnd.nextInt();
        }
        for (int t = 0; t < n; t++) {
            for (int c = 0; c < k / TK; c++) {
                int rec = qOff + (t * qStride + c) * RC;
                int sum = 0;
                for (int u = 0; u < 4; u++) {
                    int word = 0;
                    for (int j = 0; j < 4; j++) {
                        int code = rnd.nextInt(255) - 127;
                        sum += code;
                        word |= (code & 0xFF) << (8 * j);
                    }
                    r[rec + u] = word;
                }
                r[rec + 4] = Float.floatToRawIntBits(0.001f + rnd.nextFloat() * 0.05f);
                r[rec + 5] = sum;
                r[rec + 6] = 0;
            }
        }
        return r;
    }

    private static byte[] bytes(int[] words) {
        java.nio.ByteBuffer b = java.nio.ByteBuffer.allocate(words.length * 4).order(java.nio.ByteOrder.LITTLE_ENDIAN);
        b.asIntBuffer().put(words);
        return b.array();
    }

    private static int u16(byte[] b, int at) {
        return (b[at] & 0xFF) | ((b[at + 1] & 0xFF) << 8);
    }

    private static void putF16(byte[] b, int at, int bits) {
        b[at] = (byte) bits;
        b[at + 1] = (byte) (bits >>> 8);
    }

    private static double f16(int h) {
        int e = (h >>> 10) & 0x1F;
        int f = h & 0x3FF;
        double v = e == 0 ? f / 1024.0 * Math.pow(2, -14) : (1 + f / 1024.0) * Math.pow(2, e - 15);
        return (h & 0x8000) != 0 ? -v : v;
    }

    private static void close(GpuEvent e) {
        if (e != null) {
            e.close();
        }
    }
}

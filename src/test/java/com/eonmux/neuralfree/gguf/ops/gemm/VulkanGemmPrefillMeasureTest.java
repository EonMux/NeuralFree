package com.eonmux.neuralfree.gguf.ops.gemm;

import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.DeviceFeature;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.acceleration.backend.vulkan.VkComputeBackend;
import com.eonmux.neuralfree.acceleration.vulkan.VkInstanceManager;
import com.eonmux.neuralfree.acceleration.vulkan.VkPhysicalDeviceInfo;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;

/**
 * Opt-in kernel-level timing of the Vulkan prefill GEMM lanes ({@code -Dneuralfree.test.vk.gemm.measure=true}) on
 * the Vulkan device whose name contains {@code -Dneuralfree.test.vk.gemm.measure.device} (default NVIDIA). Per
 * weight type and decoder projection shape it interleaves rounds of the lanes the dispatch can pick at
 * {@value #TOKENS} tokens — the int8 MMQ GEMM with its activation staging, the fp16-tile KHR CM GEMM and the
 * VK_NV_cooperative_matrix2 CM2 GEMM with its activation staging (and its four-element decode form CM2V), whole, on
 * its 128-token tile, as the split-K launch the dispatch's heuristic ({@link SplitKGemmPlan#of(Cm2GemmPlan, int, int)})
 * makes of it where it splits and at fixed slice counts — and prints the median microseconds per launch with the
 * achieved TFLOPS. Asserts nothing; the numbers set the lane, the tile and the split.
 */
public class VulkanGemmPrefillMeasureTest {

    /** System property that turns the measurement on. */
    public static final String MEASURE_PROPERTY = "neuralfree.test.vk.gemm.measure";
    /** System property naming a substring of the measured device. */
    public static final String DEVICE_PROPERTY = "neuralfree.test.vk.gemm.measure.device";
    /** System property listing the measured weight types, comma-separated (default {@link #TYPES}). */
    public static final String TYPES_PROPERTY = "neuralfree.test.vk.gemm.measure.types";
    /** System property listing the measured shapes as comma-separated {@code ROWSxCOLS} (default {@link #SHAPES}). */
    public static final String SHAPES_PROPERTY = "neuralfree.test.vk.gemm.measure.shapes";
    /** System property that adds, per fixed split lane, its partial launch alone and the whole-K kernel over one
     *  slice's depth: the split's own cost against the work each of its slices does. */
    public static final String DIAGNOSE_PROPERTY = "neuralfree.test.vk.gemm.measure.diagnose";
    private static final int TOKENS = 512;
    private static final int ROUNDS = 7;
    private static final int RUNS_PER_ROUND = 20;
    private static final int CLOCK_WARMUP_LAUNCHES = 400;
    /** 0.6B and 4B decoder shapes {rows, cols}: q, kv, attention out, ffn in, ffn out. */
    private static final int[][] SHAPES = {
        {2048, 1024}, {1024, 1024}, {1024, 2048}, {3072, 1024}, {1024, 3072},
        {4096, 2560}, {1024, 2560}, {2560, 4096}, {9728, 2560}, {2560, 9728},
    };
    private static final GGMLType[] TYPES = {GGMLType.Q8_0, GGMLType.Q4_K, GGMLType.Q6_K};
    /** Slice counts of the fixed CM2 split lanes, whose slices round up to {@value #SPLIT_SLICE_ALIGN} elements. */
    private static final int[] MEASURED_SPLITS = {2, 3, 4};
    private static final int MAX_MEASURED_SPLITS = 4;
    /** Slice rounding of the fixed split lanes: llama.cpp's ggml_vk_matmul k_split rounding. */
    private static final int SPLIT_SLICE_ALIGN = 256;

    private VkComputeBackend backend;
    private LlmKernels kernels;
    private ComputeQueue queue;
    private MmqConfig mmqConfig;

    @Before
    public void openDevice() {
        Assume.assumeTrue("set " + MEASURE_PROPERTY + "=true to run", Boolean.getBoolean(MEASURE_PROPERTY));
        String wanted = System.getProperty(DEVICE_PROPERTY, "NVIDIA");
        VkInstanceManager manager = VkInstanceManager.acquire();
        try {
            for (VkPhysicalDeviceInfo info : manager.devices()) {
                if (info.name().contains(wanted)) {
                    backend = new VkComputeBackend(manager, info);
                    break;
                }
            }
        } finally {
            manager.release();
        }
        Assume.assumeTrue("no Vulkan device named " + wanted, backend != null);
        kernels = LlmKernels.open(backend);
        queue = backend.computeQueue();
        mmqConfig = MmqConfig.forDevice(backend.backendType(), backend.capabilities().getComputeCapability());
    }

    @After
    public void closeDevice() {
        if (backend != null) {
            backend.close();
        }
    }

    @Test
    public void measurePrefillLanes() throws InterruptedException {
        boolean cm = backend.capabilities().hasFeature(DeviceFeature.COOPERATIVE_MATRIX);
        boolean cm2 = backend.capabilities().hasFeature(DeviceFeature.COOPERATIVE_MATRIX_2);
        boolean cm2v = backend.capabilities().hasFeature(DeviceFeature.COOPERATIVE_MATRIX_2_DECODE_VECTOR);
        System.out.printf(Locale.ROOT, "device %s: cooperative matrix %s, cooperative matrix 2 %s (decode vector %s),"
                + " %d tokens%n", backend.capabilities().getDeviceName(), cm, cm2, cm2v, TOKENS);
        for (GGMLType type : measuredTypes()) {
            for (int[] shape : measuredShapes()) {
                measureShape(type, shape[0], shape[1], cm, cm2, cm2v);
            }
        }
    }

    private static List<GGMLType> measuredTypes() {
        String value = System.getProperty(TYPES_PROPERTY, "").trim();
        if (value.isEmpty()) {
            return Arrays.asList(TYPES);
        }
        List<GGMLType> types = new ArrayList<>();
        for (String name : value.split(",")) {
            types.add(GGMLType.valueOf(name.trim()));
        }
        return types;
    }

    private static List<int[]> measuredShapes() {
        String value = System.getProperty(SHAPES_PROPERTY, "").trim();
        if (value.isEmpty()) {
            return Arrays.asList(SHAPES);
        }
        List<int[]> shapes = new ArrayList<>();
        for (String shape : value.split(",")) {
            String[] dims = shape.trim().split("x");
            if (dims.length != 2) {
                throw new IllegalArgumentException(SHAPES_PROPERTY + " takes ROWSxCOLS entries, got '" + shape + "'");
            }
            shapes.add(new int[]{Integer.parseInt(dims[0]), Integer.parseInt(dims[1])});
        }
        return shapes;
    }

    private void measureShape(GGMLType type, int m, int k, boolean cm, boolean cm2, boolean cm2v)
            throws InterruptedException {
        int n = TOKENS;
        byte[] weight = OpenClMmqGemmParityTest.buildWeight(type, m, k);
        float[] x = OpenClMmqGemmParityTest.buildX(n, k);
        GpuBuffer wbuf = backend.allocateBuffer(weight.length, true);
        GpuBuffer xbuf = backend.allocateBuffer((long) x.length * Float.BYTES, true);
        Cm2GemmPlan cm2Plan = GemmKernels.hasCm2(type) ? Cm2GemmPlan.contiguous(type, m, k, n) : null;
        GpuBuffer hbuf = backend.allocateBuffer(cm2Plan == null ? 4 : cm2Plan.stagedHalves() * Short.BYTES);
        MmqGemmPlan mmq = MmqGemmPlan.contiguous(mmqConfig, type, m, k, n);
        GpuBuffer qbuf = backend.allocateBuffer(mmq.q8ElementsRequired() * Integer.BYTES);
        GpuBuffer ybuf = backend.allocateBuffer((long) n * m * Float.BYTES);
        /* the heuristic's split of the form the dispatch runs: the decode-vector form where the device has it */
        Cm2GemmPlan dispatched = cm2Plan == null || !(cm2v && GemmKernels.hasCm2DecodeVector(type)) ? cm2Plan
                : Cm2GemmPlan.of(type, true, m, k, n, k, 0, m, 0);
        SplitKGemmPlan split = dispatched == null ? null
                : SplitKGemmPlan.of(dispatched, backend.capabilities().getComputeUnits(), kernels.rowReduceLocal());
        GpuBuffer pbuf = backend.allocateBuffer(partialBufferFloats(split, n, m) * Float.BYTES);
        try {
            try (GpuEvent wv = queue.enqueueWriteBuffer(wbuf, 0, weight.length, weight);
                 GpuEvent xv = queue.enqueueWriteBuffer(xbuf, 0, (long) x.length * Float.BYTES, x, wv)) {
                xv.awaitCompletion();
            }
            List<String> names = new ArrayList<>();
            List<Runnable> lanes = new ArrayList<>();
            names.add("MMQ+stage");
            lanes.add(() -> {
                int tk = mmq.config().stageDepth();
                long records = (long) n * mmq.config().rowRecords(k);
                queue.enqueueKernelNoEvent(kernels.get(GemmKernels.QUANT_MMQ_Q8_1),
                        new long[]{(records + 63) / 64 * 64}, new long[]{64}, new Object[]{n, k, tk, k, 0, xbuf, qbuf});
                queue.enqueueKernelNoEvent(kernels.get(mmq.kernelName()), mmq.globalWorkSize(), mmq.localWorkSize(),
                        mmq.args(wbuf, qbuf, ybuf));
            });
            if (cm) {
                CmGemmPlan plan = CmGemmPlan.contiguous(type, m, k, n);
                names.add("CM");
                lanes.add(() -> queue.enqueueKernelNoEvent(kernels.get(plan.kernelName()), plan.globalWorkSize(),
                        plan.localWorkSize(), plan.args(wbuf, xbuf, ybuf)));
            }
            if (cm2 && cm2Plan != null) {
                names.add("CM2+stage");
                lanes.add(() -> {
                    queue.enqueueKernelNoEvent(kernels.get(Cm2GemmPlan.STAGE_KERNEL), cm2Plan.stageGlobalWorkSize(),
                            Cm2GemmPlan.stageLocalWorkSize(), cm2Plan.stageArgs(xbuf, hbuf));
                    queue.enqueueKernelNoEvent(kernels.get(cm2Plan.kernelName()), cm2Plan.globalWorkSize(),
                            cm2Plan.localWorkSize(), cm2Plan.args(wbuf, hbuf, ybuf));
                });
                if (cm2v && GemmKernels.hasCm2DecodeVector(type)) {
                    Cm2GemmPlan vector = Cm2GemmPlan.of(type, true, m, k, n, k, 0, m, 0);
                    names.add("CM2V+stage");
                    lanes.add(() -> {
                        queue.enqueueKernelNoEvent(kernels.get(Cm2GemmPlan.STAGE_KERNEL),
                                cm2Plan.stageGlobalWorkSize(), Cm2GemmPlan.stageLocalWorkSize(),
                                cm2Plan.stageArgs(xbuf, hbuf));
                        queue.enqueueKernelNoEvent(kernels.get(vector.kernelName()), vector.globalWorkSize(),
                                vector.localWorkSize(), vector.args(wbuf, hbuf, ybuf));
                    });
                }
                if (split != null && kernels.has(split.kernelName())) {
                    names.add(split.family() + "x" + split.splits() + "+stage");
                    lanes.add(() -> {
                        queue.enqueueKernelNoEvent(kernels.get(Cm2GemmPlan.STAGE_KERNEL),
                                cm2Plan.stageGlobalWorkSize(), Cm2GemmPlan.stageLocalWorkSize(),
                                cm2Plan.stageArgs(xbuf, hbuf));
                        queue.enqueueKernelNoEvent(kernels.get(split.kernelName()), split.partialGlobalWorkSize(),
                                split.partialLocalWorkSize(), split.partialArgs(wbuf, hbuf, pbuf));
                        queue.enqueueKernelNoEvent(kernels.get(GemmKernels.SPLIT_K_REDUCE),
                                split.reduceGlobalWorkSize(), split.reduceLocalWorkSize(),
                                split.reduceArgs(pbuf, ybuf, m, 0));
                    });
                }
                addAlignedSplits(type, m, k, cm2Plan, cm2v, names, lanes, wbuf, xbuf, hbuf, pbuf, ybuf);
                addMediumTile(type, m, k, cm2Plan, cm2v, names, lanes, wbuf, xbuf, hbuf, ybuf);
            }
            for (int i = 0; i < CLOCK_WARMUP_LAUNCHES / lanes.size(); i++) {
                for (Runnable lane : lanes) {
                    lane.run();
                }
            }
            queue.finish();
            double[][] samples = new double[lanes.size()][ROUNDS];
            for (int r = 0; r < ROUNDS; r++) {
                for (int l = 0; l < lanes.size(); l++) {
                    long t0 = System.nanoTime();
                    for (int i = 0; i < RUNS_PER_ROUND; i++) {
                        lanes.get(l).run();
                    }
                    queue.finish();
                    samples[l][r] = (System.nanoTime() - t0) / 1e3 / RUNS_PER_ROUND;
                }
            }
            StringBuilder line = new StringBuilder(String.format(Locale.ROOT, "%-5s %5dx%-5d", type, m, k));
            double flops = 2.0 * m * k * n;
            for (int l = 0; l < lanes.size(); l++) {
                double[] s = samples[l].clone();
                Arrays.sort(s);
                double us = s[s.length / 2];
                line.append(String.format(Locale.ROOT, "  %s %8.1f us (%5.1f TFLOPS)", names.get(l), us,
                        flops / us / 1e6));
            }
            System.out.println(line);
        } finally {
            wbuf.close();
            xbuf.close();
            hbuf.close();
            qbuf.close();
            ybuf.close();
            pbuf.close();
        }
    }

    /**
     * Floats of the slice planes buffer every split lane of an {@code m x n} shape writes: the fixed lanes' up to
     * {@link #MAX_MEASURED_SPLITS} planes, and the heuristic's ({@code heuristic}, null when the shape has no CM2
     * lane), which splits a small grid into up to {@link SplitKGemmPlan#MAX_SPLITS} slices.
     */
    static long partialBufferFloats(SplitKGemmPlan heuristic, int n, int m) {
        int planes = Math.max(MAX_MEASURED_SPLITS, heuristic == null ? 1 : heuristic.splits());
        return (long) planes * n * m;
    }

    /**
     * Lanes of the CM2 split-K launch at fixed slice counts ({@link #MEASURED_SPLITS}) whose slices round up to
     * {@value #SPLIT_SLICE_ALIGN} reduction elements, as llama.cpp's coopmat2 split rounds its k_split, in both decode
     * forms where the device has the four-element one: they set the CM2 split rule against the heuristic's lane.
     */
    private void addAlignedSplits(GGMLType type, int m, int k, Cm2GemmPlan cm2Plan, boolean cm2v, List<String> names,
                                  List<Runnable> lanes, GpuBuffer wbuf, GpuBuffer xbuf, GpuBuffer hbuf,
                                  GpuBuffer pbuf, GpuBuffer ybuf) {
        if (k < SplitKGemmPlan.MIN_SPLIT_K) {
            return;
        }
        List<SplitKGemmPlan.Family> families = new ArrayList<>();
        families.add(SplitKGemmPlan.Family.CM2);
        if (cm2v && GemmKernels.hasCm2DecodeVector(type)) {
            families.add(SplitKGemmPlan.Family.CM2V);
        }
        for (int splits : MEASURED_SPLITS) {
            for (int align : new int[]{SPLIT_SLICE_ALIGN, Cm2GemmPlan.TILE_DEPTH}) {
                addSplit(type, m, k, splits, align, align == SPLIT_SLICE_ALIGN ? families
                        : List.of(SplitKGemmPlan.Family.CM2), cm2Plan, names, lanes, wbuf, xbuf, hbuf, pbuf, ybuf);
            }
        }
    }

    private void addSplit(GGMLType type, int m, int k, int splits, int align, List<SplitKGemmPlan.Family> families,
                          Cm2GemmPlan cm2Plan, List<String> names, List<Runnable> lanes, GpuBuffer wbuf,
                          GpuBuffer xbuf, GpuBuffer hbuf, GpuBuffer pbuf, GpuBuffer ybuf) {
        int slice = SplitKGemmPlan.sliceLength(k, splits, align);
        if ((long) slice * (splits - 1) >= k) {
            return;
        }
        boolean diagnose = Boolean.getBoolean(DIAGNOSE_PROPERTY);
        for (SplitKGemmPlan.Family family : families) {
            SplitKGemmPlan plan = SplitKGemmPlan.of(family, type, m, k, TOKENS, k, 0, splits, slice,
                    Cm2GemmPlan.TILE_TOKENS, kernels.rowReduceLocal());
            names.add(family + "x" + splits + "/" + align + "+stage");
            lanes.add(() -> {
                queue.enqueueKernelNoEvent(kernels.get(Cm2GemmPlan.STAGE_KERNEL), cm2Plan.stageGlobalWorkSize(),
                        Cm2GemmPlan.stageLocalWorkSize(), cm2Plan.stageArgs(xbuf, hbuf));
                queue.enqueueKernelNoEvent(kernels.get(plan.kernelName()), plan.partialGlobalWorkSize(),
                        plan.partialLocalWorkSize(), plan.partialArgs(wbuf, hbuf, pbuf));
                queue.enqueueKernelNoEvent(kernels.get(GemmKernels.SPLIT_K_REDUCE), plan.reduceGlobalWorkSize(),
                        plan.reduceLocalWorkSize(), plan.reduceArgs(pbuf, ybuf, m, 0));
            });
            if (diagnose) {
                names.add(family + "x" + splits + "/" + align + " partial");
                lanes.add(() -> queue.enqueueKernelNoEvent(kernels.get(plan.kernelName()),
                        plan.partialGlobalWorkSize(), plan.partialLocalWorkSize(), plan.partialArgs(wbuf, hbuf, pbuf)));
                Cm2GemmPlan first = Cm2GemmPlan.of(type, family == SplitKGemmPlan.Family.CM2V, m, slice, TOKENS, k, 0,
                        m, 0);
                names.add(family + " k=" + slice);
                lanes.add(() -> queue.enqueueKernelNoEvent(kernels.get(first.kernelName()), first.globalWorkSize(),
                        first.localWorkSize(), first.args(wbuf, hbuf, ybuf)));
            }
        }
    }

    /** The lane of the 128-token CM2 tile ({@code _N128}) in the form the device runs, whole. */
    private void addMediumTile(GGMLType type, int m, int k, Cm2GemmPlan cm2Plan, boolean cm2v, List<String> names,
                               List<Runnable> lanes, GpuBuffer wbuf, GpuBuffer xbuf, GpuBuffer hbuf, GpuBuffer ybuf) {
        boolean vector = cm2v && GemmKernels.hasCm2DecodeVector(type);
        Cm2GemmPlan medium = Cm2GemmPlan.of(type, vector, Cm2GemmPlan.MEDIUM_TILE_TOKENS, m, k, TOKENS, k, 0, m, 0);
        if (!kernels.has(medium.kernelName())) {
            return;
        }
        names.add((vector ? "CM2V" : "CM2") + "/N128+stage");
        lanes.add(() -> {
            queue.enqueueKernelNoEvent(kernels.get(Cm2GemmPlan.STAGE_KERNEL), cm2Plan.stageGlobalWorkSize(),
                    Cm2GemmPlan.stageLocalWorkSize(), cm2Plan.stageArgs(xbuf, hbuf));
            queue.enqueueKernelNoEvent(kernels.get(medium.kernelName()), medium.globalWorkSize(),
                    medium.localWorkSize(), medium.args(wbuf, hbuf, ybuf));
        });
    }
}

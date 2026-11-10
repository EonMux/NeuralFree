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
 * VK_NV_cooperative_matrix2 CM2 GEMM with its activation staging, whole and as the split-K launch the dispatch's
 * heuristic ({@link SplitKGemmPlan#of(Cm2GemmPlan, int, int)}) makes of it where it splits — and prints the median
 * microseconds per launch with the achieved TFLOPS. Asserts nothing; the numbers set the lane, the tile and the
 * split.
 */
public class VulkanGemmPrefillMeasureTest {

    /** System property that turns the measurement on. */
    public static final String MEASURE_PROPERTY = "neuralfree.test.vk.gemm.measure";
    /** System property naming a substring of the measured device. */
    public static final String DEVICE_PROPERTY = "neuralfree.test.vk.gemm.measure.device";
    private static final int TOKENS = 512;
    private static final int ROUNDS = 7;
    private static final int RUNS_PER_ROUND = 20;
    private static final int CLOCK_WARMUP_LAUNCHES = 400;
    /** 0.6B and 4B decoder shapes {rows, cols}: q, kv, ffn in, ffn out. */
    private static final int[][] SHAPES = {
        {2048, 1024}, {1024, 1024}, {3072, 1024}, {1024, 3072},
        {4096, 2560}, {1024, 2560}, {9728, 2560}, {2560, 9728},
    };
    private static final GGMLType[] TYPES = {GGMLType.Q8_0, GGMLType.Q4_K, GGMLType.Q6_K};

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
        System.out.printf(Locale.ROOT, "device %s: cooperative matrix %s, cooperative matrix 2 %s, %d tokens%n",
                backend.capabilities().getDeviceName(), cm, cm2, TOKENS);
        for (GGMLType type : TYPES) {
            for (int[] shape : SHAPES) {
                measureShape(type, shape[0], shape[1], cm, cm2);
            }
        }
    }

    private void measureShape(GGMLType type, int m, int k, boolean cm, boolean cm2) throws InterruptedException {
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
        SplitKGemmPlan split = cm2Plan == null ? null
                : SplitKGemmPlan.of(cm2Plan, backend.capabilities().getComputeUnits(), kernels.rowReduceLocal());
        GpuBuffer pbuf = backend.allocateBuffer(split == null ? 4 : split.partialElements() * Float.BYTES);
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
                if (split != null && kernels.has(split.kernelName())) {
                    names.add("CM2x" + split.splits() + "+stage");
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
}

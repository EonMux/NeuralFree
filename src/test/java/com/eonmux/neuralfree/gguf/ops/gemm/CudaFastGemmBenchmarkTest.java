package com.eonmux.neuralfree.gguf.ops.gemm;

import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.cuda.CudaComputeBackend;
import com.eonmux.neuralfree.acceleration.cuda.CUDevice;
import com.eonmux.neuralfree.acceleration.kernels.Availability;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;

import java.util.Locale;

/**
 * Opt-in timing of the fast GEMM kernels against the tiled GEMM on the CUDA device (environment
 * {@code NEURALFREE_BENCH_GEMM=1}; the test task forwards no system properties): prints the mean
 * kernel time and the achieved TFLOP/s of every form over llama-sized projection shapes. Asserts
 * nothing; the numbers go to the benchmark note of the fast GEMM documentation.
 */
public class CudaFastGemmBenchmarkTest {

    private static final String SWITCH = "NEURALFREE_BENCH_GEMM";
    private static final int WARMUP = 3;
    private static final int RUNS = 20;

    private HWAcceleration hwAccel;
    private CudaComputeBackend backend;
    private ComputeQueue queue;
    private LlmKernels kernels;

    @Before
    public void setUp() {
        Assume.assumeTrue("set " + SWITCH + "=1 to run", "1".equals(System.getenv(SWITCH)));
        HWAcceleration.enableCuda = true;
        HWAcceleration hw = new HWAcceleration();
        if (hw.getCUDevices().isEmpty()) {
            hw.shutdown();
            Assume.assumeTrue("no CUDA device", false);
        }
        CUDevice device = hw.getCUDevices().get(0);
        CudaComputeBackend b = new CudaComputeBackend(device);
        Availability availability = LlmKernels.availability(b);
        if (!availability.available()) {
            b.close();
            hw.shutdown();
            Assume.assumeTrue("LLM kernels unavailable: " + availability.reason(), false);
        }
        hwAccel = hw;
        backend = b;
        queue = b.computeQueue();
        kernels = LlmKernels.open(b);
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
    public void fastGemmFormsAgainstTheTiledGemm() throws Exception {
        System.out.println("device " + backend.capabilities().getDeviceName() + " sm_"
                + backend.capabilities().getComputeCapability() + ", "
                + backend.capabilities().getComputeUnits() + " SMs");
        MmqConfig config = MmqConfig.forDevice(backend.backendType(), backend.capabilities().getComputeCapability());
        System.out.println("MMQ config " + config);
        int[][] shapes = {{4096, 4096, 512}, {14336, 4096, 512}, {4096, 14336, 512}, {4096, 4096, 64}};
        for (int[] s : shapes) {
            for (GGMLType type : new GGMLType[]{GGMLType.Q4_0, GGMLType.Q8_0, GGMLType.Q4_K, GGMLType.Q6_K}) {
                MmqGemmPlan mmq = MmqGemmPlan.contiguous(config, type, s[0], s[1], s[2]);
                TiledGemmPlan tiled = TiledGemmPlan.contiguous(type, s[0], s[1], s[2]);
                report(type + " " + shape(s), mmq.kernelName() + " + staging",
                        timeMmq(mmq, s), tiled.kernelName(),
                        time(tiled.kernelName(), tiled.globalWorkSize(), tiled.localWorkSize(),
                                tiled.weightBytes(), s), s);
            }
            for (GGMLType type : new GGMLType[]{GGMLType.F16, GGMLType.BF16}) {
                MmaGemmPlan mma = MmaGemmPlan.contiguous(type, s[0], s[1], s[2]);
                TiledGemmPlan tiled = TiledGemmPlan.contiguous(type, s[0], s[1], s[2]);
                report(type + " " + shape(s), mma.kernelName(), time(mma.kernelName(), mma.globalWorkSize(),
                        mma.localWorkSize(), mma.weightBytes(), s), tiled.kernelName(),
                        time(tiled.kernelName(), tiled.globalWorkSize(), tiled.localWorkSize(),
                                tiled.weightBytes(), s), s);
            }
        }
        int[][] skinny = {{1024, 8192, 8}, {512, 14336, 16}, {2048, 4096, 32}};
        int units = backend.capabilities().getComputeUnits();
        for (int[] s : skinny) {
            MmaGemmPlan plain = MmaGemmPlan.contiguous(GGMLType.F16, s[0], s[1], s[2]);
            SplitKGemmPlan split = SplitKGemmPlan.of(plain, units, 256);
            double plainMs = time(plain.kernelName(), plain.globalWorkSize(), plain.localWorkSize(),
                    plain.weightBytes(), s);
            if (split == null) {
                System.out.println(String.format(Locale.ROOT, "F16 %s: heuristic keeps the plain kernel (%.3f ms)",
                        shape(s), plainMs));
                continue;
            }
            double splitMs = timeSplit(split, s);
            System.out.println(String.format(Locale.ROOT,
                    "F16 %s: split-K x%d %.3f ms vs plain MMA %.3f ms (%.2fx)",
                    shape(s), split.splits(), splitMs, plainMs, plainMs / splitMs));
        }
        for (int[] s : new int[][]{{512, 8192, 8}, {2048, 4096, 16}}) {
            MmqGemmPlan plain = MmqGemmPlan.contiguous(config, GGMLType.Q4_0, s[0], s[1], s[2]);
            SplitKGemmPlan split = SplitKGemmPlan.of(plain, units, 256);
            double plainMs = timeMmq(plain, s);
            if (split == null) {
                System.out.println(String.format(Locale.ROOT, "Q4_0 %s: heuristic keeps the plain kernel (%.3f ms)",
                        shape(s), plainMs));
                continue;
            }
            double splitMs = timeMmqSplit(split, config, s);
            System.out.println(String.format(Locale.ROOT,
                    "Q4_0 %s: split-K x%d %.3f ms vs plain MMQ %.3f ms (%.2fx)",
                    shape(s), split.splits(), splitMs, plainMs, plainMs / splitMs));
        }
    }

    private static String shape(int[] s) {
        return "M=" + s[0] + " K=" + s[1] + " N=" + s[2];
    }

    private static void report(String label, String fast, double fastMs, String base, double baseMs, int[] s) {
        double flop = 2.0 * s[0] * s[1] * s[2];
        System.out.println(String.format(Locale.ROOT,
                "%s: %s %.3f ms (%.1f TFLOP/s) vs %s %.3f ms (%.1f TFLOP/s), %.2fx", label, fast, fastMs, flop / fastMs / 1e9, base, baseMs, flop / baseMs / 1e9, baseMs / fastMs));
    }

    /** Mean milliseconds of one launch of a 10-argument GEMM over zero-filled buffers. */
    private double time(String kernel, long[] global, long[] local, long weightBytes, int[] s) {
        GpuBuffer w = backend.allocateBuffer(weightBytes, true);
        GpuBuffer x = backend.allocateBuffer((long) s[1] * s[2] * Float.BYTES, true);
        GpuBuffer y = backend.allocateBuffer((long) s[0] * s[2] * Float.BYTES);
        try {
            Object[] args = {s[0], s[1], s[2], s[1], 0, s[0], 0, w, x, y};
            return timed(() -> queue.enqueueKernelNoEvent(kernels.get(kernel), global, local, args));
        } finally {
            w.close();
            x.close();
            y.close();
        }
    }

    private double timeSplit(SplitKGemmPlan plan, int[] s) {
        GpuBuffer w = backend.allocateBuffer(plan.weightBytes(), true);
        GpuBuffer x = backend.allocateBuffer(plan.xElementsRequired() * Float.BYTES, true);
        GpuBuffer p = backend.allocateBuffer(plan.partialElements() * Float.BYTES);
        GpuBuffer y = backend.allocateBuffer((long) s[0] * s[2] * Float.BYTES);
        try {
            Object[] partial = plan.partialArgs(w, x, p);
            Object[] reduce = plan.reduceArgs(p, y, s[0], 0);
            return timed(() -> {
                queue.enqueueKernelNoEvent(kernels.get(plan.kernelName()), plan.partialGlobalWorkSize(),
                        plan.partialLocalWorkSize(), partial);
                queue.enqueueKernelNoEvent(kernels.get(GemmKernels.SPLIT_K_REDUCE), plan.reduceGlobalWorkSize(),
                        plan.reduceLocalWorkSize(), reduce);
            });
        } finally {
            w.close();
            x.close();
            p.close();
            y.close();
        }
    }

    /** Mean milliseconds of one staged MMQ launch: the quantize pass plus the GEMM, as the dispatch runs them. */
    private double timeMmq(MmqGemmPlan plan, int[] s) {
        int tk = plan.config().stageDepth();
        GpuBuffer w = backend.allocateBuffer(plan.weightBytes(), true);
        GpuBuffer x = backend.allocateBuffer((long) s[1] * s[2] * Float.BYTES, true);
        GpuBuffer q = backend.allocateBuffer(plan.q8ElementsRequired() * Integer.BYTES);
        GpuBuffer y = backend.allocateBuffer((long) s[0] * s[2] * Float.BYTES);
        try {
            long records = (long) s[2] * plan.config().rowRecords(s[1]);
            Object[] quant = {s[2], s[1], tk, s[1], 0, x, q};
            Object[] args = plan.args(w, q, y);
            return timed(() -> {
                queue.enqueueKernelNoEvent(kernels.get(GemmKernels.QUANT_MMQ_Q8_1),
                        new long[]{(records + 255) / 256 * 256}, new long[]{256}, quant);
                queue.enqueueKernelNoEvent(kernels.get(plan.kernelName()), plan.globalWorkSize(),
                        plan.localWorkSize(), args);
            });
        } finally {
            w.close();
            x.close();
            q.close();
            y.close();
        }
    }

    /** Mean milliseconds of one staged MMQ split-K launch: the quantize pass, the slices and the reduce. */
    private double timeMmqSplit(SplitKGemmPlan plan, MmqConfig config, int[] s) {
        int tk = config.stageDepth();
        GpuBuffer w = backend.allocateBuffer(plan.weightBytes(), true);
        GpuBuffer x = backend.allocateBuffer((long) s[1] * s[2] * Float.BYTES, true);
        GpuBuffer q = backend.allocateBuffer(plan.xElementsRequired() * Integer.BYTES);
        GpuBuffer p = backend.allocateBuffer(plan.partialElements() * Float.BYTES);
        GpuBuffer y = backend.allocateBuffer((long) s[0] * s[2] * Float.BYTES);
        try {
            long records = (long) s[2] * config.rowRecords(s[1]);
            Object[] quant = {s[2], s[1], tk, s[1], 0, x, q};
            Object[] partial = plan.partialArgs(w, q, p);
            Object[] reduce = plan.reduceArgs(p, y, s[0], 0);
            return timed(() -> {
                queue.enqueueKernelNoEvent(kernels.get(GemmKernels.QUANT_MMQ_Q8_1),
                        new long[]{(records + 255) / 256 * 256}, new long[]{256}, quant);
                queue.enqueueKernelNoEvent(kernels.get(plan.kernelName()), plan.partialGlobalWorkSize(),
                        plan.partialLocalWorkSize(), partial);
                queue.enqueueKernelNoEvent(kernels.get(GemmKernels.SPLIT_K_REDUCE), plan.reduceGlobalWorkSize(),
                        plan.reduceLocalWorkSize(), reduce);
            });
        } finally {
            w.close();
            x.close();
            q.close();
            p.close();
            y.close();
        }
    }

    private double timed(Runnable launch) {
        for (int i = 0; i < WARMUP; i++) {
            launch.run();
        }
        queue.finish();
        long t0 = System.nanoTime();
        for (int i = 0; i < RUNS; i++) {
            launch.run();
        }
        queue.finish();
        return (System.nanoTime() - t0) / 1e6 / RUNS;
    }
}

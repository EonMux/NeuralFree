package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.KernelTiming;
import com.eonmux.neuralfree.acceleration.backend.cuda.CudaComputeBackend;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.ops.gemm.GemmKernels;
import com.eonmux.neuralfree.gguf.ops.gemm.MatVecMmaPlan;
import com.eonmux.neuralfree.gguf.ops.gemm.MmqConfig;
import com.eonmux.neuralfree.gguf.ops.gemm.MmqGemmPlan;
import com.eonmux.neuralfree.gguf.ops.gemm.SplitKGemmPlan;
import com.eonmux.neuralfree.gguf.ops.gemm.StreamKGemmPlan;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.EnumSet;
import java.util.List;
import java.util.Locale;

/**
 * Opt-in device-time measurement ({@code -Dneuralfree.test.mvm.measure=true}) of the decode-batch projection
 * lanes on CUDA, one harness for every lane: the counted integer-dot matvec ({@code {T}_MatVecN<n>_Q8}), its
 * two-row twin ({@code {T}_MatVec2N<n>_Q8}), the tensor-core matvec ({@code {T}_MatVecMma{8,16}_Q8}) at 2, 4 and
 * 8 warps, the MMQ 16-token twin ({@code {T}_MatMul_MMQ_S}, alone and split along K as the dispatch splits it,
 * {@code {T}_MatMul_MMQ_SKSplit} with the reduce) and the stream-k small-token tensor tile
 * ({@code {T}_MatMul_MMQ_StreamK_S} with its fixup) at 1 to 4 blocks per compute unit and at one block per tile,
 * per weight type, projection shape of the 4B and 0.6B Qwen3 decoders and token count. Each launch reads one of
 * several copies of the weight, rotating over more bytes than the L2 holds, so every launch streams its weight
 * from DRAM as a decode step does; times are the per-launch device times of the queue's event profile (every
 * dispatch of the lane's launch summed, median over the launches). The activation staging each lane reads (the
 * Q8_1 rows of the matvecs, the chunk records of the MMQ lanes) is staged once per token count and timed on its
 * own line, since a decode step shares it between the projections of one input. The printed GB/s is the weight
 * bytes over the lane's time. Asserts nothing; the numbers set {@code MatVecMmaTuning}'s rules and the CUDA
 * entries of {@code SmallBatchCrossover}.
 */
public class MatVecMmaMeasureTest {

    public static final String MEASURE_PROPERTY = "neuralfree.test.mvm.measure";
    /** Comma-separated weight types to measure (default Q8_0,Q4_K,Q6_K). */
    public static final String TYPES_PROPERTY = "neuralfree.test.mvm.types";
    /** Comma-separated token counts (default 2,4,8,9,16). */
    public static final String COUNTS_PROPERTY = "neuralfree.test.mvm.counts";
    /** Comma-separated lane groups to measure: {@code matvec} (counted, paired, tensor-core) and {@code mmq}
     *  (the 16-token twin and the stream-k small tile); default both. */
    public static final String LANES_PROPERTY = "neuralfree.test.mvm.lanes";
    private static final long ROTATION_BYTES = 256L << 20;
    /** Resident blocks per compute unit the stream-k small tile is timed at. */
    private static final int[] STREAM_K_BLOCKS_PER_UNIT = {1, 2, 3, 4};
    private static final int LAUNCHES = 24;
    private static final int[][] SHAPES = {
        {4096, 2560}, {1024, 2560}, {2560, 4096}, {9728, 2560}, {2560, 9728}, {151936, 2560},
        {2048, 1024}, {1024, 1024}, {1024, 2048}, {3072, 1024}, {1024, 3072}, {151936, 1024},
    };

    private HWAcceleration hwAccel;
    private ComputeBackend backend;
    private LlmKernels kernels;
    private int local;
    private int units;
    private String previousProfile;

    @Before
    public void openDevice() {
        Assume.assumeTrue("set " + MEASURE_PROPERTY + "=true to run", Boolean.getBoolean(MEASURE_PROPERTY));
        previousProfile = System.getProperty(KernelTiming.PROFILE_PROPERTY);
        System.setProperty(KernelTiming.PROFILE_PROPERTY, "true");
        HWAcceleration accel = new HWAcceleration(EnumSet.of(ComputeBackend.BackendType.CUDA));
        hwAccel = accel;
        Assume.assumeFalse("no CUDA device", accel.getCUDevices().isEmpty());
        backend = new CudaComputeBackend(accel.getCUDevices().get(0));
        kernels = LlmKernels.open(backend);
        local = kernels.matvecLocal();
        units = backend.capabilities().getComputeUnits();
    }

    @After
    public void closeDevice() {
        if (backend != null) {
            backend.close();
            backend = null;
        }
        if (hwAccel != null) {
            hwAccel.shutdown();
            hwAccel = null;
        }
        if (previousProfile == null) {
            System.clearProperty(KernelTiming.PROFILE_PROPERTY);
        } else {
            System.setProperty(KernelTiming.PROFILE_PROPERTY, previousProfile);
        }
    }

    @Test
    public void lanesAtDecodeBatchCounts() throws InterruptedException {
        System.out.println("MVM-MEASURE device " + backend.capabilities().getDeviceName() + ", " + units
                + " compute units, matvec local " + local);
        for (String t : System.getProperty(TYPES_PROPERTY, "Q8_0,Q4_K,Q6_K").split(",")) {
            GGMLType type = GGMLType.valueOf(t.trim());
            for (int[] shape : SHAPES) {
                if (MatVecMmaPlan.serves(type, shape[1]) || GemmKernels.isInt8Mmq(type)) {
                    measure(type, shape[0], shape[1]);
                }
            }
        }
    }

    private void measure(GGMLType type, int rows, int cols) throws InterruptedException {
        long rowBytes = (long) cols / type.getBlockSize() * type.getTypeSize();
        long bytes = rowBytes * rows;
        int copies = (int) Math.max(2, Math.min(64, (ROTATION_BYTES + bytes - 1) / bytes));
        byte[] w = MatVecMmaGpuTest.weights(type, rows, (int) rowBytes, 0xBEEF + type.ordinal());
        List<GpuBuffer> wBufs = new ArrayList<>();
        int maxN = MatVecMmaPlan.MAX_TOKENS;
        GpuBuffer xBuf = backend.allocateBuffer((long) maxN * cols * 4);
        GpuBuffer qBuf = backend.allocateBuffer((long) maxN * (cols / 32) * GpuOps.Q8_1_FLOATS * 4);
        GpuBuffer yBuf = backend.allocateBuffer((long) maxN * rows * 4);
        MmqConfig small = MmqConfig.INT8_TENSOR_SMALL;
        long rowRecords = small.rowRecords(cols);
        GpuBuffer rBuf = backend.allocateBuffer((long) maxN * rowRecords * small.recordUnits() * 4);
        long tiles = (long) (rows + small.tileRows() - 1) / small.tileRows() * ((maxN + 15) / 16);
        long maxBlocks = Math.max(tiles, (long) units * STREAM_K_BLOCKS_PER_UNIT[STREAM_K_BLOCKS_PER_UNIT.length - 1]);
        GpuBuffer fBuf = backend.allocateBuffer(maxBlocks * small.tileRows() * small.tileTokens() * 4);
        boolean matvecLanes = lanes().contains("matvec");
        boolean mmqLanes = lanes().contains("mmq") && GemmKernels.isInt8Mmq(type);
        try {
            for (int c = 0; c < copies; c++) {
                GpuBuffer b = backend.allocateBuffer(bytes);
                backend.computeQueue().enqueueWriteBuffer(b, 0, w.length, w).awaitCompletion();
                wBufs.add(b);
            }
            float[] x = MatVecMmaGpuTest.activations(maxN * cols);
            backend.computeQueue().enqueueWriteBuffer(xBuf, 0, (long) x.length * 4, x).awaitCompletion();
            backend.computeQueue().enqueueKernel(kernels.get(GpuOps.QUANTIZE_Q8_1),
                    new long[]{roundUp((long) maxN * (cols / 32))}, new long[]{local},
                    new Object[]{maxN * cols, xBuf, qBuf});
            if (mmqLanes) {
                backend.computeQueue().enqueueKernel(kernels.get(GemmKernels.QUANT_MMQ_Q8_1),
                        new long[]{roundUp(maxN * rowRecords)}, new long[]{local},
                        new Object[]{maxN, cols, small.stageDepth(), cols, 0, xBuf, rBuf});
            }
            backend.computeQueue().finish();
            for (String c : System.getProperty(COUNTS_PROPERTY, "2,4,8,9,16").split(",")) {
                int n = Integer.parseInt(c.trim());
                StringBuilder line = new StringBuilder(String.format(Locale.ROOT,
                        "MVM %-5s %6dx%-5d n=%2d floor %7.1f MB", type, rows, cols, n, bytes / 1e6));
                if (matvecLanes) {
                    matvecLanes(line, type, rows, cols, n, wBufs, qBuf, yBuf, bytes);
                }
                if (mmqLanes) {
                    mmqLanes(line, type, rows, cols, n, wBufs, rBuf, yBuf, fBuf, bytes);
                }
                System.out.println(line);
            }
            stagingLine(type, cols, xBuf, qBuf, rBuf);
        } finally {
            for (GpuBuffer b : wBufs) {
                b.close();
            }
            xBuf.close();
            qBuf.close();
            yBuf.close();
            rBuf.close();
            fBuf.close();
        }
    }

    private static String lanes() {
        return System.getProperty(LANES_PROPERTY, "matvec,mmq");
    }

    /** The counted, paired and tensor-core matvec lanes at {@code n} tokens over the staged Q8_1 rows. */
    private void matvecLanes(StringBuilder line, GGMLType type, int rows, int cols, int n, List<GpuBuffer> wBufs,
                             GpuBuffer qBuf, GpuBuffer yBuf, long bytes) throws InterruptedException {
        lane(line, "counted", type.name() + "_MatVecN" + n + "_Q8", (long) rows * local, local,
                wBufs, i -> new Object[]{rows, cols, wBufs.get(i), qBuf, yBuf}, bytes);
        lane(line, "paired", type.name() + "_MatVec2N" + n + "_Q8", (long) ((rows + 1) / 2) * local,
                local, wBufs, i -> new Object[]{rows, cols, wBufs.get(i), qBuf, yBuf}, bytes);
        if (!MatVecMmaPlan.serves(type, cols) || n < 2) {
            return;
        }
        int tile = MatVecMmaPlan.tileTokens(n);
        String kernel = MatVecMmaPlan.kernelName(type, tile);
        int tileRows = MatVecMmaPlan.tileRows(type, tile);
        for (int warps = 2; warps <= MatVecMmaPlan.MAX_WARPS; warps *= 2) {
            long groups = (rows + tileRows - 1) / tileRows;
            lane(line, "mma" + warps, kernel, groups * 32L * warps, 32L * warps, wBufs,
                    i -> new Object[]{rows, cols, n, wBufs.get(i), qBuf, yBuf}, bytes);
        }
    }

    /**
     * The MMQ lanes at {@code n} tokens over the staged chunk records: the 16-token twin of the device tile and
     * the stream-k small tensor tile with its fixup, at each resident block count and at one block per tile.
     */
    private void mmqLanes(StringBuilder line, GGMLType type, int rows, int cols, int n, List<GpuBuffer> wBufs,
                          GpuBuffer rBuf, GpuBuffer yBuf, GpuBuffer fBuf, long bytes) throws InterruptedException {
        MmqConfig device = MmqConfig.INT8_TENSOR;
        int rowRecords = device.rowRecords(cols);
        MmqGemmPlan twin = MmqGemmPlan.smallOf(device, type, rows, cols, n, rowRecords, 0, rows, 0);
        lane(line, "mmq16", twin.kernelName(), twin.globalWorkSize(), twin.localWorkSize(), wBufs,
                i -> twin.args(wBufs.get(i), rBuf, yBuf), bytes);
        SplitKGemmPlan split = SplitKGemmPlan.ofSmallTiles(
                MmqGemmPlan.of(device, type, rows, cols, n, rowRecords, 0, rows, 0), units, local);
        if (split != null && kernels.has(split.kernelName())) {
            GpuBuffer partial = backend.allocateBuffer(split.partialElements() * Float.BYTES);
            try {
                timed(line, "mmq16k" + split.splits(), wBufs.size(), i -> {
                    backend.computeQueue().enqueueKernel(kernels.get(split.kernelName()),
                            split.partialGlobalWorkSize(), split.partialLocalWorkSize(),
                            split.partialArgs(wBufs.get(i), rBuf, partial));
                    backend.computeQueue().enqueueKernel(kernels.get(GemmKernels.SPLIT_K_REDUCE),
                            split.reduceGlobalWorkSize(), split.reduceLocalWorkSize(),
                            split.reduceArgs(partial, yBuf, rows, 0));
                }, bytes);
            } finally {
                partial.close();
            }
        }
        MmqGemmPlan base = MmqGemmPlan.tensorSmallOf(type, rows, cols, n, rowRecords, 0, rows, 0);
        long steps = (long) base.rowTiles() * base.tokenTiles() * (cols / MmqConfig.INT8_TENSOR_SMALL.stageDepth());
        for (int perUnit : STREAM_K_BLOCKS_PER_UNIT) {
            int blocks = (int) Math.min(steps, (long) units * perUnit);
            streamKLane(line, "sk" + perUnit, StreamKGemmPlan.ofBlocks(base, blocks), wBufs, rBuf, yBuf, fBuf,
                    bytes);
        }
        streamKLane(line, "skTiles", StreamKGemmPlan.ofBlocks(base, base.rowTiles() * base.tokenTiles()), wBufs,
                rBuf, yBuf, fBuf, bytes);
    }

    private void streamKLane(StringBuilder line, String name, StreamKGemmPlan plan, List<GpuBuffer> wBufs,
                             GpuBuffer rBuf, GpuBuffer yBuf, GpuBuffer fBuf, long bytes)
            throws InterruptedException {
        if (!kernels.has(plan.kernelName()) || !kernels.has(plan.fixupKernelName())) {
            return;
        }
        Launch launch = i -> {
            backend.computeQueue().enqueueKernel(kernels.get(plan.kernelName()), plan.globalWorkSize(),
                    plan.localWorkSize(), plan.args(wBufs.get(i), rBuf, yBuf, fBuf));
            if (plan.splitsTiles()) {
                backend.computeQueue().enqueueKernel(kernels.get(plan.fixupKernelName()),
                        plan.fixupGlobalWorkSize(), plan.localWorkSize(), plan.fixupArgs(fBuf, yBuf));
            }
        };
        timed(line, name, wBufs.size(), launch, bytes);
    }

    /** The device time of each lane's activation staging at the measured token counts. */
    private void stagingLine(GGMLType type, int cols, GpuBuffer xBuf, GpuBuffer qBuf, GpuBuffer rBuf)
            throws InterruptedException {
        StringBuilder line = new StringBuilder(String.format(Locale.ROOT, "MVM %-5s staging cols=%d", type, cols));
        MmqConfig small = MmqConfig.INT8_TENSOR_SMALL;
        for (String c : System.getProperty(COUNTS_PROPERTY, "2,4,8,9,16").split(",")) {
            int n = Integer.parseInt(c.trim());
            long q8Blocks = (long) n * (cols / 32);
            long records = (long) n * small.rowRecords(cols);
            timed(line, "q8_1@" + n, 1, i -> backend.computeQueue().enqueueKernel(kernels.get(GpuOps.QUANTIZE_Q8_1),
                    new long[]{roundUp(q8Blocks)}, new long[]{local}, new Object[]{n * cols, xBuf, qBuf}), 0);
            timed(line, "mmq@" + n, 1, i -> backend.computeQueue().enqueueKernel(
                    kernels.get(GemmKernels.QUANT_MMQ_Q8_1), new long[]{roundUp(records)}, new long[]{local},
                    new Object[]{n, cols, small.stageDepth(), cols, 0, xBuf, rBuf}), 0);
        }
        System.out.println(line);
    }

    private interface Args {
        Object[] at(int copy);
    }

    /** The dispatches of one launch of a lane over weight copy {@code copy}. */
    private interface Launch {
        void run(int copy);
    }

    /** Appends " name us (GB/s)" for the median device time of LAUNCHES rotating 1-D launches. */
    private void lane(StringBuilder line, String name, String kernel, long global, long localSize,
                      List<GpuBuffer> wBufs, Args args, long bytes) throws InterruptedException {
        lane(line, name, kernel, new long[]{global}, new long[]{localSize}, wBufs, args, bytes);
    }

    private void lane(StringBuilder line, String name, String kernel, long[] global, long[] localSize,
                      List<GpuBuffer> wBufs, Args args, long bytes) throws InterruptedException {
        if (!kernels.has(kernel)) {
            return;
        }
        timed(line, name, wBufs.size(), i -> backend.computeQueue().enqueueKernel(kernels.get(kernel), global,
                localSize, args.at(i)), bytes);
    }

    /** Appends " name us (GB/s)": the median over LAUNCHES launches of the summed device time of their dispatches,
     *  rotating over {@code copies} weight copies (the GB/s is omitted for {@code bytes} 0). */
    private void timed(StringBuilder line, String name, int copies, Launch launch, long bytes)
            throws InterruptedException {
        KernelTiming timing = backend.computeQueue().kernelTiming();
        for (int i = 0; i < 4; i++) {
            launch.run(i % copies);
        }
        backend.computeQueue().finish();
        timing.reset();
        long[] samples = new long[LAUNCHES];
        for (int i = 0; i < LAUNCHES; i++) {
            launch.run(i % copies);
            backend.computeQueue().finish();
            samples[i] = timing.totalNanos();
            timing.reset();
        }
        java.util.Arrays.sort(samples);
        double us = samples[LAUNCHES / 2] / 1e3;
        if (bytes > 0) {
            line.append(String.format(Locale.ROOT, "  %s %7.1f (%4.0f)", name, us, bytes / us / 1e3));
        } else {
            line.append(String.format(Locale.ROOT, "  %s %6.1f", name, us));
        }
    }

    private long roundUp(long n) {
        return (n + local - 1) / local * local;
    }
}

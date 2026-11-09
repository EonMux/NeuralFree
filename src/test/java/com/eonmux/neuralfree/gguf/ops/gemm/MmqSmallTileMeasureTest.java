package com.eonmux.neuralfree.gguf.ops.gemm;

import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.cuda.CudaComputeBackend;
import com.eonmux.neuralfree.acceleration.cuda.CUDevice;
import com.eonmux.neuralfree.acceleration.kernels.Availability;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.quant.BlockLayout;
import com.eonmux.neuralfree.gguf.quant.FpBlockDecoders;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.EnumSet;
import java.util.List;
import java.util.Locale;
import java.util.Random;

/**
 * Opt-in kernel-level timing of the staged MMQ GEMM tiles against the counted integer-dot matvecs at
 * the small-batch row counts ({@code -Dneuralfree.test.mmq.tile.measure=true}), on the CUDA device
 * named by {@code -Dneuralfree.test.mmq.tile.measure.device} (index). Per weight type, projection
 * shape and token count it interleaves rounds of the whole per-projection cost each form pays
 * (activation staging included) and prints the median of each form with its effective weight
 * stream. Asserts nothing; the numbers set the tile choice and the {@code SmallBatchCrossover}
 * entries.
 */
public class MmqSmallTileMeasureTest {

    /** System property that turns the measurement on. */
    public static final String MEASURE_PROPERTY = "neuralfree.test.mmq.tile.measure";
    /** System property that selects the CUDA device by index. */
    public static final String DEVICE_PROPERTY = "neuralfree.test.mmq.tile.measure.device";
    private static final int WARMUP = 20;
    /** Interleaved rounds per comparison; the median round is reported. */
    private static final int ROUNDS = 9;
    private static final int RUNS_PER_ROUND = 10;
    /** 4B decoder shapes: qkv, FFN out, FFN in, 0.6B-class FFN in, the head. */
    private static final int[][] SHAPES = {
        {4096, 2560}, {9728, 2560}, {2560, 9728}, {1024, 1024}, {151936, 2560},
    };
    private static final GGMLType[] TYPES = {GGMLType.Q8_0, GGMLType.Q4_K};
    private static final int[] SPLITS = {2, 4, 8};
    /** Token counts the lanes are timed at: the counted-matvec band and the mid batches up to the
     *  plain tile's width (the 17..128 band the tile ladder must serve). */
    private static final int[] TOKEN_COUNTS = {9, 12, 16, 17, 20, 24, 28, 32, 48, 64, 96};

    private HWAcceleration hwAccel;
    private CudaComputeBackend backend;
    private LlmKernels kernels;
    private int local;
    private MmqConfig config;

    @Before
    public void openDevice() {
        Assume.assumeTrue("set " + MEASURE_PROPERTY + "=true to run", Boolean.getBoolean(MEASURE_PROPERTY));
        HWAcceleration.enableCuda = true;
        try {
            HWAcceleration accel = new HWAcceleration(EnumSet.of(ComputeBackend.BackendType.CUDA));
            Assume.assumeTrue("no CUDA device", !accel.getCUDevices().isEmpty());
            int index = Math.max(0, Integer.getInteger(DEVICE_PROPERTY, 0));
            Assume.assumeTrue("no CUDA device " + index, index < accel.getCUDevices().size());
            CUDevice device = accel.getCUDevices().get(index);
            CudaComputeBackend b = new CudaComputeBackend(device);
            Availability availability = LlmKernels.availability(b);
            Assume.assumeTrue("LLM kernels unavailable: " + availability.reason(), availability.available());
            hwAccel = accel;
            backend = b;
            kernels = LlmKernels.open(b);
            local = kernels.matvecLocal();
            config = MmqConfig.forDevice(ComputeBackend.BackendType.CUDA,
                    b.capabilities().getComputeCapability());
        } catch (Throwable e) {
            closeDevice();
            Assume.assumeTrue("CUDA init failed: " + e.getMessage(), false);
        }
    }

    @After
    public void closeDevice() {
        if (backend != null) {
            try {
                backend.close();
            } catch (RuntimeException ignored) {
            }
            backend = null;
        }
        if (hwAccel != null) {
            try {
                hwAccel.shutdown();
            } catch (RuntimeException ignored) {
            }
            hwAccel = null;
        }
        HWAcceleration.enableCuda = false;
    }

    /** One named timed form: the kernel launches of one lane at one token count. */
    private static final class Lane {
        final String name;
        final long weightBytes;
        final Runnable launches;

        Lane(String name, long weightBytes, Runnable launches) {
            this.name = name;
            this.weightBytes = weightBytes;
            this.launches = launches;
        }
    }

    @Test
    public void tilesAgainstCountedMatvecs() throws Exception {
        System.out.println("device " + backend.capabilities().getDeviceName() + ", compute units "
                + backend.capabilities().getComputeUnits() + ", matvec local " + local + ", mmq " + config);
        for (GGMLType type : TYPES) {
            for (int[] shape : SHAPES) {
                measure(type, shape[0], shape[1]);
            }
        }
    }

    private void measure(GGMLType type, int rows, int cols) throws Exception {
        int rowBytes = cols / type.getBlockSize() * type.getTypeSize();
        byte[] weights = weights(type, rows, rowBytes);
        float[] x = activations(cols);
        int maxTokens = TOKEN_COUNTS[TOKEN_COUNTS.length - 1];
        GpuBuffer wBuf = null;
        GpuBuffer xBuf = null;
        GpuBuffer yBuf = null;
        GpuBuffer qBuf = null;
        GpuBuffer staged = null;
        GpuBuffer partial = null;
        try {
            wBuf = backend.allocateBuffer((long) rows * rowBytes);
            xBuf = backend.allocateBuffer((long) maxTokens * cols * 4L);
            yBuf = backend.allocateBuffer((long) maxTokens * rows * 4L);
            qBuf = backend.allocateBuffer((long) maxTokens * (cols / 32) * BlockLayout.Q8_1_BYTES);
            long records = (long) maxTokens * config.rowRecords(cols);
            staged = backend.allocateBuffer(records * config.recordUnits() * 4L);
            partial = backend.allocateBuffer((long) SPLITS[SPLITS.length - 1] * rows * maxTokens * 4L);
            backend.computeQueue().enqueueWriteBuffer(wBuf, 0, weights.length, weights).awaitCompletion();
            backend.computeQueue().enqueueWriteBuffer(xBuf, 0, (long) x.length * 4, x).awaitCompletion();
            long weightBytes = (long) rows * rowBytes;
            for (int n : TOKEN_COUNTS) {
                measureTokenCount(type, rows, cols, n, weightBytes, wBuf, xBuf, yBuf, qBuf, staged, partial);
            }
        } finally {
            closeQuietly(wBuf, xBuf, yBuf, qBuf, staged, partial);
        }
    }

    private void measureTokenCount(GGMLType type, int rows, int cols, int n, long weightBytes,
                                   GpuBuffer wBuf, GpuBuffer xBuf, GpuBuffer yBuf, GpuBuffer qBuf,
                                   GpuBuffer staged, GpuBuffer partial) {
        int tk = config.stageDepth();
        int rowRecords = config.rowRecords(cols);
        long records = (long) n * rowRecords;
        Runnable stage = () -> backend.computeQueue().enqueueKernel(kernels.get(GemmKernels.QUANT_MMQ_Q8_1),
                new long[]{roundUp(records)}, new long[]{local},
                new Object[]{n, cols, tk, cols, 0, xBuf, staged});
        List<Lane> lanes = new ArrayList<>();
        MmqGemmPlan plain = MmqGemmPlan.contiguous(config, type, rows, cols, n);
        if (kernels.has(plain.kernelName())) {
            lanes.add(new Lane("mmq" + config.tileTokens(), weightBytes, () -> {
                stage.run();
                backend.computeQueue().enqueueKernel(kernels.get(plain.kernelName()), plain.globalWorkSize(),
                        plain.localWorkSize(), plain.args(wBuf, staged, yBuf));
            }));
        }
        if (config.tileTokens() > MmqConfig.TOKEN_LANES) {
            int twinTokens = MmqConfig.twinTokens(n, cols);
            if (twinTokens < config.tileTokens()) {
                MmqGemmPlan small = MmqGemmPlan.smallOf(config, type, twinTokens, rows, cols, n, rowRecords, 0,
                        rows, 0);
                if (kernels.has(small.kernelName())) {
                    lanes.add(new Lane("mmq" + twinTokens, weightBytes, () -> {
                        stage.run();
                        backend.computeQueue().enqueueKernel(kernels.get(small.kernelName()),
                                small.globalWorkSize(), small.localWorkSize(), small.args(wBuf, staged, yBuf));
                    }));
                }
            }
        }
        int rowTiles = (rows + config.tileRows() - 1) / config.tileRows();
        for (int splits : SPLITS) {
            if (splits > 1 && (long) rowTiles * splits > 4096L) {
                continue;
            }
            SplitKGemmPlan split = SplitKGemmPlan.ofMmq(type, rows, cols, n, rowRecords, 0, splits,
                    SplitKGemmPlan.sliceLength(cols, splits, config.stageDepth()), config, false, local);
            if (!kernels.has(split.kernelName()) || !kernels.has(GemmKernels.SPLIT_K_REDUCE)) {
                continue;
            }
            lanes.add(new Lane("ksplit" + splits, weightBytes, () -> {
                stage.run();
                backend.computeQueue().enqueueKernelNoEvent(kernels.get(split.kernelName()),
                        split.partialGlobalWorkSize(), split.partialLocalWorkSize(),
                        split.partialArgs(wBuf, staged, partial));
                backend.computeQueue().enqueueKernelNoEvent(kernels.get(GemmKernels.SPLIT_K_REDUCE),
                        split.reduceGlobalWorkSize(), split.reduceLocalWorkSize(),
                        split.reduceArgs(partial, yBuf, rows, 0));
            }));
        }
        if (config.tileTokens() > MmqConfig.TOKEN_LANES) {
            int twinTokens = MmqConfig.twinTokens(n, cols);
            if (twinTokens < config.tileTokens()) {
                MmqConfig twin = MmqConfig.smallTile(config, twinTokens);
                int twinRowTiles = (rows + twin.tileRows() - 1) / twin.tileRows();
                int twinTokenTiles = (n + twin.tileTokens() - 1) / twin.tileTokens();
                for (int splits : SPLITS) {
                    if (splits > 1 && (long) twinRowTiles * twinTokenTiles * splits > 4096L) {
                        continue;
                    }
                    SplitKGemmPlan split = SplitKGemmPlan.ofMmq(type, rows, cols, n, rowRecords, 0, splits,
                            SplitKGemmPlan.sliceLength(cols, splits, twin.stageDepth()), twin, true, local);
                    if (!kernels.has(split.kernelName())) {
                        continue;
                    }
                    lanes.add(new Lane("sksplit" + twinTokens + "/" + splits, weightBytes, () -> {
                        stage.run();
                        backend.computeQueue().enqueueKernelNoEvent(kernels.get(split.kernelName()),
                                split.partialGlobalWorkSize(), split.partialLocalWorkSize(),
                                split.partialArgs(wBuf, staged, partial));
                        backend.computeQueue().enqueueKernelNoEvent(kernels.get(GemmKernels.SPLIT_K_REDUCE),
                                split.reduceGlobalWorkSize(), split.reduceLocalWorkSize(),
                                split.reduceArgs(partial, yBuf, rows, 0));
                    }));
                }
            }
        }
        String runtime = type.name() + "_MatVecN_Q8";
        if (kernels.has(runtime)) {
            lanes.add(new Lane("runtime", weightBytes, () -> {
                stageCounted(n, cols, xBuf, qBuf);
                backend.computeQueue().enqueueKernel(kernels.get(runtime), new long[]{(long) rows * local},
                        new long[]{local}, new Object[]{rows, cols, n, wBuf, qBuf, yBuf});
            }));
        }
        String counted = type.name() + "_MatVecN" + n + "_Q8";
        String paired = type.name() + "_MatVec2N" + n + "_Q8";
        if (kernels.has(counted)) {
            lanes.add(new Lane("counted", weightBytes, () -> {
                stageCounted(n, cols, xBuf, qBuf);
                backend.computeQueue().enqueueKernel(kernels.get(counted), new long[]{(long) rows * local},
                        new long[]{local}, new Object[]{rows, cols, wBuf, qBuf, yBuf});
            }));
        }
        if (kernels.has(paired)) {
            long groups = ((long) rows + 1) / 2;
            lanes.add(new Lane("paired", weightBytes, () -> {
                stageCounted(n, cols, xBuf, qBuf);
                backend.computeQueue().enqueueKernel(kernels.get(paired), new long[]{groups * local},
                        new long[]{local}, new Object[]{rows, cols, wBuf, qBuf, yBuf});
            }));
        }
        if (lanes.isEmpty()) {
            return;
        }
        double[] medians = timeLanes(lanes);
        StringBuilder line = new StringBuilder(String.format(Locale.ROOT, "%-6s %6dx%-5d n=%2d",
                type, rows, cols, n));
        for (int f = 0; f < lanes.size(); f++) {
            Lane lane = lanes.get(f);
            double gbs = lane.weightBytes / (medians[f] * 1e-6) / 1e9;
            line.append(String.format(Locale.ROOT, "  %s %8.1fus(%4.0fGB/s)", lane.name, medians[f], gbs));
        }
        System.out.println(line.toString());
    }

    /** The Q8_1 rows the counted matvecs read ({@code QUANTIZE_Q8_1}, 32-element blocks). */
    private void stageCounted(int n, int cols, GpuBuffer xBuf, GpuBuffer qBuf) {
        long blocks = (long) n * (cols / 32);
        backend.computeQueue().enqueueKernel(kernels.get(GpuOps.QUANTIZE_Q8_1), new long[]{roundUp(blocks)},
                new long[]{local}, new Object[]{n * cols, xBuf, qBuf});
    }

    private double[] timeLanes(List<Lane> lanes) {
        int forms = lanes.size();
        for (int r = 0; r < WARMUP; r++) {
            for (int f = 0; f < forms; f++) {
                lanes.get((f + r) % forms).launches.run();
            }
        }
        backend.computeQueue().finish();
        long[][] rounds = new long[forms][ROUNDS];
        for (int r = 0; r < ROUNDS; r++) {
            for (int f = 0; f < forms; f++) {
                int index = (f + r) % forms;
                for (int i = 0; i < RUNS_PER_ROUND; i++) {
                    lanes.get(index).launches.run();
                }
                backend.computeQueue().finish();
                long start = System.nanoTime();
                for (int i = 0; i < RUNS_PER_ROUND; i++) {
                    lanes.get(index).launches.run();
                }
                backend.computeQueue().finish();
                rounds[index][r] = (System.nanoTime() - start) / RUNS_PER_ROUND;
            }
        }
        double[] medians = new double[forms];
        for (int f = 0; f < forms; f++) {
            long[] sorted = rounds[f].clone();
            Arrays.sort(sorted);
            medians[f] = sorted[sorted.length / 2] / 1e3;
        }
        return medians;
    }

    private long roundUp(long n) {
        return (n + local - 1) / local * local;
    }

    private static void closeQuietly(GpuBuffer... buffers) {
        for (GpuBuffer b : buffers) {
            if (b != null) {
                try {
                    b.close();
                } catch (RuntimeException ignored) {
                }
            }
        }
    }

    /** Random blocks of {@code type} with finite block scales. */
    private static byte[] weights(GGMLType type, int rows, int rowBytes) {
        Random rnd = new Random(0x51ED + type.ordinal());
        byte[] data = new byte[rows * rowBytes];
        rnd.nextBytes(data);
        int blockBytes = type.getTypeSize();
        for (int off = 0; off + blockBytes <= data.length; off += blockBytes) {
            if (type == GGMLType.Q4_K || type == GGMLType.Q5_K) {
                putHalf(data, off, 0.02f);
                putHalf(data, off + 2, 0.01f);
            } else {
                putHalf(data, off, 0.02f);
            }
        }
        return data;
    }

    private static void putHalf(byte[] data, int off, float value) {
        int bits = FpBlockDecoders.fp32ToFp16(value);
        data[off] = (byte) bits;
        data[off + 1] = (byte) (bits >>> 8);
    }

    private static float[] activations(int n) {
        Random rnd = new Random(0xC0DE);
        float[] x = new float[n];
        for (int i = 0; i < n; i++) {
            x[i] = rnd.nextFloat() * 2f - 1f;
        }
        return x;
    }
}

package com.eonmux.neuralfree.gguf.ops.gemm;

import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.cuda.CudaComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.cuda.CudaKernelHandle;
import com.eonmux.neuralfree.acceleration.cuda.CUDevice;
import com.eonmux.neuralfree.acceleration.kernels.Availability;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
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
 * Opt-in kernel-level timing of the staged MMQ GEMM at prefill token counts
 * ({@code -Dneuralfree.test.mmq.prefill.measure=true}), on the CUDA device named by
 * {@code -Dneuralfree.test.mmq.prefill.measure.device} (index). Per weight type and projection shape
 * it interleaves rounds of the staging pass alone and of the whole staged GEMM launches (plain tile
 * and split-K) at {@value #TOKENS} tokens and prints the median with the achieved TFLOPS and the
 * effective weight stream; the F16 tensor-core MMA GEMM of the same shape marks the device ceiling.
 * Asserts nothing; the numbers set the tile choice.
 */
public class MmqPrefillMeasureTest {

    /** System property that turns the measurement on. */
    public static final String MEASURE_PROPERTY = "neuralfree.test.mmq.prefill.measure";
    /** System property that selects the CUDA device by index. */
    public static final String DEVICE_PROPERTY = "neuralfree.test.mmq.prefill.measure.device";
    private static final int TOKENS = 512;
    private static final int WARMUP = 6;
    /** Interleaved rounds per comparison; the median round is reported. */
    private static final int ROUNDS = 7;
    private static final int RUNS_PER_ROUND = 5;
    /** Launches of the clock warmup, about 15 s of the 772 us mid-size F16 GEMM. */
    private static final int CLOCK_WARMUP_LAUNCHES = 20000;
    /** 4B decoder shapes: qkv, FFN out, FFN in, 0.6B-class FFN in, the head. */
    private static final int[][] SHAPES = {
        {4096, 2560}, {9728, 2560}, {2560, 9728}, {1024, 2560}, {151936, 2560},
    };
    private static final GGMLType[] TYPES = {GGMLType.Q4_K, GGMLType.Q6_K, GGMLType.Q5_K, GGMLType.Q8_0};
    private static final int[] SPLITS = {2, 4, 8};

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

    /** One named timed form: the kernel launches of one lane at the token count. */
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
    public void prefillTilesPerType() throws Exception {
        System.out.println("device " + backend.capabilities().getDeviceName() + ", compute units "
                + backend.capabilities().getComputeUnits() + ", local " + local + ", mmq " + config
                + ", shared " + config.sharedBytes() + "B");
        warmClocks();
        for (GGMLType type : TYPES) {
            for (int[] shape : SHAPES) {
                measure(type, shape[0], shape[1]);
            }
        }
    }

    /**
     * Holds the device busy for about {@value #CLOCK_WARMUP_LAUNCHES} MMA launches of a mid-size F16
     * GEMM so the DVFS governor leaves its idle clock step before the first measurement: the first
     * shapes of a cold run otherwise measure two to thirteen times their boosted time, at the
     * governor's deterministic intermediate steps.
     */
    private void warmClocks() throws Exception {
        int rows = 4096;
        int cols = 2560;
        MmaGemmPlan mma = MmaGemmPlan.of(GGMLType.F16, rows, cols, TOKENS, cols, 0, rows, 0);
        Assume.assumeTrue("no F16 MMA kernel", kernels.has(mma.kernelName()));
        GpuBuffer wF16 = backend.allocateBuffer((long) rows * cols * 2L);
        GpuBuffer xBuf = backend.allocateBuffer((long) TOKENS * cols * 4L);
        GpuBuffer yBuf = backend.allocateBuffer((long) TOKENS * rows * 4L);
        try {
            backend.computeQueue().enqueueWriteBuffer(wF16, 0, wF16.sizeInBytes(), halfWeights(rows, cols))
                    .awaitCompletion();
            for (int i = 0; i < CLOCK_WARMUP_LAUNCHES; i++) {
                backend.computeQueue().enqueueKernel(kernels.get(mma.kernelName()), mma.globalWorkSize(),
                        mma.localWorkSize(), mma.args(wF16, xBuf, yBuf));
                if (i % 512 == 511) {
                    backend.computeQueue().finish();
                }
            }
            backend.computeQueue().finish();
        } finally {
            closeQuietly(wF16, xBuf, yBuf);
        }
    }

    private void measure(GGMLType type, int rows, int cols) throws Exception {
        int rowBytes = cols / type.getBlockSize() * type.getTypeSize();
        byte[] weights = weights(type, rows, rowBytes);
        float[] x = activations(cols);
        GpuBuffer wBuf = null;
        GpuBuffer xBuf = null;
        GpuBuffer yBuf = null;
        GpuBuffer staged = null;
        try {
            wBuf = backend.allocateBuffer((long) rows * rowBytes);
            xBuf = backend.allocateBuffer((long) TOKENS * cols * 4L);
            yBuf = backend.allocateBuffer((long) TOKENS * rows * 4L);
            long records = (long) TOKENS * config.rowRecords(cols);
            staged = backend.allocateBuffer(records * config.recordUnits() * 4L);
            backend.computeQueue().enqueueWriteBuffer(wBuf, 0, weights.length, weights).awaitCompletion();
            backend.computeQueue().enqueueWriteBuffer(xBuf, 0, (long) x.length * 4, x).awaitCompletion();
            long weightBytes = (long) rows * rowBytes;
            measureShape(type, rows, cols, weightBytes, wBuf, xBuf, yBuf, staged);
        } finally {
            closeQuietly(wBuf, xBuf, yBuf, staged);
        }
    }

    private void measureShape(GGMLType type, int rows, int cols, long weightBytes, GpuBuffer wBuf,
                              GpuBuffer xBuf, GpuBuffer yBuf, GpuBuffer staged) throws Exception {
        Runnable stage = stageKernel(cols, xBuf, staged);
        List<GpuBuffer> temps = new ArrayList<>();
        List<Lane> lanes = new ArrayList<>();
        lanes.add(new Lane("stage", 0, stage));
        MmqGemmPlan plain = MmqGemmPlan.contiguous(config, type, rows, cols, TOKENS);
        if (kernels.has(plain.kernelName())) {
            lanes.add(new Lane("mmq" + config.tileTokens(), weightBytes, () -> {
                stage.run();
                backend.computeQueue().enqueueKernel(kernels.get(plain.kernelName()), plain.globalWorkSize(),
                        plain.localWorkSize(), plain.args(wBuf, staged, yBuf));
            }));
        }
        lanes.addAll(splitLanes(type, rows, cols, weightBytes, wBuf, staged, yBuf, stage, temps));
        Lane f16 = f16Lane(rows, cols, xBuf, yBuf, temps);
        if (f16 != null) {
            lanes.add(f16);
        }
        printAttributes(plain.kernelName());
        double[] medians = timeLanes(lanes);
        for (int f = 0; f < lanes.size(); f++) {
            Lane lane = lanes.get(f);
            double us = medians[f];
            String line;
            if (lane.weightBytes == 0) {
                line = String.format(Locale.ROOT, "%-5s %6dx%-5d n=%3d  %-9s %9.1fus", type, rows, cols, TOKENS,
                        lane.name, us);
            } else {
                double tflops = 2.0 * rows * cols * TOKENS / (us * 1e-6) / 1e12;
                double gbs = lane.weightBytes / (us * 1e-6) / 1e9;
                line = String.format(Locale.ROOT, "%-5s %6dx%-5d n=%3d  %-9s %9.1fus(%5.2f TFLOPS, %6.0fGB/s)",
                        type, rows, cols, TOKENS, lane.name, us, tflops, gbs);
            }
            System.out.println(line);
        }
        closeQuietly(temps.toArray(new GpuBuffer[0]));
    }

    /** The staging pass of {@code TOKENS} activation rows into the record layout. */
    private Runnable stageKernel(int cols, GpuBuffer xBuf, GpuBuffer staged) {
        int tk = config.stageDepth();
        int rowRecords = config.rowRecords(cols);
        long records = (long) TOKENS * rowRecords;
        return () -> backend.computeQueue().enqueueKernel(kernels.get(GemmKernels.QUANT_MMQ_Q8_1),
                new long[]{roundUp(records)}, new long[]{local},
                new Object[]{TOKENS, cols, tk, cols, 0, xBuf, staged});
    }

    /** The split-K lanes of the shape whose grids stay within the 4096 workgroup cap per dimension. */
    private List<Lane> splitLanes(GGMLType type, int rows, int cols, long weightBytes, GpuBuffer wBuf,
                                  GpuBuffer staged, GpuBuffer yBuf, Runnable stage, List<GpuBuffer> temps) {
        List<Lane> lanes = new ArrayList<>();
        int rowRecords = config.rowRecords(cols);
        int rowTiles = (rows + config.tileRows() - 1) / config.tileRows();
        for (int splits : SPLITS) {
            if ((long) rowTiles * splits > 4096L) {
                continue;
            }
            SplitKGemmPlan split = SplitKGemmPlan.ofMmq(type, rows, cols, TOKENS, rowRecords, 0, splits,
                    SplitKGemmPlan.sliceLength(cols, splits, config.stageDepth()), config, false, local);
            if (!kernels.has(split.kernelName()) || !kernels.has(GemmKernels.SPLIT_K_REDUCE)) {
                continue;
            }
            GpuBuffer pBuf = backend.allocateBuffer((long) splits * rows * TOKENS * 4L);
            temps.add(pBuf);
            lanes.add(new Lane("ksplit" + splits, weightBytes, () -> {
                stage.run();
                backend.computeQueue().enqueueKernelNoEvent(kernels.get(split.kernelName()),
                        split.partialGlobalWorkSize(), split.partialLocalWorkSize(),
                        split.partialArgs(wBuf, staged, pBuf));
                backend.computeQueue().enqueueKernelNoEvent(kernels.get(GemmKernels.SPLIT_K_REDUCE),
                        split.reduceGlobalWorkSize(), split.reduceLocalWorkSize(),
                        split.reduceArgs(pBuf, yBuf, rows, 0));
            }));
        }
        return lanes;
    }

    /** The F16 tensor-core MMA lane of the same shape, the device ceiling marker. */
    private Lane f16Lane(int rows, int cols, GpuBuffer xBuf, GpuBuffer yBuf, List<GpuBuffer> temps)
            throws Exception {
        MmaGemmPlan mma = MmaGemmPlan.of(GGMLType.F16, rows, cols, TOKENS, cols, 0, rows, 0);
        if (!kernels.has(mma.kernelName())) {
            return null;
        }
        GpuBuffer wF16 = backend.allocateBuffer((long) rows * cols * 2L);
        temps.add(wF16);
        backend.computeQueue().enqueueWriteBuffer(wF16, 0, wF16.sizeInBytes(), halfWeights(rows, cols))
                .awaitCompletion();
        return new Lane("f16mma", (long) rows * cols * 2L, () -> backend.computeQueue()
                .enqueueKernel(kernels.get(mma.kernelName()), mma.globalWorkSize(), mma.localWorkSize(),
                        mma.args(wF16, xBuf, yBuf)));
    }

    /** The launch attributes of one fast kernel: static local memory and largest block. */
    private void printAttributes(String kernelName) {
        if (!kernels.has(kernelName)) {
            return;
        }
        try {
            CudaKernelHandle handle = (CudaKernelHandle) kernels.get(kernelName);
            System.out.println(String.format(Locale.ROOT, "%s: shared %d B, max threads/block %d",
                    kernelName, handle.staticSharedMemoryBytes(), handle.maxThreadsPerBlock()));
        } catch (RuntimeException ignored) {
            /* non-CUDA handle: nothing to print */
        }
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

    /** Random fp16 weights with small finite magnitudes, little-endian halves. */
    private static byte[] halfWeights(int rows, int cols) {
        Random rnd = new Random(0xF16D);
        byte[] data = new byte[rows * cols * 2];
        for (int i = 0; i < data.length / 2; i++) {
            putHalf(data, 2 * i, rnd.nextFloat() * 0.04f - 0.02f);
        }
        return data;
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

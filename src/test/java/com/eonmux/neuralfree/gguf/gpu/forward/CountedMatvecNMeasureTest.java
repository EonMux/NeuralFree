package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.cuda.CudaComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ocl.OclComputeBackend;
import com.eonmux.neuralfree.acceleration.cuda.CUDevice;
import com.eonmux.neuralfree.acceleration.kernels.Availability;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.kernels.LlmKernelConstants;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.ops.gemm.GemmKernels;
import com.eonmux.neuralfree.gguf.ops.gemm.MmqConfig;
import com.eonmux.neuralfree.gguf.ops.gemm.MmqGemmPlan;
import com.eonmux.neuralfree.gguf.quant.FpBlockDecoders;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.EnumSet;
import java.util.List;
import java.util.Locale;
import java.util.Random;

/**
 * Opt-in kernel-level timing of the counted multi-column matvec forms against the lanes the small-batch
 * dispatch chooses between ({@code -Dneuralfree.test.counted.measure=true}, device index through
 * {@code -Dneuralfree.test.counted.measure.device}). Per weight type, projection shape and token count
 * n in [2, {@link LlmKernelConstants#MATVEC_N_MAX_TOKENS}] it times, in interleaved rounds of rotating
 * order (a shared GPU holds two clock states and a neighbour process's load shifts between rounds, so
 * same-order rounds would time one form on the slower state; the median round is reported), the whole
 * per-projection cost each lane pays:
 * <ul>
 *   <li>{@code runtime} &mdash; the Q8_1 quantizer plus {@code {T}_MatVecN_Q8} with the runtime token
 *       count (the incumbent multi-column lane);</li>
 *   <li>{@code counted} &mdash; quantizer plus {@code {T}_MatVecN<n>_Q8} at every count the type
 *       instantiates;</li>
 *   <li>{@code paired} &mdash; quantizer plus {@code {T}_MatVec2N<n>_Q8}, two weight rows per
 *       work-group (same counts);</li>
 *   <li>{@code mmq} &mdash; {@code LLM_Quant_MMQ_Q8_1} plus the fast GEMM lane the dispatch serves:
 *       the {@code <T>_MatMul_MMQ_S} 16-token twin for n below the device tile where the device has
 *       one, the plain tile otherwise.</li>
 * </ul>
 * Asserts nothing; the numbers set the per-kind dispatch entries of {@link SmallBatchCrossover} and the
 * counted-lane evidence of {@link Int8MatvecTuning}.
 */
@RunWith(Parameterized.class)
public class CountedMatvecNMeasureTest {

    /** System property that turns the measurement on. */
    public static final String MEASURE_PROPERTY = "neuralfree.test.counted.measure";
    /** System property that selects the backend device to measure by index (CUDA and OpenCL device order). */
    public static final String DEVICE_PROPERTY = "neuralfree.test.counted.measure.device";
    private static final int WARMUP = 20;
    /** Interleaved rounds per comparison; the median round is reported. */
    private static final int ROUNDS = 9;
    private static final int RUNS_PER_ROUND = 10;
    /**
     * A 0.6B-class decoder's FFN projection, the 4B-class projections in both orientations and the
     * output head (rows x cols). Elements below {@link SmallBatchCrossover#SHAPE_MIN_ELEMENTS} are
     * the small size class, the rest the large one.
     */
    private static final int[][] SHAPES = {
        {4096, 1024}, {19456, 2560}, {9728, 2560}, {2560, 9728}, {151936, 2560},
    };
    /** The Q8_0 type plus the main K-quant, one legacy and one i-quant type. */
    private static final GGMLType[] TYPES = {
        GGMLType.Q8_0, GGMLType.Q4_0, GGMLType.Q4_K, GGMLType.Q5_K, GGMLType.Q6_K, GGMLType.IQ4_XS,
    };

    @Parameterized.Parameters(name = "{0}")
    public static List<Object[]> backends() {
        return Arrays.asList(new Object[]{"opencl"}, new Object[]{"cuda"});
    }

    private final String api;
    private HWAcceleration hwAccel;
    private ComputeBackend backend;
    private LlmKernels kernels;
    private int local;
    private MmqConfig mmqConfig;
    private boolean previousCuda;

    public CountedMatvecNMeasureTest(String api) {
        this.api = api;
    }

    @Before
    public void openDevice() {
        Assume.assumeTrue("set " + MEASURE_PROPERTY + "=true to run", Boolean.getBoolean(MEASURE_PROPERTY));
        previousCuda = HWAcceleration.enableCuda;
        try {
            backend = openBackend();
            Availability availability = LlmKernels.availability(backend);
            Assume.assumeTrue("LLM kernels unavailable on " + api + ": " + availability.reason(),
                    availability.available());
            kernels = LlmKernels.open(backend);
            local = kernels.matvecLocal();
            mmqConfig = MmqConfig.forDevice(backend.backendType(), kernels.caps().getComputeCapability());
        } catch (Throwable e) {
            closeDevice();
            Assume.assumeTrue("no usable " + api + " GPU: " + e.getMessage(), false);
        }
    }

    private static int deviceIndex() {
        return Math.max(0, Integer.getInteger(DEVICE_PROPERTY, 0));
    }

    private ComputeBackend openBackend() {
        if (api.equals("cuda")) {
            HWAcceleration.enableCuda = true;
            HWAcceleration accel = new HWAcceleration(EnumSet.of(ComputeBackend.BackendType.CUDA));
            hwAccel = accel;
            Assume.assumeFalse("no CUDA device", accel.getCUDevices().isEmpty());
            int index = deviceIndex();
            Assume.assumeTrue("no CUDA device " + index, index < accel.getCUDevices().size());
            CUDevice device = accel.getCUDevices().get(index);
            return new CudaComputeBackend(device);
        }
        HWAcceleration accel = new HWAcceleration(EnumSet.of(ComputeBackend.BackendType.OPENCL));
        hwAccel = accel;
        Assume.assumeFalse("no OpenCL device", accel.getCLDevices().isEmpty());
        int index = deviceIndex();
        Assume.assumeTrue("no OpenCL device " + index, index < accel.getCLDevices().size());
        return new OclComputeBackend(accel.getCLDevices().get(index));
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
            hwAccel.shutdown();
            hwAccel = null;
        }
        HWAcceleration.enableCuda = previousCuda;
    }

    @Test
    public void countedMatvecFormsAgainstTheDispatchLanes() throws Exception {
        System.out.println("device " + backend.capabilities().getDeviceName() + " (" + api
                + ", index " + deviceIndex() + "), matvec local " + local + ", mmq " + mmqConfig);
        for (GGMLType type : TYPES) {
            for (int[] shape : SHAPES) {
                measure(type, shape[0], shape[1]);
            }
        }
    }

    /** One named timed form: the kernel launches of one projection lane at one token count. */
    private static final class Lane {
        final String name;
        final Runnable launches;

        Lane(String name, Runnable launches) {
            this.name = name;
            this.launches = launches;
        }
    }

    /** Interleaved-round timing of every lane of one type, shape and token count. */
    private void measure(GGMLType type, int rows, int cols) throws Exception {
        int blockSize = type.getBlockSize();
        int rowBytes = cols / blockSize * type.getTypeSize();
        byte[] weights = weights(type, rows, rowBytes);
        float[] x = activations(cols);
        GpuBuffer wBuf = backend.allocateBuffer((long) rows * rowBytes);
        GpuBuffer xBuf = backend.allocateBuffer(
                (long) LlmKernelConstants.MATVEC_N_MAX_TOKENS * cols * 4);
        GpuBuffer qBuf = backend.allocateBuffer(
                (long) LlmKernelConstants.MATVEC_N_MAX_TOKENS * (cols / 32) * GpuOps.Q8_1_FLOATS * 4);
        GpuBuffer yBuf = backend.allocateBuffer(
                (long) LlmKernelConstants.MATVEC_N_MAX_TOKENS * rows * 4);
        GpuBuffer staged = null;
        try {
            backend.computeQueue().enqueueWriteBuffer(wBuf, 0, weights.length, weights).awaitCompletion();
            backend.computeQueue().enqueueWriteBuffer(xBuf, 0, (long) x.length * 4, x).awaitCompletion();
            staged = allocateStaged(cols);
            for (int n = 2; n <= LlmKernelConstants.MATVEC_N_MAX_TOKENS; n++) {
                measureTokenCount(type, rows, cols, n, wBuf, xBuf, qBuf, yBuf, staged);
            }
        } finally {
            wBuf.close();
            xBuf.close();
            qBuf.close();
            yBuf.close();
            if (staged != null) {
                staged.close();
            }
        }
    }

    /** The staged-record buffer of one projection ({@code LLM_Quant_MMQ_Q8_1}'s output at 16 tokens), or null. */
    private GpuBuffer allocateStaged(int cols) {
        if (!kernels.has(GemmKernels.QUANT_MMQ_Q8_1) || !kernels.has(GemmKernels.SPLIT_K_REDUCE)) {
            return null;
        }
        long records = (long) LlmKernelConstants.MATVEC_N_MAX_TOKENS * mmqConfig.rowRecords(cols);
        return backend.allocateBuffer(records * mmqConfig.recordUnits() * 4L);
    }

    /** Lanes and their interleaved-round medians at one token count; one printed line. */
    private void measureTokenCount(GGMLType type, int rows, int cols, int n, GpuBuffer wBuf, GpuBuffer xBuf,
                                   GpuBuffer qBuf, GpuBuffer yBuf, GpuBuffer staged) {
        final int tokens = n;
        Runnable quantize = () -> backend.computeQueue().enqueueKernel(kernels.get(GpuOps.QUANTIZE_Q8_1),
                new long[]{roundUp((long) tokens * (cols / 32))}, new long[]{local},
                new Object[]{tokens * cols, xBuf, qBuf});
        List<Lane> lanes = new ArrayList<>();
        String runtimeKernel = type.name() + "_MatVecN_Q8";
        if (kernels.has(runtimeKernel)) {
            lanes.add(new Lane("runtime", () -> {
                quantize.run();
                backend.computeQueue().enqueueKernel(kernels.get(runtimeKernel),
                        new long[]{(long) rows * local}, new long[]{local},
                        new Object[]{rows, cols, tokens, wBuf, qBuf, yBuf});
            }));
        }
        if (n <= LlmKernelConstants.MATVEC_N_MAX_TOKENS) {
            String countedKernel = type.name() + "_MatVecN" + n + "_Q8";
            String pairedKernel = type.name() + "_MatVec2N" + n + "_Q8";
            if (kernels.has(countedKernel)) {
                lanes.add(new Lane("counted", () -> {
                    quantize.run();
                    backend.computeQueue().enqueueKernel(kernels.get(countedKernel),
                            new long[]{(long) rows * local}, new long[]{local},
                            new Object[]{rows, cols, wBuf, qBuf, yBuf});
                }));
            }
            if (kernels.has(pairedKernel)) {
                long groups = ((long) rows + 1) / 2;
                lanes.add(new Lane("paired", () -> {
                    quantize.run();
                    backend.computeQueue().enqueueKernel(kernels.get(pairedKernel),
                            new long[]{groups * local}, new long[]{local},
                            new Object[]{rows, cols, wBuf, qBuf, yBuf});
                }));
            }
        }
        if (staged != null) {
            Lane mmq = mmqLane(type, rows, cols, n, wBuf, xBuf, yBuf, staged);
            if (mmq != null) {
                lanes.add(mmq);
            }
        }
        if (lanes.isEmpty()) {
            return;
        }
        double[] medians = timeLanes(lanes);
        StringBuilder line = new StringBuilder(String.format(Locale.ROOT, "%-6s %5dx%-5d n=%2d",
                type, rows, cols, n));
        for (int f = 0; f < lanes.size(); f++) {
            line.append(String.format(Locale.ROOT, "  %s %9.2f", lanes.get(f).name, medians[f]));
        }
        System.out.println(line.append("  us").toString());
    }

    /** The fast GEMM lane the dispatch serves at {@code n}: the small twin below the device tile, else the plain tile. */
    private Lane mmqLane(GGMLType type, int rows, int cols, int n, GpuBuffer wBuf, GpuBuffer xBuf,
                         GpuBuffer yBuf, GpuBuffer staged) {
        MmqConfig tile = mmqConfig;
        String kernel = GemmKernels.matmulMmq(type);
        boolean small = mmqConfig.tileTokens() > MmqConfig.TOKEN_LANES && n < MmqConfig.TOKEN_LANES;
        if (small) {
            tile = MmqConfig.smallTile(mmqConfig);
            kernel = GemmKernels.matmulMmqSmall(type);
        }
        final String kernelName = kernel;
        if (!kernels.has(kernelName)) {
            return null;
        }
        final MmqConfig config = tile;
        final int tk = config.stageDepth();
        final int rowRecords = config.rowRecords(cols);
        final long records = (long) n * rowRecords;
        final MmqGemmPlan plan = small
                ? MmqGemmPlan.smallOf(mmqConfig, type, rows, cols, n, rowRecords, 0, rows, 0)
                : MmqGemmPlan.contiguous(config, type, rows, cols, n);
        return new Lane(small ? "mmqS" : "mmq", () -> {
            backend.computeQueue().enqueueKernel(kernels.get(GemmKernels.QUANT_MMQ_Q8_1),
                    new long[]{roundUp(records)}, new long[]{local},
                    new Object[]{n, cols, tk, cols, 0, xBuf, staged});
            backend.computeQueue().enqueueKernel(kernels.get(kernelName), plan.globalWorkSize(),
                    plan.localWorkSize(), plan.args(wBuf, staged, yBuf));
        });
    }

    /** Median-over-rounds per-lane cost with the lane order rotating every round. */
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

    /** Random blocks of {@code type} with finite block scales. */
    private static byte[] weights(GGMLType type, int rows, int rowBytes) {
        Random rnd = new Random(0x51ED + type.ordinal());
        byte[] data = new byte[rows * rowBytes];
        rnd.nextBytes(data);
        int blockBytes = type.getTypeSize();
        for (int off = 0; off + blockBytes <= data.length; off += blockBytes) {
            switch (type) {
                case Q6_K:
                    putHalf(data, off + 208, 0.01f);
                    break;
                case Q4_K:
                case Q5_K:
                    putHalf(data, off, 0.02f);
                    putHalf(data, off + 2, 0.01f);
                    break;
                default:
                    putHalf(data, off, 0.02f);
                    break;
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

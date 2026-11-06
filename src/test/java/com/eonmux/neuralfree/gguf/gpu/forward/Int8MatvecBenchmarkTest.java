package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.cuda.CudaComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ocl.OclComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.vulkan.VkComputeBackend;
import com.eonmux.neuralfree.acceleration.cuda.CUDevice;
import com.eonmux.neuralfree.acceleration.kernels.Availability;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.quant.FpBlockDecoders;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Random;

/**
 * Opt-in timing of the integer-dot (Q8_1 activation) decode matvec against the F32-activation matvec
 * of the same weight type on every available GPU backend (environment {@code NEURALFREE_BENCH_INT8=1};
 * the test task forwards no system properties). Prints the single-token time of one projection at
 * 0.6B-decoder and 7B/8B-class shapes for every type the lane serves, with and without the Q8_1
 * activation quantization the lane adds. Asserts nothing; the numbers go to the int8 matvec
 * documentation.
 */
@RunWith(Parameterized.class)
public class Int8MatvecBenchmarkTest {

    private static final String SWITCH = "NEURALFREE_BENCH_INT8";
    private static final int WARMUP = 20;
    private static final int RUNS = 100;
    /** Interleaved A/B rounds and launches per round of the paired-form comparison. */
    private static final int ROUNDS = 9;
    private static final int RUNS_PER_ROUND = 20;
    /**
     * Projection shapes of a one-token decode: a 0.6B-class decoder's attention (1024x1024) and FFN
     * (3072x1024, 1024x3072) projections, its large output head (32768x1024), a 4B-class decoder's FFN
     * projections (9728x2560, 2560x9728), and two 7B/8B-class shapes.
     */
    private static final int[][] SHAPES = {{1024, 1024}, {3072, 1024}, {1024, 3072}, {32768, 1024},
        {4096, 4096}, {9728, 2560}, {2560, 9728}, {14336, 4096}};
    /** Every weight type the Q8_1 lane serves. */
    private static final GGMLType[] TYPES = {
        GGMLType.Q4_0, GGMLType.Q4_1, GGMLType.Q5_0, GGMLType.Q5_1, GGMLType.Q8_0, GGMLType.MXFP4,
        GGMLType.NVFP4, GGMLType.Q2_K, GGMLType.Q3_K, GGMLType.Q4_K, GGMLType.Q5_K, GGMLType.Q6_K,
        GGMLType.Q8_K, GGMLType.IQ4_NL, GGMLType.IQ4_XS,
    };

    @Parameterized.Parameters(name = "{0}")
    public static List<Object[]> backends() {
        return Arrays.asList(new Object[]{"opencl"}, new Object[]{"cuda"}, new Object[]{"vulkan"});
    }

    private final String api;
    private HWAcceleration hwAccel;
    private ComputeBackend backend;
    private LlmKernels kernels;
    private int local;
    private boolean previousCuda;

    public Int8MatvecBenchmarkTest(String api) {
        this.api = api;
    }

    @Before
    public void openDevice() {
        Assume.assumeTrue("set " + SWITCH + "=1 to run", "1".equals(System.getenv(SWITCH)));
        previousCuda = HWAcceleration.enableCuda;
        try {
            backend = openBackend();
            Availability availability = LlmKernels.availability(backend);
            Assume.assumeTrue("LLM kernels unavailable on " + api + ": " + availability.reason(),
                    availability.available());
            kernels = LlmKernels.open(backend);
            local = kernels.matvecLocal();
        } catch (Throwable e) {
            closeDevice();
            Assume.assumeTrue("no usable " + api + " GPU: " + e.getMessage(), false);
        }
    }

    private ComputeBackend openBackend() {
        if (api.equals("vulkan")) {
            return new VkComputeBackend();
        }
        if (api.equals("cuda")) {
            HWAcceleration.enableCuda = true;
            HWAcceleration accel = new HWAcceleration();
            hwAccel = accel;
            Assume.assumeFalse("no CUDA device", accel.getCUDevices().isEmpty());
            CUDevice device = accel.getCUDevices().get(0);
            return new CudaComputeBackend(device);
        }
        HWAcceleration accel = new HWAcceleration();
        hwAccel = accel;
        Assume.assumeFalse("no OpenCL device", accel.getCLDevices().isEmpty());
        return new OclComputeBackend(accel.getCLDevices().get(0));
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
    public void int8LaneAgainstTheFloatMatvec() throws Exception {
        System.out.println("device " + backend.capabilities().getDeviceName() + " (" + api + "), matvec local size "
                + local);
        for (GGMLType type : TYPES) {
            for (int[] shape : SHAPES) {
                measure(type, shape[0], shape[1]);
            }
        }
    }

    /**
     * The paired integer-dot matvec ({@code {T}_MatVec2_Q8}) against the single-row form
     * ({@code {T}_MatVec_Q8}) at decode shapes, in interleaved rounds with per-round means and the median
     * reported, so a shared device's load variance cancels between the two forms.
     */
    @Test
    public void pairedInt8MatvecAgainstTheSingleRowForm() throws Exception {
        System.out.println("device " + backend.capabilities().getDeviceName() + " (" + api + "), matvec local size "
                + local);
        int[][] shapes = {{1024, 4096}, {4096, 4096}, {4096, 12288}, {12288, 4096}, {32768, 4096}};
        for (GGMLType type : new GGMLType[]{GGMLType.Q8_0, GGMLType.Q4_K, GGMLType.Q5_K, GGMLType.Q6_K,
                GGMLType.Q8_K}) {
            for (int[] shape : shapes) {
                measurePairing(type, shape[0], shape[1]);
            }
        }
    }

    /** Device time of one Q8_1 quantizer launch at the activation widths of a decode step. */
    @Test
    public void quantizerCostAtDecodeWidths() throws Exception {
        System.out.println("device " + backend.capabilities().getDeviceName() + " (" + api + "), matvec local size "
                + local);
        for (int cols : new int[]{1024, 2560, 4096, 12288}) {
            GpuBuffer xBuf = backend.allocateBuffer((long) cols * 4);
            GpuBuffer qBuf = backend.allocateBuffer((long) (cols / 32) * GpuOps.Q8_1_FLOATS * 4);
            try {
                float[] x = activations(cols);
                backend.computeQueue().enqueueWriteBuffer(xBuf, 0, (long) cols * 4, x).awaitCompletion();
                long nanos = time(() -> backend.computeQueue().enqueueKernel(kernels.get(GpuOps.QUANTIZE_Q8_1),
                        new long[]{roundUp(cols / 32)}, new long[]{local}, new Object[]{cols, xBuf, qBuf}));
                System.out.println(String.format(Locale.ROOT, "quantize %-5s %5d cols  %7.3f us",
                        api, cols, nanos / 1e3));
            } finally {
                xBuf.close();
                qBuf.close();
            }
        }
    }

    /** The paired and single-row forms over the same weights and Q8_1 row, interleaved. */
    private void measurePairing(GGMLType type, int rows, int cols) throws Exception {
        int blockSize = type.getBlockSize();
        int rowBytes = cols / blockSize * type.getTypeSize();
        byte[] weights = weights(type, rows, rowBytes);
        float[] x = activations(cols);
        GpuBuffer wBuf = backend.allocateBuffer((long) rows * rowBytes);
        GpuBuffer xBuf = backend.allocateBuffer((long) cols * 4);
        GpuBuffer qBuf = backend.allocateBuffer((long) (cols / 32) * GpuOps.Q8_1_FLOATS * 4);
        GpuBuffer yBuf = backend.allocateBuffer((long) rows * 4);
        try {
            backend.computeQueue().enqueueWriteBuffer(wBuf, 0, weights.length, weights).awaitCompletion();
            backend.computeQueue().enqueueWriteBuffer(xBuf, 0, (long) cols * 4, x).awaitCompletion();
            backend.computeQueue().enqueueKernel(kernels.get(GpuOps.QUANTIZE_Q8_1),
                    new long[]{roundUp(cols / 32)}, new long[]{local}, new Object[]{cols, xBuf, qBuf});
            backend.computeQueue().finish();
            String single = type.name() + "_MatVec_Q8";
            String paired = type.name() + "_MatVec2_Q8";
            long[] singleRounds = new long[ROUNDS];
            long[] pairedRounds = new long[ROUNDS];
            for (int r = 0; r < WARMUP; r++) {
                launch(single, rows, rows, cols, wBuf, qBuf, yBuf);
                launch(paired, ceilDiv(rows, 2), rows, cols, wBuf, qBuf, yBuf);
            }
            backend.computeQueue().finish();
            for (int r = 0; r < ROUNDS; r++) {
                singleRounds[r] = timeRound(() -> launch(single, rows, rows, cols, wBuf, qBuf, yBuf));
                pairedRounds[r] = timeRound(() -> launch(paired, ceilDiv(rows, 2), rows, cols, wBuf, qBuf, yBuf));
            }
            double singleUs = median(singleRounds) / 1e3;
            double pairedUs = median(pairedRounds) / 1e3;
            System.out.println(String.format(Locale.ROOT,
                    "%-5s %-5s %5dx%-5d single %9.3f us  paired %9.3f us  speedup %5.2fx",
                    type, api, rows, cols, singleUs, pairedUs, singleUs / pairedUs));
        } finally {
            wBuf.close();
            xBuf.close();
            qBuf.close();
            yBuf.close();
        }
    }

    private void launch(String kernel, long groups, int rows, int cols, GpuBuffer wBuf, GpuBuffer qBuf,
                        GpuBuffer yBuf) {
        backend.computeQueue().enqueueKernel(kernels.get(kernel), new long[]{groups * local}, new long[]{local},
                new Object[]{rows, cols, wBuf, qBuf, yBuf});
    }

    /** Mean wall time of {@link #RUNS_PER_ROUND} consecutive launches of one form. */
    private long timeRound(Runnable launches) {
        for (int i = 0; i < RUNS_PER_ROUND; i++) {
            launches.run();
        }
        backend.computeQueue().finish();
        long start = System.nanoTime();
        for (int i = 0; i < RUNS_PER_ROUND; i++) {
            launches.run();
        }
        backend.computeQueue().finish();
        return (System.nanoTime() - start) / RUNS_PER_ROUND;
    }

    private static long median(long[] values) {
        long[] sorted = values.clone();
        Arrays.sort(sorted);
        return sorted[sorted.length / 2];
    }

    private static int ceilDiv(int n, int d) {
        return (n + d - 1) / d;
    }

    private void measure(GGMLType type, int rows, int cols) throws Exception {
        int blockSize = type.getBlockSize();
        int rowBytes = cols / blockSize * type.getTypeSize();
        byte[] weights = weights(type, rows, rowBytes);
        float[] x = activations(cols);
        GpuBuffer wBuf = backend.allocateBuffer((long) rows * rowBytes);
        GpuBuffer xBuf = backend.allocateBuffer((long) cols * 4);
        GpuBuffer qBuf = backend.allocateBuffer((long) (cols / 32) * GpuOps.Q8_1_FLOATS * 4);
        GpuBuffer yBuf = backend.allocateBuffer((long) rows * 4);
        try {
            backend.computeQueue().enqueueWriteBuffer(wBuf, 0, weights.length, weights).awaitCompletion();
            backend.computeQueue().enqueueWriteBuffer(xBuf, 0, (long) cols * 4, x).awaitCompletion();
            long[] global = {(long) rows * local};
            long floatNanos = time(() -> backend.computeQueue().enqueueKernel(kernels.get(type.name() + "_MatVec"),
                    global, new long[]{local}, new Object[]{rows, cols, wBuf, xBuf, yBuf}));
            long quantNanos = time(() -> backend.computeQueue().enqueueKernel(kernels.get(GpuOps.QUANTIZE_Q8_1),
                    new long[]{roundUp(cols / 32)}, new long[]{local}, new Object[]{cols, xBuf, qBuf}));
            long q8Nanos = time(() -> backend.computeQueue().enqueueKernel(kernels.get(type.name() + "_MatVec_Q8"),
                    global, new long[]{local}, new Object[]{rows, cols, wBuf, qBuf, yBuf}));
            long laneNanos = quantNanos + q8Nanos;
            System.out.println(String.format(Locale.ROOT,
                    "%-5s %-5s %5dx%-5d float %8.3f ms  int8 %8.3f ms (quantize %6.3f)  speedup %5.2fx",
                    type, api, rows, cols, floatNanos / 1e6, laneNanos / 1e6, quantNanos / 1e6,
                    (double) floatNanos / (double) laneNanos));
        } finally {
            wBuf.close();
            xBuf.close();
            qBuf.close();
            yBuf.close();
        }
    }

    private long roundUp(long n) {
        return (n + local - 1) / local * local;
    }

    /** Mean wall time of {@code launches}, after {@link #WARMUP} warm-up launches and one queue finish each way. */
    private long time(Runnable launches) {
        for (int i = 0; i < WARMUP; i++) {
            launches.run();
        }
        backend.computeQueue().finish();
        long start = System.nanoTime();
        for (int i = 0; i < RUNS; i++) {
            launches.run();
        }
        backend.computeQueue().finish();
        return (System.nanoTime() - start) / RUNS;
    }

    /** Random blocks of {@code type} with finite block scales. */
    private static byte[] weights(GGMLType type, int rows, int rowBytes) {
        Random rnd = new Random(0x51ED + type.ordinal());
        byte[] data = new byte[rows * rowBytes];
        rnd.nextBytes(data);
        int blockBytes = type.getTypeSize();
        ByteBuffer b = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);
        for (int off = 0; off + blockBytes <= data.length; off += blockBytes) {
            switch (type) {
                case Q6_K:
                    putHalf(data, off + 208, 0.01f);
                    break;
                case Q2_K:
                    putHalf(data, off + 80, 0.02f);
                    putHalf(data, off + 82, 0.01f);
                    break;
                case Q3_K:
                    putHalf(data, off + 108, 0.02f);
                    break;
                case Q8_K:
                    b.putFloat(off, 0.01f);
                    break;
                case MXFP4:
                    data[off] = (byte) 127;
                    break;
                case NVFP4:
                    for (int i = 0; i < 4; i++) {
                        data[off + i] = (byte) 0x40;
                    }
                    break;
                default:
                    putHalf(data, off, 0.02f);
                    if (type == GGMLType.Q4_1 || type == GGMLType.Q5_1 || type == GGMLType.Q5_K
                            || type == GGMLType.Q4_K) {
                        putHalf(data, off + 2, 0.01f);
                    }
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

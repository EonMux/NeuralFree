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
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuPlaceholders;
import com.eonmux.neuralfree.gguf.quant.W12Codec;
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
 * Opt-in timing of the lossless W12 row kernels against the BF16 matvec of the same weights on every available
 * GPU backend (environment {@code NEURALFREE_BENCH_W12=1}; the test task forwards no system properties). Prints
 * the single-token decode time — the W12 in-register decode next to the BF16 rows over the decoded mirror — at
 * decoder-class projection shapes, plus a wide prefill's decode-into-mirror pass next to the plain BF16 batch
 * product, as tokens per second. Asserts nothing; the numbers go to the W12 documentation.
 */
@RunWith(Parameterized.class)
public class W12MatvecBenchmarkTest {

    private static final String SWITCH = "NEURALFREE_BENCH_W12";
    private static final int WARMUP = 20;
    private static final int RUNS = 60;
    private static final int PREFILL_TOKENS = 96;
    /**
     * Projection shapes of a one-token decode: a 0.6B-class decoder's attention (1024x1024) and FFN
     * (3072x1024, 1024x3072) projections, its output head (32768x1024), and a 4B-class decoder's FFN
     * (9728x2560) — every width a whole number of 64-element W12 blocks.
     */
    private static final int[][] SHAPES = {{1024, 1024}, {3072, 1024}, {1024, 3072}, {32768, 1024}, {9728, 2560}};

    @Parameterized.Parameters(name = "{0}")
    public static List<Object[]> backends() {
        return Arrays.asList(new Object[]{"opencl"}, new Object[]{"cuda"}, new Object[]{"vulkan"});
    }

    private final String api;
    private HWAcceleration hwAccel;
    private ComputeBackend backend;
    private LlmKernels kernels;
    private GpuScratch scratch;
    private GpuPlaceholders placeholders;
    private GpuOps ops;
    private boolean previousCuda;

    public W12MatvecBenchmarkTest(String api) {
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
            scratch = GpuScratch.builder(backend, PREFILL_TOKENS).slot(ScratchSlots.X,
                    (long) PREFILL_TOKENS * 9728).build();
            placeholders = GpuPlaceholders.allocate(backend, backend.computeQueue());
            ops = new GpuOps(backend.computeQueue(), kernels, scratch, placeholders);
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
        if (ops != null) {
            ops.close();
            ops = null;
        }
        if (placeholders != null) {
            placeholders.close();
            placeholders = null;
        }
        if (scratch != null) {
            scratch.close();
            scratch = null;
        }
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
    public void decodeTokensPerSecondAgainstTheBf16Matvec() throws Exception {
        System.out.println("device " + backend.capabilities().getDeviceName() + " (" + api + "), matvec local size "
                + kernels.matvecLocal());
        for (int[] shape : SHAPES) {
            measure(shape[0], shape[1]);
        }
    }

    @Test
    public void aWidePrefillAgainstTheBf16Batch() throws Exception {
        System.out.println("device " + backend.capabilities().getDeviceName() + " (" + api + "), matvec local size "
                + kernels.matvecLocal());
        for (int[] shape : SHAPES) {
            measurePrefill(shape[0], shape[1]);
        }
    }

    /** One-token decode: the W12 row decode next to the BF16 matvec over the decoded mirror. */
    private void measure(int rows, int cols) throws Exception {
        GpuWeight w12 = weight(GGMLType.W12, rows, cols);
        GpuWeight bf16 = weight(GGMLType.BF16, rows, cols);
        GpuBuffer in = backend.allocateBuffer((long) cols * 4);
        GpuBuffer out = backend.allocateBuffer((long) rows * 4);
        try {
            backend.computeQueue().enqueueWriteBuffer(in, 0, (long) cols * 4, activations(cols))
                    .awaitCompletion();
            long w12Nanos = time(() -> ops.matvec(w12, in, 0, out, 0, 1));
            long bf16Nanos = time(() -> ops.matvec(bf16, in, 0, out, 0, 1));
            System.out.println(String.format(Locale.ROOT,
                    "decode %-5s %5dx%-5d w12 %8.3f us (%9.2f tok/s)  bf16 %8.3f us (%9.2f tok/s)  ratio %5.2fx",
                    api, rows, cols, w12Nanos / 1e3, 1e9 / w12Nanos, bf16Nanos / 1e3, 1e9 / bf16Nanos,
                    (double) w12Nanos / (double) bf16Nanos));
        } finally {
            w12.shard(0).close();
            bf16.shard(0).close();
            in.close();
            out.close();
        }
    }

    /** A wide prefill: the whole-tensor decode into the BF16 mirror plus the batch product, against plain BF16. */
    private void measurePrefill(int rows, int cols) throws Exception {
        GpuWeight w12 = weight(GGMLType.W12, rows, cols);
        GpuWeight bf16 = weight(GGMLType.BF16, rows, cols);
        GpuBuffer in = backend.allocateBuffer((long) PREFILL_TOKENS * cols * 4);
        GpuBuffer out = backend.allocateBuffer((long) PREFILL_TOKENS * rows * 4);
        try {
            backend.computeQueue().enqueueWriteBuffer(in, 0, (long) PREFILL_TOKENS * cols * 4,
                    activations(PREFILL_TOKENS * cols)).awaitCompletion();
            long w12Nanos = time(() -> ops.matvec(w12, in, 0, out, 0, PREFILL_TOKENS));
            long bf16Nanos = time(() -> ops.matvec(bf16, in, 0, out, 0, PREFILL_TOKENS));
            System.out.println(String.format(Locale.ROOT,
                    "prefill %-4s %5dx%-5d w12 %9.3f ms (%9.2f tok/s)  bf16 %9.3f ms (%9.2f tok/s)  ratio %5.2fx",
                    api, rows, cols, w12Nanos / 1e6, (double) PREFILL_TOKENS * 1e9 / w12Nanos,
                    bf16Nanos / 1e6, (double) PREFILL_TOKENS * 1e9 / bf16Nanos,
                    (double) w12Nanos / (double) bf16Nanos));
        } finally {
            w12.shard(0).close();
            bf16.shard(0).close();
            in.close();
            out.close();
        }
    }

    /** A weight of one type over bf16-grid values: W12's blocks are its own encoding, BF16's the mirror bits. */
    private GpuWeight weight(GGMLType type, int rows, int cols) throws Exception {
        Random rnd = new Random(0x712 + rows);
        float[] v = new float[rows * cols];
        for (int i = 0; i < v.length; i++) {
            int sign = rnd.nextBoolean() ? 0x80000000 : 0;
            v[i] = Float.intBitsToFloat(sign | ((115 + rnd.nextInt(16)) << 23) | (rnd.nextInt(128) << 16));
        }
        byte[] data;
        if (type == GGMLType.W12) {
            data = W12Codec.encode("bench", v, rows, cols);
        } else {
            ByteBuffer bits = ByteBuffer.allocate(v.length * 2).order(ByteOrder.LITTLE_ENDIAN);
            for (float value : v) {
                bits.putShort((short) (Float.floatToRawIntBits(value) >>> 16));
            }
            data = bits.array();
        }
        GpuBuffer buf = backend.allocateBuffer(data.length);
        backend.computeQueue().enqueueWriteBuffer(buf, 0, data.length, data).awaitCompletion();
        return GpuWeight.of("bench.weight", type, new long[]{cols, rows}, buf);
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

    private static float[] activations(int n) {
        Random rnd = new Random(0xC0DE);
        float[] x = new float[n];
        for (int i = 0; i < x.length; i++) {
            x[i] = rnd.nextFloat() * 2f - 1f;
        }
        return x;
    }
}

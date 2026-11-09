package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.cuda.CudaComputeBackend;
import com.eonmux.neuralfree.acceleration.cuda.CUDevice;
import com.eonmux.neuralfree.acceleration.kernels.Availability;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuPlaceholders;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
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
 * Opt-in wall-time probe of the real {@link GpuOps#matmul} dispatch per projection shape and token
 * count ({@code -Dneuralfree.test.mmq.realpath.probe=true}): the enqueue-only wall of ten calls
 * (the host dispatch cost) and the finished wall (the GPU cost of the launched lane), staged
 * activations included. Asserts nothing.
 */
public class MmqRealPathProbeTest {

    /** System property that turns the probe on. */
    public static final String MEASURE_PROPERTY = "neuralfree.test.mmq.realpath.probe";
    private static final int[][] SHAPES = {
        {4096, 2560}, {9728, 2560}, {2560, 9728}, {1024, 2560}, {151936, 2560},
    };
    private static final GGMLType[] TYPES = {GGMLType.Q8_0, GGMLType.Q4_K};
    private static final int[] COUNTS = {24, 32, 48, 64, 96};
    private static final int WARMUP = 30;
    private static final int ROUNDS = 21;
    private static final int RUNS_PER_ROUND = 10;

    private HWAcceleration hwAccel;
    private CudaComputeBackend backend;
    private GpuOps ops;
    private GpuScratch scratch;
    private GpuPlaceholders placeholders;

    @Before
    public void openDevice() {
        Assume.assumeTrue("set " + MEASURE_PROPERTY + "=true to run", Boolean.getBoolean(MEASURE_PROPERTY));
        HWAcceleration.enableCuda = true;
        try {
            HWAcceleration accel = new HWAcceleration(EnumSet.of(ComputeBackend.BackendType.CUDA));
            Assume.assumeTrue("no CUDA device", !accel.getCUDevices().isEmpty());
            CUDevice device = accel.getCUDevices().get(0);
            CudaComputeBackend b = new CudaComputeBackend(device);
            Availability availability = LlmKernels.availability(b);
            Assume.assumeTrue("LLM kernels unavailable: " + availability.reason(), availability.available());
            hwAccel = accel;
            backend = b;
            scratch = GpuScratch.builder(b, 128)
                    .slot(ScratchSlots.X, 128L * 10240).build();
            placeholders = GpuPlaceholders.allocate(b, b.computeQueue());
            ops = new GpuOps(b.computeQueue(), LlmKernels.open(b), scratch, placeholders);
            ops.prewarmWeightKernels(GGMLType.Q8_0, true, false);
            ops.prewarmWeightKernels(GGMLType.Q4_K, true, false);
        } catch (Throwable e) {
            closeDevice();
            Assume.assumeTrue("CUDA init failed: " + e.getMessage(), false);
        }
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
            try {
                hwAccel.shutdown();
            } catch (RuntimeException ignored) {
            }
            hwAccel = null;
        }
        HWAcceleration.enableCuda = false;
    }

    @Test
    public void realPathPerShape() throws InterruptedException {
        System.out.println("device " + backend.capabilities().getDeviceName());
        for (GGMLType type : TYPES) {
            for (int[] shape : SHAPES) {
                for (int n : COUNTS) {
                    measure(type, shape[0], shape[1], n);
                }
            }
        }
    }

    private void measure(GGMLType type, int rows, int cols, int n) throws InterruptedException {
        int rowBytes = cols / type.getBlockSize() * type.getTypeSize();
        GpuBuffer wBuf = null;
        GpuBuffer in = null;
        GpuBuffer out = null;
        try {
            byte[] weights = weights(type, rows, rowBytes);
            wBuf = backend.allocateBuffer((long) rows * rowBytes);
            backend.computeQueue().enqueueWriteBuffer(wBuf, 0, weights.length, weights).awaitCompletion();
            in = backend.allocateBuffer((long) n * cols * 4);
            out = backend.allocateBuffer((long) n * rows * 4);
            GpuWeight w = GpuWeight.of("blk.0.test.weight", type, new long[]{cols, rows}, wBuf);
            for (int i = 0; i < WARMUP; i++) {
                ops.matmul(w, in, 0, out, 0, n);
            }
            backend.computeQueue().finish();
            double[] enqueue = new double[ROUNDS];
            double[] finished = new double[ROUNDS];
            for (int r = 0; r < ROUNDS; r++) {
                long t0 = System.nanoTime();
                for (int i = 0; i < RUNS_PER_ROUND; i++) {
                    ops.matmul(w, in, 0, out, 0, n);
                }
                long t1 = System.nanoTime();
                backend.computeQueue().finish();
                long t2 = System.nanoTime();
                enqueue[r] = (t1 - t0) / (double) RUNS_PER_ROUND / 1e3;
                finished[r] = (t2 - t0) / (double) RUNS_PER_ROUND / 1e3;
            }
            Arrays.sort(enqueue);
            Arrays.sort(finished);
            System.out.println(String.format(Locale.ROOT, "%-6s %6dx%-5d n=%2d  cpu %7.1fus  wall %8.1fus",
                    type, rows, cols, n, enqueue[ROUNDS / 2], finished[ROUNDS / 2]));
        } finally {
            closeQuietly(wBuf, in, out);
        }
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
}

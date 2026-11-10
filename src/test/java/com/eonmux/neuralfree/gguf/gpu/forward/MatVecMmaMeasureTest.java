package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.KernelTiming;
import com.eonmux.neuralfree.acceleration.backend.cuda.CudaComputeBackend;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.ops.gemm.MatVecMmaPlan;
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
 * lanes on CUDA: the counted integer-dot matvec ({@code {T}_MatVecN<n>_Q8}), its two-row twin
 * ({@code {T}_MatVec2N<n>_Q8}) and the tensor-core matvec ({@code {T}_MatVecMma{8,16}_Q8}) at 2, 4 and 8
 * warps, per weight type, projection shape of the 4B and 0.6B Qwen3 decoders and token
 * count. Each launch reads one of several copies of the weight, rotating over more bytes than the L2 holds, so
 * every launch streams its weight from DRAM as a decode step does; times are the per-dispatch device times of
 * the queue's event profile (median over the launches). The printed GB/s is the weight bytes over that time.
 * Asserts nothing; the numbers set {@code MatVecMmaTuning}'s rules.
 */
public class MatVecMmaMeasureTest {

    public static final String MEASURE_PROPERTY = "neuralfree.test.mvm.measure";
    /** Comma-separated weight types to measure (default Q8_0,Q4_K,Q6_K). */
    public static final String TYPES_PROPERTY = "neuralfree.test.mvm.types";
    /** Comma-separated token counts (default 2,4,8,9,16). */
    public static final String COUNTS_PROPERTY = "neuralfree.test.mvm.counts";
    private static final long ROTATION_BYTES = 256L << 20;
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
                if (MatVecMmaPlan.serves(type, shape[1])) {
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
            backend.computeQueue().finish();
            for (String c : System.getProperty(COUNTS_PROPERTY, "2,4,8,9,16").split(",")) {
                int n = Integer.parseInt(c.trim());
                StringBuilder line = new StringBuilder(String.format(Locale.ROOT,
                        "MVM %-5s %6dx%-5d n=%2d floor %7.1f MB", type, rows, cols, n, bytes / 1e6));
                lane(line, "counted", type.name() + "_MatVecN" + n + "_Q8", (long) rows * local, local,
                        wBufs, i -> new Object[]{rows, cols, wBufs.get(i), qBuf, yBuf}, bytes);
                lane(line, "paired", type.name() + "_MatVec2N" + n + "_Q8", (long) ((rows + 1) / 2) * local,
                        local, wBufs, i -> new Object[]{rows, cols, wBufs.get(i), qBuf, yBuf}, bytes);
                int tile = MatVecMmaPlan.tileTokens(n);
                String kernel = MatVecMmaPlan.kernelName(type, tile);
                int tileRows = MatVecMmaPlan.tileRows(type, tile);
                for (int warps = 2; warps <= MatVecMmaPlan.MAX_WARPS; warps *= 2) {
                    long groups = (rows + tileRows - 1) / tileRows;
                    lane(line, "mma" + warps, kernel, groups * 32L * warps, 32L * warps, wBufs,
                            i -> new Object[]{rows, cols, n, wBufs.get(i), qBuf, yBuf}, bytes);
                }
                System.out.println(line);
            }
        } finally {
            for (GpuBuffer b : wBufs) {
                b.close();
            }
            xBuf.close();
            qBuf.close();
            yBuf.close();
        }
    }

    private interface Args {
        Object[] at(int copy);
    }

    /** Appends " name us (GB/s)" for the median device time of LAUNCHES rotating launches. */
    private void lane(StringBuilder line, String name, String kernel, long global, long localSize,
                      List<GpuBuffer> wBufs, Args args, long bytes) throws InterruptedException {
        if (!kernels.has(kernel)) {
            return;
        }
        KernelTiming timing = backend.computeQueue().kernelTiming();
        for (int i = 0; i < 4; i++) {
            backend.computeQueue().enqueueKernel(kernels.get(kernel), new long[]{global}, new long[]{localSize},
                    args.at(i % wBufs.size()));
        }
        backend.computeQueue().finish();
        timing.reset();
        long[] samples = new long[LAUNCHES];
        for (int i = 0; i < LAUNCHES; i++) {
            backend.computeQueue().enqueueKernel(kernels.get(kernel), new long[]{global}, new long[]{localSize},
                    args.at(i % wBufs.size()));
            backend.computeQueue().finish();
            samples[i] = timing.totalNanos();
            timing.reset();
        }
        java.util.Arrays.sort(samples);
        double us = samples[LAUNCHES / 2] / 1e3;
        line.append(String.format(Locale.ROOT, "  %s %7.1f (%4.0f)", name, us, bytes / us / 1e3));
    }

    private long roundUp(long n) {
        return (n + local - 1) / local * local;
    }
}

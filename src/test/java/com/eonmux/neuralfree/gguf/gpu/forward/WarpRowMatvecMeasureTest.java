package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.KernelTiming;
import com.eonmux.neuralfree.acceleration.backend.cuda.CudaComputeBackend;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.EnumSet;
import java.util.List;
import java.util.Locale;

/**
 * Opt-in device-time measurement ({@code -Dneuralfree.test.warprow.measure=true}) of the CUDA single-token lanes
 * {@link WarpRowMatvec} chooses between: the float matvec ({@code {T}_MatVec}), the family's integer-dot single-row
 * and paired forms ({@code {T}_MatVec_Q8}, {@code {T}_MatVec2_Q8}) and the warp-row form ({@code {T}_MatVecRow_Q8})
 * at 1, 2 and 4 warps per work-group, and the fused gate/up pair ({@code {T}_MatVecFusedGateUp_Q8} against
 * {@code {T}_MatVecFusedGateUpRow_Q8}), over the projection shapes of the 0.6B and 4B Qwen3 decoders. Each launch
 * reads one of several copies of the weight, rotating over more bytes than the L2 holds, so every launch streams its
 * weight from DRAM as a decode step does; times are the per-dispatch device times of the queue's event profile
 * (median over the launches), and rounds alternate the lane order. The integer-dot lanes read a Q8_1 row the decode
 * step's norm emits, so the quantizer is not timed. Asserts nothing; the numbers set {@link WarpRowMatvec#warps}.
 */
public class WarpRowMatvecMeasureTest {

    public static final String MEASURE_PROPERTY = "neuralfree.test.warprow.measure";
    /** Comma-separated weight types to measure (default Q5_K,Q4_K). */
    public static final String TYPES_PROPERTY = "neuralfree.test.warprow.types";
    private static final long ROTATION_BYTES = 256L << 20;
    private static final int LAUNCHES = 24;
    private static final int ROUNDS = 3;
    private static final int[][] SHAPES = {
        {1024, 1024}, {2048, 1024}, {3072, 1024}, {1024, 3072}, {151936, 1024},
        {1024, 2560}, {4096, 2560}, {9728, 2560}, {2560, 9728}, {151936, 2560},
    };
    private static final int[] WARPS = {1, 2, WarpRowMatvec.MAX_WARPS};

    private HWAcceleration hwAccel;
    private ComputeBackend backend;
    private LlmKernels kernels;
    private int local;
    private String previousProfile;

    @Before
    public void openDevice() {
        Assume.assumeTrue("set " + MEASURE_PROPERTY + "=true to run", Boolean.getBoolean(MEASURE_PROPERTY));
        previousProfile = System.getProperty(KernelTiming.PROFILE_PROPERTY);
        System.setProperty(KernelTiming.PROFILE_PROPERTY, "true");
        HWAcceleration.enableCuda = true;
        HWAcceleration accel = new HWAcceleration(EnumSet.of(ComputeBackend.BackendType.CUDA));
        hwAccel = accel;
        Assume.assumeFalse("no CUDA device", accel.getCUDevices().isEmpty());
        backend = new CudaComputeBackend(accel.getCUDevices().get(0));
        kernels = LlmKernels.open(backend);
        local = kernels.matvecLocal();
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
    public void singleTokenLanes() throws InterruptedException {
        System.out.println("WARPROW device " + backend.capabilities().getDeviceName() + ", matvec local " + local);
        for (String t : System.getProperty(TYPES_PROPERTY, "Q5_K,Q4_K").split(",")) {
            GGMLType type = GGMLType.valueOf(t.trim());
            Assume.assumeTrue("no warp-row form of " + type, kernels.has(type.name() + "_MatVecRow_Q8"));
            for (int[] shape : SHAPES) {
                measure(type, shape[0], shape[1]);
            }
        }
    }

    private void measure(GGMLType type, int rows, int cols) throws InterruptedException {
        long bytes = (long) cols / type.getBlockSize() * type.getTypeSize() * rows;
        int copies = (int) Math.max(3, Math.min(64, (ROTATION_BYTES + bytes - 1) / bytes));
        byte[] w = MatVecMmaGpuTest.weights(type, rows, (int) (bytes / rows), 0xFACE + type.ordinal());
        List<GpuBuffer> wBufs = new ArrayList<>();
        GpuBuffer xBuf = backend.allocateBuffer((long) cols * 4);
        GpuBuffer qBuf = backend.allocateBuffer((long) (cols / 32) * GpuOps.Q8_1_FLOATS * 4);
        GpuBuffer yBuf = backend.allocateBuffer((long) rows * 4);
        try {
            for (int c = 0; c < copies; c++) {
                GpuBuffer b = backend.allocateBuffer(bytes);
                backend.computeQueue().enqueueWriteBuffer(b, 0, w.length, w).awaitCompletion();
                wBufs.add(b);
            }
            float[] x = MatVecMmaGpuTest.activations(cols);
            backend.computeQueue().enqueueWriteBuffer(xBuf, 0, (long) x.length * 4, x).awaitCompletion();
            backend.computeQueue().enqueueKernel(kernels.get(GpuOps.QUANTIZE_Q8_1),
                    new long[]{(cols / 32 + local - 1) / local * (long) local}, new long[]{local},
                    new Object[]{cols, xBuf, qBuf});
            backend.computeQueue().finish();
            String t = type.name();
            int n = wBufs.size();
            List<String> names = new ArrayList<>();
            List<Lane> lanes = new ArrayList<>();
            names.add("float");
            lanes.add(i -> launch(t + "_MatVec", rows * (long) local, local, rows, cols, wBufs.get(i), xBuf, yBuf));
            names.add("q8");
            lanes.add(i -> launch(t + "_MatVec_Q8", rows * (long) local, local, rows, cols, wBufs.get(i), qBuf,
                    yBuf));
            names.add("q8pair");
            lanes.add(i -> launch(t + "_MatVec2_Q8", (rows + 1) / 2 * (long) local, local, rows, cols, wBufs.get(i),
                    qBuf, yBuf));
            for (int warps : WARPS) {
                names.add("row" + warps);
                lanes.add(i -> launch(t + "_MatVecRow_Q8", WarpRowMatvec.globalSize(rows, warps),
                        WarpRowMatvec.localSize(warps), rows, cols, wBufs.get(i), qBuf, yBuf));
            }
            names.add("gu-q8");
            lanes.add(i -> launch(t + "_MatVecFusedGateUp_Q8", rows * (long) local, local, rows, cols, wBufs.get(i),
                    wBufs.get((i + 1) % n), qBuf, yBuf));
            for (int warps : WARPS) {
                names.add("gu-row" + warps);
                lanes.add(i -> launch(t + "_MatVecFusedGateUpRow_Q8", WarpRowMatvec.globalSize(rows, warps),
                        WarpRowMatvec.localSize(warps), rows, cols, wBufs.get(i), wBufs.get((i + 1) % n), qBuf,
                        yBuf));
            }
            double[][] samples = new double[lanes.size()][ROUNDS];
            for (int r = 0; r < ROUNDS; r++) {
                for (int k = 0; k < lanes.size(); k++) {
                    int lane = r % 2 == 0 ? k : lanes.size() - 1 - k;
                    samples[lane][r] = medianMicros(lanes.get(lane), n);
                }
            }
            StringBuilder line = new StringBuilder(String.format(Locale.ROOT, "WARPROW %-5s %6dx%-5d %7.1f MB", t,
                    rows, cols, bytes / 1e6));
            for (int k = 0; k < lanes.size(); k++) {
                Arrays.sort(samples[k]);
                line.append(String.format(Locale.ROOT, "  %s %.1f", names.get(k), samples[k][ROUNDS / 2]));
            }
            System.out.println(line);
        } finally {
            for (GpuBuffer b : wBufs) {
                b.close();
            }
            xBuf.close();
            qBuf.close();
            yBuf.close();
        }
    }

    private interface Lane {
        void run(int copy);
    }

    private void launch(String kernel, long global, long localSize, Object... args) {
        backend.computeQueue().enqueueKernel(kernels.get(kernel), new long[]{global}, new long[]{localSize}, args);
    }

    /** Median device microseconds of {@link #LAUNCHES} launches of {@code lane}, each on the next weight copy. */
    private double medianMicros(Lane lane, int copies) throws InterruptedException {
        KernelTiming timing = backend.computeQueue().kernelTiming();
        for (int i = 0; i < 4; i++) {
            lane.run(i % copies);
        }
        backend.computeQueue().finish();
        timing.reset();
        long[] samples = new long[LAUNCHES];
        for (int i = 0; i < LAUNCHES; i++) {
            lane.run(i % copies);
            backend.computeQueue().finish();
            samples[i] = timing.totalNanos();
            timing.reset();
        }
        Arrays.sort(samples);
        return samples[LAUNCHES / 2] / 1e3;
    }
}

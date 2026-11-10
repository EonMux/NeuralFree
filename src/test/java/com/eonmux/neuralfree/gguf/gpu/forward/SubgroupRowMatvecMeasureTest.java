package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.quant.FpBlockDecoders;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;

import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Random;

/**
 * Single-token matvec times of the subgroup-row forms ({@link SubgroupRowMatvec}) against the family's float
 * ({@code {T}_MatVec}) and integer-dot ({@code {T}_MatVec_Q8}, with and without its Q8_1 quantization) forms
 * and the fused gate/up pair, on one device of the forms ({@link SubgroupFormBackends}), over the projection shapes
 * of 0.6B and 4B decoders. Rounds alternate the lanes so clock states and throttling bias no lane. Times are host
 * wall times of a launch and its completion. Opt-in through {@code -Dneuralfree.test.sgrow.measure=true}; the API by
 * {@code -Dneuralfree.test.sgrow.measure.api} ({@code opencl}, the default, or {@code vulkan}) and the index among
 * that API's devices of the forms by {@code -Dneuralfree.test.sgrow.measure.device}.
 */
public class SubgroupRowMatvecMeasureTest {

    private static final String PROP = "neuralfree.test.sgrow.measure";
    private static final int[][] SHAPES = {
        {1024, 1024}, {3072, 1024}, {1024, 3072}, {151936, 1024},
        {4096, 2560}, {9728, 2560}, {2560, 9728}, {151936, 2560},
    };
    private static final GGMLType[] TYPES = {GGMLType.Q8_0, GGMLType.Q4_0, GGMLType.Q4_K, GGMLType.Q5_K,
        GGMLType.Q6_K};
    private static final int ROUNDS = 5;
    private static final int REPS = 5;

    private SubgroupFormBackends devices;
    private ComputeBackend backend;
    private LlmKernels kernels;
    private int local;

    @Before
    public void open() {
        Assume.assumeTrue("opt-in (-D" + PROP + "=true)", Boolean.getBoolean(PROP));
        ComputeBackend.BackendType api = ComputeBackend.BackendType.valueOf(
                System.getProperty(PROP + ".api", "opencl").toUpperCase(Locale.ROOT));
        devices = SubgroupFormBackends.open(api);
        List<ComputeBackend> opened = devices.backends();
        int index = Integer.getInteger(PROP + ".device", 0);
        Assume.assumeTrue("no " + api + " device " + index + " of the forms", index < opened.size());
        backend = opened.get(index);
        kernels = LlmKernels.open(backend);
        local = kernels.matvecLocal();
        Assume.assumeTrue("the device builds no subgroup-row forms", kernels.has("Q8_0_MatVecSg_Q8"));
    }

    @After
    public void close() {
        if (devices != null) {
            devices.close();
        }
    }

    @Test
    public void measure() throws Exception {
        System.out.println("SGROW " + backend.backendType() + " device=\"" + backend.capabilities().getDeviceName()
                + "\" matvecLocal=" + local);
        for (GGMLType type : TYPES) {
            for (int[] shape : SHAPES) {
                measureShape(type, shape[0], shape[1]);
            }
        }
    }

    private void measureShape(GGMLType type, int rows, int cols) throws Exception {
        long bytes = (long) rows * (cols / type.getBlockSize()) * type.getTypeSize();
        GpuBuffer w = upload(weights(type, rows, cols));
        GpuBuffer w2 = upload(weights(type, rows, cols));
        GpuBuffer x = upload(new byte[cols * 4]);
        GpuBuffer xq = backend.allocateBuffer((long) cols / 32 * GpuOps.Q8_1_FLOATS * 4);
        GpuBuffer y = backend.allocateBuffer((long) rows * 4);
        try {
            String t = type.name();
            long sgGlobal = SubgroupRowMatvec.globalSize(rows);
            Runnable[] lanes = {
                () -> enqueue(t + "_MatVec", rows * (long) local, local, rows, cols, w, x, y),
                () -> enqueue(t + "_MatVec_Q8", rows * (long) local, local, rows, cols, w, xq, y),
                () -> {
                    enqueue(GpuOps.QUANTIZE_Q8_1, roundUp(cols / 32), local, cols, x, xq);
                    enqueue(t + "_MatVecSg_Q8", sgGlobal, SubgroupRowMatvec.WORK_GROUP, rows, cols, w, xq, y);
                },
                () -> enqueue(t + "_MatVecFusedGateUp", rows * (long) local, local, rows, cols, w, w2, x, y),
                () -> {
                    enqueue(GpuOps.QUANTIZE_Q8_1, roundUp(cols / 32), local, cols, x, xq);
                    enqueue(t + "_MatVecFusedGateUpSg_Q8", sgGlobal, SubgroupRowMatvec.WORK_GROUP, rows, cols,
                            w, w2, xq, y);
                },
            };
            String[] names = {"float", "q8-family", "q8+sg", "gateup-float", "gateup-q8+sg"};
            double[][] samples = new double[lanes.length][ROUNDS];
            for (int r = 0; r < ROUNDS; r++) {
                for (int i = 0; i < lanes.length; i++) {
                    int lane = (r % 2 == 0) ? i : lanes.length - 1 - i;
                    samples[lane][r] = medianMicros(lanes[lane]);
                }
            }
            StringBuilder line = new StringBuilder(String.format(Locale.ROOT, "SGROW %-5s %6dx%-5d", t, rows, cols));
            for (int i = 0; i < lanes.length; i++) {
                Arrays.sort(samples[i]);
                double us = samples[i][ROUNDS / 2];
                double b = i >= 3 ? 2.0 * bytes : bytes;
                line.append(String.format(Locale.ROOT, "  %s=%.1fus(%.1fGB/s)", names[i], us, b / (us * 1e3)));
            }
            System.out.println(line);
        } finally {
            w.close();
            w2.close();
            x.close();
            xq.close();
            y.close();
        }
    }

    private void enqueue(String kernel, long global, int localSize, Object... args) {
        backend.computeQueue().enqueueKernel(kernels.get(kernel), new long[]{global, 1}, new long[]{localSize, 1}, args);
    }

    private long roundUp(long n) {
        return (n + local - 1) / local * local;
    }

    /** Median microseconds of one run of {@code launch} over {@link #REPS} timed runs after one warm-up. */
    private double medianMicros(Runnable launch) {
        launch.run();
        backend.computeQueue().finish();
        double[] t = new double[REPS];
        for (int i = 0; i < REPS; i++) {
            long t0 = System.nanoTime();
            launch.run();
            backend.computeQueue().finish();
            t[i] = (System.nanoTime() - t0) / 1e3;
        }
        Arrays.sort(t);
        return t[REPS / 2];
    }

    private static byte[] weights(GGMLType type, int rows, int cols) {
        int blockBytes = type.getTypeSize();
        long blocks = (long) rows * (cols / type.getBlockSize());
        byte[] data = new byte[Math.toIntExact(blocks * blockBytes)];
        new Random(rows * 31L + cols).nextBytes(data);
        int dOff = type == GGMLType.Q6_K ? 208 : 0;
        int bits = FpBlockDecoders.fp32ToFp16(0.01f);
        for (long b = 0; b < blocks; b++) {
            int base = Math.toIntExact(b * blockBytes) + dOff;
            data[base] = (byte) bits;
            data[base + 1] = (byte) (bits >>> 8);
            if (type == GGMLType.Q4_K || type == GGMLType.Q5_K) {
                data[base + 2] = (byte) bits;
                data[base + 3] = (byte) (bits >>> 8);
            }
        }
        return data;
    }

    private GpuBuffer upload(byte[] data) throws Exception {
        GpuBuffer buf = backend.allocateBuffer(data.length);
        backend.computeQueue().enqueueWriteBuffer(buf, 0, data.length, data).awaitCompletion();
        return buf;
    }
}

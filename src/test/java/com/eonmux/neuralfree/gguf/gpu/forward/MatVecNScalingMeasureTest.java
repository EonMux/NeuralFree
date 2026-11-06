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
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.Locale;
import java.util.Random;

import static org.junit.Assert.assertTrue;

/**
 * The multi-column Q8_0 matvec's own scaling: the wall time of one {@code Q8_0_MatVecN_Q8} launch
 * over token groups of one to {@link LlmKernelConstants#MATVEC_N_MAX_TOKENS} rows against the same
 * dot work split into one single-row {@code Q8_0_MatVec_Q8} launch per row, at the projection shapes
 * of a 4B decoder (fused gate/up, packed QKV, down projection). The step-level measurement
 * ({@code SmallBatchStepMeasureTest}) attributes a step's cost; this one attributes the matvec
 * kernel's share of it, separating the kernel's token-group scaling from the step around it.
 * Opt-in through {@code -Dneuralfree.test.matvecn.measure=true}; the shared suite skips in seconds.
 * Weights are random bytes with finite block scales &mdash; the timing depends on the access pattern,
 * not the values. A backend is skipped when its GPU or its kernel library is unavailable.
 */
public class MatVecNScalingMeasureTest {

    /** System property that turns the measurement on. */
    public static final String MEASURE_PROPERTY = "neuralfree.test.matvecn.measure";

    /** The weight type of the 4B decoder whose step cost motivated the multi-column lane. */
    private static final GGMLType TYPE = GGMLType.Q8_0;

    /** Projection shapes of Qwen3-Embedding-4B: {rows, cols}, in report order. */
    private static final int[][] SHAPES = {
            {19456, 2560},   // fused gate/up: two hidden-size-wide FFN matrices folded row-wise
            {6144, 2560},    // packed QKV: 32 query heads and 8+8 key/value heads of 128
            {2560, 9728},    // FFN down projection
    };
    private static final String[] SHAPE_NAMES = {"gateup", "qkv", "down"};

    /** Row counts one launch covers: the draft maximums in between and the verify shape itself. */
    private static final int[] ROW_COUNTS = {1, 2, 4, 8, 9, 16};
    /** Untimed launches per shape and row count before the timed launches. */
    private static final int WARMUP = 5;
    /** Timed launches per shape and row count; the median is reported. */
    private static final int TIMED = 11;

    private HWAcceleration hwAccel;
    private ComputeBackend backend;
    private LlmKernels kernels;
    private int local;
    private boolean previousCuda;

    @Before
    public void openDevice() {
        previousCuda = HWAcceleration.enableCuda;
        try {
            HWAcceleration.enableCuda = true;
            hwAccel = new HWAcceleration();
            Assume.assumeTrue("no CUDA device", !hwAccel.getCUDevices().isEmpty());
            backend = new CudaComputeBackend(hwAccel.getCUDevices().get(0));
            Availability availability = LlmKernels.availability(backend);
            Assume.assumeTrue("LLM kernels unavailable: " + availability.reason(),
                    availability.available());
            kernels = LlmKernels.open(backend);
            local = kernels.matvecLocal();
        } catch (Throwable e) {
            closeDevice();
            Assume.assumeTrue("no usable CUDA GPU: " + e.getMessage(), false);
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
            hwAccel.shutdown();
            hwAccel = null;
        }
        HWAcceleration.enableCuda = previousCuda;
    }

    /** The MatVecN versus per-row MatVec launch times of every shape and row count on CUDA. */
    @Test
    public void measureOnCuda() throws Exception {
        Assume.assumeTrue("opt-in (" + MEASURE_PROPERTY + "=true)",
                Boolean.getBoolean(MEASURE_PROPERTY));
        String device = hwAccel.getCUDevices().get(0).getDeviceName();
        System.out.printf(Locale.ROOT, "MATVECN-SCALE backend=CUDA device=\"%s\" local=%d%n",
                device, local);
        for (int s = 0; s < SHAPES.length; s++) {
            measureShape(SHAPES[s][0], SHAPES[s][1], SHAPE_NAMES[s]);
        }
    }

    /**
     * One shape's launch times: for every row count the median of one MatVecN launch and of the same
     * dot work as one single-row launch per row, plus the per-row ratio of the two.
     */
    private void measureShape(int rows, int cols, String name) throws Exception {
        byte[] weights = randomWeights(rows, cols);
        GpuBuffer wBuf = upload(weights);
        GpuBuffer q8 = upload(activations(cols, LlmKernelConstants.MATVEC_N_MAX_TOKENS));
        GpuBuffer yBuf = backend.allocateBuffer((long) LlmKernelConstants.MATVEC_N_MAX_TOKENS
                * rows * 4);
        String nKernel = TYPE.name() + "_MatVecN_Q8";
        String rowKernel = TYPE.name() + "_MatVec_Q8";
        assertTrue("build has no " + nKernel, kernels.has(nKernel));
        assertTrue("build has no " + rowKernel, kernels.has(rowKernel));
        try {
            for (int n : ROW_COUNTS) {
                double group = medianLaunchMillis(nKernel, rows, cols, n, wBuf, q8, yBuf);
                double perRow = 0.0;
                for (int t = 0; t < n; t++) {
                    perRow += medianLaunchMillis(rowKernel, rows, cols, 0, wBuf, q8, yBuf);
                }
                System.out.printf(Locale.ROOT,
                        "MATVECN-SCALE shape=%s rows=%d cols=%d n=%d group-ms=%.3f per-row-ms=%.3f"
                                + " group-vs-row=%.2fx%n",
                        name, rows, cols, n, group, perRow, group / Math.max(perRow, 1e-9));
            }
        } finally {
            wBuf.close();
            q8.close();
            yBuf.close();
        }
    }

    /**
     * The median wall time of the launch pattern for one token group: {@code tokens > 0} times one
     * MatVecN launch over {@code tokens} rows; {@code tokens == 0} times one single-row launch on
     * activation row {@code row} (the caller sums the per-row times of a group).
     */
    private double medianLaunchMillis(String kernel, int rows, int cols, int tokens, GpuBuffer wBuf,
                                      GpuBuffer q8, GpuBuffer yBuf) throws Exception {
        long global = (long) rows * local;
        for (int i = 0; i < WARMUP; i++) {
            launch(kernel, rows, cols, tokens, wBuf, q8, yBuf, global);
        }
        backend.computeQueue().finish();
        long[] samples = new long[TIMED];
        for (int i = 0; i < TIMED; i++) {
            long t0 = System.nanoTime();
            launch(kernel, rows, cols, tokens, wBuf, q8, yBuf, global);
            backend.computeQueue().finish();
            samples[i] = System.nanoTime() - t0;
        }
        Arrays.sort(samples);
        return samples[samples.length / 2] / 1e6;
    }

    private void launch(String kernel, int rows, int cols, int tokens, GpuBuffer wBuf,
                        GpuBuffer q8, GpuBuffer yBuf, long global) throws Exception {
        if (tokens > 0) {
            backend.computeQueue().enqueueKernel(kernels.get(kernel), new long[]{global},
                    new long[]{local}, new Object[]{rows, cols, tokens, wBuf, q8, yBuf});
        } else {
            backend.computeQueue().enqueueKernel(kernels.get(kernel), new long[]{global},
                    new long[]{local}, new Object[]{rows, cols, wBuf, q8, yBuf});
        }
    }

    /** Random blocks of {@code TYPE} with finite block scales, for {@code rows} rows. */
    private static byte[] randomWeights(int rows, int cols) {
        Random rnd = new Random(42000);
        int blocks = rows * (cols / TYPE.getBlockSize());
        byte[] data = new byte[blocks * TYPE.getTypeSize()];
        rnd.nextBytes(data);
        ByteBuffer scales = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);
        for (int b = 0; b < blocks; b++) {
            int base = b * TYPE.getTypeSize();
            int bits = 0x3C00;   // f16 1.0: finite scales keep the timing free of denormal effects
            scales.putShort(base, (short) bits);
        }
        return data;
    }

    /** The Q8_1 blocks of {@code n} zero activation rows of {@code cols} elements. */
    private static byte[] activations(int cols, int n) {
        return new byte[n * (cols / 32) * GpuOps.Q8_1_FLOATS * 4];
    }

    /** A device buffer holding {@code data}. */
    private GpuBuffer upload(byte[] data) throws Exception {
        GpuBuffer buf = backend.allocateBuffer(data.length);
        backend.computeQueue().enqueueWriteBuffer(buf, 0, data.length, data).awaitCompletion();
        return buf;
    }
}

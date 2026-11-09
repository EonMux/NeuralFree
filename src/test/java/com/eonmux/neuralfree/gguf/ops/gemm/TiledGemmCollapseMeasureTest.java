package com.eonmux.neuralfree.gguf.ops.gemm;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.ocl.OclComputeBackend;
import com.eonmux.neuralfree.acceleration.kernels.Availability;
import com.eonmux.neuralfree.acceleration.ocl.CLDevice;
import com.eonmux.neuralfree.gguf.GGMLType;
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

/**
 * The per-row cost curve of the tiled GEMM over dense F16 weights at the row counts the dispatch
 * hands it ({@code GpuProjections} serves the tiled lane above the multi-column matvec's range):
 * for one 0.6B-class and one 4B-class projection shape, the wall time of one
 * {@code F16_MatMul_Tiled} launch at every row count from the matvec crossover to 64 rows, next to
 * the float multi-column matvec ({@code F16_MatVecN}) the dispatch serves below it. The printed
 * {@code per-row-ms} must fall monotonically with the row count over 16..64 &mdash; a rising or
 * flat-and-high per-row cost there is the small-batch collapse this test exists to catch. Opt-in
 * through {@code -Dneuralfree.test.tiled.measure=true}; the shared suite skips in seconds.
 */
public class TiledGemmCollapseMeasureTest {

    /** System property that turns the measurement on. */
    public static final String MEASURE_PROPERTY = "neuralfree.test.tiled.measure";
    /** System property that selects the OpenCL device by index; 0 or negative is the default device. */
    public static final String DEVICE_PROPERTY = "neuralfree.test.tiled.measure.device";

    /** Measured shapes {M, K}: one per size class, as SmallBatchCrossoverMeasureTest picks them. */
    private static final int[][] SHAPES = {
            {4096, 1024},    // 0.6B-class packed QKV
            {19456, 2560},   // 4B-class fused gate/up
    };
    /**
     * Row counts one tiled launch covers: the matvec band the dispatch serves for context, then the
     * tile boundaries of the 32-token tile (16, 17, 32, 33) and the steps to 64 rows.
     */
    private static final int[] ROW_COUNTS = {8, 16, 17, 20, 24, 32, 33, 48, 64};
    /** Largest row count measured, which sizes the activation and output buffers. */
    private static final int MAX_ROWS = 64;
    /** Untimed launches per point before the timed launches. */
    private static final int WARMUP = 2;
    /** Timed launches per point; the median is reported. */
    private static final int TIMED = 7;

    private HWAcceleration hwAccel;
    private ComputeBackend backend;
    private LlmKernels kernels;
    private int local;

    @Before
    public void setUp() {
        Assume.assumeTrue("opt-in (" + MEASURE_PROPERTY + "=true)",
                Boolean.getBoolean(MEASURE_PROPERTY));
        try {
            GpuTestUtil.assumeOpenCLAvailable();
            hwAccel = new HWAcceleration(java.util.EnumSet.of(ComputeBackend.BackendType.OPENCL));
            int index = Math.max(0, Integer.getInteger(DEVICE_PROPERTY, 0));
            java.util.List<CLDevice> devices = hwAccel.getCLDevices();
            Assume.assumeTrue("no OpenCL device " + index + " (found " + devices.size() + ")",
                    index < devices.size());
            backend = new OclComputeBackend(devices.get(index));
            Availability availability = LlmKernels.availability(backend);
            Assume.assumeTrue("LLM kernels unavailable: " + availability.reason(),
                    availability.available());
            kernels = LlmKernels.open(backend);
            local = kernels.matvecLocal();
        } catch (Throwable e) {
            closeDevice();
            Assume.assumeTrue("no usable OpenCL GPU: " + e.getMessage(), false);
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
    }

    /** The tiled GEMM's per-row cost curve against the multi-column matvec band below it. */
    @Test
    public void tiledGemmPerRowCost() throws Exception {
        String device = backend.capabilities().getDeviceName();
        for (int[] shape : SHAPES) {
            int m = shape[0];
            int k = shape[1];
            GpuBuffer w = upload(weights(m, k));
            GpuBuffer x = upload(new float[MAX_ROWS * k]);
            GpuBuffer y = backend.allocateBuffer((long) MAX_ROWS * m * Float.BYTES);
            try {
                System.out.printf(Locale.ROOT,
                        "TILED-COLLAPSE backend=OpenCL device=\"%s\" type=F16 m=%d k=%d tile=%dx%d wg=%d%n",
                        device, m, k, TiledGemmPlan.TILE_ROWS, TiledGemmPlan.TILE_TOKENS,
                        TiledGemmPlan.WORKGROUP_SIZE);
                for (int n : ROW_COUNTS) {
                    double matvecN = n <= 16 ? matvecNMillis(m, k, n, w, x, y) : Double.NaN;
                    double tiled = tiledMillis(m, k, n, w, x, y);
                    System.out.printf(Locale.ROOT,
                            "TILED-COLLAPSE m=%d k=%d n=%d matvecn-ms=%s tiled-ms=%.3f"
                                    + " tiled-per-row-ms=%.4f tiled-token-tiles=%d%n",
                            m, k, n, Double.isNaN(matvecN) ? "n/a" : String.format(Locale.ROOT, "%.3f", matvecN),
                            tiled, tiled / n, (n + TiledGemmPlan.TILE_TOKENS - 1) / TiledGemmPlan.TILE_TOKENS);
                }
            } finally {
                w.close();
                x.close();
                y.close();
            }
        }
    }

    /** The median wall time of one {@code F16_MatMul_Tiled} launch over {@code n} rows. */
    private double tiledMillis(int m, int k, int n, GpuBuffer w, GpuBuffer x, GpuBuffer y)
            throws Exception {
        TiledGemmPlan plan = TiledGemmPlan.contiguous(GGMLType.F16, m, k, n);
        Assume.assumeTrue("device cannot run the tiled launch: " + plan.checkDevice(backend.capabilities()),
                plan.checkDevice(backend.capabilities()) == null);
        Object[] args = plan.args(w, x, y);
        return medianMillis(() -> backend.computeQueue().enqueueKernel(kernels.get(plan.kernelName()),
                plan.globalWorkSize(), plan.localWorkSize(), args));
    }

    /** The median wall time of one {@code F16_MatVecN} launch over {@code n} rows (n <= 16). */
    private double matvecNMillis(int m, int k, int n, GpuBuffer w, GpuBuffer x, GpuBuffer y)
            throws Exception {
        Object[] args = new Object[]{m, k, n, w, x, y};
        return medianMillis(() -> backend.computeQueue().enqueueKernel(kernels.get("F16_MatVecN"),
                new long[]{(long) m * local}, new long[]{local}, args));
    }

    /** The median wall milliseconds of {@code launch} over {@link #TIMED} runs after {@link #WARMUP}. */
    private double medianMillis(Runnable launch) {
        for (int i = 0; i < WARMUP; i++) {
            launch.run();
        }
        backend.computeQueue().finish();
        long[] samples = new long[TIMED];
        for (int i = 0; i < TIMED; i++) {
            long t0 = System.nanoTime();
            launch.run();
            backend.computeQueue().finish();
            samples[i] = System.nanoTime() - t0;
        }
        Arrays.sort(samples);
        return samples[samples.length / 2] / 1e6;
    }

    /** Random F16 weights with finite values near 0.5, for {@code m} rows of {@code k} columns. */
    private static byte[] weights(int m, int k) {
        ByteBuffer data = ByteBuffer.allocate(m * k * 2).order(ByteOrder.LITTLE_ENDIAN);
        Random rnd = new Random(42001);
        for (int i = 0; i < (long) m * k; i++) {
            data.putShort((short) (0x3800 + (rnd.nextInt(16) << 2)));   // f16 values near 0.5
        }
        return data.array();
    }

    /** A device buffer holding {@code data} bytes. */
    private GpuBuffer upload(byte[] data) throws Exception {
        GpuBuffer buf = backend.allocateBuffer(data.length);
        backend.computeQueue().enqueueWriteBuffer(buf, 0, data.length, data).awaitCompletion();
        return buf;
    }

    /** A device buffer of zeroed float rows. */
    private GpuBuffer upload(float[] data) throws Exception {
        ByteBuffer bb = ByteBuffer.allocate(data.length * 4).order(ByteOrder.LITTLE_ENDIAN);
        for (float v : data) {
            bb.putFloat(v);
        }
        return upload(bb.array());
    }
}

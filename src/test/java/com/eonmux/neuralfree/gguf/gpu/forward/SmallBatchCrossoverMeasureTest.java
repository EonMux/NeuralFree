package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.cuda.CudaComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ocl.OclComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.vulkan.VkComputeBackend;
import com.eonmux.neuralfree.acceleration.cuda.CUDevice;
import com.eonmux.neuralfree.acceleration.kernels.Availability;
import com.eonmux.neuralfree.acceleration.ocl.CLDevice;
import com.eonmux.neuralfree.acceleration.vulkan.VkInstanceManager;
import com.eonmux.neuralfree.acceleration.vulkan.VkPhysicalDeviceInfo;
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
import java.util.EnumSet;
import java.util.Locale;
import java.util.Random;

import static org.junit.Assert.assertTrue;

/**
 * The kernel-level lane costs behind {@link SmallBatchCrossover}: for one projection shape of each
 * size class (a 0.6B-class 1024-column projection and a 4B-class 2560-column one) the wall time of
 * the batched matvec ({@code {T}_BatchMatMul_2D}), the float multi-column matvec ({@code {T}_MatVecN})
 * and, for Q8_0, the integer-dot multi-column matvec ({@code {T}_MatVecN_Q8}) at every dispatched row
 * count, against one single-row {@code {T}_MatVec} launch. The step-level measurement
 * ({@code SmallBatchStepMeasureTest}) reports the dispatch these costs pick. Opt-in through
 * {@code -Dneuralfree.test.crossover.measure=true}; the shared suite skips in seconds. Weights are
 * random blocks with finite scales &mdash; the timing depends on the access pattern, not the values.
 * A backend is skipped when its GPU or its kernel library is unavailable.
 */
public class SmallBatchCrossoverMeasureTest {

    /** System property that turns the measurement on. */
    public static final String MEASURE_PROPERTY = "neuralfree.test.crossover.measure";
    /**
     * System property that selects the device to measure by index into the backend's device list
     * (CUDA and OpenCL device order, Vulkan physical-device order); unset, zero or negative measures
     * the device the harness picks by default (the first CUDA and OpenCL device, the first Vulkan
     * device reporting cooperative matrices).
     */
    public static final String DEVICE_PROPERTY = "neuralfree.test.crossover.measure.device";

    /** Measured shapes: {rows, cols}, one per size class (0.6B-class first). */
    private static final int[][] SHAPES = {
            {4096, 1024},    // 0.6B-class packed QKV
            {19456, 2560},   // 4B-class fused gate/up
    };
    private static final String[] SHAPE_NAMES = {"small", "large"};

    /** The measured weight types: the two the three models use. */
    private static final GGMLType[] TYPES = {GGMLType.Q8_0, GGMLType.F16};
    /** Row counts one launch covers: the dispatched multi-row counts. */
    private static final int[] ROW_COUNTS = {2, 4, 8, 9, 16};
    /** Untimed launches per point before the timed launches. */
    private static final int WARMUP = 2;
    /** Timed launches per point; the median is reported. */
    private static final int TIMED = 5;

    private HWAcceleration hwAccel;
    private ComputeBackend backend;
    private LlmKernels kernels;
    private int local;
    private boolean previousCuda;

    @Before
    public void rememberCuda() {
        previousCuda = HWAcceleration.enableCuda;
        HWAcceleration.enableCuda = true;
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

    /** Open a backend by name ("CUDA", "OpenCL", "Vulkan") and its LLM kernels; skips when unavailable. */
    private void open(String backendName) {
        try {
            int index = deviceIndex();
            if (backendName.equals("CUDA")) {
                GpuTestUtil.assumeCudaAvailable();
                hwAccel = new HWAcceleration(EnumSet.of(ComputeBackend.BackendType.CUDA));
                backend = new CudaComputeBackend(hwAccel.getCUDevices().get(assumeIndex(index, hwAccel.getCUDevices().size(),
                        backendName)));
            } else if (backendName.equals("OpenCL")) {
                GpuTestUtil.assumeOpenCLAvailable();
                hwAccel = new HWAcceleration(EnumSet.of(ComputeBackend.BackendType.OPENCL));
                backend = new OclComputeBackend(hwAccel.getCLDevices().get(assumeIndex(index, hwAccel.getCLDevices().size(),
                        backendName)));
            } else {
                GpuTestUtil.assumeVulkanAvailable();
                hwAccel = null;
                backend = vulkan(index);
            }
            Availability availability = LlmKernels.availability(backend);
            Assume.assumeTrue("LLM kernels unavailable: " + availability.reason(),
                    availability.available());
            kernels = LlmKernels.open(backend);
            local = kernels.matvecLocal();
        } catch (Throwable e) {
            closeDevice();
            Assume.assumeTrue("no usable " + backendName + " GPU: " + e.getMessage(), false);
        }
    }

    /** The selected device index, or 0 (the default device) when {@link #DEVICE_PROPERTY} is unset or negative. */
    private static int deviceIndex() {
        return Math.max(0, Integer.getInteger(DEVICE_PROPERTY, 0));
    }

    /** The device index to open: the selected one, failed by {@link org.junit.Assume} when out of range. */
    private static int assumeIndex(int index, int count, String backendName) {
        Assume.assumeTrue("no " + backendName + " device " + index + " (found " + count + ")",
                index >= 0 && index < count);
        return index;
    }

    /**
     * The Vulkan compute backend of device {@code index} when it is positive (plain physical-device
     * order), else of the first device reporting cooperative matrices, else the default device.
     */
    private ComputeBackend vulkan(int index) {
        VkInstanceManager manager = VkInstanceManager.acquire();
        try {
            if (index > 0) {
                return new VkComputeBackend(manager, manager.devices().get(assumeIndex(index,
                        manager.devices().size(), "Vulkan")));
            }
            for (VkPhysicalDeviceInfo info : manager.devices()) {
                if (info.supports(VkPhysicalDeviceInfo.Feature.COOPERATIVE_MATRIX)) {
                    return new VkComputeBackend(manager, info);
                }
            }
        } finally {
            manager.release();
        }
        return new VkComputeBackend();
    }

    private void probe(String backendName) throws Exception {
        String device = deviceName(backendName);
        for (int s = 0; s < SHAPES.length; s++) {
            for (GGMLType type : TYPES) {
                if (s == 1 && type == GGMLType.F16) {
                    continue;   // no 4B-class F16 model to decide for
                }
                measureShape(backendName, device, type, SHAPES[s][0], SHAPES[s][1], SHAPE_NAMES[s]);
            }
        }
    }

    private String deviceName(String backendName) {
        int index = deviceIndex();
        if (backendName.equals("CUDA")) {
            CUDevice d = hwAccel.getCUDevices().get(index);
            return d.getDeviceName();
        }
        if (backendName.equals("OpenCL")) {
            CLDevice d = hwAccel.getCLDevices().get(index);
            Object name = d.getDeviceInfo().get("CL_DEVICE_NAME");
            return name != null ? String.valueOf(name) : String.valueOf(d.getDeviceName());
        }
        return backend.capabilities().getDeviceName();
    }

    /** One type and shape's lane costs at every row count. */
    private void measureShape(String backendName, String device, GGMLType type, int rows, int cols,
                              String sizeName) throws Exception {
        String matvec = type.name() + "_MatVec";
        String batch2d = type == GGMLType.Q8_0 ? "LLM_Batch_Q8_0_MatMul_2D" : type.name() + "_BatchMatMul_2D";
        String matvecN = type.name() + "_MatVecN";
        String matvecNQ8 = type.name() + "_MatVecN_Q8";
        assertTrue("build has no " + matvecN, kernels.has(matvecN));
        assertTrue("build has no " + batch2d, kernels.has(batch2d));
        boolean hasInt8N = kernels.has(matvecNQ8);
        GpuBuffer wBuf = upload(weights(type, rows, cols));
        GpuBuffer xBuf = upload(new float[LlmKernelConstants.MATVEC_N_MAX_TOKENS * cols]);
        GpuBuffer q8Buf = upload(new byte[LlmKernelConstants.MATVEC_N_MAX_TOKENS
                * (cols / 32) * GpuOps.Q8_1_FLOATS * 4]);
        GpuBuffer yBuf = backend.allocateBuffer((long) LlmKernelConstants.MATVEC_N_MAX_TOKENS * rows * 4);
        try {
            double single = medianMillis(matvec, type, rows, cols, 0, wBuf, xBuf, q8Buf, yBuf, false);
            System.out.printf(Locale.ROOT,
                    "CROSSOVER-PROBE backend=%s device=\"%s\" type=%s shape=%s rows=%d cols=%d n=1"
                            + " single-ms=%.3f%n",
                    backendName, device, type, sizeName, rows, cols, single);
            for (int n : ROW_COUNTS) {
                double two = medianMillis(batch2d, type, rows, cols, n, wBuf, xBuf, q8Buf, yBuf, true);
                double group = medianMillis(matvecN, type, rows, cols, n, wBuf, xBuf, q8Buf, yBuf, false);
                Double q8n = hasInt8N
                        ? Double.valueOf(medianMillis(matvecNQ8, type, rows, cols, n, wBuf, xBuf, q8Buf, yBuf, false))
                        : null;
                System.out.printf(Locale.ROOT,
                        "CROSSOVER-PROBE backend=%s device=\"%s\" type=%s shape=%s rows=%d cols=%d n=%d"
                                + " batch2d-ms=%.3f matvecn-ms=%.3f matvecnq8-ms=%s%n",
                        backendName, device, type, sizeName, rows, cols, n, two, group,
                        q8n == null ? "n/a" : String.format(Locale.ROOT, "%.3f", q8n));
            }
        } finally {
            wBuf.close();
            xBuf.close();
            q8Buf.close();
            yBuf.close();
        }
    }

    /**
     * The median wall time of one launch pattern: {@code tokens == 0} times the single-row matvec,
     * {@code tokens > 0} times one multi-row launch over {@code tokens} rows ({@code twoDimensional}
     * launches the batched matvec's 2-D grid, the others the 1-D row grid).
     */
    private double medianMillis(String kernel, GGMLType type, int rows, int cols, int tokens,
                                GpuBuffer wBuf, GpuBuffer xBuf, GpuBuffer q8Buf, GpuBuffer yBuf,
                                boolean twoDimensional) throws Exception {
        for (int i = 0; i < WARMUP; i++) {
            launch(kernel, type, rows, cols, tokens, wBuf, xBuf, q8Buf, yBuf, twoDimensional);
        }
        backend.computeQueue().finish();
        long[] samples = new long[TIMED];
        for (int i = 0; i < TIMED; i++) {
            long t0 = System.nanoTime();
            launch(kernel, type, rows, cols, tokens, wBuf, xBuf, q8Buf, yBuf, twoDimensional);
            backend.computeQueue().finish();
            samples[i] = System.nanoTime() - t0;
        }
        Arrays.sort(samples);
        return samples[samples.length / 2] / 1e6;
    }

    private void launch(String kernel, GGMLType type, int rows, int cols, int tokens, GpuBuffer wBuf,
                        GpuBuffer xBuf, GpuBuffer q8Buf, GpuBuffer yBuf, boolean twoDimensional)
            throws Exception {
        if (tokens == 0) {
            backend.computeQueue().enqueueKernel(kernels.get(kernel), new long[]{(long) rows * local},
                    new long[]{local}, new Object[]{rows, cols, wBuf, xBuf, yBuf});
            return;
        }
        if (kernel.endsWith("_Q8")) {
            backend.computeQueue().enqueueKernel(kernels.get(kernel), new long[]{(long) rows * local},
                    new long[]{local}, new Object[]{rows, cols, tokens, wBuf, q8Buf, yBuf});
            return;
        }
        Object[] args = new Object[]{rows, cols, tokens, wBuf, xBuf, yBuf};
        long[] global = twoDimensional
                ? new long[]{rows * local, tokens}
                : new long[]{rows * local};
        long[] lcl = twoDimensional ? new long[]{local, 1} : new long[]{local};
        backend.computeQueue().enqueueKernel(kernels.get(kernel), global, lcl, args);
    }

    /** Random weight blocks of {@code type} with finite scales, for {@code rows} rows of {@code cols}. */
    private static byte[] weights(GGMLType type, int rows, int cols) {
        if (type == GGMLType.Q8_0) {
            Random rnd = new Random(42000);
            int blocks = rows * (cols / type.getBlockSize());
            byte[] data = new byte[blocks * type.getTypeSize()];
            rnd.nextBytes(data);
            ByteBuffer scales = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);
            for (int b = 0; b < blocks; b++) {
                scales.putShort(b * type.getTypeSize(), (short) 0x3C00);   // f16 1.0: finite scales
            }
            return data;
        }
        int n = rows * cols;
        ByteBuffer data = ByteBuffer.allocate(n * 2).order(ByteOrder.LITTLE_ENDIAN);
        Random rnd = new Random(42001);
        for (int i = 0; i < n; i++) {
            data.putShort((short) (0x3800 + (rnd.nextInt(16) << 2)));   // f16 values near 0.5
        }
        return data.array();
    }

    /** A device buffer holding {@code data}. */
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

    // ================================================================== tests

    /** The lane costs on the CUDA device. */
    @Test
    public void measureOnCuda() throws Exception {
        Assume.assumeTrue("opt-in (" + MEASURE_PROPERTY + "=true)",
                Boolean.getBoolean(MEASURE_PROPERTY));
        open("CUDA");
        probe("CUDA");
    }

    /** The lane costs on the OpenCL device. */
    @Test
    public void measureOnOpenCL() throws Exception {
        Assume.assumeTrue("opt-in (" + MEASURE_PROPERTY + "=true)",
                Boolean.getBoolean(MEASURE_PROPERTY));
        open("OpenCL");
        probe("OpenCL");
    }

    /** The lane costs on the Vulkan device the fast lanes enumerate first. */
    @Test
    public void measureOnVulkan() throws Exception {
        Assume.assumeTrue("opt-in (" + MEASURE_PROPERTY + "=true)",
                Boolean.getBoolean(MEASURE_PROPERTY));
        open("Vulkan");
        probe("Vulkan");
    }
}

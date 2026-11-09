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
import com.eonmux.neuralfree.gguf.kernels.LlmKernelConstants;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
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
import java.util.Random;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertTrue;

/**
 * The float-input multi-column matvec kernels ({@code {T}_MatVecN}) are bitwise the stacked
 * single-row matvecs: for every token count of the dispatch range the one-launch output equals the
 * per-token {@code {T}_MatVec} outputs element for element, on every available GPU backend, for the
 * float-input weight types the projections use. The kernel keeps each token's partial sum in the
 * same element walk and reduces it in the same work-group tree as the single-row twin, so any
 * reordering would show here as a differing bit pattern. Weights are random blocks with finite
 * scales. A backend is skipped when its GPU or its kernel library is unavailable.
 */
@RunWith(Parameterized.class)
public class FloatMatvecNParityGpuTest {

    private static final int COLS = 512;
    private static final int ROWS = 12;
    /** Token counts the dispatch serves the multi-column matvec at: odd groups and the bounds. */
    private static final int[] TOKEN_COUNTS = {2, 3, 5, 9, LlmKernelConstants.MATVEC_N_MAX_TOKENS};

    /** The float-input weight types with a {@code {T}_MatVecN} kernel the models use. */
    private static final GGMLType[] TYPES = {GGMLType.Q8_0, GGMLType.F16};

    /** Every OpenCL device (their subgroup support differs, and with it the reduction form), CUDA and Vulkan. */
    @Parameterized.Parameters(name = "{0}")
    public static List<Object[]> backends() {
        List<Object[]> apis = new ArrayList<>();
        int openClDevices = Math.max(1, openClDeviceCount());
        for (int i = 0; i < openClDevices; i++) {
            apis.add(new Object[]{OPENCL_PREFIX + i});
        }
        apis.add(new Object[]{"cuda"});
        apis.add(new Object[]{"vulkan"});
        return apis;
    }

    private static int openClDeviceCount() {
        HWAcceleration hw = new HWAcceleration(EnumSet.of(ComputeBackend.BackendType.OPENCL));
        try {
            return hw.getCLDevices().size();
        } catch (RuntimeException e) {
            return 0;
        } finally {
            hw.shutdown();
        }
    }

    private static final String OPENCL_PREFIX = "opencl:";

    private final String api;
    private HWAcceleration hwAccel;
    private ComputeBackend backend;
    private LlmKernels kernels;
    private int local;
    private boolean previousCuda;

    public FloatMatvecNParityGpuTest(String api) {
        this.api = api;
    }

    @Before
    public void openDevice() {
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
            hwAccel = new HWAcceleration();
            Assume.assumeFalse("no CUDA device", hwAccel.getCUDevices().isEmpty());
            CUDevice device = hwAccel.getCUDevices().get(0);
            return new CudaComputeBackend(device);
        }
        int index = Integer.parseInt(api.substring(OPENCL_PREFIX.length()));
        HWAcceleration accel = new HWAcceleration();
        hwAccel = accel;
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

    // ------------------------------------------------------------------ fixtures

    /** Random blocks of {@code type} with finite block scales. */
    private static byte[] randomWeights(GGMLType type, long seed) {
        Random rnd = new Random(seed);
        int blockBytes = type.getTypeSize();
        int blockSize = type.getBlockSize();
        int blocks = ROWS * (COLS / blockSize);
        byte[] data = new byte[blocks * blockBytes];
        rnd.nextBytes(data);
        if (type == GGMLType.Q8_0) {
            ByteBuffer scales = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);
            for (int b = 0; b < blocks; b++) {
                scales.putShort(b * blockBytes, (short) 0x3C00);   // f16 1.0: finite scales
            }
            return data;
        }
        ByteBuffer halves = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < blocks * blockBytes / 2; i++) {
            halves.putShort(i * 2, (short) (0x3800 + (rnd.nextInt(16) << 2)));   // f16 near 0.5
        }
        return data;
    }

    /** {@code n} activation rows of {@code cols} elements, row major. */
    private static float[] randomActivations(long seed, int n) {
        Random rnd = new Random(seed);
        float[] x = new float[n * COLS];
        for (int i = 0; i < x.length; i++) {
            x[i] = rnd.nextFloat() * 2f - 1f;
        }
        return x;
    }

    private GpuBuffer upload(byte[] data) throws Exception {
        GpuBuffer buf = backend.allocateBuffer(data.length);
        backend.computeQueue().enqueueWriteBuffer(buf, 0, data.length, data).awaitCompletion();
        return buf;
    }

    private GpuBuffer upload(float[] data) throws Exception {
        GpuBuffer buf = backend.allocateBuffer((long) data.length * 4);
        backend.computeQueue().enqueueWriteBuffer(buf, 0, (long) data.length * 4, data).awaitCompletion();
        return buf;
    }

    private float[] readFloats(GpuBuffer buf, int n) throws Exception {
        float[] out = new float[n];
        backend.computeQueue().enqueueReadBuffer(buf, 0, (long) n * 4, out).awaitCompletion();
        return out;
    }

    // ------------------------------------------------------------------ tests

    /** The one-launch multi-column output equals the stacked single-row outputs bitwise. */
    @Test
    public void multiColumnMatvecIsBitwiseTheStackedSingleRowMatvecs() throws Exception {
        for (GGMLType type : TYPES) {
            String kernel = type.name() + "_MatVecN";
            String single = type.name() + "_MatVec";
            assertTrue(api + " build has no " + kernel, kernels.has(kernel));
            assertTrue(api + " build has no " + single, kernels.has(single));
            for (int n : TOKEN_COUNTS) {
                byte[] weights = randomWeights(type, 40000L + n);
                float[] x = randomActivations(41000L + n, n);
                GpuBuffer wBuf = upload(weights);
                GpuBuffer xBuf = upload(x);
                GpuBuffer yN = backend.allocateBuffer((long) n * ROWS * 4);
                try {
                    backend.computeQueue().enqueueKernel(kernels.get(kernel),
                            new long[]{(long) ROWS * local}, new long[]{local},
                            new Object[]{ROWS, COLS, n, wBuf, xBuf, yN});
                    backend.computeQueue().finish();
                    float[] got = readFloats(yN, n * ROWS);
                    for (int t = 0; t < n; t++) {
                        GpuBuffer rowBuf = upload(Arrays.copyOfRange(x, t * COLS, (t + 1) * COLS));
                        GpuBuffer yRow = backend.allocateBuffer((long) ROWS * 4);
                        try {
                            backend.computeQueue().enqueueKernel(kernels.get(single),
                                    new long[]{(long) ROWS * local}, new long[]{local},
                                    new Object[]{ROWS, COLS, wBuf, rowBuf, yRow});
                            backend.computeQueue().finish();
                            float[] singleGot = readFloats(yRow, ROWS);
                            for (int r = 0; r < ROWS; r++) {
                                if (got[t * ROWS + r] != singleGot[r]) {
                                    assertArrayEquals(api + " " + type + " n=" + n + " token " + t
                                                    + " row " + r + " bitwise",
                                            new float[]{singleGot[r]}, new float[]{got[t * ROWS + r]}, 0f);
                                }
                            }
                        } finally {
                            rowBuf.close();
                            yRow.close();
                        }
                    }
                } finally {
                    wBuf.close();
                    xBuf.close();
                    yN.close();
                }
            }
        }
    }
}

package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.HardwareDevice;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.cuda.CudaComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ocl.OclComputeBackend;
import com.eonmux.neuralfree.acceleration.cuda.CUDevice;
import com.eonmux.neuralfree.acceleration.kernels.Availability;
import com.eonmux.neuralfree.acceleration.vulkan.VkDeviceRanking;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.ops.norm.NormKernels;
import com.eonmux.neuralfree.gguf.ops.norm.NormKernels.MixMode;
import com.eonmux.neuralfree.gguf.ops.norm.NormMath;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.EnumSet;
import java.util.List;
import java.util.Random;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/**
 * The Q8_1-emitting row norms ({@code LLM_RowNormQ8}, {@code LLM_ResidualMixNormQ8}) against the two launches they
 * fold: their plain forms ({@code LLM_RowNorm}, {@code LLM_ResidualMixNorm}) followed by
 * {@code LLM_QuantizeQ8_1} over the normalized rows. The contract is bit-exact: the same y (and mixed x) floats and
 * the same Q8_1 bytes, for every row-norm mode, with and without weight and bias, one and several rows, and row
 * lengths that are and are not a multiple of the work-group size. The quantizer runs at both grid forms the host
 * may launch (one work-item per block, and the 32-lane-per-block grid of the shuffle backends). Runs on CUDA,
 * OpenCL and Vulkan (the subgroup-shuffle, the local-memory and, on a 32-lane subgroup device, the subgroup
 * segment reductions); a backend is skipped when its GPU or its kernel library is unavailable. The plain forms are also held to the CPU reference ({@link NormMath}) at row lengths
 * inside, at and past the kernels' register cache, so both of their row passes are covered.
 */
@RunWith(Parameterized.class)
public class Q8NormFoldGpuTest {

    private static final int Q8_1_BYTES = 36;
    /** Rows inside the norm kernels' register cache (16 elements per work-item), at its limit and past it. */
    private static final int[] WIDTHS = {96, 1024, 2560, 4096, 4128};
    private static final int[] ROWS = {1, 3};

    @Parameterized.Parameters(name = "{0}")
    public static List<Object[]> backends() {
        return Arrays.asList(new Object[]{"cuda"}, new Object[]{"opencl"}, new Object[]{"vulkan"});
    }

    private final String api;
    private HWAcceleration hwAccel;
    private ComputeBackend backend;
    private LlmKernels kernels;
    private int local;
    private boolean previousCuda;
    private final List<GpuBuffer> owned = new ArrayList<>();

    public Q8NormFoldGpuTest(String api) {
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
            local = kernels.rowReduceLocal();
        } catch (Throwable e) {
            closeDevice();
            Assume.assumeTrue("no usable " + api + " GPU: " + e.getMessage(), false);
        }
    }

    private ComputeBackend openBackend() {
        if (api.equals("cuda")) {
            HWAcceleration.enableCuda = true;
            HWAcceleration accel = new HWAcceleration();
            hwAccel = accel;
            Assume.assumeFalse("no CUDA device", accel.getCUDevices().isEmpty());
            CUDevice device = accel.getCUDevices().get(0);
            return new CudaComputeBackend(device);
        }
        if (api.equals("vulkan")) {
            GpuTestUtil.assumeVulkanAvailable();
            HWAcceleration accel = new HWAcceleration(EnumSet.of(ComputeBackend.BackendType.VULKAN));
            hwAccel = accel;
            HardwareDevice device = VkDeviceRanking.fastestDevice(accel.getVkDevices(), null);
            Assume.assumeTrue("no Vulkan device enumerated", device != null);
            return accel.createBackend(device);
        }
        HWAcceleration accel = new HWAcceleration();
        hwAccel = accel;
        Assume.assumeFalse("no OpenCL device", accel.getCLDevices().isEmpty());
        return new OclComputeBackend(accel.getCLDevices().get(0));
    }

    @After
    public void closeDevice() {
        for (GpuBuffer b : owned) {
            b.close();
        }
        owned.clear();
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
    public void rowNormQ8MatchesRowNormThenQuantizer() throws Exception {
        Random rnd = new Random(41);
        for (NormKind kind : rowNormKinds()) {
            for (int width : WIDTHS) {
                for (int rows : ROWS) {
                    for (int weights = 0; weights < 3; weights++) {
                        String where = kind + " width " + width + " rows " + rows + " weights " + weights;
                        float[] x = activations(rnd, rows * width);
                        float[] wHost = weights >= 1 ? activations(rnd, width) : null;
                        float[] bHost = weights >= 2 ? activations(rnd, width) : null;
                        GpuBuffer w = wHost != null ? upload(wHost) : null;
                        GpuBuffer b = bHost != null ? upload(bHost) : null;
                        GpuBuffer placeholder = upload(new float[1]);

                        GpuBuffer xRef = upload(x);
                        GpuBuffer yRef = upload(new float[rows * width]);
                        launchNorm(NormKernels.ROW_NORM, rows, NormKernels.rowNormArgs(kind, 1e-6f, rows, width,
                                width, xRef, 0, yRef, 0, w, b, 1, placeholder));
                        float[] want = new float[rows * width];
                        NormMath.rowNorm(kind, 1e-6f, rows, width, width, 1, x, 0, wHost, bHost, want, 0);
                        assertNear(where + ": y against the CPU reference", want, readFloats(yRef, rows * width));
                        byte[] qRef = quantize(yRef, rows * width, false);
                        byte[] qRefWide = quantize(yRef, rows * width, true);
                        assertArrayEquals(where + ": the two quantizer grids", qRef, qRefWide);

                        GpuBuffer xFold = upload(x);
                        GpuBuffer yFold = upload(new float[rows * width]);
                        GpuBuffer q8 = allocate((long) rows * (width / 32) * Q8_1_BYTES);
                        Object[] args = NormKernels.rowNormArgs(kind, 1e-6f, rows, width, width, xFold, 0, yFold, 0,
                                w, b, 1, placeholder);
                        launchNorm(NormKernels.ROW_NORM_Q8, rows, NormKernels.withQ8Blocks(args, rows, width, q8));

                        assertArrayEquals(where + ": y", readFloats(yRef, rows * width),
                                readFloats(yFold, rows * width), 0.0f);
                        assertArrayEquals(where + ": Q8_1 blocks", qRef, readBytes(q8, qRef.length));
                    }
                }
            }
        }
    }

    @Test
    public void residualMixNormQ8MatchesFusedNormThenQuantizer() throws Exception {
        Random rnd = new Random(43);
        for (MixMode mix : new MixMode[]{MixMode.ADD, MixMode.SCALED}) {
            for (NormKind kind : rowNormKinds()) {
                for (int width : WIDTHS) {
                    for (int rows : ROWS) {
                        String where = mix + " " + kind + " width " + width + " rows " + rows;
                        float s = mix == MixMode.SCALED ? 0.37f : 1.0f;
                        float[] x = activations(rnd, rows * width);
                        float[] addend = activations(rnd, rows * width);
                        float[] wHost = activations(rnd, width);
                        GpuBuffer w = upload(wHost);
                        GpuBuffer placeholder = upload(new float[1]);

                        GpuBuffer xRef = upload(x);
                        GpuBuffer aRef = upload(addend);
                        GpuBuffer yRef = upload(new float[rows * width]);
                        launchNorm(NormKernels.RESIDUAL_MIX_NORM, rows, NormKernels.residualMixNormArgs(mix, s, kind,
                                1e-6f, rows, width, 0, 0, 0, xRef, aRef, w, null, 1, yRef, placeholder));
                        float[] mixed = new float[rows * width];
                        for (int i = 0; i < mixed.length; i++) {
                            mixed[i] = mix == MixMode.SCALED ? x[i] + s * addend[i] : x[i] + addend[i];
                        }
                        float[] want = new float[rows * width];
                        NormMath.rowNorm(kind, 1e-6f, rows, width, width, 1, mixed, 0, wHost, null, want, 0);
                        assertNear(where + ": mixed x against the CPU reference", mixed,
                                readFloats(xRef, rows * width));
                        assertNear(where + ": y against the CPU reference", want, readFloats(yRef, rows * width));
                        byte[] qRef = quantize(yRef, rows * width, true);

                        GpuBuffer xFold = upload(x);
                        GpuBuffer aFold = upload(addend);
                        GpuBuffer yFold = upload(new float[rows * width]);
                        GpuBuffer q8 = allocate((long) rows * (width / 32) * Q8_1_BYTES);
                        Object[] args = NormKernels.residualMixNormArgs(mix, s, kind, 1e-6f, rows, width, 0, 0, 0,
                                xFold, aFold, w, null, 1, yFold, placeholder);
                        launchNorm(NormKernels.RESIDUAL_MIX_NORM_Q8, rows,
                                NormKernels.withQ8Blocks(args, rows, width, q8));

                        assertArrayEquals(where + ": mixed x", readFloats(xRef, rows * width),
                                readFloats(xFold, rows * width), 0.0f);
                        assertArrayEquals(where + ": y", readFloats(yRef, rows * width),
                                readFloats(yFold, rows * width), 0.0f);
                        assertArrayEquals(where + ": Q8_1 blocks", qRef, readBytes(q8, qRef.length));
                    }
                }
            }
        }
    }

    // ------------------------------------------------------------------ helpers

    /** Element-wise agreement within float rounding of a row reduction: 1e-4 relative to max(1, |want|). */
    private static void assertNear(String where, float[] want, float[] got) {
        for (int i = 0; i < want.length; i++) {
            assertEquals(where + " at " + i, want[i], got[i], 1e-4f * Math.max(1.0f, Math.abs(want[i])));
        }
    }

    private static List<NormKind> rowNormKinds() {
        List<NormKind> kinds = new ArrayList<>();
        for (NormKind k : NormKind.values()) {
            if (k.isRowNorm()) {
                kinds.add(k);
            }
        }
        return kinds;
    }

    /** Activations with a heavy-tailed spread, a few zero blocks' worth of exact zeros and both signs. */
    private static float[] activations(Random rnd, int n) {
        float[] v = new float[n];
        for (int i = 0; i < n; i++) {
            double g = rnd.nextGaussian();
            v[i] = (float) (g * Math.exp(rnd.nextGaussian()));
        }
        for (int i = 0; i < Math.min(32, n); i++) {
            v[i] = 0.0f;
        }
        return v;
    }

    private void launchNorm(String kernel, int rows, Object[] args) {
        ComputeQueue q = backend.computeQueue();
        q.enqueueKernel(kernels.get(kernel), new long[]{(long) rows * local}, new long[]{local}, args);
        q.finish();
    }

    /** {@code LLM_QuantizeQ8_1} over {@code y[0 .. size)}: one work-item per block, or 32 per block when wide. */
    private byte[] quantize(GpuBuffer y, int size, boolean wide) throws Exception {
        int blocks = size / 32;
        GpuBuffer out = allocate((long) blocks * Q8_1_BYTES);
        int matvecLocal = kernels.matvecLocal();
        long items = wide ? (long) blocks * 32 : blocks;
        long global = (items + matvecLocal - 1) / matvecLocal * matvecLocal;
        ComputeQueue q = backend.computeQueue();
        q.enqueueKernel(kernels.get(GpuOps.QUANTIZE_Q8_1), new long[]{global}, new long[]{matvecLocal},
                new Object[]{size, y, out});
        q.finish();
        return readBytes(out, blocks * Q8_1_BYTES);
    }

    private GpuBuffer allocate(long bytes) {
        GpuBuffer b = backend.allocateBuffer(bytes);
        owned.add(b);
        return b;
    }

    private GpuBuffer upload(float[] data) throws Exception {
        GpuBuffer b = allocate((long) data.length * 4);
        backend.computeQueue().enqueueWriteBuffer(b, 0, (long) data.length * 4, data).awaitCompletion();
        return b;
    }

    private float[] readFloats(GpuBuffer b, int n) throws Exception {
        float[] out = new float[n];
        backend.computeQueue().enqueueReadBuffer(b, 0, (long) n * 4, out).awaitCompletion();
        return out;
    }

    private byte[] readBytes(GpuBuffer b, int n) throws Exception {
        byte[] out = new byte[n];
        backend.computeQueue().enqueueReadBuffer(b, 0, n, out).awaitCompletion();
        return out;
    }
}

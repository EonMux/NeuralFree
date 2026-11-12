package com.eonmux.neuralfree.gguf.ops.fused;

import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.cuda.CudaComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ocl.OclComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.vulkan.VkComputeBackend;
import com.eonmux.neuralfree.acceleration.cuda.CUDevice;
import com.eonmux.neuralfree.acceleration.kernels.Availability;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.arch.RopeType;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuScratch;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlots;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuPlaceholders;
import com.eonmux.neuralfree.gguf.ops.deltanet.DeltaNetKernels;
import com.eonmux.neuralfree.gguf.ops.norm.NormKernels;
import com.eonmux.neuralfree.gguf.spec.RopeSpec;
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
import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * The persistent multi-task kernel {@code LLM_MultiTask} on every GPU backend: the row-norm and norm+rotation
 * bodies against the individual kernels they replace, at a step shorter than the micro-batch the launch was built
 * for (the launch then shrinks to the step's rows). The tolerance covers only the float summation order of the
 * reductions. A backend is skipped when its GPU or its kernel library is unavailable.
 */
@RunWith(Parameterized.class)
public class MultiTaskBackendsTest {

    private static final float EPS = 1e-6f;
    private static final float TOL = 2e-5f;

    @Parameterized.Parameters(name = "{0}")
    public static List<Object[]> backends() {
        return Arrays.asList(new Object[]{"opencl"}, new Object[]{"cuda"}, new Object[]{"vulkan"});
    }

    private final String api;
    private HWAcceleration hwAccel;
    private ComputeBackend backend;
    private GpuOps ops;
    private GpuScratch scratch;
    private GpuPlaceholders placeholders;
    private int local;
    private boolean previousCuda;

    public MultiTaskBackendsTest(String api) {
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
            LlmKernels kernels = LlmKernels.open(backend);
            local = kernels.matvecLocal();
            scratch = GpuScratch.builder(backend, 8).slot(ScratchSlots.X, 8L * 64).build();
            placeholders = GpuPlaceholders.allocate(backend, backend.computeQueue());
            ops = new GpuOps(backend.computeQueue(), kernels, scratch, placeholders);
        } catch (Throwable e) {
            closeDevice();
            Assume.assumeTrue("no usable " + api + " GPU: " + e.getMessage(), false);
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
            hwAccel.shutdown();
            hwAccel = null;
        }
        HWAcceleration.enableCuda = previousCuda;
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
        hwAccel = new HWAcceleration();
        Assume.assumeFalse("no OpenCL device", hwAccel.getCLDevices().isEmpty());
        return new OclComputeBackend(hwAccel.getCLDevices().get(0));
    }

    // ------------------------------------------------------------------ fixtures

    private GpuBuffer floats(float[] data) {
        GpuBuffer b = backend.allocateBuffer((long) data.length * Float.BYTES);
        backend.computeQueue().enqueueWriteBuffer(b, 0, (long) data.length * Float.BYTES, data);
        return b;
    }

    private GpuBuffer ints(int[] data) {
        ByteBuffer le = ByteBuffer.allocate(data.length * Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        le.asIntBuffer().put(data);
        GpuBuffer b = backend.allocateBuffer(le.capacity());
        backend.computeQueue().enqueueWriteBuffer(b, 0, le.capacity(), le);
        return b;
    }

    private float[] read(GpuBuffer b, int n) {
        float[] out = new float[n];
        backend.computeQueue().enqueueReadBuffer(b, 0, (long) n * Float.BYTES, out);
        backend.computeQueue().finish();
        return out;
    }

    private static float[] random(int n, long seed) {
        Random rnd = new Random(seed);
        float[] v = new float[n];
        for (int i = 0; i < n; i++) {
            v[i] = rnd.nextFloat() * 2f - 1f;
        }
        return v;
    }

    private static void near(float[] expected, float[] actual, String what) {
        assertEquals(what + " length", expected.length, actual.length);
        for (int i = 0; i < expected.length; i++) {
            assertEquals(what + "[" + i + "]", expected[i], actual[i], TOL * (1f + Math.abs(expected[i])));
        }
    }

    private void run(String kernel, long global, Object[] args) {
        backend.computeQueue().enqueueKernel(ops.kernels().get(kernel), new long[]{global}, new long[]{local}, args);
    }

    // ------------------------------------------------------------------ tests

    @Test
    public void multiTaskRowNormAndNormRopeMatchTheIndividualKernels() {
        Assume.assumeTrue("backend has no multi-task kernel on " + api,
                GpuMultiTask.usable(ops) && GpuMultiTask.localUsable(local));
        int nTokens = 2;
        int nHeads = 2;
        int headDim = 8;
        int rowStride = nHeads * headDim;
        int total = nTokens * rowStride;
        float[] x = random(total, 31);
        float[] w = random(nHeads * headDim, 32);
        RopeSpec rope = RopeSpec.builder(RopeType.NEOX).nDims(headDim).nOffs(0).freqBase(10000f).build();

        GpuBuffer xb = floats(x);
        GpuBuffer wb = floats(w);
        GpuBuffer none = placeholders.readOnlyFloats();
        GpuBuffer pos = ints(new int[]{2, 5});
        GpuBuffer refNorm = backend.allocateBuffer((long) total * Float.BYTES);
        GpuBuffer refRope = backend.allocateBuffer((long) total * Float.BYTES);
        GpuBuffer norm = backend.allocateBuffer((long) total * Float.BYTES);
        GpuBuffer rotated = backend.allocateBuffer((long) total * Float.BYTES);

        run(NormKernels.ROW_NORM, (long) nTokens * nHeads * local,
                NormKernels.rowNormArgs(NormKind.RMS, EPS, nTokens * nHeads, headDim, headDim, xb, 0, refNorm, 0,
                        wb, null, nHeads, none));
        run(FusedKernels.RMS_NORM_ROPE, FusedKernels.rmsNormRopeGlobal(nTokens, nHeads, local),
                FusedKernels.rmsNormRopeArgs(rope, nTokens, nHeads, headDim, rowStride, headDim, 0, nTokens, 0, 0,
                        NormKernels.FLAG_HAS_W, nHeads, EPS, false, pos, none, wb, none, xb, refRope));

        GpuMultiTask batch = new GpuMultiTask(ops, nTokens + 3, "backend.test");
        try {
            batch.rowNorm(nHeads, headDim, headDim, NormKind.RMS, NormKernels.FLAG_HAS_W, nHeads, 0, 0, EPS, xb, wb,
                    none, norm);
            batch.normRope(rope, EPS, nHeads, headDim, rowStride, headDim, 0, 0, 0, NormKernels.FLAG_HAS_W, nHeads,
                    false, pos, none, wb, none, xb, rotated);
            batch.build();
            batch.run(nTokens);
        } finally {
            batch.close();
        }

        near(read(refNorm, total), read(norm, total), "multi-task row norm");
        near(read(refRope, total), read(rotated, total), "multi-task norm + rope");
        assertTrue("multi-task kernel selected on " + api, ops.fused(FusedKernels.MULTI_TASK));
    }

    /** The gate-preparation and dense F32 matvec bodies against the individual kernels that own them. */
    @Test
    public void multiTaskGatePrepAndMatvecMatchTheIndividualKernels() {
        Assume.assumeTrue("backend has no multi-task kernel on " + api,
                GpuMultiTask.usable(ops) && GpuMultiTask.localUsable(local));
        int nTokens = 2;
        int nV = 2;
        int nK = 1;
        int headDim = 1;
        int total = nTokens * nV;
        float[] alpha = random(total, 41);
        float[] beta = random(nV, 42);
        float[] dtBias = random(nV, 43);
        float[] ssmA = random(nV, 44);
        GpuBuffer ab = floats(alpha);
        GpuBuffer bb = floats(beta);
        GpuBuffer dtb = floats(dtBias);
        GpuBuffer a = floats(ssmA);
        GpuBuffer refG = backend.allocateBuffer((long) total * Float.BYTES);
        GpuBuffer refBeta = backend.allocateBuffer((long) total * Float.BYTES);
        GpuBuffer g = backend.allocateBuffer((long) total * Float.BYTES);
        GpuBuffer b = backend.allocateBuffer((long) total * Float.BYTES);

        run(DeltaNetKernels.GDN_GATE_PREP, DeltaNetKernels.gatePrepGlobal(nTokens, nV, -1, headDim, local),
                new Object[]{nTokens, nV, nK, 0, -1, headDim, 0f, nV, 0, nV, 0, 1.0f, ab, bb, dtb, a, refG,
                        refBeta});
        GpuMultiTask batch = new GpuMultiTask(ops, nTokens + 3, "backend.gate");
        try {
            batch.gatePrep(nV, nK, 0, -1, headDim, nV, 0, nV, 0, 0f, 1.0f, ab, bb, dtb, a, g, b);
            batch.build();
            batch.run(nTokens);
        } finally {
            batch.close();
        }
        near(read(refG, total), read(g, total), "multi-task gate");
        near(read(refBeta, total), read(b, total), "multi-task beta");

        int rows = 4;
        int cols = 16;
        float[] x = random(cols, 45);
        float[] weights = random(rows * cols, 46);
        GpuBuffer xb = floats(x);
        GpuBuffer wb = floats(weights);
        GpuBuffer ref = backend.allocateBuffer((long) rows * Float.BYTES);
        GpuBuffer out = backend.allocateBuffer((long) rows * Float.BYTES);
        run(GpuOps.matVecKernel(GGMLType.F32), (long) rows * local, new Object[]{rows, cols, wb, xb, ref});
        GpuWeight w = GpuWeight.of("backend.mv", GGMLType.F32, new long[]{cols, rows}, wb);
        GpuMultiTask mv = new GpuMultiTask(ops, nTokens, "backend.matvec");
        try {
            mv.matvecF32(w, xb, 0, out, 0);
            mv.build();
            mv.run(1);
        } finally {
            mv.close();
        }
        near(read(ref, rows), read(out, rows), "multi-task matvec");
    }
}

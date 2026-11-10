package com.eonmux.neuralfree.gguf.ops.fused;

import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend.BackendType;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.ocl.OclComputeBackend;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.arch.GluOp;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.arch.RopeType;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuScratch;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlots;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuPlaceholders;
import com.eonmux.neuralfree.gguf.ops.act.ActKernels;
import com.eonmux.neuralfree.gguf.ops.deltanet.DeltaNetKernels;
import com.eonmux.neuralfree.gguf.ops.hc.HcKernels;
import com.eonmux.neuralfree.gguf.ops.norm.NormKernels;
import com.eonmux.neuralfree.gguf.ops.rope.RopeKernels;
import com.eonmux.neuralfree.gguf.spec.ActParams;
import com.eonmux.neuralfree.gguf.spec.RopeSpec;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.EnumSet;
import java.util.Random;

import static com.eonmux.neuralfree.acceleration.GpuTestUtil.isOpenCLAvailable;
import static org.junit.Assert.assertEquals;

/**
 * The fused decode-path kernels of {@code kernels/llm/fused.cl} against the unfused kernel pairs they replace, and
 * the persistent multi-task kernel {@code LLM_MultiTask} against the individual kernels. Runs on OpenCL, the backend
 * that provides the multi-task kernel; the tolerance covers only the float summation order of the reductions.
 */
public class FusedKernelsGpuTest {

    private static final float TOL = 2e-5f;
    private static final float EPS = 1e-6f;

    private HWAcceleration accel;
    private ComputeBackend backend;
    private LlmKernels kernels;
    private GpuOps ops;
    private GpuScratch scratch;
    private GpuPlaceholders placeholders;
    private int local;

    @Before
    public void open() {
        Assume.assumeTrue("OpenCL GPU not available", isOpenCLAvailable());
        accel = new HWAcceleration(EnumSet.of(BackendType.OPENCL));
        Assume.assumeFalse("no OpenCL device", accel.getCLDevices().isEmpty());
        backend = new OclComputeBackend(accel.getCLDevices().get(0));
        kernels = LlmKernels.open(backend);
        local = kernels.matvecLocal();
        scratch = GpuScratch.builder(backend, 8).slot(ScratchSlots.X, 8L * 64).build();
        placeholders = GpuPlaceholders.allocate(backend, backend.computeQueue());
        ops = new GpuOps(backend.computeQueue(), kernels, scratch, placeholders);
    }

    @After
    public void close() {
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
            backend.close();
            backend = null;
        }
        if (accel != null) {
            accel.shutdown();
            accel = null;
        }
    }

    // ------------------------------------------------------------------ fixtures

    private static float[] random(int n, long seed) {
        Random rnd = new Random(seed);
        float[] v = new float[n];
        for (int i = 0; i < n; i++) {
            v[i] = rnd.nextFloat() * 2f - 1f;
        }
        return v;
    }

    private GpuBuffer buffer(float[] data) throws Exception {
        GpuBuffer b = backend.allocateBuffer((long) data.length * Float.BYTES);
        backend.computeQueue().enqueueWriteBuffer(b, 0, (long) data.length * Float.BYTES, data).awaitCompletion();
        return b;
    }

    private GpuBuffer ints(int[] data) throws Exception {
        GpuBuffer b = backend.allocateBuffer((long) data.length * Integer.BYTES);
        ByteBuffer le = ByteBuffer.allocate(data.length * Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        le.asIntBuffer().put(data);
        backend.computeQueue().enqueueWriteBuffer(b, 0, le.capacity(), le).awaitCompletion();
        return b;
    }

    private float[] read(GpuBuffer b, int n) throws Exception {
        backend.computeQueue().finish();
        float[] out = new float[n];
        backend.computeQueue().enqueueReadBuffer(b, 0, (long) n * Float.BYTES, out).awaitCompletion();
        return out;
    }

    private byte[] readBytes(GpuBuffer b, int n) throws Exception {
        backend.computeQueue().finish();
        byte[] out = new byte[n];
        backend.computeQueue().enqueueReadBuffer(b, 0, n, out).awaitCompletion();
        return out;
    }

    /** {@code count} 4-byte words of {@code value}. */
    private GpuBuffer filled(GpuBuffer b, int count, float value) throws Exception {
        float[] v = new float[count];
        java.util.Arrays.fill(v, value);
        backend.computeQueue().enqueueWriteBuffer(b, 0, (long) count * Float.BYTES, v).awaitCompletion();
        return b;
    }

    private void run(String kernel, long global, Object[] args) {
        runAt(kernel, global, local, args);
    }

    private void runAt(String kernel, long global, int workGroup, Object[] args) {
        backend.computeQueue().enqueueKernel(kernels.get(kernel), new long[]{global}, new long[]{workGroup}, args);
    }

    private static void near(float[] expected, float[] actual, String what) {
        assertEquals(what + " length", expected.length, actual.length);
        for (int i = 0; i < expected.length; i++) {
            assertEquals(what + "[" + i + "]", expected[i], actual[i], TOL * (1f + Math.abs(expected[i])));
        }
    }

    private static RopeSpec neox(int nDims) {
        return RopeSpec.builder(RopeType.NEOX).nDims(nDims).nOffs(0).freqBase(10000f).build();
    }

    // ------------------------------------------------------------------ RMS norm + RoPE

    @Test
    public void rmsNormRopeMatchesRowNormThenRope() throws Exception {
        int nTokens = 2;
        int nHeads = 2;
        int headDim = 8;
        int rowStride = nHeads * headDim;
        int total = nTokens * rowStride;
        RopeSpec rope = neox(headDim);
        float[] x = random(total, 11);
        float[] w = random(nHeads * headDim, 12);
        float[] b = random(nHeads * headDim, 13);
        GpuBuffer ff = placeholders.readOnlyFloats();
        GpuBuffer posb = ints(new int[]{3, 5});

        GpuBuffer xb = buffer(x);
        GpuBuffer wb = buffer(w);
        GpuBuffer bb = buffer(b);
        GpuBuffer ref = buffer(x);
        GpuBuffer out = backend.allocateBuffer((long) total * Float.BYTES);

        // Unfused: the RMS norm of nTokens*nHeads dense rows of headDim values, then the rotation of those rows.
        run(NormKernels.ROW_NORM, (long) nTokens * nHeads * local,
                NormKernels.rowNormArgs(NormKind.RMS, EPS, nTokens * nHeads, headDim, headDim, ref, 0, ref, 0, wb,
                        bb, nHeads, placeholders.readOnlyFloats()));
        run(RopeKernels.ROPE_EXT,
                ActKernels.global1d(RopeKernels.ropeExtWorkItems(rope.type(), nTokens, nHeads, rope.nDims()), local)[0],
                RopeKernels.ropeExtArgs(rope, nTokens, nHeads, headDim, rowStride, 0, nTokens, 0, false, posb, ff,
                        ref));

        run(FusedKernels.RMS_NORM_ROPE, FusedKernels.rmsNormRopeGlobal(nTokens, nHeads, local),
                FusedKernels.rmsNormRopeArgs(rope, nTokens, nHeads, headDim, rowStride, headDim, 0, nTokens, 0, 0,
                        NormKernels.FLAG_HAS_W | NormKernels.FLAG_HAS_B, nHeads, EPS, false, posb, ff, wb, bb, xb, out));

        near(read(ref, total), read(out, total), "rms norm + rope");
    }

    /**
     * The fused kernel over one-warp workgroups (32 work-items for a 128-value head row, the attention mixer's
     * per-row workgroup) equals the row norm and rotation at the device's matvec workgroup.
     */
    @Test
    public void rmsNormRopeOverOneWarpRowsMatchesRowNormThenRope() throws Exception {
        int nTokens = 3;
        int nHeads = 4;
        int headDim = 128;
        int rowLocal = 32;
        Assume.assumeTrue("workgroup of " + local + " is narrower than one warp row", local >= rowLocal);
        int rowStride = nHeads * headDim;
        int total = nTokens * rowStride;
        RopeSpec rope = neox(headDim);
        float[] x = random(total, 31);
        float[] w = random(nHeads * headDim, 32);
        float[] b = random(nHeads * headDim, 33);
        GpuBuffer ff = placeholders.readOnlyFloats();
        GpuBuffer posb = ints(new int[]{0, 7, 511});
        GpuBuffer xb = buffer(x);
        GpuBuffer wb = buffer(w);
        GpuBuffer bb = buffer(b);
        GpuBuffer ref = buffer(x);
        GpuBuffer out = backend.allocateBuffer((long) total * Float.BYTES);

        run(NormKernels.ROW_NORM, (long) nTokens * nHeads * local,
                NormKernels.rowNormArgs(NormKind.RMS, EPS, nTokens * nHeads, headDim, headDim, ref, 0, ref, 0, wb,
                        bb, nHeads, placeholders.readOnlyFloats()));
        run(RopeKernels.ROPE_EXT,
                ActKernels.global1d(RopeKernels.ropeExtWorkItems(rope.type(), nTokens, nHeads, rope.nDims()), local)[0],
                RopeKernels.ropeExtArgs(rope, nTokens, nHeads, headDim, rowStride, 0, nTokens, 0, false, posb, ff,
                        ref));
        runAt(FusedKernels.RMS_NORM_ROPE, FusedKernels.rmsNormRopeGlobal(nTokens, nHeads, rowLocal), rowLocal,
                FusedKernels.rmsNormRopeArgs(rope, nTokens, nHeads, headDim, rowStride, headDim, 0, nTokens, 0, 0,
                        NormKernels.FLAG_HAS_W | NormKernels.FLAG_HAS_B, nHeads, EPS, false, posb, ff, wb, bb, xb,
                        out));

        near(read(ref, total), read(out, total), "rms norm + rope over one-warp rows");
    }

    // ------------------------------------------------------------------ GDN norm + gate

    @Test
    public void gdnNormGateMatchesRowNormThenSwiglu() throws Exception {
        int nTokens = 3;
        int nHeads = 2;
        int headDim = 8;
        int rowStride = nHeads * headDim;
        int total = nTokens * rowStride;
        float[] x = random(total, 21);
        float[] z = random(total, 22);
        float[] w = random(nHeads * headDim, 23);

        GpuBuffer xb = buffer(x);
        GpuBuffer zb = buffer(z);
        GpuBuffer wb = buffer(w);
        GpuBuffer placeholder = placeholders.readOnlyFloats();
        GpuBuffer ref = backend.allocateBuffer((long) total * Float.BYTES);
        GpuBuffer out = backend.allocateBuffer((long) total * Float.BYTES);

        // Unfused: one RMS norm row per (token, head) with the head's own weight row, then silu(z) * normed.
        run(NormKernels.ROW_NORM, (long) nTokens * nHeads * local,
                NormKernels.rowNormArgs(NormKind.RMS, EPS, nTokens * nHeads, headDim, headDim, xb, 0, ref, 0, wb,
                        null, nHeads, placeholder));
        run(ActKernels.GLU, ActKernels.global1d((long) total, local)[0],
                ActKernels.gluArgs(GluOp.SWIGLU, ActParams.NONE, nTokens * nHeads, headDim, zb, 0, headDim, ref, 0,
                        headDim, out, 0, headDim));

        GpuBuffer fused = backend.allocateBuffer((long) total * Float.BYTES);
        run(FusedKernels.GDN_NORM_GATE, FusedKernels.gdnNormGateGlobal(nTokens, nHeads, local),
                FusedKernels.gdnNormGateArgs(nTokens, nHeads, headDim, rowStride, 0, 0, 0, nHeads, EPS, xb, wb, zb,
                        fused));
        near(read(out, total), read(fused, total), "gdn norm + gate");

        // A single shared weight row: every head multiplies by it.
        GpuBuffer shared = buffer(random(headDim, 24));
        run(NormKernels.ROW_NORM, (long) nTokens * nHeads * local,
                NormKernels.rowNormArgs(NormKind.RMS, EPS, nTokens * nHeads, headDim, headDim, xb, 0, ref, 0, shared,
                        null, 1, placeholder));
        GpuBuffer outShared = backend.allocateBuffer((long) total * Float.BYTES);
        run(ActKernels.GLU, ActKernels.global1d((long) total, local)[0],
                ActKernels.gluArgs(GluOp.SWIGLU, ActParams.NONE, nTokens * nHeads, headDim, zb, 0, headDim, ref, 0,
                        headDim, outShared, 0, headDim));
        GpuBuffer fusedShared = backend.allocateBuffer((long) total * Float.BYTES);
        run(FusedKernels.GDN_NORM_GATE, FusedKernels.gdnNormGateGlobal(nTokens, nHeads, local),
                FusedKernels.gdnNormGateArgs(nTokens, nHeads, headDim, rowStride, 0, 0, 0, 1, EPS, xb, shared, zb,
                        fusedShared));
        near(read(outShared, total), read(fusedShared, total), "gdn norm + gate, shared weight");
    }

    // ------------------------------------------------------------------ hyper-connection expand + norm

    @Test
    public void hcPostNormMatchesHcPostThenRowNorm() throws Exception {
        int nTokens = 2;
        int nEmbd = 64;
        int hc = 2;
        int total = nTokens * hc * nEmbd;
        for (int scope = 0; scope <= 1; scope++) {
            float[] y = random(nTokens * nEmbd, 31 + scope);
            float[] stream = random(total, 41 + scope);
            float[] post = random(nTokens * hc, 51 + scope);
            float[] comb = random(nTokens * hc * hc, 61 + scope);
            float[] gamma = random(hc * nEmbd, 71 + scope);

            GpuBuffer yb = buffer(y);
            GpuBuffer xb = buffer(stream);
            GpuBuffer pb = buffer(post);
            GpuBuffer cb = buffer(comb);
            GpuBuffer gb = buffer(gamma);
            GpuBuffer placeholder = placeholders.readOnlyFloats();
            GpuBuffer refX = backend.allocateBuffer((long) total * Float.BYTES);
            GpuBuffer refN = backend.allocateBuffer((long) total * Float.BYTES);
            GpuBuffer outX = backend.allocateBuffer((long) total * Float.BYTES);
            GpuBuffer outN = backend.allocateBuffer((long) total * Float.BYTES);

            Object[] postArgs = new Object[HcKernels.POST_ARG_COUNT];
            postArgs[HcKernels.POST_ARG_N_TOKENS] = nTokens;
            postArgs[HcKernels.POST_ARG_N_EMBD] = nEmbd;
            postArgs[HcKernels.POST_ARG_HC] = hc;
            postArgs[HcKernels.POST_ARG_HAS_COMB] = 1;
            postArgs[HcKernels.POST_ARG_POST_ACT] = 0;
            postArgs[HcKernels.POST_ARG_POST_IN_SCALE] = 0f;
            postArgs[HcKernels.POST_ARG_POST_MAG] = 0f;
            postArgs[HcKernels.POST_ARG_Y_OFF] = 0;
            postArgs[HcKernels.POST_ARG_X_OFF] = 0;
            postArgs[HcKernels.POST_ARG_POST_OFF] = 0;
            postArgs[HcKernels.POST_ARG_COMB_OFF] = 0;
            postArgs[HcKernels.POST_ARG_XOUT_OFF] = 0;
            postArgs[HcKernels.POST_ARG_Y] = yb;
            postArgs[HcKernels.POST_ARG_X] = xb;
            postArgs[HcKernels.POST_ARG_POST] = pb;
            postArgs[HcKernels.POST_ARG_COMB] = cb;
            postArgs[HcKernels.POST_ARG_XOUT] = refX;
            run(HcKernels.HC_POST, HcKernels.elementGlobal(nTokens, nEmbd, local), postArgs);

            if (scope == 0) {
                run(NormKernels.ROW_NORM, (long) nTokens * hc * local,
                        NormKernels.rowNormArgs(NormKind.RMS, EPS, nTokens * hc, nEmbd, nEmbd, refX, 0, refN, 0, gb,
                                null, hc, placeholder));
            } else {
                run(NormKernels.ROW_NORM, (long) nTokens * local,
                        NormKernels.rowNormArgs(NormKind.RMS, EPS, nTokens, hc * nEmbd, hc * nEmbd, refX, 0, refN, 0,
                                null, null, 1, placeholder));
            }

            run(FusedKernels.HC_POST_NORM, FusedKernels.hcPostNormGlobal(nTokens, local),
                    FusedKernels.hcPostNormArgs(nTokens, nEmbd, hc, true, 0, 0f, 0f, EPS, scope == 0 ? hc : 0, scope,
                            0, 0, 0, 0, 0, 0, yb, xb, pb, cb, outX, gb, outN));

            near(read(refX, total), read(outX, total), "hc post scope " + scope);
            near(read(refN, total), read(outN, total), "hc norm scope " + scope);
        }
    }

    // ------------------------------------------------------------------ persistent multi-task kernel

    @Test
    public void multiTaskRunsIndependentTasksInOneLaunch() throws Exception {
        Assume.assumeTrue("backend has no multi-task kernel", GpuMultiTask.usable(ops));
        Assume.assumeTrue("local size unsupported", GpuMultiTask.localUsable(local));

        int nTokens = 3;
        int nHeads = 2;
        int headDim = 8;
        int rowStride = nHeads * headDim;
        int total = nTokens * rowStride;
        float[] a = random(total, 81);
        float[] c = random(total, 82);
        float[] w = random(nHeads * headDim, 83);
        RopeSpec rope = neox(headDim);

        GpuBuffer ab = buffer(a);
        GpuBuffer cb = buffer(c);
        GpuBuffer wb = buffer(w);
        GpuBuffer placeholder = placeholders.readOnlyFloats();
        GpuBuffer posb = ints(new int[]{2, 4, 6});
        GpuBuffer outA = backend.allocateBuffer((long) total * Float.BYTES);
        GpuBuffer outC = backend.allocateBuffer((long) total * Float.BYTES);
        GpuBuffer outR = backend.allocateBuffer((long) total * Float.BYTES);
        GpuBuffer refA = backend.allocateBuffer((long) total * Float.BYTES);
        GpuBuffer refC = backend.allocateBuffer((long) total * Float.BYTES);
        GpuBuffer refR = backend.allocateBuffer((long) total * Float.BYTES);

        run(NormKernels.ROW_NORM, (long) nTokens * nHeads * local,
                NormKernels.rowNormArgs(NormKind.RMS, EPS, nTokens * nHeads, headDim, headDim, ab, 0, refA, 0, wb,
                        null, nHeads, placeholder));
        run(NormKernels.ROW_NORM, (long) nTokens * nHeads * local,
                NormKernels.rowNormArgs(NormKind.RMS, EPS, nTokens * nHeads, headDim, headDim, cb, 0, refC, 0, wb,
                        null, nHeads, placeholder));
        run(FusedKernels.RMS_NORM_ROPE, FusedKernels.rmsNormRopeGlobal(nTokens, nHeads, local),
                FusedKernels.rmsNormRopeArgs(rope, nTokens, nHeads, headDim, rowStride, headDim, 0, nTokens, 0, 0,
                        NormKernels.FLAG_HAS_W, nHeads, EPS, false, posb, placeholder, wb, placeholder, ab, refR));

        GpuMultiTask batch = new GpuMultiTask(ops, nTokens, "test");
        try {
            batch.rowNorm(nHeads, headDim, headDim, NormKind.RMS, NormKernels.FLAG_HAS_W, nHeads, 0, 0, EPS, ab, wb,
                    placeholder, outA);
            batch.rowNorm(nHeads, headDim, headDim, NormKind.RMS, NormKernels.FLAG_HAS_W, nHeads, 0, 0, EPS, cb, wb,
                    placeholder, outC);
            batch.normRope(rope, EPS, nHeads, headDim, rowStride, headDim, 0, 0, 0, NormKernels.FLAG_HAS_W, nHeads,
                    false, posb, placeholder, wb, placeholder, ab, outR);
            batch.build();
            batch.run(nTokens);
            backend.computeQueue().finish();

            assertEquals(3, batch.taskCount());
            near(read(refA, total), read(outA, total), "multi-task row norm A");
            near(read(refC, total), read(outC, total), "multi-task row norm C");
            near(read(refR, total), read(outR, total), "multi-task norm + rope");
        } finally {
            batch.close();
        }
    }

    /** Two multi-task launches of one step: each reads the descriptor it wrote at build, not the other's. */    @Test
    public void twoMultiTaskLaunchesKeepTheirDescriptors() throws Exception {
        Assume.assumeTrue("backend has no multi-task kernel", GpuMultiTask.usable(ops));
        Assume.assumeTrue("local size unsupported", GpuMultiTask.localUsable(local));

        int nTokens = 2;
        int nHeads = 2;
        int headDim = 8;
        int rowStride = nHeads * headDim;
        int total = nTokens * rowStride;
        float[] a = random(total, 91);
        float[] w = random(nHeads * headDim, 92);
        RopeSpec rope = neox(headDim);
        GpuBuffer ab = buffer(a);
        GpuBuffer wb = buffer(w);
        GpuBuffer placeholder = placeholders.readOnlyFloats();
        GpuBuffer posb = ints(new int[]{1, 3});
        GpuBuffer norms = backend.allocateBuffer((long) total * Float.BYTES);
        GpuBuffer rotated = backend.allocateBuffer((long) total * Float.BYTES);
        GpuBuffer refNorms = backend.allocateBuffer((long) total * Float.BYTES);
        GpuBuffer refRotated = backend.allocateBuffer((long) total * Float.BYTES);

        run(NormKernels.ROW_NORM, (long) nTokens * nHeads * local,
                NormKernels.rowNormArgs(NormKind.RMS, EPS, nTokens * nHeads, headDim, headDim, ab, 0, refNorms, 0,
                        wb, null, nHeads, placeholder));
        run(FusedKernels.RMS_NORM_ROPE, FusedKernels.rmsNormRopeGlobal(nTokens, nHeads, local),
                FusedKernels.rmsNormRopeArgs(rope, nTokens, nHeads, headDim, rowStride, headDim, 0, nTokens, 0, 0,
                        NormKernels.FLAG_HAS_W, nHeads, EPS, false, posb, placeholder, wb, placeholder, ab,
                        refRotated));

        GpuMultiTask normBatch = new GpuMultiTask(ops, nTokens, "test.norm");
        GpuMultiTask ropeBatch = new GpuMultiTask(ops, nTokens, "test.rope");
        try {
            normBatch.rowNorm(nHeads, headDim, headDim, NormKind.RMS, NormKernels.FLAG_HAS_W, nHeads, 0, 0, EPS,
                    ab, wb, placeholder, norms);
            normBatch.build();
            ropeBatch.normRope(rope, EPS, nHeads, headDim, rowStride, headDim, 0, 0, 0, NormKernels.FLAG_HAS_W,
                    nHeads, false, posb, placeholder, wb, placeholder, ab, rotated);
            ropeBatch.build();
            ropeBatch.run(nTokens);
            normBatch.run(nTokens);
            backend.computeQueue().finish();

            near(read(refNorms, total), read(norms, total), "first launch's row norm");
            near(read(refRotated, total), read(rotated, total), "second launch's norm + rope");
        } finally {
            ropeBatch.close();
            normBatch.close();
        }
    }

    /** The multi-task matvec body against {@code Float_MatVec}, the projection kernel it replaces. */
    @Test
    public void multiTaskMatvecMatchesTheMatVecKernel() throws Exception {
        Assume.assumeTrue("backend has no multi-task kernel", GpuMultiTask.usable(ops));
        int rows = 4;
        int cols = 16;
        float[] w = random(rows * cols, 97);
        float[] x = random(cols, 98);
        GpuBuffer wb = buffer(w);
        GpuBuffer xb = buffer(x);
        GpuBuffer ref = backend.allocateBuffer((long) rows * Float.BYTES);
        GpuBuffer out = backend.allocateBuffer((long) rows * Float.BYTES);
        GpuWeight weight = GpuWeight.of("mt.w", GGMLType.F32, new long[]{cols, rows}, wb);

        run(GpuOps.matVecKernel(GGMLType.F32), (long) rows * local, new Object[]{rows, cols, wb, xb, ref});

        GpuMultiTask batch = new GpuMultiTask(ops, 1, "test.matvec");
        try {
            batch.matvecF32(weight, xb, 0, out, 0);
            batch.build();
            batch.run(1);
            near(read(ref, rows), read(out, rows), "multi-task matvec");
        } finally {
            batch.close();
        }
    }

    /** The multi-task gate-preparation body against {@code LLM_GDN_GatePrep}, with OLMo Hybrid's beta scale. */
    @Test
    public void multiTaskGatePrepMatchesTheGatePrepKernel() throws Exception {
        Assume.assumeTrue("backend has no multi-task kernel", GpuMultiTask.usable(ops));
        int nTokens = 3;
        int nV = 4;
        int nK = 2;
        int headDim = 8;
        int total = nTokens * nV;
        float betaScale = 2f;
        float[] alpha = random(total, 101);
        float[] beta = random(total, 102);
        float[] dtBias = random(nV, 103);
        float[] ssmA = random(nV, 104);
        GpuBuffer ab = buffer(alpha);
        GpuBuffer bb = buffer(beta);
        GpuBuffer dtb = buffer(dtBias);
        GpuBuffer a = buffer(ssmA);
        GpuBuffer refG = backend.allocateBuffer((long) total * Float.BYTES);
        GpuBuffer refBeta = backend.allocateBuffer((long) total * Float.BYTES);
        GpuBuffer g = backend.allocateBuffer((long) total * Float.BYTES);
        GpuBuffer b = backend.allocateBuffer((long) total * Float.BYTES);

        run(DeltaNetKernels.GDN_GATE_PREP,
                DeltaNetKernels.gatePrepGlobal(nTokens, nV, -1, headDim, local),
                new Object[]{nTokens, nV, nK, 0, -1, headDim, 0f, nV, 0, nV, 0, betaScale, ab, bb, dtb, a, refG, refBeta});

        GpuMultiTask batch = new GpuMultiTask(ops, nTokens, "test.gate");
        try {
            batch.gatePrep(nV, nK, 0, -1, headDim, nV, 0, nV, 0, 0f, betaScale, ab, bb, dtb, a, g, b);
            batch.build();
            batch.run(nTokens);
            near(read(refG, total), read(g, total), "multi-task gate");
            near(read(refBeta, total), read(b, total), "multi-task beta");
        } finally {
            batch.close();
        }
    }

    // ------------------------------------------------------------------ fused KV store

    private static final int STORE_CELLS = 3;

    /** One cache row of the fused-store tests: 2 heads of 8 values in a row of 20. */
    private static final int STORE_HEAD = 8;
    private static final int STORE_DST_ROW = 20;

    private GpuBuffer intCells(int[] cells) throws Exception {
        return ints(cells);
    }

    private void runKvStore(KvDType type, int nTokens, int rowElems, GpuBuffer src, GpuBuffer cellIdx,
                            GpuBuffer dst, int dstRowStride) {
        long items = type.perTokenHead()
                ? RopeKernels.kvStoreWorkItemsPerTokenHead(type, nTokens, rowElems, STORE_HEAD)
                : RopeKernels.kvStoreWorkItems(type, nTokens, rowElems);
        run(RopeKernels.kvStore(type), ActKernels.global1d(items, local)[0],
                RopeKernels.kvStoreArgs(type, nTokens, rowElems, rowElems, 0, dstRowStride, STORE_HEAD, src, cellIdx,
                        placeholders.kvScalesOnes(), dst, placeholders.kvScalesOnes(), 0));
    }

    @Test
    public void normRopeStoreMatchesNormRopeThenKvStoreF32() throws Exception {
        Assume.assumeTrue("backend has no norm-rope store", ops.fused(FusedKernels.NORM_ROPE_STORE));
        int nTokens = 2;
        int nHeads = 2;
        int rowStride = nHeads * STORE_HEAD;
        int total = nTokens * rowStride;
        RopeSpec rope = neox(STORE_HEAD);
        float[] x = random(total, 201);
        float[] w = random(nHeads * STORE_HEAD, 202);
        float[] b = random(nHeads * STORE_HEAD, 203);
        int[] cells = {2, 0};
        int cacheElems = STORE_CELLS * STORE_DST_ROW;

        GpuBuffer xb = buffer(x);
        GpuBuffer wb = buffer(w);
        GpuBuffer bb = buffer(b);
        GpuBuffer posb = ints(new int[]{3, 5});
        GpuBuffer cellIdx = intCells(cells);
        GpuBuffer ff = placeholders.readOnlyFloats();
        GpuBuffer tmp = buffer(x);
        GpuBuffer ref = filled(backend.allocateBuffer((long) cacheElems * Float.BYTES), cacheElems, -7f);
        GpuBuffer out = filled(backend.allocateBuffer((long) cacheElems * Float.BYTES), cacheElems, -7f);

        run(NormKernels.ROW_NORM, (long) nTokens * nHeads * local,
                NormKernels.rowNormArgs(NormKind.RMS, EPS, nTokens * nHeads, STORE_HEAD, STORE_HEAD, tmp, 0, tmp, 0,
                        wb, bb, nHeads, ff));
        run(RopeKernels.ROPE_EXT,
                ActKernels.global1d(RopeKernels.ropeExtWorkItems(rope.type(), nTokens, nHeads, rope.nDims()), local)[0],
                RopeKernels.ropeExtArgs(rope, nTokens, nHeads, STORE_HEAD, rowStride, 0, nTokens, 0, false, posb, ff,
                        tmp));
        runKvStore(KvDType.F32, nTokens, rowStride, tmp, cellIdx, ref, STORE_DST_ROW);

        run(FusedKernels.NORM_ROPE_STORE, FusedKernels.normRopeStoreGlobal(nTokens, nHeads, local),
                FusedKernels.normRopeStoreArgs(rope, nTokens, nHeads, STORE_HEAD, rowStride, STORE_HEAD, 0, nTokens,
                        0, NormKernels.FLAG_HAS_W | NormKernels.FLAG_HAS_B, nHeads, EPS, false, STORE_HEAD,
                        STORE_DST_ROW, FusedKernels.KV_STORE_F32, posb, ff, wb, bb, xb, cellIdx, out));

        near(read(ref, cacheElems), read(out, cacheElems), "norm-rope store F32");
    }

    @Test
    public void normRopeStoreMatchesNormRopeThenKvStoreF16() throws Exception {
        Assume.assumeTrue("backend has no norm-rope store", ops.fused(FusedKernels.NORM_ROPE_STORE));
        int nTokens = 2;
        int nHeads = 2;
        int rowStride = nHeads * STORE_HEAD;
        int total = nTokens * rowStride;
        RopeSpec rope = neox(STORE_HEAD);
        float[] x = random(total, 211);
        float[] w = random(nHeads * STORE_HEAD, 212);
        int[] cells = {2, 0};
        int cacheBytes = STORE_CELLS * STORE_DST_ROW * 2;

        GpuBuffer xb = buffer(x);
        GpuBuffer wb = buffer(w);
        GpuBuffer posb = ints(new int[]{3, 5});
        GpuBuffer cellIdx = intCells(cells);
        GpuBuffer ff = placeholders.readOnlyFloats();
        GpuBuffer tmp = buffer(x);
        GpuBuffer ref = backend.allocateBuffer(cacheBytes);
        GpuBuffer out = backend.allocateBuffer(cacheBytes);

        run(NormKernels.ROW_NORM, (long) nTokens * nHeads * local,
                NormKernels.rowNormArgs(NormKind.RMS, EPS, nTokens * nHeads, STORE_HEAD, STORE_HEAD, tmp, 0, tmp, 0,
                        wb, null, nHeads, ff));
        run(RopeKernels.ROPE_EXT,
                ActKernels.global1d(RopeKernels.ropeExtWorkItems(rope.type(), nTokens, nHeads, rope.nDims()), local)[0],
                RopeKernels.ropeExtArgs(rope, nTokens, nHeads, STORE_HEAD, rowStride, 0, nTokens, 0, false, posb, ff,
                        tmp));
        runKvStore(KvDType.F16, nTokens, rowStride, tmp, cellIdx, ref, STORE_DST_ROW);

        run(FusedKernels.NORM_ROPE_STORE, FusedKernels.normRopeStoreGlobal(nTokens, nHeads, local),
                FusedKernels.normRopeStoreArgs(rope, nTokens, nHeads, STORE_HEAD, rowStride, STORE_HEAD, 0, nTokens,
                        0, NormKernels.FLAG_HAS_W, nHeads, EPS, false, STORE_HEAD, STORE_DST_ROW,
                        FusedKernels.KV_STORE_F16, posb, ff, wb, placeholders.readOnlyFloats(), xb, cellIdx, out));

        // The F16 stores round identically, so every byte of the cache matches (sentinel rows included).
        org.junit.Assert.assertArrayEquals("norm-rope store F16", readBytes(ref, cacheBytes),
                readBytes(out, cacheBytes));
    }

    @Test
    public void ropeStoreMatchesRopeThenKvStore() throws Exception {
        Assume.assumeTrue("backend has no rope store", ops.fused(FusedKernels.ROPE_STORE));
        int nTokens = 2;
        int nHeads = 2;
        int rowStride = nHeads * STORE_HEAD;
        int total = nTokens * rowStride;
        RopeSpec rope = neox(STORE_HEAD);
        float[] x = random(total, 221);
        int[] cells = {1, 2};
        int cacheElems = STORE_CELLS * STORE_DST_ROW;

        GpuBuffer xb = buffer(x);
        GpuBuffer posb = ints(new int[]{7, 9});
        GpuBuffer cellIdx = intCells(cells);
        GpuBuffer ff = placeholders.readOnlyFloats();
        GpuBuffer tmp = buffer(x);
        GpuBuffer ref = filled(backend.allocateBuffer((long) cacheElems * Float.BYTES), cacheElems, -7f);
        GpuBuffer out = filled(backend.allocateBuffer((long) cacheElems * Float.BYTES), cacheElems, -7f);

        run(RopeKernels.ROPE_EXT,
                ActKernels.global1d(RopeKernels.ropeExtWorkItems(rope.type(), nTokens, nHeads, rope.nDims()), local)[0],
                RopeKernels.ropeExtArgs(rope, nTokens, nHeads, STORE_HEAD, rowStride, 0, nTokens, 0, false, posb, ff,
                        tmp));
        runKvStore(KvDType.F32, nTokens, rowStride, tmp, cellIdx, ref, STORE_DST_ROW);

        run(FusedKernels.ROPE_STORE, FusedKernels.normRopeStoreGlobal(nTokens, nHeads, local),
                FusedKernels.ropeStoreArgs(rope, nTokens, nHeads, STORE_HEAD, rowStride, 0, nTokens, 0, false,
                        STORE_HEAD, STORE_DST_ROW, FusedKernels.KV_STORE_F32, posb, ff, xb, cellIdx, out));

        near(read(ref, cacheElems), read(out, cacheElems), "rope store F32");
    }
}

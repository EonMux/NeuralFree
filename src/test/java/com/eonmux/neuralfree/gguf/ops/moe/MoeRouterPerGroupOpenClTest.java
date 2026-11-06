package com.eonmux.neuralfree.gguf.ops.moe;

import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend.BackendType;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.ocl.OclComputeBackend;
import com.eonmux.neuralfree.gguf.arch.GatingFunc;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.EnumSet;
import java.util.Random;

import static com.eonmux.neuralfree.acceleration.GpuTestUtil.isOpenCLAvailable;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/**
 * {@code LLM_MoE_Router}'s per-group selection on OpenCL against {@link MoeRouting}: vLLM's InternS1-Pro router takes
 * {@code top_k / n_groups} experts from every group, groups in index order, which no group-limited selection
 * reproduces.
 */
public class MoeRouterPerGroupOpenClTest {

    private HWAcceleration accel;
    private ComputeBackend backend;
    private LlmKernels kernels;
    private int local;

    @Before
    public void open() {
        Assume.assumeTrue("OpenCL GPU not available", isOpenCLAvailable());
        accel = new HWAcceleration(EnumSet.of(BackendType.OPENCL));
        Assume.assumeFalse("no OpenCL device", accel.getCLDevices().isEmpty());
        backend = new OclComputeBackend(accel.getCLDevices().get(0));
        kernels = LlmKernels.open(backend);
        local = 32;
    }

    @After
    public void close() {
        if (backend != null) {
            backend.close();
            backend = null;
        }
        if (accel != null) {
            accel.shutdown();
            accel = null;
        }
    }

    private GpuBuffer floats(float[] data) throws Exception {
        GpuBuffer b = backend.allocateBuffer((long) data.length * Float.BYTES);
        backend.computeQueue().enqueueWriteBuffer(b, 0, (long) data.length * Float.BYTES, data).awaitCompletion();
        return b;
    }

    private GpuBuffer ints(int n) throws Exception {
        GpuBuffer b = backend.allocateBuffer((long) n * Integer.BYTES);
        backend.computeQueue().enqueueWriteBuffer(b, 0, (long) n * Integer.BYTES, new byte[n * Integer.BYTES])
                .awaitCompletion();
        return b;
    }

    private float[] readFloats(GpuBuffer b, int n) throws Exception {
        backend.computeQueue().finish();
        float[] out = new float[n];
        backend.computeQueue().enqueueReadBuffer(b, 0, (long) n * Float.BYTES, out).awaitCompletion();
        return out;
    }

    private int[] readInts(GpuBuffer b, int n) throws Exception {
        backend.computeQueue().finish();
        byte[] bytes = new byte[n * Integer.BYTES];
        backend.computeQueue().enqueueReadBuffer(b, 0, bytes.length, bytes).awaitCompletion();
        int[] out = new int[n];
        ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asIntBuffer().get(out);
        return out;
    }

    @Test
    public void everyGroupContributesItsShare() throws Exception {
        int nExpert = 8;
        int k = 4;
        int nGroups = 2;
        int nTokens = 3;
        MoeRouting.Params p = new MoeRouting.Params(nExpert, k, GatingFunc.SOFTMAX, nGroups, nGroups, true, 1.0f,
                false, MoeKernels.SEL_PROBS, 0, true);
        Random rnd = new Random(7);
        float[] logits = new float[nTokens * nExpert];
        for (int i = 0; i < logits.length; i++) {
            logits[i] = rnd.nextFloat() * 4f - 2f;
        }
        MoeRouting.Selection expected = MoeRouting.route(p, logits, nTokens, null);

        GpuBuffer logitsBuf = floats(logits);
        GpuBuffer biasBuf = floats(new float[nExpert]);
        GpuBuffer idsBuf = ints(nTokens * k);
        GpuBuffer weightsBuf = floats(new float[nTokens * k]);
        try {
            Object[] args = MoeKernels.routerArgs(p, nTokens, logitsBuf, biasBuf, idsBuf, weightsBuf);
            backend.computeQueue().enqueueKernel(kernels.get(MoeKernels.ROUTER),
                    new long[]{(long) nTokens * local}, new long[]{local}, args);
            int[] ids = readInts(idsBuf, nTokens * k);
            float[] weights = readFloats(weightsBuf, nTokens * k);
            assertArrayEquals(expected.ids(), ids);
            for (int t = 0; t < nTokens; t++) {
                int first = 0;
                int second = 0;
                for (int j = 0; j < k; j++) {
                    int id = ids[t * k + j];
                    if (id < nExpert / nGroups) {
                        first++;
                    } else {
                        second++;
                    }
                    assertEquals("weight " + t + "/" + j, expected.weight(t, j), weights[t * k + j], 1e-5f);
                }
                assertEquals("group 0 picks", k / nGroups, first);
                assertEquals("group 1 picks", k / nGroups, second);
            }
        } finally {
            logitsBuf.close();
            biasBuf.close();
            idsBuf.close();
            weightsBuf.close();
        }
    }

    /** The group-limited selection the per-group flag replaces keeps only the best groups. */
    @Test
    public void theGroupLimitedSelectionStillKeepsWholeGroups() throws Exception {
        int nExpert = 8;
        int k = 2;
        MoeRouting.Params p = new MoeRouting.Params(nExpert, k, GatingFunc.SOFTMAX, 2, 1, true, 1.0f, false,
                MoeKernels.SEL_PROBS, 0, false);
        float[] logits = {5f, 4f, 3f, 2f, 9f, 8f, 1f, 0f};
        MoeRouting.Selection expected = MoeRouting.route(p, logits, 1, null);

        GpuBuffer logitsBuf = floats(logits);
        GpuBuffer biasBuf = floats(new float[nExpert]);
        GpuBuffer idsBuf = ints(k);
        GpuBuffer weightsBuf = floats(new float[k]);
        try {
            Object[] args = MoeKernels.routerArgs(p, 1, logitsBuf, biasBuf, idsBuf, weightsBuf);
            backend.computeQueue().enqueueKernel(kernels.get(MoeKernels.ROUTER), new long[]{local},
                    new long[]{local}, args);
            assertArrayEquals(expected.ids(), readInts(idsBuf, k));
            float[] weights = readFloats(weightsBuf, k);
            for (int j = 0; j < k; j++) {
                assertEquals(expected.weight(0, j), weights[j], 1e-5f);
            }
        } finally {
            logitsBuf.close();
            biasBuf.close();
            idsBuf.close();
            weightsBuf.close();
        }
    }
}

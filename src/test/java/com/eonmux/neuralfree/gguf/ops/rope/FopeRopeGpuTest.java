package com.eonmux.neuralfree.gguf.ops.rope;

import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend.BackendType;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.ocl.OclComputeBackend;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.arch.RopeType;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.spec.FopeSpec;
import com.eonmux.neuralfree.gguf.spec.RopeSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.gguf.spec.TensorRef;
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
 * {@code LLM_RoPE_FoPE} on OpenCL against {@link CpuRope}, which evaluates vLLM's Fourier projection in double
 * precision and rounds each row to float: the kernel must produce the same cos/sin rows per head and pair, for
 * separate coefficient rows and for a wider query that shares a KV group's row.
 */
public class FopeRopeGpuTest {

    private static final float TOL = 2e-5f;

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

    private static RopeSpec spec(int nDims, int nKv, int d, boolean sep) {
        float[] inv = RopeMath.fopeInvFreq(10000f, nDims, d, 4096);
        TensorRef cos = TensorRef.of(TensorNames.weight(TensorNames.ROPE_COS_COEF), GGMLType.F32, d, d, nKv);
        TensorRef sin = TensorRef.of(TensorNames.weight(TensorNames.ROPE_SIN_COEF), GGMLType.F32, d, d, nKv);
        return RopeSpec.builder(RopeType.NEOX).nDims(nDims).freqBase(10000f)
                .fope(FopeSpec.of(nKv, sep, inv, cos, sin)).build();
    }

    private void assertMatchesCpu(RopeSpec spec, int nTokens, int nHeads, int headDim, int[] pos, float[] cosCoef,
                                  float[] sinCoef) throws Exception {
        float[] x = random(nTokens * nHeads * headDim, 31);
        int rowStride = nHeads * headDim;

        float[] expected = x.clone();
        CpuRope.apply(RopeFreqTable.of(spec, null,
                        FopeWeights.of(spec.fope(), cosCoef, sinCoef)),
                expected, 0, nTokens, nHeads, headDim, rowStride, 0, pos, 0, 1);

        GpuBuffer xb = buffer(x);
        GpuBuffer posb = ints(pos);
        GpuBuffer cosb = buffer(cosCoef);
        GpuBuffer sinb = buffer(sinCoef);
        GpuBuffer invb = buffer(spec.fope().invFreq());
        try {
            Object[] args = RopeKernels.fopeArgs(nTokens, nHeads, headDim, rowStride, spec.nDims(), 0,
                    spec.fope().numInvFreq(), spec.fope().numKeyValueHeads(), spec.fope().separateHeads(), posb, cosb,
                    sinb, invb, xb);
            long items = RopeKernels.fopeWorkItems(nTokens, nHeads);
            long global = RopeKernels.roundUp(items, local);
            backend.computeQueue().enqueueKernel(kernels.get(RopeKernels.ROPE_FOPE), new long[]{global},
                    new long[]{local}, args);
            float[] actual = read(xb, x.length);
            for (int i = 0; i < x.length; i++) {
                assertEquals("value " + i, expected[i], actual[i], TOL * (1f + Math.abs(expected[i])));
            }
        } finally {
            xb.close();
            posb.close();
            cosb.close();
            sinb.close();
            invb.close();
        }
    }

    @Test
    public void separateHeadsMatchTheCpuProjection() throws Exception {
        int headDim = 8;
        int d = 3;
        int nKv = 2;
        RopeSpec spec = spec(headDim, nKv, d, true);
        float[] cos = random(nKv * d * d, 41);
        float[] sin = random(nKv * d * d, 42);
        assertMatchesCpu(spec, 3, 2, headDim, new int[]{0, 5, 41}, cos, sin);
    }

    @Test
    public void queryGroupsShareCoefficientRows() throws Exception {
        int headDim = 16;
        int d = 4;
        int nKv = 2;
        RopeSpec spec = spec(headDim, nKv, d, true);
        float[] cos = random(nKv * d * d, 51);
        float[] sin = random(nKv * d * d, 52);
        assertMatchesCpu(spec, 2, 4, headDim, new int[]{7, 129}, cos, sin);
    }

    @Test
    public void sharedRowsBroadcastOverHeads() throws Exception {
        int headDim = 16;
        int d = 4;
        RopeSpec spec = spec(headDim, 1, d, false);
        float[] cos = random(d * d, 61);
        float[] sin = random(d * d, 62);
        assertMatchesCpu(spec, 2, 3, headDim, new int[]{0, 900}, cos, sin);
    }
}

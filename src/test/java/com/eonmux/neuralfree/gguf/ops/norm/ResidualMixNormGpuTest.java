package com.eonmux.neuralfree.gguf.ops.norm;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.HardwareDevice;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.acceleration.vulkan.VkDeviceRanking;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
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

import static org.junit.Assert.assertEquals;

/**
 * {@code LLM_ResidualMixNorm} on the GPU against a double-precision host reference: the residual mix written back
 * to x and the row norm of the sum in y, for the RMS and LAYER kinds, the ADD and SCALED mixes, weight and bias
 * rows, and both forms of the Vulkan shader (shaders/llm/norm_ext.comp): the register-resident vec4 row (a multiple
 * of 4 floats at 16-byte aligned offsets) and the scalar strided row (an odd length, or an offset off the vec4 grid).
 * The Vulkan device is the one the engine ranks first ({@link VkDeviceRanking}), the discrete GPU on a hybrid host.
 */
@RunWith(Parameterized.class)
public class ResidualMixNormGpuTest {

    private static final float TOL = 2e-5f;

    @Parameterized.Parameters(name = "{0}")
    public static List<Object[]> backends() {
        return Arrays.asList(new Object[]{ComputeBackend.BackendType.CUDA},
                new Object[]{ComputeBackend.BackendType.VULKAN});
    }

    private final ComputeBackend.BackendType api;
    private HWAcceleration hw;
    private ComputeBackend backend;
    private LlmKernels kernels;
    private final List<GpuBuffer> bufs = new ArrayList<>();

    public ResidualMixNormGpuTest(ComputeBackend.BackendType api) {
        this.api = api;
    }

    @Before
    public void open() {
        if (api == ComputeBackend.BackendType.CUDA) {
            GpuTestUtil.assumeCudaAvailable();
        } else {
            GpuTestUtil.assumeVulkanAvailable();
        }
        hw = new HWAcceleration(EnumSet.of(api));
        HardwareDevice device = api == ComputeBackend.BackendType.CUDA
                ? (hw.getCUDevices().isEmpty() ? null : hw.getCUDevices().get(0))
                : VkDeviceRanking.fastestDevice(hw.getVkDevices(), null);
        Assume.assumeTrue("no " + api + " device", device != null);
        backend = hw.createBackend(device);
        Assume.assumeTrue("LLM kernels unavailable on " + api, LlmKernels.availability(backend).available());
        kernels = LlmKernels.open(backend);
        Assume.assumeTrue("no " + NormKernels.RESIDUAL_MIX_NORM, kernels.has(NormKernels.RESIDUAL_MIX_NORM));
    }

    @After
    public void close() {
        for (GpuBuffer b : bufs) {
            b.close();
        }
        if (backend != null) {
            backend.close();
        }
        if (hw != null) {
            hw.shutdown();
        }
    }

    @Test
    public void rmsAddOverVec4Rows() {
        check(NormKind.RMS, NormKernels.MixMode.ADD, 0f, 3, 2560, 0, true, false);
    }

    @Test
    public void layerScaledWithBiasOverVec4Rows() {
        check(NormKind.LAYER, NormKernels.MixMode.SCALED, 0.5f, 2, 1024, 0, true, true);
    }

    @Test
    public void rmsOverAnOddRowTakesTheScalarRow() {
        check(NormKind.RMS, NormKernels.MixMode.ADD, 0f, 2, 2562, 0, true, false);
    }

    @Test
    public void layerOverAnUnalignedOffsetTakesTheScalarRow() {
        check(NormKind.LAYER, NormKernels.MixMode.SCALED, 2f, 2, 1024, 2, true, true);
    }

    @Test
    public void rmsWithoutWeightOverTheLargestVec4Row() {
        check(NormKind.RMS, NormKernels.MixMode.ADD, 0f, 1, 4096, 4, false, false);
    }

    private void check(NormKind kind, NormKernels.MixMode mix, float s, int nRows, int rowLen, int off,
                       boolean weight, boolean bias) {
        Random rnd = new Random(rowLen * 31L + nRows);
        int len = off + nRows * rowLen;
        float[] x = random(rnd, len);
        float[] add = random(rnd, len);
        float[] w = random(rnd, rowLen);
        float[] b = random(rnd, rowLen);
        GpuBuffer dx = upload(x);
        GpuBuffer dadd = upload(add);
        GpuBuffer dw = upload(w);
        GpuBuffer db = upload(b);
        GpuBuffer dy = upload(new float[len]);
        GpuBuffer ph = upload(new float[4]);
        float eps = 1e-6f;
        Object[] args = NormKernels.residualMixNormArgs(mix, s, kind, eps, nRows, rowLen, off, off, off, dx, dadd,
                weight ? dw : null, bias ? db : null, 1, dy, ph);
        int local = kernels.rowReduceLocal();
        GpuEvent e = backend.computeQueue().enqueueKernel(kernels.get(NormKernels.RESIDUAL_MIX_NORM),
                new long[]{(long) nRows * local}, new long[]{local}, args);
        if (e != null) {
            e.close();
        }
        float[] gotX = read(dx, len);
        float[] gotY = read(dy, len);
        for (int r = 0; r < nRows; r++) {
            int base = off + r * rowLen;
            double[] v = new double[rowLen];
            double sum = 0;
            double ss = 0;
            for (int i = 0; i < rowLen; i++) {
                float vf = mix == NormKernels.MixMode.SCALED ? x[base + i] + s * add[base + i] : x[base + i] + add[base + i];
                v[i] = vf;
                sum += vf;
                ss += (double) vf * vf;
                assertEquals(api + " x row " + r + " col " + i, vf, gotX[base + i], 0f);
            }
            double mean = kind == NormKind.LAYER ? sum / rowLen : 0;
            double scale;
            if (kind == NormKind.LAYER) {
                double dev = 0;
                for (int i = 0; i < rowLen; i++) {
                    dev += (v[i] - mean) * (v[i] - mean);
                }
                scale = 1.0 / Math.sqrt(dev / rowLen + eps);
            } else {
                scale = 1.0 / Math.sqrt(ss / rowLen + eps);
            }
            for (int i = 0; i < rowLen; i++) {
                double want = (v[i] - mean) * scale;
                if (weight) {
                    want *= w[i];
                }
                if (bias) {
                    want += b[i];
                }
                assertEquals(api + " y row " + r + " col " + i, want, gotY[base + i], TOL);
            }
        }
    }

    private static float[] random(Random rnd, int n) {
        float[] v = new float[n];
        for (int i = 0; i < n; i++) {
            v[i] = rnd.nextFloat() * 2f - 1f;
        }
        return v;
    }

    private GpuBuffer upload(float[] data) {
        GpuBuffer b = backend.allocateBuffer((long) data.length * 4, true);
        bufs.add(b);
        GpuEvent e = backend.computeQueue().enqueueWriteBuffer(b, 0, (long) data.length * 4, data);
        if (e != null) {
            e.close();
        }
        return b;
    }

    private float[] read(GpuBuffer b, int n) {
        float[] out = new float[n];
        GpuEvent e = backend.computeQueue().enqueueReadBuffer(b, 0, (long) n * 4, out);
        if (e != null) {
            e.close();
        }
        backend.computeQueue().finish();
        return out;
    }
}

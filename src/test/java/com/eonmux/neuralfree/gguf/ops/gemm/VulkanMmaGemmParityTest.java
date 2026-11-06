package com.eonmux.neuralfree.gguf.ops.gemm;

import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.DeviceFeature;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.acceleration.backend.vulkan.VkComputeBackend;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.tensor.HalfFloats;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;

import static org.junit.Assert.assertTrue;

/**
 * The cooperative-matrix {@code F16_MatMul_MMA} of {@code shaders/llm/matmul_mma.comp} on a Vulkan device with
 * {@code VK_KHR_cooperative_matrix}, against the fp64 dot: weight and activations are exact in fp16, so only fp32
 * accumulation order separates the kernel from the reference. The shape covers every row block of a 64-row tile
 * and both 16-token blocks of a 32-token tile, partial tiles in all three dimensions, and the strided, offset view
 * the projections use. Skips without such a device.
 */
public class VulkanMmaGemmParityTest {

    private static final int M = 70;   /* two 64-row tiles, the second partial */
    private static final int K = 96;   /* six 16-deep stages */
    private static final int N = 40;   /* two 32-token tiles, the second partial */

    private VkComputeBackend backend;
    private LlmKernels kernels;

    @Before
    public void setUp() {
        try {
            backend = new VkComputeBackend();
            kernels = LlmKernels.open(backend);
        } catch (Throwable e) {
            if (backend != null) {
                backend.close();
            }
            backend = null;
        }
        Assume.assumeTrue("no Vulkan GPU", backend != null);
        Assume.assumeTrue("the device reports no cooperative matrices",
                backend.capabilities().hasFeature(DeviceFeature.COOPERATIVE_MATRIX));
    }

    @After
    public void tearDown() {
        if (backend != null) {
            backend.close();
        }
    }

    @Test
    public void contiguousMmaGemmMatchesTheFp64Dot() throws Exception {
        check(MmaGemmPlan.contiguous(GGMLType.F16, M, K, N));
    }

    @Test
    public void stridedMmaGemmMatchesTheFp64Dot() throws Exception {
        check(MmaGemmPlan.of(GGMLType.F16, M, K, N, K + 8, 5, M + 3, 7));
    }

    private void check(MmaGemmPlan plan) throws Exception {
        float[] w = values(M * K, 3, 1.0f);
        float[] x = values((int) plan.xElementsRequired(), 5, 2.0f);
        ByteBuffer wb = ByteBuffer.allocate(2 * w.length).order(ByteOrder.LITTLE_ENDIAN);
        for (float f : w) {
            wb.putShort((short) HalfFloats.floatToHalf(f));
        }
        float[] out = launch(plan, wb.array(), x);
        for (int i = 0; i < out.length; i++) {
            int t = (i - plan.yOff()) / plan.yStride();
            int r = (i - plan.yOff()) % plan.yStride();
            boolean written = i >= plan.yOff() && t < N && r < M;
            if (!written) {
                assertTrue("element " + i + " outside the output view was written: " + out[i], out[i] == -7.0f);
                continue;
            }
            double acc = 0.0;
            double terms = 0.0;
            for (int k = 0; k < K; k++) {
                double term = (double) w[r * K + k] * x[plan.xOff() + t * plan.xStride() + k];
                acc += term;
                terms += Math.abs(term);
            }
            assertTrue("token " + t + " row " + r + ": kernel " + out[i] + ", expected " + acc,
                    Math.abs(out[i] - acc) <= 1e-5 * terms + 1e-3);
        }
    }

    private float[] launch(MmaGemmPlan plan, byte[] weight, float[] x) throws Exception {
        ComputeQueue queue = backend.computeQueue();
        GpuBuffer wbuf = backend.allocateBuffer(plan.weightBytes(), true);
        GpuBuffer xbuf = backend.allocateBuffer(plan.xElementsRequired() * Float.BYTES, true);
        GpuBuffer ybuf = backend.allocateBuffer(plan.yElementsRequired() * Float.BYTES);
        try {
            float[] init = new float[(int) plan.yElementsRequired()];
            Arrays.fill(init, -7.0f);
            close(queue.enqueueWriteBuffer(wbuf, 0, weight.length, weight));
            close(queue.enqueueWriteBuffer(xbuf, 0, 4L * x.length, x));
            close(queue.enqueueWriteBuffer(ybuf, 0, 4L * init.length, init));
            close(queue.enqueueKernel(kernels.get(plan.kernelName()), plan.globalWorkSize(), plan.localWorkSize(),
                    plan.args(wbuf, xbuf, ybuf)));
            float[] out = new float[init.length];
            GpuEvent read = queue.enqueueReadBuffer(ybuf, 0, 4L * out.length, out);
            queue.finish();
            close(read);
            return out;
        } finally {
            wbuf.close();
            xbuf.close();
            ybuf.close();
        }
    }

    private static void close(GpuEvent e) {
        if (e != null) {
            e.close();
        }
    }

    /** Deterministic values in {@code [-amp, amp)} on a 1/64 grid (exact in fp16 for amp <= 32). */
    private static float[] values(int n, long seed, float amp) {
        float[] v = new float[n];
        long s = seed * 0x9E3779B97F4A7C15L + 1;
        for (int i = 0; i < n; i++) {
            s ^= s << 13;
            s ^= s >>> 7;
            s ^= s << 17;
            v[i] = amp * ((int) ((s >>> 20) & 127) - 64) / 64.0f;
        }
        return v;
    }
}

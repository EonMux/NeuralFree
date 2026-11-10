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
import com.eonmux.neuralfree.gguf.arch.GluOp;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuPlaceholders;
import com.eonmux.neuralfree.gguf.quant.FpBlockDecoders;
import com.eonmux.neuralfree.gguf.spec.ActParams;
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

/**
 * The fused gate and up projection with the GLU ({@code {T}_MatVecFusedGateUp}) against the two matvecs and the
 * {@code LLM_GLU} it replaces, on every available GPU backend. Both sides run on the same device, so the expected
 * values are the unfused path's; the tolerance covers only the reduction order. A backend is skipped when its GPU
 * or its kernel library is unavailable, or when the fused form does not serve the weights there.
 */
@RunWith(Parameterized.class)
public class GpuGateUpGluGpuTest {

    private static final int COLS = 512;
    private static final int ROWS = 8;
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
    private boolean previousCuda;

    public GpuGateUpGluGpuTest(String api) {
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

    private GpuBuffer buffer(float[] data) {
        GpuBuffer b = backend.allocateBuffer((long) data.length * Float.BYTES);
        backend.computeQueue().enqueueWriteBuffer(b, 0, (long) data.length * Float.BYTES, data);
        return b;
    }

    private GpuBuffer bytes(byte[] data) {
        GpuBuffer b = backend.allocateBuffer(data.length);
        backend.computeQueue().enqueueWriteBuffer(b, 0, data.length, data);
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

    private static GpuWeight weight(String name, GGMLType type, GpuBuffer data) {
        return GpuWeight.of(name, type, new long[]{COLS, ROWS}, data);
    }

    private static byte[] quantized(GGMLType type, long seed) {
        Random rnd = new Random(seed);
        int blockBytes = type.getTypeSize();
        int blockSize = type.getBlockSize();
        byte[] data = new byte[ROWS * (COLS / blockSize) * blockBytes];
        rnd.nextBytes(data);
        for (int b = 0; b < ROWS * (COLS / blockSize); b++) {
            int base = b * blockBytes;
            switch (type) {
                case Q4_K:
                case Q5_K:
                    putHalf(data, base, 0.02f);
                    putHalf(data, base + 2, 0.01f);
                    break;
                default:
                    putHalf(data, base, 0.02f);
                    break;
            }
        }
        return data;
    }

    private static void putHalf(byte[] data, int off, float value) {
        int bits = FpBlockDecoders.fp32ToFp16(value);
        data[off] = (byte) bits;
        data[off + 1] = (byte) (bits >>> 8);
    }

    // ------------------------------------------------------------------ tests

    @Test
    public void fusedGateUpMatchesTwoMatvecsAndTheGluOnFloatWeights() {
        GpuWeight gate = weight("ffn_gate.weight", GGMLType.F32, buffer(random(COLS * ROWS, 11)));
        GpuWeight up = weight("ffn_up.weight", GGMLType.F32, buffer(random(COLS * ROWS, 12)));
        Assume.assumeTrue("fused gate+up not available on " + api,
                ops.gateUpGluUsable(gate, up, GluOp.SWIGLU, ActParams.NONE, 1));
        assertMatchesUnfused(gate, up, random(COLS, 13));
    }

    @Test
    public void fusedGateUpMatchesTwoMatvecsAndTheGluOnQuantizedWeights() {
        assertMatchesQuantized(GGMLType.Q4_K, 21, 22, 23);
    }

    /** The legacy block layout Qwen3-4B Q4_0 uses, whose row decoder differs from the K-quant one. */
    @Test
    public void fusedGateUpMatchesTwoMatvecsAndTheGluOnQ4_0Weights() {
        assertMatchesQuantized(GGMLType.Q4_0, 31, 32, 33);
    }

    private void assertMatchesQuantized(GGMLType type, long gateSeed, long upSeed, long xSeed) {
        GpuWeight gate = weight("ffn_gate.weight", type, bytes(quantized(type, gateSeed)));
        GpuWeight up = weight("ffn_up.weight", type, bytes(quantized(type, upSeed)));
        Assume.assumeTrue("fused gate+up not available on " + api,
                ops.gateUpGluUsable(gate, up, GluOp.SWIGLU, ActParams.NONE, 1));
        assertMatchesUnfused(gate, up, random(COLS, xSeed));
    }

    private void assertMatchesUnfused(GpuWeight gate, GpuWeight up, float[] x) {
        GpuBuffer in = buffer(x);
        GpuBuffer gateOut = backend.allocateBuffer((long) ROWS * Float.BYTES);
        GpuBuffer upOut = backend.allocateBuffer((long) ROWS * Float.BYTES);
        GpuBuffer expected = backend.allocateBuffer((long) ROWS * Float.BYTES);
        GpuBuffer actual = backend.allocateBuffer((long) ROWS * Float.BYTES);

        ops.matvec(gate, in, 0, gateOut, 0, 1);
        ops.matvec(up, in, 0, upOut, 0, 1);
        ops.glu(GluOp.SWIGLU, ActParams.NONE, gateOut, 0, ROWS, upOut, 0, ROWS, expected, 0, ROWS, 1, ROWS);
        ops.gateUpGlu(gate, up, GluOp.SWIGLU, ActParams.NONE, in, actual, 1);

        float[] want = read(expected, ROWS);
        float[] got = read(actual, ROWS);
        for (int i = 0; i < ROWS; i++) {
            assertEquals("fused gate+up[" + i + "] on " + api, want[i], got[i], TOL * (1f + Math.abs(want[i])));
        }
    }
}

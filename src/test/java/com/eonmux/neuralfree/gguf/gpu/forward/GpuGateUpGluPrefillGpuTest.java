package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.cuda.CudaComputeBackend;
import com.eonmux.neuralfree.acceleration.cuda.CUDevice;
import com.eonmux.neuralfree.acceleration.kernels.Availability;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.arch.GluOp;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuPlaceholders;
import com.eonmux.neuralfree.gguf.ops.gemm.MmqConfig;
import com.eonmux.neuralfree.gguf.ops.gemm.OpenClMmqGemmParityTest;
import com.eonmux.neuralfree.gguf.spec.ActParams;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;

import java.util.Arrays;

import static org.junit.Assert.assertTrue;

/**
 * The prefill form of the fused gate/up projection ({@code <T>_MatMul_MMQ_StreamK_GateUp} through
 * {@link GpuOps#gateUpGlu} with several rows) against the route it replaces on the same CUDA device: the gate and
 * up GEMMs ({@link GpuOps#matmul}) and {@code LLM_GLU}. Both read the same staged records and reduce each product
 * through the same chunk pipe; only the stream-k split points differ, so the outputs agree to the fp32 rounding of
 * the partial sums. Also checks that the form is offered only at a row count where both products would run the
 * int8 MMQ GEMM, and that a shared activation scope over the rows serves its staged records.
 */
public class GpuGateUpGluPrefillGpuTest {

    private static final int COLS = 512;
    private static final int ROWS = 320;   /* five 64-row output tiles */
    private static final int TOKENS = 200; /* one full 128-token tile and a partial */

    private HWAcceleration hwAccel;
    private CudaComputeBackend backend;
    private GpuOps ops;
    private GpuScratch scratch;
    private GpuPlaceholders placeholders;
    private boolean previousCuda;

    @Before
    public void openDevice() {
        previousCuda = HWAcceleration.enableCuda;
        try {
            HWAcceleration.enableCuda = true;
            hwAccel = new HWAcceleration();
            Assume.assumeFalse("no CUDA device", hwAccel.getCUDevices().isEmpty());
            CUDevice device = hwAccel.getCUDevices().get(0);
            backend = new CudaComputeBackend(device);
            Availability availability = LlmKernels.availability(backend);
            Assume.assumeTrue("LLM kernels unavailable: " + availability.reason(), availability.available());
            LlmKernels kernels = LlmKernels.open(backend);
            scratch = GpuScratch.builder(backend, TOKENS).slot(ScratchSlots.X, (long) TOKENS * COLS).build();
            placeholders = GpuPlaceholders.allocate(backend, backend.computeQueue());
            ops = new GpuOps(backend.computeQueue(), kernels, scratch, placeholders);
        } catch (Throwable e) {
            closeDevice();
            Assume.assumeTrue("no usable CUDA GPU: " + e.getMessage(), false);
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
            backend.close();
            backend = null;
        }
        if (hwAccel != null) {
            hwAccel.shutdown();
            hwAccel = null;
        }
        HWAcceleration.enableCuda = previousCuda;
    }

    @Test
    public void prefillGateUpMatchesTheGemmsAndTheGluOnQ8_0() {
        assertMatchesSeparate(GGMLType.Q8_0);
    }

    @Test
    public void prefillGateUpMatchesTheGemmsAndTheGluOnQ4_K() {
        assertMatchesSeparate(GGMLType.Q4_K);
    }

    @Test
    public void prefillGateUpMatchesTheGemmsAndTheGluOnQ6_K() {
        assertMatchesSeparate(GGMLType.Q6_K);
    }

    /** Rows the multi-column matvecs serve (a short speculative step) keep the separate products. */
    @Test
    public void smallRowCountsKeepTheSeparateProducts() {
        GpuWeight[] w = weights(GGMLType.Q8_0);
        for (int n = 2; n <= 8; n++) {
            assertTrue(n + " rows must not take the GEMM form",
                    !ops.gateUpGluUsable(w[0], w[1], GluOp.SWIGLU, ActParams.NONE, n));
        }
        assertTrue("a non-SWIGLU op has no fused form",
                !ops.gateUpGluUsable(w[0], w[1], GluOp.GEGLU_ERF, ActParams.NONE, TOKENS));
    }

    private void assertMatchesSeparate(GGMLType type) {
        GpuWeight[] w = weights(type);
        boolean usable = ops.gateUpGluUsable(w[0], w[1], GluOp.SWIGLU, ActParams.NONE, TOKENS);
        Assume.assumeTrue("the gate/up stream-k kernels run on the tensor tile only",
                MmqConfig.forDevice(backend.backendType(), backend.capabilities().getComputeCapability())
                        == MmqConfig.INT8_TENSOR);
        assertTrue(TOKENS + " rows of " + type + " take the fused form on the tensor tile", usable);
        GpuBuffer in = buffer(OpenClMmqGemmParityTest.buildX(TOKENS, COLS));
        int n = TOKENS * ROWS;
        GpuBuffer gateOut = backend.allocateBuffer((long) n * Float.BYTES);
        GpuBuffer upOut = backend.allocateBuffer((long) n * Float.BYTES);
        GpuBuffer expected = backend.allocateBuffer((long) n * Float.BYTES);
        GpuBuffer fused = backend.allocateBuffer((long) n * Float.BYTES);
        GpuBuffer sharedFused = backend.allocateBuffer((long) n * Float.BYTES);
        ops.matmul(w[0], in, 0, gateOut, 0, TOKENS);
        ops.matmul(w[1], in, 0, upOut, 0, TOKENS);
        ops.glu(GluOp.SWIGLU, ActParams.NONE, gateOut, 0, ROWS, upOut, 0, ROWS, expected, 0, ROWS, TOKENS, ROWS);
        ops.gateUpGlu(w[0], w[1], GluOp.SWIGLU, ActParams.NONE, in, fused, TOKENS);
        try (GpuOps.SharedQ8 share = ops.shareQ8Activations(in, 0, COLS, TOKENS)) {
            ops.matmul(w[0], in, 0, gateOut, 0, TOKENS);
            ops.gateUpGlu(w[0], w[1], GluOp.SWIGLU, ActParams.NONE, in, sharedFused, TOKENS);
        }
        float[] g = read(gateOut, n);
        float[] u = read(upOut, n);
        float[] want = read(expected, n);
        float[] got = read(fused, n);
        float[] gotShared = read(sharedFused, n);
        for (int i = 0; i < n; i++) {
            /* d(silu(g) u) = silu'(g) u dg + silu(g) du with silu' <= 1.1 and |dg|, |du| a few ulps of the
             * partial sums (the bound CudaMmqStreamKParityTest uses for split tiles) */
            double bound = 5e-3 * ((1.0 + Math.abs(g[i])) * 1.1 * Math.abs(u[i])
                    + (1.0 + Math.abs(u[i])) * Math.abs(g[i])) + 1e-6;
            assertTrue(type + " element " + i + ": fused " + got[i] + " separate " + want[i],
                    Math.abs(got[i] - want[i]) <= bound);
            assertTrue(type + " element " + i + ": the shared records give " + Float.toHexString(gotShared[i])
                            + ", the projection's own " + Float.toHexString(got[i]),
                    Float.floatToRawIntBits(gotShared[i]) == Float.floatToRawIntBits(got[i]));
        }
        for (GpuBuffer b : new GpuBuffer[]{in, gateOut, upOut, expected, fused, sharedFused}) {
            b.close();
        }
    }

    /** The gate and up weights: deterministic blocks of {@code type} with exact-in-fp16 scales. */
    private GpuWeight[] weights(GGMLType type) {
        byte[] both = OpenClMmqGemmParityTest.buildWeight(type, 2 * ROWS, COLS);
        byte[] gate = Arrays.copyOfRange(both, 0, both.length / 2);
        byte[] up = Arrays.copyOfRange(both, both.length / 2, both.length);
        return new GpuWeight[]{
                GpuWeight.of("ffn_gate.weight", type, new long[]{COLS, ROWS}, bytes(gate)),
                GpuWeight.of("ffn_up.weight", type, new long[]{COLS, ROWS}, bytes(up))};
    }

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
}

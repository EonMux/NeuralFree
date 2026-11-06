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
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuPlaceholders;
import com.eonmux.neuralfree.gguf.quant.ActFakeQuant;
import com.eonmux.neuralfree.gguf.quant.ActQuantScheme;
import com.eonmux.neuralfree.gguf.quant.CpuHadamard;
import com.eonmux.neuralfree.gguf.quant.HadamardSpec;
import com.eonmux.neuralfree.gguf.quant.InputTransform;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The GPU activation pre-scale ({@code LLM_ScaleCols}): a weight with an {@code act_pre_scale} and a static activation
 * scheme scales every input column before the fake-quantization, and the product equals the host's
 * {@link ActFakeQuant} over the scaled rows; a permuting Hadamard consumes the scaled rows without aliasing them; a
 * per-expert pre-scale has no expert index on the dense path and is refused. A backend is skipped when its GPU or
 * its kernel library is unavailable.
 */
@RunWith(Parameterized.class)
public class ActivationPreScaleGpuTest {

    private static final int COLS = 32;
    private static final int ROWS = 4;
    private static final int TOKENS = 2;
    private static final float ACT_SCALE = 0.25f;

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

    public ActivationPreScaleGpuTest(String api) {
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
            scratch = GpuScratch.builder(backend, TOKENS).slot(ScratchSlots.X, (long) TOKENS * COLS).build();
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

    private GpuBuffer upload(float[] values) throws Exception {
        GpuBuffer b = backend.allocateBuffer((long) values.length * Float.BYTES, true);
        backend.computeQueue().enqueueWriteBuffer(b, 0, (long) values.length * Float.BYTES, values)
                .awaitCompletion();
        return b;
    }

    private float[] read(GpuBuffer b, int count) throws Exception {
        float[] out = new float[count];
        backend.computeQueue().enqueueReadBuffer(b, 0, (long) count * Float.BYTES, out).awaitCompletion();
        return out;
    }

    private GpuWeight weight(float[] values) throws Exception {
        GpuBuffer b = upload(values);
        return GpuWeight.of("blk.0.attn_q.weight", GGMLType.F32, new long[]{COLS, ROWS}, b);
    }

    @Test
    public void preScaleRunsBeforeTheActivationQuantization() throws Exception {
        Random rnd = new Random(20261105L);
        float[] w = new float[ROWS * COLS];
        float[] x = new float[TOKENS * COLS];
        float[] pre = new float[COLS];
        for (int i = 0; i < w.length; i++) {
            w[i] = rnd.nextFloat() * 2f - 1f;
        }
        for (int i = 0; i < x.length; i++) {
            x[i] = rnd.nextFloat() * 4f - 2f;
        }
        for (int c = 0; c < COLS; c++) {
            pre[c] = c % 2 == 0 ? 0.5f : 2f;
        }
        InputTransform.ActQuant aq = InputTransform.ActQuant.of(ActQuantScheme.INT8_STATIC,
                new float[]{ACT_SCALE}, null, 0f);
        GpuBuffer preBuf = upload(pre);
        GpuWeight weight = weight(w).withInputTransform(InputTransform.of(null, aq, null, pre), null, preBuf);
        GpuBuffer in = upload(x);
        GpuBuffer out = backend.allocateBuffer((long) TOKENS * ROWS * Float.BYTES, true);
        ops.matmul(weight, in, 0, out, 0, TOKENS);
        backend.computeQueue().finish();
        float[] got = read(out, TOKENS * ROWS);

        float[] row = new float[COLS];
        for (int t = 0; t < TOKENS; t++) {
            System.arraycopy(x, t * COLS, row, 0, COLS);
            for (int c = 0; c < COLS; c++) {
                row[c] *= pre[c];
            }
            ActFakeQuant.apply(ActQuantScheme.INT8_STATIC, ACT_SCALE, 0, 0f, row, 0, 1, COLS);
            for (int r = 0; r < ROWS; r++) {
                float want = 0f;
                float mag = 0f;
                for (int c = 0; c < COLS; c++) {
                    float term = w[r * COLS + c] * row[c];
                    want += term;
                    mag += Math.abs(term);
                }
                assertEquals(api + " token " + t + " row " + r, want, got[t * ROWS + r],
                        Math.max(1e-5f, mag * 1e-5f));
            }
        }
    }

    @Test
    public void preScaleFeedsAPermutingHadamardWithoutOverlap() throws Exception {
        // permRep > 1 forbids an in-place rotation, so the scaled input and the rotated rows need distinct staging.
        HadamardSpec h = HadamardSpec.of(8, HadamardSpec.SignOrder.NONE, null, 2, 4, 4);
        Random rnd = new Random(20261106L);
        float[] w = new float[ROWS * COLS];
        float[] x = new float[TOKENS * COLS];
        float[] pre = new float[COLS];
        for (int i = 0; i < w.length; i++) {
            w[i] = rnd.nextFloat() * 2f - 1f;
        }
        for (int i = 0; i < x.length; i++) {
            x[i] = rnd.nextFloat() * 4f - 2f;
        }
        for (int c = 0; c < COLS; c++) {
            pre[c] = c % 3 == 0 ? 2f : 0.5f;
        }
        GpuBuffer preBuf = upload(pre);
        GpuWeight weight = weight(w).withInputTransform(InputTransform.of(h, null, null, pre), null, preBuf);
        GpuBuffer in = upload(x);
        GpuBuffer out = backend.allocateBuffer((long) TOKENS * ROWS * Float.BYTES, true);
        ops.matmul(weight, in, 0, out, 0, TOKENS);
        backend.computeQueue().finish();
        float[] got = read(out, TOKENS * ROWS);

        float[] scaled = new float[COLS];
        float[] rotated = new float[COLS];
        for (int t = 0; t < TOKENS; t++) {
            System.arraycopy(x, t * COLS, scaled, 0, COLS);
            for (int c = 0; c < COLS; c++) {
                scaled[c] *= pre[c];
            }
            CpuHadamard.apply(h, scaled, 0, rotated, 0, 1, COLS);
            for (int r = 0; r < ROWS; r++) {
                float want = 0f;
                float mag = 0f;
                for (int c = 0; c < COLS; c++) {
                    float term = w[r * COLS + c] * rotated[c];
                    want += term;
                    mag += Math.abs(term);
                }
                assertEquals(api + " token " + t + " row " + r, want, got[t * ROWS + r],
                        Math.max(1e-5f, mag * 1e-5f));
            }
        }
    }

    @Test
    public void perExpertPreScaleIsRefusedByTheDenseProjection() throws Exception {
        float[] pre = new float[2 * COLS];
        Arrays.fill(pre, 1f);
        GpuBuffer preBuf = upload(pre);
        GpuWeight weight = weight(new float[ROWS * COLS])
                .withInputTransform(InputTransform.of(null, null, null, pre), null, preBuf);
        GpuBuffer in = backend.allocateBuffer((long) COLS * Float.BYTES, true);
        GpuBuffer out = backend.allocateBuffer((long) ROWS * Float.BYTES, true);
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> ops.matvec(weight, in, 0, out, 0, 1));
        assertTrue(e.getMessage(), e.getMessage().contains("per expert"));
    }
}

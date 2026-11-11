package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.HardwareDevice;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.acceleration.kernels.Availability;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.ArrayList;
import java.util.EnumSet;
import java.util.List;
import java.util.Locale;
import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * {@code LLM_FlashAttn_Combine} (kernels/llm/flash_combine.cl, its twin shaders/llm/flash_combine.comp) against a
 * double-precision merge of the same split partials: one split, a handful, and split counts below, at and past one
 * stage of staged weights (256), head dims from 64 to 1024 (the widest is also launched with a local size too small
 * for the staged form, which runs the barrier-free form), sink logits, splits whose keys were all masked
 * (m = -inf, l = 0) and rows with no visible key at all (written as zeros). Runs on CUDA, and on OpenCL and Vulkan
 * on each device (the Intel iGPU and the NVIDIA GPU), skipping what the machine lacks.
 */
@RunWith(Parameterized.class)
public class FlashCombineKernelGpuTest {

    private static final int N_HEAD = 4;
    private static final int[] SPLITS = {1, 7, 255, 256, 257, 600};
    private static final int[] DIMS = {64, 128, 512, 1024};

    @Parameterized.Parameters(name = "{0} {1}")
    public static List<Object[]> cases() {
        List<Object[]> rows = new ArrayList<>();
        rows.add(new Object[]{ComputeBackend.BackendType.CUDA, ""});
        rows.add(new Object[]{ComputeBackend.BackendType.OPENCL, "Intel"});
        rows.add(new Object[]{ComputeBackend.BackendType.OPENCL, "NVIDIA"});
        rows.add(new Object[]{ComputeBackend.BackendType.VULKAN, "NVIDIA"});
        rows.add(new Object[]{ComputeBackend.BackendType.VULKAN, "Intel"});
        return rows;
    }

    private final ComputeBackend.BackendType api;
    private final String deviceName;
    private HWAcceleration hw;
    private ComputeBackend backend;
    private LlmKernels kernels;
    private final List<GpuBuffer> bufs = new ArrayList<>();

    public FlashCombineKernelGpuTest(ComputeBackend.BackendType api, String deviceName) {
        this.api = api;
        this.deviceName = deviceName;
    }

    @Before
    public void open() {
        switch (api) {
            case OPENCL:
                GpuTestUtil.assumeOpenCLAvailable();
                break;
            case CUDA:
                GpuTestUtil.assumeCudaAvailable();
                break;
            default:
                GpuTestUtil.assumeVulkanAvailable();
                break;
        }
        hw = new HWAcceleration(EnumSet.of(api));
        backend = openDevice();
        Assume.assumeTrue("no " + api + " " + deviceName + " device enumerated", backend != null);
        Availability a = LlmKernels.availability(backend);
        Assume.assumeTrue("LLM kernels unavailable on " + api + ": " + a.reason(), a.available());
        kernels = LlmKernels.open(backend);
    }

    /** A backend over the first device of the API whose name holds {@code deviceName}, or null. */
    private ComputeBackend openDevice() {
        List<HardwareDevice> all = new ArrayList<>();
        switch (api) {
            case OPENCL:
                all.addAll(hw.getCLDevices());
                break;
            case CUDA:
                all.addAll(hw.getCUDevices());
                break;
            default:
                all.addAll(hw.getVkDevices());
                break;
        }
        for (HardwareDevice d : all) {
            ComputeBackend b = hw.createBackend(d);
            String name = b.capabilities().getDeviceName();
            if (deviceName.isEmpty() || (name != null && name.contains(deviceName))) {
                return b;
            }
            b.close();
        }
        return null;
    }

    @After
    public void close() {
        for (GpuBuffer b : bufs) {
            b.close();
        }
        bufs.clear();
        if (backend != null) {
            backend.close();
        }
        if (hw != null) {
            hw.shutdown();
        }
    }

    @Test
    public void mergesMatchTheDoubleReference() {
        Random rnd = new Random(5);
        for (int dV : DIMS) {
            for (int splits : SPLITS) {
                for (boolean sinks : new boolean[]{false, true}) {
                    check(rnd, dV, splits, sinks, FlashAttnTiling.combineLocal());
                }
            }
        }
    }

    /** A local size too small to stage a 1024-wide row: the barrier-free form (CUDA and OpenCL take any size). */
    @Test
    public void aSmallLocalSizeRunsTheBarrierFreeForm() {
        Assume.assumeTrue("the Vulkan pipeline fixes its local size", api != ComputeBackend.BackendType.VULKAN);
        Random rnd = new Random(9);
        for (int splits : new int[]{3, 300}) {
            check(rnd, 1024, splits, true, new long[]{64});
        }
    }

    private void check(Random rnd, int dV, int splits, boolean sinks, long[] local) {
        int stride = dV + 2;
        float[] partial = new float[N_HEAD * splits * stride];
        float[] sink = new float[N_HEAD];
        for (int h = 0; h < N_HEAD; h++) {
            sink[h] = 3f * (rnd.nextFloat() - 0.5f);
            for (int s = 0; s < splits; s++) {
                int b = (h * splits + s) * stride;
                boolean masked = h == N_HEAD - 1 || rnd.nextInt(5) == 0;   // the last head sees no key at all
                for (int d = 0; d < dV; d++) {
                    partial[b + d] = masked ? 0f : 4f * (rnd.nextFloat() - 0.5f);
                }
                partial[b + dV] = masked ? Float.NEGATIVE_INFINITY : 6f * (rnd.nextFloat() - 0.5f);
                partial[b + dV + 1] = masked ? 0f : 0.5f + 3f * rnd.nextFloat();
            }
        }
        float[] got = run(partial, sink, dV, splits, sinks, local);
        String where = String.format(Locale.ROOT, "%s %s d_v=%d splits=%d sinks=%b local=%d", api, deviceName, dV,
                splits, sinks, local[0]);
        for (int h = 0; h < N_HEAD; h++) {
            double m = sinks ? sink[h] : Double.NEGATIVE_INFINITY;
            for (int s = 0; s < splits; s++) {
                m = Math.max(m, partial[(h * splits + s) * stride + dV]);
            }
            double l = sinks ? Math.exp(sink[h] - m) : 0;
            double[] o = new double[dV];
            if (m != Double.NEGATIVE_INFINITY) {
                for (int s = 0; s < splits; s++) {
                    int b = (h * splits + s) * stride;
                    double w = partial[b + dV] == Float.NEGATIVE_INFINITY ? 0 : Math.exp(partial[b + dV] - m);
                    l += partial[b + dV + 1] * w;
                    for (int d = 0; d < dV; d++) {
                        o[d] += partial[b + d] * w;
                    }
                }
            }
            for (int d = 0; d < dV; d++) {
                double want = l > 0 ? o[d] / l : 0;
                float g = got[h * dV + d];
                String at = where + " head " + h + " element " + d;
                if (h == N_HEAD - 1 && !sinks) {
                    assertEquals(at, 0f, g, 0f);
                } else {
                    assertTrue(at + ": want " + want + ", got " + g,
                            Math.abs(want - g) <= 2e-5 * Math.max(1.0, Math.abs(want)));
                }
            }
        }
    }

    private float[] run(float[] partial, float[] sink, int dV, int splits, boolean sinks, long[] local) {
        GpuBuffer p = upload(partial);
        GpuBuffer s = upload(sink);
        float[] poison = new float[N_HEAD * dV];
        java.util.Arrays.fill(poison, Float.NaN);
        GpuBuffer out = upload(poison);
        Object[] args = {N_HEAD, dV, splits, N_HEAD, sinks ? 1 : 0, 0, p, s, out};
        close(backend.computeQueue().enqueueKernel(kernels.get(AttnKernels.FLASH_COMBINE),
                new long[]{N_HEAD * local[0]}, local, args));
        float[] y = new float[N_HEAD * dV];
        close(backend.computeQueue().enqueueReadBuffer(out, 0, (long) y.length * 4, y));
        backend.computeQueue().finish();
        return y;
    }

    private GpuBuffer upload(float[] data) {
        GpuBuffer b = backend.allocateBuffer((long) data.length * 4, true);
        bufs.add(b);
        close(backend.computeQueue().enqueueWriteBuffer(b, 0, (long) data.length * 4, data));
        return b;
    }

    private static void close(GpuEvent e) {
        if (e != null) {
            e.close();
        }
    }
}

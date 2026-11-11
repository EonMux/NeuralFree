package com.eonmux.neuralfree.gguf.ops.audio;

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
import java.util.Arrays;
import java.util.EnumSet;
import java.util.List;
import java.util.Locale;
import java.util.Random;

import static org.junit.Assert.assertTrue;

/**
 * The speech attention kernels ({@code LLM_TtsCausalAttn}, {@code LLM_TtsWindowAttn}, {@code LLM_TtsFullAttn} in
 * kernels/llm/speech.cl, GLSL twins shaders/llm/tts_*_attn.comp), one work-group per (row, head), against a
 * double-precision softmax over the same visible keys: key ranges shorter and longer than one staged chunk (256
 * keys), head dims below, at and past one pass of accumulators (4 per work-item), grouped KV heads, the window band,
 * the cold-start mask (negative absolute key positions) and rows that see no key at all (written as zeros). Runs on
 * CUDA, and on OpenCL and Vulkan on each device (the Intel iGPU and the NVIDIA GPU), skipping what the machine lacks.
 */
@RunWith(Parameterized.class)
public class SpeechAttentionKernelGpuTest {

    private static final int N_HEAD = 4;
    private static final int N_HEAD_KV = 2;
    private static final int[] DIMS = {16, 64, 300};

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

    public SpeechAttentionKernelGpuTest(ComputeBackend.BackendType api, String deviceName) {
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
        releaseAll();
        if (backend != null) {
            backend.close();
        }
        if (hw != null) {
            hw.shutdown();
        }
    }

    @Test
    public void causalStepMatchesTheReference() {
        Random rnd = new Random(21);
        for (int d : DIMS) {
            for (int pos : new int[]{0, 5, 255, 256, 600}) {
                int rows = pos + 1;
                float[] q = random(rnd, N_HEAD * d);
                float[] k = random(rnd, rows * N_HEAD_KV * d);
                float[] v = random(rnd, rows * N_HEAD_KV * d);
                float[] got = run(SpeechKernels.TTS_CAUSAL_ATTN, SpeechKernels.causalAttentionGlobal(N_HEAD),
                        N_HEAD * d, q, k, v, N_HEAD, N_HEAD_KV, d, pos);
                check("causal d=" + d + " pos=" + pos, got, reference(q, k, v, 1, d, (i, j) -> j <= pos, rows));
            }
        }
    }

    @Test
    public void windowRowsMatchTheReference() {
        Random rnd = new Random(22);
        // (prefix, n_q, window, base): a short band, a band longer than a chunk, a cold start whose first rows see
        // no key at all, and a window wider than everything
        int[][] shapes = {{8, 6, 4, 8}, {300, 40, 280, 300}, {20, 12, 6, -25}, {0, 9, 1000, 0}};
        for (int d : DIMS) {
            for (int[] shape : shapes) {
                int prefix = shape[0];
                int nQ = shape[1];
                int window = shape[2];
                int base = shape[3];
                int rows = prefix + nQ;
                float[] q = random(rnd, nQ * N_HEAD * d);
                float[] k = random(rnd, rows * N_HEAD_KV * d);
                float[] v = random(rnd, rows * N_HEAD_KV * d);
                float[] got = run(SpeechKernels.TTS_WINDOW_ATTN, SpeechKernels.windowAttentionGlobal(nQ, N_HEAD),
                        nQ * N_HEAD * d, q, k, v, nQ, N_HEAD, N_HEAD_KV, d, prefix, window, base);
                Visible vis = (i, j) -> j - prefix + base >= 0 && prefix + i - j >= 0 && prefix + i - j < window;
                check(String.format(Locale.ROOT, "window d=%d prefix=%d n_q=%d window=%d base=%d", d, prefix, nQ,
                        window, base), got, reference(q, k, v, nQ, d, vis, rows));
            }
        }
    }

    @Test
    public void fullRowsMatchTheReference() {
        Random rnd = new Random(23);
        for (int d : DIMS) {
            for (int nPos : new int[]{1, 8, 257}) {
                float[] q = random(rnd, nPos * N_HEAD * d);
                float[] k = random(rnd, nPos * N_HEAD_KV * d);
                float[] v = random(rnd, nPos * N_HEAD_KV * d);
                float[] got = run(SpeechKernels.TTS_FULL_ATTN, SpeechKernels.fullAttentionGlobal(nPos, N_HEAD),
                        nPos * N_HEAD * d, q, k, v, nPos, N_HEAD, N_HEAD_KV, d);
                check("full d=" + d + " n_pos=" + nPos, got, reference(q, k, v, nPos, d, (i, j) -> true, nPos));
            }
        }
    }

    private interface Visible {
        boolean test(int row, int key);
    }

    private static double[] reference(float[] q, float[] k, float[] v, int nQ, int d, Visible vis, int nKeys) {
        double[] out = new double[nQ * N_HEAD * d];
        int kvCh = N_HEAD_KV * d;
        double scale = 1.0 / Math.sqrt(d);
        for (int i = 0; i < nQ; i++) {
            for (int h = 0; h < N_HEAD; h++) {
                int kvOff = h / (N_HEAD / N_HEAD_KV) * d;
                int qOff = (i * N_HEAD + h) * d;
                double[] s = new double[nKeys];
                double max = Double.NEGATIVE_INFINITY;
                for (int j = 0; j < nKeys; j++) {
                    if (!vis.test(i, j)) {
                        continue;
                    }
                    double dot = 0;
                    for (int e = 0; e < d; e++) {
                        dot += (double) q[qOff + e] * k[j * kvCh + kvOff + e];
                    }
                    s[j] = dot * scale;
                    max = Math.max(max, s[j]);
                }
                if (max == Double.NEGATIVE_INFINITY) {
                    continue;
                }
                double sum = 0;
                double[] acc = new double[d];
                for (int j = 0; j < nKeys; j++) {
                    if (!vis.test(i, j)) {
                        continue;
                    }
                    double w = Math.exp(s[j] - max);
                    sum += w;
                    for (int e = 0; e < d; e++) {
                        acc[e] += w * v[j * kvCh + kvOff + e];
                    }
                }
                for (int e = 0; e < d; e++) {
                    out[qOff + e] = acc[e] / sum;
                }
            }
        }
        return out;
    }

    private void check(String where, float[] got, double[] want) {
        String at = api + " " + deviceName + " " + where;
        for (int i = 0; i < want.length; i++) {
            assertTrue(at + " element " + i + ": want " + want[i] + ", got " + got[i],
                    Math.abs(want[i] - got[i]) <= 2e-5 * Math.max(1.0, Math.abs(want[i])));
        }
    }

    private float[] run(String kernel, long global, int outLen, float[] q, float[] k, float[] v, Object... scalars) {
        float[] poison = new float[outLen];
        Arrays.fill(poison, Float.NaN);
        GpuBuffer out = upload(poison);
        Object[] args = Arrays.copyOf(scalars, scalars.length + 4);
        args[scalars.length] = upload(q);
        args[scalars.length + 1] = upload(k);
        args[scalars.length + 2] = upload(v);
        args[scalars.length + 3] = out;
        close(backend.computeQueue().enqueueKernel(kernels.get(kernel), new long[]{global},
                new long[]{SpeechKernels.LOCAL}, args));
        float[] y = new float[outLen];
        close(backend.computeQueue().enqueueReadBuffer(out, 0, (long) outLen * 4, y));
        backend.computeQueue().finish();
        releaseAll();
        return y;
    }

    private static float[] random(Random rnd, int n) {
        float[] x = new float[n];
        for (int i = 0; i < n; i++) {
            x[i] = 2f * (rnd.nextFloat() - 0.5f);
        }
        return x;
    }

    private GpuBuffer upload(float[] data) {
        GpuBuffer b = backend.allocateBuffer((long) data.length * 4, true);
        bufs.add(b);
        close(backend.computeQueue().enqueueWriteBuffer(b, 0, (long) data.length * 4, data));
        return b;
    }

    private void releaseAll() {
        for (GpuBuffer b : bufs) {
            b.close();
        }
        bufs.clear();
    }

    private static void close(GpuEvent e) {
        if (e != null) {
            e.close();
        }
    }
}

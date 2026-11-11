package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.vulkan.VkComputeBackend;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.quant.FpBlockDecoders;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Random;

/**
 * Opt-in time ({@code -Dneuralfree.test.qkv.measure=true}) of the single-token Q/K/V projections on the Vulkan
 * device: three separate launches per lane (all float {@code {T}_MatVec}, all integer-dot {@code {T}_MatVec_Q8},
 * all paired {@code {T}_MatVec2_Q8}, and the Q projection paired with float K and V) against one stacked
 * {@code {T}_MatVecQKV} or {@code {T}_MatVecQKV_Q8} launch, over the attention shapes of the 0.6B and 4B Qwen3
 * decoders. Each lane runs a chain of layers whose weights rotate over more bytes than the device cache holds, as a
 * decode step streams them; the integer-dot lanes include the Q8_1 quantization of the row. The printed time is the
 * median over alternating rounds of the chain's host wall time per layer. Asserts nothing; the numbers set
 * {@link GpuProjections}'s stacked-lane rule. Types by {@code -Dneuralfree.test.qkv.types} (default Q8_0,Q4_K).
 */
public class StackedQkvMeasureTest {

    private static final String PROP = "neuralfree.test.qkv.measure";
    private static final long ROTATION_BYTES = 256L << 20;
    private static final int ROUNDS = 7;
    /** {q rows, k rows, v rows, columns}. */
    private static final int[][] SHAPES = {{2048, 1024, 1024, 1024}, {4096, 1024, 1024, 2560}};

    private ComputeBackend backend;
    private LlmKernels kernels;
    private int local;

    @Before
    public void open() {
        Assume.assumeTrue("opt-in (-D" + PROP + "=true)", Boolean.getBoolean(PROP));
        backend = new VkComputeBackend();
        kernels = LlmKernels.open(backend);
        local = kernels.matvecLocal();
    }

    @After
    public void close() {
        if (backend != null) {
            backend.close();
            backend = null;
        }
    }

    @Test
    public void measure() throws Exception {
        System.out.println("QKV device=\"" + backend.capabilities().getDeviceName() + "\" matvecLocal=" + local);
        for (String t : System.getProperty("neuralfree.test.qkv.types", "Q8_0,Q4_K").split(",")) {
            for (int[] shape : SHAPES) {
                measureShape(GGMLType.valueOf(t.trim()), shape);
            }
        }
    }

    private void measureShape(GGMLType type, int[] shape) throws Exception {
        int qRows = shape[0];
        int kRows = shape[1];
        int vRows = shape[2];
        int cols = shape[3];
        long rowBytes = (long) cols / type.getBlockSize() * type.getTypeSize();
        long setBytes = rowBytes * (qRows + kRows + vRows);
        int layers = (int) Math.max(4, Math.min(64, (ROTATION_BYTES + setBytes - 1) / setBytes));
        List<GpuBuffer[]> sets = new ArrayList<>();
        GpuBuffer x = upload(activations(cols));
        GpuBuffer xq = backend.allocateBuffer((long) cols / 32 * GpuOps.Q8_1_FLOATS * Float.BYTES);
        GpuBuffer yq = backend.allocateBuffer((long) qRows * Float.BYTES);
        GpuBuffer yk = backend.allocateBuffer((long) kRows * Float.BYTES);
        GpuBuffer yv = backend.allocateBuffer((long) vRows * Float.BYTES);
        try {
            for (int l = 0; l < layers; l++) {
                sets.add(new GpuBuffer[]{
                    upload(weights(type, qRows, cols, l)), upload(weights(type, kRows, cols, l + 100)),
                    upload(weights(type, vRows, cols, l + 200)),
                });
            }
            String t = type.name();
            String fl = t + "_MatVec";
            String q8 = t + "_MatVec_Q8";
            String paired = t + "_MatVec2_Q8";
            String[] names = {"3xfloat", "3xq8", "3xq8pair", "qpair+kvfloat", "stack-float", "stack-q8"};
            Layer[] lanes = {
                w -> {
                    single(fl, qRows, cols, w[0], x, yq);
                    single(fl, kRows, cols, w[1], x, yk);
                    single(fl, vRows, cols, w[2], x, yv);
                },
                w -> {
                    quantize(cols, x, xq);
                    single(q8, qRows, cols, w[0], xq, yq);
                    single(q8, kRows, cols, w[1], xq, yk);
                    single(q8, vRows, cols, w[2], xq, yv);
                },
                w -> {
                    quantize(cols, x, xq);
                    pair(paired, qRows, cols, w[0], xq, yq);
                    pair(paired, kRows, cols, w[1], xq, yk);
                    pair(paired, vRows, cols, w[2], xq, yv);
                },
                w -> {
                    quantize(cols, x, xq);
                    pair(paired, qRows, cols, w[0], xq, yq);
                    single(fl, kRows, cols, w[1], x, yk);
                    single(fl, vRows, cols, w[2], x, yv);
                },
                w -> stacked(GpuProjections.stackedQkvName(fl), qRows, kRows, vRows, cols, w, x, yq, yk, yv),
                w -> {
                    quantize(cols, x, xq);
                    stacked(GpuProjections.stackedQkvName(q8), qRows, kRows, vRows, cols, w, xq, yq, yk, yv);
                },
            };
            double[][] samples = new double[lanes.length][ROUNDS];
            for (int r = 0; r < ROUNDS; r++) {
                for (int i = 0; i < lanes.length; i++) {
                    int lane = r % 2 == 0 ? i : lanes.length - 1 - i;
                    samples[lane][r] = chainMicrosPerLayer(lanes[lane], sets);
                }
            }
            StringBuilder line = new StringBuilder(String.format(Locale.ROOT, "QKV %-5s %d+%d+%dx%d layers=%d",
                    t, qRows, kRows, vRows, cols, layers));
            for (int i = 0; i < lanes.length; i++) {
                Arrays.sort(samples[i]);
                line.append(String.format(Locale.ROOT, "  %s=%.2fus", names[i], samples[i][ROUNDS / 2]));
            }
            System.out.println(line);
        } finally {
            for (GpuBuffer[] set : sets) {
                for (GpuBuffer b : set) {
                    b.close();
                }
            }
            x.close();
            xq.close();
            yq.close();
            yk.close();
            yv.close();
        }
    }

    private interface Layer {
        void run(GpuBuffer[] weights);
    }

    /** Host wall microseconds per layer of one chain over every weight set, after a warm-up chain. */
    private double chainMicrosPerLayer(Layer layer, List<GpuBuffer[]> sets) {
        for (GpuBuffer[] set : sets) {
            layer.run(set);
        }
        backend.computeQueue().finish();
        long t0 = System.nanoTime();
        for (GpuBuffer[] set : sets) {
            layer.run(set);
        }
        backend.computeQueue().finish();
        return (System.nanoTime() - t0) / 1e3 / sets.size();
    }

    private void single(String kernel, int rows, int cols, GpuBuffer w, GpuBuffer in, GpuBuffer y) {
        enqueue(kernel, (long) rows * local, rows, cols, w, in, y);
    }

    private void pair(String kernel, int rows, int cols, GpuBuffer w, GpuBuffer in, GpuBuffer y) {
        enqueue(kernel, (long) (rows + 1) / 2 * local, rows, cols, w, in, y);
    }

    private void stacked(String kernel, int qRows, int kRows, int vRows, int cols, GpuBuffer[] w, GpuBuffer in,
                         GpuBuffer yq, GpuBuffer yk, GpuBuffer yv) {
        enqueue(kernel, (long) (qRows + kRows + vRows) * local, qRows, cols, kRows, vRows, w[0], w[1], w[2], in,
                yq, yk, yv);
    }

    private void quantize(int cols, GpuBuffer x, GpuBuffer xq) {
        long blocks = cols / 32;
        enqueue(GpuOps.QUANTIZE_Q8_1, (blocks + local - 1) / local * local, cols, x, xq);
    }

    private void enqueue(String kernel, long global, Object... args) {
        backend.computeQueue().enqueueKernel(kernels.get(kernel), new long[]{global}, new long[]{local}, args);
    }

    private GpuBuffer upload(byte[] data) throws Exception {
        GpuBuffer buf = backend.allocateBuffer(data.length);
        backend.computeQueue().enqueueWriteBuffer(buf, 0, data.length, data).awaitCompletion();
        return buf;
    }

    private GpuBuffer upload(float[] data) throws Exception {
        GpuBuffer buf = backend.allocateBuffer((long) data.length * Float.BYTES);
        backend.computeQueue().enqueueWriteBuffer(buf, 0, (long) data.length * Float.BYTES, data).awaitCompletion();
        return buf;
    }

    /** Random blocks of {@code type} whose scale fields hold small finite halves. */
    private static byte[] weights(GGMLType type, int rows, int cols, long seed) {
        int blockBytes = type.getTypeSize();
        int blocks = rows * (cols / type.getBlockSize());
        byte[] data = new byte[blocks * blockBytes];
        new Random(seed).nextBytes(data);
        int dOff = type == GGMLType.Q6_K ? 208 : 0;
        int bits = FpBlockDecoders.fp32ToFp16(0.01f);
        for (int b = 0; b < blocks; b++) {
            int base = b * blockBytes + dOff;
            data[base] = (byte) bits;
            data[base + 1] = (byte) (bits >>> 8);
            if (type == GGMLType.Q4_K || type == GGMLType.Q5_K) {
                data[base + 2] = (byte) bits;
                data[base + 3] = (byte) (bits >>> 8);
            }
        }
        return data;
    }

    private static float[] activations(int n) {
        Random rnd = new Random(n);
        float[] x = new float[n];
        for (int i = 0; i < n; i++) {
            x[i] = rnd.nextFloat() * 2f - 1f;
        }
        return x;
    }
}

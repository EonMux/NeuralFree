package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.vulkan.VkComputeBackend;
import com.eonmux.neuralfree.acceleration.kernels.Availability;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.quant.FpBlockDecoders;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;

import java.util.Arrays;
import java.util.Random;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertTrue;

/**
 * The stacked Q/K/V matvec kernels on the Vulkan device ({@code {T}_MatVecQKV} over the float row and
 * {@code {T}_MatVecQKV_Q8} over the Q8_1 row) against three launches of the single-weight kernel they stack
 * ({@code {T}_MatVec} / {@code {T}_MatVec_Q8}) over the same weights and row: every output must match bit for bit,
 * and no launch may write past the rows of its output. Shapes cover a 0.6B-class attention (2048 + 1024 + 1024 rows
 * of 1024), a 4B-class one (4096 + 1024 + 1024 of 2560) and uneven small row counts. Skipped without a Vulkan device
 * or the LLM kernel library.
 */
public class StackedQkvGpuTest {

    private static final GGMLType[] TYPES = {
        GGMLType.Q4_0, GGMLType.Q8_0, GGMLType.Q4_K, GGMLType.Q5_K, GGMLType.Q6_K,
    };
    /** {q rows, k rows, v rows, columns}. */
    private static final int[][] SHAPES = {
        {2048, 1024, 1024, 1024}, {4096, 1024, 1024, 2560}, {96, 32, 64, 512}, {7, 3, 5, 256},
    };
    /** Floats past each output's rows that no launch may write. */
    private static final int GUARD = 64;
    private static final float SENTINEL = -12345.0f;

    private ComputeBackend backend;
    private LlmKernels kernels;
    private int local;

    @Before
    public void openDevice() {
        try {
            backend = new VkComputeBackend();
            Availability availability = LlmKernels.availability(backend);
            Assume.assumeTrue("LLM kernels unavailable: " + availability.reason(), availability.available());
            kernels = LlmKernels.open(backend);
            local = kernels.matvecLocal();
        } catch (Throwable e) {
            closeDevice();
            Assume.assumeTrue("no usable Vulkan GPU: " + e.getMessage(), false);
        }
    }

    @After
    public void closeDevice() {
        if (backend != null) {
            try {
                backend.close();
            } catch (RuntimeException ignored) {
            }
            backend = null;
        }
    }

    @Test
    public void floatStackMatchesThreeSingleLaunchesBitForBit() throws Exception {
        for (GGMLType type : TYPES) {
            for (int[] shape : SHAPES) {
                check(type, shape, false);
            }
        }
    }

    @Test
    public void int8StackMatchesThreeSingleLaunchesBitForBit() throws Exception {
        for (GGMLType type : TYPES) {
            for (int[] shape : SHAPES) {
                check(type, shape, true);
            }
        }
    }

    private void check(GGMLType type, int[] shape, boolean int8) throws Exception {
        int qRows = shape[0];
        int kRows = shape[1];
        int vRows = shape[2];
        int cols = shape[3];
        String single = type.name() + (int8 ? "_MatVec_Q8" : "_MatVec");
        String stacked = GpuProjections.stackedQkvName(single);
        long groups = (long) qRows + kRows + vRows;
        assertTrue(stacked + " missing", kernels.has(stacked));
        long seed = 31L * type.ordinal() + cols;
        GpuBuffer wq = upload(weights(type, qRows, cols, seed));
        GpuBuffer wk = upload(weights(type, kRows, cols, seed + 1));
        GpuBuffer wv = upload(weights(type, vRows, cols, seed + 2));
        GpuBuffer x = upload(activations(cols, seed + 3));
        GpuBuffer row = int8 ? q8_1(x, cols) : x;
        try {
            float[][] want = {
                single(single, qRows, cols, wq, row), single(single, kRows, cols, wk, row),
                single(single, vRows, cols, wv, row),
            };
            GpuBuffer yq = guarded(qRows);
            GpuBuffer yk = guarded(kRows);
            GpuBuffer yv = guarded(vRows);
            backend.computeQueue().enqueueKernel(kernels.get(stacked),
                    new long[]{groups * local}, new long[]{local},
                    new Object[]{qRows, cols, kRows, vRows, wq, wk, wv, row, yq, yk, yv});
            backend.computeQueue().finish();
            String what = stacked + " " + Arrays.toString(shape);
            assertGuarded(what + " Q", want[0], read(yq, qRows + GUARD));
            assertGuarded(what + " K", want[1], read(yk, kRows + GUARD));
            assertGuarded(what + " V", want[2], read(yv, vRows + GUARD));
            yq.close();
            yk.close();
            yv.close();
        } finally {
            wq.close();
            wk.close();
            wv.close();
            if (row != x) {
                row.close();
            }
            x.close();
        }
    }

    private float[] single(String kernel, int rows, int cols, GpuBuffer w, GpuBuffer row) throws Exception {
        GpuBuffer y = backend.allocateBuffer((long) rows * Float.BYTES);
        backend.computeQueue().enqueueKernel(kernels.get(kernel), new long[]{(long) rows * local},
                new long[]{local}, new Object[]{rows, cols, w, row, y});
        backend.computeQueue().finish();
        float[] out = read(y, rows);
        y.close();
        return out;
    }

    private static void assertGuarded(String what, float[] want, float[] got) {
        assertArrayEquals(what, want, Arrays.copyOf(got, want.length), 0.0f);
        for (int i = want.length; i < got.length; i++) {
            assertTrue(what + " wrote past its rows at " + i, got[i] == SENTINEL);
        }
    }

    private GpuBuffer guarded(int rows) throws Exception {
        float[] init = new float[rows + GUARD];
        Arrays.fill(init, SENTINEL);
        return upload(init);
    }

    private float[] read(GpuBuffer buf, int count) throws Exception {
        float[] out = new float[count];
        backend.computeQueue().enqueueReadBuffer(buf, 0, (long) count * Float.BYTES, out).awaitCompletion();
        return out;
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

    /** The Q8_1 blocks of the row as the quantizer kernel writes them. */
    private GpuBuffer q8_1(GpuBuffer x, int cols) {
        int blocks = cols / GpuOps.Q8_1_BLOCK_SIZE;
        GpuBuffer q = backend.allocateBuffer((long) blocks * GpuOps.Q8_1_FLOATS * Float.BYTES);
        backend.computeQueue().enqueueKernel(kernels.get(GpuOps.QUANTIZE_Q8_1),
                new long[]{(blocks + local - 1L) / local * local}, new long[]{local}, new Object[]{cols, x, q});
        backend.computeQueue().finish();
        return q;
    }

    /** Random blocks of {@code type} whose scale fields hold small finite halves. */
    private static byte[] weights(GGMLType type, int rows, int cols, long seed) {
        Random rnd = new Random(seed);
        int blocks = rows * (cols / type.getBlockSize());
        int blockBytes = type.getTypeSize();
        byte[] data = new byte[blocks * blockBytes];
        rnd.nextBytes(data);
        for (int b = 0; b < blocks; b++) {
            int base = b * blockBytes;
            switch (type) {
                case Q4_K:
                case Q5_K:
                    putHalf(data, base, 0.02f);
                    putHalf(data, base + 2, 0.01f);
                    break;
                case Q6_K:
                    putHalf(data, base + 208, 0.01f);
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

    private static float[] activations(int n, long seed) {
        Random rnd = new Random(seed);
        float[] x = new float[n];
        for (int i = 0; i < n; i++) {
            x[i] = rnd.nextFloat() * 2f - 1f;
        }
        return x;
    }
}

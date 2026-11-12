package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.cuda.CudaComputeBackend;
import com.eonmux.neuralfree.acceleration.kernels.Availability;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.quant.FpBlockDecoders;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;

import java.util.Arrays;
import java.util.EnumSet;
import java.util.Random;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertTrue;

/**
 * The CUDA warp-row Q/K/V stacks ({@code {T}_MatVecQKVRow_Q8} and {@code {T}_MatVecQKVRowVQ6_K_Q8},
 * kernels/llm/matvec_row.cl) against three launches of the one-row form they stack ({@code {T}_MatVecRow_Q8}, the
 * V rows through the V type's form) over the same weights and Q8_1 row, at the warps per work-group the host picks
 * ({@link WarpRowMatvec#warps}): every output must match bit for bit, and no launch may write past the rows of its
 * output. Shapes cover a 0.6B-class attention (2048 + 1024 + 1024 rows of 1024), a 4B-class one
 * (4096 + 1024 + 1024 of 2560, the K-quants only: Q8_0 rows that wide keep the family) and uneven small row counts
 * whose stack ends inside a work-group. Skipped without a CUDA device or the LLM kernel library.
 */
public class StackedQkvRowGpuTest {

    /** {Q/K type, V type}. */
    private static final GGMLType[][] TYPES = {
        {GGMLType.Q8_0, GGMLType.Q8_0}, {GGMLType.Q4_K, GGMLType.Q4_K}, {GGMLType.Q5_K, GGMLType.Q5_K},
        {GGMLType.Q6_K, GGMLType.Q6_K}, {GGMLType.Q4_K, GGMLType.Q6_K}, {GGMLType.Q5_K, GGMLType.Q6_K},
    };
    /** {q rows, k rows, v rows, columns}. */
    private static final int[][] SHAPES = {
        {2048, 1024, 1024, 1024}, {4096, 1024, 1024, 2560}, {7, 3, 5, 512}, {9, 4, 2, 256},
    };
    /** Floats past each output's rows that no launch may write. */
    private static final int GUARD = 64;
    private static final float SENTINEL = -12345.0f;
    private static final int LOCAL = 256;

    private HWAcceleration hwAccel;
    private ComputeBackend backend;
    private LlmKernels kernels;

    @Before
    public void openDevice() {
        GpuTestUtil.assumeCudaAvailable();
        try {
            hwAccel = new HWAcceleration(EnumSet.of(ComputeBackend.BackendType.CUDA));
            Assume.assumeFalse("no CUDA device", hwAccel.getCUDevices().isEmpty());
            backend = new CudaComputeBackend(hwAccel.getCUDevices().get(0));
            Availability availability = LlmKernels.availability(backend);
            Assume.assumeTrue("LLM kernels unavailable: " + availability.reason(), availability.available());
            kernels = LlmKernels.open(backend);
        } catch (Throwable e) {
            closeDevice();
            Assume.assumeTrue("no usable CUDA GPU: " + e.getMessage(), false);
        }
    }

    @After
    public void closeDevice() {
        if (backend != null) {
            backend.close();
            backend = null;
        }
        if (hwAccel != null) {
            hwAccel.shutdown();
            hwAccel = null;
        }
    }

    @Test
    public void stackMatchesThreeOneRowLaunchesBitForBit() throws Exception {
        int checked = 0;
        for (GGMLType[] types : TYPES) {
            for (int[] shape : SHAPES) {
                if (WarpRowMatvec.warps(types[0], shape[0], shape[3]) > 0
                        && WarpRowMatvec.warps(types[1], shape[2], shape[3]) > 0) {
                    check(types[0], types[1], shape);
                    checked++;
                }
            }
        }
        assertTrue("shapes checked: " + checked, checked >= 18);
    }

    private void check(GGMLType qk, GGMLType vType, int[] shape) throws Exception {
        int qRows = shape[0];
        int kRows = shape[1];
        int vRows = shape[2];
        int cols = shape[3];
        String stacked = WarpRowMatvec.qkvTwinName(qk, vType);
        assertTrue(stacked + " missing", stacked != null && kernels.has(stacked));
        long seed = 31L * qk.ordinal() + 7L * vType.ordinal() + cols;
        GpuBuffer wq = upload(weights(qk, qRows, cols, seed));
        GpuBuffer wk = upload(weights(qk, kRows, cols, seed + 1));
        GpuBuffer wv = upload(weights(vType, vRows, cols, seed + 2));
        GpuBuffer x = upload(activations(cols, seed + 3));
        GpuBuffer row = q8_1(x, cols);
        try {
            float[][] want = {
                single(qk, qRows, cols, wq, row), single(qk, kRows, cols, wk, row), single(vType, vRows, cols, wv, row),
            };
            GpuBuffer yq = guarded(qRows);
            GpuBuffer yk = guarded(kRows);
            GpuBuffer yv = guarded(vRows);
            int warps = WarpRowMatvec.warps(qk, qRows, cols);
            backend.computeQueue().enqueueKernel(kernels.get(stacked),
                    new long[]{WarpRowMatvec.globalSize(qRows + kRows + vRows, warps)},
                    new long[]{WarpRowMatvec.localSize(warps)},
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
            row.close();
            x.close();
        }
    }

    /** The one-row form of {@code type} over {@code rows} rows at the host's warps for that shape. */
    private float[] single(GGMLType type, int rows, int cols, GpuBuffer w, GpuBuffer row) throws Exception {
        int warps = WarpRowMatvec.warps(type, rows, cols);
        GpuBuffer y = backend.allocateBuffer((long) rows * Float.BYTES);
        backend.computeQueue().enqueueKernel(kernels.get(type.name() + "_MatVecRow_Q8"),
                new long[]{WarpRowMatvec.globalSize(rows, warps)}, new long[]{WarpRowMatvec.localSize(warps)},
                new Object[]{rows, cols, w, row, y});
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
                new long[]{(blocks + LOCAL - 1L) / LOCAL * LOCAL}, new long[]{LOCAL}, new Object[]{cols, x, q});
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

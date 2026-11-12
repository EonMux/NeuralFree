package com.eonmux.neuralfree.gguf.ops.sparse;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.HardwareDevice;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.acceleration.kernels.Availability;
import com.eonmux.neuralfree.gguf.gpu.head.GpuArgmax;
import com.eonmux.neuralfree.gguf.gpu.head.GpuLogitsHead;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.ops.gemm.GemmKernels;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.EnumSet;
import java.util.List;
import java.util.Locale;
import java.util.Random;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * The row selection kernels of sampling and sparse attention: {@code LLM_Argmax} (kernels/llm/sampling.cl, twin
 * shaders/llm/argmax.comp) picks the first maximum of each row and never NaN unless the row holds nothing else (an
 * all-NaN row yields 0), whichever reduction tree the backend runs, and the split pair {@code LLM_ArgmaxSplit} /
 * {@code LLM_ArgmaxCombine} ({@link GpuArgmax}) picks the same index at every chunk count, ties and NaN straddling
 * chunk edges included; and the two work-group top-k kernels over the
 * shared radix select ({@code kernels/llm/common/topk_select.h}, GLSL
 * twin {@code shaders/llm/common/topk_select.glsl}) against host references, on CUDA, and on OpenCL and Vulkan on
 * each device (the Intel iGPU and the NVIDIA GPU), skipping what the machine lacks:
 * <ul>
 *   <li>{@code LLM_Row_TopK}: the k best non-NaN logits of each row in rank order (larger value first, then lower
 *       index; -INF ranks, NaN never does), -INF / -1 past the row's non-NaN count, and the row's softmax
 *       denominator. Widths below, inside and past the radix form's range (the argmax-round form) give the same
 *       answer, over rows with heavy ties, signed zeros, infinities, NaN, an all-NaN row and a 151936-wide
 *       vocabulary row.</li>
 *   <li>{@code LLM_TopK_Rows}: exactly the columns {@link CpuTopK} selects, in its order, for rows with -INF masks,
 *       NaN, ties and {@code k} past the eligible count, from 1 to 65536 columns.</li>
 * </ul>
 */
@RunWith(Parameterized.class)
public class RowSelectKernelGpuTest {

    private static final int ROWS = 4;
    private static final int[] LOCALS = {256, 64};

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

    public RowSelectKernelGpuTest(ComputeBackend.BackendType api, String deviceName) {
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
    public void argmaxPicksTheFirstMaximum() {
        Random rnd = new Random(10);
        for (int size : new int[]{1, 5, 1000, 151936}) {
            float[] x = rows(rnd, size);
            for (int local : LOCALS) {
                int[] got = runArgmax(x, size, local);
                for (int r = 0; r < ROWS; r++) {
                    int want = 0;
                    boolean found = false;
                    for (int i = 0; i < size; i++) {
                        float v = x[r * size + i];
                        if (!Float.isNaN(v) && (!found || v > x[r * size + want])) {
                            want = i;
                            found = true;
                        }
                    }
                    assertEquals(where("Argmax", size, 1, local) + " row " + r, want, got[r]);
                }
            }
        }
    }

    @Test
    public void splitArgmaxPicksTheFirstMaximumAtEveryChunkCount() {
        Random rnd = new Random(14);
        for (int local : LOCALS) {
            int chunk = GpuArgmax.UNROLL * local;
            for (int size : new int[]{1, 5, chunk - 1, chunk, chunk + 1, 3 * chunk + 7, 151936}) {
                float[] x = rows(rnd, size);
                if (size > chunk) {
                    // row 0: -0 ahead of +0 across a chunk edge (they tie, so the lower index wins); row 1: the
                    // maximum alone in the last element of the tail chunk
                    for (int i = 0; i < size; i++) {
                        x[i] = -1f - (i % 7);
                        x[size + i] = -2f;
                    }
                    x[chunk - 1] = -0f;
                    x[chunk + 1] = 0f;
                    x[size - 1] = 0f;
                    x[2 * size - 1] = 1f;
                }
                int[] want = hostArgmax(x, size);
                int[] got = runArgmaxSplit(x, size, local);
                assertArrayEquals(where("ArgmaxSplit", size, GpuArgmax.chunks(size, local), local), want, got);
                assertArrayEquals(where("Argmax", size, 1, local), want, runArgmax(x, size, local));
            }
        }
    }

    @Test
    public void rowTopKRanksLikeTheHostAtEveryWidth() {
        Random rnd = new Random(11);
        for (int size : new int[]{1, 5, 1000, 4099}) {
            float[] logits = rows(rnd, size);
            for (int k : new int[]{1, 7, 8, 40, 64, 300, 1024, 1100}) {
                for (int local : LOCALS) {
                    checkRowTopK(logits, size, k, local);
                }
            }
        }
    }

    @Test
    public void rowTopKOverAVocabularyRow() {
        Random rnd = new Random(12);
        int size = 151936;
        float[] logits = rows(rnd, size);
        for (int k : new int[]{1, 10, 40, 64, 1024}) {
            checkRowTopK(logits, size, k, 256);
        }
    }

    @Test
    public void topKRowsSelectsTheHostColumns() {
        Random rnd = new Random(13);
        for (int nCols : new int[]{1, 33, 1000, 32768, 65536}) {
            float[] scores = masked(rnd, nCols);
            for (int k : new int[]{1, 16, 256, 2048}) {
                for (int local : LOCALS) {
                    int[] want = new int[ROWS * k];
                    CpuTopK.topKRows(scores, 0, ROWS, nCols, k, want, 0);
                    int[] got = runTopKRows(scores, nCols, k, local);
                    assertArrayEquals(where("TopK_Rows", nCols, k, local), want, got);
                }
            }
        }
    }

    /**
     * Rows of {@code size} logits: row 0 uniform noise, row 1 a few repeated values (heavy ties, signed zeros,
     * both infinities), row 2 NaN-riddled ties, row 3 all NaN.
     */
    private static float[] rows(Random rnd, int size) {
        float[] x = new float[ROWS * size];
        float[] pool = {3f, 1.5f, 0f, -0f, -2f, Float.NEGATIVE_INFINITY, Float.POSITIVE_INFINITY, 1.5f};
        for (int i = 0; i < size; i++) {
            x[i] = 20f * (rnd.nextFloat() - 0.5f);
            x[size + i] = pool[rnd.nextInt(pool.length - 1)];
            x[2 * size + i] = rnd.nextInt(3) == 0 ? Float.NaN : Math.round(8f * rnd.nextFloat()) / 4f;
            x[3 * size + i] = Float.NaN;
        }
        if (size > 2) {
            x[size + size / 2] = Float.POSITIVE_INFINITY;
        }
        return x;
    }

    /** Index-scored rows with -INF masks, NaN and ties, the masked last row empty. */
    private static float[] masked(Random rnd, int nCols) {
        float[] x = new float[ROWS * nCols];
        for (int c = 0; c < nCols; c++) {
            x[c] = rnd.nextInt(4) == 0 ? Float.NEGATIVE_INFINITY : (float) rnd.nextGaussian();
            x[nCols + c] = rnd.nextInt(5) == 0 ? Float.NaN : Math.round(6f * rnd.nextFloat()) / 2f;
            x[2 * nCols + c] = c % 3 == 0 ? (c % 2 == 0 ? 0f : -0f) : Float.NEGATIVE_INFINITY;
            x[3 * nCols + c] = Float.NEGATIVE_INFINITY;
        }
        return x;
    }

    private void checkRowTopK(float[] logits, int size, int k, int local) {
        float[] gotVal = new float[ROWS * k];
        int[] gotIdx = new int[ROWS * k];
        float[] gotSum = new float[ROWS];
        runRowTopK(logits, size, k, local, gotVal, gotIdx, gotSum);
        for (int r = 0; r < ROWS; r++) {
            String at = where("Row_TopK", size, k, local) + " row " + r;
            Integer[] order = new Integer[size];
            int valid = 0;
            double max = Double.NEGATIVE_INFINITY;
            for (int i = 0; i < size; i++) {
                float v = logits[r * size + i];
                if (!Float.isNaN(v)) {
                    order[valid++] = i;
                    max = Math.max(max, v);
                }
            }
            final int row = r;
            // the kernels' total order: larger value first (float comparison, so -0 ties with +0), then lower index
            Arrays.sort(order, 0, valid, (a, b) -> {
                float va = logits[row * size + a];
                float vb = logits[row * size + b];
                return va > vb ? -1 : va < vb ? 1 : Integer.compare(a, b);
            });
            for (int j = 0; j < k; j++) {
                int wantIdx = j < valid ? order[j] : -1;
                float wantVal = j < valid ? logits[r * size + order[j]] : Float.NEGATIVE_INFINITY;
                assertEquals(at + " rank " + j + " index", wantIdx, gotIdx[r * k + j]);
                assertEquals(at + " rank " + j + " value", wantVal, gotVal[r * k + j], 0f);
            }
            double sum = 0;
            for (int i = 0; i < size; i++) {
                float v = logits[r * size + i];
                if (!Float.isNaN(v) && max != Double.NEGATIVE_INFINITY && max != Double.POSITIVE_INFINITY) {
                    sum += Math.exp(v - max);
                }
            }
            if (max != Double.NEGATIVE_INFINITY && max != Double.POSITIVE_INFINITY) {
                assertTrue(at + " sum_exp want " + sum + ", got " + gotSum[r],
                        Math.abs(sum - gotSum[r]) <= 1e-4 * Math.max(1.0, sum));
            }
        }
    }

    private void runRowTopK(float[] logits, int size, int k, int local, float[] val, int[] idx, float[] sum) {
        GpuBuffer in = upload(logits);
        GpuBuffer v = upload(new float[ROWS * k]);
        GpuBuffer i = upload(new int[ROWS * k]);
        GpuBuffer s = upload(new float[ROWS]);
        Object[] args = new Object[GemmKernels.ROWTOPK_ARG_COUNT];
        args[GemmKernels.ROWTOPK_ARG_SIZE] = size;
        args[GemmKernels.ROWTOPK_ARG_K] = k;
        args[GemmKernels.ROWTOPK_ARG_N_ROWS] = ROWS;
        args[GemmKernels.ROWTOPK_ARG_LOGITS] = in;
        args[GemmKernels.ROWTOPK_ARG_TOP_VAL] = v;
        args[GemmKernels.ROWTOPK_ARG_TOP_IDX] = i;
        args[GemmKernels.ROWTOPK_ARG_SUM_EXP] = s;
        close(backend.computeQueue().enqueueKernel(kernels.get(GemmKernels.ROW_TOPK),
                new long[]{(long) ROWS * local}, new long[]{local}, args));
        close(backend.computeQueue().enqueueReadBuffer(v, 0, (long) val.length * 4, val));
        byte[] raw = new byte[idx.length * 4];
        close(backend.computeQueue().enqueueReadBuffer(i, 0, raw.length, raw));
        close(backend.computeQueue().enqueueReadBuffer(s, 0, (long) sum.length * 4, sum));
        backend.computeQueue().finish();
        ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN).asIntBuffer().get(idx);
        releaseAll();
    }

    private int[] runArgmax(float[] x, int size, int local) {
        GpuBuffer in = upload(x);
        int[] poison = new int[ROWS];
        Arrays.fill(poison, -7);
        GpuBuffer out = upload(poison);
        close(backend.computeQueue().enqueueKernel(kernels.get(GpuLogitsHead.ARGMAX),
                new long[]{(long) ROWS * local}, new long[]{local}, new Object[]{size, in, out, ROWS}));
        int[] got = new int[ROWS];
        byte[] raw = new byte[got.length * 4];
        close(backend.computeQueue().enqueueReadBuffer(out, 0, raw.length, raw));
        backend.computeQueue().finish();
        ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN).asIntBuffer().get(got);
        releaseAll();
        return got;
    }

    /** The first maximum of each row under the kernels' order (NaN never, -0 ties +0), 0 for an all-NaN row. */
    private static int[] hostArgmax(float[] x, int size) {
        int[] want = new int[ROWS];
        for (int r = 0; r < ROWS; r++) {
            boolean found = false;
            for (int i = 0; i < size; i++) {
                float v = x[r * size + i];
                if (!Float.isNaN(v) && (!found || v > x[r * size + want[r]])) {
                    want[r] = i;
                    found = true;
                }
            }
        }
        return want;
    }

    private int[] runArgmaxSplit(float[] x, int size, int local) {
        int chunks = GpuArgmax.chunks(size, local);
        GpuBuffer in = upload(x);
        int[] poison = new int[ROWS];
        Arrays.fill(poison, -7);
        GpuBuffer out = upload(poison);
        int[] partPoison = new int[ROWS * chunks * 2];
        Arrays.fill(partPoison, -5);
        GpuBuffer part = upload(partPoison);
        close(backend.computeQueue().enqueueKernel(kernels.get(GpuArgmax.SPLIT),
                new long[]{(long) ROWS * chunks * local}, new long[]{local},
                new Object[]{size, chunks, in, part, ROWS}));
        close(backend.computeQueue().enqueueKernel(kernels.get(GpuArgmax.COMBINE),
                new long[]{(long) ROWS * local}, new long[]{local}, new Object[]{chunks, part, out, ROWS}));
        int[] got = new int[ROWS];
        byte[] raw = new byte[got.length * 4];
        close(backend.computeQueue().enqueueReadBuffer(out, 0, raw.length, raw));
        backend.computeQueue().finish();
        ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN).asIntBuffer().get(got);
        releaseAll();
        return got;
    }

    private int[] runTopKRows(float[] scores, int nCols, int k, int local) {
        GpuBuffer in = upload(scores);
        int[] poison = new int[ROWS * k];
        Arrays.fill(poison, 0x7eadbeef);
        GpuBuffer out = upload(poison);
        Object[] args = new Object[SparseKernels.ARG_TOPK_OUT + 1];
        args[SparseKernels.ARG_TOPK_N_ROWS] = ROWS;
        args[SparseKernels.ARG_TOPK_N_COLS] = nCols;
        args[SparseKernels.ARG_TOPK_K] = k;
        args[SparseKernels.ARG_TOPK_SCORES] = in;
        args[SparseKernels.ARG_TOPK_OUT] = out;
        close(backend.computeQueue().enqueueKernel(kernels.get(SparseKernels.TOPK_ROWS),
                SparseKernels.topKGlobal(ROWS, local), new long[]{local}, args));
        int[] got = new int[ROWS * k];
        byte[] raw = new byte[got.length * 4];
        close(backend.computeQueue().enqueueReadBuffer(out, 0, raw.length, raw));
        backend.computeQueue().finish();
        ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN).asIntBuffer().get(got);
        releaseAll();
        return got;
    }

    private String where(String kernel, int n, int k, int local) {
        return String.format(Locale.ROOT, "%s %s %s n=%d k=%d local=%d", kernel, api, deviceName, n, k, local);
    }

    private GpuBuffer upload(float[] data) {
        GpuBuffer b = backend.allocateBuffer((long) data.length * 4, true);
        bufs.add(b);
        close(backend.computeQueue().enqueueWriteBuffer(b, 0, (long) data.length * 4, data));
        return b;
    }

    private GpuBuffer upload(int[] data) {
        ByteBuffer bytes = ByteBuffer.allocate(data.length * 4).order(ByteOrder.LITTLE_ENDIAN);
        bytes.asIntBuffer().put(data);
        GpuBuffer b = backend.allocateBuffer((long) data.length * 4, true);
        bufs.add(b);
        close(backend.computeQueue().enqueueWriteBuffer(b, 0, (long) data.length * 4, bytes.array()));
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

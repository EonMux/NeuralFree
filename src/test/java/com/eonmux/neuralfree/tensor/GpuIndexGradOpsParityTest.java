package com.eonmux.neuralfree.tensor;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.tensor.gpu.TrainParity;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;

/**
 * {@link GpuIndexGradOps} on each GPU API against the CPU references of {@link CpuIndexing} on the host backend:
 * {@code TR_IndexAddRows_F32} in the single-rounding class (one writer per element, adding in index order, so the
 * sums equal the sequential CPU sums) across the row-block shape rule, both index modes, the skip row, INT64 and
 * strided operands; {@code TR_SegmentReduceBackward_F32} bit-identical for SUM and MAX (ties, NaN and signed zeros
 * included) and in the division class for MEAN, with empty segments and INT64 CSR arrays. Also the CPU-path
 * contract on a backend without the train library.
 */
@RunWith(Parameterized.class)
public class GpuIndexGradOpsParityTest {

    @Parameterized.Parameters(name = "{0}")
    public static Collection<Object[]> data() {
        return TrainParity.apis();
    }

    private final TrainParity.Api api;
    private TrainParity.Session session;
    private ComputeBackend gpu;
    private ComputeBackend host;
    private final List<Tensor> owned = new ArrayList<>();

    public GpuIndexGradOpsParityTest(TrainParity.Api api) {
        this.api = api;
    }

    @Before
    public void setUp() {
        host = TrainParity.hostBackend();
        session = TrainParity.open(api);
        gpu = session.backend();
    }

    @After
    public void tearDown() {
        try {
            for (int i = owned.size() - 1; i >= 0; i--) {
                owned.get(i).close();
            }
        } finally {
            try {
                if (session != null) {
                    session.close();
                }
            } finally {
                if (host != null) {
                    host.close();
                }
            }
        }
    }

    // ------------------------------------------------------------------ index add rows

    @Test
    public void indexAddRowsMatchesTheSequentialSum() {
        int[][] shapes = {{7, 5, 40}, {300, 2, 1000}, {3, 20000, 4}, {64, 1, 129}, {1, 3, 17}};
        long seed = 1;
        for (int[] s : shapes) {
            int rows = s[0];
            int dim = s[1];
            int n = s[2];
            float[] gy = TrainParity.uniform(seed++, n * dim, -3f, 3f);
            int[] idx = randomIndices(seed++, n, -2, rows + 2);
            for (boolean clamp : new boolean[]{false, true}) {
                for (int skip : new int[]{-1, rows / 2}) {
                    String what = "indexAddRows rows " + rows + " dim " + dim + " n " + n + " clamp " + clamp
                            + " skip " + skip;
                    Tensor ref = keep(CpuIndexing.indexAddRows(floats(host, gy, n, dim), ints(host, idx, n), rows,
                            clamp, skip));
                    Tensor out = keep(device(GpuIndexGradOps.indexAddRows(floats(gpu, gy, n, dim), ints(gpu, idx, n),
                            rows, clamp, skip)));
                    assertEquals(what + ": shape", ref.shape(), out.shape());
                    TrainParity.assertSingleRounding(what, ref.toFloatArray(), out.toFloatArray());
                }
            }
        }
    }

    @Test
    public void indexAddRowsTakesInt64AndStridedOperands() {
        int rows = 6;
        int dim = 4;
        int n = 10;
        float[] gy = TrainParity.uniform(11, n * dim, -2f, 2f);
        long[] idx = new long[n];
        Random r = new Random(12);
        for (int i = 0; i < n; i++) {
            int k = r.nextInt(rows + 2) - 1;
            idx[i] = ((long) r.nextInt(5) << 32) | (k & 0xffffffffL);
        }
        // The transposed [2, 5] layout stores logical index i at (i % 2) * 5 + i / 2.
        int[] narrowed = new int[n];
        for (int i = 0; i < n; i++) {
            narrowed[i] = (int) idx[(i % 2) * 5 + i / 2];
        }
        Tensor ref = keep(CpuIndexing.indexAddRows(keep(floats(host, gy, dim, n).transpose()), ints(host, narrowed,
                n), rows, false, 2));
        Tensor out = keep(device(GpuIndexGradOps.indexAddRows(keep(floats(gpu, gy, dim, n).transpose()),
                keep(int64(gpu, idx, 2, 5).transpose()), rows, false, 2)));
        Tensor refStrided = keep(CpuIndexing.indexAddRows(keep(floats(host, gy, dim, n).transpose()),
                keep(int64(host, idx, 2, 5).transpose()), rows, false, 2));
        TrainParity.assertBitIdentical("host INT64 strided indices", ref.toFloatArray(), refStrided.toFloatArray());
        TrainParity.assertSingleRounding("indexAddRows INT64 strided", ref.toFloatArray(), out.toFloatArray());
    }

    @Test
    public void indexAddRowsEmptyShapesLaunchNothing() {
        float[] gy = TrainParity.uniform(13, 3 * 2, -1f, 1f);
        Tensor zeroRows = keep(device(GpuIndexGradOps.indexAddRows(floats(gpu, gy, 3, 2), ints(gpu, new int[]{0, 1,
                2}, 3), 0, true, -1)));
        assertEquals(Shape.of(0, 2), zeroRows.shape());
        Tensor noIndices = keep(device(GpuIndexGradOps.indexAddRows(floats(gpu, new float[0], 0, 2),
                ints(gpu, new int[0], 0), 4, false, -1)));
        TrainParity.assertBitIdentical("no indices", new float[8], noIndices.toFloatArray());
        assertFalse(noIndices.requiresGrad());
    }

    // ------------------------------------------------------------------ segment backward

    @Test
    public void segmentBackwardMatchesTheReference() {
        float nan = Float.NaN;
        float[] src = {
            1f, -0f, 5f,
            3f, 0f, nan,
            3f, 0f, 5f,
            -1f, 0f, 4f,
            7f, -0f, 4f,
            3f, 9f, -2f};
        float[] gy = TrainParity.uniform(21, 4 * 3, -2f, 2f);
        int[] ptr = {0, 3, 3, 5, 6};
        int[] edges = {4, 0, 2, 1, 3, 5};
        for (GpuReduceOps.SegmentOp op : GpuReduceOps.SegmentOp.values()) {
            Tensor hs = floats(host, src, 6, 3);
            Tensor gs = floats(gpu, src, 6, 3);
            Tensor hOut = keep(CpuIndexing.segmentReduce(hs, ints(host, ptr, 5), ints(host, edges, 6), 4, op, -5f));
            Tensor gOut = floats(gpu, hOut.toFloatArray(), 4, 3);
            Tensor ref = keep(CpuIndexing.segmentReduceBackward(hs, ints(host, ptr, 5), ints(host, edges, 6), hOut,
                    floats(host, gy, 4, 3), op));
            Tensor out = keep(device(GpuIndexGradOps.segmentReduceBackward(gs, ints(gpu, ptr, 5), ints(gpu, edges,
                    6), gOut, floats(gpu, gy, 4, 3), op)));
            assertEquals("segment backward " + op + ": shape", ref.shape(), out.shape());
            if (op == GpuReduceOps.SegmentOp.MEAN) {
                TrainParity.assertDivSqrt("segment backward " + op, ref.toFloatArray(), out.toFloatArray());
            } else {
                TrainParity.assertBitIdentical("segment backward " + op, ref.toFloatArray(), out.toFloatArray());
            }
        }
    }

    @Test
    public void segmentBackwardTakesInt64CsrAndEmptyShapes() {
        float[] src = TrainParity.uniform(31, 5 * 2, -4f, 4f);
        long[] ptr = {0, 2, 2, 5};
        long[] edges = {3, (1L << 33) | 0, 1, 4, 2};
        int[] ptr32 = {0, 2, 2, 5};
        int[] edges32 = {3, 0, 1, 4, 2};
        float[] gy = TrainParity.uniform(32, 3 * 2, -1f, 1f);
        Tensor ref = keep(CpuIndexing.segmentReduceBackward(floats(host, src, 5, 2), ints(host, ptr32, 4),
                ints(host, edges32, 5), null, floats(host, gy, 3, 2), GpuReduceOps.SegmentOp.SUM));
        Tensor out = keep(device(GpuIndexGradOps.segmentReduceBackward(floats(gpu, src, 5, 2), int64(gpu, ptr, 4),
                int64(gpu, edges, 5), null, floats(gpu, gy, 3, 2), GpuReduceOps.SegmentOp.SUM)));
        TrainParity.assertBitIdentical("segment backward INT64 CSR", ref.toFloatArray(), out.toFloatArray());

        Tensor noSegments = keep(device(GpuIndexGradOps.segmentReduceBackward(floats(gpu, src, 5, 2),
                ints(gpu, new int[]{0}, 1), ints(gpu, new int[0], 0), null, floats(gpu, new float[0], 0, 2),
                GpuReduceOps.SegmentOp.MEAN)));
        TrainParity.assertBitIdentical("no segments", new float[10], noSegments.toFloatArray());
    }

    // ------------------------------------------------------------------ CPU-path contract

    @Test
    public void hostBackendDeclinesAndArgumentsFailAlike() {
        float[] gy = {1f, 2f, 3f, 4f};
        assertNull(GpuIndexGradOps.indexAddRows(floats(host, gy, 2, 2), ints(host, new int[]{0, 1}, 2), 2, false,
                -1));
        assertNull(GpuIndexGradOps.segmentReduceBackward(floats(host, gy, 2, 2), ints(host, new int[]{0, 1, 2}, 3),
                ints(host, new int[]{1, 0}, 2), null, floats(host, gy, 2, 2), GpuReduceOps.SegmentOp.SUM));
        for (ComputeBackend b : new ComputeBackend[]{host, gpu}) {
            Tensor g = floats(b, gy, 2, 2);
            assertThrows(IllegalArgumentException.class, () -> GpuIndexGradOps.indexAddRows(g, ints(b, new int[]{0},
                    1), 2, false, -1));
            assertThrows(IllegalArgumentException.class, () -> GpuIndexGradOps.indexAddRows(g, ints(b, new int[]{0,
                    1}, 2), 2, false, 2));
            assertThrows(IllegalArgumentException.class, () -> GpuIndexGradOps.segmentReduceBackward(g,
                    ints(b, new int[]{0, 1, 2}, 3), ints(b, new int[]{1, 0}, 2), null, floats(b, gy, 4),
                    GpuReduceOps.SegmentOp.SUM));
            assertThrows(NullPointerException.class, () -> GpuIndexGradOps.segmentReduceBackward(g,
                    ints(b, new int[]{0, 1, 2}, 3), ints(b, new int[]{1, 0}, 2), null, g, GpuReduceOps.SegmentOp.MAX));
        }
        assertThrows("the CPU reference rejects a repeated source row", IllegalArgumentException.class,
                () -> CpuIndexing.segmentReduceBackward(floats(host, gy, 2, 2), ints(host, new int[]{0, 1, 2}, 3),
                        ints(host, new int[]{1, 1}, 2), null, floats(host, gy, 2, 2), GpuReduceOps.SegmentOp.SUM));
    }

    // ------------------------------------------------------------------ helpers

    private Tensor device(Tensor out) {
        assertNotNull("the dispatcher computes on " + api, out);
        assertSame("result stays on the device backend", gpu, out.backend());
        return out;
    }

    private static int[] randomIndices(long seed, int n, int lo, int hi) {
        Random r = new Random(seed);
        int[] out = new int[n];
        for (int i = 0; i < n; i++) {
            out[i] = lo + r.nextInt(hi - lo);
        }
        return out;
    }

    private Tensor keep(Tensor t) {
        owned.add(t);
        return t;
    }

    private Tensor floats(ComputeBackend backend, float[] data, int... dims) {
        return keep(keep(Tensor.fromArray(backend, data)).reshape(Shape.of(dims)));
    }

    private Tensor ints(ComputeBackend backend, int[] data, int... dims) {
        return keep(keep(Tensor.fromArray(backend, data)).reshape(Shape.of(dims)));
    }

    private Tensor int64(ComputeBackend backend, long[] values, int... dims) {
        ByteBuffer buffer = ByteBuffer.allocate(values.length * Long.BYTES).order(ByteOrder.nativeOrder());
        for (long v : values) {
            buffer.putLong(v);
        }
        buffer.flip();
        return keep(Tensor.fromStorage(backend, Shape.of(dims), DType.INT64,
                TensorStorage.fromHost(backend, buffer, DType.INT64)));
    }
}

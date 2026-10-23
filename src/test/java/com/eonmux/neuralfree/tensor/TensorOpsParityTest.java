package com.eonmux.neuralfree.tensor;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.tensor.gpu.ElementwiseOp.Backward;
import com.eonmux.neuralfree.tensor.gpu.TrainParity;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.Random;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;

/**
 * {@link TensorOps} on each GPU API against {@link TensorOps} on the host reference backend, under the tolerance
 * classes of the {@linkplain com.eonmux.neuralfree.tensor.gpu numerical contract}: the facade's device compositions
 * (cat, stack and split by copies into slices, GLU and GeGLU from slice views, the in-place ops through views) and the
 * helpers it routes to the dispatchers. Every device result must stay on the device backend.
 */
@RunWith(Parameterized.class)
public class TensorOpsParityTest {

    @Parameterized.Parameters(name = "{0}")
    public static Collection<Object[]> data() {
        return TrainParity.apis();
    }

    private final TrainParity.Api api;
    private TrainParity.Session session;
    private ComputeBackend gpu;
    private ComputeBackend host;
    private final List<Tensor> owned = new ArrayList<>();

    public TensorOpsParityTest(TrainParity.Api api) {
        this.api = api;
    }

    @Before
    public void setUp() {
        ComputationGraph.current().clear();
        ComputationGraph.setGradEnabled(true);
        session = TrainParity.open(api);
        gpu = session.backend();
        host = TrainParity.hostBackend();
    }

    @After
    public void tearDown() {
        try {
            ComputationGraph.current().clear();
            ComputationGraph.setGradEnabled(true);
            for (int i = owned.size() - 1; i >= 0; i--) {
                owned.get(i).close();
            }
        } finally {
            if (session != null) {
                session.close();
            }
        }
    }

    // ------------------------------------------------------------------ cat, stack, split

    @Test
    public void catCopiesViewsIntoOutputSlices() {
        float[] a = TrainParity.uniform(1, 5 * 3, -4f, 4f);
        float[] b = TrainParity.uniform(2, 5 * 8, -4f, 4f);
        Tensor ha = keep(floats(host, a, 3, 5).transpose());
        Tensor ga = keep(floats(gpu, a, 3, 5).transpose());
        Tensor hb = keep(floats(host, b, 5, 8).slice(1, 2, 6));
        Tensor gb = keep(floats(gpu, b, 5, 8).slice(1, 2, 6));
        assertBit("cat dim 1", keep(TensorOps.cat(Arrays.asList(ha, hb, ha), 1)),
                keep(TensorOps.cat(Arrays.asList(ga, gb, ga), -1)));
        float[] c = Arrays.copyOf(b, 8 * 3);
        Tensor hc = floats(host, c, 8, 3);
        Tensor gc = floats(gpu, c, 8, 3);
        assertBit("cat dim 0", keep(TensorOps.cat(Arrays.asList(ha, hc), 0)),
                keep(TensorOps.cat(Arrays.asList(ga, gc), 0)));
    }

    @Test
    public void stackCopiesEachTensorIntoItsSlot() {
        float[] a = TrainParity.uniform(3, 4 * 6, -4f, 4f);
        float[] b = TrainParity.uniform(4, 4 * 6, -4f, 4f);
        List<Tensor> hosts = Arrays.asList(keep(floats(host, a, 6, 4).transpose()), floats(host, b, 4, 6));
        List<Tensor> gpus = Arrays.asList(keep(floats(gpu, a, 6, 4).transpose()), floats(gpu, b, 4, 6));
        for (int dim : new int[]{0, 1, 2, -1, -3}) {
            assertBit("stack dim " + dim, keep(TensorOps.stack(hosts, dim)), keep(TensorOps.stack(gpus, dim)));
        }
    }

    @Test
    public void splitReturnsIndependentContiguousCopies() {
        float[] x = TrainParity.uniform(5, 6 * 9, -4f, 4f);
        Tensor ht = keep(floats(host, x, 6, 9).transpose());
        Tensor gt = keep(floats(gpu, x, 6, 9).transpose());
        List<Tensor> hostParts = TensorOps.split(ht, 3, 0);
        owned.addAll(hostParts);
        List<Tensor> gpuParts = TensorOps.split(gt, 3, 0);
        owned.addAll(gpuParts);
        assertEquals(3, gpuParts.size());
        for (int i = 0; i < 3; i++) {
            assertTrue(gpuParts.get(i).isContiguous());
            assertBit("split part " + i, hostParts.get(i), gpuParts.get(i));
        }
        float[] before = gpuParts.get(2).toFloatArray();
        TensorOps.zeroInPlace(gt);
        assertArrayEquals(before, gpuParts.get(2).toFloatArray(), 0f);
    }

    // ------------------------------------------------------------------ gated units and in-place ops

    @Test
    public void gatedUnitsFromSliceViews() {
        float[] x = TrainParity.uniform(6, 3 * 10 * 4, -6f, 6f);
        Tensor h = keep(floats(host, x, 3, 10, 4).transpose(0, 2, 1));
        Tensor g = keep(floats(gpu, x, 3, 10, 4).transpose(0, 2, 1));
        assertGated("glu", h, 2, false, keep(TensorOps.glu(g, -1)));
        assertGated("geglu", h, 1, true, keep(TensorOps.geglu(g, -2)));
        assertGated("glu dim 0", floats(host, x, 4, 30), 0, false, keep(TensorOps.glu(floats(gpu, x, 4, 30), 0)));
    }

    @Test
    public void inPlaceOpsWriteOnlyTheViewedElements() {
        float[] base = TrainParity.uniform(7, 8 * 6, -5f, 5f);
        float[] src = TrainParity.uniform(8, 3 * 6, -5f, 5f);
        Tensor h = floats(host, base, 8, 6);
        Tensor g = floats(gpu, base, 8, 6);
        TensorOps.addInPlace(keep(h.slice(0, 2, 5)), floats(host, src, 3, 6), -0.75f);
        TensorOps.addInPlace(keep(g.slice(0, 2, 5)), floats(gpu, src, 3, 6), -0.75f);
        TensorOps.mulInPlace(keep(keep(h.transpose()).slice(0, 1, 4)), 1.5f);
        TensorOps.mulInPlace(keep(keep(g.transpose()).slice(0, 1, 4)), 1.5f);
        TensorOps.zeroInPlace(keep(h.slice(0, 6, 8)));
        TensorOps.zeroInPlace(keep(g.slice(0, 6, 8)));
        TensorOps.zeroInPlace(keep(keep(h.transpose()).slice(0, 5, 6)));
        TensorOps.zeroInPlace(keep(keep(g.transpose()).slice(0, 5, 6)));
        TrainParity.assertSingleRounding("in-place sequence", h.toFloatArray(), g.toFloatArray());
    }

    // ------------------------------------------------------------------ elementwise helpers

    @Test
    public void roundFloorAndBoolWhere() {
        float[] x = TrainParity.uniform(9, 7 * 11, -40f, 40f);
        x[3] = 0.5f;
        x[4] = -0.5f;
        x[5] = -2.5f;
        x[6] = 0.49999997f;
        x[7] = -0.0f;
        Tensor h = keep(floats(host, x, 7, 11).transpose());
        Tensor g = keep(floats(gpu, x, 7, 11).transpose());
        assertBit("round", keep(TensorOps.round(h)), keep(TensorOps.round(g)));
        assertBit("floor", keep(TensorOps.floor(h)), keep(TensorOps.floor(g)));
        float[] y = TrainParity.uniform(10, 11 * 7, -40f, 40f);
        Tensor hy = floats(host, y, 11, 7);
        Tensor gy = floats(gpu, y, 11, 7);
        Tensor hc = keep(TensorOps.gt(h, hy));
        Tensor gc = keep(TensorOps.gt(g, gy));
        assertEquals(DType.BOOL, gc.dtype());
        assertBit("where bool", keep(TensorOps.where(hc, h, hy)), keep(TensorOps.where(gc, g, gy)));
    }

    @Test
    public void activationBackwardMatchesTheReference() {
        float[] x = TrainParity.uniform(11, 9 * 13, -4f, 4f);
        float[] gyData = TrainParity.uniform(12, 9 * 13, -2f, 2f);
        Tensor hx = keep(floats(host, x, 13, 9).transpose());
        Tensor gx = keep(floats(gpu, x, 13, 9).transpose());
        Tensor hg = floats(host, gyData, 9, 13);
        Tensor gg = floats(gpu, gyData, 9, 13);
        float alpha = 0.8f;
        for (Backward op : Backward.values()) {
            float p0 = op == Backward.CLAMP ? -1f : op == Backward.LEAKY_RELU ? 0.01f : alpha;
            float p1 = op == Backward.CLAMP ? 1.5f : 0.0f;
            Tensor hostY = null;
            Tensor gpuY = null;
            if (op.readsY()) {
                hostY = keep(op == Backward.SIGMOID ? TensorOps.sigmoid(hx)
                        : op == Backward.TANH ? TensorOps.tanh(hx) : TensorOps.elu(hx, alpha));
                gpuY = floats(gpu, hostY.toFloatArray(), 9, 13);
            }
            Tensor ref = keep(TensorOps.activationBackward(op, op.readsX() ? hx : null, hostY, hg, p0, p1));
            Tensor out = keep(TensorOps.activationBackward(op, op.readsX() ? gx : null, gpuY, gg, p0, p1));
            requireDevice(out);
            boolean transcendental = op == Backward.GELU || op == Backward.SILU || op == Backward.SOFTPLUS
                    || op == Backward.MISH;
            if (transcendental) {
                TrainParity.assertTranscendental("backward " + op, ref.toFloatArray(), out.toFloatArray());
            } else {
                TrainParity.assertSingleRounding("backward " + op, ref.toFloatArray(), out.toFloatArray());
            }
        }
    }

    // ------------------------------------------------------------------ softmax family and reductions

    @Test
    public void maskedSoftmaxAndSoftmaxBackwards() {
        int rows = 6;
        int cols = 17;
        float[] x = TrainParity.uniform(13, rows * cols, -5f, 5f);
        Random r = new Random(14);
        for (int i = 0; i < x.length; i++) {
            if (i / cols == 2 || r.nextFloat() < 0.3f) {
                x[i] = Float.NEGATIVE_INFINITY;
            }
        }
        Tensor h = floats(host, x, rows, cols);
        Tensor g = floats(gpu, x, rows, cols);
        Tensor hm = keep(TensorOps.maskedSoftmax(h, -1));
        Tensor gm = keep(TensorOps.maskedSoftmax(g, 1));
        assertClose("maskedSoftmax", hm, gm);

        float[] logits = TrainParity.uniform(15, rows * cols, -5f, 5f);
        float[] grad = TrainParity.uniform(16, rows * cols, -1f, 1f);
        Tensor hy = keep(TensorOps.softmax(floats(host, logits, rows, cols), 1));
        Tensor hl = keep(TensorOps.logSoftmax(floats(host, logits, rows, cols), 1));
        Tensor gyOut = floats(gpu, hy.toFloatArray(), rows, cols);
        Tensor gl = floats(gpu, hl.toFloatArray(), rows, cols);
        Tensor hg = floats(host, grad, rows, cols);
        Tensor gg = floats(gpu, grad, rows, cols);
        assertClose("softmaxBackward", keep(TensorOps.softmaxBackward(hy, hg, -1)),
                keep(TensorOps.softmaxBackward(gyOut, gg, -1)));
        assertClose("logSoftmaxBackward", keep(TensorOps.logSoftmaxBackward(hl, hg, -1)),
                keep(TensorOps.logSoftmaxBackward(gl, gg, -1)));
    }

    @Test
    public void maxBackwardVarAndSumToShape() {
        float[] x = TrainParity.uniform(17, 4 * 3 * 5, -3f, 3f);
        x[7] = x[6];
        Tensor h = floats(host, x, 4, 3, 5);
        Tensor g = floats(gpu, x, 4, 3, 5);
        float[] grad = TrainParity.uniform(18, 4 * 3, -1f, 1f);
        Tensor hMax = keep(TensorOps.max(h, -1));
        Tensor gMax = keep(TensorOps.max(g, -1));
        assertBit("max", hMax, gMax);
        assertBit("maxBackward", keep(TensorOps.maxBackward(h, hMax, floats(host, grad, 4, 3), -1)),
                keep(TensorOps.maxBackward(g, gMax, floats(gpu, grad, 4, 3), -1)));

        for (int[] dims : new int[][]{{1}, {2}, {0, 2}, {}}) {
            float[] ref = keep(TensorOps.var(h, dims)).toFloatArray();
            float[] out = keep(TensorOps.var(g, dims)).toFloatArray();
            assertEquals(ref.length, out.length);
            for (int i = 0; i < ref.length; i++) {
                TrainParity.assertReduction("var " + Arrays.toString(dims) + " [" + i + "]", ref[i], out[i], x.length,
                        9.0);
            }
        }

        Tensor hs = keep(TensorOps.sumToShape(h, Shape.of(3, 1)));
        Tensor gs = keep(TensorOps.sumToShape(g, Shape.of(3, 1)));
        assertEquals(Shape.of(3, 1), gs.shape());
        float[] ref = hs.toFloatArray();
        float[] out = gs.toFloatArray();
        for (int i = 0; i < ref.length; i++) {
            TrainParity.assertReduction("sumToShape [" + i + "]", ref[i], out[i], 20, 20 * 3.0);
        }
        Tensor same = keep(TensorOps.sumToShape(g, g.shape()));
        assertFalse(same == g);
        assertBit("sumToShape identity", h, same);
    }

    // ------------------------------------------------------------------ index ops and products

    @Test
    public void scatterAddIndexSelectRowsAndSegmentReduce() {
        float[] input = integers(19, 5 * 4, 8);
        float[] src = integers(20, 6 * 4, 8);
        int[] index = new int[6 * 4];
        Random r = new Random(21);
        for (int i = 0; i < index.length; i++) {
            index[i] = r.nextInt(5);
        }
        assertBit("scatterAdd", keep(TensorOps.scatterAdd(floats(host, input, 5, 4), 0, ints(host, index, 6, 4),
                floats(host, src, 6, 4))), keep(TensorOps.scatterAdd(floats(gpu, input, 5, 4), 0,
                ints(gpu, index, 6, 4), floats(gpu, src, 6, 4))));

        float[] table = integers(22, 7 * 3, 50);
        int[] rows = {6, -2, 9, 0, 3, 3};
        assertBit("indexSelectRows clamp", keep(TensorOps.indexSelectRows(floats(host, table, 7, 3),
                ints(host, rows, 2, 3), true)), keep(TensorOps.indexSelectRows(floats(gpu, table, 7, 3),
                ints(gpu, rows, 2, 3), true)));

        float[] feats = integers(23, 6 * 3, 20);
        feats[4] = Float.NEGATIVE_INFINITY;
        int[] ptr = {0, 3, 3, 4, 8};
        int[] edges = {5, 0, 2, 1, 1, 3, 4, 0};
        for (GpuReduceOps.SegmentOp op : GpuReduceOps.SegmentOp.values()) {
            Tensor ref = keep(TensorOps.segmentReduce(floats(host, feats, 6, 3), ints(host, ptr, 5),
                    ints(host, edges, 8), 4, op, -7f));
            Tensor out = keep(TensorOps.segmentReduce(floats(gpu, feats, 6, 3), ints(gpu, ptr, 5),
                    ints(gpu, edges, 8), 4, op, -7f));
            requireDevice(out);
            TrainParity.assertDivSqrt("segment " + op, ref.toFloatArray(), out.toFloatArray());
        }
    }

    @Test
    public void int64IndicesNarrowOnTheDevice() {
        float[] input = integers(26, 4 * 6, 9);
        long[] index = {5, 0, 3, (1L << 32) | 2, 1, 1, 4, 0, 2, 5, 3, 1};
        Tensor hIdx = keep(int64(host, index, 3, 4).transpose());
        Tensor gIdx = keep(int64(gpu, index, 3, 4).transpose());
        assertBit("gather int64", keep(TensorOps.gather(floats(host, input, 4, 6), 1, hIdx)),
                keep(TensorOps.gather(floats(gpu, input, 4, 6), 1, gIdx)));
        float[] src = integers(27, 4 * 3, 9);
        assertBit("scatterAdd int64", keep(TensorOps.scatterAdd(floats(host, input, 4, 6), -1, hIdx,
                floats(host, src, 4, 3))), keep(TensorOps.scatterAdd(floats(gpu, input, 4, 6), -1, gIdx,
                floats(gpu, src, 4, 3))));
        assertBit("scatter int64", keep(TensorOps.scatter(floats(host, input, 4, 6), 1, hIdx,
                floats(host, src, 4, 3))), keep(TensorOps.scatter(floats(gpu, input, 4, 6), 1, gIdx,
                floats(gpu, src, 4, 3))));

        float[] feats = integers(28, 4 * 2, 20);
        long[] ptr = {0, 2, 2, 5};
        long[] edges = {3, 0, 1, 1, 2};
        Tensor ref = keep(TensorOps.segmentReduce(floats(host, feats, 4, 2), int64(host, ptr, 4),
                int64(host, edges, 5), 3, GpuReduceOps.SegmentOp.SUM, 0f));
        Tensor out = keep(TensorOps.segmentReduce(floats(gpu, feats, 4, 2), int64(gpu, ptr, 4),
                int64(gpu, edges, 5), 3, GpuReduceOps.SegmentOp.SUM, 0f));
        assertBit("segmentReduce int64 CSR", ref, out);
    }

    @Test
    public void bmmAndGradFlagsOnDevice() {
        float[] a = integers(24, 3 * 4 * 5, 6);
        float[] b = integers(25, 3 * 5 * 2, 6);
        Tensor ha = keep(floats(host, a, 3, 4, 5).withRequiresGrad(true));
        Tensor ga = keep(floats(gpu, a, 3, 4, 5).withRequiresGrad(true));
        Tensor hb = floats(host, b, 3, 5, 2);
        Tensor gb = floats(gpu, b, 3, 5, 2);
        ComputationGraph graph = ComputationGraph.current();
        int before = graph.size();
        try (ComputationGraph.StepScope step = graph.step()) {
            Tensor gBmm = TensorOps.bmm(ga, gb);
            assertTrue("a recorded result requires grad", gBmm.requiresGrad());
            assertNotNull("a recorded result carries its node's function", gBmm.gradFn());
            assertBit("bmm", TensorOps.bmm(ha, hb), gBmm);
            Tensor gMatmul = TensorOps.matmul(ga, gb);
            assertTrue(gMatmul.requiresGrad());
            assertBit("matmul batched", TensorOps.matmul(ha, hb), gMatmul);
            assertTrue(TensorOps.add(ga, ga).requiresGrad());
            assertTrue(TensorOps.mul(ga, ga).requiresGrad());
            assertTrue(TensorOps.sum(ga, 2).requiresGrad());
            assertEquals("one node per recorded call", before + 7, graph.size());
        }
        assertEquals(before, graph.size());
        try (ComputationGraph.NoGradScope off = ComputationGraph.noGrad()) {
            assertFalse(keep(TensorOps.bmm(ga, gb)).requiresGrad());
            assertFalse(keep(TensorOps.matmul(ga, gb)).requiresGrad());
            assertFalse(keep(TensorOps.add(ga, ga)).requiresGrad());
            assertFalse(keep(TensorOps.sum(ga, 2)).requiresGrad());
        }
        assertEquals("nothing records under noGrad", before, graph.size());
    }

    /**
     * Recorded index, segment and dropout ops back-propagated on the device against the host: the gradients reach
     * the leaves through IndexSelectRowsGrad (padding row skipped), SegmentReduceGrad and MaskScaleGrad, computed by
     * the backward dispatchers. Integer-valued inputs keep every sum exact, so the results are bit-identical.
     */
    @Test
    public void recordedIndexGradientsMatchTheHost() {
        float[] table = integers(29, 5 * 3, 7);
        float[] upstream = integers(30, 6 * 3, 5);
        float[] mask = {0f, 2f, 2f, 0f, 2f, 2f, 2f, 0f, 2f, 2f, 2f, 2f, 0f, 2f, 2f, 0f, 2f, 2f};
        float[] hostGrad = indexGradient(host, table, upstream, mask);
        float[] gpuGrad = indexGradient(gpu, table, upstream, mask);
        TrainParity.assertBitIdentical("index/segment/mask gradient", hostGrad, gpuGrad);
    }

    /**
     * The gradient of {@code sum(segmentReduce(maskScale(indexSelectRows(w, idx, pad 1), mask, 0.5), SUM) * up)}
     * with respect to the table {@code w} on {@code backend}.
     */
    private float[] indexGradient(ComputeBackend backend, float[] table, float[] upstream, float[] mask) {
        ComputationGraph graph = ComputationGraph.current();
        int before = graph.size();
        Tensor w = keep(floats(backend, table, 5, 3).withRequiresGrad(true));
        Tensor idx = ints(backend, new int[]{4, 1, 1, 0, 3, 2}, 6);
        Tensor ptr = ints(backend, new int[]{0, 2, 2, 6}, 4);
        Tensor edges = ints(backend, new int[]{5, 0, 3, 1, 4, 2}, 6);
        Tensor m = floats(backend, mask, 6, 3);
        Tensor up = floats(backend, Arrays.copyOf(upstream, 9), 3, 3);
        try (ComputationGraph.StepScope step = graph.step()) {
            Tensor rows = TensorOps.indexSelectRows(w, idx, false, 1);
            Tensor dropped = TensorOps.maskScale(rows, m, 0.5f);
            Tensor seg = TensorOps.segmentReduce(dropped, ptr, edges, 3, GpuReduceOps.SegmentOp.SUM, 0f);
            try (Tensor loss = TensorOps.sum(TensorOps.mul(seg, up))) {
                loss.backward();
            }
        }
        assertEquals("the step leaves no node", before, graph.size());
        assertNotNull("the gradient reaches the table", w.grad());
        float[] grad = w.grad().toFloatArray();
        graph.zeroGrad();
        for (int c = 0; c < 3; c++) {
            assertEquals("the padding row gets no gradient", 0f, grad[3 + c], 0f);
        }
        return grad;
    }

    @Test
    public void int64ViewsReadBackThroughTheDevice() {
        long[] values = {1L, -2L, 3L << 33, 4L, 5L, -(6L << 40)};
        Tensor h = keep(int64(host, values, 2, 3).transpose());
        Tensor g = keep(int64(gpu, values, 2, 3).transpose());
        assertArrayEquals(bytes(h), bytes(g));
        assertArrayEquals(h.toIntArray(), g.toIntArray());
    }

    // ------------------------------------------------------------------ assertions

    private void assertBit(String what, Tensor ref, Tensor out) {
        requireDevice(out);
        assertEquals(what + ": shape", ref.shape(), out.shape());
        assertEquals(what + ": dtype", ref.dtype(), out.dtype());
        assertEquals(what + ": requiresGrad", ref.requiresGrad(), out.requiresGrad());
        TrainParity.assertBitIdentical(what, bytes(ref), bytes(out));
    }

    private void assertClose(String what, Tensor ref, Tensor out) {
        requireDevice(out);
        assertEquals(what + ": shape", ref.shape(), out.shape());
        TrainParity.assertTranscendental(what, ref.toFloatArray(), out.toFloatArray());
    }

    /**
     * GLU ({@code gelu == false}) or GeGLU of {@code hostInput} along {@code dim} against the device result: one
     * transcendental gate times the first half, so each element may differ by the transcendental bound scaled by
     * both factors, plus the product's rounding.
     */
    private void assertGated(String what, Tensor hostInput, int dim, boolean gelu, Tensor out) {
        requireDevice(out);
        int half = hostInput.shape().dim(dim) / 2;
        Tensor ref = keep(gelu ? TensorOps.geglu(hostInput, dim) : TensorOps.glu(hostInput, dim));
        assertEquals(what + ": shape", ref.shape(), out.shape());
        float[] first = keep(hostInput.slice(dim, 0, half)).toFloatArray();
        Tensor second = keep(hostInput.slice(dim, half, 2L * half));
        float[] gate = keep(gelu ? TensorOps.gelu(second) : TensorOps.sigmoid(second)).toFloatArray();
        float[] r = ref.toFloatArray();
        float[] o = out.toFloatArray();
        for (int i = 0; i < r.length; i++) {
            double bound = 1e-5 * Math.max(1.0, Math.abs(first[i])) * Math.max(1.0, Math.abs(gate[i]))
                    + Math.ulp(r[i]);
            assertTrue(what + ": element " + i + ": ref " + r[i] + ", gpu " + o[i],
                    Math.abs((double) o[i] - r[i]) <= bound);
        }
    }

    private void requireDevice(Tensor out) {
        assertSame("result stays on the device backend", gpu, out.backend());
    }

    // ------------------------------------------------------------------ inputs

    private static float[] integers(long seed, int n, int bound) {
        Random r = new Random(seed);
        float[] out = new float[n];
        for (int i = 0; i < n; i++) {
            out[i] = r.nextInt(2 * bound + 1) - bound;
        }
        return out;
    }

    private static byte[] bytes(Tensor t) {
        ByteBuffer bb = t.toByteBuffer();
        byte[] out = new byte[bb.remaining()];
        bb.get(out);
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

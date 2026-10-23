package com.eonmux.neuralfree.tensor;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.tensor.gpu.ElementwiseOp.Backward;
import com.eonmux.neuralfree.tensor.gpu.TrainGpu;
import com.eonmux.neuralfree.tensor.gpu.TrainParity;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotSame;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The CPU reference paths of {@link TensorOps}, on a host backend where the train library is unavailable: variance
 * over any dims, stack along inner dims, BOOL {@code where} conditions, in-place writes through views, the kernels'
 * ROUND formula, and the helpers used by autograd, losses and module compositions.
 */
public class TensorOpsCpuReferenceTest {

    private ComputeBackend host;
    private final List<Tensor> owned = new ArrayList<>();

    @Before
    public void setUp() {
        ComputationGraph.current().clear();
        ComputationGraph.setGradEnabled(true);
        host = TrainParity.hostBackend();
        assertNull("the host backend has no train library", TrainGpu.forBackend(host));
    }

    @After
    public void tearDown() {
        try {
            ComputationGraph.current().clear();
        } finally {
            ComputationGraph.setGradEnabled(true);
            for (int i = owned.size() - 1; i >= 0; i--) {
                owned.get(i).close();
            }
        }
    }

    // ------------------------------------------------------------------ reductions

    @Test
    public void varReducesAnyChoiceOfDims() {
        float[] x = TrainParity.uniform(1, 2 * 3 * 4, -5f, 5f);
        Tensor t = floats(x, 2, 3, 4);
        int[][] cases = {{1}, {2}, {0, 2}, {-1, 0}, {}};
        for (int[] dims : cases) {
            Tensor v = keep(TensorOps.var(t, dims));
            boolean[] reduced = new boolean[3];
            for (int d : dims) {
                reduced[d < 0 ? d + 3 : d] = true;
            }
            if (dims.length == 0) {
                Arrays.fill(reduced, true);
            }
            assertArrayEquals("var " + Arrays.toString(dims), variance(x, new int[]{2, 3, 4}, reduced),
                    v.toFloatArray(), 1e-4f);
        }
        float[] square = TrainParity.uniform(2, 16, -5f, 5f);
        assertArrayEquals("var [4,4] dim 1", variance(square, new int[]{4, 4}, new boolean[]{false, true}),
                keep(TensorOps.var(floats(square, 4, 4), 1)).toFloatArray(), 1e-4f);
        assertEquals(Shape.vector(4), keep(TensorOps.var(floats(square, 4, 4), 1)).shape());
    }

    @Test
    public void sumRejectsRepeatedDims() {
        Tensor t = floats(new float[6], 2, 3);
        assertThrows(IllegalArgumentException.class, () -> TensorOps.mean(t, 1, -1));
        assertThrows(IndexOutOfBoundsException.class, () -> TensorOps.sum(t, 2));
    }

    @Test
    public void argReductionsFollowTheStrictComparisonRule() {
        float nan = Float.NaN;
        Tensor t = floats(new float[]{nan, 1f, 1f, 0f, nan, nan, nan, nan}, 2, 4);
        assertArrayEquals(new int[]{1, 0}, keep(TensorOps.argmax(t, 1)).toIntArray());
        assertArrayEquals(new int[]{3, 0}, keep(TensorOps.argmin(t, -1)).toIntArray());
        assertEquals(DType.INT32, keep(TensorOps.argmax(t, 0)).dtype());
    }

    @Test
    public void sumToShapeSumsBroadcastDimsIntoANewTensor() {
        float[] x = TrainParity.uniform(3, 4 * 3 * 5, -1f, 1f);
        Tensor t = floats(x, 4, 3, 5);
        Tensor rows = keep(TensorOps.sumToShape(t, Shape.of(3, 1)));
        assertEquals(Shape.of(3, 1), rows.shape());
        float[] expected = new float[3];
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 5; k++) {
                    expected[j] += x[(i * 3 + j) * 5 + k];
                }
            }
        }
        assertArrayEquals(expected, rows.toFloatArray(), 1e-5f);
        Tensor same = keep(TensorOps.sumToShape(t, t.shape()));
        assertNotSame(t, same);
        assertArrayEquals(x, same.toFloatArray(), 0f);
        Tensor rearranged = keep(TensorOps.sumToShape(floats(new float[]{1f, 2f, 3f, 4f}, 1, 4), Shape.of(4, 1)));
        assertEquals(Shape.of(4, 1), rearranged.shape());
        assertThrows(IllegalArgumentException.class, () -> TensorOps.sumToShape(t, Shape.of(7)));
    }

    // ------------------------------------------------------------------ softmax family

    @Test
    public void maskedSoftmaxGivesExactZerosForMaskedEntries() {
        float ninf = Float.NEGATIVE_INFINITY;
        Tensor t = floats(new float[]{0f, ninf, 1f, ninf, ninf, ninf}, 2, 3);
        float[] p = keep(TensorOps.maskedSoftmax(t, -1)).toFloatArray();
        float e = (float) Math.exp(0f - 1f);
        float sum = e + (float) Math.exp(1f - 1f);
        assertArrayEquals(new float[]{e / sum, 0f, 1f / sum, 0f, 0f, 0f}, p, 0f);
        assertEquals(0, Float.floatToRawIntBits(p[1]));
    }

    @Test
    public void softmaxBackwardsAcceptNegativeDims() {
        float[] x = TrainParity.uniform(4, 3 * 5, -3f, 3f);
        float[] g = TrainParity.uniform(5, 3 * 5, -1f, 1f);
        Tensor y = keep(TensorOps.softmax(floats(x, 3, 5), 1));
        Tensor logY = keep(TensorOps.logSoftmax(floats(x, 3, 5), 1));
        Tensor gy = floats(g, 3, 5);
        float[] neg = keep(TensorOps.softmaxBackward(y, gy, -1)).toFloatArray();
        assertArrayEquals(keep(TensorOps.softmaxBackward(y, gy, 1)).toFloatArray(), neg, 0f);
        float[] yv = y.toFloatArray();
        for (int r = 0; r < 3; r++) {
            float dot = 0f;
            for (int c = 0; c < 5; c++) {
                dot += yv[r * 5 + c] * g[r * 5 + c];
            }
            for (int c = 0; c < 5; c++) {
                assertEquals(yv[r * 5 + c] * (g[r * 5 + c] - dot), neg[r * 5 + c], 0f);
            }
        }
        float[] logGrad = keep(TensorOps.logSoftmaxBackward(logY, gy, -1)).toFloatArray();
        float[] lv = logY.toFloatArray();
        for (int r = 0; r < 3; r++) {
            float s = 0f;
            for (int c = 0; c < 5; c++) {
                s += g[r * 5 + c];
            }
            for (int c = 0; c < 5; c++) {
                assertEquals(g[r * 5 + c] - (float) Math.exp(lv[r * 5 + c]) * s, logGrad[r * 5 + c], 0f);
            }
        }
    }

    @Test
    public void maxBackwardRoutesToTheFirstMaximum() {
        Tensor x = floats(new float[]{1f, 3f, 3f, 2f, 5f, 4f}, 2, 3);
        Tensor max = keep(TensorOps.max(x, -1));
        Tensor gy = floats(new float[]{10f, 20f}, 2);
        assertArrayEquals(new float[]{0f, 10f, 0f, 0f, 20f, 0f},
                keep(TensorOps.maxBackward(x, max, gy, -1)).toFloatArray(), 0f);
        assertThrows(IllegalArgumentException.class,
                () -> TensorOps.maxBackward(x, max, floats(new float[3], 3), 1));
    }

    // ------------------------------------------------------------------ elementwise

    @Test
    public void roundUsesTheKernelFloorFormula() {
        float[] x = {-0.0f, 0.0f, 0.5f, -0.5f, 1.5f, -1.5f, 2.5f, -2.5f, 0.49999997f, 8388609.0f, -8388609.5f,
                Float.NaN, Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY, 3e9f, -7.25f};
        float[] r = keep(TensorOps.round(floats(x, x.length))).toFloatArray();
        assertEquals(Float.floatToRawIntBits(-0.0f), Float.floatToRawIntBits(r[0]));
        assertTrue(Float.isNaN(r[11]));
        assertEquals(Float.POSITIVE_INFINITY, r[12], 0f);
        assertEquals(Float.NEGATIVE_INFINITY, r[13], 0f);
        for (int i = 0; i < x.length; i++) {
            if (!Float.isNaN(x[i]) && !Float.isInfinite(x[i]) && Math.abs(x[i]) < 2e9f) {
                assertEquals("round " + x[i], Math.round(x[i]), (int) r[i]);
            }
        }
        float[] f = keep(TensorOps.floor(floats(x, x.length))).toFloatArray();
        for (int i = 0; i < x.length; i++) {
            assertEquals(Float.floatToRawIntBits((float) Math.floor(x[i])), Float.floatToRawIntBits(f[i]));
        }
    }

    @Test
    public void whereAcceptsFloatAndBoolConditions() {
        float[] a = {1f, 2f, 3f, 4f};
        float[] b = {-1f, -2f, -3f, -4f};
        Tensor fc = floats(new float[]{Float.NaN, -0.0f, 0.0f, 2f}, 4);
        assertArrayEquals(new float[]{1f, -2f, -3f, 4f},
                keep(TensorOps.where(fc, floats(a, 4), floats(b, 4))).toFloatArray(), 0f);
        Tensor bc = keep(Tensor.fromStorage(host, Shape.vector(4), DType.BOOL,
                TensorStorage.fromHost(host, ByteBuffer.wrap(new byte[]{0, 7, 1, 0}), DType.BOOL)));
        assertArrayEquals(new float[]{-1f, 2f, 3f, -4f},
                keep(TensorOps.where(bc, floats(a, 4), floats(b, 4))).toFloatArray(), 0f);
        assertThrows(IllegalArgumentException.class, () -> TensorOps.where(bc, floats(a, 2, 2), floats(b, 2, 2)));
    }

    @Test
    public void comparesReturnBoolBytes() {
        Tensor a = floats(new float[]{1f, Float.NaN, -0.0f, 3f}, 4);
        Tensor b = floats(new float[]{1f, Float.NaN, 0.0f, 2f}, 4);
        Tensor eq = keep(TensorOps.eq(a, b));
        assertEquals(DType.BOOL, eq.dtype());
        assertArrayEquals(new byte[]{1, 0, 1, 0}, bytes(eq));
        assertArrayEquals(new byte[]{0, 0, 0, 1}, bytes(keep(TensorOps.gt(a, b))));
        assertArrayEquals(new byte[]{0, 0, 0, 0}, bytes(keep(TensorOps.lt(a, b))));
    }

    @Test
    public void gatedUnitsMultiplyTheFirstHalfByTheGatedSecond() {
        float[] x = TrainParity.uniform(6, 2 * 4 * 3, -3f, 3f);
        Tensor t = floats(x, 2, 4, 3);
        float[] glu = keep(TensorOps.glu(t, 1)).toFloatArray();
        float[] geglu = keep(TensorOps.geglu(t, -2)).toFloatArray();
        float s = (float) Math.sqrt(2.0 / Math.PI);
        for (int o = 0; o < 2; o++) {
            for (int d = 0; d < 2; d++) {
                for (int i = 0; i < 3; i++) {
                    float a = x[(o * 4 + d) * 3 + i];
                    float g = x[(o * 4 + d + 2) * 3 + i];
                    int out = (o * 2 + d) * 3 + i;
                    assertEquals(a * (float) (1.0 / (1.0 + Math.exp(-g))), glu[out], 0f);
                    float gelu = 0.5f * g * (1.0f + (float) Math.tanh(s * (g + 0.044715f * g * g * g)));
                    assertEquals(a * gelu, geglu[out], 0f);
                }
            }
        }
        assertThrows(IllegalArgumentException.class, () -> TensorOps.glu(floats(new float[6], 2, 3), 1));
    }

    @Test
    public void activationBackwardFollowsTheGradFunctionsFormulas() {
        float[] x = {-2f, -0.0f, 0.0f, 0.5f, 3f, Float.NaN};
        float[] gy = {1f, 2f, 3f, 4f, 5f, 6f};
        Tensor xt = floats(x, 6);
        Tensor gt = floats(gy, 6);
        float[] relu = keep(TensorOps.activationBackward(Backward.RELU, xt, null, gt, 0f, 0f)).toFloatArray();
        assertArrayEquals(new float[]{0f, 0f, 0f, 4f, 5f, 0f}, relu, 0f);
        float[] abs = keep(TensorOps.activationBackward(Backward.ABS, xt, null, gt, 0f, 0f)).toFloatArray();
        assertArrayEquals("abs'(+-0) = abs'(NaN) = +0", new float[]{-1f, 0f, 0f, 4f, 5f, 0f}, abs, 0f);
        float[] clamp = keep(TensorOps.activationBackward(Backward.CLAMP, xt, null, gt, -1f, 1f)).toFloatArray();
        assertArrayEquals(new float[]{0f, 2f, 3f, 4f, 0f, 6f}, clamp, 0f);
        float[] y = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};
        float[] sig = keep(TensorOps.activationBackward(Backward.SIGMOID, null, floats(y, 6), gt, 0f, 0f))
                .toFloatArray();
        for (int i = 0; i < 6; i++) {
            assertEquals(gy[i] * y[i] * (1.0f - y[i]), sig[i], 0f);
        }
        assertThrows(NullPointerException.class,
                () -> TensorOps.activationBackward(Backward.ELU, xt, null, gt, 1f, 0f));
        assertThrows(IllegalArgumentException.class,
                () -> TensorOps.activationBackward(Backward.RELU, floats(new float[3], 3), null, gt, 0f, 0f));
    }

    // ------------------------------------------------------------------ in place

    @Test
    public void inPlaceOpsWriteThroughOffsetAndStridedViews() {
        float[] base = TrainParity.uniform(7, 6 * 4, -5f, 5f);
        float[] src = TrainParity.uniform(8, 3 * 4, -5f, 5f);
        Tensor t = floats(base, 6, 4);
        TensorOps.addInPlace(keep(t.slice(0, 2, 5)), floats(src, 3, 4), 0.5f);
        TensorOps.mulInPlace(keep(keep(t.transpose()).slice(0, 1, 3)), -2f);
        TensorOps.zeroInPlace(keep(t.slice(0, 5, 6)));
        float[] expected = base.clone();
        for (int i = 2; i < 5; i++) {
            for (int j = 0; j < 4; j++) {
                expected[i * 4 + j] += 0.5f * src[(i - 2) * 4 + j];
            }
        }
        for (int i = 0; i < 6; i++) {
            for (int j = 1; j < 3; j++) {
                expected[i * 4 + j] *= -2f;
            }
        }
        Arrays.fill(expected, 20, 24, 0f);
        assertArrayEquals(expected, t.toFloatArray(), 0f);
        assertThrows(IllegalArgumentException.class,
                () -> TensorOps.addInPlace(t, floats(new float[4], 4), 1f));
        assertThrows(UnsupportedOperationException.class,
                () -> TensorOps.zeroInPlace(keep(Tensor.fromArray(host, new int[]{1, 2}))));
    }

    // ------------------------------------------------------------------ cat, split, stack

    @Test
    public void stackPlacesEachTensorAtItsIndexOfTheNewAxis() {
        float[] a = {1f, 2f, 3f, 4f, 5f, 6f};
        float[] b = {7f, 8f, 9f, 10f, 11f, 12f};
        List<Tensor> pair = Arrays.asList(floats(a, 2, 3), floats(b, 2, 3));
        assertArrayEquals(new float[]{1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f},
                keep(TensorOps.stack(pair, 0)).toFloatArray(), 0f);
        Tensor mid = keep(TensorOps.stack(pair, 1));
        assertEquals(Shape.of(2, 2, 3), mid.shape());
        assertArrayEquals(new float[]{1f, 2f, 3f, 7f, 8f, 9f, 4f, 5f, 6f, 10f, 11f, 12f}, mid.toFloatArray(), 0f);
        Tensor last = keep(TensorOps.stack(pair, -1));
        assertEquals(Shape.of(2, 3, 2), last.shape());
        assertArrayEquals(new float[]{1f, 7f, 2f, 8f, 3f, 9f, 4f, 10f, 5f, 11f, 6f, 12f}, last.toFloatArray(), 0f);
    }

    @Test
    public void splitCopiesAndCatReassembles() {
        float[] x = TrainParity.uniform(9, 2 * 6, -1f, 1f);
        Tensor t = floats(x, 2, 6);
        List<Tensor> parts = TensorOps.split(t, 3, -1);
        owned.addAll(parts);
        assertEquals(3, parts.size());
        for (Tensor p : parts) {
            assertEquals(Shape.of(2, 2), p.shape());
            assertTrue(p.isContiguous());
        }
        Tensor joined = keep(TensorOps.cat(parts, 1));
        assertArrayEquals(x, joined.toFloatArray(), 0f);
        TensorOps.zeroInPlace(t);
        assertEquals(x[2], parts.get(1).toFloatArray()[0], 0f);
        assertThrows(IllegalArgumentException.class, () -> TensorOps.split(t, 4, 1));
        assertThrows(IllegalArgumentException.class, () -> TensorOps.split(t, 0, 1));
        assertThrows(IllegalArgumentException.class,
                () -> TensorOps.cat(Arrays.asList(t, floats(new float[3], 3, 1)), 1));
    }

    // ------------------------------------------------------------------ index ops

    @Test
    public void gatherScatterAndScatterAdd() {
        Tensor input = floats(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, 2, 3);
        Tensor index = ints(new int[]{0, 2, 2, 0}, 2, 2);
        assertArrayEquals(new float[]{1f, 3f, 6f, 4f}, keep(TensorOps.gather(input, 1, index)).toFloatArray(), 0f);
        Tensor longIndex = keep(Tensor.fromStorage(host, Shape.of(2, 2), DType.INT64,
                TensorStorage.fromHost(host, longs(0, 2, 2, 0), DType.INT64)));
        assertArrayEquals(new float[]{1f, 3f, 6f, 4f}, keep(TensorOps.gather(input, -1, longIndex)).toFloatArray(),
                0f);
        Tensor src = floats(new float[]{10f, 20f, 30f, 40f}, 2, 2);
        Tensor dup = ints(new int[]{1, 1, 0, 0}, 2, 2);
        assertArrayEquals(new float[]{1f, 20f, 3f, 40f, 5f, 6f},
                keep(TensorOps.scatter(input, 1, dup, src)).toFloatArray(), 0f);
        assertArrayEquals(new float[]{1f, 32f, 3f, 74f, 5f, 6f},
                keep(TensorOps.scatterAdd(input, 1, dup, src)).toFloatArray(), 0f);
        Tensor bad = ints(new int[]{0, 3, 1, 0}, 2, 2);
        assertThrows(IndexOutOfBoundsException.class, () -> TensorOps.gather(input, 1, bad));
        assertThrows(IndexOutOfBoundsException.class, () -> TensorOps.scatter(input, 1, bad, src));
    }

    @Test
    public void indexSelectRowsClampsOrRejects() {
        Tensor table = floats(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, 3, 2);
        Tensor idx = ints(new int[]{2, 0, 2, 1}, 2, 2);
        Tensor rows = keep(TensorOps.indexSelectRows(table, idx, false));
        assertEquals(Shape.of(2, 2, 2), rows.shape());
        assertArrayEquals(new float[]{5f, 6f, 1f, 2f, 5f, 6f, 3f, 4f}, rows.toFloatArray(), 0f);
        Tensor wide = keep(Tensor.fromStorage(host, Shape.vector(3), DType.INT64,
                TensorStorage.fromHost(host, longs(-4, 9, 1L << 32 | 1), DType.INT64)));
        assertArrayEquals(new float[]{1f, 2f, 5f, 6f, 3f, 4f},
                keep(TensorOps.indexSelectRows(table, wide, true)).toFloatArray(), 0f);
        IndexOutOfBoundsException e = assertThrows(IndexOutOfBoundsException.class,
                () -> TensorOps.indexSelectRows(table, ints(new int[]{0, 3}, 2), false));
        assertEquals("Embedding index out of range: 3", e.getMessage());
        assertThrows(IllegalArgumentException.class, () -> TensorOps.indexSelectRows(floats(new float[3], 3), idx,
                true));
        Tensor empty = keep(TensorOps.indexSelectRows(floats(new float[0], 3, 0), ints(new int[]{0, 7}, 2), false));
        assertEquals(Shape.of(2, 0), empty.shape());
    }

    @Test
    public void segmentReduceAggregatesCsrSegments() {
        int[] ptr = {0, 2, 2, 5};
        int[] edges = {1, 0, 2, 2, 1};
        float ninf = Float.NEGATIVE_INFINITY;
        float[] src = {1f, -1f, 2f, ninf, 4f, 0.5f};
        Tensor s = floats(src, 3, 2);
        Tensor p = ints(ptr, 4);
        Tensor e = ints(edges, 5);
        assertArrayEquals(new float[]{3f, ninf, -9f, -9f, 10f, ninf},
                keep(TensorOps.segmentReduce(s, p, e, 3, GpuReduceOps.SegmentOp.SUM, -9f)).toFloatArray(), 0f);
        assertArrayEquals(new float[]{1.5f, ninf, -9f, -9f, 10f / 3f, ninf},
                keep(TensorOps.segmentReduce(s, p, e, 3, GpuReduceOps.SegmentOp.MEAN, -9f)).toFloatArray(), 0f);
        assertArrayEquals(new float[]{2f, -1f, -9f, -9f, 4f, 0.5f},
                keep(TensorOps.segmentReduce(s, p, e, 3, GpuReduceOps.SegmentOp.MAX, -9f)).toFloatArray(), 0f);
        assertThrows(IllegalArgumentException.class, () -> TensorOps.segmentReduce(s, ints(new int[]{0, 3, 2, 5}, 4),
                e, 3, GpuReduceOps.SegmentOp.SUM, 0f));
        assertThrows(IllegalArgumentException.class, () -> TensorOps.segmentReduce(s, p, ints(new int[]{1, 0, 2, 2,
                3}, 5), 3, GpuReduceOps.SegmentOp.SUM, 0f));
    }

    // ------------------------------------------------------------------ products and flags

    @Test
    public void bmmAndMatmulRecordUnderGrad() {
        float[] a = TrainParity.uniform(10, 2 * 2 * 3, -1f, 1f);
        float[] b = TrainParity.uniform(11, 2 * 3 * 2, -1f, 1f);
        Tensor ta = keep(floats(a, 2, 2, 3).withRequiresGrad(true));
        Tensor tb = floats(b, 2, 3, 2);
        ComputationGraph graph = ComputationGraph.current();
        int before = graph.size();
        try (ComputationGraph.StepScope step = graph.step()) {
            Tensor bmm = TensorOps.bmm(ta, tb);
            assertTrue(bmm.requiresGrad());
            assertTrue(TensorOps.matmul(ta, tb).requiresGrad());
            float[] out = bmm.toFloatArray();
            for (int n = 0; n < 2; n++) {
                for (int i = 0; i < 2; i++) {
                    for (int j = 0; j < 2; j++) {
                        float sum = 0f;
                        for (int k = 0; k < 3; k++) {
                            sum += a[(n * 2 + i) * 3 + k] * b[(n * 3 + k) * 2 + j];
                        }
                        assertEquals(sum, out[(n * 2 + i) * 2 + j], 0f);
                    }
                }
            }
            assertTrue(TensorOps.add(ta, keep(tb.transpose(0, 2, 1))).requiresGrad());
            assertTrue(TensorOps.sub(ta, ta).requiresGrad());
            assertTrue(TensorOps.sum(ta, 1).requiresGrad());
            assertFalse("no operand requires grad", keep(TensorOps.bmm(keep(ta.detach()), tb)).requiresGrad());
            assertEquals("one node per recorded call", before + 5, graph.size());
        }
        assertEquals(before, graph.size());
        try (ComputationGraph.NoGradScope off = ComputationGraph.noGrad()) {
            assertFalse(keep(TensorOps.bmm(ta, tb)).requiresGrad());
            assertFalse(keep(TensorOps.matmul(ta, tb)).requiresGrad());
            assertFalse(keep(TensorOps.add(ta, ta)).requiresGrad());
            assertFalse(keep(TensorOps.sum(ta, 1)).requiresGrad());
        }
        assertEquals("nothing records under noGrad", before, graph.size());
    }

    @Test
    public void inPlaceOpsRefuseTensorsThatRequireGradInGradMode() {
        Tensor leaf = keep(floats(new float[]{1f, 2f, 3f}, 3).withRequiresGrad(true));
        Tensor src = floats(new float[]{1f, 1f, 1f}, 3);
        assertThrows(IllegalStateException.class, () -> TensorOps.addInPlace(leaf, src, 1f));
        assertThrows(IllegalStateException.class, () -> TensorOps.mulInPlace(leaf, 2f));
        assertThrows(IllegalStateException.class, () -> TensorOps.zeroInPlace(leaf));
        assertArrayEquals("a refused write leaves the tensor unchanged", new float[]{1f, 2f, 3f},
                leaf.toFloatArray(), 0f);
        try (ComputationGraph.NoGradScope off = ComputationGraph.noGrad()) {
            TensorOps.addInPlace(leaf, src, 1f);
            TensorOps.mulInPlace(leaf, 2f);
        }
        assertArrayEquals(new float[]{4f, 6f, 8f}, leaf.toFloatArray(), 0f);
        try (ComputationGraph.NoGradScope off = ComputationGraph.noGrad()) {
            TensorOps.zeroInPlace(leaf);
        }
        assertArrayEquals(new float[3], leaf.toFloatArray(), 0f);
    }

    // ------------------------------------------------------------------ index and segment backward

    @Test
    public void indexAddRowsSumsInIndexOrderAndSkipsRows() {
        float[] gy = {1f, 2f, 10f, 20f, 100f, 200f, 1000f, 2000f, 5f, 6f};
        Tensor g = floats(gy, 5, 2);
        Tensor idx = ints(new int[]{2, 0, 2, 7, -1}, 5);
        assertArrayEquals("error mode: out-of-range indices add nothing",
                new float[]{10f, 20f, 0f, 0f, 101f, 202f},
                keep(TensorOps.indexAddRows(g, idx, 3, false, -1)).toFloatArray(), 0f);
        assertArrayEquals("clamp mode: 7 -> 2 and -1 -> 0",
                new float[]{15f, 26f, 0f, 0f, 1101f, 2202f},
                keep(TensorOps.indexAddRows(g, idx, 3, true, -1)).toFloatArray(), 0f);
        assertArrayEquals("the skip row gets nothing", new float[]{15f, 26f, 0f, 0f, 0f, 0f},
                keep(TensorOps.indexAddRows(g, idx, 3, true, 2)).toFloatArray(), 0f);
        Tensor wide = keep(TensorOps.indexAddRows(g, keep(Tensor.fromStorage(host, Shape.of(5), DType.INT64,
                TensorStorage.fromHost(host, longs(2, 0, 2, (1L << 32) | 7, -1), DType.INT64))), 3, true, -1));
        assertArrayEquals("INT64 indices narrow by (int)", new float[]{15f, 26f, 0f, 0f, 1101f, 2202f},
                wide.toFloatArray(), 0f);
        Tensor empty = keep(TensorOps.indexAddRows(g, idx, 0, true, -1));
        assertEquals(Shape.of(0, 2), empty.shape());
        assertFalse(empty.requiresGrad());
        assertThrows(IllegalArgumentException.class, () -> TensorOps.indexAddRows(g, idx, 3, false, 3));
        assertThrows(IllegalArgumentException.class, () -> TensorOps.indexAddRows(g, ints(new int[]{0, 1}, 2), 3,
                false, -1));
        assertThrows(UnsupportedOperationException.class, () -> TensorOps.indexAddRows(g, g, 3, false, -1));
    }

    @Test
    public void segmentReduceBackwardRoutesPerOp() {
        float ninf = Float.NEGATIVE_INFINITY;
        float[] src = {1f, -0f, 4f, 7f, 4f, Float.NaN, -2f, 3f, 4f, 9f};
        Tensor s = floats(src, 5, 2);
        Tensor p = ints(new int[]{0, 3, 3, 5}, 4);
        Tensor e = ints(new int[]{2, 0, 4, 1, 3}, 5);
        Tensor gy = floats(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, 3, 2);
        assertArrayEquals("SUM copies each segment's gradient to its rows",
                new float[]{1f, 2f, 5f, 6f, 1f, 2f, 5f, 6f, 1f, 2f},
                keep(TensorOps.segmentReduceBackward(s, p, e, null, gy, GpuReduceOps.SegmentOp.SUM)).toFloatArray(),
                0f);
        assertArrayEquals("MEAN divides by the edge count",
                new float[]{1f / 3f, 2f / 3f, 2.5f, 3f, 1f / 3f, 2f / 3f, 2.5f, 3f, 1f / 3f, 2f / 3f},
                keep(TensorOps.segmentReduceBackward(s, p, e, null, gy, GpuReduceOps.SegmentOp.MEAN))
                        .toFloatArray(), 0f);
        Tensor out = keep(TensorOps.segmentReduce(s, p, e, 3, GpuReduceOps.SegmentOp.MAX, ninf));
        float[] max = keep(TensorOps.segmentReduceBackward(s, p, e, out, gy, GpuReduceOps.SegmentOp.MAX))
                .toFloatArray();
        assertArrayEquals("MAX routes to the first row (edge order) equal to the result; NaN routes nowhere",
                new float[]{0f, 0f, 5f, 6f, 1f, 0f, 0f, 0f, 0f, 0f}, max, 0f);
        assertThrows("a source row named twice", IllegalArgumentException.class,
                () -> TensorOps.segmentReduceBackward(s, p, ints(new int[]{2, 0, 2, 1, 3}, 5), null, gy,
                        GpuReduceOps.SegmentOp.SUM));
        assertThrows("MAX needs the forward output", NullPointerException.class,
                () -> TensorOps.segmentReduceBackward(s, p, e, null, gy, GpuReduceOps.SegmentOp.MAX));
        assertThrows("gradient of the wrong shape", IllegalArgumentException.class,
                () -> TensorOps.segmentReduceBackward(s, p, e, null, s, GpuReduceOps.SegmentOp.SUM));
    }

    // ------------------------------------------------------------------ helpers

    /** Population variance per kept index, in double precision. */
    private static float[] variance(float[] x, int[] dims, boolean[] reduced) {
        int rank = dims.length;
        int kept = 1;
        int length = 1;
        for (int k = 0; k < rank; k++) {
            if (reduced[k]) {
                length *= dims[k];
            } else {
                kept *= dims[k];
            }
        }
        double[] sum = new double[kept];
        double[] sq = new double[kept];
        int[] coord = new int[rank];
        for (int i = 0; i < x.length; i++) {
            int rem = i;
            for (int k = rank - 1; k >= 0; k--) {
                coord[k] = rem % dims[k];
                rem /= dims[k];
            }
            int key = 0;
            for (int k = 0; k < rank; k++) {
                if (!reduced[k]) {
                    key = key * dims[k] + coord[k];
                }
            }
            sum[key] += x[i];
            sq[key] += (double) x[i] * x[i];
        }
        float[] out = new float[kept];
        for (int r = 0; r < kept; r++) {
            double mean = sum[r] / length;
            out[r] = (float) (sq[r] / length - mean * mean);
        }
        return out;
    }

    private static ByteBuffer longs(long... values) {
        ByteBuffer buffer = ByteBuffer.allocate(values.length * Long.BYTES).order(ByteOrder.nativeOrder());
        for (long v : values) {
            buffer.putLong(v);
        }
        buffer.flip();
        return buffer;
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

    private Tensor floats(float[] data, int... dims) {
        return keep(keep(Tensor.fromArray(host, data)).reshape(Shape.of(dims)));
    }

    private Tensor ints(int[] data, int... dims) {
        return keep(keep(Tensor.fromArray(host, data)).reshape(Shape.of(dims)));
    }
}

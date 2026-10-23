package com.eonmux.neuralfree.tensor;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.autograd.GradFunctions;
import com.eonmux.neuralfree.autograd.GradientFunction;
import com.eonmux.neuralfree.tensor.gpu.TrainParity;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.fail;

/**
 * {@link GpuReduceOps} against the CPU path on each GPU API, under the tolerance classes of the
 * {@linkplain com.eonmux.neuralfree.tensor.gpu numerical contract}: sums, means and variances against a
 * double-precision reference (every layout path: in place, moved dims, column sums, chunked rows and columns, rank-7
 * views), Java max/min and arg-reductions and the max backward bit for bit, softmax forms and their backward within the
 * transcendental class, sumToShape, and segment reductions.
 */
@RunWith(Parameterized.class)
public class GpuReduceOpsParityTest {

    @Parameterized.Parameters(name = "{0}")
    public static Collection<Object[]> data() {
        return TrainParity.apis();
    }

    private final TrainParity.Api api;
    private TrainParity.Session session;
    private ComputeBackend gpu;
    private ComputeBackend host;
    private final List<Tensor> owned = new ArrayList<>();

    public GpuReduceOpsParityTest(TrainParity.Api api) {
        this.api = api;
    }

    @Before
    public void setUp() {
        session = TrainParity.open(api);
        gpu = session.backend();
        host = TrainParity.hostBackend();
    }

    @After
    public void tearDown() {
        try {
            for (Tensor t : owned) {
                t.close();
            }
        } finally {
            if (session != null) {
                session.close();
            }
        }
    }

    // ------------------------------------------------------------------ helpers

    private Tensor keep(Tensor t) {
        assertNotNull("the train library is available, so the dispatcher must not return null", t);
        owned.add(t);
        return t;
    }

    private Tensor floats(ComputeBackend b, float[] data, int... dims) {
        Tensor flat = keep(Tensor.fromArray(b, data));
        return keep(flat.reshape(Shape.of(dims)));
    }

    private static boolean[] flags(int rank, int... dims) {
        boolean[] reduced = new boolean[rank];
        if (dims.length == 0) {
            Arrays.fill(reduced, true);
        }
        for (int d : dims) {
            reduced[d < 0 ? d + rank : d] = true;
        }
        return reduced;
    }

    /**
     * Double-precision sums over the reduced dims, per output in row-major order of the kept dims: {@code [0]} the
     * sum of the terms, {@code [1]} the sum of their absolute values. With {@code center}, a term is
     * {@code (x - center[out])^2}.
     */
    private static double[][] reference(float[] x, int[] shape, boolean[] reduced, double[] center) {
        int outN = 1;
        for (int k = 0; k < shape.length; k++) {
            outN *= reduced[k] ? 1 : shape[k];
        }
        double[] sum = new double[outN];
        double[] abs = new double[outN];
        for (int i = 0; i < x.length; i++) {
            int rem = i;
            int[] coord = new int[shape.length];
            for (int k = shape.length - 1; k >= 0; k--) {
                coord[k] = rem % shape[k];
                rem /= shape[k];
            }
            int o = 0;
            for (int k = 0; k < shape.length; k++) {
                o = reduced[k] ? o : o * shape[k] + coord[k];
            }
            double v = x[i];
            if (center != null) {
                v = (v - center[o]) * (v - center[o]);
            }
            sum[o] += v;
            abs[o] += Math.abs(v);
        }
        return new double[][]{sum, abs};
    }

    /** Sum, mean and variance of {@code dev} (logical data {@code x} of {@code shape}) against the double reference. */
    private void checkSumMeanVar(String what, Tensor dev, float[] x, int[] shape, int... dims) {
        boolean[] reduced = flags(shape.length, dims);
        double[][] s = reference(x, shape, reduced, null);
        int k = s[0].length == 0 ? 0 : x.length / s[0].length;
        double[] mean = new double[s[0].length];
        double[] meanAbs = new double[s[0].length];
        for (int i = 0; i < mean.length; i++) {
            mean[i] = s[0][i] / k;
            meanAbs[i] = s[1][i] / k;
        }
        double[][] v = reference(x, shape, reduced, mean);
        for (int i = 0; i < mean.length; i++) {
            v[0][i] /= k;
        }
        Shape cpuShape = BroadcastEngine.reduceShape(Shape.of(shape), dims.length == 0 ? allDims(shape.length) : dims,
                false);
        assertReductions(what + " sum", cpuShape, s[0], s[1], k, keep(GpuReduceOps.sum(dev, dims)));
        assertReductions(what + " mean", cpuShape, mean, meanAbs, k, keep(GpuReduceOps.mean(dev, dims)));
        assertReductions(what + " var", cpuShape, v[0], v[0], k, keep(GpuReduceOps.var(dev, dims)));
    }

    private static int[] allDims(int rank) {
        int[] all = new int[rank];
        for (int i = 0; i < rank; i++) {
            all[i] = i;
        }
        return all;
    }

    private static void assertReductions(String what, Shape shape, double[] ref, double[] sumAbs, int k, Tensor dev) {
        assertEquals(what + ": shape", shape, dev.shape());
        float[] got = dev.toFloatArray();
        assertEquals(what + ": length", ref.length, got.length);
        for (int i = 0; i < ref.length; i++) {
            TrainParity.assertReduction(what + "[" + i + "]", ref[i], got[i], k, sumAbs[i]);
        }
    }

    private void assertSameFloats(String what, Tensor ref, Tensor dev) {
        assertEquals(what + ": shape", ref.shape(), dev.shape());
        TrainParity.assertBitIdentical(what, ref.toFloatArray(), dev.toFloatArray());
    }

    // ------------------------------------------------------------------ sums, means, variances

    @Test
    public void sumMeanVarOverEveryLayoutPath() {
        int[] shape = {5, 7, 300};
        float[] x = TrainParity.uniform(1, 5 * 7 * 300, -10f, 10f);
        Tensor dev = floats(gpu, x, shape);
        int[][] dimSets = {{}, {2}, {-1}, {0}, {1}, {0, 2}, {1, 2}, {0, 1}, {2, 0}};
        for (int[] dims : dimSets) {
            checkSumMeanVar("dims " + Arrays.toString(dims), dev, x, shape, dims);
        }
    }

    @Test
    public void chunkedRowsAndColumns() {
        float[] wide = TrainParity.uniform(2, 3 * 70001, -1f, 1f);
        checkSumMeanVar("rows 3 x 70001", floats(gpu, wide, 3, 70001), wide, new int[]{3, 70001}, 1);
        float[] tall = TrainParity.uniform(3, 5000 * 3, -1f, 1f);
        Tensor devTall = floats(gpu, tall, 5000, 3);
        checkSumMeanVar("columns 5000 x 3", devTall, tall, new int[]{5000, 3}, 0);
        double[][] cols = reference(tall, new int[]{5000, 3}, flags(2, 0), null);
        assertReductions("sumColumns", Shape.of(3), cols[0], cols[1], 5000, keep(GpuReduceOps.sumColumns(devTall)));
        float[] vector = TrainParity.uniform(4, 70001, -1f, 1f);
        checkSumMeanVar("vector 70001", floats(gpu, vector, 70001), vector, new int[]{70001});
        Tensor hostVector = floats(host, vector, 70001);
        Tensor devVector = floats(gpu, vector, 70001);
        assertSameFloats("chunked max", keep(TensorOps.max(hostVector, 0)), keep(GpuReduceOps.max(devVector, 0)));
        assertSameFloats("chunked min", keep(TensorOps.min(hostVector, 0)), keep(GpuReduceOps.min(devVector, 0)));
    }

    @Test
    public void stridedOffsetAndRankSevenInputs() {
        float[] x = TrainParity.uniform(5, 8 * 6 * 5, -10f, 10f);
        Tensor hostT = keep(floats(host, x, 8, 6, 5).permute(2, 0, 1));
        Tensor devT = keep(floats(gpu, x, 8, 6, 5).permute(2, 0, 1));
        checkSumMeanVar("transposed", devT, hostT.toFloatArray(), new int[]{5, 8, 6}, 1);
        assertSameFloats("transposed max", keep(TensorOps.max(hostT, 2)), keep(GpuReduceOps.max(devT, 2)));
        Tensor hostS = keep(floats(host, x, 8, 6, 5).slice(0, 2, 7));
        Tensor devS = keep(floats(gpu, x, 8, 6, 5).slice(0, 2, 7));
        checkSumMeanVar("offset slice", devS, hostS.toFloatArray(), new int[]{5, 6, 5}, 2);
        assertArrayEquals("offset slice argmax", keep(TensorOps.argmax(hostS, 2)).toIntArray(),
                keep(GpuReduceOps.argmax(devS, 2)).toIntArray());

        int[] base = {2, 3, 2, 2, 3, 2, 4};
        float[] y = TrainParity.uniform(6, 2 * 3 * 2 * 2 * 3 * 2 * 4, -10f, 10f);
        int[] reverse = {6, 5, 4, 3, 2, 1, 0};
        Tensor hostR = keep(floats(host, y, base).permute(reverse));
        Tensor devR = keep(floats(gpu, y, base).permute(reverse));
        int[] shapeR = hostR.shape().dims();
        checkSumMeanVar("rank 7 reversed", devR, hostR.toFloatArray(), shapeR, 0);
        checkSumMeanVar("rank 7 two dims", devR, hostR.toFloatArray(), shapeR, 1, 5);
        assertSameFloats("rank 7 max", keep(TensorOps.max(hostR, 3)), keep(GpuReduceOps.max(devR, 3)));
        TrainParity.assertTranscendental("rank 7 softmax", keep(TensorOps.softmax(hostR, 0)).toFloatArray(),
                keep(GpuReduceOps.softmax(devR, 0)).toFloatArray());
    }

    // ------------------------------------------------------------------ max, min, arg-reductions, max backward

    /** Rows with NaN, ties, signed zeros, infinities and an all -inf row. */
    private static float[] edgeMatrix() {
        float[] x = TrainParity.uniform(7, 6 * 9, -5f, 5f);
        x[9 + 3] = Float.NaN;
        x[18 + 2] = 4.5f;
        x[18 + 5] = 4.5f;
        for (int c = 0; c < 9; c++) {
            x[27 + c] = Float.NEGATIVE_INFINITY;
            x[36 + c] = c % 2 == 0 ? -0.0f : 0.0f;
        }
        x[45 + 7] = Float.POSITIVE_INFINITY;
        x[45 + 1] = Float.NaN;
        return x;
    }

    @Test
    public void maxMinArgMatchTheCpuBitForBit() {
        float[] x = edgeMatrix();
        Tensor hostX = floats(host, x, 6, 9);
        Tensor devX = floats(gpu, x, 6, 9);
        for (int dim : new int[]{0, 1, -1}) {
            String d = "dim " + dim;
            assertSameFloats("max " + d, keep(TensorOps.max(hostX, dim)), keep(GpuReduceOps.max(devX, dim)));
            assertSameFloats("min " + d, keep(TensorOps.min(hostX, dim)), keep(GpuReduceOps.min(devX, dim)));
            Tensor refArg = keep(TensorOps.argmax(hostX, dim));
            Tensor devArg = keep(GpuReduceOps.argmax(devX, dim));
            assertEquals("argmax " + d + ": dtype", DType.INT32, devArg.dtype());
            assertEquals("argmax " + d + ": shape", refArg.shape(), devArg.shape());
            assertArrayEquals("argmax " + d, refArg.toIntArray(), devArg.toIntArray());
            assertArrayEquals("argmin " + d, keep(TensorOps.argmin(hostX, dim)).toIntArray(),
                    keep(GpuReduceOps.argmin(devX, dim)).toIntArray());
        }
        float[] wide = TrainParity.uniform(8, 3 * 300, -5f, 5f);
        wide[17] = Float.NaN;
        wide[300 + 250] = 5f;
        wide[300 + 260] = 5f;
        Tensor hostW = floats(host, wide, 3, 300);
        Tensor devW = floats(gpu, wide, 3, 300);
        assertSameFloats("group max", keep(TensorOps.max(hostW, 1)), keep(GpuReduceOps.max(devW, 1)));
        assertArrayEquals("group argmax", keep(TensorOps.argmax(hostW, 1)).toIntArray(),
                keep(GpuReduceOps.argmax(devW, 1)).toIntArray());
        assertArrayEquals("moved argmin", keep(TensorOps.argmin(hostW, 0)).toIntArray(),
                keep(GpuReduceOps.argmin(devW, 0)).toIntArray());
    }

    @Test
    public void maxBackwardMatchesMaxGrad() {
        float[] x = edgeMatrix();
        Tensor hostX = floats(host, x, 6, 9);
        Tensor devX = floats(gpu, x, 6, 9);
        for (int dim : new int[]{1, 0}) {
            int rows = dim == 1 ? 6 : 9;
            float[] g = TrainParity.uniform(9 + dim, rows, -2f, 2f);
            Tensor hostMax = keep(TensorOps.max(hostX, dim));
            GradFunctions.MaxGrad grad = new GradFunctions.MaxGrad(dim, hostX.shape());
            grad.saveForBackward(hostX, hostMax);
            Tensor ref = keep(grad.backward(floats(host, g, rows)).get(0));
            Tensor dev = keep(GpuReduceOps.maxBackward(devX, keep(GpuReduceOps.max(devX, dim)), floats(gpu, g, rows),
                    dim));
            assertSameFloats("maxBackward dim " + dim, ref, dev);
        }
    }

    // ------------------------------------------------------------------ softmax

    @Test
    public void softmaxAndLogSoftmaxMatchTheCpu() {
        float[] x = TrainParity.uniform(10, 4 * 33 * 6, -10f, 10f);
        Tensor hostX = floats(host, x, 4, 33, 6);
        Tensor devX = floats(gpu, x, 4, 33, 6);
        for (int dim : new int[]{2, -1, 1, 0}) {
            TrainParity.assertTranscendental("softmax dim " + dim, keep(TensorOps.softmax(hostX, dim)).toFloatArray(),
                    keep(GpuReduceOps.softmax(devX, dim)).toFloatArray());
            TrainParity.assertTranscendental("logSoftmax dim " + dim,
                    keep(TensorOps.logSoftmax(hostX, dim)).toFloatArray(),
                    keep(GpuReduceOps.logSoftmax(devX, dim)).toFloatArray());
        }
    }

    /** {@code MultiHeadAttention}'s masked softmax over the last dim of a {@code [rows, cols]} array. */
    private static float[] maskedReference(float[] x, int rows, int cols) {
        float[] out = new float[x.length];
        for (int r = 0; r < rows; r++) {
            float max = Float.NEGATIVE_INFINITY;
            for (int c = 0; c < cols; c++) {
                max = Math.max(max, x[r * cols + c]);
            }
            float sum = 0.0f;
            for (int c = 0; c < cols; c++) {
                if (x[r * cols + c] != Float.NEGATIVE_INFINITY) {
                    sum += (float) Math.exp(x[r * cols + c] - max);
                }
            }
            for (int c = 0; c < cols; c++) {
                float v = x[r * cols + c];
                out[r * cols + c] = v == Float.NEGATIVE_INFINITY ? 0.0f : (float) Math.exp(v - max) / sum;
            }
        }
        return out;
    }

    @Test
    public void maskedSoftmaxZeroesMaskedEntriesAndRows() {
        int rows = 3;
        int cols = 70;
        float[] x = TrainParity.uniform(11, rows * cols, -10f, 10f);
        for (int c = 0; c < cols; c++) {
            if (c % 3 == 0) {
                x[c] = Float.NEGATIVE_INFINITY;
            }
            x[2 * cols + c] = Float.NEGATIVE_INFINITY;
        }
        float[] ref = maskedReference(x, rows, cols);
        float[] dev = keep(GpuReduceOps.maskedSoftmax(floats(gpu, x, rows, cols), 1)).toFloatArray();
        TrainParity.assertTranscendental("masked softmax", ref, dev);
        for (int i = 0; i < x.length; i++) {
            if (x[i] == Float.NEGATIVE_INFINITY) {
                assertEquals("masked entry " + i + " is exactly +0", 0, Float.floatToRawIntBits(dev[i]));
            }
        }
        Tensor devT = keep(floats(gpu, x, rows, cols).transpose());
        float[] moved = keep(GpuReduceOps.maskedSoftmax(devT, 0)).toFloatArray();
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                TrainParity.assertTranscendental("masked softmax over dim 0 [" + c + "," + r + "]",
                        new float[]{ref[r * cols + c]}, new float[]{moved[c * rows + r]});
            }
        }
    }

    @Test
    public void softmaxBackwardMatchesSoftmaxGradAndLogSoftmaxGrad() {
        float[] x = TrainParity.uniform(12, 5 * 40 * 3, -4f, 4f);
        float[] g = TrainParity.uniform(13, 5 * 40 * 3, -1f, 1f);
        Tensor hostX = floats(host, x, 5, 40, 3);
        Tensor hostG = floats(host, g, 5, 40, 3);
        Tensor devG = floats(gpu, g, 5, 40, 3);
        for (int dim : new int[]{2, 1}) {
            for (boolean log : new boolean[]{false, true}) {
                Tensor hostY = keep(log ? TensorOps.logSoftmax(hostX, dim) : TensorOps.softmax(hostX, dim));
                GradientFunction fn;
                if (log) {
                    GradFunctions.LogSoftmaxGrad lg = new GradFunctions.LogSoftmaxGrad(dim);
                    lg.saveForBackward(hostY);
                    fn = lg;
                } else {
                    GradFunctions.SoftmaxGrad sg = new GradFunctions.SoftmaxGrad(dim);
                    sg.saveForBackward(hostY);
                    fn = sg;
                }
                float[] ref = keep(fn.backward(hostG).get(0)).toFloatArray();
                Tensor devY = floats(gpu, hostY.toFloatArray(), 5, 40, 3);
                float[] dev = keep(GpuReduceOps.softmaxBackward(devY, devG, dim, log)).toFloatArray();
                TrainParity.assertTranscendental((log ? "log" : "") + "softmax backward dim " + dim, ref, dev);
            }
        }
    }

    // ------------------------------------------------------------------ sumToShape

    @Test
    public void sumToShapeReducesBroadcastDims() {
        int[] shape = {4, 3, 5};
        float[] x = TrainParity.uniform(14, 60, -10f, 10f);
        Tensor dev = floats(gpu, x, shape);
        Object[][] cases = {
                {Shape.of(3, 1), flags(3, 0, 2)},
                {Shape.of(5), flags(3, 0, 1)},
                {Shape.of(1, 3, 5), flags(3, 0)},
                {Shape.of(4, 1, 1), flags(3, 1, 2)},
                {Shape.scalar(), flags(3)},
        };
        for (Object[] c : cases) {
            Shape target = (Shape) c[0];
            double[][] ref = reference(x, shape, (boolean[]) c[1], null);
            assertReductions("sumToShape " + target, target, ref[0], ref[1], (int) (60 / target.numElements()),
                    keep(GpuReduceOps.sumToShape(dev, target)));
        }
        Tensor same = keep(GpuReduceOps.sumToShape(dev, Shape.of(shape)));
        TrainParity.assertBitIdentical("sumToShape copy", x, same.toFloatArray());
        float[] row = TrainParity.uniform(15, 4, -1f, 1f);
        Tensor rearranged = keep(GpuReduceOps.sumToShape(floats(gpu, row, 1, 4), Shape.of(4, 1)));
        assertEquals(Shape.of(4, 1), rearranged.shape());
        TrainParity.assertBitIdentical("sumToShape reshape", row, rearranged.toFloatArray());
        try {
            GpuReduceOps.sumToShape(dev, Shape.of(60));
            fail("incompatible shapes must throw");
        } catch (IllegalArgumentException expected) {
            assertNotNull(expected.getMessage());
        }
    }

    // ------------------------------------------------------------------ segments

    @Test
    public void segmentReduceMatchesTheCpuAggregations() {
        int rows = 7;
        int[] ptr = {0, 3, 3, 5, 9, 10};
        int[] edges = {6, 0, 2, 1, 1, 3, 5, 0, 4, 2};
        int nodes = ptr.length - 1;
        for (int[] featDims : new int[][]{{5}, {2, 3}}) {
            int feat = 1;
            for (int d : featDims) {
                feat *= d;
            }
            float[] src = TrainParity.uniform(16 + feat, rows * feat, -10f, 10f);
            for (int f = 0; f < feat; f++) {
                src[2 * feat + f] = Float.NEGATIVE_INFINITY;
            }
            int[] srcDims = new int[featDims.length + 1];
            srcDims[0] = rows;
            System.arraycopy(featDims, 0, srcDims, 1, featDims.length);
            Tensor devSrc = floats(gpu, src, srcDims);
            Tensor devPtr = keep(Tensor.fromArray(gpu, ptr));
            Tensor devEdges = keep(Tensor.fromArray(gpu, edges));
            float[] sum = new float[nodes * feat];
            float[] mean = new float[nodes * feat];
            float[] max = new float[nodes * feat];
            for (int t = 0; t < nodes; t++) {
                for (int f = 0; f < feat; f++) {
                    float s = 0.0f;
                    float m = Float.NEGATIVE_INFINITY;
                    for (int e = ptr[t]; e < ptr[t + 1]; e++) {
                        s += src[edges[e] * feat + f];
                        m = Math.max(m, src[edges[e] * feat + f]);
                    }
                    int count = ptr[t + 1] - ptr[t];
                    sum[t * feat + f] = count == 0 ? -1.5f : s;
                    mean[t * feat + f] = count == 0 ? -1.5f : s / count;
                    max[t * feat + f] = m == Float.NEGATIVE_INFINITY ? -1.5f : m;
                }
            }
            String what = "segment feat " + Arrays.toString(featDims);
            Tensor devSum = keep(GpuReduceOps.segmentReduce(devSrc, devPtr, devEdges, nodes,
                    GpuReduceOps.SegmentOp.SUM, -1.5f));
            int[] outDims = srcDims.clone();
            outDims[0] = nodes;
            assertEquals(what + ": shape", Shape.of(outDims), devSum.shape());
            TrainParity.assertSingleRounding(what + " sum", sum, devSum.toFloatArray());
            TrainParity.assertDivSqrt(what + " mean", mean, keep(GpuReduceOps.segmentReduce(devSrc, devPtr, devEdges,
                    nodes, GpuReduceOps.SegmentOp.MEAN, -1.5f)).toFloatArray());
            TrainParity.assertBitIdentical(what + " max", max, keep(GpuReduceOps.segmentReduce(devSrc, devPtr,
                    devEdges, nodes, GpuReduceOps.SegmentOp.MAX, -1.5f)).toFloatArray());
        }
    }

    // ------------------------------------------------------------------ edges of the contract

    @Test
    public void emptyReductionsFollowTheCpu() {
        Tensor empty = keep(GpuCoreOps.zeros(gpu, Shape.of(3, 0), DType.FLOAT32));
        TrainParity.assertBitIdentical("sum over an empty dim", new float[3],
                keep(GpuReduceOps.sum(empty, 1)).toFloatArray());
        for (float v : keep(GpuReduceOps.mean(empty, 1)).toFloatArray()) {
            assertEquals("mean over an empty dim is NaN", Float.NaN, v, 0.0f);
        }
        float[] negInf = {Float.NEGATIVE_INFINITY, Float.NEGATIVE_INFINITY, Float.NEGATIVE_INFINITY};
        TrainParity.assertBitIdentical("max over an empty dim", negInf,
                keep(GpuReduceOps.max(empty, 1)).toFloatArray());
        assertArrayEquals("argmax over an empty dim", new int[3], keep(GpuReduceOps.argmax(empty, 1)).toIntArray());
        assertEquals(Shape.of(0), keep(GpuReduceOps.sum(empty, 0)).shape());
    }

    @Test
    public void unsupportedDtypesAndUnavailableBackendsReturnNull() {
        Tensor ints = keep(Tensor.fromArray(gpu, new int[]{1, 2, 3, 4}));
        assertNull(GpuReduceOps.sum(ints));
        assertNull(GpuReduceOps.max(ints, 0));
        assertNull(GpuReduceOps.softmax(ints, 0));
        Tensor f = floats(gpu, new float[]{1f, 2f, 3f, 4f}, 2, 2);
        assertNull(GpuReduceOps.segmentReduce(f, f, keep(Tensor.fromArray(gpu, new int[]{0})), 1,
                GpuReduceOps.SegmentOp.SUM, 0f));
        assertNull(GpuReduceOps.sum(floats(host, new float[]{1f, 2f}, 2)));
        assertNull(GpuReduceOps.softmax(floats(host, new float[]{1f, 2f}, 2), 0));
    }

    @Test
    public void invalidDimsThrowOnceTheLibraryIsAvailable() {
        Tensor f = floats(gpu, new float[]{1f, 2f, 3f, 4f, 5f, 6f}, 2, 3);
        try {
            GpuReduceOps.sum(f, 1, -1);
            fail("a repeated dim must throw");
        } catch (IllegalArgumentException expected) {
            assertNotNull(expected.getMessage());
        }
        try {
            GpuReduceOps.max(f, 2);
            fail("a dim outside the rank must throw");
        } catch (IndexOutOfBoundsException expected) {
            assertNotNull(expected.getMessage());
        }
        try {
            GpuReduceOps.sumColumns(floats(gpu, new float[]{1f, 2f}, 2));
            fail("sumColumns needs a 2-D tensor");
        } catch (IllegalArgumentException expected) {
            assertNotNull(expected.getMessage());
        }
    }
}

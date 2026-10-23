package com.eonmux.neuralfree.tensor;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.tensor.gpu.TrainGpu;
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

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * {@link GpuNormOps} on each GPU API against double-precision references of the nn/norm CPU formulas (reduction
 * class {@code 1e-5 * sqrt(K) * sum|terms| + K * FLT_MIN}, with the absolute sizes of the terms that form each output
 * as the term sum), and
 * index-select bit for bit against the CPU row copy. Also checks the CPU-path contract on a backend without the
 * train library.
 */
@RunWith(Parameterized.class)
public class GpuNormOpsParityTest {

    private static final float EPS = 1e-5f;

    @Parameterized.Parameters(name = "{0}")
    public static Collection<Object[]> data() {
        return TrainParity.apis();
    }

    private final TrainParity.Api api;
    private TrainParity.Session session;
    private ComputeBackend gpu;
    private final List<Tensor> owned = new ArrayList<>();

    public GpuNormOpsParityTest(TrainParity.Api api) {
        this.api = api;
    }

    @Before
    public void setUp() {
        session = TrainParity.open(api);
        gpu = session.backend();
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

    private Tensor keep(Tensor t) {
        owned.add(t);
        return t;
    }

    private Tensor upload(ComputeBackend b, float[] data, int... dims) {
        Tensor flat = keep(Tensor.fromArray(b, data));
        return keep(flat.reshape(Shape.of(dims)));
    }

    // ------------------------------------------------------------------ row normalizations

    @Test
    public void layerNormMatchesReference() {
        int rows = 7;
        int cols = 300;
        float[] x = TrainParity.uniform(11, rows * cols, -3f, 5f);
        float[] gamma = TrainParity.uniform(12, cols, 0.5f, 1.5f);
        float[] beta = TrainParity.uniform(13, cols, -1f, 1f);
        Tensor out = keep(GpuNormOps.layerNorm(upload(gpu, x, rows, cols), cols, upload(gpu, gamma, cols),
                upload(gpu, beta, cols), EPS));
        assertRows("layerNorm", out.toFloatArray(), x, rows, cols, gamma, beta, false, EPS);

        Tensor plain = keep(GpuNormOps.layerNorm(upload(gpu, x, rows, cols), cols, null, null, EPS));
        assertRows("layerNorm without affine", plain.toFloatArray(), x, rows, cols, null, null, false, EPS);
    }

    @Test
    public void layerNormStagesStridedInput() {
        int rows = 5;
        int cols = 70;
        float[] stored = TrainParity.uniform(21, rows * cols, -2f, 2f);
        Tensor transposed = keep(upload(gpu, stored, cols, rows).transpose());
        float[] logical = new float[rows * cols];
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                logical[r * cols + c] = stored[c * rows + r];
            }
        }
        Tensor out = keep(GpuNormOps.layerNorm(transposed, cols, null, null, EPS));
        assertRows("layerNorm of a transposed view", out.toFloatArray(), logical, rows, cols, null, null, false,
                EPS);
    }

    @Test
    public void rmsNormMatchesReference() {
        int rows = 9;
        int cols = 40;
        float[] x = TrainParity.uniform(31, rows * cols, -4f, 4f);
        float[] w = TrainParity.uniform(32, cols, 0.25f, 2f);
        float eps = 1e-8f;
        Tensor out = keep(GpuNormOps.rmsNorm(upload(gpu, x, 3, 3, cols), upload(gpu, w, cols), eps));
        assertRows("rmsNorm", out.toFloatArray(), x, rows, cols, w, null, true, eps);
    }

    // ------------------------------------------------------------------ channel normalizations

    @Test
    public void batchNormTrainNormalizesAndUpdatesRunningStats() {
        int b = 3;
        int c = 4;
        int hw = 6 * 5;
        float momentum = 0.1f;
        float[] all = TrainParity.uniform(41, (b + 1) * c * hw, -2f, 3f);
        float[] x = new float[b * c * hw];
        System.arraycopy(all, c * hw, x, 0, x.length);
        Tensor input = keep(upload(gpu, all, b + 1, c, 6, 5).slice(0, 1, b + 1));
        float[] gamma = TrainParity.uniform(42, c, 0.5f, 1.5f);
        float[] beta = TrainParity.uniform(43, c, -0.5f, 0.5f);
        float[] runMean = TrainParity.uniform(44, c, -0.1f, 0.1f);
        float[] runVar = TrainParity.uniform(45, c, 0.9f, 1.1f);
        Tensor rm = upload(gpu, runMean, c);
        Tensor rv = upload(gpu, runVar, c);

        Tensor out = keep(GpuNormOps.batchNormTrain(input, upload(gpu, gamma, c), upload(gpu, beta, c), rm, rv,
                momentum, EPS));

        ChannelStats stats = ChannelStats.of(x, b, c, hw, 0);
        assertChannels("batchNormTrain", out.toFloatArray(), x, b, c, hw, stats, gamma, beta);
        float[] gotMean = rm.toFloatArray();
        float[] gotVar = rv.toFloatArray();
        double bessel = unbiasedScale(b * hw);
        for (int k = 0; k < c; k++) {
            double expMean = (1 - momentum) * (double) runMean[k] + momentum * stats.mean[k];
            double expVar = (1 - momentum) * (double) runVar[k] + momentum * stats.var[k] * bessel;
            TrainParity.assertReduction("running mean " + k, expMean, gotMean[k], b * hw,
                    (1 - momentum) * Math.abs(runMean[k]) + momentum * stats.meanTerms[k]);
            TrainParity.assertReduction("running var " + k, expVar, gotVar[k], b * hw,
                    (1 - momentum) * Math.abs(runVar[k]) + momentum * stats.varTerms[k] * bessel);
        }
    }

    @Test
    public void batchNormTrainMomentumOneKeepsCpuRunningUpdate() {
        int b = 2;
        int c = 3;
        int l = 9;
        float[] x = TrainParity.uniform(46, b * c * l, -2f, 2f);
        float[] runMean = {Float.NaN, Float.POSITIVE_INFINITY, -0.25f};
        float[] runVar = {1f, Float.NaN, 0.5f};
        Tensor rm = upload(gpu, runMean, c);
        Tensor rv = upload(gpu, runVar, c);
        keep(GpuNormOps.batchNormTrain(upload(gpu, x, b, c, l), null, null, rm, rv, 1.0f, EPS));

        ChannelStats stats = ChannelStats.of(x, b, c, l, 0);
        float[] gotMean = rm.toFloatArray();
        float[] gotVar = rv.toFloatArray();
        double bessel = unbiasedScale(b * l);
        for (int k = 0; k < c; k++) {
            assertRunning("running mean " + k, runMean[k], stats.mean[k], stats.meanTerms[k], gotMean[k], b * l);
            assertRunning("running var " + k, runVar[k], stats.var[k] * bessel, stats.varTerms[k] * bessel,
                    gotVar[k], b * l);
        }
    }

    @Test
    public void batchNormTrainWritesBackStridedRunningStats() {
        int b = 3;
        int c = 4;
        int l = 5;
        float momentum = 0.25f;
        float[] x = TrainParity.uniform(47, b * c * l, -1f, 2f);
        float[] pairs = TrainParity.uniform(48, c * 2, 0.5f, 1.5f);
        Tensor parent = upload(gpu, pairs, c, 2);
        Tensor rm = keep(parent.slice(1, 0, 1));
        Tensor rv = keep(parent.slice(1, 1, 2));
        keep(GpuNormOps.batchNormTrain(upload(gpu, x, b, c, l), null, null, rm, rv, momentum, EPS));

        ChannelStats stats = ChannelStats.of(x, b, c, l, 0);
        float[] got = parent.toFloatArray();
        double bessel = unbiasedScale(b * l);
        for (int k = 0; k < c; k++) {
            double expMean = (1 - momentum) * (double) pairs[2 * k] + momentum * stats.mean[k];
            double expVar = (1 - momentum) * (double) pairs[2 * k + 1] + momentum * stats.var[k] * bessel;
            TrainParity.assertReduction("strided running mean " + k, expMean, got[2 * k], b * l,
                    (1 - momentum) * Math.abs(pairs[2 * k]) + momentum * stats.meanTerms[k]);
            TrainParity.assertReduction("strided running var " + k, expVar, got[2 * k + 1], b * l,
                    (1 - momentum) * Math.abs(pairs[2 * k + 1]) + momentum * stats.varTerms[k] * bessel);
        }
    }

    /** PyTorch's rule: a training batch with one value per channel has no unbiased variance and is rejected. */
    @Test
    public void batchNormTrainRejectsSingleValuePerChannel() {
        float[] stats = {0.1f, 0.2f, 0.3f};
        Tensor rm = upload(gpu, stats, 3);
        Tensor rv = upload(gpu, stats, 3);
        Tensor single = upload(gpu, new float[]{1f, 2f, 3f}, 1, 3, 1);
        try {
            GpuNormOps.batchNormTrain(single, null, null, rm, rv, 0.1f, EPS);
            fail("a [1, 3, 1] training batch must be rejected");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains("more than 1 value per channel"));
        }
        TrainParity.assertBitIdentical("running mean untouched", stats, rm.toFloatArray());
        TrainParity.assertBitIdentical("running var untouched", stats, rv.toFloatArray());
    }

    /** {@code n / (n - 1)}: the factor that turns the biased batch variance into PyTorch's unbiased running term. */
    private static double unbiasedScale(int n) {
        return (double) n / (n - 1);
    }

    /** {@code momentum == 1}: the CPU's {@code 0 * running + batch}, NaN when the running value is not finite. */
    private static void assertRunning(String what, float running, double batch, double terms, float got, int k) {
        if (Float.isNaN(running) || Float.isInfinite(running)) {
            assertTrue(what + " must be NaN, got " + got, Float.isNaN(got));
        } else {
            TrainParity.assertReduction(what, batch, got, k, terms);
        }
    }

    @Test
    public void batchNormEvalUsesRunningStats() {
        int b = 2;
        int c = 3;
        int l = 17;
        float[] x = TrainParity.uniform(51, b * c * l, -2f, 2f);
        float[] mean = TrainParity.uniform(52, c, -0.5f, 0.5f);
        float[] var = TrainParity.uniform(53, c, 0.5f, 2f);
        float[] gamma = TrainParity.uniform(54, c, 0.5f, 1.5f);
        float[] beta = TrainParity.uniform(55, c, -0.5f, 0.5f);
        Tensor out = keep(GpuNormOps.batchNormEval(upload(gpu, x, b, c, l), upload(gpu, gamma, c),
                upload(gpu, beta, c), upload(gpu, mean, c), upload(gpu, var, c), EPS));
        ChannelStats given = ChannelStats.given(mean, var);
        assertChannels("batchNormEval", out.toFloatArray(), x, b, c, l, given, gamma, beta);
    }

    @Test
    public void groupNormMatchesReference() {
        int b = 2;
        int c = 6;
        int s = 4 * 3;
        int groups = 3;
        float[] x = TrainParity.uniform(61, b * c * s, -3f, 3f);
        float[] gamma = TrainParity.uniform(62, c, 0.5f, 1.5f);
        float[] beta = TrainParity.uniform(63, c, -0.5f, 0.5f);
        Tensor out = keep(GpuNormOps.groupNorm(upload(gpu, x, b, c, 4, 3), groups, upload(gpu, gamma, c),
                upload(gpu, beta, c), EPS));
        assertChannels("groupNorm", out.toFloatArray(), x, b, c, s, ChannelStats.of(x, b, c, s, groups), gamma,
                beta);
    }

    @Test
    public void groupNormLongRowsUseChunkedPartials() {
        int b = 1;
        int c = 4;
        int s = 40000;
        int groups = 2;
        float[] x = TrainParity.uniform(71, b * c * s, -1f, 2f);
        Tensor out = keep(GpuNormOps.groupNorm(upload(gpu, x, b, c, s), groups, null, null, EPS));
        assertChannels("groupNorm long rows", out.toFloatArray(), x, b, c, s, ChannelStats.of(x, b, c, s, groups),
                null, null);
    }

    // ------------------------------------------------------------------ index select

    @Test
    public void indexSelectInt32IsBitIdentical() {
        int rows = 6;
        int dim = 5;
        float[] w = TrainParity.uniform(81, rows * dim, -1f, 1f);
        int[] idx = {3, 0, 5, 5, 1, 2};
        Tensor indices = keep(keep(Tensor.fromArray(gpu, idx)).reshape(Shape.of(2, 3)));
        Tensor out = keep(GpuNormOps.indexSelectRows(upload(gpu, w, rows, dim), indices, false));
        assertArrayEquals(new int[]{2, 3, dim}, out.shape().dims());
        TrainParity.assertBitIdentical("indexSelect INT32", gatherRows(w, dim, idx, rows, false),
                out.toFloatArray());
    }

    @Test
    public void indexSelectInt64NarrowsLikeJavaCast() {
        int rows = 4;
        int dim = 3;
        float[] w = TrainParity.uniform(91, rows * dim, -1f, 1f);
        long[] wide = {(1L << 32) + 3, 0, 2, -(1L << 33) + 1};
        int[] narrow = new int[wide.length];
        ByteBuffer bytes = ByteBuffer.allocate(wide.length * Long.BYTES).order(ByteOrder.nativeOrder());
        for (int i = 0; i < wide.length; i++) {
            bytes.putLong(wide[i]);
            narrow[i] = (int) wide[i];
        }
        bytes.flip();
        Tensor indices = keep(Tensor.fromStorage(gpu, Shape.vector(wide.length), DType.INT64,
                TensorStorage.fromHost(gpu, bytes, DType.INT64)));
        Tensor out = keep(GpuNormOps.indexSelectRows(upload(gpu, w, rows, dim), indices, false));
        TrainParity.assertBitIdentical("indexSelect INT64", gatherRows(w, dim, narrow, rows, false),
                out.toFloatArray());
    }

    @Test
    public void indexSelectClampMatchesMessagePassing() {
        int rows = 5;
        int dim = 4;
        float[] w = TrainParity.uniform(101, rows * dim, -1f, 1f);
        int[] idx = {-3, 0, 4, 9, 2};
        Tensor out = keep(GpuNormOps.indexSelectRows(upload(gpu, w, rows, dim), keep(Tensor.fromArray(gpu, idx)),
                true));
        TrainParity.assertBitIdentical("indexSelect clamp", gatherRows(w, dim, idx, rows, true), out.toFloatArray());
    }

    @Test
    public void indexSelectReportsOutOfRangeAtNextSync() {
        int rows = 3;
        int dim = 2;
        float[] w = TrainParity.uniform(111, rows * dim, -1f, 1f);
        int[] idx = {1, 7, 0};
        Tensor out = keep(GpuNormOps.indexSelectRows(upload(gpu, w, rows, dim), keep(Tensor.fromArray(gpu, idx)),
                false));
        assertNotNull(out);
        try {
            session.gpu().checkErrors();
            fail("an out-of-range index must raise IndexOutOfBoundsException at the next synchronization");
        } catch (IndexOutOfBoundsException expected) {
            assertTrue(expected.getMessage(),
                    expected.getMessage().startsWith(TrainGpu.embeddingIndexMessage(7) + " ("));
        }
        float[] got = out.toFloatArray();
        for (int j = 0; j < dim; j++) {
            assertTrue("row of the bad index is zero", Float.floatToRawIntBits(got[dim + j]) == 0);
        }
    }

    // ------------------------------------------------------------------ CPU path

    @Test
    public void everyOpDeclinesWithoutTrainLibrary() {
        ComputeBackend host = TrainParity.hostBackend();
        try {
            float[] x = TrainParity.uniform(121, 2 * 4 * 3, -1f, 1f);
            float[] stats = {0.1f, 0.2f, 0.3f, 0.4f};
            Tensor in = upload(host, x, 2, 4, 3);
            Tensor rm = upload(host, stats, 4);
            Tensor rv = upload(host, stats, 4);
            assertNull(GpuNormOps.layerNorm(in, 3, null, null, EPS));
            assertNull(GpuNormOps.rmsNorm(in, null, EPS));
            assertNull(GpuNormOps.batchNormTrain(in, null, null, rm, rv, 0.1f, EPS));
            assertNull(GpuNormOps.batchNormEval(in, null, null, rm, rv, EPS));
            assertNull(GpuNormOps.groupNorm(in, 2, null, null, EPS));
            assertNull(GpuNormOps.indexSelectRows(upload(host, x, 6, 4),
                    keep(Tensor.fromArray(host, new int[]{1})), false));
            TrainParity.assertBitIdentical("running mean untouched", stats, rm.toFloatArray());
        } finally {
            TrainGpu.release(host);
        }
    }

    // ------------------------------------------------------------------ references

    /** Row-wise LayerNorm ({@code rms == false}) or RMSNorm reference, asserted element by element. */
    private static void assertRows(String what, float[] got, float[] x, int rows, int cols, float[] scale,
                                   float[] shift, boolean rms, float eps) {
        for (int r = 0; r < rows; r++) {
            int base = r * cols;
            double mean = 0;
            double sq = 0;
            for (int j = 0; j < cols; j++) {
                mean += x[base + j];
                sq += (double) x[base + j] * x[base + j];
            }
            mean /= cols;
            double var = 0;
            for (int j = 0; j < cols; j++) {
                var += (x[base + j] - mean) * (x[base + j] - mean);
            }
            var /= cols;
            double denom = rms ? Math.sqrt(sq / cols + eps) : Math.sqrt(var + eps);
            double center = rms ? 0 : mean;
            for (int j = 0; j < cols; j++) {
                double g = scale == null ? 1 : scale[j];
                double b = shift == null ? 0 : shift[j];
                double ref = (x[base + j] - center) / denom * g + b;
                double terms = (Math.abs(x[base + j]) + Math.abs(center)) / denom * Math.abs(g) + Math.abs(b);
                TrainParity.assertReduction(what + " [" + r + "," + j + "]", ref, got[base + j], cols, terms);
            }
        }
    }

    /** Asserts a channel normalization of {@code x [b, c, s]} against {@code stats} (per channel or per group). */
    private static void assertChannels(String what, float[] got, float[] x, int b, int c, int s, ChannelStats stats,
                                       float[] gamma, float[] beta) {
        for (int bi = 0; bi < b; bi++) {
            for (int ci = 0; ci < c; ci++) {
                int k = stats.index(bi, ci, c);
                double sd = Math.sqrt(stats.var[k] + EPS);
                double g = gamma == null ? 1 : gamma[ci];
                double sh = beta == null ? 0 : beta[ci];
                for (int si = 0; si < s; si++) {
                    int e = (bi * c + ci) * s + si;
                    double ref = (x[e] - stats.mean[k]) / sd * g + sh;
                    double terms = (Math.abs(x[e]) + Math.abs(stats.mean[k])) / sd * Math.abs(g) + Math.abs(sh);
                    TrainParity.assertReduction(what + " [" + e + "]", ref, got[e], stats.count, terms);
                }
            }
        }
    }

    /** CPU row copy of Embedding ({@code clamp == false}, every index in range) or MessagePassing (clamped). */
    private static float[] gatherRows(float[] w, int dim, int[] idx, int rows, boolean clamp) {
        float[] out = new float[idx.length * dim];
        for (int i = 0; i < idx.length; i++) {
            int k = clamp ? Math.max(0, Math.min(idx[i], rows - 1)) : idx[i];
            System.arraycopy(w, k * dim, out, i * dim, dim);
        }
        return out;
    }

    /** Double-precision mean and biased variance per channel ({@code groups == 0}) or per (batch, group). */
    private static final class ChannelStats {
        final double[] mean;
        final double[] var;
        final double[] meanTerms;
        final double[] varTerms;
        final int groups;
        final int count;

        private ChannelStats(int sets, int groups, int count) {
            this.mean = new double[sets];
            this.var = new double[sets];
            this.meanTerms = new double[sets];
            this.varTerms = new double[sets];
            this.groups = groups;
            this.count = count;
        }

        int index(int b, int c, int channels) {
            return groups == 0 ? c : b * groups + c / (channels / groups);
        }

        static ChannelStats given(float[] mean, float[] var) {
            ChannelStats st = new ChannelStats(mean.length, 0, 1);
            for (int k = 0; k < mean.length; k++) {
                st.mean[k] = mean[k];
                st.var[k] = var[k];
            }
            return st;
        }

        static ChannelStats of(float[] x, int b, int c, int s, int groups) {
            int sets = groups == 0 ? c : b * groups;
            int count = groups == 0 ? b * s : (c / groups) * s;
            ChannelStats st = new ChannelStats(sets, groups, count);
            double[] abs = new double[sets];
            for (int bi = 0; bi < b; bi++) {
                for (int ci = 0; ci < c; ci++) {
                    int k = st.index(bi, ci, c);
                    for (int si = 0; si < s; si++) {
                        double v = x[(bi * c + ci) * s + si];
                        st.mean[k] += v;
                        abs[k] += Math.abs(v);
                    }
                }
            }
            for (int k = 0; k < sets; k++) {
                st.mean[k] /= count;
                st.meanTerms[k] = abs[k] / count;
            }
            for (int bi = 0; bi < b; bi++) {
                for (int ci = 0; ci < c; ci++) {
                    int k = st.index(bi, ci, c);
                    for (int si = 0; si < s; si++) {
                        double v = x[(bi * c + ci) * s + si];
                        st.var[k] += (v - st.mean[k]) * (v - st.mean[k]);
                        st.varTerms[k] += (Math.abs(v) + Math.abs(st.mean[k])) * (Math.abs(v) + Math.abs(st.mean[k]));
                    }
                }
            }
            for (int k = 0; k < sets; k++) {
                st.var[k] /= count;
                st.varTerms[k] /= count;
            }
            return st;
        }
    }
}
